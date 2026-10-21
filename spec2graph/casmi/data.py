"""Competition data: schema, peak cleaning, structure table, conditioning tensors, dataset and collator."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset

from spec2graph.casmi import chem
from spec2graph.casmi.adducts import AdductVocabulary, neutral_mass, try_parse
from spec2graph.casmi.spectral import N_QUANTILES, SpectralCache, quantiles
from spec2graph.data.collate import make_training_batch_collator
from spec2graph.data.elements import atom_types_to_indices
from spec2graph.data.massspecgym import parse_peak_list
from spectral_diffusion import TrainingBatch

logger = logging.getLogger(__name__)

ALIASES = {
    "molecule_id": ("molecule_id",),
    "precursor_mz": ("precursor_mz", "precursormz", "precursor"),
    "adduct": ("adduct", "precursor_type"),
    "collision_energy": ("collision_energy_ev", "collision_energy", "ce", "nce"),
    "mzs": ("ms2_mzs", "mzs", "mz"),
    "intensities": ("ms2_normalized_intensities", "ms2_intensities", "intensities", "intensity"),
    "smiles": ("normalized_smiles", "smiles", "canonical_smiles"),
    "instrument": ("instrument_type", "instrument"),
    "library": ("ingest_lib", "library", "source"),
}
REQUIRED = ("molecule_id", "precursor_mz", "adduct", "mzs", "intensities")
MAX_SPECTRA_PER_MOLECULE = 16  # the most spectra any test-set molecule has


def prepare(df: pd.DataFrame, require_smiles: bool = False) -> pd.DataFrame:
    """Canonical column names plus ``neutral_mass`` (NaN when the adduct is unsupported) and ``polarity``."""
    lower = {str(c).lower(): c for c in df.columns}
    rename = {}
    for canon, aliases in ALIASES.items():
        if canon not in df.columns and (found := next((lower[a] for a in aliases if a in lower), None)):
            rename[found] = canon
    out = df.rename(columns=rename).copy()
    if "molecule_id" not in out.columns and "inchikey14" in out.columns:
        out["molecule_id"] = molecule_ids(out)
    if missing :=[c for c in REQUIRED + (("smiles",) if require_smiles else ()) if c not in out.columns]:
        raise RuntimeError(f"competition schema mismatch; missing {missing}")
    out["collision_energy"] = [mean_energy(v) for v in out["collision_energy"]] if "collision_energy" in out.columns else np.nan
    out["molecule_id"] = out["molecule_id"].astype(str)
    out["neutral_mass"] = [neutral_mass(mz, a) for mz, a in zip(out["precursor_mz"], out["adduct"])]
    out["polarity"] = [a.polarity if (a := try_parse(ad)) else 0 for ad in out["adduct"]]
    return out.reset_index(drop=True)


def molecule_ids(df: pd.DataFrame) -> pd.Series:
    """``library:inchikey14:chunk`` for ``train.parquet``, which has no ``molecule_id``: each structure's spectra in a
    library, split into molecules of at most ``MAX_SPECTRA_PER_MOLECULE`` like the test set's."""
    group = (df["library"].astype(str) + ":" if "library" in df.columns else "") + df["inchikey14"].astype(str)
    return group + ":" + (group.groupby(group).cumcount() // MAX_SPECTRA_PER_MOLECULE).astype(str)


def mean_energy(value: object) -> float:
    """Kaggle's ``collision_energy_ev`` is a list (several values = merged acquisitions): its finite mean, else NaN."""
    try:
        values = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return float("nan")
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def load(path: str | Path, require_smiles: bool = False) -> pd.DataFrame:
    return prepare(pd.read_parquet(path), require_smiles)


def by_molecule(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """``molecule_id -> row positions`` in file order."""
    return dict(df.reset_index(drop=True).groupby("molecule_id", sort=False).indices)


@dataclass
class Peaks:
    """Spectrum cleaning shared by training and inference."""

    top_k: int = 128
    precursor_window: float = 0.5
    min_peaks: int = 3
    sqrt_intensity: bool = False

    def clean(self, mz: np.ndarray, intensity: np.ndarray, precursor_mz: float) -> tuple[np.ndarray, np.ndarray]:
        """Drop the precursor echo and junk peaks, keep the top-k by intensity, sort by m/z, scale to a base peak of 1."""
        mz, intensity = np.asarray(mz, np.float32).ravel(), np.asarray(intensity, np.float32).ravel()
        if mz.shape != intensity.shape:
            raise ValueError("mz/intensity length mismatch")
        keep = np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0) & (mz > 0)
        if np.isfinite(precursor_mz):
            keep &= mz < precursor_mz - self.precursor_window
        mz, intensity = mz[keep], intensity[keep]
        if mz.size > self.top_k:
            top = np.argpartition(intensity, -self.top_k)[-self.top_k:]
            mz, intensity = mz[top], intensity[top]
        if mz.size == 0:
            return mz, intensity
        order = np.argsort(mz, kind="stable")
        mz, intensity = mz[order], intensity[order] / intensity.max()
        return mz, np.sqrt(intensity) if self.sqrt_intensity else intensity

    def from_row(self, row: dict) -> tuple[np.ndarray, np.ndarray]:
        return self.clean(parse_peak_list(row["mzs"]), parse_peak_list(row["intensities"]), float(row["precursor_mz"]))


def structures(smiles: Iterable[object], canonical_tautomer: bool = False, n_jobs: int = 1, progress: bool = False,
               source: Optional[str] = None) -> pd.DataFrame:
    """One row per InChIKey14: identity, formula, exact mass, heavy-atom count, every input SMILES that mapped to it
    (``aliases``) and how many inputs did."""
    counts = Counter(s for s in smiles if isinstance(s, str) and s)
    records = chem.parallel_map(partial(chem.record, canonical_tautomer=canonical_tautomer), counts, n_jobs, progress, "structures")
    rows: dict[str, dict] = {}
    for smi, rec in zip(counts, records):
        if rec is not None:
            row = rows.setdefault(rec["inchikey14"], {**rec, "aliases": [], "n_inputs": 0, "source": source})
            row["aliases"].append(smi)
            row["n_inputs"] += counts[smi]
    return pd.DataFrame(list(rows.values()))


def merge_structures(tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """One row per InChIKey14: the first table's record, with every table's aliases and input counts pooled."""
    merged = pd.concat(list(tables), ignore_index=True)
    grouped = merged.groupby("inchikey14", sort=False)
    out = merged.drop_duplicates("inchikey14").set_index("inchikey14")
    out["aliases"] = grouped["aliases"].agg(lambda lists: list(dict.fromkeys(s for aliases in lists for s in aliases)))
    out["n_inputs"] = grouped["n_inputs"].sum()
    return out.reset_index()


@dataclass
class Conditioning:
    """Per-spectrum conditioning tensors, all shaped ``(batch,)``."""

    adduct_index: torch.Tensor
    collision_energy: torch.Tensor
    collision_energy_mask: torch.Tensor
    neutral_mass: torch.Tensor
    polarity: torch.Tensor

    @classmethod
    def build(cls, adduct_index, collision_energy, neutral_mass, polarity, device: str | torch.device = "cpu") -> "Conditioning":
        """From Python lists; ``None``/NaN collision energies are masked out."""
        ce = torch.tensor([float("nan") if v is None else float(v) for v in collision_energy], dtype=torch.float32, device=device)
        mask = torch.isfinite(ce)
        as_t = lambda values, dtype: torch.tensor(list(values), dtype=dtype, device=device)
        return cls(as_t(adduct_index, torch.long), ce.masked_fill(~mask, 0.0), mask, as_t(neutral_mass, torch.float32), as_t(polarity, torch.float32))

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "Conditioning":
        return Conditioning(*(fn(t) for t in vars(self).values()))

    def to(self, device) -> "Conditioning":
        return self.map(lambda t: t.to(device))

    def select(self, idx: torch.Tensor) -> "Conditioning":
        return self.map(lambda t: t.index_select(0, idx.to(t.device)))


@dataclass
class Spectra:
    """Padded peaks plus conditioning for a set of spectra (usually one molecule)."""

    mz: torch.Tensor
    intensity: torch.Tensor
    mask: torch.Tensor
    precursor_mz: torch.Tensor
    cond: Conditioning
    positions: np.ndarray

    @property
    def n(self) -> int:
        return int(self.mz.shape[0])

    @classmethod
    def from_rows(cls, rows: pd.DataFrame, vocab: AdductVocabulary, max_peaks: int, peaks: Peaks = Peaks(),
                  device: str | torch.device = "cpu") -> Optional["Spectra"]:
        """Clean, pad and condition the usable rows; ``None`` when nothing usable remains."""
        kept = [(pos, row, *peaks.from_row(row)) for pos, row in enumerate(rows.to_dict("records")) if np.isfinite(row["neutral_mass"])]
        kept = [(pos, row, mz[:max_peaks], it[:max_peaks]) for pos, row, mz, it in kept if mz.size >= peaks.min_peaks]
        if not kept:
            return None
        rows_ = [r for _, r, _, _ in kept]
        cond = Conditioning.build([vocab.encode(r["adduct"]) for r in rows_], [r.get("collision_energy") for r in rows_],
                                  [r["neutral_mass"] for r in rows_], [r["polarity"] for r in rows_], device)
        mz, it, mask = _pad([m for _, _, m, _ in kept], [x for _, _, _, x in kept])
        t = lambda a, dtype: torch.as_tensor(a, dtype=dtype, device=device)
        return cls(t(mz, torch.float32), t(it, torch.float32), t(mask, torch.bool),
                   t([float(r["precursor_mz"]) for r in rows_], torch.float32), cond, np.array([p for p, _, _, _ in kept]))


def _pad(mzs: list[np.ndarray], intensities: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-pad ragged peak lists to ``(n, width)`` arrays plus a validity mask."""
    width = max(m.size for m in mzs)
    mz, it, mask = (np.zeros((len(mzs), width), dtype) for dtype in (np.float32, np.float32, bool))
    for i, (m, x) in enumerate(zip(mzs, intensities)):
        mz[i, : m.size], it[i, : m.size], mask[i, : m.size] = m, x, True
    return mz, it, mask


class Dataset(TorchDataset):
    """Training spectra with cached spectral targets and the conditioning fields.

    Cleaned peaks are stored flat with offsets and the per-spectrum fields as arrays, not as row dicts, so a copy fits
    in each spawned DataLoader worker.
    """

    SCALARS = ("precursor_mz", "collision_energy", "neutral_mass", "polarity")

    def __init__(self, spectra: pd.DataFrame, cache: SpectralCache, vocab: AdductVocabulary, *, max_atoms: int = 96,
                 n_quantiles: int = N_QUANTILES, peaks: Peaks = Peaks(), table: Optional[pd.DataFrame] = None,
                 adjacency: bool = False, n_jobs: int = 1):
        df = spectra if "neutral_mass" in spectra.columns else prepare(spectra, require_smiles=True)
        self.cache, self.vocab, self.n_quantiles, self.peaks, self.adjacency, self.max_atoms = cache, vocab, n_quantiles, peaks, adjacency, max_atoms
        table = structures(df["smiles"].unique(), n_jobs=n_jobs) if table is None else table
        lookup = {key: r for r in table.to_dict("records") for key in (*r["aliases"], r["smiles"])}
        drops: Counter = Counter()
        kept, self.records, self.cached = [], [], {}
        for row in df[["molecule_id", "smiles", "adduct", "mzs", "intensities", *self.SCALARS]].to_dict("records"):
            reason, cleaned = self._check(row, rec := lookup.get(row["smiles"]))
            if reason:
                drops[reason] += 1
            else:
                kept.append((row, *cleaned))
                self.records.append(rec)
        self._store(kept)
        self.drops = dict(drops)
        logger.info("Dataset: kept %d of %d spectra; dropped %s", len(self), len(df), self.drops)

    def _check(self, row: dict, rec: Optional[dict]) -> tuple[Optional[str], Optional[tuple[np.ndarray, np.ndarray]]]:
        """``(drop reason, None)`` or ``(None, cleaned peaks)``."""
        if rec is None:
            return "unparseable smiles", None
        if not np.isfinite(row["neutral_mass"]):
            return "unsupported adduct", None
        if not 0 < rec["n_heavy_atoms"] <= self.max_atoms:
            return "atom count", None
        try:
            cleaned = self.peaks.from_row(row)
        except (ValueError, TypeError):
            return "bad peaks", None
        if cleaned[0].size < self.peaks.min_peaks:
            return "too few peaks", None
        if (key := rec["inchikey"]) not in self.cached:
            lookups = (self.cache.get_or_compute, self.cache.get_or_compute_fingerprint, self.cache.spectrum)
            self.cached[key] = all(f(rec["smiles"], key) is not None for f in lookups)
        return (None, cleaned) if self.cached[key] else ("cache", None)

    def _store(self, kept: list[tuple[dict, np.ndarray, np.ndarray]]) -> None:
        flat = lambda i: np.concatenate([k[i] for k in kept]).astype(np.float32) if kept else np.zeros(0, np.float32)
        self.mz, self.intensity = flat(1), flat(2)
        self.offsets = np.cumsum([0, *(mz.size for _, mz, _ in kept)], dtype=np.int64)
        self.molecule_ids = [row["molecule_id"] for row, _, _ in kept]
        self.adduct_index = np.array([self.vocab.encode(row["adduct"]) for row, _, _ in kept], dtype=np.int64)
        self.scalars = {name: np.array([row[name] for row, _, _ in kept], dtype=np.float64) for name in self.SCALARS}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        rec, (start, stop) = self.records[i], self.offsets[i : i + 2]
        smiles, key = rec["smiles"], rec["inchikey"]
        eigvecs, atoms = self.cache.get_or_compute(smiles, key), chem.symbols(smiles)
        sample = dict(
            eigvecs=eigvecs, n_atoms=eigvecs.shape[0], mz=self.mz[start:stop], intensity=self.intensity[start:stop],
            fingerprint=self.cache.get_or_compute_fingerprint(smiles, key), atom_types=np.asarray(atom_types_to_indices(atoms), dtype=np.int64),
            atom_symbols=atoms, smiles=smiles, formula=rec["formula"], inchikey=key, inchikey14=rec["inchikey14"],
            molecule_id=self.molecule_ids[i], adduct_index=int(self.adduct_index[i]),
            spectral_quantiles=quantiles(self.cache.spectrum(smiles, key), self.n_quantiles),
            **{name: float(values[i]) for name, values in self.scalars.items()},
        )
        if self.adjacency:
            sample["adjacency"] = chem.kekulized_adjacency(smiles)
        return sample


@dataclass
class Batch:
    """A core :class:`TrainingBatch` plus conditioning, quantile targets, metadata and optional adjacencies."""

    core: TrainingBatch
    cond: Conditioning
    quantiles: torch.Tensor
    meta: dict[str, list] = field(default_factory=dict)
    adjacencies: Optional[list[np.ndarray]] = None

    def to(self, device) -> "Batch":
        core = TrainingBatch(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in vars(self.core).items()})
        return Batch(core, self.cond.to(device), self.quantiles.to(device), self.meta, self.adjacencies)


class Collator:
    """Wraps the core collator and adds the conditioning, the quantile targets and metadata."""

    META = ("molecule_id", "smiles", "inchikey14", "formula", "n_atoms", "neutral_mass")

    def __init__(self, max_atoms: int, max_peaks: int, k: int, fingerprint_bits: int, device: str = "cpu"):
        self.core = make_training_batch_collator(max_atoms=max_atoms, max_peaks=max_peaks, k=k, fingerprint_bits=fingerprint_bits, device=device)
        self.device = device
        self.args = (max_atoms, max_peaks, k, fingerprint_bits, device)

    def __reduce__(self):  # the core collator is a closure; rebuild it so spawned DataLoader workers can unpickle this
        return Collator, self.args

    def __call__(self, samples: list[dict]) -> Batch:
        samples = [s for s in samples if s["n_atoms"] > 0 and np.size(s["mz"]) > 0]
        if not samples:
            raise ValueError("empty batch after validation")
        col = lambda key: [s[key] for s in samples]
        cond = Conditioning.build(col("adduct_index"), col("collision_energy"), col("neutral_mass"), col("polarity"), self.device)
        q = torch.as_tensor(np.stack(col("spectral_quantiles")), dtype=torch.float32, device=self.device)
        return Batch(self.core(samples), cond, q, {k: col(k) for k in self.META}, col("adjacency") if "adjacency" in samples[0] else None)
