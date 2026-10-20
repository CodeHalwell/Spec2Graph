"""Competition data: parquet schema, preprocessing, structures, dataset, collator.

The Kaggle release ships ``train.parquet`` (spectra with answers) and
``test.parquet`` (spectra only). Column names are normalised through
:data:`COLUMN_ALIASES` so the rest of the layer can rely on::

    molecule_id, precursor_mz, adduct, collision_energy, mzs, intensities,
    smiles (train only), instrument, library

Derived columns added by :func:`load_casmi_parquet`:

* ``adduct_canonical`` — canonical adduct spelling (``None`` when unparseable);
* ``neutral_mass`` — monoisotopic neutral mass from precursor m/z and adduct;
* ``polarity`` — ``+1`` / ``-1``.

Structures are identified by tautomer-canonical InChIKey14 everywhere, the
same identity the metric uses. :func:`build_structure_table` produces one
row per structure with formula, exact mass and heavy-atom count.

:class:`CasmiSpectrumDataset` yields the sample dict the core collators
expect (``eigvecs``, ``mz``, ``intensity``, ``fingerprint``, ``atom_types``
...) plus the conditioning fields, and :func:`make_casmi_collator` wraps the
core collator so the model sees a :class:`CasmiBatch`.

Competition data is CC BY-NC 4.0: never commit the parquet files or the
caches derived from them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from spec2graph.casmi.adducts import AdductVocabulary, canonical_adduct_name, neutral_mass, try_parse_adduct
from spec2graph.casmi.metric import inchikey14
from spec2graph.casmi.spectral import DEFAULT_N_QUANTILES, SpectralCache, spectral_quantiles
from spec2graph.data.collate import make_training_batch_collator
from spec2graph.data.elements import atom_types_to_indices
from spec2graph.data.massspecgym import parse_peak_list
from spectral_diffusion import MAX_SMILES_LENGTH, TrainingBatch

try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover - torch is a hard dependency
    _TorchDataset = object

logger = logging.getLogger(__name__)

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "molecule_id": ("molecule_id",),
    "spectrum_id": ("spectrum_id", "spec_id", "id"),
    "precursor_mz": ("precursor_mz", "precursor_m_z", "precursormz", "precursor"),
    "adduct": ("adduct", "precursor_type", "ionization"),
    "collision_energy": ("collision_energy_ev", "collision_energy", "ce", "nce"),
    "mzs": ("ms2_mzs", "mzs", "peaks_mz", "mz"),
    "intensities": (
        "ms2_normalized_intensities",
        "ms2_intensities",
        "intensities",
        "peaks_intensity",
        "intensity",
    ),
    "smiles": ("normalized_smiles", "smiles", "canonical_smiles"),
    "instrument": ("instrument_type", "instrument"),
    "library": ("ingest_lib", "library", "source"),
}
REQUIRED_SPECTRUM_COLUMNS: tuple[str, ...] = ("molecule_id", "precursor_mz", "adduct", "mzs", "intensities")
STRUCTURE_COLUMNS: tuple[str, ...] = (
    "smiles",
    "inchikey",
    "inchikey14",
    "formula",
    "exact_mass",
    "n_heavy_atoms",
)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases to canonical column names (copy, other columns kept)."""
    rename: dict[str, str] = {}
    lower_to_actual = {str(c).lower(): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            actual = lower_to_actual.get(alias.lower())
            if actual is not None and actual not in rename:
                rename[actual] = canonical
                break
    return df.rename(columns=rename).copy()


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``adduct_canonical``, ``neutral_mass`` and ``polarity`` (copy)."""
    out = df.copy()
    adducts = out["adduct"].astype(object)
    out["adduct_canonical"] = pd.Series(
        [canonical_adduct_name(a) if isinstance(a, str) else None for a in adducts],
        index=out.index,
        dtype=object,
    )
    precursor = pd.to_numeric(out["precursor_mz"], errors="coerce")
    masses = np.full(len(out), np.nan, dtype=np.float64)
    polarity = np.zeros(len(out), dtype=np.int8)
    for i, (mz, adduct) in enumerate(zip(precursor.to_numpy(), adducts)):
        parsed = try_parse_adduct(adduct)
        if parsed is None or not np.isfinite(mz) or mz <= 0:
            continue
        masses[i] = parsed.neutral_mass(float(mz))
        polarity[i] = parsed.polarity
    out["neutral_mass"] = masses
    out["polarity"] = polarity
    return out


def load_casmi_parquet(
    path: str | Path,
    *,
    columns: Optional[Sequence[str]] = None,
    require_smiles: bool = False,
) -> pd.DataFrame:
    """Read a competition parquet file into the canonical schema."""
    df = pd.read_parquet(path, columns=list(columns) if columns else None)
    return prepare_spectra_frame(df, require_smiles=require_smiles)


def prepare_spectra_frame(df: pd.DataFrame, *, require_smiles: bool = False) -> pd.DataFrame:
    """Normalise an in-memory frame (used by tests and by the loader)."""
    df = normalise_columns(df)
    missing = [c for c in REQUIRED_SPECTRUM_COLUMNS if c not in df.columns]
    if require_smiles and "smiles" not in df.columns:
        missing.append("smiles")
    if missing:
        raise RuntimeError(
            f"Competition schema mismatch; missing {missing}. Columns: {list(df.columns)}"
        )
    if "collision_energy" not in df.columns:
        df["collision_energy"] = np.nan
    df["molecule_id"] = df["molecule_id"].astype(str)
    df = add_derived_columns(df)
    return df.reset_index(drop=True)


def group_by_molecule(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """``molecule_id -> row positions`` preserving file order."""
    groups: dict[str, list[int]] = {}
    for position, molecule_id in enumerate(df["molecule_id"].astype(str).to_numpy()):
        groups.setdefault(molecule_id, []).append(position)
    return {key: np.asarray(value, dtype=np.int64) for key, value in groups.items()}


# ----------------------------------------------------------------------
# Peak preprocessing
# ----------------------------------------------------------------------


@dataclass
class PeakPreprocessConfig:
    """Spectrum cleaning applied identically at train and test time."""

    top_k_peaks: int = 128
    max_peaks: int = 128
    # Drop peaks at or above ``precursor_mz - precursor_window`` (precursor echo).
    precursor_window: float = 0.5
    # Drop peaks below this fraction of the base peak after normalisation.
    min_relative_intensity: float = 0.0
    # Square-root compress intensities (common in spectral similarity work).
    sqrt_intensity: bool = False
    min_peaks: int = 3


def preprocess_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
    config: Optional[PeakPreprocessConfig] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean one spectrum. Returns arrays sorted by m/z, intensities in [0, 1]."""
    config = config or PeakPreprocessConfig()
    mz = np.asarray(mz, dtype=np.float32).ravel()
    intensity = np.asarray(intensity, dtype=np.float32).ravel()
    if mz.shape != intensity.shape:
        raise ValueError(f"mz and intensity length mismatch: {mz.shape} vs {intensity.shape}.")
    keep = np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0) & (mz > 0)
    if np.isfinite(precursor_mz) and precursor_mz > 0:
        keep &= mz < (float(precursor_mz) - config.precursor_window)
    mz, intensity = mz[keep], intensity[keep]
    if mz.size == 0:
        return mz, intensity
    intensity = intensity / intensity.max()
    if config.min_relative_intensity > 0:
        keep = intensity >= config.min_relative_intensity
        mz, intensity = mz[keep], intensity[keep]
    if mz.size > config.top_k_peaks:
        top = np.argpartition(intensity, -config.top_k_peaks)[-config.top_k_peaks :]
        mz, intensity = mz[top], intensity[top]
    order = np.argsort(mz, kind="stable")
    mz, intensity = mz[order], intensity[order]
    if mz.size > config.max_peaks:
        mz, intensity = mz[: config.max_peaks], intensity[: config.max_peaks]
    if config.sqrt_intensity:
        intensity = np.sqrt(intensity)
    return mz.astype(np.float32), intensity.astype(np.float32)


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


# ----------------------------------------------------------------------
# Structures
# ----------------------------------------------------------------------


def structure_record(smiles: object, *, canonical_tautomer: bool = False) -> Optional[dict[str, Any]]:
    """Identity and bulk properties of one structure, or ``None`` if unparseable.

    ``inchikey`` is the full key of the parsed molecule (used for cache
    paths); ``inchikey14`` is the tautomer-canonical first block (the
    competition identity). With ``canonical_tautomer=True`` the returned
    ``smiles`` is RDKit's canonical tautomer, which keeps one spelling per
    structure in the candidate pool.
    """
    if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > MAX_SMILES_LENGTH:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rdkit is required for structure_record()") from exc
    from spec2graph.casmi.metric import _get_tautomer_enumerator, _quiet_rdkit

    with _quiet_rdkit():
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        if canonical_tautomer:
            try:
                mol = _get_tautomer_enumerator().Canonicalize(mol)
            except Exception:  # RDKit native exception
                pass
        try:
            key = Chem.MolToInchiKey(mol)
            canonical = Chem.MolToSmiles(mol)
            formula = rdMolDescriptors.CalcMolFormula(mol)
            exact_mass = float(Descriptors.ExactMolWt(mol))
        except Exception:
            return None
    if not key:
        return None
    key14 = inchikey14(canonical)
    if key14 is None:
        return None
    return {
        "smiles": canonical,
        "inchikey": key,
        "inchikey14": key14,
        "formula": formula,
        "exact_mass": exact_mass,
        "n_heavy_atoms": int(mol.GetNumHeavyAtoms()),
    }


def build_structure_table(
    smiles: Iterable[object],
    *,
    canonical_tautomer: bool = False,
    n_jobs: int = 1,
    show_progress: bool = False,
    source: Optional[str] = None,
) -> pd.DataFrame:
    """One row per distinct InChIKey14 with :data:`STRUCTURE_COLUMNS`.

    ``n_inputs`` counts how many input SMILES mapped to each structure
    (for the training set that is the number of spectra). The first SMILES
    seen for a structure provides its representative spelling.
    """
    unique: dict[str, int] = {}
    for value in smiles:
        if isinstance(value, str) and value:
            unique[value] = unique.get(value, 0) + 1
    inputs = list(unique.keys())

    iterator: Iterable[str] = inputs
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(inputs, desc="structures", unit="smiles")
        except ImportError:
            pass

    if n_jobs == 1:
        records = [structure_record(s, canonical_tautomer=canonical_tautomer) for s in iterator]
    else:
        try:
            from joblib import Parallel, delayed

            records = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=256)(
                delayed(structure_record)(s, canonical_tautomer=canonical_tautomer) for s in iterator
            )
        except ImportError:
            logger.warning("joblib not installed; building the structure table serially.")
            records = [structure_record(s, canonical_tautomer=canonical_tautomer) for s in inputs]

    rows: dict[str, dict[str, Any]] = {}
    failed = 0
    for original, record in zip(inputs, records):
        if record is None:
            failed += 1
            continue
        key = record["inchikey14"]
        if key in rows:
            rows[key]["n_inputs"] += unique[original]
            continue
        record = dict(record)
        record["input_smiles"] = original
        record["n_inputs"] = unique[original]
        if source is not None:
            record["source"] = source
        rows[key] = record
    if failed:
        logger.info("build_structure_table: %d of %d SMILES could not be parsed.", failed, len(inputs))
    columns = list(STRUCTURE_COLUMNS) + ["input_smiles", "n_inputs"] + (["source"] if source else [])
    table = pd.DataFrame(list(rows.values()), columns=columns)
    return table.reset_index(drop=True)


def merge_structure_tables(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate tables keeping the first row per InChIKey14."""
    if not tables:
        raise ValueError("No structure tables to merge.")
    merged = pd.concat(list(tables), ignore_index=True)
    merged = merged.drop_duplicates(subset=["inchikey14"], keep="first")
    return merged.reset_index(drop=True)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------


@dataclass
class DatasetFilterStats:
    initial: int
    dropped_no_smiles: int = 0
    dropped_bad_adduct: int = 0
    dropped_unparseable: int = 0
    dropped_too_many_atoms: int = 0
    dropped_few_peaks: int = 0
    dropped_cache: int = 0
    final: int = 0

    def __str__(self) -> str:
        parts = [f"{self.initial} initial"]
        for name in (
            "dropped_no_smiles",
            "dropped_bad_adduct",
            "dropped_unparseable",
            "dropped_too_many_atoms",
            "dropped_few_peaks",
            "dropped_cache",
        ):
            value = getattr(self, name)
            if value:
                parts.append(f"-{value} {name[len('dropped_'):]}")
        parts.append(f"={self.final}")
        return " ".join(parts)


class CasmiSpectrumDataset(_TorchDataset):
    """Training/validation dataset over competition spectra.

    Each item is a dict with the fields the core collator needs
    (``eigvecs``, ``n_atoms``, ``mz``, ``intensity``, ``precursor_mz``,
    ``fingerprint``, ``atom_types``, ``atom_symbols``, ``smiles``,
    ``formula``, ``inchikey``) plus ``inchikey14``, ``molecule_id``,
    ``adduct_index``, ``collision_energy`` (NaN when unknown),
    ``neutral_mass``, ``polarity`` and ``spectral_quantiles``.

    Rows whose structure cannot be processed, whose adduct is unsupported,
    whose heavy-atom count exceeds ``max_atoms`` or whose spectrum has
    fewer than ``min_peaks`` usable peaks are dropped at construction.
    """

    def __init__(
        self,
        spectra: pd.DataFrame,
        *,
        cache: SpectralCache | str | Path,
        adduct_vocab: AdductVocabulary,
        k: int = 8,
        max_atoms: int = 96,
        fingerprint_bits: int = 2048,
        n_quantiles: int = DEFAULT_N_QUANTILES,
        peak_config: Optional[PeakPreprocessConfig] = None,
        structures: Optional[pd.DataFrame] = None,
        include_adjacency: bool = False,
        precompute: bool = True,
        n_jobs: int = 1,
    ) -> None:
        spectra = normalise_columns(spectra)
        if "smiles" not in spectra.columns:
            raise ValueError("CasmiSpectrumDataset needs a 'smiles' column (training data).")
        if isinstance(cache, (str, Path)):
            cache = SpectralCache(cache, k=k, fingerprint_bits=fingerprint_bits)
        if cache.k != k:
            raise ValueError(f"cache.k={cache.k} does not match k={k}.")
        self.cache = cache
        self.k = k
        self.max_atoms = max_atoms
        self.fingerprint_bits = fingerprint_bits
        self.n_quantiles = n_quantiles
        self.peak_config = peak_config or PeakPreprocessConfig()
        self.adduct_vocab = adduct_vocab
        self.include_adjacency = include_adjacency

        df = spectra if "neutral_mass" in spectra.columns else prepare_spectra_frame(spectra, require_smiles=True)
        stats = DatasetFilterStats(initial=len(df))

        has_smiles = df["smiles"].apply(lambda s: isinstance(s, str) and bool(s))
        stats.dropped_no_smiles = int((~has_smiles).sum())
        df = df[has_smiles]

        good_adduct = df["neutral_mass"].notna()
        stats.dropped_bad_adduct = int((~good_adduct).sum())
        df = df[good_adduct]

        # Structure lookup: one RDKit pass per distinct SMILES.
        if structures is None:
            structures = build_structure_table(df["smiles"].unique(), n_jobs=n_jobs)
        by_input = {}
        for row in structures.to_dict("records"):
            by_input.setdefault(row.get("input_smiles", row["smiles"]), row)
            by_input.setdefault(row["smiles"], row)
        records = [by_input.get(s) for s in df["smiles"].to_numpy()]
        parseable = np.array([r is not None for r in records], dtype=bool)
        stats.dropped_unparseable = int((~parseable).sum())
        df = df[parseable]
        records = [r for r in records if r is not None]

        n_atoms = np.array([r["n_heavy_atoms"] for r in records], dtype=np.int64)
        fits = (n_atoms > 0) & (n_atoms <= max_atoms)
        stats.dropped_too_many_atoms = int((~fits).sum())
        df = df[fits]
        records = [r for r, ok in zip(records, fits) if ok]

        self._df = df.reset_index(drop=True)
        self._records = records

        if precompute:
            unique = pd.DataFrame(
                {"inchikey": [r["inchikey"] for r in records], "smiles": [r["smiles"] for r in records]}
            ).drop_duplicates("inchikey")
            self.cache.precompute_all(unique, n_jobs=n_jobs, show_progress=False)
            for key, smi in zip(unique["inchikey"], unique["smiles"]):
                self.cache.get_or_compute_spectrum(smi, key)

        keep: list[int] = []
        for idx in range(len(self._df)):
            row = self._df.iloc[idx]
            record = self._records[idx]
            mz = parse_peak_list(row["mzs"])
            intensity = parse_peak_list(row["intensities"])
            if mz.shape != intensity.shape:
                stats.dropped_few_peaks += 1
                continue
            mz, _ = preprocess_peaks(mz, intensity, _safe_float(row["precursor_mz"]), self.peak_config)
            if mz.size < self.peak_config.min_peaks:
                stats.dropped_few_peaks += 1
                continue
            if self.cache.get_or_compute(record["smiles"], record["inchikey"]) is None:
                stats.dropped_cache += 1
                continue
            if self.cache.get_or_compute_fingerprint(record["smiles"], record["inchikey"]) is None:
                stats.dropped_cache += 1
                continue
            keep.append(idx)
        self._df = self._df.iloc[keep].reset_index(drop=True)
        self._records = [self._records[i] for i in keep]
        stats.final = len(self._df)
        self.stats = stats
        logger.info("CasmiSpectrumDataset: %s", stats)

    def __len__(self) -> int:
        return len(self._df)

    @property
    def frame(self) -> pd.DataFrame:
        return self._df

    @property
    def inchikey14s(self) -> np.ndarray:
        return np.asarray([r["inchikey14"] for r in self._records], dtype=object)

    @property
    def molecule_ids(self) -> np.ndarray:
        return self._df["molecule_id"].astype(str).to_numpy()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._df.iloc[idx]
        record = self._records[idx]
        smiles = record["smiles"]
        inchikey = record["inchikey"]

        eigvecs = self.cache.get_or_compute(smiles, inchikey)
        fingerprint = self.cache.get_or_compute_fingerprint(smiles, inchikey)
        spectrum = self.cache.get_or_compute_spectrum(smiles, inchikey)
        assert eigvecs is not None and fingerprint is not None and spectrum is not None

        precursor_mz = _safe_float(row["precursor_mz"])
        mz, intensity = preprocess_peaks(
            parse_peak_list(row["mzs"]), parse_peak_list(row["intensities"]), precursor_mz, self.peak_config
        )
        symbols = _atom_symbols(smiles)
        sample: dict[str, Any] = {
            "eigvecs": eigvecs.astype(np.float32, copy=False),
            "n_atoms": int(eigvecs.shape[0]),
            "mz": mz,
            "intensity": intensity,
            "precursor_mz": precursor_mz,
            "fingerprint": fingerprint.astype(np.float32, copy=False)[: self.fingerprint_bits],
            "atom_types": np.asarray(atom_types_to_indices(symbols), dtype=np.int64),
            "atom_symbols": symbols,
            "smiles": smiles,
            "formula": record["formula"],
            "inchikey": inchikey,
            "inchikey14": record["inchikey14"],
            "molecule_id": str(row["molecule_id"]),
            "adduct_index": int(self.adduct_vocab.encode(row["adduct"])),
            "collision_energy": _safe_float(row.get("collision_energy")),
            "neutral_mass": float(row["neutral_mass"]),
            "polarity": float(row["polarity"]),
            "spectral_quantiles": spectral_quantiles(spectrum, self.n_quantiles),
        }
        if self.include_adjacency:
            sample["adjacency"] = kekulized_adjacency(smiles)
        return sample


def _atom_symbols(smiles: str) -> list[str]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def kekulized_adjacency(smiles: str) -> np.ndarray:
    """Heavy-atom adjacency with integer bond orders (aromatic bonds kekulised).

    Atom order matches RDKit's parse order of ``smiles``, i.e. the order of
    the eigenvector rows. Falls back to aromatic order 1.5 rounded down to
    1 if kekulisation fails.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((0, 0), dtype=np.float32)
    n = mol.GetNumAtoms()
    adjacency = np.zeros((n, n), dtype=np.float32)
    try:
        kekulized = Chem.Mol(mol)
        Chem.Kekulize(kekulized, clearAromaticFlags=True)
        mol = kekulized
    except Exception:  # RDKit native exception
        pass
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        order = bond.GetBondTypeAsDouble()
        order = 1.0 if order == 1.5 else float(round(order))
        adjacency[i, j] = adjacency[j, i] = order
    return adjacency


# ----------------------------------------------------------------------
# Collator
# ----------------------------------------------------------------------


@dataclass
class CasmiBatch:
    """A core :class:`TrainingBatch` plus the competition conditioning."""

    batch: TrainingBatch
    adduct_index: "torch.Tensor"
    collision_energy: "torch.Tensor"
    collision_energy_mask: "torch.Tensor"
    neutral_mass: "torch.Tensor"
    polarity: "torch.Tensor"
    spectral_quantiles: "torch.Tensor"
    metadata: dict[str, list[Any]] = field(default_factory=dict)
    adjacencies: Optional[list[np.ndarray]] = None


def _usable(sample: Mapping[str, Any]) -> bool:
    return int(sample.get("n_atoms", 0)) > 0 and np.asarray(sample["mz"]).size > 0


def make_casmi_collator(
    *,
    max_atoms: int,
    max_peaks: int,
    k: int,
    fingerprint_bits: int,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    device: str = "cpu",
) -> Callable[[list[dict[str, Any]]], CasmiBatch]:
    """Return a collator producing :class:`CasmiBatch` on ``device``."""
    import torch

    core_collate = make_training_batch_collator(
        max_atoms=max_atoms,
        max_peaks=max_peaks,
        k=k,
        fingerprint_bits=fingerprint_bits,
        device=device,
    )

    def collate(samples: list[dict[str, Any]]) -> CasmiBatch:
        samples = [s for s in samples if _usable(s)]
        if not samples:
            raise ValueError("All samples in the batch were dropped during validation.")
        batch = core_collate(samples)
        n = len(samples)
        ce = np.array([_safe_float(s.get("collision_energy")) for s in samples], dtype=np.float32)
        ce_mask = np.isfinite(ce)
        ce = np.where(ce_mask, ce, 0.0).astype(np.float32)
        quantiles = np.zeros((n, n_quantiles), dtype=np.float32)
        for i, s in enumerate(samples):
            q = np.asarray(s["spectral_quantiles"], dtype=np.float32)
            quantiles[i, : min(n_quantiles, q.size)] = q[:n_quantiles]
        metadata = {
            "molecule_id": [str(s.get("molecule_id", "")) for s in samples],
            "smiles": [s["smiles"] for s in samples],
            "inchikey14": [s.get("inchikey14") for s in samples],
            "formula": [s.get("formula", "") for s in samples],
            "n_atoms": [int(s["n_atoms"]) for s in samples],
            "neutral_mass": [float(s["neutral_mass"]) for s in samples],
        }
        adjacencies = None
        if "adjacency" in samples[0]:
            adjacencies = [np.asarray(s["adjacency"], dtype=np.float32) for s in samples]
        return CasmiBatch(
            batch=batch,
            adduct_index=torch.tensor([int(s["adduct_index"]) for s in samples], dtype=torch.long, device=device),
            collision_energy=torch.from_numpy(ce).to(device),
            collision_energy_mask=torch.from_numpy(ce_mask).to(device),
            neutral_mass=torch.tensor([float(s["neutral_mass"]) for s in samples], dtype=torch.float32, device=device),
            polarity=torch.tensor([float(s["polarity"]) for s in samples], dtype=torch.float32, device=device),
            spectral_quantiles=torch.from_numpy(quantiles).to(device),
            metadata=metadata,
            adjacencies=adjacencies,
        )

    return collate


@dataclass
class SpectraTensors:
    """Padded tensors for a set of spectra (one molecule, typically)."""

    mz: "torch.Tensor"
    intensity: "torch.Tensor"
    spectrum_mask: "torch.Tensor"
    precursor_mz: "torch.Tensor"
    adduct_index: "torch.Tensor"
    collision_energy: "torch.Tensor"
    collision_energy_mask: "torch.Tensor"
    neutral_mass: "torch.Tensor"
    polarity: "torch.Tensor"
    row_positions: np.ndarray

    @property
    def n_spectra(self) -> int:
        return int(self.mz.shape[0])


def spectra_to_tensors(
    rows: pd.DataFrame,
    *,
    adduct_vocab: AdductVocabulary,
    max_peaks: int,
    peak_config: Optional[PeakPreprocessConfig] = None,
    device: str = "cpu",
) -> Optional[SpectraTensors]:
    """Preprocess and pad the spectra in ``rows`` (a prepared frame slice).

    Rows with an unsupported adduct or too few peaks are skipped; returns
    ``None`` when nothing usable remains. ``row_positions`` records which
    positional rows of ``rows`` survived.
    """
    import torch

    peak_config = peak_config or PeakPreprocessConfig()
    kept_mz: list[np.ndarray] = []
    kept_int: list[np.ndarray] = []
    meta: list[tuple[float, int, float, float, float]] = []
    positions: list[int] = []
    records = rows.to_dict("records")
    for position, row in enumerate(records):
        mass = _safe_float(row.get("neutral_mass"))
        if not np.isfinite(mass):
            continue
        try:
            mz = parse_peak_list(row["mzs"])
            intensity = parse_peak_list(row["intensities"])
        except (ValueError, TypeError):
            continue
        if mz.shape != intensity.shape:
            continue
        precursor = _safe_float(row.get("precursor_mz"))
        mz, intensity = preprocess_peaks(mz, intensity, precursor, peak_config)
        if mz.size < max(1, peak_config.min_peaks):
            continue
        kept_mz.append(mz[:max_peaks])
        kept_int.append(intensity[:max_peaks])
        meta.append(
            (
                precursor,
                int(adduct_vocab.encode(row.get("adduct"))),
                _safe_float(row.get("collision_energy")),
                mass,
                float(row.get("polarity", 1.0)),
            )
        )
        positions.append(position)
    if not kept_mz:
        return None
    n = len(kept_mz)
    width = max(int(m.shape[0]) for m in kept_mz)
    mz_arr = np.zeros((n, width), dtype=np.float32)
    int_arr = np.zeros((n, width), dtype=np.float32)
    mask = np.zeros((n, width), dtype=bool)
    for i, (m, it) in enumerate(zip(kept_mz, kept_int)):
        mz_arr[i, : m.shape[0]] = m
        int_arr[i, : it.shape[0]] = it
        mask[i, : m.shape[0]] = True
    ce = np.array([m[2] for m in meta], dtype=np.float32)
    ce_mask = np.isfinite(ce)
    return SpectraTensors(
        mz=torch.from_numpy(mz_arr).to(device),
        intensity=torch.from_numpy(int_arr).to(device),
        spectrum_mask=torch.from_numpy(mask).to(device),
        precursor_mz=torch.tensor([m[0] for m in meta], dtype=torch.float32, device=device),
        adduct_index=torch.tensor([m[1] for m in meta], dtype=torch.long, device=device),
        collision_energy=torch.from_numpy(np.where(ce_mask, ce, 0.0).astype(np.float32)).to(device),
        collision_energy_mask=torch.from_numpy(ce_mask).to(device),
        neutral_mass=torch.tensor([m[3] for m in meta], dtype=torch.float32, device=device),
        polarity=torch.tensor([m[4] for m in meta], dtype=torch.float32, device=device),
        row_positions=np.asarray(positions, dtype=np.int64),
    )


__all__ = [
    "COLUMN_ALIASES",
    "CasmiBatch",
    "CasmiSpectrumDataset",
    "DatasetFilterStats",
    "PeakPreprocessConfig",
    "REQUIRED_SPECTRUM_COLUMNS",
    "STRUCTURE_COLUMNS",
    "SpectraTensors",
    "add_derived_columns",
    "build_structure_table",
    "group_by_molecule",
    "kekulized_adjacency",
    "load_casmi_parquet",
    "make_casmi_collator",
    "merge_structure_tables",
    "normalise_columns",
    "prepare_spectra_frame",
    "preprocess_peaks",
    "spectra_to_tensors",
    "structure_record",
]
