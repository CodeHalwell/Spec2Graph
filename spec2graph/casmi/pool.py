"""Candidate structure pool with mass-window retrieval.

Nearly every answer in the competition exists in some structure database
(the training structures, COCONUT, PubChem). The pool holds, per
structure, everything the scorer needs so that scoring a mass window never
touches RDKit at prediction time:

* identity: representative SMILES, InChIKey14, formula, exact mass;
* spectral features: eigenvectors ``V_k`` (ragged, float16), element
  indices in the same atom order, spectral quantiles;
* a packed Morgan fingerprint.

Retrieval is a ``searchsorted`` on the mass-sorted array, so a 10 ppm
window over a million structures costs microseconds. Pools are saved with
``numpy.savez`` without pickling: strings are packed into a UTF-8 byte
blob with offsets, which keeps :func:`CandidatePool.load` safe to call on
a file that came from a Kaggle dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from spec2graph.casmi.spectral import DEFAULT_N_QUANTILES, smiles_to_spectral_features
from spectral_diffusion import SpectralDataProcessor

logger = logging.getLogger(__name__)

POOL_FORMAT_VERSION = 1


def _pack_strings(values: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    encoded = [str(v).encode("utf-8") for v in values]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    if encoded:
        offsets[1:] = np.cumsum([len(e) for e in encoded])
    blob = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
    return blob, offsets


def _unpack_strings(blob: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    raw = blob.tobytes()
    out = np.empty(len(offsets) - 1, dtype=object)
    for i in range(len(offsets) - 1):
        out[i] = raw[offsets[i] : offsets[i + 1]].decode("utf-8")
    return out


@dataclass
class CandidatePool:
    """Mass-sorted structure pool. Build with :meth:`build`, query with :meth:`query_mass`."""

    smiles: np.ndarray
    inchikey14: np.ndarray
    formula: np.ndarray
    exact_mass: np.ndarray
    n_atoms: np.ndarray
    quantiles: np.ndarray
    fingerprints_packed: np.ndarray
    fingerprint_bits: int
    eigvecs_flat: np.ndarray
    elements_flat: np.ndarray
    offsets: np.ndarray
    k: int
    source: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self._key_index: Optional[dict[str, int]] = None
        self._formula_index: Optional[dict[str, np.ndarray]] = None
        if not np.all(np.diff(self.exact_mass) >= 0):
            raise ValueError("CandidatePool.exact_mass must be sorted ascending.")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        structures: pd.DataFrame,
        *,
        k: int = 8,
        fingerprint_bits: int = 2048,
        n_quantiles: int = DEFAULT_N_QUANTILES,
        bond_weighting: str = "order",
        n_jobs: int = 1,
        show_progress: bool = False,
        source_column: Optional[str] = None,
    ) -> "CandidatePool":
        """Compute features for every row of a structure table.

        ``structures`` needs ``smiles``, ``inchikey14``, ``formula`` and
        ``exact_mass`` (see :func:`spec2graph.casmi.data.build_structure_table`).
        Rows RDKit rejects are dropped with a log line.
        """
        required = ("smiles", "inchikey14", "formula", "exact_mass")
        missing = [c for c in required if c not in structures.columns]
        if missing:
            raise ValueError(f"structures is missing columns {missing}.")
        if structures["inchikey14"].duplicated().any():
            raise ValueError("structures must have unique inchikey14 rows; merge tables first.")

        processor = SpectralDataProcessor(k=k, bond_weighting=bond_weighting)
        smiles_list = structures["smiles"].astype(str).tolist()

        iterator: Iterable[str] = smiles_list
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(smiles_list, desc="pool", unit="mol")
            except ImportError:
                pass

        def _work(smi: str):
            return smiles_to_spectral_features(
                smi, processor, n_quantiles=n_quantiles, fingerprint_bits=fingerprint_bits
            )

        if n_jobs == 1:
            feats = [_work(s) for s in iterator]
        else:
            try:
                from joblib import Parallel, delayed

                feats = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=256)(
                    delayed(_work)(s) for s in iterator
                )
            except ImportError:
                logger.warning("joblib not installed; building the pool serially.")
                feats = [_work(s) for s in smiles_list]

        keep = [i for i, f in enumerate(feats) if f is not None]
        if len(keep) < len(feats):
            logger.info("CandidatePool.build: dropped %d unparseable structures.", len(feats) - len(keep))
        if not keep:
            raise ValueError("No structures could be featurised.")

        masses = structures["exact_mass"].to_numpy(dtype=np.float64)[keep]
        order = np.argsort(masses, kind="stable")
        keep = [keep[i] for i in order]
        feats = [feats[i] for i in keep]

        n_atoms = np.array([f["n_atoms"] for f in feats], dtype=np.int32)
        offsets = np.zeros(len(feats) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(n_atoms)
        eigvecs_flat = np.concatenate([f["eigvecs"] for f in feats]).astype(np.float16)
        elements_flat = np.concatenate([f["elements"] for f in feats]).astype(np.int8)
        fingerprints = np.stack([f["fingerprint"] for f in feats]).astype(np.uint8)
        quantiles = np.stack([f["quantiles"] for f in feats]).astype(np.float32)

        sub = structures.iloc[keep]
        source = None
        if source_column is not None and source_column in sub.columns:
            source = sub[source_column].astype(str).to_numpy(dtype=object)

        return cls(
            smiles=sub["smiles"].astype(str).to_numpy(dtype=object),
            inchikey14=sub["inchikey14"].astype(str).to_numpy(dtype=object),
            formula=sub["formula"].astype(str).to_numpy(dtype=object),
            exact_mass=sub["exact_mass"].to_numpy(dtype=np.float64),
            n_atoms=n_atoms,
            quantiles=quantiles,
            fingerprints_packed=np.packbits(fingerprints, axis=1),
            fingerprint_bits=fingerprint_bits,
            eigvecs_flat=eigvecs_flat,
            elements_flat=elements_flat,
            offsets=offsets,
            k=k,
            source=source,
        )

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.exact_mass.shape[0])

    def eigvecs(self, index: int) -> np.ndarray:
        """``(n_atoms, k)`` float32 eigenvectors of structure ``index``."""
        start, stop = self.offsets[index], self.offsets[index + 1]
        return self.eigvecs_flat[start:stop].astype(np.float32)

    def elements(self, index: int) -> np.ndarray:
        start, stop = self.offsets[index], self.offsets[index + 1]
        return self.elements_flat[start:stop].astype(np.int64)

    def fingerprints(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """``(m, fingerprint_bits)`` float32 unpacked fingerprints."""
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            return np.zeros((0, self.fingerprint_bits), dtype=np.float32)
        bits = np.unpackbits(self.fingerprints_packed[idx], axis=1)[:, : self.fingerprint_bits]
        return bits.astype(np.float32)

    def index_of(self, inchikey14: str) -> Optional[int]:
        if self._key_index is None:
            self._key_index = {key: i for i, key in enumerate(self.inchikey14)}
        return self._key_index.get(inchikey14)

    def contains(self, inchikey14: str) -> bool:
        return self.index_of(inchikey14) is not None

    def query_mass(
        self,
        neutral_mass: float,
        ppm: float = 10.0,
        *,
        min_abs_da: float = 0.0,
        max_candidates: Optional[int] = None,
    ) -> np.ndarray:
        """Indices of structures within ``ppm`` of ``neutral_mass``.

        ``min_abs_da`` floors the tolerance in daltons (useful at low mass).
        With ``max_candidates`` the closest-in-mass structures are kept.
        """
        if not np.isfinite(neutral_mass) or neutral_mass <= 0:
            return np.zeros(0, dtype=np.int64)
        tolerance = max(float(neutral_mass) * ppm * 1e-6, min_abs_da)
        lo = np.searchsorted(self.exact_mass, neutral_mass - tolerance, side="left")
        hi = np.searchsorted(self.exact_mass, neutral_mass + tolerance, side="right")
        indices = np.arange(lo, hi, dtype=np.int64)
        if max_candidates is not None and indices.size > max_candidates:
            errors = np.abs(self.exact_mass[indices] - neutral_mass)
            indices = indices[np.argsort(errors, kind="stable")[:max_candidates]]
            indices.sort()
        return indices

    def query_formula(self, formula: str) -> np.ndarray:
        if self._formula_index is None:
            groups: dict[str, list[int]] = {}
            for i, f in enumerate(self.formula):
                groups.setdefault(str(f), []).append(i)
            self._formula_index = {f: np.asarray(v, dtype=np.int64) for f, v in groups.items()}
        return self._formula_index.get(formula, np.zeros(0, dtype=np.int64))

    def mass_error_ppm(self, neutral_mass: float, indices: np.ndarray) -> np.ndarray:
        return (self.exact_mass[indices] - neutral_mass) / neutral_mass * 1e6

    def formulas_in_window(self, neutral_mass: float, ppm: float = 10.0) -> list[str]:
        """Distinct formulas in the mass window, closest in mass first."""
        indices = self.query_mass(neutral_mass, ppm)
        if indices.size == 0:
            return []
        errors = np.abs(self.exact_mass[indices] - neutral_mass)
        seen: dict[str, float] = {}
        for i, err in zip(indices, errors):
            f = str(self.formula[i])
            if f not in seen or err < seen[f]:
                seen[f] = float(err)
        return sorted(seen, key=seen.get)

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------

    def subset(self, indices: Sequence[int] | np.ndarray) -> "CandidatePool":
        """Pool restricted to ``indices`` (mass order preserved)."""
        idx = np.sort(np.asarray(indices, dtype=np.int64))
        n_atoms = self.n_atoms[idx]
        offsets = np.zeros(idx.size + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(n_atoms)
        ranges = [np.arange(self.offsets[i], self.offsets[i + 1]) for i in idx]
        flat = np.concatenate(ranges) if ranges else np.zeros(0, dtype=np.int64)
        return CandidatePool(
            smiles=self.smiles[idx],
            inchikey14=self.inchikey14[idx],
            formula=self.formula[idx],
            exact_mass=self.exact_mass[idx],
            n_atoms=n_atoms,
            quantiles=self.quantiles[idx],
            fingerprints_packed=self.fingerprints_packed[idx],
            fingerprint_bits=self.fingerprint_bits,
            eigvecs_flat=self.eigvecs_flat[flat],
            elements_flat=self.elements_flat[flat],
            offsets=offsets,
            k=self.k,
            source=None if self.source is None else self.source[idx],
        )

    def without_keys(self, inchikey14s: Iterable[str]) -> "CandidatePool":
        """Pool with the given structures removed (simulates novel molecules)."""
        drop = set(inchikey14s)
        keep = np.array([i for i, key in enumerate(self.inchikey14) if key not in drop], dtype=np.int64)
        return self.subset(keep)

    # ------------------------------------------------------------------
    # Persistence (no pickle)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        smiles_blob, smiles_off = _pack_strings(self.smiles)
        key_blob, key_off = _pack_strings(self.inchikey14)
        formula_blob, formula_off = _pack_strings(self.formula)
        arrays = dict(
            version=np.array([POOL_FORMAT_VERSION]),
            k=np.array([self.k]),
            fingerprint_bits=np.array([self.fingerprint_bits]),
            smiles_blob=smiles_blob,
            smiles_off=smiles_off,
            key_blob=key_blob,
            key_off=key_off,
            formula_blob=formula_blob,
            formula_off=formula_off,
            exact_mass=self.exact_mass,
            n_atoms=self.n_atoms,
            quantiles=self.quantiles,
            fingerprints_packed=self.fingerprints_packed,
            eigvecs_flat=self.eigvecs_flat,
            elements_flat=self.elements_flat,
            offsets=self.offsets,
        )
        if self.source is not None:
            arrays["source_blob"], arrays["source_off"] = _pack_strings(self.source)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "CandidatePool":
        with np.load(path, allow_pickle=False) as data:
            version = int(data["version"][0])
            if version != POOL_FORMAT_VERSION:
                raise ValueError(f"Unsupported pool format version {version}.")
            source = None
            if "source_blob" in data:
                source = _unpack_strings(data["source_blob"], data["source_off"])
            return cls(
                smiles=_unpack_strings(data["smiles_blob"], data["smiles_off"]),
                inchikey14=_unpack_strings(data["key_blob"], data["key_off"]),
                formula=_unpack_strings(data["formula_blob"], data["formula_off"]),
                exact_mass=data["exact_mass"],
                n_atoms=data["n_atoms"],
                quantiles=data["quantiles"],
                fingerprints_packed=data["fingerprints_packed"],
                fingerprint_bits=int(data["fingerprint_bits"][0]),
                eigvecs_flat=data["eigvecs_flat"],
                elements_flat=data["elements_flat"],
                offsets=data["offsets"],
                k=int(data["k"][0]),
                source=source,
            )


__all__ = ["CandidatePool", "POOL_FORMAT_VERSION"]
