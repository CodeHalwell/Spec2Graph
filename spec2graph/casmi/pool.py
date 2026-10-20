"""Mass-sorted candidate structures with precomputed spectral features; pickle-free persistence."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from spec2graph.casmi import chem
from spec2graph.casmi.spectral import N_QUANTILES, features
from spectral_diffusion import SpectralDataProcessor

VERSION = 1


def _pack(values: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    blobs = [str(v).encode() for v in values]
    return np.frombuffer(b"".join(blobs), dtype=np.uint8).copy(), np.cumsum([0, *map(len, blobs)], dtype=np.int64)


def _unpack(blob: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    raw = blob.tobytes()
    return np.array([raw[a:b].decode() for a, b in zip(offsets[:-1], offsets[1:])], dtype=object)


@dataclass
class CandidatePool:
    """Structures sorted by exact mass; ragged eigenvectors and elements stored flat with offsets."""

    smiles: np.ndarray
    inchikey14: np.ndarray
    formula: np.ndarray
    exact_mass: np.ndarray
    n_atoms: np.ndarray
    quantiles: np.ndarray
    fingerprints_packed: np.ndarray
    eigvecs_flat: np.ndarray
    elements_flat: np.ndarray
    offsets: np.ndarray
    k: int
    fingerprint_bits: int

    STRINGS = ("smiles", "inchikey14", "formula")
    ARRAYS = ("exact_mass", "n_atoms", "quantiles", "fingerprints_packed", "eigvecs_flat", "elements_flat", "offsets")

    def __post_init__(self) -> None:
        if np.any(np.diff(self.exact_mass) < 0):
            raise ValueError("exact_mass must be sorted ascending")
        self._key_index = {key: i for i, key in enumerate(self.inchikey14)}

    @classmethod
    def build(cls, table: pd.DataFrame, k: int = 8, fingerprint_bits: int = 2048, n_quantiles: int = N_QUANTILES,
              n_jobs: int = 1, progress: bool = False) -> "CandidatePool":
        """Featurise a structure table (``smiles``, ``inchikey14``, ``formula``, ``exact_mass``); unparseable rows are dropped."""
        if table["inchikey14"].duplicated().any():
            raise ValueError("inchikey14 must be unique; merge tables first")
        table = table.sort_values("exact_mass", kind="stable").reset_index(drop=True)
        fn = partial(features, processor=SpectralDataProcessor(k=k), n_quantiles=n_quantiles, bits=fingerprint_bits)
        feats = chem.parallel_map(fn, table["smiles"].astype(str), n_jobs, progress, "pool")
        keep = [i for i, f in enumerate(feats) if f is not None]
        if not keep:
            raise ValueError("no structure could be featurised")
        feats, table = [feats[i] for i in keep], table.iloc[keep]
        n_atoms = np.array([f["n_atoms"] for f in feats], dtype=np.int32)
        col = lambda name: table[name].astype(str).to_numpy(dtype=object)
        return cls(col("smiles"), col("inchikey14"), col("formula"), table["exact_mass"].to_numpy(np.float64), n_atoms,
                   np.stack([f["quantiles"] for f in feats]), np.packbits(np.stack([f["fingerprint"] for f in feats]), axis=1),
                   np.concatenate([f["eigvecs"] for f in feats]).astype(np.float16),
                   np.concatenate([f["elements"] for f in feats]).astype(np.int8), np.cumsum([0, *n_atoms], dtype=np.int64), k, fingerprint_bits)

    def __len__(self) -> int:
        return len(self.exact_mass)

    def eigvecs(self, i: int) -> np.ndarray:
        return self.eigvecs_flat[self.offsets[i] : self.offsets[i + 1]].astype(np.float32)

    def elements(self, i: int) -> np.ndarray:
        return self.elements_flat[self.offsets[i] : self.offsets[i + 1]].astype(np.int64)

    def fingerprints(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return np.zeros((0, self.fingerprint_bits), np.float32)
        return np.unpackbits(self.fingerprints_packed[idx], axis=1)[:, : self.fingerprint_bits].astype(np.float32)

    def index_of(self, inchikey14: str) -> Optional[int]:
        return self._key_index.get(inchikey14)

    def query_mass(self, mass: float, ppm: float = 10.0, min_abs_da: float = 0.0, max_candidates: Optional[int] = None) -> np.ndarray:
        """Indices within ``ppm`` (floored at ``min_abs_da``) of ``mass``; the closest ones when capped."""
        if not np.isfinite(mass) or mass <= 0:
            return np.zeros(0, np.int64)
        tol = max(mass * ppm * 1e-6, min_abs_da)
        idx = np.arange(np.searchsorted(self.exact_mass, mass - tol, "left"), np.searchsorted(self.exact_mass, mass + tol, "right"), dtype=np.int64)
        if max_candidates is not None and idx.size > max_candidates:
            idx = np.sort(idx[np.argsort(np.abs(self.exact_mass[idx] - mass), kind="stable")[:max_candidates]])
        return idx

    def ppm_error(self, mass: float, idx: np.ndarray) -> np.ndarray:
        return (self.exact_mass[idx] - mass) / mass * 1e6

    def formulas_in_window(self, mass: float, ppm: float = 10.0) -> list[str]:
        """Distinct formulas in the window, closest in mass first."""
        idx = self.query_mass(mass, ppm)
        return list(dict.fromkeys(self.formula[idx[np.argsort(np.abs(self.exact_mass[idx] - mass), kind="stable")]]))

    def subset(self, idx) -> "CandidatePool":
        idx = np.sort(np.asarray(idx, dtype=np.int64))
        flat = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in idx]) if idx.size else np.zeros(0, np.int64)
        return CandidatePool(self.smiles[idx], self.inchikey14[idx], self.formula[idx], self.exact_mass[idx], self.n_atoms[idx],
                             self.quantiles[idx], self.fingerprints_packed[idx], self.eigvecs_flat[flat], self.elements_flat[flat],
                             np.cumsum([0, *self.n_atoms[idx]], dtype=np.int64), self.k, self.fingerprint_bits)

    def without(self, keys: Iterable[str]) -> "CandidatePool":
        """Pool with the given InChIKey14s removed (simulates novel molecules)."""
        drop = set(keys)
        return self.subset([i for i, key in enumerate(self.inchikey14) if key not in drop])

    def save(self, path: str | Path) -> None:
        arrays = {name: getattr(self, name) for name in self.ARRAYS}
        for name in self.STRINGS:
            arrays[f"{name}_blob"], arrays[f"{name}_off"] = _pack(getattr(self, name))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, version=VERSION, k=self.k, fingerprint_bits=self.fingerprint_bits, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "CandidatePool":
        with np.load(path, allow_pickle=False) as d:
            if int(d["version"]) != VERSION:
                raise ValueError("unsupported pool version")
            strings = [_unpack(d[f"{n}_blob"], d[f"{n}_off"]) for n in cls.STRINGS]
            return cls(*strings, *(d[n] for n in cls.ARRAYS), int(d["k"]), int(d["fingerprint_bits"]))
