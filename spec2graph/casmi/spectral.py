"""Permutation-invariant spectral descriptors of molecular graphs.

The core model predicts the subspace spanned by the first ``k`` eigenvectors
of a molecule's normalised Laplacian. Comparing that prediction with a
*candidate* structure needs an atom correspondence, which is the expensive
part (see :mod:`spec2graph.casmi.scoring`). The descriptors here need no
correspondence at all:

* :func:`laplacian_eigenvalues` — the full spectrum ``0 = λ_1 <= ... <= λ_N <= 2``.
* :func:`spectral_quantiles` — the spectrum resampled at fixed quantile
  levels, so molecules of different size map to vectors of one length.
  The distance between two quantile vectors is a one-dimensional
  Wasserstein distance between the spectral densities.

Both are invariant to atom ordering by construction, cheap enough to
precompute for a pool of a million structures, and informative about global
topology (for a connected graph ``λ_N = 2`` iff it is bipartite, i.e. it
has no odd rings; the multiplicity of ``λ = 1`` tracks pendant groups).

:class:`SpectralCache` extends the on-disk :class:`EigenvectorCache` with a
cached eigenvalue spectrum per InChIKey, reusing its sharded layout and
atomic writes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from spec2graph.data.cache import EigenvectorCache
from spec2graph.data.elements import atom_types_to_indices
from spectral_diffusion import MAX_SMILES_LENGTH, SpectralDataProcessor

logger = logging.getLogger(__name__)

DEFAULT_N_QUANTILES: int = 32
# Eigenvalues of a normalised Laplacian live in [0, 2].
SPECTRUM_MAX: float = 2.0


def laplacian_eigenvalues(adjacency: np.ndarray, processor: Optional[SpectralDataProcessor] = None) -> np.ndarray:
    """Ascending eigenvalues of the normalised Laplacian of ``adjacency``."""
    processor = processor or SpectralDataProcessor()
    laplacian = processor.compute_laplacian(np.asarray(adjacency, dtype=np.float64))
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return np.clip(np.sort(eigenvalues), 0.0, SPECTRUM_MAX).astype(np.float32)


def spectral_quantiles(eigenvalues: np.ndarray, n_quantiles: int = DEFAULT_N_QUANTILES) -> np.ndarray:
    """Resample a spectrum at ``n_quantiles`` evenly spaced quantile levels.

    A single-atom graph has the one eigenvalue ``0``; the quantile vector is
    then constant, which is the right answer rather than an error.
    """
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2; got {n_quantiles}.")
    values = np.asarray(eigenvalues, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("Cannot take quantiles of an empty spectrum.")
    levels = np.linspace(0.0, 1.0, n_quantiles)
    return np.quantile(np.sort(values), levels).astype(np.float32)


def quantile_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean absolute difference between quantile vectors (1-D Wasserstein).

    Broadcasts: ``a`` may be ``(n_q,)`` and ``b`` ``(M, n_q)``.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.abs(a - b).mean(axis=-1)


def smiles_to_spectral_features(
    smiles: str,
    processor: SpectralDataProcessor,
    *,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    fingerprint_bits: int = 2048,
    fingerprint_radius: int = 2,
) -> Optional[dict]:
    """Everything the candidate pool stores per structure, or ``None``.

    Returns ``eigvecs (N, k)``, ``eigenvalues (N,)``, ``quantiles (n_q,)``,
    ``elements`` (vocabulary indices in RDKit atom order, ``(N,)``),
    ``fingerprint (bits,)`` and ``n_atoms``. Atom order is RDKit's parse
    order of the given SMILES, the same order the eigenvectors use.
    """
    if not isinstance(smiles, str) or len(smiles) > MAX_SMILES_LENGTH:
        return None
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rdkit is required for spectral features.") from exc
    try:
        adjacency = processor.smiles_to_adjacency(smiles)
    except (ValueError, TypeError):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    laplacian = processor.compute_laplacian(adjacency)
    eigvecs = processor.extract_eigenvectors(laplacian)
    eigenvalues = np.clip(np.sort(np.linalg.eigvalsh(laplacian)), 0.0, SPECTRUM_MAX).astype(np.float32)
    try:
        fingerprint = processor.smiles_to_fingerprint(
            smiles, n_bits=fingerprint_bits, radius=fingerprint_radius
        )
    except (ValueError, TypeError):
        return None
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return {
        "eigvecs": np.asarray(eigvecs, dtype=np.float32),
        "eigenvalues": eigenvalues,
        "quantiles": spectral_quantiles(eigenvalues, n_quantiles),
        "elements": np.asarray(atom_types_to_indices(symbols), dtype=np.int8),
        "fingerprint": np.asarray(fingerprint, dtype=np.uint8),
        "n_atoms": int(mol.GetNumAtoms()),
    }


class SpectralCache(EigenvectorCache):
    """:class:`EigenvectorCache` plus a cached Laplacian eigenvalue spectrum.

    Layout adds ``{cache_dir}/laplacian_spectrum_{bond_weighting}/{shard}/{inchikey}.npy``.
    """

    def __init__(self, cache_dir: str | Path, k: int, **kwargs) -> None:
        super().__init__(cache_dir, k, **kwargs)
        self.spectrum_root = self.cache_dir / f"laplacian_spectrum_{self.bond_weighting}"

    def spectrum_path(self, inchikey: str) -> Path:
        return self.spectrum_root / self._shard(inchikey) / f"{inchikey}.npy"

    def has_spectrum(self, inchikey: str) -> bool:
        return self.spectrum_path(inchikey).exists()

    def get_or_compute_spectrum(self, smiles: str, inchikey: str) -> Optional[np.ndarray]:
        """Return the cached eigenvalue spectrum, computing it if necessary."""
        if not isinstance(smiles, str) or len(smiles) > MAX_SMILES_LENGTH:
            return None
        path = self.spectrum_path(inchikey)
        cached = self._load_npy(path)
        if cached is not None:
            return cached.astype(np.float32, copy=False)
        try:
            adjacency = self._processor.smiles_to_adjacency(smiles)
        except (ValueError, TypeError, ImportError) as exc:
            logger.warning("Failed to compute spectrum for %s (%s): %s", smiles, inchikey, exc)
            return None
        spectrum = laplacian_eigenvalues(adjacency, self._processor)
        self._atomic_save(spectrum, path)
        return spectrum

    def get_or_compute_quantiles(
        self, smiles: str, inchikey: str, n_quantiles: int = DEFAULT_N_QUANTILES
    ) -> Optional[np.ndarray]:
        spectrum = self.get_or_compute_spectrum(smiles, inchikey)
        if spectrum is None:
            return None
        return spectral_quantiles(spectrum, n_quantiles)


__all__ = [
    "DEFAULT_N_QUANTILES",
    "SPECTRUM_MAX",
    "SpectralCache",
    "laplacian_eigenvalues",
    "quantile_distance",
    "smiles_to_spectral_features",
    "spectral_quantiles",
]
