"""Permutation-invariant spectral descriptors and the per-structure feature set the pool stores."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from spec2graph.casmi import chem
from spec2graph.data.cache import EigenvectorCache
from spec2graph.data.elements import atom_types_to_indices
from spectral_diffusion import SpectralDataProcessor

N_QUANTILES = 32


def eigenvalues(adjacency: np.ndarray, processor: SpectralDataProcessor) -> np.ndarray:
    """Ascending normalised-Laplacian spectrum clipped to [0, 2] (λ_max = 2 iff the graph is bipartite)."""
    return np.clip(np.linalg.eigvalsh(processor.compute_laplacian(adjacency.astype(np.float64))), 0, 2).astype(np.float32)


def quantiles(spectrum: np.ndarray, n: int = N_QUANTILES) -> np.ndarray:
    """Spectrum resampled at ``n`` quantile levels: invariant to atom order and molecule size."""
    if n < 2 or np.size(spectrum) == 0:
        raise ValueError("need n >= 2 quantiles of a non-empty spectrum")
    return np.quantile(np.asarray(spectrum, dtype=np.float64), np.linspace(0, 1, n)).astype(np.float32)


def quantile_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean absolute difference of quantile vectors (a 1-D Wasserstein distance); broadcasts over ``b``."""
    return np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean(axis=-1)


def features(smiles: str, processor: SpectralDataProcessor, n_quantiles: int = N_QUANTILES, bits: int = 2048) -> Optional[dict]:
    """Eigenvectors, spectrum quantiles, element indices and fingerprint of one structure (RDKit atom order)."""
    if chem.mol(smiles) is None:
        return None
    adjacency = processor.smiles_to_adjacency(smiles)
    laplacian = processor.compute_laplacian(adjacency)
    return dict(
        eigvecs=processor.extract_eigenvectors(laplacian).astype(np.float32),
        quantiles=quantiles(np.clip(np.linalg.eigvalsh(laplacian), 0, 2), n_quantiles),
        elements=np.asarray(atom_types_to_indices(chem.symbols(smiles)), dtype=np.int8),
        fingerprint=processor.smiles_to_fingerprint(smiles, n_bits=bits).astype(np.uint8),
        n_atoms=adjacency.shape[0],
    )


class SpectralCache(EigenvectorCache):
    """Eigenvector/fingerprint cache plus the Laplacian spectrum per InChIKey."""

    def spectrum_path(self, inchikey: str) -> Path:
        return self.cache_dir / f"laplacian_spectrum_{self.bond_weighting}" / self._shard(inchikey) / f"{inchikey}.npy"

    def spectrum(self, smiles: str, inchikey: str) -> Optional[np.ndarray]:
        path = self.spectrum_path(inchikey)
        if (cached := self._load_npy(path)) is not None:
            return cached
        if chem.mol(smiles) is None:
            return None
        spectrum = eigenvalues(self._processor.smiles_to_adjacency(smiles), self._processor)
        self._atomic_save(spectrum, path)
        return spectrum
