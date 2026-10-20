"""Tests for :mod:`spec2graph.casmi.spectral`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")

from spec2graph.casmi.spectral import (
    SpectralCache,
    laplacian_eigenvalues,
    quantile_distance,
    smiles_to_spectral_features,
    spectral_quantiles,
)
from spectral_diffusion import SpectralDataProcessor


def _adjacency(smiles: str) -> np.ndarray:
    return SpectralDataProcessor(k=4).smiles_to_adjacency(smiles)


class TestEigenvalues:
    def test_range_and_bipartite_signature(self):
        benzene = laplacian_eigenvalues(_adjacency("c1ccccc1"))
        assert benzene.shape == (6,)
        assert benzene[0] == pytest.approx(0.0, abs=1e-6)
        assert np.all(benzene >= 0.0) and np.all(benzene <= 2.0)
        # Even ring: bipartite, so the largest eigenvalue is exactly 2.
        assert benzene[-1] == pytest.approx(2.0, abs=1e-6)
        # Odd ring: not bipartite, so it is strictly below 2.
        cyclopentane = laplacian_eigenvalues(_adjacency("C1CCCC1"))
        assert cyclopentane[-1] < 2.0 - 1e-3

    def test_permutation_invariance(self):
        adj = _adjacency("CC(C)C(=O)O")
        perm = np.random.default_rng(0).permutation(adj.shape[0])
        permuted = adj[np.ix_(perm, perm)]
        np.testing.assert_allclose(laplacian_eigenvalues(adj), laplacian_eigenvalues(permuted), atol=1e-6)


class TestQuantiles:
    def test_shape_monotone_and_size_invariant(self):
        q_small = spectral_quantiles(laplacian_eigenvalues(_adjacency("CCO")), n_quantiles=16)
        q_large = spectral_quantiles(laplacian_eigenvalues(_adjacency("Cn1cnc2c1c(=O)n(C)c(=O)n2C")), 16)
        assert q_small.shape == q_large.shape == (16,)
        assert np.all(np.diff(q_small) >= -1e-6)
        assert q_small.dtype == np.float32

    def test_single_atom(self):
        q = spectral_quantiles(np.array([0.0]), n_quantiles=4)
        np.testing.assert_allclose(q, 0.0)

    def test_validation(self):
        with pytest.raises(ValueError):
            spectral_quantiles(np.array([0.0, 1.0]), n_quantiles=1)
        with pytest.raises(ValueError):
            spectral_quantiles(np.array([]), n_quantiles=4)

    def test_distance_broadcasts(self):
        a = np.zeros(8, dtype=np.float32)
        b = np.stack([np.zeros(8), np.ones(8), np.full(8, 0.5)]).astype(np.float32)
        np.testing.assert_allclose(quantile_distance(a, b), [0.0, 1.0, 0.5])
        assert quantile_distance(a, a).shape == ()


class TestFeatures:
    def test_feature_dict(self):
        processor = SpectralDataProcessor(k=4)
        feats = smiles_to_spectral_features("Oc1ccccc1", processor, n_quantiles=8, fingerprint_bits=64)
        assert feats is not None
        assert feats["n_atoms"] == 7
        assert feats["eigvecs"].shape == (7, 4)
        assert feats["eigenvalues"].shape == (7,)
        assert feats["quantiles"].shape == (8,)
        assert feats["elements"].shape == (7,)
        assert feats["fingerprint"].shape == (64,)
        assert feats["fingerprint"].dtype == np.uint8
        assert set(np.unique(feats["fingerprint"]).tolist()) <= {0, 1}
        # Elements follow RDKit atom order: O first for "Oc1ccccc1".
        assert feats["elements"][0] == 2  # index of "O" in the vocabulary

    def test_invalid(self):
        processor = SpectralDataProcessor(k=4)
        assert smiles_to_spectral_features("not smiles", processor) is None
        assert smiles_to_spectral_features(None, processor) is None  # type: ignore[arg-type]


class TestSpectralCache:
    def test_round_trip(self, tmp_path):
        cache = SpectralCache(tmp_path, k=4, fingerprint_bits=64)
        key = "UHOVQNZJYSORNB-UHFFFAOYSA-N"
        first = cache.get_or_compute_spectrum("c1ccccc1", key)
        assert first is not None and first.shape == (6,)
        assert cache.has_spectrum(key)
        second = cache.get_or_compute_spectrum("c1ccccc1", key)
        np.testing.assert_allclose(first, second)
        q = cache.get_or_compute_quantiles("c1ccccc1", key, n_quantiles=6)
        assert q.shape == (6,)
        assert cache.get_or_compute_spectrum("garbage", "AAAAAAAAAAAAAA-UHFFFAOYSA-N") is None
        # The base-class artefacts still work through the subclass.
        assert cache.get_or_compute("c1ccccc1", key).shape == (6, 4)
