"""Tests for :mod:`spec2graph.casmi.pool`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")

from spec2graph.casmi import chem, data, spectral
from spec2graph.casmi.pool import CandidatePool
from spectral_diffusion import SpectralDataProcessor
from tests.conftest import CASMI_MOLECULES, FP_BITS, K, N_Q


@pytest.fixture
def pool() -> CandidatePool:
    return CandidatePool.build(data.structures([s for _, s in CASMI_MOLECULES] + ["garbage"]), K, FP_BITS, N_Q)


def test_build_query_and_accessors(pool):
    assert len(pool) == len(CASMI_MOLECULES) and np.all(np.diff(pool.exact_mass) >= 0) and pool.offsets[-1] == pool.n_atoms.sum()
    ethanol = chem.record("CCO")
    hits = pool.query_mass(ethanol["exact_mass"], 10.0)
    assert hits.tolist() == [pool.index_of(ethanol["inchikey14"])] and abs(pool.ppm_error(ethanol["exact_mass"], hits)[0]) < 1e-3
    assert pool.query_mass(ethanol["exact_mass"] * 1.01, 10.0).size == 0 and pool.query_mass(float("nan")).size == 0
    assert pool.query_mass(ethanol["exact_mass"], 1e6, max_candidates=3).size == 3
    assert pool.query_mass(ethanol["exact_mass"] + 0.001, 1.0).size == 0 and pool.query_mass(ethanol["exact_mass"] + 0.001, 1.0, min_abs_da=0.005).size == 1
    assert pool.formulas_in_window(chem.record("c1ccccc1")["exact_mass"]) == ["C6H6"]
    processor = SpectralDataProcessor(k=K)
    for i in range(len(pool)):
        direct = spectral.features(pool.smiles[i], processor, N_Q, FP_BITS)
        np.testing.assert_allclose(pool.eigvecs(i), direct["eigvecs"], atol=2e-3)
        np.testing.assert_array_equal(pool.elements(i), direct["elements"])
        np.testing.assert_array_equal(pool.fingerprints([i])[0], direct["fingerprint"])
    assert pool.fingerprints([]).shape == (0, FP_BITS) and pool.index_of("AAAAAAAAAAAAAA") is None


def test_subset_and_persistence(pool, tmp_path):
    key = chem.record("CCO")["inchikey14"]
    smaller = pool.without([key])
    assert len(smaller) == len(pool) - 1 and smaller.index_of(key) is None
    for i in range(len(smaller)):
        j = pool.index_of(smaller.inchikey14[i])
        np.testing.assert_array_equal(smaller.eigvecs(i), pool.eigvecs(j))
        np.testing.assert_array_equal(smaller.fingerprints([i]), pool.fingerprints([j]))
    pool.save(tmp_path / "pool.npz")
    loaded = CandidatePool.load(tmp_path / "pool.npz")
    for name in CandidatePool.STRINGS + CandidatePool.ARRAYS:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(pool, name))
    assert (loaded.k, loaded.fingerprint_bits) == (K, FP_BITS)


def test_build_validation():
    table = data.structures(["CCO", "OCC"])
    with pytest.raises(ValueError, match="unique"):
        CandidatePool.build(pd.concat([table, table]), K, FP_BITS)
    with pytest.raises(ValueError, match="sorted"):
        CandidatePool(*(np.array([]) for _ in range(3)), np.array([2.0, 1.0]), *(np.array([]) for _ in range(6)), K, FP_BITS)
