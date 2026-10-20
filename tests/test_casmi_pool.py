"""Tests for :mod:`spec2graph.casmi.pool`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")

from spec2graph.casmi.data import build_structure_table, structure_record
from spec2graph.casmi.pool import CandidatePool
from spec2graph.casmi.spectral import smiles_to_spectral_features
from spectral_diffusion import SpectralDataProcessor
from tests.conftest import CASMI_MOLECULES


@pytest.fixture
def pool() -> CandidatePool:
    table = build_structure_table([s for _, s in CASMI_MOLECULES] + ["garbage"], source="unit")
    return CandidatePool.build(table, k=4, fingerprint_bits=64, n_quantiles=8, source_column="source")


class TestBuildAndQuery:
    def test_sorted_and_sized(self, pool):
        assert len(pool) == len(CASMI_MOLECULES)
        assert np.all(np.diff(pool.exact_mass) >= 0)
        assert pool.quantiles.shape == (len(pool), 8)
        assert pool.fingerprints_packed.shape == (len(pool), 8)
        assert pool.offsets[-1] == pool.n_atoms.sum()
        assert pool.source is not None and set(pool.source) == {"unit"}

    def test_mass_window(self, pool):
        ethanol = structure_record("CCO")
        hits = pool.query_mass(ethanol["exact_mass"], ppm=10.0)
        assert hits.size == 1
        assert pool.inchikey14[hits[0]] == ethanol["inchikey14"]
        assert pool.query_mass(ethanol["exact_mass"] * 1.01, ppm=10.0).size == 0
        assert pool.query_mass(float("nan"), ppm=10.0).size == 0
        # A loose window plus max_candidates keeps the closest masses.
        wide = pool.query_mass(ethanol["exact_mass"], ppm=1e6, max_candidates=3)
        assert wide.size == 3
        assert np.all(np.diff(wide) > 0)
        assert np.abs(pool.mass_error_ppm(ethanol["exact_mass"], hits)[0]) < 1e-3

    def test_abs_floor(self, pool):
        ethanol = structure_record("CCO")
        assert pool.query_mass(ethanol["exact_mass"] + 0.001, ppm=1.0).size == 0
        assert pool.query_mass(ethanol["exact_mass"] + 0.001, ppm=1.0, min_abs_da=0.005).size == 1

    def test_formula_lookup(self, pool):
        # Phenol and toluene share no formula; benzene has C6H6.
        idx = pool.query_formula("C6H6")
        assert idx.size == 1 and pool.smiles[idx[0]] == "c1ccccc1"
        assert pool.query_formula("Xx").size == 0
        benzene_mass = structure_record("c1ccccc1")["exact_mass"]
        assert pool.formulas_in_window(benzene_mass, ppm=10.0) == ["C6H6"]

    def test_feature_accessors(self, pool):
        processor = SpectralDataProcessor(k=4)
        for i in range(len(pool)):
            smiles = pool.smiles[i]
            direct = smiles_to_spectral_features(smiles, processor, n_quantiles=8, fingerprint_bits=64)
            np.testing.assert_allclose(pool.eigvecs(i), direct["eigvecs"], atol=2e-3)
            np.testing.assert_array_equal(pool.elements(i), direct["elements"])
            np.testing.assert_array_equal(pool.fingerprints([i])[0], direct["fingerprint"])
            assert pool.eigvecs(i).shape == (pool.n_atoms[i], 4)
        assert pool.fingerprints([]).shape == (0, 64)

    def test_key_index(self, pool):
        key = structure_record("Cc1ccccc1")["inchikey14"]
        assert pool.contains(key)
        assert pool.smiles[pool.index_of(key)] == "Cc1ccccc1"
        assert not pool.contains("AAAAAAAAAAAAAA")


class TestSubsetAndPersistence:
    def test_without_keys(self, pool):
        key = structure_record("CCO")["inchikey14"]
        smaller = pool.without_keys([key])
        assert len(smaller) == len(pool) - 1
        assert not smaller.contains(key)
        # Ragged storage stays consistent after subsetting.
        for i in range(len(smaller)):
            assert smaller.eigvecs(i).shape == (smaller.n_atoms[i], 4)
            j = pool.index_of(smaller.inchikey14[i])
            np.testing.assert_array_equal(smaller.eigvecs(i), pool.eigvecs(j))
            np.testing.assert_array_equal(smaller.fingerprints([i]), pool.fingerprints([j]))

    def test_save_load_round_trip(self, pool, tmp_path):
        path = tmp_path / "pool.npz"
        pool.save(path)
        loaded = CandidatePool.load(path)
        assert len(loaded) == len(pool)
        np.testing.assert_array_equal(loaded.smiles, pool.smiles)
        np.testing.assert_array_equal(loaded.inchikey14, pool.inchikey14)
        np.testing.assert_array_equal(loaded.formula, pool.formula)
        np.testing.assert_array_equal(loaded.exact_mass, pool.exact_mass)
        np.testing.assert_array_equal(loaded.eigvecs_flat, pool.eigvecs_flat)
        np.testing.assert_array_equal(loaded.elements_flat, pool.elements_flat)
        np.testing.assert_array_equal(loaded.fingerprints_packed, pool.fingerprints_packed)
        np.testing.assert_array_equal(loaded.source, pool.source)
        assert loaded.k == 4 and loaded.fingerprint_bits == 64

    def test_build_validation(self):
        table = build_structure_table(["CCO", "OCC"])
        bad = table.copy()
        bad = __import__("pandas").concat([bad, bad], ignore_index=True)
        with pytest.raises(ValueError, match="unique"):
            CandidatePool.build(bad, k=4, fingerprint_bits=64)
        with pytest.raises(ValueError, match="missing"):
            CandidatePool.build(table.drop(columns=["formula"]), k=4, fingerprint_bits=64)
