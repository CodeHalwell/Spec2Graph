"""Tests for :mod:`spec2graph.casmi.data`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")
torch = pytest.importorskip("torch")

from spec2graph.casmi.adducts import PROTON_MASS, AdductVocabulary
from spec2graph.casmi.data import (
    CasmiBatch,
    CasmiSpectrumDataset,
    PeakPreprocessConfig,
    build_structure_table,
    group_by_molecule,
    kekulized_adjacency,
    load_casmi_parquet,
    make_casmi_collator,
    merge_structure_tables,
    normalise_columns,
    prepare_spectra_frame,
    preprocess_peaks,
    structure_record,
)
from tests.conftest import CASMI_MOLECULES


class TestFrame:
    def test_normalise_columns(self, casmi_frame):
        df = normalise_columns(casmi_frame)
        for column in ("molecule_id", "precursor_mz", "adduct", "collision_energy", "mzs", "intensities", "smiles"):
            assert column in df.columns
        assert "instrument" in df.columns and "library" in df.columns
        # Original frame untouched.
        assert "ms2_mzs" in casmi_frame.columns

    def test_prepare_adds_derived_columns(self, casmi_frame):
        df = prepare_spectra_frame(casmi_frame, require_smiles=True)
        row = df[(df["molecule_id"] == "m_ethanol")].iloc[0]
        assert row["adduct_canonical"] == "[M+H]+"
        assert row["neutral_mass"] == pytest.approx(row["precursor_mz"] - PROTON_MASS, abs=1e-5)
        assert row["polarity"] == 1
        bad = df[df["molecule_id"] == "m_bad_adduct"].iloc[0]
        assert pd.isna(bad["adduct_canonical"])
        assert np.isnan(bad["neutral_mass"])
        negative = df[df["adduct"] == "[M-H]-"].iloc[0]
        assert negative["polarity"] == -1

    def test_missing_required_column_raises(self, casmi_frame):
        with pytest.raises(RuntimeError, match="missing"):
            prepare_spectra_frame(casmi_frame.drop(columns=["adduct"]))
        with pytest.raises(RuntimeError, match="smiles"):
            prepare_spectra_frame(casmi_frame.drop(columns=["normalized_smiles"]), require_smiles=True)

    def test_parquet_round_trip(self, casmi_frame, tmp_path):
        path = tmp_path / "train.parquet"
        casmi_frame.to_parquet(path, index=False)
        df = load_casmi_parquet(path, require_smiles=True)
        assert len(df) == len(casmi_frame)
        first = df.iloc[0]
        assert isinstance(first["mzs"], np.ndarray)
        test_path = tmp_path / "test.parquet"
        casmi_frame.drop(columns=["normalized_smiles"]).to_parquet(test_path, index=False)
        test_df = load_casmi_parquet(test_path)
        assert "smiles" not in test_df.columns

    def test_group_by_molecule(self, casmi_frame):
        df = prepare_spectra_frame(casmi_frame)
        groups = group_by_molecule(df)
        assert set(groups) == set(df["molecule_id"])
        assert len(groups["m_ethanol"]) == 2
        assert np.all(df.iloc[groups["m_ethanol"]]["molecule_id"] == "m_ethanol")


class TestPeaks:
    def test_cleaning(self):
        mz = np.array([50.0, 10.0, 199.8, 150.0, 30.0, np.nan])
        intensity = np.array([5.0, 10.0, 3.0, 0.0, 2.0, 1.0])
        out_mz, out_int = preprocess_peaks(mz, intensity, precursor_mz=200.0)
        # 199.8 is within the precursor window, 150 has zero intensity, NaN dropped.
        np.testing.assert_allclose(out_mz, [10.0, 30.0, 50.0])
        np.testing.assert_allclose(out_int, [1.0, 0.2, 0.5])

    def test_top_k_and_sqrt(self):
        mz = np.arange(1.0, 11.0)
        intensity = np.arange(1.0, 11.0)
        config = PeakPreprocessConfig(top_k_peaks=3, sqrt_intensity=True)
        out_mz, out_int = preprocess_peaks(mz, intensity, precursor_mz=100.0, config=config)
        np.testing.assert_allclose(out_mz, [8.0, 9.0, 10.0])
        np.testing.assert_allclose(out_int, np.sqrt([0.8, 0.9, 1.0]))

    def test_empty_and_mismatch(self):
        mz, intensity = preprocess_peaks(np.array([300.0]), np.array([1.0]), precursor_mz=200.0)
        assert mz.size == 0 and intensity.size == 0
        with pytest.raises(ValueError):
            preprocess_peaks(np.array([1.0, 2.0]), np.array([1.0]), precursor_mz=10.0)


class TestStructures:
    def test_structure_record(self):
        record = structure_record("OCC1OC(O)C(O)C(O)C1O")
        assert record["inchikey14"] == "WQZGKKKJIJFFOK"
        assert record["formula"] == "C6H12O6"
        assert record["exact_mass"] == pytest.approx(180.0634, abs=1e-3)
        assert record["n_heavy_atoms"] == 12
        assert record["inchikey"].startswith("WQZGKKKJIJFFOK")
        assert structure_record("garbage") is None
        assert structure_record(None) is None

    def test_canonical_tautomer_representative(self):
        a = structure_record("Oc1ccccn1", canonical_tautomer=True)
        b = structure_record("O=c1cccc[nH]1", canonical_tautomer=True)
        assert a["smiles"] == b["smiles"]
        assert a["inchikey14"] == b["inchikey14"]

    def test_build_and_merge(self):
        smiles = ["CCO", "OCC", "c1ccccc1", "garbage", "Oc1ccccn1", "O=c1cccc[nH]1"]
        table = build_structure_table(smiles, source="unit")
        # Ethanol spelled two ways and the tautomer pair each collapse.
        assert len(table) == 3
        ethanol = table[table["formula"] == "C2H6O"].iloc[0]
        assert ethanol["n_inputs"] == 2
        assert ethanol["input_smiles"] == "CCO"
        assert (table["source"] == "unit").all()
        other = build_structure_table(["CCO", "CCN"])
        merged = merge_structure_tables([table, other])
        assert len(merged) == 4
        assert merged["inchikey14"].is_unique

    def test_kekulized_adjacency(self):
        adj = kekulized_adjacency("c1ccccc1")
        assert adj.shape == (6, 6)
        np.testing.assert_allclose(adj, adj.T)
        orders = sorted(adj[np.triu_indices(6, k=1)][adj[np.triu_indices(6, k=1)] > 0].tolist())
        assert orders == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        assert kekulized_adjacency("garbage").shape == (0, 0)


class TestDataset:
    def test_dataset_and_collator(self, casmi_frame, tmp_path):
        vocab = AdductVocabulary.fit(casmi_frame["adduct"])
        ds = CasmiSpectrumDataset(
            casmi_frame,
            cache=tmp_path / "cache",
            adduct_vocab=vocab,
            k=4,
            max_atoms=16,
            fingerprint_bits=64,
            n_quantiles=8,
            include_adjacency=True,
        )
        # The unsupported-adduct row is dropped; everything else survives.
        assert ds.stats.dropped_bad_adduct == 1
        assert len(ds) == len(casmi_frame) - 1
        sample = ds[0]
        for key in (
            "eigvecs", "n_atoms", "mz", "intensity", "precursor_mz", "fingerprint", "atom_types",
            "atom_symbols", "smiles", "formula", "inchikey", "inchikey14", "molecule_id",
            "adduct_index", "collision_energy", "neutral_mass", "polarity", "spectral_quantiles", "adjacency",
        ):
            assert key in sample, key
        assert sample["eigvecs"].shape == (sample["n_atoms"], 4)
        assert sample["fingerprint"].shape == (64,)
        assert sample["spectral_quantiles"].shape == (8,)
        assert sample["adjacency"].shape == (sample["n_atoms"], sample["n_atoms"])
        assert len(sample["atom_types"]) == sample["n_atoms"]
        assert set(ds.inchikey14s) == {
            structure_record(s)["inchikey14"] for _, s in CASMI_MOLECULES
        }

        collate = make_casmi_collator(max_atoms=16, max_peaks=32, k=4, fingerprint_bits=64, n_quantiles=8)
        batch = collate([ds[i] for i in range(4)])
        assert isinstance(batch, CasmiBatch)
        assert batch.batch.x_0.shape[0] == 4 and batch.batch.x_0.shape[-1] == 4
        assert batch.adduct_index.dtype == torch.long and batch.adduct_index.shape == (4,)
        assert batch.collision_energy_mask.dtype == torch.bool
        # Collision energy alternates known / unknown in the fixture.
        assert batch.collision_energy_mask.tolist() == [True, False, True, False]
        assert torch.all(batch.collision_energy[~batch.collision_energy_mask] == 0)
        assert batch.spectral_quantiles.shape == (4, 8)
        assert torch.all(batch.polarity.abs() == 1)
        assert batch.metadata["molecule_id"][0] == "m_ethanol"
        assert batch.adjacencies is not None and len(batch.adjacencies) == 4

    def test_max_atoms_filter(self, casmi_frame, tmp_path):
        vocab = AdductVocabulary.fit(casmi_frame["adduct"])
        ds = CasmiSpectrumDataset(
            casmi_frame, cache=tmp_path / "cache", adduct_vocab=vocab, k=4, max_atoms=6, fingerprint_bits=64
        )
        assert ds.stats.dropped_too_many_atoms > 0
        assert all(ds[i]["n_atoms"] <= 6 for i in range(len(ds)))

    def test_requires_smiles(self, casmi_test_frame, tmp_path):
        with pytest.raises(ValueError, match="smiles"):
            CasmiSpectrumDataset(casmi_test_frame, cache=tmp_path, adduct_vocab=AdductVocabulary(), k=4)
