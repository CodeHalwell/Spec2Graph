"""Tests for :mod:`spec2graph.casmi.data` and :mod:`spec2graph.casmi.spectral`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")
torch = pytest.importorskip("torch")

from spec2graph.casmi import chem, data, spectral
from spec2graph.casmi.adducts import PROTON_MASS, AdductVocabulary
from spectral_diffusion import SpectralDataProcessor
from tests.conftest import CASMI_MOLECULES, FP_BITS, K, MAX_ATOMS, MAX_PEAKS, N_Q


def test_prepare_and_load(casmi_frame, tmp_path):
    df = data.prepare(casmi_frame, require_smiles=True)
    assert {"molecule_id", "precursor_mz", "adduct", "collision_energy", "mzs", "intensities", "smiles", "neutral_mass", "polarity"} <= set(df.columns)
    row = df[df["molecule_id"] == "m_ethanol"].iloc[0]
    assert row["neutral_mass"] == pytest.approx(row["precursor_mz"] - PROTON_MASS, abs=1e-5) and row["polarity"] == 1
    assert np.isnan(df[df["molecule_id"] == "m_bad_adduct"]["neutral_mass"].iloc[0]) and (df[df["adduct"] == "[M-H]-"]["polarity"] == -1).all()
    with pytest.raises(RuntimeError):
        data.prepare(casmi_frame.drop(columns=["adduct"]))
    casmi_frame.to_parquet(tmp_path / "train.parquet", index=False)
    loaded = data.load(tmp_path / "train.parquet", require_smiles=True)
    assert len(loaded) == len(casmi_frame) and isinstance(loaded["mzs"].iloc[0], np.ndarray)
    groups = data.by_molecule(df)
    assert set(groups) == set(df["molecule_id"]) and (df.iloc[groups["m_ethanol"]]["molecule_id"] == "m_ethanol").all()


def test_peaks():
    mz, it = data.Peaks().clean(np.array([50.0, 10.0, 199.8, 150.0, 30.0, np.nan]), np.array([5.0, 10.0, 3.0, 0.0, 2.0, 1.0]), 200.0)
    np.testing.assert_allclose(mz, [10.0, 30.0, 50.0]) and np.testing.assert_allclose(it, [1.0, 0.2, 0.5])
    mz, it = data.Peaks(top_k=3, sqrt_intensity=True).clean(np.arange(1.0, 11.0), np.arange(1.0, 11.0), 100.0)
    np.testing.assert_allclose(mz, [8.0, 9.0, 10.0]) and np.testing.assert_allclose(it, np.sqrt([0.8, 0.9, 1.0]))
    assert data.Peaks().clean(np.array([300.0]), np.array([1.0]), 200.0)[0].size == 0
    with pytest.raises(ValueError):
        data.Peaks().clean(np.array([1.0, 2.0]), np.array([1.0]), 10.0)


def test_structures():
    table = data.structures(["CCO", "OCC", "c1ccccc1", "garbage", "Oc1ccccn1", "O=c1cccc[nH]1"], source="unit")
    assert len(table) == 3 and table["inchikey14"].is_unique and (table["source"] == "unit").all()
    ethanol = table[table["formula"] == "C2H6O"].iloc[0]
    assert ethanol["n_inputs"] == 2 and ethanol["input_smiles"] == "CCO"
    assert len(data.merge_structures([table, data.structures(["CCO", "CCN"])])) == 4


def test_spectral_descriptors(tmp_path):
    processor = SpectralDataProcessor(k=K)
    benzene = spectral.eigenvalues(processor.smiles_to_adjacency("c1ccccc1"), processor)
    assert benzene[0] == pytest.approx(0, abs=1e-6) and benzene[-1] == pytest.approx(2, abs=1e-6)  # even ring: bipartite
    assert spectral.eigenvalues(processor.smiles_to_adjacency("C1CCCC1"), processor)[-1] < 1.999  # odd ring is not
    adj = processor.smiles_to_adjacency("CC(C)C(=O)O")
    perm = np.random.default_rng(0).permutation(len(adj))
    np.testing.assert_allclose(spectral.eigenvalues(adj, processor), spectral.eigenvalues(adj[np.ix_(perm, perm)], processor), atol=1e-6)
    q = spectral.quantiles(benzene, N_Q)
    assert q.shape == (N_Q,) and np.all(np.diff(q) >= -1e-6) and spectral.quantile_distance(q, np.stack([q, q + 1])).tolist() == pytest.approx([0, 1])
    with pytest.raises(ValueError):
        spectral.quantiles(np.array([]), N_Q)
    feats = spectral.features("Oc1ccccc1", processor, N_Q, FP_BITS)
    assert feats["eigvecs"].shape == (7, K) and feats["fingerprint"].shape == (FP_BITS,) and feats["elements"][0] == 2 and feats["n_atoms"] == 7
    assert spectral.features("garbage", processor) is None
    cache = spectral.SpectralCache(tmp_path, k=K, fingerprint_bits=FP_BITS)
    key = chem.record("c1ccccc1")["inchikey"]
    first = cache.spectrum("c1ccccc1", key)
    assert first.shape == (6,) and cache.spectrum_path(key).exists() and np.allclose(first, cache.spectrum("c1ccccc1", key))
    assert cache.spectrum("garbage", "AAAAAAAAAAAAAA-UHFFFAOYSA-N") is None


def test_conditioning_and_spectra(casmi_frame):
    cond = data.Conditioning.build([1, 0], [20.0, None], [180.0, 94.0], [1.0, -1.0])
    assert cond.collision_energy_mask.tolist() == [True, False] and cond.collision_energy.tolist() == [20.0, 0.0]
    assert cond.select(torch.tensor([1])).neutral_mass.tolist() == [94.0]
    df = data.prepare(casmi_frame, require_smiles=True)
    vocab = AdductVocabulary.fit(df["adduct"])
    spectra = data.Spectra.from_rows(df[df["molecule_id"] == "m_toluene"], vocab, MAX_PEAKS)
    assert spectra.n == 2 and spectra.mz.shape == spectra.intensity.shape == spectra.mask.shape and spectra.cond.adduct_index.shape == (2,)
    assert data.Spectra.from_rows(df[df["molecule_id"] == "m_bad_adduct"], vocab, MAX_PEAKS) is None


def test_dataset_and_collator(casmi_frame, tmp_path):
    vocab = AdductVocabulary.fit(casmi_frame["adduct"])
    cache = spectral.SpectralCache(tmp_path / "cache", k=K, fingerprint_bits=FP_BITS)
    ds = data.Dataset(casmi_frame, cache, vocab, max_atoms=MAX_ATOMS, n_quantiles=N_Q, adjacency=True)
    assert ds.drops == {"unsupported adduct": 1} and len(ds) == len(casmi_frame) - 1
    sample = ds[0]
    assert sample["eigvecs"].shape == (sample["n_atoms"], K) and sample["spectral_quantiles"].shape == (N_Q,)
    assert sample["adjacency"].shape == (sample["n_atoms"],) * 2 and sample["atom_types"].dtype == np.int64
    assert {r["inchikey14"] for r in ds.records} == {chem.record(s)["inchikey14"] for _, s in CASMI_MOLECULES}
    batch = data.Collator(MAX_ATOMS, MAX_PEAKS, K, FP_BITS)([ds[i] for i in range(4)])
    assert batch.core.x_0.shape[0] == 4 and batch.cond.collision_energy_mask.tolist() == [True, False, True, False]
    assert batch.quantiles.shape == (4, N_Q) and batch.meta["molecule_id"][0] == "m_ethanol" and len(batch.adjacencies) == 4
    assert batch.to("cpu").core.x_0.shape == batch.core.x_0.shape
    assert data.Dataset(casmi_frame, cache, vocab, max_atoms=6).drops["atom count"] > 0
