"""Shared fixtures: a tiny synthetic competition dataset (real molecules, made-up spectra) and a tiny model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

CASMI_MOLECULES = [
    ("m_ethanol", "CCO"), ("m_benzene", "c1ccccc1"), ("m_acetic", "CC(=O)O"), ("m_phenol", "Oc1ccccc1"),
    ("m_toluene", "Cc1ccccc1"), ("m_pyridine", "c1ccncc1"), ("m_glucose", "OCC1OC(O)C(O)C(O)C1O"),
    ("m_caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
]
# C7H8 isomers of toluene and an unrelated small amine, for mass-window tests.
EXTRA_STRUCTURES = ["C1=CC=CC=CC1", "C1=CC2CC1C=C2", "CCN"]
K, MAX_ATOMS, MAX_PEAKS, FP_BITS, N_Q = 4, 16, 32, 64, 8


def make_casmi_frame(seed: int = 0, n_spectra_per_molecule: int = 2) -> pd.DataFrame:
    """``train.parquet``-shaped frame with the Kaggle column names, both polarities and one unsupported adduct."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    from spec2graph.casmi.adducts import parse_adduct

    rng, rows, adducts = np.random.default_rng(seed), [], ["[M+H]+", "[M-H]-", "[M+Na]+"]
    for i, (molecule_id, smiles) in enumerate(CASMI_MOLECULES):
        mass = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
        for j in range(n_spectra_per_molecule):
            adduct = adducts[(i + j) % 3]
            precursor = parse_adduct(adduct).mz(mass)
            n = int(rng.integers(5, 12))
            intensities = rng.uniform(0.05, 1.0, n)
            rows.append(dict(molecule_id=molecule_id, precursor_mz=precursor, adduct=adduct,
                             collision_energy_ev=float(rng.choice([10.0, 20.0, 40.0])) if j % 2 == 0 else np.nan,
                             ms2_mzs=np.sort(rng.uniform(15.0, max(precursor - 1.0, 16.0), n)), ms2_normalized_intensities=intensities / intensities.max(),
                             normalized_smiles=smiles, instrument_type="timsTOF", ingest_lib="synthetic"))
    rows.append(dict(molecule_id="m_bad_adduct", precursor_mz=123.0, adduct="[M+Xx]+", collision_energy_ev=20.0,
                     ms2_mzs=np.array([20.0, 40.0, 60.0, 80.0, 100.0]), ms2_normalized_intensities=np.array([0.2, 0.4, 1.0, 0.3, 0.1]),
                     normalized_smiles="CCN", instrument_type="timsTOF", ingest_lib="synthetic"))
    return pd.DataFrame(rows)


def tiny_trainer(n_adducts: int, n_timesteps: int = 10, **trainer_kw):
    """A CPU-sized conditioned model with every head enabled."""
    import torch

    from spec2graph.casmi.conditioning import CasmiModel, CasmiModelConfig, CasmiTrainer, CasmiTrainerConfig
    from spectral_diffusion import Spec2GraphDiffusion, Spec2GraphDiffusionConfig

    torch.manual_seed(0)
    core = Spec2GraphDiffusion(Spec2GraphDiffusionConfig(
        d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32, k=K, max_atoms=MAX_ATOMS,
        max_peaks=MAX_PEAKS, dropout=0.0, fingerprint_dim=FP_BITS, enable_atom_count_head=True, enable_atom_type_head=True))
    model = CasmiModel(core, CasmiModelConfig(n_adducts=n_adducts, n_quantiles=N_Q))
    return CasmiTrainer(model, CasmiTrainerConfig(n_timesteps=n_timesteps, **trainer_kw))


@pytest.fixture
def casmi_frame() -> pd.DataFrame:
    pytest.importorskip("rdkit")
    return make_casmi_frame()


@pytest.fixture
def setup(casmi_frame):
    """``(prepared frame, vocabulary, pool, trainer)`` for pipeline tests."""
    pytest.importorskip("torch")
    from spec2graph.casmi import data
    from spec2graph.casmi.adducts import AdductVocabulary
    from spec2graph.casmi.pool import CandidatePool

    df = data.prepare(casmi_frame, require_smiles=True)
    vocab = AdductVocabulary.fit(df["adduct"])
    pool = CandidatePool.build(data.structures([s for _, s in CASMI_MOLECULES] + EXTRA_STRUCTURES), K, FP_BITS, N_Q)
    return df, vocab, pool, tiny_trainer(len(vocab))
