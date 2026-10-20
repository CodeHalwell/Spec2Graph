"""Shared fixtures for the Spec2Graph test suite.

The CASMI fixtures build a tiny synthetic competition-style dataset: a few
real molecules with made-up spectra, several spectra per molecule, both
polarities, and one row with an unsupported adduct. Nothing here downloads
anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# (molecule_id, smiles) pairs — small, RDKit-friendly molecules.
CASMI_MOLECULES: list[tuple[str, str]] = [
    ("m_ethanol", "CCO"),
    ("m_benzene", "c1ccccc1"),
    ("m_acetic", "CC(=O)O"),
    ("m_phenol", "Oc1ccccc1"),
    ("m_toluene", "Cc1ccccc1"),
    ("m_pyridine", "c1ccncc1"),
    ("m_glucose", "OCC1OC(O)C(O)C(O)C1O"),
    ("m_caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
]


def _exact_mass(smiles: str) -> float:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    return float(Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles)))


def make_casmi_frame(seed: int = 0, n_spectra_per_molecule: int = 2) -> pd.DataFrame:
    """Synthetic ``train.parquet``-shaped frame using the Kaggle column names."""
    from spec2graph.casmi.adducts import parse_adduct

    rng = np.random.default_rng(seed)
    rows = []
    adducts = ["[M+H]+", "[M-H]-", "[M+Na]+"]
    for mol_idx, (molecule_id, smiles) in enumerate(CASMI_MOLECULES):
        mass = _exact_mass(smiles)
        for j in range(n_spectra_per_molecule):
            adduct = adducts[(mol_idx + j) % len(adducts)]
            precursor = parse_adduct(adduct).mz_from_neutral(mass)
            n_peaks = int(rng.integers(5, 12))
            mzs = np.sort(rng.uniform(15.0, max(precursor - 1.0, 16.0), size=n_peaks)).astype(np.float64)
            intensities = rng.uniform(0.05, 1.0, size=n_peaks)
            intensities /= intensities.max()
            rows.append(
                {
                    "molecule_id": molecule_id,
                    "precursor_mz": float(precursor),
                    "adduct": adduct,
                    "collision_energy_ev": float(rng.choice([10.0, 20.0, 40.0])) if j % 2 == 0 else np.nan,
                    "ms2_mzs": mzs,
                    "ms2_normalized_intensities": intensities,
                    "normalized_smiles": smiles,
                    "instrument_type": "timsTOF",
                    "ingest_lib": "synthetic",
                }
            )
    # One unsupported adduct row that every consumer must drop cleanly.
    rows.append(
        {
            "molecule_id": "m_bad_adduct",
            "precursor_mz": 123.0,
            "adduct": "[M+Xx]+",
            "collision_energy_ev": 20.0,
            "ms2_mzs": np.array([20.0, 40.0, 60.0, 80.0, 100.0]),
            "ms2_normalized_intensities": np.array([0.2, 0.4, 1.0, 0.3, 0.1]),
            "normalized_smiles": "CCN",
            "instrument_type": "timsTOF",
            "ingest_lib": "synthetic",
        }
    )
    return pd.DataFrame(rows)


@pytest.fixture
def casmi_frame() -> pd.DataFrame:
    pytest.importorskip("rdkit")
    return make_casmi_frame()


@pytest.fixture
def casmi_test_frame(casmi_frame: pd.DataFrame) -> pd.DataFrame:
    """A ``test.parquet``-shaped frame: the same spectra without answers."""
    return casmi_frame.drop(columns=["normalized_smiles"]).reset_index(drop=True)
