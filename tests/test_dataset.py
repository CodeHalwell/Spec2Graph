"""Tests for :mod:`spec2graph.data.dataset` and :mod:`spec2graph.data.collate`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("rdkit")

from spec2graph.data.collate import (
    CollatedEvalBatch,
    make_metadata_collator,
    make_training_batch_collator,
)
from spec2graph.data.dataset import MassSpecGymDataset
from spec2graph.data.elements import PADDING_INDEX


# Small molecules whose eigenvectors compute cleanly.
_ROWS = [
    {
        "smiles": "CCO",
        "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "formula": "C2H6O",
        "mzs": "[20.0, 25.0, 27.0, 29.0, 40.0, 45.0]",
        "intensities": "[0.1, 0.3, 0.5, 1.0, 0.4, 0.2]",
        "precursor_mz": 47.0,
        "adduct": "[M+H]+",
        "instrument_type": "Orbitrap",
        "split": "train",
    },
    {
        "smiles": "c1ccccc1",
        "inchikey": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
        "formula": "C6H6",
        "mzs": "[30.0, 50.0, 52.0, 60.0, 75.0, 77.0]",
        "intensities": "[0.2, 0.9, 0.4, 0.3, 0.6, 1.0]",
        "precursor_mz": 79.0,
        "adduct": "[M+H]+",
        "instrument_type": "Orbitrap",
        "split": "train",
    },
    {
        "smiles": "CC(=O)O",
        "inchikey": "QTBSBXVTEAMEQO-UHFFFAOYSA-N",
        "formula": "C2H4O2",
        "mzs": "[20.0, 43.0, 45.0, 55.0, 60.0]",
        "intensities": "[0.1, 1.0, 0.3, 0.5, 0.2]",
        "precursor_mz": 61.0,
        "adduct": "[M+H]+",
        "instrument_type": "Orbitrap",
        "split": "val",
    },
]


def _df() -> pd.DataFrame:
    return pd.DataFrame(_ROWS)


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    return tmp_path


class TestDataset:
    def test_length_matches_filtered_rows(self, tmp_cache: Path):
        ds = MassSpecGymDataset(
            split="train",
            cache_dir=tmp_cache,
            k=4,
            max_atoms=32,
            max_peaks=16,
            fingerprint_bits=64,
            dataframe=_df(),
        )
        assert len(ds) == 2

    def test_getitem_returns_expected_keys(self, tmp_cache: Path):
        ds = MassSpecGymDataset(
            split="train",
            cache_dir=tmp_cache,
            k=4,
            max_atoms=32,
            max_peaks=16,
            fingerprint_bits=64,
            dataframe=_df(),
        )
        sample = ds[0]
        required = {
            "eigvecs",
            "n_atoms",
            "mz",
            "intensity",
            "precursor_mz",
            "fingerprint",
            "atom_types",
            "atom_symbols",
            "smiles",
            "formula",
            "inchikey",
            "collision_energy",
        }
        assert required.issubset(sample.keys())

    def test_eigvec_shape_matches_n_atoms(self, tmp_cache: Path):
        ds = MassSpecGymDataset(
            split="train",
            cache_dir=tmp_cache,
            k=4,
            max_atoms=32,
            max_peaks=16,
            fingerprint_bits=64,
            dataframe=_df(),
        )
        for sample in ds:
            assert sample["eigvecs"].shape[0] == sample["n_atoms"]
            assert sample["eigvecs"].shape[1] == 4
            assert sample["n_atoms"] <= 32

    def test_mz_is_sorted(self, tmp_cache: Path):
        ds = MassSpecGymDataset(
            split="train",
            cache_dir=tmp_cache,
            k=4,
            max_atoms=32,
            max_peaks=16,
            fingerprint_bits=64,
            dataframe=_df(),
        )
        for sample in ds:
            mz = sample["mz"]
            assert np.all(mz[:-1] <= mz[1:])

    def test_intensity_is_normalised(self, tmp_cache: Path):
        ds = MassSpecGymDataset(
            split="train",
            cache_dir=tmp_cache,
            k=4,
            max_atoms=32,
            max_peaks=16,
            fingerprint_bits=64,
            dataframe=_df(),
        )
        for sample in ds:
            if sample["intensity"].size > 0:
                assert abs(sample["intensity"].max() - 1.0) < 1e-6

    def test_precursor_is_excluded(self, tmp_cache: Path):
        ds = MassSpecGymDataset(
            split="train",
            cache_dir=tmp_cache,
            k=4,
            max_atoms=32,
            max_peaks=16,
            fingerprint_bits=64,
            dataframe=_df(),
        )
        for sample in ds:
            cutoff = sample["precursor_mz"] - 0.5
            assert np.all(sample["mz"] < cutoff)

    def test_top_k_peaks_cap(self, tmp_cache: Path):
        ds = MassSpecGymDataset(
            split="train",
            cache_dir=tmp_cache,
            k=4,
            max_atoms=32,
            max_peaks=16,
            top_k_peaks=3,
            fingerprint_bits=64,
            dataframe=_df(),
        )
        for sample in ds:
            assert sample["mz"].size <= 3

    def test_atom_types_length_matches_n_atoms(self, tmp_cache: Path):
        ds = MassSpecGymDataset(
            split="train",
            cache_dir=tmp_cache,
            k=4,
            max_atoms=32,
            max_peaks=16,
            fingerprint_bits=64,
            dataframe=_df(),
        )
        for sample in ds:
            assert len(sample["atom_types"]) == sample["n_atoms"]

    def test_include_adjacency(self, tmp_cache: Path):
        ds = MassSpecGymDataset(
            split="train",
            cache_dir=tmp_cache,
            k=4,
            max_atoms=32,
            max_peaks=16,
            fingerprint_bits=64,
            dataframe=_df(),
            include_adjacency=True,
        )
        sample = ds[0]
        assert "adjacency" in sample
        assert sample["adjacency"].shape == (sample["n_atoms"], sample["n_atoms"])

    def test_rejects_bad_split(self, tmp_cache: Path):
        with pytest.raises(ValueError, match="split must be one of"):
            MassSpecGymDataset(
                split="training",  # typo
                cache_dir=tmp_cache,
                dataframe=_df(),
            )


class TestCollator:
    def _dataset(self, tmp_cache: Path) -> MassSpecGymDataset:
        return MassSpecGymDataset(
            split="train",
            cache_dir=tmp_cache,
            k=4,
            max_atoms=8,
            max_peaks=16,
            fingerprint_bits=64,
            dataframe=_df(),
        )

    def test_collate_produces_training_batch(self, tmp_cache: Path):
        ds = self._dataset(tmp_cache)
        collator = make_training_batch_collator(
            max_atoms=8, max_peaks=16, k=4, fingerprint_bits=64
        )
        batch = collator([ds[0], ds[1]])
        assert batch.x_0.shape == (2, batch.x_0.shape[1], 4)
        assert batch.atom_mask.dtype == torch.bool
        assert batch.spectrum_mask.dtype == torch.bool
        assert batch.atom_type_targets is not None
        assert batch.atom_type_targets.shape == batch.atom_mask.shape

    def test_masks_match_input_lengths(self, tmp_cache: Path):
        ds = self._dataset(tmp_cache)
        collator = make_training_batch_collator(
            max_atoms=8, max_peaks=16, k=4, fingerprint_bits=64
        )
        batch = collator([ds[0], ds[1]])
        for i, sample in enumerate([ds[0], ds[1]]):
            assert batch.atom_mask[i].sum().item() == sample["n_atoms"]
            assert batch.spectrum_mask[i].sum().item() == sample["mz"].size

    def test_padded_atom_type_targets_use_sentinel(self, tmp_cache: Path):
        ds = self._dataset(tmp_cache)
        collator = make_training_batch_collator(
            max_atoms=8, max_peaks=16, k=4, fingerprint_bits=64
        )
        batch = collator([ds[0], ds[1]])
        # Any position outside the atom mask must be the ignore sentinel.
        for i in range(batch.atom_mask.shape[0]):
            for j in range(batch.atom_mask.shape[1]):
                if not batch.atom_mask[i, j]:
                    assert batch.atom_type_targets[i, j].item() == PADDING_INDEX

    def test_metadata_collator_returns_metadata(self, tmp_cache: Path):
        ds = MassSpecGymDataset(
            split="train",
            cache_dir=tmp_cache,
            k=4,
            max_atoms=8,
            max_peaks=16,
            fingerprint_bits=64,
            dataframe=_df(),
            include_adjacency=True,
        )
        collator = make_metadata_collator(
            max_atoms=8, max_peaks=16, k=4, fingerprint_bits=64
        )
        collated = collator([ds[0], ds[1]])
        assert isinstance(collated, CollatedEvalBatch)
        assert collated.metadata["smiles"] == [ds[0]["smiles"], ds[1]["smiles"]]
        assert collated.metadata["formula"] == [ds[0]["formula"], ds[1]["formula"]]
        assert collated.metadata["inchikey"] == [ds[0]["inchikey"], ds[1]["inchikey"]]
        assert collated.adjacencies is not None
        assert len(collated.adjacencies) == 2

    def test_collator_drops_empty_samples(self, tmp_cache: Path):
        ds = self._dataset(tmp_cache)
        good = ds[0]
        bad = dict(good)
        bad["n_atoms"] = 0
        collator = make_training_batch_collator(
            max_atoms=8, max_peaks=16, k=4, fingerprint_bits=64
        )
        batch = collator([good, bad])
        assert batch.x_0.shape[0] == 1
