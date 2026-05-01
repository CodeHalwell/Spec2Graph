"""Tests for :mod:`spec2graph.data.massspecgym`.

Anything that hits the network (i.e. actually downloads the TSV from
Hugging Face) is marked with ``@pytest.mark.network``. Pytest does not
skip marked tests automatically — CI invokes::

    pytest tests/ -v -m "not network"

to skip them. To run only the network tests::

    pytest tests/ -v -m network

To run everything, omit ``-m`` entirely.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spec2graph.data.massspecgym import (
    REQUIRED_COLUMNS,
    FilterStats,
    filter_massspecgym,
    load_massspecgym_tsv,
    parse_peak_list,
)


def _minimal_row(**overrides) -> dict:
    """Build a single dataframe row that passes the default filters."""
    row = {
        "mzs": "[50.0, 80.0, 100.0, 120.0, 150.0, 180.0]",
        "intensities": "[0.1, 0.2, 0.3, 0.6, 1.0, 0.4]",
        "smiles": "CCO",
        "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "formula": "C2H6O",
        "precursor_mz": 47.0,
        "adduct": "[M+H]+",
        "instrument_type": "Orbitrap",
        "fold": "train",
    }
    row.update(overrides)
    return row


def _write_tsv(path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)
    return path


# ----------------------------------------------------------------------
# parse_peak_list
# ----------------------------------------------------------------------


class TestParsePeakList:
    def test_parses_stringified_list(self):
        arr = parse_peak_list("[1.0, 2.5, 3.0]")
        np.testing.assert_array_almost_equal(arr, [1.0, 2.5, 3.0])
        assert arr.dtype == np.float32

    def test_accepts_existing_list(self):
        arr = parse_peak_list([1.0, 2.0])
        np.testing.assert_array_almost_equal(arr, [1.0, 2.0])
        assert arr.dtype == np.float32

    def test_accepts_existing_ndarray(self):
        arr = parse_peak_list(np.array([1.0, 2.0], dtype=np.float64))
        assert arr.dtype == np.float32
        np.testing.assert_array_almost_equal(arr, [1.0, 2.0])

    def test_rejects_non_sequence(self):
        with pytest.raises(ValueError):
            parse_peak_list("3.14")

    def test_rejects_bad_type(self):
        with pytest.raises(TypeError):
            parse_peak_list(3.14)

    def test_no_eval_injection(self):
        # ast.literal_eval refuses names/calls — eval() would run this.
        with pytest.raises(ValueError):
            parse_peak_list("__import__('os')")

    def test_rejects_overlong_string(self):
        overlong = "[" + "1.0, " * 25000 + "1.0]"
        assert len(overlong) > 100000
        with pytest.raises(ValueError, match="exceeds maximum allowed length"):
            parse_peak_list(overlong)


# ----------------------------------------------------------------------
# load_massspecgym_tsv — schema validation via local_path
# ----------------------------------------------------------------------


class TestLoadMassSpecGymSchema:
    def test_happy_path_with_fold(self, tmp_path):
        tsv = _write_tsv(
            tmp_path / "massspecgym.tsv",
            [_minimal_row(), _minimal_row(fold="val"), _minimal_row(fold="test")],
        )
        df = load_massspecgym_tsv(local_path=tsv)
        for column in REQUIRED_COLUMNS:
            assert column in df.columns
        # "fold" should have been renamed to "split" for downstream consistency.
        assert "split" in df.columns
        assert set(df["split"]) == {"train", "val", "test"}

    def test_alternative_split_column_preserves_original(self, tmp_path, caplog):
        # When the source column is not called "split", the loader adds a
        # canonical "split" column while keeping the original column so
        # downstream code can still inspect the source-of-truth labels.
        rows = [
            _minimal_row(fold=None) | {"subset": "train"},
            _minimal_row(fold=None) | {"subset": "val"},
            _minimal_row(fold=None) | {"subset": "test"},
        ]
        for r in rows:
            del r["fold"]
        tsv = _write_tsv(tmp_path / "massspecgym.tsv", rows)
        df = load_massspecgym_tsv(local_path=tsv)
        assert "split" in df.columns
        assert "subset" in df.columns  # original preserved
        # Both columns hold the same values.
        assert list(df["split"]) == list(df["subset"])

    def test_alternative_split_column_name(self, tmp_path, caplog):
        rows = [
            _minimal_row(fold=None) | {"subset": "train"},
            _minimal_row(fold=None) | {"subset": "val"},
            _minimal_row(fold=None) | {"subset": "test"},
        ]
        for r in rows:
            del r["fold"]
        tsv = _write_tsv(tmp_path / "massspecgym.tsv", rows)

        caplog.set_level(logging.WARNING)
        df = load_massspecgym_tsv(local_path=tsv)
        assert "split" in df.columns
        assert any("Split column is named" in msg for msg in caplog.messages)

    def test_missing_required_column_raises(self, tmp_path):
        rows = [_minimal_row()]
        del rows[0]["formula"]
        tsv = _write_tsv(tmp_path / "massspecgym.tsv", rows)
        with pytest.raises(RuntimeError, match="Missing required columns"):
            load_massspecgym_tsv(local_path=tsv)

    def test_missing_split_column_raises(self, tmp_path):
        rows = [_minimal_row(fold="something_else")]
        tsv = _write_tsv(tmp_path / "massspecgym.tsv", rows)
        with pytest.raises(RuntimeError, match="Could not find a split column"):
            load_massspecgym_tsv(local_path=tsv)


# ----------------------------------------------------------------------
# filter_massspecgym
# ----------------------------------------------------------------------


class TestFilterMassSpecGym:
    def test_returns_stats_dataclass(self):
        df = pd.DataFrame([_minimal_row()])
        _, stats = filter_massspecgym(df)
        assert isinstance(stats, FilterStats)
        assert stats.initial == 1
        assert stats.final == 1

    def test_drops_wrong_adduct(self):
        rows = [_minimal_row(), _minimal_row(adduct="[M+Na]+")]
        filtered, stats = filter_massspecgym(pd.DataFrame(rows))
        assert stats.dropped_adduct == 1
        assert len(filtered) == 1
        assert filtered.iloc[0]["adduct"] == "[M+H]+"

    def test_drops_precursor_above_max(self):
        rows = [_minimal_row(precursor_mz=50.0), _minimal_row(precursor_mz=1500.0)]
        filtered, stats = filter_massspecgym(
            pd.DataFrame(rows), max_precursor_mz=1000.0
        )
        assert stats.dropped_precursor_mz == 1
        assert len(filtered) == 1
        assert filtered.iloc[0]["precursor_mz"] == 50.0

    def test_drops_sparse_spectra(self):
        rows = [
            _minimal_row(),  # 6 peaks
            _minimal_row(mzs="[50.0, 80.0]", intensities="[1.0, 0.5]"),  # 2 peaks
        ]
        filtered, stats = filter_massspecgym(pd.DataFrame(rows), min_peaks=5)
        assert stats.dropped_min_peaks == 1
        assert len(filtered) == 1

    def test_drops_long_smiles(self):
        rows = [_minimal_row(), _minimal_row(smiles="C" * 400)]
        filtered, stats = filter_massspecgym(
            pd.DataFrame(rows), max_smiles_length=200
        )
        assert stats.dropped_smiles_length == 1
        assert len(filtered) == 1

    def test_filters_compose(self):
        rows = [
            _minimal_row(),
            _minimal_row(adduct="[M+Na]+"),          # dropped: adduct
            _minimal_row(precursor_mz=1200.0),       # dropped: precursor_mz
            _minimal_row(smiles="C" * 500),          # dropped: smiles_length
            _minimal_row(mzs="[1.0]", intensities="[1.0]"),  # dropped: min_peaks
            _minimal_row(),                          # kept
        ]
        filtered, stats = filter_massspecgym(pd.DataFrame(rows))
        assert stats.initial == 6
        assert stats.final == 2
        assert stats.dropped_adduct == 1
        assert stats.dropped_precursor_mz == 1
        assert stats.dropped_smiles_length == 1
        assert stats.dropped_min_peaks == 1

    def test_drops_nan_required(self):
        rows = [_minimal_row(), _minimal_row(precursor_mz=float("nan"))]
        filtered, stats = filter_massspecgym(pd.DataFrame(rows))
        assert stats.dropped_nan_required == 1
        assert len(filtered) == 1

    def test_does_not_mutate_input(self):
        df = pd.DataFrame([_minimal_row(), _minimal_row(adduct="[M+Na]+")])
        before = df.copy()
        filter_massspecgym(df)
        pd.testing.assert_frame_equal(df, before)


# ----------------------------------------------------------------------
# Network-bound smoke test
# ----------------------------------------------------------------------


@pytest.mark.network
def test_load_real_massspecgym_smoke(tmp_path):
    """Download the real TSV and confirm schema.

    Skipped by default. Enable with ``pytest -v -m network``.
    """
    df = load_massspecgym_tsv(cache_dir=str(tmp_path))
    for column in REQUIRED_COLUMNS:
        assert column in df.columns
    assert "split" in df.columns
    assert set(df["split"].unique()) >= {"train", "val", "test"}
