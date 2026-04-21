"""Tests for :mod:`spec2graph.eval.metrics`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("myopic_mces")

from spec2graph.eval.metrics import (
    BenchmarkResults,
    MCES_SENTINEL_DISTANCE,
    aggregate_per_example,
    canonicalise,
    mces_distance,
    rank_samples_by_frequency,
    tanimoto_similarity,
    top_k_accuracy,
    top_k_mces,
    top_k_tanimoto,
    validity_rate,
)


class TestCanonicalise:
    def test_canonicalises_equivalent_smiles(self):
        # Two encodings of ethanol should canonicalise to the same string.
        a = canonicalise("CCO")
        b = canonicalise("OCC")
        assert a == b

    def test_invalid_returns_none(self):
        assert canonicalise("not-a-smiles") is None

    def test_none_returns_none(self):
        assert canonicalise(None) is None


class TestRanking:
    def test_ranks_by_frequency(self):
        # CCO sampled three times, CCC twice, CCN once.
        result = rank_samples_by_frequency(["CCO", "CCC", "CCO", "CCN", "CCC", "CCO"])
        assert result[0] == "CCO"

    def test_drops_invalid(self):
        result = rank_samples_by_frequency(["CCO", None, "!!!"])
        assert result == ["CCO"]

    def test_deduplicates_equivalent_encodings(self):
        # Two encodings of ethanol should collapse to one.
        result = rank_samples_by_frequency(["CCO", "OCC", "CCC"])
        # Only two unique molecules after canonicalisation.
        assert len(result) == 2


class TestTopKAccuracy:
    def test_match(self):
        assert top_k_accuracy("CCO", ["CCO", "CCC"], k=1) == 1.0

    def test_no_match(self):
        assert top_k_accuracy("CCO", ["CCC", "CCN"], k=2) == 0.0

    def test_ground_truth_canonicalised(self):
        # Ground truth given in non-canonical form should still match.
        assert top_k_accuracy("OCC", ["CCO"], k=1) == 1.0

    def test_outside_top_k_counts_as_miss(self):
        # Match exists but at position 3 — top-1 is a miss.
        assert top_k_accuracy("CCO", ["CCC", "CCN", "CCO"], k=1) == 0.0
        assert top_k_accuracy("CCO", ["CCC", "CCN", "CCO"], k=10) == 1.0


class TestTopKTanimoto:
    def test_identity_is_one(self):
        assert top_k_tanimoto("CCO", ["CCO"], k=1) == pytest.approx(1.0)

    def test_empty_returns_zero(self):
        assert top_k_tanimoto("CCO", [], k=1) == 0.0

    def test_takes_maximum(self):
        # CCC is close-ish to CCO; CCCCCC is less close. Max over the
        # top-k should pick the better one.
        best = top_k_tanimoto("CCO", ["CCCCCC", "CCC"], k=2)
        ccc_sim = tanimoto_similarity("CCO", "CCC")
        assert best == pytest.approx(ccc_sim)


class TestTopKMces:
    def test_identity_is_zero(self):
        # Distance from a molecule to itself must be zero.
        assert top_k_mces("CCO", ["CCO"], k=1) == pytest.approx(0.0)

    def test_empty_returns_sentinel(self):
        assert top_k_mces("CCO", [], k=1) == MCES_SENTINEL_DISTANCE

    def test_takes_minimum(self):
        # Same molecule plus a very different one; minimum is the identity.
        d = top_k_mces("CCO", ["CCCCCC", "CCO"], k=2)
        assert d == pytest.approx(0.0)


class TestValidityRate:
    def test_all_valid(self):
        assert validity_rate(["CCO", "CCC"]) == pytest.approx(1.0)

    def test_mixed(self):
        # CCO is valid, None is invalid, "!!!" fails MolFromSmiles.
        assert validity_rate(["CCO", None, "!!!"]) == pytest.approx(1 / 3)

    def test_empty(self):
        assert validity_rate([]) == 0.0


class TestBenchmarkResults:
    def _per_example(self, n: int = 5) -> list[dict]:
        return [
            {
                "top_1_accuracy": 1.0 if i == 0 else 0.0,
                "top_10_accuracy": 1.0 if i < 3 else 0.0,
                "top_1_tanimoto": 0.5 + 0.1 * (i % 2),
                "top_10_tanimoto": 0.6 + 0.1 * (i % 2),
                "top_1_mces": 2.0 + i,
                "top_10_mces": 1.0 + i,
                "validity": 1.0,
            }
            for i in range(n)
        ]

    def test_aggregate_basic(self):
        results = aggregate_per_example(
            self._per_example(5), n_samples_per_example=100, n_resamples=200
        )
        assert isinstance(results, BenchmarkResults)
        assert results.n_examples == 5
        assert results.n_samples_per_example == 100
        assert 0.0 <= results.top_1_accuracy <= 1.0
        assert 0.0 <= results.top_10_accuracy <= 1.0

    def test_aggregate_ci_brackets_mean(self):
        per_example = self._per_example(20)
        results = aggregate_per_example(
            per_example, n_samples_per_example=100, n_resamples=500
        )
        low, high = results.top_1_accuracy_ci
        assert low <= results.top_1_accuracy <= high

    def test_aggregate_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            aggregate_per_example([], n_samples_per_example=100)

    def test_to_dict_has_expected_keys(self):
        results = aggregate_per_example(
            self._per_example(3), n_samples_per_example=100, n_resamples=100
        )
        d = results.to_dict()
        for key in [
            "top_1_accuracy",
            "top_10_accuracy",
            "top_1_tanimoto",
            "top_10_tanimoto",
            "top_1_mces",
            "top_10_mces",
            "validity",
            "n_examples",
            "n_samples_per_example",
        ]:
            assert key in d

    def test_markdown_row_shape(self):
        results = aggregate_per_example(
            self._per_example(3), n_samples_per_example=100, n_resamples=100
        )
        row = results.to_markdown_row("Spec2Graph")
        assert row.startswith("| Spec2Graph ")
        # Nine pipe-separated cells.
        assert row.count("|") == 10
