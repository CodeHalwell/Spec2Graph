"""Tests for :mod:`spec2graph.casmi.metric` (official CASMI 2026 scoring)."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("rdkit")

from spec2graph.casmi.metric import (
    MAX_GUESSES,
    dedupe_by_structure,
    format_guesses,
    inchikey14,
    mrr_at_k,
    read_submission,
    reciprocal_rank,
    reciprocal_rank_from_keys,
    same_structure,
    score_submission,
    validate_submission,
    write_submission,
)

GLUCOSE_STEREO = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"
GLUCOSE_FLAT = "OCC1OC(O)C(O)C(O)C1O"


class TestInchikey14:
    def test_matches_competition_example(self):
        # The rules page states both glucose spellings reduce to this key.
        assert inchikey14(GLUCOSE_STEREO) == "WQZGKKKJIJFFOK"
        assert inchikey14(GLUCOSE_FLAT) == "WQZGKKKJIJFFOK"

    def test_tautomers_collapse(self):
        assert inchikey14("Oc1ccccn1") == inchikey14("O=c1cccc[nH]1")

    def test_protonation_state_is_ignored(self):
        # Standard InChI moves (de)protonation into the /p layer, which sits
        # outside the first block, so an acid and its anion share a key.
        assert inchikey14("CC(=O)[O-]") == inchikey14("CC(=O)O")

    def test_invalid_inputs(self):
        assert inchikey14("not a smiles") is None
        assert inchikey14("") is None
        assert inchikey14(None) is None
        assert inchikey14("C" * 5000) is None

    def test_same_structure(self):
        assert same_structure(GLUCOSE_STEREO, GLUCOSE_FLAT)
        assert not same_structure("CCO", "CCN")
        assert not same_structure("garbage", "garbage")


class TestReciprocalRank:
    def test_positions(self):
        assert reciprocal_rank_from_keys("A", ["A"]) == 1.0
        assert reciprocal_rank_from_keys("A", ["B", "A"]) == 0.5
        assert reciprocal_rank_from_keys("A", [None, None, "A"]) == pytest.approx(1 / 3)
        assert reciprocal_rank_from_keys("A", ["B"]) == 0.0
        assert reciprocal_rank_from_keys(None, ["A"]) == 0.0

    def test_only_first_k_count(self):
        guesses = ["B"] * MAX_GUESSES + ["A"]
        assert reciprocal_rank_from_keys("A", guesses) == 0.0

    def test_smiles_level(self):
        assert reciprocal_rank(GLUCOSE_STEREO, ["CCO", GLUCOSE_FLAT]) == 0.5

    def test_mrr(self):
        truth = {"m1": "CCO", "m2": "c1ccccc1", "m3": "CC(=O)O"}
        predictions = {"m1": ["CCO"], "m2": ["CCO", "C1=CC=CC=C1"], "m4": ["CC(=O)O"]}
        assert mrr_at_k(truth, predictions) == pytest.approx((1.0 + 0.5 + 0.0) / 3)

    def test_mrr_empty_truth_raises(self):
        with pytest.raises(ValueError):
            mrr_at_k({}, {})


class TestGuessFormatting:
    def test_format_guesses(self):
        assert format_guesses([" CCO ", "", "a;b", None, "CCN"]) == ["CCO", "ab", "CCN"]
        assert len(format_guesses(["C"] * 40)) == MAX_GUESSES

    def test_dedupe_by_structure(self):
        assert dedupe_by_structure([GLUCOSE_STEREO, GLUCOSE_FLAT, "garbage", "CCO"]) == [
            GLUCOSE_STEREO,
            "CCO",
        ]


class TestSubmission:
    def test_round_trip(self, tmp_path):
        predictions = {"m_1": ["CCO", "CCN"], "m_2": [], "m_3": ["garbage"]}
        path = tmp_path / "submission.csv"
        df = write_submission(predictions, path, molecule_ids=["m_1", "m_2", "m_3"])
        assert list(df.columns) == ["molecule_id", "smiles"]
        loaded = read_submission(path)
        assert loaded["m_1"] == ["CCO", "CCN"]
        # Empty and all-invalid predictions get the fallback SMILES.
        assert loaded["m_2"] == ["C"]
        assert loaded["m_3"] == ["C"]

    def test_validation_catches_rule_violations(self):
        df = pd.DataFrame(
            {
                "molecule_id": ["a", "a", "b"],
                "smiles": ["C", ";".join(["C"] * 26), ""],
            }
        )
        problems = validate_submission(df, molecule_ids=["a", "b", "c"])
        joined = " ".join(problems)
        assert "repeated molecule_id" in joined
        assert "exceed 25" in joined
        assert "empty smiles" in joined
        assert "expected molecule_ids missing" in joined

    def test_validation_missing_columns(self):
        assert validate_submission(pd.DataFrame({"x": [1]})) == [
            "missing column 'molecule_id'",
            "missing column 'smiles'",
        ]

    def test_write_rejects_bad_file(self, tmp_path):
        with pytest.raises(ValueError):
            write_submission({"a": ["C"]}, tmp_path / "s.csv", molecule_ids=["a", "a"])

    def test_score_submission(self, tmp_path):
        truth = {"m_1": "CCO", "m_2": "c1ccccc1"}
        path = tmp_path / "submission.csv"
        write_submission({"m_1": ["CCN", "CCO"], "m_2": ["C1=CC=CC=C1"]}, path, molecule_ids=list(truth))
        assert score_submission(path, truth) == pytest.approx(0.75)
