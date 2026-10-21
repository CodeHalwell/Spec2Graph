"""Tests for :mod:`spec2graph.casmi.chem` and :mod:`spec2graph.casmi.metric`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")

from spec2graph.casmi import chem
from spec2graph.casmi.metric import MAX_GUESSES, Submission, bootstrap_ci, clean_guesses, mrr, reciprocal_rank

GLUCOSE = ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "OCC1OC(O)C(O)C(O)C1O")


def test_inchikey14_matches_competition_rules():
    assert chem.inchikey14(GLUCOSE[0]) == chem.inchikey14(GLUCOSE[1]) == "WQZGKKKJIJFFOK"  # stereo ignored
    assert chem.inchikey14("Oc1ccccn1") == chem.inchikey14("O=c1cccc[nH]1")  # tautomers collapse
    assert chem.inchikey14("CC(=O)[O-]") == chem.inchikey14("CC(=O)O")  # /p layer sits outside the first block
    assert chem.inchikey14("C=C(C)C1CCC2CCC(=O)C=C2C1") == chem.inchikey14("C=C(C)[C@@H]1CCC2CCC(=O)C=C2C1")  # metric notebook regression
    assert all(chem.inchikey14(s) is None for s in ("not a smiles", "", None, "C" * 5000))


def test_chem_helpers():
    rec = chem.record("OCC1OC(O)C(O)C(O)C1O")
    assert (rec["inchikey14"], rec["formula"], rec["n_heavy_atoms"]) == ("WQZGKKKJIJFFOK", "C6H12O6", 12)
    assert rec["exact_mass"] == pytest.approx(180.0634, abs=1e-3) and rec["inchikey"].startswith("WQZGKKKJIJFFOK")
    assert chem.record("garbage") is None and chem.record("Oc1ccccn1", True)["smiles"] == chem.record("O=c1cccc[nH]1", True)["smiles"]
    adj = chem.kekulized_adjacency("c1ccccc1")
    assert np.allclose(adj, adj.T) and sorted(adj[np.triu_indices(6, 1)][adj[np.triu_indices(6, 1)] > 0]) == [1, 1, 1, 2, 2, 2]
    assert chem.kekulized_adjacency("garbage").shape == (0, 0) and chem.symbols("Oc1ccccc1")[0] == "O"
    disconnected = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
    assert chem.to_smiles(disconnected, ["C", "C", "O"]) is None
    assert chem.to_smiles(disconnected, ["C", "C", "O"], connected=False) == "CC.O"
    assert chem.parallel_map(len, ["a", "bb"]) == [1, 2] and chem.parallel_map(len, ["a", "bb"], n_jobs=2) == [1, 2]


def test_reciprocal_rank_and_mrr():
    assert reciprocal_rank("CCO", ["CCO"]) == 1.0 and reciprocal_rank("CCO", ["CCN", "garbage", "OCC"]) == pytest.approx(1 / 3)
    assert reciprocal_rank("CCO", ["CCN"] * MAX_GUESSES + ["CCO"]) == 0.0 and reciprocal_rank("garbage", ["CCO"]) == 0.0
    truth = {"m1": "CCO", "m2": "c1ccccc1", "m3": "CC(=O)O"}
    assert mrr(truth, {"m1": ["CCO"], "m2": ["CCO", "C1=CC=CC=C1"], "m4": ["CC(=O)O"]}) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        mrr({}, {})
    low, high = bootstrap_ci([0.0, 1.0, 0.5, 1.0], n=200)
    assert 0 <= low <= 0.625 <= high <= 1


def test_clean_guesses():
    assert clean_guesses([" CCO ", "", "a;b", None, GLUCOSE[0], GLUCOSE[1], "OCC"]) == ["CCO", GLUCOSE[0]]
    assert len(clean_guesses([f"C{'C' * i}" for i in range(40)])) == MAX_GUESSES


def test_submission_round_trip_and_rules(tmp_path):
    path = tmp_path / "submission.csv"
    frame = Submission({"m_1": ["CCO", "CCN"], "m_2": [], "m_3": ["garbage"]}, ["m_1", "m_2", "m_3"]).write(path)
    assert list(frame.columns) == ["molecule_id", "smiles"]
    loaded = Submission.read(path)
    assert loaded.rows == {"m_1": ["CCO", "CCN"], "m_2": ["C"], "m_3": ["C"]}
    assert loaded.score({"m_1": "OCC", "m_2": "c1ccccc1"}) == pytest.approx(0.5)
    bad = pd.DataFrame({"molecule_id": ["a", "a", "b"], "smiles": ["C", ";".join(["C"] * 26), ""]})
    problems = " ".join(Submission.problems(bad, ["a", "b", "c"]))
    assert all(p in problems for p in ("repeated", "more than 25", "empty smiles", "mismatch"))
    assert Submission.problems(pd.DataFrame({"x": [1]})) == ["missing columns ['molecule_id', 'smiles']"]
    with pytest.raises(ValueError):
        Submission({"a": ["C"] * 26}, clean=False).write(tmp_path / "x.csv")  # 26 guesses: rejected
