"""Tests for ranker, predict, validation and denovo."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")
torch = pytest.importorskip("torch")

from spec2graph.casmi import chem, data, denovo, validation
from spec2graph.casmi.conditioning import CasmiTrainer
from spec2graph.casmi.metric import Submission
from spec2graph.casmi.predict import PredictConfig, Predictor
from spec2graph.casmi.ranker import LinearRanker, rank
from spec2graph.casmi.scoring import FEATURES, CandidateScores, ScoringConfig
from spec2graph.data.elements import ELEMENT_TO_INDEX
from spectral_diffusion import SpectralDataProcessor
from tests.conftest import K, MAX_PEAKS

FAST = PredictConfig(scoring=ScoringConfig(ppm=20.0, n_samples=1, n_steps=2, max_atom_counts=2, alignment_iters=2, sinkhorn_iters=5), max_peaks=MAX_PEAKS, top_k=5)


def test_ranker(setup, tmp_path):
    _, _, pool, _ = setup
    ranker = LinearRanker.heuristic()
    features = np.full((3, len(FEATURES)), np.nan, dtype=np.float32)
    features[:, FEATURES.index("fp_loglik")] = [-0.5, -0.1, -0.9]
    assert ranker.score(features).argmax() == 1 and np.isfinite(ranker.score(features)).all()  # NaNs impute to the mean
    rng, col = np.random.default_rng(0), FEATURES.index("quantile_dist")
    groups = []
    for _ in range(30):  # the true candidate always has the largest quantile distance: the fit must flip the heuristic sign
        f, y = rng.normal(size=(6, len(FEATURES))).astype(np.float32), np.zeros(6, bool)
        f[:, col], f[0, col], y[0] = rng.uniform(0, 1, 6), 2.0, True
        groups.append((f, y))
    groups.append((np.ones((1, len(FEATURES)), dtype=np.float32), np.array([True])))  # single-candidate groups are ignored
    fitted = LinearRanker.fit(groups, epochs=200, lr=0.1)
    assert fitted.weights[col] > 0 and np.mean([fitted.score(f).argmax() == 0 for f, _ in groups[:-1]]) > 0.9
    fitted.save(tmp_path / "ranker.json")
    np.testing.assert_allclose(LinearRanker.load(tmp_path / "ranker.json").weights, fitted.weights)
    with pytest.raises(ValueError):
        LinearRanker.fit([groups[-1]])
    scores = CandidateScores("m", np.arange(len(pool)), np.zeros((len(pool), len(FEATURES)), np.float32), 100.0)
    scores.features[:, FEATURES.index("proj_overlap")] = np.linspace(0, 1, len(pool))
    assert rank(scores, pool, ranker, 3)[0] == pool.smiles[-1] and rank(CandidateScores("m", np.zeros(0, int), np.zeros((0, len(FEATURES))), 1.0), pool, ranker) == []


def test_predictor(setup, tmp_path):
    df, vocab, pool, trainer = setup
    predictor = Predictor(trainer, pool, vocab, config=FAST)
    ethanol = chem.record("CCO")
    assert predictor.nearest(ethanol["exact_mass"], 3)[0] == "CCO" and predictor.nearest(float("nan"), 3) == [] and "CCO" not in predictor.nearest(ethanol["exact_mass"], 2, frozenset([ethanol["inchikey14"]]))
    guesses, scores = predictor.predict(df[df["molecule_id"] == "m_toluene"], "m_toluene")
    assert len(guesses) == 5 and len(scores) == 3 and len({chem.inchikey14(g) for g in guesses}) == 5
    assert chem.record("Cc1ccccc1")["inchikey14"] in {chem.inchikey14(g) for g in guesses[:3]}
    predictions, collected = predictor.predict_frame(df)
    assert set(predictions) == set(df["molecule_id"]) and all(1 <= len(v) <= 5 for v in predictions.values()) and "m_bad_adduct" not in collected
    frame = predictor.submission(df, tmp_path / "submission.csv")
    assert len(frame) == df["molecule_id"].nunique() and 0 <= Submission.read(tmp_path / "submission.csv").score(validation.truth_map(df)) <= 1
    fake = lambda _trainer, _spectra, formulas: ["CCO", "COC", "garbage"] if formulas == ["C2H6O"] else []
    guesses, _ = Predictor(trainer, pool, vocab, config=FAST, denovo=fake).predict(df[df["molecule_id"] == "m_ethanol"], "m_ethanol")
    assert guesses[:2] == ["CCO", "COC"] and len(guesses) == 5  # pool first, new de novo structure next, mass top-up after


def test_validation(setup):
    df, vocab, pool, trainer = setup
    df = validation.with_keys(df)
    train_keys, holdout_keys = validation.split_keys(df["inchikey14"], 0.25, seed=1)
    train_df, holdout_df = validation.split_frame(df, holdout_keys)
    assert train_keys.isdisjoint(holdout_keys) and set(train_df["inchikey14"]).isdisjoint(holdout_df["inchikey14"]) and len(train_df) + len(holdout_df) == len(df)
    with pytest.raises(ValueError):
        validation.split_keys([], 0.1)
    predictor = Predictor(trainer, pool, vocab, config=FAST)
    truth = validation.truth_map(df)
    _, scores = predictor.predict(df[df["molecule_id"] == "m_toluene"], "m_toluene")
    assert validation.labels(scores, pool, truth["m_toluene"]).sum() == 1 and validation.labels(scores, pool, None).sum() == 0
    assert len(validation.training_groups({"m_toluene": scores}, pool, truth)) == 1
    holdout = df[df["molecule_id"].isin(["m_toluene", "m_ethanol"])].reset_index(drop=True)
    reports = validation.validate(predictor, holdout)
    known, novel = reports["known_structure"], reports["novel_structure"]
    assert known.n_molecules == 2 and known.ci_low <= known.mrr <= known.ci_high and known.candidate_recall == 1.0
    assert novel.mrr == 0.0 and novel.candidate_recall == 0.0 and "MRR@25" in str(known)  # answers removed: nothing to retrieve
    assert validation.evaluate("x", {"a": ["CCO"]}, {"a": "OCC", "b": "C"}).mrr == pytest.approx(0.5)


def _one_hot(adjacency: np.ndarray, confidence: float = 0.9) -> np.ndarray:
    n = adjacency.shape[0]
    probs = np.full((n, n, denovo.N_CLASSES), (1 - confidence) / (denovo.N_CLASSES - 1), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            probs[i, j, int(adjacency[i, j])] = confidence
    return probs


def test_bond_type_sgno_and_decode():
    torch.manual_seed(0)
    sgno = denovo.BondTypeSGNO(k=K, hidden_dim=32, num_layers=3)
    logits = sgno(torch.randn(2, 5, K), torch.randint(0, 12, (2, 5)))
    assert logits.shape == (2, 5, 5, 4) and torch.allclose(logits, logits.transpose(1, 2))
    assert torch.allclose(torch.diagonal(sgno.probabilities(torch.randn(2, 5, K), torch.zeros(2, 5, dtype=torch.long)), dim1=1, dim2=2)[:, 0], torch.ones(2, 5))
    adj, atoms, mask = chem.kekulized_adjacency("CC(=O)O"), ["C", "C", "O", "O"], torch.ones(1, 4, dtype=torch.bool)
    assert sorted(denovo.bond_targets([adj], mask)[0][torch.triu(torch.ones(4, 4, dtype=torch.bool), 1)].tolist()) == [0, 0, 0, 1, 1, 2]
    trainer, opt = denovo.BondTypeTrainer(sgno), torch.optim.Adam(sgno.parameters(), lr=1e-2)
    eigvecs, elements = torch.from_numpy(SpectralDataProcessor(k=K).process_smiles("CC(=O)O")).unsqueeze(0), torch.tensor([[ELEMENT_TO_INDEX[a] for a in atoms]])
    losses = [trainer.step(opt, eigvecs, mask, elements, [adj]) for _ in range(60)]
    assert losses[-1] < losses[0]
    decoded = denovo.decode(sgno.probabilities(eigvecs, elements)[0].detach().numpy(), atoms, h_budget=4)
    assert chem.to_smiles(decoded, atoms) == "CC(=O)O"
    for smiles in ("c1ccccc1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "C#N"):
        rec = chem.record(smiles)
        atoms = chem.symbols(rec["smiles"])
        out = chem.to_smiles(denovo.decode(_one_hot(chem.kekulized_adjacency(rec["smiles"])), atoms, h_budget=denovo.hydrogens_in(rec["formula"])), atoms)
        assert chem.inchikey14(out) == rec["inchikey14"], smiles


def test_hydrogen_budget_repair():
    atoms, single = ["C", "C", "O", "O"], np.zeros((4, 4), np.float32)
    for i, j in ((0, 1), (1, 2), (1, 3)):
        single[i, j] = single[j, i] = 1
    probs = _one_hot(single, 0.6)
    probs[1, 2, 2] = probs[2, 1, 2] = 0.3  # a double bond to atom 2 is the most plausible promotion
    assert denovo.implied_hydrogens(single, atoms) == 6
    repaired = denovo.decode(probs, atoms, h_budget=4)
    assert denovo.implied_hydrogens(repaired, atoms) == 4 and repaired[1, 2] == 2 and chem.to_smiles(repaired, atoms) == "CC(=O)O"
    demoted = denovo.decode(_one_hot(np.array([[0, 2], [2, 0]], np.float32), 0.6), ["C", "C"], h_budget=6)
    assert demoted[0, 1] == 1
    with pytest.raises(ValueError):
        denovo.decode(np.zeros((2, 2, 3)), ["C", "C"])


def test_formula_enumeration():
    glucose, caffeine = chem.record("OCC1OC(O)C(O)C(O)C1O"), chem.record("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    formulas = denovo.enumerate_formulas(glucose["exact_mass"], 5.0, ("C", "N", "O"))
    assert formulas[0] == "C6H12O6"
    for f in formulas:
        from spec2graph.eval.decode import parse_formula

        counts = {**parse_formula(f), "H": denovo.hydrogens_in(f)}
        assert abs(sum(denovo.MASSES[e] * n for e, n in counts.items()) - glucose["exact_mass"]) <= glucose["exact_mass"] * 5e-6 and denovo.rdbe(counts).is_integer()
    assert "C8H10N4O2" in denovo.enumerate_formulas(caffeine["exact_mass"], 3.0, max_results=20) and denovo.enumerate_formulas(float("nan")) == []
    assert (denovo.hydrogens_in("C2H6O"), denovo.hydrogens_in("CO2"), denovo.hydrogens_in("CHCl3"), denovo.hydrogens_in("HgO")) == (6, 0, 1, 0)
    assert denovo.rdbe({"C": 6, "H": 6}) == 4.0


def test_denovo_end_to_end(setup, tmp_path):
    df, vocab, pool, trainer = setup
    sgno = denovo.BondTypeSGNO(k=K, hidden_dim=16, num_layers=2)
    spectra = data.Spectra.from_rows(df[df["molecule_id"] == "m_acetic"], vocab, MAX_PEAKS)
    out = denovo.DeNovo(sgno, n_samples=6, n_steps=2, max_per_formula=5)(trainer, spectra, ["C2H4O2", "C3H8", "not a formula", "C200H2"])
    keys = [chem.inchikey14(s) for s in out]
    assert all(keys) and len(set(keys)) == len(keys) and all("." not in s for s in out)
    denovo.save_bond_sgno(tmp_path / "bond.pt", sgno)
    loaded = denovo.load_bond_sgno(tmp_path / "bond.pt")
    assert loaded.hparams == sgno.hparams and all(torch.equal(loaded.state_dict()[k], v) for k, v in sgno.state_dict().items())
