"""Tests for :mod:`spec2graph.casmi.ranker`, :mod:`spec2graph.casmi.predict` and
:mod:`spec2graph.casmi.validation`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")
torch = pytest.importorskip("torch")

from spec2graph.casmi.adducts import AdductVocabulary
from spec2graph.casmi.conditioning import CasmiModel, CasmiModelConfig, CasmiTrainer, CasmiTrainerConfig
from spec2graph.casmi.data import build_structure_table, prepare_spectra_frame, structure_record
from spec2graph.casmi.metric import inchikey14, read_submission, score_submission
from spec2graph.casmi.pool import CandidatePool
from spec2graph.casmi.predict import (
    LAST_RESORT_SMILES,
    PredictConfig,
    make_submission,
    nearest_mass_candidates,
    predict_frame,
    predict_molecule,
)
from spec2graph.casmi.ranker import DEFAULT_WEIGHTS, LinearRanker, rank_candidates, ranked_indices
from spec2graph.casmi.scoring import FEATURE_NAMES, CandidateScores, ScoringConfig
from spec2graph.casmi.validation import (
    add_inchikey14_column,
    evaluate_predictions,
    labels_for,
    molecule_disjoint_split,
    novelty_scenarios,
    ranker_training_groups,
    run_validation,
    split_frame,
    truth_map,
)
from spectral_diffusion import Spec2GraphDiffusion, Spec2GraphDiffusionConfig
from tests.conftest import CASMI_MOLECULES

K = 4
FAST = ScoringConfig(ppm=20.0, n_samples=1, n_steps=2, max_atom_counts=2, alignment_iters=2, sinkhorn_iters=5)


def _trainer(n_adducts: int) -> CasmiTrainer:
    torch.manual_seed(0)
    core = Spec2GraphDiffusion(
        Spec2GraphDiffusionConfig(
            d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32,
            k=K, max_atoms=16, max_peaks=32, dropout=0.0, fingerprint_dim=64,
            enable_atom_count_head=True, enable_atom_type_head=True,
        )
    )
    return CasmiTrainer(CasmiModel(core, CasmiModelConfig(n_adducts=n_adducts, n_quantiles=8)), CasmiTrainerConfig(n_timesteps=10))


@pytest.fixture
def setup(casmi_frame):
    df = prepare_spectra_frame(casmi_frame, require_smiles=True)
    vocab = AdductVocabulary.fit(df["adduct"])
    table = build_structure_table([s for _, s in CASMI_MOLECULES] + ["C1=CC=CC=CC1", "C1=CC2CC1C=C2", "CCN"])
    pool = CandidatePool.build(table, k=K, fingerprint_bits=64, n_quantiles=8)
    return df, vocab, pool, _trainer(len(vocab))


class TestRanker:
    def test_heuristic_scores_and_ordering(self):
        ranker = LinearRanker.heuristic()
        assert ranker.feature_names == FEATURE_NAMES
        features = np.full((3, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
        col = FEATURE_NAMES.index("fp_loglik")
        features[:, col] = [-0.5, -0.1, -0.9]
        scores = ranker.score(features)
        assert np.argmax(scores) == 1 and np.argmin(scores) == 2
        # NaNs impute to the mean (zero after standardisation).
        assert np.isfinite(scores).all()
        with pytest.raises(ValueError):
            ranker.score(np.zeros((2, 3)))

    def test_fit_learns_informative_feature(self, tmp_path):
        rng = np.random.default_rng(0)
        groups = []
        col = FEATURE_NAMES.index("quantile_dist")
        for _ in range(30):
            n = 6
            features = rng.normal(size=(n, len(FEATURE_NAMES))).astype(np.float32)
            labels = np.zeros(n, dtype=bool)
            labels[0] = True
            # The true candidate always has the *largest* quantile distance here,
            # the opposite of the heuristic sign, so the fit must flip it.
            features[:, col] = rng.uniform(0, 1, size=n)
            features[0, col] = 2.0
            groups.append((features, labels))
        # A single-candidate group carries no ranking information and is ignored.
        groups.append((np.ones((1, len(FEATURE_NAMES)), dtype=np.float32), np.array([True])))
        ranker = LinearRanker.fit(groups, epochs=200, lr=0.1)
        assert ranker.weights[col] > 0
        top1 = np.mean([np.argmax(ranker.score(f)) == 0 for f, _ in groups])
        assert top1 > 0.9
        path = tmp_path / "ranker.json"
        ranker.save(path)
        loaded = LinearRanker.load(path)
        np.testing.assert_allclose(loaded.weights, ranker.weights)
        np.testing.assert_allclose(loaded.mean, ranker.mean)
        with pytest.raises(ValueError):
            LinearRanker.fit([(np.zeros((2, len(FEATURE_NAMES))), np.zeros(2, dtype=bool))])
        with pytest.raises(ValueError):
            LinearRanker.fit([(np.zeros((1, len(FEATURE_NAMES))), np.ones(1, dtype=bool))])

    def test_rank_candidates_dedupes_and_caps(self, setup):
        _, _, pool, _ = setup
        ranker = LinearRanker.heuristic()
        indices = np.arange(len(pool))
        features = np.zeros((len(pool), len(FEATURE_NAMES)), dtype=np.float32)
        features[:, FEATURE_NAMES.index("proj_overlap")] = np.linspace(0, 1, len(pool))
        scores = CandidateScores("m", indices, features, FEATURE_NAMES, 100.0, 1)
        ordered = ranked_indices(scores, ranker)
        assert ordered[0] == len(pool) - 1
        top = rank_candidates(scores, pool, ranker, k=3)
        assert len(top) == 3 and top[0] == pool.smiles[len(pool) - 1]
        assert rank_candidates(CandidateScores.empty("m", 1.0, 1), pool, ranker) == []


class TestPredict:
    def test_nearest_mass(self, setup):
        _, _, pool, _ = setup
        ethanol = structure_record("CCO")
        nearest = nearest_mass_candidates(pool, ethanol["exact_mass"], k=3)
        assert nearest[0] == "CCO" and len(nearest) == 3
        assert nearest_mass_candidates(pool, float("nan"), k=3) == []
        excluded = nearest_mass_candidates(pool, ethanol["exact_mass"], k=2, exclude={ethanol["inchikey14"]})
        assert "CCO" not in excluded

    def test_predict_molecule_and_frame(self, setup, tmp_path):
        df, vocab, pool, trainer = setup
        ranker = LinearRanker.heuristic()
        config = PredictConfig(scoring=FAST, max_peaks=32, top_k=5)
        rows = df[df["molecule_id"] == "m_toluene"]
        guesses, scores = predict_molecule(trainer, pool, rows, molecule_id="m_toluene", adduct_vocab=vocab, ranker=ranker, config=config)
        assert len(guesses) == 5 and scores is not None and len(scores) == 3
        assert len({inchikey14(g) for g in guesses}) == 5
        assert structure_record("Cc1ccccc1")["inchikey14"] in {inchikey14(g) for g in guesses[:3]}

        predictions, collected = predict_frame(
            trainer, pool, df, adduct_vocab=vocab, ranker=ranker, config=config, collect_scores=True
        )
        assert set(predictions) == set(df["molecule_id"])
        assert all(1 <= len(v) <= 5 for v in predictions.values())
        # The unsupported-adduct molecule still gets a guess.
        assert predictions["m_bad_adduct"] in ([LAST_RESORT_SMILES],) or len(predictions["m_bad_adduct"]) >= 1
        assert "m_bad_adduct" not in collected
        path = tmp_path / "submission.csv"
        submission = make_submission(predictions, df, path)
        assert len(submission) == df["molecule_id"].nunique()
        loaded = read_submission(path)
        assert set(loaded) == set(predictions)
        truth = truth_map(df)
        assert 0.0 <= score_submission(path, truth) <= 1.0

    def test_denovo_hook_appends_new_structures(self, setup):
        df, vocab, pool, trainer = setup
        ranker = LinearRanker.heuristic()
        config = PredictConfig(scoring=FAST, max_peaks=32, top_k=4, denovo=True)
        rows = df[df["molecule_id"] == "m_ethanol"]

        def fake_denovo(_trainer, _tensors, formulas):
            assert formulas == ["C2H6O"]
            return ["CCO", "COC", "garbage"]

        guesses, _ = predict_molecule(
            trainer, pool, rows, molecule_id="m_ethanol", adduct_vocab=vocab, ranker=ranker, config=config, denovo_fn=fake_denovo
        )
        # Pool candidate first, then the genuinely new de novo structure, then mass top-up.
        assert guesses[0] == "CCO" and guesses[1] == "COC" and len(guesses) == 4


class TestValidation:
    def test_split_is_molecule_disjoint(self, setup):
        df, _, _, _ = setup
        df = add_inchikey14_column(df)
        train_keys, holdout_keys = molecule_disjoint_split(df["inchikey14"], holdout_fraction=0.25, seed=1)
        assert train_keys.isdisjoint(holdout_keys) and holdout_keys
        train_df, holdout_df = split_frame(df, holdout_keys)
        assert set(train_df["inchikey14"]).isdisjoint(set(holdout_df["inchikey14"]))
        assert len(train_df) + len(holdout_df) == len(df)
        with pytest.raises(ValueError):
            molecule_disjoint_split([], holdout_fraction=0.1)

    def test_scenarios_and_labels(self, setup):
        df, vocab, pool, trainer = setup
        truth = truth_map(df)
        holdout_keys = {inchikey14("Cc1ccccc1")}
        scenarios = novelty_scenarios(pool, holdout_keys)
        assert len(scenarios["novel_structure"]) == len(pool) - 1
        rows = df[df["molecule_id"] == "m_toluene"]
        _, scores = predict_molecule(
            trainer, pool, rows, molecule_id="m_toluene", adduct_vocab=vocab, ranker=LinearRanker.heuristic(),
            config=PredictConfig(scoring=FAST, max_peaks=32),
        )
        labels = labels_for(scores, pool, inchikey14(truth["m_toluene"]))
        assert labels.sum() == 1
        groups = ranker_training_groups({"m_toluene": scores}, pool, truth)
        assert len(groups) == 1 and groups[0][1].sum() == 1
        assert labels_for(scores, pool, None).sum() == 0

    def test_run_validation_reports(self, setup):
        df, vocab, pool, trainer = setup
        holdout = df[df["molecule_id"].isin(["m_toluene", "m_ethanol"])].reset_index(drop=True)
        reports = run_validation(
            trainer, pool, holdout, adduct_vocab=vocab, ranker=LinearRanker.heuristic(),
            config=PredictConfig(scoring=FAST, max_peaks=32),
        )
        known, novel = reports["known_structure"], reports["novel_structure"]
        assert known.n_molecules == novel.n_molecules == 2
        assert 0.0 <= known.mrr <= 1.0 and known.ci_low <= known.mrr <= known.ci_high
        assert known.candidate_recall == 1.0
        # With the answers removed from the pool nothing can be retrieved.
        assert novel.mrr == 0.0 and novel.candidate_recall == 0.0
        assert "MRR@25" in str(known)
        report = evaluate_predictions("x", {"a": ["CCO"]}, {"a": "OCC", "b": "C"})
        assert report.mrr == pytest.approx(0.5) and report.top1 == pytest.approx(0.5)
