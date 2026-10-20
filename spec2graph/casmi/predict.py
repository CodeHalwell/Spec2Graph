"""Spectra -> ranked SMILES -> submission, with per-molecule fallbacks so one failure never zeroes the file."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from spec2graph.casmi.adducts import AdductVocabulary
from spec2graph.casmi.chem import inchikey14
from spec2graph.casmi.conditioning import CasmiTrainer, load_checkpoint
from spec2graph.casmi.data import Peaks, Spectra, by_molecule
from spec2graph.casmi.metric import MAX_GUESSES, Submission, clean_guesses
from spec2graph.casmi.pool import CandidatePool
from spec2graph.casmi.ranker import LinearRanker, rank
from spec2graph.casmi.scoring import CandidateScores, Scorer, ScoringConfig

logger = logging.getLogger(__name__)
DeNovoFn = Callable[[CasmiTrainer, Spectra, list[str]], list[str]]


@dataclass
class PredictConfig:
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    peaks: Peaks = field(default_factory=Peaks)
    max_peaks: int = 128
    top_k: int = MAX_GUESSES
    fallback_ppm: float = 50.0  # retry window when the strict one is empty
    denovo_formulas: int = 3


def median_mass(rows: pd.DataFrame) -> float:
    masses = pd.to_numeric(rows["neutral_mass"], errors="coerce").dropna()
    return float(masses.median()) if len(masses) else float("nan")


class Predictor:
    """Scores, ranks and tops up candidates for every molecule of a prepared spectra frame."""

    def __init__(self, trainer: CasmiTrainer, pool: CandidatePool, vocab: AdductVocabulary, ranker: Optional[LinearRanker] = None,
                 config: PredictConfig = PredictConfig(), denovo: Optional[DeNovoFn] = None):
        self.trainer, self.pool, self.vocab, self.cfg, self.denovo = trainer, pool, vocab, config, denovo
        self.ranker = ranker or LinearRanker.heuristic()
        self.scorer = Scorer(trainer, pool, config.scoring)

    @classmethod
    def load(cls, checkpoint, pool, ranker=None, bond_sgno=None, device: str = "cpu", **scoring) -> "Predictor":
        """Build from files (the Kaggle-notebook entry point); keyword overrides go to :class:`ScoringConfig`."""
        model, trainer_cfg, vocab, extra = load_checkpoint(checkpoint, device)
        peaks = Peaks(top_k=int(extra.get("max_peaks", model.core.max_peaks)), sqrt_intensity=bool(extra.get("sqrt_intensity", False)))
        config = PredictConfig(scoring=ScoringConfig(**scoring), peaks=peaks, max_peaks=peaks.top_k)
        denovo = None
        if bond_sgno is not None:
            from spec2graph.casmi.denovo import DeNovo, load_bond_sgno

            denovo = DeNovo(load_bond_sgno(bond_sgno, device), n_steps=config.scoring.n_steps)
        return cls(CasmiTrainer(model, trainer_cfg, device), CandidatePool.load(pool), vocab, LinearRanker.load(ranker) if ranker else None, config, denovo)

    def with_pool(self, pool: CandidatePool) -> "Predictor":
        return Predictor(self.trainer, pool, self.vocab, self.ranker, self.cfg, self.denovo)

    def nearest(self, mass: float, k: int, exclude: frozenset[str] = frozenset()) -> list[str]:
        """``k`` pool SMILES closest in mass (any window), skipping excluded InChIKey14s."""
        if not np.isfinite(mass) or k <= 0 or not len(self.pool):
            return []
        order = np.argsort(np.abs(self.pool.exact_mass - mass), kind="stable")[: k + len(exclude)]
        return [str(self.pool.smiles[i]) for i in order if self.pool.inchikey14[i] not in exclude][:k]

    def predict(self, rows: pd.DataFrame, molecule_id: str = "") -> tuple[list[str], Optional[CandidateScores]]:
        """Ranked guesses for one molecule plus its candidate scores (``None`` without usable spectra)."""
        mass = median_mass(rows)
        spectra = Spectra.from_rows(rows, self.vocab, self.cfg.max_peaks, self.cfg.peaks, self.trainer.device)
        if spectra is None:
            return self.nearest(mass, self.cfg.top_k), None
        scores = self.scorer.score(spectra, molecule_id)
        if len(scores) == 0 and self.cfg.fallback_ppm > self.cfg.scoring.ppm:
            scores = Scorer(self.trainer, self.pool, replace(self.cfg.scoring, ppm=self.cfg.fallback_ppm)).score(spectra, molecule_id)
        guesses = rank(scores, self.pool, self.ranker, self.cfg.top_k)
        if self.denovo is not None:
            try:
                guesses += self.denovo(self.trainer, spectra, self.pool.formulas_in_window(mass, self.cfg.scoring.ppm)[: self.cfg.denovo_formulas])
            except Exception as exc:  # de novo is best effort
                logger.warning("de novo failed for %s: %s", molecule_id, exc)
        guesses = clean_guesses(guesses, self.cfg.top_k)
        keys = frozenset(filter(None, map(inchikey14, guesses)))
        return guesses + self.nearest(mass, self.cfg.top_k - len(guesses), keys), scores

    def predict_frame(self, df: pd.DataFrame, progress: bool = False) -> tuple[dict[str, list[str]], dict[str, CandidateScores]]:
        groups = by_molecule(df).items()
        predictions, scores = {}, {}
        for molecule_id, pos in (tqdm(list(groups), desc="predict", unit="mol") if progress else groups):
            rows = df.iloc[pos]
            try:
                guesses, s = self.predict(rows, molecule_id)
            except Exception as exc:
                logger.exception("prediction failed for %s: %s", molecule_id, exc)
                guesses, s = self.nearest(median_mass(rows), self.cfg.top_k), None
            predictions[molecule_id] = guesses or ["C"]
            if s is not None:
                scores[molecule_id] = s
        return predictions, scores

    def submission(self, df: pd.DataFrame, path) -> pd.DataFrame:
        predictions, _ = self.predict_frame(df, progress=True)
        return Submission(predictions, list(dict.fromkeys(df["molecule_id"]))).write(path)
