"""Molecule-disjoint validation under the competition's novelty classes.

``known_structure`` keeps the answers in the pool (classes 1 and 2; this layer has no spectral-library search, so
they coincide); ``novel_structure`` removes them (class 3), where only de novo generation can score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from spec2graph.casmi.chem import inchikey14
from spec2graph.casmi.metric import bootstrap_ci, reciprocal_ranks
from spec2graph.casmi.pool import CandidatePool
from spec2graph.casmi.predict import Predictor
from spec2graph.casmi.scoring import CandidateScores

logger = logging.getLogger(__name__)


def with_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with an ``inchikey14`` column derived from ``smiles``."""
    return df.assign(inchikey14=pd.Series([inchikey14(s) for s in df["smiles"]], index=df.index, dtype=object))


def split_keys(keys: Iterable[str], holdout_fraction: float = 0.1, seed: int = 0) -> tuple[set[str], set[str]]:
    """Distinct structure keys -> ``(train, holdout)``."""
    unique = sorted({k for k in keys if isinstance(k, str)})
    if not unique or not 0 < holdout_fraction < 1:
        raise ValueError("need structure keys and a holdout fraction in (0, 1)")
    holdout = set(np.random.default_rng(seed).choice(unique, max(1, round(holdout_fraction * len(unique))), replace=False))
    return set(unique) - holdout, holdout


def split_frame(df: pd.DataFrame, holdout_keys: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df if "inchikey14" in df.columns else with_keys(df)
    mask = df["inchikey14"].isin(holdout_keys)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def truth_map(df: pd.DataFrame) -> dict[str, str]:
    """``molecule_id -> answer SMILES`` (first row per molecule)."""
    first = df.drop_duplicates("molecule_id")
    return dict(zip(first["molecule_id"].astype(str), first["smiles"]))


def labels(scores: CandidateScores, pool: CandidatePool, answer: Optional[str]) -> np.ndarray:
    """Boolean mask of the candidates matching ``answer``."""
    key = inchikey14(answer)
    return np.asarray([pool.inchikey14[i] == key for i in scores.indices], dtype=bool) if key else np.zeros(len(scores), bool)


def training_groups(scores: dict[str, CandidateScores], pool: CandidatePool, truth: dict[str, str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(features, is_correct)`` per molecule whose answer is in its window."""
    groups = [(s.features, labels(s, pool, truth.get(m))) for m, s in scores.items()]
    return [(f, y) for f, y in groups if y.any()]


@dataclass
class Report:
    scenario: str
    n_molecules: int
    mrr: float
    ci_low: float
    ci_high: float
    top1: float
    candidate_recall: float

    def __str__(self) -> str:
        return (f"{self.scenario}: n={self.n_molecules} MRR@25={self.mrr:.4f} [{self.ci_low:.4f}, {self.ci_high:.4f}] "
                f"top1={self.top1:.3f} recall@window={self.candidate_recall:.3f}")


def evaluate(scenario: str, predictions: dict, truth: dict[str, str], scores: Optional[dict] = None,
             pool: Optional[CandidatePool] = None) -> Report:
    rr = np.array(list(reciprocal_ranks(truth, predictions).values()))
    recall = float("nan")
    if scores is not None and pool is not None:
        recall = float(np.mean([m in scores and labels(scores[m], pool, a).any() for m, a in truth.items()]))
    return Report(scenario, len(rr), float(rr.mean()), *bootstrap_ci(rr, n=500), float((rr == 1).mean()), recall)


def validate(predictor: Predictor, holdout: pd.DataFrame, scenarios: Sequence[str] = ("known_structure", "novel_structure"),
             progress: bool = False) -> dict[str, Report]:
    """Predict the held-out molecules with the answers in the pool and with them removed."""
    truth = truth_map(holdout)
    keys = {k for k in map(inchikey14, truth.values()) if k}
    pools = {"known_structure": predictor.pool, "novel_structure": predictor.pool.without(keys)}
    reports = {}
    for name in scenarios:
        predictions, scores = predictor.with_pool(pools[name]).predict_frame(holdout, progress)
        reports[name] = evaluate(name, predictions, truth, scores, pools[name])
        logger.info("%s", reports[name])
    return reports
