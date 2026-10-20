"""Molecule-disjoint validation with the competition's novelty classes.

A random split of *spectra* leaks: other spectra of the same molecule stay
in training and every score reads far too high. Splits here are by
structure (InChIKey14), and the hidden test set's three novelty classes are
simulated on the held-out molecules:

``known_structure``
    the answer is in the candidate pool (the competition's classes 1 and 2;
    this layer has no spectral-library search, so the two coincide here);
``novel_structure``
    the answer is removed from the pool (class 3), where only de novo
    generation can score and retrieval is zero by construction.

Report both, and report candidate recall alongside MRR: a structure the
window never proposes can never be ranked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from spec2graph.casmi.adducts import AdductVocabulary
from spec2graph.casmi.conditioning import CasmiTrainer
from spec2graph.casmi.metric import (
    bootstrap_mrr_ci,
    inchikey14,
    per_molecule_reciprocal_ranks,
)
from spec2graph.casmi.pool import CandidatePool
from spec2graph.casmi.predict import PredictConfig, predict_frame
from spec2graph.casmi.ranker import LinearRanker
from spec2graph.casmi.scoring import CandidateScores

logger = logging.getLogger(__name__)


def add_inchikey14_column(spectra: pd.DataFrame) -> pd.DataFrame:
    """Copy of a training frame with an ``inchikey14`` column from ``smiles``."""
    if "smiles" not in spectra.columns:
        raise ValueError("add_inchikey14_column needs a 'smiles' column.")
    out = spectra.copy()
    out["inchikey14"] = pd.Series([inchikey14(s) for s in out["smiles"]], index=out.index, dtype=object)
    return out


def molecule_disjoint_split(
    keys: Iterable[str],
    *,
    holdout_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[set[str], set[str]]:
    """Split distinct structure keys into ``(train_keys, holdout_keys)``."""
    unique = sorted({k for k in keys if isinstance(k, str) and k})
    if not unique:
        raise ValueError("No structure keys to split.")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1).")
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(len(unique))
    n_holdout = max(1, int(round(holdout_fraction * len(unique))))
    holdout = {unique[i] for i in permuted[:n_holdout]}
    train = set(unique) - holdout
    return train, holdout


def split_frame(spectra: pd.DataFrame, holdout_keys: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition a frame with an ``inchikey14`` column by structure."""
    if "inchikey14" not in spectra.columns:
        spectra = add_inchikey14_column(spectra)
    mask = spectra["inchikey14"].isin(holdout_keys)
    return spectra[~mask].reset_index(drop=True), spectra[mask].reset_index(drop=True)


def truth_map(holdout: pd.DataFrame) -> dict[str, str]:
    """``molecule_id -> answer SMILES`` (first row per molecule)."""
    out: dict[str, str] = {}
    for molecule_id, smiles in zip(holdout["molecule_id"].astype(str), holdout["smiles"]):
        out.setdefault(molecule_id, str(smiles))
    return out


def novelty_scenarios(pool: CandidatePool, holdout_keys: Iterable[str]) -> dict[str, CandidatePool]:
    return {
        "known_structure": pool,
        "novel_structure": pool.without_keys(holdout_keys),
    }


def labels_for(scores: CandidateScores, pool: CandidatePool, truth_key: Optional[str]) -> np.ndarray:
    """Boolean mask over ``scores`` candidates matching ``truth_key``."""
    if truth_key is None or len(scores) == 0:
        return np.zeros(len(scores), dtype=bool)
    return np.asarray([str(pool.inchikey14[i]) == truth_key for i in scores.indices], dtype=bool)


def ranker_training_groups(
    scores_by_molecule: dict[str, CandidateScores],
    pool: CandidatePool,
    truth: dict[str, str],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(features, labels)`` per molecule whose answer is in its window."""
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for molecule_id, scores in scores_by_molecule.items():
        key = inchikey14(truth.get(molecule_id))
        labels = labels_for(scores, pool, key)
        if labels.any():
            groups.append((scores.features, labels))
    return groups


@dataclass
class ValidationReport:
    scenario: str
    n_molecules: int
    mrr: float
    ci_low: float
    ci_high: float
    top1: float
    candidate_recall: float
    per_molecule: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "n_molecules": self.n_molecules,
            "mrr_at_25": self.mrr,
            "ci95": [self.ci_low, self.ci_high],
            "top1": self.top1,
            "candidate_recall": self.candidate_recall,
        }

    def __str__(self) -> str:
        return (
            f"{self.scenario}: n={self.n_molecules} MRR@25={self.mrr:.4f} "
            f"[{self.ci_low:.4f}, {self.ci_high:.4f}] top1={self.top1:.3f} "
            f"recall@window={self.candidate_recall:.3f}"
        )


def evaluate_predictions(
    scenario: str,
    predictions: dict[str, Sequence[str]],
    truth: dict[str, str],
    *,
    scores_by_molecule: Optional[dict[str, CandidateScores]] = None,
    pool: Optional[CandidatePool] = None,
    n_bootstrap: int = 500,
) -> ValidationReport:
    ranks = per_molecule_reciprocal_ranks(truth, predictions)
    values = list(ranks.values())
    low, high = bootstrap_mrr_ci(values, n_bootstrap=n_bootstrap)
    recall = float("nan")
    if scores_by_molecule is not None and pool is not None and truth:
        hits = 0
        for molecule_id, answer in truth.items():
            scores = scores_by_molecule.get(molecule_id)
            if scores is not None and labels_for(scores, pool, inchikey14(answer)).any():
                hits += 1
        recall = hits / len(truth)
    return ValidationReport(
        scenario=scenario,
        n_molecules=len(truth),
        mrr=float(np.mean(values)),
        ci_low=low,
        ci_high=high,
        top1=float(np.mean([v == 1.0 for v in values])),
        candidate_recall=recall,
        per_molecule=ranks,
    )


def run_validation(
    trainer: CasmiTrainer,
    pool: CandidatePool,
    holdout: pd.DataFrame,
    *,
    adduct_vocab: AdductVocabulary,
    ranker: LinearRanker,
    config: Optional[PredictConfig] = None,
    scenarios: Sequence[str] = ("known_structure", "novel_structure"),
    denovo_fn: Optional[Callable] = None,
    progress: bool = False,
) -> dict[str, ValidationReport]:
    """Predict the held-out molecules under each novelty scenario and score them."""
    if "smiles" not in holdout.columns:
        raise ValueError("Validation needs answers; the holdout frame has no 'smiles' column.")
    truth = truth_map(holdout)
    holdout_keys = {k for k in (inchikey14(s) for s in truth.values()) if k is not None}
    pools = novelty_scenarios(pool, holdout_keys)
    reports: dict[str, ValidationReport] = {}
    for name in scenarios:
        if name not in pools:
            raise ValueError(f"Unknown scenario {name!r}; expected one of {sorted(pools)}.")
        predictions, scores = predict_frame(
            trainer,
            pools[name],
            holdout,
            adduct_vocab=adduct_vocab,
            ranker=ranker,
            config=config,
            denovo_fn=denovo_fn,
            progress=progress,
            collect_scores=True,
        )
        reports[name] = evaluate_predictions(name, predictions, truth, scores_by_molecule=scores, pool=pools[name])
        logger.info("%s", reports[name])
    return reports


__all__ = [
    "ValidationReport",
    "add_inchikey14_column",
    "evaluate_predictions",
    "labels_for",
    "molecule_disjoint_split",
    "novelty_scenarios",
    "ranker_training_groups",
    "run_validation",
    "split_frame",
    "truth_map",
]
