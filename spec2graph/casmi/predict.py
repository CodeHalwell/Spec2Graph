"""End-to-end prediction: test spectra -> ranked SMILES -> ``submission.csv``.

Per molecule:

1. preprocess its spectra and build conditioning (:func:`spectra_to_tensors`);
2. score the mass-window candidates (:func:`score_molecule`);
3. rank them (:func:`rank_candidates`);
4. optionally append de novo structures for formulas the pool lacks
   (:mod:`spec2graph.casmi.denovo`, when a bond-type decoder is supplied);
5. top up to 25 with the mass-nearest pool structures.

Every molecule is wrapped in a ``try``: one failure must never zero the
whole submission, so the fallback is the mass-nearest candidates, and if
even that fails, a single valid SMILES.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

from spec2graph.casmi.adducts import AdductVocabulary
from spec2graph.casmi.conditioning import CasmiTrainer
from spec2graph.casmi.data import PeakPreprocessConfig, group_by_molecule, spectra_to_tensors
from spec2graph.casmi.metric import MAX_GUESSES, inchikey14, write_submission
from spec2graph.casmi.pool import CandidatePool
from spec2graph.casmi.ranker import LinearRanker, rank_candidates
from spec2graph.casmi.scoring import CandidateScores, ScoringConfig, score_molecule

logger = logging.getLogger(__name__)

# A valid, harmless SMILES for the very last resort.
LAST_RESORT_SMILES = "C"

# Signature of an optional de novo generator: (trainer, tensors, formulas) -> SMILES list.
DeNovoFn = Callable[[CasmiTrainer, Any, list[str]], list[str]]


@dataclass
class PredictConfig:
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    peak_config: PeakPreprocessConfig = field(default_factory=PeakPreprocessConfig)
    max_peaks: int = 128
    top_k: int = MAX_GUESSES
    # If the strict window is empty, retry once with this tolerance.
    fallback_ppm: float = 50.0
    # Append de novo structures when a generator is supplied.
    denovo: bool = False
    denovo_max_formulas: int = 3


def nearest_mass_candidates(pool: CandidatePool, neutral_mass: float, k: int, exclude: Iterable[str] = ()) -> list[str]:
    """``k`` pool SMILES closest in mass to ``neutral_mass`` (any window)."""
    if not np.isfinite(neutral_mass) or len(pool) == 0:
        return []
    excluded = set(exclude)
    position = int(np.searchsorted(pool.exact_mass, neutral_mass))
    lo, hi = position - 1, position
    out: list[str] = []
    while len(out) < k and (lo >= 0 or hi < len(pool)):
        take_lo = hi >= len(pool) or (
            lo >= 0 and abs(pool.exact_mass[lo] - neutral_mass) <= abs(pool.exact_mass[hi] - neutral_mass)
        )
        index = lo if take_lo else hi
        if take_lo:
            lo -= 1
        else:
            hi += 1
        key = str(pool.inchikey14[index])
        if key in excluded:
            continue
        excluded.add(key)
        out.append(str(pool.smiles[index]))
    return out


def _median_mass(rows: pd.DataFrame) -> float:
    masses = pd.to_numeric(rows.get("neutral_mass"), errors="coerce").dropna()
    return float(np.median(masses)) if len(masses) else float("nan")


def _top_up(guesses: list[str], pool: CandidatePool, neutral_mass: float, k: int) -> list[str]:
    if len(guesses) >= k:
        return guesses[:k]
    have = {key for key in (inchikey14(g) for g in guesses) if key is not None}
    extra = nearest_mass_candidates(pool, neutral_mass, k - len(guesses), exclude=have)
    return (guesses + extra)[:k]


def predict_molecule(
    trainer: CasmiTrainer,
    pool: CandidatePool,
    rows: pd.DataFrame,
    *,
    molecule_id: str,
    adduct_vocab: AdductVocabulary,
    ranker: LinearRanker,
    config: Optional[PredictConfig] = None,
    denovo_fn: Optional[DeNovoFn] = None,
) -> tuple[list[str], Optional[CandidateScores]]:
    """Ranked SMILES for one molecule plus the candidate scores (or ``None``)."""
    config = config or PredictConfig()
    neutral_mass = _median_mass(rows)
    tensors = spectra_to_tensors(
        rows,
        adduct_vocab=adduct_vocab,
        max_peaks=config.max_peaks,
        peak_config=config.peak_config,
        device=str(trainer.device),
    )
    if tensors is None:
        logger.warning("Molecule %s has no usable spectra; using mass-nearest candidates.", molecule_id)
        return nearest_mass_candidates(pool, neutral_mass, config.top_k), None

    scores = score_molecule(trainer, pool, tensors, molecule_id=molecule_id, config=config.scoring)
    if len(scores) == 0 and config.fallback_ppm > config.scoring.ppm:
        wide = ScoringConfig(**{**config.scoring.__dict__, "ppm": config.fallback_ppm})
        scores = score_molecule(trainer, pool, tensors, molecule_id=molecule_id, config=wide)

    guesses = rank_candidates(scores, pool, ranker, k=config.top_k)

    if config.denovo and denovo_fn is not None:
        formulas = pool.formulas_in_window(scores.neutral_mass, config.scoring.ppm)[: config.denovo_max_formulas]
        try:
            generated = denovo_fn(trainer, tensors, formulas)
        except Exception as exc:  # de novo is best-effort by design
            logger.warning("De novo generation failed for %s: %s", molecule_id, exc)
            generated = []
        have = {key for key in (inchikey14(g) for g in guesses) if key is not None}
        for smiles in generated:
            key = inchikey14(smiles)
            if key is None or key in have:
                continue
            have.add(key)
            guesses.append(smiles)

    return _top_up(guesses, pool, scores.neutral_mass, config.top_k), scores


def predict_frame(
    trainer: CasmiTrainer,
    pool: CandidatePool,
    spectra: pd.DataFrame,
    *,
    adduct_vocab: AdductVocabulary,
    ranker: LinearRanker,
    config: Optional[PredictConfig] = None,
    denovo_fn: Optional[DeNovoFn] = None,
    progress: bool = False,
    collect_scores: bool = False,
) -> tuple[dict[str, list[str]], dict[str, CandidateScores]]:
    """Predictions for every ``molecule_id`` in a prepared spectra frame."""
    config = config or PredictConfig()
    groups = group_by_molecule(spectra)
    items: Iterable[tuple[str, np.ndarray]] = groups.items()
    if progress:
        try:
            from tqdm.auto import tqdm

            items = tqdm(list(items), desc="predict", unit="mol")
        except ImportError:
            pass
    predictions: dict[str, list[str]] = {}
    scores_by_molecule: dict[str, CandidateScores] = {}
    for molecule_id, positions in items:
        rows = spectra.iloc[positions]
        try:
            guesses, scores = predict_molecule(
                trainer,
                pool,
                rows,
                molecule_id=molecule_id,
                adduct_vocab=adduct_vocab,
                ranker=ranker,
                config=config,
                denovo_fn=denovo_fn,
            )
        except Exception as exc:  # keep the submission alive whatever happens
            logger.exception("Prediction failed for %s: %s", molecule_id, exc)
            guesses, scores = [], None
            try:
                guesses = nearest_mass_candidates(pool, _median_mass(rows), config.top_k)
            except Exception:
                guesses = []
        if not guesses:
            guesses = [LAST_RESORT_SMILES]
        predictions[molecule_id] = guesses
        if collect_scores and scores is not None:
            scores_by_molecule[molecule_id] = scores
    return predictions, scores_by_molecule


def make_submission(
    predictions: dict[str, list[str]],
    spectra: pd.DataFrame,
    path: str | Path,
) -> pd.DataFrame:
    """Write ``submission.csv`` covering every molecule_id in ``spectra`` once."""
    molecule_ids = list(dict.fromkeys(spectra["molecule_id"].astype(str)))
    return write_submission(predictions, path, molecule_ids=molecule_ids, fallback_smiles=LAST_RESORT_SMILES)


__all__ = [
    "LAST_RESORT_SMILES",
    "PredictConfig",
    "make_submission",
    "nearest_mass_candidates",
    "predict_frame",
    "predict_molecule",
]
