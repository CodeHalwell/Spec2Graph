"""Rank mass-window candidates from their feature matrix.

A deliberately small model: standardised features, a linear score, trained
listwise (softmax cross-entropy over each molecule's candidate list) on
held-out molecules. With a few hundred validation molecules anything bigger
over-fits, and the point of the competition's MRR is to get the first
position right, which a listwise objective targets directly.

:meth:`LinearRanker.heuristic` gives sensible signs for every feature so the
pipeline produces a ranking before any ranker has been fitted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from spec2graph.casmi.pool import CandidatePool
from spec2graph.casmi.scoring import FEATURE_NAMES, CandidateScores

logger = logging.getLogger(__name__)

# Direction and rough magnitude of each feature's contribution before fitting.
DEFAULT_WEIGHTS: dict[str, float] = {
    "mass_ppm_abs": -0.5,
    "quantile_dist": -2.0,
    "fp_loglik": 3.0,
    "fp_soft_tanimoto": 2.0,
    "proj_overlap": 2.0,
    "proj_overlap_soft": 1.0,
    "elem_loglik": 1.5,
    "atom_count_err": -1.0,
    "n_candidates_log": 0.0,
}


@dataclass
class LinearRanker:
    """Standardise, impute missing values with the mean, take a weighted sum."""

    feature_names: tuple[str, ...]
    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    bias: float = 0.0

    @classmethod
    def heuristic(cls, feature_names: Sequence[str] = FEATURE_NAMES) -> "LinearRanker":
        names = tuple(feature_names)
        weights = np.array([DEFAULT_WEIGHTS.get(name, 0.0) for name in names], dtype=np.float64)
        return cls(names, weights, np.zeros(len(names)), np.ones(len(names)))

    def transform(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError(
                f"features must be (n, {len(self.feature_names)}); got {x.shape}."
            )
        z = (x - self.mean) / np.where(self.std > 1e-12, self.std, 1.0)
        return np.where(np.isfinite(z), z, 0.0)

    def score(self, features: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.zeros(0, dtype=np.float64)
        return self.transform(features) @ self.weights + self.bias

    @classmethod
    def fit(
        cls,
        groups: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        feature_names: Sequence[str] = FEATURE_NAMES,
        epochs: int = 300,
        lr: float = 0.05,
        l2: float = 1e-3,
        seed: int = 0,
        init: Optional["LinearRanker"] = None,
    ) -> "LinearRanker":
        """Listwise fit on ``(features, labels)`` groups, one per molecule.

        Groups without a positive label, or with a single candidate (which
        carry no ranking information and would let the L2 term pull every
        weight to zero), are ignored. Standardisation statistics are
        nan-aware means/stds over the remaining candidate rows. Raises
        :class:`ValueError` when no informative group remains; callers
        should then keep :meth:`heuristic` weights.
        """
        import torch

        names = tuple(feature_names)
        usable = [
            (np.asarray(f, dtype=np.float64), np.asarray(y, dtype=bool))
            for f, y in groups
            if np.any(y) and len(y) >= 2
        ]
        if not usable:
            raise ValueError(
                "No informative group (a positive candidate among at least two); cannot fit the ranker."
            )
        stacked = np.concatenate([f for f, _ in usable])
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
        base = cls(names, np.zeros(len(names)), mean, std)
        if init is not None:
            start = np.asarray(init.weights, dtype=np.float64)
        else:
            start = cls.heuristic(names).weights
        torch.manual_seed(seed)
        w = torch.tensor(start, dtype=torch.float64, requires_grad=True)
        optimiser = torch.optim.Adam([w], lr=lr)
        tensors = [
            (torch.tensor(base.transform(f)), torch.tensor(y))
            for f, y in usable
        ]
        for _ in range(epochs):
            optimiser.zero_grad()
            loss = torch.zeros((), dtype=torch.float64)
            for x, y in tensors:
                logits = x @ w
                log_probs = torch.log_softmax(logits, dim=0)
                loss = loss - torch.logsumexp(log_probs[y], dim=0)
            loss = loss / len(tensors) + l2 * (w**2).sum()
            loss.backward()
            optimiser.step()
        return cls(names, w.detach().numpy().copy(), mean, std)

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "weights": self.weights.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "bias": float(self.bias),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LinearRanker":
        return cls(
            tuple(payload["feature_names"]),
            np.asarray(payload["weights"], dtype=np.float64),
            np.asarray(payload["mean"], dtype=np.float64),
            np.asarray(payload["std"], dtype=np.float64),
            float(payload.get("bias", 0.0)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LinearRanker":
        return cls.from_dict(json.loads(Path(path).read_text()))


def ranked_indices(scores: CandidateScores, ranker: LinearRanker) -> np.ndarray:
    """Pool indices of the candidates, best first (stable on ties)."""
    if len(scores) == 0:
        return np.zeros(0, dtype=np.int64)
    if tuple(scores.feature_names) != tuple(ranker.feature_names):
        raise ValueError("Ranker and scores disagree on feature names.")
    values = ranker.score(scores.features)
    order = np.argsort(-values, kind="stable")
    return scores.indices[order]


def rank_candidates(
    scores: CandidateScores,
    pool: CandidatePool,
    ranker: LinearRanker,
    k: int = 25,
) -> list[str]:
    """Top-``k`` candidate SMILES, best first, one per InChIKey14."""
    seen: set[str] = set()
    out: list[str] = []
    for index in ranked_indices(scores, ranker):
        key = str(pool.inchikey14[index])
        if key in seen:
            continue
        seen.add(key)
        out.append(str(pool.smiles[index]))
        if len(out) >= k:
            break
    return out


__all__ = ["DEFAULT_WEIGHTS", "LinearRanker", "rank_candidates", "ranked_indices"]
