"""A small listwise-trained linear ranker over candidate features."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from spec2graph.casmi.pool import CandidatePool
from spec2graph.casmi.scoring import FEATURES, CandidateScores

HEURISTIC = dict(mass_ppm_abs=-0.5, quantile_dist=-2.0, fp_loglik=3.0, fp_soft_tanimoto=2.0, proj_overlap=2.0,
                 proj_overlap_soft=1.0, elem_loglik=1.5, atom_count_err=-1.0, n_candidates_log=0.0)


@dataclass
class LinearRanker:
    """Standardise (NaN -> mean), then a weighted sum. Fit listwise: softmax over each molecule's candidates."""

    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def heuristic(cls) -> "LinearRanker":
        return cls(np.array([HEURISTIC[f] for f in FEATURES]), np.zeros(len(FEATURES)), np.ones(len(FEATURES)))

    def transform(self, features: np.ndarray) -> np.ndarray:
        return np.nan_to_num((np.asarray(features, float) - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0)

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.transform(features) @ self.weights if len(features) else np.zeros(0)

    @classmethod
    def fit(cls, groups: Sequence[tuple[np.ndarray, np.ndarray]], epochs: int = 300, lr: float = 0.05, l2: float = 1e-3,
            init: Optional["LinearRanker"] = None) -> "LinearRanker":
        """``groups`` are ``(features, is_correct)`` per molecule; groups without a positive among >= 2 candidates are ignored."""
        groups = [(np.asarray(f, float), np.asarray(y, bool)) for f, y in groups if np.any(y) and len(y) >= 2]
        if not groups:
            raise ValueError("no informative group to fit on")
        stacked = np.concatenate([f for f, _ in groups])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean, std = np.nan_to_num(np.nanmean(stacked, 0)), np.nan_to_num(np.nanstd(stacked, 0), nan=1.0)
        base = cls(np.zeros(len(FEATURES)), mean, np.where(std > 1e-12, std, 1.0))
        w = torch.tensor((init or cls.heuristic()).weights, dtype=torch.float64, requires_grad=True)
        data = [(torch.tensor(base.transform(f)), torch.tensor(y)) for f, y in groups]
        opt = torch.optim.Adam([w], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = -sum(torch.log_softmax(x @ w, 0)[y].logsumexp(0) for x, y in data) / len(data) + l2 * (w**2).sum()
            loss.backward()
            opt.step()
        return cls(w.detach().numpy(), base.mean, base.std)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({k: v.tolist() for k, v in vars(self).items()}))

    @classmethod
    def load(cls, path: str | Path) -> "LinearRanker":
        return cls(**{k: np.asarray(v, dtype=float) for k, v in json.loads(Path(path).read_text()).items()})


def rank(scores: CandidateScores, pool: CandidatePool, ranker: LinearRanker, k: int = 25) -> list[str]:
    """Top-``k`` candidate SMILES, best first (pool keys are unique, so nothing to dedupe)."""
    order = np.argsort(-ranker.score(scores.features), kind="stable")[:k]
    return [str(pool.smiles[i]) for i in scores.indices[order]]
