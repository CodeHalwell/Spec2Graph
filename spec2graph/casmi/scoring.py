"""Score mass-window candidates: fingerprint agreement, spectral-quantile distance and aligned subspace overlap.

The overlap is the Spec2Graph signal proper: eigenvector draws give a soft projection ``P̂`` (legitimate because the
score is linear in ``P``), compared with each candidate's ``P_c`` after aligning atoms by Sinkhorn ascent seeded with
the atom-type head and rounded to a permutation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from spec2graph.casmi.conditioning import CasmiTrainer
from spec2graph.casmi.data import Spectra
from spec2graph.casmi.pool import CandidatePool
from spec2graph.casmi.spectral import quantile_distance

FEATURES = ("mass_ppm_abs", "quantile_dist", "fp_loglik", "fp_soft_tanimoto", "proj_overlap", "proj_overlap_soft",
            "elem_loglik", "atom_count_err", "n_candidates_log")


@dataclass
class ScoringConfig:
    ppm: float = 10.0
    min_abs_da: float = 0.002
    max_candidates: Optional[int] = 2000
    n_samples: int = 8  # eigenvector draws per spectrum
    sampler: str = "ddim"
    n_steps: int = 50
    eta: float = 0.0
    max_atom_counts: int = 8  # distinct heavy-atom counts to sample for (largest groups first)
    use_projection: bool = True
    tau: float = 0.1
    sinkhorn_iters: int = 30
    alignment_iters: int = 10
    element_weight: float = 1.0
    hard: bool = True


@dataclass
class CandidateScores:
    """Feature matrix over the pool candidates of one molecule (NaN = unavailable)."""

    molecule_id: str
    indices: np.ndarray
    features: np.ndarray
    neutral_mass: float

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.features[:, FEATURES.index(name)]

    def frame(self, pool: CandidatePool) -> pd.DataFrame:
        f = pd.DataFrame(self.features, columns=FEATURES)
        f.insert(0, "inchikey14", pool.inchikey14[self.indices])
        f.insert(0, "smiles", pool.smiles[self.indices])
        return f


def fingerprint_loglik(prob: np.ndarray, fps: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(np.asarray(prob, float), eps, 1 - eps)
    return (fps * np.log(p) + (1 - fps) * np.log1p(-p)).mean(-1).astype(np.float32)


def soft_tanimoto(prob: np.ndarray, fps: np.ndarray) -> np.ndarray:
    num, den = np.minimum(prob, fps).sum(-1), np.maximum(prob, fps).sum(-1)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0).astype(np.float32)


def log_sinkhorn(log_scores: torch.Tensor, iters: int = 30) -> torch.Tensor:
    """Log-domain Sinkhorn: log of a doubly stochastic ``(M, N, N)`` matrix."""
    for _ in range(iters):
        log_scores = log_scores - log_scores.logsumexp(-1, keepdim=True)
        log_scores = log_scores - log_scores.logsumexp(-2, keepdim=True)
    return log_scores


def hard_assignments(scores: torch.Tensor) -> torch.Tensor:
    """Round ``(M, N, N)`` score matrices to permutation matrices (Hungarian)."""
    out = torch.zeros_like(scores)
    for m, s in enumerate(scores.detach().cpu().numpy()):
        rows, cols = linear_sum_assignment(-s)
        out[m, rows, cols] = 1.0
    return out


def align_and_overlap(p_hat: torch.Tensor, elem_logprob: torch.Tensor, p_cands: torch.Tensor, cand_elements: torch.Tensor,
                      cfg: ScoringConfig = ScoringConfig()) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Maximise ``<Πᵀ P̂ Π, P_c>`` over alignments ``Π``; returns ``(hard overlap, soft overlap, element log-lik)``, each ``(M,)``.

    Overlaps are normalised by ``sqrt(tr P̂ · tr P_c)`` so they lie in [0, 1].
    """
    m, n, _ = p_cands.shape
    if p_hat.shape != (n, n) or cand_elements.shape != (m, n):
        raise ValueError("shape mismatch between p_hat, candidates and elements")
    elem_score = torch.gather(elem_logprob.expand(m, -1, -1), 2, cand_elements.long().unsqueeze(1).expand(-1, n, -1))
    tr_hat, tr_c = p_hat.diagonal().sum().clamp_min(1e-6), p_cands.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-6)
    scores = cfg.element_weight * elem_score
    log_pi = log_sinkhorn(scores / cfg.tau, cfg.sinkhorn_iters)
    for _ in range(cfg.alignment_iters):
        scores = 2 * (p_hat @ log_pi.exp() @ p_cands) * (n / tr_hat.sqrt()) + cfg.element_weight * elem_score
        log_pi = log_sinkhorn(scores / cfg.tau, cfg.sinkhorn_iters)
    overlap = lambda pi: ((pi.transpose(1, 2) @ p_hat @ pi) * p_cands).sum((1, 2)).div((tr_hat * tr_c).sqrt()).clamp(0, 1)
    pi_soft = log_pi.exp()
    pi = hard_assignments(scores) if cfg.hard else pi_soft
    return overlap(pi), overlap(pi_soft), (pi * elem_score).sum((1, 2)) / n


class Scorer:
    """Turns one molecule's spectra into a feature matrix over its mass-window candidates."""

    def __init__(self, trainer: CasmiTrainer, pool: CandidatePool, config: ScoringConfig = ScoringConfig()):
        self.trainer, self.pool, self.cfg = trainer, pool, config

    def heads(self, spectra: Spectra) -> dict[str, torch.Tensor]:
        """Head outputs averaged over the molecule's spectra; the memory stays per spectrum."""
        h = self.trainer.heads(spectra)
        for key in ("fingerprint_prob", "atom_count"):
            if key in h:
                h[key] = h[key].mean(0)
        h["quantiles"] = h["quantiles"].mean(0).sort().values
        return h

    def soft_projection(self, memory, mask, n_atoms: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean projection ``P̂ (N, N)`` and mean element log-probs ``(N, E)`` over ``n_samples`` draws per spectrum."""
        kw = dict(sampler=self.cfg.sampler, n_steps=self.cfg.n_steps, eta=self.cfg.eta)
        draws = list(self.trainer.draw(memory, mask, n_atoms, memory.shape[0] * self.cfg.n_samples, **kw))
        p_hat = torch.cat([CasmiTrainer.projection_from_embeddings(v) for v, _, _ in draws]).mean(0)
        return p_hat, torch.cat([self.trainer.atom_type_logprobs(*d) for d in draws]).mean(0)

    def score(self, spectra: Spectra, molecule_id: str = "") -> CandidateScores:
        cfg, pool = self.cfg, self.pool
        heads = self.heads(spectra)
        mass = float(spectra.cond.neutral_mass.median())
        idx = pool.query_mass(mass, cfg.ppm, cfg.min_abs_da, cfg.max_candidates)
        f = pd.DataFrame(np.nan, index=range(len(idx)), columns=list(FEATURES), dtype=np.float32)
        if len(idx) == 0:
            return CandidateScores(molecule_id, idx, f.to_numpy(np.float32), mass)
        f["mass_ppm_abs"], f["n_candidates_log"] = np.abs(pool.ppm_error(mass, idx)), math.log(len(idx))
        f["quantile_dist"] = quantile_distance(heads["quantiles"].cpu().numpy(), pool.quantiles[idx])
        if "fingerprint_prob" in heads:
            prob, fps = heads["fingerprint_prob"].cpu().numpy(), pool.fingerprints(idx)
            f["fp_loglik"], f["fp_soft_tanimoto"] = fingerprint_loglik(prob, fps), soft_tanimoto(prob, fps)
        atoms = pool.n_atoms[idx]
        if "atom_count" in heads:
            f["atom_count_err"] = np.abs(float(heads["atom_count"]) - atoms) / atoms
        if cfg.use_projection:
            for n_atoms in pd.Series(atoms).value_counts().index[: cfg.max_atom_counts]:
                self._projection_features(f, heads, idx, np.flatnonzero(atoms == n_atoms), int(n_atoms))
        return CandidateScores(molecule_id, idx, f.to_numpy(np.float32), mass)

    def _projection_features(self, f: pd.DataFrame, heads: dict, idx: np.ndarray, group: np.ndarray, n_atoms: int) -> None:
        if not 1 <= n_atoms <= self.trainer.model.max_atoms:
            return
        p_hat, elem = self.soft_projection(heads["memory"], heads["memory_mask"], n_atoms)
        stack = lambda fn: torch.as_tensor(np.stack([fn(int(i)) for i in idx[group]]), device=p_hat.device)
        hard, soft, ll = align_and_overlap(p_hat, elem, CasmiTrainer.projection_from_embeddings(stack(self.pool.eigvecs)), stack(self.pool.elements), self.cfg)
        f.loc[group, ["proj_overlap", "proj_overlap_soft", "elem_loglik"]] = torch.stack([hard, soft, ll], 1).cpu().numpy()
