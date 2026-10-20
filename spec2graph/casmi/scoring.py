"""Score mass-window candidates with the conditioned spectral model.

For one molecule (one or more spectra) the scorer produces a feature matrix
over the structures in its mass window:

``mass_ppm_abs``
    Absolute mass error of the candidate against the measured neutral mass.
``quantile_dist``
    Wasserstein distance between the predicted Laplacian spectral quantiles
    and the candidate's (permutation-invariant, needs no alignment).
``fp_loglik`` / ``fp_soft_tanimoto``
    Agreement between the fingerprint head's bit probabilities and the
    candidate fingerprint (the MIST-style retrieval signal).
``proj_overlap`` / ``proj_overlap_soft``
    The Spec2Graph signal proper. Eigenvectors are sampled from the diffusion
    model, turned into a soft projection ``P̂`` (the mean of per-sample
    projections, legitimate because the score is linear in ``P``), and
    compared with the candidate's projection ``P_c`` after aligning atoms:
    ``max_Π <Πᵀ P̂ Π, P_c>``. The alignment is found by Sinkhorn ascent over
    doubly stochastic matrices seeded by the atom-type head; the soft
    variant reports that relaxed optimum, the hard variant rounds the final
    scores to a permutation with the Hungarian algorithm, which stops wrong
    candidates profiting from averaged assignments. Both normalised to
    ``[0, 1]``.
``elem_loglik``
    Mean log-probability of the candidate's elements under the atom-type
    head through the same alignment: a learned composition check that needs
    no molecular formula.
``atom_count_err``
    Relative disagreement between the atom-count head and the candidate.
``n_candidates_log``
    Log of the window size, so the ranker can calibrate across windows.

Unavailable features are NaN; the ranker imputes them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from spec2graph.casmi.conditioning import CasmiTrainer, Conditioning
from spec2graph.casmi.data import SpectraTensors
from spec2graph.casmi.pool import CandidatePool
from spec2graph.casmi.spectral import quantile_distance

logger = logging.getLogger(__name__)

FEATURE_NAMES: tuple[str, ...] = (
    "mass_ppm_abs",
    "quantile_dist",
    "fp_loglik",
    "fp_soft_tanimoto",
    "proj_overlap",
    "proj_overlap_soft",
    "elem_loglik",
    "atom_count_err",
    "n_candidates_log",
)


@dataclass
class ScoringConfig:
    ppm: float = 10.0
    min_abs_da: float = 0.002
    max_candidates: Optional[int] = 2000
    # Eigenvector draws per spectrum for the soft projection.
    n_samples: int = 8
    sampler: str = "ddim"
    n_steps: int = 50
    eta: float = 0.0
    # Distinct heavy-atom counts to sample for (largest candidate groups first).
    max_atom_counts: int = 8
    use_projection: bool = True
    sinkhorn_tau: float = 0.1
    sinkhorn_iters: int = 30
    alignment_iters: int = 10
    element_weight: float = 1.0
    hard_assignment: bool = True
    fp_eps: float = 1e-4
    sample_batch: int = 256


@dataclass
class CandidateScores:
    """Feature matrix over pool candidates for one molecule."""

    molecule_id: str
    indices: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    neutral_mass: float
    n_spectra: int

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def feature(self, name: str) -> np.ndarray:
        return self.features[:, self.feature_names.index(name)]

    def as_frame(self, pool: CandidatePool) -> pd.DataFrame:
        frame = pd.DataFrame(self.features, columns=list(self.feature_names))
        frame.insert(0, "formula", pool.formula[self.indices])
        frame.insert(0, "inchikey14", pool.inchikey14[self.indices])
        frame.insert(0, "smiles", pool.smiles[self.indices])
        frame.insert(0, "pool_index", self.indices)
        return frame

    @classmethod
    def empty(cls, molecule_id: str, neutral_mass: float, n_spectra: int) -> "CandidateScores":
        return cls(
            molecule_id=molecule_id,
            indices=np.zeros(0, dtype=np.int64),
            features=np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
            feature_names=FEATURE_NAMES,
            neutral_mass=neutral_mass,
            n_spectra=n_spectra,
        )


# ----------------------------------------------------------------------
# Fingerprint agreement
# ----------------------------------------------------------------------


def fingerprint_loglik(prob: np.ndarray, fingerprints: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Mean per-bit Bernoulli log-likelihood of each candidate under ``prob``."""
    p = np.clip(np.asarray(prob, dtype=np.float64), eps, 1.0 - eps)
    f = np.asarray(fingerprints, dtype=np.float64)
    ll = f * np.log(p) + (1.0 - f) * np.log1p(-p)
    return ll.mean(axis=-1).astype(np.float32)


def soft_tanimoto(prob: np.ndarray, fingerprints: np.ndarray) -> np.ndarray:
    """Tanimoto between a probability vector and binary fingerprints."""
    p = np.asarray(prob, dtype=np.float64)
    f = np.asarray(fingerprints, dtype=np.float64)
    numerator = np.minimum(p, f).sum(axis=-1)
    denominator = np.maximum(p, f).sum(axis=-1)
    return np.where(denominator > 0, numerator / np.maximum(denominator, 1e-12), 0.0).astype(np.float32)


# ----------------------------------------------------------------------
# Subspace alignment
# ----------------------------------------------------------------------


def log_sinkhorn(log_scores: torch.Tensor, n_iters: int = 30) -> torch.Tensor:
    """Log-domain Sinkhorn: returns log of a doubly stochastic ``(M, N, N)`` matrix."""
    log_pi = log_scores
    for _ in range(n_iters):
        log_pi = log_pi - torch.logsumexp(log_pi, dim=-1, keepdim=True)
        log_pi = log_pi - torch.logsumexp(log_pi, dim=-2, keepdim=True)
    return log_pi


def soft_projection(eigvecs: torch.Tensor) -> torch.Tensor:
    """Mean rank-safe projection over a batch of eigenvector draws ``(D, N, k)`` -> ``(N, N)``."""
    return CasmiTrainer.projection_from_embeddings(eigvecs).mean(dim=0)


def hard_assignments(scores: torch.Tensor) -> torch.Tensor:
    """Round ``(M, N, N)`` score matrices to permutation matrices (Hungarian)."""
    from scipy.optimize import linear_sum_assignment

    scores_np = scores.detach().cpu().numpy()
    out = torch.zeros_like(scores)
    for m in range(scores_np.shape[0]):
        rows, cols = linear_sum_assignment(-scores_np[m])
        out[m, rows, cols] = 1.0
    return out


def align_and_overlap(
    p_hat: torch.Tensor,
    elem_logprob: torch.Tensor,
    p_candidates: torch.Tensor,
    candidate_elements: torch.Tensor,
    *,
    tau: float = 0.1,
    sinkhorn_iters: int = 30,
    alignment_iters: int = 10,
    element_weight: float = 1.0,
    hard: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft-align model atom slots to candidate atoms and score the subspace overlap.

    Parameters
    ----------
    p_hat:
        ``(N, N)`` soft projection predicted for this atom count.
    elem_logprob:
        ``(N, E)`` log-probabilities of each element per model slot.
    p_candidates:
        ``(M, N, N)`` projections of the candidates.
    candidate_elements:
        ``(M, N)`` element vocabulary indices per candidate atom.

    Returns
    -------
    overlap:
        ``(M,)`` normalised overlap ``<Πᵀ P̂ Π, P_c> / sqrt(tr P̂ · tr P_c)`` in
        ``[0, 1]`` under the hard (permutation) alignment when ``hard`` is
        true, otherwise a copy of the soft value.
    overlap_soft:
        ``(M,)`` the same quantity under the doubly stochastic alignment.
    elem_loglik:
        ``(M,)`` mean element log-probability under the hard alignment.
    """
    if p_hat.dim() != 2 or p_hat.shape[0] != p_hat.shape[1]:
        raise ValueError(f"p_hat must be (N, N); got {tuple(p_hat.shape)}.")
    m, n, _ = p_candidates.shape
    if p_hat.shape[0] != n:
        raise ValueError(f"Atom count mismatch: p_hat has {p_hat.shape[0]} atoms, candidates have {n}.")
    if candidate_elements.shape != (m, n):
        raise ValueError(f"candidate_elements must be (M, N)={m, n}; got {tuple(candidate_elements.shape)}.")

    # elem_score[m, i, j] = log p(element of candidate atom j | model slot i)
    gathered = elem_logprob.unsqueeze(0).expand(m, -1, -1)  # (M, N, E)
    index = candidate_elements.long().unsqueeze(1).expand(-1, n, -1)  # (M, N, N) over j
    elem_score = torch.gather(gathered, 2, index)

    trace_hat = torch.diagonal(p_hat).sum().clamp_min(1e-6)
    trace_c = torch.diagonal(p_candidates, dim1=-2, dim2=-1).sum(-1).clamp_min(1e-6)  # (M,)
    # Scale the overlap gradient so its entries are O(1) regardless of N or k.
    grad_scale = n / math.sqrt(float(trace_hat.item())) if trace_hat.item() > 0 else 1.0

    log_pi = log_sinkhorn(element_weight * elem_score / tau, sinkhorn_iters)
    p_hat_b = p_hat.unsqueeze(0)
    scores = element_weight * elem_score
    for _ in range(alignment_iters):
        pi = log_pi.exp()
        grad = 2.0 * torch.matmul(torch.matmul(p_hat_b, pi), p_candidates)  # (M, N, N)
        scores = grad * grad_scale + element_weight * elem_score
        log_pi = log_sinkhorn(scores / tau, sinkhorn_iters)

    def _overlap(pi: torch.Tensor) -> torch.Tensor:
        aligned = torch.matmul(torch.matmul(pi.transpose(1, 2), p_hat_b), pi)  # Πᵀ P̂ Π
        return ((aligned * p_candidates).sum(dim=(1, 2)) / torch.sqrt(trace_hat * trace_c)).clamp(0.0, 1.0)

    pi_soft = log_pi.exp()
    overlap_soft = _overlap(pi_soft)
    if hard:
        pi = hard_assignments(scores if alignment_iters > 0 else element_weight * elem_score)
        overlap = _overlap(pi)
    else:
        pi, overlap = pi_soft, overlap_soft
    elem_ll = (pi * elem_score).sum(dim=(1, 2)) / n
    return overlap, overlap_soft, elem_ll


# ----------------------------------------------------------------------
# Per-molecule scoring
# ----------------------------------------------------------------------


def _molecule_neutral_mass(tensors: SpectraTensors) -> float:
    return float(np.median(tensors.neutral_mass.detach().cpu().numpy()))


@torch.no_grad()
def predict_molecule_heads(trainer: CasmiTrainer, tensors: SpectraTensors) -> dict[str, torch.Tensor]:
    """Run the heads on every spectrum of a molecule and average them."""
    cond = Conditioning(
        adduct_index=tensors.adduct_index,
        collision_energy=tensors.collision_energy,
        collision_energy_mask=tensors.collision_energy_mask,
        neutral_mass=tensors.neutral_mass,
        polarity=tensors.polarity,
    )
    heads = trainer.predict_heads(tensors.mz, tensors.intensity, tensors.spectrum_mask, tensors.precursor_mz, cond)
    out: dict[str, torch.Tensor] = {"memory": heads["memory"], "memory_mask": heads["memory_mask"]}
    if "fingerprint_prob" in heads:
        out["fingerprint_prob"] = heads["fingerprint_prob"].mean(dim=0)
    if "quantiles" in heads:
        out["quantiles"] = torch.sort(heads["quantiles"].mean(dim=0)).values
    if "atom_count" in heads:
        out["atom_count"] = heads["atom_count"].mean()
    return out


@torch.no_grad()
def sample_soft_projection(
    trainer: CasmiTrainer,
    memory: torch.Tensor,
    memory_mask: torch.Tensor,
    n_atoms: int,
    config: ScoringConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw eigenvectors for every spectrum and return ``(P̂ (N, N), element log-probs (N, E))``."""
    n_spectra = memory.shape[0]
    total = n_spectra * config.n_samples
    projections: list[torch.Tensor] = []
    logprobs: list[torch.Tensor] = []
    draws_done = 0
    while draws_done < total:
        chunk = min(config.sample_batch, total - draws_done)
        # Round-robin over spectra so each chunk mixes conditioning.
        spectrum_ids = torch.arange(draws_done, draws_done + chunk, device=memory.device) % n_spectra
        mem = memory.index_select(0, spectrum_ids)
        mask = memory_mask.index_select(0, spectrum_ids)
        eigvecs = trainer.sample(
            mem, mask, n_atoms=n_atoms, sampler=config.sampler, n_steps=config.n_steps, eta=config.eta
        )
        projections.append(CasmiTrainer.projection_from_embeddings(eigvecs))
        if trainer.model.atom_type_head is not None:
            logits = trainer.atom_type_logits(eigvecs, None, mem, mask)
            logprobs.append(F.log_softmax(logits, dim=-1))
        draws_done += chunk
    p_hat = torch.cat(projections).mean(dim=0)
    if logprobs:
        elem_logprob = torch.cat(logprobs).mean(dim=0)
    else:
        n_types = trainer.model.n_element_types
        elem_logprob = torch.full((n_atoms, n_types), -math.log(n_types), device=memory.device)
    return p_hat, elem_logprob


def _candidate_projections(pool: CandidatePool, indices: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    eigvecs = torch.from_numpy(np.stack([pool.eigvecs(int(i)) for i in indices])).to(device)
    elements = torch.from_numpy(np.stack([pool.elements(int(i)) for i in indices])).to(device)
    return CasmiTrainer.projection_from_embeddings(eigvecs), elements


@torch.no_grad()
def score_molecule(
    trainer: CasmiTrainer,
    pool: CandidatePool,
    tensors: SpectraTensors,
    *,
    molecule_id: str,
    config: Optional[ScoringConfig] = None,
) -> CandidateScores:
    """Feature matrix over the mass-window candidates of one molecule."""
    config = config or ScoringConfig()
    device = trainer.device
    heads = predict_molecule_heads(trainer, tensors)
    mass = _molecule_neutral_mass(tensors)
    indices = pool.query_mass(mass, config.ppm, min_abs_da=config.min_abs_da, max_candidates=config.max_candidates)
    n_spectra = tensors.n_spectra
    if indices.size == 0:
        return CandidateScores.empty(molecule_id, mass, n_spectra)

    n_cand = indices.size
    features = np.full((n_cand, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}

    features[:, col["mass_ppm_abs"]] = np.abs(pool.mass_error_ppm(mass, indices))
    features[:, col["n_candidates_log"]] = math.log(n_cand)
    cand_atoms = pool.n_atoms[indices].astype(np.int64)

    if "quantiles" in heads:
        q = heads["quantiles"].detach().cpu().numpy()
        features[:, col["quantile_dist"]] = quantile_distance(q, pool.quantiles[indices])
    if "fingerprint_prob" in heads:
        prob = heads["fingerprint_prob"].detach().cpu().numpy()
        fps = pool.fingerprints(indices)
        features[:, col["fp_loglik"]] = fingerprint_loglik(prob, fps, config.fp_eps)
        features[:, col["fp_soft_tanimoto"]] = soft_tanimoto(prob, fps)
    if "atom_count" in heads:
        predicted = float(heads["atom_count"].item())
        features[:, col["atom_count_err"]] = np.abs(predicted - cand_atoms) / np.maximum(cand_atoms, 1)

    if config.use_projection:
        counts = pd.Series(cand_atoms).value_counts()
        max_atoms = trainer.model.max_atoms
        for n_atoms in counts.index[: config.max_atom_counts]:
            n_atoms = int(n_atoms)
            if n_atoms < 1 or n_atoms > max_atoms:
                continue
            group = np.where(cand_atoms == n_atoms)[0]
            p_hat, elem_logprob = sample_soft_projection(
                trainer, heads["memory"], heads["memory_mask"], n_atoms, config
            )
            p_cands, elements = _candidate_projections(pool, indices[group], device)
            overlap, overlap_soft, elem_ll = align_and_overlap(
                p_hat,
                elem_logprob,
                p_cands,
                elements,
                tau=config.sinkhorn_tau,
                sinkhorn_iters=config.sinkhorn_iters,
                alignment_iters=config.alignment_iters,
                element_weight=config.element_weight,
                hard=config.hard_assignment,
            )
            features[group, col["proj_overlap"]] = overlap.detach().cpu().numpy()
            features[group, col["proj_overlap_soft"]] = overlap_soft.detach().cpu().numpy()
            features[group, col["elem_loglik"]] = elem_ll.detach().cpu().numpy()

    return CandidateScores(
        molecule_id=molecule_id,
        indices=indices,
        features=features,
        feature_names=FEATURE_NAMES,
        neutral_mass=mass,
        n_spectra=n_spectra,
    )


__all__ = [
    "CandidateScores",
    "FEATURE_NAMES",
    "ScoringConfig",
    "align_and_overlap",
    "fingerprint_loglik",
    "hard_assignments",
    "log_sinkhorn",
    "predict_molecule_heads",
    "sample_soft_projection",
    "score_molecule",
    "soft_projection",
    "soft_tanimoto",
]
