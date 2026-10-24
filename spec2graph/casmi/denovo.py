"""De novo structures for formulas the pool lacks: bond-type decoder, hydrogen-budget decoding, formula enumeration."""

from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spec2graph.casmi import chem
from spec2graph.casmi.conditioning import CasmiTrainer
from spec2graph.casmi.data import Spectra
from spec2graph.data.elements import ELEMENT_TO_INDEX, N_ELEMENT_TYPES, UNKNOWN_INDEX
from spec2graph.eval.decode import assign_elements, parse_formula
from spec2graph.eval.metrics import rank_samples_by_frequency
from spectral_diffusion import VALENCY_TABLE

N_CLASSES = 4  # none, single, double, triple
FORMAT = "spec2graph-casmi-bondsgno-v1"
MASSES = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.99491461956, "S": 31.97207100, "P": 30.97376163,
          "F": 18.99840322, "Cl": 34.96885268, "Br": 78.9183371, "I": 126.904473}
VALENCE = {"C": 4, "H": 1, "N": 3, "O": 2, "S": 2, "P": 3, "F": 1, "Cl": 1, "Br": 1, "I": 1}
CAPS = {"C": 120, "N": 24, "O": 48, "S": 10, "P": 6, "F": 12, "Cl": 8, "Br": 6, "I": 4}
RATIO_TO_C = {"N": 1.3, "O": 1.2, "S": 0.8, "P": 0.3, "F": 1.5, "Cl": 0.8, "Br": 0.8, "I": 0.5}


class BondTypeSGNO(nn.Module):
    """Pairwise MLP over (eigenvector, element) pairs -> symmetric ``(B, N, N, 4)`` bond-class logits."""

    def __init__(self, k: int = 8, hidden_dim: int = 128, num_layers: int = 3, n_element_types: int = N_ELEMENT_TYPES):
        super().__init__()
        self.hparams = dict(k=k, hidden_dim=hidden_dim, num_layers=num_layers, n_element_types=n_element_types)
        self.proj_i, self.proj_j = nn.Linear(k, hidden_dim), nn.Linear(k, hidden_dim, bias=False)
        self.elem_i, self.elem_j = nn.Embedding(n_element_types, hidden_dim), nn.Embedding(n_element_types, hidden_dim)
        hidden = [m for _ in range(num_layers - 2) for m in (nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))]
        self.mlp = nn.Sequential(*hidden, nn.ReLU(), nn.Linear(hidden_dim, N_CLASSES))

    def forward(self, eigvecs: torch.Tensor, elements: torch.Tensor) -> torch.Tensor:
        elements = elements.long().clamp(0, self.elem_i.num_embeddings - 1)
        h_i, h_j = self.proj_i(eigvecs) + self.elem_i(elements), self.proj_j(eigvecs) + self.elem_j(elements)
        logits = self.mlp(h_i.unsqueeze(2) + h_j.unsqueeze(1))
        logits = 0.5 * (logits + logits.transpose(1, 2))

        # OPTIMIZATION: Avoid materializing an O(N^2) eye mask and boolean operations.
        # Instead, clone the logits and update the diagonal in-place using .diagonal().
        # We zero out the entire diagonal, then set the "none class" (index 0) to 1e4.
        out = logits.clone()
        d = out.diagonal(dim1=1, dim2=2)  # Shape: (batch, n_classes, n_atoms)
        d.fill_(0.0)
        d[:, 0, :].fill_(1e4)
        return out  # diagonal = "no bond"

    def probabilities(self, eigvecs: torch.Tensor, elements: torch.Tensor) -> torch.Tensor:
        return self(eigvecs, elements).softmax(-1)


def bond_targets(adjacencies: Sequence[np.ndarray], atom_mask: torch.Tensor) -> torch.Tensor:
    """``(B, N, N)`` bond classes from integer-order adjacencies (aromatic 1.5 -> 1)."""
    b, n = atom_mask.shape
    target = np.zeros((b, n, n), dtype=np.int64)
    for i, adj in enumerate(adjacencies):
        m = min(adj.shape[0], n)
        target[i, :m, :m] = np.clip(np.rint(np.where(np.isclose(adj[:m, :m], 1.5), 1, adj[:m, :m])), 0, N_CLASSES - 1)
    return torch.from_numpy(target).to(atom_mask.device)


class BondTypeTrainer:
    """Class-weighted cross-entropy on valid off-diagonal pairs."""

    def __init__(self, sgno: BondTypeSGNO, class_weights: Sequence[float] = (1.0, 5.0, 10.0, 20.0)):
        self.sgno, self.class_weights = sgno, torch.tensor(class_weights)

    def loss(self, eigvecs, atom_mask, elements, adjacencies) -> torch.Tensor:
        logits, targets = self.sgno(eigvecs, elements), bond_targets(adjacencies, atom_mask)
        pair = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)
        pair.diagonal(dim1=-2, dim2=-1).fill_(False)
        per_pair = F.cross_entropy(logits.flatten(0, 2), targets.flatten(), weight=self.class_weights.to(logits), reduction="none").view_as(targets)
        return (per_pair * pair).sum() / pair.sum().clamp_min(1)

    def step(self, optimiser, *args) -> float:
        self.sgno.train()
        optimiser.zero_grad()
        loss = self.loss(*args)
        loss.backward()
        optimiser.step()
        return loss.item()


def hydrogens_in(formula: str) -> int:
    m = re.search(r"H(\d*)(?![a-z])", formula)
    return int(m.group(1) or 1) if m else 0


def implied_hydrogens(adjacency: np.ndarray, atoms: Sequence[str]) -> int:
    """Hydrogens needed to fill every atom's valence after the heavy-atom bonds."""
    return int(sum(max(VALENCY_TABLE.get(a, 4) - round(float(u)), 0) for a, u in zip(atoms, adjacency.sum(1))))


def decode(probs: np.ndarray, atoms: Sequence[str], threshold: float = 0.5, h_budget: Optional[int] = None, max_repairs: int = 50) -> np.ndarray:
    """Greedy valence-constrained bond-type decode, then push the implied hydrogen count to ``h_budget``.

    Repairs promote the bond whose next class is most probable (too many H) or demote one (too few), within valence.
    """
    n = len(atoms)
    if probs.shape != (n, n, N_CLASSES):
        raise ValueError("probs must be (N, N, 4)")
    remaining = np.array([VALENCY_TABLE.get(a, 4) for a in atoms])
    adjacency = np.zeros((n, n), np.float32)
    i_idx, j_idx = np.triu_indices(n, 1)
    p_bond = 1 - probs[i_idx, j_idx, 0]
    for t in np.argsort(-p_bond, kind="stable"):
        if p_bond[t] < threshold:
            break
        i, j = i_idx[t], j_idx[t]
        if (cap := min(remaining[i], remaining[j], N_CLASSES - 1)) >= 1:
            order = int(np.argmax(probs[i, j, 1 : cap + 1])) + 1
            adjacency[i, j] = adjacency[j, i] = order
            remaining[i] -= order
            remaining[j] -= order
    for _ in range(max_repairs if h_budget is not None else 0):
        delta = int(np.sign(implied_hydrogens(adjacency, atoms) - h_budget))  # +1: too many H -> promote a bond
        if delta == 0:
            break
        cur = adjacency[i_idx, j_idx].astype(int)
        ok = (cur >= 1) & (cur < N_CLASSES - 1) & (remaining[i_idx] >= 1) & (remaining[j_idx] >= 1) if delta > 0 else cur >= 2
        if not ok.any():
            break
        t = int(np.argmax(np.where(ok, probs[i_idx, j_idx, np.clip(cur + delta, 0, N_CLASSES - 1)], -1.0)))
        i, j = i_idx[t], j_idx[t]
        adjacency[i, j] = adjacency[j, i] = cur[t] + delta
        remaining[i] -= delta
        remaining[j] -= delta
    return adjacency


def rdbe(counts: dict[str, int]) -> float:
    """Ring-double-bond equivalents ``1 + Σ n_i (v_i - 2) / 2``."""
    return 1 + 0.5 * sum(n * (VALENCE.get(e, 2) - 2) for e, n in counts.items())


def formula_string(counts: dict[str, int]) -> str:
    order = ["C", "H"] + sorted(e for e in counts if e not in ("C", "H"))
    return "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in order if counts.get(e, 0) > 0)


def enumerate_formulas(mass: float, ppm: float = 5.0, elements: Sequence[str] = ("C", "N", "O", "S", "P", "F", "Cl", "Br"),
                       max_results: int = 50) -> list[str]:
    """Even-electron formulas within ``ppm`` of a neutral mass, closest first.

    Filters: integer RDBE in [0, 40], ``0.1 <= H/C <= 3.5`` with one hydrogen of slack, element/carbon ratio caps.
    """
    if not np.isfinite(mass) or mass <= 0:
        return []
    heavy = ["C"] + [e for e in elements if e not in ("C", "H")]
    tol, hits, counts = mass * ppm * 1e-6, [], {}

    def plausible(full: dict[str, int], total: float) -> bool:
        c, h, r = full["C"], full["H"], rdbe(full)
        return c >= 1 and h >= 0 and abs(total - mass) <= tol and 0.1 * c - 1 <= h <= 3.5 * c + 1 and 0 <= r <= 40 and r.is_integer()

    def recurse(pos: int, acc: float) -> None:
        if pos == len(heavy):
            h = round((mass - acc) / MASSES["H"])
            full = {**counts, "H": h}
            if plausible(full, acc + h * MASSES["H"]):
                hits.append((abs(acc + h * MASSES["H"] - mass), formula_string(full)))
            return
        e = heavy[pos]
        cap = CAPS[e] if e == "C" else min(CAPS[e], int(RATIO_TO_C.get(e, 1) * counts["C"]) + 1)
        for n in range(cap + 1):
            if acc + n * MASSES[e] > mass + tol:
                break
            counts[e] = n
            recurse(pos + 1, acc + n * MASSES[e])
        counts.pop(e, None)

    recurse(0, 0.0)
    return list(dict.fromkeys(f for _, f in sorted(hits)))[:max_results]


class DeNovo:
    """Sample eigenvectors per formula, assign elements (core Hungarian step), decode bond types, rank by frequency."""

    def __init__(self, sgno: BondTypeSGNO, n_samples: int = 32, sampler: str = "ddim", n_steps: int = 50, eta: float = 0.0,
                 threshold: float = 0.5, max_per_formula: int = 10):
        self.sgno, self.n_samples, self.threshold, self.max_per_formula = sgno.eval(), n_samples, threshold, max_per_formula
        self.sample_kw = dict(sampler=sampler, n_steps=n_steps, eta=eta)

    @torch.no_grad()
    def __call__(self, trainer: CasmiTrainer, spectra: Spectra, formulas: Sequence[str]) -> list[str]:
        """De novo SMILES for the formulas, interleaved by rank and deduplicated by InChIKey14."""
        heads = trainer.heads(spectra)
        ranked = [self.for_formula(trainer, heads["memory"], heads["memory_mask"], f) for f in formulas]
        padded = [r + [None] * (self.max_per_formula - len(r)) for r in ranked]
        seen, out = set(), []
        for smiles in filter(None, (s for level in zip(*padded) for s in level)):
            if (key := chem.inchikey14(smiles)) and key not in seen:
                seen.add(key)
                out.append(smiles)
        return out

    def for_formula(self, trainer: CasmiTrainer, memory, mask, formula: str) -> list[str]:
        try:
            n_atoms = sum(parse_formula(formula).values())
        except ValueError:
            return []
        if not 1 <= n_atoms <= trainer.model.max_atoms:
            return []
        h_budget, smiles = hydrogens_in(formula), []
        for eigvecs, mem, msk in trainer.draw(memory, mask, n_atoms, self.n_samples, **self.sample_kw):
            atoms = [self._assign(lp, formula) for lp in trainer.atom_type_logprobs(eigvecs, mem, msk).cpu().numpy()]
            idx = torch.tensor([[ELEMENT_TO_INDEX.get(a, UNKNOWN_INDEX) for a in al] or [UNKNOWN_INDEX] * n_atoms for al in atoms], device=eigvecs.device)
            probs = self.sgno.to(eigvecs.device).probabilities(eigvecs, idx).cpu().numpy()
            smiles += [chem.to_smiles(decode(p, al, self.threshold, h_budget), al) if al else None for p, al in zip(probs, atoms)]
        return rank_samples_by_frequency(smiles)[: self.max_per_formula]

    @staticmethod
    def _assign(logprobs: np.ndarray, formula: str) -> list[str]:
        try:
            return assign_elements(logprobs, formula)
        except ValueError:
            return []


def save_bond_sgno(path, sgno: BondTypeSGNO) -> None:
    torch.save(dict(format=FORMAT, hparams=sgno.hparams, state_dict=sgno.state_dict()), path)


def load_bond_sgno(path, device: str | torch.device = "cpu") -> BondTypeSGNO:
    p = torch.load(path, map_location=device, weights_only=True)
    if p.get("format") != FORMAT:
        raise ValueError("unrecognised bond-type SGNO checkpoint")
    sgno = BondTypeSGNO(**p["hparams"])
    sgno.load_state_dict(p["state_dict"])
    return sgno.to(device).eval()
