"""De novo structures for formulas the candidate pool does not contain.

This is the part of the competition retrieval cannot touch: a molecule absent
from every structure database scores zero unless something generates it. The
pieces here extend the core decode path so an exact InChIKey14 match is at
least possible:

* :class:`BondTypeSGNO` — the core SGNO emits one logit per atom pair, i.e. a
  binary skeleton. Bond *types* (none/single/double/triple) are needed for
  the InChI hydrogen layer, so this decoder predicts four classes per pair
  from the eigenvector pair plus the two elements.
* :func:`decode_bond_types` — valence-constrained greedy decoding with an
  optional hydrogen budget from the formula: the implied hydrogen count is
  pushed to the formula's by promoting or demoting the most probable bonds.
* :func:`enumerate_formulas` — CHNOPS(+halogen) formulas within a ppm window
  of the neutral mass, filtered by ring-double-bond equivalents and element
  ratio heuristics, for molecules whose mass window has no pool formula.
* :func:`denovo_candidates` — sample eigenvectors from the conditioned
  model for each formula, assign elements with the core Hungarian step,
  decode bonds, sanitise, and rank by sampling frequency.

Everything is additive: the core SGNO, decoder and formula parser are reused
where they fit and left untouched where they do not.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spec2graph.casmi.conditioning import CasmiTrainer
from spec2graph.casmi.data import SpectraTensors
from spec2graph.casmi.metric import inchikey14
from spec2graph.casmi.scoring import predict_molecule_heads
from spec2graph.data.elements import ELEMENT_TO_INDEX, N_ELEMENT_TYPES, UNKNOWN_INDEX
from spec2graph.eval.decode import adjacency_to_rwmol, assign_elements, parse_formula
from spec2graph.eval.metrics import rank_samples_by_frequency
from spectral_diffusion import VALENCY_TABLE

logger = logging.getLogger(__name__)

N_BOND_CLASSES: int = 4  # none, single, double, triple

# Monoisotopic masses for formula enumeration.
ELEMENT_MASSES: dict[str, float] = {
    "C": 12.0,
    "H": 1.00782503207,
    "N": 14.0030740048,
    "O": 15.99491461956,
    "S": 31.97207100,
    "P": 30.97376163,
    "F": 18.99840322,
    "Cl": 34.96885268,
    "Br": 78.9183371,
    "I": 126.904473,
    "Si": 27.9769265325,
    "B": 11.0093054,
    "Se": 79.9165213,
}
# Valence used for ring-double-bond equivalents.
_RDBE_VALENCE: dict[str, int] = {"C": 4, "H": 1, "N": 3, "O": 2, "S": 2, "P": 3, "F": 1, "Cl": 1, "Br": 1, "I": 1, "Si": 4, "B": 3, "Se": 2}
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


# ----------------------------------------------------------------------
# Bond-type decoder
# ----------------------------------------------------------------------


class BondTypeSGNO(nn.Module):
    """Pairwise MLP over eigenvector pairs (and elements) -> bond-class logits.

    ``forward(eigvecs (B, N, k), elements (B, N))`` returns ``(B, N, N, 4)``
    symmetric logits with the diagonal forced to the "no bond" class. The
    first layer is split across the two atoms so the ``(B, N, N, 2k)``
    concatenation is never materialised, mirroring the core SGNO.
    """

    def __init__(
        self,
        k: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        n_element_types: int = N_ELEMENT_TYPES,
        use_elements: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2.")
        self.k = k
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_element_types = n_element_types
        self.use_elements = use_elements
        self.proj_i = nn.Linear(k, hidden_dim)
        self.proj_j = nn.Linear(k, hidden_dim, bias=False)
        if use_elements:
            self.elem_i = nn.Embedding(n_element_types, hidden_dim)
            self.elem_j = nn.Embedding(n_element_types, hidden_dim)
        layers: list[nn.Module] = []
        for _ in range(num_layers - 2):
            layers.extend([nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)])
        layers.extend([nn.ReLU(), nn.Linear(hidden_dim, N_BOND_CLASSES)])
        self.mlp = nn.Sequential(*layers)

    def forward(self, eigvecs: torch.Tensor, elements: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eigvecs.dim() != 3 or eigvecs.shape[-1] != self.k:
            raise ValueError(f"eigvecs must be (batch, n_atoms, {self.k}); got {tuple(eigvecs.shape)}.")
        h_i = self.proj_i(eigvecs)
        h_j = self.proj_j(eigvecs)
        if self.use_elements:
            if elements is None:
                raise ValueError("This BondTypeSGNO was built with use_elements=True; pass elements.")
            elements = elements.long().clamp(0, self.n_element_types - 1)
            h_i = h_i + self.elem_i(elements)
            h_j = h_j + self.elem_j(elements)
        x = h_i.unsqueeze(2) + h_j.unsqueeze(1)  # (B, N, N, hidden)
        logits = self.mlp(x)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        n = logits.shape[1]
        eye = torch.eye(n, dtype=torch.bool, device=logits.device)
        none_only = torch.zeros(N_BOND_CLASSES, device=logits.device, dtype=logits.dtype)
        none_only[0] = 1e4
        return torch.where(eye.unsqueeze(0).unsqueeze(-1), none_only, logits)

    def bond_probabilities(self, eigvecs: torch.Tensor, elements: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.softmax(self.forward(eigvecs, elements), dim=-1)

    def config_dict(self) -> dict:
        return {
            "k": self.k,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "n_element_types": self.n_element_types,
            "use_elements": self.use_elements,
        }


def bond_targets_from_adjacency(adjacencies: Sequence[np.ndarray], atom_mask: torch.Tensor) -> torch.Tensor:
    """``(B, N, N)`` long tensor of bond classes; orders are rounded, aromatic 1.5 -> 1."""
    batch_size, max_atoms = atom_mask.shape
    if len(adjacencies) != batch_size:
        raise ValueError(f"len(adjacencies)={len(adjacencies)} does not match batch size {batch_size}.")
    target = np.zeros((batch_size, max_atoms, max_atoms), dtype=np.int64)
    for i, adj in enumerate(adjacencies):
        n = min(adj.shape[0], max_atoms)
        orders = np.where(np.isclose(adj[:n, :n], 1.5), 1.0, adj[:n, :n])
        target[i, :n, :n] = np.clip(np.rint(orders), 0, N_BOND_CLASSES - 1)
    target.reshape(batch_size, -1)[:, :: max_atoms + 1] = 0
    return torch.from_numpy(target).to(atom_mask.device)


@dataclass
class BondTypeTrainerConfig:
    # Class weights for (none, single, double, triple); bonds are rare pairs.
    class_weights: tuple[float, float, float, float] = (1.0, 5.0, 10.0, 20.0)


class BondTypeTrainer:
    """Cross-entropy trainer for :class:`BondTypeSGNO` on ground-truth eigenvectors."""

    def __init__(self, sgno: BondTypeSGNO, config: Optional[BondTypeTrainerConfig] = None, device: str | torch.device = "cpu") -> None:
        self.sgno = sgno
        self.config = config or BondTypeTrainerConfig()
        self.device = device

    def compute_loss(
        self,
        eigvecs: torch.Tensor,
        atom_mask: torch.Tensor,
        elements: Optional[torch.Tensor],
        adjacencies: Sequence[np.ndarray],
    ) -> torch.Tensor:
        targets = bond_targets_from_adjacency(adjacencies, atom_mask)
        logits = self.sgno(eigvecs, elements)
        edge_mask = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)
        edge_mask.diagonal(dim1=-2, dim2=-1).fill_(False)
        if not edge_mask.any():
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        weights = torch.tensor(self.config.class_weights, device=logits.device, dtype=logits.dtype)
        per_pair = F.cross_entropy(
            logits.reshape(-1, N_BOND_CLASSES), targets.reshape(-1), weight=weights, reduction="none"
        ).reshape(targets.shape)
        mask = edge_mask.to(per_pair.dtype)
        return (per_pair * mask).sum() / mask.sum()

    def train_step(
        self,
        optimiser: torch.optim.Optimizer,
        eigvecs: torch.Tensor,
        atom_mask: torch.Tensor,
        elements: Optional[torch.Tensor],
        adjacencies: Sequence[np.ndarray],
    ) -> float:
        self.sgno.train()
        optimiser.zero_grad()
        loss = self.compute_loss(eigvecs, atom_mask, elements, adjacencies)
        loss.backward()
        optimiser.step()
        return float(loss.item())


# ----------------------------------------------------------------------
# Decoding with a hydrogen budget
# ----------------------------------------------------------------------


def formula_hydrogens(formula: str) -> int:
    """Hydrogen count in a formula string (0 if absent)."""
    total = 0
    for symbol, count in _FORMULA_TOKEN.findall(formula):
        if symbol == "H":
            total += int(count) if count else 1
    return total


def implied_hydrogens(adjacency: np.ndarray, atom_types: Sequence[str], valency_table: dict[str, int] = VALENCY_TABLE) -> int:
    """Hydrogens implied by filling every atom's valence after the heavy-atom bonds."""
    used = adjacency.sum(axis=1)
    return int(sum(max(valency_table.get(a, 4) - int(round(u)), 0) for a, u in zip(atom_types, used)))


def decode_bond_types(
    probs: np.ndarray,
    atom_types: Sequence[str],
    *,
    threshold: float = 0.5,
    h_budget: Optional[int] = None,
    valency_table: dict[str, int] = VALENCY_TABLE,
    max_repair_steps: int = 50,
) -> np.ndarray:
    """Valence-constrained greedy decode of ``(N, N, 4)`` bond-class probabilities.

    Pairs are visited by decreasing bond probability (``1 - P(none)``); each
    receives the most probable order its remaining valences allow. With
    ``h_budget`` (hydrogens from the formula) the implied hydrogen count is
    then repaired towards the budget by promoting the bond whose next class
    is most probable (too many hydrogens) or demoting the one whose lower
    class is most probable (too few), within valence.
    """
    n = len(atom_types)
    if probs.shape != (n, n, N_BOND_CLASSES):
        raise ValueError(f"probs must be ({n}, {n}, {N_BOND_CLASSES}); got {probs.shape}.")
    valence = np.array([valency_table.get(a, 4) for a in atom_types], dtype=np.int64)
    adjacency = np.zeros((n, n), dtype=np.float32)
    remaining = valence.copy()

    i_idx, j_idx = np.triu_indices(n, k=1)
    p_bond = 1.0 - probs[i_idx, j_idx, 0]
    order = np.argsort(-p_bond, kind="stable")
    for idx in order:
        if p_bond[idx] < threshold:
            break
        i, j = int(i_idx[idx]), int(j_idx[idx])
        cap = int(min(remaining[i], remaining[j], N_BOND_CLASSES - 1))
        if cap < 1:
            continue
        bond_probs = probs[i, j, 1 : cap + 1]
        chosen = int(np.argmax(bond_probs)) + 1
        adjacency[i, j] = adjacency[j, i] = chosen
        remaining[i] -= chosen
        remaining[j] -= chosen

    if h_budget is not None:
        for _ in range(max_repair_steps):
            h_now = implied_hydrogens(adjacency, atom_types, valency_table)
            if h_now == h_budget:
                break
            bonded = adjacency > 0
            if h_now > h_budget:
                # Promote the bond whose next class is most probable, within valence.
                best, best_p = None, -1.0
                for a, b in zip(i_idx, j_idx):
                    cur = int(adjacency[a, b])
                    if not bonded[a, b] or cur >= N_BOND_CLASSES - 1:
                        continue
                    if remaining[a] < 1 or remaining[b] < 1:
                        continue
                    p = probs[a, b, cur + 1]
                    if p > best_p:
                        best, best_p = (int(a), int(b)), p
                if best is None:
                    break
                a, b = best
                adjacency[a, b] += 1
                adjacency[b, a] += 1
                remaining[a] -= 1
                remaining[b] -= 1
            else:
                # Demote the multiple bond whose lower class is most probable.
                best, best_p = None, -1.0
                for a, b in zip(i_idx, j_idx):
                    cur = int(adjacency[a, b])
                    if cur < 2:
                        continue
                    p = probs[a, b, cur - 1]
                    if p > best_p:
                        best, best_p = (int(a), int(b)), p
                if best is None:
                    break
                a, b = best
                adjacency[a, b] -= 1
                adjacency[b, a] -= 1
                remaining[a] += 1
                remaining[b] += 1
    return adjacency


def adjacency_to_smiles(
    adjacency: np.ndarray, atom_types: Sequence[str], *, require_connected: bool = True
) -> Optional[str]:
    """Sanitised canonical SMILES for an integer-order adjacency, or ``None``.

    A competition answer is one molecule, so disconnected graphs (a ``.`` in
    the SMILES) are rejected unless ``require_connected`` is false.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rdkit is required.") from exc
    try:
        mol = adjacency_to_rwmol(adjacency, atom_types)
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
    except Exception as exc:  # RDKit native exceptions included
        logger.debug("Sanitisation failed: %s", exc)
        return None
    if require_connected and "." in smiles:
        return None
    return smiles


# ----------------------------------------------------------------------
# Formula enumeration
# ----------------------------------------------------------------------


DEFAULT_ENUM_ELEMENTS: tuple[str, ...] = ("C", "N", "O", "S", "P", "F", "Cl", "Br")
_DEFAULT_MAX_COUNTS: dict[str, int] = {"C": 120, "N": 24, "O": 48, "S": 10, "P": 6, "F": 12, "Cl": 8, "Br": 6, "I": 4, "Si": 4, "B": 4, "Se": 2}
# Lenient element-ratio bounds relative to carbon (after Kind & Fiehn's rules).
_RATIO_MAX: dict[str, float] = {"N": 1.3, "O": 1.2, "S": 0.8, "P": 0.3, "F": 1.5, "Cl": 0.8, "Br": 0.8, "I": 0.5}


def rdbe(counts: dict[str, int]) -> float:
    """Ring-double-bond equivalents ``1 + Σ n_i (v_i - 2) / 2``."""
    return 1.0 + 0.5 * sum(n * (_RDBE_VALENCE.get(e, 2) - 2) for e, n in counts.items())


def formula_string(counts: dict[str, int]) -> str:
    """Hill-order formula string (C, H, then alphabetical)."""
    parts = []
    for element in ["C", "H"] + sorted(e for e in counts if e not in ("C", "H")):
        n = counts.get(element, 0)
        if n <= 0:
            continue
        parts.append(f"{element}{n if n > 1 else ''}")
    return "".join(parts)


def formula_mass(counts: dict[str, int]) -> float:
    return float(sum(ELEMENT_MASSES[e] * n for e, n in counts.items()))


def enumerate_formulas(
    neutral_mass: float,
    *,
    ppm: float = 5.0,
    elements: Sequence[str] = DEFAULT_ENUM_ELEMENTS,
    max_counts: Optional[dict[str, int]] = None,
    max_results: int = 50,
    max_heavy_atoms: int = 120,
    min_carbon: int = 1,
) -> list[str]:
    """Neutral, even-electron CHNOPS(+halogen) formulas within ``ppm`` of the mass.

    Filters: integer RDBE in ``[0, 40]``, ``0.1 <= H/C <= 3.5`` (plus one
    hydrogen of slack for tiny molecules), and element-to-carbon ratio
    caps. Results are sorted by absolute mass error.
    """
    if not np.isfinite(neutral_mass) or neutral_mass <= 0:
        return []
    caps = dict(_DEFAULT_MAX_COUNTS)
    if max_counts:
        caps.update(max_counts)
    heavy = [e for e in elements if e != "H"]
    for e in heavy:
        if e not in ELEMENT_MASSES:
            raise ValueError(f"Unknown element {e!r}.")
    if "C" not in heavy:
        raise ValueError("Carbon must be among the enumerated elements.")
    heavy = ["C"] + [e for e in heavy if e != "C"]
    tolerance = neutral_mass * ppm * 1e-6
    h_mass = ELEMENT_MASSES["H"]
    results: list[tuple[float, str]] = []

    counts: dict[str, int] = {}

    def recurse(position: int, mass_so_far: float, n_heavy: int) -> None:
        if position == len(heavy):
            carbon = counts.get("C", 0)
            if carbon < min_carbon:
                return
            remainder = neutral_mass - mass_so_far
            h = int(round(remainder / h_mass))
            if h < 0:
                return
            total = mass_so_far + h * h_mass
            if abs(total - neutral_mass) > tolerance:
                return
            if h > 3.5 * carbon + 1 or (carbon > 0 and h < 0.1 * carbon - 1):
                return
            full = dict(counts)
            full["H"] = h
            value = rdbe(full)
            if value < 0 or value > 40 or abs(value - round(value)) > 1e-6:
                return
            results.append((abs(total - neutral_mass), formula_string(full)))
            return
        element = heavy[position]
        mass_e = ELEMENT_MASSES[element]
        carbon = counts.get("C", 0) if element != "C" else None
        cap = caps.get(element, 10)
        if element != "C":
            cap = min(cap, int(_RATIO_MAX.get(element, 1.0) * carbon) + 1, max_heavy_atoms - n_heavy)
        for n in range(0, cap + 1):
            new_mass = mass_so_far + n * mass_e
            if new_mass > neutral_mass + tolerance:
                break
            if n:
                counts[element] = n
            elif element in counts:
                del counts[element]
            recurse(position + 1, new_mass, n_heavy + n)
        counts.pop(element, None)

    recurse(0, 0.0, 0)
    results.sort(key=lambda item: (item[0], item[1]))
    seen: set[str] = set()
    out: list[str] = []
    for _, formula in results:
        if formula not in seen:
            seen.add(formula)
            out.append(formula)
        if len(out) >= max_results:
            break
    return out


# ----------------------------------------------------------------------
# De novo candidates
# ----------------------------------------------------------------------


@dataclass
class DeNovoConfig:
    n_samples: int = 32
    sampler: str = "ddim"
    n_steps: int = 50
    eta: float = 0.0
    threshold: float = 0.5
    max_per_formula: int = 10
    sample_batch: int = 256


@torch.no_grad()
def denovo_candidates(
    trainer: CasmiTrainer,
    tensors: SpectraTensors,
    formulas: Sequence[str],
    sgno: BondTypeSGNO,
    *,
    config: Optional[DeNovoConfig] = None,
) -> list[str]:
    """Generate and rank de novo SMILES for each formula, best first.

    Per formula: sample eigenvectors for the formula's heavy-atom count,
    assign elements with the core Hungarian step, decode bond types under
    the formula's hydrogen budget, sanitise, rank by sampling frequency.
    Formulas are interleaved in the order given so each gets its best
    structures near the top. Output is deduplicated by InChIKey14.
    """
    config = config or DeNovoConfig()
    heads = predict_molecule_heads(trainer, tensors)
    memory, memory_mask = heads["memory"], heads["memory_mask"]
    n_spectra = memory.shape[0]
    device = trainer.device
    sgno = sgno.to(device).eval()
    per_formula: list[list[str]] = []

    for formula in formulas:
        try:
            counts = parse_formula(formula)
        except ValueError:
            continue
        n_atoms = sum(counts.values())
        if n_atoms < 1 or n_atoms > trainer.model.max_atoms:
            continue
        h_budget = formula_hydrogens(formula)
        smiles_out: list[Optional[str]] = []
        total = config.n_samples
        done = 0
        while done < total:
            chunk = min(config.sample_batch, total - done)
            spectrum_ids = torch.arange(done, done + chunk, device=device) % n_spectra
            mem = memory.index_select(0, spectrum_ids)
            mask = memory_mask.index_select(0, spectrum_ids)
            eigvecs = trainer.sample(mem, mask, n_atoms=n_atoms, sampler=config.sampler, n_steps=config.n_steps, eta=config.eta)
            if trainer.model.atom_type_head is not None:
                logits = trainer.atom_type_logits(eigvecs, None, mem, mask).detach().cpu().numpy()
            else:
                logits = np.zeros((chunk, n_atoms, N_ELEMENT_TYPES), dtype=np.float32)
            element_lists: list[list[str]] = []
            element_idx = np.full((chunk, n_atoms), UNKNOWN_INDEX, dtype=np.int64)
            for s in range(chunk):
                try:
                    assigned = assign_elements(logits[s], formula)
                except ValueError:
                    assigned = []
                element_lists.append(assigned)
                if assigned:
                    element_idx[s] = [ELEMENT_TO_INDEX.get(e, UNKNOWN_INDEX) for e in assigned]
            probs = sgno.bond_probabilities(eigvecs, torch.from_numpy(element_idx).to(device)).detach().cpu().numpy()
            for s in range(chunk):
                if not element_lists[s]:
                    smiles_out.append(None)
                    continue
                adjacency = decode_bond_types(
                    probs[s], element_lists[s], threshold=config.threshold, h_budget=h_budget
                )
                smiles_out.append(adjacency_to_smiles(adjacency, element_lists[s]))
            done += chunk
        per_formula.append(rank_samples_by_frequency(smiles_out)[: config.max_per_formula])

    merged: list[str] = []
    seen: set[str] = set()
    depth = max((len(lst) for lst in per_formula), default=0)
    for rank in range(depth):
        for lst in per_formula:
            if rank < len(lst):
                key = inchikey14(lst[rank])
                if key is not None and key not in seen:
                    seen.add(key)
                    merged.append(lst[rank])
    return merged


def make_denovo_fn(sgno: BondTypeSGNO, config: Optional[DeNovoConfig] = None):
    """Adapter matching :data:`spec2graph.casmi.predict.DeNovoFn`."""

    def _fn(trainer: CasmiTrainer, tensors: SpectraTensors, formulas: list[str]) -> list[str]:
        return denovo_candidates(trainer, tensors, formulas, sgno, config=config)

    return _fn


def save_bond_sgno(path, sgno: BondTypeSGNO) -> None:
    torch.save({"format": "spec2graph-casmi-bondsgno-v1", "config": sgno.config_dict(), "state_dict": sgno.state_dict()}, path)


def load_bond_sgno(path, device: str | torch.device = "cpu") -> BondTypeSGNO:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "spec2graph-casmi-bondsgno-v1":
        raise ValueError("Unrecognised bond-type SGNO checkpoint.")
    sgno = BondTypeSGNO(**payload["config"]).to(device)
    sgno.load_state_dict(payload["state_dict"])
    return sgno.eval()


__all__ = [
    "BondTypeSGNO",
    "BondTypeTrainer",
    "BondTypeTrainerConfig",
    "DeNovoConfig",
    "DEFAULT_ENUM_ELEMENTS",
    "ELEMENT_MASSES",
    "N_BOND_CLASSES",
    "adjacency_to_smiles",
    "bond_targets_from_adjacency",
    "decode_bond_types",
    "denovo_candidates",
    "enumerate_formulas",
    "formula_hydrogens",
    "formula_mass",
    "formula_string",
    "implied_hydrogens",
    "load_bond_sgno",
    "make_denovo_fn",
    "rdbe",
    "save_bond_sgno",
]
