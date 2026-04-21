"""Eigenvectors → SMILES inference pipeline.

This module turns model outputs into molecular graphs. At a high level:

    V̂_k          atom-type logits      formula
       \\                |                /
        \\               |               /
         +--> SGNO ---> bond probs    Hungarian assignment --> atom list
                           |                                      |
                           v                                      v
                       ValencyDecoder --> adjacency --> RWMol --> SMILES

Each stage is exposed as an individual function so callers can mix and
match (e.g. a benchmark that wants to report adjacency accuracy without
going all the way to SMILES). The full convenience entrypoint is
:func:`eigvecs_to_smiles` for single samples and
:func:`batch_eigvecs_to_smiles` for batches.

**Atom ordering caveat.** The SGNO is permutation-equivariant, so the
model need not produce atoms in any canonical order. What it does need
is *consistent* orderings between rows of `V̂_k` and rows of the atom-
type logits — both come from the same decoder state, so that alignment
is automatic. The Hungarian step then assigns elements from the
molecular formula to those rows optimally. This dodges the original
formula-derived-ordering heuristic that produced nearly-random element
labels.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

import numpy as np
import torch

from spec2graph.data.elements import (
    ELEMENTS,
    ELEMENT_TO_INDEX,
    N_ELEMENT_TYPES,
    UNKNOWN_INDEX,
)
from spectral_diffusion import SpectralGraphNeuralOperator, ValencyDecoder

logger = logging.getLogger(__name__)

# Element symbol pattern. Matches one uppercase letter optionally followed
# by a single lowercase letter — the standard form of a heavy-atom symbol.
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

# Hydrogen is excluded from the heavy-atom graph — the diffusion model
# never sees it in V_k, so we must not try to place it either.
_HYDROGEN = "H"

# Canonical display ordering for well-known elements. Anything outside
# this list is sorted alphabetically at the end.
_STANDARD_ORDER = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "B", "Se")


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a molecular formula like ``'C17H26ClNO2'`` into a counts dict.

    The returned dict maps element symbols → integer counts, excluding
    hydrogen since hydrogens are implicit in the heavy-atom graph.

    Raises
    ------
    ValueError
        If ``formula`` is empty or contains no parseable element tokens.
    """
    if not formula:
        raise ValueError("formula is empty.")
    counts: dict[str, int] = {}
    for symbol, count_str in _FORMULA_TOKEN.findall(formula):
        if not symbol:
            continue
        if symbol == _HYDROGEN:
            continue
        count = int(count_str) if count_str else 1
        counts[symbol] = counts.get(symbol, 0) + count
    if not counts:
        raise ValueError(f"No heavy-atom tokens in formula {formula!r}.")
    return counts


def formula_to_atom_types(formula: str) -> list[str]:
    """Expand a formula into a list of element symbols in canonical order.

    Ordering: the elements listed in :data:`_STANDARD_ORDER` first (in
    that order), then any remaining elements sorted alphabetically.
    Hydrogen is excluded.

    This matches the original spec's heuristic for the *slot list* that
    feeds the Hungarian assignment — it's not the ordering the model
    must produce; it's the ordering of the columns in the bipartite
    cost matrix.
    """
    counts = parse_formula(formula)
    ordered: list[str] = []
    standard_set = set(_STANDARD_ORDER)
    for element in _STANDARD_ORDER:
        if element in counts:
            ordered.extend([element] * counts[element])
    remaining = sorted(e for e in counts.keys() if e not in standard_set)
    for element in remaining:
        ordered.extend([element] * counts[element])
    return ordered


def assign_elements(
    atom_type_logits: np.ndarray,
    formula: str,
) -> list[str]:
    """Assign element symbols to atom slots via balanced Hungarian matching.

    Takes a ``(n_atoms, n_element_types)`` array of log-probabilities (or
    plain logits — they only need to be monotone with probability, the
    scale is ignored by the argmin) and returns an element symbol per
    atom slot. The result respects the formula's heavy-atom counts
    exactly.

    Parameters
    ----------
    atom_type_logits:
        Per-atom element logits. Typically the output of
        :meth:`Spec2GraphDiffusion.predict_atom_types`. Shape must be
        ``(n_atoms, N_ELEMENT_TYPES)`` — the extra slot at
        ``UNKNOWN_INDEX`` is safe to include; the function never assigns
        from it because no formula slot maps to ``UNKNOWN``.
    formula:
        Molecular formula string. Heavy-atom counts must equal
        ``n_atoms``; otherwise a :class:`ValueError` is raised.

    Returns
    -------
    list[str]
        Element symbol per atom slot. ``result[i]`` is the element
        assigned to row ``i`` of ``atom_type_logits``.

    Notes
    -----
    Uses :func:`scipy.optimize.linear_sum_assignment`. The cost matrix
    has shape ``(n_atoms, n_slots)`` with ``cost[i, j] = -logits[i,
    element_to_index(slots[j])]`` — minimising cost maximises log-prob
    sum across the matched pairs.
    """
    if atom_type_logits.ndim != 2:
        raise ValueError(
            f"atom_type_logits must be 2D (n_atoms, n_element_types); got shape "
            f"{atom_type_logits.shape}."
        )
    if atom_type_logits.shape[1] < N_ELEMENT_TYPES:
        raise ValueError(
            f"atom_type_logits has {atom_type_logits.shape[1]} columns but the "
            f"vocabulary has {N_ELEMENT_TYPES} entries."
        )

    slots = formula_to_atom_types(formula)
    n_atoms = atom_type_logits.shape[0]
    if len(slots) != n_atoms:
        raise ValueError(
            f"Formula {formula!r} expands to {len(slots)} heavy atoms, but "
            f"atom_type_logits has {n_atoms} rows. Either the predicted "
            "atom count is wrong or the formula is mismatched."
        )

    # Map each slot to its vocab index. Unknown elements (not in
    # ELEMENT_TO_INDEX) would point at UNKNOWN_INDEX, which is a valid
    # column in the logits but almost certainly not what the user
    # intended — warn and continue.
    slot_indices: list[int] = []
    for symbol in slots:
        if symbol in ELEMENT_TO_INDEX:
            slot_indices.append(ELEMENT_TO_INDEX[symbol])
        else:
            logger.warning(
                "Element %s is not in the vocabulary; mapping to UNKNOWN.",
                symbol,
            )
            slot_indices.append(UNKNOWN_INDEX)

    # cost[i, j] = -logits[i, slot_indices[j]]. Built by fancy indexing.
    cost = -atom_type_logits[:, slot_indices]

    from scipy.optimize import linear_sum_assignment

    atom_idx, slot_idx = linear_sum_assignment(cost)

    result: list[Optional[str]] = [None] * n_atoms
    for a, s in zip(atom_idx, slot_idx):
        result[int(a)] = slots[int(s)]
    assert all(x is not None for x in result), (
        "Hungarian assignment left an atom unmatched; this should be impossible "
        "when the cost matrix is square."
    )
    return [r for r in result if r is not None]  # mypy-friendly


# ---------------------------------------------------------------------------
# Eigenvectors → SMILES
# ---------------------------------------------------------------------------


def adjacency_to_rwmol(
    adjacency: np.ndarray,
    atom_types: Sequence[str],
):
    """Build an RDKit :class:`~rdkit.Chem.RWMol` from adjacency + atom list.

    Bond orders are inferred from the integer entries of ``adjacency``:
    1 → single, 2 → double, 3 → triple. Values outside ``{0, 1, 2, 3}``
    raise. Returns the ``RWMol`` *before* sanitisation — the caller is
    responsible for invoking :func:`rdkit.Chem.SanitizeMol` and handling
    failures.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("rdkit is required for adjacency_to_rwmol") from exc

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"adjacency must be square 2D; got shape {adjacency.shape}."
        )
    n = adjacency.shape[0]
    if len(atom_types) != n:
        raise ValueError(
            f"atom_types length ({len(atom_types)}) must equal adjacency size ({n})."
        )

    bond_type_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }

    mol = Chem.RWMol()
    for symbol in atom_types:
        mol.AddAtom(Chem.Atom(symbol))

    for i in range(n):
        for j in range(i + 1, n):
            order = int(round(float(adjacency[i, j])))
            if order == 0:
                continue
            if order not in bond_type_map:
                raise ValueError(
                    f"Unsupported bond order {order} at ({i}, {j}); expected 0-3."
                )
            mol.AddBond(i, j, bond_type_map[order])
    return mol


def eigvecs_to_smiles(
    eigvecs: torch.Tensor,
    atom_types: Sequence[str],
    sgno: SpectralGraphNeuralOperator,
    valency_decoder: ValencyDecoder,
    *,
    threshold: float = 0.3,
    max_bond_order: int = 1,
) -> Optional[str]:
    """Decode one eigenvector tensor into a canonical SMILES string.

    Parameters
    ----------
    eigvecs:
        Tensor of shape ``(n_atoms, k)`` — a single sample with no batch
        dimension.
    atom_types:
        Element symbols per atom slot, in the same ordering as the rows
        of ``eigvecs``. Typically produced by :func:`assign_elements`.
    sgno, valency_decoder:
        A trained SGNO and the :class:`ValencyDecoder`. The SGNO must
        have been initialised with the same ``k`` as the eigenvectors.
    threshold, max_bond_order:
        Forwarded to :meth:`ValencyDecoder.decode`.

    Returns
    -------
    str | None
        Canonical SMILES, or ``None`` if RDKit sanitisation fails (e.g.
        if the predicted adjacency violates valence or produces an
        unreasonable kekulisation).
    """
    if eigvecs.dim() != 2:
        raise ValueError(
            f"eigvecs must be 2D (n_atoms, k); got shape {tuple(eigvecs.shape)}."
        )

    with torch.no_grad():
        bond_probs = sgno.bond_probabilities(eigvecs.unsqueeze(0)).squeeze(0)
    bond_probs_np = bond_probs.detach().cpu().numpy()

    adjacency = valency_decoder.decode(
        atom_types=list(atom_types),
        bond_probs=bond_probs_np,
        threshold=threshold,
        max_bond_order=max_bond_order,
    )
    return _safe_sanitise(adjacency, atom_types)


def _safe_sanitise(
    adjacency: np.ndarray, atom_types: Sequence[str]
) -> Optional[str]:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("rdkit is required to canonicalise SMILES.") from exc
    try:
        mol = adjacency_to_rwmol(adjacency, atom_types)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except (ValueError, RuntimeError) as exc:
        logger.debug("Sanitisation failed: %s", exc)
        return None
    except Exception as exc:  # RDKit raises some native exceptions.
        logger.debug("Sanitisation raised unexpected error: %s", exc)
        return None


def batch_eigvecs_to_smiles(
    eigvecs: torch.Tensor,
    atom_masks: torch.Tensor,
    formulas: Sequence[str],
    atom_type_logits: Optional[torch.Tensor],
    sgno: SpectralGraphNeuralOperator,
    valency_decoder: ValencyDecoder,
    *,
    threshold: float = 0.3,
    max_bond_order: int = 1,
    fallback_atom_types: Optional[Sequence[Sequence[str]]] = None,
) -> list[Optional[str]]:
    """Batched wrapper around :func:`eigvecs_to_smiles`.

    Handles variable atom counts per sample via ``atom_masks`` and runs
    the Hungarian assignment once per sample to turn ``atom_type_logits``
    into concrete element lists. If ``atom_type_logits`` is ``None``,
    falls back to ``fallback_atom_types[i]`` (typically derived from the
    formula via :func:`formula_to_atom_types`) — useful when the atom-
    type head is disabled.

    Returns a list of ``Optional[str]``; each ``None`` indicates that
    RDKit rejected the predicted graph.
    """
    if eigvecs.dim() != 3:
        raise ValueError(
            f"eigvecs must be 3D (batch, n_atoms, k); got {tuple(eigvecs.shape)}."
        )
    batch_size = eigvecs.shape[0]
    if len(formulas) != batch_size:
        raise ValueError(
            f"len(formulas)={len(formulas)} does not match batch size {batch_size}."
        )

    atom_type_logits_np: Optional[np.ndarray]
    if atom_type_logits is not None:
        atom_type_logits_np = atom_type_logits.detach().cpu().numpy()
    else:
        atom_type_logits_np = None

    results: list[Optional[str]] = []
    for i in range(batch_size):
        n_valid = int(atom_masks[i].sum().item())
        if n_valid == 0:
            results.append(None)
            continue
        sample_eigvecs = eigvecs[i, :n_valid, :]

        if atom_type_logits_np is not None:
            sample_logits = atom_type_logits_np[i, :n_valid, :]
            try:
                atom_types = assign_elements(sample_logits, formulas[i])
            except ValueError as exc:
                logger.debug("Element assignment failed: %s", exc)
                results.append(None)
                continue
        elif fallback_atom_types is not None:
            atom_types = list(fallback_atom_types[i])[:n_valid]
            if len(atom_types) != n_valid:
                logger.debug(
                    "Fallback atom types length (%d) != n_valid (%d); skipping.",
                    len(atom_types),
                    n_valid,
                )
                results.append(None)
                continue
        else:
            # Without atom-type logits and without a fallback we have no
            # element identity. Derive it from the formula directly —
            # this is the pre-head heuristic and is known to produce poor
            # top-k numbers, but is the right behaviour when no other
            # source of element identity exists.
            try:
                atom_types = formula_to_atom_types(formulas[i])
                if len(atom_types) != n_valid:
                    results.append(None)
                    continue
            except ValueError:
                results.append(None)
                continue

        smiles = eigvecs_to_smiles(
            eigvecs=sample_eigvecs,
            atom_types=atom_types,
            sgno=sgno,
            valency_decoder=valency_decoder,
            threshold=threshold,
            max_bond_order=max_bond_order,
        )
        results.append(smiles)
    return results
