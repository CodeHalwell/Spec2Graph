"""Element vocabulary for the atom-type head.

The model predicts per-atom element logits over a fixed vocabulary. This
module defines that vocabulary, along with the padding sentinel used by
the loss and helpers for the atom-type labels produced by the dataset.

Keeping the vocab in one place means the model, the dataset, and the
inference-time assignment code all agree on element ↔ index mapping.

The 12 elements listed here are the keys of ``VALENCY_TABLE`` in
:mod:`spectral_diffusion`, which in turn covers every heavy atom that
appears in drug-like MassSpecGym molecules. One additional "unknown"
slot catches anything outside that vocabulary (e.g. metalloid edge
cases) without silently casting them to carbon.
"""

from __future__ import annotations

from typing import Sequence

# Fixed ordering. Do not reorder — this is a model output space, and
# re-indexing would silently invalidate any trained weights.
ELEMENTS: tuple[str, ...] = (
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "Si",
    "B",
    "Se",
)

UNKNOWN_ELEMENT = "?"
UNKNOWN_INDEX = len(ELEMENTS)                  # 12
N_ELEMENT_TYPES = len(ELEMENTS) + 1            # 13

# Sentinel index passed to :class:`torch.nn.CrossEntropyLoss` via
# ``ignore_index``. Matches PyTorch's default so callers that rely on the
# default behaviour work without extra wiring.
PADDING_INDEX: int = -100

ELEMENT_TO_INDEX: dict[str, int] = {element: idx for idx, element in enumerate(ELEMENTS)}
INDEX_TO_ELEMENT: dict[int, str] = {idx: element for element, idx in ELEMENT_TO_INDEX.items()}
INDEX_TO_ELEMENT[UNKNOWN_INDEX] = UNKNOWN_ELEMENT


def element_to_index(symbol: str) -> int:
    """Return the vocab index of ``symbol``, or :data:`UNKNOWN_INDEX` if unseen.

    Callers that want strict behaviour should compare the result to
    ``UNKNOWN_INDEX`` themselves rather than relying on an exception.
    """
    return ELEMENT_TO_INDEX.get(symbol, UNKNOWN_INDEX)


def index_to_element(index: int) -> str:
    """Inverse of :func:`element_to_index`.

    Raises :class:`KeyError` for an out-of-range index. This is stricter
    than the forward direction because an unexpected index here usually
    means a bug, not noisy input.
    """
    return INDEX_TO_ELEMENT[index]


def atom_types_to_indices(atom_types: Sequence[str]) -> list[int]:
    """Map a sequence of element symbols to vocab indices.

    Parameters
    ----------
    atom_types:
        Iterable of element symbols in some canonical ordering. The
        ordering is caller-controlled — typically the canonical atom
        sequence RDKit produces for the molecule.
    """
    return [element_to_index(symbol) for symbol in atom_types]
