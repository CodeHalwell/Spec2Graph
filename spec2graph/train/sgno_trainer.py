"""Supervised training for :class:`SpectralGraphNeuralOperator`.

The diffusion trainer learns to denoise eigenvectors. The SGNO, by
contrast, needs supervision from ground-truth ``(V_k, adjacency)``
pairs. Without that, the SGNO's bond probabilities are random and the
end-to-end benchmark top-k accuracy is pinned near zero regardless of
how good the diffusion model is.

This module adds a lightweight trainer with the same shape as
:class:`DiffusionTrainer.train_step` — call ``train_step`` per batch,
inspect the loss, done. It does not own its optimizer (the caller
does), matching the existing pattern.

Loss
----
Binary cross-entropy on the off-diagonal bond logits with an adjacency
mask to skip padded atom pairs. Self-loops (diagonal) are ignored by
the SGNO and excluded here too. Ground-truth adjacency values are
binarised (any non-zero bond is treated as a positive label), which
matches the behaviour of the default ValencyDecoder at inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from spectral_diffusion import SpectralGraphNeuralOperator

logger = logging.getLogger(__name__)


@dataclass
class SGNOTrainerConfig:
    """Configuration for :class:`SGNOTrainer`."""

    # If > 0, re-weights the minority class in the BCE loss to partially
    # counter the extreme class imbalance (most pairs are non-bonds).
    pos_weight: float = 0.0


def adjacency_targets_from_batch(
    adjacencies: list[np.ndarray],
    atom_mask: torch.Tensor,
    *,
    threshold: float = 0.5,
    device: Optional[str | torch.device] = None,
) -> torch.Tensor:
    """Pad a list of ground-truth adjacencies to a batched tensor.

    Parameters
    ----------
    adjacencies:
        Per-example heavy-atom adjacency matrices (numpy, shape
        ``(n_i, n_i)``). Produced by the dataset when
        ``include_adjacency=True`` is passed.
    atom_mask:
        Boolean ``(batch, n_atoms)`` mask from the training batch;
        determines the padded-up adjacency size.
    threshold:
        Bond orders strictly greater than ``threshold`` become positive
        labels (1.0); everything else becomes 0. The default of 0.5
        keeps any integer bond order (single=1, double=2, triple=3, or
        aromatic=1.5) as "bond present".

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(batch, n_atoms, n_atoms)`` with values
        in ``{0.0, 1.0}``. Diagonal entries are zero.
    """
    batch_size, max_atoms = atom_mask.shape
    if len(adjacencies) != batch_size:
        raise ValueError(
            f"len(adjacencies)={len(adjacencies)} does not match batch size "
            f"{batch_size}."
        )

    target = np.zeros((batch_size, max_atoms, max_atoms), dtype=np.float32)
    for i, adj in enumerate(adjacencies):
        n = min(adj.shape[0], max_atoms)
        target[i, :n, :n] = (adj[:n, :n] > threshold).astype(np.float32)
    # Zero the diagonal to match the SGNO's self-loop handling.
    target[:, np.arange(max_atoms), np.arange(max_atoms)] = 0.0

    tensor = torch.from_numpy(target)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


class SGNOTrainer:
    """Supervised trainer for :class:`SpectralGraphNeuralOperator`.

    Usage::

        trainer = SGNOTrainer(sgno, config=SGNOTrainerConfig())
        optimiser = torch.optim.Adam(sgno.parameters(), lr=1e-3)

        for eval_batch in dataloader:       # CollatedEvalBatch from metadata collator
            loss = trainer.train_step(
                optimiser=optimiser,
                eigvecs=eval_batch.batch.x_0,
                atom_mask=eval_batch.batch.atom_mask,
                adjacencies=eval_batch.adjacencies,
            )
    """

    def __init__(
        self,
        sgno: SpectralGraphNeuralOperator,
        config: Optional[SGNOTrainerConfig] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.sgno = sgno
        self.config = config or SGNOTrainerConfig()
        self.device = device

    def _bce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Build an edge-level mask: valid iff both endpoints are real
        # atoms and the edge is off-diagonal. Zero the diagonal in-place
        # via a view so we don't allocate an n×n identity per call.
        edge_mask = (atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)).clone()
        edge_mask.diagonal(dim1=-2, dim2=-1).fill_(False)

        if not edge_mask.any():
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        pos_weight = None
        if self.config.pos_weight > 0:
            pos_weight = torch.tensor(
                self.config.pos_weight, device=logits.device, dtype=logits.dtype
            )

        # Flatten the unmasked entries for a clean loss mean.
        per_element = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=pos_weight
        )
        return (per_element * edge_mask.float()).sum() / edge_mask.float().sum()

    def compute_loss(
        self,
        eigvecs: torch.Tensor,
        atom_mask: torch.Tensor,
        adjacencies: list[np.ndarray],
    ) -> torch.Tensor:
        """Return the supervised BCE loss for one batch."""
        targets = adjacency_targets_from_batch(
            adjacencies, atom_mask, device=eigvecs.device
        )
        logits = self.sgno(eigvecs)
        return self._bce_loss(logits, targets, atom_mask)

    def train_step(
        self,
        optimiser: torch.optim.Optimizer,
        eigvecs: torch.Tensor,
        atom_mask: torch.Tensor,
        adjacencies: list[np.ndarray],
    ) -> float:
        """Run one gradient step. Returns the scalar loss."""
        self.sgno.train()
        optimiser.zero_grad()
        loss = self.compute_loss(eigvecs, atom_mask, adjacencies)
        loss.backward()
        optimiser.step()
        return float(loss.item())
