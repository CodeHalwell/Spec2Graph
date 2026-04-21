"""Permutation augmentation for training batches.

The learned ``atom_pos_embedding`` in :class:`Spec2GraphDiffusion` makes
the transformer decoder implicitly dependent on the atom ordering that
:class:`SpectralDataProcessor` produces. That ordering is typically
stable (RDKit canonical), but only because training always uses it.

Permutation augmentation addresses the robustness side: during training,
randomly permute each example's atom rows across ``x_0``,
``atom_mask``, ``atom_type_targets`` (if present), and any downstream
per-atom targets. Over enough steps this teaches the model that no
particular atom ordering is special, so at inference time the SGNO can
consume whatever ordering the decoder produces.

Two public entry points:

* :func:`permute_training_batch` — does the permutation in-place
  (returns a new :class:`TrainingBatch`) given an existing one.
* :func:`wrap_collator_with_permutation` — wraps a collator so every
  training step sees freshly permuted batches, matching the augmentation
  idiom used in many PyTorch pipelines.

Both are opt-in. Default training behaviour is unchanged.
"""

from __future__ import annotations

from typing import Callable

import torch

from spec2graph.data.elements import PADDING_INDEX
from spectral_diffusion import TrainingBatch


def _per_row_permutation(atom_mask: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Return a ``(batch, n_atoms)`` long tensor permuting real atoms per row.

    Only the valid positions are permuted; padded positions retain their
    original index so the mask stays bit-identical after ``gather``. This
    keeps the collator contract that padded positions never leak into the
    loss.
    """
    batch, n_atoms = atom_mask.shape
    device = atom_mask.device
    # Start with identity (so padded positions stay where they are).
    indices = torch.arange(n_atoms, device=device).unsqueeze(0).expand(batch, -1).clone()
    for i in range(batch):
        valid = atom_mask[i].nonzero(as_tuple=False).flatten()
        if valid.numel() <= 1:
            continue
        shuffle = valid[torch.randperm(valid.numel(), generator=generator, device=device)]
        indices[i, valid] = shuffle
    return indices


def permute_training_batch(
    batch: TrainingBatch,
    generator: torch.Generator | None = None,
) -> TrainingBatch:
    """Return a :class:`TrainingBatch` with atom rows randomly permuted.

    Leaves all spectrum-side fields, the fingerprint, the atom count,
    and the eigenvalue targets untouched. Permutes the atom axis of
    ``x_0``, ``atom_mask`` (redundant but keeps the relationship
    explicit) and ``atom_type_targets``.
    """
    if batch.atom_mask is None:
        raise ValueError(
            "permute_training_batch requires a non-None atom_mask on the batch."
        )

    permutation = _per_row_permutation(batch.atom_mask, generator=generator)
    batch_size, n_atoms = batch.atom_mask.shape

    # ``gather`` along the atom axis. For x_0 the permutation must be
    # broadcast to the feature dim; for the 2D tensors it applies
    # directly.
    x0_perm = batch.x_0.gather(
        1, permutation.unsqueeze(-1).expand(batch_size, n_atoms, batch.x_0.shape[-1])
    )
    mask_perm = batch.atom_mask.gather(1, permutation)

    if batch.atom_type_targets is not None:
        # Padded positions currently hold PADDING_INDEX; because the
        # permutation leaves padded positions alone they stay as
        # PADDING_INDEX, which is what we want.
        atom_type_perm = batch.atom_type_targets.gather(1, permutation)
    else:
        atom_type_perm = None

    return TrainingBatch(
        x_0=x0_perm,
        mz=batch.mz,
        intensity=batch.intensity,
        atom_mask=mask_perm,
        spectrum_mask=batch.spectrum_mask,
        precursor_mz=batch.precursor_mz,
        fingerprint_targets=batch.fingerprint_targets,
        atom_count_targets=batch.atom_count_targets,
        eigenvalue_targets=batch.eigenvalue_targets,
        atom_type_targets=atom_type_perm,
    )


def wrap_collator_with_permutation(
    collator: Callable[[list[dict]], TrainingBatch],
    generator: torch.Generator | None = None,
) -> Callable[[list[dict]], TrainingBatch]:
    """Wrap a collator so its output is permuted before being returned."""

    def wrapped(samples: list[dict]) -> TrainingBatch:
        batch = collator(samples)
        return permute_training_batch(batch, generator=generator)

    return wrapped
