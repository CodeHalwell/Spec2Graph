"""Collators that pad per-batch and wrap the result in a :class:`TrainingBatch`.

Two collators are exposed:

* :func:`make_training_batch_collator` — returns a callable suitable for
  :class:`torch.utils.data.DataLoader`'s ``collate_fn`` argument. Produces
  a :class:`TrainingBatch` with all tensors on the requested device.
* :func:`make_metadata_collator` — a superset collator that additionally
  returns a Python-side metadata dict (SMILES, formula, InChIKey, …).
  Useful for the evaluation loop, where metrics need the ground-truth
  SMILES alongside the predicted tensors.

Mask convention follows the rest of the codebase: ``True = valid``. The
collator validates this invariant and drops all-padding rows before
they hit the model's :meth:`_validate_mask`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from spec2graph.data.elements import PADDING_INDEX
from spectral_diffusion import TrainingBatch

logger = logging.getLogger(__name__)

BatchCollator = Callable[[list[dict[str, Any]]], TrainingBatch]


def _validate_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop samples with no valid atoms or peaks and log if any slip through."""
    kept: list[dict[str, Any]] = []
    for sample in samples:
        if int(sample.get("n_atoms", 0)) == 0:
            logger.warning("Dropping sample with n_atoms=0 (inchikey=%s).", sample.get("inchikey"))
            continue
        if sample["mz"].size == 0:
            logger.warning("Dropping sample with no peaks (inchikey=%s).", sample.get("inchikey"))
            continue
        kept.append(sample)
    return kept


def _pad_tensor(arr: np.ndarray, target_len: int, axis: int = 0) -> np.ndarray:
    """Zero-pad ``arr`` along ``axis`` up to ``target_len``."""
    pad_len = target_len - arr.shape[axis]
    if pad_len <= 0:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, pad_len)
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def make_training_batch_collator(
    max_atoms: int,
    max_peaks: int,
    k: int,
    fingerprint_bits: int,
    device: str = "cpu",
    include_atom_type_targets: bool = True,
) -> BatchCollator:
    """Return a collator that produces a :class:`TrainingBatch`.

    The returned callable:

    1. Drops samples with zero valid atoms or peaks.
    2. Computes per-batch ``n`` and ``p`` capped at ``max_atoms`` and
       ``max_peaks`` respectively.
    3. Zero-pads numpy arrays and fills boolean masks.
    4. Moves the result onto ``device``.
    """

    def collate(samples: list[dict[str, Any]]) -> TrainingBatch:
        samples = _validate_samples(samples)
        if not samples:
            raise ValueError("All samples in the batch were dropped during validation.")

        batch_size = len(samples)
        batch_max_atoms = min(max(int(s["n_atoms"]) for s in samples), max_atoms)
        batch_max_peaks = min(max(int(s["mz"].shape[0]) for s in samples), max_peaks)

        x_0 = np.zeros((batch_size, batch_max_atoms, k), dtype=np.float32)
        mz = np.zeros((batch_size, batch_max_peaks), dtype=np.float32)
        intensity = np.zeros((batch_size, batch_max_peaks), dtype=np.float32)
        atom_mask = np.zeros((batch_size, batch_max_atoms), dtype=bool)
        spectrum_mask = np.zeros((batch_size, batch_max_peaks), dtype=bool)
        fingerprints = np.zeros((batch_size, fingerprint_bits), dtype=np.float32)
        atom_count_targets = np.zeros(batch_size, dtype=np.float32)
        precursor_mzs = np.zeros(batch_size, dtype=np.float32)
        atom_type_targets = np.full(
            (batch_size, batch_max_atoms), PADDING_INDEX, dtype=np.int64
        )

        for i, sample in enumerate(samples):
            n = min(int(sample["n_atoms"]), batch_max_atoms)
            p = min(int(sample["mz"].shape[0]), batch_max_peaks)
            x_0[i, :n, :] = sample["eigvecs"][:n, :k]
            atom_mask[i, :n] = True
            mz[i, :p] = sample["mz"][:p]
            intensity[i, :p] = sample["intensity"][:p]
            spectrum_mask[i, :p] = True
            fingerprints[i] = sample["fingerprint"][:fingerprint_bits]
            atom_count_targets[i] = n
            precursor_mzs[i] = sample["precursor_mz"]
            atom_types = np.asarray(sample["atom_types"], dtype=np.int64)[:n]
            atom_type_targets[i, :n] = atom_types

        def _to_tensor(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.from_numpy(arr).to(dtype=dtype, device=device)

        return TrainingBatch(
            x_0=_to_tensor(x_0, torch.float32),
            mz=_to_tensor(mz, torch.float32),
            intensity=_to_tensor(intensity, torch.float32),
            atom_mask=torch.from_numpy(atom_mask).to(device=device),
            spectrum_mask=torch.from_numpy(spectrum_mask).to(device=device),
            precursor_mz=_to_tensor(precursor_mzs, torch.float32),
            fingerprint_targets=_to_tensor(fingerprints, torch.float32),
            atom_count_targets=_to_tensor(atom_count_targets, torch.float32),
            atom_type_targets=(
                torch.from_numpy(atom_type_targets).to(device=device)
                if include_atom_type_targets
                else None
            ),
        )

    return collate


@dataclass
class CollatedEvalBatch:
    """Pair of a :class:`TrainingBatch` and its Python-side metadata."""

    batch: TrainingBatch
    metadata: dict[str, list[Any]]
    adjacencies: list[np.ndarray] | None = None


def make_metadata_collator(
    max_atoms: int,
    max_peaks: int,
    k: int,
    fingerprint_bits: int,
    device: str = "cpu",
) -> Callable[[list[dict[str, Any]]], CollatedEvalBatch]:
    """Return a collator that yields :class:`CollatedEvalBatch` for evaluation.

    Adds ``smiles`` / ``formula`` / ``inchikey`` / ``atom_symbols`` lists
    alongside the padded tensors. Uses the same padding rules as
    :func:`make_training_batch_collator` so the resulting batch can still
    be fed straight into the model.
    """
    base_collator = make_training_batch_collator(
        max_atoms=max_atoms,
        max_peaks=max_peaks,
        k=k,
        fingerprint_bits=fingerprint_bits,
        device=device,
    )

    def collate(samples: list[dict[str, Any]]) -> CollatedEvalBatch:
        samples = _validate_samples(samples)
        if not samples:
            raise ValueError("All samples in the batch were dropped during validation.")
        batch = base_collator(samples)
        metadata = {
            "smiles": [s["smiles"] for s in samples],
            "formula": [s["formula"] for s in samples],
            "inchikey": [s["inchikey"] for s in samples],
            "atom_symbols": [list(s.get("atom_symbols", [])) for s in samples],
            "n_atoms": [int(s["n_atoms"]) for s in samples],
            "precursor_mz": [float(s["precursor_mz"]) for s in samples],
        }
        adjacencies = None
        if "adjacency" in samples[0]:
            adjacencies = [np.asarray(s["adjacency"], dtype=np.float32) for s in samples]
        return CollatedEvalBatch(batch=batch, metadata=metadata, adjacencies=adjacencies)

    return collate
