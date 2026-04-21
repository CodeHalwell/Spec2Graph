"""Tests for DDIM sampling and permutation augmentation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spec2graph.data.augment import (
    _per_row_permutation,
    permute_training_batch,
    wrap_collator_with_permutation,
)
from spec2graph.data.elements import PADDING_INDEX
from spec2graph.eval.ddim import _make_timestep_schedule, ddim_sample
from spectral_diffusion import (
    DiffusionTrainer,
    Spec2GraphDiffusion,
    Spec2GraphDiffusionConfig,
    TrainerConfig,
    TrainingBatch,
)


# ----------------------------------------------------------------------
# DDIM schedule
# ----------------------------------------------------------------------


class TestDDIMSchedule:
    def test_descending_order(self):
        schedule = _make_timestep_schedule(n_steps=5, max_t=50)
        assert schedule == sorted(schedule, reverse=True)

    def test_hits_endpoints(self):
        schedule = _make_timestep_schedule(n_steps=5, max_t=50)
        assert schedule[0] == 49  # highest timestep
        assert schedule[-1] == 0

    def test_deduplicates_when_oversaturated(self):
        # If n_steps > max_t, the schedule collapses to every timestep.
        schedule = _make_timestep_schedule(n_steps=100, max_t=10)
        assert schedule == list(range(9, -1, -1))

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            _make_timestep_schedule(n_steps=0, max_t=10)
        with pytest.raises(ValueError):
            _make_timestep_schedule(n_steps=5, max_t=0)


# ----------------------------------------------------------------------
# DDIM end-to-end
# ----------------------------------------------------------------------


def _tiny_trainer():
    config = Spec2GraphDiffusionConfig(
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        k=4,
        max_atoms=8,
        max_peaks=8,
        dropout=0.0,
        enable_atom_count_head=True,
    )
    model = Spec2GraphDiffusion(config).eval()
    trainer = DiffusionTrainer(
        model=model, config=TrainerConfig(n_timesteps=20), device="cpu"
    )
    return trainer


class TestDDIMSample:
    def test_output_shape(self):
        torch.manual_seed(0)
        trainer = _tiny_trainer()
        mz = torch.rand(2, 4) * 100
        intensity = torch.rand(2, 4)
        out = ddim_sample(
            trainer, mz, intensity, n_atoms=4, n_steps=5
        )
        assert out.shape == (2, 4, 4)

    def test_deterministic_at_eta_zero(self):
        # With a fixed initial noise and eta=0, DDIM is deterministic.
        torch.manual_seed(0)
        trainer = _tiny_trainer()
        mz = torch.rand(1, 4) * 100
        intensity = torch.rand(1, 4)
        x_t = torch.randn(1, 3, 4)
        a = ddim_sample(trainer, mz, intensity, n_atoms=3, n_steps=5, x_t=x_t.clone(), eta=0.0)
        b = ddim_sample(trainer, mz, intensity, n_atoms=3, n_steps=5, x_t=x_t.clone(), eta=0.0)
        torch.testing.assert_close(a, b)

    def test_stochastic_at_eta_one(self):
        torch.manual_seed(0)
        trainer = _tiny_trainer()
        mz = torch.rand(1, 4) * 100
        intensity = torch.rand(1, 4)
        x_t = torch.randn(1, 3, 4)
        a = ddim_sample(trainer, mz, intensity, n_atoms=3, n_steps=5, x_t=x_t.clone(), eta=1.0)
        b = ddim_sample(trainer, mz, intensity, n_atoms=3, n_steps=5, x_t=x_t.clone(), eta=1.0)
        # With eta=1 the per-step noise differs run to run.
        assert not torch.allclose(a, b)

    def test_rejects_too_many_steps(self):
        trainer = _tiny_trainer()
        mz = torch.rand(1, 4) * 100
        intensity = torch.rand(1, 4)
        with pytest.raises(ValueError, match="n_timesteps"):
            ddim_sample(trainer, mz, intensity, n_atoms=3, n_steps=100)

    def test_rejects_bad_eta(self):
        trainer = _tiny_trainer()
        mz = torch.rand(1, 4) * 100
        intensity = torch.rand(1, 4)
        with pytest.raises(ValueError, match="eta"):
            ddim_sample(trainer, mz, intensity, n_atoms=3, n_steps=5, eta=2.0)


# ----------------------------------------------------------------------
# Permutation augmentation
# ----------------------------------------------------------------------


def _sample_batch() -> TrainingBatch:
    x0 = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
    atom_mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)
    atom_type_targets = torch.tensor([[0, 1, 2, PADDING_INDEX]], dtype=torch.long)
    return TrainingBatch(
        x_0=x0,
        mz=torch.rand(1, 5),
        intensity=torch.rand(1, 5),
        atom_mask=atom_mask,
        spectrum_mask=torch.ones(1, 5, dtype=torch.bool),
        atom_type_targets=atom_type_targets,
    )


class TestPermuteBatch:
    def test_preserves_mask_count(self):
        torch.manual_seed(0)
        batch = _sample_batch()
        permuted = permute_training_batch(batch)
        assert permuted.atom_mask.sum().item() == batch.atom_mask.sum().item()

    def test_padded_positions_unchanged(self):
        # Padded positions must stay padded and keep their sentinel.
        torch.manual_seed(0)
        batch = _sample_batch()
        permuted = permute_training_batch(batch)
        assert permuted.atom_mask[0, 3].item() is False
        assert permuted.atom_type_targets[0, 3].item() == PADDING_INDEX

    def test_preserves_atom_set(self):
        # The multiset of real rows must be identical before/after.
        torch.manual_seed(0)
        batch = _sample_batch()
        permuted = permute_training_batch(batch)
        valid = permuted.atom_mask[0]
        permuted_rows = {tuple(r.tolist()) for r in permuted.x_0[0, valid]}
        original_rows = {tuple(r.tolist()) for r in batch.x_0[0, batch.atom_mask[0]]}
        assert permuted_rows == original_rows

    def test_atom_type_multiset_preserved(self):
        torch.manual_seed(0)
        batch = _sample_batch()
        permuted = permute_training_batch(batch)
        valid = permuted.atom_mask[0]
        assert sorted(permuted.atom_type_targets[0, valid].tolist()) == [0, 1, 2]

    def test_without_atom_mask_raises(self):
        batch = _sample_batch()
        batch.atom_mask = None
        with pytest.raises(ValueError, match="atom_mask"):
            permute_training_batch(batch)

    def test_collator_wrapper(self):
        def base(samples):
            return _sample_batch()

        wrapped = wrap_collator_with_permutation(base)
        out = wrapped([{}])
        assert out.atom_mask.sum().item() == 3
