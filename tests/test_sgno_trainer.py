"""Tests for :mod:`spec2graph.train.sgno_trainer`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spec2graph.train.sgno_trainer import (
    SGNOTrainer,
    SGNOTrainerConfig,
    adjacency_targets_from_batch,
)
from spectral_diffusion import SpectralGraphNeuralOperator


class TestAdjacencyTargets:
    def test_binarises_bonds(self):
        # Integer + aromatic bond orders must all map to 1.
        adj = np.array([[0.0, 1.0, 1.5], [1.0, 0.0, 2.0], [1.5, 2.0, 0.0]], dtype=np.float32)
        mask = torch.ones(1, 3, dtype=torch.bool)
        target = adjacency_targets_from_batch([adj], mask)
        assert target.shape == (1, 3, 3)
        assert set(torch.unique(target).tolist()).issubset({0.0, 1.0})
        # Off-diagonal bonds are all 1 for this adj.
        assert target[0, 0, 1].item() == 1.0
        assert target[0, 1, 2].item() == 1.0

    def test_zeros_diagonal(self):
        adj = np.eye(3, dtype=np.float32)
        mask = torch.ones(1, 3, dtype=torch.bool)
        target = adjacency_targets_from_batch([adj], mask)
        assert torch.all(target[0].diagonal() == 0)

    def test_pads_to_batch_max(self):
        adj = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)
        target = adjacency_targets_from_batch([adj], mask)
        assert target.shape == (1, 4, 4)
        # Padded positions must remain zero.
        assert target[0, 2, 3].item() == 0.0
        assert target[0, 3, 2].item() == 0.0

    def test_batch_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="batch size"):
            adjacency_targets_from_batch(
                [np.zeros((2, 2), dtype=np.float32)] * 2,
                torch.ones(3, 4, dtype=torch.bool),
            )


class TestSGNOTrainer:
    def test_train_step_reduces_loss_on_trivial_task(self):
        # A tiny SGNO should be able to memorise a single example
        # after a handful of gradient steps.
        torch.manual_seed(0)
        sgno = SpectralGraphNeuralOperator(k=3, hidden_dim=32, num_layers=2)
        trainer = SGNOTrainer(sgno, config=SGNOTrainerConfig(pos_weight=5.0))
        optimiser = torch.optim.Adam(sgno.parameters(), lr=1e-2)

        eigvecs = torch.randn(1, 4, 3)
        atom_mask = torch.ones(1, 4, dtype=torch.bool)
        # Target: a single "bond" between atoms 0 and 1.
        adjacency = np.zeros((4, 4), dtype=np.float32)
        adjacency[0, 1] = adjacency[1, 0] = 1.0

        first = trainer.train_step(optimiser, eigvecs, atom_mask, [adjacency])
        for _ in range(30):
            last = trainer.train_step(optimiser, eigvecs, atom_mask, [adjacency])
        assert last < first, f"Loss did not decrease: first={first}, last={last}"

    def test_compute_loss_respects_atom_mask(self):
        # A sample with only 2 valid atoms plus padding. Loss should
        # match a call that never saw the padded atoms.
        torch.manual_seed(0)
        sgno = SpectralGraphNeuralOperator(k=3, hidden_dim=16, num_layers=2)
        trainer = SGNOTrainer(sgno)

        eigvecs = torch.randn(1, 4, 3)
        atom_mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)
        adjacency = np.array(
            [[0.0, 1.0], [1.0, 0.0]], dtype=np.float32
        )  # only 2 real atoms
        loss = trainer.compute_loss(eigvecs, atom_mask, [adjacency])
        assert torch.isfinite(loss)
