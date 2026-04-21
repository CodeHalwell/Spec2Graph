"""Tests for the per-atom element (atom-type) head.

Covers:
  * Element vocabulary constants and helpers.
  * Model construction with and without the head (backward compatibility).
  * Shape and alignment of ``forward`` / ``forward_with_atom_types`` /
    ``predict_atom_types`` outputs.
  * ``compute_loss`` integration: loss wiring, ``ignore_index`` handling,
    component reporting, and the gate when the head is disabled.
  * A gradient flow check so the head actually trains.
"""

from __future__ import annotations

import pytest
import torch

from spectral_diffusion import (
    DiffusionTrainer,
    Spec2GraphDiffusion,
    Spec2GraphDiffusionConfig,
    TrainerConfig,
    TrainingBatch,
)
from spec2graph.data.elements import (
    ELEMENTS,
    N_ELEMENT_TYPES,
    PADDING_INDEX,
    UNKNOWN_ELEMENT,
    UNKNOWN_INDEX,
    atom_types_to_indices,
    element_to_index,
    index_to_element,
)


# ----------------------------------------------------------------------
# Element vocabulary
# ----------------------------------------------------------------------


class TestElementVocab:
    def test_expected_vocab_size(self):
        assert N_ELEMENT_TYPES == 13
        assert len(ELEMENTS) == 12
        assert UNKNOWN_INDEX == 12

    def test_element_to_index_roundtrip(self):
        for element in ELEMENTS:
            assert index_to_element(element_to_index(element)) == element

    def test_unknown_element_maps_to_unknown_index(self):
        assert element_to_index("Xe") == UNKNOWN_INDEX
        assert index_to_element(UNKNOWN_INDEX) == UNKNOWN_ELEMENT

    def test_padding_index_is_pytorch_default(self):
        assert PADDING_INDEX == -100

    def test_atom_types_to_indices(self):
        assert atom_types_to_indices(["C", "N", "O"]) == [0, 1, 2]

    def test_atom_types_to_indices_handles_unknown(self):
        result = atom_types_to_indices(["C", "Xe", "O"])
        assert result == [0, UNKNOWN_INDEX, 2]


# ----------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------


def _tiny_config(**overrides) -> Spec2GraphDiffusionConfig:
    base = {
        "d_model": 16,
        "nhead": 2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 32,
        "k": 4,
        "max_atoms": 8,
        "max_peaks": 12,
        "dropout": 0.0,
    }
    base.update(overrides)
    return Spec2GraphDiffusionConfig(**base)


def _tiny_inputs(batch=2, n_atoms=5, n_peaks=8, k=4, device="cpu"):
    x_t = torch.randn(batch, n_atoms, k, device=device)
    t = torch.zeros(batch, dtype=torch.long, device=device)
    mz = torch.rand(batch, n_peaks, device=device) * 300
    intensity = torch.rand(batch, n_peaks, device=device)
    atom_mask = torch.ones(batch, n_atoms, dtype=torch.bool, device=device)
    spectrum_mask = torch.ones(batch, n_peaks, dtype=torch.bool, device=device)
    return x_t, t, mz, intensity, atom_mask, spectrum_mask


class TestModelConstruction:
    def test_default_model_has_no_atom_type_head(self):
        # Defaults must be backward compatible: existing code that does
        # not set enable_atom_type_head should still work.
        model = Spec2GraphDiffusion(_tiny_config())
        assert model.atom_type_head is None
        assert model.n_element_types == N_ELEMENT_TYPES

    def test_enabled_model_has_atom_type_head(self):
        model = Spec2GraphDiffusion(_tiny_config(enable_atom_type_head=True))
        assert model.atom_type_head is not None

    def test_custom_n_element_types(self):
        config = _tiny_config(enable_atom_type_head=True, n_element_types=20)
        model = Spec2GraphDiffusion(config)
        # The head's final Linear should match the configured output dim.
        assert model.atom_type_head[-1].out_features == 20


# ----------------------------------------------------------------------
# Forward-pass behaviour
# ----------------------------------------------------------------------


class TestForwardBehaviour:
    def test_forward_returns_only_noise_unchanged(self):
        # Every existing caller treats forward() as a single-tensor
        # returner; this must stay true regardless of head state.
        model = Spec2GraphDiffusion(_tiny_config(enable_atom_type_head=True))
        model.eval()
        inputs = _tiny_inputs()
        with torch.no_grad():
            out = model(*inputs[:4], atom_mask=inputs[4], spectrum_mask=inputs[5])
        assert out.shape == (2, 5, 4)
        assert isinstance(out, torch.Tensor)

    def test_forward_with_atom_types_returns_tuple(self):
        model = Spec2GraphDiffusion(_tiny_config(enable_atom_type_head=True))
        model.eval()
        x_t, t, mz, intensity, atom_mask, spectrum_mask = _tiny_inputs()
        with torch.no_grad():
            noise, logits = model.forward_with_atom_types(
                x_t, t, mz, intensity, atom_mask, spectrum_mask
            )
        assert noise.shape == (2, 5, 4)
        assert logits.shape == (2, 5, N_ELEMENT_TYPES)

    def test_forward_with_atom_types_raises_when_head_disabled(self):
        model = Spec2GraphDiffusion(_tiny_config())  # head disabled
        x_t, t, mz, intensity, atom_mask, spectrum_mask = _tiny_inputs()
        with pytest.raises(ValueError, match="Atom-type head is disabled"):
            model.forward_with_atom_types(
                x_t, t, mz, intensity, atom_mask, spectrum_mask
            )

    def test_predict_atom_types_shape(self):
        model = Spec2GraphDiffusion(_tiny_config(enable_atom_type_head=True))
        model.eval()
        x_t, t, mz, intensity, atom_mask, spectrum_mask = _tiny_inputs()
        with torch.no_grad():
            logits = model.predict_atom_types(
                x_t, t, mz, intensity, atom_mask, spectrum_mask
            )
        assert logits.shape == (2, 5, N_ELEMENT_TYPES)

    def test_noise_output_identical_with_and_without_head_path(self):
        # forward() and forward_with_atom_types() must share the decoder
        # pass and produce identical noise predictions for the same input.
        torch.manual_seed(0)
        model = Spec2GraphDiffusion(_tiny_config(enable_atom_type_head=True))
        model.eval()
        x_t, t, mz, intensity, atom_mask, spectrum_mask = _tiny_inputs()
        with torch.no_grad():
            noise_plain = model(x_t, t, mz, intensity, atom_mask, spectrum_mask)
            noise_combined, _ = model.forward_with_atom_types(
                x_t, t, mz, intensity, atom_mask, spectrum_mask
            )
        torch.testing.assert_close(noise_plain, noise_combined)


# ----------------------------------------------------------------------
# Training loss integration
# ----------------------------------------------------------------------


def _training_inputs(batch=2, n_atoms=5, n_peaks=8, k=4):
    x_0 = torch.randn(batch, n_atoms, k)
    mz = torch.rand(batch, n_peaks) * 300
    intensity = torch.rand(batch, n_peaks)
    atom_mask = torch.ones(batch, n_atoms, dtype=torch.bool)
    spectrum_mask = torch.ones(batch, n_peaks, dtype=torch.bool)
    # Random but in-vocab element indices.
    atom_type_targets = torch.randint(0, N_ELEMENT_TYPES, (batch, n_atoms))
    return TrainingBatch(
        x_0=x_0,
        mz=mz,
        intensity=intensity,
        atom_mask=atom_mask,
        spectrum_mask=spectrum_mask,
        atom_type_targets=atom_type_targets,
    )


class TestTrainingIntegration:
    def test_loss_includes_atom_type_when_enabled(self):
        model = Spec2GraphDiffusion(_tiny_config(enable_atom_type_head=True))
        trainer = DiffusionTrainer(
            model=model,
            config=TrainerConfig(n_timesteps=10, atom_type_loss_weight=1.0),
        )
        torch.manual_seed(0)
        batch = _training_inputs()
        _, components = trainer.compute_loss(batch, return_components=True)
        assert "atom_type" in components
        assert components["atom_type"].item() > 0.0

    def test_loss_skipped_when_weight_zero(self):
        model = Spec2GraphDiffusion(_tiny_config(enable_atom_type_head=True))
        trainer = DiffusionTrainer(
            model=model,
            config=TrainerConfig(n_timesteps=10, atom_type_loss_weight=0.0),
        )
        batch = _training_inputs()
        _, components = trainer.compute_loss(batch, return_components=True)
        assert components["atom_type"].item() == 0.0

    def test_loss_skipped_when_targets_missing(self):
        model = Spec2GraphDiffusion(_tiny_config(enable_atom_type_head=True))
        trainer = DiffusionTrainer(
            model=model,
            config=TrainerConfig(n_timesteps=10, atom_type_loss_weight=1.0),
        )
        batch = _training_inputs()
        batch.atom_type_targets = None  # user did not provide labels
        _, components = trainer.compute_loss(batch, return_components=True)
        assert components["atom_type"].item() == 0.0

    def test_loss_skipped_when_head_disabled(self):
        # weight > 0 + targets present but head was never built.
        model = Spec2GraphDiffusion(_tiny_config())  # default: head off
        trainer = DiffusionTrainer(
            model=model,
            config=TrainerConfig(n_timesteps=10, atom_type_loss_weight=1.0),
        )
        batch = _training_inputs()
        _, components = trainer.compute_loss(batch, return_components=True)
        # The gate also checks model.atom_type_head; it should silently skip.
        assert components["atom_type"].item() == 0.0

    def test_loss_ignores_padded_atoms(self):
        # Using the PyTorch default ignore_index (-100) means padded atoms
        # contribute zero to the loss, which is what we want.
        model = Spec2GraphDiffusion(_tiny_config(enable_atom_type_head=True))
        trainer = DiffusionTrainer(
            model=model,
            config=TrainerConfig(n_timesteps=10, atom_type_loss_weight=1.0),
        )
        torch.manual_seed(0)
        batch = _training_inputs(n_atoms=5)
        # Real atoms for positions 0-2, padding for 3-4.
        batch.atom_mask = torch.tensor(
            [[True, True, True, False, False]] * 2, dtype=torch.bool
        )
        batch.atom_type_targets = torch.tensor(
            [[0, 1, 2, PADDING_INDEX, PADDING_INDEX]] * 2, dtype=torch.long
        )
        _, components = trainer.compute_loss(batch, return_components=True)
        # Just check the loss is finite and positive (cross-entropy over 3
        # random predictions). We don't compare an exact value because the
        # decoder is randomly initialised.
        loss_value = components["atom_type"].item()
        assert loss_value > 0.0
        assert torch.isfinite(torch.tensor(loss_value))

    def test_gradients_flow_into_head(self):
        # The head must actually receive gradients when the loss is on.
        model = Spec2GraphDiffusion(_tiny_config(enable_atom_type_head=True))
        trainer = DiffusionTrainer(
            model=model,
            config=TrainerConfig(n_timesteps=10, atom_type_loss_weight=1.0),
        )
        batch = _training_inputs()
        loss = trainer.compute_loss(batch)
        loss.backward()
        last_linear = model.atom_type_head[-1]
        assert last_linear.weight.grad is not None
        assert last_linear.weight.grad.abs().sum().item() > 0

    def test_components_dict_has_atom_type_key_always(self):
        # Even when the head is disabled, the components dict must have
        # the atom_type key so downstream logging never breaks.
        model = Spec2GraphDiffusion(_tiny_config())
        trainer = DiffusionTrainer(model=model, config=TrainerConfig(n_timesteps=10))
        batch = _training_inputs()
        _, components = trainer.compute_loss(batch, return_components=True)
        assert set(components.keys()) == {
            "noise",
            "projection",
            "orthonormality",
            "fingerprint",
            "atom_count",
            "eigenvalue",
            "atom_type",
        }
