"""Tests for :mod:`spec2graph.casmi.conditioning`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")
torch = pytest.importorskip("torch")

from spec2graph.casmi.adducts import AdductVocabulary
from spec2graph.casmi.conditioning import (
    N_CONDITION_TOKENS,
    CasmiModel,
    CasmiModelConfig,
    CasmiTrainer,
    CasmiTrainerConfig,
    Conditioning,
    load_checkpoint,
    save_checkpoint,
)
from spec2graph.casmi.data import CasmiSpectrumDataset, make_casmi_collator
from spectral_diffusion import Spec2GraphDiffusion, Spec2GraphDiffusionConfig

K, MAX_ATOMS, MAX_PEAKS, FP_BITS, N_Q = 4, 16, 32, 64, 8


def _tiny_model(n_adducts: int = 4) -> CasmiModel:
    torch.manual_seed(0)
    core = Spec2GraphDiffusion(
        Spec2GraphDiffusionConfig(
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            k=K,
            max_atoms=MAX_ATOMS,
            max_peaks=MAX_PEAKS,
            dropout=0.0,
            fingerprint_dim=FP_BITS,
            enable_atom_count_head=True,
            enable_atom_type_head=True,
        )
    )
    return CasmiModel(core, CasmiModelConfig(n_adducts=n_adducts, n_quantiles=N_Q))


@pytest.fixture
def batch(casmi_frame, tmp_path):
    vocab = AdductVocabulary.fit(casmi_frame["adduct"])
    ds = CasmiSpectrumDataset(
        casmi_frame, cache=tmp_path / "cache", adduct_vocab=vocab, k=K, max_atoms=MAX_ATOMS,
        fingerprint_bits=FP_BITS, n_quantiles=N_Q,
    )
    collate = make_casmi_collator(max_atoms=MAX_ATOMS, max_peaks=MAX_PEAKS, k=K, fingerprint_bits=FP_BITS, n_quantiles=N_Q)
    return collate([ds[i] for i in range(4)]), vocab


class TestConditioning:
    def test_build_masks_missing_collision_energy(self):
        cond = Conditioning.build([1, 0], [20.0, None], [180.0, 94.0], [1.0, -1.0])
        assert cond.collision_energy_mask.tolist() == [True, False]
        assert cond.collision_energy.tolist() == [20.0, 0.0]
        tiled = cond.repeat_interleave(3)
        assert tiled.batch_size == 6
        assert tiled.adduct_index.tolist() == [1, 1, 1, 0, 0, 0]
        assert cond.select([1]).neutral_mass.tolist() == [94.0]
        with pytest.raises(ValueError):
            Conditioning(torch.zeros(2, dtype=torch.long), torch.zeros(3), torch.zeros(2, dtype=torch.bool),
                         torch.zeros(2), torch.zeros(2)).validate()


class TestModel:
    def test_encode_shapes(self, batch):
        casmi_batch, _ = batch
        model = _tiny_model()
        tb = casmi_batch.batch
        memory, mask = model.encode(tb.mz, tb.intensity, tb.spectrum_mask, tb.precursor_mz, Conditioning.from_batch(casmi_batch))
        assert memory.shape == (4, N_CONDITION_TOKENS + tb.mz.shape[1], 16)
        assert mask.shape == (4, N_CONDITION_TOKENS + tb.mz.shape[1])
        assert mask[:, :N_CONDITION_TOKENS].all()
        assert torch.equal(mask[:, N_CONDITION_TOKENS:], tb.spectrum_mask)

    def test_missing_collision_energy_uses_learned_vector(self):
        model = _tiny_model()
        known = Conditioning.build([1], [20.0], [180.0], [1.0])
        missing = Conditioning.build([1], [None], [180.0], [1.0])
        t_known = model.condition_tokens(known)
        t_missing = model.condition_tokens(missing)
        # Adduct and mass tokens agree; the collision-energy token differs.
        assert torch.allclose(t_known[:, 0], t_missing[:, 0])
        assert torch.allclose(t_known[:, 2], t_missing[:, 2])
        assert not torch.allclose(t_known[:, 1], t_missing[:, 1])

    def test_conditioning_changes_output(self, batch):
        casmi_batch, _ = batch
        model = _tiny_model().eval()
        tb = casmi_batch.batch
        cond = Conditioning.from_batch(casmi_batch)
        other = Conditioning(cond.adduct_index, cond.collision_energy, cond.collision_energy_mask, cond.neutral_mass * 2, cond.polarity)
        trainer = CasmiTrainer(model, CasmiTrainerConfig(n_timesteps=10))
        mem_a, mask_a = trainer.encode(tb.mz, tb.intensity, tb.spectrum_mask, tb.precursor_mz, cond)
        mem_b, mask_b = trainer.encode(tb.mz, tb.intensity, tb.spectrum_mask, tb.precursor_mz, other)
        torch.manual_seed(1)
        a = trainer.sample(mem_a, mask_a, n_atoms=6, sampler="ddim", n_steps=3)
        torch.manual_seed(1)
        b = trainer.sample(mem_b, mask_b, n_atoms=6, sampler="ddim", n_steps=3)
        assert a.shape == (4, 6, K)
        assert not torch.allclose(a, b)


class TestTrainer:
    def test_rank_safe_projector(self):
        from spectral_diffusion import DiffusionTrainer

        torch.manual_seed(0)
        full = torch.randn(2, 6, 4, requires_grad=True)
        mask = torch.ones(2, 6, dtype=torch.bool)
        ours = CasmiTrainer.projection_from_embeddings(full, mask)
        core = DiffusionTrainer.projection_from_embeddings(full.detach(), mask)
        assert torch.allclose(ours, core, atol=1e-3)
        deficient = torch.randn(1, 5, 4, requires_grad=True)
        mask = torch.tensor([[True, True, True, False, False]])
        CasmiTrainer.projection_from_embeddings(deficient, mask).sum().backward()
        assert torch.isfinite(deficient.grad).all()

    def test_loss_components_and_descent(self, batch):
        casmi_batch, _ = batch
        model = _tiny_model()
        trainer = CasmiTrainer(
            model,
            CasmiTrainerConfig(
                n_timesteps=10,
                fingerprint_loss_weight=0.1,
                atom_count_loss_weight=0.05,
                atom_type_loss_weight=0.5,
                quantile_loss_weight=1.0,
                orthonormality_loss_weight=0.1,
            ),
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
        first, components = trainer.train_step(optimiser, casmi_batch, return_components=True)
        assert set(components) == {"noise", "projection", "orthonormality", "fingerprint", "atom_count", "quantile", "atom_type"}
        assert all(np.isfinite(v) for v in components.values())
        assert components["quantile"] > 0 and components["fingerprint"] > 0
        torch.manual_seed(0)
        losses = [trainer.train_step(optimiser, casmi_batch, grad_clip=1.0) for _ in range(25)]
        assert np.mean(losses[-5:]) < first

    def test_samplers_and_heads(self, batch):
        casmi_batch, _ = batch
        model = _tiny_model()
        trainer = CasmiTrainer(model, CasmiTrainerConfig(n_timesteps=10))
        tb = casmi_batch.batch
        cond = Conditioning.from_batch(casmi_batch)
        heads = trainer.predict_heads(tb.mz, tb.intensity, tb.spectrum_mask, tb.precursor_mz, cond)
        assert heads["fingerprint_prob"].shape == (4, FP_BITS)
        assert torch.all((heads["fingerprint_prob"] >= 0) & (heads["fingerprint_prob"] <= 1))
        assert heads["quantiles"].shape == (4, N_Q)
        assert torch.all(torch.diff(heads["quantiles"], dim=-1) >= 0)
        assert heads["atom_count"].shape == (4,)
        memory, mask = heads["memory"], heads["memory_mask"]
        tiled_mem, tiled_mask = memory.repeat_interleave(2, dim=0), mask.repeat_interleave(2, dim=0)
        atom_mask = torch.ones(8, 5, dtype=torch.bool)
        ddpm = trainer.sample(tiled_mem, tiled_mask, n_atoms=5, atom_mask=atom_mask, sampler="ddpm")
        ddim = trainer.sample(tiled_mem, tiled_mask, n_atoms=5, atom_mask=atom_mask, sampler="ddim", n_steps=4, eta=0.5)
        assert ddpm.shape == ddim.shape == (8, 5, K)
        assert torch.isfinite(ddpm).all() and torch.isfinite(ddim).all()
        logits = trainer.atom_type_logits(ddim, atom_mask, tiled_mem, tiled_mask)
        assert logits.shape == (8, 5, 13)
        with pytest.raises(ValueError):
            trainer.sample(memory, mask, n_atoms=MAX_ATOMS + 1)
        with pytest.raises(ValueError):
            trainer.sample(memory, mask, n_atoms=4, sampler="nope")

    def test_checkpoint_round_trip(self, batch, tmp_path):
        casmi_batch, vocab = batch
        model = _tiny_model(n_adducts=len(vocab))
        config = CasmiTrainerConfig(n_timesteps=10, quantile_loss_weight=0.5)
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, config, vocab, epoch=3, extra={"note": "unit"})
        bundle = load_checkpoint(path)
        assert bundle.epoch == 3 and bundle.extra == {"note": "unit"}
        assert bundle.trainer_config == config
        assert bundle.adduct_vocab.to_dict() == vocab.to_dict()
        for key, value in model.state_dict().items():
            assert torch.equal(bundle.model.state_dict()[key], value), key
        assert bundle.model.core.config == model.core.config
