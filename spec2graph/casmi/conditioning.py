"""Condition the core diffusion model on adduct, collision energy and neutral mass without editing it.

Three condition tokens are prepended to the encoded peaks and passed through the core decoder's existing
``memory`` argument, so every core head is reused unchanged. One new head predicts the Laplacian spectral quantiles.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spec2graph.casmi.adducts import AdductVocabulary
from spec2graph.casmi.data import Batch, Conditioning, Spectra
from spec2graph.casmi.spectral import N_QUANTILES
from spec2graph.eval.ddim import _make_timestep_schedule
from spectral_diffusion import DiffusionTrainer, FourierMzEmbedding, Spec2GraphDiffusion, Spec2GraphDiffusionConfig, TrainerConfig

N_TOKENS = 3
RIDGE = 1e-4
FORMAT = "spec2graph-casmi-v1"


@dataclass
class CasmiModelConfig:
    n_adducts: int
    n_quantiles: int = N_QUANTILES
    collision_energy_max: float = 200.0
    mass_max: float = 2000.0


class CasmiModel(nn.Module):
    """Core model + adduct/polarity, collision-energy and neutral-mass tokens + a spectral-quantile head."""

    def __init__(self, core: Spec2GraphDiffusion, config: CasmiModelConfig):
        super().__init__()
        d = core.d_model
        self.core, self.config = core, config
        self.adduct, self.polarity = nn.Embedding(config.n_adducts, d), nn.Embedding(2, d)
        self.energy, self.mass = FourierMzEmbedding(d, max_mz=config.collision_energy_max), FourierMzEmbedding(d, max_mz=config.mass_max)
        self.energy_missing = nn.Parameter(torch.randn(d) * 0.02)
        self.token_type = nn.Parameter(torch.randn(N_TOKENS, d) * 0.02)
        self.norm = nn.LayerNorm(d)
        self.quantile_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, config.n_quantiles))

    def tokens(self, cond: Conditioning) -> torch.Tensor:
        adduct = self.adduct(cond.adduct_index) + self.polarity((cond.polarity > 0).long())
        energy = self.energy(cond.collision_energy.clamp(min=0).unsqueeze(-1)).squeeze(1)
        energy = torch.where(cond.collision_energy_mask.unsqueeze(-1), energy, self.energy_missing.expand_as(energy))
        mass = self.mass(cond.neutral_mass.clamp(min=0).unsqueeze(-1)).squeeze(1)
        return self.norm(torch.stack([adduct, energy, mass], 1) + self.token_type)

    def encode(self, mz, intensity, mask, precursor_mz, cond: Conditioning) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned memory ``(B, 3 + P, d)`` and its validity mask."""
        encoded = self.core.encode_spectrum(mz, intensity, mask, precursor_mz)
        if mask is None:
            mask = torch.ones(encoded.shape[:2], dtype=torch.bool, device=encoded.device)
        ones = torch.ones(encoded.shape[0], N_TOKENS, dtype=torch.bool, device=encoded.device)
        return torch.cat([self.tokens(cond), encoded], 1), torch.cat([ones, mask], 1)

    def denoise(self, x_t, t, atom_mask, memory, memory_mask, with_atom_types: bool = False):
        """Core decoder pass; it ignores mz/intensity once ``memory`` is given and takes the memory mask as ``spectrum_mask``."""
        fn = self.core.forward_with_atom_types if with_atom_types else self.core.forward
        return fn(x_t, t, None, None, atom_mask, memory_mask, None, memory=memory)

    def pooled(self, memory, memory_mask):
        return self.core._pool_spectrum(memory, memory_mask)

    def fingerprint_logits(self, memory, memory_mask):
        return self.core.fingerprint_head(self.pooled(memory, memory_mask))

    def atom_count(self, memory, memory_mask):
        return self.core.atom_count_head(self.pooled(memory, memory_mask)).squeeze(-1)

    def quantiles(self, memory, memory_mask):
        return self.quantile_head(self.pooled(memory, memory_mask))


@dataclass
class CasmiTrainerConfig(TrainerConfig):
    quantile_loss_weight: float = 1.0
    # Proxy mode: train the encoder and the heads only, skipping the denoiser (minutes instead of hours).
    heads_only: bool = False


class CasmiTrainer(DiffusionTrainer):
    """Core trainer with the conditioned memory. ``self.model`` is the core, ``self.wrapper`` the :class:`CasmiModel`."""

    def __init__(self, model: CasmiModel, config: Optional[CasmiTrainerConfig] = None, device: str | torch.device = "cpu"):
        self.config: CasmiTrainerConfig = config or CasmiTrainerConfig()
        super().__init__(model.core, self.config, device)
        self.wrapper = model.to(device)

    @staticmethod
    def projection_from_embeddings(embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Rank-safe ``V (VᵀV + εI)⁻¹ Vᵀ``: the core's QR projector at full rank, finite gradients when rank deficient."""
        v = embeddings if mask is None else embeddings * mask.unsqueeze(-1).to(embeddings.dtype)
        eye = torch.eye(v.shape[-1], device=v.device, dtype=v.dtype)
        proj = v @ torch.linalg.solve(v.transpose(-1, -2) @ v + RIDGE * eye, eye.expand(v.shape[0], -1, -1)) @ v.transpose(-1, -2)
        if mask is not None:
            proj = proj * (mask.unsqueeze(-1) & mask.unsqueeze(-2)).to(proj.dtype)
        return 0.5 * (proj + proj.transpose(-1, -2))

    def compute_loss(self, batch: Batch, return_components: bool = False):  # type: ignore[override]
        tb, cfg, model = batch.core, self.config, self.wrapper
        memory, mask = model.encode(tb.mz, tb.intensity, tb.spectrum_mask, tb.precursor_mz, batch.cond)
        parts = {} if cfg.heads_only else self._diffusion_losses(tb, memory, mask)
        if cfg.fingerprint_loss_weight and tb.fingerprint_targets is not None and self.model.fingerprint_head is not None:
            parts["fingerprint"] = F.binary_cross_entropy_with_logits(model.fingerprint_logits(memory, mask), tb.fingerprint_targets)
        if cfg.atom_count_loss_weight and tb.atom_count_targets is not None and self.model.atom_count_head is not None:
            parts["atom_count"] = F.mse_loss(model.atom_count(memory, mask), tb.atom_count_targets.float())
        if cfg.quantile_loss_weight:
            parts["quantile"] = F.mse_loss(model.quantiles(memory, mask), batch.quantiles)
        weights = dict(noise=1.0, projection=cfg.projection_loss_weight, orthonormality=cfg.orthonormality_loss_weight,
                       atom_type=cfg.atom_type_loss_weight, fingerprint=cfg.fingerprint_loss_weight,
                       atom_count=cfg.atom_count_loss_weight, quantile=cfg.quantile_loss_weight)
        loss = sum(weights[k] * v for k, v in parts.items())
        return (loss, {k: v.detach() for k, v in parts.items()}) if return_components else loss

    def _diffusion_losses(self, tb, memory, mask) -> dict[str, torch.Tensor]:
        cfg, x_0, atom_mask = self.config, tb.x_0, tb.atom_mask
        t = torch.randint(0, self.n_timesteps, (x_0.shape[0],), device=self.device)
        x_t, noise = self.q_sample(x_0, t)
        with_types = bool(cfg.atom_type_loss_weight) and tb.atom_type_targets is not None and self.model.atom_type_head is not None
        out = self.wrapper.denoise(x_t, t, atom_mask, memory, mask, with_atom_types=with_types)
        eps, logits = out if with_types else (out, None)
        parts = {"noise": self._masked_mse(eps, noise, None if atom_mask is None else atom_mask.unsqueeze(-1))}
        if cfg.projection_loss_weight or cfg.orthonormality_loss_weight:
            view = lambda s: s[t].view(-1, 1, 1)
            x_0_pred = self._reconstruct_x0(x_t, eps, view(self.sqrt_alpha_cumprod_clamped), view(self.sqrt_one_minus_alpha_cumprod_clamped))
            if cfg.projection_loss_weight:
                pair = None if atom_mask is None else atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
                parts["projection"] = self._masked_mse(self.projection_from_embeddings(x_0_pred, atom_mask), self.projection_from_embeddings(x_0, atom_mask), pair)
            if cfg.orthonormality_loss_weight:
                parts["orthonormality"] = self._orthonormality_loss(x_0_pred, atom_mask)
        if logits is not None:
            parts["atom_type"] = F.cross_entropy(logits.flatten(0, 1), tb.atom_type_targets.flatten(), ignore_index=-100)
        return parts

    def train_step(self, optimizer, batch: Batch, return_components: bool = False, grad_clip: Optional[float] = None):  # type: ignore[override]
        self.wrapper.train()
        optimizer.zero_grad()
        out = self.compute_loss(batch, return_components)
        loss = out[0] if return_components else out
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(), grad_clip)
        optimizer.step()
        return (loss.item(), {k: v.item() for k, v in out[1].items()}) if return_components else loss.item()

    @torch.no_grad()
    def heads(self, spectra: Spectra) -> dict[str, torch.Tensor]:
        """Conditioned memory plus per-spectrum fingerprint probabilities, sorted quantiles and atom counts."""
        self.wrapper.eval()
        memory, mask = self.wrapper.encode(spectra.mz, spectra.intensity, spectra.mask, spectra.precursor_mz, spectra.cond)
        out = dict(memory=memory, memory_mask=mask, quantiles=self.wrapper.quantiles(memory, mask).sort(-1).values.clamp(0, 2))
        if self.model.fingerprint_head is not None:
            out["fingerprint_prob"] = torch.sigmoid(self.wrapper.fingerprint_logits(memory, mask))
        if self.model.atom_count_head is not None:
            out["atom_count"] = self.wrapper.atom_count(memory, mask)
        return out

    @torch.no_grad()
    def sample(self, memory, memory_mask, n_atoms: int, atom_mask=None, sampler: str = "ddim", n_steps: int = 50, eta: float = 0.0):  # type: ignore[override]
        """Reverse diffusion from a conditioned memory: ``(B, n_atoms, k)`` eigenvectors (DDPM or DDIM)."""
        self.wrapper.eval()
        if not 1 <= n_atoms <= self.model.max_atoms:
            raise ValueError(f"n_atoms must be in [1, {self.model.max_atoms}]")
        x = torch.randn(memory.shape[0], n_atoms, self.model.k, device=self.device)
        if sampler == "ddpm":
            for t in reversed(range(self.n_timesteps)):
                x = self.p_sample(x, t, None, None, atom_mask, memory_mask, None, memory=memory)
            return x
        if sampler != "ddim":
            raise ValueError(f"unknown sampler {sampler!r}")
        schedule = _make_timestep_schedule(n_steps, self.n_timesteps)
        for t, t_next in zip(schedule, schedule[1:] + [-1]):
            a_t = self.alpha_cumprod[t]
            a_next = self.alpha_cumprod[t_next] if t_next >= 0 else torch.ones((), device=self.device)
            eps = self.wrapper.denoise(x, torch.full((x.shape[0],), t, device=self.device, dtype=torch.long), atom_mask, memory, memory_mask)
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-8)
            sigma = eta * ((1 - a_next) / (1 - a_t).clamp_min(1e-8) * (1 - a_t / a_next.clamp_min(1e-8))).sqrt() if eta and t_next >= 0 else 0.0
            x = a_next.sqrt() * x0 + (1 - a_next - sigma**2).clamp_min(0).sqrt() * eps
            if eta and t_next >= 0:
                x = x + sigma * torch.randn_like(x)
        return x

    @torch.no_grad()
    def draw(self, memory, memory_mask, n_atoms: int, n_draws: int, batch: int = 256, **sample_kw) -> Iterator[tuple]:
        """Yield ``(eigvecs, memory, mask)`` chunks, spreading ``n_draws`` round-robin over the spectra in ``memory``."""
        for start in range(0, n_draws, batch):
            ids = torch.arange(start, min(start + batch, n_draws), device=memory.device) % memory.shape[0]
            yield self.sample(memory[ids], memory_mask[ids], n_atoms, **sample_kw), memory[ids], memory_mask[ids]

    @torch.no_grad()
    def atom_type_logprobs(self, eigvecs, memory, memory_mask) -> torch.Tensor:
        """Element log-probabilities ``(B, N, E)`` for clean eigenvectors; uniform when the head is absent."""
        if self.model.atom_type_head is None:
            e = self.model.n_element_types
            return torch.full((*eigvecs.shape[:2], e), -math.log(e), device=eigvecs.device)
        t = torch.zeros(eigvecs.shape[0], dtype=torch.long, device=self.device)
        return F.log_softmax(self.wrapper.denoise(eigvecs, t, None, memory, memory_mask, with_atom_types=True)[1], -1)


def save_checkpoint(path, model: CasmiModel, trainer_config: CasmiTrainerConfig, vocab: AdductVocabulary, **extra) -> None:
    """Everything needed to rebuild the model with ``torch.load(weights_only=True)``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format=FORMAT, core_config=vars(model.core.config), casmi_config=asdict(model.config),
                    trainer_config=asdict(trainer_config), adduct_vocab=vocab.index, state_dict=model.state_dict(), extra=extra), path)


def load_checkpoint(path, device: str | torch.device = "cpu") -> tuple[CasmiModel, CasmiTrainerConfig, AdductVocabulary, dict]:
    p = torch.load(path, map_location=device, weights_only=True)
    if p.get("format") != FORMAT:
        raise ValueError("unrecognised checkpoint format")
    model = CasmiModel(Spec2GraphDiffusion(Spec2GraphDiffusionConfig(**p["core_config"])), CasmiModelConfig(**p["casmi_config"]))
    model.load_state_dict(p["state_dict"])
    vocab = AdductVocabulary()
    vocab.index = p["adduct_vocab"]
    return model.to(device).eval(), CasmiTrainerConfig(**p["trainer_config"]), vocab, p["extra"]
