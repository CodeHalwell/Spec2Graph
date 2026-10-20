"""Competition conditioning for the core diffusion model, without editing it.

The competition supplies, per spectrum, an adduct, a collision energy (often
missing) and a precursor m/z from which the neutral mass follows. The core
:class:`~spectral_diffusion.Spec2GraphDiffusion` only knows about peaks and
an optional precursor m/z, so :class:`CasmiModel` wraps it:

1. the core encodes the peaks as usual (``encode_spectrum``);
2. three *condition tokens* are built, for adduct+polarity, collision energy
   and neutral mass, and prepended to the encoded peaks;
3. the concatenation is passed to the core decoder through its existing
   ``memory`` argument, with the key-padding mask extended to match.

Every core head (fingerprint, atom count, atom types) is reused unchanged
through the same ``memory``. One new head predicts the Laplacian spectral
quantiles (:mod:`spec2graph.casmi.spectral`), the permutation-invariant
descriptor the scorer compares against candidates.

:class:`CasmiTrainer` subclasses the core :class:`DiffusionTrainer` so the
DDPM schedule, ``q_sample``, ``p_sample`` and the projection/orthonormality
helpers are shared. Only the loss assembly and the samplers are redefined,
because they need the conditioned memory.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spec2graph.casmi.adducts import AdductVocabulary
from spec2graph.casmi.data import CasmiBatch
from spec2graph.casmi.spectral import DEFAULT_N_QUANTILES
from spec2graph.eval.ddim import _make_timestep_schedule
from spectral_diffusion import (
    DiffusionTrainer,
    FourierMzEmbedding,
    Spec2GraphDiffusion,
    Spec2GraphDiffusionConfig,
    TrainerConfig,
)

logger = logging.getLogger(__name__)

N_CONDITION_TOKENS: int = 3
# Ridge added to V^T V in the rank-safe projector. Negligible for orthonormal
# columns (scales the projector by 1/(1+ridge)); keeps gradients finite when a
# molecule has fewer heavy atoms than k and the eigenvector matrix is rank
# deficient, where the QR-based projector's backward pass returns NaN.
PROJECTION_RIDGE: float = 1e-4


@dataclass
class CasmiModelConfig:
    """Hyper-parameters of the conditioning wrapper."""

    n_adducts: int
    n_quantiles: int = DEFAULT_N_QUANTILES
    # Fourier scale for collision energy (eV) and neutral mass (Da).
    collision_energy_max: float = 200.0
    mass_max: float = 2000.0
    enable_quantile_head: bool = True


@dataclass
class Conditioning:
    """Per-spectrum conditioning tensors, all shaped ``(batch,)``."""

    adduct_index: torch.Tensor
    collision_energy: torch.Tensor
    collision_energy_mask: torch.Tensor
    neutral_mass: torch.Tensor
    polarity: torch.Tensor

    @classmethod
    def from_batch(cls, batch: CasmiBatch) -> "Conditioning":
        return cls(
            adduct_index=batch.adduct_index,
            collision_energy=batch.collision_energy,
            collision_energy_mask=batch.collision_energy_mask,
            neutral_mass=batch.neutral_mass,
            polarity=batch.polarity,
        )

    @classmethod
    def build(
        cls,
        adduct_index: list[int] | torch.Tensor,
        collision_energy: list[Optional[float]] | torch.Tensor,
        neutral_mass: list[float] | torch.Tensor,
        polarity: list[float] | torch.Tensor,
        device: str | torch.device = "cpu",
    ) -> "Conditioning":
        """Build from Python lists; ``None``/NaN collision energies are masked."""
        adduct = torch.as_tensor(adduct_index, dtype=torch.long, device=device)
        ce_values = [float("nan") if v is None else float(v) for v in collision_energy] if not isinstance(
            collision_energy, torch.Tensor
        ) else collision_energy
        ce = torch.as_tensor(ce_values, dtype=torch.float32, device=device)
        mask = torch.isfinite(ce)
        ce = torch.where(mask, ce, torch.zeros_like(ce))
        return cls(
            adduct_index=adduct,
            collision_energy=ce,
            collision_energy_mask=mask,
            neutral_mass=torch.as_tensor(neutral_mass, dtype=torch.float32, device=device),
            polarity=torch.as_tensor(polarity, dtype=torch.float32, device=device),
        )

    @property
    def batch_size(self) -> int:
        return int(self.adduct_index.shape[0])

    def to(self, device: str | torch.device) -> "Conditioning":
        return Conditioning(*(t.to(device) for t in self._tensors()))

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.adduct_index,
            self.collision_energy,
            self.collision_energy_mask,
            self.neutral_mass,
            self.polarity,
        )

    def repeat_interleave(self, repeats: int) -> "Conditioning":
        """Tile every row ``repeats`` times (for several samples per spectrum)."""
        return Conditioning(*(t.repeat_interleave(repeats, dim=0) for t in self._tensors()))

    def select(self, indices: torch.Tensor | list[int]) -> "Conditioning":
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.adduct_index.device)
        return Conditioning(*(t.index_select(0, idx) for t in self._tensors()))

    def validate(self) -> None:
        n = self.batch_size
        for name, tensor in zip(
            ("adduct_index", "collision_energy", "collision_energy_mask", "neutral_mass", "polarity"),
            self._tensors(),
        ):
            if tensor.dim() != 1 or tensor.shape[0] != n:
                raise ValueError(f"Conditioning.{name} must have shape ({n},); got {tuple(tensor.shape)}.")
        if self.collision_energy_mask.dtype != torch.bool:
            raise ValueError("collision_energy_mask must be boolean.")


class CasmiModel(nn.Module):
    """Core diffusion model plus competition conditioning and a quantile head."""

    def __init__(self, core: Spec2GraphDiffusion, config: CasmiModelConfig) -> None:
        super().__init__()
        if config.n_adducts < 1:
            raise ValueError("n_adducts must be >= 1 (index 0 is the unknown adduct).")
        self.core = core
        self.config = config
        d_model = core.d_model

        self.adduct_embedding = nn.Embedding(config.n_adducts, d_model)
        self.polarity_embedding = nn.Embedding(2, d_model)
        self.collision_energy_embedding = FourierMzEmbedding(d_model, max_mz=config.collision_energy_max)
        self.collision_energy_missing = nn.Parameter(torch.zeros(d_model))
        self.mass_embedding = FourierMzEmbedding(d_model, max_mz=config.mass_max)
        self.token_type = nn.Parameter(torch.zeros(N_CONDITION_TOKENS, d_model))
        self.condition_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.token_type, std=0.02)
        nn.init.normal_(self.collision_energy_missing, std=0.02)

        if config.enable_quantile_head:
            self.quantile_head: Optional[nn.Module] = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, config.n_quantiles),
            )
        else:
            self.quantile_head = None

    # ------------------------------------------------------------------
    # Convenience accessors mirroring the core
    # ------------------------------------------------------------------

    @property
    def k(self) -> int:
        return self.core.k

    @property
    def max_atoms(self) -> int:
        return self.core.max_atoms

    @property
    def d_model(self) -> int:
        return self.core.d_model

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    def condition_tokens(self, cond: Conditioning) -> torch.Tensor:
        """``(batch, N_CONDITION_TOKENS, d_model)`` tokens for the memory."""
        cond.validate()
        polarity_index = (cond.polarity > 0).long()
        adduct = self.adduct_embedding(cond.adduct_index) + self.polarity_embedding(polarity_index)

        ce_input = cond.collision_energy.clamp(min=0.0).unsqueeze(-1)  # (batch, 1)
        ce = self.collision_energy_embedding(ce_input).squeeze(1)
        missing = self.collision_energy_missing.unsqueeze(0).expand_as(ce)
        ce = torch.where(cond.collision_energy_mask.unsqueeze(-1), ce, missing)

        mass = self.mass_embedding(cond.neutral_mass.clamp(min=0.0).unsqueeze(-1)).squeeze(1)

        tokens = torch.stack([adduct, ce, mass], dim=1) + self.token_type.unsqueeze(0)
        return self.condition_norm(tokens)

    def encode(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        spectrum_mask: Optional[torch.Tensor],
        precursor_mz: Optional[torch.Tensor],
        cond: Conditioning,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned memory ``(batch, 3 + n_peaks, d)`` and its validity mask."""
        encoded = self.core.encode_spectrum(mz, intensity, spectrum_mask, precursor_mz)
        tokens = self.condition_tokens(cond)
        if tokens.shape[0] != encoded.shape[0]:
            raise ValueError(
                f"Conditioning batch ({tokens.shape[0]}) does not match spectra batch ({encoded.shape[0]})."
            )
        memory = torch.cat([tokens, encoded], dim=1)
        if spectrum_mask is None:
            spectrum_mask = torch.ones(encoded.shape[:2], dtype=torch.bool, device=encoded.device)
        token_mask = torch.ones(
            (encoded.shape[0], N_CONDITION_TOKENS), dtype=torch.bool, device=encoded.device
        )
        return memory, torch.cat([token_mask, spectrum_mask], dim=1)

    # ------------------------------------------------------------------
    # Decoder passes through the core, with the conditioned memory
    # ------------------------------------------------------------------

    def denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        atom_mask: Optional[torch.Tensor],
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        # The core ignores mz/intensity once ``memory`` is given and uses the
        # ``spectrum_mask`` argument as the memory key-padding mask.
        return self.core(x_t, t, None, None, atom_mask, memory_mask, None, memory=memory)

    def denoise_with_atom_types(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        atom_mask: Optional[torch.Tensor],
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.forward_with_atom_types(
            x_t, t, None, None, atom_mask, memory_mask, None, memory=memory
        )

    def pooled(self, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        return self.core._pool_spectrum(memory, memory_mask)

    def predict_fingerprint_logits(self, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        if self.core.fingerprint_head is None:
            raise ValueError("The core model has no fingerprint head (fingerprint_dim=0).")
        return self.core.fingerprint_head(self.pooled(memory, memory_mask))

    def predict_atom_count(self, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        if self.core.atom_count_head is None:
            raise ValueError("The core model has no atom-count head.")
        return self.core.atom_count_head(self.pooled(memory, memory_mask)).squeeze(-1)

    def predict_quantiles(self, memory: torch.Tensor, memory_mask: torch.Tensor, *, sort: bool = False) -> torch.Tensor:
        if self.quantile_head is None:
            raise ValueError("Quantile head disabled; set enable_quantile_head=True.")
        out = self.quantile_head(self.pooled(memory, memory_mask))
        return torch.sort(out, dim=-1).values if sort else out


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------


@dataclass
class CasmiTrainerConfig(TrainerConfig):
    quantile_loss_weight: float = 1.0


class CasmiTrainer(DiffusionTrainer):
    """:class:`DiffusionTrainer` whose loss and samplers use the conditioned memory.

    ``self.model`` is the *core* model (so every inherited helper keeps
    working); ``self.casmi_model`` is the wrapper. Build the optimiser over
    ``casmi_model.parameters()`` so the conditioning layers train too.
    """

    def __init__(
        self,
        model: CasmiModel,
        config: Optional[CasmiTrainerConfig] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        config = config or CasmiTrainerConfig()
        super().__init__(model=model.core, config=config, device=device)
        self.casmi_model = model
        self.quantile_loss_weight = float(getattr(config, "quantile_loss_weight", 0.0))

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def projection_from_embeddings(
        embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Rank-safe subspace projector ``P = V (VᵀV + εI)⁻¹ Vᵀ``.

        Identical (up to ``1/(1+ε)``) to the core's QR projector when ``V``
        has full column rank, but differentiable when it does not: the core's
        ``torch.linalg.qr`` backward returns NaN for rank-deficient input,
        which happens for every molecule with fewer heavy atoms than ``k``.
        Masked atoms are zeroed before and after, as in the core.
        """
        masked = embeddings if mask is None else embeddings * mask.unsqueeze(-1).to(embeddings.dtype)
        k = masked.shape[-1]
        eye = torch.eye(k, device=embeddings.device, dtype=embeddings.dtype)
        gram = masked.transpose(-1, -2) @ masked + PROJECTION_RIDGE * eye
        inverse = torch.linalg.solve(gram, eye.expand(masked.shape[0], k, k))
        proj = masked @ inverse @ masked.transpose(-1, -2)
        if mask is not None:
            proj = proj * (mask.unsqueeze(-1) & mask.unsqueeze(-2)).to(proj.dtype)
        return 0.5 * (proj + proj.transpose(-1, -2))

    def compute_loss(  # type: ignore[override]
        self,
        batch: CasmiBatch,
        return_components: bool = False,
    ):
        tb = batch.batch
        cond = Conditioning.from_batch(batch)
        x_0, atom_mask = tb.x_0, tb.atom_mask
        batch_size = x_0.shape[0]

        memory, memory_mask = self.casmi_model.encode(tb.mz, tb.intensity, tb.spectrum_mask, tb.precursor_mz, cond)

        t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)

        atom_type_active = (
            self.atom_type_loss_weight > 0
            and tb.atom_type_targets is not None
            and self.model.atom_type_head is not None
        )
        atom_type_logits = None
        if atom_type_active:
            predicted_noise, atom_type_logits = self.casmi_model.denoise_with_atom_types(
                x_t, t, atom_mask, memory, memory_mask
            )
        else:
            predicted_noise = self.casmi_model.denoise(x_t, t, atom_mask, memory, memory_mask)

        noise_loss = self._masked_mse(
            predicted_noise, noise, mask=atom_mask.unsqueeze(-1) if atom_mask is not None else None
        )
        loss = noise_loss
        zero = torch.tensor(0.0, device=self.device)
        components = {
            "noise": noise_loss.detach(),
            "projection": zero,
            "orthonormality": zero,
            "fingerprint": zero,
            "atom_count": zero,
            "quantile": zero,
            "atom_type": zero,
        }

        if self.projection_loss_weight > 0 or self.orthonormality_loss_weight > 0:
            sqrt_alpha = self.sqrt_alpha_cumprod_clamped[t].view(batch_size, 1, 1)
            sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod_clamped[t].view(batch_size, 1, 1)
            x_0_pred = self._reconstruct_x0(x_t, predicted_noise, sqrt_alpha, sqrt_one_minus)
            if self.projection_loss_weight > 0:
                proj_pred = self.projection_from_embeddings(x_0_pred, atom_mask)
                proj_target = self.projection_from_embeddings(x_0, atom_mask)
                mask_matrix = None if atom_mask is None else atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
                proj_loss = self._masked_mse(proj_pred, proj_target, mask=mask_matrix)
                loss = loss + self.projection_loss_weight * proj_loss
                components["projection"] = proj_loss.detach()
            if self.orthonormality_loss_weight > 0:
                ortho_loss = self._orthonormality_loss(x_0_pred, atom_mask)
                loss = loss + self.orthonormality_loss_weight * ortho_loss
                components["orthonormality"] = ortho_loss.detach()

        if self.fingerprint_loss_weight > 0 and tb.fingerprint_targets is not None and self.model.fingerprint_head is not None:
            fp_logits = self.casmi_model.predict_fingerprint_logits(memory, memory_mask)
            fp_loss = F.binary_cross_entropy_with_logits(fp_logits, tb.fingerprint_targets)
            loss = loss + self.fingerprint_loss_weight * fp_loss
            components["fingerprint"] = fp_loss.detach()

        if self.atom_count_loss_weight > 0 and tb.atom_count_targets is not None and self.model.atom_count_head is not None:
            count_pred = self.casmi_model.predict_atom_count(memory, memory_mask)
            count_loss = F.mse_loss(count_pred, tb.atom_count_targets.float())
            loss = loss + self.atom_count_loss_weight * count_loss
            components["atom_count"] = count_loss.detach()

        if self.quantile_loss_weight > 0 and self.casmi_model.quantile_head is not None:
            q_pred = self.casmi_model.predict_quantiles(memory, memory_mask)
            q_loss = F.mse_loss(q_pred, batch.spectral_quantiles)
            loss = loss + self.quantile_loss_weight * q_loss
            components["quantile"] = q_loss.detach()

        if atom_type_active and atom_type_logits is not None:
            at_loss = F.cross_entropy(
                atom_type_logits.reshape(-1, atom_type_logits.shape[-1]),
                tb.atom_type_targets.reshape(-1),
                ignore_index=-100,
            )
            loss = loss + self.atom_type_loss_weight * at_loss
            components["atom_type"] = at_loss.detach()

        if return_components:
            return loss, components
        return loss

    def train_step(  # type: ignore[override]
        self,
        optimizer: torch.optim.Optimizer,
        batch: CasmiBatch,
        return_components: bool = False,
        grad_clip: Optional[float] = None,
    ):
        self.casmi_model.train()
        optimizer.zero_grad()
        if return_components:
            loss, components = self.compute_loss(batch, return_components=True)
        else:
            loss, components = self.compute_loss(batch), None
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(self.casmi_model.parameters(), grad_clip)
        optimizer.step()
        if return_components:
            return loss.item(), {k: float(v.item()) for k, v in components.items()}
        return loss.item()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        spectrum_mask: Optional[torch.Tensor],
        precursor_mz: Optional[torch.Tensor],
        cond: Conditioning,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.casmi_model.eval()
        return self.casmi_model.encode(mz, intensity, spectrum_mask, precursor_mz, cond)

    @torch.no_grad()
    def predict_heads(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        spectrum_mask: Optional[torch.Tensor],
        precursor_mz: Optional[torch.Tensor],
        cond: Conditioning,
    ) -> dict[str, torch.Tensor]:
        """Fingerprint probabilities, sorted quantiles, atom count, and the memory."""
        memory, memory_mask = self.encode(mz, intensity, spectrum_mask, precursor_mz, cond)
        out: dict[str, torch.Tensor] = {"memory": memory, "memory_mask": memory_mask}
        if self.model.fingerprint_head is not None:
            out["fingerprint_prob"] = torch.sigmoid(self.casmi_model.predict_fingerprint_logits(memory, memory_mask))
        if self.casmi_model.quantile_head is not None:
            out["quantiles"] = self.casmi_model.predict_quantiles(memory, memory_mask, sort=True).clamp(0.0, 2.0)
        if self.model.atom_count_head is not None:
            out["atom_count"] = self.casmi_model.predict_atom_count(memory, memory_mask)
        return out

    @torch.no_grad()
    def sample(  # type: ignore[override]
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        *,
        n_atoms: int,
        atom_mask: Optional[torch.Tensor] = None,
        sampler: str = "ddim",
        n_steps: int = 50,
        eta: float = 0.0,
        x_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reverse diffusion from a precomputed conditioned memory.

        Returns ``(batch, n_atoms, k)`` eigenvector samples. Tile ``memory``
        (``repeat_interleave``) to draw several samples per spectrum.
        """
        self.casmi_model.eval()
        batch_size = memory.shape[0]
        if n_atoms < 1 or n_atoms > self.model.max_atoms:
            raise ValueError(f"n_atoms must be in [1, {self.model.max_atoms}]; got {n_atoms}.")
        if x_t is None:
            x_t = torch.randn(batch_size, n_atoms, self.model.k, device=self.device)
        if sampler == "ddpm":
            for t in reversed(range(self.n_timesteps)):
                x_t = self.p_sample(x_t, t, None, None, atom_mask, memory_mask, None, memory=memory)
            return x_t
        if sampler != "ddim":
            raise ValueError(f"Unknown sampler {sampler!r}; expected 'ddpm' or 'ddim'.")
        if n_steps > self.n_timesteps:
            raise ValueError(f"n_steps={n_steps} exceeds n_timesteps={self.n_timesteps}.")
        schedule = _make_timestep_schedule(n_steps, self.n_timesteps)
        alpha_cumprod = self.alpha_cumprod
        for i, t in enumerate(schedule):
            t_next = schedule[i + 1] if i + 1 < len(schedule) else -1
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            eps = self.casmi_model.denoise(x_t, t_tensor, atom_mask, memory, memory_mask)
            a_t = alpha_cumprod[t]
            a_next = alpha_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=self.device, dtype=a_t.dtype)
            x0_pred = (x_t - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t).clamp_min(1e-8)
            if eta > 0.0 and t_next >= 0:
                sigma = eta * torch.sqrt(
                    (1.0 - a_next) / (1.0 - a_t).clamp_min(1e-8) * (1.0 - a_t / a_next.clamp_min(1e-8))
                )
            else:
                sigma = torch.zeros((), device=self.device, dtype=a_t.dtype)
            direction = torch.sqrt((1.0 - a_next - sigma**2).clamp_min(0.0)) * eps
            x_t = torch.sqrt(a_next) * x0_pred + direction
            if eta > 0.0 and t_next >= 0:
                x_t = x_t + sigma * torch.randn_like(x_t)
        return x_t

    @torch.no_grad()
    def atom_type_logits(
        self,
        eigvecs: torch.Tensor,
        atom_mask: Optional[torch.Tensor],
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Element logits ``(batch, n_atoms, n_element_types)`` for clean eigenvectors."""
        self.casmi_model.eval()
        t_zero = torch.zeros(eigvecs.shape[0], dtype=torch.long, device=self.device)
        _, logits = self.casmi_model.denoise_with_atom_types(eigvecs, t_zero, atom_mask, memory, memory_mask)
        return logits


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------


@dataclass
class CheckpointBundle:
    model: CasmiModel
    trainer_config: CasmiTrainerConfig
    adduct_vocab: AdductVocabulary
    epoch: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)


def save_checkpoint(
    path: str | Path,
    model: CasmiModel,
    trainer_config: CasmiTrainerConfig,
    adduct_vocab: AdductVocabulary,
    *,
    epoch: Optional[int] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Save everything needed to rebuild the model with ``weights_only=True``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "spec2graph-casmi-v1",
        "core_config": dict(model.core.config.__dict__),
        "casmi_config": asdict(model.config),
        "trainer_config": asdict(trainer_config),
        "adduct_vocab": adduct_vocab.to_dict(),
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> CheckpointBundle:
    """Load a checkpoint written by :func:`save_checkpoint` (``weights_only=True``)."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "spec2graph-casmi-v1":
        raise ValueError(f"Unrecognised checkpoint format {payload.get('format')!r}.")
    core = Spec2GraphDiffusion(Spec2GraphDiffusionConfig(**payload["core_config"]))
    model = CasmiModel(core, CasmiModelConfig(**payload["casmi_config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return CheckpointBundle(
        model=model,
        trainer_config=CasmiTrainerConfig(**payload["trainer_config"]),
        adduct_vocab=AdductVocabulary.from_dict(payload["adduct_vocab"]),
        epoch=payload.get("epoch"),
        extra=dict(payload.get("extra") or {}),
    )


__all__ = [
    "CasmiModel",
    "CasmiModelConfig",
    "CasmiTrainer",
    "CasmiTrainerConfig",
    "CheckpointBundle",
    "Conditioning",
    "N_CONDITION_TOKENS",
    "PROJECTION_RIDGE",
    "load_checkpoint",
    "save_checkpoint",
]
