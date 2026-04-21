"""DDIM-style deterministic sampler for :class:`DiffusionTrainer`.

The default DDPM sampler runs the full ``T`` reverse steps and is
stochastic at every non-final step. DDIM (Song et al., 2021) lets us:

* **Subsample** the schedule to ``M << T`` steps, giving a large
  wall-time speedup.
* **Skip the per-step noise** (``eta = 0``), making sampling
  deterministic given the initial ``x_T``.

Both matter for the benchmark: 100 samples × 1000 reverse steps per
test spectrum is hours of GPU time; dropping to 100 × 50 DDIM steps is
roughly 20× cheaper.

The implementation reuses the trainer's α̅ schedule — it does not
redefine β or re-initialise anything on the model. It is pure
inference.
"""

from __future__ import annotations

from typing import Optional

import torch

from spectral_diffusion import DiffusionTrainer

# Maximum number of timesteps to request. Anything larger than the
# trainer's own ``n_timesteps`` is equivalent to the full schedule, but
# we bail out early rather than silently doing that.
_MAX_TIMESTEPS = 10_000


def _make_timestep_schedule(n_steps: int, max_t: int) -> list[int]:
    """Return a length-``n_steps`` descending list of timestep indices.

    Timesteps are evenly spaced between ``max_t - 1`` (inclusive) and
    ``0`` (inclusive) so the schedule always starts at the noisiest
    step and ends at the clean one. Duplicates are deduplicated — e.g.
    if ``n_steps > max_t`` the list collapses to every timestep.
    """
    import numpy as np

    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1; got {n_steps}.")
    if max_t < 1:
        raise ValueError(f"max_t must be >= 1; got {max_t}.")
    if n_steps > _MAX_TIMESTEPS:
        raise ValueError(f"n_steps={n_steps} exceeds sanity cap {_MAX_TIMESTEPS}.")
    if n_steps >= max_t:
        return list(range(max_t - 1, -1, -1))
    # Linspace from 0 to max_t - 1 inclusive, rounded to ints, then
    # reversed so we iterate from noisiest to cleanest.
    raw = np.linspace(0, max_t - 1, n_steps).round().astype(int).tolist()
    # Deduplicate while preserving the ascending order so max_t-1 and 0
    # remain the endpoints.
    seen: list[int] = []
    for t in raw:
        t = max(0, min(max_t - 1, int(t)))
        if not seen or t != seen[-1]:
            seen.append(t)
    return list(reversed(seen))


@torch.no_grad()
def ddim_sample(
    trainer: DiffusionTrainer,
    mz: torch.Tensor,
    intensity: torch.Tensor,
    *,
    n_atoms: Optional[int] = None,
    atom_mask: Optional[torch.Tensor] = None,
    spectrum_mask: Optional[torch.Tensor] = None,
    precursor_mz: Optional[torch.Tensor] = None,
    x_t: Optional[torch.Tensor] = None,
    n_steps: int = 50,
    eta: float = 0.0,
) -> torch.Tensor:
    """DDIM sample from a trained Spec2GraphDiffusion.

    Parameters
    ----------
    trainer:
        A :class:`DiffusionTrainer` holding the trained model and α̅
        schedule.
    mz, intensity, atom_mask, spectrum_mask, precursor_mz:
        Same as :meth:`DiffusionTrainer.sample`.
    n_atoms:
        Optional — number of atoms to generate. If ``None``, uses the
        model's atom-count head (same behaviour as the DDPM sampler).
    x_t:
        Optional initial noise tensor ``(batch, n_atoms, k)``. If
        ``None`` a fresh ``N(0, I)`` draw is used.
    n_steps:
        Number of DDIM steps. Must be <= ``trainer.n_timesteps``.
    eta:
        DDIM stochasticity. ``0.0`` (default) is deterministic;
        ``1.0`` reproduces the DDPM noise magnitude.

    Returns
    -------
    torch.Tensor
        Generated eigenvectors, shape ``(batch, n_atoms, k)``.
    """
    if n_steps > trainer.n_timesteps:
        raise ValueError(
            f"n_steps={n_steps} exceeds the trainer's n_timesteps="
            f"{trainer.n_timesteps}."
        )
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0, 1]; got {eta}.")

    device = trainer.device
    batch_size = mz.shape[0]
    k = trainer.model.k
    model = trainer.model

    # ------------------------------------------------------------------
    # Resolve n_atoms / atom_mask (mirrors DiffusionTrainer.sample)
    # ------------------------------------------------------------------
    if n_atoms is None:
        count_pred = model.predict_atom_count(
            mz, intensity, spectrum_mask, precursor_mz
        )
        n_atoms_per_sample = torch.clamp(
            torch.round(count_pred), 1, model.max_atoms
        ).long()
        n_atoms = int(n_atoms_per_sample.max().item())
        if atom_mask is None:
            atom_mask = (
                torch.arange(n_atoms, device=device).unsqueeze(0).expand(batch_size, -1)
                < n_atoms_per_sample.unsqueeze(1)
            )

    if x_t is None:
        x_t = torch.randn(batch_size, n_atoms, k, device=device)

    # ------------------------------------------------------------------
    # DDIM reverse loop
    # ------------------------------------------------------------------
    schedule = _make_timestep_schedule(n_steps, trainer.n_timesteps)
    alpha_cumprod = trainer.alpha_cumprod

    for i, t in enumerate(schedule):
        t_next = schedule[i + 1] if i + 1 < len(schedule) else -1

        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        eps_pred = model(
            x_t, t_tensor, mz, intensity, atom_mask, spectrum_mask, precursor_mz
        )

        a_t = alpha_cumprod[t]
        a_next = alpha_cumprod[t_next] if t_next >= 0 else torch.tensor(
            1.0, device=device, dtype=alpha_cumprod.dtype
        )

        # Tweedie estimate of x_0.
        x0_pred = (x_t - torch.sqrt(1.0 - a_t) * eps_pred) / torch.sqrt(a_t).clamp_min(
            1e-8
        )

        if eta > 0.0 and t_next >= 0:
            sigma = eta * torch.sqrt(
                (1.0 - a_next) / (1.0 - a_t).clamp_min(1e-8)
                * (1.0 - a_t / a_next.clamp_min(1e-8))
            )
        else:
            sigma = torch.zeros((), device=device, dtype=a_t.dtype)

        direction = torch.sqrt((1.0 - a_next - sigma ** 2).clamp_min(0.0)) * eps_pred
        x_t = torch.sqrt(a_next) * x0_pred + direction

        if eta > 0.0 and t_next >= 0:
            noise = torch.randn_like(x_t)
            x_t = x_t + sigma * noise

    return x_t
