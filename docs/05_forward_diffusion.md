# 05 — Forward Diffusion (Adding Noise)

Diffusion models define a fixed procedure for corrupting a clean signal with
Gaussian noise over `T` timesteps. Spec2Graph uses the standard DDPM
formulation. This doc explains the schedule and the closed-form `q_sample`
shortcut that the trainer uses.

All code references point at `DiffusionTrainer` (`spectral_diffusion.py:825`).

## The schedule

At construction (`spectral_diffusion.py:857`):

```python
self.betas = torch.linspace(beta_start, beta_end, n_timesteps).to(device)
self.alphas = 1.0 - self.betas
self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
```

Defaults are `n_timesteps=1000`, `beta_start=1e-4`, `beta_end=2e-2`. The β
values rise linearly from tiny to moderate, so at `t=0` almost no noise is
added per step, and at `t=T-1` each step is a noticeable perturbation.

From β we derive:

- `αₜ = 1 − βₜ` — how much signal survives one step.
- `ᾱₜ = ∏ αₛ` (`s ≤ t`) — how much signal survives `t` steps.
- `ᾱ_{t-1}` — shifted one step, with `ᾱ_{-1} := 1` (the clean signal).

From those we precompute everything the forward and reverse passes need:

```python
self.sqrt_alpha_cumprod            = sqrt(ᾱ)
self.sqrt_one_minus_alpha_cumprod  = sqrt(1 − ᾱ)
self.sqrt_alpha_cumprod_clamped    = clamp(..., min=PROJECTION_EPS)
self.sqrt_one_minus_alpha_cumprod_clamped = clamp(..., min=PROJECTION_EPS)
self.sqrt_recip_alpha = sqrt(1 / α)
self.posterior_variance = β · (1 − ᾱ_{t-1}) / (1 − ᾱ + 1e-8)
```

`PROJECTION_EPS = 1e-8` is a numerical floor. At `t = 0`, `sqrt_alpha_cumprod`
is close to 1 and no clamp is needed. The clamped versions are used in
`_reconstruct_x0` so we never divide by ~0.

## The closed-form noising step `q_sample`

DDPM's key trick: you don't have to simulate `t` individual noise-adding
steps to get `xₜ`. A closed form does it in one shot:

```
x_t = sqrt(ᾱ_t) · x_0  +  sqrt(1 − ᾱ_t) · ε,    ε ~ 𝒩(0, I)
```

Implementation (`spectral_diffusion.py:877`):

```python
def q_sample(self, x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alpha          = self.sqrt_alpha_cumprod[t].view(-1, 1, 1)
    sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1)
    x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    return x_t, noise
```

Three details to notice:

1. **`t` is per-sample.** Each example in the batch gets its own random `t`
   (see [08 — Training loop](./08_training_loop.md)). The `.view(-1, 1, 1)`
   reshapes the coefficients so they broadcast against `x_0`'s
   `(B, n_atoms, k)` shape.
2. **The noise is returned.** The model is trained to predict the noise,
   not the clean sample, so we need `ε` to compute the loss.
3. **Mask unaware.** `q_sample` produces noise on padding rows as well.
   That's fine — the loss masks those out. There's a small amount of
   compute waste, but keeping the noising op simple and branch-free is
   worth it.

## Why predict noise and not `x_0` directly?

Either target is mathematically valid — they're related by

```
x_0 = (x_t − sqrt(1 − ᾱ_t) · ε) / sqrt(ᾱ_t)
```

In practice, predicting ε gives a more uniform variance across `t`, which
makes training more stable. DDPM paper, Section 3.2, covers this; the
trainer follows that convention.

For the projection loss we still need an estimate of `x_0`. The trainer
reconstructs it on the fly via `_reconstruct_x0` (`spectral_diffusion.py:982`):

```python
return _safe_divide(x_t - sqrt_one_minus * eps_pred, sqrt_alpha)
```

`_safe_divide` uses `PROJECTION_EPS`-clamped denominator to avoid blowups at
`t ≈ 0`.

## Device placement

`self.betas = torch.linspace(...).to(device)` — all schedule tensors are
moved to `device` at construction time. This means you need to create the
trainer **after** the model is moved to GPU:

```python
model = Spec2GraphDiffusion(config).to("cuda")
trainer = DiffusionTrainer(model=model, config=TrainerConfig(...), device="cuda")
```

If you ever `trainer.to(new_device)`, you'll need to move the schedule
tensors too. There's no multi-device support built in — this is a
single-device trainer.

## Mental picture of the forward process

```
t=0    x_0  (clean eigenvectors)
t=250  x_t  ≈ 0.89 · x_0 + 0.45 · ε        # still recognisable
t=500  x_t  ≈ 0.50 · x_0 + 0.87 · ε        # noise dominates
t=999  x_t  ≈ 0.00 · x_0 + 1.00 · ε        # pure noise
```

(Exact coefficients depend on the β-schedule; the point is that by the last
timestep, `x_t` is indistinguishable from a `𝒩(0, I)` draw.)

The reverse process — going from `x_999` back to `x_0` — is what the
transformer learns. That's covered in
[09 — Reverse diffusion](./09_reverse_diffusion.md).

Next: [06 — Model architecture](./06_model_architecture.md) walks through
the transformer used as the denoiser.
