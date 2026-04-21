# 09 — Reverse Diffusion (Sampling)

At inference the diffusion runs in reverse: start from pure Gaussian
noise, denoise one step at a time conditioned on the mass spectrum, and
end up with a clean `V̂_k` that the decoder can turn into bonds.

Two functions implement this: `p_sample` for a single step, and `sample`
for the whole loop.

## `p_sample`: one reverse step

Source at `spectral_diffusion.py:1016`.

Given `x_t` and the current timestep `t`, produce `x_{t-1}`:

```python
with torch.no_grad():
    predicted_noise = self.model(
        x_t, t_tensor, mz, intensity, atom_mask, spectrum_mask, precursor_mz,
    )

alpha         = self.alphas[t]
alpha_cumprod = self.alpha_cumprod[t]
beta          = self.betas[t]

# Mean of p(x_{t-1} | x_t)
mean = (1.0 / torch.sqrt(alpha)) * (
    x_t - (beta / torch.sqrt(1.0 - alpha_cumprod)) * predicted_noise
)

if t > 0:
    noise    = torch.randn_like(x_t)
    variance = self.posterior_variance[t]
    x_t_minus_1 = mean + torch.sqrt(variance) * noise
else:
    x_t_minus_1 = mean

return x_t_minus_1
```

This is the DDPM paper's Algorithm 2 sample line. Two things to note:

1. **Stochastic until the last step.** At every `t > 0` we add
   `sqrt(posterior_variance) · noise`. At `t = 0` we take the mean
   deterministically — adding noise at the final step would just hurt
   sample quality.
2. **`torch.no_grad()`.** Gradients aren't needed during sampling.
   `GuidedDiffusionSampler` (see
   [11 — Advanced features](./11_advanced_features.md)) intentionally
   re-enables gradients for its Tweedie step.

## `sample`: the full loop

Source at `spectral_diffusion.py:1069`. Signature:

```python
@torch.no_grad()
def sample(
    self,
    mz, intensity,
    n_atoms=None,
    atom_mask=None,
    spectrum_mask=None,
    precursor_mz=None,
    x_t=None,
):
```

### Determining `n_atoms`

If you know how many heavy atoms the target molecule has, pass `n_atoms`
directly. If you don't, the sampler uses the optional atom-count head:

```python
if n_atoms is None:
    count_pred = self.model.predict_atom_count(
        mz, intensity, spectrum_mask, precursor_mz,
    )
    n_atoms_per_sample = torch.clamp(
        torch.round(count_pred), 1, self.model.max_atoms,
    ).long()
    n_atoms = int(n_atoms_per_sample.max().item())
    if atom_mask is None:
        atom_mask = (
            torch.arange(n_atoms, device=self.device).unsqueeze(0)
                .expand(batch_size, -1)
            < n_atoms_per_sample.unsqueeze(1)
        )
```

Two pragmatic choices here:

- **Sequence length is the max count in the batch.** Different examples can
  still predict different atom counts — they're distinguished via the
  auto-constructed `atom_mask`. Padding up to the batch max is the cheapest
  way to keep a single dense tensor.
- **Relies on the atom-count head being enabled.** `predict_atom_count`
  always exists on the model, but if you never set
  `enable_atom_count_head=True` it raises a `ValueError` at call time:
  *"Atom count head is disabled. Set enable_atom_count_head=True to enable
  predictions."* So `sample(..., n_atoms=None)` will fail fast with that
  message rather than silently producing garbage.

### Constructing `x_T`

By default, start from pure Gaussian noise of shape `(B, n_atoms, k)`:

```python
x_t = torch.randn(batch_size, n_atoms, k, device=self.device)
```

You can also pass your own `x_t` — useful for partial-noise "editing" or
deterministic reruns with a seeded tensor.

### The reverse loop

```python
for t in reversed(range(self.n_timesteps)):
    x_t = self.p_sample(
        x_t, t, mz, intensity, atom_mask, spectrum_mask, precursor_mz,
    )
return x_t
```

At `n_timesteps = 1000` that's a thousand forward passes through the
transformer. With batch size 1 on CPU this is slow; with a GPU and batch
size 32 it's tolerable but not instant. Speed-ups commonly used elsewhere:

- **DDIM sampler.** Deterministic, supports subsampling the schedule
  (e.g., 50 steps). Not implemented here but a natural drop-in.
- **Fewer training timesteps.** The demo uses `n_timesteps=100`, which
  makes sampling tractable for a quick demo.

## Using the sample

The output is `V̂_k ∈ ℝ^{B × n_atoms × k}` — the model's best estimate of
the target eigenvectors, matched to the input spectrum.

You can evaluate the subspace similarity against known targets with
`projection_from_embeddings`:

```python
target_proj = DiffusionTrainer.projection_from_embeddings(x_0[:1], mask=atom_mask[:1])
gen_proj    = DiffusionTrainer.projection_from_embeddings(generated.cpu())
sim = F.mse_loss(gen_proj, target_proj)
```

That's exactly what the demo prints at the end as "projection similarity
(lower is better)".

For a full molecule reconstruction, feed `V̂_k` into a decoder — see
[10 — Prediction decoders](./10_prediction_decoders.md).

## Implementation subtleties worth knowing

### Mask handling during sampling

The model's `forward` validates masks on every call. That includes every
one of the ~1000 steps in `p_sample`. The cost is negligible but make sure
the masks you pass are compatible with the batch at all times — if you
construct them on the fly from predicted atom counts, double-check you
aren't creating all-False rows.

### Timestep tensor

`p_sample` wraps `t` as `torch.full((B,), t, device=..., dtype=torch.long)`.
The model's `TimestepEmbedding` expects a 1-D long tensor per batch item,
so this broadcast is required — passing a scalar silently fails because
the sinusoidal expansion treats `(B, d_model)` and `(d_model,)` shapes
very differently.

### Running on CPU

Everything supports CPU; the schedule tensors live on `device` and the
model moves with `.to(device)`. With defaults, one `p_sample` call on
CPU is milliseconds; the whole reverse loop is tens of seconds for
`n_timesteps=1000`. Fine for debugging, not for production.

### Deterministic sampling

To reproduce a sample exactly, seed `torch.manual_seed` before calling
`sample` — but only if you also pass a pre-computed `x_t`. The two sources
of randomness in the loop are the initial noise and the per-step noise,
and both go through the default PyTorch RNG.

Next: [10 — Prediction decoders](./10_prediction_decoders.md) turns the
generated eigenvectors into an actual molecular graph.
