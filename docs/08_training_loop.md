# 08 — Training Loop

The outermost layer that users touch is `DiffusionTrainer.train_step`
(`spectral_diffusion.py:1249`). This doc walks through what happens in one
step end-to-end and shows how to extend it to a full training script.

## `train_step` in detail

```python
def train_step(self, optimizer, batch, return_components=False):
    self.model.train()
    optimizer.zero_grad()
    loss, components = (
        self.compute_loss(batch, return_components=True)
        if return_components
        else (self.compute_loss(batch, return_components=False), None)
    )
    loss.backward()
    optimizer.step()
    if return_components:
        return loss.item(), {k: v.item() for k, v in components.items()}
    return loss.item()
```

The chain of calls:

1. `self.model.train()` — puts dropout/BatchNorm into training mode.
2. `optimizer.zero_grad()` — clears the previous step's gradients.
3. `compute_loss(batch)` — does the heavy lifting:
    - Samples a random timestep per batch item.
    - Calls `q_sample` to noise `x_0`.
    - Runs a forward pass of the model.
    - Computes the weighted sum of all enabled loss terms
      (see [07 — Training losses](./07_training_losses.md)).
4. `loss.backward()` — autograd over the entire graph, including through
   `_reconstruct_x0` and `projection_from_embeddings`.
5. `optimizer.step()` — gradient descent update.

Nothing exotic — this is the shape of every PyTorch training step you've
ever written.

## What `compute_loss` does each step

Reading `compute_loss` line by line (`spectral_diffusion.py:1137`):

### a. Pull tensors out of the batch

```python
x_0, mz, intensity = batch.x_0, batch.mz, batch.intensity
atom_mask, spectrum_mask = batch.atom_mask, batch.spectrum_mask
precursor_mz = batch.precursor_mz
fingerprint_targets   = batch.fingerprint_targets
atom_count_targets    = batch.atom_count_targets
eigenvalue_targets    = batch.eigenvalue_targets
```

### b. Sample a timestep and noise

```python
t = torch.randint(0, self.n_timesteps, (B,), device=self.device)
noise = torch.randn_like(x_0)
x_t, _ = self.q_sample(x_0, t, noise)
```

Each example in the batch gets its own timestep. At batch size 32, you
cover 32 different noise levels per step — this is intentional and keeps
the gradient well-conditioned across the schedule.

### c. Forward through the model

```python
predicted_noise = self.model(
    x_t, t, mz, intensity, atom_mask, spectrum_mask, precursor_mz,
)
```

Predicts the noise that was added.

### d. Accumulate losses

Noise loss first (always on):

```python
noise_loss = self._masked_mse(
    predicted_noise, noise,
    mask=atom_mask.unsqueeze(-1) if atom_mask is not None else None,
)
loss = noise_loss
```

Projection + orthonormality reuse `_reconstruct_x0` to get a clean estimate:

```python
if self.projection_loss_weight > 0 or self.orthonormality_loss_weight > 0:
    sqrt_alpha = self.sqrt_alpha_cumprod_clamped[t].view(B, 1, 1)
    sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod_clamped[t].view(B, 1, 1)
    x_0_pred = self._reconstruct_x0(x_t, predicted_noise, sqrt_alpha, sqrt_one_minus)
```

Then each optional term is added only if its weight is > 0 and the needed
targets are present:

```python
if self.projection_loss_weight > 0 and x_0_pred is not None:
    ...
    loss += self.projection_loss_weight * proj_loss

if self.orthonormality_loss_weight > 0 and x_0_pred is not None:
    ...
    loss += self.orthonormality_loss_weight * ortho_loss

if self.fingerprint_loss_weight > 0 and fingerprint_targets is not None:
    ...
    loss += self.fingerprint_loss_weight * fingerprint_loss

# same pattern for atom_count and eigenvalue
```

Notice the guard on targets: if you forgot to pass `fingerprint_targets`
but set `fingerprint_loss_weight > 0`, the fingerprint term is silently
skipped. That's forgiving, but it also means a misconfigured training run
can look like it's working when it isn't. Verify loss components
(`return_components=True`) match your expectation for the first few steps.

### e. Return

`compute_loss` returns either a scalar `loss` or `(loss, components)`.
Keep in mind the components dictionary is **detached** — it's for
logging, not backprop.

## A full training loop

The demo (`spectral_diffusion.py:2075`) shows the minimal version:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for step in range(n_steps):
    loss, components = trainer.train_step(
        optimizer, batch, return_components=True,
    )
    if (step + 1) % 5 == 0:
        print(
            f"   Step {step + 1}/{n_steps}, total loss: {loss:.4f} | "
            f"noise: {components['noise']:.4f}, "
            f"proj: {components['projection']:.4f}, "
            f"fp: {components['fingerprint']:.4f}, "
            f"atoms: {components['atom_count']:.4f}"
        )
```

Scaling this up to a real training script typically means adding:

```python
from torch.utils.data import DataLoader

for epoch in range(num_epochs):
    for raw_batch in dataloader:          # your dataset yields dicts/tuples
        batch = TrainingBatch(**raw_batch).to(device)
        loss, comps = trainer.train_step(optimizer, batch, return_components=True)
        # log, step LR scheduler, evaluate, checkpoint, etc.
    # optional: evaluate / sample at end of epoch
```

Things not built into the trainer but worth adding in a real run:

- **LR scheduling** — `torch.optim.lr_scheduler.CosineAnnealingLR` or warmup
  + cosine. The trainer doesn't touch the optimizer beyond `zero_grad` and
  `step`, so plug in whatever.
- **Gradient clipping** — `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
  between `backward()` and `step()`. The projection loss occasionally
  produces spiky gradients.
- **Mixed precision** — `torch.amp.autocast` + `GradScaler`. You'd need to
  wrap `compute_loss` with autocast and insert `scaler.scale(loss).backward()`.
  The trainer doesn't currently do this itself.
- **EMA of model weights** — a very common DDPM trick that smooths sample
  quality. Easy to bolt on outside the trainer.
- **Checkpointing** — `torch.save({"model": model.state_dict(), "optimizer":
  optimizer.state_dict(), "config": config})`.

## Common issues and diagnoses

- **Noise loss stays high, projection low.** You're underfitting the
  denoising task. Increase model capacity, reduce projection weight
  temporarily, or check that the forward process is working correctly
  (plot `x_t` vs `x_0`).
- **Projection loss stays high, noise loss low.** The predicted noise is
  accurate in magnitude but not pointing the right way for subspace
  recovery. Try enabling the orthonormality term.
- **NaNs.** Almost always stem from a padded batch with an all-False row
  slipping past `_validate_mask`, or an `eigh` of a badly-conditioned
  Laplacian. Instrument with `torch.isnan(x).any()` at each stage.
- **Exploding losses early.** Lower the LR, add gradient clipping, or
  warm up the projection weight from 0 to its target.

Next: [09 — Reverse diffusion](./09_reverse_diffusion.md) turns the
trained model around to generate eigenvectors from a spectrum.
