# 07 — Training Losses

Training Spec2Graph is a weighted sum of up to five loss terms. Only the
first is required; the rest are opt-in via `TrainerConfig`
(`spectral_diffusion.py:813`):

```python
@dataclass
class TrainerConfig:
    n_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    projection_loss_weight: float = 1.0
    orthonormality_loss_weight: float = 0.0
    fingerprint_loss_weight: float = 0.0
    atom_count_loss_weight: float = 0.0
    eigenvalue_loss_weight: float = 0.0
```

All of the math below is implemented in `DiffusionTrainer.compute_loss`
(`spectral_diffusion.py:1137`).

## 1. Noise MSE (required, primary)

The DDPM objective:

```
L_noise = E_t ‖ ε̂(x_t, t, spectrum) − ε ‖²
```

Implementation (`spectral_diffusion.py:1176`):

```python
noise_loss = self._masked_mse(
    predicted_noise, noise,
    mask=atom_mask.unsqueeze(-1) if atom_mask is not None else None,
)
loss = noise_loss
```

`_masked_mse` (`spectral_diffusion.py:965`) averages only over valid atoms:

```python
diff = (pred - target) ** 2
per_item = (diff * mask).view(diff.shape[0], -1).sum(dim=1)
denom = mask.view(mask.shape[0], -1).sum(dim=1).clamp_min(PROJECTION_EPS)
return (per_item / denom).mean()
```

That prevents padded positions from inflating the denominator (a mistake
that would tell you training is going well when really the loss is just
getting diluted by zeros).

This term alone would train a valid DDPM, but it's weak because the target
eigenvectors suffer from the sign/rotation ambiguity described in
[02 — Data processing](./02_data_processing.md). That's where the next
term comes in.

## 2. Projection loss `‖P̂_k − P_k‖²_F` (required by convention)

Default weight is `projection_loss_weight = 1.0`, so this is on by default.
The trainer:

1. Reconstructs a Tweedie-style estimate of the clean sample from the
   predicted noise (`_reconstruct_x0`, L982).
2. Builds the projection matrix from the predicted and ground-truth
   eigenvectors via QR-orthonormalization (`projection_from_embeddings`, L900).
3. Computes a mask-aware MSE of the two projections.

```python
x_0_pred = self._reconstruct_x0(x_t, predicted_noise, sqrt_alpha, sqrt_one_minus)
proj_pred   = self.projection_from_embeddings(x_0_pred, atom_mask)
proj_target = self.projection_from_embeddings(x_0,      atom_mask)
mask_matrix = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
proj_loss = self._masked_mse(proj_pred, proj_target, mask=mask_matrix)
loss += self.projection_loss_weight * proj_loss
```

### Why QR?

Raw `V̂ V̂ᵀ` is only a true orthogonal projector if `V̂` is orthonormal,
which the model isn't guaranteed to produce (especially early in training).
Taking the QR of `V̂`'s column space gives a proper orthonormal basis
`Q` first, then `P = Q Qᵀ`. Both predictions and targets go through the
same operation so they stay comparable.

### Why a mask matrix for the MSE?

`P_k` is `(n_atoms, n_atoms)`. Padded atoms should contribute nothing.
The outer product `atom_mask ⊗ atom_mask` gives a row/col mask: only
entries where both endpoints are real atoms count.

### A warning

For large `n_atoms`, allocating the full `(B, N, N)` tensor is expensive.
`projection_from_embeddings` warns once when `n_atoms >
PROJECTION_WARNING_THRESHOLD = 256` (L35).

## 3. Orthonormality regularizer (optional)

Even though the subspace-invariant projection loss handles the ambiguity
issue, sometimes you also want the predicted eigenvectors to be
approximately orthonormal — e.g., if you plan to use them directly rather
than via `P_k`. Enable this with `orthonormality_loss_weight > 0`.

Implementation (`spectral_diffusion.py:993`):

```python
gram = V̂ᵀ V̂                     # (B, k, k)
I_k  = identity(k)
L_ortho = mean( (gram - I_k)² )
```

This penalty encourages `V̂` to sit on the Stiefel manifold (columns
orthonormal). Padded atoms are zeroed out first, so they don't artificially
inflate or deflate the Gram matrix.

## 4. Fingerprint head (optional)

Enable with:
- `Spec2GraphDiffusionConfig.fingerprint_dim = 2048` (or whatever bit count).
- `TrainerConfig.fingerprint_loss_weight > 0`.
- Provide `TrainingBatch.fingerprint_targets` as a `(B, fp_dim)` binary
  tensor.

The head runs **only on the spectrum** — no diffusion, no eigenvectors:

```python
fp_logits = self.model.predict_fingerprint(mz, intensity, spectrum_mask, precursor_mz)
fingerprint_loss = F.binary_cross_entropy_with_logits(fp_logits, fingerprint_targets)
loss += self.fingerprint_loss_weight * fingerprint_loss
```

This gives the spectrum encoder a second, easier gradient signal: predict
which Morgan bits the molecule has. It acts as a regularizer and makes the
encoder's internal representation more chemistry-aware.

## 5. Atom count head (optional)

Enable with:
- `Spec2GraphDiffusionConfig.enable_atom_count_head = True`.
- `TrainerConfig.atom_count_loss_weight > 0`.
- Provide `TrainingBatch.atom_count_targets` (float, shape `(B,)`).

Loss: plain MSE on a scalar regression. Why bother? At inference, if you
don't know the atom count in advance, the sampler can call
`model.predict_atom_count` to decide how many atom slots to generate
(see [09 — Reverse diffusion](./09_reverse_diffusion.md)).

## 6. Eigenvalue head (optional, Phase 4)

Enable with:
- `Spec2GraphDiffusionConfig.enable_eigenvalue_head = True`.
- `TrainerConfig.eigenvalue_loss_weight > 0`.
- Provide `TrainingBatch.eigenvalue_targets` shape `(B, k)`.

```python
eig_pred = self.model.predict_eigenvalues(mz, intensity, spectrum_mask, precursor_mz)
eigenvalue_loss = F.mse_loss(eig_pred, eigenvalue_targets)
loss += self.eigenvalue_loss_weight * eigenvalue_loss
```

Laplacian eigenvalues encode graph invariants (connectivity, bipartiteness,
approximate edge count). Predicting them both regularizes the encoder and
enables the eigenvalue-conditioned decoder in
[11 — Advanced features](./11_advanced_features.md).

## Combined objective

```
L = L_noise
  + w_proj   · L_proj            (projection, default 1.0)
  + w_ortho  · L_ortho           (optional)
  + w_fp     · L_fingerprint     (optional)
  + w_count  · L_atom_count      (optional)
  + w_eig    · L_eigenvalues     (optional)
```

## Inspecting individual components

`compute_loss` accepts `return_components=True`:

```python
loss, comps = trainer.compute_loss(batch, return_components=True)
# comps = {
#   "noise": ..., "projection": ..., "orthonormality": ...,
#   "fingerprint": ..., "atom_count": ..., "eigenvalue": ...,
# }
```

The demo prints noise, projection, fingerprint, and atom_count side by
side so you can watch each term evolve. This is very useful when you're
tuning the weights — if the projection loss flatlines but noise keeps
falling, the subspace signal isn't being learned.

## A note on the reconstruction shortcut

Both `L_proj` and `L_ortho` need `x_0` on the prediction side. Rather than
run a separate forward pass to predict `x_0` directly, the trainer uses
Tweedie's formula on the predicted noise:

```python
x_0_pred = (x_t − sqrt(1 − ᾱ_t) · ε̂) / sqrt(ᾱ_t)
```

This is **exact** algebra on the closed-form `q_sample` equation, so no
extra approximation is introduced — it's just the same relationship
rearranged.

Next: [08 — Training loop](./08_training_loop.md) assembles all of this
into a single `train_step`.
