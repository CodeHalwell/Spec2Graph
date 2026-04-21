# 04 — Batching and Masks

Molecules have different atom counts. Spectra have different peak counts.
To run in a single batched forward pass we pad each tensor to the batch
maximum and use boolean masks to tell the model which positions are real.

This doc covers the conventions, the validation checks, and the end-to-end
construction of a padded batch.

## Mask convention: **True = valid**

Every user-facing mask in Spec2Graph uses `True = valid, False = padding`.
This is enforced by `_validate_mask` (`spectral_diffusion.py:570`):

```python
if mask.dtype != torch.bool:
    raise ValueError(f"{name} must be a boolean tensor with True indicating valid entries.")
if mask.dim() != 2 or mask.shape[1] != expected_length:
    raise ValueError(f"{name} must have shape (batch, length); got {mask.shape}.")
if not mask.any(dim=1).all():
    raise ValueError(f"{name} must contain at least one valid element per batch item.")
```

Those three checks catch the most common mistakes in practice:

1. Passing a float/int mask instead of bool.
2. Passing a mask with the wrong length.
3. Accidentally creating an all-padding row (the model has nothing to attend
   to and silently produces garbage).

The mask is only inverted (`~mask`) right before being handed to PyTorch's
transformer layers, which use `True = ignore`. Users never need to think
about that flip.

## The two masks you pass

| Name | Shape | Meaning |
|------|-------|---------|
| `atom_mask` | `(B, n_atoms)` | Which eigenvector rows are real atoms. |
| `spectrum_mask` | `(B, n_peaks)` | Which spectrum positions are real peaks. |

Both are propagated through:

- `encode_spectrum` — `spectrum_mask` becomes `src_key_padding_mask`.
- `forward` / `decoder` — `atom_mask` becomes `tgt_key_padding_mask`,
  `spectrum_mask` becomes `memory_key_padding_mask`.
- `_masked_mse` — the noise loss is normalized by the number of valid
  atoms, not the padded length.
- `projection_from_embeddings` — masked atoms are zeroed before the QR
  decomposition, and masked rows/cols of `P_k` are zeroed out after.
- `_pool_spectrum` — mean-pool over valid peaks only.
- `_orthonormality_loss` — masked atoms are zeroed before computing
  `VᵀV`.

## Constructing a padded batch

The pattern used in `create_synthetic_demo_dataset`
(`spectral_diffusion.py:1908`) is the canonical recipe:

```python
# Variable-length per example
eig_list       # list of (n_i, k)
mz_list        # list of (p_i,)
intensity_list # list of (p_i,)

max_atoms = max(e.shape[0] for e in eig_list)
max_peaks = max(len(m) for m in mz_list)

x0            = np.zeros((B, max_atoms, k), dtype=np.float32)
atom_mask     = np.zeros((B, max_atoms), dtype=bool)
mz            = np.zeros((B, max_peaks), dtype=np.float32)
intensity     = np.zeros((B, max_peaks), dtype=np.float32)
spectrum_mask = np.zeros((B, max_peaks), dtype=bool)

for i in range(B):
    n = eig_list[i].shape[0]
    p = len(mz_list[i])
    x0[i, :n, :]            = eig_list[i]
    atom_mask[i, :n]        = True
    mz[i, :p]               = mz_list[i]
    intensity[i, :p]        = intensity_list[i]
    spectrum_mask[i, :p]    = True
```

Two things worth noting:

1. **Zero padding is safe.** The Laplacian eigenvector tensor is zero-padded
   because masked positions are multiplied out before any projection or
   orthonormality computation. Zero padding for m/z also works because those
   positions are masked out of attention; the Fourier features at m/z=0
   never reach the loss.
2. **No NaNs, ever.** Pad with zeros, not NaNs. Every numerical op treats
   zero cleanly; NaNs will propagate through log-sum-exp softmax and
   silently corrupt gradients even on masked positions.

## Alternative: build masks from sequence lengths

When you have `n_atoms` and `n_peaks` as 1-D tensors, use the following
idiomatic broadcast:

```python
atom_mask     = torch.arange(max_atoms).unsqueeze(0) < n_atoms.unsqueeze(1)
spectrum_mask = torch.arange(max_peaks).unsqueeze(0) < n_peaks.unsqueeze(1)
```

Both produce the same shape and meaning as the loop above.

## Packaging into a `TrainingBatch`

All the tensors plus any optional targets ride in a
`TrainingBatch` dataclass (`spectral_diffusion.py:800`):

```python
@dataclass
class TrainingBatch:
    x_0: torch.Tensor                       # (B, n_atoms, k)
    mz: torch.Tensor                        # (B, n_peaks)
    intensity: torch.Tensor                 # (B, n_peaks)
    atom_mask: Optional[torch.Tensor]       # (B, n_atoms) bool
    spectrum_mask: Optional[torch.Tensor]   # (B, n_peaks) bool
    precursor_mz: Optional[torch.Tensor]    # (B,)
    fingerprint_targets: Optional[torch.Tensor]  # (B, fp_dim)
    atom_count_targets: Optional[torch.Tensor]   # (B,)
    eigenvalue_targets: Optional[torch.Tensor]   # (B, k)
```

Every field except `x_0`, `mz`, and `intensity` is optional. If you don't
want a particular loss or aux head, just leave the corresponding field as
`None` — the loss for that term will be skipped automatically (see
[07 — Training losses](./07_training_losses.md)).

## Why validation is strict

`_validate_mask` runs on every forward pass. The cost is a handful of tensor
ops, but it catches entire classes of silent training failures:

- A spectrum row where every peak is padded (would produce NaN attention
  weights).
- A mask passed in as `float32` (PyTorch would silently cast and get the
  semantics wrong).
- A mask whose length doesn't match the tensor (off-by-one in a collator).

Debug once, catch forever.

Next: [05 — Forward diffusion](./05_forward_diffusion.md) explains how
clean eigenvectors become noisy targets.
