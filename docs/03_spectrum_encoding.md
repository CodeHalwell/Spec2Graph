# 03 — Spectrum Encoding

Each training example pairs eigenvectors (the target) with a mass spectrum
(the conditioning). This doc covers how raw `(m/z, intensity)` peak lists
become tensors the transformer can consume.

Four embeddings are involved:

1. `FourierMzEmbedding` — positional features for m/z.
2. `IntensityEmbedding` — linear projection of peak intensities.
3. `TimestepEmbedding` — for the diffusion step `t` (covered here because it
   mirrors the Fourier idea).
4. `atom_pos_embedding` — a learned `nn.Embedding` for atom indices
   (defined inside `Spec2GraphDiffusion`).

## FourierMzEmbedding

Defined at `spectral_diffusion.py:280`.

Mass-spec peaks live on a continuous axis (m/z ∈ roughly `[0, 2000]`). A
naïve embedding would truncate to integers or slam them through a single
`Linear` that has to learn everything. Fourier features instead project each
m/z onto a bank of sinusoids at geometrically spaced frequencies, giving the
transformer a smooth, scale-aware position code.

### Frequency bank

```python
freqs = torch.exp(torch.linspace(
    math.log(1.0), math.log(max_mz / 2.0), num_freqs // 2
))
```

`num_freqs = 64` by default, so we get 32 frequency bands geometrically
spread between 1.0 and `max_mz / 2 = 1000`. The lowest frequency captures
peaks that differ by hundreds of Da; the highest resolves near-integer
m/z differences.

### Precomputed scaling

```python
self.register_buffer(
    "scaled_freqs",
    self._compute_scaled_freqs(freqs),     # freqs * 2π / max_mz
    persistent=False,
)
```

The buffer is non-persistent because it's derived from `freqs`. An override of
`_load_from_state_dict` (`spectral_diffusion.py:321`) rebuilds it whenever a
checkpoint is loaded, which keeps old checkpoints (saved before the precompute
optimization) fully compatible.

### Forward pass

```python
angles = mz.unsqueeze(-1) * self.scaled_freqs           # (B, n_peaks, 32)
features = torch.cat([sin(angles), cos(angles)], dim=-1)  # (B, n_peaks, 64)
return self.proj(features)                                # (B, n_peaks, d_model)
```

So each peak ends up with a `d_model`-dimensional embedding where the first
layer has built-in sensitivity to mass scale.

## IntensityEmbedding

Defined at `spectral_diffusion.py:364`. Simpler: a single `Linear(1,
d_model)` applied to `intensity.unsqueeze(-1)`. Intensities are typically
normalized to `[0, 1]` or log-scaled beforehand; the module doesn't
prescribe a scheme.

## Combining them

Inside `Spec2GraphDiffusion.encode_spectrum`
(`spectral_diffusion.py:580`):

```python
mz_emb  = self.mz_embedding(mz)          # (B, n_peaks, d_model)
int_emb = self.intensity_embedding(intensity)
spectrum_emb = mz_emb + int_emb          # additive fusion
```

Additive fusion keeps parameter count low and works well as a first pass.
If you later want multiplicative gating or concatenation followed by a
projection, this is the natural hook point.

The result is then passed to `self.spectrum_encoder` (a
`nn.TransformerEncoder`) with a padding mask derived from `spectrum_mask`:

```python
src_key_padding_mask = ~mask    # PyTorch uses True=ignore
encoded = self.spectrum_encoder(spectrum_emb, src_key_padding_mask=...)
encoded = self.encoder_norm(encoded)
```

Note the inversion of the mask — throughout Spec2Graph, user-facing masks
use `True = valid`. Only at the transformer boundary do we flip the
convention to match PyTorch.

## Optional: precursor conditioning

If you enable `enable_precursor_conditioning=True` in
`Spec2GraphDiffusionConfig`, the model additionally takes a single
precursor m/z per batch item. The encode path then does:

```python
precursor_emb = self.precursor_embedding(precursor_mz.unsqueeze(-1))   # (B, 1, d_model)
pooled = self._pool_spectrum(encoded, spectrum_mask)                   # (B, d_model)
fused = self.precursor_fusion(torch.cat([pooled, precursor_emb.squeeze(1)], dim=-1))
encoded = encoded + fused.unsqueeze(1)      # broadcast to all peak positions
```

Two things to notice:

- `_pool_spectrum` does **mask-aware mean pooling**
  (`spectral_diffusion.py:733`). It divides by the number of valid peaks, not
  by the padded length.
- The fused vector is **broadcast** back to every peak position. This gives
  every peak token access to the precursor context without resizing the
  sequence.

## TimestepEmbedding

Defined at `spectral_diffusion.py:390`. The diffusion model needs to know
*which* noise level it's denoising. We give it a standard sinusoidal
embedding followed by a small MLP:

```python
half_dim = d_model // 2
freqs = torch.exp(-math.log(max_period) * torch.arange(half_dim) / half_dim)
```

The forward pass is:

```python
args = t.float().unsqueeze(-1) * self.freqs           # (B, half_dim)
embedding = torch.cat([cos(args), sin(args)], dim=-1) # (B, 2*half_dim)
return self.mlp(embedding)                            # (B, d_model)
```

This timestep embedding is later **added** to the decoder input tokens (same
d_model) so every atom position is aware of the current noise level.

## Atom position embedding

Not in a separate class — it's `self.atom_pos_embedding = nn.Embedding(max_atoms, d_model)`
inside `Spec2GraphDiffusion` (`spectral_diffusion.py:505`). Atoms don't have
an inherent sequential position, but the transformer still needs a way to
distinguish "atom 0" from "atom 1" as tokens. A learned positional embedding
indexed by position does the job.

This is one of the reasons the model is **not permutation-invariant** by
design: the atom ordering is taken as given (typically the RDKit canonical
ordering). That tradeoff is flagged as a risk in `ROADMAP.md`.

## Summary of shapes

At the end of spectrum encoding, inside the model forward pass:

| Tensor | Shape |
|--------|-------|
| `mz` | `(B, n_peaks)` |
| `intensity` | `(B, n_peaks)` |
| `spectrum_mask` (optional) | `(B, n_peaks)` bool |
| `precursor_mz` (optional) | `(B,)` |
| `memory` after `encode_spectrum` | `(B, n_peaks, d_model)` |
| `t` | `(B,)` long |
| `t_emb` | `(B, d_model)` |

Next: [04 — Batching and masks](./04_batching_and_masks.md) explains how
variable-length examples are stitched together into a single batch.
