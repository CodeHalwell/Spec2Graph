# 06 — Model Architecture

`Spec2GraphDiffusion` (`spectral_diffusion.py:464`) is the neural network at
the heart of the project. It's a standard transformer encoder/decoder with
a few conditioning tricks: the encoder sees the mass spectrum, the decoder
sees noisy eigenvectors together with the diffusion timestep, and
cross-attention bridges the two.

This doc describes the layers, the forward pass, and the optional auxiliary
heads.

## Configuration

Everything tunable lives in `Spec2GraphDiffusionConfig`
(`spectral_diffusion.py:447`):

```python
@dataclass
class Spec2GraphDiffusionConfig:
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    k: int = 8                              # number of eigenvectors predicted
    max_atoms: int = 64                     # max sequence length on decoder side
    max_peaks: int = 100                    # max sequence length on encoder side
    dropout: float = 0.1
    fingerprint_dim: int = 0                # 0 disables fingerprint head
    enable_atom_count_head: bool = False
    enable_eigenvalue_head: bool = False
    enable_precursor_conditioning: bool = False
```

`max_atoms` and `max_peaks` are not just hyperparameters — `atom_pos_embedding`
is a `nn.Embedding(max_atoms, d_model)`, so exceeding `max_atoms` at runtime
raises a clear error (`spectral_diffusion.py:684`).

## Components, one by one

### Spectrum side (the encoder)

| Piece | Purpose |
|-------|---------|
| `FourierMzEmbedding(d_model)` | Project each m/z to a `d_model` vector using sinusoidal features. |
| `IntensityEmbedding(d_model)` | Linear projection of intensity. |
| Sum them | Combined peak representation. |
| `TransformerEncoder(num_encoder_layers)` | Self-attention over peaks (padded positions masked). |
| `LayerNorm` (`encoder_norm`) | Stabilize the output. |
| (optional) `precursor_embedding` + `precursor_fusion` | Inject a global precursor-m/z token. |

See [03 — Spectrum encoding](./03_spectrum_encoding.md) for the details.

### Atom side (the decoder)

| Piece | Purpose |
|-------|---------|
| `eigenvec_in = Linear(k, d_model)` | Embed noisy eigenvector rows as tokens. |
| `atom_pos_embedding` | Learned positional embedding across `max_atoms`. |
| `TimestepEmbedding(d_model)` | Sinusoidal + MLP embedding for `t`. |
| `TransformerDecoder(num_decoder_layers)` | Self-attends over atoms and cross-attends to spectrum memory. |
| `LayerNorm` (`decoder_norm`) | Stabilize. |
| `eigenvec_out = Linear(d_model, k)` | Project back to eigenvector space — predicts **noise**, not eigenvectors. |

### Auxiliary heads (optional)

All three heads operate on the pooled spectrum representation, so they
don't need the decoder. They are light MLPs wired up conditionally in
`__init__`:

| Head | Gate | Output |
|------|------|--------|
| `fingerprint_head` | `fingerprint_dim > 0` | `(B, fingerprint_dim)` logits — Morgan fingerprint prediction. |
| `atom_count_head` | `enable_atom_count_head=True` | `(B,)` regression — number of atoms. |
| `eigenvalue_head` | `enable_eigenvalue_head=True` | `(B, k)` regression — Laplacian eigenvalues. |

Each has a corresponding `predict_*` method that re-runs `encode_spectrum`
and `_pool_spectrum` before invoking the head.

## The main `forward` pass

Source at `spectral_diffusion.py:657`.

Inputs:
- `x_t`: noisy eigenvectors `(B, n_atoms, k)`
- `t`: timestep `(B,)` long
- `mz`, `intensity`: `(B, n_peaks)`
- `atom_mask`, `spectrum_mask`: `(B, *)` bool, True=valid
- `precursor_mz`: `(B,)` or None

Output:
- `ε̂`: predicted noise `(B, n_atoms, k)`

The flow inside the method:

```python
# 1. encode spectrum
memory = self.encode_spectrum(mz, intensity, spectrum_mask, precursor_mz)
# 2. timestep embedding
t_emb = self.time_embedding(t)                          # (B, d_model)
# 3. embed noisy eigenvectors
x_emb = self.eigenvec_in(x_t)                           # (B, n_atoms, d_model)
# 4. add positional + timestep
pos_emb = self.atom_pos_embedding(torch.arange(n_atoms))  # (n_atoms, d_model)
x_emb = x_emb + pos_emb + t_emb.unsqueeze(1)
# 5. decode with cross-attention to memory
tgt_key_padding_mask    = ~atom_mask     if atom_mask     is not None else None
memory_key_padding_mask = ~spectrum_mask if spectrum_mask is not None else None
decoded = self.decoder(
    x_emb, memory,
    tgt_key_padding_mask=tgt_key_padding_mask,
    memory_key_padding_mask=memory_key_padding_mask,
)
decoded = self.decoder_norm(decoded)
# 6. project back to eigenvector space (predicts noise)
return self.eigenvec_out(decoded)
```

Four things worth understanding:

1. **Timestep addition, not concatenation.** `t_emb` has shape
   `(B, d_model)` and is added with `.unsqueeze(1)` so the same timestep
   contribution is applied to every atom token. This keeps the sequence
   length unchanged and is how most DDPM denoisers wire `t` in.
2. **Positional embedding is on atom indices.** This is what prevents
   the model from being permutation-invariant. If you want order-invariance,
   you'd need to replace this with a set-style encoding and rethink the
   atom_count head.
3. **Output shape equals input shape.** The decoder is essentially a
   conditional denoiser: given a noisy `V_k`, predict the noise you'd need
   to remove.
4. **Cross-attention bridges the two modalities.** Each layer of
   `TransformerDecoder` has a cross-attention block that lets atom tokens
   pull features from spectrum tokens. This is where "condition on the
   spectrum" is implemented.

## Input validation

`encode_spectrum` checks:

- `mz.dim() == 2`, `intensity.dim() == 2`
- `mz.shape == intensity.shape`
- `mz.shape[1] <= max_peaks`
- `precursor_mz.shape == (B,)` if provided

`forward` also checks `n_atoms <= max_atoms` and validates masks via
`_validate_mask`. These are all cheap and fire before any heavy compute,
so you get an instant, descriptive error instead of a shape-broadcasting
surprise deep in the transformer.

## Parameter count at default config

The demo (`run_demo`) prints the parameter count after construction. With the
defaults (`d_model=256`, 4+4 layers, `k=8`, `max_atoms=64`, `max_peaks=100`)
you get ~5M parameters. The demo halves that by using `d_model=128` and 2+2
layers.

## Extending the model

Common modifications map onto specific hooks:

- **Different m/z encoding.** Swap `FourierMzEmbedding` for a learned MLP
  or a Gaussian-mixture kernel. Just preserve the `(B, n_peaks, d_model)`
  output shape.
- **New conditioning signal.** Copy the precursor-conditioning path:
  embed it, mean-pool the encoded spectrum, fuse, broadcast. This is the
  cleanest way to add new global context.
- **Alternative decoder.** You can replace the `TransformerDecoder` with
  a graph neural operator (the SGNO lives in the same file for exactly
  this reason), but you'd need to handle the timestep and positional
  conditioning yourself.

Next: [07 — Training losses](./07_training_losses.md) covers what the model
is actually optimised for.
