# 01 — Pipeline Overview

Spec2Graph learns a map from a mass spectrum to a molecular graph. It does
this in two conceptual halves:

1. **Spectrum → eigenvectors.** A diffusion model takes mass-spec peaks and
   generates the top-`k` eigenvectors of the target molecule's Laplacian.
2. **Eigenvectors → graph.** A decoder turns those eigenvectors into a bond
   matrix (optionally constrained by chemistry rules) to produce a predicted
   molecule.

This document sketches the whole path at a high level. Each later doc zooms
into one stage.

## Inputs and outputs

| Stage | Input | Output |
|-------|-------|--------|
| Data processing | SMILES string | `V_k` eigenvectors `(N, k)` |
| Spectrum encoding | `(mz, intensity)` peak lists | Encoded memory `(B, n_peaks, d_model)` |
| Forward diffusion | `x_0 = V_k` + timestep `t` | Noisy `x_t` + the noise `ε` |
| Model forward | `x_t`, `t`, spectrum | Predicted noise `ε̂` |
| Loss | `ε̂`, `ε`, `x_0` | Scalar loss for backprop |
| Sampling | Random noise + spectrum | Generated `V̂_k` |
| Decode | `V̂_k` | Adjacency matrix / graph |

## The training path in one diagram

```text
                              ┌──────────────────────────────────┐
 SMILES ──► RDKit mol ──► adjacency ──► Laplacian ──► eigvecs V_k │ data processing
                              └──────────────────┬───────────────┘
                                                 │ (N, k)
                                                 ▼
 (mz, intensity) ──► Fourier + linear ──► TransformerEncoder ──► memory (B, n_peaks, d_model)
                                                 │                              │
                                                 ▼                              │
 t ~ Uniform{0,T-1}   ε ~ 𝒩(0, I)                                               │
              │             │                                                   │
              ▼             ▼                                                   │
              x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε                                      │
                             │                                                  │
                             ▼                                                  │
 atom_pos + time_emb ──► TransformerDecoder ◄── cross-attend to memory ─────────┘
                             │
                             ▼
                       ε̂ (B, N, k)
                             │
                             ▼
        ┌────────────────────┴──────────────────────┐
        │   noise MSE                                │ primary loss
        │   projection loss ‖P̂_k − P_k‖²_F          │ subspace-invariant
        │   orthonormality ‖V̂ᵀV̂ − I‖²_F             │ keeps outputs orthonormal
        │   (optional) fingerprint / atom-count /   │ aux heads on spectrum
        │   eigenvalue heads                         │
        └──────────────────────┬────────────────────┘
                               ▼
                           loss.backward()
                           optimizer.step()
```

At inference, the diffusion runs in reverse — start from pure noise, denoise
for `T` steps using the spectrum as conditioning, then feed the resulting
eigenvectors into a decoder to obtain bonds.

## Where everything lives

Every class below is defined in
[`spectral_diffusion.py`](../spectral_diffusion.py). The rest of this doc set
references line numbers in that file.

| Component | Class | Role |
|-----------|-------|------|
| Data processing | `SpectralDataProcessor` (L54) | SMILES → eigenvectors / projection |
| m/z embedding | `FourierMzEmbedding` (L280) | Sinusoidal features for peak positions |
| Intensity embedding | `IntensityEmbedding` (L364) | Linear embedding of intensities |
| Timestep embedding | `TimestepEmbedding` (L390) | Sinusoidal + MLP for diffusion step `t` |
| Core model | `Spec2GraphDiffusion` (L464) | Transformer encoder/decoder + heads |
| Trainer | `DiffusionTrainer` (L825) | DDPM schedule, losses, sampling |
| Graph decoder | `SpectralGraphNeuralOperator` (L1297) | Eigenvectors → adjacency logits |
| Validity scorer | `DenseGNNDiscriminator` (L1456) | Scores chemical plausibility |
| Guided sampler | `GuidedDiffusionSampler` (L1517) | Steers diffusion with discriminator |
| Valency decoding | `ValencyDecoder` (L1673) | Greedy, valency-constrained adjacency |
| Eigenvalue head | `EigenvalueConditioner` (L1783) | Optional eigenvalue conditioning |

## The one non-obvious design choice

Laplacian eigenvectors are **ambiguous**: a sign flip or a rotation inside a
degenerate eigenspace yields a different `V_k` for the same graph. Regressing
onto raw `V_k` is therefore ill-posed. Spec2Graph sidesteps this by training
an extra loss on the **projection matrix** `P_k = V_k V_k^T`, which is
invariant under those transformations. See
[`07_training_losses.md`](./07_training_losses.md) for the full treatment.

## Reading path suggestions

- "I want to understand training." Read 02 → 05 → 06 → 07 → 08.
- "I want to understand inference." Read 02 → 06 → 09 → 10.
- "I want to add a new auxiliary loss." Read 06 → 07 → 08.
- "I want to swap the decoder." Read 02 → 10 → 11.
