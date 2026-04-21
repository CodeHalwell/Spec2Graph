# Spec2Graph Training Pipeline — Documentation

This folder walks through Spec2Graph end-to-end, from a raw mass spectrum and
SMILES string to a predicted molecular graph. Each document covers one stage
of the pipeline in detail so you can jump to the part you care about without
reading the whole codebase.

The full pipeline lives in a single file, [`spectral_diffusion.py`](../spectral_diffusion.py),
and every stage referenced below points back to specific line numbers there.

## Read in order

1. [01 — Pipeline overview](./01_pipeline_overview.md)
   The 10,000-ft view: what comes in, what comes out, and how the pieces fit.
2. [02 — Data processing: SMILES to eigenvectors](./02_data_processing.md)
   `SpectralDataProcessor`: adjacency, normalized Laplacian, sign-canonicalized
   eigenvectors, and the projection matrix `P_k`.
3. [03 — Spectrum encoding](./03_spectrum_encoding.md)
   Fourier m/z features, intensity projection, and the timestep embedding.
4. [04 — Batching and masks](./04_batching_and_masks.md)
   Padding, mask conventions (`True = valid`), and how validation catches
   bad batches early.
5. [05 — Forward diffusion (adding noise)](./05_forward_diffusion.md)
   The DDPM β-schedule, `q_sample`, and the math behind `x_t`.
6. [06 — Model architecture](./06_model_architecture.md)
   `Spec2GraphDiffusion`: transformer encoder/decoder, conditioning,
   auxiliary heads.
7. [07 — Training losses](./07_training_losses.md)
   Noise MSE, subspace-invariant projection loss, orthonormality regularizer,
   and the optional fingerprint / atom-count / eigenvalue heads.
8. [08 — Training loop](./08_training_loop.md)
   `DiffusionTrainer.train_step` — how a single gradient update is assembled.
9. [09 — Reverse diffusion (sampling)](./09_reverse_diffusion.md)
   `p_sample` and `sample`: going from noise back to eigenvectors at inference.
10. [10 — Prediction decoders](./10_prediction_decoders.md)
    SGNO kernel decoder + valency-constrained adjacency decoding.
11. [11 — Advanced features](./11_advanced_features.md)
    Guided diffusion with a GNN discriminator and eigenvalue-conditioned
    decoders.
12. [12 — End-to-end walkthrough](./12_end_to_end_walkthrough.md)
    Tying it all together with the exact calls from `run_demo()`.

## Who this is for

- **New contributors** who want a map of the code before diving in.
- **Researchers** who want to understand *why* Spec2Graph predicts a projection
  matrix instead of raw eigenvectors.
- **Debuggers** who want to pinpoint which stage of the pipeline is failing.

## Conventions used throughout

- Code references use the `path:line` format (e.g. `spectral_diffusion.py:163`)
  so you can jump directly to the implementation.
- Tensor shapes are annotated inline, e.g. `x_0: (batch, n_atoms, k)`.
- Masks follow the **True = valid** convention everywhere. When a
  `TransformerEncoder` is involved, the mask is inverted (`~mask`) because
  PyTorch's `src_key_padding_mask` uses the opposite convention.
- Symbols:
  - `N` or `n_atoms` — number of atoms in a molecule
  - `k` — number of Laplacian eigenvectors kept
  - `T` — number of diffusion timesteps (`n_timesteps`, default 1000)
  - `V_k` — `(N, k)` matrix of eigenvectors
  - `P_k = V_k V_kᵀ` — `(N, N)` projection onto the eigen-subspace
