# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Spec2Graph maps an MS/MS mass spectrum to a molecular graph by predicting a
**spectral embedding** (the top-`k` eigenvectors of the molecule's normalized
graph Laplacian) as an intermediate, then reconstructing bonds/adjacency from
that embedding. A DDPM diffusion model generates the eigenvectors conditioned on
the spectrum.

The central design decision — read this before touching the model or losses: do
**not** train on raw eigenvectors. They are sign/rotation ambiguous (especially
with degenerate eigenvalues). Instead train on the subspace-invariant projection
`P_k = V_k V_kᵀ` plus an orthonormality regularizer. This is why the loss is
"projection-aware" rather than a plain MSE on eigenvectors. `ROADMAP.md` and
`README.md` explain the theory ("clever but cursed").

## Repository layout (two tiers)

- **`spectral_diffusion.py`** — the monolithic core (~2300 lines). Contains
  `SpectralDataProcessor` (SMILES → Laplacian → sign-canonicalized eigenvectors
  → `P_k`), the Fourier/intensity/timestep embeddings, `Spec2GraphDiffusion`
  (transformer encoder/decoder + auxiliary heads), `DiffusionTrainer` (DDPM
  schedule, `train_step`, `q_sample`, `p_sample`, `sample`),
  `SpectralGraphNeuralOperator` (SGNO bond decoder), `ValencyDecoder`, guided
  diffusion, eigenvalue conditioning, and `run_demo()`. The `__main__` block
  runs the synthetic demo.
- **`spec2graph/`** — an **additive** package layered on top. Per its
  `__init__.py`, everything here (data loading, caching, evaluation, CLI
  scripts) must **not mutate the core module** — it only imports from
  `spectral_diffusion`. Subpackages:
  - `data/` — MassSpecGym download/filter (`massspecgym.py`), on-disk
    eigenvector/fingerprint cache (`cache.py`), `MassSpecGymDataset`
    (`dataset.py`), collators (`collate.py`), permutation augmentation
    (`augment.py`), element vocabulary (`elements.py`).
  - `eval/` — `benchmark.py` (the de novo benchmark loop), `metrics.py`
    (top-k Tanimoto / MCES, validity), `decode.py` (formula parsing, adjacency
    decode), `ddim.py` (DDIM sampler).
  - `train/sgno_trainer.py` — `SGNOTrainer` for adjacency supervision of the SGNO.
  - `scripts/` — CLI entry points (see Commands).
- **`docs/`** — 15 numbered walkthroughs of each pipeline stage, cross-referenced
  to `spectral_diffusion.py` by `path:line`. Start at `docs/README.md`.
- **`.jules/`** — accumulated agent learnings: `bolt.md` (performance) and
  `sentinel.md` (security). Read these before optimizing or touching parsing/IO;
  see Conventions below.

## Commands

```bash
pip install -r requirements.txt          # Python 3.10+, torch>=2.6, rdkit

python spectral_diffusion.py             # run the synthetic end-to-end demo

pytest                                    # full suite (227 tests)
pytest -m "not network"                   # skip tests that download data
pytest -m network                         # only the network/download tests
pytest tests/test_metrics.py             # single file
pytest tests/test_metrics.py::test_name  # single test
```

CLI pipeline (all idempotent / re-runnable; run as modules):

```bash
python -m spec2graph.scripts.prepare_data --cache-dir ~/.cache/spec2graph --k 8 --n-jobs 8
python -m spec2graph.scripts.train --cache-dir ~/.cache/spec2graph --output-dir runs/baseline --epochs 20
python -m spec2graph.scripts.evaluate --checkpoint runs/baseline/epoch_020.pt \
    --sgno-checkpoint runs/baseline/sgno_epoch_020.pt --cache-dir ~/.cache/spec2graph --split test
```

Tests downloading from the internet are gated behind the `network` marker
(`pytest.ini`); CI-style runs should use `-m "not network"`.

## Conventions

- **Masks use `True = valid`** everywhere (`atom_mask`, `spectrum_mask`). Batch
  items with all-False entries raise at runtime. When passing to PyTorch
  `TransformerEncoder`, masks are inverted (`~mask`) because `src_key_padding_mask`
  uses the opposite convention.
- **Shared-encoder caching:** auxiliary heads (`predict_fingerprint`,
  `predict_atom_count`, `predict_eigenvalues`, atom-type) accept an optional
  pre-computed `encoded` tensor. In the training loop, run the encoder once and
  pass the cached output to every head — never let each head re-run the encoder
  (causes O(N²)/O(N³) regressions). See `.jules/bolt.md`.
- **Diffusion sampling:** static spectrum context (`mz`, `intensity`) is encoded
  once outside the reverse loop and the precomputed `memory` is passed into the
  denoiser per timestep — do not re-encode inside the T-step loop.
- **Performance idioms:** zero diagonals with `.diagonal(...).zero_()` not an
  `eye()` mask; rewrite `D^{-1/2} A D^{-1/2} X` associatively to avoid dense
  O(N²) intermediates; precompute ground-truth RDKit `Mol`/fingerprints outside
  metric loops; `@functools.lru_cache` on repeated SMILES canonicalization/FP
  helpers in eval.
- **Security invariants (do not regress — see `.jules/sentinel.md`):**
  - Enforce `len(smiles) > MAX_SMILES_LENGTH` (2000) before any
    `Chem.MolFromSmiles`, and validate `mol.GetNumAtoms() > 0` after parsing.
  - `torch.load` must use `weights_only=True` for checkpoints.
  - Validate external identifiers used in file paths (e.g. `inchikey`) against a
    strict regex before building cache paths — guards against path traversal.
  - Cap string length before `ast.literal_eval` on dataset peak lists, and bound
    chemical-formula length/atom counts in `parse_formula`.

## Working notes

- The diffusion model and the SGNO bond decoder are trained separately;
  evaluation needs both checkpoints. A randomly initialized SGNO yields near-zero
  top-k accuracy.
- The `SpectralKernel`/SGNO description in `README.md` "Model Details" is partly
  a proposed/alternative architecture — the shipped decoder is the transformer
  `Spec2GraphDiffusion`. Trust the code over prose when they differ.
- When editing the core model, keep `spec2graph/` working by import only; do not
  push package-specific concerns down into `spectral_diffusion.py`.
