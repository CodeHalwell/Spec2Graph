# 02 — Data Processing: SMILES to Eigenvectors

Before any ML runs, we need a numerical target. Spec2Graph's target is the
top-`k` eigenvectors of a molecule's normalized Laplacian. This doc walks
through the four functions of `SpectralDataProcessor`
(`spectral_diffusion.py:54`) that turn a SMILES string into that target.

The pipeline is:

```text
SMILES  ──►  adjacency  ──►  Laplacian  ──►  eigenvectors V_k  (──► P_k = V_k V_kᵀ)
```

## Step 1 — SMILES → adjacency matrix

`smiles_to_adjacency` (`spectral_diffusion.py:78`) parses the SMILES with
RDKit and writes one row/column per atom. Bond weights come from one of
three modes chosen at construction time:

- `"unweighted"` — every bond has weight 1.
- `"order"` (default) — `single=1, double=2, triple=3, aromatic=1.5`.
- `"aromatic"` — aromatic bonds get 1.5, everything else 1.

Input checks you should be aware of:

- SMILES must be a `str`.
- Length is capped at `MAX_SMILES_LENGTH = 2000` (L40). This prevents DoS
  from pathological inputs that would blow up the downstream `O(N³)`
  eigendecomposition.
- Invalid SMILES raises `ValueError`.
- A molecule with zero atoms also raises — we never want to build a 0×0
  Laplacian silently.

The returned matrix is symmetric `float32` of shape `(N, N)` where `N` is the
number of heavy atoms.

## Step 2 — adjacency → normalized Laplacian

`compute_laplacian` (`spectral_diffusion.py:128`) builds:

```
L = I − D^{−1/2} · A · D^{−1/2}
```

with `D` the diagonal degree matrix. The implementation is defensive about
isolated atoms:

```python
eps = 1e-12
degree_inv_sqrt = np.where(degree > eps, 1.0 / np.sqrt(degree), 0.0)
```

So atoms with zero degree contribute `0` rather than a NaN. That keeps the
Laplacian well-defined even when you pass in disconnected fragments.

The normalized form is convenient for two reasons:

1. Eigenvalues are bounded in `[0, 2]`, which keeps training numerically
   stable.
2. Eigenvectors of the normalized Laplacian are a smooth "coordinate system"
   on the graph — closeness in eigenvector space correlates with being
   neighbors, which is exactly what the SGNO decoder will exploit later
   (see [10 — Prediction decoders](./10_prediction_decoders.md)).

## Step 3 — Laplacian → top-`k` eigenvectors

`extract_eigenvectors` (`spectral_diffusion.py:150`) calls
`np.linalg.eigh` (dense symmetric eigendecomposition) and then does three
things that matter:

### 3a. Sort ascending and skip the zero eigenvalue

For a connected graph the smallest eigenvalue of the normalized Laplacian is
exactly 0, with the constant eigenvector. That carries no structural
information, so we skip over it:

```python
start_idx = int((eigenvalues < eps).sum())
```

For disconnected graphs, the multiplicity of 0 equals the number of connected
components, and we skip all of them.

### 3b. Select `k` vectors (pad with zeros if fewer exist)

If the molecule has fewer non-trivial eigenvectors than `k`, the output is
right-padded with zero columns so downstream shapes are always `(N, k)`.

### 3c. Sign canonicalization

Eigenvectors are only defined up to a sign: `−v` is just as valid as `v`.
That breaks any regression target. To make the target **deterministic** for a
given graph, the code flips each eigenvector so its first non-negligible
entry is positive:

```python
mask = np.abs(selected) > 1e-10
has_nonzero = mask.any(axis=0)
idx = mask.argmax(axis=0)          # index of first |entry| > 1e-10
zero_cols = ~has_nonzero
if zero_cols.any():                # fallback for ~zero vectors
    idx[zero_cols] = np.abs(selected[:, zero_cols]).argmax(axis=0)
first_vals = selected[idx, col_idx]
signs = np.where(first_vals < 0, -1, 1)
selected *= signs
```

This helps training, but it **doesn't solve the rotation ambiguity** when
eigenvalues are degenerate. That's the job of the projection loss below.

## Step 4 (optional) — projection matrix `P_k`

`projection_matrix` (`spectral_diffusion.py:205`) computes:

```
P_k = V_k · V_kᵀ                      ∈ ℝ^{N×N}
```

`P_k` is the orthogonal projector onto the span of `V_k`. Two key properties:

- **Sign-invariant.** Flip any column of `V_k`: `V_kᵀ` flips the same column,
  so the product `V_k V_kᵀ` is unchanged.
- **Rotation-invariant within an eigenspace.** For any orthogonal `Q`
  mixing columns inside a degenerate eigenspace,
  `(V_k Q)(V_k Q)ᵀ = V_k Q Qᵀ V_kᵀ = V_k V_kᵀ`.

So `P_k` is a *subspace-invariant* target: two equivalent decompositions of
the same graph produce the same `P_k`. That makes it a much better regression
target than raw eigenvectors. The diffusion model itself still predicts
eigenvectors (for computational reasons), but the loss includes a penalty
on `P̂_k − P_k` — see [07 — Training losses](./07_training_losses.md).

## Step 5 (optional) — Morgan fingerprint

`smiles_to_fingerprint` (`spectral_diffusion.py:233`) builds a Morgan
(circular) fingerprint of the molecule (default 2048 bits, radius 2) as a
binary vector. It's used when you enable the auxiliary fingerprint head —
see [07 — Training losses](./07_training_losses.md).

## Putting it together: `process_smiles`

The convenience method `process_smiles` (`spectral_diffusion.py:255`) glues
the three mandatory steps together:

```python
processor = SpectralDataProcessor(k=8)
eigvecs = processor.process_smiles("c1ccccc1")             # (6, 8)
eigvecs, proj = processor.process_smiles("c1ccccc1", return_projection=True)
# proj.shape == (6, 6)
```

## Common gotchas

- **Variable atom counts.** Different molecules have different `N`. The
  demo pads them per-batch (see [04 — Batching and masks](./04_batching_and_masks.md)).
- **`k` larger than available eigenvectors.** `extract_eigenvectors` silently
  zero-pads. `projection_matrix` separately checks `k <= n_vectors` — if you
  call it with an already-padded matrix, be mindful that the trailing zero
  columns contribute nothing to `P_k`.
- **RDKit required.** `process_smiles` raises a clear `ImportError` at call
  time if RDKit is missing. Every model in the repo that doesn't need SMILES
  input will still import successfully without it.
- **Cost.** `eigh` is `O(N³)`. That's fine for drug-like molecules but
  explodes for peptides and polymers; `MAX_SMILES_LENGTH` caps the worst
  case.

## What this produces for the rest of the pipeline

After this stage, every training example is represented by:

- `V_k ∈ ℝ^{N × k}` — the clean target the diffusion model learns to denoise.
- Optionally `P_k ∈ ℝ^{N × N}` — used only by the projection loss; the
  trainer recomputes it from `V_k` at training time, so you don't usually
  need to store it alongside the data.

Next: [03 — Spectrum encoding](./03_spectrum_encoding.md) covers how the
conditioning spectrum is prepared.
