# 13 — Per-Atom Element Head

The diffusion model predicts eigenvectors. The eigenvectors alone are not
enough to build a molecule — each row of `V̂_k` is an atom slot, but
nothing says which element occupies that slot. This document covers the
optional **atom-type head** that closes that gap.

## The problem the head solves

The SGNO decoder is already permutation-equivariant: if two atom slots
swap rows in `V̂_k`, it swaps the corresponding rows and columns in the
predicted adjacency. So *within* the decoder pipeline, you don't need
the atom ordering to match any canonical scheme — it just has to be
*self-consistent*.

What you *do* still need is an answer to: **which element is each
row?** Without that, you can't construct an RDKit molecule and you
can't compute SMILES similarity.

One workaround is to derive the atom list from the molecular formula
(e.g. `"C17H26ClNO2"` → `["C"]*17 + ["N"] + ["O"]*2 + ["Cl"]`) and
assume it matches the model's output ordering. That assumption is
almost always wrong, which means predicted molecules end up labelled
with the wrong elements even when the adjacency is perfect.

The atom-type head gives the model a proper answer: an extra linear
layer on top of the decoder state that predicts, for each atom slot,
a distribution over elements. At inference you pair that with the
known formula via a Hungarian-style assignment and get labels that
both respect the element counts *and* align with the predicted
adjacency.

## Architecture

Inside `Spec2GraphDiffusion.__init__`, when
`config.enable_atom_type_head=True`:

```python
self.atom_type_head = nn.Sequential(
    nn.Linear(config.d_model, config.d_model),
    nn.ReLU(),
    nn.Linear(config.d_model, config.n_element_types),
)
```

`n_element_types` defaults to 13 — the 12 elements in
`spectral_diffusion.VALENCY_TABLE` plus one "unknown" slot. The
vocabulary lives in `spec2graph/data/elements.py`.

The head reads from the **decoder output**, not the encoder output.
That matters: the eigenvector projection (`eigenvec_out`) and the
atom-type projection both consume the same per-atom decoder state, so
row `i` of the noise prediction and row `i` of the element logits
refer to the same atom slot. That alignment is what makes the
Hungarian assignment tractable at inference.

## Forward-pass API

Three entry points, each with a specific use case:

| Method | Returns | When to use |
|--------|---------|-------------|
| `forward(...)` | `ε̂` `(B, N, k)` — unchanged behaviour | Existing callers. Single-tensor return preserved for backward compatibility. |
| `forward_with_atom_types(...)` | `(ε̂, logits)` with `logits: (B, N, n_element_types)` | Training, when both losses are active — avoids a duplicate decoder pass. |
| `predict_atom_types(...)` | `logits` only | Inference, when only the element logits matter. |

The three methods share a single internal helper, `_decode_atoms`,
that runs the encoder/decoder stack once. This keeps the training
cost of the head negligible: adding the atom-type loss doesn't
duplicate the decoder pass.

## Training

### Loss term

Cross-entropy on per-atom logits with PyTorch's default
`ignore_index=-100`. Implemented in `DiffusionTrainer.compute_loss`:

```python
atom_type_loss = F.cross_entropy(
    atom_type_logits.reshape(-1, n_element_types),
    atom_type_targets.reshape(-1),
    ignore_index=-100,
)
loss = loss + self.atom_type_loss_weight * atom_type_loss
```

Padded atoms in a batch must carry the sentinel `-100` in their
target index so the loss skips them naturally.

### Gating

The term is added to the total loss only when **all three** of these
hold:

1. `TrainerConfig.atom_type_loss_weight > 0`
2. `TrainingBatch.atom_type_targets is not None`
3. `model.atom_type_head is not None`

Any missing piece means the term is silently skipped and contributes
zero to the loss. This matches the existing pattern used by the
fingerprint, atom-count and eigenvalue heads.

### Providing targets

Populate `atom_type_targets` on the training batch. Shape:
`(batch, n_atoms)` long tensor. Build the per-molecule label using
`spec2graph.data.elements.atom_types_to_indices`:

```python
from spec2graph.data.elements import atom_types_to_indices, PADDING_INDEX

# rdkit.Chem.MolFromSmiles(smiles).GetAtoms() yields atoms in RDKit's
# canonical ordering — the same ordering SpectralDataProcessor uses
# for V_k, so the label rows line up with x_0's rows.
symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
indices = atom_types_to_indices(symbols)                  # list[int], length n_atoms

# Pad to match the batch's n_atoms with the CrossEntropyLoss sentinel.
padded = indices + [PADDING_INDEX] * (max_atoms - len(indices))
```

## Inference

The head is what makes formula-aware decoding possible. The full
inference pipeline (covered in a future `decode.py` module) is:

1. Run reverse diffusion to get `V̂_k`.
2. Call `predict_atom_types(...)` on the final step to get logits
   `(N, n_element_types)`.
3. Build the element-slot list from the molecular formula, e.g.
   `["C"] * 17 + ["N"] + ["O"] * 2 + ["Cl"]`.
4. Construct a cost matrix `cost[i, j] = -logits[i, element_to_index(slots[j])]`.
5. Solve via `scipy.optimize.linear_sum_assignment` — a balanced
   bipartite matching between atom slots and formula slots.
6. Apply the assignment: each atom slot gets the element it was
   matched to.

The Hungarian step guarantees the output respects the formula's
element counts *exactly*. Individual atom predictions that conflict
with the formula (e.g. the head says "carbon" but the formula has
already allocated all carbons) are resolved optimally by the
assignment rather than by naïve argmax.

## Configuration reference

`Spec2GraphDiffusionConfig`:

| Field | Default | Description |
|-------|---------|-------------|
| `enable_atom_type_head` | `False` | Turn the head on. |
| `n_element_types` | `13` | Output dim; defaults to 12 known + 1 unknown. |

`TrainerConfig`:

| Field | Default | Description |
|-------|---------|-------------|
| `atom_type_loss_weight` | `0.0` | Weight for the cross-entropy term. |

`TrainingBatch`:

| Field | Default | Description |
|-------|---------|-------------|
| `atom_type_targets` | `None` | `(B, N)` long tensor; `-100` for padded atoms. |

All defaults preserve legacy behaviour — a model built with no
configuration changes is bit-for-bit compatible with pre-head
checkpoints (the head is simply not constructed).

## Design notes worth keeping in mind

- **Why a dedicated head instead of a larger eigenvec output?**
  Element identity and eigenvector geometry are qualitatively
  different outputs; a shared projection couples them in a way that
  hurts both. Separate heads with shared backbone features is the
  standard recipe (it mirrors the fingerprint/eigenvalue heads
  already in the model).
- **Why not augment training with permuted atom orderings?**
  Permutation augmentation is the next obvious robustness upgrade
  but is orthogonal to getting the head working. Both outputs
  (noise and element logits) come from the same decoder state, so
  they're aligned *by construction*. Augmentation is a follow-up,
  not a prerequisite.
- **Why the extra "unknown" vocab slot?**
  Out-of-vocab elements would otherwise silently collapse to index
  0 (carbon), which masks data-quality issues. Routing them to an
  explicit unknown slot makes them easy to flag in training logs.
