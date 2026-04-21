# 15 — MassSpecGym Benchmarking

This chapter walks through the de novo MassSpecGym challenge: given a
mass spectrum at inference time, predict the molecule that produced
it. We cover the sampling protocol, the four metrics, how the atom-
type head and the Hungarian assignment fit in, and how to run the end-
to-end benchmark.

## The de novo challenge in one paragraph

Given a mass spectrum with its molecular formula, sample candidate
molecules from the conditional generative model, rank them by sampling
frequency, keep the top-K, and measure how similar the top-K candidates
are to the ground truth. MassSpecGym reports four headline metrics:

- **Top-k exact accuracy** — percent of examples where the canonical
  ground-truth SMILES appears in the top-k predictions.
- **Top-k max Morgan Tanimoto** — "close enough" metric tolerant of
  minor structural variation.
- **Top-k min MCES** — maximum common edge subgraph distance (from
  `myopic-mces`).
- **Validity rate** — fraction of predicted SMILES that RDKit can
  parse and sanitise.

## The 100-sample sampling protocol

DiffMS, FOAM, FlowMS, FRIGID all follow this convention: for each test
spectrum, draw **100 candidate molecules**, deduplicate on canonical
SMILES, rank by frequency. We follow it too.

In code (`spec2graph/eval/benchmark.py`):

```python
predicted_smiles = batch_eigvecs_to_smiles(
    eigvecs=eigvecs,                    # (n_samples, n_atoms, k)
    atom_masks=atom_masks,              # (n_samples, n_atoms)
    formulas=[formula] * n_samples,
    atom_type_logits=atom_type_logits,  # (n_samples, n_atoms, n_elems)
    sgno=sgno,
    valency_decoder=valency_decoder,
)
ranked = rank_samples_by_frequency(predicted_smiles)
```

`rank_samples_by_frequency`:

1. Drops `None` entries (sanitisation failures).
2. Canonicalises the rest via RDKit.
3. Counts occurrences; sorts descending; breaks ties by first-seen
   order for determinism.

## The inference pipeline, step by step

From `V̂_k` to SMILES (see `spec2graph/eval/decode.py`):

```
V̂_k     atom_type_logits     formula
   \\            |              /
    SGNO   Hungarian assignment
   bond probs         atom_types
        \\          /
      ValencyDecoder
             |
          adjacency
             |
          RWMol + SanitizeMol
             |
          canonical SMILES
```

### Hungarian element assignment

The original heuristic — "derive atom types from the formula in a fixed
order" — produced near-random element labels. The model isn't trained to
output atoms in that order, so row `i` of `V̂_k` rarely matches element
`atoms[i]`.

The atom-type head (chapter 13) fixes this. At inference we:

1. Call `predict_atom_types(...)` to get `(n_atoms, n_element_types)`
   logits, aligned with the eigenvector rows.
2. Build a formula-derived list of element slots, e.g. `["C"] * 17 +
   ["N"] + ["O"] * 2 + ["Cl"]`.
3. Solve the balanced bipartite matching with
   `scipy.optimize.linear_sum_assignment`, minimising
   `cost[i, j] = -logits[i, element_to_index(slots[j])]`.

This guarantees:

- The output respects the formula's element counts exactly.
- The match is globally optimal under the log-probability sum.
- Atoms the head is uncertain about still receive a valid element —
  whichever one is left in the pool when the Hungarian runs.

### ValencyDecoder + RWMol

The SGNO's bond probabilities are turned into a hard adjacency by the
existing `ValencyDecoder` (see chapter 10), which respects per-element
valency budgets. We build an RDKit `RWMol`, invoke `SanitizeMol`, and
canonicalise the resulting SMILES. Sanitisation failures return `None`
and count as invalid for the validity metric.

## Running the benchmark

`benchmark_model` in `spec2graph/eval/benchmark.py` wires everything
together:

```python
from spec2graph.eval.benchmark import BenchmarkConfig, benchmark_model

results = benchmark_model(
    trainer=trainer,
    sgno=sgno,
    dataset=test_ds,
    valency_decoder=ValencyDecoder(),
    config=BenchmarkConfig(
        n_samples_per_spectrum=100,
        sampler="ddim",         # 20× faster than full DDPM
        ddim_n_steps=50,
        ddim_eta=0.0,
        limit=None,             # run on the full test split
    ),
    device="cuda",
    jsonl_path="runs/baseline/results.jsonl",
)
print(results.to_markdown_row("Spec2Graph"))
```

### Development knobs

`BenchmarkConfig` exposes two parameters intended strictly for
development-speed runs:

- `limit: int | None` — cap the number of test examples. Use `20–500`
  while iterating; `None` for the final report.
- `n_timesteps_override: int | None` — temporarily shorten the reverse
  diffusion schedule. Handy for "does the harness even run?" checks;
  **must be `None` for published numbers**.

The `sampler="ddim"` knob is legitimate in production — DDIM converges
to indistinguishable samples at `ddim_n_steps >= 50` and is roughly
20× faster than the default 1000-step DDPM. Switch to `"ddpm"` only if
you suspect sampler-quality issues.

### Per-example JSONL log

Every run writes one JSON object per test example to the JSONL path.
Each row contains the ground-truth SMILES, the top-10 predicted SMILES,
and every per-example metric. Use it to:

- Resume a partial run (skip examples whose `idx` is already present).
- Inspect worst-case failures (sort by `top_1_mces`).
- Compute custom post-hoc metrics without rerunning the sampler.

## Reference scores on MassSpecGym (de novo, test split)

Published baselines at the time of writing (early-to-mid 2026 —
confirm with the original papers before citing). Columns are top-1 /
top-10 exact match accuracy (%):

| Model | Top-1 | Top-10 | Notes |
|---|---|---|---|
| Random chemical generation | 0.0 | 0.0 | MassSpecGym paper baseline |
| SMILES Transformer | 0.0 | 0.0 | MassSpecGym paper baseline |
| SELFIES Transformer | 0.0 | 0.0 | MassSpecGym paper baseline |
| MADGEN | ~0 | low single digits | Scaffold-based, formula-conditioned |
| DiffMS | ~2.3 | ~4.3 | Graph diffusion under formula constraint |
| Test-time tuned LLM (Alberts et al. 2026) | ~3.2 | — | Transformer + test-time finetuning |
| FOAM | 1.5 | ~10.3 | DiffMS + forward-model optimisation |
| FRIGID | ~18 | — | Scaled diffusion |
| MIST + MolForge | ~31 | ~40 | Fingerprint-first pipeline |
| **Spec2Graph (this repo)** | TBD | TBD | To be populated |

Architecturally Spec2Graph is closest to DiffMS — both are graph
diffusion models conditioned on spectra and formulas. Matching DiffMS's
numbers is the first honest milestone.

## Known limitations

- **Atom-ordering heuristic deprecated.** The formula-derived atom
  ordering is still available via the `fallback_atom_types` parameter
  of `batch_eigvecs_to_smiles` — use it only when the atom-type head
  is disabled and you know what you're doing.
- **SGNO supervision is required.** A randomly-initialised SGNO gives
  random bond probabilities, pinning top-k accuracy near zero. Train
  one via `scripts.train --train-sgno` before running a real
  benchmark.
- **Permutation augmentation recommended but not default.** The model
  is trained on RDKit's canonical atom ordering. The atom-type head
  handles element identity, and the SGNO is permutation-equivariant,
  but the decoder's `atom_pos_embedding` is an implicit bias toward
  the training-time ordering. Enable `--permute-atoms` in
  `scripts.train` for a robustness bump.
- **100 × 1000 reverse steps is slow.** DDIM with 50 steps and `eta=0`
  is the recommended production sampler. See chapter 16 (DDIM) for the
  background.
- **MS ambiguity is intrinsic.** Structural isomers share very similar
  spectra — no decoder can disambiguate them without extra context.
  That's why top-k matters.

## Reproducing a benchmark run

```bash
# 1. Prime the cache (idempotent).
python -m spec2graph.scripts.prepare_data \\
    --cache-dir ~/.cache/spec2graph --k 8 --n-jobs 8

# 2. Train diffusion model + SGNO.
python -m spec2graph.scripts.train \\
    --cache-dir ~/.cache/spec2graph \\
    --output-dir runs/baseline \\
    --epochs 20 --batch-size 32 --lr 1e-4 \\
    --permute-atoms --train-sgno

# 3. Benchmark on test split.
python -m spec2graph.scripts.evaluate \\
    --checkpoint runs/baseline/epoch_020.pt \\
    --sgno-checkpoint runs/baseline/sgno_epoch_020.pt \\
    --cache-dir ~/.cache/spec2graph \\
    --split test --n-samples 100 \\
    --sampler ddim --ddim-steps 50 \\
    --output runs/baseline/test_results.json \\
    --jsonl runs/baseline/test_predictions.jsonl
```

Expect the benchmark step to take several hours on a single GPU at
`n-samples=100`. During development, add `--limit 500` for a ~10-minute
dry run.
