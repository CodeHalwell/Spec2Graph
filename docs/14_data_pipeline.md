# 14 — MassSpecGym Data Pipeline

This chapter covers how Spec2Graph consumes real mass-spec data from
[MassSpecGym](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym).
Everything described here lives under `spec2graph/data/` and is additive
— the original synthetic demo (`spectral_diffusion.run_demo`) still works
unchanged.

## Why a dedicated pipeline

The MassSpecGym TSV is ~262 MB, has ~231k rows, and ships with a specific
schema. Rather than hand-roll CSV parsing in each training script we
centralise:

- Download + schema validation (`load_massspecgym_tsv`).
- Filtering (`filter_massspecgym`).
- Per-molecule eigenvector + fingerprint caching (`EigenvectorCache`).
- A Torch `Dataset` that yields dicts (`MassSpecGymDataset`).
- A collator that pads per-batch and produces `TrainingBatch` tensors
  (`make_training_batch_collator`).

## Stage 1: download + schema validation

`spec2graph.data.load_massspecgym_tsv` (`spec2graph/data/massspecgym.py`)
pulls the TSV from Hugging Face Hub (`roman-bushuiev/MassSpecGym`, path
`data/MassSpecGym.tsv`). On first use it caches under the user's HF
cache directory; subsequent calls are lazy.

After loading, the function verifies that every column in
`REQUIRED_COLUMNS` is present:

```
mzs, intensities, smiles, inchikey, formula,
precursor_mz, adduct, instrument_type
```

It also detects the train/val/test split column, preferring `fold` and
falling back to any low-cardinality string column that contains the
three split labels. The detected column is *copied* into a canonical
`split` column (not renamed), so downstream code can rely on `split`
while still inspecting the original.

`load_massspecgym_tsv(local_path=...)` reads from a local path instead
of the network, which is how the test suite works without downloads.

## Stage 2: filtering

`filter_massspecgym(df, ...)` applies the Phase-0 MVP filters described
in `ROADMAP.md`:

| Filter | Default | Rationale |
|---|---|---|
| Adduct | `[M+H]+` | MVP scope; extend per-run for larger evaluations. |
| Precursor m/z | `<= 1000` | Drug-like upper bound. |
| Min peaks | `>= 5` | Meaningful spectrum. |
| Max SMILES length | `<= 200` | Well under `MAX_SMILES_LENGTH = 2000`. |

The function returns a new dataframe plus a `FilterStats` dataclass
tracking drops per filter — useful for sanity-checking a dataset run
didn't shed 90% of rows silently.

Edge-case handling:

- NaNs in `adduct`, `precursor_mz`, `smiles`, `mzs`, or `intensities` are
  dropped up-front so they can't trip the downstream predicates.
- Non-numeric `precursor_mz` is coerced via `pd.to_numeric(errors="coerce")`,
  which folds the bad rows in with the NaN drops.

## Stage 3: eigenvector / fingerprint caching

`EigenvectorCache` (covered in detail in chapter 02) stores results
keyed by InChIKey under `{cache_dir}/eigvecs_k{k}_{bond_weighting}/` and
`{cache_dir}/fingerprints_{n_bits}bit/`. Files are sharded by the first
two characters of the InChIKey to avoid a single 29k-file directory.

The writer uses atomic writes (`os.fdopen(fd, "wb")` → `np.save` →
`os.replace`) to guard against interrupted processes producing torn
cache entries. The reader detects corrupted files and recomputes them
rather than raising.

`precompute_all` parallelises the first-fill across joblib workers.
After it runs, subsequent training-time lookups are O(disk read).

## Stage 4: the `Dataset`

`MassSpecGymDataset(split, cache_dir, ...)` produces one dict per call
to `__getitem__`. The per-example preprocessing exactly matches
MassSpecGym's evaluation protocol:

1. **Drop precursor echo.** Any peak with `m/z >= precursor_mz - 0.5`
   is removed. This is what the upstream benchmark does, and if we
   don't match it we inflate our own scores.
2. **Normalise intensities** so `max == 1.0`. Absolute intensities aren't
   comparable across instruments.
3. **Keep the top-K peaks** by intensity, default 128. Sparse tails
   contribute little and bloat padding budgets.
4. **Re-sort by m/z ascending.** The transformer doesn't care about
   ordering but downstream diagnostic code often does.

Atom-side outputs:

- `eigvecs` — `(n_atoms, k)` float32 via `EigenvectorCache`.
- `atom_types` — `(n_atoms,)` int64 in RDKit canonical atom order,
  matching the eigenvector rows. This is what the atom-type head
  supervises (see chapter 13).
- `atom_symbols` — the same thing as a Python string list, kept for
  the eval-time Hungarian assignment.

Adjacency targets (optional):

- Pass `include_adjacency=True` to also return a heavy-atom adjacency
  matrix per sample. The SGNO trainer uses this; the diffusion trainer
  does not.

The constructor filters the dataframe down to rows whose InChIKey
successfully caches. Any row with an unparseable SMILES, a heavy-atom
count exceeding `max_atoms`, or a cache write failure is silently
dropped — `len(dataset)` reflects only loadable examples.

## Stage 5: the collators

Two collators, both in `spec2graph/data/collate.py`:

- `make_training_batch_collator` — returns a callable suitable for
  `DataLoader(collate_fn=...)`. Output is a `TrainingBatch` on the
  requested device, with `atom_type_targets` padded using the
  `-100` sentinel (`PADDING_INDEX`) so CrossEntropyLoss skips them.
- `make_metadata_collator` — wraps the training collator and additionally
  returns the per-sample `smiles` / `formula` / `inchikey` as Python
  lists. Required by the evaluation loop, which needs the ground truth
  alongside the tensor outputs.

Mask convention: `True = valid`, consistent with the core model. Padded
positions on the atom axis are zero in the eigenvector tensor and
`-100` in the atom-type targets.

## CLI entry point

`python -m spec2graph.scripts.prepare_data --cache-dir ~/.cache/spec2graph`
is a one-off idempotent precompute that exercises the entire pipeline
end-to-end. Rerunning only fills missing entries.

## Failure-mode checklist

If your training run blows up mid-epoch, the most common causes are:

1. **Cache not primed.** Run `prepare_data` with matching `--k` and
   `--bond-weighting`. Mismatched `k` silently rebuilds entries because
   the cache directory path encodes `k`.
2. **A sample with all peaks stripped.** The `min_peaks` filter should
   prevent this, but a high `max_precursor_mz` combined with a
   narrow `--min-peaks` can leave rows with 0 peaks post-precursor-echo
   filtering. The dataset raises `RuntimeError` in that case rather
   than silently padding.
3. **Atom count > `max_atoms`.** The constructor drops these, but if
   you hot-swap the config mid-run the shape mismatch surfaces at
   collate time.

## Known limitations

- **Cospectrality.** Two distinct molecules can share a spectrum. The
  pipeline doesn't deduplicate test examples by `formula + spectrum`,
  so the per-example metrics treat genuine ambiguity as a failure.
- **Adduct scope.** Phase-0 defaults to `[M+H]+`. Extend via
  `--adduct '[M+Na]+'` etc., but remember the filter intersects — you
  must opt in to each adduct you want.
- **No mzML / MSP parsing.** Both formats are out of scope for this
  initial pipeline; the Hugging Face TSV is the only supported source.
