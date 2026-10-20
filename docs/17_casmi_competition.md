# 17 — Enveda CASMI 2026 competition layer

`spec2graph/casmi/` adapts Spec2Graph to the Kaggle competition
[Enveda CASMI 2026 – Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
without changing the core model. This chapter explains what the competition
asks for, why the spectral-subspace model is used as a *scoring function*
rather than a generator, and how to run the pipeline end to end.

## The brief

| | |
|---|---|
| Input | LC-MS/MS spectra (timsTOF), 1–16 per `molecule_id`, with precursor m/z, adduct and collision energy. No molecular formula. |
| Output | Up to 25 SMILES per molecule, best first, in `submission.csv`. |
| Metric | MRR@25. A guess is correct when its tautomer-canonical InChIKey14 equals the answer's (RDKit pinned at 2026.03.3). |
| Rules | Offline notebook, 9 h, public pretrained models allowed, data CC BY-NC 4.0 (no redistribution). |
| Dates | Entry/merger 7 Dec 2026, final submissions 14 Dec 2026. |

Two facts shape the design. Almost every answer exists in some structure
database (training structures, COCONUT, PubChem), so retrieval and ranking
dominate; and a de novo exact match is rare for every published model, so
a generator alone would score near zero. The layer therefore uses the model
to **score mass-window candidates**, and only generates de novo structures
for formulas no pool contains.

## Mapping Spec2Graph onto the task

```
spectra of one molecule ──► CasmiModel.encode ──► conditioned memory
                                                      │
          ┌──────────── heads ───────────────┬────────┴───────── sampling ─────────┐
          ▼                                  ▼                                     ▼
 fingerprint probabilities        spectral quantiles              eigenvector draws V̂  →  P̂ = mean(QQᵀ)
          │                                  │                                     │  + atom-type log-probs
          ▼                                  ▼                                     ▼
   fp_loglik, soft Tanimoto         Wasserstein distance            Sinkhorn alignment to each candidate
          └──────────────────────────────────┴───────────────── features ─────────┘
                                             ▼
                        LinearRanker (listwise fit on held-out molecules) → top 25
```

* **Conditioning** (`conditioning.py`). `CasmiModel` wraps the core
  `Spec2GraphDiffusion`: three condition tokens (adduct + polarity,
  collision energy with a learned "missing" vector, neutral mass) are
  prepended to the encoded peaks and passed through the core's existing
  `memory` argument. Every core head is reused unchanged; one new head
  predicts the Laplacian **spectral quantiles**. `CasmiTrainer` subclasses
  `DiffusionTrainer`, sharing the DDPM schedule and helpers, and overrides
  only the loss assembly and the samplers. `CasmiTrainerConfig.heads_only`
  is a **proxy mode**: it trains the encoder and the fingerprint,
  atom-count and quantile heads without the denoiser, so the conditioning
  and data pipeline can be checked in minutes before a diffusion run.
* **Spectral quantiles** (`spectral.py`). The eigenvalue spectrum of the
  normalised Laplacian resampled at fixed quantile levels: permutation- and
  size-invariant, so it can be compared with any candidate without an atom
  correspondence. Bipartite (odd-ring-free) graphs have λ_max = 2, pendant
  groups raise the multiplicity of λ = 1, so it carries real topology.
* **Projection overlap** (`scoring.py`, class `Scorer`). Because the score
  is linear in `P`, averaging per-sample projections gives a legitimate
  expected projection `P̂`. Candidates are compared through
  `max_Π <Πᵀ P̂ Π, P_c>` with `Π` found by Sinkhorn ascent seeded by the
  atom-type head and rounded to a permutation with the Hungarian algorithm.
  The same alignment yields an element log-likelihood, a learned
  composition check that needs no formula. This is the one signal on the
  board that nobody else has; whether it is complementary to fingerprints
  is the experiment.
* **Candidate pool** (`pool.py`). Mass-sorted structures with precomputed
  `V_k`, elements, quantiles and packed fingerprints; `searchsorted`
  retrieval in a ppm window; saved without pickling.
* **Ranker and prediction** (`ranker.py`, `predict.py`). `LinearRanker`:
  standardised features, mean imputation, a linear score trained listwise
  (softmax over each molecule's candidates) on held-out molecules, with
  heuristic weights before any fit. `Predictor` runs score, rank, de novo
  fill and mass-nearest top-up per molecule inside a try/except and writes
  the `Submission`; `Predictor.load` builds it from files.
* **De novo** (`denovo.py`). `BondTypeSGNO` predicts four bond classes per
  atom pair (the core SGNO is binary), `decode` enforces valence and the
  formula's hydrogen budget, `enumerate_formulas` proposes CHNOPS(+halogen)
  formulas within a ppm window, and `DeNovo` samples, assigns elements with
  the core Hungarian step, rejects disconnected graphs and ranks by
  frequency. Hooked into prediction behind `--bond-sgno`.
* **Metric** (`metric.py`, `chem.py`). Tautomer canonicalisation +
  InChIKey14, MRR@25, and a `Submission` class that enforces the rejection
  rules. Standard InChI puts (de)protonation in the `/p` layer, so an acid
  and its anion share a key. `chem.py` holds the shared RDKit helpers
  (length-guarded parsing, structure records, kekulised adjacency).

## Validation

`validation.py` splits by structure (InChIKey14) and simulates the hidden
test set's novelty classes on the held-out molecules (`validate`):

| Scenario | Meaning | Who can score |
|---|---|---|
| `known_structure` | answer present in the pool (competition classes 1 and 2) | ranker |
| `novel_structure` | answer removed from the pool (class 3) | de novo only |

Candidate recall is reported alongside MRR: a structure the window never
proposes can never be ranked. Teams found that chemistry-matched holdouts
(natural-product sources) predicted the leaderboard better than
instrument-matched ones; choose the holdout accordingly.

## Commands

```bash
# 1. Assets: adduct vocabulary, structure table, candidate pool, caches.
python -m spec2graph.scripts.casmi_prepare \
    --train-parquet data/train.parquet --cache-dir ~/.cache/spec2graph-casmi \
    --output-dir assets/casmi --extra-structures external/coconut.csv --n-jobs 8

# 2. Train (molecule-disjoint holdout written to runs/casmi/holdout_keys.json).
python -m spec2graph.scripts.casmi_train \
    --train-parquet data/train.parquet --assets-dir assets/casmi \
    --cache-dir ~/.cache/spec2graph-casmi --output-dir runs/casmi \
    --epochs 10 --batch-size 128 --train-bond-sgno

# 3. Validate under both scenarios and fit the ranker on half the holdout.
python -m spec2graph.scripts.casmi_validate \
    --checkpoint runs/casmi/epoch_010.pt --pool assets/casmi/pool.npz \
    --train-parquet data/train.parquet --holdout-keys runs/casmi/holdout_keys.json \
    --fit-ranker runs/casmi/ranker.json --bond-sgno runs/casmi/bond_sgno_epoch_010.pt \
    --output runs/casmi/validation.json

# 4. Submission (this is the offline Kaggle notebook body).
python -m spec2graph.scripts.casmi_predict \
    --checkpoint runs/casmi/epoch_010.pt --pool assets/casmi/pool.npz \
    --ranker runs/casmi/ranker.json --bond-sgno runs/casmi/bond_sgno_epoch_010.pt \
    --test-parquet data/test.parquet --output submission.csv
```

All four are idempotent and runnable on CPU for development (`--max-train-rows`,
`--limit`, `--n-steps`, `--heads-only`). Cognitive complexity is kept under
15 per function: `complexipy spec2graph/casmi`.

## Data handling

The competition data is CC BY-NC 4.0 with explicit permission for academic
research, and the rules forbid redistribution. Keep `train.parquet`,
`test.parquet`, the caches and the pool out of version control (see
`.gitignore`); upload weights, pool and ranker to Kaggle as a private
dataset for the notebook. Weights trained on the data inherit the
non-commercial restriction.

## Notes and known limitations

* **Rank-safe projector.** The core projection loss uses `torch.linalg.qr`,
  whose backward pass is NaN when the eigenvector matrix is rank deficient,
  i.e. for any molecule with fewer heavy atoms than `k`. `CasmiTrainer`
  overrides `projection_from_embeddings` with `V (VᵀV + εI)⁻¹ Vᵀ`, identical
  up to `1/(1+ε)` at full rank and differentiable always. The core is
  unchanged; see `tests/test_casmi_model.py::test_rank_safe_projector`.
* **Multiple spectra per molecule** are aggregated by averaging head outputs
  and pooling eigenvector draws across spectra. A joint set encoder over the
  collision-energy series is the natural next step.
* **No spectral-library search** yet. Classes 1 and 2 therefore coincide in
  validation. A cosine/entropy similarity to training spectra would be a
  cheap extra ranker feature.
* **De novo payoff is bounded** by the hidden class-3 fraction; it is where
  the project is different, not where it earns most points.
* **RDKit version.** Pin `rdkit==2026.3.3` for final local validation so the
  tautomer canonicalisation matches the metric notebook.
