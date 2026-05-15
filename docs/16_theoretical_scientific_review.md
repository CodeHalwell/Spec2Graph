# 16 — Theoretical and Scientific Review

## Review scope and criteria

This review evaluates Spec2Graph from four scientific angles:

1. **Problem identifiability**: whether the spectrum-to-graph inverse problem is well-posed enough for learning.
2. **Mathematical consistency**: whether the objective and representation choices align with graph spectral theory.
3. **Experimental rigor**: whether the current implementation/documentation supports reproducible, falsifiable claims.
4. **Modeling risk profile**: expected failure modes, and whether mitigations are principled.

The review is constrained to what is currently implemented and documented in this repository.

## Executive assessment

Spec2Graph is **theoretically motivated and directionally strong** in its core choice to train on a subspace-invariant objective (`P_k = V_kV_k^T`) instead of raw eigenvectors. This directly addresses sign ambiguity and eigenspace rotation degeneracy, which are central spectral-learning pitfalls.

Scientifically, the project is currently best characterized as a **well-constructed research prototype** rather than a validated method. The main reason is not architectural weakness; it is the current absence (in docs and tests) of a complete empirical protocol that can disentangle gains from representation choices vs. decoder heuristics, data assumptions, and constraints.

## 1) Theoretical strengths

### 1.1 Correct treatment of eigenvector non-identifiability

The repository explicitly acknowledges sign/rotation ambiguity and uses projection-space supervision as a robust surrogate target. This is the right invariance class for spectral embeddings and is consistent with linear subspace geometry (Grassmannian perspective).

**Why this matters:** direct L2 loss on eigenvectors is not stable under equivalent bases; projection loss is basis-invariant and therefore much better aligned with the true object of interest.

### 1.2 Sound Laplacian construction and numerical guards

The normalized Laplacian implementation uses stable degree handling and avoids unnecessary dense operations where possible. This is mathematically consistent for weighted undirected molecular graphs.

### 1.3 Acknowledgment of fundamental inverse-problem ambiguity

The docs explicitly call out cospectrality and MS ambiguity. This is scientifically important because it sets realistic expectations: the task should be framed as candidate generation/ranking under uncertainty, not deterministic exact recovery.

### 1.4 Useful decomposition of pipeline stages

Separating spectrum encoder, subspace prediction, and graph decoding is beneficial for ablations and diagnosis. It enables testing whether errors originate from conditioning signal quality, diffusion denoising, or graph reconstruction.

## 2) Theoretical limitations and open risks

### 2.1 Spectral truncation is information-losing

Using top-k eigenvectors can miss local structural distinctions important for chemistry, especially if `k` is small or atom counts vary significantly. This is a known tradeoff between robustness and expressiveness.

### 2.2 Projection loss does not ensure chemically valid graphs

Subspace closeness is not equivalent to graph validity, nor to exact adjacency recovery. Different graphs can produce similar low-rank spectral subspaces; downstream decoding and constraints do substantial work.

### 2.3 Potential mismatch between training objective and final utility

If evaluation emphasizes exact graph match, optimizing projection similarity may not correlate tightly enough unless the decoder is jointly optimized for chemical validity and discriminative adjacency recovery.

### 2.4 Dependency on atom ordering/size conventions

Even with symmetric adjacency decoding, practical training can still be sensitive to atom ordering and padding conventions, especially in batched setups with variable `N`.

## 3) Experimental rigor: current state vs. recommended standard

### What is already good

- The docs define a phase-based roadmap and known risks.
- The codebase includes tests across dataset, metrics, decoding, diffusion variants, and benchmarking scaffolding.

### What is still needed for strong scientific claims

1. **Primary benchmark protocol lock-in**
   - Fixed splits (scaffold and random), fixed seeds, and explicit leakage checks.
2. **Ablation matrix**
   - Raw-eigenvector loss vs projection loss.
   - With/without orthonormality regularization.
   - With/without auxiliary heads.
   - Alternative decoders under equal compute budget.
3. **Uncertainty-aware metrics**
   - Top-k candidate quality, calibration, abstention coverage/accuracy.
4. **Data realism checks**
   - Instrument domain shift, adduct diversity, and collision-energy variability.
5. **Statistical reporting**
   - Mean ± CI across seeds; significance-aware comparisons for key endpoints.

Without this layer, conclusions remain plausible but not yet decisively demonstrated.

## 4) Reproducibility and reliability review

### Positives

- Clear component structure and documentation map.
- Runtime validation for masks and selected numerical safety checks.

### Gaps to address for publication-grade reproducibility

- Add an explicit experiment manifest (dataset version hashes, split files, seeds, hyperparameters).
- Add one-command reproduction for core tables/figures.
- Archive generated artifacts (metrics JSON, predictions, logs) with schema checks.

## 5) Security and robustness considerations (scientific operations)

The implementation includes practical hardening such as SMILES length caps and graceful RDKit dependency checks. For training/evaluation pipelines, the largest residual risk is supply-chain and data provenance (dependency pinning, source integrity, and preprocessing determinism).

## 6) Priority recommendations

### High priority (next iteration)

1. **Finalize benchmark protocol** and freeze reference splits + seeds.
2. **Run mandatory ablations** around subspace invariance claims.
3. **Report candidate-set metrics** (Top-1/Top-10, validity, calibration).
4. **Publish reproducibility bundle** (commands + artifacts + checksums).

### Medium priority

1. Add spectral+chemical hybrid losses that more directly align with final graph quality.
2. Investigate permutation-invariant or canonicalization-aware decoding strategies.
3. Quantify scaling behavior vs atom count and spectrum length.

## Bottom line

Spec2Graph has a **credible theoretical foundation** centered on the correct treatment of spectral non-identifiability and a modular architecture suitable for rigorous research. Its main limitation is not conceptual novelty but **evidence maturity**: the project needs a locked experimental protocol and ablation-heavy validation to substantiate scientific claims about superiority and robustness.
