# Spec2Graph Roadmap and Theory

This document captures the practical plan and core theory for Spec2Graph: mapping an MS/MS spectrum to a molecular graph by predicting a spectral embedding (Laplacian eigenvectors/subspace) as an intermediate, then reconstructing the adjacency/bonds from that embedding. It summarizes the “clever but cursed” aspects of spectral methods, the subspace-invariant target choice, and a staged implementation roadmap.

## Why Spec2Graph is both clever and cursed

- **Eigenvectors are not unique:** Each eigenvector can flip sign; repeated eigenvalues allow arbitrary rotations inside an eigenspace.
- **Cospectral graphs exist:** Distinct graphs can share the same spectrum, so spectral info alone is not unique.
- **MS/MS ambiguity:** Isomers can share very similar spectra, so identifiability is limited even before the spectral embedding step.

### Key practical move

Do **not** train on raw eigenvectors. Train on subspace-invariant targets:

\[
P_k = V_k V_k^\top
\]

where \(V_k\) are the first \(k\) eigenvectors. \(P_k\) is invariant to sign flips and rotations within degenerate eigenspaces, making learning far less fragile. Include an orthonormality regulariser or explicit normalisation when predicting embeddings.

## Phase roadmap

**Phase 0 — MVP scope**
- Start with ESI(+), \[M+H\]^+, high-res MS/MS, ≤ 500 Da.
- Assume formula + adduct are known for the MVP.

**Phase 1 — Data & preprocessing**
- Parse MSP/mzML → list of (m/z, intensity) peaks; normalise, denoise, keep top-K peaks.
- Build a “peak graph”: nodes = peaks; edge features = Δm/z (k-NN in m/z or windowed).
- Targets: SMILES → molecular graph via RDKit; compute Laplacian \(L\) (choose normalised/unnormalised consistently); compute \(V_k\); store \(P_k = V_k V_k^\top\) (and optionally eigenvalues).

**Phase 2 — Handle variable atoms**
- Condition on formula to fix heavy-atom count \(N\) and allowed atom types. Later add a formula/adduct predictor head.

**Phase 3 — Model 1: spectrum encoder → spectral subspace**
- Encoder options: peak-graph GNN, Set Transformer, or Transformer over peaks (+ Δm/z bias).
- Decoder: produce node embedding matrix \(\hat{E} \in \mathbb{R}^{N \times k}\).
- Losses: subspace loss \(\|\hat{P}_k - P_k\|_F^2\); orthonormality penalty; optionally predict eigenvalues.

**Phase 4 — Model 2: embeddings → bonds**
- Link prediction over atom pairs with chemistry constraints.
- Enforce symmetry and valence (via constrained decoding, ILP/flow, or a repair stage).

**Phase 5 — Candidate generation + ranking**
- Sample diverse graphs (stochastic edges, beam search; later diffusion).
- Rank by model scores, formula/mass filters, optional fingerprint head.

**Phase 6 — Round-trip verification**
- Validate candidates by predicting MS/MS or fragments and rerank via similarity metrics.

**Phase 7 — Conditional diffusion over graphs (optional)**
- Discrete diffusion (DiGress-style) over bond types conditioned on spectrum latent and/or node embeddings.

**Phase 8 — Evaluation**
- Use scaffold/molecule-level splits.
- Report Top-1/Top-10 exact match, best-of-K fingerprint Tanimoto, validity, and “abstain accuracy”.
- Baselines: library-search style and ablations (no peak-graph edges, no formula conditioning, no round-trip, etc.).

## MVP ladder

1. **MVP A:** Known formula + peak-graph encoder → link predictor → valid graph.
2. **MVP B:** Add subspace loss via \(P_k\) (Spec2Graph core) and show robustness gains.
3. **MVP C:** Add candidate sampling + reranking + confidence/abstention.
4. **MVP D:** Add formula/adduct prediction head (multi-task).
5. **MVP E:** Conditional diffusion for diversity/dark space.

## Risks and mitigations

- **Eigenvector degeneracy / sign flips:** Train on \(P_k\); use orthonormalisation.
- **Atom permutation:** Use canonical atom ordering or permutation-invariant decoding with symmetric edge prediction.
- **Cospectrality / MS ambiguity:** Embrace top-K outputs + reranking + abstention.
- **Data mismatch:** Pretrain on synthetic, finetune on real; augment with peak dropout/intensity jitter.
