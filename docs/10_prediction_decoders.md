# 10 — Prediction Decoders

Once reverse diffusion yields `V̂_k`, we still need to turn those
eigenvectors into a molecular graph — i.e., an adjacency matrix. Spec2Graph
ships two decoders that handle different stages of that:

1. **`SpectralGraphNeuralOperator` (SGNO)** — neural, differentiable,
   produces continuous bond probabilities.
2. **`ValencyDecoder`** — symbolic, discrete, enforces chemical valency.

You typically use both: SGNO first, then `ValencyDecoder` on its output.

## SpectralGraphNeuralOperator

Defined at `spectral_diffusion.py:1297`.

### Idea

Treat each predicted eigenvector row `E_i ∈ ℝ^k` as **coordinates** for
atom `i` on a continuous manifold. Two atoms connected by a bond should
live "close together" in this coordinate system. A small MLP evaluates a
learnable kernel on every pair:

```
bond_logit(i, j) = MLP([E_i ; E_j])
```

After symmetrizing and zeroing the diagonal, that becomes an adjacency
logit matrix.

### Construction

```python
SpectralGraphNeuralOperator(k=8, hidden_dim=128, num_layers=3)
```

- `k` must match the eigenvector dimension from the diffusion model.
- `hidden_dim` and `num_layers` control the size of the pairwise MLP.

The MLP structure is:

```
Linear(2k, hidden) → ReLU → Linear(hidden, hidden) → ReLU → … → Linear(hidden, 1)
```

### Forward pass (the memory-efficient bit)

A naïve implementation would allocate the full `(B, N, N, 2k)`
concatenation tensor and forward it through the first linear layer. That
uses `O(N² · 2k)` memory per batch item. Instead, the code splits the
first layer's weight in half, applies each half independently to the node
embeddings, and adds them via broadcasting:

```python
first_layer = self.pairwise_mlp[0]
w_i = first_layer.weight[:, :k]
w_j = first_layer.weight[:, k:]

out_i = F.linear(embeddings, w_i)       # (B, N, hidden)
out_j = F.linear(embeddings, w_j)       # (B, N, hidden)
if first_layer.bias is not None:
    out_i = out_i + first_layer.bias
x = out_i.unsqueeze(2) + out_j.unsqueeze(1)   # (B, N, N, hidden)
```

This is mathematically identical to the concat+linear version but never
materializes the `2k` intermediate. For `N ≈ 64` the saving is modest;
for `N ≈ 256` it's substantial.

### Outputs

Three complementary methods:

| Method | Returns |
|--------|---------|
| `forward(E)` | Logits `(B, N, N)`, symmetric, diagonal zeroed. |
| `bond_probabilities(E)` | `sigmoid(logits)`, diagonal zeroed. |
| `decode_to_adjacency(E, threshold=0.5)` | Binary adjacency `(B, N, N)`. |

`_zero_diagonal` (`spectral_diffusion.py:1327`) enforces a strict square-
matrix contract: it raises if you hand it a non-square tensor. This
catches bugs where the upstream code accidentally rectangularized a batch.

### Training the SGNO

The SGNO is **not** trained by `DiffusionTrainer`. If you want it to learn
meaningful adjacency logits you need a separate training loop with a
binary cross-entropy or BCEWithLogitsLoss target from ground-truth
adjacency matrices — typically on pairs `(V_k_target, A_target)`.

The standard recipe is:

```python
sgno = SpectralGraphNeuralOperator(k=k).to(device)
opt = torch.optim.Adam(sgno.parameters(), lr=1e-4)

for batch in dataloader:
    V_k = batch.eigenvectors          # ground-truth V_k for the molecule
    A   = batch.adjacency             # ground-truth adjacency
    logits = sgno(V_k)
    loss = F.binary_cross_entropy_with_logits(logits, A)
    opt.zero_grad(); loss.backward(); opt.step()
```

At inference, you feed it the *predicted* `V̂_k` from the diffusion
sampler.

## ValencyDecoder

Defined at `spectral_diffusion.py:1673`. This is a deterministic
post-processing step that turns continuous bond probabilities into a
chemically valid adjacency matrix.

### Why it exists

Naïvely thresholding `bond_probabilities` at 0.5 can produce atoms with
impossible valency — e.g., a carbon bonded to six other atoms. Real
chemistry has hard constraints:

```python
VALENCY_TABLE = {
    "C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "F": 1, "Cl": 1, "Br": 1,
    "I": 1, "Si": 4, "B": 3, "Se": 2,
}
```

### Algorithm

A greedy "edge-by-edge" decoder:

```python
candidates = [(prob, i, j) for i<j if prob[i,j] >= threshold]
candidates.sort(reverse=True)              # strongest bonds first

remaining = [valency[a] for a in atom_types]
for prob, i, j in candidates:
    feasible = min(remaining[i], remaining[j], max_bond_order)
    if feasible > 0:
        adjacency[i, j] = feasible
        adjacency[j, i] = feasible
        remaining[i] -= feasible
        remaining[j] -= feasible
```

Strongest-first ensures the most confident bonds are placed before the
budget runs out. `max_bond_order=1` only ever places single bonds; increase
it if you want to allow doubles/triples (you'll also need your probabilities
to encode order, not just presence).

### Signature

```python
decoder = ValencyDecoder()
adj = decoder.decode(
    atom_types=["C", "C", "O"],
    bond_probs=np.array([[0.1, 0.9, 0.8],
                         [0.9, 0.1, 0.3],
                         [0.8, 0.3, 0.1]]),
    threshold=0.3,
    max_bond_order=1,
)
```

Or batched:

```python
adjs = decoder.decode_batch(
    atom_types_batch=[["C", "C", "O"], ["N", "N"]],
    bond_probs_batch=probs,                 # (B, N_max, N_max)
    threshold=0.3,
)
```

Note: `decode_batch` expects a list of atom-type lists — one per molecule
— because the atom count can vary. Each result is a plain `numpy` array.

### `threshold` and `max_bond_order`

- Low threshold (e.g., 0.2): more candidate edges, fuller molecules,
  slower.
- High threshold (e.g., 0.7): sparser outputs; atoms may end up
  disconnected.
- `max_bond_order > 1`: allows multi-bond assignment in a single
  iteration. Only useful if your probabilities actually distinguish
  order.

### Unknown elements

`get_valency` (`spectral_diffusion.py:1696`) returns `default_valency=4`
for any element not in `VALENCY_TABLE`. For exotic elements, subclass
or pass a custom `valency_table` at construction.

## Putting SGNO and ValencyDecoder together

```python
# After reverse diffusion
V_hat = trainer.sample(mz, intensity, n_atoms=N, spectrum_mask=spec_mask)

# Neural decoding to continuous probabilities
sgno = SpectralGraphNeuralOperator(k=k).to(device)
bond_probs = sgno.bond_probabilities(V_hat)                         # (B, N, N)

# Symbolic decoding to chemically valid graph
atom_types = [...]  # list of element symbols per atom (from formula prediction, e.g.)
valency_decoder = ValencyDecoder()
adj = valency_decoder.decode(
    atom_types=atom_types,
    bond_probs=bond_probs[0].cpu().numpy(),
    threshold=0.3,
    max_bond_order=1,
)
```

This gives you a valid single-bond adjacency ready to feed back into
RDKit (`Chem.RWMol`) for SMILES canonicalisation.

## Where this stops and where you take over

Neither decoder infers *what* elements the atoms are — you provide
`atom_types` to `ValencyDecoder`. In a full pipeline, the element
assignment typically comes from a formula prediction head or from the
precursor formula if it's known a priori. That's noted in `ROADMAP.md`
under Phase 2.

Next: [11 — Advanced features](./11_advanced_features.md) covers the
optional guidance and eigenvalue-conditioning modules.
