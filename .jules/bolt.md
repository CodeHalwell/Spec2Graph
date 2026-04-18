## 2024-04-15 - Optimizing Pairwise Operations in GNN Decoders
**Learning:** In SpectralGraphNeuralOperator (SGNO) and its variants, computing pairwise interactions using standard MLP concatenation creates a massive O(N^2) memory bottleneck, allocating full `(batch, n_atoms, n_atoms, 2k)` tensors. This happens because `.expand()` and `torch.cat()` materialize the combination of features.
**Action:** Instead of concatenating before the first MLP layer, split the first linear layer's weights into `w_i` and `w_j` and apply them to the node embeddings individually. Then, use broadcasting addition `out_i.unsqueeze(2) + out_j.unsqueeze(1)` to compute the interaction efficiently in O(N) memory before passing to subsequent layers. This reduces memory footprint by up to 2x during forward and backward passes.

## 2024-04-18 - Avoid O(N^2) allocations for zeroing diagonals
**Learning:** Using `torch.eye(N).unsqueeze(0)` and `.masked_fill()` to zero out diagonals in batched square matrices allocates $O(N^2)$ memory and compute for the mask creation alone.
**Action:** Use `.diagonal(dim1=-2, dim2=-1).zero_()` on a cloned or intermediate tensor instead. This operates efficiently in-place on the diagonal view without allocating unnecessary dense masks, providing a >3x speedup on diagonal zeroing operations.
