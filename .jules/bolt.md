## 2024-04-15 - Optimizing Pairwise Operations in GNN Decoders
**Learning:** In SpectralGraphNeuralOperator (SGNO) and its variants, computing pairwise interactions using standard MLP concatenation creates a massive O(N^2) memory bottleneck, allocating full `(batch, n_atoms, n_atoms, 2k)` tensors. This happens because `.expand()` and `torch.cat()` materialize the combination of features.
**Action:** Instead of concatenating before the first MLP layer, split the first linear layer's weights into `w_i` and `w_j` and apply them to the node embeddings individually. Then, use broadcasting addition `out_i.unsqueeze(2) + out_j.unsqueeze(1)` to compute the interaction efficiently in O(N) memory before passing to subsequent layers. This reduces memory footprint by up to 2x during forward and backward passes.
## 2024-05-24 - PyTorch Persistent Buffers
**Learning:** When pre-computing constants in PyTorch modules (like frequency embeddings) and using `register_buffer`, always use `persistent=False` if the buffer is purely derived from config parameters and isn't a learned weight. Otherwise, loading older checkpoints with `strict=True` will fail due to unexpected keys.
**Action:** Use `self.register_buffer("name", tensor, persistent=False)` for derived constants that shouldn't be serialized in checkpoints.

## 2024-05-24 - PyTorch In-Place Diagonal Zeroing
**Learning:** Zeroing a matrix diagonal by generating a `torch.eye()` mask and applying `masked_fill` is computationally expensive because it creates intermediate tensors and iterates over the mask. PyTorch allows zeroing the diagonal directly using `.diagonal(dim1=-2, dim2=-1).zero_()`. To preserve autograd graphs without breaking gradient propagation, ensure this is done on a cloned or mathematically intermediate tensor.
**Action:** Replace `matrix.masked_fill(torch.eye(n).unsqueeze(0), 0.0)` with a direct, in-place diagonal zeroing on a cloned tensor: `res = matrix.clone(); res.diagonal(dim1=-2, dim2=-1).zero_()`.
