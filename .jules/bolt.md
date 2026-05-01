## 2024-05-24 - Avoiding O(N^2) allocations for diagonal masking
**Learning:** In PyTorch, using `masked_fill` with an identity matrix (like `torch.eye(N)`) to zero out the diagonal of a batched square matrix incurs a significant O(N^2) memory allocation and computational overhead, especially when called repeatedly in the forward pass of O(N^2) graph neural networks.
**Action:** Always prefer using `.diagonal(dim1=-2, dim2=-1).zero_()` to zero out diagonals in place. If the original tensor must be preserved, use `.clone()` first. This avoids materializing the O(N^2) mask tensor entirely.

## 2024-05-25 - Avoid dynamic tensor allocation in forward pass
**Learning:** In PyTorch, allocating tensors dynamically during the forward pass (e.g., using `torch.arange(N, device=device)`) inside a loop (such as a diffusion model's sampling loop) causes device synchronization and memory allocation overhead.
**Action:** When a tensor's values only depend on a known maximum length, pre-allocate it in the module's `__init__` as a non-persistent buffer (`self.register_buffer('name', tensor, persistent=False)`). During the forward pass, simply slice the buffer dynamically to the required length (e.g., `self.name[:N]`). The `persistent=False` flag guarantees backward compatibility with existing checkpoints.
## 2024-05-26 - Precompute ground truth properties in evaluation metrics
**Learning:** In evaluation metric loops (e.g., `top_k_tanimoto` or similar), repeatedly parsing the ground truth SMILES and generating its Morgan fingerprint inside the loop for every top-k prediction is an O(N) performance bottleneck.
**Action:** Always precompute properties, like fingerprints or `RDKit` molecule objects, for the ground truth molecule outside the prediction loop to eliminate redundant parsing and generation overhead.

## 2024-05-26 - Avoid negligible micro-optimizations on small tensors
**Learning:** Replacing a one-off `torch.arange(N, device=device)` call with a slice of a pre-allocated tensor buffer `positions[:N]` when `N` is small and the call happens only once per forward pass (not deep in a heavy loop) is a negligible micro-optimization. Furthermore, blindly slicing pre-allocated positional buffers is unsafe due to potential shape and semantic mismatches.
**Action:** Do not implement micro-optimizations that offer no measurable impact or require unsafe assumptions about model properties. Focus on $O(N)$ vs $O(N^2)$ structural bottlenecks or redundant computations.
## 2025-02-20 - [GNN Adjacency Normalization Optimization]
**Learning:** In PyTorch GNNs, when aggregating features using a dense normalized adjacency matrix (e.g., `D^{-1/2} A D^{-1/2} X`), computing `adj_norm = adj * D_inv_sqrt * D_inv_sqrt.transpose()` forces an O(N^2) memory allocation and broadcasting overhead.
**Action:** Use the associative property of matrix multiplication to rewrite it as `D^{-1/2} (A (D^{-1/2} X))`. First scale features (`x_scaled = x * D_inv_sqrt`), perform message passing `torch.bmm(adj, x_scaled)`, and finally scale the output `agg = agg * D_inv_sqrt`. This avoids the O(N^2) dense matrix and saves significant time and memory.
