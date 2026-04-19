## 2024-05-24 - Avoiding O(N^2) allocations for diagonal masking
**Learning:** In PyTorch, using `masked_fill` with an identity matrix (like `torch.eye(N)`) to zero out the diagonal of a batched square matrix incurs a significant O(N^2) memory allocation and computational overhead, especially when called repeatedly in the forward pass of O(N^2) graph neural networks.
**Action:** Always prefer using `.diagonal(dim1=-2, dim2=-1).zero_()` to zero out diagonals in place. If the original tensor must be preserved, use `.clone()` first. This avoids materializing the O(N^2) mask tensor entirely.
