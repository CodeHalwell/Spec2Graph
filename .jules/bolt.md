## 2024-05-24 - Avoiding O(N^2) allocations for diagonal masking
**Learning:** In PyTorch, using `masked_fill` with an identity matrix (like `torch.eye(N)`) to zero out the diagonal of a batched square matrix incurs a significant O(N^2) memory allocation and computational overhead, especially when called repeatedly in the forward pass of O(N^2) graph neural networks.
**Action:** Always prefer using `.diagonal(dim1=-2, dim2=-1).zero_()` to zero out diagonals in place. If the original tensor must be preserved, use `.clone()` first. This avoids materializing the O(N^2) mask tensor entirely.

## 2024-05-25 - Avoid dynamic tensor allocation in forward pass
**Learning:** In PyTorch, allocating tensors dynamically during the forward pass (e.g., using `torch.arange(N, device=device)`) inside a loop (such as a diffusion model's sampling loop) causes device synchronization and memory allocation overhead.
**Action:** When a tensor's values only depend on a known maximum length, pre-allocate it in the module's `__init__` as a non-persistent buffer (`self.register_buffer('name', tensor, persistent=False)`). During the forward pass, simply slice the buffer dynamically to the required length (e.g., `self.name[:N]`). The `persistent=False` flag guarantees backward compatibility with existing checkpoints.
