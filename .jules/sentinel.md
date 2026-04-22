## 2024-05-18 - Missing Length Bounds on RDKit Parsing and Transformer Inputs
**Vulnerability:** External string inputs (SMILES) passed to RDKit's parser and tensor peak arrays passed to the Spec2Graph Transformer did not have upper bounds, leading to potential Denial of Service (DoS) attacks via memory exhaustion. RDKit and O(N^2) Transformer mechanisms are vulnerable to extreme inputs.
**Learning:** In machine learning processing pipelines, memory-intensive modules (like molecular parsing and self-attention) must implement hard size limits on inputs before computation, even if downstream layers process it.
**Prevention:** Always implement an initial size check (e.g., `len(smiles) <= MAX_LEN` or `mz.shape[1] <= max_peaks`) at the very edge of the API or forward pass method to drop excessively large payloads immediately.
## 2026-04-22 - Fix insecure weights loading vulnerability

**Vulnerability:** Insecure deserialization via `torch.load` with `weights_only=False` in `spec2graph/scripts/evaluate.py`.
**Learning:** PyTorch's `torch.load` uses Python's `pickle` module by default, which can execute arbitrary code if an attacker provides a malicious checkpoint file. Setting `weights_only=True` is the required defense against this.
**Prevention:** Always use `weights_only=True` when calling `torch.load` to load PyTorch checkpoints from untrusted sources, restricting deserialization strictly to tensors, primitive types, and dictionaries.
