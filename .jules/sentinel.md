## 2024-05-18 - Missing Length Bounds on RDKit Parsing and Transformer Inputs
**Vulnerability:** External string inputs (SMILES) passed to RDKit's parser and tensor peak arrays passed to the Spec2Graph Transformer did not have upper bounds, leading to potential Denial of Service (DoS) attacks via memory exhaustion. RDKit and O(N^2) Transformer mechanisms are vulnerable to extreme inputs.
**Learning:** In machine learning processing pipelines, memory-intensive modules (like molecular parsing and self-attention) must implement hard size limits on inputs before computation, even if downstream layers process it.
**Prevention:** Always implement an initial size check (e.g., `len(smiles) <= MAX_LEN` or `mz.shape[1] <= max_peaks`) at the very edge of the API or forward pass method to drop excessively large payloads immediately.

## 2024-05-18 - RDKit Silent Acceptance of Empty SMILES
**Vulnerability:** Empty strings `""` passed to `Chem.MolFromSmiles()` are parsed as valid `Mol` objects but possess exactly 0 atoms, which bypassing `mol is None` checks and causes downstream errors or unhandled behaviors during representation generation (like fingerprints).
**Learning:** Checking if an RDKit object `is None` is insufficient to validate external SMILES input, as technically 0-atom valid objects can be constructed.
**Prevention:** Always validate `mol.GetNumAtoms() > 0` immediately after instantiating an RDKit `Mol` from an untrusted source, even if parsing succeeded without error.
