## 2024-05-24 - Missing Input Length Limits
**Vulnerability:** The application did not restrict the length of SMILES strings parsed by RDKit or the number of peaks (`mz` arrays) passed into O(N^2) complexity Transformer modules.
**Learning:** External inputs like strings passed to parsers and large arrays given to O(N^2) complexity modules can lead to memory exhaustion and cause a Denial of Service (DoS).
**Prevention:** Always enforce hard length limits on input strings directly tied to parsers or memory-heavy components like Transformers.
