## 2025-02-20 - Insecure Deserialization in Checkpoint Loading
**Vulnerability:** Found `torch.load` being used with `weights_only=False` when loading `args.checkpoint` and `args.sgno_checkpoint` in `spec2graph/scripts/evaluate.py`. This exposes the application to Remote Code Execution (RCE) via insecure deserialization using Python's `pickle` module if a malicious checkpoint file is loaded.
**Learning:** PyTorch models and state dictionaries often load standard checkpoints that only contain dicts and tensors. Setting `weights_only=False` explicitly bypasses the safer, restricted unpickler. It's common in ML scripts to mistakenly set `weights_only=False` to bypass PyTorch FutureWarnings without realizing the security implications.
**Prevention:** Always enforce `weights_only=True` in `torch.load` calls when loading untrusted checkpoint models, as standard model weights do not require arbitrary object deserialization.

## 2025-02-21 - Denial of Service via Zero-Atom Molecules
**Vulnerability:** Found `rdkit.Chem.MolFromSmiles("")` parsing empty strings into valid `Mol` objects with 0 atoms. Downstream logic assumed molecules had >0 atoms, which could lead to exceptions and crashes.
**Learning:** RDKit considers an empty SMILES valid but atomless. ML datasets or APIs might occasionally yield empty SMILES leading to unexpected behavior.
**Prevention:** Explicitly validate `mol.GetNumAtoms() > 0` after parsing SMILES strings using `Chem.MolFromSmiles()` to prevent processing zero-atom molecules.

## 2026-04-29 - Path Traversal in Cache Key Processing
**Vulnerability:** Found `inchikey` inputs from external datasets used directly to build cache directory and file paths (`inchikey[:2]` and `{inchikey}.npy`) in `spec2graph/data/cache.py` without proper input validation. This allows arbitrary file writes or reads outside the cache directory via path traversal (e.g., `../../etc/passwd`).
**Learning:** Even identifiers sourced from supposedly structured datasets (like TSVs) should not be trusted if they are used in file paths. Malicious modification of the dataset could exploit downstream cache-generation logic.
**Prevention:** Always validate external identifiers used in file paths against a strict allowed-character format (e.g., via regex `^[A-Z0-9\-]+$`) before appending them to file paths.

## 2025-02-22 - Denial of Service via Memory Exhaustion in String Parsing
**Vulnerability:** Found `ast.literal_eval` parsing arbitrarily long stringified lists of mass spectrum peaks (`mzs` and `intensities`) from the MassSpecGym dataset in `spec2graph/data/massspecgym.py`. Passing an extremely large string could lead to CPU/Memory exhaustion and cause a Denial of Service (DoS) attack.
**Learning:** While `ast.literal_eval` safely prevents Remote Code Execution (RCE) compared to `eval`, it does not inherently protect against memory or CPU exhaustion if the payload size is unbounded. External data sources (like public datasets) can contain maliciously large payloads to exploit these parsers.
**Prevention:** Always enforce a strict hard length limit check (e.g., `len(value) > 100000`) before parsing any stringified Python literals with `ast.literal_eval` to prevent DoS via memory/CPU exhaustion.

## 2023-10-27 - Fix DoS via unbounded SMILES parsing
**Vulnerability:** SMILES parsing via `Chem.MolFromSmiles(smiles)` lacked a strict length limit in `spec2graph/data/dataset.py` and `spec2graph/eval/metrics.py`, which could allow malicious inputs to trigger excessive CPU/Memory consumption (DoS).
**Learning:** Pathological strings passed to external C++ binding libraries can cause O(N^3) complexity operations. Hard length limits on strings must be checked immediately before external library boundaries.
**Prevention:** Enforce a strict length limit (e.g., checking against `MAX_SMILES_LENGTH` = 2000) on all external inputs before passing them to external dependencies like RDKit.

## 2025-05-24 - DoS Vulnerability in `parse_formula` via Regex and Long String Count Exhaustion
**Vulnerability:** Found `parse_formula` parsing arbitrarily long formula strings (e.g., `C9999999999999999999`) from MassSpecGym TSV dataset. Extremely large string lengths lead to Denial of Service (DoS) attacks through memory/CPU exhaustion when extracting counts and propagating them to lists using python string multiplication operators.
**Learning:** Even though `parse_formula` uses a simple and seemingly safe regular expression (`[A-Z][a-z]?\d*`), parsing excessively long integers from external inputs without limits can lead to `OverflowError` or CPU/memory exhaustion when these values are used in subsequent computations like `ordered.extend([element] * counts[element])`.
**Prevention:** When parsing stringified counts or multipliers from external inputs (like chemical formulas), enforce both a maximum string length and a maximum parsed integer/count limit to prevent memory exhaustion (DoS) during subsequent list expansions.
