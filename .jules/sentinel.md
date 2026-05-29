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
## 2024-05-23 - Formula Parsing DoS via Missing Input Length and Bounds Validation
**Vulnerability:** The `parse_formula` function in `spec2graph/eval/decode.py` accepted unbound chemical formula strings and lacked protections against parsing exceedingly large total atom counts (e.g. "C1000000000"). This could trigger OOM failures (memory exhaustion via `formula_to_atom_types` which attempts to list expansion O(N)) when parsing a small malicious input, leading to a Denial of Service.
**Learning:** Functions that parse mathematical or chemical syntax representations (like formulas) and subsequently allocate downstream resources scaled to the extracted components must defend against excessive input sizes and unreasonable magnitude parsing. Just because a string describes a valid syntax does not mean the system can safely construct the described state.
**Prevention:** Defend against DoS memory allocation attacks by explicitly capping the string length of chemical formulas and the maximum supported parsed values (e.g., maximum supported heavy atoms).

## 2025-02-23 - Fix DoS via unbounded SMILES parsing in metrics evaluation
**Vulnerability:** SMILES parsing via `Chem.MolFromSmiles(gt_smiles)` lacked a strict length limit check in the `top_k_mces` metric function within `spec2graph/eval/metrics.py`. This omission could allow malicious inputs to trigger excessive CPU/Memory consumption (DoS) when constructing the ground truth `Mol` object.
**Learning:** Functions that parse molecular representations like SMILES into heavy downstream structures (like `Mol` objects in RDKit) must strictly enforce input length limits to defend against DoS, even inside evaluation or metric functions where inputs might be assumed safe.
**Prevention:** Always enforce a strict hard length limit check (e.g., `len(smiles) > MAX_SMILES_LENGTH`) before calling parsing functions like `Chem.MolFromSmiles` anywhere in the codebase.
