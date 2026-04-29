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
