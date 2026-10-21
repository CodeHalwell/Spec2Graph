"""RDKit helpers shared by the competition layer: one place for parsing guards and quiet logging."""

from __future__ import annotations

import functools
from typing import Callable, Iterable, Optional, TypeVar

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm.auto import tqdm

from spec2graph.eval.decode import adjacency_to_rwmol
from spectral_diffusion import MAX_SMILES_LENGTH

_TAUTOMERS = rdMolStandardize.TautomerEnumerator()
T, R = TypeVar("T"), TypeVar("R")


def mol(smiles: object, canonical_tautomer: bool = False) -> Optional[Chem.Mol]:
    """Parse a SMILES under the project's length guard; ``None`` when invalid or empty."""
    if not isinstance(smiles, str) or not 0 < len(smiles.strip()) <= MAX_SMILES_LENGTH:
        return None
    with rdBase.BlockLogs():
        m = Chem.MolFromSmiles(smiles.strip())
        if m is None or m.GetNumAtoms() == 0:
            return None
        if canonical_tautomer:
            try:
                m = _TAUTOMERS.Canonicalize(m)
            except Exception:  # RDKit native error: keep the input tautomer
                pass
    return m


@functools.lru_cache(maxsize=1 << 16)
def inchikey14(smiles: Optional[str]) -> Optional[str]:
    """The metric notebook's ``canonicalize_smiles_to_ik14``: stereo stripped, then tautomer-canonical InChIKey first block.

    Stripping first matters: a stereocentre can pin a double bond, so the same skeleton drawn with and without it
    would otherwise canonicalise to different tautomers and different keys.
    """
    m = mol(smiles)
    if m is None:
        return None
    try:
        with rdBase.BlockLogs():
            Chem.RemoveStereochemistry(m)
            key = Chem.MolToInchiKey(_TAUTOMERS.Canonicalize(m))
    except (ValueError, RuntimeError, TypeError):
        return None
    return key[:14] if len(key) >= 14 else None


def record(smiles: object, canonical_tautomer: bool = False) -> Optional[dict]:
    """Identity and bulk properties of one structure, or ``None``."""
    m = mol(smiles, canonical_tautomer)
    if m is None or (key14 := inchikey14(canonical := Chem.MolToSmiles(m))) is None:
        return None
    with rdBase.BlockLogs():
        return dict(smiles=canonical, inchikey=Chem.MolToInchiKey(m), inchikey14=key14,
                    formula=rdMolDescriptors.CalcMolFormula(m), exact_mass=float(Descriptors.ExactMolWt(m)),
                    n_heavy_atoms=m.GetNumHeavyAtoms())


def symbols(smiles: str) -> list[str]:
    m = mol(smiles)
    return [a.GetSymbol() for a in m.GetAtoms()] if m else []


def kekulized_adjacency(smiles: str) -> np.ndarray:
    """Heavy-atom adjacency with integer bond orders, atoms in RDKit parse order (aromatic -> 1 if kekulisation fails)."""
    m = mol(smiles)
    if m is None:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
    except Exception:
        pass
    adjacency = np.zeros((m.GetNumAtoms(),) * 2, dtype=np.float32)
    for b in m.GetBonds():
        order = b.GetBondTypeAsDouble()
        adjacency[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = adjacency[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = 1 if order == 1.5 else int(order)
    return adjacency


def to_smiles(adjacency: np.ndarray, atoms: list[str], connected: bool = True) -> Optional[str]:
    """Sanitised canonical SMILES for an integer-order adjacency; ``None`` if RDKit rejects it (or it is disconnected)."""
    try:
        rw = adjacency_to_rwmol(adjacency, atoms)
        with rdBase.BlockLogs():
            Chem.SanitizeMol(rw)
        smiles = Chem.MolToSmiles(rw)
    except Exception:
        return None
    return None if connected and "." in smiles else smiles


def parallel_map(fn: Callable[[T], R], items: Iterable[T], n_jobs: int = 1, progress: bool = False, desc: str = "") -> list[R]:
    """Map with optional joblib processes and a progress bar."""
    items = tqdm(list(items), desc=desc, unit="item") if progress else list(items)
    if n_jobs == 1:
        return [fn(x) for x in items]
    return Parallel(n_jobs=n_jobs, prefer="processes", batch_size=256)(delayed(fn)(x) for x in items)
