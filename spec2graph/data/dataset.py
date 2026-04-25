"""MassSpecGym Torch :class:`Dataset` for Spec2Graph training and eval.

Wraps :func:`load_massspecgym_tsv` and :class:`EigenvectorCache` with the
per-example preprocessing described in the agent instructions — precursor
echo removal, top-K peak filtering, intensity normalisation, atom-type
extraction in RDKit canonical order, and adjacency extraction for
supervised SGNO training.

The constructor filters the dataframe down to rows whose InChIKey has a
populated cache entry, so :meth:`__len__` reflects only examples that
can be loaded cleanly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from spec2graph.data.cache import EigenvectorCache
from spec2graph.data.elements import atom_types_to_indices
from spec2graph.data.massspecgym import (
    filter_massspecgym,
    load_massspecgym_tsv,
    parse_peak_list,
)

try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover - torch is a hard dep of the repo
    _TorchDataset = object

logger = logging.getLogger(__name__)


def _adjacency_from_smiles(smiles: str) -> Optional[np.ndarray]:
    """Return the heavy-atom adjacency matrix for a SMILES, or ``None``.

    Bond orders are real-valued to match :class:`SpectralDataProcessor`
    (single=1, double=2, triple=3, aromatic=1.5). The ordering of atoms
    matches the RDKit canonical ordering, which in turn matches the
    ordering used for eigenvector computation.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "rdkit is required for MassSpecGymDataset. Install via conda or "
            "`pip install rdkit`."
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n = mol.GetNumAtoms()
    if n == 0:
        return None
    adjacency = np.zeros((n, n), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        weight = float(bond.GetBondTypeAsDouble())
        adjacency[i, j] = weight
        adjacency[j, i] = weight
    return adjacency


def _atom_symbols_from_smiles(smiles: str) -> Optional[list[str]]:
    """Return a list of element symbols in RDKit canonical order."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("rdkit is required.") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if mol.GetNumAtoms() == 0:
        return None
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


class MassSpecGymDataset(_TorchDataset):
    """Torch-compatible dataset producing dicts of numpy arrays.

    Parameters map directly onto the agent instructions for Task 3. The
    dataset loads the MassSpecGym TSV, filters it, caches eigenvectors /
    fingerprints via :class:`EigenvectorCache`, and returns one dict per
    call to :meth:`__getitem__`.

    The constructor-time step that filters out InChIKeys with no cache
    entry means :meth:`__len__` reflects only examples the dataset can
    successfully load. Rows that fail mid-epoch would otherwise surface
    as surprise :class:`KeyError`s or ``None``-valued tensors.
    """

    def __init__(
        self,
        split: str,
        cache_dir: str | Path,
        *,
        k: int = 8,
        max_atoms: int = 64,
        max_peaks: int = 128,
        top_k_peaks: int = 128,
        fingerprint_bits: int = 2048,
        adducts: tuple[str, ...] = ("[M+H]+",),
        max_precursor_mz: float = 1000.0,
        min_peaks: int = 5,
        max_smiles_length: int = 200,
        bond_weighting: str = "order",
        dataframe: Optional[pd.DataFrame] = None,
        download_cache_dir: Optional[str] = None,
        precompute: bool = False,
        include_adjacency: bool = False,
        validate_cache: bool = False,
        cache_precompute_jobs: int = -1,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"split must be one of 'train', 'val', 'test'; got {split!r}."
            )
        self.split = split
        self.cache_dir = Path(cache_dir)
        self.k = k
        self.max_atoms = max_atoms
        self.max_peaks = max_peaks
        self.top_k_peaks = top_k_peaks
        self.fingerprint_bits = fingerprint_bits
        self.bond_weighting = bond_weighting
        self.include_adjacency = include_adjacency

        # Accept a pre-built dataframe so tests can avoid the network.
        if dataframe is None:
            dataframe = load_massspecgym_tsv(cache_dir=download_cache_dir)

        # Restrict to the requested split first so filters only work on
        # the relevant subset (cheaper and more meaningful stats).
        if "split" not in dataframe.columns:
            raise RuntimeError(
                "Dataframe is missing the canonical 'split' column; call "
                "load_massspecgym_tsv to produce one."
            )
        subset = dataframe[dataframe["split"] == split]
        if subset.empty:
            raise ValueError(f"No rows in split {split!r}.")

        filtered, stats = filter_massspecgym(
            subset,
            adducts=adducts,
            max_precursor_mz=max_precursor_mz,
            min_peaks=min_peaks,
            max_smiles_length=max_smiles_length,
        )
        logger.info("MassSpecGymDataset[%s]: %s", split, stats)

        # Also drop anything whose heavy-atom count exceeds max_atoms;
        # those examples cannot be collated without truncating, which
        # would invalidate the eigenvector target.
        filtered = filtered.copy()
        filtered["n_heavy_atoms"] = filtered["smiles"].apply(
            lambda s: _heavy_atom_count(s) if isinstance(s, str) else 0
        )
        before_n_atoms = len(filtered)
        filtered = filtered[
            (filtered["n_heavy_atoms"] > 0)
            & (filtered["n_heavy_atoms"] <= max_atoms)
        ]
        dropped_size = before_n_atoms - len(filtered)
        if dropped_size:
            logger.info(
                "MassSpecGymDataset[%s]: dropped %d rows with n_atoms<=0 or >%d.",
                split,
                dropped_size,
                max_atoms,
            )
        filtered = filtered.reset_index(drop=True)

        self.cache = EigenvectorCache(
            cache_dir=self.cache_dir,
            k=k,
            bond_weighting=bond_weighting,
            fingerprint_bits=fingerprint_bits,
        )

        # Optionally prime the cache in parallel. ``precompute=True`` is
        # the recommended path when the cache directory is empty; it
        # uses joblib to run eigendecomposition across workers.
        if precompute:
            self.cache.precompute_all(
                filtered[["inchikey", "smiles"]].drop_duplicates(),
                n_jobs=cache_precompute_jobs,
                show_progress=False,
            )

        # Build the index mapping.
        #
        # Two regimes:
        #
        # * ``validate_cache=False`` (default): check file existence via
        #   ``has_eigvec``/``has_fingerprint`` — just a ``stat()`` per
        #   row, which is microseconds. This assumes callers have
        #   already primed the cache via ``precompute=True`` or the
        #   ``prepare_data`` CLI. Rows whose cache files are missing are
        #   dropped up-front so ``__getitem__`` never surprises us.
        # * ``validate_cache=True``: go deeper and actually load each
        #   cached file, re-running RDKit when the file is absent.
        #   Slower but guarantees every row survives ``__getitem__``.
        indexable_rows: list[int] = []
        missing: list[str] = []
        for idx, row in filtered.iterrows():
            inchikey = row["inchikey"]
            if validate_cache:
                if self.cache.get_or_compute(row["smiles"], inchikey) is None:
                    missing.append(inchikey)
                    continue
                if self.cache.get_or_compute_fingerprint(row["smiles"], inchikey) is None:
                    missing.append(inchikey)
                    continue
            else:
                if not self.cache.has_eigvec(inchikey):
                    missing.append(inchikey)
                    continue
                if not self.cache.has_fingerprint(inchikey):
                    missing.append(inchikey)
                    continue
            # Pre-apply the same peak filter __getitem__ uses so rows
            # that lose all their peaks to the precursor-echo cutoff
            # are dropped up-front rather than raising mid-epoch. This
            # is cheap: one pass over the stringified peak list.
            if not _has_usable_peaks_post_cutoff(row):
                missing.append(inchikey)
                continue
            indexable_rows.append(int(idx))

        if missing:
            level = logger.warning if validate_cache else logger.info
            level(
                "MassSpecGymDataset[%s]: %d rows skipped (missing cache entries "
                "or no peaks below precursor-echo cutoff).",
                split,
                len(missing),
            )

        self._df = filtered.loc[indexable_rows].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._df.iloc[idx]
        smiles: str = row["smiles"]
        inchikey: str = row["inchikey"]

        eigvecs = self.cache.get_or_compute(smiles, inchikey)
        fingerprint = self.cache.get_or_compute_fingerprint(smiles, inchikey)
        assert eigvecs is not None and fingerprint is not None, (
            "Constructor should have filtered uncacheable rows."
        )

        mz = parse_peak_list(row["mzs"])
        intensity = parse_peak_list(row["intensities"])
        if mz.shape != intensity.shape:
            raise ValueError(
                f"mz and intensity shape mismatch at idx={idx}: "
                f"{mz.shape} vs {intensity.shape}"
            )
        precursor_mz = float(row["precursor_mz"])

        mz, intensity = _filter_and_normalise_peaks(
            mz,
            intensity,
            precursor_mz=precursor_mz,
            top_k=self.top_k_peaks,
            max_peaks=self.max_peaks,
        )
        if mz.size == 0:
            raise RuntimeError(
                f"All peaks filtered out for idx={idx}. The constructor's "
                "min_peaks filter should have prevented this; indicates a "
                "filter misconfiguration."
            )

        symbols = _atom_symbols_from_smiles(smiles) or []
        atom_types = np.asarray(atom_types_to_indices(symbols), dtype=np.int64)

        sample: dict[str, Any] = {
            "eigvecs": eigvecs.astype(np.float32, copy=False),
            "n_atoms": int(eigvecs.shape[0]),
            "mz": mz.astype(np.float32, copy=False),
            "intensity": intensity.astype(np.float32, copy=False),
            "precursor_mz": precursor_mz,
            "fingerprint": fingerprint.astype(np.float32, copy=False),
            "atom_types": atom_types,
            "atom_symbols": symbols,
            "smiles": smiles,
            "formula": str(row.get("formula", "")),
            "inchikey": inchikey,
            "collision_energy": _safe_float(row.get("collision_energy")),
        }

        if self.include_adjacency:
            adjacency = _adjacency_from_smiles(smiles)
            if adjacency is None:
                # This should not happen — the constructor would have
                # filtered the row — but defend against later edits.
                adjacency = np.zeros(
                    (sample["n_atoms"], sample["n_atoms"]), dtype=np.float32
                )
            sample["adjacency"] = adjacency

        return sample


def _has_usable_peaks_post_cutoff(row: pd.Series) -> bool:
    """Return True iff the row still has peaks after the precursor-echo filter.

    Mirrors the ``m/z < precursor_mz - 0.5`` rule applied in
    :meth:`MassSpecGymDataset.__getitem__` so we can drop dead rows at
    construction time instead of raising mid-epoch.
    """
    try:
        mz = parse_peak_list(row["mzs"])
        precursor_mz = float(row["precursor_mz"])
    except (ValueError, SyntaxError, TypeError):
        return False
    cutoff = precursor_mz - 0.5
    return bool((mz < cutoff).any())


def _heavy_atom_count(smiles: str) -> int:
    """Return the number of heavy atoms or 0 if RDKit can't parse."""
    try:
        from rdkit import Chem
    except ImportError:
        return 0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetNumAtoms()


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _filter_and_normalise_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    precursor_mz: float,
    top_k: int,
    max_peaks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the canonical MassSpecGym eval-time preprocessing to a spectrum.

    1. Drop peaks with m/z >= precursor_mz - 0.5 (precursor echo).
    2. Normalise intensities so max == 1.0.
    3. Keep the top-K peaks by intensity.
    4. Re-sort ascending by m/z.
    5. Cap at ``max_peaks`` (defensive).
    """
    mask = mz < (precursor_mz - 0.5)
    mz = mz[mask]
    intensity = intensity[mask]
    if mz.size == 0:
        return mz, intensity

    peak_max = intensity.max()
    if peak_max > 0:
        intensity = intensity / peak_max
    else:
        logger.warning(
            "Spectrum had all-zero intensities; skipping normalisation."
        )

    if mz.size > top_k:
        top_indices = np.argpartition(intensity, -top_k)[-top_k:]
        mz = mz[top_indices]
        intensity = intensity[top_indices]

    order = np.argsort(mz, kind="stable")
    mz = mz[order]
    intensity = intensity[order]

    if mz.size > max_peaks:
        mz = mz[:max_peaks]
        intensity = intensity[:max_peaks]
    return mz, intensity
