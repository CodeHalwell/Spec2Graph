"""On-disk cache for Laplacian eigenvectors and Morgan fingerprints.

Running :meth:`SpectralDataProcessor.process_smiles` for every training
example costs an ``eigh`` call per row. MassSpecGym has ~29k unique
InChIKeys, so pre-computing once and reading from disk saves roughly an
hour per training run.

Layout
------
Files are sharded by the first two characters of the InChIKey to avoid
putting tens of thousands of files in a single directory::

    {cache_dir}/eigvecs_k{k}_{bond_weighting}/{inchikey[:2]}/{inchikey}.npy
    {cache_dir}/fingerprints_{n_bits}bit/{inchikey[:2]}/{inchikey}.npy

Corruption handling
-------------------
If a cached ``.npy`` file fails to load (truncated write, cosmic ray,
manual edit), the cache logs a warning, deletes the file and recomputes
from scratch rather than raising to the caller.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from spectral_diffusion import SpectralDataProcessor

logger = logging.getLogger(__name__)

_DEFAULT_FP_BITS = 2048
_DEFAULT_FP_RADIUS = 2


class EigenvectorCache:
    """Disk-backed cache keyed by InChIKey.

    Parameters
    ----------
    cache_dir:
        Root directory. Created on first use if it does not already exist.
    k:
        Number of eigenvectors retained per molecule. Must match the ``k``
        used when training the model.
    bond_weighting:
        Passed through to :class:`SpectralDataProcessor`.
    fingerprint_bits / fingerprint_radius:
        Morgan fingerprint parameters used by
        :meth:`get_or_compute_fingerprint`.
    """

    def __init__(
        self,
        cache_dir: str | os.PathLike,
        k: int,
        bond_weighting: str = "order",
        fingerprint_bits: int = _DEFAULT_FP_BITS,
        fingerprint_radius: int = _DEFAULT_FP_RADIUS,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.k = k
        self.bond_weighting = bond_weighting
        self.fingerprint_bits = fingerprint_bits
        self.fingerprint_radius = fingerprint_radius

        # One processor instance is enough — it is stateless apart from
        # its configuration.
        self._processor = SpectralDataProcessor(k=k, bond_weighting=bond_weighting)

        self.eigvec_root = self.cache_dir / f"eigvecs_k{k}_{bond_weighting}"
        self.fingerprint_root = self.cache_dir / f"fingerprints_{fingerprint_bits}bit"

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shard(inchikey: str) -> str:
        """Return the two-character shard prefix used in the directory layout.

        InChIKeys are always at least 2 characters; we uppercase to keep
        the sharding case-insensitive on Windows. An empty or missing
        key raises :class:`ValueError` — cache keys must be deterministic.
        """
        if not inchikey or len(inchikey) < 2:
            raise ValueError(
                f"InChIKey must have length >= 2 for sharding; got {inchikey!r}."
            )
        return inchikey[:2].upper()

    def eigvec_path(self, inchikey: str) -> Path:
        return self.eigvec_root / self._shard(inchikey) / f"{inchikey}.npy"

    def fingerprint_path(self, inchikey: str) -> Path:
        return self.fingerprint_root / self._shard(inchikey) / f"{inchikey}.npy"

    # ------------------------------------------------------------------
    # Low-level I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_save(array: np.ndarray, destination: Path) -> None:
        """Write ``array`` to ``destination`` atomically via a temp file.

        Guards against torn writes from interrupted processes — a reader
        will either see the old file, the new file, or a missing file,
        never a half-written one.

        Writes through the raw file descriptor so ``np.save`` does not
        apply its string-path ``.npy`` suffix logic. Passing the path as
        a string made ``np.save`` write to ``tmp_path + ".npy"``, leaving
        the ``mkstemp`` placeholder empty and causing the later
        ``os.replace`` to promote that empty file to the destination.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        # mkstemp creates the file on the same filesystem as destination,
        # which is a prerequisite for atomic os.replace.
        fd, tmp_path = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                np.save(handle, array, allow_pickle=False)
            os.replace(tmp_path, destination)
        except BaseException:
            # If anything between mkstemp and replace raised (including
            # KeyboardInterrupt), remove the stale temp file so the next
            # call does not trip over it.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _load_npy(self, path: Path) -> Optional[np.ndarray]:
        """Load a cached ``.npy`` file, or return ``None`` if corrupted.

        A corrupted file is deleted so the next call triggers a recompute.
        """
        if not path.exists():
            return None
        try:
            return np.load(path, allow_pickle=False)
        except (ValueError, OSError, EOFError) as exc:
            logger.warning(
                "Cache file %s is corrupted (%s); deleting and recomputing.",
                path,
                exc,
            )
            try:
                path.unlink()
            except OSError:
                pass
            return None

    # ------------------------------------------------------------------
    # Eigenvectors
    # ------------------------------------------------------------------

    def get_or_compute(
        self, smiles: str, inchikey: str
    ) -> Optional[np.ndarray]:
        """Return cached eigenvectors, computing them if necessary.

        Returns ``None`` if :meth:`SpectralDataProcessor.process_smiles`
        rejects the SMILES — callers should drop that row rather than
        failing the batch.
        """
        path = self.eigvec_path(inchikey)

        cached = self._load_npy(path)
        if cached is not None:
            return cached.astype(np.float32, copy=False)

        try:
            eigvecs = self._processor.process_smiles(smiles)
        except (ValueError, TypeError, ImportError) as exc:
            logger.warning(
                "Failed to compute eigenvectors for %s (inchikey=%s): %s",
                smiles,
                inchikey,
                exc,
            )
            return None

        eigvecs = np.asarray(eigvecs, dtype=np.float32)
        self._atomic_save(eigvecs, path)
        return eigvecs

    # ------------------------------------------------------------------
    # Fingerprints
    # ------------------------------------------------------------------

    def get_or_compute_fingerprint(
        self, smiles: str, inchikey: str
    ) -> Optional[np.ndarray]:
        """Return cached Morgan fingerprint, computing it if necessary."""
        path = self.fingerprint_path(inchikey)

        cached = self._load_npy(path)
        if cached is not None:
            return cached.astype(np.float32, copy=False)

        try:
            fingerprint = self._processor.smiles_to_fingerprint(
                smiles,
                n_bits=self.fingerprint_bits,
                radius=self.fingerprint_radius,
            )
        except (ValueError, TypeError, ImportError) as exc:
            logger.warning(
                "Failed to compute fingerprint for %s (inchikey=%s): %s",
                smiles,
                inchikey,
                exc,
            )
            return None

        fingerprint = np.asarray(fingerprint, dtype=np.float32)
        self._atomic_save(fingerprint, path)
        return fingerprint

    # ------------------------------------------------------------------
    # Bulk precomputation
    # ------------------------------------------------------------------

    def precompute_all(
        self,
        df: pd.DataFrame,
        n_jobs: int = -1,
        include_fingerprints: bool = True,
        show_progress: bool = True,
    ) -> dict[str, int]:
        """Precompute eigenvectors (and optionally fingerprints) for ``df``.

        Deduplicates by InChIKey so each molecule is processed at most
        once. Uses :mod:`joblib` for parallelism when ``n_jobs != 1``.

        Returns a dict of counts: ``{"requested", "computed", "skipped",
        "failed"}`` for logging / reporting.
        """
        if "inchikey" not in df.columns or "smiles" not in df.columns:
            raise ValueError(
                "Dataframe must contain 'inchikey' and 'smiles' columns for "
                "precomputation."
            )

        unique = df.drop_duplicates(subset=["inchikey"])[["inchikey", "smiles"]]
        # Filter out rows where the cache already has BOTH artefacts —
        # this is what makes the script idempotent.
        pairs: list[tuple[str, str]] = []
        for row in unique.itertuples(index=False):
            has_eigvec = self.eigvec_path(row.inchikey).exists()
            has_fingerprint = (
                not include_fingerprints
                or self.fingerprint_path(row.inchikey).exists()
            )
            if not (has_eigvec and has_fingerprint):
                pairs.append((row.inchikey, row.smiles))

        counts = {
            "requested": len(unique),
            "computed": 0,
            "skipped": len(unique) - len(pairs),
            "failed": 0,
        }
        if not pairs:
            logger.info("Cache already populated for all %d inchikeys.", counts["requested"])
            return counts

        iterator: Iterable[tuple[str, str]] = pairs
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(pairs, desc="precompute", unit="mol")
            except ImportError:
                logger.debug("tqdm not installed; falling back to silent iteration.")

        def _work(inchikey: str, smiles: str) -> tuple[str, bool]:
            eig = self.get_or_compute(smiles, inchikey)
            if eig is None:
                return inchikey, False
            if include_fingerprints:
                fp = self.get_or_compute_fingerprint(smiles, inchikey)
                if fp is None:
                    return inchikey, False
            return inchikey, True

        # joblib is optional at call time — if it is missing we fall back
        # to a serial loop. Users running tests without joblib installed
        # should still get a working cache.
        if n_jobs == 1:
            results = [_work(k, s) for k, s in iterator]
        else:
            try:
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=n_jobs, prefer="processes")(
                    delayed(_work)(k, s) for k, s in iterator
                )
            except ImportError:
                logger.warning(
                    "joblib not installed; running precomputation serially."
                )
                results = [_work(k, s) for k, s in iterator]

        for _, ok in results:
            if ok:
                counts["computed"] += 1
            else:
                counts["failed"] += 1

        logger.info(
            "precompute_all: requested=%d computed=%d skipped=%d failed=%d",
            counts["requested"],
            counts["computed"],
            counts["skipped"],
            counts["failed"],
        )
        return counts

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def has_eigvec(self, inchikey: str) -> bool:
        return self.eigvec_path(inchikey).exists()

    def has_fingerprint(self, inchikey: str) -> bool:
        return self.fingerprint_path(inchikey).exists()
