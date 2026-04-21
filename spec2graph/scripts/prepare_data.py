"""One-off preparation script: download MassSpecGym + precompute cache.

Usage::

    python -m spec2graph.scripts.prepare_data \\
        --cache-dir ~/.cache/spec2graph \\
        --k 8 \\
        --n-jobs 8

Idempotent — rerunning only fills in missing cache entries.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spec2graph.data.cache import EigenvectorCache
from spec2graph.data.massspecgym import (
    filter_massspecgym,
    load_massspecgym_tsv,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MassSpecGym and prime the eigenvector / fingerprint cache."
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Root of the on-disk cache.",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        type=Path,
        help="Optional hugging_face_hub cache directory for the raw TSV.",
    )
    parser.add_argument("--k", type=int, default=8, help="Number of eigenvectors per molecule.")
    parser.add_argument(
        "--bond-weighting",
        choices=("unweighted", "order", "aromatic"),
        default="order",
    )
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--adduct",
        action="append",
        default=None,
        help='Adduct to keep (may be passed multiple times). Default ("[M+H]+",).',
    )
    parser.add_argument("--max-precursor-mz", type=float, default=1000.0)
    parser.add_argument("--min-peaks", type=int, default=5)
    parser.add_argument("--max-smiles-length", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    adducts = tuple(args.adduct) if args.adduct else ("[M+H]+",)

    logger.info("Loading MassSpecGym (download if needed)...")
    df = load_massspecgym_tsv(cache_dir=str(args.download_dir) if args.download_dir else None)

    logger.info("Filtering...")
    filtered, stats = filter_massspecgym(
        df,
        adducts=adducts,
        max_precursor_mz=args.max_precursor_mz,
        min_peaks=args.min_peaks,
        max_smiles_length=args.max_smiles_length,
    )
    logger.info("Filter stats: %s", stats)

    logger.info(
        "Precomputing eigenvectors + fingerprints into %s (k=%d)...",
        args.cache_dir,
        args.k,
    )
    cache = EigenvectorCache(
        cache_dir=args.cache_dir,
        k=args.k,
        bond_weighting=args.bond_weighting,
        fingerprint_bits=args.fingerprint_bits,
    )
    counts = cache.precompute_all(
        filtered[["inchikey", "smiles"]].drop_duplicates(),
        n_jobs=args.n_jobs,
    )
    logger.info("Precompute complete: %s", counts)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
