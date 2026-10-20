"""Prepare competition assets: adduct vocabulary, structure table, candidate pool, caches.

Usage::

    python -m spec2graph.scripts.casmi_prepare \\
        --train-parquet data/train.parquet \\
        --cache-dir ~/.cache/spec2graph-casmi \\
        --output-dir assets/casmi \\
        --extra-structures external/coconut.csv \\
        --k 8 --n-jobs 8

Writes ``adducts.json``, ``structures.parquet`` and ``pool.npz`` into the
output directory and primes the eigenvector / fingerprint / spectrum cache
for every training structure. Idempotent: existing cache entries are kept.

``--extra-structures`` accepts CSV or parquet files with a ``smiles`` column
(COCONUT, PubChem subsets, ...). Candidate pools built from them inherit the
licence of their source; competition data stays under CC BY-NC 4.0.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from spec2graph.casmi.adducts import AdductVocabulary
from spec2graph.casmi.data import build_structure_table, load_casmi_parquet, merge_structure_tables
from spec2graph.casmi.pool import CandidatePool
from spec2graph.casmi.spectral import SpectralCache

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Spec2Graph assets for the CASMI 2026 competition.")
    parser.add_argument("--train-parquet", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--extra-structures", type=Path, action="append", default=[], help="CSV/parquet with a 'smiles' column; repeatable.")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--n-quantiles", type=int, default=32)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--canonical-tautomer", action="store_true", help="Store RDKit canonical tautomers as representative SMILES (slower).")
    parser.add_argument("--skip-cache", action="store_true", help="Do not prime the training cache (pool only).")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Development knob: only read this many spectra.")
    return parser.parse_args(argv)


def _read_structures(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    column = lower.get("smiles") or lower.get("canonical_smiles") or lower.get("normalized_smiles")
    if column is None:
        raise ValueError(f"{path} has no SMILES column; found {list(df.columns)}.")
    return df[[column]].rename(columns={column: "smiles"})


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - wiring only
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading %s", args.train_parquet)
    train = load_casmi_parquet(args.train_parquet, require_smiles=True)
    if args.max_train_rows:
        train = train.iloc[: args.max_train_rows]
    logger.info("%d spectra, %d molecule ids", len(train), train["molecule_id"].nunique())

    vocab = AdductVocabulary.fit(train["adduct"])
    vocab.save(args.output_dir / "adducts.json")
    logger.info("Adduct vocabulary: %s", vocab.names)

    logger.info("Building structure table from training SMILES...")
    tables = [
        build_structure_table(
            train["smiles"], canonical_tautomer=args.canonical_tautomer, n_jobs=args.n_jobs, show_progress=True, source="train"
        )
    ]
    for path in args.extra_structures:
        logger.info("Adding structures from %s", path)
        extra = _read_structures(path)
        tables.append(
            build_structure_table(
                extra["smiles"], canonical_tautomer=args.canonical_tautomer, n_jobs=args.n_jobs, show_progress=True, source=path.stem
            )
        )
    structures = merge_structure_tables(tables)
    structures.to_parquet(args.output_dir / "structures.parquet", index=False)
    logger.info("%d distinct structures (InChIKey14)", len(structures))

    logger.info("Building candidate pool...")
    pool = CandidatePool.build(
        structures,
        k=args.k,
        fingerprint_bits=args.fingerprint_bits,
        n_quantiles=args.n_quantiles,
        n_jobs=args.n_jobs,
        show_progress=True,
        source_column="source",
    )
    pool.save(args.output_dir / "pool.npz")
    logger.info("Pool saved with %d structures", len(pool))

    if not args.skip_cache:
        logger.info("Priming spectral cache for training structures...")
        cache = SpectralCache(args.cache_dir, k=args.k, fingerprint_bits=args.fingerprint_bits)
        train_structures = structures[structures["source"] == "train"]
        cache.precompute_all(train_structures[["inchikey", "smiles"]], n_jobs=args.n_jobs, show_progress=True)
        for key, smiles in zip(train_structures["inchikey"], train_structures["smiles"]):
            cache.get_or_compute_spectrum(smiles, key)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
