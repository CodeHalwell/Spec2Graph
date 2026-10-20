"""Build competition assets: adduct vocabulary, structure table, candidate pool and the training cache.

python -m spec2graph.scripts.casmi_prepare --train-parquet data/train.parquet --cache-dir ~/.cache/spec2graph-casmi \\
    --output-dir assets/casmi [--extra-structures external/coconut.csv ...] --n-jobs 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from spec2graph.casmi import data
from spec2graph.casmi.adducts import AdductVocabulary
from spec2graph.casmi.pool import CandidatePool
from spec2graph.casmi.spectral import SpectralCache


def read_smiles(path: Path) -> pd.Series:
    df = pd.read_parquet(path) if path.suffix in (".parquet", ".pq") else pd.read_csv(path)
    column = next((c for c in df.columns if c.lower() in ("smiles", "canonical_smiles", "normalized_smiles")), None)
    if column is None:
        raise ValueError(f"{path}: no SMILES column")
    return df[column]


def main(argv=None) -> int:  # pragma: no cover - wiring only
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-parquet", required=True, type=Path)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--extra-structures", type=Path, action="append", default=[], help="CSV/parquet with a SMILES column; repeatable")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--fingerprint-bits", type=int, default=2048)
    p.add_argument("--n-quantiles", type=int, default=32)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--canonical-tautomer", action="store_true", help="store RDKit canonical tautomers as representative SMILES")
    p.add_argument("--max-train-rows", type=int, default=None, help="development knob")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    a.output_dir.mkdir(parents=True, exist_ok=True)

    train = data.load(a.train_parquet, require_smiles=True).iloc[: a.max_train_rows]
    AdductVocabulary.fit(train["adduct"]).save(a.output_dir / "adducts.json")
    tables = [data.structures(train["smiles"], a.canonical_tautomer, a.n_jobs, True, "train")]
    tables += [data.structures(read_smiles(f), a.canonical_tautomer, a.n_jobs, True, f.stem) for f in a.extra_structures]
    table = data.merge_structures(tables)
    table.to_parquet(a.output_dir / "structures.parquet", index=False)
    CandidatePool.build(table, a.k, a.fingerprint_bits, a.n_quantiles, a.n_jobs, True).save(a.output_dir / "pool.npz")

    cache = SpectralCache(a.cache_dir, k=a.k, fingerprint_bits=a.fingerprint_bits)
    own = table[table["source"] == "train"]
    cache.precompute_all(own[["inchikey", "smiles"]], n_jobs=a.n_jobs)
    for key, smiles in zip(own["inchikey"], own["smiles"]):
        cache.spectrum(smiles, key)
    logging.info("%d spectra, %d structures; assets written to %s", len(train), len(table), a.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
