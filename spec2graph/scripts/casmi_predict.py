"""Write ``submission.csv`` for the test set (the body of the offline Kaggle notebook).

python -m spec2graph.scripts.casmi_predict --checkpoint runs/casmi/epoch_010.pt --pool assets/casmi/pool.npz \\
    --ranker runs/casmi/ranker.json [--bond-sgno runs/casmi/bond_sgno_epoch_010.pt] \\
    --test-parquet data/test.parquet --output submission.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from spec2graph.casmi import data
from spec2graph.casmi.predict import Predictor


def main(argv=None) -> int:  # pragma: no cover - wiring only
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--checkpoint", "--pool", "--test-parquet", "--output"):
        p.add_argument(name, required=True, type=Path)
    for name in ("--ranker", "--bond-sgno"):
        p.add_argument(name, type=Path, default=None)
    p.add_argument("--ppm", type=float, default=10.0)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--no-projection", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="development knob: first N molecules only")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if a.device == "mps":
        torch.backends.mha.set_fastpath_enabled(False)  # the encoder's eval fast path needs an op MPS lacks

    predictor = Predictor.load(a.checkpoint, a.pool, a.ranker, a.bond_sgno, a.device, ppm=a.ppm, n_samples=a.n_samples,
                               n_steps=a.n_steps, use_projection=not a.no_projection)
    test = data.load(a.test_parquet)
    if a.limit:
        test = test[test["molecule_id"].isin(list(dict.fromkeys(test["molecule_id"]))[: a.limit])].reset_index(drop=True)
    submission = predictor.submission(test, a.output)
    logging.info("wrote %s (%d molecules, pool of %d structures)", a.output, len(submission), len(predictor.pool))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
