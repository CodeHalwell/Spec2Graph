"""Validate on held-out molecules under both novelty scenarios; optionally fit the ranker on half of them.

python -m spec2graph.scripts.casmi_validate --checkpoint runs/casmi/epoch_010.pt --pool assets/casmi/pool.npz \\
    --train-parquet data/train.parquet --holdout-keys runs/casmi/holdout_keys.json --fit-ranker runs/casmi/ranker.json \\
    [--bond-sgno runs/casmi/bond_sgno_epoch_010.pt] --output runs/casmi/validation.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from spec2graph.casmi import data, validation
from spec2graph.casmi.predict import Predictor
from spec2graph.casmi.ranker import LinearRanker
from spec2graph.casmi.scoring import FEATURES

log = logging.getLogger(__name__)


def main(argv=None) -> int:  # pragma: no cover - wiring only
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--checkpoint", "--pool", "--train-parquet", "--output"):
        p.add_argument(name, required=True, type=Path)
    for name in ("--holdout-keys", "--ranker", "--fit-ranker", "--bond-sgno"):
        p.add_argument(name, type=Path, default=None)
    p.add_argument("--holdout-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-molecules", type=int, default=None)
    p.add_argument("--ppm", type=float, default=10.0)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--no-projection", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    predictor = Predictor.load(a.checkpoint, a.pool, a.ranker, a.bond_sgno, a.device, ppm=a.ppm, n_samples=a.n_samples,
                               n_steps=a.n_steps, use_projection=not a.no_projection)
    train = validation.with_keys(data.load(a.train_parquet, require_smiles=True))
    keys = set(json.loads(a.holdout_keys.read_text())) if a.holdout_keys else validation.split_keys(train["inchikey14"], a.holdout_fraction, a.seed)[1]
    _, holdout = validation.split_frame(train, keys)
    molecules = list(dict.fromkeys(holdout["molecule_id"]))
    np.random.default_rng(a.seed).shuffle(molecules)
    molecules = molecules[: a.limit_molecules]
    half = len(molecules) // 2 if a.fit_ranker else 0
    frame = lambda ids: holdout[holdout["molecule_id"].isin(ids)].reset_index(drop=True)

    if a.fit_ranker:
        fit = frame(molecules[:half])
        _, scores = predictor.predict_frame(fit, progress=True)
        groups = validation.training_groups(scores, predictor.pool, validation.truth_map(fit))
        try:
            predictor.ranker = LinearRanker.fit(groups, init=predictor.ranker)
        except ValueError as exc:
            log.warning("ranker not fitted (%s); keeping the current weights", exc)
        predictor.ranker.save(a.fit_ranker)
        log.info("ranker weights: %s", dict(zip(FEATURES, np.round(predictor.ranker.weights, 3))))

    reports = validation.validate(predictor, frame(molecules[half:]), progress=True)
    for report in reports.values():
        print(report)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({k: asdict(r) for k, r in reports.items()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
