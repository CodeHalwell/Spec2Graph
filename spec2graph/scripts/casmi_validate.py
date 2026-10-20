"""Molecule-disjoint validation under the novelty scenarios, optional ranker fit.

Usage::

    python -m spec2graph.scripts.casmi_validate \\
        --checkpoint runs/casmi/epoch_010.pt \\
        --pool assets/casmi/pool.npz \\
        --train-parquet data/train.parquet \\
        --holdout-keys runs/casmi/holdout_keys.json \\
        --fit-ranker runs/casmi/ranker.json \\
        --output runs/casmi/validation.json

The held-out molecules are split in two: the ranker is fitted on one half
and every reported number comes from the other, so the MRR is not inflated
by fitting and scoring on the same molecules. ``--bond-sgno`` enables de
novo candidates for the ``novel_structure`` scenario.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from spec2graph.casmi.conditioning import load_checkpoint
from spec2graph.casmi.data import PeakPreprocessConfig, load_casmi_parquet
from spec2graph.casmi.denovo import DeNovoConfig, load_bond_sgno, make_denovo_fn
from spec2graph.casmi.pool import CandidatePool
from spec2graph.casmi.predict import PredictConfig, predict_frame
from spec2graph.casmi.ranker import LinearRanker
from spec2graph.casmi.scoring import ScoringConfig
from spec2graph.casmi.validation import (
    add_inchikey14_column,
    molecule_disjoint_split,
    novelty_scenarios,
    ranker_training_groups,
    run_validation,
    split_frame,
    truth_map,
)
from spec2graph.casmi.conditioning import CasmiTrainer

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Spec2Graph on held-out CASMI molecules.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--train-parquet", required=True, type=Path)
    parser.add_argument("--holdout-keys", type=Path, default=None, help="JSON list from casmi_train; otherwise a fresh split is drawn.")
    parser.add_argument("--holdout-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-molecules", type=int, default=None)
    parser.add_argument("--ranker", type=Path, default=None, help="Existing ranker.json to evaluate.")
    parser.add_argument("--fit-ranker", type=Path, default=None, help="Fit a ranker on half the holdout and save it here.")
    parser.add_argument("--bond-sgno", type=Path, default=None)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--no-projection", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - wiring only
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    bundle = load_checkpoint(args.checkpoint, device=args.device)
    trainer = CasmiTrainer(bundle.model, bundle.trainer_config, device=args.device)
    pool = CandidatePool.load(args.pool)
    max_peaks = int(bundle.extra.get("max_peaks", bundle.model.core.max_peaks))
    peak_config = PeakPreprocessConfig(top_k_peaks=max_peaks, max_peaks=max_peaks, sqrt_intensity=bool(bundle.extra.get("sqrt_intensity", False)))

    train = add_inchikey14_column(load_casmi_parquet(args.train_parquet, require_smiles=True))
    if args.holdout_keys is not None:
        holdout_keys = set(json.loads(args.holdout_keys.read_text()))
    else:
        _, holdout_keys = molecule_disjoint_split(train["inchikey14"], holdout_fraction=args.holdout_fraction, seed=args.seed)
    _, holdout = split_frame(train, holdout_keys)
    molecule_ids = list(dict.fromkeys(holdout["molecule_id"]))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(molecule_ids)
    if args.limit_molecules:
        molecule_ids = molecule_ids[: args.limit_molecules]
    half = len(molecule_ids) // 2
    fit_ids, eval_ids = set(molecule_ids[:half]), set(molecule_ids[half:]) if args.fit_ranker else set(molecule_ids)
    logger.info("Holdout molecules: %d (fit %d, eval %d)", len(molecule_ids), len(fit_ids) if args.fit_ranker else 0, len(eval_ids))

    config = PredictConfig(
        scoring=ScoringConfig(ppm=args.ppm, n_samples=args.n_samples, n_steps=args.n_steps, use_projection=not args.no_projection),
        peak_config=peak_config,
        max_peaks=max_peaks,
        denovo=args.bond_sgno is not None,
    )
    denovo_fn = make_denovo_fn(load_bond_sgno(args.bond_sgno, device=args.device), DeNovoConfig(n_steps=args.n_steps)) if args.bond_sgno else None

    ranker = LinearRanker.load(args.ranker) if args.ranker else LinearRanker.heuristic()
    if args.fit_ranker:
        fit_frame = holdout[holdout["molecule_id"].isin(fit_ids)].reset_index(drop=True)
        _, scores = predict_frame(trainer, pool, fit_frame, adduct_vocab=bundle.adduct_vocab, ranker=ranker, config=config, progress=True, collect_scores=True)
        groups = ranker_training_groups(scores, pool, truth_map(fit_frame))
        logger.info("Fitting ranker on %d molecules with the answer in window", len(groups))
        try:
            ranker = LinearRanker.fit(groups, init=ranker)
        except ValueError as exc:
            logger.warning("Ranker not fitted (%s); keeping the current weights.", exc)
        ranker.save(args.fit_ranker)
        logger.info("Ranker weights: %s", dict(zip(ranker.feature_names, np.round(ranker.weights, 3))))

    eval_frame = holdout[holdout["molecule_id"].isin(eval_ids)].reset_index(drop=True)
    reports = run_validation(
        trainer, pool, eval_frame, adduct_vocab=bundle.adduct_vocab, ranker=ranker, config=config, denovo_fn=denovo_fn, progress=True
    )
    for report in reports.values():
        print(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({name: r.to_dict() for name, r in reports.items()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
