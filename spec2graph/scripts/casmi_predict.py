"""Write ``submission.csv`` for the competition test set.

Usage (also the body of the offline Kaggle notebook)::

    python -m spec2graph.scripts.casmi_predict \\
        --checkpoint /kaggle/input/spec2graph-casmi/epoch_010.pt \\
        --pool /kaggle/input/spec2graph-casmi/pool.npz \\
        --ranker /kaggle/input/spec2graph-casmi/ranker.json \\
        --bond-sgno /kaggle/input/spec2graph-casmi/bond_sgno.pt \\
        --test-parquet /kaggle/input/enveda-CASMI26-molecule-id-mass-spectra/test.parquet \\
        --output /kaggle/working/submission.csv

Every molecule is predicted inside a try/except with a mass-nearest
fallback, and the file is validated against the competition's rejection
rules before it is written.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from spec2graph.casmi.conditioning import CasmiTrainer, load_checkpoint
from spec2graph.casmi.data import PeakPreprocessConfig, load_casmi_parquet
from spec2graph.casmi.denovo import DeNovoConfig, load_bond_sgno, make_denovo_fn
from spec2graph.casmi.metric import validate_submission
from spec2graph.casmi.pool import CandidatePool
from spec2graph.casmi.predict import PredictConfig, make_submission, predict_frame
from spec2graph.casmi.ranker import LinearRanker
from spec2graph.casmi.scoring import ScoringConfig

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict CASMI 2026 test structures with Spec2Graph.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--test-parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ranker", type=Path, default=None)
    parser.add_argument("--bond-sgno", type=Path, default=None, help="Enables de novo candidates for off-pool formulas.")
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--fallback-ppm", type=float, default=50.0)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--no-projection", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Development knob: only the first N molecules.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - wiring only
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    bundle = load_checkpoint(args.checkpoint, device=args.device)
    trainer = CasmiTrainer(bundle.model, bundle.trainer_config, device=args.device)
    pool = CandidatePool.load(args.pool)
    ranker = LinearRanker.load(args.ranker) if args.ranker else LinearRanker.heuristic()
    max_peaks = int(bundle.extra.get("max_peaks", bundle.model.core.max_peaks))
    peak_config = PeakPreprocessConfig(top_k_peaks=max_peaks, max_peaks=max_peaks, sqrt_intensity=bool(bundle.extra.get("sqrt_intensity", False)))

    test = load_casmi_parquet(args.test_parquet)
    if args.limit:
        keep = list(dict.fromkeys(test["molecule_id"]))[: args.limit]
        test = test[test["molecule_id"].isin(keep)].reset_index(drop=True)
    logger.info("%d test spectra over %d molecules; pool of %d structures", len(test), test["molecule_id"].nunique(), len(pool))

    config = PredictConfig(
        scoring=ScoringConfig(ppm=args.ppm, n_samples=args.n_samples, n_steps=args.n_steps, use_projection=not args.no_projection),
        peak_config=peak_config,
        max_peaks=max_peaks,
        fallback_ppm=args.fallback_ppm,
        denovo=args.bond_sgno is not None,
    )
    denovo_fn = make_denovo_fn(load_bond_sgno(args.bond_sgno, device=args.device), DeNovoConfig(n_steps=args.n_steps)) if args.bond_sgno else None

    predictions, _ = predict_frame(
        trainer, pool, test, adduct_vocab=bundle.adduct_vocab, ranker=ranker, config=config, denovo_fn=denovo_fn, progress=True
    )
    submission = make_submission(predictions, test, args.output)
    problems = validate_submission(submission, molecule_ids=list(dict.fromkeys(test["molecule_id"])))
    if problems:
        logger.error("Submission problems: %s", problems)
        return 1
    logger.info("Wrote %s (%d rows)", args.output, len(submission))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
