"""Evaluation entry point — runs the full MassSpecGym benchmark.

Usage::

    python -m spec2graph.scripts.evaluate \\
        --checkpoint runs/baseline/epoch_020.pt \\
        --sgno-checkpoint runs/baseline/sgno_epoch_020.pt \\
        --cache-dir ~/.cache/spec2graph \\
        --split test \\
        --n-samples 100 \\
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from spec2graph.data.dataset import MassSpecGymDataset
from spec2graph.eval.benchmark import BenchmarkConfig, benchmark_model
from spectral_diffusion import (
    DiffusionTrainer,
    Spec2GraphDiffusion,
    Spec2GraphDiffusionConfig,
    SpectralGraphNeuralOperator,
    TrainerConfig,
    ValencyDecoder,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Spec2Graph on MassSpecGym.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Diffusion model checkpoint.")
    parser.add_argument(
        "--sgno-checkpoint",
        type=Path,
        default=None,
        help="SGNO checkpoint. If omitted a randomly initialised SGNO is used, "
        "which yields near-zero top-k accuracy.",
    )
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddim")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--decode-threshold", type=float, default=0.3)
    parser.add_argument("--max-bond-order", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-name", default="Spec2Graph")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - wiring only
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    logger.info("Loading checkpoint %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config_dict = ckpt["config"]
    trainer_dict = ckpt.get("trainer_config", {})

    config = Spec2GraphDiffusionConfig(**config_dict)
    model = Spec2GraphDiffusion(config).to(args.device)
    model.load_state_dict(ckpt["model"])

    trainer_config = TrainerConfig(**trainer_dict) if trainer_dict else TrainerConfig()
    trainer = DiffusionTrainer(model=model, config=trainer_config, device=args.device)

    sgno = SpectralGraphNeuralOperator(k=config.k).to(args.device)
    if args.sgno_checkpoint is not None:
        logger.info("Loading SGNO checkpoint %s", args.sgno_checkpoint)
        sgno_ckpt = torch.load(args.sgno_checkpoint, map_location=args.device, weights_only=False)
        sgno.load_state_dict(sgno_ckpt["model"])
    else:
        logger.warning(
            "No SGNO checkpoint provided; the SGNO is randomly initialised and "
            "top-k accuracy will be ~0. Train an SGNO via --train-sgno in "
            "scripts.train before running a real benchmark."
        )

    dataset = MassSpecGymDataset(
        split=args.split,
        cache_dir=args.cache_dir,
        k=config.k,
        max_atoms=config.max_atoms,
        max_peaks=config.max_peaks,
        fingerprint_bits=config.fingerprint_dim or 2048,
    )

    bench_config = BenchmarkConfig(
        n_samples_per_spectrum=args.n_samples,
        batch_size=args.batch_size,
        sampler=args.sampler,
        ddim_n_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        decode_threshold=args.decode_threshold,
        max_bond_order=args.max_bond_order,
        limit=args.limit,
        progress=True,
    )
    results = benchmark_model(
        trainer=trainer,
        sgno=sgno,
        dataset=dataset,
        valency_decoder=ValencyDecoder(),
        config=bench_config,
        device=args.device,
        jsonl_path=args.jsonl,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results.to_dict(), indent=2))
    print(results.to_markdown_row(args.model_name))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
