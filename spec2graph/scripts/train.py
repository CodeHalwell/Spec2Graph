"""Training entry point for Spec2Graph on MassSpecGym.

Usage::

    python -m spec2graph.scripts.train \\
        --cache-dir ~/.cache/spec2graph \\
        --output-dir runs/baseline \\
        --epochs 20 \\
        --batch-size 32 \\
        --lr 1e-4

Trains both the diffusion model and, optionally, the SGNO from adjacency
supervision. Saves a checkpoint per epoch and optionally runs a small
validation sample each epoch.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spec2graph.data.augment import wrap_collator_with_permutation
from spec2graph.data.collate import (
    make_metadata_collator,
    make_training_batch_collator,
)
from spec2graph.data.dataset import MassSpecGymDataset
from spec2graph.train.sgno_trainer import SGNOTrainer, SGNOTrainerConfig
from spectral_diffusion import (
    DiffusionTrainer,
    Spec2GraphDiffusion,
    Spec2GraphDiffusionConfig,
    SpectralGraphNeuralOperator,
    TrainerConfig,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Spec2Graph on MassSpecGym.")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--num-decoder-layers", type=int, default=4)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-atoms", type=int, default=64)
    parser.add_argument("--max-peaks", type=int, default=128)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--n-timesteps", type=int, default=1000)
    parser.add_argument("--atom-type-weight", type=float, default=0.5)
    parser.add_argument("--fingerprint-weight", type=float, default=0.1)
    parser.add_argument("--atom-count-weight", type=float, default=0.05)
    parser.add_argument("--permute-atoms", action="store_true")
    parser.add_argument("--train-sgno", action="store_true")
    parser.add_argument("--sgno-hidden-dim", type=int, default=128)
    parser.add_argument("--sgno-layers", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - wiring only
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading train / val splits...")
    train_ds = MassSpecGymDataset(
        split="train",
        cache_dir=args.cache_dir,
        k=args.k,
        max_atoms=args.max_atoms,
        max_peaks=args.max_peaks,
        fingerprint_bits=args.fingerprint_bits,
        include_adjacency=args.train_sgno,
    )
    val_ds = MassSpecGymDataset(
        split="val",
        cache_dir=args.cache_dir,
        k=args.k,
        max_atoms=args.max_atoms,
        max_peaks=args.max_peaks,
        fingerprint_bits=args.fingerprint_bits,
    )
    logger.info("Train size: %d, Val size: %d", len(train_ds), len(val_ds))

    config = Spec2GraphDiffusionConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        k=args.k,
        max_atoms=args.max_atoms,
        max_peaks=args.max_peaks,
        fingerprint_dim=args.fingerprint_bits,
        enable_atom_count_head=True,
        enable_atom_type_head=True,
    )
    model = Spec2GraphDiffusion(config).to(args.device)
    trainer = DiffusionTrainer(
        model=model,
        config=TrainerConfig(
            n_timesteps=args.n_timesteps,
            atom_type_loss_weight=args.atom_type_weight,
            fingerprint_loss_weight=args.fingerprint_weight,
            atom_count_loss_weight=args.atom_count_weight,
        ),
        device=args.device,
    )
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    base_collator = make_training_batch_collator(
        max_atoms=args.max_atoms,
        max_peaks=args.max_peaks,
        k=args.k,
        fingerprint_bits=args.fingerprint_bits,
        device=args.device,
    )
    collator = wrap_collator_with_permutation(base_collator) if args.permute_atoms else base_collator

    sgno = SGNOTrainer = None
    sgno_optim = None
    if args.train_sgno:
        sgno_module = SpectralGraphNeuralOperator(
            k=args.k, hidden_dim=args.sgno_hidden_dim, num_layers=args.sgno_layers
        ).to(args.device)
        sgno = SGNOTrainer = sgno_module  # type: ignore
        from spec2graph.train.sgno_trainer import SGNOTrainer as _SGNOTrainer
        sgno_trainer = _SGNOTrainer(sgno_module, config=SGNOTrainerConfig(pos_weight=5.0), device=args.device)
        sgno_optim = torch.optim.Adam(sgno_module.parameters(), lr=args.lr)
        eval_collator = make_metadata_collator(
            max_atoms=args.max_atoms,
            max_peaks=args.max_peaks,
            k=args.k,
            fingerprint_bits=args.fingerprint_bits,
            device=args.device,
        )
    else:
        sgno_trainer = None
        eval_collator = None

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    sgno_loader = None
    if sgno_trainer is not None:
        sgno_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=eval_collator,
            num_workers=args.num_workers,
        )

    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        for step, batch in enumerate(loader):
            loss = trainer.train_step(optim, batch)
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            losses.append(loss)
        epoch_loss = sum(losses) / max(len(losses), 1)
        logger.info("Epoch %d: train loss = %.4f", epoch + 1, epoch_loss)

        if sgno_trainer is not None and sgno_loader is not None:
            sgno_losses: list[float] = []
            for collated in sgno_loader:
                sl = sgno_trainer.train_step(
                    sgno_optim,
                    collated.batch.x_0,
                    collated.batch.atom_mask,
                    collated.adjacencies,
                )
                sgno_losses.append(sl)
            logger.info(
                "Epoch %d: sgno loss = %.4f",
                epoch + 1,
                sum(sgno_losses) / max(len(sgno_losses), 1),
            )

        # Checkpoint
        checkpoint_path = args.output_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "config": config.__dict__,
                "trainer_config": trainer.config.__dict__,
            },
            checkpoint_path,
        )
        if sgno_trainer is not None:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": sgno_module.state_dict(),  # type: ignore
                    "optimizer": sgno_optim.state_dict() if sgno_optim else None,
                },
                args.output_dir / f"sgno_epoch_{epoch + 1:03d}.pt",
            )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
