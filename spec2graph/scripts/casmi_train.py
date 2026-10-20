"""Train the conditioned Spec2Graph model (and optionally the bond-type SGNO).

Usage::

    python -m spec2graph.scripts.casmi_train \\
        --train-parquet data/train.parquet \\
        --assets-dir assets/casmi \\
        --cache-dir ~/.cache/spec2graph-casmi \\
        --output-dir runs/casmi \\
        --epochs 10 --batch-size 128 --lr 1e-4 --train-bond-sgno

The split is molecule-disjoint by InChIKey14; the held-out keys are written
to ``holdout_keys.json`` so validation and ranker fitting never see training
structures. One checkpoint per epoch (``epoch_NNN.pt``) plus
``bond_sgno_epoch_NNN.pt`` when the bond-type decoder is trained.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spec2graph.casmi.adducts import AdductVocabulary
from spec2graph.casmi.conditioning import CasmiModel, CasmiModelConfig, CasmiTrainer, CasmiTrainerConfig, save_checkpoint
from spec2graph.casmi.data import CasmiSpectrumDataset, PeakPreprocessConfig, load_casmi_parquet, make_casmi_collator
from spec2graph.casmi.denovo import BondTypeSGNO, BondTypeTrainer, save_bond_sgno
from spec2graph.casmi.spectral import SpectralCache
from spec2graph.casmi.validation import add_inchikey14_column, molecule_disjoint_split, split_frame
from spectral_diffusion import Spec2GraphDiffusion, Spec2GraphDiffusionConfig

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Spec2Graph for CASMI 2026.")
    parser.add_argument("--train-parquet", required=True, type=Path)
    parser.add_argument("--assets-dir", required=True, type=Path, help="Output of casmi_prepare (adducts.json, structures.parquet).")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--holdout-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--num-decoder-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-atoms", type=int, default=96)
    parser.add_argument("--max-peaks", type=int, default=128)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--n-quantiles", type=int, default=32)
    parser.add_argument("--n-timesteps", type=int, default=1000)
    parser.add_argument("--projection-weight", type=float, default=1.0)
    parser.add_argument("--orthonormality-weight", type=float, default=0.0)
    parser.add_argument("--fingerprint-weight", type=float, default=0.5)
    parser.add_argument("--atom-count-weight", type=float, default=0.05)
    parser.add_argument("--atom-type-weight", type=float, default=0.5)
    parser.add_argument("--quantile-weight", type=float, default=1.0)
    parser.add_argument("--train-bond-sgno", action="store_true")
    parser.add_argument("--bond-hidden-dim", type=int, default=128)
    parser.add_argument("--bond-layers", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Development knob.")
    parser.add_argument("--sqrt-intensity", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - wiring only
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vocab = AdductVocabulary.load(args.assets_dir / "adducts.json")
    structures = None
    structures_path = args.assets_dir / "structures.parquet"
    if structures_path.exists():
        import pandas as pd

        structures = pd.read_parquet(structures_path)

    train = load_casmi_parquet(args.train_parquet, require_smiles=True)
    if args.max_train_rows:
        train = train.iloc[: args.max_train_rows]
    train = add_inchikey14_column(train)
    train_keys, holdout_keys = molecule_disjoint_split(train["inchikey14"], holdout_fraction=args.holdout_fraction, seed=args.seed)
    train_df, holdout_df = split_frame(train, holdout_keys)
    (args.output_dir / "holdout_keys.json").write_text(json.dumps(sorted(holdout_keys)))
    logger.info("Train spectra: %d (%d structures); holdout spectra: %d (%d structures)", len(train_df), len(train_keys), len(holdout_df), len(holdout_keys))

    peak_config = PeakPreprocessConfig(top_k_peaks=args.max_peaks, max_peaks=args.max_peaks, sqrt_intensity=args.sqrt_intensity)
    cache = SpectralCache(args.cache_dir, k=args.k, fingerprint_bits=args.fingerprint_bits)
    dataset = CasmiSpectrumDataset(
        train_df,
        cache=cache,
        adduct_vocab=vocab,
        k=args.k,
        max_atoms=args.max_atoms,
        fingerprint_bits=args.fingerprint_bits,
        n_quantiles=args.n_quantiles,
        peak_config=peak_config,
        structures=structures,
        include_adjacency=args.train_bond_sgno,
        precompute=True,
        n_jobs=max(1, args.num_workers),
    )
    logger.info("Dataset: %s", dataset.stats)

    core = Spec2GraphDiffusion(
        Spec2GraphDiffusionConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            k=args.k,
            max_atoms=args.max_atoms,
            max_peaks=args.max_peaks,
            fingerprint_dim=args.fingerprint_bits,
            enable_atom_count_head=True,
            enable_atom_type_head=True,
        )
    )
    model = CasmiModel(core, CasmiModelConfig(n_adducts=len(vocab), n_quantiles=args.n_quantiles)).to(args.device)
    trainer_config = CasmiTrainerConfig(
        n_timesteps=args.n_timesteps,
        projection_loss_weight=args.projection_weight,
        orthonormality_loss_weight=args.orthonormality_weight,
        fingerprint_loss_weight=args.fingerprint_weight,
        atom_count_loss_weight=args.atom_count_weight,
        atom_type_loss_weight=args.atom_type_weight,
        quantile_loss_weight=args.quantile_weight,
    )
    trainer = CasmiTrainer(model, trainer_config, device=args.device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    bond_sgno = bond_trainer = bond_optimiser = None
    if args.train_bond_sgno:
        bond_sgno = BondTypeSGNO(k=args.k, hidden_dim=args.bond_hidden_dim, num_layers=args.bond_layers).to(args.device)
        bond_trainer = BondTypeTrainer(bond_sgno, device=args.device)
        bond_optimiser = torch.optim.Adam(bond_sgno.parameters(), lr=1e-3)

    collate = make_casmi_collator(
        max_atoms=args.max_atoms, max_peaks=args.max_peaks, k=args.k, fingerprint_bits=args.fingerprint_bits,
        n_quantiles=args.n_quantiles, device="cpu",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers)

    def _to_device(batch):
        from dataclasses import replace

        tb = batch.batch
        moved_tb = type(tb)(**{
            k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in tb.__dict__.items()
        })
        return replace(
            batch,
            batch=moved_tb,
            adduct_index=batch.adduct_index.to(args.device),
            collision_energy=batch.collision_energy.to(args.device),
            collision_energy_mask=batch.collision_energy_mask.to(args.device),
            neutral_mass=batch.neutral_mass.to(args.device),
            polarity=batch.polarity.to(args.device),
            spectral_quantiles=batch.spectral_quantiles.to(args.device),
        )

    for epoch in range(1, args.epochs + 1):
        totals: dict[str, float] = {}
        n_batches = 0
        bond_losses: list[float] = []
        for batch in loader:
            batch = _to_device(batch)
            loss, components = trainer.train_step(optimiser, batch, return_components=True, grad_clip=args.grad_clip)
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + value
            totals["total"] = totals.get("total", 0.0) + loss
            n_batches += 1
            if bond_trainer is not None and batch.adjacencies is not None:
                bond_losses.append(
                    bond_trainer.train_step(
                        bond_optimiser, batch.batch.x_0, batch.batch.atom_mask, batch.batch.atom_type_targets.clamp(min=0), batch.adjacencies
                    )
                )
        summary = ", ".join(f"{k}={v / max(n_batches, 1):.4f}" for k, v in totals.items())
        logger.info("Epoch %d: %s", epoch, summary)
        if bond_losses:
            logger.info("Epoch %d: bond-type loss = %.4f", epoch, sum(bond_losses) / len(bond_losses))
        save_checkpoint(
            args.output_dir / f"epoch_{epoch:03d}.pt", model, trainer_config, vocab, epoch=epoch, optimizer=optimiser,
            extra={"max_peaks": args.max_peaks, "sqrt_intensity": args.sqrt_intensity},
        )
        if bond_sgno is not None:
            save_bond_sgno(args.output_dir / f"bond_sgno_epoch_{epoch:03d}.pt", bond_sgno)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
