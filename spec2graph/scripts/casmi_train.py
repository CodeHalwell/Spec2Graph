"""Train the conditioned Spec2Graph model (and optionally the bond-type SGNO) on a molecule-disjoint split.

python -m spec2graph.scripts.casmi_train --train-parquet data/train.parquet --assets-dir assets/casmi \\
    --cache-dir ~/.cache/spec2graph-casmi --output-dir runs/casmi --epochs 10 --train-bond-sgno
Use --heads-only for a quick proxy run (encoder + fingerprint/atom-count/quantile heads, no denoiser).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from spec2graph.casmi import data, validation
from spec2graph.casmi.adducts import AdductVocabulary
from spec2graph.casmi.conditioning import CasmiModel, CasmiModelConfig, CasmiTrainer, CasmiTrainerConfig, save_checkpoint
from spec2graph.casmi.denovo import BondTypeSGNO, BondTypeTrainer, save_bond_sgno
from spec2graph.casmi.spectral import SpectralCache
from spectral_diffusion import Spec2GraphDiffusion, Spec2GraphDiffusionConfig

log = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--train-parquet", "--assets-dir", "--cache-dir", "--output-dir"):
        p.add_argument(name, required=True, type=Path)
    ints = dict(epochs=10, batch_size=64, d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024,
                k=8, max_atoms=96, max_peaks=128, fingerprint_bits=2048, n_quantiles=32, n_timesteps=1000, bond_hidden_dim=128,
                bond_layers=3, num_workers=0, seed=0)
    floats = dict(holdout_fraction=0.05, lr=1e-4, projection_weight=1.0, orthonormality_weight=0.0, fingerprint_weight=0.5,
                  atom_count_weight=0.05, atom_type_weight=0.5, quantile_weight=1.0, grad_clip=1.0)
    for name, default in {**ints, **floats}.items():
        p.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=default)
    p.add_argument("--max-train-rows", type=int, default=None, help="development knob")
    for flag in ("--train-bond-sgno", "--sqrt-intensity", "--heads-only"):
        p.add_argument(flag, action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def build(a: argparse.Namespace, vocab: AdductVocabulary) -> tuple[CasmiTrainer, CasmiTrainerConfig]:
    core = Spec2GraphDiffusion(Spec2GraphDiffusionConfig(
        d_model=a.d_model, nhead=a.nhead, num_encoder_layers=a.num_encoder_layers, num_decoder_layers=a.num_decoder_layers,
        dim_feedforward=a.dim_feedforward, k=a.k, max_atoms=a.max_atoms, max_peaks=a.max_peaks, fingerprint_dim=a.fingerprint_bits,
        enable_atom_count_head=True, enable_atom_type_head=True))
    config = CasmiTrainerConfig(
        n_timesteps=a.n_timesteps, projection_loss_weight=a.projection_weight, orthonormality_loss_weight=a.orthonormality_weight,
        fingerprint_loss_weight=a.fingerprint_weight, atom_count_loss_weight=a.atom_count_weight, atom_type_loss_weight=a.atom_type_weight,
        quantile_loss_weight=a.quantile_weight, heads_only=a.heads_only)
    return CasmiTrainer(CasmiModel(core, CasmiModelConfig(n_adducts=len(vocab), n_quantiles=a.n_quantiles)), config, a.device), config


def train_epoch(loader, trainer: CasmiTrainer, optimiser, bond, bond_optimiser, a: argparse.Namespace) -> str:
    """One pass over the loader; returns the mean loss components as a log string."""
    totals, bond_losses, n = {}, [], 0
    for batch in loader:
        batch = batch.to(a.device)
        loss, parts = trainer.train_step(optimiser, batch, return_components=True, grad_clip=a.grad_clip)
        for key, value in {**parts, "total": loss}.items():
            totals[key] = totals.get(key, 0.0) + value
        n += 1
        if bond and batch.adjacencies is not None:
            tb = batch.core
            bond_losses.append(bond.step(bond_optimiser, tb.x_0, tb.atom_mask, tb.atom_type_targets.clamp(min=0), batch.adjacencies))
    summary = ", ".join(f"{k}={v / max(n, 1):.4f}" for k, v in totals.items())
    return summary + (f", bond={sum(bond_losses) / len(bond_losses):.4f}" if bond_losses else "")


def main(argv=None) -> int:  # pragma: no cover - wiring only
    a = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    torch.manual_seed(a.seed)
    a.output_dir.mkdir(parents=True, exist_ok=True)

    vocab = AdductVocabulary.load(a.assets_dir / "adducts.json")
    table = pd.read_parquet(a.assets_dir / "structures.parquet") if (a.assets_dir / "structures.parquet").exists() else None
    train = validation.with_keys(data.load(a.train_parquet, require_smiles=True).iloc[: a.max_train_rows])
    _, holdout_keys = validation.split_keys(train["inchikey14"], a.holdout_fraction, a.seed)
    train_df, holdout_df = validation.split_frame(train, holdout_keys)
    (a.output_dir / "holdout_keys.json").write_text(json.dumps(sorted(holdout_keys)))
    log.info("train spectra %d, holdout spectra %d (%d structures)", len(train_df), len(holdout_df), len(holdout_keys))

    cache = SpectralCache(a.cache_dir, k=a.k, fingerprint_bits=a.fingerprint_bits)
    dataset = data.Dataset(train_df, cache, vocab, max_atoms=a.max_atoms, n_quantiles=a.n_quantiles, table=table,
                           peaks=data.Peaks(top_k=a.max_peaks, sqrt_intensity=a.sqrt_intensity), adjacency=a.train_bond_sgno, n_jobs=max(1, a.num_workers))
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
                        collate_fn=data.Collator(a.max_atoms, a.max_peaks, a.k, a.fingerprint_bits))
    trainer, config = build(a, vocab)
    optimiser = torch.optim.AdamW(trainer.wrapper.parameters(), lr=a.lr)
    bond = BondTypeTrainer(BondTypeSGNO(a.k, a.bond_hidden_dim, a.bond_layers).to(a.device)) if a.train_bond_sgno else None
    bond_optimiser = torch.optim.Adam(bond.sgno.parameters(), lr=1e-3) if bond else None

    for epoch in range(1, a.epochs + 1):
        log.info("epoch %d: %s", epoch, train_epoch(loader, trainer, optimiser, bond, bond_optimiser, a))
        save_checkpoint(a.output_dir / f"epoch_{epoch:03d}.pt", trainer.wrapper, config, vocab, epoch=epoch, max_peaks=a.max_peaks, sqrt_intensity=a.sqrt_intensity)
        if bond:
            save_bond_sgno(a.output_dir / f"bond_sgno_epoch_{epoch:03d}.pt", bond.sgno)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
