"""End-to-end MassSpecGym benchmark harness.

Runs the sampling + decoding + metric aggregation loop described in
``AGENT_INSTRUCTIONS.md`` Task 7:

1. For each test spectrum, draw ``n_samples_per_spectrum`` samples via
   :meth:`DiffusionTrainer.sample` or :func:`ddim_sample`.
2. Decode each sample with the SGNO + ``ValencyDecoder``.
3. Rank canonical SMILES by sampling frequency.
4. Compute per-spectrum top-k accuracy / Tanimoto / MCES / validity.
5. Aggregate into :class:`BenchmarkResults` with bootstrap CIs.

Per-example results are streamed to a JSONL file so a partial run can
be resumed or inspected.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from spec2graph.data.collate import make_metadata_collator
from spec2graph.data.dataset import MassSpecGymDataset
from spec2graph.eval.ddim import ddim_sample
from spec2graph.eval.decode import batch_eigvecs_to_smiles
from spec2graph.eval.metrics import (
    BenchmarkResults,
    aggregate_per_example,
    canonicalise,
    rank_samples_by_frequency,
    top_k_accuracy,
    top_k_mces,
    top_k_tanimoto,
    validity_rate,
)
from spectral_diffusion import (
    DiffusionTrainer,
    SpectralGraphNeuralOperator,
    ValencyDecoder,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for :func:`benchmark_model`.

    ``limit`` and ``n_timesteps_override`` are explicit development
    knobs — the final reported numbers must set ``limit=None`` and
    leave ``n_timesteps_override=None`` so the full reverse diffusion
    schedule runs.
    """

    n_samples_per_spectrum: int = 100
    batch_size: int = 16
    sampler: str = "ddpm"  # or "ddim"
    ddim_n_steps: int = 50
    ddim_eta: float = 0.0
    decode_threshold: float = 0.3
    max_bond_order: int = 1
    limit: Optional[int] = None
    n_timesteps_override: Optional[int] = None
    use_atom_type_head: bool = True
    known_atom_count: bool = True
    progress: bool = True


@dataclass
class JSONLRecord:
    idx: int
    inchikey: str
    formula: str
    gt_smiles: str
    predictions: list[str]
    top_1_accuracy: float
    top_10_accuracy: float
    top_1_tanimoto: float
    top_10_tanimoto: float
    top_1_mces: float
    top_10_mces: float
    validity: float


def _extract_n_atoms(formula: str, fallback: int) -> int:
    """Sum heavy-atom counts from a formula string, falling back if empty."""
    from spec2graph.eval.decode import parse_formula
    try:
        counts = parse_formula(formula)
        total = sum(counts.values())
        return total if total > 0 else fallback
    except ValueError:
        return fallback


def _sample_spectrum(
    trainer: DiffusionTrainer,
    mz: torch.Tensor,
    intensity: torch.Tensor,
    n_samples: int,
    n_atoms: int,
    spectrum_mask: torch.Tensor,
    precursor_mz: torch.Tensor,
    *,
    sampler: str,
    ddim_n_steps: int,
    ddim_eta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``n_samples`` samples for a single spectrum.

    Tiles the 1-row conditioning to a length-``n_samples`` batch,
    runs the chosen sampler, and returns ``(eigvecs, atom_mask)``.
    """
    repeated_mz = mz.expand(n_samples, -1).contiguous()
    repeated_intensity = intensity.expand(n_samples, -1).contiguous()
    repeated_spec_mask = spectrum_mask.expand(n_samples, -1).contiguous()
    repeated_precursor = precursor_mz.expand(n_samples).contiguous()
    atom_mask = torch.ones(n_samples, n_atoms, dtype=torch.bool, device=mz.device)

    if sampler == "ddpm":
        eigvecs = trainer.sample(
            repeated_mz,
            repeated_intensity,
            n_atoms=n_atoms,
            atom_mask=atom_mask,
            spectrum_mask=repeated_spec_mask,
            precursor_mz=repeated_precursor,
        )
    elif sampler == "ddim":
        eigvecs = ddim_sample(
            trainer,
            repeated_mz,
            repeated_intensity,
            n_atoms=n_atoms,
            atom_mask=atom_mask,
            spectrum_mask=repeated_spec_mask,
            precursor_mz=repeated_precursor,
            n_steps=ddim_n_steps,
            eta=ddim_eta,
        )
    else:
        raise ValueError(f"Unknown sampler {sampler!r}; expected 'ddpm' or 'ddim'.")
    return eigvecs, atom_mask


def benchmark_model(
    trainer: DiffusionTrainer,
    sgno: SpectralGraphNeuralOperator,
    dataset: MassSpecGymDataset,
    valency_decoder: ValencyDecoder,
    *,
    config: Optional[BenchmarkConfig] = None,
    device: str | torch.device = "cpu",
    jsonl_path: Optional[str | Path] = None,
) -> BenchmarkResults:
    """Run the full benchmark loop and return aggregated metrics.

    See :class:`BenchmarkConfig` for the parameters. Per-example
    records are appended to ``jsonl_path`` if provided, one JSON object
    per line.
    """
    if config is None:
        config = BenchmarkConfig()

    saved_n_timesteps = None
    if config.n_timesteps_override is not None:
        if config.n_timesteps_override > trainer.n_timesteps:
            raise ValueError(
                f"n_timesteps_override={config.n_timesteps_override} exceeds "
                f"trainer.n_timesteps={trainer.n_timesteps}."
            )
        saved_n_timesteps = trainer.n_timesteps
        trainer.n_timesteps = config.n_timesteps_override

    n_examples = len(dataset)
    if config.limit is not None:
        n_examples = min(n_examples, config.limit)

    per_example: list[dict[str, Any]] = []

    iterator: range = range(n_examples)
    if config.progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="benchmark", unit="spectrum")
        except ImportError:
            pass

    jsonl_handle = None
    if jsonl_path is not None:
        jsonl_path = Path(jsonl_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = jsonl_path.open("w")

    trainer.model.eval()
    sgno.eval()

    try:
        for idx in iterator:
            sample = dataset[idx]
            gt_smiles = sample["smiles"]
            formula = sample["formula"]
            inchikey = sample["inchikey"]

            n_atoms = sample["n_atoms"]
            if config.known_atom_count:
                n_atoms = _extract_n_atoms(formula, fallback=n_atoms)

            mz = torch.from_numpy(sample["mz"]).to(device).unsqueeze(0)
            intensity = torch.from_numpy(sample["intensity"]).to(device).unsqueeze(0)
            spectrum_mask = torch.ones_like(mz, dtype=torch.bool)
            precursor_mz = torch.tensor(
                [sample["precursor_mz"]], device=device, dtype=torch.float32
            )

            # Sample n_samples eigenvector tensors for this spectrum.
            eigvecs, atom_masks = _sample_spectrum(
                trainer,
                mz,
                intensity,
                n_samples=config.n_samples_per_spectrum,
                n_atoms=n_atoms,
                spectrum_mask=spectrum_mask,
                precursor_mz=precursor_mz,
                sampler=config.sampler,
                ddim_n_steps=config.ddim_n_steps,
                ddim_eta=config.ddim_eta,
            )

            # Optional atom-type logits for the Hungarian decoder.
            atom_type_logits = None
            if config.use_atom_type_head and trainer.model.atom_type_head is not None:
                t_zero = torch.zeros(
                    config.n_samples_per_spectrum, device=device, dtype=torch.long
                )
                with torch.no_grad():
                    atom_type_logits = trainer.model.predict_atom_types(
                        eigvecs,
                        t_zero,
                        mz.expand(config.n_samples_per_spectrum, -1).contiguous(),
                        intensity.expand(config.n_samples_per_spectrum, -1).contiguous(),
                        atom_masks,
                        spectrum_mask.expand(config.n_samples_per_spectrum, -1).contiguous(),
                        precursor_mz.expand(config.n_samples_per_spectrum).contiguous(),
                    )

            predicted_smiles = batch_eigvecs_to_smiles(
                eigvecs=eigvecs,
                atom_masks=atom_masks,
                formulas=[formula] * config.n_samples_per_spectrum,
                atom_type_logits=atom_type_logits,
                sgno=sgno,
                valency_decoder=valency_decoder,
                threshold=config.decode_threshold,
                max_bond_order=config.max_bond_order,
            )

            ranked = rank_samples_by_frequency(predicted_smiles)

            record: dict[str, Any] = {
                "idx": idx,
                "inchikey": inchikey,
                "formula": formula,
                "gt_smiles": canonicalise(gt_smiles) or gt_smiles,
                "predictions": ranked[:10],  # only keep top-10 in the log
                "top_1_accuracy": top_k_accuracy(gt_smiles, ranked, k=1),
                "top_10_accuracy": top_k_accuracy(gt_smiles, ranked, k=10),
                "top_1_tanimoto": top_k_tanimoto(gt_smiles, ranked, k=1),
                "top_10_tanimoto": top_k_tanimoto(gt_smiles, ranked, k=10),
                "top_1_mces": top_k_mces(gt_smiles, ranked, k=1),
                "top_10_mces": top_k_mces(gt_smiles, ranked, k=10),
                "validity": validity_rate(predicted_smiles),
            }
            per_example.append(record)

            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps(record) + "\n")
                jsonl_handle.flush()

        return aggregate_per_example(
            per_example, n_samples_per_example=config.n_samples_per_spectrum
        )
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()
        if saved_n_timesteps is not None:
            trainer.n_timesteps = saved_n_timesteps


__all__ = ["BenchmarkConfig", "benchmark_model"]
