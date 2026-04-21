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


def _sample_super_batch(
    trainer: DiffusionTrainer,
    mz: torch.Tensor,
    intensity: torch.Tensor,
    spectrum_mask: torch.Tensor,
    precursor_mz: torch.Tensor,
    atom_mask: torch.Tensor,
    max_n_atoms: int,
    *,
    sampler: str,
    ddim_n_steps: int,
    ddim_eta: float,
) -> torch.Tensor:
    """Draw a batch of samples. Shape: ``(super_batch, max_n_atoms, k)``.

    Conditioning inputs are expected to already be at super-batch scale
    — callers tile per-spectrum conditioning ``n_samples`` times and
    concatenate across spectra before calling.
    """
    if sampler == "ddpm":
        return trainer.sample(
            mz,
            intensity,
            n_atoms=max_n_atoms,
            atom_mask=atom_mask,
            spectrum_mask=spectrum_mask,
            precursor_mz=precursor_mz,
        )
    if sampler == "ddim":
        return ddim_sample(
            trainer,
            mz,
            intensity,
            n_atoms=max_n_atoms,
            atom_mask=atom_mask,
            spectrum_mask=spectrum_mask,
            precursor_mz=precursor_mz,
            n_steps=ddim_n_steps,
            eta=ddim_eta,
        )
    raise ValueError(f"Unknown sampler {sampler!r}; expected 'ddpm' or 'ddim'.")


def _tile_spectra(
    samples: list[dict[str, Any]],
    n_samples: int,
    *,
    known_atom_count: bool,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Build super-batch conditioning tensors for a set of spectra.

    Each sample is tiled ``n_samples`` times so all of its candidate
    draws share conditioning. All spectra in the batch are padded to the
    longest peak list and the largest heavy-atom count in the batch.
    """
    batch_n_atoms = []
    for sample in samples:
        n = sample["n_atoms"]
        if known_atom_count:
            n = _extract_n_atoms(sample["formula"], fallback=n)
        batch_n_atoms.append(int(n))
    max_n_atoms = max(batch_n_atoms)

    max_peaks = max(int(s["mz"].shape[0]) for s in samples)

    n_spectra = len(samples)
    super_batch = n_spectra * n_samples

    mz = torch.zeros(super_batch, max_peaks, device=device, dtype=torch.float32)
    intensity = torch.zeros_like(mz)
    spectrum_mask = torch.zeros(super_batch, max_peaks, device=device, dtype=torch.bool)
    atom_mask = torch.zeros(super_batch, max_n_atoms, device=device, dtype=torch.bool)
    precursor_mz = torch.zeros(super_batch, device=device, dtype=torch.float32)

    for i, sample in enumerate(samples):
        start = i * n_samples
        stop = start + n_samples
        p = int(sample["mz"].shape[0])
        mz[start:stop, :p] = torch.from_numpy(sample["mz"]).to(device)
        intensity[start:stop, :p] = torch.from_numpy(sample["intensity"]).to(device)
        spectrum_mask[start:stop, :p] = True
        atom_mask[start:stop, : batch_n_atoms[i]] = True
        precursor_mz[start:stop] = sample["precursor_mz"]

    return {
        "mz": mz,
        "intensity": intensity,
        "spectrum_mask": spectrum_mask,
        "atom_mask": atom_mask,
        "precursor_mz": precursor_mz,
        "max_n_atoms": max_n_atoms,
        "batch_n_atoms": batch_n_atoms,
    }


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

    # Resumable JSONL: if the file exists, read existing idx values and
    # skip them. New records append to the existing file. This is what
    # the docstring advertised; "w" mode would have silently discarded
    # prior partial runs.
    already_done: set[int] = set()
    jsonl_handle = None
    if jsonl_path is not None:
        jsonl_path = Path(jsonl_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if jsonl_path.exists():
            for line in jsonl_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec.get("idx"), int):
                        already_done.add(rec["idx"])
                        per_example.append(rec)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSONL line in %s; record will "
                        "be overwritten on next complete run.",
                        jsonl_path,
                    )
            if already_done:
                logger.info(
                    "Resuming: %d previously computed examples will be skipped.",
                    len(already_done),
                )
        jsonl_handle = jsonl_path.open("a")

    trainer.model.eval()
    sgno.eval()

    # Super-batch: process ``batch_size`` spectra at a time. Each spectrum
    # contributes ``n_samples_per_spectrum`` rows to the sampler call,
    # padded to the common max n_atoms of its super-batch. Decoding is
    # still per-spectrum because the ValencyDecoder is not batched.
    batch_size = max(1, int(config.batch_size))
    n_samples = config.n_samples_per_spectrum

    pending_ids: list[int] = []
    pending_samples: list[dict[str, Any]] = []

    def _flush() -> None:
        if not pending_samples:
            return
        conditioning = _tile_spectra(
            pending_samples,
            n_samples=n_samples,
            known_atom_count=config.known_atom_count,
            device=device,
        )
        super_eigvecs = _sample_super_batch(
            trainer,
            mz=conditioning["mz"],
            intensity=conditioning["intensity"],
            spectrum_mask=conditioning["spectrum_mask"],
            precursor_mz=conditioning["precursor_mz"],
            atom_mask=conditioning["atom_mask"],
            max_n_atoms=conditioning["max_n_atoms"],
            sampler=config.sampler,
            ddim_n_steps=config.ddim_n_steps,
            ddim_eta=config.ddim_eta,
        )

        super_atom_type_logits = None
        if config.use_atom_type_head and trainer.model.atom_type_head is not None:
            t_zero = torch.zeros(
                super_eigvecs.shape[0], device=device, dtype=torch.long
            )
            with torch.no_grad():
                super_atom_type_logits = trainer.model.predict_atom_types(
                    super_eigvecs,
                    t_zero,
                    conditioning["mz"],
                    conditioning["intensity"],
                    conditioning["atom_mask"],
                    conditioning["spectrum_mask"],
                    conditioning["precursor_mz"],
                )

        # Slice the super-batch back into per-spectrum groups.
        for i, sample in enumerate(pending_samples):
            start = i * n_samples
            stop = start + n_samples
            spectrum_eigvecs = super_eigvecs[start:stop]
            spectrum_atom_masks = conditioning["atom_mask"][start:stop]
            spectrum_logits = (
                super_atom_type_logits[start:stop]
                if super_atom_type_logits is not None
                else None
            )
            predicted_smiles = batch_eigvecs_to_smiles(
                eigvecs=spectrum_eigvecs,
                atom_masks=spectrum_atom_masks,
                formulas=[sample["formula"]] * n_samples,
                atom_type_logits=spectrum_logits,
                sgno=sgno,
                valency_decoder=valency_decoder,
                threshold=config.decode_threshold,
                max_bond_order=config.max_bond_order,
            )
            ranked = rank_samples_by_frequency(predicted_smiles)
            gt_smiles = sample["smiles"]
            record: dict[str, Any] = {
                "idx": pending_ids[i],
                "inchikey": sample["inchikey"],
                "formula": sample["formula"],
                "gt_smiles": canonicalise(gt_smiles) or gt_smiles,
                "predictions": ranked[:10],
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

        pending_ids.clear()
        pending_samples.clear()

    try:
        for idx in iterator:
            if idx in already_done:
                continue
            pending_ids.append(idx)
            pending_samples.append(dataset[idx])
            if len(pending_samples) >= batch_size:
                _flush()
        _flush()

        return aggregate_per_example(
            per_example, n_samples_per_example=n_samples
        )
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()
        if saved_n_timesteps is not None:
            trainer.n_timesteps = saved_n_timesteps


__all__ = ["BenchmarkConfig", "benchmark_model"]
