"""End-to-end smoke test for :mod:`spec2graph.eval.benchmark`.

Trains for 2 steps on 2 examples, samples 2 molecules per spectrum on
2 test examples, runs :func:`benchmark_model`. Must complete in under
30 seconds on CPU and produce finite metric values. Catches shape /
wiring regressions across the whole pipeline.
"""

from __future__ import annotations

from pathlib import Path

import json
import pandas as pd
import pytest
import torch

pytest.importorskip("rdkit")
pytest.importorskip("scipy")

from spec2graph.data.collate import make_training_batch_collator
from spec2graph.data.dataset import MassSpecGymDataset
from spec2graph.eval.benchmark import BenchmarkConfig, benchmark_model
from spec2graph.eval.metrics import BenchmarkResults
from spec2graph.train.sgno_trainer import SGNOTrainer
from spectral_diffusion import (
    DiffusionTrainer,
    Spec2GraphDiffusion,
    Spec2GraphDiffusionConfig,
    SpectralGraphNeuralOperator,
    TrainerConfig,
    ValencyDecoder,
)


_ROWS = [
    {
        "smiles": "CCO",
        "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "formula": "C2H6O",
        "mzs": "[20.0, 25.0, 27.0, 29.0, 40.0, 45.0]",
        "intensities": "[0.1, 0.3, 0.5, 1.0, 0.4, 0.2]",
        "precursor_mz": 47.0,
        "adduct": "[M+H]+",
        "instrument_type": "Orbitrap",
        "split": "train",
    },
    {
        "smiles": "c1ccccc1",
        "inchikey": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
        "formula": "C6H6",
        "mzs": "[30.0, 50.0, 52.0, 60.0, 75.0, 77.0]",
        "intensities": "[0.2, 0.9, 0.4, 0.3, 0.6, 1.0]",
        "precursor_mz": 79.0,
        "adduct": "[M+H]+",
        "instrument_type": "Orbitrap",
        "split": "test",
    },
    {
        "smiles": "CC(=O)O",
        "inchikey": "QTBSBXVTEAMEQO-UHFFFAOYSA-N",
        "formula": "C2H4O2",
        "mzs": "[20.0, 43.0, 45.0, 55.0, 60.0]",
        "intensities": "[0.1, 1.0, 0.3, 0.5, 0.2]",
        "precursor_mz": 61.0,
        "adduct": "[M+H]+",
        "instrument_type": "Orbitrap",
        "split": "test",
    },
]


def _df() -> pd.DataFrame:
    return pd.DataFrame(_ROWS)


def test_end_to_end_benchmark_smoke(tmp_path: Path):
    df = _df()
    train_ds = MassSpecGymDataset(
        split="train",
        cache_dir=tmp_path,
        k=4,
        max_atoms=8,
        max_peaks=16,
        top_k_peaks=16,
        fingerprint_bits=64,
        dataframe=df,
        include_adjacency=True,
    )
    test_ds = MassSpecGymDataset(
        split="test",
        cache_dir=tmp_path,
        k=4,
        max_atoms=8,
        max_peaks=16,
        top_k_peaks=16,
        fingerprint_bits=64,
        dataframe=df,
    )
    assert len(train_ds) >= 1
    assert len(test_ds) >= 1

    config = Spec2GraphDiffusionConfig(
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        k=4,
        max_atoms=8,
        max_peaks=16,
        dropout=0.0,
        fingerprint_dim=64,
        enable_atom_count_head=True,
        enable_atom_type_head=True,
    )
    model = Spec2GraphDiffusion(config)
    trainer = DiffusionTrainer(
        model=model,
        config=TrainerConfig(
            n_timesteps=10,
            atom_type_loss_weight=1.0,
            fingerprint_loss_weight=0.1,
            atom_count_loss_weight=0.05,
        ),
    )
    sgno = SpectralGraphNeuralOperator(k=4, hidden_dim=16, num_layers=2)
    sgno_trainer = SGNOTrainer(sgno)
    sgno_optim = torch.optim.Adam(sgno.parameters(), lr=1e-2)

    collator = make_training_batch_collator(
        max_atoms=8, max_peaks=16, k=4, fingerprint_bits=64
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Two "epochs", one step each, on the tiny train split.
    batch = collator([train_ds[0]])
    for _ in range(2):
        loss = trainer.train_step(optim, batch)
        assert torch.isfinite(torch.tensor(loss))

    # Two SGNO training steps on the same example.
    sample = train_ds[0]
    eigvec_batch = torch.from_numpy(sample["eigvecs"]).unsqueeze(0)
    atom_mask = torch.ones(1, sample["n_atoms"], dtype=torch.bool)
    for _ in range(2):
        sgno_loss = sgno_trainer.train_step(
            sgno_optim, eigvec_batch, atom_mask, [sample["adjacency"]]
        )
        assert sgno_loss >= 0

    # Run the benchmark on the test split with a tiny sample count.
    bench_config = BenchmarkConfig(
        n_samples_per_spectrum=2,
        sampler="ddim",
        ddim_n_steps=5,
        ddim_eta=0.0,
        decode_threshold=0.3,
        limit=len(test_ds),
        progress=False,
    )
    jsonl_path = tmp_path / "results.jsonl"
    results = benchmark_model(
        trainer=trainer,
        sgno=sgno,
        dataset=test_ds,
        valency_decoder=ValencyDecoder(),
        config=bench_config,
        jsonl_path=jsonl_path,
    )

    assert isinstance(results, BenchmarkResults)
    # Every scalar must be finite.
    for key, value in results.to_dict().items():
        if isinstance(value, float):
            assert torch.isfinite(torch.tensor(value)), f"{key} is not finite: {value}"
    assert results.n_examples == len(test_ds)
    assert results.n_samples_per_example == 2

    # JSONL must contain one line per test example.
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == len(test_ds)
    for line in lines:
        parsed = json.loads(line)
        for key in [
            "idx",
            "inchikey",
            "formula",
            "gt_smiles",
            "predictions",
            "top_1_accuracy",
        ]:
            assert key in parsed
