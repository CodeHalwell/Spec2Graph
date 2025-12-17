import numpy as np
import pytest
import torch

from spectral_diffusion import (
    DiffusionTrainer,
    SpectralDataProcessor,
    Spec2GraphDiffusion,
)


def test_projection_invariance_under_basis_change():
    torch.manual_seed(0)
    embeddings = torch.randn(2, 4, 3)
    random_matrix = torch.randn(3, 3)
    random_matrix = random_matrix + torch.eye(3)  # ensure invertible

    proj_original = DiffusionTrainer.projection_from_embeddings(embeddings)
    proj_rotated = DiffusionTrainer.projection_from_embeddings(
        embeddings @ random_matrix
    )

    assert torch.allclose(proj_original, proj_rotated, atol=1e-5)


def test_masked_mse_normalisation_respects_padding():
    pred_short = torch.tensor([[[1.0], [2.0], [0.0]]])
    target_short = torch.zeros_like(pred_short)
    mask_short = torch.tensor([[1, 1, 0]], dtype=torch.bool)

    pred_long = torch.tensor([[[1.0], [2.0], [0.0], [0.0], [0.0]]])
    target_long = torch.zeros_like(pred_long)
    mask_long = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool)

    loss_short = DiffusionTrainer._masked_mse(pred_short, target_short, mask_short.unsqueeze(-1))
    loss_long = DiffusionTrainer._masked_mse(pred_long, target_long, mask_long.unsqueeze(-1))

    assert pytest.approx(loss_short.item(), rel=1e-5) == loss_long.item()


def test_extract_eigenvectors_skips_all_zero_eigenvalues_for_disconnected_graph():
    adjacency = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.float32,
    )
    processor = SpectralDataProcessor(k=3)
    laplacian = processor.compute_laplacian(adjacency)

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    start_idx = int((eigenvalues < 1e-9).sum())
    expected = eigenvectors[:, start_idx : start_idx + 3]

    selected = processor.extract_eigenvectors(laplacian, canonicalize=False)

    assert np.allclose(selected[:, : expected.shape[1]], expected)


def test_forward_raises_when_atoms_exceed_max():
    model = Spec2GraphDiffusion(k=2, max_atoms=2, max_peaks=4)
    x_t = torch.randn(1, 3, 2)
    t = torch.zeros(1, dtype=torch.long)
    mz = torch.zeros(1, 4)
    intensity = torch.zeros(1, 4)

    with pytest.raises(ValueError):
        model(x_t, t, mz, intensity)


def test_q_sample_reconstruct_is_stable():
    model = Spec2GraphDiffusion(k=2, max_atoms=4, max_peaks=4)
    trainer = DiffusionTrainer(model, n_timesteps=5)
    x0 = torch.randn(1, 4, 2)
    t = torch.tensor([1])
    x_t, noise = trainer.q_sample(x0, t)
    reconstructed = trainer._reconstruct_x0(
        x_t,
        noise,
        trainer.sqrt_alpha_cumprod_clamped[t].view(1, 1, 1),
        trainer.sqrt_one_minus_alpha_cumprod_clamped[t].view(1, 1, 1),
    )

    assert torch.all(torch.isfinite(reconstructed))
    assert torch.allclose(x0, reconstructed, atol=1e-5)


def test_bond_weighting_distinguishes_double_bond():
    single = SpectralDataProcessor(bond_weighting="order").smiles_to_adjacency("CC")
    double = SpectralDataProcessor(bond_weighting="order").smiles_to_adjacency("C=C")

    assert single[0, 1] == pytest.approx(1.0)
    assert double[0, 1] == pytest.approx(2.0)
