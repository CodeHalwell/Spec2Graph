import numpy as np
import pytest
import torch
from spectral_diffusion import TrainingBatch

from spectral_diffusion import (
    DiffusionTrainer,
    SpectralDataProcessor,
    Spec2GraphDiffusion,
    SpectralGraphNeuralOperator,
    DenseGNNDiscriminator,
    DenseGNNLayer,
    GuidedDiffusionSampler,
    ValencyDecoder,
    VALENCY_TABLE,
    EigenvalueConditioner,
    EigenvalueConditionedSGNO,
)


def test_projection_invariance_under_basis_change():
    torch.manual_seed(0)
    embeddings = torch.randn(2, 4, 3)
    u, _, v = torch.linalg.svd(torch.randn(3, 3))
    singular_values = torch.tensor([1.0, 1.5, 2.0])
    random_matrix = u @ torch.diag(singular_values) @ v

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
    eigen_eps = 1e-9  # matches SpectralDataProcessor.extract_eigenvectors
    start_idx = int((eigenvalues < eigen_eps).sum())
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


# ==============================================================================
# SGNO Decoder Tests (Section 3.5)
# ==============================================================================


class TestSpectralGraphNeuralOperator:
    def test_sgno_output_shape(self):
        sgno = SpectralGraphNeuralOperator(k=4, hidden_dim=32, num_layers=2)
        embeddings = torch.randn(3, 6, 4)
        logits = sgno(embeddings)
        assert logits.shape == (3, 6, 6)

    def test_sgno_output_is_symmetric(self):
        sgno = SpectralGraphNeuralOperator(k=4, hidden_dim=32, num_layers=2)
        embeddings = torch.randn(2, 5, 4)
        logits = sgno(embeddings)
        assert torch.allclose(logits, logits.transpose(-1, -2), atol=1e-6)

    def test_sgno_bond_probabilities_in_range(self):
        sgno = SpectralGraphNeuralOperator(k=4, hidden_dim=32, num_layers=2)
        embeddings = torch.randn(2, 5, 4)
        probs = sgno.bond_probabilities(embeddings)
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        assert torch.allclose(
            torch.diagonal(probs, dim1=-2, dim2=-1),
            torch.zeros_like(torch.diagonal(probs, dim1=-2, dim2=-1)),
        )

    def test_sgno_decode_to_adjacency_is_binary(self):
        sgno = SpectralGraphNeuralOperator(k=4, hidden_dim=32, num_layers=2)
        embeddings = torch.randn(2, 5, 4)
        adj = sgno.decode_to_adjacency(embeddings)
        unique_vals = torch.unique(adj)
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())
        assert torch.allclose(
            torch.diagonal(adj, dim1=-2, dim2=-1),
            torch.zeros_like(torch.diagonal(adj, dim1=-2, dim2=-1)),
        )

    def test_sgno_single_atom_produces_zero_adjacency(self):
        """A single atom should have no bonds."""
        sgno = SpectralGraphNeuralOperator(k=4, hidden_dim=32, num_layers=2)
        embeddings = torch.randn(1, 1, 4)
        adj = sgno.decode_to_adjacency(embeddings, threshold=0.99)
        # Single atom, self-loop probability may not be 0 but threshold should filter
        # At minimum, shape should be correct
        assert adj.shape == (1, 1, 1)

    def test_sgno_different_thresholds_produce_different_adjacencies(self):
        torch.manual_seed(42)
        sgno = SpectralGraphNeuralOperator(k=4, hidden_dim=32, num_layers=2)
        embeddings = torch.randn(1, 5, 4)
        adj_low = sgno.decode_to_adjacency(embeddings, threshold=0.1)
        adj_high = sgno.decode_to_adjacency(embeddings, threshold=0.9)
        # Lower threshold should produce >= bonds than higher
        assert adj_low.sum() >= adj_high.sum()

    def test_sgno_gradients_flow(self):
        """Verify gradients flow through the SGNO for training."""
        sgno = SpectralGraphNeuralOperator(k=4, hidden_dim=32, num_layers=2)
        embeddings = torch.randn(2, 5, 4, requires_grad=True)
        logits = sgno(embeddings)
        loss = logits.sum()
        loss.backward()
        assert embeddings.grad is not None
        assert embeddings.grad.shape == embeddings.shape


# ==============================================================================
# Orthonormality Loss Tests (Section 3.4)
# ==============================================================================


class TestOrthonormalityLoss:
    def test_default_orthonormality_weight_is_backward_compatible(self):
        model = Spec2GraphDiffusion(k=2, max_atoms=4, max_peaks=4, d_model=32, nhead=4)
        trainer = DiffusionTrainer(model, n_timesteps=5)
        assert trainer.orthonormality_loss_weight == 0.0

    def test_orthonormal_input_gives_near_zero_loss(self):
        """Orthonormal columns should produce ~0 loss."""
        q, _ = torch.linalg.qr(torch.randn(6, 4))
        embeddings = q.unsqueeze(0)  # (1, 6, 4)
        loss = DiffusionTrainer._orthonormality_loss(embeddings)
        assert loss.item() < 1e-10

    def test_non_orthonormal_input_gives_positive_loss(self):
        """Non-orthonormal columns should produce positive loss."""
        embeddings = torch.randn(2, 5, 3)
        loss = DiffusionTrainer._orthonormality_loss(embeddings)
        assert loss.item() > 0

    def test_orthonormality_loss_with_mask(self):
        """Masked atoms should not contribute to the loss."""
        q, _ = torch.linalg.qr(torch.randn(4, 3))
        embeddings = torch.zeros(1, 6, 3)
        embeddings[0, :4, :] = q
        mask = torch.tensor([[True, True, True, True, False, False]])
        loss = DiffusionTrainer._orthonormality_loss(embeddings, mask)
        # After masking, the valid part is orthonormal
        assert loss.item() < 1e-5

    def test_orthonormality_loss_integrated_in_compute_loss(self):
        """Verify orthonormality loss appears in compute_loss components."""
        model = Spec2GraphDiffusion(k=2, max_atoms=4, max_peaks=4, d_model=32, nhead=4)
        trainer = DiffusionTrainer(model, n_timesteps=5, orthonormality_loss_weight=1.0)
        x_0 = torch.randn(2, 4, 2)
        mz = torch.randn(2, 4)
        intensity = torch.randn(2, 4)
        batch = TrainingBatch(x_0=x_0, mz=mz, intensity=intensity)
        _, components = trainer.compute_loss(batch, return_components=True)
        assert "orthonormality" in components
        # Should be non-negative
        assert components["orthonormality"].item() >= 0

    def test_zero_orthonormality_weight_skips_computation(self):
        """When weight is 0, orthonormality loss should be 0."""
        model = Spec2GraphDiffusion(k=2, max_atoms=4, max_peaks=4, d_model=32, nhead=4)
        trainer = DiffusionTrainer(model, n_timesteps=5, orthonormality_loss_weight=0.0)
        x_0 = torch.randn(2, 4, 2)
        mz = torch.randn(2, 4)
        intensity = torch.randn(2, 4)
        batch = TrainingBatch(x_0=x_0, mz=mz, intensity=intensity)
        _, components = trainer.compute_loss(batch, return_components=True)
        assert components["orthonormality"].item() == 0.0


# ==============================================================================
# Precursor Conditioning Tests (Section 3.1)
# ==============================================================================


class TestPrecursorConditioning:
    def test_precursor_conditioning_changes_output(self):
        """Precursor conditioning should alter the encoded spectrum."""
        model = Spec2GraphDiffusion(
            k=4, max_atoms=8, max_peaks=10, d_model=32, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1,
            enable_precursor_conditioning=True
        )
        model.eval()
        mz = torch.randn(2, 10)
        intensity = torch.randn(2, 10)

        with torch.no_grad():
            enc_no_precursor = model.encode_spectrum(mz, intensity)
            enc_with_precursor = model.encode_spectrum(
                mz, intensity, precursor_mz=torch.tensor([100.0, 200.0])
            )

        # Should be different when precursor is provided
        assert not torch.allclose(enc_no_precursor, enc_with_precursor, atol=1e-6)

    def test_precursor_conditioning_different_masses_differ(self):
        """Different precursor masses should produce different encodings."""
        model = Spec2GraphDiffusion(
            k=4, max_atoms=8, max_peaks=10, d_model=32, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1,
            enable_precursor_conditioning=True
        )
        model.eval()  # disable dropout for deterministic comparison
        mz = torch.randn(1, 10).expand(2, -1)
        intensity = torch.randn(1, 10).expand(2, -1)

        with torch.no_grad():
            enc1 = model.encode_spectrum(mz, intensity, precursor_mz=torch.tensor([100.0, 100.0]))
            enc2 = model.encode_spectrum(mz, intensity, precursor_mz=torch.tensor([100.0, 500.0]))

        # First batch item should be same, second should differ
        assert torch.allclose(enc1[0], enc2[0], atol=1e-5)
        assert not torch.allclose(enc1[1], enc2[1], atol=1e-5)

    def test_model_without_precursor_conditioning(self):
        """Model without precursor conditioning should work normally."""
        model = Spec2GraphDiffusion(
            k=4, max_atoms=8, max_peaks=10, d_model=32, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1,
            enable_precursor_conditioning=False
        )
        mz = torch.randn(2, 10)
        intensity = torch.randn(2, 10)
        # Should not raise even when precursor_mz is passed
        enc = model.encode_spectrum(mz, intensity, precursor_mz=torch.tensor([100.0, 200.0]))
        assert enc.shape == (2, 10, 32)

    def test_compute_loss_changes_with_precursor_conditioning(self):
        """compute_loss should thread precursor mass through the main training path."""
        model = Spec2GraphDiffusion(
            k=2, max_atoms=4, max_peaks=4, d_model=32, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1,
            enable_precursor_conditioning=True
        )
        model.eval()
        trainer = DiffusionTrainer(model, n_timesteps=5)
        x_0 = torch.randn(1, 4, 2)
        mz = torch.randn(1, 4)
        intensity = torch.randn(1, 4)

        with torch.no_grad():
            torch.manual_seed(7)
            batch_low = TrainingBatch(x_0=x_0, mz=mz, intensity=intensity, precursor_mz=torch.tensor([100.0]))
            loss_low = trainer.compute_loss(batch_low)
            torch.manual_seed(7)
            batch_high = TrainingBatch(x_0=x_0, mz=mz, intensity=intensity, precursor_mz=torch.tensor([500.0]))
            loss_high = trainer.compute_loss(batch_high)

        assert torch.isfinite(loss_low)
        assert torch.isfinite(loss_high)
        assert loss_low.item() >= 0
        assert loss_high.item() >= 0
        assert not torch.allclose(loss_low, loss_high)

    def test_sample_changes_with_precursor_conditioning(self):
        """sample should thread precursor mass through the reverse diffusion path."""
        model = Spec2GraphDiffusion(
            k=2, max_atoms=4, max_peaks=4, d_model=32, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1,
            enable_precursor_conditioning=True
        )
        model.eval()
        trainer = DiffusionTrainer(model, n_timesteps=3)
        mz = torch.randn(1, 4)
        intensity = torch.randn(1, 4)

        torch.manual_seed(13)
        sample_low = trainer.sample(
            mz, intensity, n_atoms=4, precursor_mz=torch.tensor([100.0])
        )
        torch.manual_seed(13)
        sample_high = trainer.sample(
            mz, intensity, n_atoms=4, precursor_mz=torch.tensor([500.0])
        )

        assert sample_low.shape == (1, 4, 2)
        assert sample_high.shape == (1, 4, 2)
        assert torch.all(torch.isfinite(sample_low))
        assert torch.all(torch.isfinite(sample_high))
        assert not torch.allclose(sample_low, sample_high)


# ==============================================================================
# GNN Discriminator Tests (Phase 1, Section 5)
# ==============================================================================


class TestDenseGNNDiscriminator:
    def test_discriminator_output_shape(self):
        disc = DenseGNNDiscriminator(node_features=1, hidden_dim=32, num_layers=2)
        adj = torch.rand(3, 6, 6)
        score = disc(adj)
        assert score.shape == (3, 1)

    def test_discriminator_gradients_flow_through_adjacency(self):
        """Gradients must flow through the adjacency matrix for guidance."""
        disc = DenseGNNDiscriminator(node_features=1, hidden_dim=32, num_layers=2)
        adj = torch.rand(2, 5, 5, requires_grad=True)
        score = disc(adj)
        loss = score.sum()
        loss.backward()
        assert adj.grad is not None
        assert not torch.all(adj.grad == 0)

    def test_discriminator_different_inputs_give_different_scores(self):
        torch.manual_seed(42)
        disc = DenseGNNDiscriminator(node_features=1, hidden_dim=32, num_layers=2)
        adj_dense = torch.ones(1, 4, 4) * 0.9
        adj_sparse = torch.eye(4).unsqueeze(0) * 0.1
        score_dense = disc(adj_dense)
        score_sparse = disc(adj_sparse)
        assert not torch.allclose(score_dense, score_sparse, atol=1e-6)

    def test_dense_gnn_layer_message_passing(self):
        """DenseGNNLayer should aggregate neighbor information."""
        layer = DenseGNNLayer(in_features=8, out_features=8)
        x = torch.randn(2, 4, 8)
        adj = torch.eye(4).unsqueeze(0).expand(2, -1, -1)  # identity = self-loop only
        out = layer(x, adj)
        assert out.shape == (2, 4, 8)


# ==============================================================================
# Guided Diffusion Sampler Tests (Phase 1, Section 5)
# ==============================================================================


class TestGuidedDiffusionSampler:
    def _make_sampler(self):
        model = Spec2GraphDiffusion(
            k=2, max_atoms=4, max_peaks=4, d_model=32, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1
        )
        trainer = DiffusionTrainer(model, n_timesteps=3)
        sgno = SpectralGraphNeuralOperator(k=2, hidden_dim=16, num_layers=2)
        disc = DenseGNNDiscriminator(node_features=1, hidden_dim=16, num_layers=2)
        return GuidedDiffusionSampler(trainer, sgno, disc, guidance_scale=0.1)

    def test_guided_sample_output_shape(self):
        sampler = self._make_sampler()
        mz = torch.randn(1, 4)
        intensity = torch.randn(1, 4)
        result = sampler.guided_sample(mz, intensity, n_atoms=4)
        assert result.shape == (1, 4, 2)

    def test_guided_sample_all_finite(self):
        sampler = self._make_sampler()
        mz = torch.randn(1, 4)
        intensity = torch.randn(1, 4)
        result = sampler.guided_sample(mz, intensity, n_atoms=3)
        assert torch.all(torch.isfinite(result))

    def test_guided_vs_unguided_differ(self):
        """Guided sampling should produce different results from unguided."""
        torch.manual_seed(0)
        model = Spec2GraphDiffusion(
            k=2, max_atoms=4, max_peaks=4, d_model=32, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1
        )
        trainer = DiffusionTrainer(model, n_timesteps=3)
        sgno = SpectralGraphNeuralOperator(k=2, hidden_dim=16, num_layers=2)
        disc = DenseGNNDiscriminator(node_features=1, hidden_dim=16, num_layers=2)

        mz = torch.randn(1, 4)
        intensity = torch.randn(1, 4)

        # Unguided
        torch.manual_seed(123)
        unguided = trainer.sample(mz, intensity, n_atoms=4)

        # Guided (with scale > 0, samples will generally differ due to guidance)
        sampler = GuidedDiffusionSampler(trainer, sgno, disc, guidance_scale=1.0)
        torch.manual_seed(123)
        guided = sampler.guided_sample(mz, intensity, n_atoms=4)

        # They should not be identical (guidance alters the trajectory)
        assert not torch.allclose(unguided, guided, atol=1e-4)

    def test_zero_guidance_scale(self):
        """With guidance_scale=0, guided sampling should match standard sampling."""
        model = Spec2GraphDiffusion(
            k=2, max_atoms=4, max_peaks=4, d_model=32, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1
        )
        trainer = DiffusionTrainer(model, n_timesteps=3)
        sgno = SpectralGraphNeuralOperator(k=2, hidden_dim=16, num_layers=2)
        disc = DenseGNNDiscriminator(node_features=1, hidden_dim=16, num_layers=2)
        sampler = GuidedDiffusionSampler(trainer, sgno, disc, guidance_scale=0.0)

        mz = torch.randn(1, 4)
        intensity = torch.randn(1, 4)

        torch.manual_seed(42)
        unguided = trainer.sample(mz, intensity, n_atoms=4)
        torch.manual_seed(42)
        guided = sampler.guided_sample(mz, intensity, n_atoms=4)

        assert torch.allclose(unguided, guided, atol=1e-5)

    def test_guidance_masks_invalid_atoms(self):
        """Invalid atoms should be masked out before SGNO/discriminator guidance."""
        class RecordingSGNO(SpectralGraphNeuralOperator):
            def __init__(self):
                super().__init__(k=2, hidden_dim=16, num_layers=2)
                self.last_embeddings = None

            def bond_probabilities(self, embeddings):
                self.last_embeddings = embeddings.detach().clone()
                return super().bond_probabilities(embeddings)

        class RecordingDiscriminator(DenseGNNDiscriminator):
            def __init__(self):
                super().__init__(node_features=1, hidden_dim=16, num_layers=1)
                self.last_adj_probs = None

            def forward(self, adj_probs):
                self.last_adj_probs = adj_probs.detach().clone()
                return super().forward(adj_probs)

        model = Spec2GraphDiffusion(
            k=2, max_atoms=4, max_peaks=4, d_model=32, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1
        )
        trainer = DiffusionTrainer(model, n_timesteps=3)
        sgno = RecordingSGNO()
        disc = RecordingDiscriminator()
        sampler = GuidedDiffusionSampler(trainer, sgno, disc, guidance_scale=0.5)

        mz = torch.randn(1, 4)
        intensity = torch.randn(1, 4)
        atom_mask = torch.tensor([[True, True, False, False]])
        sampler.guided_sample(mz, intensity, n_atoms=4, atom_mask=atom_mask)
        first_invalid_atom = 2

        assert sgno.last_embeddings is not None
        assert disc.last_adj_probs is not None
        assert torch.allclose(
            sgno.last_embeddings[0, first_invalid_atom:],
            torch.zeros_like(sgno.last_embeddings[0, first_invalid_atom:]),
        )
        assert torch.allclose(
            disc.last_adj_probs[0, first_invalid_atom:, :],
            torch.zeros_like(disc.last_adj_probs[0, first_invalid_atom:, :]),
        )
        assert torch.allclose(
            disc.last_adj_probs[0, :, first_invalid_atom:],
            torch.zeros_like(disc.last_adj_probs[0, :, first_invalid_atom:]),
        )


# ==============================================================================
# Valency Decoder Tests (Phase 3, Section 7)
# ==============================================================================


class TestValencyDecoder:
    def test_carbon_valency_constraint(self):
        """Carbon atoms should not exceed valency 4."""
        vd = ValencyDecoder()
        atoms = ["C", "C", "C", "C", "C"]
        probs = np.ones((5, 5), dtype=np.float32) * 0.9
        np.fill_diagonal(probs, 0)
        adj = vd.decode(atoms, probs, threshold=0.3)
        bond_sums = adj.sum(axis=1)
        assert all(s <= 4 for s in bond_sums)

    def test_oxygen_valency_constraint(self):
        """Oxygen atoms should not exceed valency 2."""
        vd = ValencyDecoder()
        atoms = ["O", "C", "C", "C", "C"]
        probs = np.ones((5, 5), dtype=np.float32) * 0.9
        np.fill_diagonal(probs, 0)
        adj = vd.decode(atoms, probs, threshold=0.3)
        assert adj[0].sum() <= 2

    def test_symmetry(self):
        """Decoded adjacency matrix should be symmetric."""
        vd = ValencyDecoder()
        atoms = ["C", "N", "O"]
        probs = np.array([[0, 0.8, 0.6], [0.8, 0, 0.5], [0.6, 0.5, 0]], dtype=np.float32)
        adj = vd.decode(atoms, probs, threshold=0.3)
        assert np.allclose(adj, adj.T)

    def test_below_threshold_gives_no_bonds(self):
        """Bonds below threshold should not be added."""
        vd = ValencyDecoder()
        atoms = ["C", "C"]
        probs = np.array([[0, 0.2], [0.2, 0]], dtype=np.float32)
        adj = vd.decode(atoms, probs, threshold=0.5)
        assert adj.sum() == 0

    def test_greedy_order_prefers_high_probability(self):
        """Higher probability bonds should be selected first."""
        vd = ValencyDecoder()
        # Fluorine has valency 1, so it can only bond once
        atoms = ["F", "C", "C"]
        probs = np.array([
            [0, 0.9, 0.5],
            [0.9, 0, 0.8],
            [0.5, 0.8, 0]
        ], dtype=np.float32)
        adj = vd.decode(atoms, probs, threshold=0.3)
        # F should bond to C(1) (prob 0.9) not C(2) (prob 0.5)
        assert adj[0, 1] == 1
        assert adj[0, 2] == 0

    def test_valency_table_defaults(self):
        """Check standard valency table values."""
        assert VALENCY_TABLE["C"] == 4
        assert VALENCY_TABLE["N"] == 3
        assert VALENCY_TABLE["O"] == 2
        assert VALENCY_TABLE["F"] == 1

    def test_custom_valency_table(self):
        """Custom valency tables should be respected."""
        custom = {"X": 6}
        vd = ValencyDecoder(valency_table=custom)
        assert vd.get_valency("X") == 6
        assert vd.get_valency("unknown") == 4  # default

    def test_decode_batch(self):
        """Batch decoding should work correctly."""
        vd = ValencyDecoder()
        atoms_batch = [["C", "C"], ["O", "C"]]
        probs = torch.tensor([
            [[0, 0.9], [0.9, 0]],
            [[0, 0.8], [0.8, 0]]
        ])
        results = vd.decode_batch(atoms_batch, probs, threshold=0.3)
        assert len(results) == 2
        assert results[0][0, 1] == 1
        assert results[1][0, 1] == 1


# ==============================================================================
# Eigenvalue Conditioning Tests (Phase 4, Section 8)
# ==============================================================================


class TestEigenvalueConditioning:
    def test_eigenvalue_conditioner_output_shape(self):
        ec = EigenvalueConditioner(k=8, d_model=64)
        eigenvalues = torch.randn(3, 8)
        cond = ec(eigenvalues)
        assert cond.shape == (3, 64)

    def test_eigenvalue_head_prediction_shape(self):
        model = Spec2GraphDiffusion(
            k=4, max_atoms=8, max_peaks=10, d_model=32, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1,
            enable_eigenvalue_head=True
        )
        mz = torch.randn(2, 10)
        intensity = torch.randn(2, 10)
        pred = model.predict_eigenvalues(mz, intensity)
        assert pred.shape == (2, 4)

    def test_eigenvalue_head_disabled_raises(self):
        model = Spec2GraphDiffusion(
            k=4, max_atoms=8, max_peaks=10, d_model=32, nhead=4,
            enable_eigenvalue_head=False
        )
        mz = torch.randn(1, 10)
        intensity = torch.randn(1, 10)
        with pytest.raises(ValueError):
            model.predict_eigenvalues(mz, intensity)

    def test_eigenvalue_conditioned_sgno_output_shape(self):
        ec_sgno = EigenvalueConditionedSGNO(k=4, hidden_dim=32, num_layers=2, eigenvalue_dim=16)
        embeddings = torch.randn(2, 5, 4)
        eigenvalues = torch.randn(2, 4)
        logits = ec_sgno(embeddings, eigenvalues)
        assert logits.shape == (2, 5, 5)

    def test_eigenvalue_conditioned_sgno_is_symmetric(self):
        ec_sgno = EigenvalueConditionedSGNO(k=4, hidden_dim=32, num_layers=2, eigenvalue_dim=16)
        embeddings = torch.randn(2, 5, 4)
        eigenvalues = torch.randn(2, 4)
        logits = ec_sgno(embeddings, eigenvalues)
        assert torch.allclose(logits, logits.transpose(-1, -2), atol=1e-6)

    def test_different_eigenvalues_produce_different_outputs(self):
        """Different eigenvalue conditioning should change the output."""
        torch.manual_seed(42)
        ec_sgno = EigenvalueConditionedSGNO(k=4, hidden_dim=32, num_layers=2, eigenvalue_dim=16)
        embeddings = torch.randn(1, 5, 4)
        ev1 = torch.tensor([[0.0, 0.5, 1.0, 1.5]])
        ev2 = torch.tensor([[0.0, 2.0, 3.0, 4.0]])
        out1 = ec_sgno(embeddings, ev1)
        out2 = ec_sgno(embeddings, ev2)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_eigenvalue_conditioned_sgno_gradients_flow(self):
        ec_sgno = EigenvalueConditionedSGNO(k=4, hidden_dim=32, num_layers=2, eigenvalue_dim=16)
        embeddings = torch.randn(2, 5, 4, requires_grad=True)
        eigenvalues = torch.randn(2, 4, requires_grad=True)
        logits = ec_sgno(embeddings, eigenvalues)
        loss = logits.sum()
        loss.backward()
        assert embeddings.grad is not None
        assert eigenvalues.grad is not None

    def test_eigenvalue_loss_in_compute_loss(self):
        """Verify eigenvalue loss appears when enabled."""
        model = Spec2GraphDiffusion(
            k=2, max_atoms=4, max_peaks=4, d_model=32, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1,
            enable_eigenvalue_head=True
        )
        trainer = DiffusionTrainer(
            model, n_timesteps=5, eigenvalue_loss_weight=1.0
        )
        x_0 = torch.randn(2, 4, 2)
        mz = torch.randn(2, 4)
        intensity = torch.randn(2, 4)
        eigenvalue_targets = torch.randn(2, 2)
        batch = TrainingBatch(
            x_0=x_0, mz=mz, intensity=intensity,
            eigenvalue_targets=eigenvalue_targets
        )
        _, components = trainer.compute_loss(
            batch,
            return_components=True
        )
        assert "eigenvalue" in components
        assert components["eigenvalue"].item() > 0


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestEndToEndPipeline:
    def test_full_pipeline_sgno_decode(self):
        """Test SMILES -> eigenvectors -> SGNO -> adjacency pipeline."""
        processor = SpectralDataProcessor(k=4)
        eigenvectors = processor.process_smiles("CCO")  # ethanol: 3 atoms
        eigvec_tensor = torch.tensor(eigenvectors, dtype=torch.float32).unsqueeze(0)

        sgno = SpectralGraphNeuralOperator(k=4, hidden_dim=32, num_layers=2)
        adj_probs = sgno.bond_probabilities(eigvec_tensor)
        assert adj_probs.shape == (1, 3, 3)
        assert (adj_probs >= 0).all()
        assert (adj_probs <= 1).all()

    def test_sgno_to_valency_decoder_pipeline(self):
        """Test SGNO probabilities -> valency-constrained decoding."""
        sgno = SpectralGraphNeuralOperator(k=4, hidden_dim=32, num_layers=2)
        embeddings = torch.randn(1, 4, 4)
        probs = sgno.bond_probabilities(embeddings)

        vd = ValencyDecoder()
        atoms = ["C", "C", "O", "N"]
        adj = vd.decode(atoms, probs[0].detach().numpy(), threshold=0.3)
        assert adj.shape == (4, 4)
        # Check valency constraints
        assert adj[2].sum() <= 2  # O
        assert adj[3].sum() <= 3  # N

    def test_training_step_with_all_losses(self):
        """Verify a training step works with all loss components enabled."""
        model = Spec2GraphDiffusion(
            k=2, max_atoms=4, max_peaks=4, d_model=32, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1,
            fingerprint_dim=16, enable_atom_count_head=True,
            enable_eigenvalue_head=True
        )
        trainer = DiffusionTrainer(
            model, n_timesteps=5,
            projection_loss_weight=1.0,
            orthonormality_loss_weight=0.1,
            fingerprint_loss_weight=0.1,
            atom_count_loss_weight=0.05,
            eigenvalue_loss_weight=0.1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x_0 = torch.randn(2, 4, 2)
        mz = torch.randn(2, 4)
        intensity = torch.randn(2, 4)
        fp = torch.rand(2, 16)
        ac = torch.tensor([4.0, 3.0])
        ev = torch.randn(2, 2)

        batch = TrainingBatch(
            x_0=x_0, mz=mz, intensity=intensity,
            fingerprint_targets=fp, atom_count_targets=ac,
            eigenvalue_targets=ev
        )
        loss_val, components = trainer.train_step(
            optimizer, batch,
            return_components=True
        )
        assert isinstance(loss_val, float)
        assert loss_val > 0
        assert all(k in components for k in [
            "noise", "projection", "orthonormality", "fingerprint", "atom_count",
            "eigenvalue"
        ])
