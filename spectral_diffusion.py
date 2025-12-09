"""
Spectral Diffusion Model for MS Spectrum to Graph Prediction.

This module implements a complete Spectral Diffusion Model that learns to generate
the Spectral Embedding (eigenvectors) of a molecule's Laplacian matrix, conditioned
on Mass Spectrum data.

Key Components:
- SpectralDataProcessor: Converts SMILES to eigenvectors with sign canonicalization
- FourierMzEmbedding: Fourier positional embeddings for m/z values
- Spec2GraphDiffusion: Transformer + diffusion denoiser model
- DiffusionTrainer: Training loop with DDPM schedule and sampling

Usage Example:
    Run `python spectral_diffusion.py` to see a complete example.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from rdkit import Chem


class SpectralDataProcessor:
    """Processes molecular SMILES to spectral graph representations."""

    def __init__(self, k: int = 8):
        """
        Initialize the spectral data processor.

        Args:
            k: Number of top eigenvectors to extract
        """
        self.k = k

    def smiles_to_adjacency(self, smiles: str) -> np.ndarray:
        """
        Convert a SMILES string to an adjacency matrix.

        Args:
            smiles: SMILES representation of the molecule

        Returns:
            Adjacency matrix as numpy array
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        num_atoms = mol.GetNumAtoms()
        adjacency = np.zeros((num_atoms, num_atoms), dtype=np.float32)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0

        return adjacency

    def compute_laplacian(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Compute the normalized Laplacian matrix.

        Args:
            adjacency: Adjacency matrix

        Returns:
            Normalized Laplacian matrix
        """
        degree = np.sum(adjacency, axis=1)
        # Avoid division by zero using epsilon for numerical stability
        eps = 1e-12
        degree_inv_sqrt = np.where(degree > eps, 1.0 / np.sqrt(degree), 0.0)
        D_inv_sqrt = np.diag(degree_inv_sqrt)

        # Normalized Laplacian: I - D^(-1/2) A D^(-1/2)
        identity = np.eye(adjacency.shape[0])
        laplacian = identity - D_inv_sqrt @ adjacency @ D_inv_sqrt

        return laplacian

    def extract_eigenvectors(
        self, laplacian: np.ndarray, canonicalize: bool = True
    ) -> np.ndarray:
        """
        Extract top k eigenvectors from the Laplacian matrix.

        Args:
            laplacian: Laplacian matrix
            canonicalize: Whether to apply sign canonicalization

        Returns:
            Matrix of shape (n_atoms, k) containing top k eigenvectors
        """
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        # Sort by eigenvalues (ascending - smallest first)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # For connected graphs, the smallest eigenvalue of the normalized Laplacian is zero, corresponding to the constant eigenvector.
        # Skip it if the smallest eigenvalue is close to zero.
        n_atoms = laplacian.shape[0]
        eps = 1e-9
        start_idx = 1 if eigenvalues[0] < eps else 0

        # Extract k eigenvectors starting from start_idx
        k_actual = min(self.k, n_atoms - start_idx)
        end_idx = start_idx + k_actual
        selected = eigenvectors[:, start_idx:end_idx]

        # Sign canonicalization: ensure first non-zero element is positive
        if canonicalize:
            for i in range(selected.shape[1]):
                col = selected[:, i]
                abs_col = np.abs(col)
                if np.any(abs_col > 1e-10):
                    idx = np.argmax(abs_col > 1e-10)
                else:
                    idx = np.argmax(abs_col)
                if col[idx] < 0:
                    selected[:, i] = -col

        # Pad with zeros if needed
        if selected.shape[1] < self.k:
            padding = np.zeros((n_atoms, self.k - selected.shape[1]))
            selected = np.hstack([selected, padding])

        return selected.astype(np.float32)

    def process_smiles(self, smiles: str) -> np.ndarray:
        """
        Full pipeline: SMILES to spectral embedding.

        Args:
            smiles: SMILES string

        Returns:
            Spectral embedding of shape (n_atoms, k)
        """
        adjacency = self.smiles_to_adjacency(smiles)
        laplacian = self.compute_laplacian(adjacency)
        eigenvectors = self.extract_eigenvectors(laplacian)
        return eigenvectors


class FourierMzEmbedding(nn.Module):
    """Fourier positional embedding for m/z values."""

    def __init__(self, d_model: int, max_mz: float = 2000.0, num_freqs: int = 64):
        """
        Initialize Fourier m/z embedding.

        Args:
            d_model: Output embedding dimension
            max_mz: Maximum m/z value for normalization
            num_freqs: Number of frequency components
        """
        super().__init__()
        self.d_model = d_model
        self.max_mz = max_mz
        self.num_freqs = num_freqs

        # Create frequency bands
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0), math.log(max_mz / 2.0), num_freqs // 2
            )
        )
        self.register_buffer("freqs", freqs)

        # Projection layer
        self.proj = nn.Linear(num_freqs, d_model)

    def forward(self, mz: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier embedding for m/z values.

        Args:
            mz: Tensor of m/z values, shape (batch, n_peaks)

        Returns:
            Embeddings of shape (batch, n_peaks, d_model)
        """
        # Normalize m/z values
        mz_norm = mz / self.max_mz

        # Compute Fourier features
        # Shape: (batch, n_peaks, num_freqs // 2)
        angles = mz_norm.unsqueeze(-1) * self.freqs * 2 * math.pi

        # Sin and cos features
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        # Project to d_model
        return self.proj(features)


class IntensityEmbedding(nn.Module):
    """Embedding layer for peak intensities."""

    def __init__(self, d_model: int):
        """
        Initialize intensity embedding.

        Args:
            d_model: Output embedding dimension
        """
        super().__init__()
        self.proj = nn.Linear(1, d_model)

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        """
        Compute embedding for intensity values.

        Args:
            intensity: Tensor of intensity values, shape (batch, n_peaks)

        Returns:
            Embeddings of shape (batch, n_peaks, d_model)
        """
        return self.proj(intensity.unsqueeze(-1))


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion."""

    def __init__(self, d_model: int, max_period: int = 10000):
        """
        Initialize timestep embedding.

        Args:
            d_model: Output embedding dimension
            max_period: Maximum period for sinusoidal encoding
        """
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute timestep embedding.

        Args:
            t: Timestep tensor, shape (batch,)

        Returns:
            Embedding of shape (batch, d_model)
        """
        half_dim = self.d_model // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        args = t.float().unsqueeze(-1) * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.d_model % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return self.mlp(embedding)


class Spec2GraphDiffusion(nn.Module):
    """
    Transformer-based diffusion model for spectrum to graph conversion.

    Takes mass spectrum data and predicts denoised spectral embeddings.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        k: int = 8,
        max_atoms: int = 64,
        max_peaks: int = 100,
        dropout: float = 0.1,
    ):
        """
        Initialize the Spec2Graph diffusion model.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            num_decoder_layers: Number of transformer decoder layers
            dim_feedforward: Feedforward dimension
            k: Number of eigenvectors
            max_atoms: Maximum number of atoms
            max_peaks: Maximum number of spectrum peaks
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.k = k
        self.max_atoms = max_atoms
        self.max_peaks = max_peaks

        # Spectrum encoding
        self.mz_embedding = FourierMzEmbedding(d_model)
        self.intensity_embedding = IntensityEmbedding(d_model)

        # Timestep embedding
        self.time_embedding = TimestepEmbedding(d_model)

        # Eigenvector embedding (project k-dim eigenvector to d_model)
        self.eigenvec_in = nn.Linear(k, d_model)
        self.eigenvec_out = nn.Linear(d_model, k)

        # Positional embeddings for atoms
        self.atom_pos_embedding = nn.Embedding(max_atoms, d_model)

        # Transformer encoder for spectrum
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.spectrum_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Transformer decoder for eigenvectors
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Layer norms
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

    def encode_spectrum(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode mass spectrum data.

        Args:
            mz: m/z values, shape (batch, n_peaks)
            intensity: Intensity values, shape (batch, n_peaks)
            mask: Optional mask for padding, shape (batch, n_peaks)

        Returns:
            Encoded spectrum, shape (batch, n_peaks, d_model)
        """
        # Combine m/z and intensity embeddings
        mz_emb = self.mz_embedding(mz)
        int_emb = self.intensity_embedding(intensity)
        spectrum_emb = mz_emb + int_emb

        # Apply transformer encoder
        if mask is not None:
            # Convert to attention mask format (True = ignore)
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None

        encoded = self.spectrum_encoder(
            spectrum_emb, src_key_padding_mask=src_key_padding_mask
        )
        return self.encoder_norm(encoded)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        spectrum_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: predict noise given noisy eigenvectors and spectrum.

        Args:
            x_t: Noisy eigenvectors, shape (batch, n_atoms, k)
            t: Timesteps, shape (batch,)
            mz: m/z values, shape (batch, n_peaks)
            intensity: Intensity values, shape (batch, n_peaks)
            atom_mask: Mask for valid atoms, shape (batch, n_atoms)
            spectrum_mask: Mask for valid peaks, shape (batch, n_peaks)

        Returns:
            Predicted noise, shape (batch, n_atoms, k)
        """
        batch_size, n_atoms, _ = x_t.shape

        # Encode spectrum
        memory = self.encode_spectrum(mz, intensity, spectrum_mask)

        # Get timestep embedding
        t_emb = self.time_embedding(t)  # (batch, d_model)

        # Embed noisy eigenvectors
        x_emb = self.eigenvec_in(x_t)  # (batch, n_atoms, d_model)

        # Add positional embeddings
        positions = torch.arange(n_atoms, device=x_t.device)
        pos_emb = self.atom_pos_embedding(positions)  # (n_atoms, d_model)
        x_emb = x_emb + pos_emb

        # Add timestep embedding
        x_emb = x_emb + t_emb.unsqueeze(1)

        # Prepare masks
        if atom_mask is not None:
            tgt_key_padding_mask = ~atom_mask
        else:
            tgt_key_padding_mask = None

        if spectrum_mask is not None:
            memory_key_padding_mask = ~spectrum_mask
        else:
            memory_key_padding_mask = None

        # Decode
        decoded = self.decoder(
            x_emb,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        decoded = self.decoder_norm(decoded)

        # Project back to eigenvector space
        output = self.eigenvec_out(decoded)

        return output


class DiffusionTrainer:
    """Training and sampling utilities for DDPM diffusion."""

    def __init__(
        self,
        model: Spec2GraphDiffusion,
        n_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        """
        Initialize the diffusion trainer.

        Args:
            model: The Spec2GraphDiffusion model
            n_timesteps: Number of diffusion timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            device: Device to use
        """
        self.model = model
        self.n_timesteps = n_timesteps
        self.device = device

        # DDPM schedule
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)

        # Pre-compute useful values
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)
        # Add small epsilon to denominator for numerical stability (avoids div by zero at t=0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod + 1e-8)
        )

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from q(x_t | x_0).

        Args:
            x_0: Clean data, shape (batch, n_atoms, k)
            t: Timesteps, shape (batch,)
            noise: Optional pre-computed noise

        Returns:
            Tuple of (noisy data, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        spectrum_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample from p(x_{t-1} | x_t).

        Args:
            x_t: Current noisy data
            t: Current timestep (integer)
            mz: m/z values
            intensity: Intensity values
            atom_mask: Atom mask
            spectrum_mask: Spectrum mask

        Returns:
            Denoised sample
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        # Predict noise
        with torch.no_grad():
            predicted_noise = self.model(
                x_t, t_tensor, mz, intensity, atom_mask, spectrum_mask
            )

        # Compute x_{t-1}
        alpha = self.alphas[t]
        alpha_cumprod = self.alpha_cumprod[t]
        beta = self.betas[t]

        # Mean of p(x_{t-1} | x_t)
        mean = (1.0 / torch.sqrt(alpha)) * (
            x_t - (beta / torch.sqrt(1.0 - alpha_cumprod)) * predicted_noise
        )

        if t > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t]
            x_t_minus_1 = mean + torch.sqrt(variance) * noise
        else:
            x_t_minus_1 = mean

        return x_t_minus_1

    @torch.no_grad()
    def sample(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        n_atoms: int,
        atom_mask: Optional[torch.Tensor] = None,
        spectrum_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate eigenvectors by reverse diffusion.

        Args:
            mz: m/z values, shape (batch, n_peaks)
            intensity: Intensity values, shape (batch, n_peaks)
            n_atoms: Number of atoms to generate
            atom_mask: Optional atom mask
            spectrum_mask: Optional spectrum mask

        Returns:
            Generated eigenvectors, shape (batch, n_atoms, k)
        """
        batch_size = mz.shape[0]
        k = self.model.k

        # Start from pure noise
        x_t = torch.randn(batch_size, n_atoms, k, device=self.device)

        # Reverse diffusion
        for t in reversed(range(self.n_timesteps)):
            x_t = self.p_sample(x_t, t, mz, intensity, atom_mask, spectrum_mask)

        return x_t

    def compute_loss(
        self,
        x_0: torch.Tensor,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        spectrum_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            x_0: Clean eigenvectors, shape (batch, n_atoms, k)
            mz: m/z values, shape (batch, n_peaks)
            intensity: Intensity values, shape (batch, n_peaks)
            atom_mask: Optional atom mask
            spectrum_mask: Optional spectrum mask

        Returns:
            MSE loss
        """
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device)

        # Add noise
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)

        # Predict noise
        predicted_noise = self.model(
            x_t, t, mz, intensity, atom_mask, spectrum_mask
        )

        # Compute loss (only on valid atoms if mask provided)
        if atom_mask is not None:
            mask = atom_mask.unsqueeze(-1).float()
            loss = F.mse_loss(predicted_noise * mask, noise * mask)
        else:
            loss = F.mse_loss(predicted_noise, noise)

        return loss

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        x_0: torch.Tensor,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        spectrum_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Single training step.

        Args:
            optimizer: Optimizer
            x_0: Clean eigenvectors
            mz: m/z values
            intensity: Intensity values
            atom_mask: Optional atom mask
            spectrum_mask: Optional spectrum mask

        Returns:
            Loss value
        """
        self.model.train()
        optimizer.zero_grad()

        loss = self.compute_loss(x_0, mz, intensity, atom_mask, spectrum_mask)
        loss.backward()
        optimizer.step()

        return loss.item()


def create_benzene_demo_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create demo data for benzene molecule with simulated spectrum.

    Returns:
        Tuple of (eigenvectors, mz_values, intensities)
    """
    # Process benzene SMILES
    processor = SpectralDataProcessor(k=8)
    smiles = "c1ccccc1"  # Benzene

    eigenvectors = processor.process_smiles(smiles)

    # Simulated mass spectrum for benzene (MW ~78)
    # Major peaks: molecular ion (78), loss of H (77), C4H3 (51), C3H3 (39) -- common fragmentation patterns for benzene in mass spectrometry (m/z values in parentheses)
    mz_values = np.array([39.0, 50.0, 51.0, 52.0, 63.0, 74.0, 76.0, 77.0, 78.0, 79.0])
    intensities = np.array([0.3, 0.1, 0.4, 0.15, 0.1, 0.05, 0.2, 0.6, 1.0, 0.1])

    return eigenvectors, mz_values, intensities


def run_demo():
    """Run a demonstration of the Spectral Diffusion Model."""
    print("=" * 60)
    print("Spectral Diffusion Model Demo")
    print("=" * 60)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create demo data
    print("\n1. Creating demo data for benzene molecule...")
    eigenvectors, mz_values, intensities = create_benzene_demo_data()
    print(f"   Eigenvector shape: {eigenvectors.shape}")
    print(f"   Number of spectrum peaks: {len(mz_values)}")

    # Convert to tensors
    x_0 = torch.tensor(eigenvectors, dtype=torch.float32).unsqueeze(0).to(device)
    mz = torch.tensor(mz_values, dtype=torch.float32).unsqueeze(0).to(device)
    intensity = torch.tensor(intensities, dtype=torch.float32).unsqueeze(0).to(device)

    # Create model
    print("\n2. Creating Spec2GraphDiffusion model...")
    model = Spec2GraphDiffusion(
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        k=8,
        max_atoms=32,
        max_peaks=50,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Create trainer with fewer timesteps for demo
    print("\n3. Creating diffusion trainer...")
    trainer = DiffusionTrainer(
        model=model,
        n_timesteps=100,  # Reduced for demo
        device=device,
    )

    # Training loop
    print("\n4. Training for a few steps (demo only)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_steps = 50
    for step in range(n_steps):
        loss = trainer.train_step(optimizer, x_0, mz, intensity)
        if (step + 1) % 10 == 0:
            print(f"   Step {step + 1}/{n_steps}, Loss: {loss:.4f}")

    # Sample
    print("\n5. Generating sample eigenvectors...")
    model.eval()
    n_atoms = eigenvectors.shape[0]
    generated = trainer.sample(mz, intensity, n_atoms)

    print(f"   Generated shape: {generated.shape}")
    print(f"   Original eigenvectors (first atom):\n   {eigenvectors[0, :]}")
    print(f"   Generated eigenvectors (first atom):\n   {generated[0, 0, :].cpu().numpy()}")

    # Compute similarity
    orig_flat = torch.tensor(eigenvectors).flatten()
    gen_flat = generated[0].cpu().flatten()
    cosine_sim = F.cosine_similarity(orig_flat.unsqueeze(0), gen_flat.unsqueeze(0))
    print(f"\n   Cosine similarity (original vs generated): {cosine_sim.item():.4f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
