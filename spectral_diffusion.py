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
from typing import Tuple, Optional, Union
import warnings

# Numerical floor to avoid divide-by-zero when reconstructing clean samples
PROJECTION_EPS = 1e-8
# Threshold above which projection matrices become memory-heavy (n_atoms^2)
PROJECTION_WARNING_THRESHOLD = 256
# Emit projection warning only once to avoid log spam during training
PROJECTION_WARNING_EMITTED = False

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    _HAS_RDKIT = True
except ImportError:  # pragma: no cover - handled at runtime with clear error
    Chem = None
    DataStructs = None
    AllChem = None
    _HAS_RDKIT = False


class SpectralDataProcessor:
    """Processes molecular SMILES to spectral graph representations."""

    def __init__(self, k: int = 8, bond_weighting: str = "order"):
        """
        Initialize the spectral data processor.

        Args:
            k: Number of top eigenvectors to extract
            bond_weighting: How to weight bonds when building the adjacency.
                - "unweighted": all bonds weight 1.0
                - "order": use bond order (single=1, double=2, triple=3, aromatic=1.5)
                - "aromatic": aromatic=1.5, else 1.0
        """
        self.k = k
        self.bond_weighting = bond_weighting

    @staticmethod
    def _require_rdkit():
        if not _HAS_RDKIT:
            raise ImportError(
                "RDKit is required for molecular processing. Please install RDKit to use SpectralDataProcessor."
            )

    def smiles_to_adjacency(self, smiles: str) -> np.ndarray:
        """
        Convert a SMILES string to an adjacency matrix.

        Args:
            smiles: SMILES representation of the molecule

        Returns:
            Adjacency matrix as numpy array
        """
        self._require_rdkit()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        num_atoms = mol.GetNumAtoms()
        adjacency = np.zeros((num_atoms, num_atoms), dtype=np.float32)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if self.bond_weighting == "unweighted":
                weight = 1.0
            elif self.bond_weighting == "order":
                weight = float(bond.GetBondTypeAsDouble())
            elif self.bond_weighting == "aromatic":
                weight = 1.5 if bond.GetIsAromatic() else 1.0
            else:
                raise ValueError(
                    f"Unsupported bond_weighting='{self.bond_weighting}'. "
                    "Choose from ['unweighted', 'order', 'aromatic']."
                )
            adjacency[i, j] = weight
            adjacency[j, i] = weight

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
        start_idx = int((eigenvalues < eps).sum())

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

    def projection_matrix(self, eigenvectors: np.ndarray) -> np.ndarray:
        """
        Compute the spectral projection matrix P_k = V_k V_k^T to obtain a
        subspace-invariant target (robust to sign flips and degenerate eigenspaces).

        Args:
            eigenvectors: Eigenvector matrix of shape (n_atoms, k)

        Returns:
            Projection matrix of shape (n_atoms, n_atoms)

        Note:
            This validates only dimensionality (2D); callers are responsible for
            providing semantically correct (n_atoms, k) inputs.
        """
        if eigenvectors.ndim != 2:
            raise ValueError(
                f"eigenvectors must be 2D (n_atoms, k), got {eigenvectors.ndim}D with shape {eigenvectors.shape}"
            )
        return (eigenvectors @ eigenvectors.T).astype(np.float32)

    def smiles_to_fingerprint(
        self, smiles: str, n_bits: int = 128, radius: int = 2
    ) -> np.ndarray:
        """Compute Morgan fingerprint for a SMILES string."""
        self._require_rdkit()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def process_smiles(
        self, smiles: str, return_projection: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Full pipeline: SMILES to spectral embedding (and optionally projection).

        Args:
            smiles: SMILES string
            return_projection: If True, also return P_k = V_k V_k^T

        Returns:
            Spectral embedding of shape (n_atoms, k) or a tuple
            (eigenvectors, projection_matrix)
        """
        adjacency = self.smiles_to_adjacency(smiles)
        laplacian = self.compute_laplacian(adjacency)
        eigenvectors = self.extract_eigenvectors(laplacian)

        if return_projection:
            projection = self.projection_matrix(eigenvectors)
            return eigenvectors, projection

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
        fingerprint_dim: int = 0,
        enable_atom_count_head: bool = False,
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
            fingerprint_dim: Optional size of fingerprint prediction head (0 to disable)
            enable_atom_count_head: Whether to add an atom-count prediction head for sampling
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

        # Optional auxiliary heads
        if fingerprint_dim > 0:
            self.fingerprint_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, fingerprint_dim),
            )
        else:
            self.fingerprint_head = None

        if enable_atom_count_head:
            self.atom_count_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )
        else:
            self.atom_count_head = None

    @staticmethod
    def _validate_mask(mask: torch.Tensor, name: str) -> None:
        """Ensure masks follow the True=valid convention and have at least one valid token."""
        if mask.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean tensor with True indicating valid entries.")
        if mask.dim() < 2:
            raise ValueError(f"{name} must have shape (batch, length); got {mask.shape}.")
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError(f"{name} must contain at least one valid element per batch item.")

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
            mask: Optional boolean mask for padding (True = valid), shape (batch, n_peaks)

        Returns:
            Encoded spectrum, shape (batch, n_peaks, d_model)
        """
        # Combine m/z and intensity embeddings
        mz_emb = self.mz_embedding(mz)
        int_emb = self.intensity_embedding(intensity)
        spectrum_emb = mz_emb + int_emb

        # Apply transformer encoder
        if mask is not None:
            self._validate_mask(mask, "spectrum_mask")
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
            atom_mask: Boolean mask for valid atoms (True = valid), shape (batch, n_atoms)
            spectrum_mask: Boolean mask for valid peaks (True = valid), shape (batch, n_peaks)

        Returns:
            Predicted noise, shape (batch, n_atoms, k)
        """
        batch_size, n_atoms, _ = x_t.shape

        if n_atoms > self.max_atoms:
            raise ValueError(
                f"Number of atoms ({n_atoms}) exceeds configured max_atoms ({self.max_atoms}). "
                "Increase max_atoms or truncate input."
            )

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
            self._validate_mask(atom_mask, "atom_mask")
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

    def _pool_spectrum(
        self, encoded: torch.Tensor, spectrum_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mean-pool spectrum embeddings respecting padding mask."""
        if spectrum_mask is None:
            return encoded.mean(dim=1)
        spectrum_mask = spectrum_mask.float().unsqueeze(-1)
        denom = torch.clamp(spectrum_mask.sum(dim=1), min=PROJECTION_EPS)
        return (encoded * spectrum_mask).sum(dim=1) / denom

    def predict_fingerprint(
        self, mz: torch.Tensor, intensity: torch.Tensor, spectrum_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict fingerprint logits from spectrum input."""
        if self.fingerprint_head is None:
            raise ValueError("Fingerprint head is disabled. Set fingerprint_dim>0 to enable.")
        encoded = self.encode_spectrum(mz, intensity, spectrum_mask)
        pooled = self._pool_spectrum(encoded, spectrum_mask)
        return self.fingerprint_head(pooled)

    def predict_atom_count(
        self, mz: torch.Tensor, intensity: torch.Tensor, spectrum_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict atom count (regression logit) from spectrum input."""
        if self.atom_count_head is None:
            raise ValueError(
                "Atom count head is disabled. Set enable_atom_count_head=True to enable predictions."
            )
        encoded = self.encode_spectrum(mz, intensity, spectrum_mask)
        pooled = self._pool_spectrum(encoded, spectrum_mask)
        return self.atom_count_head(pooled).squeeze(-1)


class DiffusionTrainer:
    """Training and sampling utilities for DDPM diffusion."""

    def __init__(
        self,
        model: Spec2GraphDiffusion,
        n_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cpu",
        projection_loss_weight: float = 1.0,
        fingerprint_loss_weight: float = 0.0,
        atom_count_loss_weight: float = 0.0,
    ):
        """
        Initialize the diffusion trainer.

        Args:
            model: The Spec2GraphDiffusion model
            n_timesteps: Number of diffusion timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            device: Device to use
            projection_loss_weight: Weight for subspace-invariant projection loss
            fingerprint_loss_weight: Weight for optional fingerprint auxiliary loss
            atom_count_loss_weight: Weight for optional atom-count auxiliary loss
        """
        self.model = model
        self.n_timesteps = n_timesteps
        self.device = device
        self.projection_loss_weight = projection_loss_weight
        self.fingerprint_loss_weight = fingerprint_loss_weight
        self.atom_count_loss_weight = atom_count_loss_weight

        # DDPM schedule
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)

        # Pre-compute useful values
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_alpha_cumprod_clamped = torch.clamp(
            self.sqrt_alpha_cumprod, min=PROJECTION_EPS
        )
        self.sqrt_one_minus_alpha_cumprod_clamped = torch.clamp(
            self.sqrt_one_minus_alpha_cumprod, min=PROJECTION_EPS
        )
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

    @staticmethod
    def projection_from_embeddings(
        embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute projection matrix P_k = Q Q^T using a QR-orthonormalised basis of
        the embedding columns. Masked atoms are zeroed before the projection is
        computed; the input tensor is not modified in place. If ``mask`` is None,
        all atoms are treated as valid. This materialises a (batch, n_atoms,
        n_atoms) tensor, which can be memory intensive for large n_atoms; consider
        chunking if scaling up.

        Args:
            embeddings: Tensor of shape (batch, n_atoms, k)
            mask: Optional boolean mask of shape (batch, n_atoms)

        Returns:
            Projection matrices of shape (batch, n_atoms, n_atoms)
        """
        n_atoms = embeddings.shape[1]
        global PROJECTION_WARNING_EMITTED
        if n_atoms > PROJECTION_WARNING_THRESHOLD and not PROJECTION_WARNING_EMITTED:
            warnings.warn(
                "projection_from_embeddings materialises a (batch, n_atoms, n_atoms) tensor; "
                f"consider chunking when n_atoms > {PROJECTION_WARNING_THRESHOLD}.",
                RuntimeWarning,
            )
            PROJECTION_WARNING_EMITTED = True

        masked_embeddings = (
            embeddings if mask is None else embeddings * mask.unsqueeze(-1).float()
        )

        proj = torch.zeros(
            embeddings.shape[0],
            embeddings.shape[1],
            embeddings.shape[1],
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        valid_batch = (
            torch.ones(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
            if mask is None
            else mask.sum(dim=1) > 0
        )

        if valid_batch.any():
            q, _ = torch.linalg.qr(masked_embeddings[valid_batch], mode="reduced")
            proj_valid = q @ q.transpose(-1, -2)
            proj[valid_batch] = proj_valid

        if mask is not None:
            mask_matrix = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            proj = proj * mask_matrix.float()

        # Enforce symmetry to reduce numerical drift
        proj = 0.5 * (proj + proj.transpose(-1, -2))

        return proj

    @staticmethod
    def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """Numerically stable division with clamped denominator."""
        return numerator / torch.clamp(denominator, min=PROJECTION_EPS)

    @staticmethod
    def _masked_mse(
        pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mask-aware MSE normalised by the number of valid entries."""
        if mask is None:
            return F.mse_loss(pred, target)

        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.float()
        diff = (pred - target) ** 2
        per_item = (diff * mask).view(diff.shape[0], -1).sum(dim=1)
        denom = mask.view(mask.shape[0], -1).sum(dim=1).clamp_min(PROJECTION_EPS)
        return (per_item / denom).mean()

    @staticmethod
    def _reconstruct_x0(
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        sqrt_alpha: torch.Tensor,
        sqrt_one_minus_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct clean sample x_0 from noisy x_t and predicted noise."""
        return DiffusionTrainer._safe_divide(
            x_t - sqrt_one_minus_alpha * predicted_noise, sqrt_alpha
        )

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
        n_atoms: Optional[int] = None,
        atom_mask: Optional[torch.Tensor] = None,
        spectrum_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate eigenvectors by reverse diffusion.

        Args:
            mz: m/z values, shape (batch, n_peaks)
            intensity: Intensity values, shape (batch, n_peaks)
            n_atoms: Number of atoms to generate (if None, predicted from spectrum)
            atom_mask: Optional atom mask
            spectrum_mask: Optional spectrum mask
            
        Note:
            When ``n_atoms`` is None, this relies on a trained atom-count head
            (`enable_atom_count_head=True`) to provide reasonable predictions.

        Returns:
            Generated eigenvectors, shape (batch, n_atoms, k)
        """
        batch_size = mz.shape[0]
        k = self.model.k

        if n_atoms is None:
            count_pred = self.model.predict_atom_count(mz, intensity, spectrum_mask)
            n_atoms_per_sample = torch.clamp(
                torch.round(count_pred), 1, self.model.max_atoms
            ).long()
            n_atoms = int(n_atoms_per_sample.max().item())
            if atom_mask is None:
                atom_mask = (
                    torch.arange(n_atoms, device=self.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                    < n_atoms_per_sample.unsqueeze(1)
                )

        if n_atoms > self.model.max_atoms:
            raise ValueError(
                f"Requested n_atoms ({n_atoms}) exceeds model.max_atoms ({self.model.max_atoms})."
            )

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
        fingerprint_targets: Optional[torch.Tensor] = None,
        atom_count_targets: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Compute training loss.

        Args:
            x_0: Clean eigenvectors, shape (batch, n_atoms, k)
            mz: m/z values, shape (batch, n_peaks)
            intensity: Intensity values, shape (batch, n_peaks)
            atom_mask: Optional atom mask
            spectrum_mask: Optional spectrum mask
            fingerprint_targets: Optional fingerprint labels for auxiliary loss
            atom_count_targets: Optional atom-count labels for auxiliary loss
            return_components: If True, also return component losses

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
        noise_loss = self._masked_mse(
            predicted_noise, noise, mask=atom_mask.unsqueeze(-1) if atom_mask is not None else None
        )

        loss = noise_loss
        proj_loss = torch.tensor(0.0, device=self.device)
        fingerprint_loss = torch.tensor(0.0, device=self.device)
        atom_count_loss = torch.tensor(0.0, device=self.device)

        # Subspace-invariant projection loss to mitigate eigenvector sign/rotation ambiguity
        if self.projection_loss_weight > 0:
            sqrt_alpha = self.sqrt_alpha_cumprod_clamped[t].view(batch_size, 1, 1)
            sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod_clamped[t].view(
                batch_size, 1, 1
            )

            x_0_pred = self._reconstruct_x0(
                x_t, predicted_noise, sqrt_alpha, sqrt_one_minus_alpha
            )

            proj_pred = self.projection_from_embeddings(x_0_pred, atom_mask)
            proj_target = self.projection_from_embeddings(x_0, atom_mask)
            mask_matrix = (
                None if atom_mask is None else atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
            )
            proj_loss = self._masked_mse(proj_pred, proj_target, mask=mask_matrix)
            loss = loss + self.projection_loss_weight * proj_loss

        if self.fingerprint_loss_weight > 0 and fingerprint_targets is not None:
            fp_logits = self.model.predict_fingerprint(mz, intensity, spectrum_mask)
            fingerprint_loss = F.binary_cross_entropy_with_logits(fp_logits, fingerprint_targets)
            loss = loss + self.fingerprint_loss_weight * fingerprint_loss

        if self.atom_count_loss_weight > 0 and atom_count_targets is not None:
            atom_pred = self.model.predict_atom_count(mz, intensity, spectrum_mask)
            atom_count_loss = F.mse_loss(atom_pred, atom_count_targets.float())
            loss = loss + self.atom_count_loss_weight * atom_count_loss

        if return_components:
            return loss, {
                "noise": noise_loss.detach(),
                "projection": proj_loss.detach(),
                "fingerprint": fingerprint_loss.detach(),
                "atom_count": atom_count_loss.detach(),
            }

        return loss

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        x_0: torch.Tensor,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        spectrum_mask: Optional[torch.Tensor] = None,
        fingerprint_targets: Optional[torch.Tensor] = None,
        atom_count_targets: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[float, Tuple[float, dict]]:
        """
        Single training step.

        Args:
            optimizer: Optimizer
            x_0: Clean eigenvectors
            mz: m/z values
            intensity: Intensity values
            atom_mask: Optional atom mask
            spectrum_mask: Optional spectrum mask
            fingerprint_targets: Optional fingerprint labels
            atom_count_targets: Optional atom-count labels
            return_components: If True, also return component losses

        Returns:
            Loss value
        """
        self.model.train()
        optimizer.zero_grad()

        loss, components = (
            self.compute_loss(
                x_0,
                mz,
                intensity,
                atom_mask,
                spectrum_mask,
                fingerprint_targets,
                atom_count_targets,
                return_components=True,
            )
            if return_components
            else (
                self.compute_loss(
                    x_0,
                    mz,
                    intensity,
                    atom_mask,
                    spectrum_mask,
                    fingerprint_targets,
                    atom_count_targets,
                    return_components=False,
                ),
                None,
            )
        )
        loss.backward()
        optimizer.step()

        if return_components:
            return loss.item(), {k: v.item() for k, v in components.items()}

        return loss.item()


def create_synthetic_demo_dataset(
    batch_size: int = 32, k: int = 8, max_peaks: int = 40, fingerprint_bits: int = 128
) -> Tuple[torch.Tensor, ...]:
    """
    Create a small synthetic dataset with padding, masks, and fingerprints.
 
    Returns:
        Tuple of (eigenvectors, mz_values, intensities, atom_mask, spectrum_mask, fingerprints, atom_counts)
    """
    rng = np.random.default_rng(0)
    smiles_pool = [
        "c1ccccc1",  # benzene
        "CCO",  # ethanol
        "CC(=O)O",  # acetic acid
        "C#N",  # hydrogen cyanide
        "CCN",  # ethylamine
        "CCCC",  # butane
        "C1=CC=CN=C1",  # pyridine
    ]
    processor = SpectralDataProcessor(k=k, bond_weighting="order")

    eigen_list = []
    mz_list = []
    intensity_list = []
    atom_counts = []
    fingerprints = []

    max_atoms = 0
    for i in range(batch_size):
        smiles = smiles_pool[i % len(smiles_pool)]
        eig = processor.process_smiles(smiles)
        eigen_list.append(eig)
        atom_counts.append(eig.shape[0])
        fingerprints.append(processor.smiles_to_fingerprint(smiles, n_bits=fingerprint_bits))
        max_atoms = max(max_atoms, eig.shape[0])

        low = 6
        high = max(low + 1, max_peaks // 2 + 2)
        n_peaks = int(rng.integers(low=low, high=high))
        base_mass = 20 + 10 * eig.shape[0]
        mz_vals = np.sort(
            base_mass + rng.uniform(-5, 5, size=n_peaks).astype(np.float32)
        ).astype(np.float32)
        intensities = rng.random(n_peaks).astype(np.float32)
        mz_list.append(mz_vals)
        intensity_list.append(intensities)

    max_peaks_actual = max(max(len(arr), 1) for arr in mz_list)
    max_peaks_final = max(max_peaks_actual, max_peaks // 2)

    x0 = np.zeros((batch_size, max_atoms, k), dtype=np.float32)
    atom_mask = np.zeros((batch_size, max_atoms), dtype=bool)
    mz = np.zeros((batch_size, max_peaks_final), dtype=np.float32)
    intensity = np.zeros((batch_size, max_peaks_final), dtype=np.float32)
    spectrum_mask = np.zeros((batch_size, max_peaks_final), dtype=bool)
    fp_arr = np.zeros((batch_size, fingerprint_bits), dtype=np.float32)

    for i in range(batch_size):
        n_atoms = eigen_list[i].shape[0]
        x0[i, :n_atoms, :] = eigen_list[i]
        atom_mask[i, :n_atoms] = True

        n_peaks = len(mz_list[i])
        mz[i, :n_peaks] = mz_list[i]
        intensity[i, :n_peaks] = intensity_list[i]
        spectrum_mask[i, :n_peaks] = True
        fp_arr[i] = fingerprints[i]

    return (
        torch.tensor(x0, dtype=torch.float32),
        torch.tensor(mz, dtype=torch.float32),
        torch.tensor(intensity, dtype=torch.float32),
        torch.tensor(atom_mask),
        torch.tensor(spectrum_mask),
        torch.tensor(fp_arr, dtype=torch.float32),
        torch.tensor(atom_counts, dtype=torch.float32),
    )


def run_demo():
    """Run a demonstration of the Spectral Diffusion Model."""
    print("=" * 60)
    print("Spectral Diffusion Model Demo")
    print("=" * 60)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create demo data
    print("\n1. Creating synthetic demo dataset with padding and masks...")
    (
        x_0,
        mz,
        intensity,
        atom_mask,
        spectrum_mask,
        fp_targets,
        atom_counts,
    ) = create_synthetic_demo_dataset(batch_size=32, k=8, max_peaks=40, fingerprint_bits=128)
    x_0, mz, intensity, atom_mask, spectrum_mask, fp_targets, atom_counts = (
        x_0.to(device),
        mz.to(device),
        intensity.to(device),
        atom_mask.to(device),
        spectrum_mask.to(device),
        fp_targets.to(device),
        atom_counts.to(device),
    )
    print(f"   Eigenvector batch shape (with padding): {x_0.shape}")
    print(f"   Spectrum batch shape (with padding): {mz.shape}")
    print(f"   Example atom_mask valid atoms: {int(atom_mask[0].sum().item())}/{atom_mask.shape[1]}")
    print(
        f"   Example spectrum_mask valid peaks: {int(spectrum_mask[0].sum().item())}/{spectrum_mask.shape[1]}"
    )

    # Create model
    print("\n2. Creating Spec2GraphDiffusion model...")
    model = Spec2GraphDiffusion(
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        k=x_0.shape[-1],
        max_atoms=x_0.shape[1],
        max_peaks=mz.shape[1],
        dropout=0.1,
        fingerprint_dim=fp_targets.shape[-1],
        enable_atom_count_head=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Create trainer with fewer timesteps for demo
    print("\n3. Creating diffusion trainer...")
    trainer = DiffusionTrainer(
        model=model,
        n_timesteps=100,  # Reduced for demo
        device=device,
        projection_loss_weight=1.0,
        fingerprint_loss_weight=0.1,
        atom_count_loss_weight=0.05,
    )

    # Training loop
    print("\n4. Training for a few steps on the padded batch (demo only)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_steps = 20
    for step in range(n_steps):
        loss, components = trainer.train_step(
            optimizer,
            x_0,
            mz,
            intensity,
            atom_mask=atom_mask,
            spectrum_mask=spectrum_mask,
            fingerprint_targets=fp_targets,
            atom_count_targets=atom_counts,
            return_components=True,
        )
        if (step + 1) % 5 == 0:
            print(
                f"   Step {step + 1}/{n_steps}, total loss: {loss:.4f} | "
                f"noise: {components['noise']:.4f}, proj: {components['projection']:.4f}, "
                f"fp: {components['fingerprint']:.4f}, atoms: {components['atom_count']:.4f}"
            )

    # Sample
    print("\n5. Generating sample eigenvectors...")
    model.eval()
    generated = trainer.sample(
        mz[:1], intensity[:1], n_atoms=None, spectrum_mask=spectrum_mask[:1]
    )
    target_proj = DiffusionTrainer.projection_from_embeddings(
        x_0[:1], mask=atom_mask[:1]
    ).cpu()
    gen_proj = DiffusionTrainer.projection_from_embeddings(generated.cpu()).cpu()
    projection_similarity = F.mse_loss(gen_proj, target_proj).item()

    print(f"   Generated shape: {generated.shape}")
    print(f"   Projection similarity (lower is better): {projection_similarity:.4f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
