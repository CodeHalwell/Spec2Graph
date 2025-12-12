# Spec2Graph

MS spectrum to graph prediction of chemicals using Spectral Diffusion Models.

## Overview

Spec2Graph implements a **Spectral Diffusion Model** that learns to generate the Spectral Embedding (eigenvectors) of a molecule's Laplacian matrix, conditioned on Mass Spectrum data. This enables direct generation of high-quality spectral graph embeddings from MS spectrum data for downstream molecular reconstruction and analysis.

### Why this is both clever and cursed

- Eigenvectors are **sign/rotation ambiguous** (especially with degenerate eigenvalues); raw eigenvectors are unstable targets.
- Distinct graphs can be **cospectral**, so spectral information alone is not uniquely identifying.
- MS/MS is already ambiguous for isomers, so we stack multiple identifiability limits.

**Practical fix:** train on the subspace-invariant projection matrix  
\(P_k = V_k V_k^\top\) (first \(k\) eigenvectors) and add an orthonormality regularizer, rather than using raw eigenvectors. This is implemented via the projection-aware loss in `DiffusionTrainer` and the `projection_matrix` helper in `SpectralDataProcessor`.

For a detailed implementation roadmap, see [ROADMAP.md](./ROADMAP.md).

### Key Features

- **Input**: Mass spectrum peaks encoded with transformer + Fourier positional embeddings
- **Output**: Top k eigenvectors of Laplacian matrix
- **Diffusion**: DDPM formulation (noisy → denoised spectral embedding)
- **Demo**: Includes a benzene molecule example with simulated spectrum

## Architecture

The implementation consists of four main components:

1. **SpectralDataProcessor**: Converts SMILES molecular representations to eigenvectors with sign canonicalization
2. **FourierMzEmbedding**: Fourier positional embeddings for m/z values
3. **Spec2GraphDiffusion**: Transformer encoder-decoder with diffusion denoising
4. **DiffusionTrainer**: Training loop with DDPM schedule and sampling

## Installation

```bash
# Clone the repository
git clone https://github.com/CodeHalwell/Spec2Graph.git
cd Spec2Graph

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch >= 2.6.0
- NumPy >= 1.20.0
- RDKit >= 2023.3.1

## Usage

### Quick Demo

Run the included demo to see the model in action:

```bash
python spectral_diffusion.py
```

This will:
1. Create demo data for a benzene molecule with simulated mass spectrum
2. Build a Spec2GraphDiffusion model
3. Train for a few iterations
4. Generate sample eigenvectors via reverse diffusion

### Example Code

```python
from spectral_diffusion import (
    SpectralDataProcessor,
    Spec2GraphDiffusion,
    DiffusionTrainer
)
import torch
import numpy as np

# Process a molecule
processor = SpectralDataProcessor(k=8)
eigenvectors = processor.process_smiles("c1ccccc1")  # Benzene

# Create model
model = Spec2GraphDiffusion(
    d_model=256,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    k=8,
)

# Create trainer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
trainer = DiffusionTrainer(model=model, n_timesteps=1000, device=device)

# Prepare spectrum data (example)
mz = torch.tensor([[39.0, 51.0, 77.0, 78.0]]).to(device)
intensity = torch.tensor([[0.3, 0.4, 0.6, 1.0]]).to(device)
x_0 = torch.tensor(eigenvectors).unsqueeze(0).to(device)

# Training step
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss = trainer.train_step(optimizer, x_0, mz, intensity)
print(f"Loss: {loss}")

# Generate new eigenvectors
generated = trainer.sample(mz, intensity, n_atoms=6)
print(f"Generated shape: {generated.shape}")
```

## Model Details

### Spectral Graph Representation

The model represents molecular graphs through their spectral embeddings:
- Converts SMILES to adjacency matrix via RDKit
- Computes normalized Laplacian: `L = I - D^(-1/2) A D^(-1/2)`
- Extracts top k eigenvectors with sign canonicalization

### Diffusion Process

Uses DDPM (Denoising Diffusion Probabilistic Models):
- Forward process: Gradually adds Gaussian noise to eigenvectors
- Reverse process: Learns to denoise conditioned on spectrum data
- Training objective: Predict the noise added at each timestep

### Transformer Architecture

- **Encoder**: Processes mass spectrum peaks with Fourier m/z embeddings
- **Decoder**: Cross-attends to encoded spectrum while processing noisy eigenvectors
- **Timestep conditioning**: Sinusoidal embeddings added to decoder input

## API Reference

### SpectralDataProcessor

```python
processor = SpectralDataProcessor(k=8)
eigenvectors = processor.process_smiles("CCO")  # Ethanol
```

### Spec2GraphDiffusion

```python
model = Spec2GraphDiffusion(
    d_model=256,        # Model dimension
    nhead=8,            # Attention heads
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=1024,
    k=8,                # Number of eigenvectors
    max_atoms=64,
    max_peaks=100,
    dropout=0.1,
)
```

### DiffusionTrainer

```python
trainer = DiffusionTrainer(
    model=model,
    n_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    device="cuda",
    projection_loss_weight=1.0,  # subspace-invariant P_k supervision
)
```

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{spec2graph2024,
  title={Spec2Graph: Spectral Diffusion Model for MS Spectrum to Graph Prediction},
  author={CodeHalwell},
  year={2024},
  url={https://github.com/CodeHalwell/Spec2Graph}
}
```
