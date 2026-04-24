# 12 — End-to-End Walkthrough

This doc ties every previous chapter together. It maps the exact calls in
`run_demo()` (`spectral_diffusion.py:1987`) to the pipeline stages, then
shows how you'd extend the demo into a "real" workflow that reaches a
molecular graph.

## The five stages of `run_demo`

```
[1] create_synthetic_demo_dataset   — builds a padded batch
[2] Spec2GraphDiffusion(config)     — constructs the model
[3] DiffusionTrainer(model, cfg)    — wraps the model in a DDPM trainer
[4] trainer.train_step(...) × 20    — runs a short training loop
[5] trainer.sample(...)             — draws one sample and scores it
```

Let's walk through each.

### Stage 1 — dataset construction

`create_synthetic_demo_dataset` (`spectral_diffusion.py:1908`) picks SMILES
from a tiny pool (`benzene`, `ethanol`, `acetic acid`, `HCN`, `ethylamine`,
`butane`, `pyridine`), runs each through `SpectralDataProcessor`, fabricates
a plausible-looking mass spectrum, and pads everything.

Outputs shape (with `batch_size=32`, `k=8`, `max_peaks=40`,
`fingerprint_bits=128`):

```
x_0           : (32, max_atoms, 8)      # ground-truth eigenvectors
mz            : (32, ~22)               # padded m/z
intensity     : (32, ~22)               # padded intensities
atom_mask     : (32, max_atoms) bool    # True = valid atom
spectrum_mask : (32, ~22) bool          # True = valid peak
fp_targets    : (32, 128)               # Morgan fingerprints
atom_counts   : (32,) float             # heavy-atom counts
```

Documented in:
- [02 — Data processing](./02_data_processing.md) — how each SMILES becomes
  `V_k`.
- [04 — Batching and masks](./04_batching_and_masks.md) — how everything
  gets padded.

### Stage 2 — model

```python
config = Spec2GraphDiffusionConfig(
    d_model=128,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=512,
    k=x_0.shape[-1],
    max_atoms=x_0.shape[1],
    max_peaks=mz.shape[1],
    dropout=0.1,
    fingerprint_dim=fp_targets.shape[-1],     # enables fingerprint head
    enable_atom_count_head=True,              # enables atom count head
)
model = Spec2GraphDiffusion(config).to(device)
```

Note how the demo derives `max_atoms` and `max_peaks` from the batch
itself. This is fine for a demo but not recommended for production —
you'd typically set these to global maxima expected across your dataset.

Documented in: [06 — Model architecture](./06_model_architecture.md).

### Stage 3 — trainer

```python
config = TrainerConfig(
    n_timesteps=100,            # short schedule for demo speed
    projection_loss_weight=1.0,
    fingerprint_loss_weight=0.1,
    atom_count_loss_weight=0.05,
)
trainer = DiffusionTrainer(model=model, config=config, device=device)
```

With `n_timesteps=100`, one full sample takes 100 reverse steps instead
of 1000. That's why the demo finishes in seconds.

Documented in:
- [05 — Forward diffusion](./05_forward_diffusion.md) — the schedule.
- [07 — Training losses](./07_training_losses.md) — what the weights mean.

### Stage 4 — training loop

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
batch = TrainingBatch(
    x_0=x_0, mz=mz, intensity=intensity,
    atom_mask=atom_mask, spectrum_mask=spectrum_mask,
    fingerprint_targets=fp_targets,
    atom_count_targets=atom_counts,
)
for step in range(20):
    loss, components = trainer.train_step(
        optimizer, batch, return_components=True,
    )
```

20 steps is not enough to actually converge — the demo just shows the
machinery. The printed components let you verify every loss term is
non-zero and finite.

Documented in: [08 — Training loop](./08_training_loop.md).

### Stage 5 — sampling and evaluation

```python
model.eval()
generated = trainer.sample(
    mz[:1], intensity[:1],
    n_atoms=None,                       # predicted from atom_count head
    spectrum_mask=spectrum_mask[:1],
)
target_proj = DiffusionTrainer.projection_from_embeddings(x_0[:1], mask=atom_mask[:1]).cpu()
gen_proj    = DiffusionTrainer.projection_from_embeddings(generated.cpu())
projection_similarity = F.mse_loss(gen_proj, target_proj).item()
```

Because `n_atoms=None`, the sampler uses the atom-count head to decide
how many atom slots to allocate. Evaluation is a projection-matrix MSE,
which is the subspace-invariant quantity described in
[02 — Data processing](./02_data_processing.md).

Documented in: [09 — Reverse diffusion](./09_reverse_diffusion.md).

## Going from the demo to a full molecule prediction

The demo stops at eigenvectors. To produce a molecular graph you need one
more pair of stages:

```python
from spectral_diffusion import (
    SpectralGraphNeuralOperator,
    ValencyDecoder,
)

# Assume the SGNO has been trained separately on (V_k, adjacency) pairs.
sgno = SpectralGraphNeuralOperator(k=8).to(device)
sgno.load_state_dict(torch.load("sgno.pt", weights_only=True))
sgno.eval()

with torch.no_grad():
    bond_probs = sgno.bond_probabilities(generated)    # (1, N, N)

# Atom types must come from somewhere — formula prediction, adduct parsing,
# or priors. Here we pretend we know them.
atom_types = ["C"] * generated.shape[1]

valency_decoder = ValencyDecoder()
adjacency = valency_decoder.decode(
    atom_types=atom_types,
    bond_probs=bond_probs[0].cpu().numpy(),
    threshold=0.3,
    max_bond_order=1,
)
```

Documented in: [10 — Prediction decoders](./10_prediction_decoders.md).

If you want guided sampling or eigenvalue conditioning, wrap the above in
the constructs described in [11 — Advanced features](./11_advanced_features.md):

```python
from spectral_diffusion import (
    DenseGNNDiscriminator,
    GuidedDiffusionSampler,
    EigenvalueConditionedSGNO,
)

discriminator = DenseGNNDiscriminator().to(device)
discriminator.load_state_dict(torch.load("discriminator.pt", weights_only=True))
discriminator.eval()

guided = GuidedDiffusionSampler(
    trainer=trainer, sgno=sgno, discriminator=discriminator, guidance_scale=0.5,
)
generated = guided.guided_sample(
    mz[:1], intensity[:1], n_atoms=int(atom_counts[0]),
    spectrum_mask=spectrum_mask[:1],
)
```

## Checklist: "Is my training set up correctly?"

Walk through this before your first long training run.

- [ ] Each example's `eigenvectors` uses the same `k` and `bond_weighting`
      as your `Spec2GraphDiffusion(k=...)` and `SpectralDataProcessor`.
- [ ] `atom_mask` and `spectrum_mask` are **bool** tensors with at least
      one True per row.
- [ ] Your `max_atoms` in the config is ≥ the longest molecule in your
      dataset — otherwise training hits a ValueError.
- [ ] Your `max_peaks` is ≥ the longest spectrum — same reason.
- [ ] `TrainerConfig.projection_loss_weight > 0` (default) — this is the
      term that fixes eigenvector ambiguity.
- [ ] If you plan to sample with `n_atoms=None`, you've set
      `enable_atom_count_head=True` and `atom_count_loss_weight > 0` and
      are passing `atom_count_targets`.
- [ ] You've printed `return_components=True` for the first few steps
      and confirmed each enabled loss is finite and non-zero.
- [ ] Batches are on the same `device` as the trainer/model.
- [ ] Your model was moved to `device` *before* constructing the
      `DiffusionTrainer` (because the schedule tensors go to `device` at
      construction time).

## What's deliberately not covered

Several things that would be part of a production pipeline are not
implemented in this repo yet:

- A real dataset loader (MSP/mzML parsing).
- An SGNO or discriminator training loop.
- DDIM-style accelerated samplers.
- Element-aware prediction — the code always treats atoms as anonymous
  slots and expects you to provide `atom_types` externally.
- Formula/adduct conditioning.

The `ROADMAP.md` covers the planned phases. This docs set describes
what's already in the code.

## Where to go next

- To dig deeper into the math, revisit
  [05 — Forward diffusion](./05_forward_diffusion.md) and
  [07 — Training losses](./07_training_losses.md).
- To extend the model, look at
  [06 — Model architecture](./06_model_architecture.md) (hooks for new
  conditioning) and
  [10 — Prediction decoders](./10_prediction_decoders.md) (for decoder
  swaps).
- To move toward real data, the dataset constructor in
  [04 — Batching and masks](./04_batching_and_masks.md) is the pattern
  you want to replicate with your MSP/mzML loader.
