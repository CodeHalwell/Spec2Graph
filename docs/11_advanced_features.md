# 11 — Advanced Features

Beyond the core diffusion-train-sample loop, Spec2Graph includes two
optional systems that layer extra structure onto the pipeline:

1. **Guided sampling with a GNN discriminator** — pushes reverse diffusion
   toward chemically plausible graphs.
2. **Eigenvalue conditioning** — injects predicted Laplacian eigenvalues
   into the decoder to tighten the topology prior.

Both are opt-in and independent of the training loop.

## Guided diffusion via a GNN discriminator

The idea: during reverse diffusion, at each step we have an estimate of
the clean eigenvectors via Tweedie's formula. Decode those through the
SGNO into a soft adjacency, score the result with a trained
"chemistry discriminator", and use the gradient of that score to nudge
the reverse step toward more valid structures.

The plumbing spans three classes.

### `DenseGNNDiscriminator`

Defined at `spectral_diffusion.py:1456`. A small, differentiable GNN that
consumes a **continuous** adjacency probability matrix and outputs a
scalar validity score.

Why dense? Standard `torch_geometric` GNNs operate on sparse edge_index
representations, which aren't differentiable w.r.t. edge *existence*. We
want gradients flowing through `adj_probs`, so the layer uses dense
matrix multiplication:

```python
class DenseGNNLayer(nn.Module):
    def forward(self, x, adj):
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=PROJECTION_EPS)
        degree_inv_sqrt = torch.rsqrt(degree)
        adj_norm = adj * degree_inv_sqrt * degree_inv_sqrt.transpose(-1, -2)
        agg = torch.bmm(adj_norm, x)
        out = self.linear(agg)
        return F.relu(self.norm(out))
```

The discriminator stacks `num_layers` of these with residual connections,
then mean-pools across atoms and runs a linear classifier:

```python
x = self.node_proj(degree)                  # degree-based initial features
for layer in self.gnn_layers:
    x = x + layer(x, adj_probs)             # residual connection
return self.classifier(x.mean(dim=1))       # (B, 1) scalar per graph
```

Training the discriminator is **your** job — typically as a binary
classifier on (valid graph vs. randomly perturbed graph) pairs. The repo
doesn't ship a training loop for it.

### `SpectralGraphNeuralOperator` again

Already covered in [10 — Prediction decoders](./10_prediction_decoders.md).
For guidance we use `bond_probabilities` to get a **soft** adjacency that
the discriminator can differentiate through.

### `GuidedDiffusionSampler`

Defined at `spectral_diffusion.py:1517`. Construction:

```python
sampler = GuidedDiffusionSampler(
    trainer=trainer,           # must have a trained Spec2GraphDiffusion
    sgno=sgno,                 # trained or at least meaningful
    discriminator=discriminator,
    guidance_scale=1.0,        # how strongly to steer
)
```

The core of the loop at each timestep:

```python
# 1. Standard reverse step (no grad)
with torch.no_grad():
    eps_pred = trainer.model(x_t, t_tensor, mz, intensity, ...)
mu = (1/sqrt(α)) · (x_t − (β/sqrt(1−ᾱ)) · eps_pred)

# 2. Tweedie estimate with gradient enabled
if guidance_scale > 0 and t > 0:
    x_t_grad = x_t.detach().requires_grad_(True)
    eps_for_tweedie = trainer.model(x_t_grad, t_tensor, mz, intensity, ...)
    x0_hat = (x_t_grad − sqrt(1−ᾱ) · eps_for_tweedie) / sqrt(ᾱ)

    # Apply atom mask so padded atoms don't contribute
    if atom_mask is not None:
        x0_hat = x0_hat * atom_mask.unsqueeze(-1).float()

    # 3. Decode and score
    adj_probs = sgno.bond_probabilities(x0_hat)
    if atom_mask is not None:
        edge_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
        adj_probs = adj_probs * edge_mask
    validity_log_prob = F.logsigmoid(discriminator(adj_probs))

    # 4. Backprop through the discriminator to get a gradient on x_t
    grad = torch.autograd.grad(validity_log_prob.sum(), x_t_grad)[0]

    mu = mu + guidance_scale * grad

# 5. Sample with standard variance (except at t=0)
if t > 0:
    noise = torch.randn_like(x_t)
    x_t = mu + sqrt(posterior_variance[t]) * noise
else:
    x_t = mu
```

Things to understand:

- **Two forward passes per step.** One inside `no_grad` for the standard
  mean, one with gradients enabled to take the Tweedie derivative. That
  ~doubles inference cost per step.
- **`guidance_scale` is sensitive.** Too small and the guidance does
  nothing; too large and the gradient dominates the reverse dynamics
  and produces artifacts. Start in the `[0.1, 1.0]` range.
- **The Tweedie shortcut is exact.** As noted in
  [05 — Forward diffusion](./05_forward_diffusion.md), it's the analytic
  inverse of `q_sample`, not an approximation.
- **Mask everything.** Padded atom rows/cols of `x0_hat` and `adj_probs`
  are zeroed before the discriminator sees them, so padding can't bias
  the score.

### When it helps

Best use case: the diffusion model alone produces eigenvectors whose
decoded adjacency *mostly* looks like a molecule but occasionally
produces nonsense (isolated atoms, impossible valency). Guidance steers
away from those. If the base model is simply underfit, guidance can't
save you.

## Eigenvalue conditioning

The second advanced feature is wiring predicted eigenvalues into the
decoder. Three classes collaborate.

### `EigenvalueConditioner`

Defined at `spectral_diffusion.py:1783`. A tiny MLP:

```python
nn.Sequential(
    nn.Linear(k, d_model),
    nn.ReLU(),
    nn.Linear(d_model, d_model),
)
```

Input: predicted eigenvalues `(B, k)`. Output: a conditioning vector
`(B, d_model)` that encodes graph-level spectral invariants.

### `Spec2GraphDiffusion.eigenvalue_head`

Defined inline (`spectral_diffusion.py:554`) and wired up when
`enable_eigenvalue_head=True`. It predicts `(B, k)` eigenvalues from the
pooled spectrum encoding:

```python
eig_pred = model.predict_eigenvalues(mz, intensity, spectrum_mask, precursor_mz)
```

Training this head is turned on by `TrainerConfig.eigenvalue_loss_weight > 0`
with `TrainingBatch.eigenvalue_targets` provided. See
[07 — Training losses](./07_training_losses.md).

### `EigenvalueConditionedSGNO`

Defined at `spectral_diffusion.py:1820`. An extension of the plain SGNO
whose pairwise MLP takes `[E_i ; E_j ; eigenvalue_cond]` instead of just
`[E_i ; E_j]`:

```python
in_dim = 2*k + eigenvalue_dim
```

The forward pass splits the first linear layer into three parts
(w_i, w_j, w_eig) and combines them with broadcasting, so it never
materializes the full `(B, N, N, 2k + eigenvalue_dim)` tensor — same
optimization as the base SGNO.

Usage:

```python
decoder = EigenvalueConditionedSGNO(k=k, eigenvalue_dim=64).to(device)
logits = decoder(predicted_eigenvectors, predicted_eigenvalues)
probs  = decoder.bond_probabilities(predicted_eigenvectors, predicted_eigenvalues)
```

### Why it helps

Laplacian eigenvalues of a molecular graph encode strong constraints:

- The multiplicity of 0 equals the number of connected components.
- The largest eigenvalue relates to max degree and bipartiteness.
- The spectral sum equals the number of edges (for normalized
  Laplacians with specific normalizations).

Passing these through to the decoder means "the adjacency we're about
to decode should produce this spectrum". In practice, it acts like a
soft constraint that biases the decoder's edge distribution toward
topologies consistent with the predicted invariants.

## Summary

| Feature | Requires training | What it gives you |
|---------|-------------------|-------------------|
| Guided sampling | A trained discriminator + trained SGNO | Sharper, more chemically valid samples |
| Eigenvalue conditioning | `eigenvalue_head` trained + conditioned SGNO | Decoder that's aware of spectral invariants |

Both are compatible. A full inference pipeline could:

1. Run `trainer.sample` (or `GuidedDiffusionSampler.guided_sample`) to get
   `V̂_k`.
2. Call `model.predict_eigenvalues` on the same spectrum.
3. Feed both into an `EigenvalueConditionedSGNO` to get bond probabilities.
4. Finish with `ValencyDecoder` for a valid graph.

Next: [12 — End-to-end walkthrough](./12_end_to_end_walkthrough.md) ties
every chapter together with a minimal runnable example.
