"""Tests for :mod:`spec2graph.casmi.conditioning` and :mod:`spec2graph.casmi.scoring`."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("rdkit")
torch = pytest.importorskip("torch")

from spec2graph.casmi import chem, data, spectral
from spec2graph.casmi.adducts import AdductVocabulary
from spec2graph.casmi.conditioning import N_TOKENS, CasmiTrainer, load_checkpoint, save_checkpoint
from spec2graph.casmi.scoring import FEATURES, Scorer, ScoringConfig, align_and_overlap, fingerprint_loglik, log_sinkhorn, soft_tanimoto
from spec2graph.data.elements import ELEMENT_TO_INDEX, N_ELEMENT_TYPES
from spectral_diffusion import DiffusionTrainer, SpectralDataProcessor
from tests.conftest import FP_BITS, K, MAX_ATOMS, MAX_PEAKS, N_Q, tiny_trainer

FAST = ScoringConfig(ppm=20.0, n_samples=2, n_steps=3, max_atom_counts=4, alignment_iters=2, sinkhorn_iters=5)


@pytest.fixture
def batch(casmi_frame, tmp_path):
    vocab = AdductVocabulary.fit(casmi_frame["adduct"])
    ds = data.Dataset(casmi_frame, spectral.SpectralCache(tmp_path / "cache", k=K, fingerprint_bits=FP_BITS), vocab, max_atoms=MAX_ATOMS, n_quantiles=N_Q)
    return data.Collator(MAX_ATOMS, MAX_PEAKS, K, FP_BITS)([ds[i] for i in range(4)]), vocab


def test_encode_and_conditioning(batch):
    b, vocab = batch
    trainer = tiny_trainer(len(vocab))
    tb = b.core
    memory, mask = trainer.wrapper.encode(tb.mz, tb.intensity, tb.spectrum_mask, tb.precursor_mz, b.cond)
    assert memory.shape == (4, N_TOKENS + tb.mz.shape[1], 16) and mask[:, :N_TOKENS].all() and torch.equal(mask[:, N_TOKENS:], tb.spectrum_mask)
    known, missing = (data.Conditioning.build([1], [ce], [180.0], [1.0]) for ce in (20.0, None))
    t_known, t_missing = trainer.wrapper.tokens(known), trainer.wrapper.tokens(missing)
    assert torch.allclose(t_known[:, 0], t_missing[:, 0]) and not torch.allclose(t_known[:, 1], t_missing[:, 1])  # learned "missing" vector
    doubled = b.cond.map(lambda t: t)
    doubled.neutral_mass = b.cond.neutral_mass * 2
    mem_b, mask_b = trainer.wrapper.encode(tb.mz, tb.intensity, tb.spectrum_mask, tb.precursor_mz, doubled)
    torch.manual_seed(1)
    a = trainer.sample(memory, mask, 6, n_steps=3)
    torch.manual_seed(1)
    assert a.shape == (4, 6, K) and not torch.allclose(a, trainer.sample(mem_b, mask_b, 6, n_steps=3))


def test_rank_safe_projector():
    torch.manual_seed(0)
    full, mask = torch.randn(2, 6, K, requires_grad=True), torch.ones(2, 6, dtype=torch.bool)
    assert torch.allclose(CasmiTrainer.projection_from_embeddings(full, mask), DiffusionTrainer.projection_from_embeddings(full.detach(), mask), atol=1e-3)
    deficient = torch.randn(1, 5, K, requires_grad=True)  # 3 valid rows, k=4: rank deficient
    CasmiTrainer.projection_from_embeddings(deficient, torch.tensor([[True, True, True, False, False]])).sum().backward()
    assert torch.isfinite(deficient.grad).all()


def test_training_modes(batch):
    b, vocab = batch
    trainer = tiny_trainer(len(vocab), fingerprint_loss_weight=0.1, atom_count_loss_weight=0.05, atom_type_loss_weight=0.5, orthonormality_loss_weight=0.1)
    opt = torch.optim.Adam(trainer.wrapper.parameters(), lr=1e-2)
    first, parts = trainer.train_step(opt, b, return_components=True)
    assert set(parts) == {"noise", "projection", "orthonormality", "fingerprint", "atom_count", "quantile", "atom_type"} and all(map(np.isfinite, parts.values()))
    losses = [trainer.train_step(opt, b, grad_clip=1.0) for _ in range(25)]
    assert np.mean(losses[-5:]) < first
    proxy = tiny_trainer(len(vocab), heads_only=True, fingerprint_loss_weight=0.1)
    _, parts = proxy.train_step(torch.optim.Adam(proxy.wrapper.parameters()), b, return_components=True)
    assert set(parts) == {"fingerprint", "quantile"}  # no denoiser in proxy mode


def test_heads_samplers_and_checkpoint(batch, tmp_path):
    b, vocab = batch
    trainer = tiny_trainer(len(vocab))
    spectra = data.Spectra(b.core.mz, b.core.intensity, b.core.spectrum_mask, b.core.precursor_mz, b.cond, np.arange(4))
    heads = trainer.heads(spectra)
    assert heads["fingerprint_prob"].shape == (4, FP_BITS) and heads["quantiles"].shape == (4, N_Q) and heads["atom_count"].shape == (4,)
    assert torch.all(torch.diff(heads["quantiles"], dim=-1) >= 0)
    memory, mask = heads["memory"], heads["memory_mask"]
    atom_mask = torch.ones(4, 5, dtype=torch.bool)
    for kw in (dict(sampler="ddpm"), dict(sampler="ddim", n_steps=4, eta=0.5)):
        out = trainer.sample(memory, mask, 5, atom_mask, **kw)
        assert out.shape == (4, 5, K) and torch.isfinite(out).all()
    chunks = list(trainer.draw(memory, mask, 5, n_draws=7, batch=4, n_steps=2))
    assert [c[0].shape[0] for c in chunks] == [4, 3] and trainer.atom_type_logprobs(chunks[0][0], chunks[0][1], chunks[0][2]).shape == (4, 5, N_ELEMENT_TYPES)
    with pytest.raises(ValueError):
        trainer.sample(memory, mask, MAX_ATOMS + 1)
    save_checkpoint(tmp_path / "ckpt.pt", trainer.wrapper, trainer.config, vocab, epoch=3)
    model, config, loaded_vocab, extra = load_checkpoint(tmp_path / "ckpt.pt")
    assert extra == {"epoch": 3} and config == trainer.config and loaded_vocab.index == vocab.index
    assert all(torch.equal(model.state_dict()[k], v) for k, v in trainer.wrapper.state_dict().items())


def _projection(smiles: str, perm=None):
    eigvecs, elements = SpectralDataProcessor(k=K).process_smiles(smiles), np.array([ELEMENT_TO_INDEX[s] for s in chem.symbols(smiles)])
    if perm is not None:
        eigvecs, elements = eigvecs[perm], elements[perm]
    return CasmiTrainer.projection_from_embeddings(torch.from_numpy(eigvecs).unsqueeze(0))[0], torch.from_numpy(elements)


def _confident(elements: torch.Tensor) -> torch.Tensor:
    logprob = torch.full((len(elements), N_ELEMENT_TYPES), math.log(0.02 / (N_ELEMENT_TYPES - 1)))
    logprob[torch.arange(len(elements)), elements] = math.log(0.98)
    return logprob


def test_fingerprint_scores_and_sinkhorn():
    prob, fps = np.array([0.9, 0.1, 0.9, 0.1]), np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.float32)
    assert fingerprint_loglik(prob, fps).argsort().tolist() == [1, 2, 0] and soft_tanimoto(prob, fps).argsort().tolist() == [1, 2, 0]
    assert soft_tanimoto(np.zeros(4), np.zeros((1, 4)))[0] == 0.0
    pi = log_sinkhorn(torch.randn(3, 5, 5), 50).exp()
    assert torch.allclose(pi.sum(-1), torch.ones(3, 5), atol=1e-4) and torch.allclose(pi.sum(-2), torch.ones(3, 5), atol=1e-4)


def test_alignment_recovers_permuted_truth():
    perm = np.random.default_rng(0).permutation(7)
    p_true, e_true = _projection("Cc1ccccc1", perm)
    p_c, e_c = zip(*(_projection(s) for s in ["Cc1ccccc1", "CCCCCCC", "C1CCCCCC1", "CC(C)CCCC"]))  # all-carbon C7 graphs
    hard, soft, ll = align_and_overlap(p_true, _confident(e_true), torch.stack(p_c), torch.stack(e_c))
    assert hard.argmax() == soft.argmax() == 0 and hard[0] > 0.99 and torch.all(hard[1:] < hard[0]) and torch.allclose(ll, ll[0], atol=1e-4)
    p_true, e_true = _projection("Cc1ccccc1")
    p_c, e_c = zip(*(_projection(s) for s in ["Cc1ccccc1", "Oc1ccccc1", "Fc1ccccc1"]))  # same weighted graph, different elements
    hard, _, ll = align_and_overlap(p_true, _confident(e_true), torch.stack(p_c), torch.stack(e_c))
    assert torch.allclose(hard, hard[0], atol=0.05) and ll.argmax() == 0 and ll[0] > ll[1] + 0.1
    with pytest.raises(ValueError):
        align_and_overlap(torch.eye(3), torch.zeros(3, N_ELEMENT_TYPES), torch.zeros(2, 4, 4), torch.zeros(2, 4))


def test_scorer(setup):
    df, vocab, pool, trainer = setup
    scorer = Scorer(trainer, pool, FAST)
    rows = lambda m: df[df["molecule_id"] == m]
    scores = scorer.score(data.Spectra.from_rows(rows("m_toluene"), vocab, MAX_PEAKS), "m_toluene")
    assert scores.features.shape == (3, len(FEATURES)) and chem.record("Cc1ccccc1")["inchikey14"] in set(pool.inchikey14[scores.indices])  # C7H8 isomers
    assert all(np.isfinite(scores[name]).all() for name in FEATURES) and np.all((scores["proj_overlap"] >= 0) & (scores["proj_overlap"] <= 1))
    assert {"smiles", "inchikey14", *FEATURES} <= set(scores.frame(pool).columns)
    empty = Scorer(trainer, pool.without(pool.inchikey14), FAST).score(data.Spectra.from_rows(rows("m_caffeine"), vocab, MAX_PEAKS))
    assert len(empty) == 0 and empty.features.shape == (0, len(FEATURES))
    no_proj = Scorer(trainer, pool, ScoringConfig(use_projection=False)).score(data.Spectra.from_rows(rows("m_ethanol"), vocab, MAX_PEAKS))
    assert len(no_proj) == 1 and np.isnan(no_proj["proj_overlap"]).all() and np.isfinite(no_proj["fp_loglik"]).all()
