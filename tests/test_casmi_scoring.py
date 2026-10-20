"""Tests for :mod:`spec2graph.casmi.scoring`."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("rdkit")
torch = pytest.importorskip("torch")

from spec2graph.casmi.adducts import AdductVocabulary
from spec2graph.casmi.conditioning import CasmiModel, CasmiModelConfig, CasmiTrainer, CasmiTrainerConfig
from spec2graph.casmi.data import (
    build_structure_table,
    group_by_molecule,
    prepare_spectra_frame,
    spectra_to_tensors,
    structure_record,
)
from spec2graph.casmi.pool import CandidatePool
from spec2graph.casmi.scoring import (
    FEATURE_NAMES,
    CandidateScores,
    ScoringConfig,
    align_and_overlap,
    fingerprint_loglik,
    log_sinkhorn,
    score_molecule,
    soft_tanimoto,
)
from spec2graph.data.elements import ELEMENT_TO_INDEX, N_ELEMENT_TYPES
from spectral_diffusion import Spec2GraphDiffusion, Spec2GraphDiffusionConfig, SpectralDataProcessor
from tests.conftest import CASMI_MOLECULES

K = 4


class TestFingerprintScores:
    def test_loglik_prefers_matching_fingerprint(self):
        prob = np.array([0.9, 0.1, 0.9, 0.1])
        fps = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.float32)
        ll = fingerprint_loglik(prob, fps)
        assert ll[0] > ll[2] > ll[1]
        assert ll.shape == (3,)
        st = soft_tanimoto(prob, fps)
        assert st[0] > st[2] > st[1]
        assert soft_tanimoto(np.zeros(4), np.zeros((1, 4)))[0] == 0.0


class TestAlignment:
    def test_sinkhorn_is_doubly_stochastic(self):
        torch.manual_seed(0)
        pi = log_sinkhorn(torch.randn(3, 5, 5), n_iters=50).exp()
        assert torch.allclose(pi.sum(-1), torch.ones(3, 5), atol=1e-4)
        assert torch.allclose(pi.sum(-2), torch.ones(3, 5), atol=1e-4)

    def _projection(self, smiles: str, perm: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        processor = SpectralDataProcessor(k=K)
        from rdkit import Chem

        eigvecs = processor.process_smiles(smiles)
        mol = Chem.MolFromSmiles(smiles)
        elements = np.array([ELEMENT_TO_INDEX[a.GetSymbol()] for a in mol.GetAtoms()])
        if perm is not None:
            eigvecs, elements = eigvecs[perm], elements[perm]
        v = torch.from_numpy(eigvecs).unsqueeze(0)
        return CasmiTrainer.projection_from_embeddings(v)[0], torch.from_numpy(elements)

    def test_recovers_permuted_truth_against_isomers(self):
        # All-carbon C7 isomers with different graphs: only the subspace can tell them apart.
        candidates = ["Cc1ccccc1", "CCCCCCC", "C1CCCCCC1", "CC(C)CCCC"]
        rng = np.random.default_rng(0)
        perm = rng.permutation(7)
        p_true, elems_true = self._projection("Cc1ccccc1", perm)
        # Confident one-hot element log-probabilities in the permuted order.
        logprob = torch.full((7, N_ELEMENT_TYPES), math.log(0.02 / (N_ELEMENT_TYPES - 1)))
        logprob[torch.arange(7), elems_true] = math.log(0.98)
        p_c, e_c = zip(*(self._projection(s) for s in candidates))
        overlap, overlap_soft, elem_ll = align_and_overlap(p_true, logprob, torch.stack(p_c), torch.stack(e_c))
        assert overlap.shape == overlap_soft.shape == elem_ll.shape == (4,)
        assert torch.argmax(overlap).item() == 0 and torch.argmax(overlap_soft).item() == 0
        assert overlap[0] > 0.99 and overlap_soft[0] > 0.99
        assert torch.all(overlap[1:] < overlap[0])
        # Identical compositions: element log-likelihoods are all the same.
        assert torch.allclose(elem_ll, elem_ll[0], atol=1e-4)

    def test_element_term_separates_same_graph(self):
        # Toluene, phenol and fluorobenzene share one weighted heavy-atom graph;
        # only the elements differ, and the alignment must notice.
        p_true, elems_true = self._projection("Cc1ccccc1")
        logprob = torch.full((7, N_ELEMENT_TYPES), math.log(0.02 / (N_ELEMENT_TYPES - 1)))
        logprob[torch.arange(7), elems_true] = math.log(0.98)
        p_c, e_c = zip(*(self._projection(s) for s in ["Cc1ccccc1", "Oc1ccccc1", "Fc1ccccc1"]))
        overlap, _, elem_ll = align_and_overlap(p_true, logprob, torch.stack(p_c), torch.stack(e_c))
        assert torch.allclose(overlap, overlap[0], atol=0.05)
        assert torch.argmax(elem_ll).item() == 0
        assert elem_ll[0] > elem_ll[1] + 0.1

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            align_and_overlap(torch.eye(3), torch.zeros(3, N_ELEMENT_TYPES), torch.zeros(2, 4, 4), torch.zeros(2, 4))
        with pytest.raises(ValueError):
            align_and_overlap(torch.eye(3), torch.zeros(3, N_ELEMENT_TYPES), torch.zeros(2, 3, 3), torch.zeros(2, 2))


def _tiny_trainer(n_adducts: int, fingerprint_bits: int = 64, n_quantiles: int = 8) -> CasmiTrainer:
    torch.manual_seed(0)
    core = Spec2GraphDiffusion(
        Spec2GraphDiffusionConfig(
            d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32,
            k=K, max_atoms=16, max_peaks=32, dropout=0.0, fingerprint_dim=fingerprint_bits,
            enable_atom_count_head=True, enable_atom_type_head=True,
        )
    )
    model = CasmiModel(core, CasmiModelConfig(n_adducts=n_adducts, n_quantiles=n_quantiles))
    return CasmiTrainer(model, CasmiTrainerConfig(n_timesteps=10))


class TestScoreMolecule:
    def test_features_for_fixture_molecule(self, casmi_frame):
        df = prepare_spectra_frame(casmi_frame, require_smiles=True)
        vocab = AdductVocabulary.fit(df["adduct"])
        # Cycloheptatriene and norbornadiene are C7H8 isomers of toluene.
        table = build_structure_table([s for _, s in CASMI_MOLECULES] + ["C1=CC=CC=CC1", "C1=CC2CC1C=C2"])
        pool = CandidatePool.build(table, k=K, fingerprint_bits=64, n_quantiles=8)
        trainer = _tiny_trainer(len(vocab))
        groups = group_by_molecule(df)
        rows = df.iloc[groups["m_toluene"]]
        tensors = spectra_to_tensors(rows, adduct_vocab=vocab, max_peaks=32)
        assert tensors is not None and tensors.n_spectra == 2
        scores = score_molecule(
            trainer, pool, tensors, molecule_id="m_toluene",
            config=ScoringConfig(ppm=20.0, n_samples=2, n_steps=3, max_atom_counts=4),
        )
        assert isinstance(scores, CandidateScores)
        assert scores.feature_names == FEATURE_NAMES
        assert scores.features.shape == (len(scores), len(FEATURE_NAMES))
        truth_key = structure_record("Cc1ccccc1")["inchikey14"]
        keys = pool.inchikey14[scores.indices]
        assert truth_key in set(keys)
        # The C7H8 isomers share the mass window and all got projection features.
        assert len(scores) == 3
        for name in FEATURE_NAMES:
            assert np.isfinite(scores.feature(name)).all(), name
        assert np.all((scores.feature("proj_overlap") >= 0) & (scores.feature("proj_overlap") <= 1))
        frame = scores.as_frame(pool)
        assert {"smiles", "inchikey14", "formula", "pool_index", *FEATURE_NAMES} <= set(frame.columns)

    def test_empty_window(self, casmi_frame):
        df = prepare_spectra_frame(casmi_frame, require_smiles=True)
        vocab = AdductVocabulary.fit(df["adduct"])
        pool = CandidatePool.build(build_structure_table(["CCO"]), k=K, fingerprint_bits=64, n_quantiles=8)
        trainer = _tiny_trainer(len(vocab))
        rows = df.iloc[group_by_molecule(df)["m_caffeine"]]
        tensors = spectra_to_tensors(rows, adduct_vocab=vocab, max_peaks=32)
        scores = score_molecule(trainer, pool, tensors, molecule_id="m_caffeine", config=ScoringConfig(n_samples=1, n_steps=2))
        assert len(scores) == 0 and scores.features.shape == (0, len(FEATURE_NAMES))

    def test_projection_can_be_disabled(self, casmi_frame):
        df = prepare_spectra_frame(casmi_frame, require_smiles=True)
        vocab = AdductVocabulary.fit(df["adduct"])
        pool = CandidatePool.build(build_structure_table([s for _, s in CASMI_MOLECULES]), k=K, fingerprint_bits=64, n_quantiles=8)
        trainer = _tiny_trainer(len(vocab))
        rows = df.iloc[group_by_molecule(df)["m_ethanol"]]
        tensors = spectra_to_tensors(rows, adduct_vocab=vocab, max_peaks=32)
        scores = score_molecule(trainer, pool, tensors, molecule_id="m_ethanol", config=ScoringConfig(use_projection=False))
        assert len(scores) == 1
        assert np.isnan(scores.feature("proj_overlap")).all()
        assert np.isfinite(scores.feature("fp_loglik")).all()
