"""Tests for :mod:`spec2graph.casmi.denovo`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")
torch = pytest.importorskip("torch")

from spec2graph.casmi.adducts import AdductVocabulary
from spec2graph.casmi.conditioning import CasmiModel, CasmiModelConfig, CasmiTrainer, CasmiTrainerConfig
from spec2graph.casmi.data import group_by_molecule, kekulized_adjacency, prepare_spectra_frame, spectra_to_tensors, structure_record
from spec2graph.casmi.denovo import (
    N_BOND_CLASSES,
    BondTypeSGNO,
    BondTypeTrainer,
    DeNovoConfig,
    adjacency_to_smiles,
    bond_targets_from_adjacency,
    decode_bond_types,
    denovo_candidates,
    enumerate_formulas,
    formula_hydrogens,
    formula_mass,
    implied_hydrogens,
    load_bond_sgno,
    make_denovo_fn,
    rdbe,
    save_bond_sgno,
)
from spec2graph.casmi.metric import inchikey14
from spec2graph.data.elements import ELEMENT_TO_INDEX
from spectral_diffusion import Spec2GraphDiffusion, Spec2GraphDiffusionConfig, SpectralDataProcessor

K = 4


class TestBondTypeSGNO:
    def test_forward_symmetric_with_empty_diagonal(self):
        torch.manual_seed(0)
        sgno = BondTypeSGNO(k=K, hidden_dim=16, num_layers=3)
        eigvecs = torch.randn(2, 5, K)
        elements = torch.randint(0, 12, (2, 5))
        logits = sgno(eigvecs, elements)
        assert logits.shape == (2, 5, 5, N_BOND_CLASSES)
        assert torch.allclose(logits, logits.transpose(1, 2))
        probs = sgno.bond_probabilities(eigvecs, elements)
        diag = torch.diagonal(probs, dim1=1, dim2=2)  # (2, 4, 5)
        assert torch.allclose(diag[:, 0], torch.ones(2, 5))
        with pytest.raises(ValueError):
            sgno(eigvecs)  # elements required
        BondTypeSGNO(k=K, hidden_dim=8, num_layers=2, use_elements=False)(eigvecs)

    def test_targets_and_training(self):
        adj = kekulized_adjacency("CC(=O)O")  # single, double, single
        mask = torch.ones(1, 4, dtype=torch.bool)
        targets = bond_targets_from_adjacency([adj], mask)
        assert targets.shape == (1, 4, 4)
        assert sorted(targets[0][torch.triu(torch.ones(4, 4, dtype=torch.bool), 1)].tolist()) == [0, 0, 0, 1, 1, 2]
        aromatic = np.array([[0, 1.5], [1.5, 0]], dtype=np.float32)
        assert bond_targets_from_adjacency([aromatic], torch.ones(1, 2, dtype=torch.bool))[0, 0, 1] == 1

        torch.manual_seed(0)
        sgno = BondTypeSGNO(k=K, hidden_dim=32, num_layers=3)
        trainer = BondTypeTrainer(sgno)
        optimiser = torch.optim.Adam(sgno.parameters(), lr=1e-2)
        eigvecs = torch.from_numpy(SpectralDataProcessor(k=K).process_smiles("CC(=O)O")).unsqueeze(0)
        elements = torch.tensor([[ELEMENT_TO_INDEX[s] for s in ("C", "C", "O", "O")]])
        first = trainer.train_step(optimiser, eigvecs, mask, elements, [adj])
        for _ in range(60):
            last = trainer.train_step(optimiser, eigvecs, mask, elements, [adj])
        assert last < first
        probs = sgno.bond_probabilities(eigvecs, elements)[0].detach().numpy()
        decoded = decode_bond_types(probs, ["C", "C", "O", "O"], h_budget=4)
        assert adjacency_to_smiles(decoded, ["C", "C", "O", "O"]) == "CC(=O)O"


class TestDecode:
    def _one_hot(self, adjacency: np.ndarray, confidence: float = 0.9) -> np.ndarray:
        n = adjacency.shape[0]
        probs = np.full((n, n, N_BOND_CLASSES), (1 - confidence) / (N_BOND_CLASSES - 1), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                probs[i, j, int(adjacency[i, j])] = confidence
        return probs

    def test_reconstructs_from_confident_probabilities(self):
        for smiles in ("CC(=O)O", "c1ccccc1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "C#N"):
            record = structure_record(smiles)
            adj = kekulized_adjacency(record["smiles"])
            from rdkit import Chem

            atoms = [a.GetSymbol() for a in Chem.MolFromSmiles(record["smiles"]).GetAtoms()]
            decoded = decode_bond_types(self._one_hot(adj), atoms, h_budget=formula_hydrogens(record["formula"]))
            out = adjacency_to_smiles(decoded, atoms)
            assert out is not None and inchikey14(out) == record["inchikey14"], smiles

    def test_hydrogen_budget_repairs_bond_orders(self):
        # Skeleton of acetic acid with every bond single: implied H is 6, budget 4.
        atoms = ["C", "C", "O", "O"]
        single = np.zeros((4, 4), dtype=np.float32)
        single[0, 1] = single[1, 0] = 1
        single[1, 2] = single[2, 1] = 1
        single[1, 3] = single[3, 1] = 1
        probs = self._one_hot(single, confidence=0.6)
        # Make the double bond to atom 2 slightly more plausible than to atom 3.
        probs[1, 2, 2] = probs[2, 1, 2] = 0.3
        assert implied_hydrogens(single, atoms) == 6
        repaired = decode_bond_types(probs, atoms, h_budget=4)
        assert implied_hydrogens(repaired, atoms) == 4
        assert repaired[1, 2] == 2
        assert adjacency_to_smiles(repaired, atoms) == "CC(=O)O"
        # Too few hydrogens: demote a double bond.
        double_everywhere = np.array([[0, 2], [2, 0]], dtype=np.float32)
        probs2 = self._one_hot(double_everywhere, confidence=0.6)
        demoted = decode_bond_types(probs2, ["C", "C"], h_budget=6)
        assert demoted[0, 1] == 1 and implied_hydrogens(demoted, ["C", "C"]) == 6

    def test_validation(self):
        with pytest.raises(ValueError):
            decode_bond_types(np.zeros((2, 2, 3)), ["C", "C"])
        assert adjacency_to_smiles(np.array([[0, 5], [5, 0]], dtype=np.float32), ["C", "C"]) is None
        # Disconnected graphs are not single molecules.
        disconnected = np.zeros((3, 3), dtype=np.float32)
        disconnected[0, 1] = disconnected[1, 0] = 1
        assert adjacency_to_smiles(disconnected, ["C", "C", "O"]) is None
        assert adjacency_to_smiles(disconnected, ["C", "C", "O"], require_connected=False) == "CC.O"


class TestFormulaEnumeration:
    def test_glucose_window(self):
        record = structure_record("OCC1OC(O)C(O)C(O)C1O")
        formulas = enumerate_formulas(record["exact_mass"], ppm=5.0, elements=("C", "N", "O"))
        assert "C6H12O6" in formulas
        assert formulas[0] == "C6H12O6"
        for formula in formulas:
            from spec2graph.eval.decode import parse_formula

            counts = parse_formula(formula)
            counts["H"] = formula_hydrogens(formula)
            assert abs(formula_mass(counts) - record["exact_mass"]) <= record["exact_mass"] * 5e-6
            assert rdbe(counts) >= 0 and abs(rdbe(counts) - round(rdbe(counts))) < 1e-6

    def test_caffeine_with_default_elements(self):
        record = structure_record("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
        formulas = enumerate_formulas(record["exact_mass"], ppm=3.0, max_results=20)
        assert "C8H10N4O2" in formulas
        assert len(formulas) <= 20
        assert enumerate_formulas(float("nan")) == []
        with pytest.raises(ValueError):
            enumerate_formulas(100.0, elements=("N", "O"))

    def test_helpers(self):
        assert formula_hydrogens("C2H6O") == 6 and formula_hydrogens("C6H6") == 6 and formula_hydrogens("CO2") == 0
        assert rdbe({"C": 6, "H": 6}) == 4.0
        assert rdbe({"C": 2, "H": 6, "O": 1}) == 0.0


class TestDeNovoCandidates:
    def test_end_to_end_with_random_weights(self, casmi_frame):
        df = prepare_spectra_frame(casmi_frame, require_smiles=True)
        vocab = AdductVocabulary.fit(df["adduct"])
        torch.manual_seed(0)
        core = Spec2GraphDiffusion(
            Spec2GraphDiffusionConfig(
                d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32,
                k=K, max_atoms=16, max_peaks=32, dropout=0.0, fingerprint_dim=64,
                enable_atom_count_head=True, enable_atom_type_head=True,
            )
        )
        trainer = CasmiTrainer(CasmiModel(core, CasmiModelConfig(n_adducts=len(vocab), n_quantiles=8)), CasmiTrainerConfig(n_timesteps=10))
        sgno = BondTypeSGNO(k=K, hidden_dim=16, num_layers=2)
        rows = df.iloc[group_by_molecule(df)["m_acetic"]]
        tensors = spectra_to_tensors(rows, adduct_vocab=vocab, max_peaks=32)
        config = DeNovoConfig(n_samples=6, n_steps=2, max_per_formula=5)
        out = denovo_candidates(trainer, tensors, ["C2H4O2", "C3H8", "not a formula", "C200H2"], sgno, config=config)
        assert isinstance(out, list)
        keys = [inchikey14(s) for s in out]
        assert all(k is not None for k in keys) and len(set(keys)) == len(keys)
        fn = make_denovo_fn(sgno, config)
        assert isinstance(fn(trainer, tensors, ["C2H4O2"]), list)

    def test_bond_sgno_checkpoint(self, tmp_path):
        sgno = BondTypeSGNO(k=K, hidden_dim=8, num_layers=2)
        path = tmp_path / "bond.pt"
        save_bond_sgno(path, sgno)
        loaded = load_bond_sgno(path)
        assert loaded.config_dict() == sgno.config_dict()
        for key, value in sgno.state_dict().items():
            assert torch.equal(loaded.state_dict()[key], value)
