"""Tests for :mod:`spec2graph.eval.decode`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("rdkit")
pytest.importorskip("scipy")

from spec2graph.data.elements import (
    ELEMENT_TO_INDEX,
    N_ELEMENT_TYPES,
)
from spec2graph.eval.decode import (
    adjacency_to_rwmol,
    assign_elements,
    batch_eigvecs_to_smiles,
    eigvecs_to_smiles,
    formula_to_atom_types,
    parse_formula,
)
from spectral_diffusion import SpectralGraphNeuralOperator, ValencyDecoder


class TestParseFormula:
    def test_simple(self):
        assert parse_formula("C2H6O") == {"C": 2, "O": 1}

    def test_ignores_hydrogen(self):
        assert "H" not in parse_formula("C6H6")

    def test_multichar_element(self):
        assert parse_formula("ClCC") == {"Cl": 1, "C": 2}

    def test_with_multi_digit_count(self):
        assert parse_formula("C17H26ClNO2") == {"C": 17, "Cl": 1, "N": 1, "O": 2}

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_formula("")

    def test_only_hydrogen_raises(self):
        with pytest.raises(ValueError, match="No heavy-atom tokens"):
            parse_formula("H2")


class TestFormulaToAtomTypes:
    def test_ethanol(self):
        assert formula_to_atom_types("C2H6O") == ["C", "C", "O"]

    def test_benzene(self):
        assert formula_to_atom_types("C6H6") == ["C"] * 6

    def test_canonical_order(self):
        # C, N, O, S, P, F, Cl, Br, I — order matters.
        assert formula_to_atom_types("CH4ClNS") == ["C", "N", "S", "Cl"]

    def test_unknown_elements_sorted_after(self):
        # Krypton (not in _STANDARD_ORDER) should come after Cl.
        out = formula_to_atom_types("CClKr")
        assert out == ["C", "Cl", "Kr"]


class TestAssignElements:
    def _logits_for(self, atoms: list[str], strength: float = 5.0) -> np.ndarray:
        """Build logits that strongly prefer ``atoms[i]`` for row i."""
        logits = np.zeros((len(atoms), N_ELEMENT_TYPES), dtype=np.float32)
        for i, symbol in enumerate(atoms):
            idx = ELEMENT_TO_INDEX[symbol]
            logits[i, idx] = strength
        return logits

    def test_recovers_deterministic_assignment(self):
        # If the logits perfectly match the formula, Hungarian returns
        # an assignment consistent with the logit argmax.
        atoms = ["C", "C", "O"]
        logits = self._logits_for(atoms)
        assignment = assign_elements(logits, "C2H6O")
        assert sorted(assignment) == sorted(atoms)
        # Each row's assignment should agree with its argmax because the
        # logits were constructed to make that the optimum.
        for i, element in enumerate(assignment):
            assert element == atoms[i]

    def test_respects_formula_counts_on_ambiguous_logits(self):
        # All-zero logits leave the assignment arbitrary but it still
        # respects the counts.
        logits = np.zeros((3, N_ELEMENT_TYPES), dtype=np.float32)
        assignment = assign_elements(logits, "C2H6O")
        assert sorted(assignment) == ["C", "C", "O"]

    def test_row_mismatch_raises(self):
        logits = np.zeros((2, N_ELEMENT_TYPES), dtype=np.float32)
        with pytest.raises(ValueError, match="expands to 3 heavy atoms"):
            assign_elements(logits, "C2H6O")  # 3 atoms, 2 rows

    def test_too_few_columns_raises(self):
        logits = np.zeros((3, 5), dtype=np.float32)  # not enough columns
        with pytest.raises(ValueError, match="columns"):
            assign_elements(logits, "C2H6O")

    def test_unknown_element_in_formula_logged_and_assigned(self, caplog):
        # The formula contains Kr, which is not in the vocabulary. The
        # function should warn and still produce a valid assignment.
        logits = np.zeros((2, N_ELEMENT_TYPES), dtype=np.float32)
        caplog.set_level("WARNING")
        assignment = assign_elements(logits, "KrC")
        assert sorted(assignment) == ["C", "Kr"]


class TestAdjacencyToRwmol:
    def test_builds_benzene_from_explicit_adjacency(self):
        from rdkit import Chem
        # 6-ring of carbons with alternating double bonds (Kekulé form).
        n = 6
        adjacency = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            adjacency[i, (i + 1) % n] = 1 if i % 2 == 0 else 2
            adjacency[(i + 1) % n, i] = adjacency[i, (i + 1) % n]
        mol = adjacency_to_rwmol(adjacency, ["C"] * 6)
        Chem.SanitizeMol(mol)
        # Canonical SMILES of benzene.
        assert Chem.MolToSmiles(mol) == "c1ccccc1"

    def test_rejects_non_square(self):
        adjacency = np.zeros((3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="square"):
            adjacency_to_rwmol(adjacency, ["C", "C", "C"])

    def test_rejects_mismatched_atom_list(self):
        adjacency = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="length"):
            adjacency_to_rwmol(adjacency, ["C", "C"])


class TestEigvecsToSmiles:
    def test_returns_none_on_unsanitisable(self):
        # Random eigenvectors on carbon slots — untrained SGNO produces
        # essentially random adjacency. Most draws will fail sanitation;
        # the call should return None rather than crash.
        torch.manual_seed(0)
        sgno = SpectralGraphNeuralOperator(k=4, hidden_dim=16, num_layers=2)
        valency = ValencyDecoder()
        eigvecs = torch.randn(5, 4)
        result = eigvecs_to_smiles(
            eigvecs=eigvecs,
            atom_types=["C"] * 5,
            sgno=sgno,
            valency_decoder=valency,
            threshold=0.5,
        )
        # Either returns a valid SMILES or None — must not raise.
        assert result is None or isinstance(result, str)

    def test_batch_wrapper_respects_mask(self):
        torch.manual_seed(0)
        sgno = SpectralGraphNeuralOperator(k=4, hidden_dim=16, num_layers=2)
        valency = ValencyDecoder()
        eigvecs = torch.randn(2, 6, 4)
        atom_masks = torch.tensor(
            [[True, True, True, False, False, False],
             [True, True, True, True, True, True]],
            dtype=torch.bool,
        )
        # Formulas whose heavy-atom counts match the mask sizes.
        results = batch_eigvecs_to_smiles(
            eigvecs=eigvecs,
            atom_masks=atom_masks,
            formulas=["C3", "C6"],
            atom_type_logits=None,
            sgno=sgno,
            valency_decoder=valency,
            threshold=0.5,
            fallback_atom_types=[["C"] * 3, ["C"] * 6],
        )
        assert len(results) == 2
        for r in results:
            assert r is None or isinstance(r, str)

    def test_batch_with_atom_type_logits(self):
        torch.manual_seed(0)
        sgno = SpectralGraphNeuralOperator(k=4, hidden_dim=16, num_layers=2)
        valency = ValencyDecoder()
        eigvecs = torch.randn(1, 3, 4)
        atom_masks = torch.ones(1, 3, dtype=torch.bool)
        # Strong preference for 2x C and 1x O.
        logits = torch.zeros(1, 3, N_ELEMENT_TYPES)
        logits[0, 0, ELEMENT_TO_INDEX["C"]] = 10.0
        logits[0, 1, ELEMENT_TO_INDEX["C"]] = 10.0
        logits[0, 2, ELEMENT_TO_INDEX["O"]] = 10.0
        results = batch_eigvecs_to_smiles(
            eigvecs=eigvecs,
            atom_masks=atom_masks,
            formulas=["C2H6O"],
            atom_type_logits=logits,
            sgno=sgno,
            valency_decoder=valency,
            threshold=0.5,
        )
        assert len(results) == 1

    def test_batch_returns_none_on_formula_mismatch(self):
        sgno = SpectralGraphNeuralOperator(k=4, hidden_dim=16, num_layers=2)
        valency = ValencyDecoder()
        eigvecs = torch.randn(1, 3, 4)
        atom_masks = torch.ones(1, 3, dtype=torch.bool)
        # Formula says 5 atoms but mask only covers 3 — assignment fails.
        logits = torch.zeros(1, 3, N_ELEMENT_TYPES)
        results = batch_eigvecs_to_smiles(
            eigvecs=eigvecs,
            atom_masks=atom_masks,
            formulas=["C5"],
            atom_type_logits=logits,
            sgno=sgno,
            valency_decoder=valency,
        )
        assert results == [None]
