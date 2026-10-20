"""Tests for :mod:`spec2graph.casmi.adducts`."""

from __future__ import annotations

import pytest

from spec2graph.casmi.adducts import (
    PROTON_MASS,
    AdductVocabulary,
    adduct_polarity,
    canonical_adduct_name,
    neutral_mass,
    parse_adduct,
    try_parse_adduct,
)


class TestParse:
    @pytest.mark.parametrize(
        "name, expected_shift_from_mz",
        [
            ("[M+H]+", -PROTON_MASS),
            ("[M-H]-", PROTON_MASS),
            ("[M+Na]+", -22.989218),
            ("[M+K]+", -38.963158),
            ("[M+NH4]+", -18.033823),
            ("[M+H-H2O]+", 17.003289),
            ("[M+Cl]-", -34.969402),
            ("[M+HCOO]-", -44.998203),
        ],
    )
    def test_singly_charged_neutral_mass(self, name, expected_shift_from_mz):
        mz = 300.0
        # Published adduct tables round the electron mass; 1e-5 Da is 0.03 ppm at 300 Da.
        assert neutral_mass(mz, name) == pytest.approx(mz + expected_shift_from_mz, abs=1e-5)

    def test_multiply_charged(self):
        adduct = parse_adduct("[M+2H]2+")
        assert adduct.charge == 2
        # m/z = (M + 2 protons) / 2
        assert adduct.neutral_mass(151.0) == pytest.approx(2 * 151.0 - 2 * PROTON_MASS, abs=1e-6)
        assert adduct.mz_from_neutral(adduct.neutral_mass(151.0)) == pytest.approx(151.0)

    def test_dimer(self):
        adduct = parse_adduct("[2M+H]+")
        assert adduct.multiplier == 2
        assert adduct.neutral_mass(2 * 100.0 + PROTON_MASS) == pytest.approx(100.0, abs=1e-6)

    def test_alias_formate(self):
        assert neutral_mass(300.0, "[M+FA-H]-") == pytest.approx(neutral_mass(300.0, "[M+HCOO]-"), abs=1e-6)
        assert canonical_adduct_name("[M+FA-H]-") == "[M+HCOO]-"

    def test_charge_spellings(self):
        assert canonical_adduct_name("[M+H]1+") == "[M+H]+"
        assert canonical_adduct_name("[M-H]1-") == "[M-H]-"
        assert canonical_adduct_name("[M+H]+1") == "[M+H]+"
        assert canonical_adduct_name(" M+H ") == "[M+H]+"
        assert canonical_adduct_name("[M+2H]2+") == "[M+2H]2+"

    def test_radical_cation(self):
        assert neutral_mass(100.0, "[M]+") == pytest.approx(100.0 + 0.000549, abs=1e-6)

    def test_unknown_component(self):
        with pytest.raises(ValueError):
            parse_adduct("[M+Xx]+")
        assert try_parse_adduct("[M+Xx]+") is None
        assert try_parse_adduct("") is None
        assert try_parse_adduct(None) is None
        assert neutral_mass(100.0, "nonsense") is None

    def test_bad_mz(self):
        assert neutral_mass(float("nan"), "[M+H]+") is None
        assert neutral_mass(-5.0, "[M+H]+") is None
        assert neutral_mass("abc", "[M+H]+") is None

    def test_polarity(self):
        assert adduct_polarity("[M+H]+") == 1
        assert adduct_polarity("[M-H]-") == -1
        assert adduct_polarity("[M-2H]2-") == -1
        assert adduct_polarity("garbage") is None


class TestVocabulary:
    def test_fit_encode_round_trip(self, tmp_path):
        vocab = AdductVocabulary.fit(["[M+H]+", "[M+H]+", "[M-H]-", "[M+H]1+", "junk"])
        assert len(vocab) == 3  # unk + 2 real
        assert vocab.encode("[M+H]+") == 1  # most frequent first
        assert vocab.encode("[M+H]1+") == 1
        assert vocab.encode("[M-H]-") == 2
        assert vocab.encode("[M+Na]+") == 0
        assert vocab.encode(None) == 0
        assert "[M+H]+" in vocab and "junk" not in vocab
        path = tmp_path / "adducts.json"
        vocab.save(path)
        loaded = AdductVocabulary.load(path)
        assert loaded.to_dict() == vocab.to_dict()
        assert loaded.names == [AdductVocabulary.UNKNOWN, "[M+H]+", "[M-H]-"]
