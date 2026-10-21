"""Tests for :mod:`spec2graph.casmi.adducts`."""

from __future__ import annotations

import pytest

from spec2graph.casmi.adducts import PROTON_MASS, AdductVocabulary, neutral_mass, parse_adduct, try_parse


@pytest.mark.parametrize("name, shift", [("[M+H]+", -PROTON_MASS), ("[M-H]-", PROTON_MASS), ("[M+Na]+", -22.989218),
                                         ("[M+NH4]+", -18.033823), ("[M+H-H2O]+", 17.003289), ("[M+Cl]-", -34.969402), ("[M+HCOO]-", -44.998203)])
def test_singly_charged(name, shift):
    assert neutral_mass(300.0, name) == pytest.approx(300.0 + shift, abs=1e-5)  # tables round the electron mass


def test_charges_multipliers_and_aliases():
    a = parse_adduct("[M+2H]2+")
    assert a.charge == 2 and a.neutral_mass(151.0) == pytest.approx(302.0 - 2 * PROTON_MASS, abs=1e-6) and a.mz(a.neutral_mass(151.0)) == pytest.approx(151.0)
    assert parse_adduct("[2M+H]+").neutral_mass(200.0 + PROTON_MASS) == pytest.approx(100.0, abs=1e-6)
    assert neutral_mass(300.0, "[M+FA-H]-") == pytest.approx(neutral_mass(300.0, "[M+HCOO]-"), abs=1e-6)
    assert [parse_adduct(s).name for s in ("[M+H]1+", "[M-H]1-", "[M+H]+1", " M+H ", "M-H", "[M+2H]2+")] == ["[M+H]+", "[M-H]-", "[M+H]+", "[M+H]+", "[M-H]-", "[M+2H]2+"]
    assert neutral_mass(100.0, "[M]+") == pytest.approx(100.000549, abs=1e-6)
    assert parse_adduct("[M-2H]2-").polarity == -1 and parse_adduct("[M+K]+").polarity == 1


def test_competition_test_set_adducts():
    names = ("[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+", "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-")
    assert all(try_parse(name) for name in names)
    assert parse_adduct("[M+CH2O2-H]-").name == "[M+HCOO]-"
    assert neutral_mass(300.0, "[M-2H2O+H]+") == pytest.approx(300.0 - PROTON_MASS + 2 * 18.0105646863, abs=1e-6)


def test_unusable_inputs():
    with pytest.raises(ValueError):
        parse_adduct("[M+Xx]+")
    assert try_parse("[M+Xx]+") is None and try_parse("") is None and try_parse(None) is None
    assert all(neutral_mass(mz, a) != neutral_mass(mz, a) for mz, a in ((100.0, "nonsense"), (float("nan"), "[M+H]+"), (-5.0, "[M+H]+"), ("abc", "[M+H]+")))


def test_vocabulary(tmp_path):
    vocab = AdductVocabulary.fit(["[M+H]+", "[M+H]+", "[M-H]-", "[M+H]1+", "junk"])
    assert len(vocab) == 3 and vocab.encode("[M+H]1+") == 1 and vocab.encode("[M-H]-") == 2
    assert vocab.encode("[M+Na]+") == 0 and vocab.encode(None) == 0
    vocab.save(tmp_path / "adducts.json")
    assert AdductVocabulary.load(tmp_path / "adducts.json").index == vocab.index
