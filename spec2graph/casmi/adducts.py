"""Adduct arithmetic: precursor m/z <-> neutral monoisotopic mass.

The competition gives each spectrum a precursor m/z and an adduct string
such as ``[M+H]+``, ``[M-H]-``, ``[M+Na]+`` or ``[M+2H]2+`` but no
molecular formula. Everything downstream (candidate retrieval, formula
enumeration, mass-error features) works on the neutral mass, so this
module converts between the two.

Rather than a fixed table, adduct strings are parsed generically::

    [nM (+|-) count component ... ] charge sign

The ion mass is ``n * M + sum(component masses) - z * m_e`` with ``z``
signed (a cation has lost an electron), and ``m/z = ion mass / |z|``.
Component masses are neutral monoisotopic masses, so ``[M+H]+`` resolves
to the familiar ``M + 1.007276`` once the electron is subtracted.

Unknown components raise :class:`ValueError` so bad rows can be dropped
explicitly instead of producing a silently wrong mass.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

ELECTRON_MASS: float = 0.000548579909
PROTON_MASS: float = 1.007276466621

# Neutral monoisotopic masses of the pieces that appear in adduct strings.
COMPONENT_MASSES: dict[str, float] = {
    "H": 1.00782503207,
    "H2": 2.01565006414,
    "Na": 22.9897692809,
    "K": 38.96370668,
    "Li": 7.01600455,
    "NH4": 18.03437413,
    "NH3": 17.02654910,
    "H2O": 18.0105646863,
    "O": 15.99491462,
    "Cl": 34.96885268,
    "Br": 78.9183371,
    "I": 126.904473,
    "F": 18.99840322,
    "HCOO": 44.99765427,
    "HCO2": 44.99765427,
    "FA": 46.00547930,
    "HCOOH": 46.00547930,
    "CH3COO": 59.01330434,
    "CH3CO2": 59.01330434,
    "OAc": 59.01330434,
    "Hac": 60.02112937,
    "HAc": 60.02112937,
    "AcOH": 60.02112937,
    "CH3COOH": 60.02112937,
    "ACN": 41.02654910,
    "CH3CN": 41.02654910,
    "MeOH": 32.02621475,
    "CH3OH": 32.02621475,
    "IPA": 60.05751488,
    "DMSO": 78.01393598,
    "CO2": 43.98982924,
    "CO": 27.99491462,
    "TFA": 113.99286,
    "C2HF3O2": 113.99286,
    "HSO4": 96.96010,
    "H2PO4": 96.96962,
}

_ADDUCT_RE = re.compile(
    r"^\[(?P<mult>\d*)M(?P<body>(?:[+-]\d*[A-Za-z][A-Za-z0-9]*)*)\]"
    r"(?:(?P<charge_a>\d*)(?P<sign_a>[+-])|(?P<sign_b>[+-])(?P<charge_b>\d*))$"
)
_TOKEN_RE = re.compile(r"([+-])(\d*)([A-Za-z][A-Za-z0-9]*)")

# Spellings seen in public libraries that the regex does not cover directly.
ADDUCT_ALIASES: dict[str, str] = {
    "M+H": "[M+H]+",
    "M-H": "[M-H]-",
    "M+NH4": "[M+NH4]+",
    "M+Na": "[M+Na]+",
    "M+K": "[M+K]+",
    "[M+H]": "[M+H]+",
    "[M-H]": "[M-H]-",
    "[M]+.": "[M]+",
    "[M]-.": "[M]-",
    "[M+H-H2O]+": "[M+H-H2O]+",
    "[M-H2O+H]+": "[M+H-H2O]+",
    "[M-H-H2O]-": "[M-H-H2O]-",
    "[M-H2O-H]-": "[M-H-H2O]-",
    "[M+FA-H]-": "[M+HCOO]-",
    "[M+HCOOH-H]-": "[M+HCOO]-",
    "[M+Hac-H]-": "[M+CH3COO]-",
    "[M+HAc-H]-": "[M+CH3COO]-",
    "[M+CH3COOH-H]-": "[M+CH3COO]-",
}


@dataclass(frozen=True)
class Adduct:
    """A parsed adduct. ``charge`` is signed; ``shift`` is the neutral mass added."""

    name: str
    multiplier: int
    charge: int
    shift: float

    @property
    def polarity(self) -> int:
        return 1 if self.charge > 0 else -1

    def neutral_mass(self, precursor_mz: float) -> float:
        """Neutral monoisotopic mass of M given the ion's m/z."""
        ion_mass = float(precursor_mz) * abs(self.charge)
        return (ion_mass - self.shift + self.charge * ELECTRON_MASS) / self.multiplier

    def mz_from_neutral(self, neutral_mass: float) -> float:
        """Inverse of :meth:`neutral_mass`."""
        ion_mass = self.multiplier * float(neutral_mass) + self.shift - self.charge * ELECTRON_MASS
        return ion_mass / abs(self.charge)


def normalise_adduct_name(name: str) -> str:
    """Normalise spacing/aliases so equivalent spellings share one key."""
    if not isinstance(name, str):
        raise ValueError(f"Adduct must be a string; got {type(name).__name__}.")
    cleaned = re.sub(r"\s+", "", name)
    cleaned = ADDUCT_ALIASES.get(cleaned, cleaned)
    if cleaned and not cleaned.startswith("[") and "M" in cleaned:
        # Bare "M+H" style without an explicit charge: assume singly charged,
        # with the sign taken from the leading operation.
        polarity = "-" if cleaned.endswith("-") or "M-H" in cleaned else "+"
        cleaned = f"[{cleaned.rstrip('+-')}]{polarity}"
        cleaned = ADDUCT_ALIASES.get(cleaned, cleaned)
    return cleaned


@functools.lru_cache(maxsize=1024)
def parse_adduct(name: str) -> Adduct:
    """Parse an adduct string. Raises :class:`ValueError` when unsupported."""
    cleaned = normalise_adduct_name(name)
    match = _ADDUCT_RE.match(cleaned)
    if match is None:
        raise ValueError(f"Unrecognised adduct {name!r}.")
    multiplier = int(match.group("mult") or 1)
    sign = match.group("sign_a") or match.group("sign_b")
    charge_digits = match.group("charge_a") or match.group("charge_b") or ""
    magnitude = int(charge_digits) if charge_digits else 1
    if magnitude == 0:
        raise ValueError(f"Adduct {name!r} has zero charge.")
    charge = magnitude if sign == "+" else -magnitude

    shift = 0.0
    for op, count_str, component in _TOKEN_RE.findall(match.group("body")):
        if component not in COMPONENT_MASSES:
            raise ValueError(f"Unknown adduct component {component!r} in {name!r}.")
        count = int(count_str) if count_str else 1
        mass = COMPONENT_MASSES[component] * count
        shift += mass if op == "+" else -mass

    canonical = f"[{multiplier if multiplier > 1 else ''}M{match.group('body')}]"
    canonical += f"{magnitude if magnitude > 1 else ''}{sign}"
    return Adduct(name=canonical, multiplier=multiplier, charge=charge, shift=shift)


def try_parse_adduct(name: object) -> Optional[Adduct]:
    """Like :func:`parse_adduct` but returns ``None`` instead of raising."""
    if not isinstance(name, str) or not name.strip():
        return None
    try:
        return parse_adduct(name)
    except ValueError:
        return None


def neutral_mass(precursor_mz: float, adduct: str) -> Optional[float]:
    """Neutral mass for a precursor m/z and adduct, or ``None`` if unsupported."""
    parsed = try_parse_adduct(adduct)
    if parsed is None:
        return None
    try:
        value = float(precursor_mz)
    except (TypeError, ValueError):
        return None
    if value != value or value <= 0:  # NaN or non-physical
        return None
    return parsed.neutral_mass(value)


def adduct_polarity(adduct: str) -> Optional[int]:
    """``+1`` for cations, ``-1`` for anions, ``None`` when unparseable."""
    parsed = try_parse_adduct(adduct)
    return None if parsed is None else parsed.polarity


def canonical_adduct_name(adduct: str) -> Optional[str]:
    """Canonical spelling (e.g. ``[M+H]1+`` -> ``[M+H]+``) or ``None``."""
    parsed = try_parse_adduct(adduct)
    return None if parsed is None else parsed.name


class AdductVocabulary:
    """Integer vocabulary over canonical adduct names for an embedding table.

    Index ``0`` is reserved for unknown/unparseable adducts so the model
    always has a well-defined row to fall back to at test time.
    """

    UNKNOWN: str = "<unk>"

    def __init__(self, names: Iterable[str] = ()) -> None:
        self._index: dict[str, int] = {self.UNKNOWN: 0}
        for name in names:
            self.add(name)

    def add(self, name: str) -> int:
        canonical = canonical_adduct_name(name)
        if canonical is None:
            return 0
        if canonical not in self._index:
            self._index[canonical] = len(self._index)
        return self._index[canonical]

    @classmethod
    def fit(cls, names: Iterable[str], min_count: int = 1) -> "AdductVocabulary":
        counts: dict[str, int] = {}
        for name in names:
            canonical = canonical_adduct_name(name)
            if canonical is None:
                continue
            counts[canonical] = counts.get(canonical, 0) + 1
        ordered = sorted(
            (name for name, count in counts.items() if count >= min_count),
            key=lambda n: (-counts[n], n),
        )
        return cls(ordered)

    def encode(self, name: object) -> int:
        canonical = canonical_adduct_name(name) if isinstance(name, str) else None
        if canonical is None:
            return 0
        return self._index.get(canonical, 0)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, name: object) -> bool:
        return self.encode(name) != 0

    @property
    def names(self) -> list[str]:
        return [name for name, _ in sorted(self._index.items(), key=lambda kv: kv[1])]

    def to_dict(self) -> dict[str, int]:
        return dict(self._index)

    @classmethod
    def from_dict(cls, mapping: dict[str, int]) -> "AdductVocabulary":
        vocab = cls()
        for name, index in sorted(mapping.items(), key=lambda kv: kv[1]):
            if name == cls.UNKNOWN:
                continue
            vocab._index[name] = index
        return vocab

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "AdductVocabulary":
        return cls.from_dict(json.loads(Path(path).read_text()))


__all__ = [
    "ADDUCT_ALIASES",
    "Adduct",
    "AdductVocabulary",
    "COMPONENT_MASSES",
    "ELECTRON_MASS",
    "PROTON_MASS",
    "adduct_polarity",
    "canonical_adduct_name",
    "neutral_mass",
    "normalise_adduct_name",
    "parse_adduct",
    "try_parse_adduct",
]
