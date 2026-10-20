"""Adduct arithmetic: precursor m/z <-> neutral mass, plus an adduct vocabulary for embeddings."""

from __future__ import annotations

import functools
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

ELECTRON_MASS = 0.000548579909
PROTON_MASS = 1.007276466621
# Neutral monoisotopic masses of the pieces that appear in adduct strings.
COMPONENT_MASSES = {
    "H": 1.00782503207, "Na": 22.9897692809, "K": 38.96370668, "Li": 7.01600455, "NH4": 18.03437413, "NH3": 17.02654910,
    "H2O": 18.0105646863, "Cl": 34.96885268, "Br": 78.9183371, "I": 126.904473, "F": 18.99840322,
    "HCOO": 44.99765427, "FA": 46.00547930, "HCOOH": 46.00547930, "CH3COO": 59.01330434, "Hac": 60.02112937,
    "HAc": 60.02112937, "CH3COOH": 60.02112937, "ACN": 41.02654910, "CH3CN": 41.02654910, "MeOH": 32.02621475,
    "CH3OH": 32.02621475, "DMSO": 78.01393598, "CO2": 43.98982924, "CO": 27.99491462, "TFA": 113.99286,
}
ALIASES = {
    "[M+FA-H]-": "[M+HCOO]-", "[M+HCOOH-H]-": "[M+HCOO]-", "[M+Hac-H]-": "[M+CH3COO]-", "[M+HAc-H]-": "[M+CH3COO]-",
    "[M+CH3COOH-H]-": "[M+CH3COO]-", "[M-H2O+H]+": "[M+H-H2O]+", "[M-H2O-H]-": "[M-H-H2O]-", "[M]+.": "[M]+", "[M]-.": "[M]-",
}
_ADDUCT = re.compile(r"^\[(\d*)M((?:[+-]\d*[A-Za-z][A-Za-z0-9]*)*)\](?:(\d*)([+-])|([+-])(\d*))$")
_TOKEN = re.compile(r"([+-])(\d*)([A-Za-z][A-Za-z0-9]*)")


@dataclass(frozen=True)
class Adduct:
    """``[nM ± components]z±``: ion mass = n·M + shift − z·mₑ and m/z = ion mass / |z|."""

    name: str
    multiplier: int
    charge: int
    shift: float

    @property
    def polarity(self) -> int:
        return 1 if self.charge > 0 else -1

    def neutral_mass(self, mz: float) -> float:
        return (mz * abs(self.charge) - self.shift + self.charge * ELECTRON_MASS) / self.multiplier

    def mz(self, neutral_mass: float) -> float:
        return (self.multiplier * neutral_mass + self.shift - self.charge * ELECTRON_MASS) / abs(self.charge)


def _normalise(name: str) -> str:
    s = re.sub(r"\s+", "", name)
    if s and not s.startswith("["):  # bare "M+H" / "M-H": singly charged, sign from the first operation
        s = f"[{s.rstrip('+-')}]{'-' if s.startswith('M-') else '+'}"
    return ALIASES.get(s, s)


@functools.lru_cache(maxsize=1024)
def parse_adduct(name: str) -> Adduct:
    """Parse an adduct string; ``ValueError`` for unknown components or syntax."""
    if (m := _ADDUCT.match(_normalise(name))) is None:
        raise ValueError(f"unrecognised adduct {name!r}")
    mult, body, z1, s1, s2, z2 = m.groups()
    sign, magnitude, mult = s1 or s2, int(z1 or z2 or 1), int(mult or 1)
    if magnitude == 0:
        raise ValueError(f"zero charge in {name!r}")
    shift = 0.0
    for op, count, component in _TOKEN.findall(body):
        if component not in COMPONENT_MASSES:
            raise ValueError(f"unknown component {component!r} in {name!r}")
        shift += (1 if op == "+" else -1) * int(count or 1) * COMPONENT_MASSES[component]
    canonical = f"[{mult if mult > 1 else ''}M{body}]{magnitude if magnitude > 1 else ''}{sign}"
    return Adduct(canonical, mult, magnitude if sign == "+" else -magnitude, shift)


def try_parse(name: object) -> Optional[Adduct]:
    try:
        return parse_adduct(name) if isinstance(name, str) and name.strip() else None
    except ValueError:
        return None


def neutral_mass(precursor_mz: object, adduct: object) -> float:
    """Neutral monoisotopic mass, or NaN when the adduct or m/z is unusable."""
    try:
        mz = float(precursor_mz)
    except (TypeError, ValueError):
        return float("nan")
    a = try_parse(adduct)
    return a.neutral_mass(mz) if a and mz > 0 else float("nan")


class AdductVocabulary:
    """Canonical adduct name -> embedding index; index 0 is reserved for unknown adducts."""

    UNKNOWN = "<unk>"

    def __init__(self, names: Iterable[str] = ()):
        self.index = {self.UNKNOWN: 0}
        for a in filter(None, map(try_parse, names)):
            self.index.setdefault(a.name, len(self.index))

    @classmethod
    def fit(cls, names: Iterable[str]) -> "AdductVocabulary":
        counts = Counter(a.name for a in map(try_parse, names) if a)
        return cls(name for name, _ in counts.most_common())

    def encode(self, name: object) -> int:
        a = try_parse(name)
        return self.index.get(a.name, 0) if a else 0

    def __len__(self) -> int:
        return len(self.index)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.index))

    @classmethod
    def load(cls, path: str | Path) -> "AdductVocabulary":
        vocab = cls()
        vocab.index = json.loads(Path(path).read_text())
        return vocab
