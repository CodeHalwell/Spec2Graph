"""Official scoring: MRR@25 on tautomer-canonical InChIKey14, plus the submission file rules."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from spec2graph.casmi.chem import inchikey14

MAX_GUESSES = 25
SEP = ";"
METRIC_RDKIT_VERSION = "2026.03.3"


def reciprocal_rank(answer: str, guesses: Sequence[str], k: int = MAX_GUESSES) -> float:
    """1/position of the first guess matching ``answer``; invalid guesses keep their slot."""
    truth = inchikey14(answer)
    return next((1.0 / i for i, g in enumerate(guesses[:k], 1) if truth and inchikey14(g) == truth), 0.0)


def reciprocal_ranks(truth: Mapping[str, str], predictions: Mapping[str, Sequence[str]], k: int = MAX_GUESSES) -> dict[str, float]:
    return {m: reciprocal_rank(a, list(predictions.get(m, ())), k) for m, a in truth.items()}


def mrr(truth: Mapping[str, str], predictions: Mapping[str, Sequence[str]], k: int = MAX_GUESSES) -> float:
    if not truth:
        raise ValueError("truth is empty")
    return float(np.mean(list(reciprocal_ranks(truth, predictions, k).values())))


def bootstrap_ci(values: Sequence[float], n: int = 1000, seed: int = 0, confidence: float = 0.95) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    means = np.random.default_rng(seed).choice(v, size=(n, v.size)).mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def clean_guesses(guesses: Sequence[str], k: int = MAX_GUESSES) -> list[str]:
    """Valid SMILES only, one per InChIKey14, capped at ``k``."""
    seen: set[str] = set()
    out: list[str] = []
    for g in guesses:
        g = g.strip().replace(SEP, "") if isinstance(g, str) else ""
        if (key := inchikey14(g)) and key not in seen:
            seen.add(key)
            out.append(g)
        if len(out) == k:
            break
    return out


class Submission:
    """Ranked guesses per ``molecule_id`` in the competition's file format."""

    def __init__(self, predictions: Mapping[str, Sequence[str]], molecule_ids: Sequence[str] | None = None,
                 fallback: str = "C", clean: bool = True):
        ids = list(molecule_ids) if molecule_ids is not None else list(predictions)
        prep = clean_guesses if clean else list
        self.rows = {str(m): prep(predictions.get(m, ())) or [fallback] for m in ids}

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"molecule_id": list(self.rows), "smiles": [SEP.join(g) for g in self.rows.values()]})

    @staticmethod
    def problems(frame: pd.DataFrame, molecule_ids: Sequence[str] | None = None) -> list[str]:
        """Reasons the competition would reject ``frame`` (empty when valid)."""
        if missing := [c for c in ("molecule_id", "smiles") if c not in frame.columns]:
            return [f"missing columns {missing}"]
        ids, smiles = frame["molecule_id"].astype(str), frame["smiles"].astype(str)
        checks = {
            "empty": frame.empty,
            "null values": frame.isna().any().any(),
            "repeated molecule_id": ids.duplicated().any(),
            "empty smiles": (smiles.str.strip() == "").any(),
            f"more than {MAX_GUESSES} guesses": (smiles.str.count(SEP) >= MAX_GUESSES).any(),
            "molecule_id set mismatch": molecule_ids is not None and set(ids) != set(map(str, molecule_ids)),
        }
        return [reason for reason, failed in checks.items() if failed]

    def write(self, path: str | Path) -> pd.DataFrame:
        frame = self.frame()
        if problems := self.problems(frame):
            raise ValueError(f"submission would be rejected: {problems}")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return frame

    @classmethod
    def read(cls, path: str | Path) -> "Submission":
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if problems := cls.problems(frame):
            raise ValueError(f"invalid submission: {problems}")
        return cls({m: s.split(SEP) for m, s in zip(frame["molecule_id"], frame["smiles"])}, clean=False)

    def score(self, truth: Mapping[str, str], k: int = MAX_GUESSES) -> float:
        return mrr(truth, self.rows, k)
