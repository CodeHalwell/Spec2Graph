"""MassSpecGym de novo metrics.

Implements the four metrics the MassSpecGym leaderboard tracks:

* Top-k exact match accuracy (``top_k_accuracy``)
* Top-k max Morgan fingerprint Tanimoto (``top_k_tanimoto``)
* Top-k min MCES distance (``top_k_mces``)
* Validity rate (``validity_rate``)

Plus an :class:`BenchmarkResults` dataclass that aggregates everything
with bootstrap 95% confidence intervals and a helper to render the
benchmark row as markdown.

All metrics canonicalise SMILES via RDKit before comparing — a
prediction that looks different as a string but is the same molecule
counts as a match. Skipping this step is the single biggest source of
silent under-counting in naive implementations.
"""

from __future__ import annotations

import contextlib
import io
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# MCES returns a "large" distance for graphs that cannot be matched within
# the threshold. We use a consistent sentinel so aggregates stay finite.
MCES_SENTINEL_DISTANCE: float = 100.0


# ----------------------------------------------------------------------
# SMILES canonicalisation
# ----------------------------------------------------------------------


def canonicalise(smiles: Optional[str]) -> Optional[str]:
    """Return the RDKit-canonical form of ``smiles`` or ``None``.

    ``None`` input and unparseable strings both return ``None`` so
    callers can uniformly treat failed predictions as invalid.
    """
    if smiles is None:
        return None
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("rdkit is required for canonicalise()") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except (ValueError, RuntimeError):
        return None


# ----------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------


def rank_samples_by_frequency(
    smiles_list: Sequence[Optional[str]],
) -> list[str]:
    """Filter out invalid entries, canonicalise, dedupe by frequency.

    Returns SMILES ordered by sampling frequency (descending). Ties are
    broken by first-occurrence order so the output is deterministic
    given a fixed input.
    """
    canonical = [canonicalise(s) for s in smiles_list]
    counter: Counter[str] = Counter()
    # Iterate in order so ties resolve in favour of earlier-seen items
    # (Counter preserves insertion order in Python 3.7+).
    for c in canonical:
        if c is None:
            continue
        counter[c] += 1
    return [s for s, _ in counter.most_common()]


# ----------------------------------------------------------------------
# Morgan fingerprint + Tanimoto
# ----------------------------------------------------------------------


def _morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_similarity(
    smiles_a: str,
    smiles_b: str,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    """Morgan fingerprint Tanimoto similarity between two SMILES."""
    from rdkit import DataStructs

    fp_a = _morgan_fp(smiles_a, radius=radius, n_bits=n_bits)
    fp_b = _morgan_fp(smiles_b, radius=radius, n_bits=n_bits)
    if fp_a is None or fp_b is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


# ----------------------------------------------------------------------
# MCES
# ----------------------------------------------------------------------


def mces_distance(
    smiles_a: str,
    smiles_b: str,
    *,
    threshold: float = 10.0,
) -> float:
    """Myopic MCES (maximum common edge subgraph) distance.

    Wraps :func:`myopic_mces.MCES`. The upstream library writes solver
    progress to stdout; we redirect that to a string buffer and discard
    it so benchmark runs don't drown the terminal.
    """
    try:
        from myopic_mces import MCES
    except ImportError as exc:
        raise ImportError(
            "myopic-mces is required for MCES metrics; "
            "install via `pip install myopic-mces`."
        ) from exc

    # MCES returns (index, distance, time, exact_flag). The distance is
    # what we want; index/time/exact_flag are ignored.
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            result = MCES(smiles_a, smiles_b, threshold=threshold, catch_errors=True)
    except Exception as exc:  # myopic_mces can raise on ILP issues
        logger.debug("MCES raised: %s", exc)
        return MCES_SENTINEL_DISTANCE

    # Expected shape: (index, distance, time, exact_flag)
    if not isinstance(result, tuple) or len(result) < 2:
        return MCES_SENTINEL_DISTANCE
    distance = float(result[1])
    if np.isnan(distance) or np.isinf(distance):
        return MCES_SENTINEL_DISTANCE
    return distance


# ----------------------------------------------------------------------
# Per-example metrics
# ----------------------------------------------------------------------


def top_k_accuracy(
    gt_smiles: str,
    predictions: Sequence[str],
    k: int,
) -> float:
    """1.0 if canonical ground truth is in the top-k predictions, else 0.0.

    ``predictions`` is assumed to be already ranked (strongest first) and
    already canonicalised — typically the output of
    :func:`rank_samples_by_frequency`.
    """
    gt = canonicalise(gt_smiles)
    if gt is None:
        return 0.0
    return 1.0 if gt in predictions[:k] else 0.0


def top_k_tanimoto(
    gt_smiles: str,
    predictions: Sequence[str],
    k: int,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    """Max Morgan Tanimoto similarity over the top-k predictions.

    Returns ``0.0`` if ``predictions`` is empty or all invalid.
    """
    best = 0.0
    for pred in predictions[:k]:
        sim = tanimoto_similarity(gt_smiles, pred, radius=radius, n_bits=n_bits)
        if sim > best:
            best = sim
    return best


def top_k_mces(
    gt_smiles: str,
    predictions: Sequence[str],
    k: int,
    *,
    threshold: float = 10.0,
) -> float:
    """Min MCES distance over the top-k predictions.

    Returns :data:`MCES_SENTINEL_DISTANCE` if ``predictions`` is empty.
    """
    if not predictions:
        return MCES_SENTINEL_DISTANCE
    best = MCES_SENTINEL_DISTANCE
    for pred in predictions[:k]:
        d = mces_distance(gt_smiles, pred, threshold=threshold)
        if d < best:
            best = d
    return best


def validity_rate(predictions: Sequence[Optional[str]]) -> float:
    """Fraction of predictions that are non-None and parse with RDKit.

    An empty input returns ``0.0`` rather than raising — a benchmark run
    with no valid predictions should cleanly report 0% validity, not
    crash.
    """
    if not predictions:
        return 0.0
    valid = sum(1 for s in predictions if canonicalise(s) is not None)
    return valid / len(predictions)


# ----------------------------------------------------------------------
# Aggregate results
# ----------------------------------------------------------------------


def _bootstrap_mean_ci(
    values: Sequence[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) bootstrap CI for ``values``.

    Uses simple percentile bootstrap with replacement. ``confidence``
    defaults to 0.95, matching MassSpecGym's baseline reporting.
    """
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(values, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(arr)
    resampled = rng.choice(arr, size=(n_resamples, n), replace=True)
    means = resampled.mean(axis=1)
    lower = float(np.percentile(means, 100 * (1 - confidence) / 2))
    upper = float(np.percentile(means, 100 * (1 + confidence) / 2))
    return float(arr.mean()), lower, upper


@dataclass
class BenchmarkResults:
    """Aggregated metrics for a full benchmark run."""

    top_1_accuracy: float
    top_10_accuracy: float
    top_1_tanimoto: float
    top_10_tanimoto: float
    top_1_mces: float
    top_10_mces: float
    validity: float
    n_examples: int
    n_samples_per_example: int

    # 95% bootstrap CIs for the top-k accuracy metrics.
    top_1_accuracy_ci: tuple[float, float] = (float("nan"), float("nan"))
    top_10_accuracy_ci: tuple[float, float] = (float("nan"), float("nan"))

    # Per-example series, useful for downstream analysis / reruns.
    per_example: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "top_1_accuracy": self.top_1_accuracy,
            "top_10_accuracy": self.top_10_accuracy,
            "top_1_tanimoto": self.top_1_tanimoto,
            "top_10_tanimoto": self.top_10_tanimoto,
            "top_1_mces": self.top_1_mces,
            "top_10_mces": self.top_10_mces,
            "validity": self.validity,
            "n_examples": self.n_examples,
            "n_samples_per_example": self.n_samples_per_example,
            "top_1_accuracy_ci": list(self.top_1_accuracy_ci),
            "top_10_accuracy_ci": list(self.top_10_accuracy_ci),
        }

    def to_markdown_row(self, model_name: str) -> str:
        """Render a single markdown table row suitable for the results doc."""
        return (
            f"| {model_name} "
            f"| {100 * self.top_1_accuracy:.2f} "
            f"| {100 * self.top_10_accuracy:.2f} "
            f"| {self.top_1_tanimoto:.3f} "
            f"| {self.top_10_tanimoto:.3f} "
            f"| {self.top_1_mces:.2f} "
            f"| {self.top_10_mces:.2f} "
            f"| {100 * self.validity:.1f} "
            f"| {self.n_examples} |"
        )


def aggregate_per_example(
    per_example: Sequence[dict],
    n_samples_per_example: int,
    *,
    n_resamples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> BenchmarkResults:
    """Aggregate per-example metric dicts into a :class:`BenchmarkResults`.

    Each dict must expose the keys ``top_1_accuracy``, ``top_10_accuracy``,
    ``top_1_tanimoto``, ``top_10_tanimoto``, ``top_1_mces``,
    ``top_10_mces``, ``validity``. See :func:`benchmark_model` for the
    canonical producer.
    """
    if not per_example:
        raise ValueError("per_example is empty; nothing to aggregate.")

    def _col(key: str) -> list[float]:
        return [float(r[key]) for r in per_example]

    mean_top1, lo1, hi1 = _bootstrap_mean_ci(
        _col("top_1_accuracy"), n_resamples=n_resamples, rng=rng
    )
    mean_top10, lo10, hi10 = _bootstrap_mean_ci(
        _col("top_10_accuracy"), n_resamples=n_resamples, rng=rng
    )

    return BenchmarkResults(
        top_1_accuracy=mean_top1,
        top_10_accuracy=mean_top10,
        top_1_tanimoto=float(np.mean(_col("top_1_tanimoto"))),
        top_10_tanimoto=float(np.mean(_col("top_10_tanimoto"))),
        top_1_mces=float(np.mean(_col("top_1_mces"))),
        top_10_mces=float(np.mean(_col("top_10_mces"))),
        validity=float(np.mean(_col("validity"))),
        n_examples=len(per_example),
        n_samples_per_example=n_samples_per_example,
        top_1_accuracy_ci=(lo1, hi1),
        top_10_accuracy_ci=(lo10, hi10),
        per_example=list(per_example),
    )
