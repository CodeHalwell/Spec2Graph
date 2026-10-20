"""Official Enveda CASMI 2026 scoring and submission handling.

The competition scores **Mean Reciprocal Rank at 25** (MRR@25). For each
``molecule_id`` the submission lists up to 25 SMILES, best guess first. A
guess is correct when, after RDKit tautomer canonicalisation, the first
block of its InChIKey (the *InChIKey14*, i.e. the connectivity layer)
equals that of the answer. Stereochemistry and tautomer form are ignored.
So is acid/base protonation state: standard InChI records it in the ``/p``
layer, which sits outside the first block, so an acid and its anion share
a key. Charges that change connectivity (quaternary ammonium, for example)
do not collapse.

The competition pins RDKit at ``2026.03.3`` for the canonicalisation. Local
scores computed with another RDKit release can differ on a handful of
tautomer edge cases, so treat the metric notebook as the reference and
pin the same version for final validation.

Submission rules enforced by :func:`validate_submission`:

* columns ``molecule_id`` and ``smiles``, header included;
* every ``molecule_id`` exactly once, no nulls or empty fields;
* at most 25 semicolon-separated guesses per molecule.

Nothing here touches the core model. The only project dependency is the
``MAX_SMILES_LENGTH`` guard, which this module applies before every RDKit
parse (see ``.jules/sentinel.md``).
"""

from __future__ import annotations

import contextlib
import functools
import logging
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from spectral_diffusion import MAX_SMILES_LENGTH

logger = logging.getLogger(__name__)

MAX_GUESSES: int = 25
GUESS_SEPARATOR: str = ";"
METRIC_RDKIT_VERSION: str = "2026.03.3"
MOLECULE_ID_COLUMN: str = "molecule_id"
SMILES_COLUMN: str = "smiles"

# RDKit's tautomer enumerator is expensive to build; create it once.
_TAUTOMER_ENUMERATOR = None


@contextlib.contextmanager
def _quiet_rdkit() -> Iterator[None]:
    """Silence RDKit's C++ logging for the duration of the block."""
    try:
        from rdkit import rdBase
    except ImportError:  # pragma: no cover - rdkit is a hard dependency
        yield
        return
    blocker = getattr(rdBase, "BlockLogs", None)
    if blocker is None:  # pragma: no cover - very old RDKit
        yield
        return
    with blocker():
        yield


def _get_tautomer_enumerator():
    global _TAUTOMER_ENUMERATOR
    if _TAUTOMER_ENUMERATOR is None:
        from rdkit.Chem.MolStandardize import rdMolStandardize

        _TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
    return _TAUTOMER_ENUMERATOR


@functools.lru_cache(maxsize=1 << 16)
def inchikey14(smiles: Optional[str]) -> Optional[str]:
    """Return the tautomer-canonical InChIKey14 of ``smiles`` or ``None``.

    Mirrors the competition's matching rule: parse, canonicalise the
    tautomer with :class:`rdkit.Chem.MolStandardize.rdMolStandardize.TautomerEnumerator`,
    take the first 14 characters of the InChIKey. Anything RDKit cannot
    parse, an empty molecule, or an over-long string returns ``None`` so
    callers can treat it as an invalid guess.
    """
    if not isinstance(smiles, str):
        return None
    smiles = smiles.strip()
    if not smiles or len(smiles) > MAX_SMILES_LENGTH:
        return None
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rdkit is required for inchikey14()") from exc

    with _quiet_rdkit():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        try:
            mol = _get_tautomer_enumerator().Canonicalize(mol)
        except Exception as exc:  # RDKit raises native exceptions here.
            logger.debug("Tautomer canonicalisation failed for %s: %s", smiles, exc)
        try:
            key = Chem.MolToInchiKey(mol)
        except Exception as exc:
            logger.debug("InChIKey generation failed for %s: %s", smiles, exc)
            return None
    if not key or len(key) < 14:
        return None
    return key[:14]


def same_structure(smiles_a: Optional[str], smiles_b: Optional[str]) -> bool:
    """True when both SMILES reduce to the same InChIKey14."""
    key_a = inchikey14(smiles_a)
    return key_a is not None and key_a == inchikey14(smiles_b)


# ----------------------------------------------------------------------
# MRR@25
# ----------------------------------------------------------------------


def reciprocal_rank_from_keys(
    truth_key: Optional[str],
    guess_keys: Sequence[Optional[str]],
    k: int = MAX_GUESSES,
) -> float:
    """Reciprocal rank of the first guess whose key equals ``truth_key``.

    Only the first ``k`` guesses count. Invalid guesses (``None``) still
    occupy their position, exactly as the competition scores them.
    """
    if truth_key is None:
        return 0.0
    for position, key in enumerate(guess_keys[:k], start=1):
        if key is not None and key == truth_key:
            return 1.0 / position
    return 0.0


def reciprocal_rank(
    truth_smiles: str,
    guesses: Sequence[str],
    k: int = MAX_GUESSES,
) -> float:
    """Reciprocal rank of ``truth_smiles`` within ``guesses`` (SMILES)."""
    return reciprocal_rank_from_keys(
        inchikey14(truth_smiles), [inchikey14(g) for g in guesses[:k]], k=k
    )


def mrr_at_k(
    truth: Mapping[str, str],
    predictions: Mapping[str, Sequence[str]],
    k: int = MAX_GUESSES,
) -> float:
    """Mean reciprocal rank over every molecule in ``truth``.

    ``truth`` maps ``molecule_id -> answer SMILES``; ``predictions`` maps
    ``molecule_id -> ranked guesses``. Molecules without a prediction
    score zero, as they would on the leaderboard.
    """
    if not truth:
        raise ValueError("truth is empty; nothing to score.")
    total = 0.0
    for molecule_id, answer in truth.items():
        guesses = predictions.get(molecule_id, ())
        total += reciprocal_rank(answer, list(guesses), k=k)
    return total / len(truth)


def per_molecule_reciprocal_ranks(
    truth: Mapping[str, str],
    predictions: Mapping[str, Sequence[str]],
    k: int = MAX_GUESSES,
) -> dict[str, float]:
    """Per-molecule reciprocal ranks, handy for paired comparisons."""
    return {
        molecule_id: reciprocal_rank(answer, list(predictions.get(molecule_id, ())), k=k)
        for molecule_id, answer in truth.items()
    }


# ----------------------------------------------------------------------
# Submission files
# ----------------------------------------------------------------------


def format_guesses(guesses: Sequence[str], k: int = MAX_GUESSES) -> list[str]:
    """Clean a guess list: strip, drop empties and separators, cap at ``k``.

    Does **not** deduplicate by structure; use :func:`dedupe_by_structure`
    first if you want each InChIKey14 to occupy one slot only.
    """
    cleaned: list[str] = []
    for guess in guesses:
        if not isinstance(guess, str):
            continue
        guess = guess.strip().replace(GUESS_SEPARATOR, "")
        if guess:
            cleaned.append(guess)
        if len(cleaned) >= k:
            break
    return cleaned


def dedupe_by_structure(guesses: Sequence[str], k: int = MAX_GUESSES) -> list[str]:
    """Keep the first SMILES per InChIKey14, drop unparseable ones, cap at ``k``.

    A rank occupied by a duplicate of an earlier guess is wasted, so this
    is the last step before writing a submission.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for guess in format_guesses(guesses, k=len(guesses)):
        key = inchikey14(guess)
        if key is None or key in seen:
            continue
        seen.add(key)
        kept.append(guess)
        if len(kept) >= k:
            break
    return kept


def write_submission(
    predictions: Mapping[str, Sequence[str]],
    path: str | Path,
    molecule_ids: Optional[Sequence[str]] = None,
    *,
    fallback_smiles: str = "C",
    dedupe: bool = True,
) -> pd.DataFrame:
    """Write ``submission.csv`` and return the dataframe that was written.

    ``molecule_ids`` is the authoritative list (normally the test set's
    ids). Molecules without usable guesses receive ``fallback_smiles`` so
    the file is never rejected for an empty field. The result is validated
    with :func:`validate_submission` before writing; problems raise.
    """
    ids = list(molecule_ids) if molecule_ids is not None else list(predictions.keys())
    rows = []
    for molecule_id in ids:
        guesses = list(predictions.get(molecule_id, ()))
        guesses = dedupe_by_structure(guesses) if dedupe else format_guesses(guesses)
        if not guesses:
            guesses = [fallback_smiles]
        rows.append({MOLECULE_ID_COLUMN: molecule_id, SMILES_COLUMN: GUESS_SEPARATOR.join(guesses)})
    df = pd.DataFrame(rows, columns=[MOLECULE_ID_COLUMN, SMILES_COLUMN])
    problems = validate_submission(df, molecule_ids=ids)
    if problems:
        raise ValueError("Submission would be rejected: " + "; ".join(problems))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def validate_submission(
    df: pd.DataFrame,
    molecule_ids: Optional[Sequence[str]] = None,
) -> list[str]:
    """Return a list of rejection reasons (empty when the file is valid)."""
    problems: list[str] = []
    for column in (MOLECULE_ID_COLUMN, SMILES_COLUMN):
        if column not in df.columns:
            problems.append(f"missing column {column!r}")
    if problems:
        return problems
    if len(df) == 0:
        problems.append("submission is empty")
        return problems
    if df[MOLECULE_ID_COLUMN].isna().any() or df[SMILES_COLUMN].isna().any():
        problems.append("null values present")
    ids = df[MOLECULE_ID_COLUMN].astype(str)
    duplicated = ids[ids.duplicated()].unique().tolist()
    if duplicated:
        problems.append(f"repeated molecule_id: {duplicated[:5]}")
    smiles = df[SMILES_COLUMN].astype(str)
    if (smiles.str.strip() == "").any():
        problems.append("empty smiles field")
    too_many = (smiles.str.count(GUESS_SEPARATOR) + 1 > MAX_GUESSES).sum()
    if too_many:
        problems.append(f"{int(too_many)} rows exceed {MAX_GUESSES} guesses")
    if molecule_ids is not None:
        expected = set(map(str, molecule_ids))
        present = set(ids)
        missing = expected - present
        extra = present - expected
        if missing:
            problems.append(f"{len(missing)} expected molecule_ids missing")
        if extra:
            problems.append(f"{len(extra)} unexpected molecule_ids present")
    return problems


def read_submission(path: str | Path) -> dict[str, list[str]]:
    """Read a submission file into ``molecule_id -> guesses``."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    problems = validate_submission(df)
    if problems:
        raise ValueError("Invalid submission: " + "; ".join(problems))
    return {
        str(row[MOLECULE_ID_COLUMN]): [
            g for g in str(row[SMILES_COLUMN]).split(GUESS_SEPARATOR) if g
        ]
        for row in df.to_dict("records")
    }


def score_submission(
    submission: pd.DataFrame | str | Path | Mapping[str, Sequence[str]],
    truth: Mapping[str, str],
    k: int = MAX_GUESSES,
) -> float:
    """MRR@k of a submission (dataframe, path or mapping) against ``truth``."""
    if isinstance(submission, (str, Path)):
        predictions = read_submission(submission)
    elif isinstance(submission, pd.DataFrame):
        predictions = {
            str(row[MOLECULE_ID_COLUMN]): str(row[SMILES_COLUMN]).split(GUESS_SEPARATOR)
            for row in submission.to_dict("records")
        }
    else:
        predictions = {str(key): list(value) for key, value in submission.items()}
    return mrr_at_k(truth, predictions, k=k)


def bootstrap_mrr_ci(
    reciprocal_ranks: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval of the mean reciprocal rank."""
    values = np.asarray(list(reciprocal_ranks), dtype=np.float64)
    if values.size == 0:
        raise ValueError("No reciprocal ranks supplied.")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        means[i] = values[rng.integers(0, values.size, values.size)].mean()
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


__all__ = [
    "GUESS_SEPARATOR",
    "MAX_GUESSES",
    "METRIC_RDKIT_VERSION",
    "MOLECULE_ID_COLUMN",
    "SMILES_COLUMN",
    "bootstrap_mrr_ci",
    "dedupe_by_structure",
    "format_guesses",
    "inchikey14",
    "mrr_at_k",
    "per_molecule_reciprocal_ranks",
    "read_submission",
    "reciprocal_rank",
    "reciprocal_rank_from_keys",
    "same_structure",
    "score_submission",
    "validate_submission",
    "write_submission",
]
