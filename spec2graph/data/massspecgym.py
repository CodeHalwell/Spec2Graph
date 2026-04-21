"""MassSpecGym download, schema validation and filtering.

This module is the single entry point for fetching the MassSpecGym
benchmark (`roman-bushuiev/MassSpecGym` on the Hugging Face Hub) and
converting it into a :class:`pandas.DataFrame` suitable for downstream
training and evaluation.

We deliberately do **not** import from :mod:`spectral_diffusion` here — the
loader returns raw dataframes and leaves eigenvector computation to
:mod:`spec2graph.data.cache`.

Example
-------
>>> df = load_massspecgym_tsv()                     # doctest: +SKIP
>>> filtered, stats = filter_massspecgym(df)        # doctest: +SKIP
>>> print(stats)                                    # doctest: +SKIP
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MASSSPECGYM_REPO_ID = "roman-bushuiev/MassSpecGym"
MASSSPECGYM_TSV_PATH = "data/MassSpecGym.tsv"

# Columns that must be present before the loader returns. If any of these
# are missing the schema has shifted and downstream code cannot proceed.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "mzs",
    "intensities",
    "smiles",
    "inchikey",
    "formula",
    "precursor_mz",
    "adduct",
    "instrument_type",
)

# Candidate names for the train/val/test split column. MassSpecGym has
# historically used "fold"; we look for common alternatives as a safety net.
_SPLIT_COLUMN_CANDIDATES: tuple[str, ...] = ("fold", "split", "subset")
_SPLIT_VALUES = {"train", "val", "test"}


def _hf_hub_download(cache_dir: Optional[str] = None) -> str:
    """Download the MassSpecGym TSV and return its local path.

    The import of :mod:`huggingface_hub` is deferred so tests that never
    call this function can run without the dependency installed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download MassSpecGym. "
            "Install it via `pip install huggingface-hub`."
        ) from exc

    return hf_hub_download(
        repo_id=MASSSPECGYM_REPO_ID,
        filename=MASSSPECGYM_TSV_PATH,
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def _detect_split_column(df: pd.DataFrame) -> str:
    """Return the name of the column holding train/val/test labels.

    Preference order: the canonical name ``fold``, then any other column
    containing all three expected values. Raises :class:`RuntimeError` if
    no suitable column is found.
    """
    for candidate in _SPLIT_COLUMN_CANDIDATES:
        if candidate in df.columns:
            values = set(df[candidate].dropna().unique())
            if _SPLIT_VALUES.issubset(values):
                if candidate != "fold":
                    logger.warning(
                        "Split column is named %r rather than the expected 'fold'. "
                        "Using it anyway.",
                        candidate,
                    )
                return candidate

    # Last resort: scan plausibly split-like columns. A split column is
    # always a low-cardinality object/string column, so we restrict the
    # scan to those and only compute the unique-set when nunique is small
    # — this avoids hashing 200k-row text columns like ``mzs`` or
    # ``smiles`` on every load.
    #
    # ``_MAX_SPLIT_CARDINALITY`` is the largest cardinality worth
    # considering; MassSpecGym uses 3 (train/val/test) but we leave head
    # room in case future releases add an extra fold.
    _MAX_SPLIT_CARDINALITY = 8
    # Include "str" explicitly for forward-compatibility with pandas 3,
    # which no longer treats string dtypes as a subset of "object".
    object_columns = df.select_dtypes(
        include=["object", "string", "str", "category"]
    ).columns
    for column in object_columns:
        if df[column].nunique(dropna=True) > _MAX_SPLIT_CARDINALITY:
            continue
        values = set(df[column].dropna().astype(str).unique())
        if _SPLIT_VALUES.issubset(values):
            logger.warning(
                "Split column auto-detected as %r; expected one of %s.",
                column,
                _SPLIT_COLUMN_CANDIDATES,
            )
            return column

    raise RuntimeError(
        "Could not find a split column in the MassSpecGym dataframe. "
        f"Expected one of {_SPLIT_COLUMN_CANDIDATES} or any column with the "
        f"values {_SPLIT_VALUES}. Available columns: {list(df.columns)}."
    )


def load_massspecgym_tsv(
    cache_dir: Optional[str] = None,
    local_path: Optional[str | os.PathLike] = None,
) -> pd.DataFrame:
    """Download MassSpecGym (if necessary) and return it as a dataframe.

    Parameters
    ----------
    cache_dir:
        Directory to pass to :func:`huggingface_hub.hf_hub_download`. When
        ``None`` the default Hugging Face cache is used.
    local_path:
        If provided, read from this path instead of downloading. Useful for
        tests that pre-write a small subset to a temporary directory.

    Returns
    -------
    pd.DataFrame
        The full dataset. Includes a ``split`` column (potentially renamed
        from ``fold`` or similar) so downstream code can filter by split
        without re-detecting the column name.

    Raises
    ------
    RuntimeError
        If any of :data:`REQUIRED_COLUMNS` or a split column is missing.
    """
    if local_path is not None:
        path = Path(local_path)
    else:
        path = Path(_hf_hub_download(cache_dir=cache_dir))

    logger.info("Reading MassSpecGym TSV from %s", path)
    df = pd.read_csv(path, sep="\t")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            "MassSpecGym schema mismatch. Missing required columns: "
            f"{missing}. Available columns: {list(df.columns)}."
        )

    split_column = _detect_split_column(df)
    if split_column != "split":
        # Copy (do not rename) so the original column remains available —
        # downstream callers may want to inspect the source-of-truth
        # split labels in addition to the canonical "split" alias.
        df = df.copy()
        df["split"] = df[split_column]

    return df


def parse_peak_list(value: object) -> np.ndarray:
    """Parse a serialised peak list from MassSpecGym into a numpy array.

    MassSpecGym stores ``mzs`` and ``intensities`` as stringified Python
    lists (``"[1.0, 2.0, 3.0]"``). We use :func:`ast.literal_eval` — never
    :func:`eval` — to keep the parse safe.

    Accepts existing list/tuple/ndarray inputs for convenience (some loader
    paths may deserialise eagerly) and always returns a 1-D ``float32``
    array.
    """
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float32).ravel()
    if isinstance(value, (list, tuple)):
        return np.asarray(list(value), dtype=np.float32)
    if not isinstance(value, str):
        raise TypeError(
            f"Cannot parse peak list of type {type(value).__name__}; "
            "expected str, list, tuple or ndarray."
        )
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(
            f"Parsed peak list is not a sequence; got {type(parsed).__name__}."
        )
    return np.asarray(parsed, dtype=np.float32)


@dataclass
class FilterStats:
    """Counts of rows dropped by :func:`filter_massspecgym`, for logging."""

    initial: int
    dropped_adduct: int = 0
    dropped_precursor_mz: int = 0
    dropped_min_peaks: int = 0
    dropped_smiles_length: int = 0
    dropped_nan_required: int = 0
    final: int = 0
    per_filter: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.per_filter = {
            "adduct": self.dropped_adduct,
            "precursor_mz": self.dropped_precursor_mz,
            "min_peaks": self.dropped_min_peaks,
            "smiles_length": self.dropped_smiles_length,
            "nan_required": self.dropped_nan_required,
        }

    def __str__(self) -> str:
        parts = [f"{self.initial} initial"]
        for name, count in self.per_filter.items():
            if count:
                parts.append(f"-{count} {name}")
        parts.append(f"={self.final}")
        return " ".join(parts)


def _count_peaks(value: object) -> int:
    """Return the number of peaks in a serialised peak list, or 0 on failure."""
    try:
        return len(parse_peak_list(value))
    except (ValueError, SyntaxError, TypeError):
        return 0


def filter_massspecgym(
    df: pd.DataFrame,
    adducts: Sequence[str] = ("[M+H]+",),
    max_precursor_mz: float = 1000.0,
    min_peaks: int = 5,
    max_smiles_length: int = 200,
) -> tuple[pd.DataFrame, FilterStats]:
    """Apply Phase 0 MVP filters to a MassSpecGym dataframe.

    Each filter is applied in order and the row count after each step is
    logged. Returns a new dataframe (no in-place mutation) and a
    :class:`FilterStats` summarising drops per filter.

    Parameters
    ----------
    df:
        The output of :func:`load_massspecgym_tsv`.
    adducts:
        Keep only rows whose ``adduct`` is in this sequence. Default is
        ``("[M+H]+",)``, matching Spec2Graph's Phase 0 scope.
    max_precursor_mz:
        Drop rows whose precursor m/z exceeds this value.
    min_peaks:
        Drop spectra with fewer than this many peaks. Applied after any
        peak parsing so malformed rows are also dropped.
    max_smiles_length:
        Drop rows whose SMILES string exceeds this length. The hard cap
        in :mod:`spectral_diffusion` is ``MAX_SMILES_LENGTH = 2000``; the
        default 200 keeps training tractable.
    """
    initial = len(df)
    current = df.copy()

    # Drop rows that are missing any required value. This protects the
    # filters below from edge-case NaNs; adduct/instrument NaN would
    # otherwise short-circuit the ``isin`` check. ``intensities`` is in
    # the list so that spectra without intensity data can't slip through
    # and produce NaNs downstream.
    required_for_filtering = (
        "adduct",
        "precursor_mz",
        "smiles",
        "mzs",
        "intensities",
    )
    before = len(current)
    current = current.dropna(subset=list(required_for_filtering))
    dropped_nan = before - len(current)

    # 1. Adduct filter
    adduct_set = set(adducts)
    before = len(current)
    current = current[current["adduct"].isin(adduct_set)]
    dropped_adduct = before - len(current)

    # 2. Precursor m/z filter. Use pd.to_numeric(errors='coerce') so a
    # stray non-numeric value (e.g. "NA" as a string) becomes NaN and
    # gets dropped by the comparison, rather than raising ValueError.
    before = len(current)
    precursor_numeric = pd.to_numeric(current["precursor_mz"], errors="coerce")
    current = current[precursor_numeric <= float(max_precursor_mz)]
    dropped_precursor = before - len(current)

    # 3. SMILES length filter
    before = len(current)
    current = current[current["smiles"].str.len() <= max_smiles_length]
    dropped_smiles = before - len(current)

    # 4. Minimum peak count filter (parses the stringified list)
    before = len(current)
    peak_counts = current["mzs"].apply(_count_peaks)
    current = current[peak_counts >= min_peaks]
    dropped_min_peaks = before - len(current)

    stats = FilterStats(
        initial=initial,
        dropped_nan_required=dropped_nan,
        dropped_adduct=dropped_adduct,
        dropped_precursor_mz=dropped_precursor,
        dropped_smiles_length=dropped_smiles,
        dropped_min_peaks=dropped_min_peaks,
        final=len(current),
    )

    logger.info("filter_massspecgym: %s", stats)
    return current.reset_index(drop=True), stats
