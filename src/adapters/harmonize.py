"""Cross-dataset cell type harmonization.

Functions to strip side suffixes from cell type labels, build side columns,
load harmonization mappings, apply them to neuron DataFrames, and produce
coverage reports across datasets.
"""

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Side-suffix patterns
# ---------------------------------------------------------------------------

# Each pattern is (regex, side_normalization_fn)
# The regex must capture the side so we can normalize it.
_SIDE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"_(L|R)$", re.IGNORECASE), lambda m: m.group(1).upper()),
    (re.compile(r"_(left|right)$", re.IGNORECASE), lambda m: "L" if m.group(1).lower() == "left" else "R"),
    (re.compile(r"\((L|R)\)$", re.IGNORECASE), lambda m: m.group(1).upper()),
    (re.compile(r"\((left|right)\)$", re.IGNORECASE), lambda m: "L" if m.group(1).lower() == "left" else "R"),
    (re.compile(r"-(L|R)$", re.IGNORECASE), lambda m: m.group(1).upper()),
]

_SIDE_NORMALIZE: dict[str, str] = {
    "L": "L",
    "R": "R",
    "left": "L",
    "right": "R",
}

# Values in the side column that we consider unresolved (will try to extract
# from the type column).
_UNRESOLVED_SIDE = {None, "", "unknown", "C", "center", "midline", "mid", "U"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def strip_side_suffixes(type_str: str) -> tuple[str, str | None]:
    """Remove a side suffix from *type_str* and return the cleaned type and side.

    Parameters
    ----------
    type_str : str
        Raw cell-type label, e.g. ``"ORN_DA1_L"`` or ``"PN_glomerulus(left)"``.

    Returns
    -------
    tuple[str, str | None]
        ``(type_without_side, detected_side)`` where *detected_side* is
        ``"L"``, ``"R"``, or ``None`` when no side suffix was found.
    """
    if not isinstance(type_str, str) or not type_str.strip():
        return type_str, None

    for pattern, normalizer in _SIDE_PATTERNS:
        m = pattern.search(type_str)
        if m:
            side = normalizer(m)
            cleaned = pattern.sub("", type_str).strip()
            logger.warning(
                "Stripped side suffix from '%s' → '%s' (side=%s)",
                type_str,
                cleaned,
                side,
            )
            return cleaned, side

    return type_str, None


def build_side_column(
    neurons: pd.DataFrame,
    type_col: str = "cell_type_raw",
    side_col: str = "side",
) -> pd.Series:
    """Build a side column by extracting side from cell-type labels where missing.

    For every row whose *side_col* value is null or ``"unknown"`` / ``"C"``,
    attempt to extract a side (``"L"`` / ``"R"``) from *type_col* via
    :func:`strip_side_suffixes`.  Rows that already have ``"L"`` or ``"R"``
    are left unchanged; everything else becomes ``"C"`` (center / midline /
    unknown).

    Parameters
    ----------
    neurons : pd.DataFrame
        DataFrame of neuron metadata.
    type_col : str
        Column that contains raw cell-type labels.  Default ``"cell_type_raw"``.
    side_col : str
        Column that contains (possibly incomplete) side labels.
        Default ``"side"``.

    Returns
    -------
    pd.Series
        A Series with values ``"L"``, ``"R"``, or ``"C"``, aligned to
        *neurons*.
    """
    side = neurons[side_col].copy() if side_col in neurons.columns else pd.Series("C", index=neurons.index)

    # Normalise known values
    side = side.apply(_normalize_side)

    # Identify rows where side is unresolved
    unresolved_mask = side.isin(_UNRESOLVED_SIDE) | side.isna()

    if unresolved_mask.any():
        # Try to extract side from the type column
        extracted = neurons.loc[unresolved_mask, type_col].apply(strip_side_suffixes)
        side.loc[unresolved_mask] = extracted.apply(lambda t: t[1] if t[1] else "C")

    # Everything still null/empty becomes "C"
    side = side.fillna("C").replace({"": "C"})

    resolved_count = (side.isin({"L", "R"}) & unresolved_mask.reindex(side.index, fill_value=False)).sum()
    if resolved_count:
        logger.info(
            "Resolved side for %d / %d neurons from '%s' column.",
            resolved_count,
            len(neurons),
            type_col,
        )

    return side


def load_harmonization_map(csv_path: str) -> dict[tuple[str, str], str]:
    """Load the cell-type harmonization CSV, returning a lookup dict.

    The CSV is expected to have columns::

        source_dataset, source_type, canonical_type

    Parameters
    ----------
    csv_path : str
        Path to the harmonization CSV file.

    Returns
    -------
    dict[tuple[str, str], str]
        Mapping ``(source_dataset, source_type) → canonical_type``.
        If the CSV does not exist a template is created and an empty dict
        is returned.
    """
    path = Path(csv_path)

    if not path.exists():
        _create_template(path)
        logger.warning(
            "Harmonization CSV not found at '%s' – created template. "
            "Please populate it and re-run.",
            path,
        )
        return {}

    df = pd.read_csv(path)

    required_cols = {"source_dataset", "source_type", "canonical_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Harmonization CSV at '{path}' is missing required columns: {missing}"
        )

    mapping: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        key = (str(row["source_dataset"]), str(row["source_type"]))
        mapping[key] = str(row["canonical_type"])

    logger.info("Loaded %d harmonization mappings from '%s'.", len(mapping), path)
    return mapping


def apply_harmonization(
    neurons: pd.DataFrame,
    dataset_tag: str,
    mapping: dict[tuple[str, str], str],
    type_col: str = "cell_type_raw",
) -> pd.DataFrame:
    """Add a ``cell_type_harmonized`` column to *neurons* using *mapping*.

    For each row the lookup key is ``(dataset_tag, value_in_type_col)``.
    When a mapping exists the canonical type is used; otherwise the raw
    type is kept as-is.

    Parameters
    ----------
    neurons : pd.DataFrame
        Neuron metadata.
    dataset_tag : str
        Tag identifying the source dataset (e.g. ``"flywire"``, ``"mcns"``).
    mapping : dict[tuple[str, str], str]
        Mapping from :func:`load_harmonization_map`.
    type_col : str
        Column that holds the raw (per-dataset) cell type.  Default
        ``"cell_type_raw"``.

    Returns
    -------
    pd.DataFrame
        *neurons* with the new ``cell_type_harmonized`` column appended
        (a copy is **not** made; the input is mutated).
    """
    raw_types = neurons[type_col].astype(str)

    harmonized = raw_types.map(
        lambda ct: mapping.get((dataset_tag, ct), ct)
    )

    neurons["cell_type_harmonized"] = harmonized

    mapped_count = (harmonized != raw_types).sum()
    logger.info(
        "Harmonized %d / %d neurons for dataset '%s'.",
        mapped_count,
        len(neurons),
        dataset_tag,
    )

    return neurons


def coverage_report(
    neurons_by_dataset: dict[str, pd.DataFrame],
    type_col: str = "cell_type_harmonized",
) -> pd.DataFrame:
    """Produce a coverage table showing how many datasets each harmonized type
    appears in, with per-dataset neuron counts.

    Parameters
    ----------
    neurons_by_dataset : dict[str, pd.DataFrame]
        Mapping ``{dataset_tag: neurons_df}``.  Each DataFrame must contain
        the *type_col* column (default ``"cell_type_harmonized"``).
    type_col : str
        Column containing the harmonized cell type.

    Returns
    -------
    pd.DataFrame
        Columns: *cell_type_harmonized*, *n_datasets_present*, and one
        ``{tag}_count`` column per dataset.  Sorted by *n_datasets_present*
        descending.
    """
    records: list[dict] = []
    all_types: set[str] = set()

    # Count neurons per type per dataset
    counts: dict[str, pd.Series] = {}
    for tag, df in neurons_by_dataset.items():
        if type_col not in df.columns:
            raise KeyError(
                f"DataFrame for dataset '{tag}' is missing column '{type_col}'"
            )
        cnt = df[type_col].value_counts()
        counts[tag] = cnt
        all_types.update(cnt.index)

    dataset_tags = sorted(neurons_by_dataset.keys())

    for ct in sorted(all_types):
        row: dict = {"cell_type_harmonized": ct, "n_datasets_present": 0}
        for tag in dataset_tags:
            col_name = f"{tag}_count"
            val = counts[tag].get(ct, 0)
            row[col_name] = val
            if val > 0:
                row["n_datasets_present"] += 1
        records.append(row)

    report = pd.DataFrame(records)
    report = report.sort_values("n_datasets_present", ascending=False, ignore_index=True)

    # ---- summary ----
    n_multi = (report["n_datasets_present"] >= 2).sum()
    total_neurons = sum(
        df[type_col].notna().sum() for df in neurons_by_dataset.values()
    )
    logger.info(
        "Coverage report: %d harmonized types, %d present in >= 2 datasets, "
        "%d total neurons covered.",
        len(report),
        n_multi,
        total_neurons,
    )

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_side(value) -> str:
    """Normalise a side value to 'L', 'R', or the original value."""
    if isinstance(value, str):
        return _SIDE_NORMALIZE.get(value.lower(), value)
    return value


def _create_template(path: Path) -> None:
    """Write a template harmonization CSV with the expected header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    template = "source_dataset,source_type,canonical_type\n"
    path.write_text(template)
