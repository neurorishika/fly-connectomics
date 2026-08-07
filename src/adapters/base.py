"""
Abstract base class for connectome dataset adapters.

Each adapter wraps one dataset (FlyWire, MCNS, BANC, …) and exposes
harmonised :func:`load` → (neurons, edges) DataFrames, plus validation
and QC helpers.

Importable symbols
------------------
- :class:`ConnectomeAdapter` – ABC that all adapters must implement.
- :func:`normalize_side` – canonicalise side labels ('L', 'R', 'C', None).
- :func:`glom_from_type` – extract glomerulus name from ORN type strings.
- :func:`is_orn_type` – test whether a cell-type string looks like an ORN.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

_SIDE_MAP: Dict[str, Optional[str]] = {
    "l": "L",
    "left": "L",
    "r": "R",
    "right": "R",
    "m": "C",
    "midline": "C",
    "c": "C",
    "central": "C",
}


def normalize_side(val: Any) -> Optional[str]:
    """Map common side representations to canonical ``'L'`` / ``'R'`` / ``'C'``.

    Parameters
    ----------
    val : Any
        A string or something convertible via ``str()``.

    Returns
    -------
    str or None
        One of ``'L'``, ``'R'``, ``'C'``, or ``None`` if the value is
        not recognised, missing, or null.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    key = str(val).strip().lower()
    return _SIDE_MAP.get(key, None)


# Pattern:  ORN_<glomerulus>  or  ORN_<glomerulus>_<suffix>
_ORN_PATTERN: re.Pattern[str] = re.compile(r"^ORN_([A-Za-z0-9]+)(?:_\S+)?$")


def is_orn_type(type_str: Any) -> bool:
    """Return ``True`` if *type_str* matches the ORN naming convention.

    Recognised forms: ``ORN_DA1``, ``ORN_VA1d``, ``ORN_DC3_something``.
    """
    if not isinstance(type_str, str):
        return False
    return _ORN_PATTERN.match(type_str) is not None


def glom_from_type(type_str: Any) -> Optional[str]:
    """Extract the glomerulus name from an ORN type string.

    Parameters
    ----------
    type_str : Any
        A cell-type string (e.g. ``'ORN_DA1'``, ``'ORN_VA1d_lPN'``).

    Returns
    -------
    str or None
        The glomerulus portion (``'DA1'``) or ``None`` when the string
        does not look like an ORN type.
    """
    if not isinstance(type_str, str):
        return None
    m = _ORN_PATTERN.match(type_str)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# ConnectomeDataset — container for harmonised output
# ---------------------------------------------------------------------------

@dataclass
class ConnectomeDataset:
    """Container returned by every adapter's :meth:`load`.

    Attributes
    ----------
    neurons : pd.DataFrame
        Indexed by ``neuron_id``, columns per the adapter schema.
    edges : pd.DataFrame
        Columns: ``pre_id``, ``post_id``, ``syn_count``.
    dataset_name : str
        Human-readable name.
    dataset_tag : str
        Short tag (``'flywire'``, ``'mcns'``, ``'banc'``).
    """

    neurons: pd.DataFrame
    edges: pd.DataFrame
    dataset_name: str
    dataset_tag: str


# ---------------------------------------------------------------------------
# extract_side_from_type — resolve side from cell type strings
# ---------------------------------------------------------------------------

# Patterns that indicate a side suffix in a cell type string.
# Ordered by priority: more-specific patterns first.
_SIDE_SUFFIX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"_(L|R)$", re.IGNORECASE), lambda m: m.group(1).upper()),
    (re.compile(r"_\((L|R)\)$", re.IGNORECASE), lambda m: m.group(1).upper()),
    (re.compile(r"\((L|R)\)$", re.IGNORECASE), lambda m: m.group(1).upper()),
    (re.compile(r"-(L|R)$", re.IGNORECASE), lambda m: m.group(1).upper()),
]


def extract_side_from_type(type_str: Any) -> Optional[str]:
    """Try to extract a side label (``'L'`` / ``'R'``) from a cell-type string.

    Returns ``None`` if no side suffix is recognised.
    """
    if not isinstance(type_str, str):
        return None
    s = type_str.strip()
    for pat, fn in _SIDE_SUFFIX_PATTERNS:
        m = pat.search(s)
        if m:
            return fn(m)
    return None


def strip_side_suffix(type_str: Any) -> str:
    """Remove known side suffixes from a cell-type string.

    Returns the cleaned string.  Logs a debug message when a suffix is stripped.
    """
    if not isinstance(type_str, str):
        return ""
    s = type_str.strip()
    original = s
    for pat, _fn in _SIDE_SUFFIX_PATTERNS:
        s, n = pat.subn("", s)
        if n:
            logger.debug(f"Stripped side suffix from '{original}' → '{s.strip()}'")
    result = s.strip()
    if result != original:
        logger.info(f"Side suffix stripped: '{original}' → '{result}'")
    return result


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------

class ConnectomeAdapter(ABC):
    """Abstract base class for a connectome-dataset adapter.

    Subclasses must implement :meth:`dataset_name`, :meth:`dataset_tag`,
    and :meth:`load`.

    Parameters
    ----------
    data_dir : str
        Directory where raw / preprocessed dataset files are stored.
    config : Config
        The project-wide configuration dataclass.
    """

    # Supported dataset tag values (informational; not enforced).
    _VALID_TAGS: Tuple[str, ...] = ("flywire", "mcns", "banc")

    def __init__(self, data_dir: str, config: Config) -> None:
        self.data_dir: str = data_dir
        self.config: Config = config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Human-readable dataset name (e.g. ``'FlyWire'``, ``'BANC'``)."""
        ...

    @property
    @abstractmethod
    def dataset_tag(self) -> str:
        """Short tag — one of ``'flywire'``, ``'mcns'``, ``'banc'``."""
        ...

    @abstractmethod
    def load(self) -> ConnectomeDataset:
        """Load and return a :class:`ConnectomeDataset` with harmonised
        (neurons, edges) DataFrames.

        Returns
        -------
        ConnectomeDataset
            Container with ``neurons`` (indexed by ``neuron_id``, columns:
            ``neuron_id``, ``dataset``, ``side``, ``cell_type``,
            ``cell_type_raw``, ``super_class``, ``class``, ``nt_type``,
            ``region``) and ``edges`` (columns: ``pre_id``, ``post_id``,
            ``syn_count``).
        """
        ...

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    # Required columns for each dataframe (neuron_id may be the index)
    _NEURON_COLS: Tuple[str, ...] = (
        "dataset", "side", "cell_type",
        "cell_type_raw", "super_class", "class", "nt_type", "region",
    )
    _NEURON_ID_NAME: str = "neuron_id"
    _EDGE_COLS: Tuple[str, ...] = (
        "pre_id", "post_id", "syn_count",
    )

    # Allowed side values (string category)
    _VALID_SIDES: Tuple[str, ...] = ("L", "R", "C")

    def validate_output(
        self, neurons: pd.DataFrame, edges: pd.DataFrame
    ) -> None:
        """Run assertions on the shape, dtypes, values, and integrity
        of the harmonised output.

        Raises
        ------
        AssertionError
            If any validation check fails.
        """
        # ---- shape ----
        assert isinstance(neurons, pd.DataFrame), "neurons must be a DataFrame"
        assert isinstance(edges, pd.DataFrame), "edges must be a DataFrame"
        assert neurons.shape[0] > 0, "neurons must have at least 1 row"
        assert edges.shape[0] > 0, "edges must have at least 1 row"

        # ---- columns present ----
        # neuron_id may be the index; check index name
        assert neurons.index.name == self._NEURON_ID_NAME, (
            f"neurons index must be named '{self._NEURON_ID_NAME}', got '{neurons.index.name}'"
        )
        missing_n = set(self._NEURON_COLS) - set(neurons.columns)
        assert not missing_n, f"neurons missing columns: {missing_n}"
        missing_e = set(self._EDGE_COLS) - set(edges.columns)
        assert not missing_e, f"edges missing columns: {missing_e}"

        # ---- dtypes ----
        # neuron_id is the index; check its dtype
        neuron_id_vals = neurons.index
        assert pd.api.types.is_integer_dtype(neuron_id_vals) or pd.api.types.is_object_dtype(
            neuron_id_vals
        ), "neuron_id (index) must be int64 or object (str)"
        assert str(neurons["dataset"].dtype) == "category", "dataset must be category"
        assert str(neurons["side"].dtype) == "category", "side must be category"
        assert pd.api.types.is_string_dtype(
            neurons["cell_type"]
        ), "cell_type must be string"
        assert pd.api.types.is_string_dtype(
            neurons["cell_type_raw"]
        ), "cell_type_raw must be string"
        assert str(neurons["super_class"].dtype) == "category", "super_class must be category"
        assert pd.api.types.is_string_dtype(neurons["class"]), "class must be string"
        # nt_type may be category (nullable)
        assert str(neurons["nt_type"].dtype) == "category", "nt_type must be category"
        assert pd.api.types.is_string_dtype(neurons["region"]), "region must be string"

        # edges: pre_id dtype must match neuron_id (index) dtype
        assert edges["pre_id"].dtype == neuron_id_vals.dtype, (
            f"pre_id dtype {edges['pre_id'].dtype} != neuron_id dtype "
            f"{neuron_id_vals.dtype}"
        )
        assert edges["post_id"].dtype == neuron_id_vals.dtype, (
            f"post_id dtype {edges['post_id'].dtype} != neuron_id dtype "
            f"{neuron_id_vals.dtype}"
        )
        assert pd.api.types.is_integer_dtype(
            edges["syn_count"]
        ), "syn_count must be integer"

        # syn_count bounds
        assert (edges["syn_count"] >= 1).all(), "syn_count must be >= 1"
        assert edges["syn_count"].dtype == np.int32, "syn_count must be int32"

        # ---- side values ----
        side_vals = set(neurons["side"].cat.categories.tolist())
        invalid = side_vals - set(self._VALID_SIDES)
        assert not invalid, f"unexpected side values: {invalid}"

        # ---- nulls ----
        assert neurons.index.notna().all(), "neuron_id must not be null"
        assert neurons["side"].notna().all(), "side must not be null"
        # nt_type is allowed to have nulls (nullable category)

        # ---- referential integrity ----
        neuron_ids = set(neurons.index)
        pre_ids = set(edges["pre_id"])
        post_ids = set(edges["post_id"])
        assert pre_ids.issubset(neuron_ids), "some pre_id not found in neurons"
        assert post_ids.issubset(neuron_ids), "some post_id not found in neurons"

    # ------------------------------------------------------------------
    # QC report
    # ------------------------------------------------------------------

    def qc_report(
        self, neurons: pd.DataFrame, edges: pd.DataFrame
    ) -> Dict[str, object]:
        """Return a dictionary of summary counts and statistics.

        Parameters
        ----------
        neurons : pd.DataFrame
            Validated neurons dataframe.
        edges : pd.DataFrame
            Validated edges dataframe.

        Returns
        -------
        dict
            Keys include ``total_neurons``, ``total_edges``, ``n_L``,
            ``n_R``, ``n_C``, ``n_sensory``, ``n_olfactory``,
            ``n_orn_glomeruli``, ``mean_syn_per_edge``, and more.
        """
        side_counts = neurons["side"].value_counts()
        n_L: int = int(side_counts.get("L", 0))
        n_R: int = int(side_counts.get("R", 0))
        n_C: int = int(side_counts.get("C", 0))

        # Sensory / olfactory counts (off super_class)
        n_sensory: int = int((neurons["super_class"] == "sensory").sum())
        n_olfactory: int = int(
            neurons["class"].str.lower().str.contains("olfactory", na=False).sum()
        )

        # ORN glomeruli
        orn_mask: pd.Series[bool] = neurons["cell_type"].apply(is_orn_type)
        n_orn: int = int(orn_mask.sum())
        orn_glom_set: set = set(
            neurons.loc[orn_mask, "cell_type"].map(glom_from_type).dropna()
        )
        n_orn_glomeruli: int = len(orn_glom_set)

        # Mean syn per edge
        mean_syn: float = float(edges["syn_count"].mean())

        return {
            "total_neurons": neurons.shape[0],
            "total_edges": edges.shape[0],
            "n_L": n_L,
            "n_R": n_R,
            "n_C": n_C,
            "n_sensory": n_sensory,
            "n_olfactory": n_olfactory,
            "n_orn": n_orn,
            "n_orn_glomeruli": n_orn_glomeruli,
            "mean_syn_per_edge": round(mean_syn, 4),
        }
