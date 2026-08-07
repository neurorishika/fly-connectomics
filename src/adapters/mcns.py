"""Male CNS (neuPrint) connectome adapter.

Loads neuronal metadata and synapse-level connectivity from cached
neuPrint exports under ``data/Male_CNS/`` and exposes them as a
standardised :class:`ConnectomeDataset`.

**Does not** use the ``neuprint-python`` client dynamically — all data
must have been pre-cached by the companion notebook.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .base import (
    ConnectomeAdapter,
    ConnectomeDataset,
    extract_side_from_type,
    glom_from_type,
    is_orn_type,
    normalize_side,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# module-level constants
# ═══════════════════════════════════════════════════════════════════════════

# predictedNt (long form) → canonical abbreviation (nullable)
_NT_MAP: dict[str, Optional[str]] = {
    "gaba": "GABA",
    "acetylcholine": "ACh",
    "glutamate": "Glu",
    "dopamine": "DA",
    "histamine": "HA",
    "serotonin": "5-HT",
    "octopamine": "OA",
    "unclear": None,
}

# Class column value → simplified super_class
_CLASS_TO_SUPER: dict[str, str] = {
    # sensory
    "olfactory": "sensory",
    "ORN": "sensory",
    "visual": "sensory",
    "mechanosensory": "sensory",
    "mechanosensory_tactile": "sensory",
    "mechanosensory_proprioceptive": "sensory",
    "mechanosensory_tbc": "sensory",
    "gustatory": "sensory",
    "hygrosensory": "sensory",
    "chemosensory": "sensory",
    "thermosensory": "sensory",
    "unknown_sensory": "sensory",
    # central
    "ALPN": "central",
    "ALLN": "central",
    "ALIN": "central",
    "ALON": "central",
    "MBON": "central",
    "DAN": "central",
    "Kenyon_Cell": "central",
    "CX": "central",
    "ol_bilateral": "central",
    "SEZPN": "central",
}

# Class values that we re-label as 'olfactory'
_OLFACTORY_CLASSES: set[str] = {
    "olfactory",
    "ORN",
    "ALPN",
    "ALLN",
}

# Side-suffix patterns used to strip side from flywireType.
# Applied in priority order.
_SIDE_STRIP_PATTERNS: list[re.Pattern] = [
    re.compile(r"_(L|R)$"),
    re.compile(r"_\((L|R)\)$"),
    re.compile(r"\((L|R)\)$"),
    re.compile(r"-(L|R)$"),
]


# ═══════════════════════════════════════════════════════════════════════════
# helpers (module-private)
# ═══════════════════════════════════════════════════════════════════════════

def _map_nt(raw: object) -> Optional[str]:
    """Map a ``predictedNt`` value to a canonical abbreviation."""
    if not isinstance(raw, str):
        return None
    return _NT_MAP.get(raw.strip().lower(), None)


def _derive_super_class(class_str: object) -> Optional[str]:
    """Map a raw ``class`` value to a simplified super-class."""
    if not isinstance(class_str, str):
        return "central"  # default for unclassified
    c = class_str.strip()
    # Exact match first
    if c in _CLASS_TO_SUPER:
        return _CLASS_TO_SUPER[c]
    # Substring / heuristic fallbacks — be inclusive
    c_lower = c.lower()
    if any(tok in c_lower for tok in ("motor", "mn", "efferent")):
        return "motor"
    if any(tok in c_lower for tok in ("sensory", "receptor", "gustatory",
                                        "mechano", "hygro", "chemo", "thermo",
                                        "visual", "photo", "olfact")):
        return "sensory"
    if any(tok in c_lower for tok in ("ascending", "descending")):
        return "central"
    # Default
    return "central"


def _derive_class(class_str: object) -> Optional[str]:
    """Derive the adapter-level ``class`` field.

    Returns ``'olfactory'`` for ORN / ALPN / ALLN types; otherwise the
    raw class string (or ``None``).
    """
    if not isinstance(class_str, str):
        return None
    c = class_str.strip()
    if c in _OLFACTORY_CLASSES:
        return "olfactory"
    return c


def _strip_side_from_type(type_str: object) -> str:
    """Remove known side suffixes from a flywireType string."""
    if not isinstance(type_str, str):
        return ""
    s = type_str.strip()
    for pat in _SIDE_STRIP_PATTERNS:
        s = pat.sub("", s).strip()
    return s


# ═══════════════════════════════════════════════════════════════════════════
# adapter
# ═══════════════════════════════════════════════════════════════════════════

class MCNSAdapter(ConnectomeAdapter):
    """Adapter for the **Male CNS** (neuPrint) dataset.

    Expected cached files under ``{data_dir}/Male_CNS/``:

    * ``neurons.pkl`` — pickle of ``(metadata_df, roi_df)``
    * ``connectivity_matrix.npz`` — CSR sparse matrix (primary)
    * ``connectome-weights-male-cns-v1.0-minconf-0.5.feather`` — fallback #1
    * ``orn_to_alpnln_AL_roi_adj.feather`` — fallback #2 (partial)
    * ``n_bodyIds_bodyId_idx_maps.pkl`` — bodyId ↔ index mapping for the
      full neuPrint ID space (needed to resolve CSR indices).
    """

    dataset_name = "male-cns"
    dataset_tag = "mcns"

    # ── load ──────────────────────────────────────────────────────────────

    def load(self) -> ConnectomeDataset:
        base = Path(self.data_dir) / "Male_CNS"

        # ------------------------------------------------------------------
        # 1. neurons
        # ------------------------------------------------------------------
        print(f"[{self.dataset_name}] Loading neurons from {base / 'neurons.pkl'} ...")
        with open(base / "neurons.pkl", "rb") as f:
            neurons_tuple = pickle.load(f)

        raw: pd.DataFrame = neurons_tuple[0].copy()
        n_total = len(raw)
        print(f"[{self.dataset_name}] Loaded {n_total:,} neuron rows from pickle")

        # --- neuron_id / id ------------------------------------------------
        raw["id"] = raw["bodyId"].astype("int64")

        # --- side ----------------------------------------------------------
        side_root = raw["rootSide"].apply(normalize_side)
        side_soma = raw["somaSide"].apply(normalize_side)
        raw["side"] = side_root.fillna(side_soma).fillna("C")
        # Also extract side from flywireType to fill remaining gaps
        side_from_type = raw["flywireType"].apply(extract_side_from_type)
        still_missing = raw["side"].isin([None, "C", ""]) | raw["side"].isna()
        if still_missing.any():
            raw.loc[still_missing, "side"] = side_from_type[still_missing].fillna("C")

        n_dropped = (raw["side"].isna() | (raw["side"] == "")).sum()
        if n_dropped > 0:
            print(
                f"[{self.dataset_name}] Dropping {n_dropped} neurons with "
                f"undetermined side (out of {n_total:,})"
            )
            raw = raw[raw["side"].notna() & (raw["side"] != "")].copy()

        side_counts = raw["side"].value_counts().to_dict()
        print(f"[{self.dataset_name}] Side distribution: {side_counts}")

        # --- cell_type_raw -------------------------------------------------
        raw["cell_type_raw"] = raw["flywireType"].fillna("").astype(str)

        # --- cell_type (side-stripped) -------------------------------------
        raw["cell_type"] = raw["cell_type_raw"].apply(_strip_side_from_type)
        n_stripped = (raw["cell_type"] != raw["cell_type_raw"]).sum()
        print(
            f"[{self.dataset_name}] Stripped side suffix from "
            f"{n_stripped:,} / {len(raw):,} cell types"
        )

        # --- super_class ---------------------------------------------------
        raw["super_class"] = raw["class"].apply(_derive_super_class)
        super_counts = raw["super_class"].value_counts().to_dict()
        print(f"[{self.dataset_name}] Super-class distribution: {super_counts}")

        # --- class ---------------------------------------------------------
        raw["class"] = raw["class"].apply(_derive_class)
        class_counts = raw["class"].value_counts().to_dict()
        print(f"[{self.dataset_name}] Class distribution (top 10): "
              f"{dict(sorted(class_counts.items(), key=lambda x: -x[1])[:10])}")

        # --- nt_type -------------------------------------------------------
        raw["nt_type"] = raw["predictedNt"].apply(_map_nt)
        nt_counts = raw["nt_type"].value_counts(dropna=False).to_dict()
        print(f"[{self.dataset_name}] NT type distribution: {nt_counts}")

        # --- region --------------------------------------------------------
        raw["region"] = "brain"

        # --- build final neuron table --------------------------------------
        neurons = raw[[
            "id",
            "side",
            "cell_type_raw",
            "cell_type",
            "super_class",
            "class",
            "nt_type",
            "region",
        ]].copy()

        neurons["dataset"] = self.dataset_tag
        neurons["dataset"] = neurons["dataset"].astype("category")
        neurons["side"] = neurons["side"].astype("category")
        neurons["super_class"] = neurons["super_class"].astype("category")
        neurons["nt_type"] = neurons["nt_type"].astype("category")
        neurons["neuron_id"] = neurons["id"]
        neurons = neurons.set_index("neuron_id", drop=False)

        print(f"[{self.dataset_name}] Final neuron table: {len(neurons):,} rows "
              f"x {len(neurons.columns)} columns")

        # ------------------------------------------------------------------
        # 2. edges
        # ------------------------------------------------------------------
        edges = self._load_edges(base, neurons)

        # ------------------------------------------------------------------
        # 3. QC
        # ------------------------------------------------------------------
        self._run_qc(neurons, edges)

        return ConnectomeDataset(
            neurons=neurons,
            edges=edges,
            dataset_name=self.dataset_name,
            dataset_tag=self.dataset_tag,
        )

    # ── edge loading ──────────────────────────────────────────────────────

    def _load_edges(self, base: Path, neurons: pd.DataFrame) -> pd.DataFrame:
        """Load edges from the best available source.

        Tries, in order:
        1. ``connectome-weights-male-cns-v1.0-minconf-0.5.feather`` (long-format, fast)
        2. ``connectivity_matrix.npz`` (CSR; very slow for full-space, only as fallback)
        3. ``orn_to_alpnln_AL_roi_adj.feather`` (ORN → ALPN/LN only; warning)
        """
        neuron_ids = set(neurons["id"].values)

        # -- attempt 1: long-format feather (fastest) ------------------------
        edges = self._try_feather_edges(base, neuron_ids)
        if edges is not None:
            return edges

        # -- attempt 2: CSR matrix (slow fallback) ---------------------------
        edges = self._try_csr_edges(base, neuron_ids, neurons)
        if edges is not None:
            return edges

        # -- attempt 3: ORN → ALPN/LN partial -------------------------------
        edges = self._try_orn_partial_edges(base, neuron_ids)
        if edges is not None:
            return edges

        # -- absolute fallback: empty edge list -----------------------------
        print(
            f"[{self.dataset_name}] WARNING: No edge source could be loaded. "
            f"Returning empty edge list."
        )
        return pd.DataFrame(columns=["pre_id", "post_id", "syn_count"])

    # ── CSR path ──────────────────────────────────────────────────────────

    @staticmethod
    def _try_csr_edges(
        base: Path, neuron_ids: set[int], neurons: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """Attempt to extract edges from the sparse CSR matrix.

        The CSR may be either:
        - **neuron-order-indexed**: shape == (N, N) where N = len(neurons);
          row/col *i* corresponds to ``neurons.iloc[i]``.
        - **full-space indexed**: shape == (M, M) where M is the total
          neuPrint body-ID space; resolved via
          ``n_bodyIds_bodyId_idx_maps.pkl``.
        """
        npz_path = base / "connectivity_matrix.npz"
        map_path = base / "n_bodyIds_bodyId_idx_maps.pkl"

        if not npz_path.exists():
            print(f"[mcns] {npz_path.name} not found, skipping CSR path.")
            return None

        print(f"[mcns] Loading CSR matrix from {npz_path.name} ...")
        try:
            loader = np.load(npz_path)
            data = loader["data"]
            indices = loader["indices"]
            indptr = loader["indptr"]
            shape = tuple(loader["shape"])
            csr = csr_matrix((data, indices, indptr), shape=shape)
            loader.close()
        except Exception as exc:
            print(f"[mcns] Failed to load CSR matrix: {exc}")
            return None

        print(f"[mcns] CSR shape={csr.shape}, nnz={csr.nnz:,}")

        n_neurons = len(neurons)

        # --- Case 1: CSR matches neuron count → neuron-order-indexed -------
        if csr.shape[0] == csr.shape[1] == n_neurons:
            print("[mcns] CSR indexed by neuron order (direct mapping).")
            return MCNSAdapter._csr_neuron_order(neurons=neurons, csr=csr)

        # --- Case 2: full-space CSR → resolve via index maps ----------------
        idx_to_body: dict[int, int] = {}
        if map_path.exists():
            print(f"[mcns] Loading index maps from {map_path.name} ...")
            try:
                with open(map_path, "rb") as f:
                    _, _, _, idx_to_body = pickle.load(f)
            except Exception as exc:
                print(f"[mcns] Failed to load index maps: {exc}")

        if not idx_to_body:
            print(
                "[mcns] CSR is full-space but no index maps found; "
                "cannot resolve body IDs. Falling back."
            )
            return None

        return MCNSAdapter._csr_full_space(
            csr=csr, neuron_ids=neuron_ids, idx_to_body=idx_to_body,
        )

    @staticmethod
    def _csr_neuron_order(neurons: pd.DataFrame, csr: csr_matrix) -> pd.DataFrame:
        """Extract edges from a CSR matrix indexed by neuron DataFrame order.

        Row/col *i* maps to ``neurons.iloc[i]["id"]``.
        M[i, j] = synapses from neuron *j* → neuron *i*.
        """
        body_ids = neurons["id"].values
        coo = csr.tocoo()
        mask = coo.data >= 1
        pre_idx = coo.col[mask]
        post_idx = coo.row[mask]
        syn = coo.data[mask]

        edges = pd.DataFrame({
            "pre_id": body_ids[pre_idx],
            "post_id": body_ids[post_idx],
            "syn_count": syn,
        })

        print(
            f"[mcns] CSR (neuron-order) path: {len(edges):,} edges "
            f"(≥ 1 synapse) from {edges['pre_id'].nunique():,} pre "
            f"× {edges['post_id'].nunique():,} post neurons."
        )
        return edges

    @staticmethod
    def _csr_full_space(
        csr: csr_matrix,
        neuron_ids: set[int],
        idx_to_body: dict[int, int],
    ) -> Optional[pd.DataFrame]:
        """Extract edges from a full neuPrint-space CSR matrix."""
        # Build bodyId → index (reverse of idx_to_body)
        body_to_idx: dict[int, int] = {}
        for idx, bid in idx_to_body.items():
            body_to_idx[bid] = idx

        # Get global indices for our neurons
        our_indices: set[int] = set()
        for bid in neuron_ids:
            idx = body_to_idx.get(bid)
            if idx is not None:
                our_indices.add(idx)

        if not our_indices:
            print("[mcns] No neuron bodyIds found in index maps. Falling back.")
            return None

        print(
            f"[mcns] Resolved {len(our_indices):,} / {len(neuron_ids):,} "
            f"neurons in full-space CSR."
        )

        # Build set→list for column filtering
        our_indices_list = sorted(our_indices)
        our_indices_set = set(our_indices_list)

        rows_list: list[int] = []
        cols_list: list[int] = []
        data_list: list[int] = []

        # Iterate over our post-synaptic rows
        for post_idx in our_indices_list:
            row_start = csr.indptr[post_idx]
            row_end = csr.indptr[post_idx + 1]
            if row_start == row_end:
                continue
            row_cols = csr.indices[row_start:row_end]
            row_data = csr.data[row_start:row_end]
            # Filter to pre-synaptic neurons also in our set
            mask = np.isin(row_cols, list(our_indices_set))
            if not mask.any():
                continue
            filtered_cols = row_cols[mask]
            filtered_data = row_data[mask]
            rows_list.extend([post_idx] * len(filtered_cols))
            cols_list.extend(filtered_cols.tolist())
            data_list.extend(filtered_data.tolist())

        if not rows_list:
            print("[mcns] No edges found for our neurons in CSR. Falling back.")
            return None

        # Map indices back to bodyIds
        pre_body = [idx_to_body[c] for c in cols_list]
        post_body = [idx_to_body[r] for r in rows_list]

        edges = pd.DataFrame({
            "pre_id": pre_body,
            "post_id": post_body,
            "syn_count": data_list,
        })

        edges = edges[edges["syn_count"] >= 1].copy()
        print(
            f"[mcns] CSR path: {len(edges):,} edges "
            f"(≥ 1 synapse) from {edges['pre_id'].nunique():,} pre "
            f"× {edges['post_id'].nunique():,} post neurons."
        )
        return edges

    # ── feather fallback paths ────────────────────────────────────────────

    @staticmethod
    def _try_feather_edges(
        base: Path, neuron_ids: set[int],
    ) -> Optional[pd.DataFrame]:
        """Load edges from the long-format feather file."""
        feather_path = base / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
        if not feather_path.exists():
            print(f"[mcns] {feather_path.name} not found, skipping feather path.")
            return None

        print(f"[mcns] Loading edges from {feather_path.name} ...")
        try:
            raw = pd.read_feather(feather_path)
        except Exception as exc:
            print(f"[mcns] Failed to load feather edges: {exc}")
            return None

        print(f"[mcns] Feather raw edges: {len(raw):,}")

        # Rename to canonical columns
        raw = raw.rename(columns={
            "body_pre": "pre_id",
            "body_post": "post_id",
            "weight": "syn_count",
        })

        # Filter to neurons in our table
        n_before = len(raw)
        raw = raw[
            raw["pre_id"].isin(neuron_ids)
            & raw["post_id"].isin(neuron_ids)
        ].copy()
        n_after = len(raw)
        print(
            f"[mcns] Feather edge filter: {n_after:,} / {n_before:,} edges kept "
            f"({n_after / max(n_before, 1):.2%})"
        )

        # Filter syn_count >= 1
        raw = raw[raw["syn_count"] >= 1]

        # Collapse duplicate (pre, post) pairs (shouldn't exist but be safe)
        raw = (
            raw.groupby(["pre_id", "post_id"], as_index=False)["syn_count"]
               .sum()
        )

        print(
            f"[mcns] Feather path: {len(raw):,} edges "
            f"from {raw['pre_id'].nunique():,} pre "
            f"× {raw['post_id'].nunique():,} post neurons."
        )
        return raw

    @staticmethod
    def _try_orn_partial_edges(
        base: Path, neuron_ids: set[int],
    ) -> Optional[pd.DataFrame]:
        """Load the partial ORN → ALPN/LN adjacency as last-resort edges."""
        partial_path = base / "orn_to_alpnln_AL_roi_adj.feather"
        if not partial_path.exists():
            print(f"[mcns] {partial_path.name} not found, skipping partial path.")
            return None

        print(
            f"[mcns] WARNING: Using partial ORN→ALPN/LN edge list from "
            f"{partial_path.name} — this is NOT the full connectome!"
        )
        try:
            raw = pd.read_feather(partial_path)
        except Exception as exc:
            print(f"[mcns] Failed to load partial edges: {exc}")
            return None

        raw = raw.rename(columns={
            "bodyId_pre": "pre_id",
            "bodyId_post": "post_id",
            "weight": "syn_count",
        })

        # Filter to neurons in our table
        n_before = len(raw)
        raw = raw[
            raw["pre_id"].isin(neuron_ids)
            & raw["post_id"].isin(neuron_ids)
        ].copy()
        n_after = len(raw)
        print(
            f"[mcns] Partial edge filter: {n_after:,} / {n_before:,} edges kept "
            f"({n_after / max(n_before, 1):.2%})"
        )

        raw = raw[raw["syn_count"] >= 1]
        raw = (
            raw.groupby(["pre_id", "post_id"], as_index=False)["syn_count"]
               .sum()
        )

        print(
            f"[mcns] Partial path: {len(raw):,} edges (ORN→ALPN/LN only) "
            f"from {raw['pre_id'].nunique():,} pre "
            f"× {raw['post_id'].nunique():,} post neurons."
        )
        return raw

    # ── QC ────────────────────────────────────────────────────────────────

    @staticmethod
    def _run_qc(neurons: pd.DataFrame, edges: pd.DataFrame) -> None:
        """Run assertions and print QC summary."""

        tag = "MCNS"

        # -- neurons --
        assert neurons["id"].notna().all(), f"{tag}: null neuron ids"
        assert neurons["id"].is_unique, (
            f"{tag}: duplicate neuron ids "
            f"({neurons['id'].duplicated().sum()})"
        )
        valid_sides = {"L", "R", "C"}
        side_vals = set(neurons["side"].dropna().unique())
        assert side_vals.issubset(valid_sides), (
            f"{tag}: unexpected side values: {side_vals - valid_sides}"
        )
        assert neurons["cell_type_raw"].notna().all(), f"{tag}: null cell_type_raw"
        assert neurons["cell_type"].notna().all(), f"{tag}: null cell_type"
        assert set(neurons["region"].unique()).issubset({"brain"}), (
            f"{tag}: unexpected region values: {neurons['region'].unique().tolist()}"
        )

        # -- edges (if any) --
        if len(edges) > 0:
            assert edges["pre_id"].notna().all(), f"{tag}: null pre_id in edges"
            assert edges["post_id"].notna().all(), f"{tag}: null post_id in edges"
            assert edges["syn_count"].notna().all(), f"{tag}: null syn_count in edges"
            assert (edges["syn_count"] > 0).all(), (
                f"{tag}: non-positive syn_count values: "
                f"{edges[edges['syn_count'] <= 0].shape[0]}"
            )

            # referential integrity
            neuron_ids = set(neurons["id"].values)
            orphan_pre = set(edges["pre_id"].unique()) - neuron_ids
            orphan_post = set(edges["post_id"].unique()) - neuron_ids
            assert not orphan_pre, (
                f"{tag}: {len(orphan_pre)} pre_ids in edges not in neurons"
            )
            assert not orphan_post, (
                f"{tag}: {len(orphan_post)} post_ids in edges not in neurons"
            )

        # -- summary --
        print(
            f"[{tag}] QC passed: "
            f"{len(neurons):,} neurons, "
            f"{len(edges):,} edges, "
            f"side counts={neurons['side'].value_counts().to_dict()}, "
            f"region counts={neurons['region'].value_counts().to_dict()}"
        )
