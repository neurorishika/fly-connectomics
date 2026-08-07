"""FlyWire (FAFB, CAVE) connectome adapter.

Loads neuronal metadata and synapse-level connectivity from the FlyWire
dataset and exposes them as a standardised :class:`ConnectomeDataset`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import (
    ConnectomeAdapter,
    ConnectomeDataset,
    extract_side_from_type,
    normalize_side,
    strip_side_suffix,
)

# ═══════════════════════════════════════════════════════════════════════════
# module-level constants
# ═══════════════════════════════════════════════════════════════════════════

# Predicted NT type abbreviation → canonical lower-case label
_NT_MAP: dict[str, Optional[str]] = {
    'ACH':  'acetylcholine',
    'GABA': 'gaba',
    'GLUT': 'glutamate',
    'SER':  'serotonin',
    'DA':   'dopamine',
    'OCT':  'octopamine',
    'UNK':  None,
}

# Substring patterns for super_class assignment
# Applied in order; first match wins.
_SUPER_CLASS_RULES: list[tuple[str, str]] = [
    # (pattern in class_raw lower-case, super_class)
    ('motor',          'motor'),
    ('descending',     'descending'),
    ('ascending',      'ascending'),
    ('sensory',        'sensory'),
    ('ocellar',        'sensory'),
    ('visual',         'sensory'),
    ('optic_lobes',    'sensory'),
    ('olfactory',      'sensory'),       # ORNs
    ('kenyon_cell',    'central'),       # KC
    ('alpn',           'central'),
    ('alln',           'central'),
    ('alin',           'central'),
    ('alon',           'central'),
    ('lhln',           'central'),
    ('lhcent',         'central'),
    ('dan',            'central'),
    ('mbon',           'central'),
    ('mbin',           'central'),
    ('mal',            'central'),
    ('cx',             'central'),
    ('tubu',           'central'),
    ('tpn',            'central'),
    ('pars_intercerebralis', 'central'),
    ('pars_lateralis', 'central'),
    ('bilateral',      'central'),
]

# Class values that are considered olfactory (AL / LH related)
_OLFACTORY_CLASSES: set[str] = {
    'olfactory',
    'alpn',
    'alln',
    'alin',
    'alon',
    'lhln',
    'lhcent',
}


# ═══════════════════════════════════════════════════════════════════════════
# helper functions (module-private)
# ═══════════════════════════════════════════════════════════════════════════

def _map_nt(raw: object) -> Optional[str]:
    """Map a FlyWire ``nt_type`` abbreviation to a canonical name."""
    if not isinstance(raw, str):
        return None
    return _NT_MAP.get(raw.strip().upper(), None)


def _derive_super_class(class_str: object) -> Optional[str]:
    """Map a FlyWire ``class`` value to a simplified super-class string."""
    if not isinstance(class_str, str):
        return None
    c = class_str.strip().lower()
    for pattern, super_cls in _SUPER_CLASS_RULES:
        if pattern in c:
            return super_cls
    return None


def _derive_class(class_str: object) -> Optional[str]:
    """Map a FlyWire ``class`` value to the adapter's ``class`` field.

    Returns ``'olfactory'`` for ORN / ALPN / ALLN / PN / LN types;
    otherwise the raw class string unchanged (or None).
    """
    if not isinstance(class_str, str):
        return None
    c = class_str.strip().lower()
    if c in _OLFACTORY_CLASSES:
        return 'olfactory'
    return c  # raw class string


# ═══════════════════════════════════════════════════════════════════════════
# adapter
# ═══════════════════════════════════════════════════════════════════════════

class FlyWireAdapter(ConnectomeAdapter):
    """Adapter for the **FlyWire** (FAFB, CAVE) connectome dataset.

    Expected files under ``{data_dir}/FlyWire/``:

    * ``classification.csv.gz``
    * ``neurons.csv.gz``
    * ``connections_princeton_no_threshold.csv.gz``
    """

    dataset_name = 'flywire'
    dataset_tag  = 'flywire'

    # ── load ──────────────────────────────────────────────────────────────

    def load(self) -> ConnectomeDataset:
        base = Path(self.data_dir) / 'FlyWire'

        # ------------------------------------------------------------------
        # 1. neurons — merge classification + neurotransmitter info
        # ------------------------------------------------------------------
        print(f"[{self.dataset_name}] Loading classification …")
        cla = pd.read_csv(base / 'classification.csv.gz')
        print(f"[{self.dataset_name}]   classification: {len(cla):,} rows")

        # Rename to canonical column names
        cla = cla.rename(columns={
            'root_id':         'id',
            'hemibrain_type':  'type',
        })

        # --- load neurotransmitter data -----------------------------------
        print(f"[{self.dataset_name}] Loading neurons (NT) …")
        nts = pd.read_csv(base / 'neurons.csv.gz')
        print(f"[{self.dataset_name}]   neurons: {len(nts):,} rows")

        # Merge NT info onto classification (left join on root_id, before rename)
        nts_sub = nts[['root_id', 'nt_type']].rename(columns={'root_id': 'id'})
        raw = cla.merge(nts_sub, on='id', how='left')

        # --- neuron_id ----------------------------------------------------
        raw['id'] = raw['id'].astype('int64')

        # --- side ---------------------------------------------------------
        raw['side'] = raw['side'].apply(normalize_side)

        # Also try to extract side from type as fallback
        side_from_type = raw['type'].apply(extract_side_from_type)
        still_missing = raw['side'].isna() | (raw['side'] == 'C')
        if still_missing.any():
            raw.loc[still_missing, 'side'] = side_from_type[still_missing].fillna('C')

        # --- drop rows with null side -------------------------------------
        before_drop = len(raw)
        raw = raw[raw['side'].notna()].copy()
        dropped = before_drop - len(raw)
        if dropped:
            print(
                f"[{self.dataset_name}]   dropped {dropped:,} rows "
                f"with null side ({dropped / before_drop:.2%})"
            )
        print(
            f"[{self.dataset_name}]   side counts: "
            f"{raw['side'].value_counts().to_dict()}"
        )

        # --- cell_type_raw / cell_type ------------------------------------
        raw['cell_type_raw'] = raw['type'].astype(str)
        raw['cell_type']     = raw['cell_type_raw'].apply(strip_side_suffix)

        # --- super_class --------------------------------------------------
        raw['super_class'] = raw['class'].apply(_derive_super_class)

        # --- class --------------------------------------------------------
        raw['class_derived'] = raw['class'].apply(_derive_class)

        # --- nt_type ------------------------------------------------------
        raw['nt_type'] = raw['nt_type'].apply(_map_nt)

        # --- region -------------------------------------------------------
        raw['region'] = 'brain'  # FlyWire is brain-only

        # --- build final neuron table -------------------------------------
        neurons = raw[[
            'id',
            'side',
            'cell_type_raw',
            'cell_type',
            'super_class',
            'class_derived',
            'nt_type',
            'region',
        ]].rename(columns={'class_derived': 'class'}).copy()

        neurons['dataset'] = self.dataset_tag
        neurons['dataset'] = neurons['dataset'].astype('category')
        neurons['side'] = neurons['side'].astype('category')
        neurons['super_class'] = neurons['super_class'].astype('category')
        neurons['nt_type'] = neurons['nt_type'].astype('category')
        # index = neuron_id (same as id)
        neurons['neuron_id'] = neurons['id']
        neurons = neurons.set_index('neuron_id', drop=False)

        print(
            f"[{self.dataset_name}]   built neuron table: "
            f"{len(neurons):,} neurons"
        )

        # ------------------------------------------------------------------
        # 2. edges — chunked read from large connections file
        # ------------------------------------------------------------------
        print(
            f"[{self.dataset_name}] Loading connections "
            f"(chunked, chunksize=3,000,000) …"
        )
        chunk_iter = pd.read_csv(
            base / 'connections_princeton_no_threshold.csv.gz',
            chunksize=3_000_000,
        )

        # Build set of valid neuron ids for filtering
        valid_ids: set[int] = set(neurons['id'].values)

        edge_chunks: list[pd.DataFrame] = []
        total_read = 0
        total_kept = 0

        for i, chunk in enumerate(chunk_iter):
            chunk = chunk.rename(columns={
                'pre_root_id':  'pre_id',
                'post_root_id': 'post_id',
                'syn_count':    'syn_count',
            })

            total_read += len(chunk)

            # Keep only edges where both ends are in the neuron table
            keep = (
                chunk['pre_id'].isin(valid_ids)
                & chunk['post_id'].isin(valid_ids)
            )
            chunk_filtered = chunk.loc[keep, ['pre_id', 'post_id', 'syn_count']].copy()
            total_kept += len(chunk_filtered)

            # Group within chunk to reduce memory
            if len(chunk_filtered):
                chunk_filtered = (
                    chunk_filtered
                    .groupby(['pre_id', 'post_id'], as_index=False)['syn_count']
                    .sum()
                )
                edge_chunks.append(chunk_filtered)

            # Progress every 5 chunks
            if (i + 1) % 5 == 0:
                print(
                    f"[{self.dataset_name}]   chunk {i + 1}: "
                    f"{total_read:,} rows read, "
                    f"{total_kept:,} kept so far"
                )

        print(
            f"[{self.dataset_name}]   finished reading: "
            f"{total_read:,} total rows, "
            f"{total_kept:,} kept "
            f"({total_kept / max(total_read, 1):.2%})"
        )

        # Concatenate all chunks and do a final group-by to deduplicate
        # across chunk boundaries
        if edge_chunks:
            edges = pd.concat(edge_chunks, ignore_index=True)
            edges = (
                edges.groupby(['pre_id', 'post_id'], as_index=False)['syn_count']
                     .sum()
            )
        else:
            edges = pd.DataFrame(columns=['pre_id', 'post_id', 'syn_count'])

        print(
            f"[{self.dataset_name}]   built edge table: "
            f"{len(edges):,} edges"
        )

        # ------------------------------------------------------------------
        # 3. assertions / QC
        # ------------------------------------------------------------------
        self._run_qc(neurons, edges)

        return ConnectomeDataset(
            neurons=neurons,
            edges=edges,
            dataset_name=self.dataset_name,
            dataset_tag=self.dataset_tag,
        )

    # ── QC ────────────────────────────────────────────────────────────────

    @staticmethod
    def _run_qc(neurons: pd.DataFrame, edges: pd.DataFrame) -> None:
        """Run assertions on loaded data."""

        # -- neurons --
        assert neurons['id'].notna().all(), \
            'FlyWire: null neuron ids found'
        assert neurons['id'].is_unique, \
            f'FlyWire: duplicate neuron ids ({neurons["id"].duplicated().sum()})'
        assert set(neurons['side'].dropna().unique()).issubset({'L', 'R', 'C'}), \
            f'FlyWire: unexpected side values: {neurons["side"].value_counts(dropna=False).to_dict()}'
        assert neurons['cell_type_raw'].notna().all(), \
            'FlyWire: null cell_type_raw'
        assert neurons['cell_type'].notna().all(), \
            'FlyWire: null cell_type'
        assert (neurons['region'] == 'brain').all(), \
            f'FlyWire: unexpected region values: {neurons["region"].unique().tolist()}'

        # -- edges --
        assert edges['pre_id'].notna().all(), \
            'FlyWire: null pre_id in edges'
        assert edges['post_id'].notna().all(), \
            'FlyWire: null post_id in edges'
        assert edges['syn_count'].notna().all(), \
            'FlyWire: null syn_count in edges'
        assert (edges['syn_count'] > 0).all(), \
            f'FlyWire: non-positive syn_count values: {edges[edges["syn_count"] <= 0].shape[0]}'

        # -- referential integrity --
        neuron_ids = set(neurons['id'].values)
        orphan_pre = set(edges['pre_id'].unique()) - neuron_ids
        orphan_post = set(edges['post_id'].unique()) - neuron_ids
        assert not orphan_pre, \
            f'FlyWire: {len(orphan_pre)} pre_ids in edges not in neurons'
        assert not orphan_post, \
            f'FlyWire: {len(orphan_post)} post_ids in edges not in neurons'

        # -- summary --
        print(
            f"[flywire.FlyWireAdapter] QC passed: "
            f"{len(neurons):,} neurons, "
            f"{len(edges):,} edges, "
            f"side counts={neurons['side'].value_counts().to_dict()}"
        )
