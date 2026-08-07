"""BANC (Brain And Nerve Cord, CAVE) connectome adapter.

Loads neuronal metadata and synapse-level connectivity from the BANC dataset
and exposes them as a standardised :class:`ConnectomeDataset`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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

# Predicted NT type → canonical lower-case label
_NT_MAP: dict[str, Optional[str]] = {
    'ACH': 'acetylcholine',
    'GABA': 'gaba',
    'GLUT': 'glutamate',
    'SER': 'serotonin',
    'HIST': 'histamine',
    'DA': 'dopamine',
    'OCT': 'octopamine',
    'TYR': 'tyramine',
}

# Class string → simplified super_class
_CLASS_TO_SUPER: dict[str, str] = {
    'olfactory_receptor_neuron': 'sensory',
    'antennal_lobe_projection_neuron': 'central',
    'antennal_lobe_local_neuron': 'central',
    'kenyon_cell': 'central',
    'mushroom_body_output_neuron': 'central',
    'motor': 'motor',
    'motor_neuron': 'motor',
    'leg_motor_neuron': 'motor',
    'neck_motor_neuron': 'motor',
    'wing_motor_neuron': 'motor',
    'haltere_motor_neuron': 'motor',
    'abdomen_motor_neuron': 'motor',
    'antenna_motor_neuron': 'motor',
    'proboscis_motor_neuron': 'motor',
    'pharynx_motor_neuron': 'motor',
    'crop_motor_neuron': 'motor',
    'uterus_motor_neuron': 'motor',
    'eye_motor_neuron': 'motor',
    'salivary_motor_neuron': 'motor',
    'hind_leg_motor_neuron': 'motor',
    'thoracic_abdominal_segmental_motor_neuron': 'motor',
    'unknown_motor_neuron': 'motor',
    'unknown_thoracic_abdominal_motor_neuron': 'motor',
    'ascending': 'ascending',
    'ascending_neuron': 'ascending',
    'descending': 'descending',
    'descending_neuron': 'descending',
}

# Class string that should produce class='olfactory'
_OLFACTORY_CLASSES: set[str] = {
    'olfactory_receptor_neuron',
    'antennal_lobe_projection_neuron',
    'antennal_lobe_local_neuron',
}

# Nerve substrings that strongly imply nerve_cord
_NERVE_CORD_NERVE_TOKENS: tuple[str, ...] = (
    'leg', 'wing', 'haltere', 'abdominal', 'thoracic', 'ventral_cervical',
    'prosternal', 'metathoracic', 'mesothoracic', 'prothoracic',
)

# Class substrings that strongly imply nerve_cord
_NERVE_CORD_CLASS_TOKENS: tuple[str, ...] = (
    'motor', 'leg_', 'wing_', 'haltere_', 'abdom',
    'thoracic_abdominal', 'neck_motor', 'ventral_nerve_cord',
    'ascending', 'taste_bristle', 'taste_peg', 'chordotonal',
    'campaniform', 'hair_plate', 'strand', 'multidendritic',
    'reproductive_tract',
)


# ═══════════════════════════════════════════════════════════════════════════
# helper functions (module-private)
# ═══════════════════════════════════════════════════════════════════════════

def _map_nt(raw: object) -> Optional[str]:
    """Map a BANC ``Predicted NT type`` abbreviation to a canonical name."""
    if not isinstance(raw, str):
        return None
    return _NT_MAP.get(raw.strip().upper(), None)


def _derive_super_class(class_str: object) -> Optional[str]:
    """Map a BANC ``Class`` value to a simplified super-class string."""
    if not isinstance(class_str, str):
        return None
    c = class_str.strip().lower()
    return _CLASS_TO_SUPER.get(c, None)


def _derive_class(class_str: object) -> Optional[str]:
    """Map a BANC ``Class`` value to the adapter's ``class`` field.

    Returns ``'olfactory'`` for ORN / ALPN / ALLN types; otherwise the
    raw class string unchanged (or None).
    """
    if not isinstance(class_str, str):
        return None
    c = class_str.strip().lower()
    if c in _OLFACTORY_CLASSES:
        return 'olfactory'
    return c  # raw class string


def _derive_region(class_str: object, nerve: object) -> str:
    """Return ``'brain'`` or ``'nerve_cord'`` for a neuron.

    Heuristics (in order):
    1. If *class_str* contains known nerve-cord class tokens → ``'nerve_cord'``
    2. If *nerve* contains known nerve-cord nerve tokens → ``'nerve_cord'``
    3. Default → ``'brain'``
    """
    # 1 – class-based
    if isinstance(class_str, str):
        c = class_str.strip().lower()
        for tok in _NERVE_CORD_CLASS_TOKENS:
            if tok in c:
                return 'nerve_cord'
        # descending neurons have cell bodies in the brain
        if c in ('descending', 'descending_neuron'):
            return 'brain'

    # 2 – nerve-based
    if isinstance(nerve, str):
        n = nerve.strip().lower()
        for tok in _NERVE_CORD_NERVE_TOKENS:
            if tok in n:
                return 'nerve_cord'

    # 3 – default
    return 'brain'


# ═══════════════════════════════════════════════════════════════════════════
# adapter
# ═══════════════════════════════════════════════════════════════════════════

class BANCAdapter(ConnectomeAdapter):
    """Adapter for the **BANC** (Brain And Nerve Cord) CAVE dataset.

    Expected files under ``{data_dir}/BANC/``:

    * ``neurons.csv.gz``
    * ``connections_princeton.csv.gz``
    """

    dataset_name = 'banc'
    dataset_tag  = 'banc'

    # ── load ──────────────────────────────────────────────────────────────

    def load(self) -> ConnectomeDataset:
        base = Path(self.data_dir) / 'BANC'

        # ------------------------------------------------------------------
        # 1. neurons
        # ------------------------------------------------------------------
        raw = pd.read_csv(base / 'neurons.csv.gz')

        # rename to canonical column names
        raw = raw.rename(columns={
            'Root ID':             'id',
            'Primary Cell Type':   'type',
            'Soma side':           'side_raw',
            'Class':               'class_raw',
            'Sub Class':           'sub_class',
            'Predicted NT type':   'predicted_nt',
            'Nerve':               'nerve',
            'Body Part':           'body_part',
            'Super Class':         'super_class_original',
        })

        # --- neuron_id ----------------------------------------------------
        raw['id'] = raw['id'].astype('int64')

        # --- side ---------------------------------------------------------
        side_from_col  = raw['side_raw'].apply(normalize_side)
        side_from_type = raw['type'].apply(extract_side_from_type)
        raw['side']    = side_from_col.fillna(side_from_type)

        # --- cell_type_raw / cell_type ------------------------------------
        raw['cell_type_raw'] = raw['type'].astype(str)
        raw['cell_type']     = raw['cell_type_raw'].apply(strip_side_suffix)

        # --- super_class --------------------------------------------------
        raw['super_class'] = raw['class_raw'].apply(_derive_super_class)

        # --- class --------------------------------------------------------
        raw['class'] = raw['class_raw'].apply(_derive_class)

        # --- nt_type ------------------------------------------------------
        raw['nt_type'] = raw['predicted_nt'].apply(_map_nt)

        # --- region -------------------------------------------------------
        raw['region'] = raw.apply(
            lambda r: _derive_region(r['class_raw'], r.get('nerve')),
            axis=1,
        )

        # --- build final neuron table -------------------------------------
        neurons = raw[[
            'id',
            'side',
            'cell_type_raw',
            'cell_type',
            'super_class',
            'class',
            'nt_type',
            'region',
        ]].copy()

        neurons['dataset'] = self.dataset_tag
        neurons['dataset'] = neurons['dataset'].astype('category')
        neurons['side'] = neurons['side'].astype('category')
        neurons['super_class'] = neurons['super_class'].astype('category')
        neurons['nt_type'] = neurons['nt_type'].astype('category')
        # index = neuron_id (same as id)
        neurons['neuron_id'] = neurons['id']
        neurons = neurons.set_index('neuron_id', drop=False)

        # ------------------------------------------------------------------
        # 2. edges
        # ------------------------------------------------------------------
        edges = pd.read_csv(base / 'connections_princeton.csv.gz')
        edges = edges.rename(columns={
            'pre_root_id':  'pre_id',
            'post_root_id': 'post_id',
            'syn_count':    'syn_count',
        })

        # keep only edges whose both ends are in the neuron table
        valid_ids = set(neurons['id'].values)
        n_before  = len(edges)
        edges     = edges[
            edges['pre_id'].isin(valid_ids)
            & edges['post_id'].isin(valid_ids)
        ].copy()
        n_after = len(edges)
        if n_before > 0:
            fraction = n_after / n_before
        else:
            fraction = 1.0
        print(
            f"[{self.dataset_name}] edge filter: "
            f"{n_after:,} / {n_before:,} edges kept "
            f"({fraction:.2%})"
        )

        # group by (pre_id, post_id), sum syn_count
        edges = (
            edges.groupby(['pre_id', 'post_id'], as_index=False)['syn_count']
                 .sum()
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
            'BANC: null neuron ids found'
        assert neurons['id'].is_unique, \
            f'BANC: duplicate neuron ids ({neurons["id"].duplicated().sum()})'
        assert set(neurons['side'].dropna().unique()).issubset({'L', 'R', 'C'}), \
            f'BANC: unexpected side values: {neurons["side"].value_counts(dropna=False).to_dict()}'
        assert neurons['cell_type_raw'].notna().all(), \
            'BANC: null cell_type_raw'
        assert neurons['cell_type'].notna().all(), \
            'BANC: null cell_type'
        assert set(neurons['region'].unique()).issubset({'brain', 'nerve_cord'}), \
            f'BANC: unexpected region values: {neurons["region"].unique().tolist()}'

        # -- edges --
        assert edges['pre_id'].notna().all(), \
            'BANC: null pre_id in edges'
        assert edges['post_id'].notna().all(), \
            'BANC: null post_id in edges'
        assert edges['syn_count'].notna().all(), \
            'BANC: null syn_count in edges'
        assert (edges['syn_count'] > 0).all(), \
            f'BANC: non-positive syn_count values: {edges[edges["syn_count"] <= 0].shape[0]}'

        # -- referential integrity --
        neuron_ids = set(neurons['id'].values)
        orphan_pre = set(edges['pre_id'].unique()) - neuron_ids
        orphan_post = set(edges['post_id'].unique()) - neuron_ids
        assert not orphan_pre, \
            f'BANC: {len(orphan_pre)} pre_ids in edges not in neurons'
        assert not orphan_post, \
            f'BANC: {len(orphan_post)} post_ids in edges not in neurons'

        # -- summary --
        print(
            f"[{__name__}.BANCAdapter] QC passed: "
            f"{len(neurons):,} neurons, "
            f"{len(edges):,} edges, "
            f"side counts={neurons['side'].value_counts().to_dict()}, "
            f"region counts={neurons['region'].value_counts().to_dict()}"
        )
