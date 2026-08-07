"""
Seed definition module for ORN lateralization decay analysis.

Defines ORN seed vectors, control seed sets, and null-model seed generators.
All seed-construction functions operate on the harmonised neuron DataFrame
(indexed by ``neuron_id``) and the ``id_to_idx`` mapping produced during
graph construction.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .adapters.base import is_orn_type, glom_from_type, strip_side_suffix
from .config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ORN subtype enumeration
# ---------------------------------------------------------------------------


def enumerate_orn_subtypes(
    neurons: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify ORN subtypes and return seed-group information.

    Filters *neurons* to rows where ``class`` contains ``'olfactory'``,
    ``super_class == 'sensory'``, and ``cell_type`` matches the ORN naming
    pattern (via :func:`is_orn_type`).  Groups by side-stripped
    ``cell_type`` × ``side`` and validates each glomerulus against the
    thresholds in ``config.seeds``.

    Parameters
    ----------
    neurons : pd.DataFrame
        Indexed by ``neuron_id`` (int64).  Must have columns: ``class``,
        ``super_class``, ``cell_type``, ``side``.
    config : Config
        Analysis configuration; the ``seeds`` sub-config provides
        ``min_orns_per_side`` and ``max_side_imbalance_ratio``.

    Returns
    -------
    seeds_info : pd.DataFrame
        Columns: ``cell_type`` (side-stripped), ``side`` (``'L'``/``'R'``),
        ``n_orns``, ``orn_ids`` (list of ``neuron_id``).
        Only glomeruli that pass the validation checks.
    excluded : pd.DataFrame
        Same schema as *seeds_info*, for glomeruli that were excluded
        because they fell below ``min_orns_per_side``.
    """
    # -- filter to olfactory sensory ORNs -----------------------------------
    mask_olfactory = neurons["class"].str.lower().str.contains(
        "olfactory", na=False
    )
    mask_sensory = neurons["super_class"] == "sensory"
    mask_orn_type = neurons["cell_type"].apply(is_orn_type)

    orn_mask = mask_olfactory & mask_sensory & mask_orn_type
    orns = neurons.loc[orn_mask].copy()

    logger.info(
        "ORN filter: %s / %s neurons retained", orn_mask.sum(), len(neurons)
    )

    # -- strip side suffix to get glomerulus name ---------------------------
    orns["glom"] = orns["cell_type"].apply(strip_side_suffix)

    # -- group by glomerulus × side -----------------------------------------
    records: List[dict] = []
    for (glom, side), group in orns.groupby(["glom", "side"], observed=True):
        records.append(
            {
                "cell_type": glom,
                "side": side,
                "n_orns": len(group),
                "orn_ids": list(group.index),
            }
        )

    seeds_info = pd.DataFrame(records)

    if seeds_info.empty:
        logger.warning("No ORN subtypes found — check neuron metadata.")
        empty_df = pd.DataFrame(
            columns=["cell_type", "side", "n_orns", "orn_ids"]
        )
        return empty_df, empty_df.copy()

    # -- validation ---------------------------------------------------------
    distinct_glom = seeds_info["cell_type"].nunique()
    total_orns = int(seeds_info["n_orns"].sum())

    print(f"\n{'=' * 60}")
    print("ORN SEED VALIDATION")
    print(f"{'=' * 60}")
    print(f"Distinct glomeruli (cell_type values): {distinct_glom}")
    print(f"Total ORNs: {total_orns}")
    print()

    max_imbalance = config.seeds.max_side_imbalance_ratio
    min_orns = config.seeds.min_orns_per_side

    excluded_rows: List[dict] = []
    kept_rows: List[dict] = []

    for glom in sorted(seeds_info["cell_type"].unique()):
        sub = seeds_info[seeds_info["cell_type"] == glom]
        n_L = int(sub.loc[sub["side"] == "L", "n_orns"].sum())
        n_R = int(sub.loc[sub["side"] == "R", "n_orns"].sum())

        # -- imbalance flag -------------------------------------------------
        imbalance_flag = ""
        if n_L > 0 and n_R > 0:
            ratio = max(n_L, n_R) / min(n_L, n_R)
            if ratio > max_imbalance:
                imbalance_flag = f" ⚠ IMBALANCE (ratio={ratio:.1f})"
        elif n_L == 0 and n_R == 0:
            imbalance_flag = " ⚠ NO NEURONS"
        elif n_L == 0 or n_R == 0:
            imbalance_flag = " ⚠ MISSING SIDE"

        # -- minimum-ORNs check ---------------------------------------------
        exclude_this = False
        exclude_flag = ""
        if n_L < min_orns or n_R < min_orns:
            exclude_flag = f" ✗ EXCLUDED (< {min_orns}/side)"
            exclude_this = True

        status = (
            f"L={n_L:3d}  R={n_R:3d}{imbalance_flag}{exclude_flag}"
        )
        print(f"  {glom:24s}  {status}")

        if exclude_this:
            excluded_rows.extend(sub.to_dict("records"))
        else:
            kept_rows.extend(sub.to_dict("records"))

    n_kept_glom = len({r["cell_type"] for r in kept_rows}) if kept_rows else 0
    n_excl_glom = (
        len({r["cell_type"] for r in excluded_rows}) if excluded_rows else 0
    )

    print(
        f"\nKept:     {len(kept_rows):3d} seed groups across "
        f"{n_kept_glom} glomeruli"
    )
    print(
        f"Excluded: {len(excluded_rows):3d} seed groups across "
        f"{n_excl_glom} glomeruli"
    )
    print(f"{'=' * 60}\n")

    seeds_info_kept = (
        pd.DataFrame(kept_rows)
        if kept_rows
        else pd.DataFrame(columns=["cell_type", "side", "n_orns", "orn_ids"])
    )
    seeds_info_excluded = (
        pd.DataFrame(excluded_rows)
        if excluded_rows
        else pd.DataFrame(columns=["cell_type", "side", "n_orns", "orn_ids"])
    )

    return seeds_info_kept, seeds_info_excluded


# ---------------------------------------------------------------------------
# Seed vector construction
# ---------------------------------------------------------------------------


def build_seed_vector(
    neuron_ids: List[int],
    id_to_idx: Dict[int, int],
    n_neurons: int,
) -> NDArray[np.float64]:
    """Create a dense seed vector with equal mass on each neuron.

    Parameters
    ----------
    neuron_ids : list of int
        Neuron IDs (matching the ``neurons`` DataFrame index) to include
        in the seed.
    id_to_idx : dict
        Mapping from ``neuron_id`` → matrix row index (0 … n_neurons-1).
    n_neurons : int
        Total number of neurons (dimension of the weight matrix).

    Returns
    -------
    np.ndarray
        Shape ``(n_neurons,)``, dtype ``float64``, summing to 1.0.
        Each included neuron receives mass ``1 / len(neuron_ids)``.
    """
    seed = np.zeros(n_neurons, dtype=np.float64)

    n = len(neuron_ids)
    if n == 0:
        return seed

    mass = 1.0 / n
    for nid in neuron_ids:
        idx = id_to_idx[nid]
        seed[idx] = mass

    # -- assertions ---------------------------------------------------------
    assert abs(seed.sum() - 1.0) < 1e-12, (
        f"Seed sum = {seed.sum():.15f}, expected 1.0"
    )
    nonzero_indices = set(np.where(seed > 0)[0])
    expected_indices = {id_to_idx[nid] for nid in neuron_ids}
    assert nonzero_indices == expected_indices, (
        "Nonzero entries do not match intended neuron set"
    )

    return seed


# ---------------------------------------------------------------------------
# Seed matrix — one column per seed group
# ---------------------------------------------------------------------------


def build_seed_matrix(
    seeds_info: pd.DataFrame,
    id_to_idx: Dict[int, int],
    n_neurons: int,
) -> Tuple[NDArray[np.float64], List[str]]:
    """Build a dense ``(n_neurons, n_seeds)`` matrix of seed vectors.

    Parameters
    ----------
    seeds_info : pd.DataFrame
        Output of :func:`enumerate_orn_subtypes` (the *kept* portion).
        Must have column ``orn_ids`` (list of neuron IDs).
    id_to_idx : dict
        Mapping from ``neuron_id`` → matrix row index.
    n_neurons : int
        Total number of neurons.

    Returns
    -------
    S : np.ndarray
        Shape ``(n_neurons, n_seeds)``, dtype ``float64``.
    labels : list of str
        Column labels of the form ``'{cell_type}_{side}'``, one per seed.
    """
    n_seeds = len(seeds_info)
    S = np.zeros((n_neurons, n_seeds), dtype=np.float64)
    labels: List[str] = []

    for col_idx, (_, row) in enumerate(seeds_info.iterrows()):
        orn_ids: list = row["orn_ids"]
        cell_type: str = row["cell_type"]
        side: str = row["side"]

        seed_vec = build_seed_vector(orn_ids, id_to_idx, n_neurons)
        S[:, col_idx] = seed_vec
        labels.append(f"{cell_type}_{side}")

    logger.info(
        "Built seed matrix: %d neurons × %d seeds", S.shape[0], S.shape[1]
    )

    return S, labels


# ---------------------------------------------------------------------------
# Control seed sets
# ---------------------------------------------------------------------------


def select_control_seeds(
    neurons: pd.DataFrame,
    config: Config,
) -> Dict[str, list]:
    """Select control neuron sets: visual and mechanosensory.

    Visual control
        Neurons whose ``class`` string contains ``'visual'`` or
        ``'photoreceptor'``, restricted to one side (prefers ``'R'``).

    Mechanosensory control
        Neurons whose ``cell_type`` or ``class`` contains ``'johnston'``
        (Johnston's organ), restricted to one side.

    Parameters
    ----------
    neurons : pd.DataFrame
        Indexed by ``neuron_id``.
    config : Config
        Analysis configuration (not currently used; reserved for future
        control-tuning parameters).

    Returns
    -------
    dict[str, list]
        Keys ``'visual'`` and ``'mechanosensory'``, each mapping to a
        list of ``neuron_id`` values.
    """
    # -- visual -------------------------------------------------------------
    vis_mask = neurons["class"].str.lower().str.contains(
        "visual|photoreceptor", na=False, regex=True
    )
    vis_neurons = neurons.loc[vis_mask]
    vis_ids: list = []

    if len(vis_neurons) > 0:
        # Pick one side only (prefer R)
        for side in ("R", "L"):
            side_ids = vis_neurons.loc[
                vis_neurons["side"] == side
            ].index.tolist()
            if side_ids:
                vis_ids = side_ids
                break
        if not vis_ids:  # fallback: C-only or no side at all
            vis_ids = vis_neurons.index.tolist()

    logger.info("Visual control seeds: %d neurons", len(vis_ids))

    # -- mechanosensory (Johnston's organ) ----------------------------------
    mech_mask = neurons["cell_type"].str.lower().str.contains(
        "johnston", na=False
    ) | neurons["class"].str.lower().str.contains("johnston", na=False)
    mech_neurons = neurons.loc[mech_mask]
    mech_ids: list = []

    if len(mech_neurons) > 0:
        for side in ("R", "L"):
            side_ids = mech_neurons.loc[
                mech_neurons["side"] == side
            ].index.tolist()
            if side_ids:
                mech_ids = side_ids
                break
        if not mech_ids:
            mech_ids = mech_neurons.index.tolist()

    logger.info("Mechanosensory control seeds: %d neurons", len(mech_ids))

    return {"visual": vis_ids, "mechanosensory": mech_ids}


# ---------------------------------------------------------------------------
# Random null seeds
# ---------------------------------------------------------------------------


def generate_random_null_seeds(
    neurons: pd.DataFrame,
    seeds_info: pd.DataFrame,
    n_repeats: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """Generate size-matched random same-side sensory null seeds.

    For each real ORN seed group in *seeds_info*, draw *n_repeats* random
    sets of sensory neurons of identical size, restricted to the same
    side.

    Parameters
    ----------
    neurons : pd.DataFrame
        Indexed by ``neuron_id``.  Must have columns ``super_class`` and
        ``side``.
    seeds_info : pd.DataFrame
        Output of :func:`enumerate_orn_subtypes` (the *kept* portion).
        Must have columns ``side`` and ``n_orns``.
    n_repeats : int
        Number of random null sets to draw per real seed group.
    rng : np.random.Generator
        Seeded random generator for reproducibility.

    Returns
    -------
    list[list[int]]
        Each element is a list of ``neuron_id`` values forming one null
        seed.  Total length ≤ ``len(seeds_info) * n_repeats``
        (fewer if any side pool is too small).
    """
    sensory_mask = neurons["super_class"] == "sensory"
    sensory = neurons.loc[sensory_mask]

    null_seeds: List[List[int]] = []

    for _, row in seeds_info.iterrows():
        side: str = row["side"]
        size: int = int(row["n_orns"])

        # Same-side sensory pool (exclude central)
        pool = (
            sensory.loc[sensory["side"] == side]
            .index.tolist()
        )

        if len(pool) < size:
            logger.warning(
                "Skipping null seeds for side=%s: need %d neurons but "
                "only %d same-side sensory neurons available.",
                side,
                size,
                len(pool),
            )
            continue

        for _ in range(n_repeats):
            draw = rng.choice(pool, size=size, replace=False)
            null_seeds.append(list(draw))

    logger.info(
        "Generated %d random null seeds (%d real groups × %d repeats)",
        len(null_seeds),
        len(seeds_info),
        n_repeats,
    )

    return null_seeds
