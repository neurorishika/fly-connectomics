"""
Lateralization index (LI) computation and decay fitting.

Primary analysis is at the **cell_type × side** level, not single-neuron
level.  Per-hop influence arrays are first aggregated to type×side groups;
ipsi/contra means are formed by averaging over both stimulation sides; LI
is computed with a noise-floor guard; exponential decay is fit to the
population-averaged |LI| and a crossover hop is identified per target type.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .adapters.base import strip_side_suffix
from .config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Aggregate per-hop influence to cell_type × side
# ---------------------------------------------------------------------------


def aggregate_to_cell_type(
    per_hop: NDArray[np.float32],
    neurons: pd.DataFrame,
    id_to_idx: dict,
) -> dict:
    """Aggregate per-hop influence from individual neurons to (cell_type, side).

    Parameters
    ----------
    per_hop : np.ndarray
        Shape ``(n_hops + 1, n_neurons, n_seeds)``, dtype float32, as
        returned by :func:`~.propagate.propagate_batch`.
    neurons : pd.DataFrame
        Indexed by ``neuron_id``.  Must have columns ``cell_type`` and
        ``side``.  Only rows whose index appears in *id_to_idx* are used.
    id_to_idx : dict
        Mapping ``neuron_id → matrix row index``.

    Returns
    -------
    dict
        Keys:

        - ``by_type_side`` : :class:`pd.DataFrame`
            Columns: ``cell_type``, ``side``, ``hop``, ``seed_idx``,
            ``influence``.  *influence* is the mean per-hop influence
            across all neurons in the (cell_type, side) group.
        - ``type_side_index`` : dict
            Mapping ``(cell_type, side) → integer index`` for the unique
            groups discovered.
        - ``neuron_counts`` : :class:`pd.DataFrame`
            Columns: ``cell_type``, ``side``, ``n_neurons``.
    """
    n_hops_plus_1, n_neurons, n_seeds = per_hop.shape
    n_hops = n_hops_plus_1 - 1

    # Reverse mapping: matrix index → neuron_id
    idx_to_id: Dict[int, int] = {idx: nid for nid, idx in id_to_idx.items()}

    # Build arrays of cell_type and side aligned to matrix indices (vectorized)
    # Create a neuron_id array aligned to matrix indices
    neuron_id_arr = np.empty(n_neurons, dtype=object)
    for idx, nid in idx_to_id.items():
        neuron_id_arr[idx] = nid

    # Use pandas .loc with the array of IDs to get columns in one shot
    neuron_id_set = set(neurons.index)
    valid_mask = np.array([nid is not None and nid in neuron_id_set for nid in neuron_id_arr])
    cell_type_arr = np.empty(n_neurons, dtype=object)
    side_arr = np.empty(n_neurons, dtype=object)
    cell_type_arr[~valid_mask] = None
    side_arr[~valid_mask] = None

    if valid_mask.any():
        valid_ids = neuron_id_arr[valid_mask]
        # Get cell_type and side as numpy arrays via .loc (vectorized)
        cell_type_arr[valid_mask] = neurons.loc[valid_ids, 'cell_type'].values
        side_arr[valid_mask] = neurons.loc[valid_ids, 'side'].values

    # Unique (cell_type, side) groups (excluding None)
    valid = cell_type_arr != None  # noqa: E711
    pairs = list(
        sorted(set(zip(cell_type_arr[valid], side_arr[valid])))
    )
    type_side_index: Dict[Tuple[str, str], int] = {
        pair: i for i, pair in enumerate(pairs)
    }
    n_groups = len(pairs)

    # Map each neuron to its group index (-1 = invalid)
    # Fast path: use a dict from (cell_type, side) → group index
    pair_to_group = type_side_index  # already a dict
    neuron_to_group = np.full(n_neurons, -1, dtype=np.int32)
    for i in range(n_neurons):
        if cell_type_arr[i] is not None:
            neuron_to_group[i] = pair_to_group.get((cell_type_arr[i], side_arr[i]), -1)

    # Neuron counts per group
    counts = np.bincount(
        neuron_to_group[neuron_to_group >= 0], minlength=n_groups
    )
    neuron_counts_df = pd.DataFrame(
        {
            "cell_type": [p[0] for p in pairs],
            "side": [p[1] for p in pairs],
            "n_neurons": counts,
        }
    )

    # --- Accumulate mean influence per group (vectorized inner loop) ---
    records: List[dict] = []
    
    # Pre-compute valid indices (neurons that have a type/side assigned)
    valid_idx = neuron_to_group >= 0
    valid_groups = neuron_to_group[valid_idx]
    n_valid = len(valid_groups)

    for hop in range(n_hops_plus_1):
        hop_data = per_hop[hop]  # (n_neurons, n_seeds)

        for seed_idx in range(n_seeds):
            infl = hop_data[:, seed_idx]  # (n_neurons,)
            
            # Sum influence per group (only for valid neurons)
            group_sums = np.bincount(
                valid_groups,
                weights=infl[valid_idx],
                minlength=n_groups,
            )
            group_means = group_sums / np.maximum(counts, 1)
            
            # Only record groups with nonzero influence
            nonzero = np.where(group_means > 0)[0]
            for gi in nonzero:
                records.append(
                    {
                        "cell_type": pairs[gi][0],
                        "side": pairs[gi][1],
                        "hop": hop,
                        "seed_idx": seed_idx,
                        "influence": float(group_means[gi]),
                    }
                )

    by_type_side = pd.DataFrame(records)

    logger.info(
        "Aggregated %d hops × %d seeds → %d (cell_type, side) groups "
        "(%d records)",
        n_hops_plus_1,
        n_seeds,
        n_groups,
        len(by_type_side),
    )

    return {
        "by_type_side": by_type_side,
        "type_side_index": type_side_index,
        "neuron_counts": neuron_counts_df,
    }


# ---------------------------------------------------------------------------
# 2. Fold ipsi / contra by averaging over stimulation sides
# ---------------------------------------------------------------------------


def fold_ipsi_contra(
    influence_by_type: pd.DataFrame,
    seed_info: pd.DataFrame,
) -> pd.DataFrame:
    """Average over stimulation sides to produce ipsi- and contralateral
    influence per (seed glomerulus, target type, hop).

    For seed glomerulus *A* and target type *T* at hop *n*::

        I_ipsi(T)  = mean[ I(A_L → T_L),  I(A_R → T_R) ]
        I_contra(T) = mean[ I(A_L → T_R),  I(A_R → T_L) ]

    Averaging over both stimulation sides cancels first-order reconstruction
    asymmetry between hemispheres.

    Parameters
    ----------
    influence_by_type : pd.DataFrame
        Output of :func:`aggregate_to_cell_type` (the ``"by_type_side"``
        key).  Must have columns: ``cell_type``, ``side``, ``hop``,
        ``seed_idx``, ``influence``.
    seed_info : pd.DataFrame
        Must map ``seed_idx`` → ``cell_type`` (glomerulus name, side-stripped)
        and ``side`` (``'L'`` / ``'R'``).  Typically derived from the output
        of :func:`~.seeds.enumerate_orn_subtypes`.

    Returns
    -------
    pd.DataFrame
        Columns: ``seed_glomerulus``, ``target_type``, ``hop``,
        ``I_ipsi``, ``I_contra``.  One row per unique combination.
    """
    # Build seed lookup: seed_idx → (glomerulus, side)
    seed_lookup: Dict[int, Tuple[str, str]] = {}
    for _, row in seed_info.iterrows():
        # seed_info may have seed_idx as column or index
        sidx = row.get("seed_idx", row.name)
        if isinstance(sidx, (int, np.integer)):
            seed_lookup[int(sidx)] = (str(row["cell_type"]), str(row["side"]))

    # Build lookup DataFrames for vectorized mapping
    seed_idx_vals = np.array(list(seed_lookup.keys()))
    glom_vals = np.array([v[0] for v in seed_lookup.values()])
    side_vals = np.array([v[1] for v in seed_lookup.values()])
    seed_to_glom = dict(zip(seed_idx_vals, glom_vals))
    seed_to_side = dict(zip(seed_idx_vals, side_vals))

    # Merge seed metadata into influence data (vectorized map)
    df = influence_by_type.copy()
    df["seed_glom"] = df["seed_idx"].map(seed_to_glom)
    df["seed_side"] = df["seed_idx"].map(seed_to_side)

    # Drop rows where seed mapping failed
    n_before = len(df)
    df = df.dropna(subset=["seed_glom", "seed_side"])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(
            "Dropped %d rows with unmatched seed_idx (missing seed_info)", n_dropped
        )

    # For matching target types across sides, we need a side-free target
    # type label.  Strip known side suffixes from cell_type.
    df["target_type_base"] = df["cell_type"].apply(_safe_strip_side)

    # Determine unique seeds and their sides
    seed_sides: Dict[str, Dict[str, int]] = {}  # glom → {side: seed_idx}
    for sidx, (glom, side) in seed_lookup.items():
        seed_sides.setdefault(glom, {})[side] = sidx

    # Vectorized fold using pivot_table instead of groupby+iterrows
    # Pivot: rows = (seed_glom, target_type_base, hop), columns = (seed_side, side)
    pivot = df.pivot_table(
        index=["seed_glom", "target_type_base", "hop"],
        columns=["seed_side", "side"],
        values="influence",
        aggfunc="mean",
    )

    # Compute I_ipsi = mean(LL, RR), I_contra = mean(LR, RL)
    has_LL = ("L", "L") in pivot.columns
    has_RR = ("R", "R") in pivot.columns
    has_LR = ("L", "R") in pivot.columns
    has_RL = ("R", "L") in pivot.columns

    ipsi_parts = []
    if has_LL:
        ipsi_parts.append(pivot[("L", "L")])
    if has_RR:
        ipsi_parts.append(pivot[("R", "R")])

    contra_parts = []
    if has_LR:
        contra_parts.append(pivot[("L", "R")])
    if has_RL:
        contra_parts.append(pivot[("R", "L")])

    if ipsi_parts:
        I_ipsi = pd.concat(ipsi_parts, axis=1).mean(axis=1)
    else:
        I_ipsi = pd.Series(np.nan, index=pivot.index)

    if contra_parts:
        I_contra = pd.concat(contra_parts, axis=1).mean(axis=1)
    else:
        I_contra = pd.Series(np.nan, index=pivot.index)

    result = pd.DataFrame({
        "I_ipsi": I_ipsi.values,
        "I_contra": I_contra.values,
    }, index=pivot.index).reset_index()

    result = result.rename(columns={
        "seed_glom": "seed_glomerulus",
        "target_type_base": "target_type",
    })

    n_seeds = df["seed_glom"].nunique()
    n_targets = df["target_type_base"].nunique()
    logger.info(
        "fold_ipsi_contra: %d seeds × %d target types → %d rows",
        n_seeds,
        n_targets,
        len(result),
    )

    return result


def _safe_strip_side(type_str: Any) -> str:
    """Strip side suffix from a type string, returning the original on failure."""
    if not isinstance(type_str, str):
        return str(type_str)
    stripped = strip_side_suffix(type_str)
    return stripped if stripped else type_str


# ---------------------------------------------------------------------------
# 3. Lateralization index
# ---------------------------------------------------------------------------


def compute_li(
    df: pd.DataFrame,
    noise_floor: float,
) -> pd.DataFrame:
    """Compute lateralization index LI = (I_ipsi − I_contra) / (I_ipsi + I_contra).

    Where ``I_ipsi + I_contra < noise_floor`` the LI is set to ``NaN``
    and the row is flagged ``above_floor = False``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``I_ipsi`` and ``I_contra`` (output of
        :func:`fold_ipsi_contra`).
    noise_floor : float
        Minimum denominator magnitude.  Rows whose summed influence falls
        below this value are masked.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with two additional columns: ``LI`` (float) and
        ``above_floor`` (bool).
    """
    result = df.copy()
    total = result["I_ipsi"] + result["I_contra"]

    above = total >= noise_floor
    result["above_floor"] = above
    result["LI"] = np.where(
        above,
        (result["I_ipsi"] - result["I_contra"]) / total,
        np.nan,
    )

    # --- report masking ---
    n_total = len(result)
    n_masked = int((~above).sum())
    logger.info(
        "LI masking: %d / %d rows below noise floor (%.2f%%)",
        n_masked,
        n_total,
        100.0 * n_masked / max(n_total, 1),
    )
    # Per-hop breakdown (only if 'hop' column exists)
    if "hop" in result.columns:
        for hop in sorted(result["hop"].unique()):
            hop_mask = result["hop"] == hop
            n_hop = int(hop_mask.sum())
            n_masked_hop = int((~above & hop_mask).sum())
            if n_hop:
                logger.debug(
                    "  hop %2d: %d / %d masked (%.1f%%)",
                    hop,
                    n_masked_hop,
                    n_hop,
                    100.0 * n_masked_hop / n_hop,
                )

    return result


# ---------------------------------------------------------------------------
# 4. Noise floor estimation
# ---------------------------------------------------------------------------


def compute_noise_floor(
    neurons: pd.DataFrame,
    edges: pd.DataFrame,
) -> float:
    """Estimate the reconstruction noise floor from L/R asymmetry.

    Extracts the L→L and R→R induced subgraphs, matches homologous
    (pre_type, post_type) pairs, and computes the distribution of
    relative weight differences::

        |w_LL − w_RR| / (w_LL + w_RR)

    The noise floor is the *p*-th percentile of this distribution,
    where *p* defaults to 95 (set in ``config.lateralization``).

    Parameters
    ----------
    neurons : pd.DataFrame
        Indexed by ``neuron_id``.  Must have columns ``cell_type``, ``side``.
    edges : pd.DataFrame
        Columns: ``pre_id``, ``post_id``, ``syn_count``.

    Returns
    -------
    float
        The 95th percentile of the relative weight-difference distribution.
        If fewer than 10 homologous pairs are found the function warns and
        returns ``0.0``.
    """
    # --- Build neuron-side and neuron-type lookups ---
    neuron_side: Dict[Any, str] = {}
    neuron_type: Dict[Any, str] = {}
    for nid, row in neurons.iterrows():
        neuron_side[nid] = str(row["side"])
        neuron_type[nid] = str(row["cell_type"])

    # --- Filter edges to same-side pairs ---
    edges_copy = edges.copy()
    edges_copy["pre_side"] = edges_copy["pre_id"].map(neuron_side)
    edges_copy["post_side"] = edges_copy["post_id"].map(neuron_side)
    edges_copy["pre_type"] = edges_copy["pre_id"].map(neuron_type)
    edges_copy["post_type"] = edges_copy["post_id"].map(neuron_type)

    # Drop edges with missing side/type info
    edges_copy = edges_copy.dropna(subset=["pre_side", "post_side", "pre_type", "post_type"])

    ll_edges = edges_copy[
        (edges_copy["pre_side"] == "L") & (edges_copy["post_side"] == "L")
    ]
    rr_edges = edges_copy[
        (edges_copy["pre_side"] == "R") & (edges_copy["post_side"] == "R")
    ]

    # --- Aggregate mean syn_count per (pre_type, post_type) per side ---
    def _agg_mean(edf: pd.DataFrame) -> pd.DataFrame:
        """Mean syn_count per (pre_type, post_type)."""
        if edf.empty:
            return pd.DataFrame(columns=["pre_type", "post_type", "weight"])
        grouped = (
            edf.groupby(["pre_type", "post_type"], observed=True)["syn_count"]
            .mean()
            .reset_index()
        )
        grouped.rename(columns={"syn_count": "weight"}, inplace=True)
        return grouped

    ll_agg = _agg_mean(ll_edges)
    rr_agg = _agg_mean(rr_edges)

    # --- Match homologous pairs ---
    merged = pd.merge(
        ll_agg,
        rr_agg,
        on=["pre_type", "post_type"],
        how="inner",
        suffixes=("_LL", "_RR"),
    )

    if len(merged) < 10:
        logger.warning(
            "Noise floor: only %d homologous type pairs found (< 10); "
            "returning 0.0",
            len(merged),
        )
        return 0.0

    w_ll = merged["weight_LL"].values.astype(np.float64)
    w_rr = merged["weight_RR"].values.astype(np.float64)

    denom = w_ll + w_rr
    valid = denom > 0
    rel_diff = np.full_like(denom, np.nan)
    rel_diff[valid] = np.abs(w_ll[valid] - w_rr[valid]) / denom[valid]

    rel_diff = rel_diff[~np.isnan(rel_diff)]

    # 95th percentile
    noise = float(np.percentile(rel_diff, 95))

    logger.info(
        "Noise floor: %.6f (95th percentile of |w_LL − w_RR|/(w_LL+w_RR) "
        "over %d homologous type pairs, %d valid pairs)",
        noise,
        len(merged),
        len(rel_diff),
    )

    return noise


# ---------------------------------------------------------------------------
# 5. Exponential decay fit
# ---------------------------------------------------------------------------


def fit_decay(
    df: pd.DataFrame,
    config: Config,
) -> dict:
    """Fit exponential decay Lbar(n) = L₀·exp(−n/λ) to population |LI|.

    For each hop *n*, the population-averaged |LI| is::

        Lbar(n) = Σ_T w_T(n) · |LI(T, n)|  /  Σ_T w_T(n)

    where ``w_T = I_ipsi + I_contra`` and the sum runs over all target
    types (pooled across seed glomeruli).  A weighted least-squares fit of
    ``log Lbar(n) = log L₀ − n/λ`` yields *λ* and *L₀*.

    Bootstrap confidence intervals for *λ* are obtained by resampling
    target types with replacement (``config.lateralization.bootstrap_draws``
    draws).

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compute_li` — must have columns ``target_type``,
        ``hop``, ``I_ipsi``, ``I_contra``, ``LI``, ``above_floor``.
    config : Config
        Analysis configuration.  Uses ``lateralization.bootstrap_draws``
        and ``lateralization.min_r_squared``.

    Returns
    -------
    dict
        Keys: ``lambda``, ``lambda_ci_lo``, ``lambda_ci_hi``, ``L0``,
        ``r2``, ``hops_used``, ``flagged`` (bool — True when R² <
        ``min_r_squared``).
    """
    lat_cfg = config.lateralization

    # --- Pool across seed glomeruli: per (target_type, hop) ---
    pooled = (
        df.groupby(["target_type", "hop"], observed=True)
        .agg(I_ipsi=("I_ipsi", "mean"), I_contra=("I_contra", "mean"))
        .reset_index()
    )
    pooled["w"] = pooled["I_ipsi"] + pooled["I_contra"]
    pooled["LI"] = np.where(
        pooled["w"] > 0,
        (pooled["I_ipsi"] - pooled["I_contra"]) / pooled["w"],
        np.nan,
    )
    pooled["abs_LI"] = np.abs(pooled["LI"])

    hops_all = sorted(pooled["hop"].unique())
    # Exclude hop 0 (seed layer — LI is degenerate)
    hops_fit = [h for h in hops_all if h >= 1]
    if len(hops_fit) < 3:
        raise ValueError(
            f"Need at least 3 hops with h ≥ 1 for decay fit; "
            f"got {len(hops_fit)}"
        )

    # --- Point estimate: Lbar(n) ---
    def _lbar(pooled_df: pd.DataFrame, hops: List[int]) -> Tuple[
        NDArray[np.float64], NDArray[np.float64]
    ]:
        """Return (Lbar array, weights array) for given hops."""
        lbar_vals = []
        weights = []
        for h in hops:
            sub = pooled_df[pooled_df["hop"] == h]
            w_sum = sub["w"].sum()
            if w_sum > 0:
                lb = (sub["w"] * sub["abs_LI"].fillna(0)).sum() / w_sum
            else:
                lb = np.nan
            lbar_vals.append(lb)
            weights.append(w_sum)
        return np.array(lbar_vals, dtype=np.float64), np.array(weights, dtype=np.float64)

    lbar, w_hop = _lbar(pooled, hops_fit)

    # Remove hops where Lbar is NaN or ≤ 0 (can't take log)
    valid_lbar = (~np.isnan(lbar)) & (lbar > 0)
    hops_used = [h for h, v in zip(hops_fit, valid_lbar) if v]
    lbar_valid = lbar[valid_lbar]
    w_valid = w_hop[valid_lbar]

    if len(lbar_valid) < 3:
        raise ValueError(
            f"Only {len(lbar_valid)} hops have valid Lbar > 0; "
            f"need ≥ 3 for fit"
        )

    log_lbar = np.log(lbar_valid)
    x = np.array(hops_used, dtype=np.float64)

    # Weighted least squares: y = a + b*x, where a = log(L0), b = -1/λ
    A = np.column_stack([np.ones_like(x), x])
    W_diag = w_valid / w_valid.sum()  # normalize weights
    W = np.diag(W_diag)

    # (Aᵀ W A)⁻¹ Aᵀ W y
    AWA = A.T @ W @ A
    AWy = A.T @ W @ log_lbar
    try:
        coeffs = np.linalg.solve(AWA, AWy)
    except np.linalg.LinAlgError:
        # Fallback: unweighted
        logger.warning("WLS singular; falling back to unweighted OLS")
        coeffs = np.linalg.lstsq(A, log_lbar, rcond=None)[0]

    a, b = coeffs[0], coeffs[1]
    L0 = float(np.exp(a))
    lam = float(-1.0 / b) if b < 0 else np.inf

    # R²
    y_pred = A @ coeffs
    ss_res = float(np.sum(W_diag * (log_lbar - y_pred) ** 2))
    ss_tot = float(np.sum(W_diag * (log_lbar - np.average(log_lbar, weights=W_diag)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    # --- Bootstrap CI for λ ---
    target_types = pooled["target_type"].unique()
    n_types = len(target_types)
    rng = np.random.default_rng(config.seed)

    lam_boot = []
    for _ in range(lat_cfg.bootstrap_draws):
        # Resample target types with replacement
        resampled_types = rng.choice(target_types, size=n_types, replace=True)
        boot_df = pooled[pooled["target_type"].isin(resampled_types)]

        lbar_b, _ = _lbar(boot_df, hops_fit)
        valid_b = (~np.isnan(lbar_b)) & (lbar_b > 0)
        if valid_b.sum() < 3:
            continue
        log_lbar_b = np.log(lbar_b[valid_b])
        x_b = np.array([h for h, v in zip(hops_fit, valid_b) if v], dtype=np.float64)

        A_b = np.column_stack([np.ones_like(x_b), x_b])
        try:
            coeffs_b = np.linalg.lstsq(A_b, log_lbar_b, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        b_b = coeffs_b[1]
        if b_b < 0:
            lam_boot.append(float(-1.0 / b_b))

    if len(lam_boot) >= 100:
        lam_ci_lo = float(np.percentile(lam_boot, 2.5))
        lam_ci_hi = float(np.percentile(lam_boot, 97.5))
    else:
        logger.warning(
            "Only %d bootstrap samples produced valid λ (need ≥ 100); "
            "CI may be unreliable",
            len(lam_boot),
        )
        lam_ci_lo = float(np.percentile(lam_boot, 2.5)) if lam_boot else np.nan
        lam_ci_hi = float(np.percentile(lam_boot, 97.5)) if lam_boot else np.nan

    flagged = bool(r2 < lat_cfg.min_r_squared) if not np.isnan(r2) else True

    if flagged:
        logger.warning(
            "Decay fit R² = %.4f < min_r_squared = %.3f — FLAGGED",
            r2,
            lat_cfg.min_r_squared,
        )

    logger.info(
        "Decay fit: λ = %.2f [%.2f, %.2f], L₀ = %.4f, R² = %.4f, "
        "hops used = %s",
        lam,
        lam_ci_lo,
        lam_ci_hi,
        L0,
        r2,
        hops_used,
    )

    return {
        "lambda": lam,
        "lambda_ci_lo": lam_ci_lo,
        "lambda_ci_hi": lam_ci_hi,
        "L0": L0,
        "r2": r2,
        "hops_used": hops_used,
        "flagged": flagged,
    }


# ---------------------------------------------------------------------------
# 6. Crossover hop
# ---------------------------------------------------------------------------


def compute_crossover(
    df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Identify the crossover hop where |LI| drops below *threshold* and
    stays below.

    For each target type the function finds the smallest hop *n* ≥ 1 such
    that |LI| < *threshold* **and** |LI| remains below the threshold for
    all subsequent hops.  Types that never cross are right-censored
    (``crossover_hop = NaN``, ``censored = True``).

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compute_li`.  Must have columns ``target_type``,
        ``hop``, and ``LI``.  Rows with ``LI = NaN`` are treated as if
        |LI| is below the threshold (the influence is too faint to measure).
    threshold : float
        The |LI| threshold (e.g. ``config.lateralization.crossover_threshold``).

    Returns
    -------
    pd.DataFrame
        Columns: ``target_type``, ``crossover_hop``, ``censored``.
        One row per target type.
    """
    # Pool across seeds: mean LI per (target_type, hop)
    li_by_type = (
        df.groupby(["target_type", "hop"], observed=True)["LI"]
        .mean()
        .reset_index()
    )

    # NaN LI → treat as below threshold (influence too faint)
    li_by_type["abs_LI"] = np.abs(li_by_type["LI"].values)
    li_by_type["below"] = li_by_type["abs_LI"].fillna(0.0) < threshold

    records: List[dict] = []
    for ttype, grp in li_by_type.groupby("target_type", observed=True):
        grp_sorted = grp.sort_values("hop")
        hops = grp_sorted["hop"].values
        below = grp_sorted["below"].values

        # Start from hop ≥ 1
        crossover_hop: Optional[float] = None
        censored = True
        for i in range(len(hops)):
            h = int(hops[i])
            if h < 1:
                continue
            # Check if below threshold AND stays below for all later hops
            if below[i] and np.all(below[i:]):
                crossover_hop = float(h)
                censored = False
                break

        records.append(
            {
                "target_type": ttype,
                "crossover_hop": crossover_hop,
                "censored": censored,
            }
        )

    result = pd.DataFrame(records)

    n_censored = int(result["censored"].sum())
    n_total = len(result)
    logger.info(
        "Crossover: %d / %d target types censored (%.1f%%)",
        n_censored,
        n_total,
        100.0 * n_censored / max(n_total, 1),
    )

    return result
