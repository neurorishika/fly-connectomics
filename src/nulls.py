"""
Null models for the ORN lateralization decay analysis.

Provides four null models (plus robustness) to assess the statistical
significance and stability of lateralisation indices and decay constants.

Functions
---------
1. ``compute_noise_floor_lr`` – L/R intrinsic asymmetry baseline
2. ``degree_preserving_rewiring`` – configuration-model edge rewiring
3. ``random_seed_null`` – z-score / p-value from random-sensory-seed null
4. ``robustness_sweep`` – parameter-sweep rank-stability matrix
"""

from __future__ import annotations

import itertools
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

from .config import Config
from .graph import build_weight_matrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _side_mask(neurons: pd.DataFrame, side: str) -> np.ndarray:
    """Boolean array for neurons whose ``side`` equals *side*."""
    return (neurons["side"] == side).values


def _induced_subgraph_edges(
    edges: pd.DataFrame,
    pre_side: str,
    post_side: str,
    neurons: pd.DataFrame,
) -> pd.DataFrame:
    """Return edges where pre is on *pre_side* and post is on *post_side*.

    Parameters
    ----------
    edges : pd.DataFrame
        Columns ``pre_id``, ``post_id``, ``syn_count``.
    neurons : pd.DataFrame
        Indexed by ``neuron_id``; must have categorical ``side`` column.
    pre_side, post_side : str
        One of ``'L'``, ``'R'``.

    Returns
    -------
    pd.DataFrame
        Subset of *edges*.
    """
    pre_ids = neurons.index[_side_mask(neurons, pre_side)]
    post_ids = neurons.index[_side_mask(neurons, post_side)]
    mask = edges["pre_id"].isin(pre_ids) & edges["post_id"].isin(post_ids)
    return edges.loc[mask].copy()


def _default_propagate(
    W: csr_matrix,
    seed_indices: np.ndarray,
    alpha: float,
    n_hops: int,
) -> np.ndarray:
    """Iterative random-walk propagation.

    .. math::

        v_{k+1} = \\alpha W v_k + (1-\\alpha) s

    where *s* is a uniform distribution over *seed_indices*.

    Parameters
    ----------
    W : csr_matrix
        Column-stochastic weight matrix (shape ``n×n``).
    seed_indices : np.ndarray
        Integer indices of seed neurons.
    alpha : float
        Teleportation parameter (must satisfy ``α·ρ(W) < 1``).
    n_hops : int
        Number of propagation steps.

    Returns
    -------
    v : np.ndarray
        Influence vector of length *n* after *n_hops* steps.
    """
    n = W.shape[0]
    s = np.zeros(n, dtype=W.dtype)
    if len(seed_indices) > 0:
        s[seed_indices] = 1.0 / len(seed_indices)
    v = s.copy()
    for _ in range(n_hops):
        v = alpha * (W @ v) + (1.0 - alpha) * s
    return v


def _homologous_pairs(
    neurons: pd.DataFrame,
) -> List[Tuple[list, list]]:
    """Find homologous neuron groups across L/R.

    Groups neurons by ``cell_type`` and returns, for each type that has
    at least one neuron on each side, a tuple ``(indices_L, indices_R)``
    where each element is a list of positional indices.

    Parameters
    ----------
    neurons : pd.DataFrame
        Must have columns ``cell_type``, ``side`` and be indexed by
        positional index (0..n-1).

    Returns
    -------
    pairs : list of (list, list)
        Each pair is ``([idx_L, ...], [idx_R, ...])``.
    """
    # Ensure we're working with a positional index
    df = neurons.copy()
    if df.index.name != "positional":
        df = df.reset_index(drop=True)

    pairs = []
    for ctype, group in df.groupby("cell_type"):
        left = group.index[group["side"] == "L"].tolist()
        right = group.index[group["side"] == "R"].tolist()
        if left and right:
            pairs.append((left, right))
    return pairs


# ---------------------------------------------------------------------------
# 1. Noise floor via L/R intrinsic asymmetry
# ---------------------------------------------------------------------------

def compute_noise_floor_lr(
    neurons: pd.DataFrame,
    edges: pd.DataFrame,
    config: Config,
    *,
    propagate_fn: Optional[Callable] = None,
    alpha: Optional[float] = None,
    n_hops: Optional[int] = None,
) -> dict:
    """Estimate the intrinsic L/R reconstruction noise floor.

    Two complementary estimates are computed:

    **Structural** — for every homologous type-pair (same *cell_type*,
    different *side*), the relative difference in direct edge weights to
    homologous post-synaptic targets is measured.  The distribution of
    ``|w_L - w_R| / (w_L + w_R + ε)`` quantifies weight-level asymmetry.

    **Propagation** — the L→L and R→R subgraphs are built, and random-walk
    propagation is run from each homologous seed set.  The absolute
    difference in stationary influence on every homologous target pair
    forms the *effective* noise distribution.

    Both distributions are returned; the 95th percentile of the structural
    (and, if available, propagation) distribution is reported as the noise
    floor.

    Parameters
    ----------
    neurons : pd.DataFrame
        Indexed by ``neuron_id``; columns ``side``, ``cell_type``.
    edges : pd.DataFrame
        Columns ``pre_id``, ``post_id``, ``syn_count``.
    config : Config
    propagate_fn : callable, optional
        Signature ``(W, seed_indices, alpha, n_hops) -> np.ndarray``.
        If ``None``, a built-in iterative random walk is used.
    alpha : float, optional
        Teleportation parameter (default ``config.propagation.default_alpha``).
    n_hops : int, optional
        Propagation steps (default ``config.propagation.n_hops``).

    Returns
    -------
    dict
        Keys:
        - ``noise_floor`` (float): 95th percentile of structural differences.
        - ``noise_floor_propagation`` (float or None)
        - ``structural_diffs`` (np.ndarray)
        - ``propagation_diffs`` (np.ndarray or None)
        - ``percentile`` (float): percentile used.
        - ``n_homologous_pairs`` (int)
        - ``n_homologous_targets`` (int)
    """
    alpha = alpha if alpha is not None else config.propagation.default_alpha
    n_hops = n_hops if n_hops is not None else config.propagation.n_hops
    percentile = config.lateralization.noise_floor_percentile

    # ---- positional indexing ----
    neuron_ids = neurons.index.values
    pos_neurons = neurons.reset_index(drop=True)
    pos_neurons.index.name = "positional"

    # ---- homologous pairs ----
    pairs = _homologous_pairs(pos_neurons)
    logger.info(f"Found {len(pairs):,} homologous type-pairs (L/R)")

    if not pairs:
        return {
            "noise_floor": np.nan,
            "noise_floor_propagation": None,
            "structural_diffs": np.array([]),
            "propagation_diffs": None,
            "percentile": percentile,
            "n_homologous_pairs": 0,
            "n_homologous_targets": 0,
            "warning": "No homologous L/R pairs found",
        }

    # ---- structural differences (direct edge weights) ----
    structural_diffs: List[float] = []
    n_target_pairs = 0

    for left_idxs, right_idxs in pairs:
        # Outgoing edges from the left neurons
        left_ids = set(neuron_ids[left_idxs])
        right_ids = set(neuron_ids[right_idxs])

        # Get target neurons and their sides
        for t_ctype, t_group in pos_neurons.groupby("cell_type"):
            t_left = t_group.index[t_group["side"] == "L"].tolist()
            t_right = t_group.index[t_group["side"] == "R"].tolist()
            if not t_left or not t_right:
                continue
            # Use the first neuron for each target side
            tL_id = neuron_ids[t_left[0]]
            tR_id = neuron_ids[t_right[0]]

            # Summed input from left seeds → left target
            wL = float(
                edges.loc[
                    edges["pre_id"].isin(left_ids) & (edges["post_id"] == tL_id),
                    "syn_count",
                ].sum()
            )
            # Summed input from right seeds → right target
            wR = float(
                edges.loc[
                    edges["pre_id"].isin(right_ids) & (edges["post_id"] == tR_id),
                    "syn_count",
                ].sum()
            )

            denom = wL + wR
            if denom > 0:
                structural_diffs.append(abs(wL - wR) / denom)
            n_target_pairs += 1

    structural_diffs_arr = np.array(structural_diffs, dtype=np.float64)
    noise_floor = (
        float(np.percentile(structural_diffs_arr, percentile))
        if len(structural_diffs_arr) > 0
        else np.nan
    )

    logger.info(
        f"Structural noise floor ({percentile}th pct): {noise_floor:.4f} "
        f"(n={len(structural_diffs_arr):,} target pairs)"
    )

    # ---- propagation-based differences ----
    propagation_diffs: Optional[np.ndarray] = None
    noise_floor_prop: Optional[float] = None

    try:
        # Build L→L and R→R weight matrices
        syn_threshold = config.graph.syn_threshold
        norm = config.graph.normalization

        # L subgraph
        neurons_L = neurons[neurons["side"] == "L"].copy()
        edges_LL = _induced_subgraph_edges(edges, "L", "L", neurons)
        if len(neurons_L) > 0 and len(edges_LL) > 0:
            W_L, _ = build_weight_matrix(
                neurons_L, edges_LL, config,
                syn_threshold=syn_threshold, normalization=norm,
            )
        else:
            W_L = None

        # R subgraph
        neurons_R = neurons[neurons["side"] == "R"].copy()
        edges_RR = _induced_subgraph_edges(edges, "R", "R", neurons)
        if len(neurons_R) > 0 and len(edges_RR) > 0:
            W_R, _ = build_weight_matrix(
                neurons_R, edges_RR, config,
                syn_threshold=syn_threshold, normalization=norm,
            )
        else:
            W_R = None

        if W_L is not None and W_R is not None:
            _prop = propagate_fn if propagate_fn is not None else _default_propagate
            prop_diffs_list: List[float] = []

            # Build side-specific id→idx maps
            l_ids = neurons_L.index.values
            r_ids = neurons_R.index.values
            l_to_idx = {nid: i for i, nid in enumerate(l_ids)}
            r_to_idx = {nid: i for i, nid in enumerate(r_ids)}

            for left_idxs, right_idxs in pairs:
                # Map to L/R subgraph indices
                seed_L = np.array(
                    [l_to_idx[neuron_ids[i]]
                     for i in left_idxs if neuron_ids[i] in l_to_idx],
                    dtype=np.int32,
                )
                seed_R = np.array(
                    [r_to_idx[neuron_ids[i]]
                     for i in right_idxs if neuron_ids[i] in r_to_idx],
                    dtype=np.int32,
                )
                if len(seed_L) == 0 or len(seed_R) == 0:
                    continue

                vL = _prop(W_L, seed_L, alpha, n_hops)
                vR = _prop(W_R, seed_R, alpha, n_hops)

                # Compare influence on homologous targets
                for t_ctype, t_group in pos_neurons.groupby("cell_type"):
                    t_left_ids = t_group.index[t_group["side"] == "L"].tolist()
                    t_right_ids = t_group.index[t_group["side"] == "R"].tolist()
                    if not t_left_ids or not t_right_ids:
                        continue

                    tL_nid = neuron_ids[t_left_ids[0]]
                    tR_nid = neuron_ids[t_right_ids[0]]

                    if tL_nid in l_to_idx and tR_nid in r_to_idx:
                        inf_L = float(vL[l_to_idx[tL_nid]])
                        inf_R = float(vR[r_to_idx[tR_nid]])
                        prop_diffs_list.append(abs(inf_L - inf_R))

            if prop_diffs_list:
                propagation_diffs = np.array(prop_diffs_list, dtype=np.float64)
                noise_floor_prop = float(
                    np.percentile(propagation_diffs, percentile)
                )
                logger.info(
                    f"Propagation noise floor ({percentile}th pct): "
                    f"{noise_floor_prop:.6f} (n={len(prop_diffs_list):,})"
                )
    except Exception as exc:
        logger.warning(f"Propagation-based noise floor failed: {exc}")

    return {
        "noise_floor": noise_floor,
        "noise_floor_propagation": noise_floor_prop,
        "structural_diffs": structural_diffs_arr,
        "propagation_diffs": propagation_diffs,
        "percentile": percentile,
        "n_homologous_pairs": len(pairs),
        "n_homologous_targets": n_target_pairs,
    }


# ---------------------------------------------------------------------------
# 2. Degree-preserving rewiring (configuration model)
# ---------------------------------------------------------------------------

def degree_preserving_rewiring(
    edges: pd.DataFrame,
    neurons: pd.DataFrame,
    n_rewirings: int,
    preserve_blocks: bool,
    rng: np.random.Generator,
) -> List[pd.DataFrame]:
    """Generate rewired edge sets via double-edge swaps (XSwap).

    Preserves the in-degree and out-degree of every neuron exactly.
    When *preserve_blocks* is ``True``, edges are partitioned into the
    four laterality blocks (L→L, L→R, R→L, R→R) and rewiring is applied
    within each block independently, preserving block-level densities.

    Algorithm
    ---------
    For each block, ``10 × n_edges`` candidate swaps are attempted.
    A swap picks two distinct edges ``(a→b)`` and ``(c→d)`` and replaces
    them with ``(a→d)`` and ``(c→b)``, provided neither replacement edge
    already exists and ``a≠c``, ``b≠d``.

    Parameters
    ----------
    edges : pd.DataFrame
        Columns ``pre_id``, ``post_id``, ``syn_count``.
    neurons : pd.DataFrame
        Indexed by ``neuron_id``; must have categorical ``side`` column.
    n_rewirings : int
        Number of independent rewired edge sets to return.
    preserve_blocks : bool
        If ``True``, rewire within each laterality block separately.
    rng : np.random.Generator
        Seeded random generator for reproducibility.

    Returns
    -------
    list of pd.DataFrame
        *n_rewirings* edge DataFrames, each with the same columns and
        ``syn_count`` values as the input (only topology changes).
    """
    # Determine blocks
    sides = ["L", "R"]
    if preserve_blocks:
        block_specs = [
            (pre, post)
            for pre in sides
            for post in sides
        ]
    else:
        block_specs = [(None, None)]  # single block = all edges

    # Pre-extract edges per block as list-of-tuples for fast manipulation
    # We keep syn_count attached so we can reconstruct DataFrames.

    def _extract_block(pre_side, post_side):
        if pre_side is None:
            sub = edges.copy()
        else:
            sub = _induced_subgraph_edges(edges, pre_side, post_side, neurons)
        # Return as list of (pre_id, post_id, syn_count)
        return list(
            zip(sub["pre_id"].values, sub["post_id"].values, sub["syn_count"].values)
        )

    blocks_raw = {spec: _extract_block(*spec) for spec in block_specs}

    logger.info(
        "Degree-preserving rewiring: %d blocks, %d rewiring(s)",
        len(block_specs), n_rewirings,
    )
    for spec, elist in blocks_raw.items():
        logger.info("  Block %s: %d edges", spec, len(elist))

    rewired_list: List[pd.DataFrame] = []

    for i_rew in range(n_rewirings):
        all_rewired: List[Tuple] = []

        for spec, elist in blocks_raw.items():
            if len(elist) < 2:
                all_rewired.extend(elist)
                continue

            # Work with mutable list of [pre, post, syn]
            working = [[e[0], e[1], e[2]] for e in elist]
            n_edges = len(working)
            n_swaps = 10 * n_edges

            # Build fast lookup for edge existence (pre_id, post_id)
            edge_set = set((e[0], e[1]) for e in working)

            swaps_done = 0
            for _ in range(n_swaps):
                # Pick two distinct edges
                i, j = rng.integers(0, n_edges, size=2)
                if i == j:
                    continue

                a, b, syn_ab = working[i]
                c, d, syn_cd = working[j]

                # Degenerate check
                if a == c or b == d:
                    continue

                # Proposed new edges
                if (a, d) in edge_set or (c, b) in edge_set:
                    continue

                # Perform swap
                edge_set.remove((a, b))
                edge_set.remove((c, d))
                edge_set.add((a, d))
                edge_set.add((c, b))

                working[i] = [a, d, syn_ab]
                working[j] = [c, b, syn_cd]
                swaps_done += 1

            logger.debug(
                "  Rewiring %d/%d block %s: %d/%d swaps accepted",
                i_rew + 1, n_rewirings, spec, swaps_done, n_swaps,
            )
            all_rewired.extend(tuple(e) for e in working)

        df_rew = pd.DataFrame(
            all_rewired,
            columns=["pre_id", "post_id", "syn_count"],
        )
        df_rew["syn_count"] = df_rew["syn_count"].astype(np.int32)
        rewired_list.append(df_rew)

    return rewired_list


# ---------------------------------------------------------------------------
# 3. Random-seed null model
# ---------------------------------------------------------------------------

def random_seed_null(
    lambda_obs: float,
    lambda_null_dist: List[float],
) -> dict:
    """Assess significance of an observed λ against a null distribution.

    The null distribution is typically obtained by re-running the
    lateralisation pipeline with size-matched *random* sensory neuron
    seeds (instead of the true ORN seeds) many times.

    Parameters
    ----------
    lambda_obs : float
        Observed lateralisation decay constant for a glomerulus.
    lambda_null_dist : list of float
        Null λ values (from random seeds).

    Returns
    -------
    dict
        Keys:
        - ``z_score`` (float): ``(λ_obs - μ_null) / σ_null``
        - ``p_value`` (float): two-sided empirical p-value
        - ``null_mean`` (float)
        - ``null_std`` (float)
        - ``n_null`` (int)
        - ``lambda_obs`` (float)
    """
    null_arr = np.asarray(lambda_null_dist, dtype=np.float64)
    if len(null_arr) == 0:
        return {
            "z_score": np.nan,
            "p_value": np.nan,
            "null_mean": np.nan,
            "null_std": np.nan,
            "n_null": 0,
            "lambda_obs": lambda_obs,
            "warning": "Empty null distribution",
        }

    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr, ddof=1))

    # z-score (guard against zero std)
    if null_std > 0:
        z_score = (lambda_obs - null_mean) / null_std
    else:
        z_score = float("inf") if lambda_obs != null_mean else 0.0

    # Two-sided empirical p-value
    n_null = len(null_arr)
    n_extreme = int(np.sum(np.abs(null_arr - null_mean) >= np.abs(lambda_obs - null_mean)))
    p_value = n_extreme / n_null
    # Floor to 1/n_null to avoid p=0 when obs is more extreme than all nulls
    p_value = max(p_value, 1.0 / n_null)

    return {
        "z_score": z_score,
        "p_value": p_value,
        "null_mean": null_mean,
        "null_std": null_std,
        "n_null": n_null,
        "lambda_obs": lambda_obs,
    }


# ---------------------------------------------------------------------------
# 4. Robustness sweep
# ---------------------------------------------------------------------------

def robustness_sweep(
    datasets: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    config: Config,
    *,
    pipeline_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """Assess λ-rank stability under parameter variation.

    For every combination of ``syn_threshold`` × ``normalization`` ×
    ``alpha`` in the robustness config, the full pipeline is re-run (or a
    user-supplied *pipeline_fn*) and a vector of λ values (one per
    glomerulus) is collected.  Pairwise Spearman rank correlations are
    computed between every parameter combination, yielding a square
    matrix.

    If the minimum off-diagonal correlation is below 0.8, a prominent
    warning is logged.

    Parameters
    ----------
    datasets : dict
        ``{name: (neurons_df, edges_df)}``.  The first dataset is used
        for the sweep (multi-dataset averaging is deferred to the
        pipeline).
    config : Config
    pipeline_fn : callable, optional
        ``(neurons, edges, syn_threshold, normalization, alpha) -> lambda_vec``
        where ``lambda_vec`` is a 1-D array or dict of λ values keyed by
        glomerulus.  If ``None``, a simple built-in pipeline that only
        builds the weight matrix is used as a placeholder (the returned
        matrix will contain NaNs for λ-dependent entries — a warning is
        issued).

    Returns
    -------
    pd.DataFrame
        Square symmetric DataFrame.  Row/column labels are
        ``"syn=N_norm=name_alpha=0.X"`` strings.
        Values are Spearman ρ (rank correlation).
    """
    if pipeline_fn is None:
        logger.warning(
            "No pipeline_fn provided to robustness_sweep; λ cannot be "
            "computed. Returning a placeholder correlation matrix "
            "(all NaNs). Provide a callable with signature "
            "(neurons, edges, syn_threshold, normalization, alpha) -> "
            "lambda_vec."
        )

    syn_sweep = config.robustness.syn_threshold_sweep
    norm_sweep = config.robustness.normalization_variants
    alpha_sweep = config.robustness.alpha_sweep

    param_grid = list(itertools.product(syn_sweep, norm_sweep, alpha_sweep))
    logger.info(
        "Robustness sweep: %d parameter combinations "
        "(syn_threshold × normalization × alpha)",
        len(param_grid),
    )

    # Use the first dataset
    name, (neurons, edges) = next(iter(datasets.items()))
    logger.info("Using dataset '%s' for robustness sweep", name)

    # Collect λ vectors
    lambda_vectors: Dict[str, np.ndarray] = {}

    for syn_th, norm, alpha in param_grid:
        label = f"syn={syn_th}_norm={norm}_alpha={alpha}"

        if pipeline_fn is None:
            # Placeholder: store NaN vector
            lambda_vectors[label] = np.array([np.nan])
            continue

        try:
            lam = pipeline_fn(neurons, edges, syn_th, norm, alpha)
            if isinstance(lam, dict):
                # Assume dict keyed by glomerulus → λ
                lam_arr = np.array(list(lam.values()), dtype=np.float64)
            else:
                lam_arr = np.asarray(lam, dtype=np.float64)
            lambda_vectors[label] = lam_arr
        except Exception as exc:
            logger.warning("Pipeline failed for %s: %s", label, exc)
            lambda_vectors[label] = np.array([np.nan])

    # Build correlation matrix
    n_combos = len(param_grid)
    labels = [
        f"syn={syn_th}_norm={norm}_alpha={alpha}"
        for syn_th, norm, alpha in param_grid
    ]
    corr_mat = np.full((n_combos, n_combos), np.nan)

    for i, j in itertools.combinations_with_replacement(range(n_combos), 2):
        vi = lambda_vectors.get(labels[i], np.array([np.nan]))
        vj = lambda_vectors.get(labels[j], np.array([np.nan]))

        # Align on common glomeruli (for dict-based lambdas, alignment
        # is handled by the pipeline; for arrays we require same length)
        if len(vi) != len(vj) or len(vi) < 3:
            corr_mat[i, j] = np.nan
            corr_mat[j, i] = np.nan
            continue

        # Drop pairs where either is NaN
        mask = np.isfinite(vi) & np.isfinite(vj)
        if mask.sum() < 3:
            corr_mat[i, j] = np.nan
            corr_mat[j, i] = np.nan
            continue

        rho, _ = spearmanr(vi[mask], vj[mask])
        corr_mat[i, j] = rho
        corr_mat[j, i] = rho

    corr_df = pd.DataFrame(corr_mat, index=labels, columns=labels)

    # ---- stability check ----
    off_diag = corr_mat[~np.eye(n_combos, dtype=bool)]
    off_diag_finite = off_diag[np.isfinite(off_diag)]
    if len(off_diag_finite) > 0:
        min_corr = float(np.nanmin(off_diag_finite))
        mean_corr = float(np.nanmean(off_diag_finite))
        logger.info(
            "Rank stability: min ρ=%.4f, mean ρ=%.4f (off-diagonal)",
            min_corr, mean_corr,
        )
        if min_corr < 0.8:
            logger.warning(
                "⚠️  RANK INSTABILITY DETECTED: minimum off-diagonal "
                "Spearman ρ = %.4f < 0.8.  λ rankings are sensitive to "
                "parameter choices.  Report prominently and consider "
                "using the median/summary ranking.",
                min_corr,
            )
    else:
        logger.warning("No valid off-diagonal correlations to assess stability")

    return corr_df
