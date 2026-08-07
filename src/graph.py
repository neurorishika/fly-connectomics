"""Graph construction from connectome edges.

Builds a sparse column-stochastic weight matrix W where W[j,i] is the
input fraction of neuron j coming from neuron i.

W[j,i] = syn(i -> j) / sum_k syn(k -> j)

Denominator = total input onto j, computed BEFORE thresholding.
Thresholding is applied to the numerator only.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import eigs

from .config import Config

logger = logging.getLogger(__name__)


def build_weight_matrix(
    neurons: pd.DataFrame,
    edges: pd.DataFrame,
    config: Config,
    *,
    syn_threshold: Optional[int] = None,
    normalization: Optional[str] = None,
    signed: Optional[bool] = None,
    dtype: Optional[str] = None,
) -> Tuple[csr_matrix, dict]:
    """Build the column-stochastic weight matrix W.

    Parameters
    ----------
    neurons : pd.DataFrame
        Indexed by neuron_id. Must have columns ``nt_type`` (for signed mode).
    edges : pd.DataFrame
        Columns: ``pre_id``, ``post_id``, ``syn_count``.
    config : Config
        Analysis configuration.
    syn_threshold : int, optional
        Minimum synapses for an edge. Defaults to config.graph.syn_threshold.
    normalization : str, optional
        One of 'input_fraction', 'raw', 'log_syn', 'sqrt_syn'.
        Defaults to config.graph.normalization.
    signed : bool, optional
        If True, multiply presynaptic columns by +1 (ACh) or -1 (GABA/Glu).
        Defaults to config.graph.signed.
    dtype : str, optional
        Defaults to config.graph.dtype.

    Returns
    -------
    W : csr_matrix
        Square sparse matrix, shape (n_neurons, n_neurons), where
        W[j, i] = normalized synaptic weight from i to j.
    info : dict
        Diagnostic keys: n_neurons, n_edges_raw, n_edges_filtered,
        spectral_radius, n_zero_input, n_cholinergic, n_gabaergic,
        n_glutamatergic, n_unknown_nt.
    """
    # --- defaults from config ---
    syn_threshold = syn_threshold if syn_threshold is not None else config.graph.syn_threshold
    normalization = normalization or config.graph.normalization
    signed = signed if signed is not None else config.graph.signed
    dtype_str = dtype or config.graph.dtype
    dt = np.dtype(dtype_str)

    # Build neuron_id -> matrix index mapping
    neuron_ids = neurons.index.values
    id_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
    n = len(neuron_ids)

    # --- filter edges by threshold ---
    edges_f = edges[edges["syn_count"] >= syn_threshold].copy()
    n_raw = len(edges)
    n_filt = len(edges_f)
    logger.info(
        f"Edge filter: {n_filt:,}/{n_raw:,} edges kept "
        f"(syn_threshold={syn_threshold}, {n_filt/max(n_raw,1):.1%})"
    )

    # Map to matrix indices
    pre_idx = edges_f["pre_id"].map(id_to_idx).values
    post_idx = edges_f["post_id"].map(id_to_idx).values
    syn = edges_f["syn_count"].values.astype(dt)

    # Drop edges where either end is not in the neuron table
    valid = (~np.isnan(pre_idx)) & (~np.isnan(post_idx))
    pre_idx = pre_idx[valid].astype(np.int32)
    post_idx = post_idx[valid].astype(np.int32)
    syn = syn[valid]
    n_mapped = len(syn)
    logger.info(f"Mapped edges: {n_mapped:,} (dropped {n_filt - n_mapped:,} orphan edges)")

    # --- signed mode: apply NT multiplier ---
    sign_vector = None
    if signed:
        sign_vector = _build_sign_vector(neurons, id_to_idx, dt)
        syn = syn * sign_vector[pre_idx]  # multiply by presynaptic sign

    # --- construct raw sparse matrix (COO -> CSR) ---
    # W[j, i] = raw weight from i to j, so we use (row=post, col=pre)
    W_raw = csr_matrix(
        (syn, (post_idx, pre_idx)),
        shape=(n, n),
        dtype=dt,
    )

    # --- compute total input per neuron (column sums of W_raw treated as row sums of W^T) ---
    # Actually: total input onto neuron j = sum over all presynaptic i of raw_weight(i->j)
    # In our matrix W[j,i] = raw_weight(i->j), so total input onto j = sum_i W[j,i]
    # = row sum of W
    total_input = np.asarray(W_raw.sum(axis=1)).flatten()

    # Guard against divide-by-zero
    n_zero_input = int((total_input == 0).sum())
    if n_zero_input > 0:
        logger.info(
            f"Neurons with zero total input: {n_zero_input:,}/{n:,} "
            f"({n_zero_input/n:.1%})"
        )
    total_input_safe = np.where(total_input > 0, total_input, 1.0)

    # --- apply normalization ---
    if normalization == "input_fraction":
        # W[j,i] = raw(i->j) / total_input(j)
        # Equivalent to dividing each row j by its sum
        inv_input = 1.0 / total_input_safe
        D_inv = sparse.diags(inv_input, dtype=dt)
        W = D_inv @ W_raw
    elif normalization == "raw":
        W = W_raw.astype(dt)
        W.data[:] = W.data.astype(dt)  # ensure dtype
    elif normalization == "log_syn":
        W_data = np.log1p(W_raw.data)
        W = csr_matrix((W_data, W_raw.indices, W_raw.indptr), shape=W_raw.shape, dtype=dt)
        # Re-normalize rows
        total = np.asarray(W.sum(axis=1)).flatten()
        total_safe = np.where(total > 0, total, 1.0)
        D_inv = sparse.diags(1.0 / total_safe, dtype=dt)
        W = D_inv @ W
    elif normalization == "sqrt_syn":
        W_data = np.sqrt(W_raw.data)
        W = csr_matrix((W_data, W_raw.indices, W_raw.indptr), shape=W_raw.shape, dtype=dt)
        total = np.asarray(W.sum(axis=1)).flatten()
        total_safe = np.where(total > 0, total, 1.0)
        D_inv = sparse.diags(1.0 / total_safe, dtype=dt)
        W = D_inv @ W
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    # --- assertions ---
    assert W.shape == (n, n), f"W shape {W.shape} != ({n},{n})"
    assert W.nnz > 0, "W is empty"
    if normalization == "input_fraction":
        assert float(W.data.min()) >= 0, f"W has negative entries: min={W.data.min()}"
        # Row sums should be ≈1 (except zero-input rows)
        row_sums = np.asarray(W.sum(axis=1)).flatten()
        nonzero_rows = row_sums > 0
        if nonzero_rows.any():
            max_dev = np.max(np.abs(row_sums[nonzero_rows] - 1.0))
            assert max_dev < 1e-5, f"Row sums deviate from 1.0 by up to {max_dev}"

    # --- spectral diagnostics ---
    rho, converged = _spectral_radius(W)
    logger.info(f"Spectral radius ρ(W) = {rho:.6f} (converged={converged})")

    # --- NT counts (for signed mode) ---
    nt_counts = _count_nt_types(neurons)
    n_unknown = nt_counts.get(None, 0) + nt_counts.get("unknown", 0)

    info = {
        "n_neurons": n,
        "n_edges_raw": n_raw,
        "n_edges_filtered": n_filt,
        "n_edges_mapped": n_mapped,
        "spectral_radius": float(rho),
        "n_zero_input": n_zero_input,
        "n_cholinergic": nt_counts.get("ACh", 0) + nt_counts.get("acetylcholine", 0),
        "n_gabaergic": nt_counts.get("GABA", 0) + nt_counts.get("gaba", 0),
        "n_glutamatergic": nt_counts.get("Glu", 0) + nt_counts.get("glutamate", 0),
        "n_unknown_nt": n_unknown,
        "dtype": dtype_str,
        "normalization": normalization,
        "syn_threshold": syn_threshold,
        "signed": signed,
    }

    return W, info


def _build_sign_vector(
    neurons: pd.DataFrame,
    id_to_idx: dict,
    dtype: np.dtype,
) -> np.ndarray:
    """Build a per-neuron sign vector: +1 excitatory, -1 inhibitory.

    ACh → +1, GABA → -1, Glu → -1, all others → +1.
    Neurons without NT prediction default to +1.
    """
    n = len(id_to_idx)
    signs = np.ones(n, dtype=dtype)

    nt_col = neurons.get("nt_type", pd.Series(index=neurons.index))
    excitatory = {"ACh", "acetylcholine", "DA", "dopamine", "5-HT", "serotonin",
                  "OA", "octopamine", "HA", "histamine"}
    inhibitory = {"GABA", "gaba", "Glu", "glutamate"}

    for nid, idx in id_to_idx.items():
        nt = nt_col.get(nid, None)
        if isinstance(nt, float) and np.isnan(nt):
            nt = None
        if nt in inhibitory:
            signs[idx] = -1.0
        elif nt in excitatory:
            signs[idx] = 1.0
        # else: unknown → +1 (default)

    n_inhib = int((signs < 0).sum())
    n_exc = int((signs > 0).sum())
    logger.info(f"Sign vector: {n_exc:,} excitatory, {n_inhib:,} inhibitory")
    return signs


def _spectral_radius(W: csr_matrix) -> Tuple[float, bool]:
    """Compute leading eigenvalue magnitude (spectral radius).

    Tries ARPACK (eigs); falls back to power iteration on failure.
    """
    try:
        vals, _ = eigs(W, k=1, which="LM", maxiter=500, tol=1e-6)
        rho = float(np.abs(vals[0]))
        converged = True
    except Exception:
        # Fallback: power iteration
        logger.warning("eigs failed, falling back to power iteration")
        rho = _power_iteration(W, n_iter=200)
        converged = False
    return rho, converged


def _power_iteration(W: csr_matrix, n_iter: int = 200) -> float:
    """Estimate spectral radius via power iteration."""
    n = W.shape[0]
    v = np.random.randn(n).astype(W.dtype)
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        v_new = W @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-15:
            break
        v = v_new / norm
    # Rayleigh quotient
    return float(np.abs(v @ (W @ v)))


def _count_nt_types(neurons: pd.DataFrame) -> dict:
    """Count neurons per neurotransmitter type."""
    if "nt_type" not in neurons.columns:
        return {}
    counts = neurons["nt_type"].value_counts(dropna=False)
    return counts.to_dict()


def validate_alpha(alpha: float, rho: float) -> None:
    """Assert α·ρ(W) < 1 and print effective decay per hop."""
    product = alpha * rho
    assert product < 1.0, (
        f"α·ρ(W) = {alpha}·{rho:.4f} = {product:.4f} ≥ 1.0 — "
        f"propagation will not converge. Reduce α."
    )
    logger.info(
        f"α·ρ(W) = {product:.4f} < 1.0 ✓ "
        f"(effective decay per hop: {product:.4f})"
    )


def ablate_seed_feedback(W: csr_matrix, seed_indices: np.ndarray) -> csr_matrix:
    """Zero the rows corresponding to seed neurons so they cannot receive
    feedback from the network.

    Parameters
    ----------
    W : csr_matrix
        Original weight matrix.
    seed_indices : np.ndarray
        Integer indices of seed neurons to ablate.

    Returns
    -------
    W_ablated : csr_matrix
        Copy of W with seed rows zeroed.
    """
    W_abl = W.copy()
    # Convert to lil for efficient row zeroing
    W_lil = W_abl.tolil()
    W_lil[seed_indices, :] = 0
    W_abl = W_lil.tocsr()
    W_abl.eliminate_zeros()
    logger.info(
        f"Ablated seed feedback: zeroed {len(seed_indices):,} rows "
        f"({len(seed_indices)/W.shape[0]:.1%} of neurons)"
    )
    return W_abl
