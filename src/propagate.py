"""Per-hop random-walk propagation on the connectome weight matrix.

Computes the influence delivered at each synaptic distance (hop) for a given
seed vector or batch of seed vectors. Also provides resolvent verification
against the exact linear solve.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye as speye

from .graph import validate_alpha


def propagate(
    W: csr_matrix,
    seed: np.ndarray,
    n_hops: int,
    alpha: float,
    *,
    spectral_radius: Optional[float] = None,
) -> np.ndarray:
    """Per-hop influence of a single seed vector.

    Parameters
    ----------
    W : csr_matrix
        Square weight matrix, shape (n_neurons, n_neurons).
    seed : np.ndarray
        1-D seed vector, shape (n_neurons,).
    n_hops : int
        Number of synaptic steps to simulate (≥ 0).
    alpha : float
        Attenuation factor per hop.  Must satisfy
        ``alpha * spectral_radius(W) < 1`` for the resolvent to converge.
    spectral_radius : float, optional
        Pre-computed spectral radius of *W*.  When provided,
        :func:`~.graph.validate_alpha` is called to assert
        ``alpha * spectral_radius < 1``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_hops + 1, n_neurons)`` with dtype float32.
        ``result[h]`` = ``alpha**h * (W**h @ seed)`` — the influence
        contributed by paths of exactly *h* synaptic steps.
        ``result[0]`` is the seed itself.
    """
    if spectral_radius is not None:
        validate_alpha(alpha, spectral_radius)

    n = W.shape[0]
    seed = np.asarray(seed, dtype=np.float32)
    if seed.ndim != 1 or seed.shape[0] != n:
        raise ValueError(f"seed must be 1-D with length {n}, got shape {seed.shape}")

    result = np.empty((n_hops + 1, n), dtype=np.float32)
    x = seed.copy()
    result[0] = x

    for h in range(1, n_hops + 1):
        x = alpha * (W @ x)
        # Ensure float32 after sparse matvec (scipy may upcast)
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        result[h] = x

    return result


def propagate_batch(
    W: csr_matrix,
    seed_matrix: np.ndarray,
    n_hops: int,
    alpha: float,
    *,
    spectral_radius: Optional[float] = None,
) -> np.ndarray:
    """Per-hop influence for multiple seed vectors in one pass.

    Parameters
    ----------
    W : csr_matrix
        Square weight matrix, shape (n_neurons, n_neurons).
    seed_matrix : np.ndarray
        2-D array of shape ``(n_neurons, n_seeds)``, where each column is
        an independent seed vector.
    n_hops : int
        Number of synaptic steps to simulate (≥ 0).
    alpha : float
        Attenuation factor per hop.
    spectral_radius : float, optional
        Pre-computed spectral radius of *W*.  When provided,
        :func:`~.graph.validate_alpha` is called to assert
        ``alpha * spectral_radius < 1``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_hops + 1, n_neurons, n_seeds)`` with dtype float32.
        ``result[h, :, k]`` is the per-hop influence for seed *k* at hop *h*.
    """
    if spectral_radius is not None:
        validate_alpha(alpha, spectral_radius)

    n = W.shape[0]
    seed_matrix = np.asarray(seed_matrix, dtype=np.float32)
    if seed_matrix.ndim != 2 or seed_matrix.shape[0] != n:
        raise ValueError(
            f"seed_matrix must be 2-D with first dimension {n}, "
            f"got shape {seed_matrix.shape}"
        )
    n_seeds = seed_matrix.shape[1]

    result = np.empty((n_hops + 1, n, n_seeds), dtype=np.float32)
    X = seed_matrix.copy()
    result[0] = X

    for h in range(1, n_hops + 1):
        X = alpha * (W @ X)
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        result[h] = X

    return result


def cumulative_influence(per_hop: np.ndarray) -> np.ndarray:
    """Sum per-hop influence across all hops (including hop 0).

    Parameters
    ----------
    per_hop : np.ndarray
        Array of shape ``(n_hops + 1, n_neurons)``, as returned by
        :func:`propagate`.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(n_neurons,)`` — the total influence delivered
        by all paths up to *n_hops* steps.
    """
    return np.sum(per_hop, axis=0)


def resolvent_check(
    W: csr_matrix,
    seed: np.ndarray,
    alpha: float,
    n_hops: int,
    tol: float = 1e-5,
) -> bool:
    """Verify that the cumulative random-walk sum matches the resolvent.

    Compares  ``Σ_{h=0}^{n_hops} α^h W^h s``  with the exact solution of
    ``(I - αW) x = s`` obtained via sparse direct solve.

    Parameters
    ----------
    W : csr_matrix
        Square weight matrix.
    seed : np.ndarray
        1-D seed vector, shape (n_neurons,).
    alpha : float
        Attenuation factor.
    n_hops : int
        Number of hops used in the truncated sum.
    tol : float
        Absolute tolerance for the element-wise comparison.

    Returns
    -------
    bool
        True if ``max(abs(cumulative - exact)) < tol``.

    Raises
    ------
    AssertionError
        If the agreement is worse than *tol*.
    """
    n = W.shape[0]
    seed = np.asarray(seed, dtype=np.float32)
    if seed.ndim != 1 or seed.shape[0] != n:
        raise ValueError(f"seed must be 1-D with length {n}, got shape {seed.shape}")

    per_hop = propagate(W, seed, n_hops, alpha)
    cum = cumulative_influence(per_hop)

    # Solve (I - alpha*W) x = seed  →  x = inv(I - alpha*W) @ seed
    I = speye(n, dtype=np.float64, format="csr")
    A = I - alpha * W
    exact = spsolve(A, seed.astype(np.float64))

    diff = np.max(np.abs(cum - exact))
    ok = bool(diff < tol)
    if not ok:
        raise AssertionError(
            f"Resolvent check failed: max |cumulative - exact| = {diff:.2e} ≥ tol={tol}"
        )
    return ok
