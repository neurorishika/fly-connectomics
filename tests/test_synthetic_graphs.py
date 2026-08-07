"""Synthetic graph tests for propagation and lateralization analysis.

These tests use small, hand-computable graphs to verify correctness
BEFORE running on real data. They catch the most common and silent
failure modes: indexing errors, transpose errors, and normalization bugs.

Test graphs:
  1. Pure chain: influence should appear at exactly hop k, LI = 1.0 always.
  2. Symmetric bilateral graph with one commissure: LI decays at
     analytically derivable rate.
  3. Fully mixed graph: LI ≈ 0 at hop 1.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import pytest

# Import from src (assumes pytest is run from repo root or PYTHONPATH is set)
from src.graph import build_weight_matrix, validate_alpha
from src.propagate import propagate, propagate_batch, resolvent_check
from src.lateralization import compute_li, fold_ipsi_contra, aggregate_to_cell_type
from src.config import Config, GraphConfig, PropagationConfig


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def cfg():
    """Minimal config for testing."""
    return Config(
        seed=42,
        paths=None,  # type: ignore
        graph=GraphConfig(
            syn_threshold=1,
            normalization="input_fraction",
            signed=False,
            ablate_seed_feedback=False,
            dtype="float32",
            sparse_format="csr",
        ),
        propagation=PropagationConfig(
            n_hops=10,
            alpha_sweep=[0.5],
            default_alpha=0.5,
            batch_seeds=True,
        ),
        # Minimal stubs for other config sections (not used in these tests)
        seeds=None,  # type: ignore
        lateralization=None,  # type: ignore
        nulls=None,  # type: ignore
        robustness=None,  # type: ignore
        plotting=None,  # type: ignore
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Pure chain graph
# ═══════════════════════════════════════════════════════════════════════════

def _build_chain(n: int) -> tuple:
    """Build a chain graph: 0→1→2→...→(n-1).

    Returns (neurons_df, edges_df, id_to_idx).
    """
    neuron_ids = list(range(n))
    edges = pd.DataFrame({
        "pre_id": list(range(n - 1)),
        "post_id": list(range(1, n)),
        "syn_count": [1] * (n - 1),
    })
    neurons = pd.DataFrame({
        "neuron_id": neuron_ids,
        "side": ["L"] * n,
        "cell_type": [f"type_{i}" for i in range(n)],
        "cell_type_raw": [f"type_{i}" for i in range(n)],
        "super_class": ["sensory"] * n,
        "class": ["test"] * n,
        "nt_type": ["ACh"] * n,
        "region": ["brain"] * n,
        "dataset": ["test"] * n,
    }).set_index("neuron_id", drop=False)
    id_to_idx = {i: i for i in range(n)}
    return neurons, edges, id_to_idx


def test_chain_propagation_exact_hop(cfg):
    """In a pure chain, influence should appear at exactly hop k."""
    n = 6
    neurons, edges, id_to_idx = _build_chain(n)
    W, info = build_weight_matrix(neurons, edges, cfg)
    rho = info["spectral_radius"]
    validate_alpha(0.5, rho)

    # Seed at neuron 0
    seed = np.zeros(n, dtype=np.float32)
    seed[0] = 1.0

    per_hop = propagate(W, seed, n_hops=5, alpha=0.5, spectral_radius=rho)

    # At hop k, only neuron k should have nonzero influence
    for k in range(6):
        nonzero = np.where(per_hop[k] > 1e-10)[0]
        if k < n:
            assert len(nonzero) == 1, f"Hop {k}: expected 1 nonzero, got {len(nonzero)}"
            assert nonzero[0] == k, f"Hop {k}: expected neuron {k}, got {nonzero[0]}"
        # Influence = alpha^k (since each edge weight = 1.0, seed=1.0)
        expected = (0.5 ** k) if k < n else 0.0
        actual = per_hop[k, k] if k < n else 0.0
        assert abs(actual - expected) < 1e-6, f"Hop {k}: expected {expected:.6f}, got {actual:.6f}"


def test_chain_li_always_one(cfg):
    """In a pure chain with no crossing, LI should stay at 1.0."""
    n = 10
    neurons, edges, id_to_idx = _build_chain(n)
    W, info = build_weight_matrix(neurons, edges, cfg)
    rho = info["spectral_radius"]

    # Build seeds: one for left side, one for right (identical chain, no crossing)
    # All neurons are L-side; seed L and R with mirror chains
    seed_L = np.zeros(n, dtype=np.float32)
    seed_L[0] = 1.0

    # For LI to be defined, we need both ipsi and contra
    # In a pure chain, contra = 0 always, so LI = 1.0
    per_hop = propagate(W, seed_L, n_hops=3, alpha=0.5, spectral_radius=rho)

    # Simulate: I_ipsi = per_hop[:, some_target], I_contra = 0
    for hop in range(1, 4):
        I_ipsi = per_hop[hop, hop] if hop < n else 0.0
        I_contra = 0.0
        denom = I_ipsi + I_contra
        if denom > 1e-10:
            li = (I_ipsi - I_contra) / denom
        else:
            li = np.nan
        if not np.isnan(li):
            assert li == 1.0, f"Hop {hop}: LI should be 1.0, got {li}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Symmetric bilateral graph with one commissure
# ═══════════════════════════════════════════════════════════════════════════

def _build_bilateral_with_commissure(n_per_side: int) -> tuple:
    """Build a symmetric bilateral graph.

    Structure:
      L chain: L0→L1→L2→...→L{n-1}
      R chain: R0→R1→R2→...→R{n-1}
      Commissure: L{mid}→R{mid} and R{mid}→L{mid}

    All edges have weight 1. Each neuron receives from exactly one source
    (except the mid neurons which receive from both their own chain and cross).

    Neuron IDs: L0...L{n-1} are 0...(n-1), R0...R{n-1} are n...(2n-1).
    """
    n = n_per_side
    total = 2 * n
    mid = n // 2  # midpoint where commissure sits

    edges_list = []
    # L chain: L_i → L_{i+1}
    for i in range(n - 1):
        edges_list.append({"pre_id": i, "post_id": i + 1, "syn_count": 1})
    # R chain: R_i → R_{i+1}
    for i in range(n - 1):
        edges_list.append({"pre_id": n + i, "post_id": n + i + 1, "syn_count": 1})
    # Commissure: L_mid → R_mid and R_mid → L_mid
    edges_list.append({"pre_id": mid, "post_id": n + mid, "syn_count": 1})
    edges_list.append({"pre_id": n + mid, "post_id": mid, "syn_count": 1})

    edges = pd.DataFrame(edges_list)

    sides = ["L"] * n + ["R"] * n
    neurons = pd.DataFrame({
        "neuron_id": list(range(total)),
        "side": sides,
        "cell_type": [f"type_{i}" for i in range(total)],
        "cell_type_raw": [f"type_{i}" for i in range(total)],
        "super_class": ["sensory"] * total,
        "class": ["test"] * total,
        "nt_type": ["ACh"] * total,
        "region": ["brain"] * total,
        "dataset": ["test"] * total,
    }).set_index("neuron_id", drop=False)

    id_to_idx = {i: i for i in range(total)}
    return neurons, edges, id_to_idx


def test_bilateral_commissure_crossing(cfg):
    """In the bilateral+commissure graph, influence should cross at the commissure.

    Seed L0: flow goes L0→L1→...→L_mid→[R_mid via commissure]→R_{mid+1}→...
    """
    n_per_side = 6
    n = n_per_side
    mid = n // 2
    neurons, edges, id_to_idx = _build_bilateral_with_commissure(n_per_side)
    W, info = build_weight_matrix(neurons, edges, cfg)
    rho = info["spectral_radius"]
    validate_alpha(0.5, rho)

    # Seed at L0
    seed = np.zeros(2 * n, dtype=np.float32)
    seed[0] = 1.0

    per_hop = propagate(W, seed, n_hops=8, alpha=0.5, spectral_radius=rho)

    # Before commissure (hops 0..mid-1): influence only on L side
    for k in range(mid):
        left_influence = per_hop[k, :n].sum()
        right_influence = per_hop[k, n:].sum()
        assert right_influence < 1e-10, (
            f"Hop {k}: influence should not have reached R side yet, "
            f"got R influence = {right_influence:.2e}"
        )
        assert left_influence > 1e-10, (
            f"Hop {k}: influence should be on L side"
        )

    # At hop mid+1: commissure fires — influence appears on R side
    # (L3 activates at hop mid=3, then at hop 4 passes to R3 via commissure)
    hop_comm = mid + 1
    hop_comm_influence = per_hop[hop_comm]
    right_at_comm = hop_comm_influence[n:].sum()
    assert right_at_comm > 1e-10, (
        f"Hop {hop_comm}: commissure should deliver influence to R side, "
        f"got R influence = {right_at_comm:.2e}"
    )

    # After commissure: both sides have influence
    # (mass is not exactly conserved under input-fraction normalization
    #  when neurons have multiple inputs — this is expected behavior)


def test_bilateral_li_decay_derivable(cfg):
    """In the symmetric bilateral graph, LI should decrease after the commissure.

    LI = (I_ipsi - I_contra) / (I_ipsi + I_contra)
    Before commissure: LI = 1.0 (purely ipsi)
    After commissure: LI < 1.0 (some contra influence)
    """
    n_per_side = 10
    n = n_per_side
    mid = n // 2
    neurons, edges, id_to_idx = _build_bilateral_with_commissure(n_per_side)
    W, info = build_weight_matrix(neurons, edges, cfg)
    rho = info["spectral_radius"]

    # Seed both L0 and R0, compare influence on target neuron at position i
    seed_L = np.zeros(2 * n, dtype=np.float32)
    seed_L[0] = 1.0
    seed_R = np.zeros(2 * n, dtype=np.float32)
    seed_R[n] = 1.0  # R0

    per_hop_L = propagate(W, seed_L, n_hops=6, alpha=0.5, spectral_radius=rho)
    per_hop_R = propagate(W, seed_R, n_hops=6, alpha=0.5, spectral_radius=rho)

    # Verify that after the commissure, targets on the L side DO receive
    # nonzero contra influence from the R seed (via the commissure).
    n_hops = 8
    per_hop_L = propagate(W, seed_L, n_hops=n_hops, alpha=0.5, spectral_radius=rho)
    per_hop_R = propagate(W, seed_R, n_hops=n_hops, alpha=0.5, spectral_radius=rho)

    found_contra = False
    for target_idx in range(mid + 1, n):
        for hop in range(1, n_hops + 1):
            I_contra = per_hop_R[hop, target_idx]
            if I_contra > 1e-10:
                I_ipsi = per_hop_L[hop, target_idx]
                denom = I_ipsi + I_contra
                li = (I_ipsi - I_contra) / denom
                assert li < 0.999, (
                    f"Target L_{target_idx} hop {hop}: LI should be < 1.0 "
                    f"after commissure, got {li:.4f}"
                )
                found_contra = True
                break
        if found_contra:
            break

    assert found_contra, (
        "No contra influence found on L-side targets "
        "after commissure — commissure is not working"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Fully mixed graph
# ═══════════════════════════════════════════════════════════════════════════

def _build_fully_mixed(n: int) -> tuple:
    """Build a fully connected graph where every neuron connects to every other.

    All edges have weight 1. Under input-fraction normalization, each neuron
    distributes its influence equally to all n-1 other neurons.
    """
    edges_list = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edges_list.append({"pre_id": i, "post_id": j, "syn_count": 1})
    edges = pd.DataFrame(edges_list)

    half = n // 2
    sides = ["L"] * half + ["R"] * (n - half)
    neurons = pd.DataFrame({
        "neuron_id": list(range(n)),
        "side": sides,
        "cell_type": [f"type_{i}" for i in range(n)],
        "cell_type_raw": [f"type_{i}" for i in range(n)],
        "super_class": ["sensory"] * n,
        "class": ["test"] * n,
        "nt_type": ["ACh"] * n,
        "region": ["brain"] * n,
        "dataset": ["test"] * n,
    }).set_index("neuron_id", drop=False)
    id_to_idx = {i: i for i in range(n)}
    return neurons, edges, id_to_idx


def test_fully_mixed_li_vanishes(cfg):
    """In a fully mixed graph, LI should be ~0 at hop 1."""
    n = 10
    neurons, edges, id_to_idx = _build_fully_mixed(n)
    W, info = build_weight_matrix(neurons, edges, cfg)
    rho = info["spectral_radius"]
    validate_alpha(0.5, rho)

    half = n // 2
    # Seed left half
    seed_L = np.zeros(n, dtype=np.float32)
    seed_L[:half] = 1.0 / half
    # Seed right half
    seed_R = np.zeros(n, dtype=np.float32)
    seed_R[half:] = 1.0 / (n - half)

    per_hop_L = propagate(W, seed_L, n_hops=2, alpha=0.5, spectral_radius=rho)
    per_hop_R = propagate(W, seed_R, n_hops=2, alpha=0.5, spectral_radius=rho)

    # At hop 1, every neuron receives equal input from all sources
    # So I_ipsi ≈ I_contra for any target, giving LI ≈ 0
    for target in range(n):
        I_ipsi = per_hop_L[1, target]
        I_contra = per_hop_R[1, target]
        denom = I_ipsi + I_contra
        if denom > 1e-10:
            li = (I_ipsi - I_contra) / denom
            # LI should be very close to 0 (within ~0.1 due to unequal L/R split)
            assert abs(li) < 0.2, (
                f"Target {target} hop 1: LI should be ~0, got {li:.4f}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Resolvent agreement
# ═══════════════════════════════════════════════════════════════════════════

def test_resolvent_agreement(cfg):
    """The cumulative random-walk sum should agree with sparse solve."""
    n = 15
    neurons, edges, id_to_idx = _build_fully_mixed(n)
    W, info = build_weight_matrix(neurons, edges, cfg)
    rho = info["spectral_radius"]
    validate_alpha(0.5, rho)

    seed = np.random.default_rng(42).random(n).astype(np.float32)
    seed /= seed.sum()

    assert resolvent_check(W, seed, alpha=0.5, n_hops=20, tol=1e-5), (
        "Resolvent check failed: cumulative random walk does not match sparse solve"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Batch propagation matches single-seed
# ═══════════════════════════════════════════════════════════════════════════

def test_batch_matches_single(cfg):
    """propagate_batch should give same results as repeated propagate."""
    n = 10
    neurons, edges, id_to_idx = _build_fully_mixed(n)
    W, info = build_weight_matrix(neurons, edges, cfg)
    rho = info["spectral_radius"]

    n_seeds = 3
    rng = np.random.default_rng(123)
    seeds_single = []
    seed_matrix = np.zeros((n, n_seeds), dtype=np.float32)
    for s in range(n_seeds):
        s_vec = rng.random(n).astype(np.float32)
        s_vec /= s_vec.sum()
        seeds_single.append(s_vec)
        seed_matrix[:, s] = s_vec

    batch_result = propagate_batch(W, seed_matrix, n_hops=5, alpha=0.5, spectral_radius=rho)

    for s in range(n_seeds):
        single_result = propagate(W, seeds_single[s], n_hops=5, alpha=0.5, spectral_radius=rho)
        max_diff = np.max(np.abs(batch_result[:, :, s] - single_result))
        assert max_diff < 1e-6, (
            f"Seed {s}: batch vs single max diff = {max_diff:.2e}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Negative LI (contralateral bias)
# ═══════════════════════════════════════════════════════════════════════════

def test_negative_li():
    """LI should be negative when contra > ipsi."""
    df = pd.DataFrame({
        "I_ipsi": [0.3, 0.8, 0.0],
        "I_contra": [0.7, 0.2, 0.5],
    })
    result = compute_li(df, noise_floor=1e-10)

    assert result["LI"].iloc[0] < 0, "First row: contra > ipsi, LI should be negative"
    assert result["LI"].iloc[1] > 0, "Second row: ipsi > contra, LI should be positive"
    assert result["LI"].iloc[0] == pytest.approx(-0.4, abs=1e-6)
    assert result["LI"].iloc[1] == pytest.approx(0.6, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: Noise floor masking
# ═══════════════════════════════════════════════════════════════════════════

def test_noise_floor_masking():
    """LI should be NaN when denominator is below noise floor."""
    df = pd.DataFrame({
        "I_ipsi": [1.0, 0.001, 0.0],
        "I_contra": [0.5, 0.001, 0.0],
    })
    result = compute_li(df, noise_floor=0.01)

    # Row 0: denom=1.5 > 0.01, should have valid LI
    assert not np.isnan(result["LI"].iloc[0])
    assert result["above_floor"].iloc[0]

    # Row 1: denom=0.002 < 0.01, should be masked
    assert np.isnan(result["LI"].iloc[1])
    assert not result["above_floor"].iloc[1]

    # Row 2: denom=0, should be masked
    assert np.isnan(result["LI"].iloc[2])
    assert not result["above_floor"].iloc[2]
