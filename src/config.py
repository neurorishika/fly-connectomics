"""
Configuration module for the ORN lateralization decay analysis.

Loads ``config.yaml`` from the repository root and exposes a frozen
:class:`Config` dataclass so all parameters are typed and discoverable.

Typical usage::

    from src.config import load_config, print_config
    cfg = load_config()
    print_config(cfg)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the absolute path to the repository root.

    The root is the parent of ``src/`` (i.e. one level above this file).
    """
    return Path(__file__).resolve().parent.parent


def _resolve(p: str, root: Path) -> Path:
    """If *p* is a relative path, resolve it against *root*; otherwise
    return it as an absolute ``Path``.  Does **not** check existence.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    return (root / path).resolve()


# ---------------------------------------------------------------------------
# Nested config dataclasses — one per YAML section
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathsConfig:
    """File-system paths used by the analysis."""

    data_dir: Path
    cache_dir: Path
    results_dir: Path
    tables_dir: Path
    figures_dir: Path
    harmonize_csv: Path


@dataclass(frozen=True)
class GraphConfig:
    """Graph construction parameters."""

    syn_threshold: int
    normalization: str
    signed: bool
    ablate_seed_feedback: bool
    dtype: str
    sparse_format: str


@dataclass(frozen=True)
class PropagationConfig:
    """Random-walk propagation parameters."""

    n_hops: int
    alpha_sweep: List[float]
    default_alpha: float
    batch_seeds: bool


@dataclass(frozen=True)
class SeedsConfig:
    """Seed-neuron selection rules."""

    min_orns_per_side: int
    max_side_imbalance_ratio: float
    random_null_seeds: int


@dataclass(frozen=True)
class LateralizationConfig:
    """Lateralization index and decay-fitting parameters."""

    noise_floor_percentile: float
    li_floor_fraction: float
    crossover_threshold: float
    bootstrap_draws: int
    min_r_squared: float


@dataclass(frozen=True)
class NullsConfig:
    """Null-model parameters."""

    n_rewirings: int
    preserve_block_structure: bool


@dataclass(frozen=True)
class RobustnessConfig:
    """Robustness-sweep parameter grids."""

    syn_threshold_sweep: List[int]
    normalization_variants: List[str]
    alpha_sweep: List[float]


@dataclass(frozen=True)
class PlottingConfig:
    """Plotting and style parameters."""

    dpi: int
    formats: List[str]
    palette: str
    dataset_colors: Dict[str, str]
    figure_size_default: Tuple[int, int]


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """All analysis parameters loaded from ``config.yaml``.

    Attributes are named after the top-level YAML sections: ``paths``,
    ``graph``, ``propagation``, ``seeds``, ``lateralization``, ``nulls``,
    ``robustness``, ``plotting``, and the scalar ``seed``.
    """

    paths: PathsConfig
    graph: GraphConfig
    propagation: PropagationConfig
    seeds: SeedsConfig
    lateralization: LateralizationConfig
    nulls: NullsConfig
    robustness: RobustnessConfig
    plotting: PlottingConfig
    seed: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from a YAML file.

    Parameters
    ----------
    path : str or None
        Path to a YAML config file.  When ``None`` (default), the file
        ``config.yaml`` in the repository root is used.

    Returns
    -------
    Config
        A frozen dataclass exposing every parameter as a typed attribute.
        Relative path values are resolved to absolute paths rooted at the
        repository root.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    """
    if path is None:
        yaml_path = _repo_root() / "config.yaml"
    else:
        yaml_path = Path(path).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh)

    root = _repo_root() if path is None else yaml_path.parent

    # --- paths section ---
    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        data_dir=_resolve(paths_raw.get("data_dir", "data"), root),
        cache_dir=_resolve(paths_raw.get("cache_dir", "cache"), root),
        results_dir=_resolve(paths_raw.get("results_dir", "results"), root),
        tables_dir=_resolve(paths_raw.get("tables_dir", "results/tables"), root),
        figures_dir=_resolve(paths_raw.get("figures_dir", "results/figures"), root),
        harmonize_csv=_resolve(paths_raw.get("harmonize_csv", ""), root),
    )

    # --- graph section ---
    g = raw.get("graph", {})
    graph = GraphConfig(
        syn_threshold=int(g.get("syn_threshold", 5)),
        normalization=str(g.get("normalization", "input_fraction")),
        signed=bool(g.get("signed", False)),
        ablate_seed_feedback=bool(g.get("ablate_seed_feedback", True)),
        dtype=str(g.get("dtype", "float32")),
        sparse_format=str(g.get("sparse_format", "csr")),
    )

    # --- propagation section ---
    p = raw.get("propagation", {})
    propagation = PropagationConfig(
        n_hops=int(p.get("n_hops", 10)),
        alpha_sweep=[float(x) for x in p.get("alpha_sweep", [0.5])],
        default_alpha=float(p.get("default_alpha", 0.5)),
        batch_seeds=bool(p.get("batch_seeds", True)),
    )

    # --- seeds section ---
    s = raw.get("seeds", {})
    seeds = SeedsConfig(
        min_orns_per_side=int(s.get("min_orns_per_side", 3)),
        max_side_imbalance_ratio=float(s.get("max_side_imbalance_ratio", 2.0)),
        random_null_seeds=int(s.get("random_null_seeds", 100)),
    )

    # --- lateralization section ---
    lat = raw.get("lateralization", {})
    lateralization = LateralizationConfig(
        noise_floor_percentile=float(lat.get("noise_floor_percentile", 95)),
        li_floor_fraction=float(lat.get("li_floor_fraction", 0.01)),
        crossover_threshold=float(lat.get("crossover_threshold", 0.1)),
        bootstrap_draws=int(lat.get("bootstrap_draws", 1000)),
        min_r_squared=float(lat.get("min_r_squared", 0.7)),
    )

    # --- nulls section ---
    n = raw.get("nulls", {})
    nulls = NullsConfig(
        n_rewirings=int(n.get("n_rewirings", 100)),
        preserve_block_structure=bool(n.get("preserve_block_structure", True)),
    )

    # --- robustness section ---
    r = raw.get("robustness", {})
    robustness = RobustnessConfig(
        syn_threshold_sweep=[int(x) for x in r.get("syn_threshold_sweep", [1, 3, 5, 10])],
        normalization_variants=[str(x) for x in r.get("normalization_variants", ["input_fraction"])],
        alpha_sweep=[float(x) for x in r.get("alpha_sweep", [0.5])],
    )

    # --- plotting section ---
    pl = raw.get("plotting", {})
    plotting = PlottingConfig(
        dpi=int(pl.get("dpi", 300)),
        formats=[str(x) for x in pl.get("formats", ["svg", "png"])],
        palette=str(pl.get("palette", "colorblind")),
        dataset_colors=dict(pl.get("dataset_colors", {})),
        figure_size_default=tuple(pl.get("figure_size_default", [8, 5])),
    )

    # --- scalar seed ---
    seed = int(raw.get("seed", 42))

    # Set numpy random seed for reproducibility
    np.random.seed(seed)

    return Config(
        paths=paths,
        graph=graph,
        propagation=propagation,
        seeds=seeds,
        lateralization=lateralization,
        nulls=nulls,
        robustness=robustness,
        plotting=plotting,
        seed=seed,
    )


def print_config(cfg: Config) -> None:
    """Pretty-print a :class:`Config` instance as YAML.

    Parameters
    ----------
    cfg : Config
        The configuration to print.
    """
    # Rebuild a dict mirroring the original YAML structure for clean output.
    d: Dict[str, Any] = {
        "paths": {
            "data_dir": str(cfg.paths.data_dir),
            "cache_dir": str(cfg.paths.cache_dir),
            "results_dir": str(cfg.paths.results_dir),
            "tables_dir": str(cfg.paths.tables_dir),
            "figures_dir": str(cfg.paths.figures_dir),
            "harmonize_csv": str(cfg.paths.harmonize_csv),
        },
        "graph": {
            "syn_threshold": cfg.graph.syn_threshold,
            "normalization": cfg.graph.normalization,
            "signed": cfg.graph.signed,
            "ablate_seed_feedback": cfg.graph.ablate_seed_feedback,
            "dtype": cfg.graph.dtype,
            "sparse_format": cfg.graph.sparse_format,
        },
        "propagation": {
            "n_hops": cfg.propagation.n_hops,
            "alpha_sweep": cfg.propagation.alpha_sweep,
            "default_alpha": cfg.propagation.default_alpha,
            "batch_seeds": cfg.propagation.batch_seeds,
        },
        "seeds": {
            "min_orns_per_side": cfg.seeds.min_orns_per_side,
            "max_side_imbalance_ratio": cfg.seeds.max_side_imbalance_ratio,
            "random_null_seeds": cfg.seeds.random_null_seeds,
        },
        "lateralization": {
            "noise_floor_percentile": cfg.lateralization.noise_floor_percentile,
            "li_floor_fraction": cfg.lateralization.li_floor_fraction,
            "crossover_threshold": cfg.lateralization.crossover_threshold,
            "bootstrap_draws": cfg.lateralization.bootstrap_draws,
            "min_r_squared": cfg.lateralization.min_r_squared,
        },
        "nulls": {
            "n_rewirings": cfg.nulls.n_rewirings,
            "preserve_block_structure": cfg.nulls.preserve_block_structure,
        },
        "robustness": {
            "syn_threshold_sweep": cfg.robustness.syn_threshold_sweep,
            "normalization_variants": cfg.robustness.normalization_variants,
            "alpha_sweep": cfg.robustness.alpha_sweep,
        },
        "plotting": {
            "dpi": cfg.plotting.dpi,
            "formats": cfg.plotting.formats,
            "palette": cfg.plotting.palette,
            "dataset_colors": cfg.plotting.dataset_colors,
            "figure_size_default": list(cfg.plotting.figure_size_default),
        },
        "seed": cfg.seed,
    }
    print(yaml.dump(d, sort_keys=False, default_flow_style=False))
