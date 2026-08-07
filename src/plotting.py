"""
Plotting module for the ORN lateralization decay analysis.

Every function accepts data + :class:`~.config.Config`, produces a
matplotlib Figure, saves it to disk (SVG + PNG at ``config.plotting.dpi``),
and returns ``(fig, ax)`` for notebook chaining.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy import stats

from .config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_style(config: Config) -> None:
    """Apply consistent matplotlib + seaborn style from *config*.

    Parameters
    ----------
    config : Config
        ``config.plotting.palette`` names the seaborn palette
        (default ``"colorblind"``).  ``config.plotting.dpi`` sets the
        save resolution.  ``config.plotting.figure_size_default`` sets
        ``rcParams["figure.figsize"]``.

    Notes
    -----
    Call once at the top of the analysis notebook (before any plotting).
    """
    palette = config.plotting.palette
    dpi = config.plotting.dpi
    figsize = config.plotting.figure_size_default

    sns.set_theme(
        style="ticks",
        palette=palette,
        rc={
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.figsize": figsize,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        },
    )

    logger.info("Plotting style set: palette=%s, dpi=%d", palette, dpi)


# ---------------------------------------------------------------------------
# Figure 1 — LI vs synaptic distance (hop)
# ---------------------------------------------------------------------------

def plot_li_vs_hop(
    li_data: pd.DataFrame,
    config: Config,
    noise_floor: float,
    filename: str,
) -> Tuple[Figure, Axes]:
    """LI magnitude versus hop, one line per ORN subtype, faceted by dataset.

    Parameters
    ----------
    li_data : pd.DataFrame
        Long-form DataFrame with columns:
        ``dataset``, ``orn_subtype``, ``hop`` (int), ``abs_li`` (float).
    config : Config
    noise_floor : float
        Horizontal band floor value, e.g. the 95th percentile of |LI|
        from reconstructions.
    filename : str
        Stem for the output file (without extension).  Saved to
        ``config.paths.figures_dir / filename.{fmt}`` for each format
        in ``config.plotting.formats``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    datasets = sorted(li_data["dataset"].unique())
    n_datasets = len(datasets)
    dataset_colors = config.plotting.dataset_colors

    # Build a consistent palette mapping for ORN subtypes
    subtypes = sorted(li_data["orn_subtype"].unique())
    n_subtypes = len(subtypes)
    palette = sns.color_palette(config.plotting.palette, n_subtypes)
    subtype_color = dict(zip(subtypes, palette))

    fig, axes = plt.subplots(
        1,
        n_datasets,
        figsize=(5.5 * n_datasets, 4.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]  # 1-D array

    for ax, dataset in zip(axes, datasets):
        sub = li_data[li_data["dataset"] == dataset]
        color = dataset_colors.get(dataset.lower(), "#333333")

        for st in subtypes:
            st_sub = sub[sub["orn_subtype"] == st]
            if st_sub.empty:
                continue
            st_sub = st_sub.sort_values("hop")
            ax.plot(
                st_sub["hop"],
                st_sub["abs_li"],
                color=subtype_color[st],
                alpha=0.5,
                linewidth=0.8,
                marker=".",
                markersize=3,
            )

        # Noise floor band
        ax.axhspan(0, noise_floor, color="gray", alpha=0.12, zorder=0)
        ax.axhline(noise_floor, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)

        ax.set_title(dataset, fontweight="bold", color=color)
        ax.set_xlabel("Hop")
        ax.set_ylabel("|LI|")
        ax.set_xlim(left=-0.2)
        ax.set_ylim(bottom=-0.02)

    # Shared legend for ORN subtypes (single-column, compact)
    handles = [
        plt.Line2D([0], [0], color=subtype_color[st], linewidth=1.5, label=st)
        for st in subtypes
    ]
    fig.legend(
        handles=handles,
        title="ORN subtype",
        loc="center right",
        bbox_to_anchor=(1.12, 0.5),
        frameon=False,
        fontsize=7,
        title_fontsize=8,
    )

    fig.suptitle("|LI| decay with synaptic distance", y=1.01, fontweight="bold")
    fig.tight_layout()

    _save_fig(fig, config, filename)
    return fig, axes


# ---------------------------------------------------------------------------
# Figure 2 — λ ranking across ORN subtypes
# ---------------------------------------------------------------------------

def plot_lambda_ranking(
    lambda_data: pd.DataFrame,
    config: Config,
    controls: dict,
    filename: str,
) -> Tuple[Figure, Axes]:
    """Bar chart of lateralization decay rate λ per ORN subtype.

    Parameters
    ----------
    lambda_data : pd.DataFrame
        Columns: ``orn_subtype`` (str), ``lambda`` (float),
        ``lambda_ci_low`` (float), ``lambda_ci_high`` (float),
        ``dataset`` (str, optional — if multiple datasets, the mean
        across datasets is used).
    config : Config
    controls : dict
        Mapping ``{label: lambda_value}`` for control modalities
        (e.g. visual, mechanosensory).  Drawn as horizontal reference
        lines.
    filename : str
        Output file stem.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    # Aggregate across datasets if needed
    if "dataset" in lambda_data.columns and lambda_data["dataset"].nunique() > 1:
        agg = (
            lambda_data.groupby("orn_subtype", as_index=False)
            .agg(lambda_val=("lambda", "mean"), ci_low=("lambda_ci_low", "mean"), ci_high=("lambda_ci_high", "mean"))
        )
    else:
        agg = lambda_data.rename(
            columns={"lambda": "lambda_val", "lambda_ci_low": "ci_low", "lambda_ci_high": "ci_high"}
        )[["orn_subtype", "lambda_val", "ci_low", "ci_high"]]

    # Sort by λ descending
    agg = agg.sort_values("lambda_val", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = np.arange(len(agg))
    values = agg["lambda_val"].values
    yerr_low = values - agg["ci_low"].values
    yerr_high = agg["ci_high"].values - values

    palette = sns.color_palette(config.plotting.palette, 1)
    bar_color = palette[0]

    ax.barh(y_pos, values, xerr=[yerr_low, yerr_high], color=bar_color, alpha=0.85,
            edgecolor="white", linewidth=0.5, capsize=2, error_kw={"linewidth": 0.8})

    # Control reference lines
    control_styles = {"visual": ("#D55E00", "--"), "mechanosensory": ("#0072B2", "-.")}
    for label, lam in controls.items():
        style = control_styles.get(label, ("#666666", ":"))
        ax.axvline(lam, color=style[0], linestyle=style[1], linewidth=1.2, alpha=0.8,
                   label=f"{label} (λ={lam:.3f})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(agg["orn_subtype"].values, fontsize=8)
    ax.set_xlabel("Decay rate  λ")
    ax.set_ylabel("ORN subtype")
    ax.set_title("Lateralization decay rate by ORN subtype", fontweight="bold")

    if controls:
        ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax.invert_yaxis()
    sns.despine(ax=ax)
    fig.tight_layout()

    _save_fig(fig, config, filename)
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 3 — Cross-dataset scatter of λ
# ---------------------------------------------------------------------------

def plot_cross_dataset_scatter(
    lambda_data: pd.DataFrame,
    config: Config,
    filename: str,
) -> Tuple[Figure, List[Axes]]:
    """Pairwise scatter plots of λ across datasets (FlyWire vs MCNS,
    FlyWire vs BANC, MCNS vs BANC), restricted to harmonised ORN types
    present in both datasets.

    Parameters
    ----------
    lambda_data : pd.DataFrame
        Must contain columns: ``orn_subtype``, ``dataset``, ``lambda``.
    config : Config
    filename : str
        Output file stem.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        The three scatter axes.
    """
    datasets_present = sorted(lambda_data["dataset"].unique())
    pairs = [("FlyWire", "MCNS"), ("FlyWire", "BANC"), ("MCNS", "BANC")]
    pairs = [(a, b) for (a, b) in pairs if a in datasets_present and b in datasets_present]

    if len(pairs) == 0:
        raise ValueError(
            f"Need at least 2 datasets in lambda_data; found {datasets_present}"
        )

    dataset_colors = config.plotting.dataset_colors
    n_pairs = len(pairs)

    fig, axes = plt.subplots(1, n_pairs, figsize=(5.5 * n_pairs, 4.5), squeeze=False)
    axes = axes[0]

    for ax, (ds_a, ds_b) in zip(axes, pairs):
        a_data = lambda_data[lambda_data["dataset"] == ds_a].set_index("orn_subtype")["lambda"]
        b_data = lambda_data[lambda_data["dataset"] == ds_b].set_index("orn_subtype")["lambda"]
        common = a_data.index.intersection(b_data.index)
        a_vals = a_data.loc[common]
        b_vals = b_data.loc[common]

        if len(common) < 3:
            ax.text(0.5, 0.5, f"Too few shared types\n(n={len(common)})",
                    transform=ax.transAxes, ha="center", va="center", fontsize=10)
            ax.set_title(f"{ds_a} vs {ds_b}")
            continue

        rho, pval = stats.spearmanr(a_vals, b_vals)
        color_a = dataset_colors.get(ds_a.lower(), "#333")
        color_b = dataset_colors.get(ds_b.lower(), "#333")

        ax.scatter(a_vals, b_vals, c=color_a, alpha=0.7, edgecolors="white", linewidth=0.3, s=40)

        # Diagonal (y=x)
        lims = [
            min(a_vals.min(), b_vals.min()) - 0.02,
            max(a_vals.max(), b_vals.max()) + 0.02,
        ]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Annotation
        ax.annotate(
            f"ρ = {rho:.3f}\np = {pval:.2e}",
            xy=(0.05, 0.92),
            xycoords="axes fraction",
            fontsize=9,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray", linewidth=0.5),
        )

        ax.set_xlabel(f"λ  ({ds_a})")
        ax.set_ylabel(f"λ  ({ds_b})")
        ax.set_title(f"{ds_a}  vs  {ds_b}", fontweight="bold")
        ax.set_aspect("equal")

    fig.suptitle("Cross-dataset agreement of decay rate λ", y=1.01, fontweight="bold")
    fig.tight_layout()

    _save_fig(fig, config, filename)
    return fig, list(axes)


# ---------------------------------------------------------------------------
# Figure 4 — LI heatmap for a representative ORN subtype
# ---------------------------------------------------------------------------

def plot_li_heatmap(
    li_data: pd.DataFrame,
    seed_glomerulus: str,
    config: Config,
    filename: str,
) -> Tuple[Figure, Axes]:
    """Heatmap of |LI| across target cell types versus hop for a single
    seed ORN subtype.

    Parameters
    ----------
    li_data : pd.DataFrame
        Columns: ``dataset``, ``orn_subtype``, ``hop`` (int),
        ``target_type`` (str), ``abs_li`` (float).
    seed_glomerulus : str
        The ORN subtype / glomerulus name to plot (e.g. ``"ORN_DA1"``).
    config : Config
    filename : str
        Output file stem.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    sub = li_data[li_data["orn_subtype"] == seed_glomerulus].copy()
    if sub.empty:
        raise ValueError(f"No LI data for seed glomerulus: {seed_glomerulus}")

    datasets = sorted(sub["dataset"].unique())
    n_datasets = len(datasets)

    fig, axes = plt.subplots(
        1,
        n_datasets,
        figsize=(6 * n_datasets, 5),
        squeeze=False,
    )
    axes = axes[0]

    for ax, dataset in enumerate(datasets):
        ds_sub = sub[sub["dataset"] == dataset]
        pivot = ds_sub.pivot_table(
            index="target_type", columns="hop", values="abs_li", aggfunc="mean"
        )
        # Sort rows by mean LI across hops
        pivot["_mean"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("_mean", ascending=False).drop(columns="_mean")

        if pivot.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{dataset} — {seed_glomerulus}")
            continue

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="viridis",
            cbar_kws={"label": "|LI|", "shrink": 0.8},
            linewidths=0.2,
            linecolor="white",
            xticklabels=2,
            yticklabels=(pivot.shape[0] <= 40),
        )
        ax.set_title(f"{dataset}  —  {seed_glomerulus}", fontweight="bold")
        ax.set_xlabel("Hop")
        ax.set_ylabel("Target cell type" if pivot.shape[0] <= 40 else "")

    fig.suptitle("LI heatmap across target cell types", y=1.02, fontweight="bold")
    fig.tight_layout()

    _save_fig(fig, config, filename)
    return fig, axes


# ---------------------------------------------------------------------------
# Figure 5 — Robustness matrix (rank correlations across parameter settings)
# ---------------------------------------------------------------------------

def plot_robustness_matrix(
    corr_matrix: pd.DataFrame,
    config: Config,
    filename: str,
) -> Tuple[Figure, Axes]:
    """Heatmap of Spearman rank correlations of λ across robustness
    parameter settings.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Square matrix (index = columns = parameter labels) of correlation
        values.
    config : Config
    filename : str
        Output file stem.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    n = corr_matrix.shape[0]
    figsize = max(6, n * 0.45)

    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.9))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        ax=ax,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=(n <= 15),
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
        square=True,
    )

    ax.set_title("Rank-correlation of λ across parameter settings", fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()

    _save_fig(fig, config, filename)
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 6 — Null distributions
# ---------------------------------------------------------------------------

def plot_null_distributions(
    null_data: dict,
    config: Config,
    filename: str,
) -> Tuple[Figure, np.ndarray]:
    """Observed λ against null-model distributions.

    Parameters
    ----------
    null_data : dict
        Keys are ORN subtype labels (str).  Each value is a dict with:
        ``observed`` (float), ``rewiring`` (list of float),
        ``random_seeds`` (list of float).
    config : Config
    filename : str
        Output file stem.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    subtypes = sorted(null_data.keys())
    n = len(subtypes)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 2.8 * n_rows), squeeze=False,
    )

    for idx, st in enumerate(subtypes):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        d = null_data[st]
        observed = d["observed"]
        rewiring = d.get("rewiring", [])
        random_seeds = d.get("random_seeds", [])

        bins = 30
        if len(rewiring) > 0:
            ax.hist(rewiring, bins=bins, alpha=0.5, color="#0173B2", label="Rewiring null", density=True)
        if len(random_seeds) > 0:
            ax.hist(random_seeds, bins=bins, alpha=0.5, color="#DE8F05", label="Random-seed null", density=True)

        ax.axvline(observed, color="black", linewidth=1.5, linestyle="--", label=f"Observed λ={observed:.3f}")
        ax.set_title(st, fontsize=9, fontweight="bold")
        ax.set_xlabel("λ")
        ax.set_ylabel("Density" if col == 0 else "")
        if idx == 0 and (len(rewiring) > 0 or len(random_seeds) > 0):
            ax.legend(fontsize=6, frameon=False, loc="upper right")

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle("Null-model comparison of decay rate λ", y=1.01, fontweight="bold")
    fig.tight_layout()

    _save_fig(fig, config, filename)
    return fig, axes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: Figure, config: Config, filename: str) -> None:
    """Save *fig* in every format listed in ``config.plotting.formats``.

    Parameters
    ----------
    fig : Figure
    config : Config
    filename : str
        Stem (no extension).  Written to ``config.paths.figures_dir``.
    """
    figures_dir = Path(config.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dpi = config.plotting.dpi
    for fmt in config.plotting.formats:
        ext = fmt.lstrip(".")
        out_path = figures_dir / f"{filename}.{ext}"
        fig.savefig(out_path, dpi=dpi, format=ext, bbox_inches="tight")
        logger.info("Saved %s", out_path)
