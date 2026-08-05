#!/usr/bin/env python3
"""
ALPN Glomerulus Volume Mapping — MCNS + FAFB
==============================================
Computes 3-D convex-hull volumes for every antennal-lobe glomerulus, split by
antennal-lobe side (AL_R / AL_L), using two complementary synapse sets:

  (A) ALPN postsynaptic sites  ("PN inputs")
        · Input sites on the antennal-lobe projection neurons.
        · Glomerulus identity comes directly from the ALPN cell type.

  (B) ORN presynaptic sites  ("ORN outputs")
        · ORN → ALPN presynapse locations (dense; tile the glomerular neuropil).
        · Each presynapse is assigned to a glomerulus + AL side via its
          postsynaptic ALPN partner.  ORNs project *bilaterally*, so the
          post-partner (ALPNs are unilateral) is what cleanly separates the
          left- and right-AL clouds.

Why group by the post-partner ALPN?
-----------------------------------
ORN somata sit on one side but their axons innervate *both* antennal lobes, so
grouping ORN presynapses by ORN soma-side merges the two ALs into one ~200 µm
blob.  The cognate ALPN is unilateral, so its (glomerulus, side) is the correct
label for the presynapse and naturally splits AL_R from AL_L.

Datasets  (verified live, 2026-06)
----------------------------------
  MCNS  — Male CNS connectome → neuprint-python
          https://neuprint.janelia.org   dataset: male-cns:v0.9
          coordinates in 8 nm voxels.
  FAFB  — FlyWire / Full Adult Female Brain → CAVEclient
          datastack: flywire_fafb_public   materialization: 783
          coordinates already in nm.  Cell typing from the public Codex
          classification dump (data/FlyWire/classification.csv.gz, v783).

Tokens
------
  NEUPRINT_TOKEN  — read from a .env file at the repo root (python-dotenv) or
                    the environment.  https://neuprint.janelia.org/account
  CAVE token      — taken from the locally cached CAVE secret
                    (~/.cloudvolume/secrets).  No env var needed; if you have
                    never authenticated run:  caveclient + client.auth ... .

Install
-------
pip install neuprint-python caveclient python-dotenv numpy pandas scipy matplotlib
"""

from __future__ import annotations
import os
import sys
import json
import time
import warnings
from pathlib import Path

# ── Import-path guard ─────────────────────────────────────────────────────────
# A data folder named ``neuprint/`` lives next to the analysis notebooks.  When
# this code runs from that directory (e.g. inside Jupyter), the notebook's own
# directory sits first on ``sys.path`` and that local folder shadows the
# pip-installed ``neuprint-python`` package.  De-prioritise (don't remove) the
# current directory so installed packages win, while local helper modules
# remain importable as a fallback.
_cwd = os.path.realpath(os.getcwd())
sys.path = ([p for p in sys.path if os.path.realpath(p or ".") != _cwd]
            + [p for p in sys.path if os.path.realpath(p or ".") == _cwd])

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── MCNS / neuprint ───────────────────────────────────────────────────────────
NP_SERVER   = "https://neuprint.janelia.org"
NP_DATASET  = "male-cns:v0.9"
NP_AL_ROIS  = ["AL(R)", "AL(L)"]
NP_VOXEL_NM = 8                        # nm per voxel in the male-cns EM volume

# ── FAFB / FlyWire ────────────────────────────────────────────────────────────
FW_DATASTACK = "flywire_fafb_public"
FW_MAT_VER   = 783                     # public materialisation matching the dump
FW_SYN_TABLE = "synapses_nt_v1"
# Cell typing comes from the Codex classification dump (root_ids match v783):
FW_CLASSIFICATION = "data/FlyWire/classification.csv.gz"
FW_AL_MIDLINE_X   = 520_000           # nm; x < midline → left AL, else right AL.
#                                       Splits synapses by location (not by the
#                                       PN's annotated side) so bilateral PNs
#                                       don't merge the two antennal lobes.

# ── Analysis ──────────────────────────────────────────────────────────────────
SYNAPSE_METHOD   = "both"              # "pn_inputs" | "orn_outputs" | "both"
RUN_DATASETS     = ("MCNS", "FAFB")    # subset to ("MCNS",) etc. to run one
MIN_SYNAPSES     = 25                  # skip a (glomerulus, side) with fewer pts
OUTLIER_QTILE    = 0.05                # drop farthest 5 % (radial) before the hull;
#                                        raise to trim neurite tails harder, lower
#                                        to keep more of the cloud.  Convex-hull
#                                        volume is inherently spread-sensitive.
BATCH_SIZE       = 50                  # ids per CAVE query batch (smaller = gentler)
CAVE_RETRIES     = 5                   # retries per CAVE query (transient 500s)
GLOMERULUS_LIMIT = None                # int → only the first N glomeruli (debug)

OUTPUT_DIR = Path("glomerulus_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# LABELLING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def norm_side(raw) -> str:
    """Normalise an AL-side label to 'R' / 'L' (else the raw string)."""
    s = str(raw)
    if "right" in s or "(R)" in s:
        return "R"
    if "left" in s or "(L)" in s:
        return "L"
    return s


def glom_from_orn(t: str) -> str | None:
    """'ORN_DA1' → 'DA1'."""
    if not isinstance(t, str) or not t.startswith("ORN_"):
        return None
    g = t[len("ORN_"):]
    return g or None


def glom_from_alpn(t: str) -> str | None:
    """'DA1_lPN' → 'DA1'  (uniglomerular-PN prefix)."""
    if not isinstance(t, str) or "_" not in t:
        return None
    g = t.split("_")[0]
    return g or None


# Multiglomerular PNs (hemibrain 'M_*'/'MZ_*' tracts, or comma/'+'-joined
# names like 'VP1d,VP4') innervate many glomeruli; lumping their synapses into
# one pseudo-glomerulus balloons the hull, so they are excluded.
_MULTIGLOM = {"M", "MZ"}


def is_uniglom(g) -> bool:
    if not isinstance(g, str) or g in _MULTIGLOM:
        return False
    if any(ch in g for ch in ",+/ "):
        return False
    if g.startswith("CB") and g[2:].isdigit():     # provisional cell-body ID
        return False
    return True


def _locate(rel: str) -> Path:
    """Find a repo-relative data file whether run from repo root or analysis/."""
    here = Path(rel)
    for base in (Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parent):
        cand = base / rel
        if cand.exists():
            return cand
    return here   # let the caller raise a clear FileNotFoundError


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

def trimmed_hull(pts: np.ndarray, q: float = OUTLIER_QTILE):
    """
    Drop the farthest fraction q of points (by Euclidean distance from the
    median centre) before fitting the hull.

    A convex hull is extremely sensitive to outliers: a handful of stray
    synapses (along a PN's neurite, or mis-localised) make the hull stretch
    into a long spike.  A radial trim removes those — unlike a per-axis
    quantile trim, it also catches diagonal outliers.
    """
    centre = np.median(pts, axis=0)
    dist   = np.linalg.norm(pts - centre, axis=1)
    thresh = np.quantile(dist, 1.0 - q)
    mask   = dist <= thresh
    p      = pts[mask] if mask.sum() >= 4 else pts
    return ConvexHull(p), p


def compute_glom_volumes(syn_df: pd.DataFrame, coord_unit_nm: float = 1.0) -> dict:
    """
    Trimmed convex-hull volume for every (glomerulus, side) group.

    syn_df must have columns [glomerulus, side, x, y, z] in one coordinate unit.
    Returns dict keyed by (glomerulus, side) → metrics.
    """
    to_um   = coord_unit_nm / 1_000.0
    results: dict = {}

    for (glom, side), grp in syn_df.groupby(["glomerulus", "side"]):
        pts = grp[["x", "y", "z"]].values.astype(np.float64)
        if len(pts) < MIN_SYNAPSES:
            continue
        try:
            hull, pts_t = trimmed_hull(pts)
        except Exception as exc:
            print(f"    ConvexHull failed {glom}/{side}: {exc}")
            continue
        centroid = pts_t.mean(axis=0)
        results[(glom, side)] = dict(
            hull        = hull,
            pts         = pts_t,
            raw_pts     = pts,
            centroid    = centroid,
            centroid_nm = centroid * coord_unit_nm,
            vol_um3     = hull.volume * to_um ** 3,
            area_um2    = hull.area   * to_um ** 2,
            n_syn       = len(pts),
            n_hull      = len(pts_t),
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND A — MCNS via neuprint-python
# ══════════════════════════════════════════════════════════════════════════════

def run_neuprint(method: str = "both") -> dict[str, pd.DataFrame]:
    from dotenv import load_dotenv, find_dotenv
    from neuprint import (Client, fetch_neurons, fetch_synapses,
                          NeuronCriteria as NC, SynapseCriteria as SC)

    load_dotenv(find_dotenv(usecwd=True) or _locate(".env"))
    token = os.getenv("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError("NEUPRINT_TOKEN not found in environment or .env")

    print(f"\n{'─'*64}\n  MCNS  ·  {NP_SERVER}  ·  {NP_DATASET}\n{'─'*64}")
    c = Client(NP_SERVER, dataset=NP_DATASET, token=token)
    print("  Connected  ✓")

    # ── neuron → glomerulus maps ──────────────────────────────────────────────
    alpn, _ = fetch_neurons(NC(class_="ALPN"), client=c)
    alpn["glomerulus"] = alpn["type"].map(glom_from_alpn)
    alpn = alpn.dropna(subset=["glomerulus"])
    alpn = alpn[alpn["glomerulus"].map(is_uniglom)]      # uniglomerular only
    alpn_glom = dict(zip(alpn["bodyId"], alpn["glomerulus"]))
    alpn_ids  = alpn["bodyId"].tolist()

    orn, _ = fetch_neurons(NC(type="ORN_.*", regex=True), client=c)
    orn["glomerulus"] = orn["type"].map(glom_from_orn)
    orn = orn.dropna(subset=["glomerulus"])
    orn_glom = dict(zip(orn["bodyId"], orn["glomerulus"]))
    orn_ids  = orn["bodyId"].tolist()
    print(f"  ALPNs: {len(alpn_ids)} ({alpn['glomerulus'].nunique()} gloms)  |  "
          f"ORNs: {len(orn_ids)} ({orn['glomerulus'].nunique()} gloms)")

    if GLOMERULUS_LIMIT:
        keep = sorted(set(alpn["glomerulus"]))[:GLOMERULUS_LIMIT]
        alpn_ids = alpn[alpn["glomerulus"].isin(keep)]["bodyId"].tolist()
        orn_ids  = orn[orn["glomerulus"].isin(keep)]["bodyId"].tolist()
        print(f"  [debug] limited to glomeruli: {keep}")

    frames: dict[str, pd.DataFrame] = {}

    # ── A1: ALPN postsynaptic (input) sites ───────────────────────────────────
    if method in ("pn_inputs", "both"):
        print("  [A1] ALPN postsynaptic sites …")
        syn = fetch_synapses(NC(bodyId=alpn_ids),
                             SC(type="post", rois=NP_AL_ROIS, primary_only=True),
                             client=c)
        df = pd.DataFrame({
            "glomerulus": syn["bodyId"].map(alpn_glom).values,
            "side"      : syn["roi"].map(norm_side).values,
            "x": syn["x"].values, "y": syn["y"].values, "z": syn["z"].values,
        }).dropna(subset=["glomerulus"])
        df = df[df["side"].isin(["R", "L"])]
        df.to_csv(OUTPUT_DIR / "mcns_pn_inputs.csv", index=False)
        frames["pn_inputs"] = df
        print(f"    {len(df):,} synapses")

    # ── A2: ORN presynaptic (output) sites in the AL ──────────────────────────
    #  The synapse ROI (AL(R)/AL(L)) gives the side directly, so bilateral ORN
    #  axons split correctly between the two antennal lobes by location.
    if method in ("orn_outputs", "both"):
        print("  [A2] ORN presynaptic sites in AL …")
        syn = fetch_synapses(NC(bodyId=orn_ids),
                             SC(type="pre", rois=NP_AL_ROIS, primary_only=True),
                             client=c)
        df = pd.DataFrame({
            "glomerulus": syn["bodyId"].map(orn_glom).values,
            "side"      : syn["roi"].map(norm_side).values,
            "x": syn["x"].values, "y": syn["y"].values, "z": syn["z"].values,
        }).dropna(subset=["glomerulus"])
        df = df[df["side"].isin(["R", "L"])]
        df.to_csv(OUTPUT_DIR / "mcns_orn_outputs.csv", index=False)
        frames["orn_outputs"] = df
        print(f"    {len(df):,} ORN presynapses")

    return frames


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND B — FAFB / FlyWire via CAVEclient  (coords in nm)
# ══════════════════════════════════════════════════════════════════════════════

def run_flywire(method: str = "both") -> dict[str, pd.DataFrame]:
    from caveclient import CAVEclient

    print(f"\n{'─'*64}\n  FAFB  ·  {FW_DATASTACK}  ·  mat {FW_MAT_VER}\n{'─'*64}")
    cave = CAVEclient(FW_DATASTACK)          # cached CAVE secret → auth
    print("  Connected  ✓")

    # ── neuron typing from the Codex classification dump (v783 root_ids) ──────
    cls_path = _locate(FW_CLASSIFICATION)
    if not cls_path.exists():
        raise FileNotFoundError(f"FlyWire classification not found: {cls_path}")
    cls = pd.read_csv(cls_path, dtype={"root_id": "int64"})

    alpn = cls[(cls["class"] == "ALPN") &
               (cls["sub_class"] == "uniglomerular")].copy()   # uniglomerular only
    alpn["glomerulus"] = alpn["hemibrain_type"].astype(str).map(glom_from_alpn)
    alpn["side"]       = alpn["side"].map(norm_side)
    alpn = alpn.dropna(subset=["glomerulus"])
    alpn = alpn[alpn["glomerulus"].map(is_uniglom)]

    orn = cls[cls["hemibrain_type"].astype(str).str.startswith("ORN_")].copy()
    orn["glomerulus"] = orn["hemibrain_type"].astype(str).map(glom_from_orn)
    orn = orn.dropna(subset=["glomerulus"])
    orn = orn[orn["glomerulus"].map(is_uniglom)]
    orn_glom = dict(zip(orn["root_id"], orn["glomerulus"]))

    if GLOMERULUS_LIMIT:
        keep = sorted(set(alpn["glomerulus"]))[:GLOMERULUS_LIMIT]
        alpn = alpn[alpn["glomerulus"].isin(keep)]
        orn  = orn[orn["glomerulus"].isin(keep)]
        print(f"  [debug] limited to glomeruli: {keep}")

    # root_id → (glomerulus, side) for the unilateral ALPNs
    alpn_info = {r: (g, s) for r, g, s in
                 zip(alpn["root_id"], alpn["glomerulus"], alpn["side"])}
    alpn_ids = alpn["root_id"].tolist()
    orn_ids  = orn["root_id"].tolist()
    print(f"  ALPNs: {len(alpn_ids)} ({alpn['glomerulus'].nunique()} gloms)  |  "
          f"ORNs: {len(orn_ids)} ({orn['glomerulus'].nunique()} gloms)")

    def _query_batch(batch, filter_col):
        """One CAVE batch with retry/backoff — the materialization server
        intermittently drops large COPY queries with a 500."""
        for attempt in range(CAVE_RETRIES):
            try:
                return cave.materialize.query_table(
                    FW_SYN_TABLE,
                    filter_in_dict={filter_col: batch},
                    select_columns=["pre_pt_root_id", "post_pt_root_id",
                                    "pre_pt_position", "post_pt_position"],
                    materialization_version=FW_MAT_VER)
            except Exception as exc:
                if attempt == CAVE_RETRIES - 1:
                    raise
                wait = 2 ** attempt
                print(f"\n    [retry {attempt+1}/{CAVE_RETRIES-1} in {wait}s] "
                      f"{type(exc).__name__}: {str(exc)[:80]}")
                time.sleep(wait)

    def fetch_syn(id_list, filter_col):
        parts = []
        for i in range(0, len(id_list), BATCH_SIZE):
            batch = id_list[i:i + BATCH_SIZE]
            ch = _query_batch(batch, filter_col)
            for col in ("pre_pt_root_id", "post_pt_root_id"):
                ch[col] = ch[col].astype("int64")
            parts.append(ch)
            print(f"    … {min(i+BATCH_SIZE, len(id_list))}/{len(id_list)}",
                  end="\r")
        print()
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _frame(glom_series, coords) -> pd.DataFrame:
        """Assemble a [glomerulus, side, x, y, z] frame; side from x-location."""
        df = pd.DataFrame({
            "glomerulus": np.asarray(glom_series),
            "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2],
        }).dropna(subset=["glomerulus"])
        df["side"] = np.where(df["x"] < FW_AL_MIDLINE_X, "L", "R")
        return df.reset_index(drop=True)

    frames: dict[str, pd.DataFrame] = {}

    # ── B1: ORN → ALPN presynapses (ORN outputs, and AL bounds for clipping) ──
    #  ORN presynapses are AL-only; their per-(glomerulus, side) bounding box
    #  defines the antennal-lobe region used to clip the PN-input cloud below.
    #  Fetched whenever PN inputs are requested too, since FlyWire synapses carry
    #  no ROI tag (unlike neuprint) and we have no other AL spatial restriction.
    print("  [B1] ORN→ALPN presynaptic sites …")
    raw = fetch_syn(orn_ids, "pre_pt_root_id")
    bbox: dict = {}
    if len(raw):
        raw = raw[raw["post_pt_root_id"].isin(alpn_info)]
        # Glomerulus from the ORN's own identity (avoids cross-glomerular
        # ORN→PN synapses scattering points); side from the synapse location.
        coords = np.vstack(raw["pre_pt_position"].values).astype(np.float64)
        orn_df = _frame(raw["pre_pt_root_id"].map(orn_glom).values, coords)
        for (g, s), grp in orn_df.groupby(["glomerulus", "side"]):
            p = grp[["x", "y", "z"]].values
            lo, hi = p.min(0), p.max(0)
            mar = (hi - lo) * 0.10
            bbox[(g, s)] = (lo - mar, hi + mar)
        if method in ("orn_outputs", "both"):
            orn_df.to_csv(OUTPUT_DIR / "fafb_orn_outputs.csv", index=False)
            frames["orn_outputs"] = orn_df
            print(f"    {len(orn_df):,} ORN presynapses")

    # ── B2: ALPN postsynapses (PN inputs), clipped to each glomerulus's AL box ─
    if method in ("pn_inputs", "both"):
        print("  [B2] ALPN postsynaptic sites …")
        raw = fetch_syn(alpn_ids, "post_pt_root_id")
        if len(raw):
            coords = np.vstack(raw["post_pt_position"].values).astype(np.float64)
            df = _frame(raw["post_pt_root_id"].map(
                lambda r: alpn_info[r][0]).values, coords)
            if bbox:
                keep  = np.zeros(len(df), dtype=bool)
                pts   = df[["x", "y", "z"]].values
                gcol  = df["glomerulus"].values
                scol  = df["side"].values
                for (g, s), (lo, hi) in bbox.items():
                    m = (gcol == g) & (scol == s)
                    if m.any():
                        keep[m] = np.all((pts[m] >= lo) & (pts[m] <= hi), axis=1)
                n0 = len(df); df = df[keep].reset_index(drop=True)
                print(f"    AL-clip kept {len(df):,}/{n0:,} (dropped LH/calyx)")
            else:
                print("    [warn] no ORN bbox — PN inputs left un-clipped")
            df.to_csv(OUTPUT_DIR / "fafb_pn_inputs.csv", index=False)
            frames["pn_inputs"] = df
            print(f"    {len(df):,} synapses")

    return frames


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT + ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def export_hulls(glom: dict, tag: str):
    hull_dir = OUTPUT_DIR / f"{tag}_hull_meshes"
    hull_dir.mkdir(exist_ok=True)
    summary = {}
    for (g, side), d in glom.items():
        key   = f"{g}__AL_{side}"
        verts = d["pts"][d["hull"].vertices]
        np.save(hull_dir / f"{key}_vertices.npy", verts)
        np.save(hull_dir / f"{key}_faces.npy",    d["hull"].simplices)
        np.save(hull_dir / f"{key}_all_pts.npy",  d["raw_pts"])
        summary[key] = dict(
            glomerulus=g, side=side,
            centroid_nm=d["centroid_nm"].tolist(),
            volume_um3=round(d["vol_um3"], 3),
            surface_um2=round(d["area_um2"], 3),
            n_synapses=d["n_syn"],
            n_hull_verts=int(len(verts)),
            n_hull_faces=int(len(d["hull"].simplices)),
        )
    with open(hull_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    Mesh files → {hull_dir}/")
    return summary


def build_table(glom: dict, method: str) -> pd.DataFrame:
    rows = []
    for (g, side), d in glom.items():
        rows.append(dict(
            glomerulus=g, side=side, method=method,
            centroid_x_nm=round(d["centroid_nm"][0], 1),
            centroid_y_nm=round(d["centroid_nm"][1], 1),
            centroid_z_nm=round(d["centroid_nm"][2], 1),
            volume_um3=round(d["vol_um3"], 4),
            surface_um2=round(d["area_um2"], 4),
            n_synapses=d["n_syn"],
        ))
    return (pd.DataFrame(rows).sort_values(["side", "glomerulus"])
            if rows else pd.DataFrame(
                columns=["glomerulus", "side", "method", "volume_um3"]))


def plot_glomeruli_3d(glom: dict, title: str, filepath: Path,
                      alpha: float = 0.18, max_show: int = 80):
    if not glom:
        return
    items  = list(glom.items())[:max_show]
    colors = cm.tab20(np.linspace(0, 1, max(len(items), 1)))
    fig = plt.figure(figsize=(14, 11))
    ax  = fig.add_subplot(111, projection="3d")
    for color, ((g, side), d) in zip(colors, items):
        faces = [d["pts"][s] for s in d["hull"].simplices]
        ax.add_collection3d(Poly3DCollection(
            faces, alpha=alpha, facecolor=color, edgecolor=(0, 0, 0, 0.05)))
        ax.scatter(*d["centroid"], c=[color], s=28, zorder=5)
        ax.text(*d["centroid"], g, fontsize=5, ha="center")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"    3-D plot → {filepath.name}")


def plot_volume_comparison(pn_tbl, orn_tbl, label, side):
    m = pd.merge(
        pn_tbl[pn_tbl["side"] == side][["glomerulus", "volume_um3"]]
            .rename(columns={"volume_um3": "PN_input"}),
        orn_tbl[orn_tbl["side"] == side][["glomerulus", "volume_um3"]]
            .rename(columns={"volume_um3": "ORN_output"}),
        on="glomerulus", how="inner").sort_values("PN_input", ascending=False)
    if m.empty:
        return
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(max(10, len(m) * 0.45), 5))
    ax.bar(x - 0.2, m["PN_input"],   0.38, label="PN input (postsynaptic)",  color="#4E91C2")
    ax.bar(x + 0.2, m["ORN_output"], 0.38, label="ORN output (presynaptic)", color="#E07B54")
    ax.set_xticks(x); ax.set_xticklabels(m["glomerulus"], rotation=90, fontsize=6.5)
    ax.set_ylabel("Convex-hull volume (µm³)")
    ax.set_title(f"{label}  AL_{side}  — PN inputs vs ORN outputs")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = OUTPUT_DIR / f"{label.lower()}_AL_{side}_comparison.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"    Comparison plot → {out.name}")


def process(label: str, syn_frames: dict, coord_unit_nm: float) -> dict:
    print(f"\n  ── {label}: computing glomerulus volumes ──")
    tables: dict[str, pd.DataFrame] = {}
    for method, df in syn_frames.items():
        if df is None or df.empty:
            continue
        print(f"  [{method}]  {len(df):,} synapses  "
              f"({df['glomerulus'].nunique()} glomeruli, sides {sorted(df['side'].unique())})")
        glom = compute_glom_volumes(df, coord_unit_nm=coord_unit_nm)
        print(f"    → {len(glom)} (glomerulus, side) hulls computed")
        tag = f"{label}_{method}"
        tbl = build_table(glom, method)
        tbl.to_csv(OUTPUT_DIR / f"{tag}_volumes.csv", index=False)
        print(f"    CSV → {tag}_volumes.csv")
        export_hulls(glom, tag)
        for side in sorted(df["side"].unique()):
            sub = {k: v for k, v in glom.items() if k[1] == side}
            plot_glomeruli_3d(sub, title=f"{label}  AL_{side}  {method}",
                              filepath=OUTPUT_DIR / f"{tag}_AL_{side}.png")
        tables[method] = tbl

    if "pn_inputs" in tables and "orn_outputs" in tables:
        sides = sorted(set(pd.concat(tables.values())["side"]))
        for side in sides:
            plot_volume_comparison(tables["pn_inputs"], tables["orn_outputs"],
                                   label, side)
    return tables


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    all_results: dict = {}
    if "MCNS" in RUN_DATASETS:
        all_results["MCNS"] = process("MCNS", run_neuprint(SYNAPSE_METHOD),
                                      coord_unit_nm=NP_VOXEL_NM)
    if "FAFB" in RUN_DATASETS:
        all_results["FAFB"] = process("FAFB", run_flywire(SYNAPSE_METHOD),
                                      coord_unit_nm=1.0)

    print(f"\n{'═'*64}\n  Output directory: {OUTPUT_DIR.resolve()}\n{'═'*64}")
    for ds, tbls in all_results.items():
        for method, tbl in tbls.items():
            if len(tbl):
                print(f"  {ds:6s}  {method:14s}  {len(tbl):3d} (glom,side)   "
                      f"vol {tbl['volume_um3'].min():.1f}–{tbl['volume_um3'].max():.1f} µm³")
    return all_results


if __name__ == "__main__":
    main()
