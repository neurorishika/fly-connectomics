# Uniform ORN/PN/LN Lateralization Across MCNS / FAFB / BANC

Date: 2026-06-13

## Goal

A single self-contained notebook (`analysis/ORN-lateralization-multi.ipynb`) that computes
three lateralization metrics identically across the three public Drosophila connectome
datasets — Male-CNS (MCNS, neuPrint), FAFB (FlyWire), and BANC — from local data files,
and produces directly comparable cross-dataset figures plus a combined tidy table.

## Metrics (all are contra/ipsi synapse-count ratios, per ORN type)

1. **ORN axon laterality** — for each ORN, (output synapses in the *contralateral* AL) /
   (output synapses in the *ipsilateral* AL). Ipsi AL = the AL on the ORN's soma side.
   Reference frame = **AL neuropil side** (option A).
2. **ALPN input bias (P_PN)** — per ORN type, pool all ORN→ALPN synapses and compute
   (synapses from the *contra* antenna) / (synapses from the *ipsi* antenna).
   Ipsi = ORN soma side == the **AL the synapse sits in** (the PN's dendrite side).
3. **ALLN input bias (P_LN)** — identical to (2) for ALLNs.

Metrics 2 & 3 reference the **AL neuropil side of each synapse**, NOT the post-neuron's
soma side. This matters for bilateral PNs (e.g. `V_ilPN`, `VL1_ilPN`) whose soma sits
contralateral to their dendrites: a soma-side reference miscounts their ipsilateral ORN
input as "contra". FAFB/BANC have per-edge `neuropil`; MCNS lacks it, so MCNS P_PN/P_LN use
a one-time cached neuPrint fetch (`data/Male_CNS/orn_to_alpnln_AL_roi_adj.feather`) of every
ORN→ALPN/ALLN connection resolved by AL(L)/AL(R).

Ratio convention: > 1 = contra-dominant, = 1 = symmetric, < 1 = ipsi-dominant.
A small epsilon (1e-9) guards divide-by-zero; ratios where ipsi == 0 are flagged.

## Data sources (all local)

| Dataset | Neuron metadata | Edge table | ORN id / type | Side | ALPN class | ALLN class |
|---|---|---|---|---|---|---|
| MCNS | `data/Male_CNS/neurons.pkl` (`[0]` df, has `roiInfo`) | `connectome-weights-male-cns-v0.9-minconf-0.5.feather` (`body_pre,body_post,weight`) | `flywireType` `ORN_*` | `rootSide`→`somaSide` | `class=="ALPN"` | `class=="ALLN"` |
| FAFB | `data/FlyWire/classification.csv.gz` | `connections_princeton_no_threshold.csv.gz` (`pre_root_id,post_root_id,neuropil,syn_count`) | `hemibrain_type` `ORN_*` | `side` | `class=="ALPN"` | `class=="ALLN"` |
| BANC | `data/BANC/neurons.pickle.gz` (dict: root_id→attrs) | `connections_princeton.csv.gz` (same cols) | `class=="olfactory_receptor_neuron"`, glom from `resolved_type` `ORN_*` | `side` | `class=="antennal_lobe_projection_neuron"` | `..._local_neuron` |

Metric 1 attribution by AL neuropil:
- MCNS: `roiInfo['AL(L)'/'AL(R)']['downstream']` (output synapse counts per AL).
- FAFB/BANC: edge `neuropil` in {`AL_L`,`AL_R`}, summed `syn_count`.

## Architecture

- `norm_side()` maps `L/R/left/right` → `left/right`; `M/midline/unknown` → drop.
- `glom()` strips the `ORN_` prefix → glomerulus label (e.g. `ORN_DA1` → `DA1`).
- `load_mcns()`, `load_fafb()`, `load_banc()` each return two standardized frames:
  - **metric1 frame**: `dataset, neuron_id, glomerulus, side, ipsi_syn, contra_syn`
  - **edges23 frame**: `dataset, glomerulus, post_role∈{ALPN,ALLN}, weight, ipsi(bool)`
- ORNs are identified uniformly as type names starting with `ORN_` (with a valid side).
- Shared `summarize_*()` functions pool the frames per (dataset, glomerulus[, post_role])
  into ipsi/contra totals and a ratio. The math is dataset-agnostic.
- Glomerulus harmonization keeps the **intersection of glomeruli present in all three**
  datasets for the comparison figures; the full table is saved unfiltered.

## Outputs

- `analysis/ORN-lateralization-multi.csv` — tidy: one row per
  `dataset, glomerulus, metric ∈ {ORN_axon, ALPN_input, ALLN_input}` with
  `ipsi_syn, contra_syn, ratio, n` (n = #ORNs or #edges contributing).
- Figures (matplotlib/seaborn):
  - 3 grouped bar plots (one per metric): x = glomerulus (common set), hue = dataset,
    y = contra/ipsi ratio on a log scale with a reference line at 1.
  - Metric-1 per-ORN distribution box+strip, faceted by dataset (matches existing style).

## Non-goals

- No programmatic/live fetching (no CAVE token); local files only.
- No partner-count or (R−L)/(R+L) index — synapse counts and the simple ratio only.
- No glomeruli outside the common-to-all-three set in the comparison figures.
