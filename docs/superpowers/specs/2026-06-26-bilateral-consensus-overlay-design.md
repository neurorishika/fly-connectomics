# Bilateral Consensus + Cross-Dataset AL Overlay (ORN outputs)

**Date:** 2026-06-26
**Notebook:** `analysis/volume.ipynb`
**Scope:** ORN presynaptic output point clouds only (`*_orn_outputs.csv`), MCNS + FAFB.

## Goal

For each dataset, use bilateral symmetry to collapse the left and right antennal
lobes (ALs) into a single **consensus** AL per glomerulus, then place the MCNS
and FAFB consensus ALs together in **FAFB coordinate space** — MCNS where FAFB's
left AL sits, FAFB where its right AL sits — for direct, mirror-posed visual and
quantitative comparison.

## Bug fix (precondition)

`_locate()` references `Path(__file__)`, which is undefined when the module body
runs inside a Jupyter kernel → `NameError`. Guard the `__file__` base so the
function falls back to `cwd` / `cwd.parent` when `__file__` is absent. This is
what currently aborts `run_flywire()`.

## Inputs

Reuse the existing ORN-output frames (in-memory `frames["orn_outputs"]`, or the
on-disk `mcns_orn_outputs.csv` / `fafb_orn_outputs.csv`). Columns:
`glomerulus, side, x, y, z`.

- MCNS coordinates are in **8 nm voxels** → multiply by `NP_VOXEL_NM` to get nm.
- FAFB coordinates are already in **nm**.

All registration and volume math is done in **nm**, reported in **µm³**.

## Stage 1 — Per-dataset bilateral consensus

Run independently for MCNS and FAFB.

1. **Centroids:** per (glomerulus, side), compute the point-cloud centroid (nm).
2. **Mirror left → right frame:** reflect the left-AL points across the dataset
   midline plane (reflect the X axis about the AL midline; FAFB uses
   `FW_AL_MIDLINE_X`, MCNS uses the median X of its AL points). This yields the
   left glomeruli posed in a "pseudo-right" frame.
3. **Global affine registration (12-DOF):** fit ONE affine transform mapping the
   mirrored-left glomerulus centroids onto the matching right centroids, using
   only glomeruli present on **both** sides as correspondences (least-squares).
   Apply the same transform to every mirrored-left point. Report centroid RMSE
   before/after.
4. **Pool:** for each glomerulus, union the registered-left points with the
   native-right points → the consensus cloud, in the dataset's right-AL frame.

## Stage 2 — Exclusive consensus volumes

Enforces the requirement that **no region of space belongs to two glomeruli**.

1. **Voxel grid:** isotropic voxels over the AL bounding box. Default
   `VOXEL_UM = 2.0` (tunable).
2. **Occupancy:** each glomerulus consensus cloud → set of occupied voxels
   (voxel contains ≥1 point), with a light binary morphological closing to fill
   the cloud interior (so the volume reflects the filled glomerular region, not
   just synapse-occupied voxels).
3. **Resolve contested voxels:** a voxel claimed by >1 glomerulus is assigned to
   the glomerulus with the **nearest centroid**. Each voxel ends up owned by
   exactly one glomerulus.
4. **Volume:** `exclusive_voxel_count × VOXEL_UM³` per glomerulus.

This replaces the convex-hull volume with an exclusive, partition-based volume.

## Stage 3 — Cross-dataset overlay in FAFB space

1. **MCNS → FAFB affine (12-DOF):** fit one affine mapping MCNS consensus
   centroids → FAFB consensus centroids (shared glomeruli only). Apply to all
   MCNS consensus voxels/points → MCNS posed in FAFB's right-AL frame. Report
   RMSE.
2. **Place the two ALs:** keep FAFB consensus at its native (right-AL) location.
   Mirror the FAFB-frame MCNS consensus back across the FAFB midline so it lands
   where FAFB's **left** AL sits.
3. Result: one FAFB-coordinate space — FAFB-derived AL on the right, MCNS-derived
   AL on the left, mirror-posed and directly comparable.

## Outputs (to `OUTPUT_DIR`)

- `consensus_volumes.csv` — `dataset, glomerulus, volume_um3, n_voxels,
  centroid_x_nm, centroid_y_nm, centroid_z_nm`.
- Registration QC printed + saved (`consensus_registration_qc.json`): per-step
  centroid RMSE (MCNS L→R, FAFB L→R, MCNS→FAFB) and n correspondences.
- `consensus_overlay_3d.png` — combined FAFB-space 3-D plot (MCNS left /
  FAFB right), colored by glomerulus.
- `consensus_volume_scatter.png` — per-glomerulus MCNS vs FAFB exclusive volume.

## Design choices (locked)

- Consensus = **union of both sides, made spatially exclusive** (no region in two
  glomeruli).
- Within-dataset alignment = **global transform from glomerulus centroids**.
- Transform class = **full affine (12-DOF)** for both within- and cross-dataset.
- Correspondences = glomeruli present on both sides / in both datasets only.

## Non-goals

- PN-input clouds (this analysis is ORN outputs only).
- BANC dataset.
- Non-rigid / deformable registration.
- Changing the existing per-side hull pipeline (Stages run in new cells alongside it).
