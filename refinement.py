"""
ioah3.refinement
================
Stage 3 of IOAH3: adaptive hierarchical refinement.

For each included cell, assigns a target H3 resolution based on importance
quantiles (paper eq. 5), propagates refinement to k-ring-1 neighbours to
avoid isolated fine cells, then replaces each coarse cell with its H3
children at the target resolution.

Paper reference: Section 3.3 (Adaptive Hierarchical Refinement).
"""

import numpy as np
import pandas as pd
import h3
from pathlib import Path
from tqdm import tqdm

from .config import (
    CACHE_DIR,
    COARSE_RES,
    DEM_RASTER_TIF,
    HAZARD_RASTER_TIF,
    MAX_REFINEMENT_RES,
    POP_RASTER_TIF,
    USE_VIENNA_ONLY,
    VIENNA_BOUNDS,
    DEM_DOWNSAMPLE_FACTOR,
)


def _assign_target_resolutions(important: pd.DataFrame) -> pd.DataFrame:
    """
    Assign target H3 resolution per cell using importance quantiles
    (paper eq. 5):

        r*(h) = r_max          if s(h) > q_80
              = 9              if q_65 < s(h) ≤ q_80
              = r_0            otherwise

    where q_40, q_65, q_80 are percentiles of s(h) over included cells.
    """
    score_s = important["importance_score"]
    quantiles = np.percentile(score_s, [40, 65, 80])
    conditions = [score_s > quantiles[2], score_s > quantiles[1]]
    choices    = [MAX_REFINEMENT_RES, 9]
    important = important.copy()
    important["target_res"] = np.select(conditions, choices, default=COARSE_RES)
    print(f"    Quantiles q40={quantiles[0]:.4f}  q65={quantiles[1]:.4f}  q80={quantiles[2]:.4f}")
    return important


def _propagate_refinement(refinement_targets: dict) -> dict:
    """
    Neighbour propagation (paper Section 3.3):
    for every cell with r*(h) > r_0, promote all k-ring-1 neighbours to
    at least r*(h).  This prevents isolated fine-resolution cells.
    """
    expanded = refinement_targets.copy()
    for h, r in list(refinement_targets.items()):
        if r > COARSE_RES:
            try:
                for n in map(str, h3.k_ring(h, 1)):
                    if n != h:
                        expanded[n] = max(expanded.get(n, 0), r)
            except Exception:
                pass
    return expanded


def automatic_hierarchical_refinement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Stage 3 of IOAH3.

    1. Pre-load fine-resolution rasters once (pop, elev, haz at MAX_REFINEMENT_RES).
    2. Assign target resolutions per cell (eq. 5).
    3. Propagate to neighbours.
    4. Replace each coarse cell by its H3 children; sample child features
       from pre-loaded fine raster lookup tables.

    Returns expanded DataFrame with per-cell columns:
        h3, pop, mean_elev, elev_roughness, haz, resolution,
        importance_score, included, excluded.
    """
    from .features import raster_to_h3

    print("\n" + "=" * 70)
    print("STAGE 3: ADAPTIVE HIERARCHICAL REFINEMENT")
    print("=" * 70)

    important = df[df["included"]].copy()
    if len(important) == 0:
        return pd.DataFrame()

    # Pre-load fine lookup tables once
    print("\n  🚀 Pre-loading fine rasters …")
    bbox = VIENNA_BOUNDS if USE_VIENNA_ONLY else None

    pop_fine  = raster_to_h3(POP_RASTER_TIF,  MAX_REFINEMENT_RES, stat="sum",  bbox=bbox)
    elev_fine = raster_to_h3(DEM_RASTER_TIF,  MAX_REFINEMENT_RES, stat="mean",
                              downsample=DEM_DOWNSAMPLE_FACTOR, bbox=bbox)
    haz_fine  = raster_to_h3(HAZARD_RASTER_TIF, MAX_REFINEMENT_RES, stat="mean", bbox=bbox)

    pop_dict            = dict(zip(pop_fine["h3"].astype(str),  pop_fine["val"]))
    mean_elev_dict      = dict(zip(elev_fine["h3"].astype(str), elev_fine["mean_elev"]))
    elev_roughness_dict = dict(zip(elev_fine["h3"].astype(str), elev_fine["elev_roughness"]))
    haz_dict            = dict(zip(haz_fine["h3"].astype(str),  haz_fine["val"]))
    print("  ✅ Fine lookup tables ready")

    # Assign and propagate target resolutions
    important = _assign_target_resolutions(important)
    refinement_targets = dict(zip(important["h3"], important["target_res"]))
    expanded_targets   = _propagate_refinement(refinement_targets)

    df_indexed = df.set_index("h3")
    res_dist   = {COARSE_RES: 0, 9: 0, MAX_REFINEMENT_RES: 0}
    refined_rows = []

    for h, target_res in tqdm(expanded_targets.items(), desc="    Refining"):
        if h not in df_indexed.index:
            continue
        row   = df_indexed.loc[h]
        score = float(row.get("importance_score", 0.0))

        if target_res > COARSE_RES:
            try:
                children = h3.h3_to_children(h, target_res)
                for child in map(str, children):
                    refined_rows.append({
                        "h3":               child,
                        "pop":              pop_dict.get(child,            float(row.get("pop", 0))),
                        "mean_elev":        mean_elev_dict.get(child,      float(row.get("mean_elev", 0.0))),
                        "elev_roughness":   elev_roughness_dict.get(child, float(row.get("elev_roughness", 0.0))),
                        "haz":              haz_dict.get(child,            float(row.get("haz", 0))),
                        "resolution":       target_res,
                        "importance_score": score,
                        "included":         True,
                        "excluded":         False,
                    })
                    res_dist[target_res] += 1
            except Exception:
                refined_rows.append({**row.to_dict(), "h3": h,
                                     "resolution": COARSE_RES,
                                     "included": True, "excluded": False})
                res_dist[COARSE_RES] += 1
        else:
            refined_rows.append({**row.to_dict(), "h3": h,
                                 "resolution": COARSE_RES,
                                 "included": True, "excluded": False})
            res_dist[COARSE_RES] += 1

    # Append excluded cells at coarse resolution (background context)
    excluded_records               = df[df["excluded"]].copy()
    excluded_records["resolution"] = COARSE_RES
    excluded_records["included"]   = False
    excluded_records["excluded"]   = True
    refined_rows.extend(excluded_records.to_dict("records"))
    res_dist[COARSE_RES] += len(excluded_records)

    df_result = pd.DataFrame(refined_rows)
    print(
        f"\n  Res {COARSE_RES}: {res_dist[COARSE_RES]:,} | "
        f"Res 9: {res_dist[9]:,} | Res {MAX_REFINEMENT_RES}: {res_dist[MAX_REFINEMENT_RES]:,}"
    )
    return df_result
