"""
ioah3.pipeline
==============
Top-level IOAH3 pipeline: orchestrates Stages 1–3 and saves outputs.

Usage
-----
    from ioah3.pipeline import run
    substrate = run()

Or from the command line via scripts/run_ioah3.py.

Paper reference: Algorithm 1 (full pipeline pseudocode).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    CACHE_DIR,
    COARSE_RES,
    DEM_RASTER_TIF,
    DEM_DOWNSAMPLE_FACTOR,
    HAZARD_RASTER_TIF,
    MAX_REFINEMENT_RES,
    OSM_PBF,
    POP_RASTER_TIF,
    USE_VIENNA_ONLY,
    VIZ_ONLY,
    VIENNA_BOUNDS,
)
from .features import (
    ensure_raster_4326,
    extract_osm_infrastructure,
    raster_to_h3,
    patch_coarse_roughness,
    compute_pca_importance,
)
from .graphcut import compute_importance_graph_cut, remove_isolated_inclusions
from .refinement import automatic_hierarchical_refinement
from .checkpoint import save_checkpoint, load_checkpoint


def run() -> pd.DataFrame:
    """
    Run the full IOAH3 pipeline and return the final substrate DataFrame.

    Stages
    ------
    1. Feature extraction + PCA importance scoring
    2. MRF graph-cut cell selection
    3. Adaptive hierarchical refinement
    """
    print("\n" + "=" * 70)
    print("IOAH3 PIPELINE")
    print(f"MODE: {'Vienna Only' if USE_VIENNA_ONLY else 'Whole Domain'}")
    print("=" * 70)

    # ── VIZ_ONLY fast path ────────────────────────────────────────────────────
    if VIZ_ONLY:
        print("\n  ⚡ VIZ_ONLY — loading checkpoint …")
        ck = load_checkpoint()
        if ck is not None:
            df_combined, df_cells, elev_result, haz_result, road_h3_set, elev_h3_polygons = ck
            _patch_checkpoint_roughness(elev_result)
            from .visualization import create_map
            create_map(df_combined, df_cells, road_h3_set=road_h3_set,
                       elev_result=elev_result, haz_result=haz_result,
                       elev_h3_polygons=elev_h3_polygons)
            return df_combined
        print("  ⚠️  Checkpoint missing — running full pipeline …\n")

    start = time.time()
    bbox  = VIENNA_BOUNDS if USE_VIENNA_ONLY else None

    # ── Reproject rasters to EPSG:4326 ───────────────────────────────────────
    POP_4326 = ensure_raster_4326(POP_RASTER_TIF)
    DEM_4326 = ensure_raster_4326(DEM_RASTER_TIF)
    HAZ_4326 = ensure_raster_4326(HAZARD_RASTER_TIF)

    # ── Stage 1a: OSM extraction ──────────────────────────────────────────────
    infra_df    = extract_osm_infrastructure(OSM_PBF, coarse_res=COARSE_RES)
    road_h3_set = set(infra_df[infra_df["road"]     > 0]["h3"].astype(str))
    poi_h3_set  = set(infra_df[infra_df["poi"]      > 0]["h3"].astype(str))
    bldg_h3_set = set(infra_df[infra_df["building"] > 0]["h3"].astype(str))

    # ── Stage 1b: Raster aggregation ─────────────────────────────────────────
    pop_result  = raster_to_h3(POP_4326, COARSE_RES, stat="sum",  bbox=bbox)
    elev_result = raster_to_h3(DEM_4326, COARSE_RES, stat="mean",
                                downsample=DEM_DOWNSAMPLE_FACTOR, bbox=bbox)
    haz_result  = raster_to_h3(HAZ_4326, COARSE_RES, stat="mean", bbox=bbox)

    # Roughness fix: aggregate from fine cache if coarse roughness is zero
    elev_result = _maybe_patch_roughness(elev_result, DEM_4326)

    # ── Build coarse feature table ────────────────────────────────────────────
    df_cells = _build_cell_table(pop_result, elev_result, haz_result, infra_df)

    # ── Stage 1c: PCA importance scoring ─────────────────────────────────────
    df_pca = _build_pca_table(infra_df, df_cells)
    unary_score = compute_pca_importance(df_pca)

    # ── Stage 2: Graph-cut cell selection ────────────────────────────────────
    df_importance = compute_importance_graph_cut(
        df_pca, road_h3_set, poi_h3_set, unary_score)
    df_importance = remove_isolated_inclusions(df_importance)

    # ── Stage 3: Hierarchical refinement ─────────────────────────────────────
    df_combined = automatic_hierarchical_refinement(df_importance)

    # Post-refinement roughness patch (fine cache now available)
    elev_result = _maybe_patch_roughness(elev_result, DEM_4326, post=True)

    # ── Save checkpoint and outputs ───────────────────────────────────────────
    from .visualization import prefetch_h3_polygons, create_map
    elev_h3_polygons = prefetch_h3_polygons(elev_result["h3"].astype(str).tolist())

    save_checkpoint(df_combined, df_cells, elev_result, haz_result,
                    road_h3_set, elev_h3_polygons)
    _save_substrate(df_combined)

    create_map(df_combined, df_cells, road_h3_set=road_h3_set,
               elev_result=elev_result, haz_result=haz_result,
               elev_h3_polygons=elev_h3_polygons)

    print(f"\n✅ IOAH3 COMPLETE  ({time.time() - start:.1f}s)")
    print(f"  Cells: {len(df_combined):,}")
    return df_combined


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _maybe_patch_roughness(elev_result, dem_4326_path, post=False):
    rough_max = (float(elev_result["elev_roughness"].max())
                 if "elev_roughness" in elev_result.columns else 0.0)
    if rough_max > 0:
        return elev_result
    label = "Post-refinement" if post else "Pre-refinement"
    print(f"\n  ⚠️  {label}: coarse roughness=0 — attempting patch …")
    fine_cache = CACHE_DIR / f"{Path(DEM_RASTER_TIF).stem}_h3_res{MAX_REFINEMENT_RES}.parquet"
    if fine_cache.exists():
        elev_result = patch_coarse_roughness(elev_result, fine_cache)
        coarse_cache = CACHE_DIR / f"{Path(dem_4326_path).stem}_h3_res{COARSE_RES}.parquet"
        elev_result.to_parquet(coarse_cache)
        print(f"  💾 Patched coarse cache saved: {coarse_cache.name}")
    else:
        print(f"  ℹ️  Fine cache not yet available ({fine_cache.name})")
    return elev_result


def _patch_checkpoint_roughness(elev_result):
    if ("elev_roughness" in elev_result.columns
            and float(elev_result["elev_roughness"].max()) > 0):
        return elev_result
    fine_cache = CACHE_DIR / f"{Path(DEM_RASTER_TIF).stem}_h3_res{MAX_REFINEMENT_RES}.parquet"
    return patch_coarse_roughness(elev_result, fine_cache)


def _build_cell_table(pop_result, elev_result, haz_result, infra_df):
    import h3 as _h3

    df_cells       = pop_result[["h3", "val"]].copy().rename(columns={"val": "pop"})
    df_cells["h3"] = df_cells["h3"].astype(str)

    elev_idx = elev_result.set_index("h3")
    haz_map  = dict(zip(haz_result["h3"].astype(str), haz_result["val"]))

    df_cells["mean_elev"]      = df_cells["h3"].map(elev_idx["mean_elev"]).fillna(0)
    df_cells["elev_roughness"] = df_cells["h3"].map(elev_idx["elev_roughness"]).fillna(0)
    df_cells["haz"]            = df_cells["h3"].map(haz_map).fillna(0)

    infra_h3_str           = infra_df["h3"].astype(str)
    df_cells["poi_count"]  = df_cells["h3"].map(dict(zip(infra_h3_str, infra_df["poi"]))).fillna(0)
    df_cells["bldg_count"] = df_cells["h3"].map(dict(zip(infra_h3_str, infra_df["building"]))).fillna(0)
    df_cells["road_count"] = df_cells["h3"].map(dict(zip(infra_h3_str, infra_df["road"]))).fillna(0)

    # Remove cells with no signal
    has_infra = (
        (df_cells["road_count"] > 0)
        | (df_cells["poi_count"]  > 0)
        | (df_cells["bldg_count"] > 0)
    )
    df_cells = df_cells[
        has_infra | (df_cells["pop"] > 0) | (df_cells["haz"] > 0.001)
    ].copy()

    # Normalised density features (paper Section 3.1)
    df_cells["cell_area"]        = df_cells["h3"].map(
        lambda h: _h3.cell_area(h, unit="km^2") + 1e-9)
    df_cells["road_density"]     = df_cells["road_count"]  / df_cells["cell_area"]
    df_cells["building_density"] = df_cells["bldg_count"]  / df_cells["cell_area"]
    df_cells["poi_density"]      = df_cells["poi_count"]   / df_cells["cell_area"]

    def _norm(col): return col / (col.max() + 1e-8)
    df_cells["road_density_norm"]     = _norm(df_cells["road_density"])
    df_cells["building_density_norm"] = _norm(df_cells["building_density"])
    df_cells["poi_density_norm"]      = _norm(df_cells["poi_density"])

    return df_cells


def _build_pca_table(infra_df, df_cells):
    df_pca       = infra_df.copy()
    df_pca["h3"] = df_pca["h3"].astype(str)
    df_pca = df_pca.merge(
        df_cells[[
            "h3", "pop", "haz", "elev_roughness",
            "road_density_norm", "building_density_norm", "poi_density_norm",
        ]],
        on="h3", how="left",
    ).fillna(0.0)
    return df_pca


def _save_substrate(df_combined):
    cache_file = CACHE_DIR / "substrate.parquet"
    incl_mask  = df_combined["included"].astype(bool)
    excl_mask  = df_combined["excluded"].astype(bool)

    df_excl = df_combined[excl_mask].copy()
    if len(df_excl) == 0:
        # If MRF included everything, use coarse-only cells as proxy
        coarse_only = df_combined["resolution"] == COARSE_RES
        df_excl = df_combined[coarse_only].copy()
        df_excl["excluded"] = True; df_excl["included"] = False

    df_excl.to_parquet(CACHE_DIR / "substrate_excluded.parquet")
    df_incl = df_combined[incl_mask] if incl_mask.any() else df_combined
    df_incl.to_parquet(cache_file)
    print(f"  💾 Included: {len(df_incl):,}  Excluded: {len(df_excl):,}")
