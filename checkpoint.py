"""
ioah3.checkpoint
================
Save and load the visualisation checkpoint so the map can be re-rendered
without re-running the full pipeline (VIZ_ONLY mode).
"""

import pickle
from pathlib import Path

import pandas as pd

from .config import CACHE_DIR


_FILES = {
    "df_combined":      CACHE_DIR / "viz_df_combined.parquet",
    "df_cells":         CACHE_DIR / "viz_df_cells.parquet",
    "elev_result":      CACHE_DIR / "viz_elev_result.parquet",
    "haz_result":       CACHE_DIR / "viz_haz_result.parquet",
    "road_h3_set":      CACHE_DIR / "viz_road_h3_set.pkl",
    "elev_h3_polygons": CACHE_DIR / "viz_elev_h3_polygons.pkl",
}


def save_checkpoint(df_combined, df_cells, elev_result, haz_result,
                    road_h3_set, elev_h3_polygons):
    print("\n   Saving checkpoint …")
    df_combined.to_parquet(_FILES["df_combined"])
    df_cells.to_parquet(_FILES["df_cells"])
    elev_result.to_parquet(_FILES["elev_result"])
    haz_result.to_parquet(_FILES["haz_result"])
    with open(_FILES["road_h3_set"], "wb") as f:
        pickle.dump(road_h3_set, f)
    with open(_FILES["elev_h3_polygons"], "wb") as f:
        pickle.dump(elev_h3_polygons, f)
    print(f"   Checkpoint saved to {CACHE_DIR}")


def load_checkpoint():
    missing = [str(p) for p in _FILES.values() if not p.exists()]
    if missing:
        print("    Checkpoint incomplete — missing:")
        for m in missing:
            print(f"       {m}")
        return None
    print("   Loading checkpoint …")
    df_combined      = pd.read_parquet(_FILES["df_combined"])
    df_cells         = pd.read_parquet(_FILES["df_cells"])
    elev_result      = pd.read_parquet(_FILES["elev_result"])
    haz_result       = pd.read_parquet(_FILES["haz_result"])
    with open(_FILES["road_h3_set"], "rb") as f:
        road_h3_set = pickle.load(f)
    with open(_FILES["elev_h3_polygons"], "rb") as f:
        elev_h3_polygons = pickle.load(f)
    return df_combined, df_cells, elev_result, haz_result, road_h3_set, elev_h3_polygons
