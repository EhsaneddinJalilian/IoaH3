"""
ioah3.config
============
Central configuration for the IOAH3 pipeline.

Edit the paths and flags here before running.
"""

from pathlib import Path

# ── Data paths ────────────────────────────────────────────────────────────────
POP_RASTER_TIF    = Path(r"C:\work/My_papers/GeoData/data/raw/population_worldpop.tif")
DEM_RASTER_TIF    = Path(r"C:\work/My_papers/GeoData/data/raw/Elevation_dhm_at_lamb_10m_2018.tif")
HAZARD_RASTER_TIF = Path(r"C:\work/My_papers/GeoData/data/raw/flood_prob.tif")
OSM_PBF           = Path(r"C:\work/My_papers/GeoData/data/raw/osm/austria-260128.osm.pbf")

CACHE_DIR = Path("data/processed")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Resolution settings ───────────────────────────────────────────────────────
COARSE_RES            = 7    # base H3 resolution (≈ 5.16 km²)
MAX_REFINEMENT_RES    = 10   # maximum refinement H3 resolution (≈ 0.015 km²)

# ── Processing settings ───────────────────────────────────────────────────────
N_WORKERS             = 2    # parallel raster workers
DEM_DOWNSAMPLE_FACTOR = 6    # downsampling factor for mean_elev pass
SLOPE_SAMPLE_STEP     = 3    # pixel stride for roughness sampling

# ── Geographic filter ─────────────────────────────────────────────────────────
VIENNA_BOUNDS = {
    "min_lat": 47.70, "max_lat": 48.50,
    "min_lon": 14.00, "max_lon": 16.70,
}
USE_VIENNA_ONLY = False       # True = restrict to VIENNA_BOUNDS

# ── Workflow flags ────────────────────────────────────────────────────────────
VIZ_ONLY             = False  # True = load checkpoint and re-render map only
FORCE_RECOMPUTE_ELEV = False  # True = delete stale elevation parquet and recompute
