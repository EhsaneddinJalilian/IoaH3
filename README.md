# IOAH3: Importance-Driven Adaptive Spatial Partitioning

> **IOAH3: Importance-Driven Adaptive Spatial Partitioning via Graph-Cut Optimisation over H3 Hierarchical Grids**  

## Overview

IOAH3 constructs data-driven, multi-resolution spatial partitions of geo-referenced domains
using [Uber H3](https://h3geo.org/) hexagonal grids. It addresses the Modifiable Areal Unit
Problem (MAUP) by replacing fixed-resolution grids with importance-adaptive partitions that
concentrate fine cells where informational content is highest.

The pipeline runs in three stages:

| Stage | Module | Paper Section |
|-------|--------|---------------|
| 1. Feature extraction + PCA importance scoring | `ioah3/features.py` | §3.1 |
| 2. MRF graph-cut cell selection | `ioah3/graphcut.py` | §3.2 |
| 3. Adaptive hierarchical refinement | `ioah3/refinement.py` | §3.3 |

## Repository Structure

```
ioah3/
├── config.py          # paths and flags
├── features.py        # Stage 1: OSM extraction, raster→H3, PCA scoring
├── graphcut.py        # Stage 2: MRF energy + max-flow solver
├── refinement.py      # Stage 3: hierarchical refinement + propagation
├── pipeline.py        # top-level orchestrator
├── checkpoint.py      # save/load viz checkpoint
└── visualization.py   # five-layer Folium map

scripts/
└── run_ioah3.py       # CLI entry point
```

## Installation

```bash
pip install -r requirements.txt
```

osmium-tool must also be available for OSM PBF parsing:
```bash
# Ubuntu / Debian
sudo apt install osmium-tool python3-osmium
```

## Data Requirements

Place the following files as configured in `ioah3/config.py`:

| File | Description |
|------|-------------|
| `data/raw/population_worldpop.tif` | WorldPop population raster |
| `data/raw/Elevation_dhm_at_lamb_10m_2018.tif` | DEM (any CRS; auto-reprojected) |
| `data/raw/flood_prob.tif` | Hazard probability raster |
| `data/raw/osm/austria-260128.osm.pbf` | OSM PBF extract |

## Usage

```bash
# Full pipeline
python scripts/run_ioah3.py

# Re-render map from checkpoint only
python scripts/run_ioah3.py --viz-only

# Restrict to Vienna bounding box
python scripts/run_ioah3.py --vienna-only

# Force re-computation of elevation cache
python scripts/run_ioah3.py --force-elev
```

Or from Python:

```python
from ioah3 import run
substrate = run()
```

## Outputs

All outputs are written to `data/processed/`:

| File | Description |
|------|-------------|
| `substrate.parquet` | Included cells with per-cell resolution and features |
| `substrate_excluded.parquet` | Excluded (background) cells |
| `substrate_map.html` | Interactive five-layer Folium map |
| `pillar1_spatial_graph.json` | Spatial constraint graph |
| `viz_*.parquet` / `.pkl` | Visualisation checkpoint |

## Notes on Elevation Roughness

The DEM is reprojected to EPSG:4326 before coarse-resolution aggregation. Bilinear
resampling during reprojection suppresses high-frequency slope detail, causing the
coarse roughness to read as zero. The pipeline automatically detects this and
aggregates roughness from the fine-resolution (res-10) cache, which is computed from
the original Lambert-projected DEM.

## Citation

If you use this code, please cite the accompanying preprint (link to be added after upload).

## License

MIT
