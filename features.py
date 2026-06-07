"""
ioah3.features
==============
Stage 1 of IOAH3: multi-source feature extraction and PCA importance scoring.

Covers:
  - OSM road / POI / building density extraction via osmium streaming
  - Raster → H3 aggregation (population, elevation, hazard)
  - Elevation roughness fix: aggregate from fine (res-10) to coarse (res-7)
  - PCA importance scoring over the four standardised feature signals

Paper reference: Section 3.1 (Feature Extraction and Importance Scoring).
"""

import time
import warnings
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import rasterio
import rasterio.enums
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.transform import xy
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import from_bounds as window_from_bounds
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .config import (
    CACHE_DIR,
    COARSE_RES,
    DEM_DOWNSAMPLE_FACTOR,
    MAX_REFINEMENT_RES,
    N_WORKERS,
    SLOPE_SAMPLE_STEP,
    USE_VIENNA_ONLY,
    VIENNA_BOUNDS,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Raster reprojection helper
# ---------------------------------------------------------------------------

def ensure_raster_4326(src_raster_path: Path) -> Path:
    """Reproject *src_raster_path* to EPSG:4326 if needed; return path."""
    src_raster_path = Path(src_raster_path)
    reprojected_file = CACHE_DIR / f"{src_raster_path.stem}_EPSG4326.tif"
    if reprojected_file.exists():
        print(f"  ✓ Cached: {reprojected_file.name}")
        return reprojected_file
    try:
        with rasterio.open(src_raster_path) as src:
            if src.crs and src.crs.to_epsg() == 4326:
                return src_raster_path
            print(f"  🔄 Reprojecting: {src_raster_path.name}")
            dst_crs = "EPSG:4326"
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({"crs": dst_crs, "transform": transform,
                           "width": width, "height": height})
            with rasterio.open(reprojected_file, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                    )
            return reprojected_file
    except Exception as e:
        print(f"  ❌ Reprojection failed: {e}")
        return None


# ---------------------------------------------------------------------------
# OSM extraction
# ---------------------------------------------------------------------------

def extract_osm_infrastructure(pbf_path: Path, coarse_res: int = COARSE_RES) -> pd.DataFrame:
    """
    Stream an OSM PBF file with osmium and aggregate road / POI / building
    counts per H3 cell at *coarse_res*.

    Returns DataFrame with columns: h3, road, poi, building.
    """
    import osmium

    start_time = time.time()
    pbf_path = str(pbf_path)
    cache_name = "vienna" if USE_VIENNA_ONLY else "full"
    cache_file = CACHE_DIR / f"osm_infra_ultrafast_{cache_name}_res{coarse_res}.parquet"

    if cache_file.exists():
        print(f"  ✓ Cached OSM infrastructure ({cache_name})")
        return pd.read_parquet(cache_file)

    print("  ⚡ Streaming OSM → H3 …")
    important_highways = {
        "motorway", "trunk", "primary", "secondary",
        "tertiary", "unclassified", "residential",
    }

    if USE_VIENNA_ONLY:
        _min_lon = VIENNA_BOUNDS["min_lon"]; _max_lon = VIENNA_BOUNDS["max_lon"]
        _min_lat = VIENNA_BOUNDS["min_lat"]; _max_lat = VIENNA_BOUNDS["max_lat"]
        def in_bbox(lon, lat):
            return _min_lon <= lon <= _max_lon and _min_lat <= lat <= _max_lat
    else:
        def in_bbox(lon, lat):
            return True

    class InfraHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.counts = defaultdict(lambda: [0, 0, 0])
            self.processed = 0

        def node(self, n):
            if not n.location.valid():
                return
            lon, lat = n.location.lon, n.location.lat
            if not in_bbox(lon, lat):
                return
            if "amenity" in n.tags or "shop" in n.tags:
                self.counts[h3.geo_to_h3(lat, lon, coarse_res)][1] += 1

        def way(self, w):
            if not w.nodes:
                return
            if "highway" in w.tags and w.tags["highway"] in important_highways:
                coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
                step = max(1, len(coords) // 8)
                for i in range(0, len(coords), step):
                    lon, lat = coords[i]
                    if in_bbox(lon, lat):
                        self.counts[h3.geo_to_h3(lat, lon, coarse_res)][0] += 1
            if "building" in w.tags:
                coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
                step = max(1, len(coords) // 6)
                for i in range(0, len(coords), step):
                    lon, lat = coords[i]
                    if in_bbox(lon, lat):
                        self.counts[h3.geo_to_h3(lat, lon, coarse_res)][2] += 1
            self.processed += 1
            if self.processed % 500_000 == 0:
                print(f"    ⏳ {self.processed:,} ways …")

    handler = InfraHandler()
    handler.apply_file(pbf_path, locations=True)

    rows = [(h, v[0], v[1], v[2]) for h, v in handler.counts.items()]
    df = pd.DataFrame(rows, columns=["h3", "road", "poi", "building"])
    df.to_parquet(cache_file)
    print(f"    ✅ {len(df):,} H3 cells  ⏱ {time.time() - start_time:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Raster → H3 parallel aggregation
# ---------------------------------------------------------------------------

def _process_window(args):
    """
    Two-pass window worker:
      pass-1 (downsampled) → mean_elev
      pass-2 (native, stride-sampled) → elev_roughness (slope magnitude)

    Returns dict {h3_str: [elev_sum, elev_count, slope_sum, slope_count]}.
    """
    window, raster_path, res, downsample = args
    result = {}
    try:
        with rasterio.open(raster_path) as src:
            nodata = src.nodata

            # Pass 1: downsampled → mean elevation
            out_h = max(1, int(window.height / downsample))
            out_w = max(1, int(window.width / downsample))
            data_down = src.read(
                1, window=window,
                out_shape=(out_h, out_w),
                resampling=rasterio.enums.Resampling.average,
                masked=True,
            )
            if np.ma.is_masked(data_down) and data_down.mask.all():
                return result

            scale_x = window.width / out_w
            scale_y = window.height / out_h
            win_tf = src.window_transform(window)
            win_tf_down = Affine(
                win_tf.a * scale_x, win_tf.b, win_tf.c,
                win_tf.d, win_tf.e * scale_y, win_tf.f,
            )
            transformer = Transformer.from_crs(
                src.crs or "EPSG:4326", "EPSG:4326", always_xy=True)
            filled_down = (data_down.filled(0) if np.ma.is_masked(data_down)
                           else np.array(data_down, dtype=float))
            valid_down = (~data_down.mask if np.ma.is_masked(data_down)
                          else ~np.isnan(data_down))

            for r, c in zip(*np.where(valid_down)):
                val = filled_down[r, c]
                if nodata is not None and abs(float(val) - float(nodata)) < 1e-6:
                    continue
                px, py = xy(win_tf_down, r, c)
                lon, lat = transformer.transform(px, py)
                cell = h3.geo_to_h3(lat, lon, res)
                if cell not in result:
                    result[cell] = [0.0, 0, 0.0, 0]
                result[cell][0] += float(val)
                result[cell][1] += 1

            # Pass 2: native resolution → slope / roughness
            data_full = src.read(1, window=window, masked=True)
            if np.ma.is_masked(data_full) and data_full.mask.all():
                return result

            win_tf_full = src.window_transform(window)
            filled_full = (
                data_full.filled(
                    np.nanmedian(data_full.compressed())
                    if data_full.compressed().size > 0 else 0.0
                )
                if np.ma.is_masked(data_full)
                else np.array(data_full, dtype=float)
            )
            if filled_full.shape[0] >= 2 and filled_full.shape[1] >= 2:
                dzdx, dzdy = np.gradient(filled_full)
                slope_full = np.sqrt(dzdx ** 2 + dzdy ** 2)
            else:
                slope_full = np.zeros_like(filled_full)

            valid_full = (~data_full.mask if np.ma.is_masked(data_full)
                          else ~np.isnan(data_full))
            step = SLOPE_SAMPLE_STEP
            rows_s = np.arange(0, filled_full.shape[0], step)
            cols_s = np.arange(0, filled_full.shape[1], step)
            rr, cc = np.meshgrid(rows_s, cols_s, indexing="ij")
            rr = rr.flatten(); cc = cc.flatten()
            sel = valid_full[rr, cc]
            rr = rr[sel]; cc = cc[sel]

            for r, c in zip(rr, cc):
                val = filled_full[r, c]
                if nodata is not None and abs(float(val) - float(nodata)) < 1e-6:
                    continue
                px, py = xy(win_tf_full, r, c)
                lon, lat = transformer.transform(px, py)
                cell = h3.geo_to_h3(lat, lon, res)
                if cell not in result:
                    result[cell] = [0.0, 0, 0.0, 0]
                result[cell][2] += float(slope_full[r, c])
                result[cell][3] += 1

    except Exception as e:
        print(f"⚠️  Window {window} failed: {e}")
    return result


def raster_to_h3(raster_path, res, stat="sum", downsample=1, bbox=None) -> pd.DataFrame:
    """
    Convert a raster to H3-aggregated values.

    Returns DataFrame with columns: h3, val, mean_elev, elev_roughness.
    Results are cached as parquet under CACHE_DIR.
    """
    from .config import FORCE_RECOMPUTE_ELEV

    cache_file = (
        CACHE_DIR /
        f"{Path(raster_path).stem}_h3_res{res}{'_vienna' if bbox else ''}.parquet"
    )
    is_elev = any(k in str(cache_file).lower()
                  for k in ("elevation", "dhm", "dem", "srtm"))

    if cache_file.exists():
        if FORCE_RECOMPUTE_ELEV and is_elev:
            print(f"  🗑️  FORCE_RECOMPUTE_ELEV — deleting: {cache_file.name}")
            cache_file.unlink(missing_ok=True)
        else:
            df = pd.read_parquet(cache_file)
            rough_ok = (
                "elev_roughness" in df.columns
                and float(df["elev_roughness"].max()) > 0
            )
            if is_elev and not rough_ok:
                print(f"  ✓ Cached (roughness=0, will be patched): {cache_file.name}")
                return df
            info = (f"  roughness max={float(df['elev_roughness'].max()):.4f}"
                    if "elev_roughness" in df.columns else "")
            print(f"  ✓ Cached: {Path(raster_path).name} ({len(df):,} cells{info})")
            return df

    print(f"  🚀 Converting {Path(raster_path).name} → Res {res}")
    try:
        with rasterio.open(raster_path) as src:
            windows = (
                [window_from_bounds(
                    bbox["min_lon"], bbox["min_lat"],
                    bbox["max_lon"], bbox["max_lat"],
                    src.transform,
                )]
                if bbox else [w[1] for w in src.block_windows(1)]
            )

        worker_args = [(w, str(raster_path), res, downsample) for w in windows]
        global_agg = {}

        with Pool(N_WORKERS) as p:
            for local in tqdm(
                p.imap_unordered(_process_window, worker_args, chunksize=1),
                total=len(worker_args), desc="    Blocks", leave=True, ncols=80,
            ):
                for cell, vals in local.items():
                    if cell not in global_agg:
                        global_agg[cell] = [0.0, 0, 0.0, 0]
                    g = global_agg[cell]
                    g[0] += vals[0]; g[1] += vals[1]
                    g[2] += vals[2]; g[3] += vals[3]

        rows_out = []
        for cell, (es, ec, ss, sc) in global_agg.items():
            rows_out.append((
                cell,
                es if stat == "sum" else (es / ec if ec > 0 else 0.0),
                es / ec if ec > 0 else 0.0,
                ss / sc if sc > 0 else 0.0,
            ))
        df = pd.DataFrame(rows_out, columns=["h3", "val", "mean_elev", "elev_roughness"])
        df.to_parquet(cache_file)
        print(f"    ✅ {len(df):,} cells  roughness max={float(df['elev_roughness'].max()):.4f}")
        return df

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame(columns=["h3", "val", "mean_elev", "elev_roughness"])


# ---------------------------------------------------------------------------
# Elevation roughness fix: aggregate fine → coarse
# ---------------------------------------------------------------------------

def aggregate_roughness_fine_to_coarse(fine_df: pd.DataFrame,
                                        coarse_res: int = COARSE_RES) -> dict:
    """
    The reprojected EPSG:4326 DEM loses slope detail due to bilinear
    resampling, so the coarse aggregation yields roughness=0.
    This function aggregates roughness from a fine-resolution (res-10)
    cache to coarse (res-7) by averaging over H3 parent cells.

    Returns dict {coarse_h3_str: mean_roughness}.
    """
    print("  🔧 Aggregating fine→coarse roughness …")
    t0 = time.time()
    fine_df = fine_df.copy()
    fine_df["h3"] = fine_df["h3"].astype(str)

    if "elev_roughness" not in fine_df.columns:
        print("  ⚠️  fine_df has no elev_roughness column")
        return {}

    nz = fine_df[fine_df["elev_roughness"] > 0]
    print(f"     Fine cells with roughness > 0: {len(nz):,} / {len(fine_df):,}")
    if len(nz) == 0:
        return {}

    coarse_sum = {}; coarse_count = {}
    for h, r in tqdm(zip(nz["h3"].values, nz["elev_roughness"].values),
                     total=len(nz), desc="     Aggregating", ncols=80):
        try:
            parent = h3.h3_to_parent(str(h), coarse_res)
        except Exception:
            continue
        coarse_sum[parent] = coarse_sum.get(parent, 0.0) + float(r)
        coarse_count[parent] = coarse_count.get(parent, 0) + 1

    result = {p: coarse_sum[p] / coarse_count[p]
              for p in coarse_sum if coarse_count[p] > 0}
    rmax = max(result.values()) if result else 0.0
    print(f"     → {len(result):,} coarse cells  max={rmax:.4f}  t={time.time()-t0:.1f}s")
    return result


def patch_coarse_roughness(elev_result: pd.DataFrame,
                            fine_cache_path: Path,
                            coarse_res: int = COARSE_RES) -> pd.DataFrame:
    """Patch roughness=0 in *elev_result* using aggregated fine-res cache."""
    if not Path(fine_cache_path).exists():
        print(f"  ⚠️  Fine elev cache not found: {fine_cache_path}")
        return elev_result
    fine_df = pd.read_parquet(fine_cache_path)
    fine_rmax = float(fine_df["elev_roughness"].max()) if "elev_roughness" in fine_df.columns else 0.0
    if fine_rmax == 0:
        return elev_result
    coarse_roughness = aggregate_roughness_fine_to_coarse(fine_df, coarse_res)
    if not coarse_roughness:
        return elev_result
    elev_result = elev_result.copy()
    elev_result["h3"] = elev_result["h3"].astype(str)
    elev_result["elev_roughness"] = elev_result["h3"].map(coarse_roughness).fillna(0.0)
    print(f"     Patched roughness max={float(elev_result['elev_roughness'].max()):.4f}")
    return elev_result


# ---------------------------------------------------------------------------
# PCA importance scoring  (paper eq. 2)
# ---------------------------------------------------------------------------

def compute_pca_importance(df_pca: pd.DataFrame) -> np.ndarray:
    """
    Standardise the four feature signals and project onto the first
    principal component.  Min-max normalise the result to [0, 1].

    Features used (paper Section 3.1):
      poi_density_norm, building_density_norm, road_density_norm, elev_roughness

    Returns 1-D array of importance scores, one per row of df_pca.
    """
    features = [
        "poi_density_norm",
        "building_density_norm",
        "road_density_norm",
        "elev_roughness",
    ]
    for f in features:
        df_pca[f] = df_pca[f].fillna(0.0)

    X = StandardScaler().fit_transform(df_pca[features])
    pc1 = PCA(n_components=1).fit_transform(X).flatten()
    scores = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
    print(f"  ✅ PCA scores: {len(scores):,} cells  "
          f"min={scores.min():.4f}  max={scores.max():.4f}")
    return scores
