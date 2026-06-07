"""
ioah3.graphcut
==============
Stage 2 of IOAH3: MRF graph-cut cell selection.

Constructs a binary MRF over the H3 base grid and solves for the
included / excluded labelling via max-flow / min-cut (PyMaxflow).

Paper reference: Section 3.2 (Graph-Cut Cell Selection) and eq. (2)–(4).
"""

import numpy as np
import pandas as pd
import h3
import maxflow

from .config import COARSE_RES


def compute_importance_graph_cut(
    df_coarse: pd.DataFrame,
    road_h3_set: set,
    poi_h3_set: set,
    unary_score: np.ndarray,
    lambda_smooth: float = 1.0,
) -> pd.DataFrame:
    """
    Solve the MRF energy (paper eq. 2):

        E(l) = Σ_h ψ_h(l_h)  +  λ Σ_{(h,h')} φ_{hh'}(l_h, l_h')

    Unary term  (eq. 2, paper):
        ψ_h(0) = 1 - s(h)   [cost of including low-importance cell]
        ψ_h(1) = 0           [no cost for excluding]

    Pairwise / smoothness weight  (eq. 4, paper):
        w_{hh'} = exp( -|p(h)-p(h')| - |η(h)-η(h')| )

    Parameters
    ----------
    df_coarse : DataFrame with columns h3, pop, haz, (elev_roughness)
    road_h3_set : set of H3 strings that contain road segments
    poi_h3_set  : set of H3 strings that contain POIs
    unary_score : 1-D array of importance scores s(h) ∈ [0,1]
    lambda_smooth : smoothness weight λ (default 1.0, paper eq. 2)

    Returns
    -------
    df_coarse with added columns: included, excluded, importance_score
    """
    print("\n  ✨ MRF Graph-Cut Optimisation …")

    df = df_coarse.copy()
    df["h3"] = df["h3"].astype(str)

    if "pop" not in df.columns or "haz" not in df.columns:
        raise KeyError(f"Missing columns. Available: {list(df.columns)}")

    df["has_road"] = df["h3"].isin(road_h3_set)
    df["has_poi"]  = df["h3"].isin(poi_h3_set)

    # Elevation exception flag (used for diagnostics, not the energy term)
    roughness = (df["elev_roughness"].values
                 if "elev_roughness" in df.columns else np.zeros(len(df)))
    med = np.median(roughness)
    mad = np.median(np.abs(roughness - med)) + 1e-8
    z = np.abs((roughness - med) / mad)
    df["has_elev_exception"] = z > np.percentile(z, 90)

    h3_list   = df["h3"].tolist()
    index_map = {h: i for i, h in enumerate(h3_list)}

    # Build graph
    g = maxflow.Graph[float]()
    g.add_nodes(len(df))

    # Unary terms: source capacity = s(h), sink capacity = 0
    for i, score in enumerate(unary_score):
        g.add_tedge(i, float(score), 0.0)

    # Pairwise (smoothness) terms — H3 k-ring-1 adjacency
    print("    ⚡ Building pairwise edges …")
    pop_dict = dict(zip(df["h3"], df["pop"]))
    haz_dict = dict(zip(df["h3"], df["haz"]))

    for h in h3_list:
        i = index_map[h]
        try:
            neighbors = h3.k_ring(h, 1)
            neighbors.discard(h)
            for n in map(str, neighbors):
                if n not in index_map:
                    continue
                j = index_map[n]
                d = abs(pop_dict[h] - pop_dict[n]) + abs(haz_dict[h] - haz_dict[n])
                w = lambda_smooth * np.exp(-d)
                g.add_edge(i, j, w, w)
        except Exception:
            continue

    print("    🔥 Running max-flow …")
    g.maxflow()

    labels = np.array([g.get_segment(i) for i in range(len(df))])
    df["included"]         = labels == 0   # source-side = included
    df["excluded"]         = labels == 1
    df["importance_score"] = unary_score

    n_incl = int(df["included"].sum())
    print(f"    Included: {n_incl:,} ({100 * n_incl / len(df):.1f}%)")
    return df


def remove_isolated_inclusions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-processing connectivity filter (paper Section 3.2, last paragraph):
    remove included cells that have no included H3 neighbour.

    Returns filtered DataFrame with included and excluded columns updated.
    """
    df = df.copy()
    df["h3"] = df["h3"].astype(str)
    h3_set_incl = set(df.loc[df["included"], "h3"])

    def _count_incl_neighbors(h):
        try:
            return sum(1 for n in map(str, h3.k_ring(h, 1))
                       if n != h and n in h3_set_incl)
        except Exception:
            return 0

    df_incl = df[df["included"]].copy()
    df_incl["neighbor_count"] = df_incl["h3"].map(_count_incl_neighbors)
    df_incl = df_incl[df_incl["neighbor_count"] > 0]

    df_result = pd.concat([df_incl, df[df["excluded"]]])
    removed = len(df[df["included"]]) - len(df_incl)
    if removed:
        print(f"    Connectivity filter: removed {removed:,} isolated cells")
    return df_result
