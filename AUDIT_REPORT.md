# IOAH3 — Draft vs. Code Audit Report

**Preprint draft:** `main__9_.tex`  
**Code:** `Main_final_code.py`  
**Scope:** Stages 1–3 of the IOAH3 pipeline (everything the paper claims to describe).  
**Not in scope:** Visualisation, spatial constraint graph, checkpoint I/O — the paper does not describe these and the code upload excludes them. ✅

---

## ✅ Points of Agreement

### Stage 1 — Feature Extraction (§3.1)

| Paper claim | Code |
|---|---|
| Four feature signals: road density, POI density, building density, elevation roughness | `features = ["poi_density_norm", "building_density_norm", "road_density_norm", "elev_roughness"]` |
| Road classes: motorway, trunk, primary, secondary, tertiary, unclassified, residential | `important_highways = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential"}` |
| POI: OSM amenity or shop tags | `if "amenity" in n.tags or "shop" in n.tags` |
| Building density: OSM building footprint nodes normalised by cell area | `bldg_count / cell_area` |
| Roughness = mean gradient magnitude (slope std over DEM pixels) | `slope_full = np.sqrt(dzdx**2 + dzdy**2)` then mean over cell |
| Standardise to zero mean / unit variance before PCA | `StandardScaler().fit_transform(...)` |
| Importance score = first PC, min-max normalised to [0,1] | `PCA(n_components=1)` + `(pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)` |

### Stage 2 — Graph-Cut (§3.2)

| Paper claim | Code |
|---|---|
| Binary MRF labelling: included (0) / excluded (1) | `g.get_segment(i)` == 0 → included |
| Unary: ψ_h(0) = 1 − s(h), ψ_h(1) = 0 | `g.add_tedge(i, score, 0.0)` |
| Pairwise weight: exp(−\|p(h)−p(h')\| − \|η(h)−η(h')\|) | `w = np.exp(-abs(pop[h]-pop[n]) - abs(haz[h]-haz[n]))` ✅ |
| Adjacency: H3 k-ring-1 (six hexagonal neighbours) | `h3.k_ring(h, 1)` |
| λ = 1.0 | `lambda_smooth = 1.0` (hardcoded) |
| Solved via max-flow / min-cut (PyMaxflow / Boykov-Kolmogorov) | `maxflow.Graph[float]()` → `g.maxflow()` |
| Connectivity filter: remove included cells with < 1 included neighbour | `df_blue = df_blue[df_blue["neighbor_count"] > 0]` |

### Stage 3 — Refinement (§3.3)

| Paper claim | Code |
|---|---|
| Quantile thresholds: 40th, 65th, 80th percentiles | `np.percentile(..., [40, 65, 80])` |
| r*(h) = r_max if s(h) > q_80; 9 if q_65 < s(h) ≤ q_80; r_0 otherwise | `np.select([score > q[2], score > q[1]], [MAX_REFINEMENT_RES, 9], default=COARSE_RES)` |
| r_0 = 7, r_max = 10 | `COARSE_RES = 7`, `MAX_REFINEMENT_RES = 10` |
| Neighbour propagation via k-ring-1 | `h3.k_ring(h, 1)` |
| Children sampled from fine rasters pre-loaded once | `pop_fine`, `elev_fine`, `haz_fine` at `MAX_REFINEMENT_RES` loaded before refinement loop |
| Background excluded cells retained at coarse resolution | `excluded_records["resolution"] = COARSE_RES` |

---

## ⚠️ Discrepancies to Fix Before Preprint Upload

### D1 — Unary term sign convention (§3.2, eq. 2)

**Paper (eq. 2):**  
> ψ_h(0) = 1 − s(h) [cost of including a low-importance cell]

This implies label 0 = *included* carries a unary cost. In a min-cut graph, the source edge carries the cost of *labelling as source-side (included)*. The code does:

```python
g.add_tedge(i, score, 0.0)   # source capacity = s(h), sink = 0
```

In PyMaxflow, `add_tedge(i, source_cap, sink_cap)`: the source capacity represents the cost of assigning node *i* to the sink side (excluded). So the code minimises the cost of *excluding* high-importance cells, which is equivalent to the paper's formulation *only if* the labelling convention is reversed.

**Verdict:** The code is correct (source-side = included, consistent with `labels == 0 → included`), but the paper defines ψ_h(0) as the cost of *including*, which reverses the cut convention from what PyMaxflow uses. The paper should either clarify the cut convention explicitly or restate the unary as:

> ψ_h(source) = s(h), ψ_h(sink) = 0 [PyMaxflow convention]

**Action:** Add one sentence to §3.2 noting that in the implementation the source capacity equals s(h) per the PyMaxflow convention, which is equivalent to the stated formulation.

---

### D2 — d = 4 features stated; mean elevation silently excluded (§3.1)

**Paper:** "These four features (d = 4) are standardised…"  and lists road density, POI density, building density, elevation roughness.

**Code:** Mean elevation (`mean_elev`) is computed and stored per cell but is explicitly *excluded* from the PCA feature matrix, consistent with the paper's footnote about it being "retained as a cell attribute for display and refinement propagation but excluded from the PCA feature set."

**Verdict:** Consistent, but the paper should make this more explicit earlier in §3.1 (the current language buries it mid-paragraph). No code change needed.

---

### D3 — Smoothness weight uses normalised vs raw population/hazard (§3.2, eq. 4)

**Paper (eq. 4):**  
> w_{hh'} = exp(−|p(h)−p(h')| − |η(h)−η(h')|)  
> with p(h) the *normalised* population count and η(h) the *normalised* hazard value.

**Code:**
```python
d = abs(pop_dict[h] - pop_dict[n]) + abs(haz_dict[h] - haz_dict[n])
w = np.exp(-d)
```
where `pop_dict` and `haz_dict` are populated from `df["pop"]` and `df["haz"]` — these are the raw aggregated raster values, **not normalised to [0,1]**.

**Impact:** If raw population values (e.g. 0–50,000 inhabitants per cell) are used, the exponent `−|p(h)−p(h')|` will be a very large negative number for any pair of cells with differing populations, collapsing nearly all smoothness weights to ≈ 0. This effectively disables the pairwise term and reduces the cut to a purely unary selection.

**Action:** Either (a) normalise pop and haz before populating the dicts (add `df["pop"] = normalize(df["pop"])` before the graph-cut), or (b) update the paper to state that raw values are used and the exponential decay acts as a soft threshold. **This is a real numerical bug that should be fixed in the code before publication.**

Suggested fix:
```python
df["pop_norm"] = df["pop"] / (df["pop"].max() + 1e-8)
df["haz_norm"] = df["haz"] / (df["haz"].max() + 1e-8)
pop_dict = dict(zip(df["h3"], df["pop_norm"]))
haz_dict = dict(zip(df["h3"], df["haz_norm"]))
```

---

### D4 — Connectivity filter threshold (§3.2)

**Paper:** "An additional connectivity filter removes included cells with fewer than *one* included neighbour."

**Code:**
```python
df_blue = df_blue[df_blue["neighbor_count"] > 0]
```
`neighbor_count > 0` means ≥ 1 neighbour, which matches "fewer than one" (i.e. remove those with 0). ✅ Correct, but the phrasing is ambiguous. The paper should say "removes cells with zero included neighbours" or "retains only cells with at least one included neighbour" to avoid any ambiguity about whether 1 means "at least 1" or "more than 1".

---

### D5 — Algorithm 1 omits the connectivity filter step

**Algorithm 1** in the paper (lines 1–10) does not include the isolated-singleton connectivity filter. The filter is mentioned in §3.2 prose but is missing from the pseudocode, which is what a reader would implement.

**Action:** Add a step after line 4 (solve max-flow):
```
4a: Remove included cells h with ∑_{h' ∈ k-ring(h,1)} 1[ℓ*(h')=0] = 0
```

---

### D6 — Base resolution in refinement quantiles (§3.3, eq. 5)

**Paper (eq. 5):** Quantiles are computed over `{s(h) : ℓ_h = 0}` (included cells).

**Code:**
```python
important = df[df["included"]].copy()
quantiles = np.percentile(important["importance_score"], [40, 65, 80])
```
This correctly conditions on included cells. ✅ However, the paper's label convention uses ℓ_h = 0 for included (source side), so the notation in eq. 5 is consistent with the stated convention in §3.2 — just flagging it as something a reader might trip over.

---

### D7 — Roughness computation uses native-CRS DEM for fine, reprojected for coarse

The paper describes elevation roughness as a single feature computed from gradient magnitude. The code uses a two-resolution scheme: the fine (res-10) cache is computed from the original Lambert-projected DEM (correct pixel geometry), while the coarse (res-7) value is computed from the bilinear-resampled EPSG:4326 reprojection (which zeros out roughness due to smoothing).

The pipeline patches this via `patch_coarse_roughness()` / `aggregate_roughness_fine_to_coarse()`, but this correction is **completely absent from the paper**.

**Action:** Add a remark in §3.1 or §4 noting that:
> In the Austria case study, the DEM is supplied in Lambert conformal conic projection. After reprojection to EPSG:4326 for H3 alignment, bilinear resampling suppresses sub-cell gradient variation, yielding near-zero coarse roughness. We therefore compute roughness at the fine resolution (r = 10) from the original projected DEM and aggregate by averaging over H3 parent cells.

This is important for reproducibility.

---

## Summary Table

| ID | Severity | Type | Description |
|----|----------|------|-------------|
| D1 | Low | Notation | Unary term sign convention differs from PyMaxflow's edge direction — clarify |
| D2 | Low | Clarity | mean_elev exclusion from PCA should be stated earlier |
| D3 | **High** | **Bug** | Pop/haz not normalised before smoothness weight — disables pairwise term |
| D4 | Low | Clarity | Connectivity filter threshold wording ambiguous |
| D5 | Medium | Missing | Connectivity filter absent from Algorithm 1 pseudocode |
| D6 | Low | Notation | ℓ_h=0 vs included boolean — fine, but worth a note |
| D7 | Medium | Missing | Roughness fine-to-coarse aggregation not documented in paper |

**Critical before upload: fix D3 in the code (or justify raw values in paper) and add D7 to the paper.**
