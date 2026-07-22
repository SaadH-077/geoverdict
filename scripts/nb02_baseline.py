"""Build notebook 02 — the 2020 forest baseline and the map-disagreement finding."""

from nbbuild import bootstrap_cells, code, md, save

cells = [
    md("""
# 02 — Was it forest on 31 December 2020? Two official answers

**Question this notebook answers:** for each validated plot, what fraction was
forest at the EUDR cutoff — and *how much does the answer depend on which
official forest map you consult?*

**Why the baseline decides everything.** Under EUDR, only land that **was
forest on 2020-12-31 and was cleared afterwards** creates non-compliance.
A plot that was already pasture at the cutoff is compliant *no matter what
happens on it later*. So before any change detection, every plot needs a
baseline forest status — and here the pipeline meets an uncomfortable fact of
this domain: there is no single ground truth. There are *maps*, each with its
own sensor, definition and error profile:

| | **JRC Global Forest Cover 2020** | **Hansen Global Forest Change** |
|---|---|---|
| Who | EU Joint Research Centre | Univ. of Maryland |
| Status | The map the EU built *for EUDR* | The de-facto research standard |
| Forest definition | ≥ 10% canopy, ≥ 0.5 ha, height ≥ 5 m at 2020 | canopy ≥ *threshold* in 2000, minus mapped loss 2001–2020 |
| Construction | direct 2020 classification | 2000 baseline + 20 years of accumulated loss detection |

Note the structural difference: Hansen's 2020 state is an *accounting
identity* (2000 cover minus detected loss), so 20 years of missed or
phantom loss accumulate into its 2020 answer. JRC classifies 2020 directly.
Neither is "right" — and the disagreement between them, **measured per plot**,
is this chapter's finding.

**Produces**
- `outputs/baseline.csv` — per plot: forest fraction (both maps), post-2020 Hansen loss fraction
- `outputs/baseline_provenance.json` — exact asset versions used
- `figures/g02_*.png`

**Expected runtime:** ~5 minutes (all computation is server-side on Earth
Engine; only per-plot statistics travel). Requires a (free) Earth Engine
account — the first cell below walks through the one-time auth.
"""),
    *bootstrap_cells(),
    md("""
### Earth Engine setup — and why GEE for *this* chapter

The reference maps are tens of terabytes as rasters. We need exactly one
number per plot per map: the mean of a binary mask over a polygon. Earth
Engine computes that reduction next to the data and ships back kilobytes —
this is the workload GEE is uniquely right for. (Chapter 03's time series
also reduce server-side; chapter 04's pixel chips, where we need full control
of the raw data, come via the STAC/COG path instead. Choosing the access path
per workload is a design decision this project makes explicitly.)

**One-time setup:** create a (free) Google Cloud project with Earth Engine
enabled at [code.earthengine.google.com](https://code.earthengine.google.com),
then put its id in `EE_PROJECT` below. The `Authenticate()` flow runs once
per Colab VM.
"""),
    code("""
EE_PROJECT = ""   # <- your Earth Engine cloud project id, e.g. "ee-yourname"

from geoverdict import gee
gee.init(project=EE_PROJECT or None)
print("Earth Engine initialised")
"""),
    code("""
import geopandas as gpd
import numpy as np
import pandas as pd

plots = gpd.read_file(cfg.OUTPUT_DIR / "plots_analysis.geojson")
print(f"analysis set from chapter 01: {len(plots)} plots")
plots.head(3)
"""),
    md("""
### Per-plot forest fractions, from both maps

One server-side call per batch: for every plot, the mean of (a) JRC GFC2020
forest mask, (b) the Hansen-derived "forest at cutoff" mask (canopy ≥ 30% in
2000 AND not lost by 2020), and (c) the fraction mapped by Hansen as lost
**after** 2020 — which chapters 03–05 will use as the weak reference label.

The canopy threshold is a *definition choice*, not a constant of nature —
30% is the FAO-aligned convention. We compute the baseline at 10/30/50% too,
because a compliance verdict that flips when a definition knob moves is
something an engineering team must know about itself, not discover from a
customer dispute.
"""),
    code("""
geoms = list(plots.geometry)
ids = list(plots.plot_id)

baseline = gee.forest_baseline_fractions(geoms, ids)
assets = dict(baseline.attrs.get("assets", {}))
print("asset versions used:", assets)

# definition sensitivity: same reduction at three canopy thresholds
sens = {}
for thr in (10, 30, 50):
    df_thr = (baseline if thr == 30
              else gee.forest_baseline_fractions(geoms, ids, canopy_threshold=thr))
    sens[thr] = df_thr.set_index("plot_id")["forest_frac_hansen"]

baseline = baseline.merge(plots[["plot_id", "area_ha"]], on="plot_id")
baseline["hansen_frac_c10"] = baseline.plot_id.map(sens[10])
baseline["hansen_frac_c50"] = baseline.plot_id.map(sens[50])
baseline.to_csv(cfg.OUTPUT_DIR / "baseline.csv", index=False)

import json
from datetime import datetime, timezone
prov = {"source": "Google Earth Engine", "identifiers": list(assets.values()),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameters": {"scale_m": 10, "canopy_thresholds": [10, 30, 50]}}
(cfg.OUTPUT_DIR / "baseline_provenance.json").write_text(json.dumps(prov, indent=2))
baseline.describe().round(3)
"""),
    code("""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

ax = axes[0]
ax.scatter(baseline.forest_frac_hansen, baseline.forest_frac_jrc, s=10, alpha=0.5,
           color=viz.PALETTE["forest"])
ax.plot([0, 1], [0, 1], ls="--", lw=1, color=viz.PALETTE["insufficient"])
for v in (0.30,):
    ax.axvline(v, color=viz.PALETTE["warn"], lw=0.8, ls=":")
    ax.axhline(v, color=viz.PALETTE["warn"], lw=0.8, ls=":")
ax.set_xlabel("Hansen forest fraction at cutoff")
ax.set_ylabel("JRC GFC2020 forest fraction")
ax.set_title("Two official answers, per plot\\n(off-diagonal quadrants = verdict depends on the map)")

ax = axes[1]
both = baseline[["forest_frac_jrc", "forest_frac_hansen"]].dropna()
ax.hist(both.forest_frac_jrc, bins=30, alpha=0.6, label="JRC GFC2020",
        color=viz.PALETTE["accent"])
ax.hist(both.forest_frac_hansen, bins=30, alpha=0.6, label="Hansen (30% canopy)",
        color=viz.PALETTE["forest"])
ax.set_xlabel("forest fraction at cutoff"); ax.set_ylabel("plots")
ax.legend(); ax.set_title("Distribution of baseline forest fractions")
fig.tight_layout()
viz.save(fig, "g02_forest_fractions")
plt.show()
"""),
    md("""
### The finding: how often does the compliance answer depend on the map?

We call a plot "forest at cutoff" when its forest fraction crosses 30% (the
same constant the verdict layer uses, chosen so that clearing the forested
part of a mixed plot still registers). The number that matters is the share
of plots whose forest/non-forest **status flips between the two maps** — and,
separately, the share that flips when only Hansen's canopy-definition knob
moves between 10% and 50%.

Also broken out by plot size, for a structural reason worth predicting in
advance: both products are built from ~30 m Landsat pixels, so a 1–4 ha plot
contains only a dozen such pixels and mixed edges dominate — *disagreement
should concentrate in exactly the smallholder plots that compliance is
hardest for.* Let's see if the data agrees.
"""),
    code("""
THR = 0.30
b = baseline.dropna(subset=["forest_frac_jrc", "forest_frac_hansen"]).copy()
b["jrc_forest"] = b.forest_frac_jrc >= THR
b["han_forest"] = b.forest_frac_hansen >= THR
b["maps_disagree"] = b.jrc_forest != b.han_forest
b["def_disagree"] = (b.hansen_frac_c10 >= THR) != (b.hansen_frac_c50 >= THR)

overall = b.maps_disagree.mean()
by_def = b.def_disagree.mean()
print(f"plots whose forest-at-cutoff status flips between JRC and Hansen: {overall:.1%}")
print(f"plots whose status flips between canopy definitions 10% vs 50%:   {by_def:.1%}")

bins = [0, 4, 10, 30, 100, np.inf]
labels = ["<=4 ha", "4-10 ha", "10-30 ha", "30-100 ha", ">100 ha"]
b["size_bin"] = pd.cut(b.area_ha, bins=bins, labels=labels)
by_size = b.groupby("size_bin", observed=True)["maps_disagree"].agg(["mean", "size"])
print(by_size)

cfg.append_result({"notebook": "02", "name": "baseline_disagreement",
                   "threshold": THR,
                   "map_disagreement_rate": float(overall),
                   "definition_disagreement_rate": float(by_def),
                   "by_size": {str(k): float(v) for k, v in by_size["mean"].items()},
                   "assets": assets})
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

ax = axes[0]
ax.bar(by_size.index.astype(str), by_size["mean"], color=viz.PALETTE["warn"])
for x, (v, n) in enumerate(zip(by_size["mean"], by_size["size"])):
    ax.text(x, v, f"{v:.0%}\\n(n={n})", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("share of plots where the maps disagree")
ax.set_title("Disagreement concentrates in small plots\\n(30 m source pixels vs plot size)")

ax = axes[1]
colors = np.where(b.maps_disagree, viz.PALETTE["warn"], viz.PALETTE["forest"])
cent = b.merge(plots[["plot_id", "geometry"]], on="plot_id").geometry.centroid
ax.scatter(cent.x, cent.y, c=colors, s=8, alpha=0.8)
ax.set_title("Where the maps disagree (orange) over the AOI")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_aspect(1/np.cos(np.radians(-7.2)))
fig.tight_layout()
viz.save(fig, "g02_disagreement")
plt.show()
"""),
    md("""
### Why this matters (and what a production system does about it)

If X% of plots get a different compliance-relevant baseline depending on
which official map is consulted, then for those plots **the verdict is a
property of the map choice, not of the land** — and a system that silently
picks one map is hiding a judgement call inside a lookup. GeoVerdict's
response, implemented in chapter 05: baseline disagreement is *itself a
signal* that caps the verdict at MEDIUM and states the conflict in the
evidence bundle, so a human sees exactly which authority said what.

Also worth naming: our plots are *randomly placed* over a deforestation
frontier — a portfolio of real farms clusters along roads and rivers, on
exactly the fragmented forest edges where maps disagree most. This estimate
is therefore more likely a floor than a ceiling for real submissions.

**Next:** chapter 03 stops asking maps and starts asking the satellite —
six years of Sentinel-2 over every plot, and a detector that finds the month
the forest signal broke.
"""),
]

save(cells, "02_forest_baseline.ipynb")
