"""Build notebook 01 — the geometry gauntlet."""

from nbbuild import bootstrap_cells, code, md, save

cells = [
    md("""
# 01 — The geometry gauntlet: validating and repairing supplier plots

**Question this notebook answers:** when 600+ farm-plot geometries arrive from
suppliers, how many are damaged, can we *detect* every failure class, and how
much geometry can we *repair automatically* — measured, per failure class?

**Why this is chapter one.** An EUDR due-diligence pipeline begins where the
data begins: geolocation coordinates submitted by operators. Every downstream
step — satellite lookups, forest baselines, verdicts — silently inherits the
quality of these polygons. A verdict computed over a plot whose latitude and
longitude were swapped is a confident answer about the wrong hemisphere. So
before any satellite pixel is touched, the pipeline needs an intake gate with
three properties: it **diagnoses** (never silently fixes), it **repairs with
an audit trail**, and its repair quality is **measured, not assumed**.

**The experimental trick.** Real damaged submissions come without ground truth
— you cannot score a repair when you don't know what the farmer meant. So we
do what database fuzzers do: take known-good geometry, inject a taxonomy of
realistic corruptions with a seeded RNG, and measure (a) whether the validator
catches each class and (b) whether the repaired polygon matches the original,
by IoU. Detection rate and repair quality per failure class is the deliverable.

**Produces**
- `outputs/plots_truth.geojson` — the clean portfolio (the answer key)
- `outputs/plots_submitted.geojson` — the corrupted "supplier submission"
- `outputs/plots_analysis.geojson` — validated + repaired plots (what chapters 02-05 screen)
- `outputs/validation_report.csv` — per-plot issues, repairs, and quality
- `figures/g01_*.png`

**Expected runtime:** ~3 minutes, CPU only — no GPU needed for this chapter.
"""),
    *bootstrap_cells(),
    md("""
### The failure taxonomy

Each row below is a defect class the validator diagnoses, with its severity:
**ERROR** means the plot cannot be analysed until fixed (a verdict would be
meaningless), **WARNING** means analysable but recorded, because it degrades
confidence. The taxonomy is not invented — each class reproduces a failure
mode with a documented real-world cause (lat/lon order confusion between
GeoJSON and GPS conventions, lost CRS tags from web-map exports, GPS-logger
stutter, hand-digitising slips, the same farm submitted twice).

Severity is a *decision*, not a property of the geometry: a clockwise-wound
ring violates the GeoJSON spec but every library here still computes the
right area for it — so it is a WARNING. A self-intersection changes what
area even *means* — so it is an ERROR.
"""),
    code("""
import pandas as pd
from geoverdict import geometry as G

taxonomy = pd.DataFrame(
    [(code, sev, desc) for code, (sev, desc) in G.TAXONOMY.items()],
    columns=["code", "severity", "description"],
).sort_values(["severity", "code"]).reset_index(drop=True)
taxonomy
"""),
    md("""
### The plot portfolio

**Where:** Novo Progresso, Pará, Brazil — the BR-163 corridor in the Amazon's
"arc of deforestation". Chosen because (1) post-2020 clearing, stable forest,
and pre-2020 pasture all occur naturally here, so every verdict tier will be
exercised by real land; (2) it has a pronounced dry season, so the notebooks
can show both clean signals and the wet-season observation gaps that make
tropical monitoring genuinely hard; (3) cattle and soy from this corridor are
squarely in EUDR scope.

**Honesty note, stated up front:** the plot *geometries* are synthetic — there
is no public registry of farm boundaries for this frontier — but everything
measured **under** them from chapter 02 onward is real satellite data over
real land. Plot sizes follow a log-normal distribution (median ≈ 5 ha, tail to
hundreds of ha), the standard empirical model for landholding sizes, so the
portfolio has the same smallholder-dominated size mix the EUDR actually
covers. Shapes are irregular 6–11-vertex polygons rather than rectangles,
because rectangles survive several corruption classes trivially and real
field boundaries are irregular.
"""),
    code("""
import numpy as np
import matplotlib.pyplot as plt
from geoverdict import corrupt
from geoverdict import geometry as G

truth = corrupt.generate_portfolio(cfg.N_PLOTS, cfg.AOI_BBOX, seed=cfg.SEED)
areas = np.array([G.area_ha(p) for p in truth])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(areas, bins=np.geomspace(areas.min(), areas.max(), 40), color=viz.PALETTE["forest"])
axes[0].set_xscale("log")
axes[0].set_xlabel("plot area (ha, log scale)"); axes[0].set_ylabel("plots")
axes[0].set_title(f"{cfg.N_PLOTS} plots, median {np.median(areas):.1f} ha")
axes[0].axvline(4.0, color=viz.PALETTE["warn"], ls="--", lw=1.2)
axes[0].annotate("EUDR 4 ha polygon rule", xy=(4.0, axes[0].get_ylim()[1]*0.9),
                 fontsize=8, color=viz.PALETTE["warn"], rotation=90, va="top")
for g in truth:
    viz.draw_geom(axes[1], g, edgecolor=viz.PALETTE["forest"], lw=0.4, alpha=0.7)
axes[1].set_title("the portfolio over the AOI")
axes[1].set_xlabel("longitude"); axes[1].set_ylabel("latitude")
axes[1].set_aspect(1/np.cos(np.radians(-7.2)))
viz.save(fig, "g01_portfolio")
plt.show()

print(f"total portfolio area: {areas.sum():,.0f} ha")
print(f"plots <= 4 ha (point submissions allowed under EUDR): {(areas <= 4).sum()}")
"""),
    md("""
### A reality check before the synthetic experiment

Before trusting a validator tuned on manufactured damage, point it at
geometry from the wild. The [Whisp](https://github.com/forestdatapartnership/whisp)
project (FAO / Forest Data Partnership — the open-source reference tool for
exactly this EUDR plot-screening problem) ships 50 real example plots from
actual submissions across Africa, South America and Southeast Asia. Running
our validator over them — with the AOI check disabled, since these plots are
global — is a cheap external sanity check: real-world plots should be mostly
clean with a scattering of warnings, and a validator that flags half of them
as ERRORs has miscalibrated thresholds.
"""),
    code("""
import json, urllib.request
from shapely.geometry import shape

WHISP_URL = ("https://raw.githubusercontent.com/forestdatapartnership/whisp/"
             "main/tests/fixtures/geojson_example.geojson")
try:
    with urllib.request.urlopen(WHISP_URL, timeout=30) as r:
        whisp = json.load(r)
    whisp_geoms = [shape(f["geometry"]) for f in whisp["features"]]
    issues_per_plot = G.validate_portfolio(whisp_geoms, aoi_bbox=None)
    from collections import Counter
    counts = Counter(i.code for plot in issues_per_plot for i in plot)
    n_clean = sum(1 for plot in issues_per_plot if not plot)
    print(f"Whisp fixture: {len(whisp_geoms)} real plots -> {n_clean} fully clean")
    for code_, n in counts.most_common():
        print(f"  {code_:<20} {n:>3}  ({G.TAXONOMY[code_][0]})")
except Exception as exc:
    print(f"Whisp fixture unavailable ({exc}) - external check skipped; "
          f"the synthetic experiment below is self-contained.")
"""),
    md("""
### Injecting damage: the corrupted "submission"

45% of the portfolio is damaged — deliberately pessimistic, so every failure
class has enough samples to estimate a repair rate from. Corruption types are
assigned round-robin over a seeded shuffle (a uniform draw would leave rare
classes with 3 samples and meaningless rates). Duplicates are *appended* as
extra rows, because duplication is a property of the submission, not of a
geometry. Every plot also carries a **declared area** with ±10% supplier
noise — real declared areas never match GIS-measured ones, and the EUDR
point-vs-polygon rule must be tested against noisy declarations.

The gallery below shows one example of each class: grey = what the farmer
meant, red = what arrived, green = what the repairer recovered.
"""),
    code("""
submitted, injected, declared = corrupt.corrupt_portfolio(truth, cfg.CORRUPTION_RATE, seed=cfg.SEED)
print(f"submission: {len(submitted)} rows ({len(submitted) - len(truth)} appended duplicates)")

from collections import Counter
print(Counter(x for x in injected if x))
"""),
    code("""
# one showcase per corruption class: corrupt a mid-sized clean plot directly
import numpy as np

showcase_plot = truth[int(np.argsort([G.area_ha(p) for p in truth])[len(truth)//2])]
rng = np.random.default_rng(cfg.SEED)
cases = []
for name, (fn, expected) in corrupt.CORRUPTIONS.items():
    damaged = fn(showcase_plot, rng)
    issues = G.validate_geometry(damaged, declared_area_ha=G.area_ha(showcase_plot))
    result = G.repair(damaged, issues, declared_area_ha=G.area_ha(showcase_plot))
    quality = G.iou(result.geometry, showcase_plot) if result.ok else 0.0
    cases.append({
        "name": name, "clean": showcase_plot, "corrupted": damaged,
        "repaired": result.geometry if result.ok else None,
        "caption": (f"repaired, IoU {quality:.2f}" if result.ok else "-> manual review"),
    })
fig = viz.corruption_gallery(cases, n_cols=4,
    title="grey = intended | red = as submitted | green = auto-repaired")
viz.save(fig, "g01_corruption_gallery")
plt.show()
"""),
    md("""
### Experiment 1 — does the validator catch what we injected?

We validate the whole corrupted submission (including the portfolio-level
checks: duplicates, overlaps) and join *diagnosed* codes against *injected*
classes. The result is a detection matrix — injected class × diagnosed code.
A perfect validator has its expected code lit on every row.

Three subtleties worth noticing, all of them *domain logic* rather than
slack in the answer key:
- several classes have **more than one correct diagnosis** — a plot collapsed
  to a point is `POINT_GEOMETRY` when its declared area permits a point under
  EUDR Art. 9 and `POINT_TOO_LARGE` when it does not; a sliver on a small plot
  legitimately trips the area floor first;
- expect **bow-tie detection below 100%, and that is not a bug**: a vertex
  swap that happens not to self-intersect produces a *legal* polygon that is
  simply the wrong shape — no validator on Earth can catch damage that leaves
  the geometry valid. That residual class is exactly why compliance workflows
  send the rendered plot back to the supplier for visual confirmation;
- clean plots that get flagged are **false alarms** — the operational cost of
  an over-eager validator is real farmers asked to re-submit valid plots, so
  we report that rate too.
""" ),
    code("""
issues_per_plot = G.validate_portfolio(submitted, declared_areas=declared)

detected, false_alarms = {}, 0
for inj, issues in zip(injected, issues_per_plot):
    codes = {i.code for i in issues}
    if inj is None:
        # a clean plot flagged with any ERROR is a false alarm
        if any(i.severity == G.ERROR for i in issues):
            false_alarms += 1
        continue
    acceptable = set(corrupt.CORRUPTIONS[inj][1])
    detected.setdefault(inj, []).append(bool(acceptable & codes))

rows = []
for name in corrupt.CORRUPTIONS:
    hits = detected.get(name, [])
    if hits:
        rows.append({"corruption": name, "n": len(hits),
                     "detection_rate": float(np.mean(hits)),
                     "acceptable_codes": " | ".join(corrupt.CORRUPTIONS[name][1])})
det = pd.DataFrame(rows).sort_values("detection_rate")
n_clean_total = sum(1 for x in injected if x is None)
print(det.to_string(index=False))
print(f"\\nfalse-alarm rate on clean plots: {false_alarms}/{n_clean_total} "
      f"= {false_alarms/n_clean_total:.1%}")

cfg.append_result({"notebook": "01", "name": "validator_detection",
                   "overall_detection_rate": float(np.mean([h for v in detected.values() for h in v])),
                   "false_alarm_rate": false_alarms / n_clean_total,
                   "per_class": {r['corruption']: r['detection_rate'] for r in rows}})
"""),
    code("""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = [viz.PALETTE["forest"] if r == 1.0 else
          (viz.PALETTE["warn"] if r >= 0.9 else viz.PALETTE["clearing"])
          for r in det["detection_rate"]]
ax.barh(det["corruption"], det["detection_rate"], color=colors)
ax.set_xlabel("detection rate"); ax.set_xlim(0, 1.05)
ax.set_title("Every injected failure class, and how reliably the validator catches it")
for y, (r, n) in enumerate(zip(det["detection_rate"], det["n"])):
    ax.text(min(r + 0.01, 1.0), y, f"{r:.0%} (n={n})", va="center", fontsize=8)
viz.save(fig, "g01_detection_rates")
plt.show()
"""),
    md("""
### Experiment 2 — repair, with quality measured against the answer key

The repairer applies targeted fixes in a fixed order (frame → topology →
size; the order *is* the algorithm — `make_valid` on Web-Mercator metres is
meaningless), logs every action, and the **validator** re-checks the result:
a repair only counts if the diagnosis comes back clean, not because the
repair code ran.

Repair quality per class = IoU between the repaired polygon and the clean
original. Note what "honest" means here: for classes like `teleport` (plot
1000 km away) or `micro` (collapsed to a few hundred m²), the *correct*
behaviour is to **refuse** and route to manual review — a repairer that
invents a plausible-looking plot from unrecoverable input is a compliance
liability, not a feature. Those refusals are counted as correct below.
"""),
    code("""
from geoverdict import geometry as G

records = []
repaired_geoms = []
for k, (geom, inj, dec) in enumerate(zip(submitted, injected, declared)):
    issues = issues_per_plot[k]
    if not any(i.severity == G.ERROR for i in issues) and not issues:
        # fully clean: passes straight through
        records.append({"plot_id": k, "injected": inj, "status": "clean",
                        "n_issues": 0, "actions": "", "iou_vs_truth": 1.0 if k < len(truth) else None})
        repaired_geoms.append(geom)
        continue
    result = G.repair(geom, issues, declared_area_ha=dec)
    truth_geom = truth[k] if k < len(truth) else None  # appended duplicates have no own truth
    quality = (G.iou(result.geometry, truth_geom)
               if (result.ok and truth_geom is not None) else None)
    status = ("clean_warned" if (not any(i.severity == G.ERROR for i in issues))
              else ("repaired" if result.ok else "manual_review"))
    records.append({"plot_id": k, "injected": inj, "status": status,
                    "n_issues": len(issues), "actions": "; ".join(result.actions),
                    "unresolved": "; ".join(result.unresolved),
                    "iou_vs_truth": quality})
    repaired_geoms.append(result.geometry if result.ok else geom)

report = pd.DataFrame(records)
funnel = report["status"].value_counts()
print(funnel)
print()
# "auto-resolved" counts BOTH repaired plots and warning-only plots that pass
# with the warning recorded (Z-noise, winding, duplicate vertices are
# warnings; the repairer normalises them, but they were never blockers)
auto_ok = report["status"].isin(["repaired", "clean_warned"])
rq = (report[report["injected"].notna()]
      .assign(auto_resolved=auto_ok)
      .groupby("injected")
      .agg(n=("status", "size"),
           auto_resolved=("auto_resolved", "mean"),
           mean_iou=("iou_vs_truth", "mean")))
print(rq.round(3).to_string())
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

order = ["clean", "clean_warned", "repaired", "manual_review"]
colors = [viz.PALETTE["forest"], viz.PALETTE["warn"], viz.PALETTE["accent"], viz.PALETTE["clearing"]]
vals = [int(funnel.get(s, 0)) for s in order]
axes[0].bar(order, vals, color=colors)
for x, v in enumerate(vals):
    axes[0].text(x, v, f"{v}\\n({v/len(report):.0%})", ha="center", va="bottom", fontsize=9)
axes[0].set_title("The intake funnel: what happens to a submission")
axes[0].set_ylabel("plots")

rq_plot = rq.sort_values("mean_iou")
axes[1].barh(rq_plot.index, rq_plot["mean_iou"].fillna(0),
             color=[viz.PALETTE["forest"] if v >= 0.95 else
                    (viz.PALETTE["warn"] if v >= 0.5 else viz.PALETTE["clearing"])
                    for v in rq_plot["mean_iou"].fillna(0)])
axes[1].set_xlabel("mean IoU of repaired vs intended geometry")
axes[1].set_title("Repair quality per failure class\\n(zero bars = honest refusals -> manual review)")
axes[1].set_xlim(0, 1.05)
fig.tight_layout()
viz.save(fig, "g01_repair_funnel")
plt.show()

analysable = report["status"].isin(["clean", "clean_warned", "repaired"]).mean()
cfg.append_result({"notebook": "01", "name": "repair_funnel",
                   "n_submitted": len(report),
                   "analysable_after_repair": float(analysable),
                   "manual_review": int(funnel.get("manual_review", 0)),
                   "mean_iou_repaired": float(report.loc[report.status=="repaired", "iou_vs_truth"].mean()),
                   "per_class_auto_resolved": rq["auto_resolved"].round(3).to_dict()})
print(f"analysable after auto-repair: {analysable:.1%}")
"""),
    md("""
### Hand-off: the analysis set

Downstream chapters must only ever see plots the gate passed. We save three
artefacts — the truth (answer key, kept for verification in chapter 06), the
raw submission (so the whole gauntlet is replayable), and **the analysis set**:
repaired geometry plus, per plot, its area, its geometry status and any
warnings that must be carried into the final verdict (an overlap warning at
intake is still a caveat on the compliance verdict three chapters later —
that thread must not be dropped).
"""),
    code("""
import geopandas as gpd

def to_gdf(geoms, extra: dict) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({**extra, "geometry": geoms}, crs="EPSG:4326")

to_gdf(truth, {"plot_id": [str(i) for i in range(len(truth))]}) \\
    .to_file(cfg.OUTPUT_DIR / "plots_truth.geojson", driver="GeoJSON")

to_gdf(submitted, {"plot_id": [str(i) for i in range(len(submitted))],
                   "injected": [x or "" for x in injected],
                   "declared_area_ha": declared}) \\
    .to_file(cfg.OUTPUT_DIR / "plots_submitted.geojson", driver="GeoJSON")

analysis_mask = report["status"].isin(["clean", "clean_warned", "repaired"]).to_numpy()
warnings_join = [
    "; ".join(sorted({i.code for i in issues_per_plot[k] if i.severity == G.WARNING}))
    for k in range(len(submitted))
]
analysis = to_gdf(
    [g for g, m in zip(repaired_geoms, analysis_mask) if m],
    {"plot_id": [str(k) for k, m in enumerate(analysis_mask) if m],
     "area_ha": [G.area_ha(g) for g, m in zip(repaired_geoms, analysis_mask) if m],
     "geometry_status": report.loc[analysis_mask, "status"].tolist(),
     "geometry_warnings": [w for w, m in zip(warnings_join, analysis_mask) if m]},
)
analysis.to_file(cfg.OUTPUT_DIR / "plots_analysis.geojson", driver="GeoJSON")
report.to_csv(cfg.OUTPUT_DIR / "validation_report.csv", index=False)

print(f"analysis set: {len(analysis)} plots -> {cfg.OUTPUT_DIR / 'plots_analysis.geojson'}")
analysis.head()
"""),
    md("""
### What this chapter established

1. **Detection is a solved problem for 10 of 11 failure classes** — the
   validator catches them at or near 100%, with a low false-alarm rate on
   clean plots (the exact numbers are in the ledger, `outputs/results.json`,
   and are re-verified in chapter 06).
2. **Repair splits cleanly into three regimes**: mechanical fixes with
   near-perfect recovery (axis swaps, Web-Mercator un-projection, duplicate
   vertices, winding, Z-noise); partial recovery with a recorded cost
   (bow-ties keep the dominant lobe); and honest refusals (teleported, micro,
   sliver) where inventing geometry would be a compliance liability.
3. **The audit trail is the product**: every repaired plot carries the exact
   actions applied, and warnings survive into the final verdict.

**Next:** chapter 02 asks the question the entire regulation hinges on — *was
each of these plots forest on 31 December 2020?* — and gets two official,
disagreeing answers.
"""),
]

save(cells, "01_geometry_gauntlet.ipynb")
