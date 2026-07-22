"""Build notebook 05 — verdicts, risk fusion, evidence bundles, DDS roll-up."""

from nbbuild import bootstrap_cells, code, md, save

cells = [
    md("""
# 05 — The verdict: risk fusion and audit-ready evidence

**Question this notebook answers:** given everything the pipeline now knows
about each plot — geometry audit, two forest baselines, a breakpoint
detector, a calibrated CNN — what is the *defensible* risk verdict, and what
does the auditor-facing evidence for it look like?

**Why the fusion is rules, not another model.** Two reasons, one statistical
and one institutional. Statistical: there is no labelled dataset of
adjudicated EUDR outcomes to learn a fusion from — a learned combiner would
be trained on proxies of proxies. Institutional: a due-diligence decision
must survive an auditor asking *why*. "HIGH because: forest at cutoff on
both baselines; NBR breakpoint 2022-07 confirmed by 4 observations; CNN
p=0.94" survives that question. "HIGH because an ensemble said 0.83" does
not. So the machine learning lives in the *evidence* (detectors, baselines,
calibration) and the *fusion* stays transparent — which mirrors how
production compliance systems are actually structured.

**The four tiers** (full logic in `geoverdict/risk.py`, unit-tested):
- **LOW** — analysable, and no credible post-cutoff deforestation signal
  (including: was simply not forest at the cutoff — clearing pasture is not
  deforestation under EUDR).
- **MEDIUM** — analysable, but the evidence conflicts (maps disagree on the
  baseline; detectors disagree with each other) or confidence is degraded.
- **HIGH** — forest at cutoff *and* a corroborated post-cutoff clearing.
- **INSUFFICIENT_EVIDENCE** — the pipeline *admits ignorance*: unrepaired
  geometry, or too few cloud-free observations to honestly say "no change".
  This tier is the most important design decision in the file: a silent LOW
  on an unobservable plot is how non-compliant beef gets certified.

**Produces**
- `outputs/verdicts.csv`, `outputs/evidence/plot_*.json`, `outputs/dds_summary.json`
- `figures/g05_*.png`

**Expected runtime:** ~4 minutes, CPU.
"""),
    *bootstrap_cells(),
    code("""
import geopandas as gpd
import numpy as np
import pandas as pd

plots = gpd.read_file(cfg.OUTPUT_DIR / "plots_analysis.geojson")
baseline = pd.read_csv(cfg.OUTPUT_DIR / "baseline.csv", dtype={"plot_id": str})
detections = pd.read_csv(cfg.OUTPUT_DIR / "ts_detections.csv", dtype={"plot_id": str})
cnn = pd.read_csv(cfg.OUTPUT_DIR / "cnn_predictions.csv", dtype={"plot_id": str})

df = (plots.merge(baseline, on="plot_id", suffixes=("", "_b"))
           .merge(detections, on="plot_id", how="left")
           .merge(cnn[["plot_id", "model_prob"]], on="plot_id", how="left"))
print(f"{len(df)} plots entering the verdict layer")
print(f"  with time-series screening: {df.break_detected.notna().sum()}")
print(f"  with CNN probability:       {df.model_prob.notna().sum()}")
"""),
    md("""
Note what the merge just made visible: not every plot has every signal (the
time-series arm screened a stratified subset; chips were lost to cloud for a
few plots). The fusion handles partial evidence *explicitly* — a missing
model probability changes which rule fires, it does not crash a join or
silently default to 0.5. Production pipelines live in this partial-evidence
regime permanently.
"""),
    code("""
from geoverdict import risk

verdicts = []
for r in df.itertuples():
    v = risk.assess_plot(
        plot_id=r.plot_id,
        geometry_ok=True,   # everything in the analysis set passed chapter 01
        geometry_warnings=[w for w in str(getattr(r, "geometry_warnings", "") or "").split("; ") if w],
        forest_frac_jrc=None if pd.isna(r.forest_frac_jrc) else float(r.forest_frac_jrc),
        forest_frac_hansen=None if pd.isna(r.forest_frac_hansen) else float(r.forest_frac_hansen),
        ts_break_detected=None if pd.isna(r.break_detected) else bool(r.break_detected),
        ts_break_date=None if pd.isna(r.break_date) else str(r.break_date),
        ts_obs_density=None if pd.isna(r.obs_density) else float(r.obs_density),
        model_prob=None if pd.isna(r.model_prob) else float(r.model_prob),
        hansen_loss_post_frac=None if pd.isna(r.hansen_loss_post_frac) else float(r.hansen_loss_post_frac),
    )
    verdicts.append(v)

vdf = pd.DataFrame([{"plot_id": v.plot_id, "tier": v.tier,
                     "reasons": " | ".join(v.reasons)} for v in verdicts])
vdf.to_csv(cfg.OUTPUT_DIR / "verdicts.csv", index=False)
tier_counts = vdf.tier.value_counts()
print(tier_counts)
cfg.append_result({"notebook": "05", "name": "tier_counts",
                   **{k: int(v) for k, v in tier_counts.items()}})
"""),
    code("""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

geo = plots.set_index("plot_id").geometry
viz.portfolio_map(axes[0], [geo.get(v.plot_id) for v in verdicts],
                  [v.tier for v in verdicts], viz.TIER_COLORS, legend_title="verdict")
axes[0].set_title("The portfolio, screened: every plot has a verdict")

order = ["LOW", "MEDIUM", "HIGH", "INSUFFICIENT_EVIDENCE"]
vals = [int(tier_counts.get(t, 0)) for t in order]
axes[1].bar(order, vals, color=[viz.TIER_COLORS[t] for t in order])
for x, v_ in enumerate(vals):
    axes[1].text(x, v_, f"{v_}\\n({v_/len(vdf):.0%})", ha="center", va="bottom")
axes[1].set_title("Tier distribution")
axes[1].tick_params(axis="x", labelsize=8)
fig.tight_layout()
viz.save(fig, "g05_verdict_map")
plt.show()
"""),
    md("""
### Why did each plot get its tier? Reasons are data, not prose

Every verdict carries machine-readable reasons. Aggregating them turns the
portfolio into an *operations dashboard*: how much MEDIUM comes from
baseline-map disagreement (a data-sourcing problem) vs detector disagreement
(a modelling problem) vs thin observation records (a physics problem)? Each
cause has a different owner and a different fix — this chart is what a team
lead would actually triage from.
"""),
    code("""
def reason_category(reason: str) -> str:
    if "baselines disagree" in reason: return "baseline maps disagree"
    if "detector disagrees" in reason or "without a confirmed" in reason: return "detectors disagree"
    if "monitoring months" in reason or "record is thin" in reason: return "thin observation record"
    if "breakpoint" in reason and "concurs" in reason: return "corroborated clearing"
    if "non-forest at the EUDR cutoff" in reason: return "not forest at cutoff"
    if "geometry" in reason: return "geometry caveat"
    if "no detector" in reason: return "clean monitored forest"
    return "other"

cats = []
for v in verdicts:
    for reason in v.reasons:
        cats.append({"tier": v.tier, "category": reason_category(reason)})
cat_df = pd.DataFrame(cats)
pivot = cat_df.value_counts(["category", "tier"]).unstack(fill_value=0)[
    [c for c in order if c in cat_df.tier.unique()]]

fig, ax = plt.subplots(figsize=(9, 4.5))
bottom = np.zeros(len(pivot))
for tier in pivot.columns:
    ax.barh(pivot.index, pivot[tier], left=bottom, color=viz.TIER_COLORS[tier], label=tier)
    bottom += pivot[tier].to_numpy()
ax.legend(fontsize=8)
ax.set_title("Why plots got their verdicts - evidence categories by tier")
ax.set_xlabel("number of reasons recorded")
fig.tight_layout()
viz.save(fig, "g05_reason_categories")
plt.show()
"""),
    md("""
### The business translation: what does screening cost?

The tier distribution *is* an analyst workload: every HIGH and MEDIUM (and
arguably every INSUFFICIENT) lands on a human's desk. At an assumed 15
minutes of review per flagged plot, the chart below converts the portfolio
into review-hours — and shows the trade the thresholds are making. This is
the number a deployment discussion actually turns on, and it is also the
frame in which "INSUFFICIENT_EVIDENCE means buy radar data or wait a season"
becomes a costed decision instead of a shrug.
"""),
    code("""
REVIEW_MIN = 15
needs_review = vdf.tier.isin(["HIGH", "MEDIUM", "INSUFFICIENT_EVIDENCE"])
per_1000 = 1000 * needs_review.mean()
hours_per_1000 = per_1000 * REVIEW_MIN / 60

print(f"plots needing human review: {needs_review.sum()}/{len(vdf)} "
      f"= {per_1000:.0f} per 1,000 plots")
print(f"analyst workload at {REVIEW_MIN} min/plot: {hours_per_1000:.0f} h per 1,000 plots screened")
print(f"fully automated (LOW): {(~needs_review).mean():.0%} of the portfolio")

cfg.append_result({"notebook": "05", "name": "screening_cost",
                   "review_per_1000": float(per_1000),
                   "hours_per_1000": float(hours_per_1000),
                   "automated_frac": float((~needs_review).mean())})
"""),
    md("""
### Evidence bundles: what the auditor opens

For every plot the pipeline emits a JSON bundle — verdict + reasons, the
geometry audit trail, both baseline values, the monthly NDVI/NBR series
*with its gaps preserved* (the months we could not observe are part of the
evidence), the breakpoint, the calibrated model probability, and provenance
down to satellite scene ids and map asset versions. The one-page figure is
the same content made legible in ten seconds.

We render three showcase bundles spanning the interesting tiers: a
corroborated HIGH, a MEDIUM where the official maps disagree, and an
INSUFFICIENT where the honest answer is "the optical record cannot say".
"""),
    code("""
import json
from geoverdict import evidence, timeseries as ts

monthly_df = pd.read_parquet(cfg.OUTPUT_DIR / "series_monthly.parquet")
monthly_df.columns = pd.MultiIndex.from_tuples(
    [tuple(c.split("|")) for c in monthly_df.columns], names=["plot_id", "index"])
prov = [json.loads((cfg.OUTPUT_DIR / "baseline_provenance.json").read_text()),
        {"source": "sentinel-2-l2a via Earth Search STAC + COPERNICUS/S2_SR_HARMONIZED (GEE)",
         "identifiers": ["see chips.npz / series_raw.parquet"],
         "created": "", "parameters": {"chip_bands": list(cfg.CHIP_BANDS)}}]

by_id = {v.plot_id: v for v in verdicts}
screened = [pid for pid in detections.plot_id if pid in by_id]

def first_where(pred):
    for pid in screened:
        if pred(by_id[pid]):
            return pid
    return None

showcase = {
    "high": first_where(lambda v: v.tier == "HIGH"),
    "medium": first_where(lambda v: v.tier == "MEDIUM"),
    "insufficient": first_where(lambda v: v.tier == "INSUFFICIENT_EVIDENCE"),
}
print("showcase plots:", showcase)

n_saved = 0
for pid in screened:
    v = by_id[pid]
    ndvi_s = monthly_df[(pid, "ndvi")] if (pid, "ndvi") in monthly_df.columns else None
    nbr_s = monthly_df[(pid, "nbr")] if (pid, "nbr") in monthly_df.columns else None
    br = ts.detect_break(nbr_s, cfg.CUTOFF_DATE).to_dict() if nbr_s is not None else None
    prow = plots.set_index("plot_id").loc[pid]
    bundle = evidence.build_bundle(
        v, {"plot_id": pid, "area_ha": float(prow.area_ha),
            "centroid": [float(prow.geometry.centroid.x), float(prow.geometry.centroid.y)],
            "geometry_status": str(prow.geometry_status),
            "geometry_warnings": str(prow.geometry_warnings or "")},
        ndvi_series=ndvi_s, nbr_series=nbr_s, break_result=br, provenance=prov)
    evidence.save_bundle(bundle)
    n_saved += 1
print(f"saved {n_saved} evidence bundles -> {cfg.EVIDENCE_DIR}")
"""),
    code("""
# chips for the showcase figures come from the chapter-04 cache
z = np.load(cfg.OUTPUT_DIR / "chips.npz", allow_pickle=True)
chip_lookup = {s: i for i, s in enumerate(z["sample_id"])}

for name, pid in showcase.items():
    if pid is None:
        continue
    bundle = json.loads((cfg.EVIDENCE_DIR / f"plot_{pid}.json").read_text())
    i = chip_lookup.get(pid)
    fig = viz.evidence_figure(bundle,
                              chip_t1=z["x1"][i] if i is not None else None,
                              chip_t2=z["x2"][i] if i is not None else None)
    viz.save(fig, f"g05_evidence_{name}")
    plt.show()
"""),
    md("""
### The portfolio roll-up: what feeds a Due Diligence Statement

Under EUDR, the operator files a DDS asserting negligible risk *for the
portfolio*, backed by per-plot geolocation and evidence. The roll-up below is
the shape of that support: tier counts, and every non-LOW plot listed by id
with its reasons — because those are exactly the plots a human must resolve
before the statement is signed. (This is deliberately *support for* a DDS,
not a DDS: the legal document has registry fields — operator identity, HS
codes, quantities — that are out of scope for a screening prototype, and
pretending otherwise would be the kind of overclaim this project avoids.)
"""),
    code("""
summary = evidence.portfolio_summary(verdicts, cfg.OUTPUT_DIR / "dds_summary.json")
print(json.dumps({k: v for k, v in summary.items() if k != "attention_required"}, indent=2))
print(f"\\nplots requiring attention: {len(summary['attention_required'])} "
      f"(first 3 shown)")
for item in summary["attention_required"][:3]:
    print(f"\\n  plot {item['plot_id']} [{item['tier']}]")
    for reason in item["reasons"]:
        print(f"    - {reason}")
"""),
    md("""
### What this chapter established

1. **The pipeline ends in a decision, with reasons** — every plot carries a
   tier and the evidence trail that produced it; nothing terminates in a bare
   probability.
2. **Partial evidence is handled explicitly** — plots outside the
   time-series subset, or whose chips were clouded out, get verdicts from
   the evidence that exists, with the gaps named.
3. **INSUFFICIENT_EVIDENCE is a feature** — the share of the portfolio the
   optical record genuinely cannot certify is now a measured number with a
   costed remedy (radar, or another dry season), not a silent LOW.
4. **The evidence bundle is the product**: verdict, series, chips,
   provenance — the artefact a compliance officer, or a court, would open.

**Next and last:** chapter 06 turns on the pipeline's own headlights —
re-verifying every ledger claim, ablating the detector's knobs, quantifying
seed variance, and demonstrating (by doing it wrong on purpose) how much a
random split would have flattered every number in this project.
"""),
]

save(cells, "05_verdicts_evidence.ipynb")
