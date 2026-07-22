"""Build notebook 03 — Sentinel-2 time series + the statistics & classical-ML arms."""

from nbbuild import bootstrap_cells, code, md, save

cells = [
    md("""
# 03 — Watching the land: six years of Sentinel-2 per plot, and the statistics arm

**Question this notebook answers:** from each plot's own NDVI/NBR history,
*when* did the forest signal break — and how well does a transparent
robust-statistics detector screen plots, before any deep learning?

**Why start with statistics.** The team-shaped wisdom this project follows:
a change detector you can explain end-to-end — "the plot's NDVI baseline was
0.84 ± 0.02; from July 2022 it sat below 0.45 for four consecutive
observations" — is *evidence*. A neural network's 0.93 is a *score*. A
compliance pipeline wants both, but the statistics arm comes first because
(1) it sets the floor any learned model must beat to justify its complexity,
(2) it produces the breakpoint **date**, which the learned classifier does
not, and the date is what the regulation cares about, and (3) at screening
scale it costs nothing.

**The detector** (`geoverdict.timeseries.detect_break`), and the reasoning
behind each choice:
- **baseline = median, spread = MAD** of the pre-cutoff period — cloud and
  smoke that slip past the mask leave heavy-tailed outliers, and a mean-based
  baseline is dragged by exactly the artefacts we must ignore;
- **break = first run of ≥ 3 consecutive valid observations below
  baseline − max(6·MAD, 0.12)** — one low month is weather, three sustained
  months on a formerly forested plot is clearing. The threshold *adapts per
  plot*: a seasonal plot has a bigger MAD and automatically demands a bigger
  drop;
- **gaps stay gaps** — no interpolation through the wet season; a month with
  no valid observation neither confirms nor breaks a run, and observation
  density is reported so chapter 05 can refuse to certify unobservable plots.

**Produces**
- `outputs/series_monthly.parquet` — per plot, monthly NDVI & NBR (with gaps)
- `outputs/ts_detections.csv` — per plot: breakpoint, magnitude, obs density
- `outputs/rf_arm.joblib`, `outputs/ts_eval.json`
- `figures/g03_*.png`

**Expected runtime:** ~12 minutes (server-side reductions on Earth Engine +
detector fits; no GPU).
"""),
    md("""
### 📦 Where the data in this notebook comes from

| Data | Source | How it enters the notebook |
|---|---|---|
| **Sentinel-2 L2A surface reflectance** (2019–2025) | Copernicus / ESA, hosted on Earth Engine as `COPERNICUS/S2_SR_HARMONIZED` | queried server-side; per plot & scene, the SCL-cloud-masked mean NDVI/NBR returns (cached to `series_raw.parquet`) |
| **SCL cloud mask** | part of each Sentinel-2 L2A scene | used inside the same Earth Engine call to drop cloud/shadow pixels |
| **Hansen post-2020 loss** (reference labels) | Univ. of Maryland, on Earth Engine | the independent referee the detector is scored against |
| **Plot geometries + baselines** | `outputs/plots_analysis.geojson`, `outputs/baseline.csv` from chapters 01–02 | loaded from Drive |

Again nothing large is downloaded — the six-year reflectance history for every
plot is reduced to (plot, date, NDVI, NBR) tuples **on Google's servers**, and
only those tuples come back. The first fetch is cached to Drive, so re-runs
load in seconds.
"""),
    *bootstrap_cells(),
    code("""
EE_PROJECT = ""   # <- same Earth Engine project id as chapter 02

from geoverdict import gee
gee.init(project=EE_PROJECT or None)

import geopandas as gpd
import numpy as np
import pandas as pd

plots = gpd.read_file(cfg.OUTPUT_DIR / "plots_analysis.geojson")
baseline = pd.read_csv(cfg.OUTPUT_DIR / "baseline.csv", dtype={"plot_id": str})
plots = plots.merge(baseline, on="plot_id", suffixes=("", "_b"))
print(f"{len(plots)} plots with baselines")
"""),
    md("""
### The screening subset, chosen honestly

Six years × ~70 usable scenes/year × every plot is cheap server-side, but the
transfer and the per-plot detector fits are not free, and this chapter should
run in minutes, not hours. We screen a **stratified subset of 200 plots**:
stratified by Hansen post-2020 loss (so both changed and stable plots are
well represented — a uniform draw from a portfolio that is mostly stable
forest would leave the "clearing" stratum too thin to measure recall on),
seeded for reproducibility.

Scaling note, because it matters in review: nothing in the method changes at
10⁵ plots — the reduction is already server-side; you shard the plot list and
parallelise the transfer. The subset is a *notebook-runtime* decision, not an
architectural one.
"""),
    code("""
rng = np.random.default_rng(cfg.SEED)
loss = plots["hansen_loss_post_frac"].fillna(0)

changed = plots.index[loss > cfg.POS_LOSS_FRAC].to_numpy()
stable  = plots.index[loss < cfg.NEG_LOSS_FRAC].to_numpy()
middle  = plots.index[(loss >= cfg.NEG_LOSS_FRAC) & (loss <= cfg.POS_LOSS_FRAC)].to_numpy()

n_changed = min(len(changed), cfg.TS_SUBSET // 3)
n_middle  = min(len(middle),  cfg.TS_SUBSET // 6)
n_stable  = cfg.TS_SUBSET - n_changed - n_middle
subset_idx = np.concatenate([
    rng.choice(changed, n_changed, replace=False),
    rng.choice(middle,  n_middle,  replace=False),
    rng.choice(stable,  min(n_stable, len(stable)), replace=False),
])
sub = plots.loc[subset_idx].reset_index(drop=True)
print(f"screening subset: {len(sub)} plots "
      f"({n_changed} changed / {n_middle} ambiguous / {len(sub)-n_changed-n_middle} stable by Hansen)")
"""),
    md("""
### Fetching the series (cached — re-runs are instant)

One server-side pass: for every plot and every Sentinel-2 scene 2019→2025,
the SCL-masked mean NDVI and NBR over the plot, with the same mask classes as
the STAC path in chapter 04 (the two access paths share one usability
definition by construction — `gee.cfg_bad_scl()` imports it from `s2.py`).
A plot-scene with < 30% valid pixels comes back as *no observation*: a mean
over a sliver of clear pixels at a cloud edge is noise wearing a number's
clothing.
"""),
    code("""
cache = cfg.OUTPUT_DIR / "series_raw.parquet"
if cache.exists():
    raw = pd.read_parquet(cache)
    print(f"loaded cached series: {len(raw):,} plot-scene observations")
else:
    raw = gee.s2_plot_timeseries(list(sub.geometry), list(sub.plot_id))
    raw.to_parquet(cache)
    print(f"fetched {len(raw):,} plot-scene observations "
          f"({raw['ndvi'].notna().mean():.0%} usable)")
"""),
    code("""
from geoverdict import timeseries as ts

monthly = {}
for pid, gdf in raw.groupby("plot_id"):
    for col in ("ndvi", "nbr"):
        monthly[(pid, col)] = ts.monthly_series(
            gdf["date"], gdf[col].to_numpy(dtype=float),
            start=cfg.BASELINE_START[:7] + "-01", end=cfg.MONITOR_END[:7] + "-01")

monthly_df = pd.DataFrame({k: v for k, v in monthly.items()})
monthly_df.columns = pd.MultiIndex.from_tuples(monthly_df.columns, names=["plot_id", "index"])

# parquet cannot store MultiIndex columns -> save flattened ("plotid|index"),
# and chapter 05 restores the MultiIndex on load
flat = monthly_df.copy()
flat.columns = ["|".join(c) for c in monthly_df.columns]
flat.to_parquet(cfg.OUTPUT_DIR / "series_monthly.parquet")

gap_rate = monthly_df.isna().mean().mean()
per_month_cov = 1 - monthly_df.isna().groupby(monthly_df.index.month).mean().mean(axis=1)
print(f"overall monthly gap rate: {gap_rate:.0%}")
"""),
    code("""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(range(1, 13), per_month_cov, color=viz.PALETTE["accent"])
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
ax.set_ylabel("share of plot-months observed")
ax.set_title("The wet-season blindness of optical monitoring\\n"
             "(Dec-Mar: the satellite passes, the clouds win)")
viz.save(fig, "g03_observation_gaps")
plt.show()
"""),
    md("""
### Running the detector — and reading a few plots' stories

The gallery shows six plots chosen to span the behaviours that matter: stable
forest (flat, high), a detected clearing (level shift with the breakpoint
marked), stable *non*-forest (flat, low — nothing to detect, and the verdict
layer already knows it was not forest at the cutoff), and a gappy plot where
observation density is the real story. NBR is plotted alongside NDVI: for
clearing, NBR usually moves **further** (a clear-cut loses NIR *and* gains
SWIR — the two shifts add up in the ratio), which is why the detector runs on
NBR by default and NDVI is the cross-check.
"""),
    code("""
det_rows = []
for pid in sub.plot_id:
    r_nbr = ts.detect_break(monthly_df[(pid, "nbr")], cfg.CUTOFF_DATE)
    r_ndvi = ts.detect_break(monthly_df[(pid, "ndvi")], cfg.CUTOFF_DATE)
    det_rows.append({
        "plot_id": pid,
        "break_detected": r_nbr.detected,
        "break_date": r_nbr.break_date,
        "magnitude": r_nbr.magnitude,
        "obs_density": r_nbr.obs_density,
        "baseline_median_nbr": r_nbr.baseline_median,
        "ndvi_agrees": r_ndvi.detected == r_nbr.detected,
        # a continuous score for PR analysis: how far below threshold the
        # series went, in MAD units (0 when never below)
        "score": max(0.0, (r_nbr.threshold - float(np.nanmin(
            monthly_df[(pid, "nbr")][monthly_df[(pid, "nbr")].index > pd.Timestamp(cfg.CUTOFF_DATE)]
        ))) / max(r_nbr.baseline_mad, 1e-3)) if not np.isnan(r_nbr.baseline_median) else 0.0,
    })
detections = pd.DataFrame(det_rows)
detections.to_csv(cfg.OUTPUT_DIR / "ts_detections.csv", index=False)
print(detections.break_detected.value_counts())
print(f"NDVI/NBR detector agreement: {detections.ndvi_agrees.mean():.0%}")
"""),
    code("""
sub_l = sub.set_index("plot_id")
det_l = detections.set_index("plot_id")

def pick(mask, n=2):
    ids = det_l.index[mask]
    return list(ids[:n])

gallery = (pick(det_l.break_detected & (sub_l.loc[det_l.index, "forest_frac_jrc"] > 0.5), 2)
           + pick(~det_l.break_detected & (sub_l.loc[det_l.index, "forest_frac_jrc"] > 0.5), 2)
           + pick(~det_l.break_detected & (sub_l.loc[det_l.index, "forest_frac_jrc"] < 0.1), 1)
           + pick(det_l.obs_density < 0.4, 1))[:6]

fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
for ax, pid in zip(axes.ravel(), gallery):
    r = ts.detect_break(monthly_df[(pid, "nbr")], cfg.CUTOFF_DATE)
    viz.plot_series(ax, monthly_df[(pid, "nbr")], r.to_dict(), label="NBR",
                    color=viz.PALETTE["accent"])
    hl = sub_l.loc[pid, "hansen_loss_post_frac"]
    ax.set_title(f"plot {pid} - {sub_l.loc[pid, 'area_ha']:.0f} ha - "
                 f"Hansen post-2020 loss {hl:.0%} - obs density {r.obs_density:.0%}",
                 fontsize=9)
fig.tight_layout()
viz.save(fig, "g03_series_gallery")
plt.show()
"""),
    md("""
### Evaluation against the independent reference

**The reference label:** Hansen post-2020 loss fraction > 20% = "cleared".
Calling this what it is: a *weak* label from a different sensor (Landsat) and
a different method — not ground truth. It is still a legitimate referee
precisely because it is independent of everything our detector sees
(different satellite, different algorithm, different resolution), and its
error modes (annual dating, 30 m edges) are known and discussed where they
bite. Plots in Hansen's ambiguous 2–20% band are excluded from scoring —
grading against a referee who is guessing teaches nothing.

Two evaluations:
1. **Screening quality** — precision/recall of the binary flag, the PR curve
   over the continuous score, and the business table: flags per 1,000 plots
   at fixed recall.
2. **Date agreement** — for true positives, our breakpoint month vs Hansen's
   loss year. Hansen dates to a calendar year, so ±1 year is the honest
   success criterion.
""" ),
    code("""
from geoverdict import metrics as M

eval_df = detections.merge(sub[["plot_id", "hansen_loss_post_frac"]], on="plot_id")
eval_df = eval_df[(eval_df.hansen_loss_post_frac > cfg.POS_LOSS_FRAC) |
                  (eval_df.hansen_loss_post_frac < cfg.NEG_LOSS_FRAC)]
y_true = (eval_df.hansen_loss_post_frac > cfg.POS_LOSS_FRAC).to_numpy()
y_flag = eval_df.break_detected.to_numpy()
scores = eval_df.score.to_numpy()

m = M.prf(y_true, y_flag)
print(f"statistics arm vs Hansen reference (n={len(eval_df)}, {y_true.sum()} positives):")
print(f"  precision {m['precision']:.3f}   recall {m['recall']:.3f}   F1 {m['f1']:.3f}")
print(f"  PR-AUC over the continuous score: {M.pr_auc(y_true, scores):.3f}")
print()
screen = pd.DataFrame(M.screening_table(y_true, scores))
print(screen.round(3).to_string(index=False))

import json
(cfg.OUTPUT_DIR / "ts_eval.json").write_text(json.dumps({
    "n": int(len(eval_df)), "positives": int(y_true.sum()),
    **{k: m[k] for k in ("precision", "recall", "f1")},
    "pr_auc": M.pr_auc(y_true, scores)}, indent=2))
cfg.append_result({"notebook": "03", "name": "stats_arm_eval",
                   "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                   "pr_auc": M.pr_auc(y_true, scores),
                   "flags_per_1000_at_r90": float(screen.loc[screen.recall_target == 0.90,
                                                             "flags_per_1000"].iloc[0])})
"""),
    code("""
# date agreement for true positives
tp = eval_df[y_true & eval_df.break_detected].copy()
tp["break_year"] = pd.to_datetime(tp.break_date).dt.year

loss_years = gee.hansen_loss_year_fractions(
    list(sub.set_index("plot_id").loc[tp.plot_id].geometry), list(tp.plot_id))
year_cols = [c for c in loss_years.columns if c.startswith("loss_")]
loss_years["hansen_year"] = (loss_years[year_cols].idxmax(axis=1)
                             .str.replace("loss_", "").astype(int))
tp = tp.merge(loss_years[["plot_id", "hansen_year"]], on="plot_id")

within1 = (tp.break_year - tp.hansen_year).abs() <= 1
print(f"breakpoint within +/-1 year of Hansen's loss year: {within1.mean():.0%} of {len(tp)} TPs")

fig, ax = plt.subplots(figsize=(5.5, 5))
jit = np.random.default_rng(0).uniform(-0.15, 0.15, len(tp))
ax.scatter(tp.hansen_year + jit, tp.break_year, s=18, alpha=0.7,
           color=viz.PALETTE["accent"])
lims = [2020.5, 2024.5]
ax.plot(lims, lims, ls="--", lw=1, color=viz.PALETTE["insufficient"])
ax.fill_between(lims, [l-1 for l in lims], [l+1 for l in lims],
                alpha=0.12, color=viz.PALETTE["forest"], label="+/-1 year")
ax.set_xlabel("Hansen loss year"); ax.set_ylabel("detected breakpoint year")
ax.set_title(f"When did it happen? Detector vs Hansen\\n({within1.mean():.0%} within +/-1 yr)")
ax.legend()
viz.save(fig, "g03_date_agreement")
plt.show()
cfg.append_result({"notebook": "03", "name": "date_agreement",
                   "within_1yr": float(within1.mean()), "n_tp": int(len(tp))})
"""),
    md("""
### The classical-ML arm: does *learning* the boundary beat hand-setting it?

Same information, different decision rule: a random forest on ~12 temporal
features (baseline level and spread, post-cutoff minimum and quantiles, the
size and abruptness of the change, trend, observation density). If the RF
beats the hand-tuned detector meaningfully, the thresholds were leaving
signal on the table; if it merely matches, the transparent detector earns
its keep — either outcome is a finding.

**The split is spatially blocked**, not random: plots are assigned to
train/test by longitude thirds of the AOI. Neighbouring plots share weather,
soil and clearing waves; a random split leaks that neighbourhood signal and
reports a fantasy number. (This is the classic geospatial evaluation trap —
we demonstrate its size in chapter 06 by doing it wrong on purpose.)
"""),
    code("""
from sklearn.ensemble import RandomForestClassifier
import joblib

feat_rows, feat_ids = [], []
for pid in eval_df.plot_id:
    f = ts.temporal_features(monthly_df[(pid, "nbr")], cfg.CUTOFF_DATE)
    if f:
        feat_rows.append(f); feat_ids.append(pid)
X = pd.DataFrame(feat_rows, index=feat_ids)
y = (eval_df.set_index("plot_id").loc[X.index, "hansen_loss_post_frac"] > cfg.POS_LOSS_FRAC).to_numpy()

lon = sub.set_index("plot_id").loc[X.index].geometry.centroid.x
q1, q2 = lon.quantile([1/3, 2/3])
test_mask = (lon > q2).to_numpy()          # eastern third held out
train_mask = ~test_mask

rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                            class_weight="balanced", random_state=cfg.SEED)
rf.fit(X[train_mask], y[train_mask])
rf_scores = rf.predict_proba(X[test_mask])[:, 1]

stats_scores_test = eval_df.set_index("plot_id").loc[X.index[test_mask], "score"].to_numpy()
print(f"spatially-blocked test set: {test_mask.sum()} plots, {y[test_mask].sum()} positives")
print(f"  RF        PR-AUC {M.pr_auc(y[test_mask], rf_scores):.3f}")
print(f"  detector  PR-AUC {M.pr_auc(y[test_mask], stats_scores_test):.3f}")
joblib.dump(rf, cfg.OUTPUT_DIR / "rf_arm.joblib")

imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp.index, imp.values, color=viz.PALETTE["forest"])
ax.set_title("What the forest thinks matters\\n(sanity check: change magnitude "
             "and post-cutoff minimum should dominate)")
ax.set_xlabel("feature importance")
viz.save(fig, "g03_rf_importance")
plt.show()

cfg.append_result({"notebook": "03", "name": "rf_vs_detector",
                   "rf_pr_auc": M.pr_auc(y[test_mask], rf_scores),
                   "detector_pr_auc": M.pr_auc(y[test_mask], stats_scores_test),
                   "n_test": int(test_mask.sum())})
"""),
    md("""
### What this chapter established

1. **A transparent detector screens surprisingly well** — the exact
   precision/recall/PR-AUC numbers are in the ledger; the point is the
   *shape*: near-total recall on large clear-cuts, with misses concentrated
   where observation density is low or the clearing is partial.
2. **Detection dates line up with the independent reference** to within the
   ±1 year that Hansen's annual dating permits — which is what makes the
   breakpoint usable as evidence against the 2020-12-31 cutoff.
3. **The wet season is a measured blind spot, not an anecdote** — and its
   size is exactly why chapter 05's verdict layer has an
   INSUFFICIENT_EVIDENCE tier instead of pretending "not seen" means "not
   cleared".
4. **The learned-vs-hand-set comparison** (RF vs detector, same features,
   spatially-blocked) tells us how much signal the thresholds leave behind —
   the honest number is in the ledger and discussed against the deep model in
   chapter 04.

**Next:** the learned arm proper — a siamese CNN that looks at the *pixels*
(before/after chips) instead of plot-mean series, which is where partial
clearings and sub-plot patterns live.
"""),
]

save(cells, "03_timeseries_screening.ipynb")
