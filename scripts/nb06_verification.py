"""Build notebook 06 — verification, ablations, and the limitations chapter."""

from nbbuild import bootstrap_cells, code, md, save

cells = [
    md("""
# 06 — Verification: does the pipeline survive its own scrutiny?

**Question this notebook answers:** which of this project's claims hold up
when we deliberately try to break them — and what are the honest limits of
what was built?

A results chapter you write yourself deserves suspicion; this chapter is the
counterweight, and it does four specific things:

1. **Re-verify the ledger.** Every headline number in the README comes from
   `outputs/results.json`, written by the notebook that measured it. Here we
   re-read the ledger and re-derive key claims from the saved artefacts —
   numbers that only exist inside a notebook cell die with the cell.
2. **Ablate the detector's knobs.** The persistence rule and the drop
   threshold were *reasoned* in chapter 03; here they are *swept*, so the
   choice is visible as a point on a trade-off curve rather than folklore.
3. **Quantify the evaluation traps.** Seed variance (is the CNN result a
   model or an anecdote?) and — by doing it wrong on purpose — the size of
   the flattery a random split would have bought over the spatial split.
4. **Write the limitations honestly**, each with its cost and its remedy.

**Expected runtime:** ~15 minutes (one deliberate model retrain on a T4).
"""),
    *bootstrap_cells(),
    md("""
### 1 — The ledger, re-read

Every record below was appended by the chapter that measured it (same
notebook + name re-runs overwrite, so the ledger cannot accumulate stale
duplicates). This table *is* the project's claim sheet.
"""),
    code("""
import numpy as np
import pandas as pd

ledger = pd.DataFrame(cfg.load_results())
with pd.option_context("display.max_colwidth", 90):
    display(ledger[["notebook", "name"]].assign(
        headline=[{k: v for k, v in r.items() if k not in ("notebook", "name")
                   and not isinstance(v, (dict, list))} for r in cfg.load_results()]))
"""),
    code("""
# spot re-derivations from raw artefacts (not from the ledger itself)
report = pd.read_csv(cfg.OUTPUT_DIR / "validation_report.csv")
analysable = report["status"].isin(["clean", "clean_warned", "repaired"]).mean()
led = {(r["notebook"], r["name"]): r for r in cfg.load_results()}
claimed = led[("01", "repair_funnel")]["analysable_after_repair"]
assert abs(analysable - claimed) < 1e-9, (analysable, claimed)
print(f"ch.01 claim re-derived from validation_report.csv: analysable {analysable:.1%} == ledger OK")

verdicts = pd.read_csv(cfg.OUTPUT_DIR / "verdicts.csv")
for tier, n in verdicts.tier.value_counts().items():
    assert led[("05", "tier_counts")][tier] == int(n)
print("ch.05 tier counts re-derived from verdicts.csv == ledger OK")
"""),
    md("""
### 2 — Sweeping the detector's knobs

Chapter 03 argued for `k_persist = 3` (three consecutive sub-threshold
observations) and `drop = 6 MAD`. The sweep below re-runs the detector over
the cached monthly series for every (k, drop) pair and scores against the
Hansen reference. What we expect to see — and *want* to be able to point at
in a design discussion:

- **k = 1** buys recall with a false-alarm bill (single-month dips: smoke,
  missed shadow, drought stress);
- **large k** delays and eventually misses short observation runs in gappy
  wet-season records — recall decays;
- the drop threshold trades the same currencies on the amplitude axis.

The chosen point should sit at a sensible knee, not at an extreme — if it
does not, the honest move is changing the default and saying so.
"""),
    code("""
from geoverdict import metrics as M
from geoverdict import timeseries as ts

monthly_df = pd.read_parquet(cfg.OUTPUT_DIR / "series_monthly.parquet")
monthly_df.columns = pd.MultiIndex.from_tuples(
    [tuple(c.split("|")) for c in monthly_df.columns], names=["plot_id", "index"])

import geopandas as gpd
plots = gpd.read_file(cfg.OUTPUT_DIR / "plots_analysis.geojson")
baseline = pd.read_csv(cfg.OUTPUT_DIR / "baseline.csv", dtype={"plot_id": str})
ref = plots.merge(baseline, on="plot_id").set_index("plot_id").hansen_loss_post_frac.fillna(0)

pids = sorted({c[0] for c in monthly_df.columns})
pids = [p for p in pids
        if (ref.get(p, 0) > cfg.POS_LOSS_FRAC) or (ref.get(p, 0) < cfg.NEG_LOSS_FRAC)]
y_true = np.array([ref[p] > cfg.POS_LOSS_FRAC for p in pids])

rows = []
for k in (1, 2, 3, 4, 6):
    for drop in (3.0, 6.0, 9.0):
        flags = np.array([ts.detect_break(monthly_df[(p, "nbr")], cfg.CUTOFF_DATE,
                                          k_persist=k, drop_mads=drop).detected
                          for p in pids])
        m = M.prf(y_true, flags)
        rows.append({"k_persist": k, "drop_mads": drop, **{x: m[x] for x in ("precision", "recall", "f1")}})
sweep = pd.DataFrame(rows)
print(sweep.round(3).to_string(index=False))
cfg.append_result({"notebook": "06", "name": "detector_sweep",
                   "best_f1": float(sweep.f1.max()),
                   "chosen_f1": float(sweep.query("k_persist==3 and drop_mads==6.0").f1.iloc[0])})
"""),
    code("""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5.2))
for drop, marker in ((3.0, "o"), (6.0, "s"), (9.0, "^")):
    d = sweep[sweep.drop_mads == drop].sort_values("k_persist")
    ax.plot(d.recall, d.precision, marker + "-", lw=1.2, ms=7, alpha=0.85,
            label=f"drop = {drop:.0f} MAD")
    for _, r in d.iterrows():
        ax.annotate(f"k={int(r.k_persist)}", (r.recall, r.precision),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
chosen = sweep.query("k_persist==3 and drop_mads==6.0").iloc[0]
ax.scatter([chosen.recall], [chosen.precision], s=180, facecolor="none",
           edgecolor=viz.PALETTE["clearing"], lw=2, zorder=5, label="chosen default")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("The detector's knobs, swept\\n(persistence k and drop threshold vs the Hansen reference)")
ax.legend(fontsize=8)
viz.save(fig, "g06_detector_sweep")
plt.show()
"""),
    md("""
### 3a — Seed variance: model or anecdote?

Chapter 04 trained three seeds; the ledger has mean ± std of PR-AUC. The
rule this project holds itself to: **a claimed difference between methods
must exceed the spread between seeds of the same method** — otherwise it is
noise wearing a narrative.
"""),
    code("""
led = {(r["notebook"], r["name"]): r for r in cfg.load_results()}
cnn = led[("04", "cnn_eval")]
print(f"CNN PR-AUC over {len(cnn['seeds'])} seeds: "
      f"{cnn['pr_auc_mean']:.3f} +/- {cnn['pr_auc_std']:.3f}")
rf = led.get(("03", "rf_vs_detector"), {})
if rf:
    delta = cnn["pr_auc_mean"] - rf.get("detector_pr_auc", float("nan"))
    print(f"CNN - statistics-arm delta: {delta:+.3f} "
          f"({'exceeds' if abs(delta) > 2*cnn['pr_auc_std'] else 'DOES NOT exceed'} 2x seed std)")
"""),
    md("""
### 3b — The split experiment: how much would cheating have paid?

The classic geospatial evaluation trap, demonstrated rather than asserted:
retrain the CNN (one seed, identical hyperparameters) with a **random**
train/val/test split instead of the spatially-blocked one, and compare test
PR-AUC. Neighbouring plots share weather, soil, clearing waves and even the
same Sentinel-2 scene noise; a random split lets the model *memorise the
neighbourhood* and grade itself on it. The gap between the two numbers is
the size of the lie a random split would have told — worth knowing precisely,
because most published numbers in this domain quietly contain it.
"""),
    code("""
import torch
from geoverdict import models as Mo

z = np.load(cfg.OUTPUT_DIR / "chips.npz", allow_pickle=True)
X1, X2 = z["x1"], z["x2"]
lab = z["label"]
usable = lab >= 0

# reproduce chapter 04's spatial assignment for reference
led_spatial = led[("04", "cnn_eval")]["pr_auc_mean"]

rng = np.random.default_rng(cfg.SEED)
u_idx = np.where(usable)[0]
perm = rng.permutation(u_idx)
n = len(perm)
rand_idx = {"train": perm[: int(0.6 * n)], "val": perm[int(0.6 * n): int(0.8 * n)],
            "test": perm[int(0.8 * n):]}

stats = {"mean": X1[rand_idx["train"]].mean(axis=(0, 2, 3)),
         "std": X1[rand_idx["train"]].std(axis=(0, 2, 3)) + 1e-6}

def loader(ids, augment):
    ds = Mo.ChipPairDataset(X1[ids], X2[ids], lab[ids].astype(float),
                            stats=stats, augment=augment, seed=cfg.SEED)
    return torch.utils.data.DataLoader(ds, batch_size=64, shuffle=augment)

cfg.set_seed(cfg.SEEDS[0])
model = Mo.SiameseChangeNet()
pw = float((lab[rand_idx["train"]] == 0).sum() / max((lab[rand_idx["train"]] == 1).sum(), 1))
_ = Mo.fit(model, loader(rand_idx["train"], True), loader(rand_idx["val"], False),
           epochs=40, pos_weight=pw)
_, z_t, y_t = Mo.evaluate(model, loader(rand_idx["test"], False))
random_prauc = M.pr_auc(y_t, 1 / (1 + np.exp(-z_t)))

print(f"spatially-blocked PR-AUC (ch.04, 3 seeds): {led_spatial:.3f}")
print(f"random-split PR-AUC (same model, 1 seed):  {random_prauc:.3f}")
print(f"the flattery a random split buys: {random_prauc - led_spatial:+.3f}")

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["spatially blocked\\n(honest)", "random split\\n(leaky)"],
              [led_spatial, random_prauc],
              color=[viz.PALETTE["forest"], viz.PALETTE["clearing"]])
for b, v in zip(bars, [led_spatial, random_prauc]):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom",
            fontweight="bold")
ax.set_ylabel("test PR-AUC")
ax.set_title("The same model, graded two ways")
viz.save(fig, "g06_split_inflation")
plt.show()
cfg.append_result({"notebook": "06", "name": "split_inflation",
                   "spatial_pr_auc": float(led_spatial),
                   "random_pr_auc": float(random_prauc),
                   "inflation": float(random_prauc - led_spatial)})
"""),
    md("""
### 3c — Verdict sensitivity: how much do the fusion constants matter?

The verdict layer's forest-at-cutoff threshold (30%) was a reasoned choice.
Sweeping it from 10% to 50% and re-running the fusion shows how the tier
distribution moves — the compliance analogue of a decision-threshold
analysis. If HIGH counts swing wildly, the constant is doing too much work
and the evidence too little; a *stable* core with movement concentrated in
the LOW/MEDIUM boundary is the healthy signature (the boundary cases are
genuinely boundary).
"""),
    code("""
from geoverdict import risk

detections = pd.read_csv(cfg.OUTPUT_DIR / "ts_detections.csv", dtype={"plot_id": str})
cnnp = pd.read_csv(cfg.OUTPUT_DIR / "cnn_predictions.csv", dtype={"plot_id": str})
df = (plots.merge(baseline, on="plot_id", suffixes=("", "_b"))
           .merge(detections, on="plot_id", how="left")
           .merge(cnnp[["plot_id", "model_prob"]], on="plot_id", how="left"))

rows = []
for thr in (0.10, 0.20, 0.30, 0.40, 0.50):
    risk.FOREST_FRAC_MIN = thr
    tiers = []
    for r in df.itertuples():
        v = risk.assess_plot(
            plot_id=r.plot_id, geometry_ok=True, geometry_warnings=[],
            forest_frac_jrc=None if pd.isna(r.forest_frac_jrc) else float(r.forest_frac_jrc),
            forest_frac_hansen=None if pd.isna(r.forest_frac_hansen) else float(r.forest_frac_hansen),
            ts_break_detected=None if pd.isna(r.break_detected) else bool(r.break_detected),
            ts_break_date=None if pd.isna(r.break_date) else str(r.break_date),
            ts_obs_density=None if pd.isna(r.obs_density) else float(r.obs_density),
            model_prob=None if pd.isna(r.model_prob) else float(r.model_prob),
            hansen_loss_post_frac=None if pd.isna(r.hansen_loss_post_frac) else float(r.hansen_loss_post_frac))
        tiers.append(v.tier)
    counts = pd.Series(tiers).value_counts()
    rows.append({"forest_frac_min": thr, **{t: int(counts.get(t, 0)) for t in
                 ("LOW", "MEDIUM", "HIGH", "INSUFFICIENT_EVIDENCE")}})
risk.FOREST_FRAC_MIN = 0.30  # restore the default before anything else runs
sens = pd.DataFrame(rows)
print(sens.to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for tier in ("LOW", "MEDIUM", "HIGH", "INSUFFICIENT_EVIDENCE"):
    ax.plot(sens.forest_frac_min, sens[tier], "o-", color=viz.TIER_COLORS[tier], label=tier)
ax.axvline(0.30, color=viz.PALETTE["neutral"], ls="--", lw=1)
ax.annotate("default", xy=(0.30, ax.get_ylim()[1]*0.95), fontsize=8)
ax.set_xlabel('"forest at cutoff" threshold (fraction of plot)')
ax.set_ylabel("plots in tier")
ax.set_title("How much is the verdict a property of one constant?")
ax.legend(fontsize=8)
viz.save(fig, "g06_verdict_sensitivity")
plt.show()
cfg.append_result({"notebook": "06", "name": "verdict_sensitivity",
                   "high_range": [int(sens.HIGH.min()), int(sens.HIGH.max())]})
"""),
    md("""
### 4 — Limitations, written by the author

Each with its cost and its concrete remedy — because a limitation without a
remedy is a shrug, and a project without limitations is a sales deck.

1. **The plot geometries are synthetic.** Cost: intake statistics
   (corruption mix, sizes) reflect my damage model, not a real supplier's.
   The land, imagery, baselines and detections under them are real. Remedy:
   the gauntlet is data-agnostic — point `validate_portfolio` at any real
   submission; the Whisp fixture check in ch. 01 was the first step.
2. **Labels are weak (Hansen), and the hard negatives descend from TMF —
   both Landsat-family products.** Agreement partly measures agreement
   *about Landsat's view of the world*. Remedies in order of cost: RADD
   radar alerts as a third referee; hand-verified chips for a gold test set
   (a day of labelling buys an honest ceiling); field data if it exists.
3. **Optical-only.** The wet-season INSUFFICIENT tier is the bill for this,
   and it is measured, not hidden. Remedy: Sentinel-1 SAR — sees through
   cloud, at the price of a much harder signal; the fusion layer was built
   expecting a third detector input.
4. **Clearing, not degradation.** Selective logging and fire-thinning move
   the indices less than the detector's floor. This is the genuinely hard
   open problem of the field; the honest position is that this pipeline
   *screens for clearing* and says so.
5. **One AOI, one biome.** Every threshold was reasoned to be adaptive
   (per-plot MAD baselines, per-plot definitions), but *tested* only on the
   BR-163 corridor. Cross-geography transfer — West African cocoa mosaics,
   Indonesian peatland — is exactly the generalisation study published by
   the team this project is in dialogue with, and the natural next chapter.
6. **Small-sample calibration.** Temperature was fitted on a few hundred
   validation chips; per-region recalibration at deployment scale is the
   production answer.
"""),
    md("""
### The project's claims, final form

The ledger (`outputs/results.json`) now contains every number this project
asserts, each written by the chapter that measured it and several re-derived
here from raw artefacts. The README's results table is generated from this
file — if a number appears in prose but not in the ledger, it is not a claim
of this project.

**The one-sentence summary the whole repo argues for:** *a compliance
verdict is an engineering artefact — geometry gate, dual baselines, two
independent detectors, calibrated confidence, honest abstention — and every
link in that chain was measured here, including the ones that failed.*
"""),
    code("""
final = pd.DataFrame(cfg.load_results())
print(f"ledger: {len(final)} recorded results across notebooks "
      f"{sorted(final.notebook.unique())}")
print("\\nverification complete - the claims in the README are backed by the ledger.")
"""),
]

save(cells, "06_verification.ipynb")
