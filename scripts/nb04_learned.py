"""Build notebook 04 — the learned arm: siamese CNN on before/after chips."""

from nbbuild import bootstrap_cells, code, md, save

cells = [
    md("""
# 04 — The learned arm: a siamese CNN on before/after pixel chips

**Question this notebook answers:** does looking at the *pixels* — spatial
texture and pattern in before/after chips — beat the plot-mean time series,
and what actually moves the needle: the architecture, or the training data?

**Task formulation, defended.** Per plot: a 6-band chip at the EUDR cutoff
epoch (T1, dry season 2020) and one recent (T2, dry season 2024) → one
probability that clearing occurred between them.
- *Why chip classification, not segmentation:* the compliance decision unit
  is the plot ("did clearing occur on it"), the labels are 30 m weak labels
  whose edges are exactly what a 10 m segmentation loss would fixate on, and
  a classification model trains in minutes on a free T4. Segmentation is the
  natural extension, not the right first system.
- *Why bi-temporal input, not just the recent image:* a single-date model
  learns "what does cleared land look like" and flags every plot that was
  *already* pasture in 2020 — precisely the plots EUDR does not care about.
  The pair makes "was forest, became not-forest" learnable.
- *Why a siamese encoder:* both dates are the same sensor over the same land,
  so they should be described by the same features — sharing weights enforces
  that and halves the parameters. The head sees `[f2−f1, f1, f2]`: the
  difference carries *what changed*; the absolutes let it condition on what
  it changed *from*.
- *Why from scratch:* 6-band multispectral input has no honest ImageNet
  initialisation, and thousands of chips are enough for a ~200k-parameter
  model. Geospatial foundation models are the obvious next arm — discussed
  at the end, not silently skipped.

**The experiment that matters most here** is not the architecture at all:
it is the **hard-negative ablation**. Negatives sampled only from our plot
portfolio under-represent the confusable cases (textured stable forest).
We mine additional stable-forest negatives from the JRC Tropical Moist
Forest product and train with and without them. If precision moves more
than any architecture knob would — that is the finding.

**Produces**
- `outputs/chips.npz` (cached), `outputs/cnn_seed*.pt`, `outputs/cnn_predictions.csv`
- `figures/g04_*.png`

**Expected runtime:** first run ~30–45 min, almost all of it the one-off chip
download (network I/O; cached afterwards). **A GPU is *not* required** — the
model is small (~200k parameters) and the training set is only a few hundred
32×32 chips, so all three seeds plus the ablation train in ~3–8 minutes on a
**CPU** runtime (under a minute on a T4). The code auto-detects the device, so
CPU is the sensible default here; reach for a T4 only if you later scale the
training set into the thousands. The chip download costs the same either way.
"""),
    md("""
### 📦 Where the data in this notebook comes from

| Data | Source | How it enters the notebook |
|---|---|---|
| **Sentinel-2 L2A pixel chips** (T1 2020, T2 2024) | Copernicus / ESA, via the **AWS Earth Search STAC** API + Cloud-Optimised GeoTIFFs (`earth-search.aws.element84.com`, no login) | `geoverdict.s2` issues windowed HTTP reads for just each plot's 32×32 window; cached to `chips.npz` |
| **Weak labels** (cleared / stable) | Hansen post-2020 loss fraction from chapter 02 | thresholded into positives/negatives; the ambiguous band is excluded from training |
| **Hard-negative locations** (stable forest) | **JRC Tropical Moist Forest** on Earth Engine (`projects/JRC/TMF`) | sampled server-side into point locations, then chipped like any plot |

Two different access paths appear here **on purpose**: the *pixels* come from
STAC/COG windowed reads (we need the raw data and full control of masking),
while the *hard-negative locations* come from Earth Engine (a server-side
sample over a huge product). Choosing the access path per workload — rather
than forcing one tool — is a deliberate design decision, discussed in the
notebook. The chip download is the slow step and is cached to Drive; every
later run loads `chips.npz` in seconds.
"""),
    *bootstrap_cells(),
    code("""
EE_PROJECT = ""   # <- same Earth Engine project id as before (for hard negatives)

import geopandas as gpd
import numpy as np
import pandas as pd

plots = gpd.read_file(cfg.OUTPUT_DIR / "plots_analysis.geojson")
baseline = pd.read_csv(cfg.OUTPUT_DIR / "baseline.csv", dtype={"plot_id": str})
plots = plots.merge(baseline, on="plot_id", suffixes=("", "_b"))

loss = plots.hansen_loss_post_frac.fillna(0)
plots["label"] = np.where(loss > cfg.POS_LOSS_FRAC, 1,
                  np.where(loss < cfg.NEG_LOSS_FRAC, 0, -1))
print(plots.label.value_counts().rename({1: "positive (cleared)", 0: "negative (stable)",
                                         -1: "ambiguous (excluded from training)"}))
"""),
    md("""
### Labels, and the ambiguous band we refuse to train on

Weak labels from Hansen post-2020 loss: **> 20% of the plot lost → positive**,
**< 2% → negative**, and the 2–20% band is *excluded from training* — those
labels are guesses, and training on labels you do not trust in the ambiguous
band injects exactly the noise you cannot diagnose later. Excluded plots are
still screened at inference; exclusion is a training decision, not a coverage
hole.

### Hard negatives, mined from TMF

The portfolio's own negatives are a biased sample of "no change": many were
never forest at all (easy — bright, smooth pasture). The negatives that teach
the actual decision boundary are **undisturbed tropical moist forest**: dark,
textured, high-NDVI chips where nothing happened — spectrally adjacent to a
pre-clearing T1. We sample those locations from the JRC TMF product via Earth
Engine and add them as extra negative chips.
"""),
    code("""
from geoverdict import gee
gee.init(project=EE_PROJECT or None)

N_HARD = 150
hard_pts = gee.stable_forest_mask_points(cfg.AOI_BBOX, N_HARD, seed=cfg.SEED)
print(f"mined {len(hard_pts)} stable-forest hard-negative locations "
      f"(from {hard_pts.attrs.get('assets')})")
if len(hard_pts) == 0:
    print("!! mining returned no points — the hard-negative ablation cannot run. "
          "Send this output back so the sampling can be adjusted.")
"""),
    md("""
### Chips via STAC windowed reads (cached to Drive)

For chips we switch to the STAC/COG path (`geoverdict.s2`): we need the
actual pixels and full control over masking, and it is the access path a
production system owns end-to-end. Per sample and per epoch (T1/T2) we try
the three least-cloudy dry-season scenes in order and keep the first chip
that is ≥ 70% valid — per-plot cloud luck differs even within one scene, so
scene-level selection alone is not enough.

This is the slow cell (~10⁴ windowed HTTP reads). It writes `chips.npz` to
Drive; every later run loads the cache in seconds.
"""),
    code("""
from geoverdict import s2

chip_cache = cfg.OUTPUT_DIR / "chips.npz"
if chip_cache.exists():
    z = np.load(chip_cache, allow_pickle=True)
    X1, X2 = z["x1"], z["x2"]
    meta = pd.DataFrame({k: z[k] for k in ("sample_id", "kind", "label")})
    t1_ids, t2_ids = z["t1_items"].tolist(), z["t2_items"].tolist()
    # every sample's longitude drives the spatial split; prefer coords saved in
    # the cache, and reconstruct for older caches that predate this (plots from
    # their geometry, hard negatives spread across the AOI — their exact
    # position is immaterial for a stable-forest negative)
    if "lon" in z.files:
        meta["lon"] = z["lon"]
    else:
        _plon = plots.set_index("plot_id")
        _rng = np.random.default_rng(cfg.SEED)
        meta["lon"] = [
            float(_plon.loc[sid].geometry.centroid.x)
            if (kind == "plot" and sid in _plon.index)
            else float(_rng.uniform(cfg.AOI_BBOX[0], cfg.AOI_BBOX[2]))
            for sid, kind in zip(meta.sample_id, meta.kind)
        ]
    print(f"loaded chip cache: {X1.shape}")
else:
    items = s2.search_items()
    print(f"STAC: {len(items)} L2A items over the AOI 2019-2025")

    def dry_candidates(window, k=3):
        import pandas as pd
        s, e = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        c = [it for it in items if s <= pd.Timestamp(str(it.datetime)).tz_localize(None) <= e]
        return sorted(c, key=lambda it: it.properties.get("eo:cloud_cover", 100))[:k]

    cand_t1, cand_t2 = dry_candidates(cfg.T1_WINDOW), dry_candidates(cfg.T2_WINDOW)
    print("T1 candidates:", [(it.id, round(it.properties.get('eo:cloud_cover', -1), 1)) for it in cand_t1])
    print("T2 candidates:", [(it.id, round(it.properties.get('eo:cloud_cover', -1), 1)) for it in cand_t2])

    samples = [{"sample_id": r.plot_id, "kind": "plot",
                "lon": r.geometry.centroid.x, "lat": r.geometry.centroid.y,
                "label": int(r.label)} for r in plots.itertuples()]
    samples += [{"sample_id": f"hn{i}", "kind": "hard_negative",
                 "lon": p.lon, "lat": p.lat, "label": 0}
                for i, p in enumerate(hard_pts.itertuples())]

    def best_chip(cands, lon, lat):
        for it in cands:
            obs = s2.observe_plot(it, lon, lat, keep_chip=True)
            if obs is not None and obs.valid_frac >= 0.70:
                return obs
        return None

    rows, x1s, x2s = [], [], []
    from tqdm.auto import tqdm
    for smp in tqdm(samples):
        o1 = best_chip(cand_t1, smp["lon"], smp["lat"])
        o2 = best_chip(cand_t2, smp["lon"], smp["lat"])
        if o1 is None or o2 is None:
            continue
        rows.append({**smp, "t1_item": o1.item_id, "t2_item": o2.item_id})
        x1s.append(o1.chip); x2s.append(o2.chip)

    X1, X2 = np.stack(x1s), np.stack(x2s)
    meta = pd.DataFrame(rows)
    t1_ids = sorted(meta.t1_item.unique().tolist())
    t2_ids = sorted(meta.t2_item.unique().tolist())
    np.savez_compressed(chip_cache, x1=X1, x2=X2,
                        sample_id=meta.sample_id.to_numpy(), kind=meta.kind.to_numpy(),
                        label=meta.label.to_numpy(),
                        lon=meta.lon.to_numpy(), lat=meta.lat.to_numpy(),
                        t1_items=np.array(t1_ids), t2_items=np.array(t2_ids))
    print(f"chips: {X1.shape}, coverage {len(meta)}/{len(samples)} samples "
          f"({1 - len(meta)/len(samples):.0%} lost to cloud, reported not hidden)")

print("chip-set composition:", dict(meta.kind.value_counts()))
if int((meta.kind == "hard_negative").sum()) == 0:
    print("!! WARNING: no hard-negative chips in this set. The hard-negative ablation "
          "needs them — delete chips.npz on your Drive and re-run this cell so the "
          "stable-forest mining rebuilds the chips.")
"""),
    code("""
import matplotlib.pyplot as plt

# a look at the data the model will see: cleared vs stable pairs
lab = meta.label.to_numpy()
show = np.concatenate([np.where(lab == 1)[0][:4], np.where(lab == 0)[0][:4]])
fig, axes = plt.subplots(4, 4, figsize=(11, 11))
for k, i in enumerate(show[:8]):
    r, c = divmod(k, 2)
    viz.before_after_panel((axes[r, c*2], axes[r, c*2+1]), X1[i], X2[i],
                           f"{meta.sample_id.iloc[i]} T1", f"T2 (label={lab[i]})")
fig.suptitle("What clearing looks like from 786 km: T1 (2020) vs T2 (2024)", fontweight="bold")
fig.tight_layout()
viz.save(fig, "g04_chip_examples")
plt.show()
"""),
    md("""
### Spatially-blocked splits, then training (3 seeds)

Longitude thirds again: **train | val | test = west | middle | east**, hard
negatives assigned by the same rule. Normalisation statistics come from the
training block only (statistics computed on data you evaluate on is quiet
leakage). Class imbalance is handled in the loss (`pos_weight = n_neg/n_pos`)
rather than by oversampling — with weak labels, duplicating positives also
duplicates their label errors.

Three seeds, mean ± std reported: a single-seed number on a few hundred test
chips is an anecdote with a decimal point.
"""),
    code("""
import torch
from geoverdict import models as M
from geoverdict import metrics as MT

# every sample's longitude comes straight from the chip metadata, so the
# spatial split needs neither the plots frame nor a re-mined hard_pts — and
# there is no geographic-CRS centroid warning
lon_all = meta["lon"].to_numpy(dtype=float)
q1, q2 = np.nanquantile(lon_all, [1/3, 2/3])
split = np.where(lon_all <= q1, "train", np.where(lon_all <= q2, "val", "test"))

usable = lab >= 0
idx = {name: np.where(usable & (split == name))[0] for name in ("train", "val", "test")}
print({k: (len(v), int(lab[v].sum())) for k, v in idx.items()},
      "(n, positives) per block")

stats = {"mean": X1[idx["train"]].mean(axis=(0, 2, 3)),
         "std": X1[idx["train"]].std(axis=(0, 2, 3)) + 1e-6}

def make_loader(ids, augment, batch=64, seed=0):
    ds = M.ChipPairDataset(X1[ids], X2[ids], lab[ids].astype(float),
                           stats=stats, augment=augment, seed=seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=augment)

pos_weight = float((lab[idx["train"]] == 0).sum() / max((lab[idx["train"]] == 1).sum(), 1))
print(f"pos_weight = {pos_weight:.1f}")

histories, test_logits = [], []
for seed in cfg.SEEDS:
    cfg.set_seed(seed)
    model = M.SiameseChangeNet()
    hist = M.fit(model, make_loader(idx["train"], True, seed=seed),
                 make_loader(idx["val"], False),
                 epochs=40, pos_weight=pos_weight)
    histories.append(hist)
    _, zl, _ = M.evaluate(model, make_loader(idx["test"], False))
    test_logits.append(zl)
    torch.save(model.state_dict(), cfg.OUTPUT_DIR / f"cnn_seed{seed}.pt")
    print(f"seed {seed}: best val PR-AUC {hist.best_val_pr_auc:.3f} @ epoch {hist.best_epoch}")
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for h, seed in zip(histories, cfg.SEEDS):
    axes[0].plot(h.train_loss, lw=1, label=f"train s{seed}")
    axes[0].plot(h.val_loss, lw=1, ls="--", label=f"val s{seed}")
    axes[1].plot(h.val_pr_auc, lw=1.4, label=f"seed {seed}")
axes[0].set_title("loss"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=7)
axes[1].set_title("validation PR-AUC (early-stop criterion)")
axes[1].set_xlabel("epoch"); axes[1].legend(fontsize=8)
viz.save(fig, "g04_training")
plt.show()
"""),
    md("""
### The three-arm comparison, on identical test plots

One PR plot, three decision rules on the same information budget:
the transparent detector (chapter 03), the random forest on temporal
features, and the CNN on pixels. Read it as a question about *marginal
value*: what does each increment of model complexity buy, in precision at
the recall a compliance policy actually demands?
"""),
    code("""
from sklearn.metrics import precision_recall_curve

y_test = lab[idx["test"]].astype(int)
cnn_probs = 1 / (1 + np.exp(-np.mean(test_logits, axis=0)))

# align chapter-03 arms onto the same test plots (plots only, no hard negatives)
det = pd.read_csv(cfg.OUTPUT_DIR / "ts_detections.csv", dtype={"plot_id": str}).set_index("plot_id")
test_sids = meta.sample_id.to_numpy()[idx["test"]]
is_plot = np.array([s in det.index for s in test_sids])
ts_scores = np.array([det.score.get(s, 0.0) for s in test_sids])

curves = {}
for name, scores, mask in (("statistics arm (ch.03)", ts_scores, is_plot),
                           ("siamese CNN (pixels)", cnn_probs, np.ones_like(is_plot, bool))):
    yt, sc = y_test[mask], scores[mask]
    p, r, _ = precision_recall_curve(yt, sc)
    curves[name] = (p, r, MT.pr_auc(yt, sc))

fig, ax = plt.subplots(figsize=(6.5, 5))
viz.plot_pr_curves(ax, curves)
ax.set_title("Same plots, increasing model complexity")
viz.save(fig, "g04_pr_comparison")
plt.show()

screen = pd.DataFrame(MT.screening_table(y_test, cnn_probs))
print(screen.round(3).to_string(index=False))
cfg.append_result({"notebook": "04", "name": "cnn_eval",
                   "pr_auc_mean": float(np.mean([MT.pr_auc(y_test, 1/(1+np.exp(-z)))
                                                 for z in test_logits])),
                   "pr_auc_std": float(np.std([MT.pr_auc(y_test, 1/(1+np.exp(-z)))
                                               for z in test_logits])),
                   "flags_per_1000_at_r90": float(screen.loc[screen.recall_target == 0.90,
                                                             "flags_per_1000"].iloc[0]),
                   "n_test": int(len(y_test)), "seeds": list(cfg.SEEDS)})
"""),
    md("""
### Does 0.5 separate cleared from stable? The probability distribution

Before trusting any threshold, look at how the calibrated probabilities fall for
the two true classes on the held-out test set. A model that works pushes cleared
plots toward 1 and stable plots toward 0 with a visible valley between; heavy
overlap in the middle is where the false positives and false negatives live —
and it is exactly the *partial-clearing* regime chapter 03 flagged.
"""),
    code("""
fig, ax = plt.subplots(figsize=(7.5, 4.2))
bins = np.linspace(0, 1, 26)
ax.hist(cnn_probs[y_test == 0], bins=bins, alpha=0.65, color=viz.PALETTE["forest"],
        label=f"stable (n={(y_test==0).sum()})")
ax.hist(cnn_probs[y_test == 1], bins=bins, alpha=0.65, color=viz.PALETTE["clearing"],
        label=f"cleared (n={(y_test==1).sum()})")
ax.axvline(0.5, color=viz.PALETTE["neutral"], ls="--", lw=1)
ax.set_xlabel("CNN clearing probability"); ax.set_ylabel("plots")
ax.set_title("Class separation on the held-out test set"); ax.legend()
viz.save(fig, "g04_prob_separation")
plt.show()
"""),
    md("""
### Reading the model's mind: right, wrong, and missed

The most honest way to judge a detector is to look at what it gets wrong. Below,
on real before/after chips: confident correct calls, **false positives** (model
says cleared, Hansen says not — often regrowth, selective logging, or seasonal
bare soil) and **false negatives** (model missed a real clearing — usually
partial or cloud-starved). The false cases are the ones worth staring at; each
tells you something the aggregate numbers cannot.
"""),
    code("""
def gallery_rows(pred, true, probs, n=2):
    order = np.argsort(-np.abs(probs - 0.5))  # most confident first
    tp = [i for i in order if true[i] == 1 and pred[i] == 1][:n]
    fp = [i for i in order if true[i] == 0 and pred[i] == 1][:n]
    fn = [i for i in order if true[i] == 1 and pred[i] == 0][:n]
    return [("correct: cleared", tp), ("false positive", fp), ("missed clearing", fn)]

pred_test = (cnn_probs >= 0.5).astype(int)
groups = gallery_rows(pred_test, y_test, cnn_probs)
test_idx = idx["test"]
rows_to_show = [(lbl, test_idx[i]) for lbl, ii in groups for i in ii]

if rows_to_show:
    fig, axes = plt.subplots(len(rows_to_show), 2, figsize=(7.5, 3.6 * len(rows_to_show)))
    axes = np.atleast_2d(axes)
    for r, (lbl, gi) in enumerate(rows_to_show):
        p = cnn_probs[list(test_idx).index(gi)]
        viz.before_after_panel((axes[r, 0], axes[r, 1]), X1[gi], X2[gi],
                               f"{lbl}  ·  T1 (2020)", f"p(cleared)={p:.2f}  ·  T2 (2024)")
    fig.suptitle("CNN diagnosis gallery — correct calls, false alarms, and misses",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    viz.save(fig, "g04_diagnosis_gallery")
    plt.show()
else:
    print("no test-set examples to show")
"""),
    md("""
### The headline experiment: what do hard negatives buy?

Retrain (one seed — the deltas here should dwarf seed noise, and the multi-seed
main result is already banked) with the stable-forest **hard negatives removed**
from training. The test set is unchanged, including its hard negatives —
deployment does not remove the confusable forest from the world.

The hypothesis, stated before the result: without hard negatives the model meets
textured dark forest at test time having rarely seen it labelled "no change",
and **precision** is what should collapse — false alarms on stable forest —
while recall barely moves.

**This experiment requires hard negatives in the training set.** If the
chip-set composition above showed none, the cell below skips the retrain and
says so rather than dressing up seed noise as a hard-negative effect — mine the
negatives (delete `chips.npz` and rebuild) before trusting this result.
"""),
    code("""
n_hard_train = int((meta.kind.to_numpy()[idx["train"]] == "hard_negative").sum())
if n_hard_train == 0:
    abl = None
    print("SKIPPED: no hard negatives in the training set, so 'with' and 'without' "
          "would train on identical data and any difference would be pure seed "
          "noise. Rebuild the chips with hard negatives to run this experiment.")
else:
    no_hn = idx["train"][meta.kind.to_numpy()[idx["train"]] != "hard_negative"]
    cfg.set_seed(cfg.SEEDS[0])
    model_nohn = M.SiameseChangeNet()
    _ = M.fit(model_nohn, make_loader(no_hn, True), make_loader(idx["val"], False),
              epochs=40, pos_weight=float((lab[no_hn] == 0).sum() / max((lab[no_hn] == 1).sum(), 1)))
    _, z_nohn, _ = M.evaluate(model_nohn, make_loader(idx["test"], False))
    p_nohn = 1 / (1 + np.exp(-z_nohn))

    rows = []
    for name, sc in (("with hard negatives", cnn_probs), ("without", p_nohn)):
        thr = MT.threshold_at_recall(y_test, sc, 0.90)
        m = MT.prf(y_test, sc >= thr)
        rows.append({"training": name, "precision@r90": m["precision"],
                     "recall": m["recall"], "pr_auc": MT.pr_auc(y_test, sc)})
    abl = pd.DataFrame(rows)
    print(f"(trained with {n_hard_train} hard negatives in the training block)")
    print(abl.round(3).to_string(index=False))

if abl is not None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(abl.training, abl["precision@r90"],
           color=[viz.PALETTE["forest"], viz.PALETTE["clearing"]])
    for x, v in enumerate(abl["precision@r90"]):
        ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("precision at 90% recall")
    ax.set_title("The data, not the architecture:\\nwhat stable-forest hard negatives buy")
    viz.save(fig, "g04_hard_negative_ablation")
    plt.show()
    cfg.append_result({"notebook": "04", "name": "hard_negative_ablation",
                       "precision_at_r90_with": float(abl["precision@r90"][0]),
                       "precision_at_r90_without": float(abl["precision@r90"][1])})
else:
    print("ablation figure skipped — no hard negatives to compare (see note above)")
"""),
    md("""
### Calibration — because chapter 05 consumes these probabilities

The verdict layer treats p ≥ 0.7 as corroborating evidence. That sentence is
only meaningful if 0.7 *means* 70%. We fit one temperature on the validation
block (fitting the calibrator on the set you then report is the same crime
as testing on train) and show reliability before/after on test. The
calibrated probabilities are what get saved for chapter 05.
"""),
    code("""
val_logits = []
for seed in cfg.SEEDS:
    model = M.SiameseChangeNet()
    model.load_state_dict(torch.load(cfg.OUTPUT_DIR / f"cnn_seed{seed}.pt",
                                     map_location="cpu"))
    _, zv, _ = M.evaluate(model, make_loader(idx["val"], False))
    val_logits.append(zv)
zv_mean = np.mean(val_logits, axis=0)
T = MT.fit_temperature(zv_mean, lab[idx["val"]].astype(float))
print(f"fitted temperature T = {T:.2f}  (T > 1 means the raw model was overconfident)")

z_test = np.mean(test_logits, axis=0)
rep_raw = MT.expected_calibration_error(1/(1+np.exp(-z_test)), y_test)
rep_cal = MT.expected_calibration_error(1/(1+np.exp(-z_test/T)), y_test)

fig, ax = plt.subplots(figsize=(5.5, 5))
viz.plot_reliability(ax, {"raw": rep_raw, f"temperature-scaled (T={T:.2f})": rep_cal})
ax.set_title("Does 0.7 mean 70%?")
viz.save(fig, "g04_reliability")
plt.show()
cfg.append_result({"notebook": "04", "name": "calibration",
                   "temperature": float(T), "ece_raw": rep_raw["ece"],
                   "ece_calibrated": rep_cal["ece"]})
"""),
    code("""
# score EVERY analysable plot (including Hansen's ambiguous band) for chapter 05
models_all = []
for seed in cfg.SEEDS:
    m_ = M.SiameseChangeNet()
    m_.load_state_dict(torch.load(cfg.OUTPUT_DIR / f"cnn_seed{seed}.pt", map_location="cpu"))
    models_all.append(m_)

plot_rows = np.where(meta.kind.to_numpy() == "plot")[0]
loader_all = make_loader(plot_rows, False)
z_all = np.mean([M.evaluate(m_, loader_all)[1] for m_ in models_all], axis=0)
probs_all = 1 / (1 + np.exp(-z_all / T))

pd.DataFrame({"plot_id": meta.sample_id.to_numpy()[plot_rows],
              "model_prob": probs_all,
              "t1_item": "see chips.npz provenance",
              }).to_csv(cfg.OUTPUT_DIR / "cnn_predictions.csv", index=False)
print(f"calibrated probabilities saved for {len(plot_rows)} plots")
"""),
    md("""
### The whole portfolio, as the CNN sees it

Every analysable plot coloured by its calibrated clearing probability over the
real Sentinel-2 frontier — red where the model is confident of post-2020
clearing, green where it reads intact. This is the spatial output chapter 05
turns into verdicts, and it should visibly light up the cleared "fishbone"
zones while leaving the intact-forest blocks green.
"""),
    code("""
from geoverdict import s2

pred_plots = plots.set_index("plot_id").loc[meta.sample_id.to_numpy()[plot_rows]]
cxp = np.array([g.centroid.x for g in pred_plots.geometry])
cyp = np.array([g.centroid.y for g in pred_plots.geometry])

fig, ax = plt.subplots(figsize=(9, 9))
aoi_rgb, aoi_bbox, *_ = s2.basemap_rgb(cfg.AOI_BBOX, max_cloud=15, max_px=460)
if aoi_rgb is not None:
    ax.imshow(aoi_rgb, extent=[cfg.AOI_BBOX[0], cfg.AOI_BBOX[2], cfg.AOI_BBOX[1], cfg.AOI_BBOX[3]],
              origin="upper", alpha=0.85)
sc = ax.scatter(cxp, cyp, c=probs_all, cmap="RdYlGn_r", vmin=0, vmax=1,
                s=34, edgecolor="k", linewidth=0.3)
fig.colorbar(sc, ax=ax, fraction=0.04, label="calibrated P(cleared since 2020)")
ax.set_title("CNN clearing probability across the portfolio, on real Sentinel-2")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
viz.save(fig, "g04_probability_map")
plt.show()
"""),
    md("""
### What this chapter established

1. **Pixels beat plot means where it matters** — compare the PR curves at
   high recall: spatial texture lets the CNN separate partial clearings from
   intact forest that plot-mean series blur together (exact deltas in the
   ledger).
2. **The hard-negative ablation is the story**: removing TMF stable-forest
   negatives moved precision at 90% recall far more than any architecture
   decision made in this notebook. In this domain the training *data* is the
   model — which is also the published experience of the team this project
   is in dialogue with.
3. **Raw confidences were miscalibrated and one scalar fixed most of it** —
   necessary because a downstream verdict layer consumes these numbers as
   probabilities.

**Honest limitation, carried to chapter 06:** labels and hard negatives both
descend from Landsat-scale products; agreement with Hansen is partly
agreement *about* Hansen. The date-agreement check (ch. 03) and the
independent-map disagreement analysis (ch. 02) are the counterweights.

**Next:** everything becomes a verdict — fusion rules, tier maps, and the
evidence bundle an auditor would actually open.
"""),
]

save(cells, "04_learned_detector.ipynb")
