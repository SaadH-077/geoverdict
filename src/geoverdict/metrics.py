"""Evaluation: PR-centric metrics, calibration, and the business translation.

THE THREE RULES OF THIS MODULE:

1. NEVER LEAD WITH ACCURACY. Post-2020 clearing affects a small minority of
   plots; a model that says "no change" to everything scores ~95% accuracy
   and detects nothing. Everything here is precision/recall/F1 and PR-AUC,
   which stay honest under imbalance.

2. NORMALISE PER PLOT, NOT PER PIXEL. Plots range from ~1 ha to hundreds of
   ha. Pixel-pooled metrics let one 300 ha ranch outvote fifty smallholder
   farms — and the smallholders are exactly who EUDR compliance is hardest
   for. `plot_normalised_prf` therefore computes each plot's confusion counts
   normalised by that plot's valid-pixel count before summing, so every plot
   contributes equally regardless of size. (This follows the sample-normalised
   metric design in Mammadov et al., AGILE 2026, adopted here because the
   argument for it is independently compelling.)

3. TRANSLATE TO BUSINESS UNITS. A compliance team experiences the model as
   "how many plots land on an analyst's desk, and how many real clearings
   slip through" — so `screening_table` reports flags-per-1000-plots at fixed
   recall, which is the number a deployment decision actually turns on.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import auc, precision_recall_curve


# --------------------------------------------------------------------------
# Plot-level binary metrics
# --------------------------------------------------------------------------

def prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "tn": int((~y_true & ~y_pred).sum())}


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve — the threshold-free headline.

    PR-AUC rather than ROC-AUC: with few positives, ROC-AUC is inflated by
    the ocean of easy true negatives and two very different models can share
    a ROC-AUC of 0.97. The PR curve only rewards what happens on and around
    the positives.
    """
    p, r, _ = precision_recall_curve(np.asarray(y_true).astype(int), np.asarray(scores))
    return float(auc(r, p))


def plot_normalised_prf(
    per_plot_counts: list[dict[str, float]],
) -> dict[str, float]:
    """Precision/recall/F1 where every plot contributes equally.

    Input: per plot, raw pixel confusion counts {tp, fp, fn, valid}. Each
    plot's counts are divided by its own valid-pixel count, then summed —
    a 400 ha ranch and a 2 ha farm each contribute exactly one unit of
    evidence. Compare against naive pixel pooling to SEE the size bias
    (notebook 04 plots both).
    """
    tp = fp = fn = 0.0
    for c in per_plot_counts:
        v = max(float(c.get("valid", 0)), 1.0)
        tp += c.get("tp", 0) / v
        fp += c.get("fp", 0) / v
        fn += c.get("fn", 0) / v
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
# WHY CALIBRATION IS A COMPLIANCE FEATURE, NOT A NICETY. The verdict layer
# consumes probabilities ("0.9 => high risk"). If the model says 0.9 and is
# right 60% of the time, every downstream risk threshold silently means
# something different from what the policy says. ECE quantifies that gap;
# temperature scaling is the one-parameter fix that cannot reorder decisions
# (it is monotone), which makes it safe to apply post-hoc.

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 12) -> dict:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, centers, accs, confs, weights = 0.0, [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if m.sum() == 0:
            continue
        conf, acc, w = probs[m].mean(), labels[m].mean(), m.mean()
        ece += w * abs(acc - conf)
        centers.append((lo + hi) / 2); accs.append(acc); confs.append(conf); weights.append(w)
    return {"ece": float(ece), "bin_centers": centers, "bin_acc": accs,
            "bin_conf": confs, "bin_weight": weights}


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """One scalar T minimising NLL of sigmoid(logits / T) on a held-out split.

    Fit on VALIDATION, evaluate on TEST — fitting the calibrator on the data
    you then report calibration for is the same crime as testing on train.
    Golden-section search over log T: the NLL is unimodal in T, so a bracketed
    search is exact enough and dependency-free.
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)

    def nll(t: float) -> float:
        z = logits / t
        # numerically stable log(1+exp)
        return float(np.mean(np.logaddexp(0.0, z) - labels * z))

    lo, hi = 0.05, 20.0
    phi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    for _ in range(80):
        if nll(c) < nll(d):
            b = d
        else:
            a = c
        c, d = b - phi * (b - a), a + phi * (b - a)
    return float((a + b) / 2)


# --------------------------------------------------------------------------
# The business translation
# --------------------------------------------------------------------------

def threshold_at_recall(y_true: np.ndarray, scores: np.ndarray, recall_target: float = 0.90) -> float:
    """Lowest score threshold that still achieves the recall target.

    The operating point is chosen by POLICY (miss at most 10% of clearings),
    and the threshold is derived from it — never the reverse. A compliance
    product cannot justify "0.5 because sigmoid".
    """
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    sorted_true = y_true[order]
    cum_tp = np.cumsum(sorted_true)
    total_pos = max(int(y_true.sum()), 1)
    k = int(np.searchsorted(cum_tp, np.ceil(recall_target * total_pos)))
    k = min(k, len(scores) - 1)
    return float(scores[order][k])


def screening_table(y_true: np.ndarray, scores: np.ndarray,
                    recall_targets=(0.80, 0.90, 0.95)) -> list[dict]:
    """For each recall policy: threshold, precision, and flags per 1,000 plots.

    'Flags per 1,000' is the analyst-workload number: every flagged plot is a
    manual review. Two models with equal F1 can differ 3x here, and the
    cheaper one wins the deployment argument.
    """
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    rows = []
    for rt in recall_targets:
        thr = threshold_at_recall(y_true, scores, rt)
        flag = scores >= thr
        m = prf(y_true, flag)
        rows.append({
            "recall_target": rt,
            "threshold": thr,
            "achieved_recall": m["recall"],
            "precision": m["precision"],
            "flags_per_1000": 1000.0 * flag.mean(),
            "missed_clearings_per_1000": 1000.0 * ((y_true & ~flag).sum() / len(y_true)),
        })
    return rows
