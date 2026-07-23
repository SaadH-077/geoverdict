"""The verdict layer: fuse every stage's evidence into a risk tier with reasons.

WHY RULES AND NOT A LEARNED RISK MODEL. Two reasons, one statistical and one
institutional:
  * Statistical — there is no labelled "compliance outcome" to learn from
    here (no dataset of plots with adjudicated EUDR decisions), so a learned
    fusion would be trained on proxies of proxies.
  * Institutional — a due-diligence decision must be EXPLAINABLE TO AN
    AUDITOR. "HIGH because: forest at cutoff per both baseline maps; NBR
    breakpoint 2022-07 confirmed by 4 observations; CNN probability 0.94"
    survives a compliance review. "HIGH because a gradient-boosted ensemble
    said 0.83" does not. The ML lives in the EVIDENCE (detectors, baselines);
    the FUSION stays transparent. This mirrors how production compliance
    systems are actually structured.

THE FOUR TIERS:
  LOW           — analysable, no credible post-cutoff deforestation signal.
  MEDIUM        — analysable, but evidence conflicts or confidence is reduced
                  (baseline maps disagree; detectors disagree; plot overlaps).
  HIGH          — forest at cutoff AND a corroborated post-cutoff clearing.
  INSUFFICIENT  — cannot honestly conclude (geometry unrepaired, too few
                  cloud-free observations, baseline unreliable). This tier is
                  the pipeline admitting ignorance — the single most important
                  property of a screening system, because a silent "LOW" on an
                  unobservable plot is how non-compliant cocoa gets certified.

Every rule that fires appends a human-readable reason; the reasons ARE the
explanation, and the evidence bundle prints them verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

LOW, MEDIUM, HIGH, INSUFFICIENT = "LOW", "MEDIUM", "HIGH", "INSUFFICIENT_EVIDENCE"

# Decision constants — named, with rationale, because "magic number in an if"
# is the first thing a reviewer of a compliance system attacks.
FOREST_FRAC_MIN = 0.30      # >=30% of plot pixels forested at cutoff => the plot
                            # "contains forest" in the sense that clearing it
                            # would be deforestation. Below that it was already
                            # non-forest land at the cutoff.
FOREST_FRAC_CLEAR = 0.10    # <=10% on BOTH baselines => confidently non-forest.
OBS_DENSITY_MIN = 0.25      # <25% of monitoring months observed => the optical
                            # record is too thin to call "no change" honestly.
MODEL_PROB_HIGH = 0.70      # CNN clearing probability treated as corroborating.
MODEL_PROB_LOW = 0.30       # ...and below this, as contradicting.


@dataclass
class Verdict:
    plot_id: str
    tier: str = INSUFFICIENT   # safe default: a verdict never accidentally reads as LOW
    reasons: list[str] = field(default_factory=list)
    inputs: dict = field(default_factory=dict)   # the raw evidence, for the bundle

    def to_dict(self) -> dict:
        return {"plot_id": self.plot_id, "tier": self.tier,
                "reasons": self.reasons, "inputs": self.inputs}


def assess_plot(
    plot_id: str,
    geometry_ok: bool,
    geometry_warnings: list[str],
    forest_frac_jrc: float | None,
    forest_frac_hansen: float | None,
    ts_break_detected: bool | None,
    ts_break_date: str | None,
    ts_obs_density: float | None,
    model_prob: float | None,
    hansen_loss_post_frac: float | None = None,
) -> Verdict:
    """One plot in, one tier + reasons out. Pure function: trivially testable.

    EVIDENCE HIERARCHY (and why):
      1. Geometry gate — a verdict about the wrong shape is a verdict about
         someone else's land. Unrepaired geometry => INSUFFICIENT, always.
      2. Baseline gate — EUDR risk only exists on land that WAS forest at the
         cutoff. Both baseline maps below the clear threshold => LOW without
         even consulting the detectors ("it was pasture in 2020" ends the
         analysis). Maps disagreeing is itself a MEDIUM signal — the official
         answer depends on which official map you trust.
      3. Observability gate — no detector can clear a plot it cannot see.
      4. Detector fusion — time-series break and CNN probability agree => the
         strongest claim; disagree => MEDIUM with both stated.
    """
    v = Verdict(plot_id=str(plot_id))
    v.inputs = {
        "geometry_ok": geometry_ok,
        "forest_frac_jrc": forest_frac_jrc,
        "forest_frac_hansen": forest_frac_hansen,
        "ts_break_detected": ts_break_detected,
        "ts_break_date": ts_break_date,
        "ts_obs_density": ts_obs_density,
        "model_prob": model_prob,
        "hansen_loss_post_frac": hansen_loss_post_frac,
    }

    # 1 — geometry gate
    if not geometry_ok:
        v.tier = INSUFFICIENT
        v.reasons.append("geometry could not be validated or repaired; the analysed footprint would not be trustworthy")
        return v
    for w in geometry_warnings:
        v.reasons.append(f"geometry warning carried forward: {w}")

    # 2 — baseline gate
    have_baseline = forest_frac_jrc is not None and forest_frac_hansen is not None
    if not have_baseline:
        v.tier = INSUFFICIENT
        v.reasons.append("no forest-baseline value could be computed at the 2020-12-31 cutoff")
        return v

    jrc_forest = forest_frac_jrc >= FOREST_FRAC_MIN
    han_forest = forest_frac_hansen >= FOREST_FRAC_MIN
    baselines_disagree = jrc_forest != han_forest

    if forest_frac_jrc <= FOREST_FRAC_CLEAR and forest_frac_hansen <= FOREST_FRAC_CLEAR:
        v.tier = LOW
        v.reasons.append(
            f"non-forest at the EUDR cutoff on both baselines "
            f"(JRC {forest_frac_jrc:.0%}, Hansen {forest_frac_hansen:.0%} forest cover) — "
            "post-2020 change on this plot cannot constitute deforestation")
        return v

    if baselines_disagree:
        v.reasons.append(
            f"official baselines disagree on cutoff forest status "
            f"(JRC {forest_frac_jrc:.0%} vs Hansen {forest_frac_hansen:.0%}) — verdict depends on map choice")

    # 3 — observability gate
    ts_screened = ts_break_detected is not None  # None => plot never time-series screened
    thin_record = ts_obs_density is not None and ts_obs_density < OBS_DENSITY_MIN
    if thin_record and not ts_break_detected and (model_prob is None or model_prob < MODEL_PROB_HIGH):
        v.tier = INSUFFICIENT
        v.reasons.append(
            f"only {ts_obs_density:.0%} of monitoring months have a usable observation — "
            "'no change detected' would be unsupported by the optical record")
        return v

    # 4 — detector fusion
    ts_pos = bool(ts_break_detected)
    model_pos = model_prob is not None and model_prob >= MODEL_PROB_HIGH
    model_neg = model_prob is not None and model_prob <= MODEL_PROB_LOW
    label_pos = hansen_loss_post_frac is not None and hansen_loss_post_frac > 0.05

    n_pos = sum([ts_pos, model_pos, label_pos])

    if ts_pos and (model_pos or model_prob is None):
        v.tier = HIGH
        v.reasons.append(
            f"sustained spectral breakpoint at {ts_break_date} on land that was forest at the cutoff"
            + (f"; learned detector concurs (p={model_prob:.2f})" if model_pos else ""))
        if label_pos:
            v.reasons.append(f"independent product corroboration: Hansen maps {hansen_loss_post_frac:.0%} of the plot as post-2020 loss")
        return v

    if n_pos >= 1:
        v.tier = MEDIUM
        if ts_pos and model_neg:
            v.reasons.append(
                f"time-series breakpoint at {ts_break_date} but the learned detector disagrees "
                f"(p={model_prob:.2f}) — plausible degradation, seasonal signal, or detector miss; needs review")
        elif model_pos and not ts_pos:
            v.reasons.append(
                f"learned detector flags clearing (p={model_prob:.2f}) without a confirmed time-series break — "
                "possibly recent, partial, or below the persistence rule; needs review")
        elif label_pos:
            v.reasons.append(
                f"Hansen maps {hansen_loss_post_frac:.0%} of the plot as post-2020 loss but neither "
                "in-pipeline detector confirms — vintage or resolution mismatch; needs review")
        return v

    # 5 — no clearing signal fired. Before certifying LOW, insist the plot was
    # actually screened for change: a plot that was forest at the cutoff but
    # never time-series screened has not been *checked*, so "no clearing" is
    # unsupported and the honest tier is INSUFFICIENT, not LOW. (If the learned
    # arm had flagged it, the MEDIUM branch above would already have returned.)
    if not ts_screened:
        v.tier = INSUFFICIENT
        v.reasons.append(
            "forest at the cutoff but not screened for post-cutoff change "
            "(outside the time-series subset) — no basis to certify 'no clearing'")
        return v

    if baselines_disagree or thin_record or geometry_warnings:
        v.tier = MEDIUM
        if thin_record:
            v.reasons.append(f"no clearing signal, but the record is thin ({ts_obs_density:.0%} of months observed)")
        if not v.reasons:
            v.reasons.append("no clearing signal, with reduced-confidence caveats above")
        return v

    v.tier = LOW
    v.reasons.append(
        "forest at cutoff, monitored throughout the post-cutoff window, and no "
        "detector — statistical or learned — found a clearing signal")
    return v
