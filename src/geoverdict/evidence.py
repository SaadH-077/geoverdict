"""Evidence bundles: the audit-ready artefact behind every verdict.

WHY THIS FILE IS THE POINT OF THE PROJECT. A probability is not evidence.
Under EUDR, an operator files a Due Diligence Statement and must be able to
show authorities WHY a plot was assessed as it was. So for every plot the
pipeline emits a machine-readable bundle containing: the verdict and its
reasons, the geometry audit trail (what arrived, what was repaired, how),
the forest-baseline values from both official maps, the spectral time
series with the detected breakpoint, the model probability, and full data
provenance (which satellite scenes, which map vintages, which code version).

The JSON is the record; the one-page figure (viz.evidence_figure) is the
same content made legible to a human reviewer in ten seconds.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

SCHEMA_VERSION = "1.0"


def build_bundle(
    verdict,                     # risk.Verdict
    plot_record: dict,           # id, area_ha, centroid, geometry audit trail
    ndvi_series: pd.Series | None = None,
    nbr_series: pd.Series | None = None,
    break_result: dict | None = None,
    provenance: list[dict] | None = None,
) -> dict:
    """Assemble one plot's bundle. Everything JSON-native, nothing lossy.

    Time series are stored as explicit (month, value|null) pairs rather than
    arrays, so a bundle read in isolation — by an auditor's tooling, years
    later — is self-describing. Nulls are preserved: the months we could NOT
    observe are part of the evidence.
    """
    def series_records(s: pd.Series | None) -> list[dict] | None:
        if s is None:
            return None
        return [
            {"month": str(ts.date()), "value": (None if np.isnan(v) else round(float(v), 4))}
            for ts, v in s.items()
        ]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "regulation": {
            "framework": "EUDR (Regulation (EU) 2023/1115)",
            "cutoff_date": cfg.CUTOFF_DATE,
            "monitoring_window": [cfg.MONITOR_START, cfg.MONITOR_END],
        },
        "plot": plot_record,
        "verdict": verdict.to_dict(),
        "signals": {
            "ndvi_monthly": series_records(ndvi_series),
            "nbr_monthly": series_records(nbr_series),
            "breakpoint": break_result,
        },
        "provenance": provenance or [],
        "disclaimer": (
            "Research prototype. Screening evidence, not a legal determination; "
            "HIGH/MEDIUM tiers indicate plots requiring human review."
        ),
    }


def save_bundle(bundle: dict, directory: Path | str | None = None) -> Path:
    directory = Path(directory or cfg.EVIDENCE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"plot_{bundle['plot']['plot_id']}.json"
    path.write_text(json.dumps(bundle, indent=2, default=cfg._json_default), encoding="utf-8")
    return path


def portfolio_summary(verdicts: list, path: Path | str | None = None) -> dict:
    """The DDS-support roll-up: tier counts + every non-LOW plot listed.

    This is the shape of what feeds a Due Diligence Statement: the operator
    asserts negligible risk FOR THE PORTFOLIO, backed by per-plot evidence
    for every exception. LOW plots appear as a count; everything else appears
    by name, because those are the plots a human must look at.
    """
    tiers: dict[str, int] = {}
    for v in verdicts:
        tiers[v.tier] = tiers.get(v.tier, 0) + 1
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_plots": len(verdicts),
        "tier_counts": tiers,
        "attention_required": [
            {"plot_id": v.plot_id, "tier": v.tier, "reasons": v.reasons}
            for v in verdicts if v.tier != "LOW"
        ],
    }
    if path is not None:
        Path(path).write_text(json.dumps(summary, indent=2, default=cfg._json_default), encoding="utf-8")
    return summary


def leading_reason(verdict) -> str:
    """The reason that actually DROVE the tier, not a carried-forward caveat.

    Geometry warnings and baseline-disagreement notes are appended before the
    tier-deciding reason, so `reasons[0]` for a HIGH plot can misleadingly read
    "geometry warning ..." instead of "sustained spectral breakpoint ...". This
    prefers the first non-caveat reason so the report headlines the real driver.
    """
    caveats = ("geometry warning", "baselines disagree")
    primary = [r for r in verdict.reasons if not any(c in r for c in caveats)]
    text = (primary[0] if primary else (verdict.reasons[0] if verdict.reasons else ""))
    return text.split(" — ")[0].split(" (")[0]


# Tier presentation: icon + one-line meaning, ordered for a risk report.
_TIER_META = [
    ("HIGH", "🟥", "corroborated post-2020 clearing on land that was forest at the cutoff"),
    ("MEDIUM", "⚠️", "conflicting or partial evidence — human review required"),
    ("INSUFFICIENT_EVIDENCE", "◻️", "not screened or unobservable — the pipeline abstains"),
    ("LOW", "✅", "negligible risk — no post-cutoff clearing found"),
]


def format_dds_report(verdicts: list, areas: dict | None = None, review_min: int = 15,
                      max_rows: int = 25) -> str:
    """A clean, human-readable EUDR screening summary in Markdown.

    This is the portfolio-level *support for* a Due Diligence Statement — the
    document a compliance officer reads before signing. It is deliberately not a
    raw JSON dump: a risk overview table, the analyst-workload translation, and a
    ranked table of the plots that actually need a human, each with its single
    most important reason. Full per-plot evidence lives in the JSON bundles.
    """
    n = max(len(verdicts), 1)
    counts = {}
    for v in verdicts:
        counts[v.tier] = counts.get(v.tier, 0) + 1
    total_ha = sum(areas.values()) if areas else None

    L = [
        "# EUDR Due Diligence — Screening Summary",
        "",
        "**Framework:** EUDR (Regulation (EU) 2023/1115)  ·  **Deforestation cut-off:** 31 December 2020  ",
        f"**Generated:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}  ",
        f"**Portfolio:** {len(verdicts)} land parcels"
        + (f"  ·  {total_ha:,.0f} ha total" if total_ha is not None else ""),
        "",
        "## Risk overview",
        "",
        "| | Tier | Parcels | Share | Assessment |",
        "|:--:|:--|--:|--:|:--|",
    ]
    for tier, icon, meaning in _TIER_META:
        c = counts.get(tier, 0)
        L.append(f"| {icon} | **{tier.replace('_EVIDENCE','')}** | {c} | {c/n:.0%} | {meaning} |")

    need = sum(counts.get(t, 0) for t in ("HIGH", "MEDIUM", "INSUFFICIENT_EVIDENCE"))
    hours_per_1000 = need / n * 1000 * review_min / 60
    L += [
        "",
        f"> **Cleared automatically (LOW):** {counts.get('LOW', 0)} parcels "
        f"({counts.get('LOW', 0)/n:.0%}).  "
        f"**Require human review:** {need} parcels ({need/n:.0%}) — "
        f"≈ {hours_per_1000:.0f} analyst-hours per 1,000 parcels at {review_min} min each.",
        "",
        "## Parcels requiring attention",
        "",
        "| Parcel | Tier | Area (ha) | Leading reason |",
        "|:--|:--|--:|:--|",
    ]
    rank = {"HIGH": 0, "MEDIUM": 1, "INSUFFICIENT_EVIDENCE": 2}
    attention = sorted((v for v in verdicts if v.tier != "LOW"),
                       key=lambda v: rank.get(v.tier, 3))
    for v in attention[:max_rows]:
        area = f"{areas.get(v.plot_id, float('nan')):.1f}" if areas else "—"
        L.append(f"| {v.plot_id} | {v.tier.replace('_EVIDENCE','')} | {area} | {leading_reason(v)[:80]} |")
    if len(attention) > max_rows:
        L.append(f"| … | | | *and {len(attention) - max_rows} more — full list in `dds_summary.json`* |")

    L += [
        "",
        "---",
        "",
        "*Screening evidence only — not a legal determination. Every HIGH / MEDIUM / "
        "INSUFFICIENT parcel above must be resolved by a human before a Due Diligence "
        "Statement is signed. Full per-parcel evidence bundles (imagery, spectral time "
        "series, provenance) are in `outputs/evidence/`.*",
    ]
    return "\n".join(L)
