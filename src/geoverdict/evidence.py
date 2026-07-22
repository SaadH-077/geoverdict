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
