"""Every figure, one style. Plot helpers shared by all notebooks.

WHY A VIZ MODULE. Two goals: (1) every figure in the repo looks like it came
from the same report — one font scale, one palette, one grid style — because
visual consistency is what makes a seven-figure argument readable as ONE
argument; (2) figures a reviewer might reproduce live in code, not in cells,
so the notebook shows the call and the caption, and the styling noise stays
out of the narrative.

Colour semantics are FIXED across the whole project:
  green  = forest / LOW risk / no-change
  orange = MEDIUM risk / warnings
  red    = clearing / HIGH risk / errors
  grey   = insufficient / masked / no-data
A reader who has seen one figure has learned the legend for all of them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import config as cfg

PALETTE = {
    "forest": "#2f9e44", "clearing": "#e03131", "warn": "#f08c00",
    "insufficient": "#868e96", "water": "#1971c2", "neutral": "#495057",
    "accent": "#7048e8",
}
TIER_COLORS = {"LOW": PALETTE["forest"], "MEDIUM": PALETTE["warn"],
               "HIGH": PALETTE["clearing"], "INSUFFICIENT_EVIDENCE": PALETTE["insufficient"]}


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.facecolor": "white",
    })


def save(fig, name: str, directory: Path | str | None = None) -> Path:
    directory = Path(directory or cfg.FIGURE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    print(f"figure saved: {path}")
    return path


# --------------------------------------------------------------------------
# Geometry drawing
# --------------------------------------------------------------------------

def draw_geom(ax, geom, *, facecolor="none", edgecolor=PALETTE["neutral"],
              lw=1.2, alpha=1.0, zorder=2, label=None):
    """Draw any shapely geometry on a matplotlib axis (no geopandas needed).

    Handles Polygon/MultiPolygon rings and Points; drawing raw rings rather
    than patches keeps self-INTERSECTING (invalid) inputs drawable — which
    is exactly what the corruption gallery needs to show.
    """
    from shapely.geometry import MultiPolygon, Point, Polygon

    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Point):
        ax.plot(geom.x, geom.y, "o", color=edgecolor, ms=6, zorder=zorder, label=label)
        return
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    for k, p in enumerate(polys):
        if not isinstance(p, Polygon):
            continue
        # slice to (x, y) explicitly: a Z-carrying ring (e.g. the add_z
        # corruption) yields 3 columns, and zip(*coords) would then unpack
        # into three names and crash. Taking columns 0:2 draws any geometry,
        # 2D or 3D, which is exactly what the corruption gallery needs.
        ring = np.asarray(p.exterior.coords, dtype=float)
        xs, ys = ring[:, 0], ring[:, 1]
        if facecolor != "none":
            ax.fill(xs, ys, facecolor=facecolor, alpha=alpha * 0.5, zorder=zorder - 1)
        ax.plot(xs, ys, color=edgecolor, lw=lw, alpha=alpha, zorder=zorder,
                label=label if k == 0 else None)


def plot_on_basemap(ax, rgb, bbox, geoms_styles, title: str = ""):
    """Draw one or more geometries over a Sentinel-2 basemap in lon/lat space.

    geoms_styles: list of (geom, style_dict) where style_dict is kwargs for
    draw_geom. imshow uses extent = the lon/lat bbox and origin='upper' (the
    read is north-up), so the polygon coordinates and the pixels share one
    coordinate system and line up by construction.
    """
    if rgb is not None:
        ax.imshow(rgb, extent=[bbox[0], bbox[2], bbox[1], bbox[3]], origin="upper")
    else:
        ax.text(0.5, 0.5, "no cloud-free\nscene found", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color=PALETTE["insufficient"])
        ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
    for geom, style in geoms_styles:
        draw_geom(ax, geom, **style)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=9)


def repair_before_after(cases: list[dict], title: str = ""):
    """Paired before/after panels: 'as submitted' vs 'after repair', per case.

    Each case: {name, clean, corrupted, repaired, iou}. Left panel draws the
    faint intended shape (grey) under the damaged submission (red); right panel
    draws the same faint intended shape under the repair (green) — so the eye
    sees the green land back onto the grey. This is the money shot of the
    chapter: a repair you can visually verify.
    """
    n = len(cases)
    fig, axes = plt.subplots(n, 2, figsize=(7.2, 3.1 * n))
    axes = np.atleast_2d(axes)
    for row, case in enumerate(cases):
        axL, axR = axes[row, 0], axes[row, 1]
        for ax in (axL, axR):
            draw_geom(ax, case.get("clean"), edgecolor=PALETTE["insufficient"], lw=2.6, alpha=0.55)
        draw_geom(axL, case.get("corrupted"), edgecolor=PALETTE["clearing"], lw=1.6)
        draw_geom(axR, case.get("repaired"), edgecolor=PALETTE["forest"], lw=1.8)
        axL.set_ylabel(case["name"], fontsize=9, fontweight="bold")
        if row == 0:
            axL.set_title("as submitted", fontsize=10, color=PALETTE["clearing"])
            axR.set_title("after repair", fontsize=10, color=PALETTE["forest"])
        iou = case.get("iou")
        axR.set_xlabel(f"IoU vs intended: {iou:.3f}" if iou is not None else "→ manual review",
                       fontsize=8)
        for ax in (axL, axR):
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.grid(False)
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def corruption_gallery(cases: list[dict], n_cols: int = 4, title: str = ""):
    """Grid of (clean, corrupted, repaired) overlays, one panel per failure class.

    cases: [{name, clean, corrupted, repaired, caption}]. The clean truth is
    grey, the corruption red, the repair green — a reader sees at a glance
    which classes come back and which do not.
    """
    n = len(cases)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, case in zip(axes, cases):
        draw_geom(ax, case.get("clean"), edgecolor=PALETTE["insufficient"], lw=2.2, alpha=0.9)
        draw_geom(ax, case.get("corrupted"), edgecolor=PALETTE["clearing"], lw=1.2, alpha=0.9)
        draw_geom(ax, case.get("repaired"), edgecolor=PALETTE["forest"], lw=1.4, alpha=0.95)
        ax.set_title(case["name"], fontsize=10)
        if case.get("caption"):
            ax.set_xlabel(case["caption"], fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        ax.grid(False)
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Chips and maps
# --------------------------------------------------------------------------

def chip_rgb(chip: np.ndarray, bands=cfg.CHIP_BANDS, low: float = 2, high: float = 98) -> np.ndarray:
    """(C,H,W) reflectance -> displayable RGB with a percentile stretch.

    Sentinel-2 reflectance over land lives in roughly [0, 0.4]; mapped
    straight to [0,1] every chip looks nearly black. The 2-98% stretch is
    display-only and PER CHIP — fine for looking, never for analysis, which
    is why it lives here in viz and nowhere near the model code.
    """
    idx = [bands.index(b) for b in ("B04", "B03", "B02")]
    rgb = np.stack([chip[i] for i in idx], axis=-1)
    lo, hi = np.percentile(rgb, [low, high])
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def before_after_panel(ax_pair, chip_t1, chip_t2, title_t1="T1 (cutoff epoch)",
                       title_t2="T2 (recent)"):
    for ax, chip, title in zip(ax_pair, (chip_t1, chip_t2), (title_t1, title_t2)):
        ax.imshow(chip_rgb(chip))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)


def portfolio_map(ax, geoms, values, cmap_colors: dict, aoi_bbox=cfg.AOI_BBOX,
                  legend_title: str = ""):
    """Plot centroids over the AOI, coloured by a categorical value (e.g. tier)."""
    seen = set()
    for g, v in zip(geoms, values):
        if g is None or g.is_empty:
            continue
        c = g.centroid
        color = cmap_colors.get(v, PALETTE["neutral"])
        ax.plot(c.x, c.y, "o", color=color, ms=4, alpha=0.8,
                label=v if v not in seen else None)
        seen.add(v)
    ax.set_xlim(aoi_bbox[0] - 0.02, aoi_bbox[2] + 0.02)
    ax.set_ylim(aoi_bbox[1] - 0.02, aoi_bbox[3] + 0.02)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_aspect(1.0 / np.cos(np.radians((aoi_bbox[1] + aoi_bbox[3]) / 2)))
    if legend_title:
        ax.legend(title=legend_title, fontsize=8, loc="upper right")


# --------------------------------------------------------------------------
# Time series
# --------------------------------------------------------------------------

def plot_series(ax, series, break_result=None, cutoff: str = cfg.CUTOFF_DATE,
                label: str = "NDVI", color: str = PALETTE["forest"]):
    """One plot's monthly series with baseline band, cutoff line, breakpoint.

    Gaps are drawn as gaps (no line through NaN months): the missing wet-
    season observations are part of the story, and interpolating them away
    in the FIGURE would misrepresent what the detector actually saw.
    """
    import pandas as pd

    s = series.copy()
    ax.plot(s.index, s.values, ".-", color=color, lw=1.0, ms=3.5, label=label)
    ax.axvline(pd.Timestamp(cutoff), color=PALETTE["neutral"], ls="--", lw=1.2)
    ax.annotate("EUDR cutoff", xy=(pd.Timestamp(cutoff), ax.get_ylim()[1]),
                fontsize=8, color=PALETTE["neutral"], ha="left", va="top",
                xytext=(4, -2), textcoords="offset points")
    if break_result is not None and break_result.get("detected"):
        bd = pd.Timestamp(break_result["break_date"])
        ax.axvline(bd, color=PALETTE["clearing"], lw=1.4)
        ax.annotate(f"break {break_result['break_date']}", xy=(bd, ax.get_ylim()[0]),
                    fontsize=8, color=PALETTE["clearing"], ha="left", va="bottom",
                    xytext=(4, 2), textcoords="offset points")
        thr = break_result.get("threshold")
        if thr is not None and np.isfinite(thr):
            ax.axhline(thr, color=PALETTE["warn"], lw=0.9, ls=":",
                       label=f"detection threshold {thr:.2f}")
    ax.set_ylabel(label)


# --------------------------------------------------------------------------
# Evaluation figures
# --------------------------------------------------------------------------

def plot_pr_curves(ax, curves: dict[str, tuple[np.ndarray, np.ndarray, float]]):
    """curves: name -> (precision, recall, pr_auc)."""
    colors = [PALETTE["neutral"], PALETTE["accent"], PALETTE["clearing"],
              PALETTE["forest"], PALETTE["warn"]]
    for (name, (p, r, a)), c in zip(curves.items(), colors):
        ax.plot(r, p, lw=1.8, color=c, label=f"{name} (PR-AUC {a:.3f})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower left")


def plot_reliability(ax, reports: dict[str, dict]):
    """Reliability diagram(s): reports: name -> expected_calibration_error() dict."""
    ax.plot([0, 1], [0, 1], color=PALETTE["insufficient"], lw=1.0, ls="--",
            label="perfect calibration")
    colors = [PALETTE["accent"], PALETTE["forest"], PALETTE["clearing"], PALETTE["warn"]]
    for (name, rep), c in zip(reports.items(), colors):
        ax.plot(rep["bin_conf"], rep["bin_acc"], "o-", color=c, lw=1.4, ms=4,
                label=f"{name} (ECE {rep['ece']:.3f})")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")


# --------------------------------------------------------------------------
# The evidence figure — one page, one plot, everything an auditor needs
# --------------------------------------------------------------------------

def evidence_figure(bundle: dict, chip_t1=None, chip_t2=None):
    """Render one evidence bundle as a single reviewable page.

    Layout: header (verdict + reasons), before/after chips, NDVI and NBR
    series with the breakpoint, and the provenance footer. Ten-second
    readability is the design goal: tier colour and the breakpoint line
    carry the story; everything else is supporting detail.
    """
    import pandas as pd

    tier = bundle["verdict"]["tier"]
    color = TIER_COLORS.get(tier, PALETTE["neutral"])
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.4, 1.4], hspace=0.55, wspace=0.35)

    # header
    ax_h = fig.add_subplot(gs[0, :]); ax_h.axis("off")
    plot = bundle["plot"]
    ax_h.text(0.0, 0.95, f"Plot {plot['plot_id']}  —  {plot.get('area_ha', float('nan')):.1f} ha",
              fontsize=14, fontweight="bold", va="top")
    ax_h.text(0.0, 0.55, f"VERDICT: {tier}", fontsize=13, fontweight="bold", color=color, va="top")
    reasons = bundle["verdict"]["reasons"]
    ax_h.text(0.0, 0.30, "\n".join(f"• {r}" for r in reasons[:4]), fontsize=8.5, va="top",
              color=PALETTE["neutral"], wrap=True)

    # chips
    if chip_t1 is not None and chip_t2 is not None:
        ax1, ax2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        before_after_panel((ax1, ax2), chip_t1, chip_t2)

    # series
    sig = bundle["signals"]
    for row_col, key, label in ((gs[1, 2:], "ndvi_monthly", "NDVI"), (gs[2, 2:], "nbr_monthly", "NBR")):
        recs = sig.get(key)
        if not recs:
            continue
        ax = fig.add_subplot(row_col)
        s = pd.Series({pd.Timestamp(r["month"]): (np.nan if r["value"] is None else r["value"])
                       for r in recs}).sort_index()
        plot_series(ax, s, sig.get("breakpoint"), label=label,
                    color=PALETTE["forest"] if label == "NDVI" else PALETTE["accent"])

    # provenance footer
    ax_f = fig.add_subplot(gs[2, :2]); ax_f.axis("off")
    prov_lines = []
    for p in bundle.get("provenance", [])[:5]:
        ids = p.get("identifiers", [])
        shown = ", ".join(str(x) for x in ids[:2]) + (f" (+{len(ids)-2} more)" if len(ids) > 2 else "")
        prov_lines.append(f"• {p.get('source', '?')}: {shown}")
    ax_f.text(0, 0.9, "Data provenance", fontsize=9, fontweight="bold", va="top")
    ax_f.text(0, 0.72, "\n".join(prov_lines) or "—", fontsize=7.5, va="top", color=PALETTE["neutral"])
    ax_f.text(0, 0.05, bundle.get("disclaimer", ""), fontsize=7, style="italic",
              color=PALETTE["insufficient"], va="bottom")
    return fig
