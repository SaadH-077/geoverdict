"""Plot-geometry validation and repair: the intake gate of the pipeline.

WHY THIS MODULE EXISTS. EUDR due diligence starts with geolocation data
submitted by operators — thousands of farm polygons drawn by hand in web
tools, exported from mobile GPS apps, or copy-pasted between spreadsheets.
In practice a large share of them arrive damaged: self-intersecting rings,
swapped latitude/longitude, coordinates left in a projected CRS, duplicate
vertices, plots collapsed to points. Nothing downstream (satellite lookups,
forest baselines, verdicts) is meaningful until the geometry is trustworthy,
so the very first machine-learning problem in this domain is not a model at
all — it is data validation with measurable repair quality.

DESIGN: DIAGNOSE, THEN REPAIR, THEN MEASURE.
  1. `validate_geometry` / `validate_portfolio` return a list of typed
     `Issue`s per plot — a diagnosis, never a silent fix. Silent fixes are
     how a swapped-axes plot in the Gulf of Guinea gets a confident forest
     verdict.
  2. `repair` applies an ordered sequence of targeted fixes and records
     every action it took. An auditor can replay the exact transformation.
  3. Repair quality is measured (in notebook 01) as IoU between the repaired
     polygon and the known-good original, per failure class — a number, not
     a vibe.

Everything here depends only on shapely + numpy, so the whole module is unit
tested without a geospatial stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.strtree import STRtree
from shapely.validation import explain_validity

from . import config as cfg

# --------------------------------------------------------------------------
# The failure taxonomy
# --------------------------------------------------------------------------
# Severity semantics:
#   ERROR   — the plot cannot be analysed until this is fixed; a verdict on it
#             would be meaningless (wrong place, wrong shape, no shape).
#   WARNING — the plot is analysable, but the defect must be recorded because
#             it degrades confidence (e.g. overlapping with a sibling plot).

ERROR = "ERROR"
WARNING = "WARNING"

TAXONOMY: dict[str, tuple[str, str]] = {
    # code                 severity  human description
    "EMPTY_GEOMETRY":     (ERROR,   "geometry is missing or empty"),
    "NONFINITE_COORD":    (ERROR,   "coordinates contain NaN or infinity"),
    "WRONG_TYPE":         (ERROR,   "geometry is not a polygon or point (e.g. a line)"),
    "POINT_TOO_LARGE":    (ERROR,   "point submission but declared area > 4 ha (EUDR Art. 9 requires a polygon)"),
    "POINT_GEOMETRY":     (WARNING, "point submission (allowed under EUDR for plots <= 4 ha)"),
    "HAS_Z":              (WARNING, "vertices carry a Z coordinate (altitude noise from GPS exports)"),
    "LIKELY_PROJECTED":   (ERROR,   "coordinate magnitudes far outside lon/lat range — CRS not WGS84 (typically Web Mercator metres)"),
    "SWAPPED_AXES":       (ERROR,   "latitude and longitude appear interchanged"),
    "INVALID_RING":       (ERROR,   "ring is topologically invalid (self-intersection / bow-tie)"),
    "REPEATED_VERTICES":  (WARNING, "consecutive duplicate vertices"),
    "CW_WINDING":         (WARNING, "exterior ring wound clockwise (violates RFC 7946)"),
    "TOO_SMALL":          (ERROR,   f"area below {cfg.MIN_PLOT_HA} ha — digitising slip, not a plot"),
    "TOO_LARGE":          (ERROR,   f"area above {cfg.MAX_PLOT_HA} ha — unit or digitising error"),
    "SLIVER":             (ERROR,   "degenerate needle shape — thinner than a pixel, cannot be observed"),
    "OUTSIDE_AOI":        (ERROR,   "plot falls outside the declared sourcing region"),
    "DUPLICATE_PLOT":     (ERROR,   "near-identical to another plot in the same submission"),
    "OVERLAPPING_PLOTS":  (WARNING, "overlaps another plot in the same submission"),
}


@dataclass(frozen=True)
class Issue:
    code: str
    severity: str
    message: str

    @classmethod
    def of(cls, code: str, detail: str = "") -> "Issue":
        sev, desc = TAXONOMY[code]
        msg = f"{desc}" + (f" [{detail}]" if detail else "")
        return cls(code=code, severity=sev, message=msg)


# --------------------------------------------------------------------------
# Measurement helpers
# --------------------------------------------------------------------------

_EARTH_R = 6_371_008.8  # mean Earth radius, metres


def area_ha(geom: BaseGeometry) -> float:
    """Geodesic area in hectares, computed on the sphere.

    WHY NOT `geom.area`: shapely's area on lon/lat coordinates is in
    "square degrees", a unit that changes size with latitude and means
    nothing. The usual fix is projecting to a local UTM zone, but that drags
    in pyproj and a zone-selection step. Instead we evaluate the spherical
    shoelace formula directly (the same approach turf.js and Earth Engine's
    planar fallback use). Against an ellipsoidal calculation the error is
    below ~0.3% — irrelevant for thresholds like "is this bigger than 4 ha",
    and the trade is a dependency-free, unit-testable function.
    """
    if geom is None or geom.is_empty:
        return 0.0
    if isinstance(geom, MultiPolygon):
        return float(sum(area_ha(p) for p in geom.geoms))
    if not isinstance(geom, Polygon):
        return 0.0

    def ring_area(coords) -> float:
        xs = np.radians(np.asarray([c[0] for c in coords], dtype=np.float64))
        ys = np.radians(np.asarray([c[1] for c in coords], dtype=np.float64))
        # spherical shoelace: sum (λ2-λ1)(2 + sin φ1 + sin φ2) over edges
        total = np.sum((xs[1:] - xs[:-1]) * (2.0 + np.sin(ys[:-1]) + np.sin(ys[1:])))
        return abs(total * _EARTH_R * _EARTH_R / 2.0)

    a = ring_area(geom.exterior.coords)
    for interior in geom.interiors:
        a -= ring_area(interior.coords)
    return float(max(a, 0.0) / 10_000.0)


def iou(a: BaseGeometry, b: BaseGeometry) -> float:
    """Intersection-over-union of two geometries (0 = disjoint, 1 = identical).

    This is the repair-quality metric: a repair that "succeeds" by returning
    a valid polygon somewhere else entirely scores ~0 and is counted as a
    failure, which is exactly the honesty the funnel needs.
    """
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
    except shapely.errors.GEOSException:
        a2, b2 = shapely.make_valid(a), shapely.make_valid(b)
        inter = a2.intersection(b2).area
        union = a2.union(b2).area
    return float(inter / union) if union > 0 else 0.0


def thinness(geom: Polygon) -> float:
    """Polsby-Popper compactness 4*pi*A/P^2: 1 for a circle, ->0 for a needle.

    Used for the sliver check. The threshold (0.02) is calibrated so that
    genuinely elongated riverside fields pass while degenerate needles a few
    metres wide — which no 10 m satellite pixel can observe — fail.
    """
    p = geom.length
    return float(4.0 * math.pi * geom.area / (p * p)) if p > 0 else 0.0


def swap_xy(geom: BaseGeometry) -> BaseGeometry:
    return shapely.transform(geom, lambda c: c[:, ::-1])


def mercator_to_wgs84(geom: BaseGeometry) -> BaseGeometry:
    """Inverse Web-Mercator (EPSG:3857 -> EPSG:4326), closed-form.

    WHY THIS SPECIFIC UN-PROJECTION EXISTS. When a submission arrives with
    coordinates like (-6_180_000, -790_000) the CRS was lost somewhere in a
    web tool, and in the wild that CRS is almost always Web Mercator, because
    that is what every web map serves. Inverting it is two lines of closed-form
    math, so the repair attempts it, then *verifies* the result lands in a
    plausible place before accepting — an unverified guess would be worse
    than rejecting the plot.
    """
    R = 6_378_137.0

    def inv(c: np.ndarray) -> np.ndarray:
        lon = np.degrees(c[:, 0] / R)
        lat = np.degrees(2.0 * np.arctan(np.exp(c[:, 1] / R)) - math.pi / 2.0)
        return np.column_stack([lon, lat])

    return shapely.transform(geom, inv)


def point_to_plot(point: Point, declared_area_ha: float) -> Polygon:
    """Expand a legitimate point submission into a circular analysis footprint.

    EUDR permits points for plots <= 4 ha, but a point has no area to observe
    satellite data over. The standard practice is to buffer to the declared
    area. Degrees are converted to metres with the cos(latitude) correction —
    at plot scale (< 250 m radius) the planar approximation is exact to
    centimetres.
    """
    radius_m = math.sqrt(declared_area_ha * 10_000.0 / math.pi)
    lat = point.y
    deg_lat = radius_m / 111_320.0
    deg_lon = radius_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    # buffer in a locally-scaled frame so the circle is round in metres
    scaled = shapely.transform(point, lambda c: c * np.array([1.0 / deg_lon, 1.0 / deg_lat]))
    circle = scaled.buffer(1.0, quad_segs=16)
    return shapely.transform(circle, lambda c: c * np.array([deg_lon, deg_lat]))


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _coords_array(geom: BaseGeometry) -> np.ndarray:
    return shapely.get_coordinates(geom, include_z=False)


def _in_bbox(x: float, y: float, bbox, pad: float) -> bool:
    return (bbox[0] - pad) <= x <= (bbox[2] + pad) and (bbox[1] - pad) <= y <= (bbox[3] + pad)


def validate_geometry(
    geom: BaseGeometry | None,
    declared_area_ha: float | None = None,
    aoi_bbox: tuple[float, float, float, float] | None = cfg.AOI_BBOX,
    aoi_pad_deg: float = 1.0,
) -> list[Issue]:
    """Diagnose one geometry. Returns [] only for a clean, analysable plot.

    ORDER MATTERS: structural checks run first because later checks are
    meaningless on broken input (asking the area of a NaN polygon answers
    nothing). Checks after a disqualifying structural failure are skipped
    rather than allowed to emit misleading secondary noise.
    """
    issues: list[Issue] = []

    # -- existence -----------------------------------------------------------
    if geom is None or geom.is_empty:
        return [Issue.of("EMPTY_GEOMETRY")]

    coords = _coords_array(geom)
    if not np.isfinite(coords).all():
        return [Issue.of("NONFINITE_COORD")]

    # -- type ----------------------------------------------------------------
    if isinstance(geom, Point):
        if declared_area_ha is not None and declared_area_ha > cfg.POINT_MAX_AREA_HA:
            return [Issue.of("POINT_TOO_LARGE", f"declared {declared_area_ha:.1f} ha")]
        return [Issue.of("POINT_GEOMETRY")]
    if not isinstance(geom, (Polygon, MultiPolygon)):
        return [Issue.of("WRONG_TYPE", geom.geom_type)]

    if shapely.has_z(geom):
        issues.append(Issue.of("HAS_Z"))
        geom = shapely.force_2d(geom)
        coords = _coords_array(geom)

    # -- coordinate frame ------------------------------------------------------
    max_abs = float(np.abs(coords).max())
    if max_abs > 360.0:
        # Magnitudes in the thousands+ are metres of some projected CRS. This is
        # disqualifying for every later check (area, AOI), so return here.
        issues.append(Issue.of("LIKELY_PROJECTED", f"|coord| up to {max_abs:,.0f}"))
        return issues

    out_of_range = (np.abs(coords[:, 0]) > 180.0).any() or (np.abs(coords[:, 1]) > 90.0).any()
    cx, cy = float(coords[:, 0].mean()), float(coords[:, 1].mean())
    if aoi_bbox is not None:
        in_aoi = _in_bbox(cx, cy, aoi_bbox, aoi_pad_deg)
        swapped_in_aoi = _in_bbox(cy, cx, aoi_bbox, aoi_pad_deg)
        # Heuristic, and stated as one: swapped axes are only *detectable*
        # relative to an expectation of where the plots should be. Both
        # orderings are numerically legal coordinates; a sourcing region is
        # what breaks the symmetry. Without an AOI this check cannot fire.
        if not in_aoi and swapped_in_aoi:
            issues.append(Issue.of("SWAPPED_AXES", f"centroid ({cx:.2f}, {cy:.2f})"))
            return issues
        if not in_aoi:
            issues.append(Issue.of("OUTSIDE_AOI", f"centroid ({cx:.2f}, {cy:.2f})"))
            return issues
    elif out_of_range:
        issues.append(Issue.of("SWAPPED_AXES", "coordinates out of lon/lat range"))
        return issues

    # -- topology ---------------------------------------------------------------
    if not geom.is_valid:
        issues.append(Issue.of("INVALID_RING", explain_validity(geom)))

    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    for p in polys:
        ring = np.asarray(p.exterior.coords)
        if len(ring) - len(shapely.get_coordinates(shapely.remove_repeated_points(p.exterior))) > 0:
            issues.append(Issue.of("REPEATED_VERTICES"))
            break
    if geom.is_valid:
        for p in polys:
            if not p.exterior.is_ccw:
                issues.append(Issue.of("CW_WINDING"))
                break

    # -- size and shape (only meaningful if the topology is sound) --------------
    ref = shapely.make_valid(geom) if not geom.is_valid else geom
    a = area_ha(ref)
    if a < cfg.MIN_PLOT_HA:
        issues.append(Issue.of("TOO_SMALL", f"{a:.3f} ha"))
    elif a > cfg.MAX_PLOT_HA:
        issues.append(Issue.of("TOO_LARGE", f"{a:,.0f} ha"))
    elif isinstance(ref, Polygon) and thinness(ref) < 0.02:
        issues.append(Issue.of("SLIVER", f"compactness {thinness(ref):.3f}"))

    return issues


def validate_portfolio(
    geoms: list[BaseGeometry | None],
    declared_areas: list[float | None] | None = None,
    aoi_bbox=cfg.AOI_BBOX,
    duplicate_iou: float = 0.95,
    overlap_frac: float = 0.02,
) -> list[list[Issue]]:
    """Per-plot issues, including the cross-plot checks a single geometry cannot see.

    Duplicates and overlaps are *portfolio* properties: the same farm submitted
    twice under two supplier IDs, or two neighbours both claiming the boundary
    strip. They are found with an STRtree in O(n log n); the pairwise-everything
    alternative is O(n^2) and stops being funny at real submission sizes.
    """
    declared_areas = declared_areas or [None] * len(geoms)
    per_plot = [
        validate_geometry(g, declared_areas[i], aoi_bbox=aoi_bbox)
        for i, g in enumerate(geoms)
    ]

    # Only structurally sound polygons participate in cross-plot checks.
    ok_idx = [
        i for i, (g, iss) in enumerate(zip(geoms, per_plot))
        if g is not None and not g.is_empty
        and isinstance(g, (Polygon, MultiPolygon))
        and not any(x.code in ("LIKELY_PROJECTED", "SWAPPED_AXES", "OUTSIDE_AOI",
                               "NONFINITE_COORD", "INVALID_RING") for x in iss)
    ]
    if len(ok_idx) < 2:
        return per_plot

    valid_geoms = [shapely.make_valid(geoms[i]) for i in ok_idx]
    tree = STRtree(valid_geoms)
    flagged_dup: set[int] = set()
    for local_i, gi in enumerate(valid_geoms):
        for local_j in tree.query(gi, predicate="intersects"):
            local_j = int(local_j)
            if local_j <= local_i:
                continue
            gj = valid_geoms[local_j]
            pair_iou = iou(gi, gj)
            i_global, j_global = ok_idx[local_i], ok_idx[local_j]
            if pair_iou >= duplicate_iou:
                # the *second* occurrence is the duplicate — deterministic and
                # keeps exactly one copy analysable
                if j_global not in flagged_dup:
                    per_plot[j_global].append(Issue.of("DUPLICATE_PLOT", f"IoU {pair_iou:.2f} with plot {i_global}"))
                    flagged_dup.add(j_global)
            else:
                inter = gi.intersection(gj).area
                if inter / min(gi.area, gj.area) > overlap_frac:
                    detail_i = f"{100 * inter / gi.area:.0f}% with plot {j_global}"
                    detail_j = f"{100 * inter / gj.area:.0f}% with plot {i_global}"
                    per_plot[i_global].append(Issue.of("OVERLAPPING_PLOTS", detail_i))
                    per_plot[j_global].append(Issue.of("OVERLAPPING_PLOTS", detail_j))
    return per_plot


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------

@dataclass
class RepairResult:
    geometry: BaseGeometry | None
    actions: list[str] = field(default_factory=list)
    resolved: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    ok: bool = False          # True => analysable after repair
    notes: str = ""


def _largest_polygon(geom: BaseGeometry) -> tuple[BaseGeometry, float]:
    """Largest polygonal piece and the fraction of total area it retains.

    `make_valid` on a bow-tie returns the two lobes as a MultiPolygon. Keeping
    the largest lobe assumes the big lobe is the intended field and the small
    one is the digitising slip — usually true, and crucially the retained-area
    fraction is RECORDED so a 55/45 split (where the assumption is dubious)
    is visible in the audit trail instead of vanishing.
    """
    polys: list[Polygon] = []
    if isinstance(geom, Polygon):
        return geom, 1.0
    for part in getattr(geom, "geoms", []):
        if isinstance(part, Polygon):
            polys.append(part)
        elif isinstance(part, MultiPolygon):
            polys.extend(part.geoms)
    if not polys:
        return None, 0.0
    total = sum(p.area for p in polys)
    best = max(polys, key=lambda p: p.area)
    return best, (best.area / total if total > 0 else 0.0)


def repair(
    geom: BaseGeometry | None,
    issues: list[Issue],
    declared_area_ha: float | None = None,
    aoi_bbox=cfg.AOI_BBOX,
) -> RepairResult:
    """Apply targeted fixes in a fixed order and log every action.

    THE ORDER IS THE ALGORITHM: coordinate-frame fixes must precede topology
    fixes (a bow-tie in Web Mercator metres must be un-projected before
    `make_valid` means anything), and size checks must come last because area
    is only defined once the frame and topology are sound. After the fixes,
    the geometry is re-validated from scratch — a repair is only claimed as
    successful if the *validator* agrees, not because the repair code ran.
    """
    res = RepairResult(geometry=geom)
    codes = {i.code for i in issues}

    if "EMPTY_GEOMETRY" in codes or "NONFINITE_COORD" in codes or "WRONG_TYPE" in codes:
        res.unresolved = sorted(codes)
        res.notes = "nothing to repair from — plot must be re-submitted"
        return res
    if "POINT_TOO_LARGE" in codes:
        res.unresolved = sorted(codes)
        res.notes = "EUDR requires a polygon at this size — re-submission required"
        return res
    if "DUPLICATE_PLOT" in codes:
        # A duplicate is a PORTFOLIO defect: the fix is dropping this row, not
        # transforming its geometry. Re-validating the single geometry would
        # come back clean and silently launder the duplicate into the analysis
        # set — so this is an explicit refusal, and the first occurrence is
        # the copy that stays analysable.
        res.unresolved = sorted(codes)
        res.notes = "duplicate submission — excluded; the first occurrence remains in the analysis set"
        return res

    g = geom

    if "HAS_Z" in codes:
        g = shapely.force_2d(g)
        res.actions.append("dropped Z coordinates")

    if "SWAPPED_AXES" in codes:
        g = swap_xy(g)
        res.actions.append("swapped x/y axes")

    if "LIKELY_PROJECTED" in codes:
        candidate = mercator_to_wgs84(g)
        c = shapely.get_coordinates(candidate).mean(axis=0)
        if aoi_bbox is not None and _in_bbox(float(c[0]), float(c[1]), aoi_bbox, 1.0):
            g = candidate
            res.actions.append("inverse Web-Mercator projection (verified against AOI)")
        else:
            res.unresolved = sorted(codes)
            res.notes = "projected CRS could not be identified — needs the original CRS from the supplier"
            return res

    if "POINT_GEOMETRY" in codes and isinstance(g, Point):
        area = declared_area_ha if declared_area_ha else 1.0
        g = point_to_plot(g, area)
        res.actions.append(f"buffered point to declared {area:.1f} ha footprint")

    if "REPEATED_VERTICES" in codes:
        g = shapely.remove_repeated_points(g)
        res.actions.append("removed repeated vertices")

    if "INVALID_RING" in codes or not g.is_valid:
        fixed = shapely.make_valid(g)
        largest, kept = _largest_polygon(fixed)
        if largest is None:
            res.unresolved = sorted(codes)
            res.notes = "make_valid produced no polygonal part"
            return res
        g = largest
        res.actions.append(f"make_valid, kept largest part ({kept:.0%} of area)")

    if isinstance(g, Polygon) and not g.exterior.is_ccw:
        g = orient(g, sign=1.0)
        if "CW_WINDING" in codes:
            res.actions.append("re-oriented exterior ring counter-clockwise")

    # -- re-validate: the validator, not the repairer, declares success --------
    remaining = validate_geometry(g, declared_area_ha, aoi_bbox=aoi_bbox)
    remaining_codes = {i.code for i in remaining}
    res.geometry = g
    res.resolved = sorted(codes - remaining_codes)
    res.unresolved = sorted(remaining_codes)
    res.ok = not any(i.severity == ERROR for i in remaining)
    if not res.ok and not res.notes:
        res.notes = "residual errors after repair — routed to manual review"
    return res
