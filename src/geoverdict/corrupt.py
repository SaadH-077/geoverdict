"""The corruption harness: manufacture realistic damage with known ground truth.

WHY MANUFACTURE DAMAGE AT ALL. Real supplier submissions are messy, but they
come without ground truth — when a polygon self-intersects you do not know
what the farmer *meant*, so you cannot score a repair. The standard testing
move (identical to how database fuzzers and data-quality suites work) is to
take known-good geometry, apply a taxonomy of realistic corruptions with a
seeded RNG, and then measure the validator (does it catch each class?) and
the repairer (does the repaired shape match the original, by IoU?). Every
corruption here reproduces a failure mode with a documented real-world cause.

The output of this module is the experiment of notebook 01: detection rate
and repair quality PER FAILURE CLASS — the kind of table an engineering team
can actually act on ("we auto-fix 100% of winding errors, 0% of unknown-CRS
plots; invest in supplier tooling for the latter").
"""

from __future__ import annotations

import math

import numpy as np
import shapely
import shapely.affinity
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

from .geometry import swap_xy

# --------------------------------------------------------------------------
# Individual corruptions
# --------------------------------------------------------------------------
# Each takes (polygon, rng) -> geometry. Names line up with the validator's
# taxonomy so notebook 01 can join "what we injected" against "what was
# diagnosed" — that join IS the detection-rate experiment.


def corrupt_swap_axes(g: Polygon, rng: np.random.Generator) -> BaseGeometry:
    """(lat, lon) instead of (lon, lat) — the single most common GeoJSON bug.

    GeoJSON mandates lon,lat order; humans, GPS apps, and Google Maps URLs
    all speak lat,lon. Every geospatial team on Earth has met this one.
    """
    return swap_xy(g)


def corrupt_web_mercator(g: Polygon, rng: np.random.Generator) -> BaseGeometry:
    """Coordinates exported in EPSG:3857 metres with the CRS tag lost."""
    R = 6_378_137.0

    def fwd(c: np.ndarray) -> np.ndarray:
        x = np.radians(c[:, 0]) * R
        y = R * np.log(np.tan(math.pi / 4.0 + np.radians(c[:, 1]) / 2.0))
        return np.column_stack([x, y])

    return shapely.transform(g, fwd)


def corrupt_bowtie(g: Polygon, rng: np.random.Generator) -> BaseGeometry:
    """Swap two non-adjacent vertices so the ring self-intersects.

    This is what hand-digitising produces when a vertex is dragged across
    the polygon, and what naive vertex-ordering code produces routinely.
    """
    coords = list(g.exterior.coords)[:-1]
    if len(coords) < 4:
        return g
    i = int(rng.integers(0, len(coords) - 2))
    j = min(i + 2, len(coords) - 1)
    coords[i], coords[j] = coords[j], coords[i]
    return Polygon(coords + [coords[0]])


def corrupt_repeated_vertices(g: Polygon, rng: np.random.Generator) -> BaseGeometry:
    """Duplicate a few vertices in place — classic GPS-logger stutter."""
    coords = list(g.exterior.coords)[:-1]
    out = []
    dup_at = set(rng.choice(len(coords), size=min(3, len(coords)), replace=False).tolist())
    for k, c in enumerate(coords):
        out.append(c)
        if k in dup_at:
            out.extend([c, c])
    return Polygon(out + [out[0]])


def corrupt_cw_winding(g: Polygon, rng: np.random.Generator) -> BaseGeometry:
    """Exterior ring wound clockwise — RFC 7946 violation from shapefile exports."""
    return orient(g, sign=-1.0)


def corrupt_add_z(g: Polygon, rng: np.random.Generator) -> BaseGeometry:
    """Attach altitude noise as a Z coordinate, as mobile GPS exports do."""
    return shapely.force_3d(g, z=float(rng.uniform(80, 400)))


def corrupt_collapse_to_point(g: Polygon, rng: np.random.Generator) -> BaseGeometry:
    """The whole plot arrives as a single coordinate.

    Sometimes legitimate (EUDR allows points <= 4 ha), sometimes a lost
    polygon. The validator must arbitrate using the declared area.
    """
    return g.centroid


def corrupt_micro(g: Polygon, rng: np.random.Generator) -> BaseGeometry:
    """Shrink to a few hundred m^2 — a fat-fingered digitising slip."""
    c = g.centroid
    return shapely.affinity.scale(g, xfact=0.02, yfact=0.02, origin=(c.x, c.y))


def corrupt_sliver(g: Polygon, rng: np.random.Generator) -> BaseGeometry:
    """Crush one axis to a needle a few metres wide, stretching the other so
    the AREA stays plot-sized — the point of a sliver is that it passes any
    naive area check while being unobservable at 10 m resolution."""
    c = g.centroid
    return shapely.affinity.scale(g, xfact=20.0, yfact=0.004, origin=(c.x, c.y))


def corrupt_teleport(g: Polygon, rng: np.random.Generator) -> BaseGeometry:
    """Translate the plot ~1000 km away — a pasted row from the wrong sheet."""
    return shapely.affinity.translate(g, xoff=float(rng.uniform(8, 15)), yoff=float(rng.uniform(8, 15)))


def corrupt_duplicate(g: Polygon, rng: np.random.Generator) -> BaseGeometry:
    """Near-identical copy (tiny jitter): same farm submitted twice.

    Applied by the portfolio builder as an EXTRA row, not a replacement —
    duplication is a portfolio-level defect.
    """
    dx, dy = rng.normal(0, 1e-5, size=2)
    return shapely.affinity.translate(g, xoff=float(dx), yoff=float(dy))


# What the validator is EXPECTED to diagnose for each injected corruption —
# the answer key for the detection-rate experiment in notebook 01. Several
# classes have MORE THAN ONE correct diagnosis, and the multiplicity is
# domain logic, not slack: a plot collapsed to a point is POINT_GEOMETRY when
# its declared area permits a point under EUDR Art. 9 and POINT_TOO_LARGE
# when it does not; a sliver on a small plot legitimately trips the area
# floor before the compactness check; a near-duplicate that drifted slightly
# reads as a heavy overlap. Any listed code counts as a correct detection.
CORRUPTIONS: dict[str, tuple] = {
    # name                 (function,                 acceptable diagnoses)
    "swap_axes":           (corrupt_swap_axes,        ("SWAPPED_AXES",)),
    "web_mercator":        (corrupt_web_mercator,     ("LIKELY_PROJECTED",)),
    "bowtie":              (corrupt_bowtie,           ("INVALID_RING",)),
    "repeated_vertices":   (corrupt_repeated_vertices, ("REPEATED_VERTICES",)),
    "cw_winding":          (corrupt_cw_winding,       ("CW_WINDING",)),
    "add_z":               (corrupt_add_z,            ("HAS_Z",)),
    "collapse_to_point":   (corrupt_collapse_to_point, ("POINT_GEOMETRY", "POINT_TOO_LARGE")),
    "micro":               (corrupt_micro,            ("TOO_SMALL",)),
    "sliver":              (corrupt_sliver,           ("SLIVER", "TOO_SMALL")),
    "teleport":            (corrupt_teleport,         ("OUTSIDE_AOI",)),
    "duplicate":           (corrupt_duplicate,        ("DUPLICATE_PLOT", "OVERLAPPING_PLOTS")),
}


# --------------------------------------------------------------------------
# Portfolio generation
# --------------------------------------------------------------------------

def make_plot(center_lon: float, center_lat: float, area_ha: float, rng: np.random.Generator) -> Polygon:
    """One irregular convex-ish field polygon of roughly the requested area.

    Fields are modelled as noisy polygons around an ellipse rather than
    rectangles, because rectangles are trivially valid under every corruption
    (a bow-tie needs >= 4 irregular vertices to be interesting) and because
    real smallholder boundaries are irregular. The construction: sample 6-11
    angles, radii jittered ±35%, then rescale to hit the target area exactly.
    """
    n = int(rng.integers(6, 12))
    angles = np.sort(rng.uniform(0, 2 * math.pi, size=n))
    radii = 1.0 + rng.uniform(-0.35, 0.35, size=n)
    aspect = float(rng.uniform(0.6, 1.0))
    xs, ys = np.cos(angles) * radii, np.sin(angles) * radii * aspect
    poly = Polygon(np.column_stack([xs, ys]))
    if not poly.is_valid:
        poly = poly.convex_hull

    # scale to the requested area, working in metres then back to degrees
    target_m2 = area_ha * 10_000.0
    deg_lat = 1.0 / 111_320.0
    deg_lon = 1.0 / (111_320.0 * max(math.cos(math.radians(center_lat)), 1e-6))
    current_m2 = poly.area / (deg_lat * deg_lon)  # unit-poly area in "m^2" after deg scaling
    s = math.sqrt(target_m2 / poly.area)
    poly = shapely.affinity.scale(poly, xfact=s * deg_lon, yfact=s * deg_lat, origin=(0, 0))
    del current_m2
    poly = shapely.affinity.rotate(poly, float(rng.uniform(0, 180)), origin=(0, 0))
    return shapely.affinity.translate(poly, xoff=center_lon, yoff=center_lat)


def generate_portfolio(
    n_plots: int,
    bbox: tuple[float, float, float, float],
    seed: int,
    area_lognorm_mean: float = 1.6,
    area_lognorm_sigma: float = 0.9,
    min_ha: float = 1.0,
    max_ha: float = 400.0,
) -> list[Polygon]:
    """A seeded portfolio of plot polygons scattered over the AOI.

    HONESTY NOTE (stated in the notebook too): the *geometries* are synthetic
    — there is no public registry of farm boundaries for this frontier — but
    everything measured UNDER them from notebook 02 onward is real satellite
    data over real land. Plot sizes follow a log-normal (median ~5 ha, tail to
    hundreds of ha), matching the size mix of smallholder + ranch parcels the
    EUDR actually covers; the log-normal is the standard empirical model for
    landholding size distributions.
    """
    rng = np.random.default_rng(seed)
    lon0, lat0, lon1, lat1 = bbox
    pad = 0.02  # keep plots off the AOI edge so chips never fall outside
    plots: list[Polygon] = []
    while len(plots) < n_plots:
        area = float(np.clip(rng.lognormal(area_lognorm_mean, area_lognorm_sigma), min_ha, max_ha))
        lon = float(rng.uniform(lon0 + pad, lon1 - pad))
        lat = float(rng.uniform(lat0 + pad, lat1 - pad))
        plots.append(make_plot(lon, lat, area, rng))
    return plots


def corrupt_portfolio(
    plots: list[Polygon],
    corruption_rate: float,
    seed: int,
) -> tuple[list, list[str | None], list[float]]:
    """Damage a fraction of the portfolio; return (geoms, injected_labels, declared areas).

    Corruption types are assigned round-robin over the shuffled victim set so
    every class gets an almost equal sample (a uniform draw would leave rare
    classes with too few examples to estimate a repair rate from). Declared
    areas are reported for every plot — with ±10% supplier noise, because real
    declared areas never match GIS-measured ones exactly, and the point/area
    arbitration (EUDR 4 ha rule) should be tested against noisy declarations.
    """
    rng = np.random.default_rng(seed)
    names = [n for n in CORRUPTIONS if n != "duplicate"]
    n_bad = int(round(len(plots) * corruption_rate))
    victims = rng.permutation(len(plots))[:n_bad]

    geoms: list = list(plots)
    injected: list[str | None] = [None] * len(plots)
    from .geometry import area_ha as _area
    declared = [float(_area(p) * rng.uniform(0.9, 1.1)) for p in plots]

    for k, idx in enumerate(victims):
        name = names[k % len(names)]
        fn, _ = CORRUPTIONS[name]
        geoms[idx] = fn(plots[idx], rng)
        injected[idx] = name

    # duplicates are appended as extra rows referencing a clean plot
    n_dup = max(2, n_bad // len(names))
    clean_idx = [i for i in range(len(plots)) if injected[i] is None]
    for idx in rng.choice(clean_idx, size=min(n_dup, len(clean_idx)), replace=False):
        geoms.append(corrupt_duplicate(plots[int(idx)], rng))
        injected.append("duplicate")
        declared.append(declared[int(idx)])

    return geoms, injected, declared
