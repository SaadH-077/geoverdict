"""Sentinel-2 access: STAC search and windowed COG reads, per plot.

THE ACCESS PATTERN, AND WHY IT IS THE ONLY ONE THAT SCALES. A Sentinel-2
tile is 110x110 km and hundreds of MB; a plot is a few hundred metres. The
pipeline therefore never downloads scenes — it queries the Earth Search
STAC API for scene metadata, then issues HTTP range reads against the
Cloud-Optimised GeoTIFFs for just the plot's window (a few kB each). This
is the same data path production EUDR systems use, and it is the only
reason screening hundreds of plots fits in a free Colab session.

Adapted from the author's earlier sentinel2-landcover-mapping project
(s2map.stac), reworked from scene-level to plot-level access.

All geospatial imports are function-local so the module imports (and the
rest of the package tests) without rasterio/pystac installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config as cfg

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

ASSET_KEYS: dict[str, str] = {
    "B02": "blue", "B03": "green", "B04": "red", "B08": "nir",
    "B11": "swir16", "B12": "swir22",
}
SCL_ASSET = "scl"

# SCL classes masked as unusable: no-data, saturated, cloud shadow, medium
# cloud, high cloud, thin cirrus. Dark-area and snow are real surfaces and
# stay. Shadows are masked because a cloud shadow over forest mimics the
# spectral DROP a clearing produces — the single nastiest false-positive
# source in optical change detection.
BAD_SCL = (0, 1, 3, 8, 9, 10)


@dataclass
class PlotObservation:
    """One plot's band values for one scene date: masked means + pixel chip."""

    item_id: str
    datetime: str
    cloud_cover_scene: float
    band_means: dict[str, float]      # mean over valid plot pixels
    valid_frac: float                 # usable fraction of plot pixels
    chip: np.ndarray | None = None    # (C, H, W) reflectance, plot-centred
    chip_valid: np.ndarray | None = None


def search_items(
    bbox=cfg.AOI_BBOX,
    date_range: str = f"{cfg.BASELINE_START}/{cfg.MONITOR_END}",
    max_cloud: float = 80.0,
    url: str = EARTH_SEARCH_URL,
) -> list:
    """All L2A items over the AOI, sorted by time.

    max_cloud is deliberately PERMISSIVE (80%): scene-level cloud cover is
    tile-wide, and a 60%-cloudy tile is often clear over a given plot. The
    per-plot SCL mask — not the scene metadata — decides usability. Filtering
    hard here would throw away exactly the wet-season observations the
    tropics are short of.
    """
    from pystac_client import Client

    client = Client.open(url)
    search = client.search(
        collections=[COLLECTION], bbox=list(bbox), datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=None,
    )
    items = list(search.items())
    return sorted(items, key=lambda it: str(it.datetime))


def plot_chip_bbox(centroid_lon: float, centroid_lat: float,
                   chip_px: int = cfg.CHIP_SIZE) -> tuple[float, float, float, float]:
    """A chip_px * 10 m square, centred on the plot, in lon/lat."""
    half_m = chip_px * 10.0 / 2.0
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * max(math.cos(math.radians(centroid_lat)), 1e-6))
    return (centroid_lon - dlon, centroid_lat - dlat, centroid_lon + dlon, centroid_lat + dlat)


def read_window(href: str, bbox, target_shape=None, resampling: str = "bilinear"):
    """Windowed read of one COG asset, reprojecting the lon/lat bbox to its CRS.

    Bilinear for reflectance (a continuous physical quantity); nearest for
    SCL (categorical — interpolating class codes averages 'cloud' with
    'vegetation' into a class that does not exist).
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    method = {"bilinear": Resampling.bilinear, "nearest": Resampling.nearest}[resampling]
    with rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=5)
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        out_shape = target_shape or (max(int(round(window.height)), 1), max(int(round(window.width)), 1))
        data = src.read(1, window=window, out_shape=out_shape, resampling=method, boundless=True, fill_value=0)
        return data


def observe_plot(
    item,
    centroid_lon: float,
    centroid_lat: float,
    bands=cfg.CHIP_BANDS,
    chip_px: int = cfg.CHIP_SIZE,
    keep_chip: bool = False,
    scale: float = 1e-4,
) -> PlotObservation | None:
    """Read one scene's chip over one plot; return None if unusably cloudy.

    The band means feed the time series (notebook 03); the chips feed the
    CNN (notebook 04). Same read serves both, so the two arms are guaranteed
    to have seen identical data — comparisons between them are then about
    the METHOD, not the ingestion.

    A <30% valid chip is rejected outright: a mean over a handful of clear
    pixels at a cloud edge is noise wearing a number's clothing.
    """
    bbox = plot_chip_bbox(centroid_lon, centroid_lat, chip_px)
    shape = (chip_px, chip_px)
    try:
        scl = read_window(item.assets[SCL_ASSET].href, bbox, shape, resampling="nearest")
    except Exception:
        return None
    valid = ~np.isin(scl, BAD_SCL)
    valid_frac = float(valid.mean())
    if valid_frac < 0.30:
        return None

    layers, means = [], {}
    for b in bands:
        try:
            arr = read_window(item.assets[ASSET_KEYS[b]].href, bbox, shape).astype(np.float32) * scale
        except Exception:
            return None
        means[b] = float(arr[valid].mean())
        if keep_chip:
            layers.append(arr)

    return PlotObservation(
        item_id=item.id,
        datetime=str(item.datetime),
        cloud_cover_scene=float(item.properties.get("eo:cloud_cover", float("nan"))),
        band_means=means,
        valid_frac=valid_frac,
        chip=(np.stack(layers) if keep_chip else None),
        chip_valid=(valid if keep_chip else None),
    )


def geom_view_bbox(geom, margin_frac: float = 0.45, min_span_deg: float = 0.0025):
    """A square-ish lon/lat window around a geometry, with margin, for display.

    Used to fetch a satellite basemap that shows a plot IN CONTEXT — the plot
    plus a margin of its surroundings — rather than cropped to its own edge.
    `min_span_deg` (~275 m) keeps a tiny plot from producing a uselessly
    zoomed thumbnail.
    """
    minx, miny, maxx, maxy = geom.bounds
    span = max(maxx - minx, maxy - miny, min_span_deg)
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    half = span * (0.5 + margin_frac)
    return (cx - half, cy - half, cx + half, cy + half)


def basemap_rgb(
    bbox,
    date_range: str = "2023-01-01/2024-12-31",
    max_cloud: float = 20.0,
    max_px: int = 256,
    compose: int = 4,
    url: str = EARTH_SEARCH_URL,
):
    """Fetch a true-colour Sentinel-2 basemap over a lon/lat bbox, for display.

    Returns (rgb, bbox, item_id, datetime) where rgb is an (H, W, 3) array
    percentile-stretched to [0, 1], meant purely for `imshow(rgb, extent=...)`
    with a geometry drawn on top. Returns (None, bbox, ...) if nothing usable
    covers the window — the caller reports the gap rather than drawing an empty
    box.

    COMPOSITING. A single Sentinel-2 scene is 110x110 km, so it covers a small
    plot fully but may only partially overlap a large AOI window, leaving
    no-data gaps. `compose` reads up to N least-cloudy scenes and fills each
    pixel from the first scene that has data there — a simple most-recent-valid
    mosaic that gives full coverage for wide windows while staying cheap for
    small ones (it stops as soon as the window is covered).

    This is display-only imagery: the stretch is per-tile and cosmetic, which
    is exactly why it lives here and nowhere near anything that measures.
    """
    try:
        from pystac_client import Client

        client = Client.open(url)
        items = sorted(
            client.search(collections=[COLLECTION], bbox=list(bbox), datetime=date_range,
                          query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=30).items(),
            key=lambda it: it.properties.get("eo:cloud_cover", 100.0),
        )
    except Exception:
        # missing pystac/rasterio, network failure, or no items — the caller
        # draws a "no scene" panel and reports coverage rather than crashing
        return None, bbox, "", ""
    if not items:
        return None, bbox, "", ""

    aspect = (bbox[3] - bbox[1]) / max(bbox[2] - bbox[0], 1e-9)
    h = max(int(round(max_px * min(aspect, 4.0))), 16)
    shape = (h, max_px)

    composite = None
    filled = None
    primary = items[0]
    for it in items[: max(compose, 1)]:
        try:
            chans = [read_window(it.assets[ASSET_KEYS[b]].href, bbox, shape).astype(np.float32) * 1e-4
                     for b in ("B04", "B03", "B02")]
        except Exception:
            continue
        rgb_i = np.dstack(chans)
        valid_i = rgb_i.sum(axis=2) > 0  # boundless reads fill no-data with 0
        if composite is None:
            composite, filled = rgb_i, valid_i
        else:
            gap = (~filled) & valid_i
            composite[gap] = rgb_i[gap]
            filled = filled | valid_i
        if filled.mean() > 0.98:  # window essentially covered — stop reading
            break
    if composite is None:
        return None, bbox, primary.id, str(primary.datetime)

    valid = composite.sum(axis=2) > 0
    if valid.any():
        lo, hi = np.nanpercentile(composite[valid], [2, 98])
        composite = np.clip((composite - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    composite[~valid] = np.nan  # let no-data render transparent, not black
    return composite, bbox, primary.id, str(primary.datetime)


def least_cloudy_per_period(items: list, periods: list[tuple[str, str]]) -> list:
    """Pick the least-cloudy item inside each (start, end) window.

    Used to choose the T1 (cutoff-epoch) and T2 (recent) chip dates in
    notebook 04. Dry-season windows are passed in, because in the Amazon a
    'least cloudy June-September scene' is usually genuinely clear, while an
    annual minimum can still be 40% cloud.
    """
    import pandas as pd

    chosen = []
    for start, end in periods:
        cands = [it for it in items
                 if pd.Timestamp(start) <= pd.Timestamp(str(it.datetime)).tz_localize(None) <= pd.Timestamp(end)]
        if not cands:
            chosen.append(None)
            continue
        chosen.append(min(cands, key=lambda it: it.properties.get("eo:cloud_cover", 100.0)))
    return chosen
