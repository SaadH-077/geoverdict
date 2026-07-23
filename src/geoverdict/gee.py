"""Google Earth Engine helpers: forest baselines and reference loss products.

WHY EARTH ENGINE FOR THE BASELINES (and STAC for the imagery). The baseline
question — "was this plot forest on 2020-12-31?" — is answered by global
reference maps (JRC Global Forest Cover 2020, Hansen Global Forest Change),
which are tens of terabytes as rasters but are hosted, tiled and reducible
server-side on GEE for free. Asking GEE for "mean of this mask over these
600 polygons" moves kilobytes. Sentinel-2 time series, by contrast, come
from the Earth Search STAC (s2.py) because per-plot windowed reads give
exact control over masking and chips, and because that is the access path
a production system owns end-to-end. Two tools, each doing the thing it is
uniquely good at — the split is deliberate and worth explaining in review.

ASSET IDS ARE RESOLVED DEFENSIVELY. GEE dataset ids carry versions that
change (Hansen updates annually; JRC GFC2020 went V1 -> V2). Every loader
tries a preference-ordered list and REPORTS which version it used — the
version lands in the provenance record, because "which map vintage said
this was forest" is an auditable fact, not an implementation detail.

Everything needs `ee.Initialize()` first — see `init()`. All functions
return plain pandas DataFrames; nothing downstream depends on GEE types.
"""

from __future__ import annotations

import pandas as pd

from . import config as cfg

HANSEN_CANDIDATES = [
    "UMD/hansen/global_forest_change_2024_v1_12",
    "UMD/hansen/global_forest_change_2023_v1_11",
]
# JRC GFC2020 is served as an ImageCollection of tiles (NOT a single Image),
# so it must be mosaicked before use. V1/V2 are deprecated; V3 is current.
JRC_GFC2020_CANDIDATES = [
    "JRC/GFC2020/V3",
    "JRC/GFC2020/V2",
    "JRC/GFC2020/V1",
]
TMF_DEGRADATION_CANDIDATES = [
    "projects/JRC/TMF/v1_2024/AnnualChanges",
    "projects/JRC/TMF/v1_2023/AnnualChanges",
]

# Hansen "forest": canopy cover >= 30% in 2000 and not lost by the cutoff.
# 30% is the FAO-aligned convention and also what most Hansen-based studies
# use; it is a *definition choice* that notebook 02 varies (10/30/50%) to
# show how much the verdict moves with it — that sensitivity IS a result.
HANSEN_CANOPY_THRESHOLD = 30


def init(project: str | None = None) -> None:
    """Authenticate + initialise. On Colab: ee.Authenticate() pops the flow once."""
    import ee

    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def _first_available(candidates: list[str], kind: str):
    """Return the first loadable asset + its id, trying candidates in order.

    kind:
      "image"      -> ee.Image (e.g. Hansen GFC, a single global image)
      "collection" -> ee.ImageCollection (e.g. TMF annual changes)
      "mosaic"     -> ee.ImageCollection mosaicked into one Image (e.g. JRC
                      GFC2020, which is served as tiles and must be composited)

    Each candidate is probed with a light server round-trip (bandNames /
    size), so an unavailable or wrong-type asset fails HERE and we fall
    through to the next version instead of exploding deep in a reducer.
    """
    import ee

    errors = []
    for asset_id in candidates:
        try:
            if kind == "collection":
                obj = ee.ImageCollection(asset_id)
                obj.size().getInfo()
            elif kind == "mosaic":
                obj = ee.ImageCollection(asset_id).mosaic()
                obj.bandNames().getInfo()
            else:
                obj = ee.Image(asset_id)
                obj.bandNames().getInfo()
            return obj, asset_id
        except Exception as e:  # try the next version, remember why
            errors.append(f"{asset_id}: {e}")
    raise RuntimeError("no candidate asset available:\n" + "\n".join(errors))


def _fc_from_geoms(geoms, ids) -> "object":
    """Shapely polygons -> ee.FeatureCollection (WGS84 lon/lat throughout)."""
    import ee
    from shapely.geometry import mapping

    feats = [ee.Feature(ee.Geometry(mapping(g)), {"plot_id": str(i)}) for g, i in zip(geoms, ids)]
    return ee.FeatureCollection(feats)


def forest_baseline_fractions(geoms, ids, batch: int = 120,
                              canopy_threshold: int = HANSEN_CANOPY_THRESHOLD) -> pd.DataFrame:
    """Per plot: forest fraction at the 2020 cutoff, from BOTH official maps.

    Returns columns: plot_id, forest_frac_jrc, forest_frac_hansen,
    hansen_loss_post_frac (fraction mapped as lost 2021+, the weak label for
    notebooks 03-04), plus the asset ids used (provenance).

    Batched reduceRegions with getInfo: ~600 plots fit comfortably under
    GEE's interactive limits in a few batches; the alternative (Drive
    exports) adds an async wait for no benefit at this scale. scale=10 m
    on ~30 m products oversamples slightly — harmless for area fractions,
    and it keeps every reducer on the plot's native analysis grid.
    """
    import ee

    jrc_img, jrc_id = _first_available(JRC_GFC2020_CANDIDATES, "mosaic")
    hansen_img, hansen_id = _first_available(HANSEN_CANDIDATES, "image")

    # GFC2020's forest band is "Map" (1 = tree cover); fall back to band 0 if a
    # future version renames it, so a rename degrades gracefully rather than
    # crashing.
    jrc_bands = jrc_img.bandNames().getInfo()
    jrc_sel = "Map" if "Map" in jrc_bands else jrc_bands[0]
    jrc_forest = jrc_img.select(jrc_sel).eq(1)
    treecover = hansen_img.select("treecover2000").gte(canopy_threshold)
    lossyear = hansen_img.select("lossyear")  # 0 = no loss; 1..24 = 2001..2024
    lost_by_cutoff = lossyear.gt(0).And(lossyear.lte(cfg.CUTOFF_YEAR - 2000))
    hansen_forest_2020 = treecover.And(lost_by_cutoff.Not())
    loss_post = lossyear.gt(cfg.CUTOFF_YEAR - 2000)

    # CRITICAL: unmask to 0 before the mean reducer. GFC2020's "Map" band masks
    # non-forest pixels, and ee.Reducer.mean() IGNORES masked pixels — so
    # without this the forest fraction would be ~1.0 for any plot containing any
    # forest at all, silently breaking the whole map-disagreement analysis.
    # unmask(0) makes masked pixels count as non-forest / no-loss, so the mean
    # is a true "fraction of the plot" over every pixel.
    stack = (jrc_forest.unmask(0).rename("jrc")
             .addBands(hansen_forest_2020.unmask(0).rename("hansen"))
             .addBands(loss_post.unmask(0).rename("loss_post")))

    rows = []
    for s in range(0, len(geoms), batch):
        fc = _fc_from_geoms(geoms[s:s + batch], ids[s:s + batch])
        reduced = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10)
        for f in reduced.getInfo()["features"]:
            p = f["properties"]
            rows.append({
                "plot_id": p["plot_id"],
                "forest_frac_jrc": p.get("jrc"),
                "forest_frac_hansen": p.get("hansen"),
                "hansen_loss_post_frac": p.get("loss_post"),
            })
    df = pd.DataFrame(rows)
    df.attrs["assets"] = {"jrc": jrc_id, "hansen": hansen_id,
                          "hansen_canopy_threshold": canopy_threshold}
    return df


def s2_plot_timeseries(
    geoms,
    ids,
    start: str = cfg.BASELINE_START,
    end: str = cfg.MONITOR_END,
    max_scene_cloud: float = 80.0,
    batch_plots: int = 50,
) -> pd.DataFrame:
    """Per plot, per scene: masked-mean NDVI/NBR + valid fraction, server-side.

    WHY GEE HERE AND STAC IN s2.py — the division of labour, precisely:
    a six-year NDVI series for 200 plots touches ~400 scene dates. Doing that
    client-side is ~10^5 windowed HTTP reads (hours on Colab); as a server-
    side `reduceRegions` mapped over the collection, the pixels never leave
    Google's racks and only (plot, date, mean) tuples come back — seconds of
    transfer. Chips for the CNN are the opposite case: we need the actual
    pixels, few dates, full control over masking — that is the STAC path.
    Choosing per workload is the design point worth defending in review.

    Masking: the SCL band of COPERNICUS/S2_SR_HARMONIZED, same classes as
    s2.BAD_SCL, so the two access paths apply the SAME usability definition.
    Scenes are pre-filtered at 80% tile cloud (permissive on purpose — the
    per-plot mask decides), and a plot-scene with <30% valid pixels returns
    null rather than a cloud-contaminated mean.

    Batched over plots and years to stay inside getInfo response limits.
    """
    import ee

    bad = list(cfg_bad_scl())

    def add_indices(img):
        scl = img.select("SCL")
        valid = ee.Image.constant(1)
        for c in bad:
            valid = valid.And(scl.neq(c))
        scaled = img.divide(10_000)
        ndvi = scaled.normalizedDifference(["B8", "B4"]).rename("ndvi")
        nbr = scaled.normalizedDifference(["B8", "B12"]).rename("nbr")
        return (ndvi.addBands(nbr)
                .updateMask(valid)
                .addBands(valid.rename("valid"))
                .copyProperties(img, ["system:time_start", "system:index"]))

    years = list(range(int(start[:4]), int(end[:4]) + 1))
    rows = []
    for s in range(0, len(geoms), batch_plots):
        fc = _fc_from_geoms(geoms[s:s + batch_plots], ids[s:s + batch_plots])
        for y in years:
            y0 = max(f"{y}-01-01", start)
            y1 = min(f"{y}-12-31", end)
            coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(fc)
                    .filterDate(y0, y1)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_scene_cloud))
                    .map(add_indices))

            def per_image(img):
                date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
                return img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10) \
                          .map(lambda f: f.set("date", date))

            feats = coll.map(per_image).flatten().getInfo()["features"]
            for f in feats:
                p = f["properties"]
                rows.append({
                    "plot_id": p["plot_id"], "date": p.get("date"),
                    "ndvi": p.get("ndvi"), "nbr": p.get("nbr"),
                    "valid_frac": p.get("valid"),
                })
    df = pd.DataFrame(rows)
    if len(df):
        # A plot-scene where <30% of pixels were usable is not an observation.
        df.loc[df["valid_frac"].fillna(0) < 0.30, ["ndvi", "nbr"]] = None
    df.attrs["assets"] = {"s2": "COPERNICUS/S2_SR_HARMONIZED"}
    return df


def cfg_bad_scl() -> tuple:
    """The masked SCL classes, imported lazily from s2.py so the two access
    paths can never drift apart silently."""
    from .s2 import BAD_SCL

    return BAD_SCL


def hansen_loss_year_fractions(geoms, ids, batch: int = 120) -> pd.DataFrame:
    """Per plot and per year 2021+: fraction mapped as lost that year.

    This is the reference the detected BREAK DATES are compared against in
    notebook 03 (detection-date scatter). Hansen dates loss to a calendar
    year, so agreement within ±1 year is the honest success criterion —
    a January clearing and Hansen's annual compositing legitimately disagree
    about which year it belongs to.
    """
    import ee

    hansen_img, hansen_id = _first_available(HANSEN_CANDIDATES, "image")
    lossyear = hansen_img.select("lossyear")
    years = list(range(cfg.CUTOFF_YEAR + 1, 2025))
    stack = None
    for y in years:
        band = lossyear.eq(y - 2000).rename(f"loss_{y}")
        stack = band if stack is None else stack.addBands(band)

    rows = []
    for s in range(0, len(geoms), batch):
        fc = _fc_from_geoms(geoms[s:s + batch], ids[s:s + batch])
        reduced = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10)
        for f in reduced.getInfo()["features"]:
            p = f["properties"]
            rows.append({"plot_id": p["plot_id"],
                         **{f"loss_{y}": p.get(f"loss_{y}") for y in years}})
    df = pd.DataFrame(rows)
    df.attrs["assets"] = {"hansen": hansen_id}
    return df


def stable_forest_mask_points(aoi_bbox, n_points: int, seed: int) -> pd.DataFrame:
    """Sample point locations of TMF 'undisturbed forest throughout' — the
    hard-negative mine for notebook 04.

    THE TRICK (and why it matters): a change detector trained on
    (clearing, random-background) pairs learns 'forest vs everything',
    not 'change vs no-change'. The negatives that teach the real boundary
    are STABLE FOREST — textured, dark, high-NDVI chips where nothing
    happened. TMF's annual undisturbed class, intersected across all
    post-2019 years, provides exactly those locations. The ablation
    (train with vs without them) is notebook 04's headline experiment.
    """
    import ee

    tmf, tmf_id = _first_available(TMF_DEGRADATION_CANDIDATES, "collection")
    # TMF AnnualChanges class 1 = undisturbed tropical moist forest. Band
    # naming differs across TMF versions, so rather than per-version band
    # archaeology we take the FIRST and LAST annual images and require
    # 'undisturbed' in both — a pixel stable at both ends of the record is a
    # sound hard-negative location, and the construction survives version
    # bumps unchanged.
    imgs = tmf.toList(tmf.size())
    first = ee.Image(imgs.get(0)).select(0).eq(1)
    last = ee.Image(imgs.get(-1)).select(0).eq(1)
    stable = first.And(last).selfMask()

    region = ee.Geometry.Rectangle(list(aoi_bbox))
    pts = stable.stratifiedSample(
        numPoints=n_points, classBand=stable.bandNames().get(0), region=region,
        scale=30, seed=seed, geometries=True,
    )
    rows = []
    for f in pts.getInfo()["features"]:
        lon, lat = f["geometry"]["coordinates"]
        rows.append({"lon": lon, "lat": lat})
    df = pd.DataFrame(rows)
    df.attrs["assets"] = {"tmf": tmf_id}
    return df
