"""GeoVerdict — from messy plot geometries to audit-ready EUDR verdicts.

A geospatial ML pipeline prototype: validate and repair supplier plot
geometries, establish each plot's forest status at the EUDR cutoff
(2020-12-31), screen post-cutoff Sentinel-2 time series for clearing with a
statistics arm and a learned arm, fuse the evidence into a risk tier with
stated reasons, and emit an auditor-readable evidence bundle per plot.

Modules (in pipeline order):
    geometry    validation taxonomy + measured repair
    corrupt     seeded corruption harness (ground truth for repair quality)
    gee         forest baselines / reference products via Earth Engine
    s2          Sentinel-2 STAC search + per-plot windowed COG reads
    timeseries  monthly compositing + breakpoint detection (statistics arm)
    models      siamese change CNN (learned arm)
    metrics     PR-centric + plot-normalised metrics, calibration, business units
    risk        transparent evidence fusion -> verdict tiers
    evidence    per-plot audit bundles + portfolio roll-up
    viz         one style for every figure
"""

__version__ = "0.1.0"
