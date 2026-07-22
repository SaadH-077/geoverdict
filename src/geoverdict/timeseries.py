"""Per-plot spectral time series: compositing, gap handling, breakpoint detection.

WHY A STATISTICS ARM AT ALL. The job here is "did this plot lose its forest
after 2020-12-31, and when?". Before any neural network gets an opinion, a
robust-statistics detector answers the same question from the NDVI/NBR time
series directly. Three reasons this arm is not a strawman:
  1. It is interpretable end-to-end: the evidence bundle can show an auditor
     the exact series and the exact date the signal broke.
  2. It sets the floor. If deep learning cannot beat median-and-MAD
     thresholding, that is a finding worth reporting, not hiding.
  3. It is what production systems actually run first — cheap screening over
     millions of plots, with expensive models reserved for the ambiguous rest.

THE DETECTOR (design choices, in order):
  * Baseline = median of the pre-cutoff period; spread = MAD. Median/MAD
    instead of mean/std because undetected cloud and smoke leave heavy-tailed
    outliers, and a mean-based baseline is dragged by exactly the artefacts
    the detector must ignore.
  * A break = the FIRST run of >= k consecutive valid observations below
    baseline - drop. The persistence rule is the anti-false-positive device:
    a single cloudy or burned-haze month dips once; a clearing stays down.
    k trades detection delay against false alarms, and notebook 06 ablates it.
  * Gaps stay gaps. Interpolating through a 4-month wet-season hole invents
    observations exactly where the tropics have none; the detector instead
    counts consecutive VALID points and reports observation density so the
    verdict can say "insufficient evidence" honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def normalized_difference(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (a - b) / (a + b + eps)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """(B08-B04)/(B08+B04): chlorophyll absorbs red, leaf structure reflects NIR."""
    return normalized_difference(nir, red)


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """(B08-B12)/(B08+B12) — the burn/clearing index.

    NBR reacts harder than NDVI to clearing because B12 (2.2 um SWIR) responds
    to dryness and exposed soil/ash: a clear-cut both loses NIR (canopy) and
    gains SWIR (dry debris), so the two shifts add up in the ratio. This is
    why the pipeline tracks both: NDVI for general greenness, NBR for the
    clearing signature specifically.
    """
    return normalized_difference(nir, swir2)


def monthly_series(
    dates: pd.DatetimeIndex | list,
    values: np.ndarray,
    valid: np.ndarray | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Collapse irregular scene-level observations to a regular monthly median.

    WHY MONTHLY MEDIAN. Sentinel-2 revisits every ~5 days but the tropics
    cloud most of them out; the per-month *median of valid observations* is
    the standard compromise: regular spacing for the detector, median for
    robustness to residual cloud the SCL mask missed. Months with no valid
    observation become NaN — visible gaps, not invented values.
    """
    s = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(dates))
    if valid is not None:
        s = s[np.asarray(valid, dtype=bool)]
    s = s.dropna()
    out = s.resample("MS").median()
    if start or end:
        idx = pd.date_range(start or out.index.min(), end or out.index.max(), freq="MS")
        out = out.reindex(idx)
    return out


@dataclass
class BreakResult:
    detected: bool
    break_date: str | None          # first month of the confirmed run
    magnitude: float                # baseline_median - run median (index units)
    baseline_median: float
    baseline_mad: float
    n_confirm: int                  # valid obs in the confirming run
    obs_density: float              # fraction of monitoring months with a valid obs
    threshold: float                # the absolute level that had to be crossed

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def detect_break(
    series: pd.Series,
    cutoff: str,
    k_persist: int = 3,
    drop_mads: float = 6.0,
    min_drop_abs: float = 0.12,
    min_baseline_obs: int = 5,
) -> BreakResult:
    """First sustained post-cutoff drop below the plot's own pre-cutoff baseline.

    PARAMETER REASONING (each is ablated in notebook 06):
      k_persist=3     — three consecutive valid months below threshold. One
                        month = could be smoke/cloud; two = drought is possible;
                        three sustained months of a collapsed index over a plot
                        that used to be forest is clearing. Cost: up to ~3
                        months detection delay, irrelevant for annual compliance.
      drop_mads=6     — 6 scaled-MADs is far outside natural phenology for
                        evergreen tropical forest (NDVI seasonal swing there is
                        small); deciduous plots produce a larger MAD and thus
                        automatically demand a larger absolute drop. The
                        threshold ADAPTS per plot instead of being global.
      min_drop_abs    — floor for near-constant baselines where MAD ~ 0 and
                        any noise would count as 6 MADs.
      min_baseline_obs — below ~5 pre-cutoff observations the baseline itself
                        is unreliable; the result then says "insufficient",
                        never "no change". Absence of evidence, stated as such.
    """
    series = series.sort_index()
    cutoff_ts = pd.Timestamp(cutoff)
    base = series[series.index <= cutoff_ts].dropna()
    post = series[series.index > cutoff_ts]

    n_months = max(len(post), 1)
    obs_density = float(post.notna().sum() / n_months)

    if len(base) < min_baseline_obs:
        return BreakResult(False, None, 0.0, float("nan"), float("nan"), 0, obs_density, float("nan"))

    med = float(base.median())
    mad = float(1.4826 * (base - med).abs().median())  # 1.4826: MAD -> sigma for a normal
    drop = max(drop_mads * mad, min_drop_abs)
    threshold = med - drop

    run: list[pd.Timestamp] = []
    for ts, v in post.items():
        if np.isnan(v):
            continue  # a gap neither confirms nor breaks a run: no observation
        if v < threshold:
            run.append(ts)
            if len(run) >= k_persist:
                run_vals = post.loc[run].astype(float)
                return BreakResult(
                    detected=True,
                    break_date=str(run[0].date()),
                    magnitude=float(med - run_vals.median()),
                    baseline_median=med,
                    baseline_mad=mad,
                    n_confirm=len(run),
                    obs_density=obs_density,
                    threshold=threshold,
                )
        else:
            run = []
    return BreakResult(False, None, 0.0, med, mad, len(run), obs_density, threshold)


def temporal_features(series: pd.Series, cutoff: str) -> dict[str, float]:
    """Hand-crafted features for the classical-ML arm (random forest).

    The feature set encodes the same physics the detector uses — baseline
    level, post-cutoff minimum, the size and abruptness of the change — plus
    shape statistics a single threshold cannot express (variance change,
    trend). Deliberately ~a dozen features, not hundreds: the RF arm exists
    to test whether LEARNING the decision boundary beats HAND-SETTING it,
    on the same information.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    base = series[series.index <= cutoff_ts].dropna()
    post = series[series.index > cutoff_ts].dropna()
    if len(base) == 0 or len(post) == 0:
        return {}

    def slope(s: pd.Series) -> float:
        if len(s) < 3:
            return 0.0
        x = (s.index - s.index[0]).days.to_numpy(dtype=float) / 365.25
        return float(np.polyfit(x, s.to_numpy(dtype=float), 1)[0])

    return {
        "base_median": float(base.median()),
        "base_mad": float((base - base.median()).abs().median()),
        "base_min": float(base.min()),
        "post_median": float(post.median()),
        "post_min": float(post.min()),
        "post_p10": float(post.quantile(0.10)),
        "delta_median": float(base.median() - post.median()),
        "delta_min": float(base.median() - post.min()),
        "post_slope": slope(post),
        "post_std": float(post.std(ddof=0)),
        "obs_density_post": float(len(post) / max((series.index > cutoff_ts).sum(), 1)),
        "n_base_obs": float(len(base)),
    }
