"""Breakpoint detector on constructed series where the truth is known."""

import numpy as np
import pandas as pd
import pytest

from geoverdict import timeseries as ts

CUTOFF = "2020-12-31"


def make_series(values: dict[str, float]) -> pd.Series:
    return pd.Series({pd.Timestamp(k): v for k, v in values.items()}).sort_index()


def forest_months(start="2019-01-01", end="2020-12-01", level=0.85, noise=0.02, seed=0):
    idx = pd.date_range(start, end, freq="MS")
    rng = np.random.default_rng(seed)
    return pd.Series(level + rng.normal(0, noise, len(idx)), index=idx)


class TestIndices:
    def test_ndvi_dense_forest_high(self):
        assert ts.ndvi(np.array([0.35]), np.array([0.03]))[0] > 0.8

    def test_nbr_reacts_harder_than_ndvi_to_clearing(self):
        # forest -> cleared: NIR drops, red rises a little, SWIR2 rises a lot
        ndvi_before = ts.ndvi(np.array([0.35]), np.array([0.03]))[0]
        ndvi_after = ts.ndvi(np.array([0.20]), np.array([0.10]))[0]
        nbr_before = ts.nbr(np.array([0.35]), np.array([0.08]))[0]
        nbr_after = ts.nbr(np.array([0.20]), np.array([0.25]))[0]
        assert (nbr_before - nbr_after) > (ndvi_before - ndvi_after)


class TestMonthly:
    def test_median_and_gaps(self):
        dates = ["2021-01-03", "2021-01-18", "2021-03-10"]
        s = ts.monthly_series(dates, np.array([0.8, 0.6, 0.5]),
                              start="2021-01-01", end="2021-03-01")
        assert s.iloc[0] == pytest.approx(0.7)   # median of the two January obs
        assert np.isnan(s.iloc[1])               # February: a gap, not an invention
        assert s.iloc[2] == pytest.approx(0.5)

    def test_invalid_observations_excluded(self):
        dates = ["2021-01-03", "2021-01-18"]
        s = ts.monthly_series(dates, np.array([0.8, -0.5]), valid=np.array([True, False]))
        assert s.iloc[0] == pytest.approx(0.8)


class TestDetector:
    def test_clearing_detected_at_right_month(self):
        base = forest_months()
        post_idx = pd.date_range("2021-01-01", "2022-12-01", freq="MS")
        post_vals = [0.85] * 6 + [0.25] * 18       # clearing in July 2021
        s = pd.concat([base, pd.Series(post_vals, index=post_idx)])
        r = ts.detect_break(s, CUTOFF)
        assert r.detected
        assert r.break_date == "2021-07-01"
        assert r.magnitude > 0.4

    def test_single_dip_is_not_a_clearing(self):
        base = forest_months()
        post_idx = pd.date_range("2021-01-01", "2021-12-01", freq="MS")
        post_vals = [0.85, 0.85, 0.15, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
        s = pd.concat([base, pd.Series(post_vals, index=post_idx)])
        assert not ts.detect_break(s, CUTOFF).detected  # persistence rule holds

    def test_gaps_do_not_break_a_run(self):
        base = forest_months()
        post_idx = pd.date_range("2021-01-01", "2021-12-01", freq="MS")
        post_vals = [0.85, 0.25, np.nan, 0.25, np.nan, 0.25, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
        s = pd.concat([base, pd.Series(post_vals, index=post_idx)])
        r = ts.detect_break(s, CUTOFF)
        assert r.detected and r.break_date == "2021-02-01"

    def test_stable_forest_stays_clean(self):
        idx = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
        rng = np.random.default_rng(1)
        s = pd.Series(0.85 + rng.normal(0, 0.03, len(idx)), index=idx)
        assert not ts.detect_break(s, CUTOFF).detected

    def test_thin_baseline_reports_insufficient_not_no_change(self):
        idx = pd.date_range("2020-10-01", "2022-12-01", freq="MS")
        s = pd.Series(0.85, index=idx)
        r = ts.detect_break(s, CUTOFF)
        assert not r.detected and np.isnan(r.baseline_median)

    def test_threshold_adapts_to_seasonal_plots(self):
        # deciduous-like plot with a big seasonal swing must NOT trigger
        idx = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
        seasonal = 0.6 + 0.2 * np.sin(np.arange(len(idx)) * 2 * np.pi / 12)
        r = ts.detect_break(pd.Series(seasonal, index=idx), CUTOFF)
        assert not r.detected


class TestFeatures:
    def test_features_present_and_signed(self):
        base = forest_months()
        post_idx = pd.date_range("2021-01-01", "2022-12-01", freq="MS")
        s = pd.concat([base, pd.Series([0.3] * len(post_idx), index=post_idx)])
        f = ts.temporal_features(s, CUTOFF)
        assert f["delta_median"] > 0.4
        assert set(f) >= {"base_median", "post_min", "delta_min", "post_slope"}
