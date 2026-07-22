"""The validator and repairer, checked against known damage.

The load-bearing test is the round trip at the bottom: for every corruption
class, corrupt a clean plot, validate (must diagnose the expected code),
repair, and require either a high-IoU recovery or an explicit, honest
refusal. That property is the entire contract of notebook 01.
"""

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon

from geoverdict import config as cfg
from geoverdict import corrupt, geometry as G


class TestArea:
    def test_known_area(self, clean_plot):
        # cross-check the spherical shoelace against a planar cos(lat) estimate,
        # which is accurate at this scale
        coords = np.asarray(clean_plot.exterior.coords)
        lat = coords[:, 1].mean()
        mx = coords[:, 0] * 111_320 * np.cos(np.radians(lat))
        my = coords[:, 1] * 111_320
        planar_ha = 0.5 * abs(np.dot(mx, np.roll(my, -1)) - np.dot(my, np.roll(mx, -1))) / 10_000
        assert G.area_ha(clean_plot) == pytest.approx(planar_ha, rel=0.01)

    def test_empty_is_zero(self):
        assert G.area_ha(Polygon()) == 0.0


class TestValidate:
    def test_clean_plot_passes(self, clean_plot):
        assert G.validate_geometry(clean_plot) == []

    def test_none_and_empty(self):
        assert G.validate_geometry(None)[0].code == "EMPTY_GEOMETRY"
        assert G.validate_geometry(Polygon())[0].code == "EMPTY_GEOMETRY"

    def test_point_rules_follow_eudr_article_9(self):
        p = Point(-55.4, -7.2)
        codes_small = {i.code for i in G.validate_geometry(p, declared_area_ha=2.0)}
        codes_large = {i.code for i in G.validate_geometry(p, declared_area_ha=25.0)}
        assert "POINT_GEOMETRY" in codes_small       # allowed <= 4 ha
        assert "POINT_TOO_LARGE" in codes_large      # polygon required > 4 ha

    def test_severities_come_from_taxonomy(self, clean_plot):
        bad = corrupt.corrupt_bowtie(clean_plot, np.random.default_rng(0))
        issues = G.validate_geometry(bad)
        assert any(i.code == "INVALID_RING" and i.severity == G.ERROR for i in issues)


class TestPortfolio:
    def test_duplicate_flagged_once_second_occurrence(self, clean_plot, rng):
        dup = corrupt.corrupt_duplicate(clean_plot, rng)
        issues = G.validate_portfolio([clean_plot, dup])
        assert not any(i.code == "DUPLICATE_PLOT" for i in issues[0])
        assert any(i.code == "DUPLICATE_PLOT" for i in issues[1])

    def test_overlap_flagged_on_both(self, clean_plot):
        shifted = shapely.affinity.translate(clean_plot, xoff=0.001)
        issues = G.validate_portfolio([clean_plot, shifted])
        for plot_issues in issues:
            assert any(i.code in ("OVERLAPPING_PLOTS", "DUPLICATE_PLOT") for i in plot_issues)


class TestRepairRoundTrip:
    """corrupt -> validate -> repair, per failure class."""

    RECOVERABLE_IOU = {
        # classes where the repaired polygon must land back on the original
        "swap_axes": 0.99, "web_mercator": 0.95, "repeated_vertices": 0.999,
        "cw_winding": 0.999, "add_z": 0.999, "bowtie": 0.55,
        # bowtie: make_valid keeps the largest lobe — most of the shape, not all;
        # the point of the threshold is "same field", not "identical ring"
    }
    HONEST_REFUSALS = {"micro", "sliver", "teleport"}  # repair must NOT pretend

    @pytest.mark.parametrize("name", list(RECOVERABLE_IOU) + sorted(HONEST_REFUSALS))
    def test_round_trip(self, clean_plot, name):
        fn, expected_codes = corrupt.CORRUPTIONS[name]
        rng = np.random.default_rng(cfg.SEED)
        damaged = fn(clean_plot, rng)

        issues = G.validate_geometry(damaged)
        codes = {i.code for i in issues}
        assert codes & set(expected_codes), \
            f"{name}: validator missed {expected_codes}, saw {codes}"

        result = G.repair(damaged, issues)
        if name in self.RECOVERABLE_IOU:
            assert result.ok, f"{name}: repair failed: {result.notes} / {result.unresolved}"
            recovered = G.iou(result.geometry, clean_plot)
            assert recovered >= self.RECOVERABLE_IOU[name], f"{name}: IoU {recovered:.3f}"
        else:
            assert not result.ok, f"{name}: repair should refuse, not invent a plot"

    def test_point_collapse_uses_declared_area(self, clean_plot):
        pt = clean_plot.centroid
        issues = G.validate_geometry(pt, declared_area_ha=3.0)
        result = G.repair(pt, issues, declared_area_ha=3.0)
        assert result.ok
        assert G.area_ha(result.geometry) == pytest.approx(3.0, rel=0.05)

    def test_repair_logs_actions(self, clean_plot, rng):
        damaged = corrupt.corrupt_swap_axes(clean_plot, rng)
        result = G.repair(damaged, G.validate_geometry(damaged))
        assert result.actions, "an auditable repair must log what it did"
