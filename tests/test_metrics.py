"""Metrics: hand-checkable cases, plus the properties the design claims."""

import numpy as np
import pytest

from geoverdict import metrics as M


class TestPRF:
    def test_hand_computed(self):
        y = np.array([1, 1, 0, 0, 1])
        p = np.array([1, 0, 1, 0, 1])
        m = M.prf(y, p)
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["recall"] == pytest.approx(2 / 3)

    def test_all_negative_predictions_score_zero_not_crash(self):
        m = M.prf(np.array([1, 0]), np.array([0, 0]))
        assert m["f1"] == 0.0


class TestPlotNormalised:
    def test_big_plot_cannot_outvote_small_ones(self):
        # one huge plot fully wrong, three small plots fully right
        counts = [
            {"tp": 0, "fp": 90_000, "fn": 0, "valid": 90_000},   # 900 ha of FP
            {"tp": 100, "fp": 0, "fn": 0, "valid": 100},
            {"tp": 100, "fp": 0, "fn": 0, "valid": 100},
            {"tp": 100, "fp": 0, "fn": 0, "valid": 100},
        ]
        norm = M.plot_normalised_prf(counts)
        # pixel pooling would give precision 300/90300 ~ 0.003; normalised: 3/4
        assert norm["precision"] == pytest.approx(0.75)


class TestCalibration:
    def test_perfect_calibration_low_ece(self, ):
        rng = np.random.default_rng(0)
        probs = rng.uniform(0, 1, 20_000)
        labels = (rng.uniform(0, 1, 20_000) < probs).astype(float)
        assert M.expected_calibration_error(probs, labels)["ece"] < 0.02

    def test_temperature_fixes_overconfidence(self):
        rng = np.random.default_rng(0)
        z_true = rng.normal(0, 1.5, 20_000)
        labels = (rng.uniform(0, 1, 20_000) < 1 / (1 + np.exp(-z_true))).astype(float)
        overconfident = z_true * 3.0  # same ranking, inflated confidence
        t = M.fit_temperature(overconfident, labels)
        assert t == pytest.approx(3.0, rel=0.15)
        before = M.expected_calibration_error(1 / (1 + np.exp(-overconfident)), labels)["ece"]
        after = M.expected_calibration_error(1 / (1 + np.exp(-overconfident / t)), labels)["ece"]
        assert after < before / 3


class TestScreening:
    def test_threshold_hits_recall_target(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(0, 1, 5000) < 0.05
        scores = np.where(y, rng.uniform(0.4, 1.0, 5000), rng.uniform(0.0, 0.7, 5000))
        for row in M.screening_table(y, scores):
            assert row["achieved_recall"] >= row["recall_target"] - 0.01

    def test_better_model_flags_fewer_plots(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(0, 1, 5000) < 0.05
        sharp = np.where(y, rng.uniform(0.8, 1.0, 5000), rng.uniform(0.0, 0.3, 5000))
        blunt = np.where(y, rng.uniform(0.4, 1.0, 5000), rng.uniform(0.0, 0.7, 5000))
        f_sharp = M.screening_table(y, sharp, (0.9,))[0]["flags_per_1000"]
        f_blunt = M.screening_table(y, blunt, (0.9,))[0]["flags_per_1000"]
        assert f_sharp < f_blunt
