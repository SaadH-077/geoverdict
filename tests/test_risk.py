"""The verdict layer: every gate, every tier, reasons always present."""

from geoverdict import risk


def base_kwargs(**over):
    kw = dict(
        plot_id="p1", geometry_ok=True, geometry_warnings=[],
        forest_frac_jrc=0.9, forest_frac_hansen=0.9,
        ts_break_detected=False, ts_break_date=None, ts_obs_density=0.8,
        model_prob=0.05, hansen_loss_post_frac=0.0,
    )
    kw.update(over)
    return kw


class TestGates:
    def test_bad_geometry_is_insufficient_regardless_of_signals(self):
        v = risk.assess_plot(**base_kwargs(geometry_ok=False, ts_break_detected=True,
                                           model_prob=0.99))
        assert v.tier == risk.INSUFFICIENT

    def test_nonforest_at_cutoff_is_low_even_with_change(self):
        # pasture in 2020 that changed in 2022 is NOT deforestation under EUDR
        v = risk.assess_plot(**base_kwargs(forest_frac_jrc=0.02, forest_frac_hansen=0.05,
                                           ts_break_detected=True, ts_break_date="2022-05-01",
                                           model_prob=0.95))
        assert v.tier == risk.LOW

    def test_unobservable_plot_never_gets_low(self):
        v = risk.assess_plot(**base_kwargs(ts_obs_density=0.1))
        assert v.tier == risk.INSUFFICIENT

    def test_unscreened_forest_plot_is_insufficient_not_low(self):
        # forest at cutoff but never time-series screened (ts fields None) and the
        # learned arm did not flag it -> we have not actually checked it
        v = risk.assess_plot(**base_kwargs(ts_break_detected=None, ts_break_date=None,
                                           ts_obs_density=None, model_prob=0.1,
                                           hansen_loss_post_frac=0.0))
        assert v.tier == risk.INSUFFICIENT
        assert any("not screened" in r for r in v.reasons)

    def test_screened_clean_forest_plot_is_still_low(self):
        # ts_break_detected == False means screened and no break -> LOW is honest
        v = risk.assess_plot(**base_kwargs(ts_break_detected=False, ts_obs_density=0.8,
                                           model_prob=0.1, hansen_loss_post_frac=0.0))
        assert v.tier == risk.LOW


class TestFusion:
    def test_agreeing_detectors_give_high(self):
        v = risk.assess_plot(**base_kwargs(ts_break_detected=True,
                                           ts_break_date="2022-07-01", model_prob=0.93))
        assert v.tier == risk.HIGH

    def test_disagreeing_detectors_give_medium(self):
        v = risk.assess_plot(**base_kwargs(ts_break_detected=True,
                                           ts_break_date="2022-07-01", model_prob=0.1))
        assert v.tier == risk.MEDIUM

    def test_baseline_map_disagreement_caps_at_medium(self):
        v = risk.assess_plot(**base_kwargs(forest_frac_jrc=0.85, forest_frac_hansen=0.05))
        assert v.tier == risk.MEDIUM

    def test_clean_forest_plot_is_low(self):
        v = risk.assess_plot(**base_kwargs())
        assert v.tier == risk.LOW


class TestExplainability:
    def test_every_verdict_has_reasons(self):
        for kw in (base_kwargs(), base_kwargs(geometry_ok=False),
                   base_kwargs(ts_break_detected=True, ts_break_date="2022-01-01",
                               model_prob=0.9)):
            v = risk.assess_plot(**kw)
            assert v.reasons, f"tier {v.tier} produced no reasons"

    def test_inputs_are_preserved_for_the_bundle(self):
        v = risk.assess_plot(**base_kwargs(model_prob=0.42))
        assert v.inputs["model_prob"] == 0.42
