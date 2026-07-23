"""The evidence bundle and DDS report — structure and honesty checks."""

from geoverdict import evidence, risk


def _verdict(pid, tier, reasons):
    v = risk.Verdict(plot_id=pid, tier=tier)
    v.reasons = reasons
    return v


def sample_verdicts():
    return (
        [_verdict(str(i), "LOW", ["forest at cutoff, monitored throughout"]) for i in range(6)]
        + [_verdict(str(10 + i), "MEDIUM", ["baselines disagree"]) for i in range(3)]
        + [_verdict(str(20 + i), "INSUFFICIENT_EVIDENCE", ["not screened for post-cutoff change"]) for i in range(2)]
        + [_verdict("99", "HIGH", ["sustained spectral breakpoint at 2022-07-01"])]
    )


class TestDDSReport:
    def test_report_has_all_sections(self):
        r = evidence.format_dds_report(sample_verdicts())
        assert "# EUDR Due Diligence" in r
        assert "Risk overview" in r
        assert "Parcels requiring attention" in r
        assert "31 December 2020" in r  # the cutoff must be stated

    def test_counts_and_workload_present(self):
        vs = sample_verdicts()
        r = evidence.format_dds_report(vs)
        assert "12 land parcels" in r          # 6+3+2+1
        assert "analyst-hours" in r            # the workload translation

    def test_attention_table_excludes_low_and_ranks_high_first(self):
        r = evidence.format_dds_report(sample_verdicts())
        table = r.split("Parcels requiring attention")[1]
        assert "| 99 | HIGH" in table          # the HIGH plot is listed
        # LOW plots (ids 0..5) must not appear as their own attention rows
        for low_id in ("| 0 |", "| 1 |"):
            assert low_id not in table

    def test_areas_render_when_provided(self):
        vs = sample_verdicts()
        r = evidence.format_dds_report(vs, areas={"99": 42.5})
        assert "42.5" in r
        assert "ha total" in r


class TestPortfolioSummary:
    def test_attention_lists_only_non_low(self):
        s = evidence.portfolio_summary(sample_verdicts())
        assert s["n_plots"] == 12
        assert all(item["tier"] != "LOW" for item in s["attention_required"])
        assert s["tier_counts"]["LOW"] == 6
