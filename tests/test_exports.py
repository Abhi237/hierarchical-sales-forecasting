"""Tableau export contract: files, columns, keys, labelling of assumptions, size, text hygiene."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from hier_forecast.config import rpath

EXPECTED = {
    "hierarchy_nodes.csv": ["node_id", "node_label", "level_id", "level_name", "layer", "total_revenue_usd", "share_of_total_revenue_pct"],
    "actuals_weekly.csv": ["node_id", "level_id", "week_idx", "week_start", "revenue_usd", "units"],
    "calendar_weeks.csv": ["week_idx", "week_start", "week_end", "event_names", "role"],
    "actuals_daily_recent.csv": ["level_id", "node_id", "date", "week_idx", "units", "revenue", "node_label"],
    "forecast_vs_actual.csv": ["model", "model_label", "level_id", "node_id", "origin_week_idx", "horizon_weeks", "target_week_start", "forecast_usd",
                               "lo80_usd", "hi80_usd", "lo95_usd", "hi95_usd", "actual_usd", "error_usd", "in_80_interval", "in_95_interval"],
    "metric_scorecard.csv": ["layer", "model", "model_label", "model_family", "level_id", "level_name", "split", "metric", "value", "ci_lo", "ci_hi"],
    "reconciliation_comparison.csv": ["engine", "structure", "base_model", "method", "model", "level_id", "split", "metric", "value"],
    "coherence_gap.csv": ["model", "level_id", "mean_abs_rel_gap"],
    "interval_coverage.csv": ["model", "level_id", "split", "nominal", "coverage", "miscalibration", "mean_width_over_revenue"],
    "interval_coverage_by_horizon.csv": ["model", "level_id", "horizon_weeks", "coverage_80", "coverage_95"],
    "metrics_by_origin.csv": ["model", "level_id", "split", "origin_week_idx", "WAPE", "MASE"],
    "decision_planning.csv": ["model", "node_id", "actual_revenue_usd", "plan_at_q90_usd", "ASSUMED_cost_point_plan_usd", "ASSUMED_plan_quantile"],
    "decision_sensitivity.csv": ["model", "level_id", "plan_rule", "ASSUMED_gross_margin", "ASSUMED_carrying_cost", "ASSUMED_cost_pct_of_revenue"],
    "origins_design.csv": ["origin_week_idx", "split", "train_through", "forecast_first_week_start"],
    "cleaning_log.csv": ["step", "metric", "value", "note"],
}


def read(cfg, name):
    return pd.read_csv(rpath(cfg, "tableau", "data", name))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_file_has_documented_columns_and_rows(built, name):
    df = read(built, name)
    assert len(df) > 0
    missing = set(EXPECTED[name]) - set(df.columns)
    assert not missing, missing


def test_no_duplicate_keys_in_forecast_vs_actual(built):
    f = read(built, "forecast_vs_actual.csv")
    assert not f.duplicated(["model", "node_id", "origin_week_idx", "horizon_weeks"]).any()


def test_forecast_vs_actual_only_has_test_origins(built):
    f = read(built, "forecast_vs_actual.csv")
    d = read(built, "origins_design.csv")
    test = set(d[d["split"].str.startswith("test")]["origin_week_idx"])
    assert set(f["origin_week_idx"]) == test


def test_forecast_vs_actual_actuals_match_weekly_actuals(built):
    f = read(built, "forecast_vs_actual.csv")
    a = read(built, "actuals_weekly.csv")
    m = f.merge(a, left_on=["node_id", "target_week_idx"], right_on=["node_id", "week_idx"])
    assert len(m) == len(f)
    assert np.allclose(m["actual_usd"], m["revenue_usd"], atol=0.01)


def test_interval_flags_are_consistent_with_bounds(built):
    f = read(built, "forecast_vs_actual.csv")
    ok80 = ((f["actual_usd"] >= f["lo80_usd"]) & (f["actual_usd"] <= f["hi80_usd"])).astype(int)
    assert (ok80 == f["in_80_interval"]).mean() > 0.999    # rounding to cents can flip a boundary case


def test_hierarchy_nodes_cover_all_levels(built):
    h = read(built, "hierarchy_nodes.csv")
    assert set(h["level_id"]) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}
    assert h["node_id"].is_unique
    top = h[h["level_id"] == 1]
    assert top["share_of_total_revenue_pct"].iloc[0] == pytest.approx(100.0)
    for lvl in range(2, 10):
        assert h[h["level_id"] == lvl]["share_of_total_revenue_pct"].sum() == pytest.approx(100.0, abs=1e-6)


def test_actuals_weekly_children_sum_to_total(built):
    a = read(built, "actuals_weekly.csv")
    tot = a[a["node_id"] == "T"].set_index("week_idx")["revenue_usd"]
    for lvl in (2, 3, 9):
        s = a[a["level_id"] == lvl].groupby("week_idx")["revenue_usd"].sum()
        assert np.allclose(s, tot, atol=0.05)


def test_assumed_parameters_are_labelled_and_match_config(built):
    d = read(built, "decision_planning.csv")
    assumed = [c for c in d.columns if c.startswith("ASSUMED")]
    assert len(assumed) >= 4
    assert (d["ASSUMED_gross_margin_on_unmet_revenue"] == built["decision"]["assumed_gross_margin"]).all()
    s = read(built, "decision_sensitivity.csv")
    assert s["is_default_assumption"].sum() > 0
    assert set(s["plan_rule"]) == {"point forecast (q50)", "upper 80% bound (q90)", "upper 95% bound (q97.5)"}


def test_decision_costs_are_nonnegative_and_buffer_positive(built):
    d = read(built, "decision_planning.csv")
    assert (d["ASSUMED_cost_point_plan_usd"] >= 0).all() and (d["ASSUMED_cost_q90_plan_usd"] >= 0).all()
    assert (d["safety_buffer_usd"] >= -1e-6).all()
    assert d["share_weeks_actual_within_q90"].between(0, 1).all()


def test_scorecard_is_tidy_long_format(built):
    s = read(built, "metric_scorecard.csv")
    assert not s.duplicated(["layer", "model", "level_id", "split", "metric"]).any()
    assert s["value"].notna().mean() > 0.8


def test_interval_coverage_miscalibration_definition(built):
    c = read(built, "interval_coverage.csv")
    assert np.allclose(c["miscalibration"], c["coverage"] - c["nominal"] / 100, atol=1e-4)
    assert set(c["nominal"]) == {80, 95}


def test_tableau_csvs_fit_public_limits(built):
    import pathlib
    total = sum(p.stat().st_size for p in pathlib.Path(rpath(built, "tableau", "data")).glob("*.csv"))
    assert total < 100e6
    assert all(p.stat().st_size < 25e6 for p in pathlib.Path(rpath(built, "tableau", "data")).glob("*.csv"))


def test_exports_have_no_emoji_or_em_dash(built):
    import pathlib
    bad = re.compile("[\u2014\U0001F300-\U0001FAFF\u2600-\u27BF]")
    for p in pathlib.Path(rpath(built, "tableau", "data")).glob("*.csv"):
        assert not bad.search(p.read_text(encoding="utf-8")), p.name


def test_figures_render(built):
    from hier_forecast.plots import make_figures
    done = make_figures(built)
    assert "wape_by_level.png" in done and "backtest_design.png" in done
    for f in done:
        assert (rpath(built, "docs", "figures", f)).stat().st_size > 2000
