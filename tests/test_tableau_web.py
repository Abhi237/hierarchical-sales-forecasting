"""tableau/web/*.csv: regenerate from the committed tableau/data exports, check structure, no NaN in key columns, and tie-out to the sources."""
from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "tableau" / "data"

_spec = importlib.util.spec_from_file_location("build_tableau_web", ROOT / "scripts" / "build_tableau_web.py")
btw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(btw)

KEYS = {
    "scorecard.csv": ["model_label", "level_order", "level_name", "wape_pct", "wape_ci_lo", "wape_ci_hi", "wape_ratio_vs_snaive", "rank_in_level", "is_best_in_level"],
    "forecast_explorer.csv": ["node_label", "level_name", "model_label", "target_week_start", "horizon_weeks", "forecast_usd", "actual_usd", "lo80_usd", "hi80_usd",
                              "lo95_usd", "hi95_usd", "in_80", "in_95"],
    "reconciliation.csv": ["base_model", "method", "method_order", "level_order", "wape_pct", "wape_ratio_vs_base", "wape_ratio_ci_lo", "wape_ratio_ci_hi"],
    "calibration.csv": ["model_label", "level_order", "nominal_pct", "coverage", "coverage_ci_lo", "coverage_ci_hi", "nominal_coverage", "miscalibration",
                        "mean_width_over_revenue"],
    "planning_buffer.csv": ["model_label", "level_order", "buffer_pct_of_revenue", "realised_share_within_q90", "ASSUMED_cost_reduction_pct"],
}


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("web")
    btw.build_all(d)
    return d


def read(out, name):
    return pd.read_csv(out / name)


def test_all_five_files_are_written(out):
    assert sorted(p.name for p in out.glob("*.csv")) == sorted(KEYS)


@pytest.mark.parametrize("name", sorted(KEYS))
def test_key_columns_have_no_nan_and_no_index_column(out, name):
    d = read(out, name)
    assert len(d) > 0
    assert not any(c.startswith("Unnamed") for c in d.columns)
    assert d[KEYS[name]].notna().all().all()
    assert not d.duplicated().any()


@pytest.mark.parametrize("name", sorted(KEYS))
def test_files_are_utf8_comma_separated_and_small(out, name):
    raw = (out / name).read_bytes()
    raw.decode("utf-8")
    assert raw.split(b"\n")[0].count(b",") >= 4
    assert len(raw) < 15e6


def test_committed_files_match_regeneration(out):
    for name in KEYS:
        committed = ROOT / "tableau" / "web" / name
        assert committed.exists(), name
        assert committed.read_bytes() == (out / name).read_bytes(), f"{name} is stale: run python scripts/build_tableau_web.py"


def test_scorecard_ties_out_to_metric_scorecard(out):
    d = read(out, "scorecard.csv")
    s = pd.read_csv(SRC / "metric_scorecard.csv")
    s = s[(s["split"] == "test") & (s["metric"] == "WAPE") & s["layer"].str.startswith("hier")].set_index(["model", "level_id"])
    for r in d.sample(25, random_state=1).itertuples():
        src = s.loc[(r.model, r.level_order)]
        assert r.wape_pct == pytest.approx(src["value"], abs=1e-6)
        assert r.wape_ci_lo == pytest.approx(src["ci_lo"], abs=1e-6) and r.wape_ci_hi == pytest.approx(src["ci_hi"], abs=1e-6)


def test_scorecard_design(out):
    d = read(out, "scorecard.csv")
    assert set(d["level_order"]) == set(range(1, 10))
    assert not d.duplicated(["model_label", "level_order"]).any()
    assert not d["model"].str.startswith(("py_", "geo_")).any() and "lgbm_conformal" not in set(d["model"])
    assert (d.groupby("level_order")["is_best_in_level"].apply(lambda s: (s == "Yes").sum()) >= 1).all()
    best = d[d["is_best_in_level"] == "Yes"]
    assert (best["rank_in_level"] == 1).all()
    assert (best["wape_pct"] == d.groupby("level_order")["wape_pct"].min().reindex(best["level_order"]).to_numpy()).all()
    assert set(d.loc[d["model"] == "snaive", "better_than_snaive"]) == {"No"}          # ratio is exactly 1 for the reference itself
    assert ((d["wape_ratio_vs_snaive"] < 1) == (d["better_than_snaive"] == "Yes")).all()
    assert set(d.loc[d["is_baseline"] == "Yes", "model"]) == {"naive", "snaive", "ma4"}
    assert (d["wape_ci_lo"] <= d["wape_pct"]).all() and (d["wape_pct"] <= d["wape_ci_hi"]).all()


def test_forecast_explorer_ties_out_to_forecast_vs_actual(out):
    e = read(out, "forecast_explorer.csv")
    f = pd.read_csv(SRC / "forecast_vs_actual.csv")
    assert set(e["level_order"]) == {1, 2, 3} and set(e["horizon_weeks"]) == {1, 4}
    assert e["model_label"].nunique() == 6 and e["target_week_start"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()
    o = pd.read_csv(SRC / "origins_design.csv")
    assert set(e["origin_week_idx"]) <= set(o.loc[o["split"].str.startswith("test"), "origin_week_idx"])
    key = ["model_label", "node_id", "origin_week_idx", "horizon_weeks"]
    m = e.merge(f, on=key, suffixes=("", "_src"), validate="1:1")
    assert len(m) == len(e)
    for c in ["forecast_usd", "actual_usd", "lo80_usd", "hi80_usd", "lo95_usd", "hi95_usd"]:
        assert (m[c] - m[f"{c}_src"]).abs().max() < 1e-4
    assert (m["in_80"] == m["in_80_interval"]).all() and (m["in_95"] == m["in_95_interval"]).all()
    assert (e["lo95_usd"] <= e["lo80_usd"]).all() and (e["hi80_usd"] <= e["hi95_usd"]).all()


def test_reconciliation_ties_out(out):
    d = read(out, "reconciliation.csv")
    r = pd.read_csv(SRC / "reconciliation_comparison.csv")
    r = r[(r["split"] == "test") & (r["engine"] == "R fable") & (r["structure"] == "full 154 nodes") & (r["metric"] == "WAPE_ratio_vs_base")]
    assert set(d["base_model"]) == {"ARIMA", "ETS"} and len(d) == 2 * 6 * 9
    for x in d.sample(15, random_state=2).itertuples():
        s = r[(r["base_model"] == f"R {x.base_model}") & (r["method"] == x.method) & (r["level_id"] == x.level_order)]
        assert len(s) == 1 and x.wape_ratio_vs_base == pytest.approx(s["value"].iloc[0], abs=1e-6)
    b = d[d["method"] == "base (unreconciled)"]
    assert (b["wape_ratio_vs_base"] == 1).all() and (b["method_order"] == 1).all()


def test_calibration_ties_out(out):
    d = read(out, "calibration.csv")
    c = pd.read_csv(SRC / "interval_coverage.csv")
    c = c[(c["split"] == "test") & c["layer"].str.startswith("hier")].set_index(["model", "level_id", "nominal"])
    assert set(d["nominal_pct"]) == {80, 95} and set(d["level_order"]) == set(range(1, 10))
    for x in d.sample(30, random_state=3).itertuples():
        s = c.loc[(x.model, x.level_order, x.nominal_pct)]
        assert x.coverage == pytest.approx(s["coverage"], abs=1e-6) and x.mean_width_over_revenue == pytest.approx(s["mean_width_over_revenue"], abs=1e-6)
    assert (d["nominal_coverage"] == d["nominal_pct"] / 100).all()
    assert (d["miscalibration"] - (d["coverage"] - d["nominal_coverage"])).abs().max() < 2e-5


def test_planning_buffer_ties_out(out):
    d = read(out, "planning_buffer.csv")
    p = pd.read_csv(SRC / "decision_planning.csv")
    assert set(d["level_order"]) == set(range(1, 10)) and not d.duplicated(["model", "level_order"]).any()
    t = p[(p["model"] == "r_arima_mint") & (p["level_id"] == 9)]
    x = d[(d["model"] == "r_arima_mint") & (d["level_order"] == 9)].iloc[0]
    assert x["buffer_pct_of_revenue"] == pytest.approx(t["safety_buffer_usd"].sum() / t["actual_revenue_usd"].sum(), abs=1e-6)
    assert x["realised_share_within_q90"] == pytest.approx(t["share_weeks_actual_within_q90"].mean(), abs=1e-6)
    assert x["ASSUMED_cost_reduction_pct"] == pytest.approx(1 - t["ASSUMED_cost_q90_plan_usd"].sum() / t["ASSUMED_cost_point_plan_usd"].sum(), abs=1e-6)
    assert (d["ASSUMED_gross_margin"] == 0.25).all() and (d["ASSUMED_carrying_cost"] == 0.05).all() and (d["ASSUMED_plan_quantile"] == 0.9).all()
    assert all(c.startswith("ASSUMED_") for c in d.columns if "cost" in c.lower() and c != "n_nodes")
