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
    "scorecard.csv": ["model_label", "level_order", "level_name", "wape_pct", "wape_ci_lo", "wape_ci_hi", "wape_ratio_vs_snaive", "rank_in_level", "is_best_in_level", "colour_group", "ma4_beats_ets_in_level"],
    "forecast_explorer.csv": ["node_label", "level_name", "model_label", "target_week_start", "horizon_weeks", "forecast_usd", "actual_usd", "lo80_usd", "hi80_usd",
                              "lo95_usd", "hi95_usd", "in_80", "in_95", "band_width_usd", "series_label", "colour_group"],
    "reconciliation.csv": ["base_model", "method", "method_order", "level_order", "wape_pct", "wape_ratio_vs_base", "wape_ratio_ci_lo", "wape_ratio_ci_hi"],
    "calibration.csv": ["model_label", "level_order", "nominal_pct", "coverage", "coverage_ci_lo", "coverage_ci_hi", "nominal_coverage", "miscalibration",
                        "mean_width_over_revenue", "colour_group", "is_best_interval_in_level", "under_covers"],
    "planning_buffer.csv": ["model_label", "level_order", "buffer_pct_of_revenue", "realised_share_within_q90", "ASSUMED_cost_reduction_pct"],
    "kpis.csv": ["kpi_order", "kpi_label", "kpi_value", "kpi_display", "kpi_note"],
    "findings.csv": ["finding_order", "section", "section_name", "finding_type", "finding_headline", "finding_detail", "is_assumed", "evidence_file"],
}


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("web")
    btw.build_all(d)
    return d


def read(out, name):
    return pd.read_csv(out / name)


def test_all_seven_files_are_written(out):
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


# ---------------------------------------------------------------- story columns, KPIs and findings (independent recomputation from tableau/data)

def _wape(level, split="test"):
    s = pd.read_csv(SRC / "metric_scorecard.csv")
    s = s[(s["split"] == split) & (s["metric"] == "WAPE") & s["layer"].str.startswith("hier") & (s["level_id"] == level)]
    return s.set_index("model")["value"]


def test_colour_groups_follow_the_colour_semantics(out):
    d = read(out, "scorecard.csv")
    assert set(d["colour_group"]) <= {"Baseline", "Statistical", "Machine learning", "Reconciled"}
    m = dict(zip(d["model"], d["colour_group"], strict=True))
    assert m["snaive"] == m["naive"] == m["ma4"] == "Baseline"
    assert m["ets"] == m["sarima_airline"] == m["r_arima_base"] == "Statistical"
    assert m["lgbm_q"] == "Machine learning"
    assert m["r_arima_mint"] == m["r_ets_mint"] == "Reconciled"
    r = read(out, "reconciliation.csv")
    assert set(r.loc[r["method"] == "base (unreconciled)", "colour_group"]) == {"Statistical"}
    assert set(r.loc[r["method"] != "base (unreconciled)", "colour_group"]) == {"Reconciled"}
    c = read(out, "calibration.csv")
    assert dict(zip(c["model"], c["colour_group"], strict=True))["lgbm_conformal"] == "Machine learning"


def test_ma4_beats_ets_flag_matches_sources(out):
    d = read(out, "scorecard.csv")
    for lvl in range(1, 10):
        w = _wape(lvl)
        x = d[d["level_order"] == lvl]
        assert set(x["ma4_beats_ets_in_level"]) == {"Yes" if w["ma4"] < w["ets"] else "No"}
        assert x["ma4_minus_ets_wape"].iloc[0] == pytest.approx(w["ma4"] - w["ets"], abs=2e-5)


def test_calibration_flags(out):
    d = read(out, "calibration.csv")
    assert ((d["coverage"] < d["nominal_coverage"] - 0.05) == (d["under_covers"] == "Yes")).all()
    for (_, _), g in d.groupby(["level_order", "nominal_pct"]):
        best = g.loc[g["is_best_interval_in_level"] == "Yes"]
        assert len(best) >= 1
        assert (best["miscalibration"].abs() <= g["miscalibration"].abs().min() + 1e-5).all()
        assert (g.loc[g["is_best_interval_in_level"] == "No", "miscalibration"].abs() >= best["miscalibration"].abs().min() - 1e-5).all()


def test_explorer_band_width_and_series_label(out):
    e = read(out, "forecast_explorer.csv")
    assert ((e["hi80_usd"] - e["lo80_usd"]) - e["band_width_usd"]).abs().max() < 0.011
    assert (e["band_width_usd"] > 0).all()
    assert (e["series_label"] == e["model_label"] + " | " + e["node_label"]).all()
    assert e["series_label"].nunique() == e["model_label"].nunique() * e["node_id"].nunique()


def test_kpis_recompute_from_sources(out):
    k = read(out, "kpis.csv").set_index("kpi_order")
    w1, snaive1 = _wape(1), _wape(1)["snaive"]
    cand = w1.drop([m for m in w1.index if m.startswith(("py_", "geo_")) or m == "lgbm_conformal"])
    best = cand.idxmin()
    assert k.loc[1, "kpi_value"] == pytest.approx(cand.min(), abs=1e-5) and k.loc[1, "kpi_display"] == f"{100 * cand.min():.1f}%"
    assert k.loc[2, "kpi_value"] == pytest.approx(1 - cand.min() / snaive1, abs=1e-4)   # source ratio is rounded to 5 decimals
    labels = pd.read_csv(SRC / "metric_scorecard.csv").drop_duplicates("model").set_index("model")["model_label"]
    assert labels[best] in k.loc[1, "kpi_note"]
    wins = sum(_wape(lvl).drop([m for m in _wape(lvl).index if m.startswith(("py_", "geo_")) or m == "lgbm_conformal"]).idxmin() == best for lvl in range(1, 10))
    assert k.loc[3, "kpi_value"] == wins and k.loc[3, "kpi_display"] == f"{wins} of 9"
    f = pd.read_csv(SRC / "forecast_vs_actual.csv")
    best3 = labels[_wape(3).drop([m for m in _wape(3).index if m.startswith(("py_", "geo_")) or m == "lgbm_conformal"]).idxmin()]
    st = f[(f["level_id"] == 3) & f["model_label"].isin([best3, "Seasonal naive (52 wk)"])].groupby(["node_id", "model_label"]).apply(
        lambda g: g["abs_error_usd"].sum() / g["actual_usd"].sum(), include_groups=False).unstack()
    n_beat = int((st[best3] < st["Seasonal naive (52 wk)"]).sum())
    assert k.loc[4, "kpi_display"] == f"{n_beat} of {len(st)}" and k.loc[4, "kpi_value"] == pytest.approx(n_beat / len(st))
    c = pd.read_csv(SRC / "interval_coverage.csv")
    c = c[(c["split"] == "test") & (c["layer"].str.startswith("hier")) & (c["level_id"] == 1) & (c["nominal"] == 80)]
    c = c[c["model"].isin(btw.CALIBRATION_MODELS)]
    near = c.loc[(c["coverage"] - 0.8).abs().idxmin()]
    assert k.loc[5, "kpi_value"] == pytest.approx(near["coverage"], abs=1e-5) and near["model_label"] in k.loc[5, "kpi_note"]
    r = pd.read_csv(SRC / "reconciliation_comparison.csv")
    r = r[(r["split"] == "test") & (r["engine"] == "R fable") & (r["structure"] == "full 154 nodes") & (r["metric"] == "WAPE_ratio_vs_base")]
    mint = r[(r["base_model"] == "R ARIMA") & (r["method"] == "MinT shrink") & (r["level_id"] == 1)]["value"].iloc[0]
    assert k.loc[6, "kpi_value"] == pytest.approx(mint, abs=1e-5) and k.loc[6, "kpi_display"] == f"{mint:.3f}"
    o = pd.read_csv(SRC / "origins_design.csv")
    assert k.loc[7, "kpi_value"] == o["split"].str.startswith("test").sum()
    assert (k["kpi_display"].str.len() > 0).all() and k["kpi_value"].notna().all()


def test_findings_numbers_trace_to_data(out):
    d = read(out, "findings.csv")
    assert list(d["finding_order"]) == list(range(1, len(d) + 1))
    assert set(d["section"]) == {2, 3, 4, 5, 6}
    assert (d["section"] == 6).sum() >= 3 and (d["section"] == 6).sum() <= 4
    for sec in (2, 3, 4, 5):
        assert (d["section"] == sec).any()
    text = lambda sec, i=0: (d[d["section"] == sec].iloc[i]["finding_headline"], d[d["section"] == sec].iloc[i]["finding_detail"])  # noqa: E731
    w = {lvl: _wape(lvl) for lvl in range(1, 10)}
    # section 2: moving average beats ETS count and the widest gap
    n_beat = sum(w[lvl]["ma4"] < w[lvl]["ets"] for lvl in w)
    h, det = text(2, 1)
    assert f"beats ETS at {n_beat} of 9 levels" in h
    gap = max(w, key=lambda lvl: w[lvl]["ets"] - w[lvl]["ma4"])
    assert f"{100 * w[gap]['ma4']:.1f}%" in det and f"{100 * w[gap]['ets']:.1f}%" in det
    # section 2 headline: the winner count
    cand = {lvl: x.drop([m for m in x.index if m.startswith(("py_", "geo_")) or m == "lgbm_conformal"]).idxmin() for lvl, x in w.items()}
    top = max(set(cand.values()), key=list(cand.values()).count)
    assert f"at {list(cand.values()).count(top)} of 9 levels" in text(2, 0)[0]
    # section 3: LightGBM Total band
    f = pd.read_csv(SRC / "forecast_vs_actual.csv")
    f = f[(f["level_id"] == 1) & (f["model_label"] == "LightGBM quantile")]
    h, det = text(3)
    assert f"{int(f['in_80_interval'].sum())} of {len(f)} Total forecasts ({100 * f['in_80_interval'].mean():.1f}%)" in h
    assert f"{100 * ((f['hi80_usd'] - f['lo80_usd']) / f['actual_usd']).mean():.1f}%" in det
    # section 4: bottom-up at Total and MinT range
    r = pd.read_csv(SRC / "reconciliation_comparison.csv")
    r = r[(r["split"] == "test") & (r["engine"] == "R fable") & (r["structure"] == "full 154 nodes") & (r["metric"] == "WAPE_ratio_vs_base") & (r["base_model"] == "R ARIMA")]
    bu = r[(r["method"] == "bottom-up") & (r["level_id"] == 1)]["value"].iloc[0]
    mint = r[r["method"] == "MinT shrink"]["value"]
    h, det = text(4)
    assert f"worse by {100 * (bu - 1):.1f}% at Total" in h
    assert f"{100 * (1 - mint.max()):.1f}% to {100 * (1 - mint.min()):.1f}%" in det
    assert (mint < 1).all() and h.startswith("Only MinT shrink")
    # section 5: under-coverage count at 80%
    c = pd.read_csv(SRC / "interval_coverage.csv")
    c = c[(c["split"] == "test") & c["layer"].str.startswith("hier") & (c["nominal"] == 80) & c["model"].isin(btw.CALIBRATION_MODELS)]
    n_under = int((c["coverage"] < 0.8 - 0.05).sum())
    assert f"{n_under} of {len(c)} model and level cases" in text(5)[0]
    # section 6: planning buffer figures for MinT ARIMA come from decision_planning.csv
    p = pd.read_csv(SRC / "decision_planning.csv")
    t = p[(p["model"] == "r_arima_mint") & (p["level_id"] == 1)]
    buf = t["safety_buffer_usd"].sum() / t["actual_revenue_usd"].sum()
    red = 1 - t["ASSUMED_cost_q90_plan_usd"].sum() / t["ASSUMED_cost_point_plan_usd"].sum()
    rec = d[d["section"] == 6]
    assumed = rec[rec["is_assumed"] == "Yes"]
    assert len(assumed) == 1 and assumed.iloc[0]["finding_headline"].startswith("ASSUMED")
    assert f"buffer of {100 * buf:.1f}% of revenue at Total" in assumed.iloc[0]["finding_headline"]
    assert f"cuts the ASSUMED cost by {100 * red:.0f}% at Total" in assumed.iloc[0]["finding_headline"]
    assert "25% margin" in assumed.iloc[0]["finding_detail"] and "5% carrying" in assumed.iloc[0]["finding_detail"] and "q90" in assumed.iloc[0]["finding_detail"]
    share = t["share_weeks_actual_within_q90"].mean()
    assert f"only {100 * share:.0f}% of Total weeks" in " ".join(rec["finding_headline"])
    assert "6 test origins" in " ".join(rec["finding_detail"])
    # cost words and dollar signs appear only in the ASSUMED row
    others = d[d["is_assumed"] == "No"]
    txt = (others["finding_headline"] + " " + others["finding_detail"]).str.lower()
    assert not txt.str.contains("cost|\\$|margin|carrying").any()


def test_story_files_text_hygiene(out):
    bad = __import__("re").compile("[\u2014\u2013\U0001F300-\U0001FAFF\u2600-\u27BF]")
    for name in ("kpis.csv", "findings.csv"):
        txt = (out / name).read_text(encoding="utf-8")
        assert not bad.search(txt), name
        assert len(txt) < 20_000
