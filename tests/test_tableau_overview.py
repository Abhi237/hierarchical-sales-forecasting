"""tableau/web overview CSVs: regenerate, check shape and ranges, tie out to forecast_vs_actual and metrics.py WAPE."""
from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / "tableau" / "web"
DATA = ROOT / "tableau" / "data"
_spec = importlib.util.spec_from_file_location("build_tableau_overview", ROOT / "scripts" / "build_tableau_overview.py")
bto = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bto)

FILES = [
    "overview_kpis.csv", "kpi_strip.csv", "accuracy_by_area.csv", "forecast_plain.csv",
    "business_impact.csv", "forecast_context.csv", "methodology_notes.csv", "action_panel.csv",
    "story_headlines.csv",
]


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("overview")
    bto.build_all(WEB, d, DATA)
    return d


@pytest.mark.parametrize("name", FILES)
def test_committed_files_match_regeneration(out, name):
    pd.testing.assert_frame_equal(pd.read_csv(out / name), pd.read_csv(WEB / name))


def _src():
    f = pd.read_csv(DATA / "forecast_vs_actual.csv")
    return f[f.level_name.isin(["Total", "State", "Store"])]


def test_overview_kpis(out):
    k = pd.read_csv(out / "overview_kpis.csv")
    assert list(k.columns) == ["order", "label", "value_display", "caption"]
    tot = pd.read_csv(WEB / "scorecard.csv").query("level_name == 'Total'").set_index("model_label").wape_pct
    w, w_sn = tot[bto.DEFAULT_MODEL], tot[bto.BASELINE_MODEL]
    assert k.value_display[0] == f"{100 * (1 - w):.1f}%"
    assert k.value_display[1] == f"{100 * (1 - w / w_sn):.1f}%"
    company_acc = pd.read_csv(out / "accuracy_by_area.csv").query("Level == 'Company'")["Accuracy (%)"].iloc[0]
    assert k.value_display[0] == f"{company_acc:.1f}%"
    src = _src().query("model == @bto.DEFAULT_ID and level_name == 'Total' and horizon_weeks == 1")
    assert k.value_display[2] == f"{int(src.in_80_interval.sum())} of {len(src)}"


def test_kpi_strip(out):
    k = pd.read_csv(out / "kpi_strip.csv")
    assert len(k) == 14 * 5
    assert set(k.Level) == {"Company", "State", "Store"}
    assert k["Interval nominal (%)"].eq(80).all()
    for c in [c for c in k.columns if c.startswith("Desc ")]:
        assert k[c].str.len().max() <= 45
    k["wa"] = k["Weeks ahead"].astype(str)
    company = k[(k.Level == "Company") & (k.wa == "All")].iloc[0]
    src = _src().query("model == @bto.DEFAULT_ID and level_name == 'Total'")
    assert abs(company["Actual revenue ($)"] - src.actual_usd.sum()) < 0.02
    assert abs(company["Forecast revenue ($)"] - src.forecast_usd.sum()) < 0.02
    assert company["Revenue variance ($)"] == pytest.approx(
        company["Forecast revenue ($)"] - company["Actual revenue ($)"], abs=0.02)
    assert company["Forecast accuracy (%)"] == pytest.approx(
        100 * (1 - src.abs_error_usd.sum() / src.actual_usd.sum()), abs=0.05)
    assert company["Interval coverage"] == f"{int(src.in_80_interval.sum())} of {len(src)}"
    h1 = k[(k.Level == "Company") & (k.wa == "1")].iloc[0]
    s1 = src[src.horizon_weeks == 1]
    assert int(h1["Weeks total"]) == len(s1) == 6
    assert h1["Interval coverage"] == f"{int(s1.in_80_interval.sum())} of {len(s1)}"


def test_accuracy_by_area(out):
    a = pd.read_csv(out / "accuracy_by_area.csv")
    assert a.Level.value_counts().to_dict() == {"Company": 1, "State": 3, "Store": 10}
    assert {"Rank (within level)", "Strength label", "Revenue variance ($)"}.issubset(a.columns)
    assert a.query("Level == 'Store' and `Strength label` == 'Strongest'").Store.iloc[0] == "CA_1"
    assert a.query("Level == 'Store' and `Strength label` == 'Weakest'").Store.iloc[0] == "CA_2"
    assert (a["Accuracy (%)"] > a["Last year's pattern accuracy (%)"]).all()
    sc = pd.read_csv(WEB / "scorecard.csv").query("model_label == @bto.DEFAULT_MODEL").set_index("level_name").wape_pct
    assert abs(a.query("Level == 'Company'")["Accuracy (%)"].iloc[0] - 100 * (1 - sc["Total"])) <= 0.05 + 1e-9


def test_forecast_plain(out):
    f = pd.read_csv(out / "forecast_plain.csv")
    src = _src().query("model == @bto.DEFAULT_ID")
    assert len(f) == len(src) == 14 * 24
    assert set(f["Weeks ahead"]) == {1, 2, 3, 4}
    assert not f.duplicated(["Level", "State", "Store", "Week"]).any()
    assert ((f["Inside likely range"] == "Yes") == (
        (f["Actual weekly revenue"] >= f["Likely range low"]) & (f["Actual weekly revenue"] <= f["Likely range high"]))).all()
    assert (f["Revenue variance ($)"] - (f["Forecast weekly revenue"] - f["Actual weekly revenue"])).abs().max() < 0.02
    assert (f.groupby("Level")["Top 3 largest miss"].apply(lambda s: (s == "Yes").sum()) == 3).all()
    company = f[f.Level == "Company"]
    stores = f[f.Level == "Store"].groupby("Week")["Actual weekly revenue"].sum()
    assert (company.set_index("Week")["Actual weekly revenue"] - stores).abs().max() < 1


def test_business_impact_and_context(out):
    b = pd.read_csv(out / "business_impact.csv")
    assert len(b) == 5 and b.order.tolist() == [1, 2, 3, 4, 5]
    src = _src().query("model == @bto.DEFAULT_ID and level_name == 'Total'")
    sn = _src().query("model == @bto.BASELINE_ID and level_name == 'Total'")
    assert abs(float(b.value_usd[0].replace("$", "").replace(",", "")) - src.abs_error_usd.sum()) < 0.02
    assert abs(float(b.value_usd[3].replace("$", "").replace(",", "")) - sn.abs_error_usd.sum()) < 0.02
    ctx = pd.read_csv(out / "forecast_context.csv")
    assert "Not available" in ctx.value.astype(str).tolist()
    assert "No genuinely future forecast exists" in " ".join(ctx.detail)


def test_methodology_and_actions(out):
    m = pd.read_csv(out / "methodology_notes.csv")
    assert list(m.columns) == ["Topic", "Plain explanation"]
    assert len(m) >= 7
    text = " ".join(m["Plain explanation"])
    assert "100 minus" in text and "10th to 90th" in text and "4 of 6" in text and "17 of 24" in text
    assert "0.02" in text or "within 0.02" in text
    a = pd.read_csv(out / "action_panel.csv")
    assert set(a.kind) == {"Insight", "Action", "Watch"}
    assert "CA_2" in a.text.iloc[1] and "WI_2" in a.text.iloc[1] and "95%" in a.text.iloc[1]


def test_no_emoji_or_em_dash():
    for name in FILES:
        txt = (WEB / name).read_text()
        assert "\u2014" not in txt and not any(0x1F300 <= ord(c) <= 0x1FAFF for c in txt), name
    assert "\u2014" not in (ROOT / "scripts" / "build_tableau_overview.py").read_text()
