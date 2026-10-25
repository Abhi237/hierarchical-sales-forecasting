"""tableau/web overview CSVs: regenerate from the committed tableau/web sources, check shape and ranges, tie out to the sources."""
from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / "tableau" / "web"
_spec = importlib.util.spec_from_file_location("build_tableau_overview", ROOT / "scripts" / "build_tableau_overview.py")
bto = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bto)
FILES = ["overview_kpis.csv", "accuracy_by_area.csv", "forecast_plain.csv", "story_headlines.csv"]
DATA = ROOT / "tableau" / "data"


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("overview")
    bto.build_all(WEB, d, DATA)
    return d


@pytest.mark.parametrize("name", FILES)
def test_committed_files_match_regeneration(out, name):
    pd.testing.assert_frame_equal(pd.read_csv(out / name), pd.read_csv(WEB / name))


def test_overview_kpis(out):
    k = pd.read_csv(out / "overview_kpis.csv")
    assert list(k.columns) == ["order", "label", "value_display", "caption"]
    assert k.order.tolist() == [1, 2, 3, 4]
    src = pd.read_csv(WEB / "kpis.csv").set_index("kpi_order")
    tot = pd.read_csv(WEB / "scorecard.csv").query("level_name == 'Total'").set_index("model_label").wape_pct
    w, w_sn = tot[bto.DEFAULT_MODEL], tot[bto.BASELINE_MODEL]
    assert k.value_display[0] == f"{100 * (1 - w):.1f}%"
    assert k.value_display[1] == f"{100 * (1 - w / w_sn):.1f}%"
    acc = pd.read_csv(WEB / "accuracy_by_area.csv")
    company = acc[acc.Level == "Company"]["Accuracy (%)"].iloc[0]
    assert k.value_display[0] == f"{company:.1f}%"
    assert k.value_display[3] == src.loc[4, "kpi_display"]
    fx = pd.read_csv(WEB / "forecast_explorer.csv")
    t = fx[(fx.model_label == bto.DEFAULT_MODEL) & (fx.level_name == "Total") & (fx.horizon_weeks == 1)]
    assert k.value_display[2] == f"{int(t.in_80.sum())} of {len(t)}"
    assert k.caption.str.len().max() < 160 and k.caption.notna().all()


def _src():
    f = pd.read_csv(DATA / "forecast_vs_actual.csv")
    return f[f.level_name.isin(["Total", "State", "Store"])]


def test_accuracy_by_area(out):
    a = pd.read_csv(out / "accuracy_by_area.csv")
    assert list(a.columns) == ["level_order", "Level", "State", "Store", "Accuracy (%)", "Last year's pattern accuracy (%)"]
    assert a.Level.value_counts().to_dict() == {"Company": 1, "State": 3, "Store": 10}
    assert set(a.State) == {"All states", "California", "Texas", "Wisconsin"}
    assert (a[a.Level != "Store"].Store == "All stores").all()
    st = a[a.Level == "Store"]
    assert all(s.startswith({"California": "CA_", "Texas": "TX_", "Wisconsin": "WI_"}[g]) for g, s in zip(st.State, st.Store))
    assert a["Accuracy (%)"].between(0, 100).all()
    assert (a["Accuracy (%)"] > a["Last year's pattern accuracy (%)"]).all()
    # company and state rows agree with the scorecard level WAPE (pooled over nodes, horizons 1-4, 6 origins)
    sc = pd.read_csv(WEB / "scorecard.csv").query("model_label == @bto.DEFAULT_MODEL").set_index("level_name").wape_pct
    assert abs(a[a.Level == "Company"]["Accuracy (%)"].iloc[0] - 100 * (1 - sc["Total"])) <= 0.05 + 1e-9
    f = _src()
    d = f[(f.model == bto.DEFAULT_ID) & (f.node_label == "CA_2")]
    assert abs(st[st.Store == "CA_2"]["Accuracy (%)"].iloc[0] - 100 * (1 - d.abs_error_usd.sum() / d.actual_usd.sum())) <= 0.05 + 1e-9


def test_forecast_plain(out):
    f = pd.read_csv(out / "forecast_plain.csv")
    assert list(f.columns) == ["Week", "Level", "State", "Store", "Weeks ahead", "Forecast made after", "Actual weekly revenue",
                               "Forecast weekly revenue", "Likely range low", "Likely range high", "Inside likely range"]
    src = _src().query("model == @bto.DEFAULT_ID")
    assert len(f) == len(src) == 14 * 24
    assert set(f["Weeks ahead"]) == {1, 2, 3, 4} and f.Week.nunique() == 24
    # one row per area and week, so a Week filter never double counts
    assert not f.duplicated(["Level", "State", "Store", "Week"]).any()
    assert f.notna().all().all() and (f["Likely range low"] <= f["Likely range high"]).all()
    inside = (f["Actual weekly revenue"] >= f["Likely range low"]) & (f["Actual weekly revenue"] <= f["Likely range high"])
    assert ((f["Inside likely range"] == "Yes") == inside).all()
    assert abs(f["Actual weekly revenue"].sum() - src.actual_usd.sum()) < 1
    company = f[f.Level == "Company"]
    stores = f[f.Level == "Store"].groupby("Week")["Actual weekly revenue"].sum()
    assert (company.set_index("Week")["Actual weekly revenue"] - stores).abs().max() < 1   # stores add up to the company


def test_story_headlines(out):
    h = pd.read_csv(out / "story_headlines.csv")
    assert list(h.columns) == ["story_point", "title", "headline", "source_numbers", "source_file"]
    assert h.story_point.tolist() == [1, 2, 3, 4] and h.headline.notna().all()
    k = pd.read_csv(out / "overview_kpis.csv")
    acc = pd.read_csv(out / "accuracy_by_area.csv")
    assert k.value_display[0] in h.headline[0] and k.value_display[1] in h.headline[0]
    so = acc[acc.Level == "Store"].sort_values("Accuracy (%)")
    assert so.Store.iloc[0] in h.headline[1] and so.Store.iloc[-1] in h.headline[1]
    for word in ("WAPE", "ARIMA", "MinT", "SARIMA", "naive", "quantile", "model"):
        assert not h.headline.str.contains(word).any(), word


def test_no_emoji_or_em_dash():
    for name in FILES + ["README.md"]:
        txt = (WEB / name).read_text()
        assert "\u2014" not in txt and not any(0x1F300 <= ord(c) <= 0x1FAFF for c in txt), name
    for path in ("README.md", "scripts/build_tableau_overview.py"):
        assert "\u2014" not in (ROOT / path).read_text(), path
