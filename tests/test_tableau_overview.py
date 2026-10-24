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
FILES = ["overview_kpis.csv", "accuracy_by_area.csv", "forecast_plain.csv"]


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("overview")
    bto.build_all(WEB, d)
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
    company = acc[(acc.area == "Company total") & (acc.series == bto.OUR)].accuracy_pct.iloc[0]
    assert k.value_display[0] == f"{company:.1f}%"
    assert k.value_display[3] == src.loc[4, "kpi_display"]
    fx = pd.read_csv(WEB / "forecast_explorer.csv")
    t = fx[(fx.model_label == bto.DEFAULT_MODEL) & (fx.level_name == "Total") & (fx.horizon_weeks == 1)]
    assert k.value_display[2] == f"{int(t.in_80.sum())} of {len(t)}"
    assert k.caption.str.len().max() < 160 and k.caption.notna().all()


def test_accuracy_by_area(out):
    a = pd.read_csv(out / "accuracy_by_area.csv")
    assert list(a.columns) == ["area_order", "area", "series", "accuracy_pct"]
    assert len(a) == 18 and set(a.series) == {bto.OUR, bto.LAST_YEAR}
    assert a.accuracy_pct.between(0, 100).all() and a.area.nunique() == 9
    p = a.pivot(index="area_order", columns="series", values="accuracy_pct")
    assert (p[bto.OUR] > p[bto.LAST_YEAR]).all()
    sc = pd.read_csv(WEB / "scorecard.csv")
    for model, series in [(bto.DEFAULT_MODEL, bto.OUR), (bto.BASELINE_MODEL, bto.LAST_YEAR)]:
        w = sc[sc.model_label == model].set_index("level_order").wape_pct
        assert (p[series] - 100 * (1 - w)).abs().max() <= 0.05 + 1e-9


def test_forecast_plain(out):
    f = pd.read_csv(out / "forecast_plain.csv")
    assert list(f.columns) == ["week", "area", "horizon_weeks", "actual_revenue", "forecast_revenue", "range_low", "range_high", "inside_range"]
    fx = pd.read_csv(WEB / "forecast_explorer.csv")
    src = fx[fx.model_label == bto.DEFAULT_MODEL]
    assert len(f) == len(src) == 168 and f.area.nunique() == 14
    assert set(f.horizon_weeks) == {1, 4} and set(f.inside_range) == {"Yes", "No"}
    assert f.notna().all().all() and (f.range_low <= f.range_high).all() and (f.actual_revenue > 0).all()
    inside = (f.actual_revenue >= f.range_low) & (f.actual_revenue <= f.range_high)
    assert ((f.inside_range == "Yes") == inside).all()
    assert (f.inside_range == "Yes").sum() == int(src.in_80.sum())
    assert abs(f.actual_revenue.sum() - src.actual_usd.sum()) < 1


def test_no_emoji_or_em_dash():
    for name in FILES + ["README.md"]:
        txt = (WEB / name).read_text()
        assert "\u2014" not in txt and not any(0x1F300 <= ord(c) <= 0x1FAFF for c in txt), name
    for path in ("README.md", "scripts/build_tableau_overview.py"):
        assert "\u2014" not in (ROOT / path).read_text(), path
