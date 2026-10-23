"""Build the plain-language Overview sources for the Tableau dashboard (tableau/web/overview_*.csv).

Every value is derived from the existing tableau/web exports (scorecard.csv, kpis.csv, forecast_explorer.csv);
nothing is modelled or typed here. Run after scripts/build_tableau_web.py (`make tableau-web`).
"""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / "tableau" / "web"

DEFAULT_MODEL = "R fable ARIMA MinT shrink"
BASELINE_MODEL = "Seasonal naive (52 wk)"
OUR, LAST_YEAR = "Our forecast", "Last year's pattern"

AREA_NAMES = {
    "Total": "Company total",
    "State": "By state",
    "Store": "By store",
    "Category": "By category",
    "Department": "By department",
    "State x Category": "By state and category",
    "State x Department": "By state and department",
    "Store x Category": "By store and category",
    "Store x Department": "By store and department",
}
STATE_NAMES = {"CA": "California", "TX": "Texas", "WI": "Wisconsin"}


def node_area(level_name: str, node_label: str) -> str:
    if level_name == "Total":
        return "Company total"
    if level_name == "State":
        return f"State: {STATE_NAMES.get(node_label, node_label)}"
    if level_name == "Store":
        return f"Store {node_label}"
    raise ValueError(f"unexpected level in forecast explorer: {level_name}")


def build_kpis(web: pathlib.Path) -> pd.DataFrame:
    k = pd.read_csv(web / "kpis.csv").set_index("kpi_order")
    fx = pd.read_csv(web / "forecast_explorer.csv")
    best_wape = float(k.loc[1, "kpi_value"])
    cut = float(k.loc[2, "kpi_value"])
    stores = k.loc[4, "kpi_display"]
    t = fx[(fx.model_label == DEFAULT_MODEL) & (fx.level_name == "Total") & (fx.horizon_weeks == 1)]
    inside, weeks = int(t.in_80.sum()), len(t)
    rows = [
        ("Forecast accuracy at company level", f"{100 * (1 - best_wape):.1f}%",
         f"Our best company-wide forecast missed actual weekly revenue by {100 * best_wape:.1f}% on average, "
         "looking up to 4 weeks ahead."),
        ("More accurate than last year's pattern", f"{100 * cut:.1f}%",
         f"Our company-wide forecast errors are {100 * cut:.1f}% smaller than simply repeating last year's sales."),
        ("Weeks actual revenue landed in the likely range", f"{inside} of {weeks}",
         "Company-wide, one week ahead: weeks where actual revenue fell inside our likely range (80% band)."),
        ("Stores where our forecast beats last year's pattern", str(stores),
         "Stores where our forecast was more accurate than repeating last year's sales."),
    ]
    return pd.DataFrame([(i + 1, *r) for i, r in enumerate(rows)], columns=["order", "label", "value_display", "caption"])


def build_accuracy(web: pathlib.Path) -> pd.DataFrame:
    s = pd.read_csv(web / "scorecard.csv")
    out = []
    for model, series in [(DEFAULT_MODEL, OUR), (BASELINE_MODEL, LAST_YEAR)]:
        d = s[s.model_label == model].sort_values("level_order")
        for r in d.itertuples():
            out.append((int(r.level_order), AREA_NAMES[r.level_name], series, round(100 * (1 - r.wape_pct), 1)))
    df = pd.DataFrame(out, columns=["area_order", "area", "series", "accuracy_pct"])
    return df.sort_values(["area_order", "series"], ascending=[True, False]).reset_index(drop=True)


def build_forecast(web: pathlib.Path) -> pd.DataFrame:
    fx = pd.read_csv(web / "forecast_explorer.csv")
    d = fx[fx.model_label == DEFAULT_MODEL].sort_values(["level_order", "node_label", "horizon_weeks", "target_week_start"])
    return pd.DataFrame({
        "week": d.target_week_start.values,
        "area": [node_area(a, b) for a, b in zip(d.level_name, d.node_label)],
        "horizon_weeks": d.horizon_weeks.astype(int).values,
        "actual_revenue": d.actual_usd.round(2).values,
        "forecast_revenue": d.forecast_usd.round(2).values,
        "range_low": d.lo80_usd.round(2).values,
        "range_high": d.hi80_usd.round(2).values,
        "inside_range": ["Yes" if v == 1 else "No" for v in d.in_80],
    })


def build_all(src: pathlib.Path = WEB, dst: pathlib.Path = WEB) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    build_kpis(src).to_csv(dst / "overview_kpis.csv", index=False)
    build_accuracy(src).to_csv(dst / "accuracy_by_area.csv", index=False)
    build_forecast(src).to_csv(dst / "forecast_plain.csv", index=False)


if __name__ == "__main__":
    build_all()
    print("tableau/web: overview_kpis.csv, accuracy_by_area.csv, forecast_plain.csv written")
