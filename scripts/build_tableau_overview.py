"""Build the plain-language Overview sources for the Tableau dashboard (tableau/web/overview_*.csv).

Every value is derived from the existing tableau/web exports (scorecard.csv, kpis.csv, forecast_explorer.csv).
All cards use the default model (R fable ARIMA MinT shrink), the same as accuracy_by_area.csv and forecast_plain.csv;
nothing is modelled or typed here. Run after scripts/build_tableau_web.py (`make tableau-web`).
"""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / "tableau" / "web"

DEFAULT_MODEL = "R fable ARIMA MinT shrink"
BASELINE_MODEL = "Seasonal naive (52 wk)"

DATA = ROOT / "tableau" / "data"
DEFAULT_ID, BASELINE_ID = "r_arima_mint", "snaive"
STATE_NAMES = {"CA": "California", "TX": "Texas", "WI": "Wisconsin"}
ALL_STATES, ALL_STORES = "All states", "All stores"
LEVEL_NAMES = {"Total": "Company", "State": "State", "Store": "Store"}
ACC, LY = "Accuracy (%)", "Last year's pattern accuracy (%)"
LOW_ACCURACY_PCT = 95.0   # stores below this accuracy are flagged in the "what to do" headline


def geo(level_name: str, node_label: str) -> tuple[str, str]:
    """(State, Store) plain columns for a Total, State or Store node."""
    if level_name == "Total":
        return ALL_STATES, ALL_STORES
    if level_name == "State":
        return STATE_NAMES[node_label], ALL_STORES
    if level_name == "Store":
        return STATE_NAMES[node_label.split("_")[0]], node_label
    raise ValueError(f"unexpected level: {level_name}")


def geo_forecasts(data: pathlib.Path) -> pd.DataFrame:
    """All horizons (1-4) and test origins for the Company, State and Store nodes, from tableau/data/forecast_vs_actual.csv."""
    f = pd.read_csv(data / "forecast_vs_actual.csv")
    return f[f.level_name.isin(LEVEL_NAMES) & f.model.isin([DEFAULT_ID, BASELINE_ID])].copy()


def accuracy_pct(d: pd.DataFrame) -> float:
    return 100 * (1 - d.abs_error_usd.sum() / d.actual_usd.sum())


def build_kpis(web: pathlib.Path) -> pd.DataFrame:
    k = pd.read_csv(web / "kpis.csv").set_index("kpi_order")
    fx = pd.read_csv(web / "forecast_explorer.csv")
    sc = pd.read_csv(web / "scorecard.csv")
    tot = sc[sc.level_name == "Total"].set_index("model_label").wape_pct
    best_wape = float(tot[DEFAULT_MODEL])
    cut = 1 - best_wape / float(tot[BASELINE_MODEL])
    stores = k.loc[4, "kpi_display"]
    t = fx[(fx.model_label == DEFAULT_MODEL) & (fx.level_name == "Total") & (fx.horizon_weeks == 1)]
    inside, weeks = int(t.in_80.sum()), len(t)
    rows = [
        ("Forecast accuracy at company level", f"{100 * (1 - best_wape):.1f}%",
         f"Our company-wide forecast missed actual weekly revenue by {100 * best_wape:.1f}% on average, "
         "looking up to 4 weeks ahead."),
        ("More accurate than last year's pattern", f"{100 * cut:.1f}%",
         f"Our company-wide forecast errors are {100 * cut:.1f}% smaller than simply repeating last year's sales."),
        ("Weeks actual revenue landed in the likely range", f"{inside} of {weeks}",
         "Company-wide, one week ahead: weeks where actual revenue fell inside our likely range (80% band)."),
        ("Stores where our forecast beats last year's pattern", str(stores),
         "Stores where our forecast was more accurate than repeating last year's sales."),
    ]
    return pd.DataFrame([(i + 1, *r) for i, r in enumerate(rows)], columns=["order", "label", "value_display", "caption"])


def build_accuracy(data: pathlib.Path) -> pd.DataFrame:
    f = geo_forecasts(data)
    rows = []
    for (lvl, node), g in f.groupby(["level_id", "node_label"], sort=True):
        name = g.level_name.iloc[0]
        state, store = geo(name, node)
        rows.append((int(lvl), LEVEL_NAMES[name], state, store,
                     round(accuracy_pct(g[g.model == DEFAULT_ID]), 1), round(accuracy_pct(g[g.model == BASELINE_ID]), 1)))
    return pd.DataFrame(rows, columns=["level_order", "Level", "State", "Store", "Accuracy (%)", "Last year's pattern accuracy (%)"])


def build_forecast(data: pathlib.Path) -> pd.DataFrame:
    f = geo_forecasts(data)
    d = f[f.model == DEFAULT_ID].sort_values(["level_id", "node_label", "target_week_start"])
    sg = [geo(a, b) for a, b in zip(d.level_name, d.node_label)]
    return pd.DataFrame({
        "Week": d.target_week_start.values,
        "Level": d.level_name.map(LEVEL_NAMES).values,
        "State": [s for s, _ in sg],
        "Store": [s for _, s in sg],
        "Weeks ahead": d.horizon_weeks.astype(int).values,
        "Forecast made after": d.forecast_made_after.values,
        "Actual weekly revenue": d.actual_usd.round(2).values,
        "Forecast weekly revenue": d.forecast_usd.round(2).values,
        "Likely range low": d.lo80_usd.round(2).values,
        "Likely range high": d.hi80_usd.round(2).values,
        "Inside likely range": ["Yes" if v == 1 else "No" for v in d.in_80_interval],
    })


def _usd(x: float) -> str:
    return f"${x / 1e6:.2f} million"


def build_headlines(web: pathlib.Path, data: pathlib.Path) -> pd.DataFrame:
    k = build_kpis(web).set_index("order")
    acc = build_accuracy(data)
    st = acc[acc.Level == "State"].sort_values("Accuracy (%)")
    so = acc[acc.Level == "Store"].sort_values("Accuracy (%)")
    co = acc[acc.Level == "Company"].iloc[0]
    f = geo_forecasts(data)
    tot = f[(f.model == DEFAULT_ID) & (f.level_name == "Total")]
    by_h = tot.groupby("horizon_weeks").apply(accuracy_pct, include_groups=False)
    last = tot[tot.origin_week_idx == tot.origin_week_idx.max()].sort_values("target_week_start")
    n_weeks = tot.target_week_start.nunique()
    stores_beat = int((so["Accuracy (%)"] > so["Last year's pattern accuracy (%)"]).sum())
    states_beat = int((st["Accuracy (%)"] > st["Last year's pattern accuracy (%)"]).sum())
    low = so[so["Accuracy (%)"] < LOW_ACCURACY_PCT]
    gap = last.forecast_usd.sum() / last.actual_usd.sum() - 1
    rows = [
        ("Overall accuracy",
         f"Company weekly revenue forecasts were {co['Accuracy (%)']:.1f}% accurate over {n_weeks} tested weeks, with errors "
         f"{k.loc[2, 'value_display']} smaller than repeating last year's sales.",
         f"company accuracy {co['Accuracy (%)']:.1f}%; last year's pattern {co[LY]:.1f}%; "
         f"error reduction {k.loc[2, 'value_display']}; tested weeks {n_weeks}",
         "accuracy_by_area.csv; overview_kpis.csv"),
        ("Strongest and weakest areas",
         f"{st.iloc[-1].State} is the most accurate state ({st.iloc[-1]['Accuracy (%)']:.1f}%) and {st.iloc[0].State} the least "
         f"({st.iloc[0]['Accuracy (%)']:.1f}%); by store, {so.iloc[-1].Store} is strongest ({so.iloc[-1]['Accuracy (%)']:.1f}%) and "
         f"{so.iloc[0].Store} weakest ({so.iloc[0]['Accuracy (%)']:.1f}%). {states_beat} of {len(st)} states and {stores_beat} of "
         f"{len(so)} stores beat last year's pattern.",
         "; ".join(f"{s if lv == 'State' else so_} {a:.1f}%"
                  for lv, s, so_, a in acc[acc.Level != "Company"][["Level", "State", "Store", ACC]].itertuples(index=False)),
         "accuracy_by_area.csv"),
        ("Next 4 weeks outlook",
         f"Company accuracy stays between {by_h.min():.1f}% and {by_h.max():.1f}% from 1 to 4 weeks ahead. In the latest tested "
         f"4 weeks ({last.target_week_start.iloc[0]} to {last.target_week_start.iloc[-1]}) the forecast was {_usd(last.forecast_usd.sum())} "
         f"against {_usd(last.actual_usd.sum())} actual ({100 * abs(gap):.1f}% {'below' if gap < 0 else 'above'}), inside the likely range in "
         f"{int(last.in_80_interval.sum())} of {len(last)} weeks.",
         "; ".join(f"{h} week{'s' if h > 1 else ''} ahead {v:.1f}%" for h, v in by_h.items())
         + f"; latest forecast {last.forecast_usd.sum():.2f}; latest actual {last.actual_usd.sum():.2f}",
         "forecast_plain.csv"),
        ("What to do",
         f"Use the forecast for company and state revenue planning ({co['Accuracy (%)']:.1f}% accurate, ahead of last year's pattern "
         f"everywhere), and hold a wider buffer at "
         + (" and ".join(low.Store) if len(low) else "no store")
         + f", the {len(low)} stores below {LOW_ACCURACY_PCT:.0f}% accuracy, reviewing them weekly.",
         "; ".join(f"{s} {a:.1f}%" for s, a in zip(low.Store, low[ACC])) + f"; threshold {LOW_ACCURACY_PCT:.0f}% (chosen cut-off)",
         "accuracy_by_area.csv"),
    ]
    return pd.DataFrame([(i + 1, *r) for i, r in enumerate(rows)],
                        columns=["story_point", "title", "headline", "source_numbers", "source_file"])


def build_all(src: pathlib.Path = WEB, dst: pathlib.Path = WEB, data: pathlib.Path = DATA) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    build_kpis(src).to_csv(dst / "overview_kpis.csv", index=False)
    build_accuracy(data).to_csv(dst / "accuracy_by_area.csv", index=False)
    build_forecast(data).to_csv(dst / "forecast_plain.csv", index=False)
    build_headlines(src, data).to_csv(dst / "story_headlines.csv", index=False)


if __name__ == "__main__":
    build_all()
    print("tableau/web: overview_kpis.csv, accuracy_by_area.csv, forecast_plain.csv, story_headlines.csv written")
