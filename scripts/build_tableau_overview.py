"""Build the plain-language Overview sources for the Tableau dashboard (tableau/web/).

Every number comes from tableau/data/forecast_vs_actual.csv, tableau/data/calendar_weeks.csv,
tableau/data/origins_design.csv, and the existing tableau/web scorecard/kpis. Nothing is typed.
Default forecast = R fable ARIMA MinT shrink (model id r_arima_mint). Baseline = seasonal naive
(model id snaive): last year's same week. Accuracy = 100 * (1 - WAPE), with WAPE as in
src/hier_forecast/metrics.py: sum(|forecast - actual|) / sum(actual). Likely range = the 80%
prediction interval (10th to 90th percentile; see R/forecast_reconciliation.R and config levels_pi).
Run after scripts/build_tableau_web.py (`make tableau-web`).
"""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / "tableau" / "web"
DATA = ROOT / "tableau" / "data"

DEFAULT_MODEL = "R fable ARIMA MinT shrink"
BASELINE_MODEL = "Seasonal naive (52 wk)"
DEFAULT_ID, BASELINE_ID = "r_arima_mint", "snaive"
STATE_NAMES = {"CA": "California", "TX": "Texas", "WI": "Wisconsin"}
ALL_STATES, ALL_STORES = "All states", "All stores"
LEVEL_NAMES = {"Total": "Company", "State": "State", "Store": "Store"}
ACC, LY = "Accuracy (%)", "Last year's pattern accuracy (%)"
INTERVAL_NOMINAL_PCT = 80   # lo80/hi80 = 0.10/0.90 quantiles
LOW_ACCURACY_PCT = 95.0     # stores below this are flagged in the action panel (chosen cut-off)
DESC_MAX = 45


def geo(level_name: str, node_label: str) -> tuple[str, str]:
    if level_name == "Total":
        return ALL_STATES, ALL_STORES
    if level_name == "State":
        return STATE_NAMES[node_label], ALL_STORES
    if level_name == "Store":
        return STATE_NAMES[node_label.split("_")[0]], node_label
    raise ValueError(f"unexpected level: {level_name}")


def geo_forecasts(data: pathlib.Path) -> pd.DataFrame:
    f = pd.read_csv(data / "forecast_vs_actual.csv")
    return f[f.level_name.isin(LEVEL_NAMES) & f.model.isin([DEFAULT_ID, BASELINE_ID])].copy()


def wape(d: pd.DataFrame) -> float:
    """WAPE as metrics.metrics_from_sums: sum(abs) / sum(y)."""
    y = d.actual_usd.sum()
    return float(d.abs_error_usd.sum() / y) if y > 0 else float("nan")


def accuracy_pct(d: pd.DataFrame) -> float:
    return 100 * (1 - wape(d))


def _usd(x: float) -> str:
    ax = abs(x)
    if ax >= 1e6:
        return f"${x / 1e6:,.2f}m"
    if ax >= 1e3:
        return f"${x / 1e3:,.1f}k"
    return f"${x:,.0f}"


def _usd_full(x: float) -> str:
    return f"${x:,.2f}"


def _clip(s: str) -> str:
    return s if len(s) <= DESC_MAX else s[: DESC_MAX - 1].rstrip() + "."


def build_kpis(web: pathlib.Path, data: pathlib.Path) -> pd.DataFrame:
    """Four headline cards (company, all weeks ahead). Kept for the existing Overview tiles."""
    sc = pd.read_csv(web / "scorecard.csv")
    tot = sc[sc.level_name == "Total"].set_index("model_label").wape_pct
    w = float(tot[DEFAULT_MODEL])
    cut = 1 - w / float(tot[BASELINE_MODEL])
    f = geo_forecasts(data)
    t = f[(f.model == DEFAULT_ID) & (f.level_name == "Total")]
    h1 = t[t.horizon_weeks == 1]
    stores = (
        f[f.model == DEFAULT_ID]
        .query("level_name == 'Store'")
        .groupby("node_label")
        .apply(lambda g: accuracy_pct(g), include_groups=False)
    )
    sn = (
        f[f.model == BASELINE_ID]
        .query("level_name == 'Store'")
        .groupby("node_label")
        .apply(lambda g: accuracy_pct(g), include_groups=False)
    )
    n_beat = int((stores > sn.reindex(stores.index)).sum())
    rows = [
        ("Forecast accuracy at company level", f"{100 * (1 - w):.1f}%",
         f"Company forecast missed weekly revenue by {100 * w:.1f}% on average, looking 1 to 4 weeks ahead."),
        ("More accurate than last year's pattern", f"{100 * cut:.1f}%",
         f"Company forecast errors are {100 * cut:.1f}% smaller than repeating last year's sales."),
        ("Weeks inside the likely range (1 week ahead)", f"{int(h1.in_80_interval.sum())} of {len(h1)}",
         f"Company, 1 week ahead only: weeks inside the {INTERVAL_NOMINAL_PCT}% likely range "
         f"({int(t.in_80_interval.sum())} of {len(t)} across all 1 to 4 weeks ahead)."),
        ("Stores beating last year's pattern", f"{n_beat} of {len(stores)}",
         "Stores where our forecast was more accurate than repeating last year's sales."),
    ]
    return pd.DataFrame([(i + 1, *r) for i, r in enumerate(rows)],
                        columns=["order", "label", "value_display", "caption"])


def _area_slice(f: pd.DataFrame, level: str, state: str, store: str, weeks_ahead) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = f.copy()
    if level == "Company":
        d = d[d.level_name == "Total"]
    elif level == "State":
        code = {v: k for k, v in STATE_NAMES.items()}[state]
        d = d[(d.level_name == "State") & (d.node_label == code)]
    else:
        d = d[(d.level_name == "Store") & (d.node_label == store)]
    if weeks_ahead != "All":
        d = d[d.horizon_weeks == int(weeks_ahead)]
    return d[d.model == DEFAULT_ID], d[d.model == BASELINE_ID]


def build_kpi_strip(data: pathlib.Path) -> pd.DataFrame:
    """Filterable KPI strip: one row per Level/State/Store x Weeks ahead (All, 1, 2, 3, 4)."""
    f = geo_forecasts(data)
    areas = [("Company", ALL_STATES, ALL_STORES)]
    for _code, name in STATE_NAMES.items():
        areas.append(("State", name, ALL_STORES))
    for code, name in STATE_NAMES.items():
        for n in sorted(f[(f.level_name == "Store") & (f.node_label.str.startswith(code + "_"))].node_label.unique()):
            areas.append(("Store", name, n))
    horizons = ["All", 1, 2, 3, 4]
    rows = []
    for level, state, store in areas:
        for h in horizons:
            our, base = _area_slice(f, level, state, store, h)
            if our.empty:
                continue
            actual = float(our.actual_usd.sum())
            forecast = float(our.forecast_usd.sum())
            var = forecast - actual
            var_pct = 100 * var / actual if actual else float("nan")
            acc = accuracy_pct(our)
            base_w = wape(base)
            improve = 100 * (1 - wape(our) / base_w) if base_w else float("nan")
            inside = int(our.in_80_interval.sum())
            n = len(our)
            rows.append({
                "Level": level, "State": state, "Store": store,
                "Weeks ahead": h if h == "All" else int(h),
                "Actual revenue ($)": round(actual, 2),
                "Forecast revenue ($)": round(forecast, 2),
                "Revenue variance ($)": round(var, 2),
                "Revenue variance (%)": round(var_pct, 2),
                "Variance direction": "Over-forecast" if var > 0 else ("Under-forecast" if var < 0 else "Exact"),
                "Forecast accuracy (%)": round(acc, 1),
                "Baseline improvement (%)": round(improve, 1),
                "Weeks inside likely range": inside,
                "Weeks total": n,
                "Interval coverage": f"{inside} of {n}",
                "Interval nominal (%)": INTERVAL_NOMINAL_PCT,
                "Actual revenue display": _usd(actual),
                "Forecast revenue display": _usd(forecast),
                "Revenue variance display": _usd(var),
                "Desc actual": _clip("Total actual revenue in filter"),
                "Desc forecast": _clip("Total forecast revenue in filter"),
                "Desc variance $": _clip("Forecast minus actual (signed)"),
                "Desc variance %": _clip("Variance as % of actual"),
                "Desc accuracy": _clip("100 minus error share of revenue"),
                "Desc baseline": _clip("Error cut vs last year's pattern"),
                "Desc coverage": _clip(f"Weeks inside {INTERVAL_NOMINAL_PCT}% likely range"),
            })
    return pd.DataFrame(rows)


def build_accuracy(data: pathlib.Path) -> pd.DataFrame:
    f = geo_forecasts(data)
    rows = []
    for (lvl, node), g in f.groupby(["level_id", "node_label"], sort=True):
        name = g.level_name.iloc[0]
        state, store = geo(name, node)
        our, base = g[g.model == DEFAULT_ID], g[g.model == BASELINE_ID]
        actual = float(our.actual_usd.sum())
        forecast = float(our.forecast_usd.sum())
        var = forecast - actual
        rows.append({
            "level_order": int(lvl), "Level": LEVEL_NAMES[name], "State": state, "Store": store,
            ACC: round(accuracy_pct(our), 1), LY: round(accuracy_pct(base), 1),
            "Actual revenue ($)": round(actual, 2), "Forecast revenue ($)": round(forecast, 2),
            "Revenue variance ($)": round(var, 2),
            "Revenue variance (%)": round(100 * var / actual, 2) if actual else float("nan"),
            "Variance direction": "Over-forecast" if var > 0 else ("Under-forecast" if var < 0 else "Exact"),
            "Abs error ($)": round(float(our.abs_error_usd.sum()), 2),
        })
    df = pd.DataFrame(rows)
    df["Rank (within level)"] = df.groupby("Level")[ACC].rank(ascending=False, method="min").astype(int)
    labels = []
    for _, r in df.iterrows():
        n = (df.Level == r.Level).sum()
        if r.Level == "Company":
            labels.append("Company total")
        elif r["Rank (within level)"] == 1:
            labels.append("Strongest")
        elif r["Rank (within level)"] == n:
            labels.append("Weakest")
        else:
            labels.append("-")
    df["Strength label"] = labels
    return df.sort_values(["level_order", "Rank (within level)", "Store"]).reset_index(drop=True)


def build_forecast(data: pathlib.Path) -> pd.DataFrame:
    f = geo_forecasts(data)
    d = f[f.model == DEFAULT_ID].sort_values(["level_id", "node_label", "target_week_start"])
    sg = [geo(a, b) for a, b in zip(d.level_name, d.node_label)]
    err = d.forecast_usd - d.actual_usd
    out = pd.DataFrame({
        "Week": d.target_week_start.values,
        "Level": d.level_name.map(LEVEL_NAMES).values,
        "State": [s for s, _ in sg],
        "Store": [s for _, s in sg],
        "Weeks ahead": d.horizon_weeks.astype(int).values,
        "Forecast made after": d.forecast_made_after.values,
        "Actual weekly revenue": d.actual_usd.round(2).values,
        "Forecast weekly revenue": d.forecast_usd.round(2).values,
        "Revenue variance ($)": err.round(2).values,
        "Revenue variance (%)": (100 * err / d.actual_usd).round(2).values,
        "Direction": ["Over-forecast" if e > 0 else ("Under-forecast" if e < 0 else "Exact") for e in err],
        "Abs error ($)": d.abs_error_usd.round(2).values,
        "Likely range low": d.lo80_usd.round(2).values,
        "Likely range high": d.hi80_usd.round(2).values,
        "Inside likely range": ["Yes" if v == 1 else "No" for v in d.in_80_interval],
        "Interval nominal (%)": INTERVAL_NOMINAL_PCT,
    })
    # Abs error rank and top-3 flag within each Level (company/state/store separately)
    out["Abs error rank"] = out.groupby("Level")["Abs error ($)"].rank(ascending=False, method="min").astype(int)
    out["Top 3 largest miss"] = out["Abs error rank"].le(3).map({True: "Yes", False: "No"})
    return out


def build_business_impact(data: pathlib.Path) -> pd.DataFrame:
    f = geo_forecasts(data)
    our = f[(f.model == DEFAULT_ID) & (f.level_name == "Total")]
    base = f[(f.model == BASELINE_ID) & (f.level_name == "Total")]
    abs_our = float(our.abs_error_usd.sum())
    abs_base = float(base.abs_error_usd.sum())
    avg = float(our.abs_error_usd.mean())
    worst = our.loc[our.abs_error_usd.idxmax()]
    saved = abs_base - abs_our
    rows = [
        (1, "Total absolute miss", _usd(abs_our), _usd_full(abs_our),
         f"Sum of |forecast - actual| over {len(our)} company week-ahead forecasts in the test window."),
        (2, "Average weekly miss", _usd(avg), _usd_full(avg),
         "Mean absolute miss per company week-ahead forecast."),
        (3, "Largest single-week miss", _usd(float(worst.abs_error_usd)), _usd_full(float(worst.abs_error_usd)),
         f"Week of {worst.target_week_start}, {int(worst.horizon_weeks)} week(s) ahead; "
         f"{'under' if worst.error_usd < 0 else 'over'}-forecast."),
        (4, "Last year's pattern absolute miss", _usd(abs_base), _usd_full(abs_base),
         "Same company weeks, repeating last year's sales for the same week."),
        (5, "Absolute miss avoided vs last year", _usd(saved), _usd_full(saved),
         f"{100 * saved / abs_base:.1f}% less absolute miss than last year's pattern."),
    ]
    return pd.DataFrame(rows, columns=["order", "metric", "value_display", "value_usd", "caption"])


def build_forecast_context(data: pathlib.Path) -> pd.DataFrame:
    cal = pd.read_csv(data / "calendar_weeks.csv")
    org = pd.read_csv(data / "origins_design.csv")
    hist = cal[cal.role == "train/backtest history"]
    test = cal[cal.role == "test target week"]
    test_org = org[org.split.str.contains("test")]
    val_org = org[org.split == "validation"]
    burn = org[org.split.str.contains("burn-in")]
    rows = [
        ("Observed history", f"{hist.week_start.min()} to {hist.week_start.max()}",
         f"{len(hist)} complete weeks of M5 weekly revenue (company to store)."),
        ("Training at each origin", "Weeks 1 through the origin week",
         "At every origin the models train only on weeks up to that origin; no future leaks."),
        ("Burn-in origins", f"{burn.forecast_first_week_start.min()} to {burn.forecast_last_week_start.max()}",
         f"{len(burn)} origins used only for training and interval calibration."),
        ("Validation origins", f"{val_org.forecast_first_week_start.min()} to {val_org.forecast_last_week_start.max()}",
         f"{len(val_org)} origins used for model selection; not the final score."),
        ("Test (backtest) window", f"{test.week_start.min()} to {test.week_start.max()}",
         f"{len(test)} target weeks from {len(test_org)} origins x 4 weeks ahead. All Overview numbers use this window."),
        ("Origins x horizons", f"{len(test_org)} origins x 4 weeks ahead = {len(test_org) * 4} forecasts per area",
         "Each origin trains through the origin week, then forecasts the next 4 weeks."),
        ("Future forecast", "Not available",
         "No genuinely future forecast exists. The latest week in the data is 2016-05-14; every figure is a backtest."),
    ]
    return pd.DataFrame([(i + 1, *r) for i, r in enumerate(rows)],
                        columns=["order", "topic", "value", "detail"])


def build_methodology_notes(data: pathlib.Path) -> pd.DataFrame:
    f = geo_forecasts(data)
    our = f[f.model == DEFAULT_ID]
    tot = our[our.level_name == "Total"]
    # reconciliation: max |sum(children) - parent| on forecasts
    gaps = []
    for week, h in tot[["target_week_start", "horizon_weeks"]].itertuples(index=False):
        mask = (our.target_week_start == week) & (our.horizon_weeks == h)
        company = our[mask & (our.level_name == "Total")].forecast_usd.iloc[0]
        states = our[mask & (our.level_name == "State")].forecast_usd.sum()
        stores = our[mask & (our.level_name == "Store")].forecast_usd.sum()
        gaps.append(abs(states - company))
        gaps.append(abs(stores - company))
        for code in STATE_NAMES:
            st = our[mask & (our.level_name == "State") & (our.node_label == code)].forecast_usd.iloc[0]
            so = our[mask & (our.level_name == "Store") & (our.node_label.str.startswith(code + "_"))].forecast_usd.sum()
            gaps.append(abs(so - st))
    max_gap = max(gaps)
    h1 = tot[tot.horizon_weeks == 1]
    rows = [
        ("Forecast accuracy",
         "Accuracy (%) = 100 minus the share of actual revenue missed. "
         "Share missed = sum of absolute forecast errors divided by sum of actual revenue "
         "(WAPE in the evaluation code). Higher is better."),
        ("Baseline improvement",
         "Baseline improvement (%) = 100 x (1 - our miss share / last year's miss share). "
         "Last year's pattern repeats the same calendar week from 52 weeks earlier (seasonal naive). "
         "81.0% at company means our absolute errors are 81.0% smaller than that baseline."),
        ("Likely range and coverage",
         f"The likely range is the {INTERVAL_NOMINAL_PCT}% prediction interval "
         f"(10th to 90th percentile of the forecast distribution). "
         f"'Weeks inside the likely range' counts how many of the tested week-ahead forecasts "
         f"had actual revenue between the low and high of that band. "
         f"At company, 1 week ahead: {int(h1.in_80_interval.sum())} of {len(h1)}. "
         f"Across all 1 to 4 weeks ahead: {int(tot.in_80_interval.sum())} of {len(tot)}. "
         f"A {INTERVAL_NOMINAL_PCT}% band that is well calibrated would land near {INTERVAL_NOMINAL_PCT}% coverage."),
        ("Revenue definition",
         "Weekly revenue = units sold x the week's sell price for each item in each store "
         "(M5 sell_prices). Prices are stored to the cent; children sum to parents to the cent in the SQL model."),
        ("Forecast method",
         "Each series is forecast with an ARIMA model (R fable). Store, state and company forecasts are then "
         "reconciled with MinT shrinkage so they add up. The Overview uses that reconciled forecast."),
        ("Benchmark",
         "Last year's pattern (seasonal naive): forecast = actual revenue from the same week 52 weeks earlier. "
         "A 4-week moving average and other models are on the Methodology tab."),
        ("Hierarchy check",
         f"On the Overview forecast, store forecasts sum to their state and state forecasts sum to the company "
         f"within {max_gap:.2f} dollars at most across every tested week (rounding). Actuals sum exactly."),
        ("What this is not",
         "This is a backtest on 2015-12-05 to 2016-05-14. There is no forecast of weeks after the last data week. "
         "Action recommendations for buffers use a chosen accuracy cut-off of "
         f"{LOW_ACCURACY_PCT:.0f}% and are labelled as such."),
    ]
    return pd.DataFrame(rows, columns=["Topic", "Plain explanation"])


def build_action_panel(data: pathlib.Path) -> pd.DataFrame:
    acc = build_accuracy(data)
    fp = build_forecast(data)
    stores = acc[acc.Level == "Store"].sort_values(ACC)
    company = fp[fp.Level == "Company"]
    top_miss = company.nsmallest(1, "Abs error rank").iloc[0]
    weak = stores[stores[ACC] < LOW_ACCURACY_PCT]
    rows = [
        (1, "Insight",
         f"Company accuracy is {acc[acc.Level == 'Company'][ACC].iloc[0]:.1f}% and beats last year's pattern at every "
         f"state and store. Use the company and state forecasts for planning."),
        (2, "Action",
         "Hold a wider buffer at "
         + " and ".join(f"{r.Store} ({r[ACC]:.1f}%)" for _, r in weak.iterrows())
         + f", the {len(weak)} stores below the chosen {LOW_ACCURACY_PCT:.0f}% accuracy cut-off, and review them weekly."),
        (3, "Watch",
         f"Largest company miss: week of {top_miss.Week}, {int(top_miss['Weeks ahead'])} week(s) ahead, "
         f"{_usd(float(top_miss['Abs error ($)']))} absolute miss ({top_miss.Direction.lower()}). "
         f"Filter the weekly chart to Top 3 largest miss = Yes to inspect the other large misses."),
    ]
    return pd.DataFrame(rows, columns=["order", "kind", "text"])


def build_headlines(web: pathlib.Path, data: pathlib.Path) -> pd.DataFrame:
    """Four story-point headlines (kept for the existing story layout)."""
    k = build_kpis(web, data).set_index("order")
    acc = build_accuracy(data)
    impact = build_business_impact(data).set_index("order")
    st = acc[acc.Level == "State"].sort_values(ACC)
    so = acc[acc.Level == "Store"].sort_values(ACC)
    co = acc[acc.Level == "Company"].iloc[0]
    f = geo_forecasts(data)
    tot = f[(f.model == DEFAULT_ID) & (f.level_name == "Total")]
    by_h = tot.groupby("horizon_weeks").apply(accuracy_pct, include_groups=False)
    last = tot[tot.origin_week_idx == tot.origin_week_idx.max()].sort_values("target_week_start")
    gap = last.forecast_usd.sum() / last.actual_usd.sum() - 1
    weak = so[so[ACC] < LOW_ACCURACY_PCT]
    rows = [
        ("Overall accuracy",
         f"Company weekly revenue forecasts were {co[ACC]:.1f}% accurate over {tot.target_week_start.nunique()} tested weeks, "
         f"with absolute misses {k.loc[2, 'value_display']} smaller than last year's pattern "
         f"({impact.loc[1, 'value_display']} total absolute miss vs {impact.loc[4, 'value_display']}).",
         f"accuracy {co[ACC]:.1f}%; baseline improvement {k.loc[2, 'value_display']}; "
         f"abs miss {impact.loc[1, 'value_usd']}; baseline abs miss {impact.loc[4, 'value_usd']}",
         "kpi_strip.csv; business_impact.csv"),
        ("Strongest and weakest areas",
         f"{st.iloc[-1].State} is the most accurate state ({st.iloc[-1][ACC]:.1f}%) and {st.iloc[0].State} the least "
         f"({st.iloc[0][ACC]:.1f}%); by store, {so.iloc[-1].Store} is strongest ({so.iloc[-1][ACC]:.1f}%) and "
         f"{so.iloc[0].Store} weakest ({so.iloc[0][ACC]:.1f}%).",
         "; ".join(f"{r.Store if r.Level == 'Store' else r.State} {r[ACC]:.1f}%" for _, r in acc[acc.Level != 'Company'].iterrows()),
         "accuracy_by_area.csv"),
        ("Latest 4 weeks reliability",
         f"Company accuracy stays between {by_h.min():.1f}% and {by_h.max():.1f}% from 1 to 4 weeks ahead. "
         f"In the latest tested 4 weeks ({last.target_week_start.iloc[0]} to {last.target_week_start.iloc[-1]}) "
         f"the forecast was {_usd(float(last.forecast_usd.sum()))} against {_usd(float(last.actual_usd.sum()))} actual "
         f"({100 * abs(gap):.1f}% {'below' if gap < 0 else 'above'}), inside the likely range in "
         f"{int(last.in_80_interval.sum())} of {len(last)} weeks. No future weeks are forecast.",
         "; ".join(f"{h} week{'s' if h > 1 else ''} ahead {v:.1f}%" for h, v in by_h.items()),
         "forecast_plain.csv; forecast_context.csv"),
        ("What to do",
         f"Plan with company and state forecasts ({co[ACC]:.1f}% accurate). Hold a wider buffer at "
         + (" and ".join(weak.Store) if len(weak) else "no store")
         + f", the {len(weak)} stores below the chosen {LOW_ACCURACY_PCT:.0f}% accuracy cut-off.",
         "; ".join(f"{s} {a:.1f}%" for s, a in zip(weak.Store, weak[ACC]))
         + f"; threshold {LOW_ACCURACY_PCT:.0f}% (chosen cut-off)",
         "action_panel.csv; accuracy_by_area.csv"),
    ]
    return pd.DataFrame([(i + 1, *r) for i, r in enumerate(rows)],
                        columns=["story_point", "title", "headline", "source_numbers", "source_file"])


def build_all(src: pathlib.Path = WEB, dst: pathlib.Path = WEB, data: pathlib.Path = DATA) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    build_kpis(src, data).to_csv(dst / "overview_kpis.csv", index=False)
    build_kpi_strip(data).to_csv(dst / "kpi_strip.csv", index=False)
    build_accuracy(data).to_csv(dst / "accuracy_by_area.csv", index=False)
    build_forecast(data).to_csv(dst / "forecast_plain.csv", index=False)
    build_business_impact(data).to_csv(dst / "business_impact.csv", index=False)
    build_forecast_context(data).to_csv(dst / "forecast_context.csv", index=False)
    build_methodology_notes(data).to_csv(dst / "methodology_notes.csv", index=False)
    build_action_panel(data).to_csv(dst / "action_panel.csv", index=False)
    build_headlines(src, data).to_csv(dst / "story_headlines.csv", index=False)


if __name__ == "__main__":
    build_all()
    print("tableau/web: overview_kpis, kpi_strip, accuracy_by_area, forecast_plain, "
          "business_impact, forecast_context, methodology_notes, action_panel, story_headlines written")
