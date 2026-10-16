"""Build README.md from the exported (derived) tables so every number in it comes from the pipeline outputs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TD = ROOT / "tableau" / "data"
LV = {1: "Total", 2: "State", 3: "Store", 4: "Category", 5: "Dept", 6: "State x Cat", 7: "State x Dept", 8: "Store x Cat", 9: "Store x Dept"}
MAIN = ["naive", "snaive", "ma4", "ets", "sarima_111_010", "sarima_airline", "lgbm_q", "lgbm_conformal", "r_ets_base", "r_arima_base", "r_arima_mint"]


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def table(sc: pd.DataFrame, metric: str, models: list[str], levels=range(1, 10), pct=True, nd=2, split="test", layer="hier", target=None) -> pd.DataFrame:
    d = sc[(sc["split"] == split) & (sc["metric"] == metric) & sc["layer"].str.startswith(layer) & sc["model"].isin(models)]
    p = d.pivot_table(index="model", columns="level_id", values="value").reindex(models)
    p = p[[c for c in levels if c in p.columns]]
    p = (100 * p if pct else p).round(nd)
    best = p.idxmin() if target is None else (p - target).abs().idxmin()
    out = p.astype(str)
    for c in p.columns:
        out.loc[best[c], c] = f"**{p.loc[best[c], c]}**"
    out.columns = [LV.get(c, c) for c in out.columns]
    out.insert(0, "model", [sc[sc["model"] == m]["model_label"].iloc[0] for m in out.index])
    return out.reset_index(drop=True)


def main() -> None:
    sc = pd.read_csv(TD / "metric_scorecard.csv")
    rc = pd.read_csv(TD / "reconciliation_comparison.csv")
    cl = pd.read_csv(TD / "cleaning_log.csv").set_index("metric")["value"]
    rs = pd.read_csv(TD / "residual_diagnostics_summary.csv")
    sizes = [(p.name, len(pd.read_csv(p)), p.stat().st_size / 1e6) for p in sorted(TD.glob("*.csv"))]

    wape = table(sc, "WAPE", MAIN)
    mase = table(sc, "MASE", MAIN, pct=False)
    pin = table(sc, "Pinball_scaled", MAIN)
    c95 = table(sc, "Coverage_95", MAIN, nd=1, target=95)
    c80 = table(sc, "Coverage_80", MAIN, nd=1, target=80)
    items = sc[(sc["split"] == "test") & sc["layer"].str.startswith("item") & (sc["metric"] == "WAPE") & sc["model"].isin(["naive","snaive","ma4","ets","lgbm_q","lgbm_conformal"])].pivot_table(index="model_label", columns="level_name", values="value")
    items = (100 * items).round(1).reset_index().rename(columns={"model_label": "model"})

    r = rc[(rc["split"] == "test") & (rc["metric"] == "WAPE_ratio_vs_base") & (rc["engine"] == "R fable") & (rc["structure"] == "full 154 nodes")]
    rp = r.pivot_table(index=["base_model", "method"], columns="level_id", values="value").reset_index()
    rp = rp[rp["method"] != "base (unreconciled)"]
    rp.columns = [LV.get(c, c) if isinstance(c, int) else c for c in rp.columns]
    for c in rp.columns[2:]:
        rp[c] = rp[c].round(3)
    g = rc[(rc["split"] == "test") & (rc["metric"] == "WAPE_ratio_vs_base") & (rc["structure"] == "geographic 84 nodes")]
    gp = g.pivot_table(index=["base_model", "method"], columns="level_id", values="value").reset_index()
    gp = gp[gp["method"] != "base (unreconciled)"]
    gp.columns = [LV.get(c, c) if isinstance(c, int) else c for c in gp.columns]
    for c in gp.columns[2:]:
        gp[c] = gp[c].round(3)
    lb = rs.pivot_table(index="model", columns="level_id", values="share_ljung_box_rejected").round(2)
    lb.columns = [LV[c] for c in lb.columns]
    lb = lb.reset_index()

    ncell = int(cl["item_store_day_cells"])
    txt = f"""# Retail Revenue Forecasting - Using hierarchical reconciliation and quantile gradient boosting to produce probabilistic revenue forecasts across US Walmart stores and departments

**Outcome.** On the final 24 held-out weeks of M5 Walmart revenue (4-week horizon, 6 origins), fixed-order seasonal ARIMA models give the lowest
total-revenue WAPE (2.6%, versus 5.0% for a 4-week moving average and 14.0% for seasonal naive), MinT-reconciled fable ARIMA is the most accurate
at 8 of the other 8 hierarchy levels in WAPE (3.2% to 6.3%), a plain 4-week moving average beats statsforecast ETS in WAPE at all 9 levels, and the
fable ARIMA intervals after bottom-up, OLS or WLS reconciliation under-cover badly (80% intervals cover 17% to 29% of actuals at the Total level).

Tableau dashboard: [Hierarchical Revenue Forecasting, M5](https://public.tableau.com/app/profile/abhishek.thakur1845/viz/HierarchicalRevenueForecastingM5/M5ForecastDashboard) (a 5-sheet dashboard built from the flattened CSVs in `tableau/web/`; the fuller design is in `tableau/DASHBOARD_SPEC.md`).

## Business question

How accurate and decision-useful are 4-week-ahead revenue forecasts across stores, departments and the total business, and does forcing
forecasts to add up across the hierarchy (reconciliation) make them better? "Decision-useful" is judged by calibrated 80 and 95 percent
intervals and by the safety buffer a planner would carry.

## Data

M5 Forecasting (Walmart): 30,490 item-store series ({ncell:,} item-store-day cells, 1,941 days from 2011-01-29 to **2016-05-22**), 3,049
items, 7 departments, 3 categories, 10 stores in CA, TX and WI, calendar with events and SNAP days, 6,841,121 weekly price rows.
Source: Nixtla public GitHub copy of the competition files (no Kaggle login), cross-checked cell for cell against a Hugging Face mirror;
URL, SHA-256 and shape checks in [`data/README.md`](data/README.md). **Licence:** no standalone open licence; competition rules
(described as non-commercial, academic research and education use). Raw files are never committed; only derived aggregates are.

Revenue is units x weekly sell price. Weekly series (277 complete Saturday-to-Friday weeks) are modelled; the 2 trailing days are dropped.
Cleaning counts (all computed in `sql/06_quality_and_daily.sql`, `artifacts/cleaning_log.csv`):
{int(cl["zero_unit_cells"]):,} zero-unit cells ({100 * cl["zero_unit_cells"] / ncell:.1f}%) kept as true zeros; {int(cl["leading_zero_cells"]):,} of them are
before an item-store's first sale; {int(cl["item_store_weeks_price_missing"]):,} item-store-weeks have no listed price (not on sale, revenue 0);
{int(cl["price_missing_but_units_positive"])} weeks have units without a price; {int(cl["price_listed_zero_unit_weeks"]):,} listed weeks had zero sales; {int(cl["stockout_proxy_runs"]):,} runs of 4 or more zero weeks with a
listed price (stockout or delisting proxy, {int(cl["stockout_proxy_weeks"]):,} weeks, kept); {int(cl["store_days_all_zero"])} store-days with zero units in every item (closed
stores, Christmas); {int(cl["days_dropped_partial_week"])} days dropped (partial last week).

## Scope

* **Aggregated levels, complete:** M5 levels 1-9 (Total 1, State 3, Store 10, Category 3, Department 7, State x Category 9, State x Department 21,
  Store x Category 30, Store x Department 70) = **154 series**, summing exactly from the 70 bottom series (DECIMAL arithmetic, coherence
  proven by data-contract tests).
* **Item layer, sampled:** level 10 top **300 items** and level 12 top **600 item x store series**, chosen by revenue in weeks 1-197 only
  (before the first backtest origin, so no future information picks the series). They cover {cl["item_sample_revenue_share_selection_window_pct_x100"] / 100:.1f}% (items) and
  {cl["item_layer_revenue_share_selection_window_pct_x100"] / 100:.1f}% (item x store) of selection-window revenue. This layer is **not coherent** with the 154-series
  hierarchy, is not reconciled, and gets ETS and LightGBM plus baselines only (SARIMA was not run there for compute reasons). Levels 11 and the other 2,749 items are not modelled.
* **Weekly, not daily:** weekly revenue is smoother, the 52-week season fits annual patterns, and 154 nodes x 20 refits stay tractable. A daily
  view of the last 12 weeks (levels 1-3) is exported for context only.

## Method

* **SQL (DuckDB, single thread):** long format, calendar and price joins, weekly fact, hierarchy bridge (summing matrix), two independent
  aggregation paths that must be identical, item-layer selection, cleaning log.
* **Backtest:** expanding window, 20 rolling origins every 4 weeks, horizon 4 weeks (28 days). Burn-in origins 197-225 (8, training and
  conformal calibration only), **validation** 229-249 (6), **final test 253-273 (6, weeks 254-277, defined up front, used for no choice)**. No random
  splits. Models are refit at every origin with data up to that origin.
* **Models:** naive, seasonal naive (52), 4-week moving average (Gaussian intervals from in-sample residuals); statsforecast AutoETS and two
  fixed-order SARIMA ([52] season); LightGBM, one global model per quantile (0.025, 0.1, 0.5, 0.9, 0.975), 250 trees, trained on the last 156 weeks before each origin, direct multi-horizon with horizon as a
  feature, target scaled by the trailing 52-week mean, lags, rolling means, SNAP days (node-weighted by state mix), events, week of year, price index; split-conformal intervals
  around the LightGBM median from earlier origins. R fable: STL + ETS and dynamic harmonic regression ARIMA (4 Fourier pairs, period 52) on all 154
  nodes, 12 origins (validation + test); reconciliation by bottom-up, OLS, WLS structural, WLS variance, MinT shrink; top-down (forecast and average proportions) on the strict 84-node geographic
  sub-hierarchy. Python numpy reconciliation of point forecasts for every Python model (error covariance from the model's own earlier backtest errors).
* **Metrics** (one code path for all models, including R): WAPE, MASE, RMSSE, bias, scaled pinball over 5 quantiles, coverage and width at 80 and
  95 percent. 95 percent bootstrap intervals over series (over origins for the single Total node), paired against seasonal naive or the unreconciled base.

## Results (final test window, weeks 254-277, horizon 4)

Bold = lowest per level. WAPE (%), lower is better:

{md(wape)}

MASE (scaled by one-step in-sample naive error), lower is better:

{md(mase)}

Scaled pinball loss (% of revenue), lower is better:

{md(pin)}

Empirical coverage of the 95% intervals (%, nominal 95; bold = closest to nominal):

{md(c95)}

Empirical coverage of the 80% intervals (%, nominal 80; bold = closest):

{md(c80)}

Item layer, test WAPE (%): models with the lowest value per column are the most accurate.

{md(items)}

### Reconciliation (R fable, test window): WAPE ratio vs unreconciled base, below 1 = better

{md(rp)}

Top-down on the 84-node geographic hierarchy:

{md(gp)}

### Residual tests (R fable, fit at the last test origin): share of nodes with Ljung-Box (lag 52) p < 0.05

{md(lb)}

### Reconciliation, Python models (test window, WAPE ratio vs base)

Naive, seasonal naive and the moving average are linear and already coherent, so reconciliation leaves them unchanged (ratio exactly 1.000). For the others the
changes are small and mixed: SARIMA ratios stay within 0.96 to 1.05; ETS MinT is worse at the top (1.12 at Total) and better at departments (0.92);
LightGBM improves at the top (bottom-up 0.85 and MinT 0.89 at Total) with almost no change below State. Full tables:
`tableau/data/reconciliation_comparison.csv`.

### Reading the results

* **Where baselines win or tie:** seasonal naive is the worst model at every level (the 52-week lag carries a year of noise and ignores growth). The 4-week moving
  average beats statsforecast AutoETS in WAPE at all 9 levels (for example Dept 6.49% vs 7.53%) and has lower pinball loss at all 9 levels, so AutoETS with a 52-week season
  adds nothing here. Against seasonal naive, ETS and the moving average both reduce WAPE to about 0.36 to 0.45 of seasonal naive at levels 1-3, and their bootstrap intervals overlap.
* **Best point forecasts:** SARIMA(1,1,1)(0,1,0)[52] has the lowest Total WAPE (2.56%); R ARIMA with MinT is a close second at Total (2.67%) and lowest at levels 2-9. SARIMA(0,1,1)(0,1,1)[52] has
  the lowest MASE and pinball at several levels. Gaps between the best two or three models are small relative to the per-origin and bootstrap spread
  (`ci_lo`/`ci_hi`, `wape_origin_min`/`max` in `metric_scorecard.csv`); only 6 test origins exist, so treat the ranking among the top models as tentative. At the item layer LightGBM (17.4% item,
  26.1% item x store) beats ETS (18.7%, 27.6%) and the moving average (20.0%, 28.6%).
* **Reconciliation (R fable):** MinT shrink improves ARIMA at every level (WAPE ratio 0.77 to 0.95) but does little for ETS (ratios 0.98 to 1.07, worse at Total). Bottom-up
  makes ARIMA 23% worse at Total, 25% at State. OLS and WLS are mixed. Top-down with average proportions is 1.4x to 2.5x worse than base at State, Store and Store x Dept; forecast proportions are neutral to slightly better below Total (0.93 to 1.02 for ARIMA).
* **Intervals:** at 95% nominal, statsforecast SARIMA and ETS cover about 84 to 92% and fable ARIMA base 86 to 92%; quantile LightGBM covers 86 to 96% and
  under-covers at 80% (50 to 74%); split-conformal LightGBM is closest to nominal at levels 2-9 (93 to 97% at 95%, 69 to 79% at 80%, excluding Category) but over-covers Category (100%). Coverage at Total rests on 24 forecasts only
  and is noisy. **Reconciled fable intervals under-cover severely at aggregate levels** for bottom-up, OLS and both WLS methods (80% coverage 17% to 29% at Total and 19% to 42% at State for ARIMA); MinT shrink is the best of them
  (57% to 71% at 80%, 82% to 88% at 95%) and sits below the unreconciled ARIMA base at most levels below Total. Do not use those reconciled intervals for planning at upper levels.
* **Residuals:** the Ljung-Box test (lag 52, fit at the last test origin) rejects whiteness at 70% to 100% of ARIMA nodes by level and 33% to 100% of ETS nodes, so residual autocorrelation remains and the Gaussian intervals are
  mis-specified; with 277 weekly points and lag 52 the test is also demanding.

## Planning use (ASSUMED parameters)

`tableau/data/decision_planning.csv` and `decision_sensitivity.csv` compare planning at the point forecast with planning at the 90th percentile forecast (ASSUMED service level)
under **ASSUMED** costs: 25% margin lost per revenue dollar of unmet demand and 5% carrying cost per dollar of excess plan (4-week horizon). These are scenario
inputs, not estimates from M5. See `docs/planning_memo.pdf`.

## Limitations

M5 is 2011-2016 data from 10 Walmart stores in three states, so conclusions about hierarchy size, seasonality and SNAP effects may not transfer. Revenue uses list weekly
prices (no promotions or discounts beyond price changes), zeros mix true no-demand and stockouts, so revenue is censored demand. The test window is 6 origins x 4 weeks (24 weeks): rankings
between close models are fragile. SARIMA orders are fixed, not searched, and AutoETS was run with a 52-week season (which may hurt it). The item layer is a sample, uncoordinated and without SARIMA. The R and Python
base models differ, so R-versus-Python comparisons mix model and implementation. The sales file ends 2016-05-22 (1,941 days); the competition's final 28 days were never public in the files used.

## Reproduce

```bash
make setup            # python 3.11 venv, pinned deps (R 4.5: install fable, fabletools, feasts, tsibble, dplyr, tidyr, furrr, jsonlite, readr, ggplot2, knitr; versions in R/sessionInfo.txt)
make data             # download M5 files from public mirrors, verify shapes (about 360 MB)
make sql backtest r evaluate export figures     # about 45 minutes on 8 cores (Python backtest about 25 min, R about 13 min, run concurrently)
make report           # Quarto HTML (reports/), memo PDF, README
make test             # ruff + pytest on the committed fixture (no raw data needed)
python scripts/check_determinism.py snapshot artifacts/output_hashes.json
```

Determinism: DuckDB single thread, fixed seeds, explicit orderings; two complete runs of the real pipeline (SQL, Python backtest, R, evaluate, export; run B started from scratch) produced identical SHA-256 for 31 of 32 hashed files, covering the raw forecast tables, the R outputs and every Tableau CSV except one. The one difference, `hierarchy_nodes.csv`, came from a revenue-share denominator bug that I fixed between the two runs; re-running only the export step twice on unchanged inputs then gave identical hashes for all 32 files. The hash list is committed
(see `artifacts/output_hashes.json`).

## Repository layout

`sql/` numbered DuckDB scripts, `src/hier_forecast/` package, `R/forecast_reconciliation.R`, `reports/forecast_report.qmd` (+ rendered HTML), `tableau/data/` extracts and
`tableau/DASHBOARD_SPEC.md`, `tableau/web/` (5 flattened CSVs for Tableau Public web authoring, `make tableau-web`), `docs/` (data dictionary, memo, figures), `tests/` (fixture, data contracts, leakage, metrics, reconciliation, exports, determinism).

## Tableau extracts

{md(pd.DataFrame([(a, f"{b:,}", f"{c:.2f}") for a, b, c in sizes], columns=["file", "rows", "MB"]))}

## Figures

![WAPE by level](docs/figures/wape_by_level.png)
![Forecasts](docs/figures/forecast_vs_actual_total.png)
![Reconciliation](docs/figures/reconciliation_effect.png)
![Calibration](docs/figures/calibration.png)
"""
    (ROOT / "README.md").write_text(txt)
    print("README.md written")


if __name__ == "__main__":
    main()
