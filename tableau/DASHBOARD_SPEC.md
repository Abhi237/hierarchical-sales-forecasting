# Tableau Public dashboard specification

Title (workbook and first dashboard): **Retail Revenue Forecasting - Using hierarchical reconciliation and quantile gradient boosting to
produce probabilistic revenue forecasts across US Walmart stores and departments**

Audience: a retail planning lead who must decide which forecast to trust at which level (total, state, store, department) and how much
buffer to plan around it. Data: M5 Walmart weekly revenue, 2011-01-29 to 2016-05-20, 10 stores in CA, TX, WI. Every file in
`tableau/data/` is a derived aggregate; no raw M5 file is used. Tableau Public limits: all CSVs together are far below 100 MB; do not use
live connections; connect to the CSVs as extracts. Published at https://public.tableau.com/app/profile/abhishek.thakur1845/viz/HierarchicalRevenueForecastingM5/M5ForecastDashboard (see the next section for what was actually built).

## Published version (what was actually built)

The dashboard is published at https://public.tableau.com/app/profile/abhishek.thakur1845/viz/HierarchicalRevenueForecastingM5/M5ForecastDashboard.

The published version is a leaner, single dashboard with 5 sheets, built in Tableau Public web authoring from the five flattened CSVs in
`tableau/web/` (see `tableau/web/README.md`). The sheets are:

1. WAPE heat map by model and hierarchy level (`scorecard.csv`).
2. WAPE ratio vs seasonal naive (`scorecard.csv`).
3. Forecast vs actual, with the default view set to Total, LightGBM quantile and horizon 1 week (`forecast_explorer.csv`).
4. Interval coverage vs the nominal 80% (`calibration.csv`).
5. Reconciliation impact, WAPE ratio vs the unreconciled base (`reconciliation.csv`).

No calculated fields were built in Tableau, because every value (rates as fractions, ranks, ratios, Yes/No flags, buffers) is pre-calculated by
`scripts/build_tableau_web.py`. `planning_buffer.csv` is available for a planning sheet but is not a sheet on the published dashboard.

The fuller 5-dashboard design below (data model with relationships, parameters, calculated fields, executive scorecard, forecast explorer,
reconciliation impact, calibration and planning takeaways) remains in this file as design intent and was not built as written.

No emojis and no em dashes anywhere in titles, captions or tooltips.

## 1. Data model (Data Source tab)

Connect all CSVs in `tableau/data/` as separate text-file data sources and relate them as follows (Relationships, not joins; Tableau
keeps each at its own grain).

| Primary table | Related to | On |
|---|---|---|
| `metric_scorecard` | `hierarchy_nodes` (levels only: use `level_id`, no node) | none; keep as its own source |
| `forecast_vs_actual` | `hierarchy_nodes` | `node_id` = `node_id` |
| `forecast_vs_actual` | `actuals_weekly` | `node_id` = `node_id` AND `target_week_idx` = `week_idx` (use only for history context) |
| `actuals_weekly` | `calendar_weeks` | `week_idx` = `week_idx` |
| `decision_planning` | `hierarchy_nodes` | `node_id` |
| other files | standalone sources | filtered by `level_id`, `model`, `split` actions |

Sources used per dashboard are listed below; each worksheet uses exactly one source to avoid accidental fan-out.

## 2. Palette and typography

| Use | Hex |
|---|---|
| Actual revenue line | `#111111` |
| Naive | `#9AA0A6`; Seasonal naive `#5F6368`; Moving average `#BDC1C6` |
| ETS `#E69F00`; SARIMA airline `#009E73`; SARIMA(1,1,1)(0,1,0) `#66C2A5` |
| LightGBM quantile `#0072B2`; LightGBM + conformal `#56B4E9` |
| R fable ETS (base) `#D55E00`; R fable ETS MinT `#CC79A7`; R ARIMA (base) `#8C510A`; R ARIMA MinT `#B2182B` |
| Interval bands | same hue as the model, 25 percent opacity (95 percent), 45 percent opacity (80 percent) |
| Better / worse than seasonal naive | blue `#0072B2` / orange `#D55E00` (diverging, centred at ratio 1) |
| Background | white; text `#202124`; gridlines `#E8EAED` |
| Font | Tableau default (Tableau Book) |

Colour legend for models is a shared parameter-driven palette: build one discrete colour assignment on `model` and copy it to every sheet.

## 3. Global parameters, filters and calculated fields

Parameters
- `p_split` (string list): `test`, `validation`; default `test`. Test is the final held-out window (6 origins, weeks 254-277).
- `p_metric` (string list): `WAPE`, `MASE`, `RMSSE`, `Bias`, `Pinball_scaled`; default `WAPE`.
- `p_nominal` (integer list): 80, 95; default 95.
- `p_horizon` (integer list): 1, 2, 3, 4; default 4.

Calculated fields (on `metric_scorecard`)
- `Selected metric value` = `IF [metric] = [p_metric] THEN [value] END`
- `Selected ci_lo` = `IF [metric] = [p_metric] THEN [ci_lo] END`, `Selected ci_hi` likewise.
- `WAPE pct` = `IF [metric] = "WAPE" THEN [value] END` (format 0.0%).
- `Skill vs seasonal naive` = `IF [metric] = "WAPE_ratio_vs_snaive" THEN [value] END`
- `Better than seasonal naive?` = `[Skill vs seasonal naive] < 1`
- `Is baseline` = `[model_family] = "baseline"`
- `Rank within level` = `RANK(MIN([Selected metric value]), "asc")` computed per `level_name` and `split`
- `Winner flag` = `[Rank within level] = 1`
- `Baseline wins level` = `{ FIXED [level_id], [split] : MIN(IF [Is baseline] AND [metric]="WAPE" THEN [value] END) } <= { FIXED [level_id], [split] : MIN(IF NOT [Is baseline] AND [metric]="WAPE" THEN [value] END) }`

Calculated fields (on `forecast_vs_actual`)
- `Selected lo` = `IF [p_nominal] = 95 THEN [lo95_usd] ELSE [lo80_usd] END`; `Selected hi` likewise with `hi95_usd` / `hi80_usd`.
- `Inside selected interval` = `IF [p_nominal] = 95 THEN [in_95_interval] ELSE [in_80_interval] END`
- `Coverage` = `AVG([Inside selected interval])`
- `Abs pct error` = `[abs_error_usd] / [actual_usd]`
- `Horizon filter` = `[horizon_weeks] = [p_horizon]`
- `Revenue (thousand USD)` = `[forecast_usd] / 1000`

Calculated fields (on `decision_planning`)
- `Buffer pct` = `SUM([safety_buffer_usd]) / SUM([actual_revenue_usd])`
- `ASSUMED cost reduction` = `1 - SUM([ASSUMED_cost_q90_plan_usd]) / SUM([ASSUMED_cost_point_plan_usd])`

Filters shown as quick filters unless stated: `split` (parameter), `level_name` (single value on dashboards 2 and 5, multi on 1),
`model_label` (multi), `node_label` (single, cascading from `level_name`, dashboard 2), `model_family` (multi, dashboard 1).

## 4. Dashboards

Size: Automatic with fixed height 900 px for desktop; all sheets below sit on tiled containers. Each dashboard has a one-line takeaway
text box at the top (written after publishing from the memo; the numbers must be read from the sheets, not typed from here).

### Dashboard 1: Executive scorecard

Title: "Which model is most accurate at each level of the hierarchy?"
Source: `metric_scorecard.csv` (rows with `layer` = "hierarchy (levels 1-9)").
- Sheet 1a, heat table: Rows `model_label` sorted by mean `WAPE pct`; Columns `level_name` (sorted by `level_id`); Text and Colour `WAPE pct`; diverging colour centred on the seasonal naive value of the same column is not available, so use a sequential single-hue blue (lighter = lower error). Filter `split` = `p_split`, `metric` = WAPE, exclude `model_family` containing "reconciled (Python".
- Sheet 1b, dot plot with intervals: Columns `Selected metric value` with error bars from `Selected ci_lo` to `Selected ci_hi`; Rows `model_label`; Detail `level_name`; small multiples by `level_name`. Tooltip: model, level, value, 95 percent bootstrap interval, `ci_method`, `n_nodes`.
- Sheet 1c, skill bar: Columns `Skill vs seasonal naive` per `model_label`, Rows `level_name`; reference line at 1; colour by `Better than seasonal naive?`. Caption: "Ratio of WAPE to seasonal naive, paired bootstrap interval in tooltip".
- Sheet 1d, KPI strip: for the selected level the best model by `Rank within level`, its WAPE, and the WAPE of the best baseline (`Is baseline`). Text: "Best baseline: <model> at <WAPE>".
- Sheet 1e, per-origin spread: source `metrics_by_origin.csv`, Columns `origin_week_end`, Rows `WAPE`, Colour `model_label`, filter `level_name`; shows how stable each ranking is across origins.
Actions: clicking a `level_name` cell filters dashboards 2 to 5 (Use as Filter, all sheets on workbook).
Callout (text, static wording, no numbers): "Where a bar is above the reference line the model is worse than seasonal naive. Baselines are kept in the chart on purpose."

### Dashboard 2: Forecast explorer

Title: "Forecast versus actual with prediction intervals"
Source: `forecast_vs_actual.csv` (+ `actuals_weekly.csv` for history).
- Filters: `level_name` (single), `node_label` (single, depends on level), `model_label` (multi, default LightGBM quantile, SARIMA airline, Seasonal naive), `p_nominal`, `p_horizon`.
- Sheet 2a, main line: Columns `target_week_start` (continuous, week); Rows `forecast_usd`; Colour `model_label`; one line per model. Dual-axis layer 2: actual from `actual_usd` (black, `#111111`, thicker). Interval band: two additional measures `Selected lo` and `Selected hi` drawn as Area with "Lines" dual-axis "Band" technique (area between lo and hi using a `Band width` = `[Selected hi] - [Selected lo]` stacked on `Selected lo` with transparent lower area), opacity 25 percent, colour by model. Filter `Horizon filter` = TRUE. Tooltip: week, model, forecast, actual, interval bounds, `Inside selected interval`.
- Sheet 2b, history context: source `actuals_weekly.csv`, `week_start` vs `revenue_usd` for the same `node_id`, last 104 weeks, shaded test window using `calendar_weeks.role`.
- Sheet 2c, error by horizon: `horizon_weeks` on Columns, `AVG([abs_error_usd])` on Rows, colour model.
- Sheet 2d, recent daily view: source `actuals_daily_recent.csv` (levels 1 to 3 only), `date` vs `revenue`, colour `level_name`; shows daily rhythm in the last 12 weeks, caption "Daily actuals; forecasts are weekly".
- Sheet 2e, scatter: `forecast_usd` vs `actual_usd` for the selected node and models, 45 degree reference line (`Reference line` from 0 to max actual).

### Dashboard 3: Reconciliation impact

Title: "Does forcing forecasts to add up make them better?"
Sources: `reconciliation_comparison.csv`, `coherence_gap.csv`.
- Filters: `engine` (R fable / Python), `base_model`, `structure`, `split` (parameter).
- Sheet 3a, ratio bars: Columns `level_name`, Rows `value` where `metric` = "WAPE_ratio_vs_base", Colour `method`; error bars `ci_lo` to `ci_hi`; reference line at 1; colours bottom-up `#D55E00`, OLS `#E69F00`, WLS structural `#009E73`, WLS variance `#56B4E9`, MinT shrink `#0072B2`, top-down (forecast proportions) `#CC79A7`, top-down (average proportions) `#8C510A`.
- Sheet 3b, absolute table: `method` x `level_name`, text = `value` for `metric` = "WAPE" (format 0.00%), small multiples by `base_model`.
- Sheet 3c, incoherence: source `coherence_gap.csv`, Columns `level_name`, Rows `mean_abs_rel_gap` (percent), Colour `model_label`; caption "How far unreconciled forecasts are from adding up".
- Sheet 3d, interval effect (R only): `metric` in (Coverage_95, Width_95_rel, Pinball_scaled) by `method`, to show whether reconciliation changed calibration; Python rows have no interval metrics and are filtered out by `engine` = "R fable".
Note shown on the sheet: "Top-down is only defined on the strict 84-node geographic hierarchy (levels 1, 2, 3, 9)."

### Dashboard 4: Calibration

Title: "Are the prediction intervals honest?"
Sources: `interval_coverage.csv`, `interval_coverage_by_horizon.csv`, `residual_diagnostics_summary.csv`, `residual_diagnostics_nodes.csv`.
- Sheet 4a, coverage vs nominal: Columns `level_name`, Rows `coverage`, Colour `model_label`, panel by `nominal` (80, 95); reference line at `nominal_coverage` (use a dual-axis constant line, dotted); error bars `coverage_ci_lo` to `coverage_ci_hi`. Filter `split` = `p_split`.
- Sheet 4b, miscalibration heat table: Rows `model_label`, Columns `level_name`, Colour `miscalibration` (diverging, blue = over-covers, orange = under-covers, centre 0), text format +0.00.
- Sheet 4c, width versus coverage: scatter of `mean_width_over_revenue` (x) against `coverage` (y), one mark per model x level, shape `nominal`. Narrower with equal coverage is better.
- Sheet 4d, coverage by horizon: `horizon_weeks` x `coverage_95`, colour `model_label`, filter level.
- Sheet 4e, residual tests: bars of `share_ljung_box_rejected` by `level_name`, colour `model` (ets, arima), reference line at 0.05. Tooltip with `median_ljung_box_p`, `median_lag1_acf`, `share_shapiro_rejected`. Caption: "Ljung-Box lag 52 at the last test origin; share of nodes with p below 0.05".

### Dashboard 5: Planning memo takeaways

Title: "What buffer should planning carry?"
Sources: `decision_planning.csv`, `decision_sensitivity.csv`, `origins_design.csv`, `cleaning_log.csv`.
Banner (always visible, bold): "All cost figures use ASSUMED parameters (gross margin on unmet revenue, carrying cost on excess plan). They are scenario inputs, not estimates from M5."
- Sheet 5a, buffer by level: `level_name` x `Buffer pct`, colour `model_label`; data from rows where `ASSUMED_plan_quantile` = 0.9 (the buffer is the gap between the upper 80 percent bound and the point forecast).
- Sheet 5b, realised service level: `share_weeks_actual_within_q90` averaged by `level_name` and `model_label`, reference line at 0.90 (nominal one-sided target of the upper 80 percent bound).
- Sheet 5c, ASSUMED cost scenario: source `decision_sensitivity.csv`; Columns `ASSUMED_gross_margin`, Rows `ASSUMED_carrying_cost`, Text `ASSUMED_cost_pct_of_revenue`, Colour same; filter `plan_rule`, `level_name`, `model_label`; mark the default assumption with `is_default_assumption` = 1 (bold border).
- Sheet 5d, cost reduction from planning at q90 versus point: `ASSUMED cost reduction` by `level_name`, `model_label`. Title suffix "(ASSUMED costs)".
- Sheet 5e, data and method notes (text sheet): backtest design (`origins_design.csv` table: origin, split, train_through), dataset scope, cleaning counts (selected rows of `cleaning_log.csv`: zero-unit share, weeks without price, partial week dropped, store-days closed), and limits (2011-2016 data, 10 stores, revenue = units x list price, no promotions beyond price, SNAP and events only as covariates).
- Takeaway text box (filled after publishing, one sentence per bullet): which model for which level, whether reconciliation helped, which intervals under-cover, what buffer percentage the best calibrated model implies. Numbers come from sheets 1 to 4.

## 5. Publishing checklist

1. Create extracts, check that every worksheet filter uses the `split` parameter default `test`.
2. Add the data-source caption on every dashboard: "Source: M5 Forecasting (Walmart), derived aggregates only. Forecasts are backtests, not live predictions."
3. Hide unused fields; rename `*_usd` fields to readable names; set number formats ($#,##0 for revenue, 0.0% for WAPE).
4. Publish to Tableau Public. Done for the lean version: the URL is in the README and at the top of this file.
