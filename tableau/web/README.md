# tableau/web: flattened sources for Tableau Public web authoring

Five CSVs, one per worksheet source. No joins, relationships or calculated fields are needed in Tableau: every value is pre-calculated by
`scripts/build_tableau_web.py` (`make tableau-web`) from the existing exports in `tableau/data/` (no new modelling). All rows are the **final test
window** (6 origins, target weeks 2015-12-05 to 2016-05-14), M5 hierarchy levels 1-9, dollars in USD. Every rate is a **fraction** (0.0256 = 2.56%): apply
one percent number format in Tableau. Dates are ISO (YYYY-MM-DD), UTF-8, comma separated, no index column. `level_order` 1-9 sorts the levels:
1 Total, 2 State, 3 Store, 4 Category, 5 Dept, 6 State x Cat, 7 State x Dept, 8 Store x Cat, 9 Store x Dept.

| file | rows | grain | dashboard sheet (see `tableau/DASHBOARD_SPEC.md`) |
|---|---|---|---|
| `scorecard.csv` | 99 | model x level | Executive scorecard: WAPE heat map by model and level, best-in-level highlight, ratio vs seasonal naive |
| `forecast_explorer.csv` | 1,008 | node x model x origin x horizon (1 and 4) | Forecast explorer: forecast vs actual with 80% and 95% bands, for Total, 3 states, 10 stores |
| `reconciliation.csv` | 108 | base model x method x level | Reconciliation impact: WAPE ratio vs unreconciled base with bootstrap intervals |
| `calibration.csv` | 360 | model x level x nominal (80, 95) | Calibration: coverage vs nominal, with intervals, and relative interval width |
| `planning_buffer.csv` | 135 | model x level | Planning takeaways: safety buffer, realised service, ASSUMED cost change |

## Columns

**scorecard.csv** (models: 3 baselines, statsforecast ETS and 2 SARIMA, LightGBM quantile, fable ETS and ARIMA base and MinT; excluded: the duplicate
`lgbm_conformal` point forecast, the Python point-only reconciled variants and the geographic sub-hierarchy variants)
`model_label, model, model_family, is_baseline (Yes/No), level_order, level_name, wape_pct, wape_ci_lo, wape_ci_hi` (95% bootstrap over series; over origins
for Total), `wape_ratio_vs_snaive, wape_ratio_ci_lo, wape_ratio_ci_hi, mase, rank_in_level` (1 = lowest WAPE among the 11 models, ties share a rank),
`is_best_in_level, better_than_snaive` (Yes when the ratio is below 1; Seasonal naive itself is No).

**forecast_explorer.csv** (levels 1-3: 14 nodes; models: Seasonal naive, 4-week moving average, SARIMA(0,1,1)(0,1,1)[52] (the SARIMA with the lowest mean test WAPE
over levels 1-3), LightGBM quantile, fable ARIMA MinT, fable ETS MinT; horizons 1 and 4 weeks)
`level_order, level_name, node_id, node_label, model_order, model_label, model_family, origin_week_idx, forecast_made_after, target_week_idx, target_week_start,
horizon_weeks, forecast_usd, actual_usd, lo80_usd, hi80_usd, lo95_usd, hi95_usd, in_80, in_95` (1 = actual inside the interval, 0 = outside; the mean is the coverage).

**reconciliation.csv** (R fable on the full 154-node hierarchy; the top-down methods on the 84-node geographic sub-hierarchy are not included)
`base_model` (ARIMA or ETS)`, method_order, method, is_reconciled, level_order, level_name, wape_pct, wape_ci_lo, wape_ci_hi, wape_ratio_vs_base,
wape_ratio_ci_lo, wape_ratio_ci_hi, better_than_base`. `method_order`: 1 base, 2 bottom-up, 3 OLS, 4 WLS structural, 5 WLS variance, 6 MinT shrink.

**calibration.csv** (the 11 scorecard models plus LightGBM + conformal and the R bottom-up, OLS and WLS reconciliations)
`model_label, model, model_family, is_reconciled, level_order, level_name, nominal_pct` (80 or 95)`, nominal_coverage, coverage, coverage_ci_lo, coverage_ci_hi,
miscalibration` (coverage minus nominal)`, mean_width_over_revenue`.

**planning_buffer.csv** (15 models with 90th-percentile forecasts, sums over the nodes of each level and the 24 test weeks)
`model_label, model, model_family, level_order, level_name, n_nodes, actual_revenue_usd, buffer_usd, buffer_pct_of_revenue` (q90 plan minus point plan, over revenue)`,
realised_share_within_q90` (share of node-weeks with actual at or below the q90 plan)`, q90_service_gap` (that share minus 0.90)`, ASSUMED_gross_margin, ASSUMED_carrying_cost,
ASSUMED_plan_quantile, ASSUMED_cost_point_plan_usd, ASSUMED_cost_q90_plan_usd, ASSUMED_cost_point_plan_pct_of_revenue, ASSUMED_cost_q90_plan_pct_of_revenue,
ASSUMED_cost_reduction_pct` (1 minus q90 cost over point cost). The ASSUMED_ parameters (25% margin lost on unmet revenue, 5% carrying cost on excess, q90) are scenario
inputs from `config/`, not estimates from the data.
