# Data dictionary

All money is US dollars. Week index `w` / `week_idx` runs 1..277 (week 1 starts Saturday 2011-01-29, week 277 ends Friday 2016-05-20).
An origin `o` means the model saw weeks 1..o and forecast weeks o+1..o+4. Revenue = units x weekly sell price; an item-store-week with
no listed price is not on sale and has revenue 0.

## Hierarchy (M5 levels)

| level_id | level_name | nodes | in modelled hierarchy |
|---|---|---|---|
| 1 | Total | 1 | yes |
| 2 | State | 3 | yes |
| 3 | Store | 10 | yes |
| 4 | Category | 3 | yes |
| 5 | Department | 7 | yes |
| 6 | State x Category | 9 | yes |
| 7 | State x Department | 21 | yes |
| 8 | Store x Category | 30 | yes |
| 9 | Store x Department | 70 | yes (bottom, 154 nodes in total) |
| 10 | Item (top 300 by revenue) | 300 | sample, not reconciled |
| 11 | Item x State | 9,147 | not modelled |
| 12 | Item x Store (top 600 by revenue) | 600 | sample, not reconciled |

`node_id` patterns: `T`, `ST:CA`, `S:CA_1`, `C:FOODS`, `D:FOODS_1`, `ST:CA|C:FOODS`, `ST:CA|D:FOODS_1`, `S:CA_1|C:FOODS`,
`S:CA_1|D:FOODS_1`, `I:FOODS_3_090`, `IS:FOODS_3_090|CA_1`.

## DuckDB tables (data/processed/m5.duckdb, git-ignored)

| Table | Grain | Key columns |
|---|---|---|
| stg_sales_long | item-store x day | series_id, d_num, units |
| dim_series | item-store | series_id, item_id, dept_id, cat_id, store_id, state_id |
| dim_week | complete week | w, wm_yr_wk, week_start, week_end, snap_days_ca/tx/wi, n_ev_* event counts, is_*_week flags, event_names |
| fact_item_store_week | item-store x week | units, sell_price (NULL = not listed), price_listed, revenue DECIMAL(18,2) |
| dim_bottom, fact_bottom_week | store x department x week | bottom_id, units, revenue |
| dim_node, bridge_node_bottom | 154 nodes, summing matrix S in long form | node_id, level_id, bottom_id |
| fact_node_week | node x week (direct GROUP BY) | node_id, w, units, revenue |
| fact_node_week_bridge | node x week (S x bottom) | must equal fact_node_week exactly |
| dim_edge | partition edges used by the coherence contract | parent_id, child_id |
| sel_item, sel_item_store, dim_item_node, fact_item_node_week | item layer sample | selection uses weeks 1..first origin only |
| node_state_share | node x state revenue mix (selection window) | share |
| cleaning_log | one row per counted fact | step, metric, value, note |
| fact_recent_daily | levels 1-3 x day, last 12 weeks | revenue, units |

## Tableau files (tableau/data/)

| File | Grain | Columns |
|---|---|---|
| hierarchy_nodes.csv | node | node_id, node_label, level_id, level_name, state, store, category, department, n_bottom_series, layer, total_revenue_usd, avg_weekly_revenue_usd, share_of_total_revenue_pct, selection_window_revenue_usd |
| actuals_weekly.csv | node x week (levels 1-9) | node_id, level_id, week_idx, week_start, revenue_usd, units |
| calendar_weeks.csv | week | week_idx, week_start, week_end, snap_days_ca/tx/wi, event_names, is_christmas_week, role |
| actuals_daily_recent.csv | node (levels 1-3) x day, last 12 weeks | level_id, node_id, date, week_idx, units, revenue, node_label, level_name |
| forecast_vs_actual.csv | model x node x test origin x horizon | model, model_label, model_family, level_id, level_name, node_id, node_label, origin_week_idx, forecast_made_after, horizon_weeks, target_week_idx, target_week_start, forecast_usd, lo80_usd, hi80_usd, lo95_usd, hi95_usd, actual_usd, error_usd (forecast minus actual), abs_error_usd, in_80_interval, in_95_interval |
| metric_scorecard.csv | model x level x split x metric | layer, model, model_label, model_family, level_id, level_name, split, metric, value, ci_lo, ci_hi (95% bootstrap), ci_method, n_nodes, n_origins, wape_origin_min, wape_origin_max |
| reconciliation_comparison.csv | method x base model x level x split x metric | engine, structure, base_model, method, model, model_label, level_id, level_name, split, metric (adds WAPE_ratio_vs_base), value, ci_lo, ci_hi, ci_method |
| coherence_gap.csv | model x level | mean_abs_rel_gap = mean of abs(F_node - S x F_bottom) / F_node for base forecasts |
| interval_coverage.csv | model x level x split x nominal | nominal, nominal_coverage, coverage, coverage_ci_lo/hi, miscalibration, mean_width_over_revenue |
| interval_coverage_by_horizon.csv | model x level x split x horizon | coverage_80, coverage_95, width_80/95_over_revenue, n |
| metrics_by_origin.csv | model x level x origin | WAPE, MASE, RMSSE, Bias, Pinball_scaled, Coverage_80/95, Width_80/95_rel |
| residual_diagnostics_nodes.csv / _summary.csv | R fable fit at the last test origin | lb_stat, lb_pvalue (Ljung-Box lag 52), lb_reject_5pct, lag1_acf, shapiro_p, share_ljung_box_rejected, ... |
| decision_planning.csv | model x node (test window) | plan and shortfall columns, ASSUMED_* cost columns |
| decision_sensitivity.csv | model x level x plan rule x ASSUMED margin x ASSUMED carrying cost | ASSUMED_cost_pct_of_revenue |
| origins_design.csv | origin | split, train_through, forecast window |
| cleaning_log.csv | counted fact | step, metric, value, note |

## Metric definitions

Computed once, in `src/hier_forecast/metrics.py`, for every model including the R fable ones (R forecasts are scored by the Python code).

| Metric | Definition |
|---|---|
| WAPE | sum of absolute errors / sum of actuals, pooled over nodes, origins and horizons of the group |
| MASE | mean over rows of abs error / in-sample one-step naive MAE of that node at that origin (weeks 1..origin) |
| RMSSE | mean over node-origins of sqrt(mean squared error / in-sample one-step naive MSE) |
| Bias | sum(forecast - actual) / sum(actual); positive = over-forecast |
| Pinball_scaled | mean pinball loss over quantiles 0.025, 0.1, 0.5, 0.9, 0.975, summed over rows, divided by sum of actuals |
| Coverage_80 / 95 | share of rows with actual inside the central interval |
| Width_80 / 95_rel | sum of interval widths / sum of actuals |
| WAPE_ratio_vs_snaive | WAPE of model / WAPE of seasonal naive, paired bootstrap interval (same resample for both) |
| WAPE_ratio_vs_base | WAPE of reconciled model / WAPE of its unreconciled base, paired bootstrap |

Bootstrap: 1,000 resamples, seed 7. The resampling unit is the node for levels with at least 3 nodes, and the origin for single-node
levels (Total). Intervals describe sampling variation over series (or origins), not model-fit uncertainty.

## Model names

| model | description |
|---|---|
| naive, snaive, ma4 | last value, value 52 weeks earlier, mean of last 4 weeks; Gaussian intervals from in-sample h-step residuals |
| ets | statsforecast AutoETS, season length 52 |
| sarima_111_010, sarima_airline | ARIMA(1,1,1)(0,1,0)[52] and ARIMA(0,1,1)(0,1,1)[52], orders fixed a priori |
| lgbm_q | one global LightGBM per quantile (0.025, 0.1, 0.5, 0.9, 0.975), horizon as a feature, target scaled by trailing 52-week mean |
| lgbm_conformal | LightGBM median with split-conformal intervals from earlier origins |
| r_ets_*, r_arima_* | R fable STL+ETS and dynamic harmonic regression ARIMA (4 Fourier pairs, period 52); suffix base, bu, ols, wls_struct, wls_var, mint |
| geo_r_*_td_fp, td_ap | top-down on the strictly hierarchical geographic sub-hierarchy (84 nodes) |
| py_<model>_<method> | Python numpy reconciliation of point forecasts (intervals not defined) |
