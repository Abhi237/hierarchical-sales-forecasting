# tableau/web: flattened sources for Tableau Public web authoring

Seven CSVs, one per worksheet or text-tile source (five chart sources, plus `kpis.csv` and `findings.csv` for the story text). No joins, relationships or calculated fields are needed in Tableau: every value is pre-calculated by
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
| `kpis.csv` | 7 | one KPI per row | Section 1, headline KPI tiles (`kpi_display` is the formatted text, `kpi_value` the number, `kpi_note` the footnote) |
| `findings.csv` | 9 | one finding per row | Sections 2 to 6, the finding-style titles, callouts and recommendations (`section` 2 to 6, `finding_headline` as the title, `finding_detail` as the sentence below it) |

Story order for one long scrolling dashboard: (1) KPIs, (2) model by level heat map and the moving-average-vs-ETS callout, (3) forecast vs actual with the 80% band,
(4) reconciliation impact, (5) interval coverage vs the 80% target, (6) recommendations. Every number in `kpis.csv` and `findings.csv` is computed by
`scripts/build_tableau_web.py` from `tableau/data/` and checked by `tests/test_tableau_web.py`; none is typed. Colour semantics for `colour_group`:
Baseline grey, Statistical blue, Machine learning orange, Reconciled teal.

## Columns

**scorecard.csv** (models: 3 baselines, statsforecast ETS and 2 SARIMA, LightGBM quantile, fable ETS and ARIMA base and MinT; excluded: the duplicate
`lgbm_conformal` point forecast, the Python point-only reconciled variants and the geographic sub-hierarchy variants)
`model_label, model, model_family, colour_group, is_baseline (Yes/No), level_order, level_name, wape_pct, wape_ci_lo, wape_ci_hi` (95% bootstrap over series; over origins
for Total), `wape_ratio_vs_snaive, wape_ratio_ci_lo, wape_ratio_ci_hi, mase, rank_in_level` (1 = lowest WAPE among the 11 models, ties share a rank),
`is_best_in_level, better_than_snaive` (Yes when the ratio is below 1; Seasonal naive itself is No), `ma4_beats_ets_in_level` (Yes/No, same on every row of a level), `ma4_minus_ets_wape` (moving average WAPE minus ETS WAPE; negative means the moving average wins).

**forecast_explorer.csv** (levels 1-3: 14 nodes; models: Seasonal naive, 4-week moving average, SARIMA(0,1,1)(0,1,1)[52] (the SARIMA with the lowest mean test WAPE
over levels 1-3), LightGBM quantile, fable ARIMA MinT, fable ETS MinT; horizons 1 and 4 weeks)
`level_order, level_name, node_id, node_label, series_label` (model and node, `Model | Node`)`, model_order, model_label, model_family, colour_group, origin_week_idx, forecast_made_after, target_week_idx, target_week_start,
horizon_weeks, forecast_usd, actual_usd, lo80_usd, hi80_usd, band_width_usd` (hi80 minus lo80)`, lo95_usd, hi95_usd, in_80, in_95` (1 = actual inside the interval, 0 = outside; the mean is the coverage).

**reconciliation.csv** (R fable on the full 154-node hierarchy; the top-down methods on the 84-node geographic sub-hierarchy are not included)
`base_model` (ARIMA or ETS)`, method_order, method, colour_group, is_reconciled, level_order, level_name, wape_pct, wape_ci_lo, wape_ci_hi, wape_ratio_vs_base,
wape_ratio_ci_lo, wape_ratio_ci_hi, better_than_base`. `method_order`: 1 base, 2 bottom-up, 3 OLS, 4 WLS structural, 5 WLS variance, 6 MinT shrink.

**calibration.csv** (the 11 scorecard models plus LightGBM + conformal and the R bottom-up, OLS and WLS reconciliations)
`model_label, model, model_family, colour_group, is_reconciled, level_order, level_name, nominal_pct` (80 or 95)`, nominal_coverage, coverage, coverage_ci_lo, coverage_ci_hi,
miscalibration` (coverage minus nominal)`, abs_miscalibration, is_best_interval_in_level` (Yes for the model closest to nominal in that level and nominal, ties all Yes)`, under_covers` (Yes when coverage is below nominal minus 0.05)`, mean_width_over_revenue`.

**planning_buffer.csv** (15 models with 90th-percentile forecasts, sums over the nodes of each level and the 24 test weeks)
`model_label, model, model_family, colour_group, level_order, level_name, n_nodes, actual_revenue_usd, buffer_usd, buffer_pct_of_revenue` (q90 plan minus point plan, over revenue)`,
realised_share_within_q90` (share of node-weeks with actual at or below the q90 plan)`, q90_service_gap` (that share minus 0.90)`, ASSUMED_gross_margin, ASSUMED_carrying_cost,
ASSUMED_plan_quantile, ASSUMED_cost_point_plan_usd, ASSUMED_cost_q90_plan_usd, ASSUMED_cost_point_plan_pct_of_revenue, ASSUMED_cost_q90_plan_pct_of_revenue,
ASSUMED_cost_reduction_pct` (1 minus q90 cost over point cost). The ASSUMED_ parameters (25% margin lost on unmet revenue, 5% carrying cost on excess, q90) are scenario
inputs from `config/`, not estimates from the data.

**kpis.csv** `kpi_order, kpi_label, kpi_value, kpi_display, kpi_note`. KPIs: best WAPE at Total (and which model); relative WAPE cut vs seasonal naive at Total; levels won by the best
Total model; stores where the best Store-level model beats seasonal naive (per-store WAPE from `forecast_vs_actual.csv`, horizons 1 to 4); 80% coverage of the model closest to the
target at Total (from 24 forecasts, noisy); MinT shrink WAPE ratio vs the ARIMA base at Total; number of test origins. Definitions are in each `kpi_note`.

**findings.csv** `finding_order, section` (2 to 6)`, section_name, finding_type` (headline, callout or recommendation)`, finding_headline, finding_detail, model_label, level_name,
is_assumed` (Yes only for the row that quotes cost figures, whose headline starts with ASSUMED)`, evidence_file` (the export the numbers come from).
