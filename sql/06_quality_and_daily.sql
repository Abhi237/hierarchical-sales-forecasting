-- 06_quality_and_daily: cleaning log (every count is computed, none typed in) and the recent daily view.

CREATE OR REPLACE TABLE series_first_sale AS
SELECT series_id, min(d_num) FILTER (WHERE units > 0) AS first_sale_d FROM stg_sales_long GROUP BY series_id;

CREATE OR REPLACE TABLE stockout_runs AS
-- proxy: a listed-price item-store series with >= 4 consecutive zero-unit weeks after its first sale (possible stockout or delisting)
WITH x AS (
  SELECT f.series_id, f.w, f.units, f.price_listed,
         sum(CASE WHEN f.units > 0 THEN 1 ELSE 0 END) OVER (PARTITION BY f.series_id ORDER BY f.w) AS grp
  FROM fact_item_store_week f),
z AS (
  SELECT series_id, grp, count(*) AS run_len, min(w) AS w0
  FROM x WHERE units = 0 AND price_listed GROUP BY series_id, grp)
SELECT z.series_id, z.grp, z.run_len FROM z WHERE z.grp > 0 AND z.run_len >= 4;

CREATE OR REPLACE TABLE cleaning_log AS
SELECT 1 AS step, 'item_store_series' AS metric, CAST(count(*) AS BIGINT) AS value, 'rows in dim_series (item x store)' AS note FROM dim_series
UNION ALL SELECT 2, 'day_columns_loaded', max(d_num), 'd_1 .. d_n of the sales file' FROM stg_sales_long
UNION ALL SELECT 3, 'item_store_day_cells', count(*), 'series x days (long format)' FROM stg_sales_long
UNION ALL SELECT 4, 'null_or_negative_units', count(*) FILTER (WHERE units IS NULL OR units < 0), 'must be 0' FROM stg_sales_long
UNION ALL SELECT 5, 'zero_unit_cells', count(*) FILTER (WHERE units = 0), 'kept as true zeros (demand or stockout cannot be separated)' FROM stg_sales_long
UNION ALL SELECT 6, 'leading_zero_cells', (SELECT count(*) FROM stg_sales_long l JOIN series_first_sale f USING (series_id) WHERE l.d_num < coalesce(f.first_sale_d, 1000000)),
       'zero cells before an item-store first sale (not yet on sale); kept, revenue 0' 
UNION ALL SELECT 7, 'series_with_no_sales_ever', count(*) FILTER (WHERE first_sale_d IS NULL), 'item-store series with all zeros' FROM series_first_sale
UNION ALL SELECT 8, 'calendar_days_in_scope', count(*), 'calendar rows with d_num <= n_days' FROM stg_calendar_scope
UNION ALL SELECT 9, 'complete_weeks_modelled', count(*), 'weeks with all 7 days in scope' FROM dim_week
UNION ALL SELECT 10, 'days_dropped_partial_week', (SELECT count(*) FROM stg_calendar_scope) - (SELECT count(*) FROM map_day_week), 'trailing days of an incomplete week, excluded from weekly series'
UNION ALL SELECT 11, 'item_store_weeks', count(*), 'rows in fact_item_store_week' FROM fact_item_store_week
UNION ALL SELECT 12, 'item_store_weeks_price_missing', count(*) FILTER (WHERE NOT price_listed), 'weeks without a listed price (not on sale); revenue = 0' FROM fact_item_store_week
UNION ALL SELECT 13, 'price_missing_but_units_positive', count(*) FILTER (WHERE NOT price_listed AND units > 0), 'units sold with no price: revenue would be unknown (must be 0)' FROM fact_item_store_week
UNION ALL SELECT 14, 'price_listed_zero_unit_weeks', count(*) FILTER (WHERE price_listed AND units = 0), 'listed but no sales (true zero demand or stockout), kept as zero revenue' FROM fact_item_store_week
UNION ALL SELECT 15, 'stockout_proxy_runs', count(*), 'runs of >= 4 consecutive zero weeks with a listed price after first sale' FROM stockout_runs
UNION ALL SELECT 16, 'stockout_proxy_weeks', coalesce(sum(run_len), 0), 'weeks inside those runs' FROM stockout_runs
UNION ALL SELECT 17, 'store_days_all_zero', (SELECT count(*) FROM (SELECT s.store_id, l.d_num FROM stg_sales_long l JOIN dim_series s USING (series_id) GROUP BY 1, 2 HAVING sum(l.units) = 0)),
       'store-days with zero units in every item (store closed, e.g. Christmas); kept as zero revenue'
UNION ALL SELECT 18, 'total_units', sum(units), 'sum of all units, weekly fact' FROM fact_item_store_week
UNION ALL SELECT 19, 'total_revenue_usd', CAST(round(sum(revenue)) AS BIGINT), 'rounded to dollars' FROM fact_item_store_week
UNION ALL SELECT 20, 'bottom_series', count(*), 'store x department bottom series' FROM dim_bottom
UNION ALL SELECT 21, 'hierarchy_nodes', count(*), 'aggregated nodes, levels 1-9' FROM dim_node
UNION ALL SELECT 22, 'item_layer_nodes', count(*), 'sampled item and item x store nodes (levels 10 and 12)' FROM dim_item_node
UNION ALL SELECT 23, 'item_layer_revenue_share_selection_window_pct_x100', CAST(round(10000 * (SELECT sum(selection_revenue) FROM sel_item_store) / (SELECT sum(revenue) FROM fact_item_store_week WHERE w <= {{selection_end_week}}))  AS BIGINT), 'top item x store series share of all item-store revenue in the selection window (percent x 100)'
UNION ALL SELECT 24, 'item_sample_revenue_share_selection_window_pct_x100', CAST(round(10000 * (SELECT sum(selection_revenue) FROM sel_item) / (SELECT sum(revenue) FROM fact_item_store_week WHERE w <= {{selection_end_week}}))  AS BIGINT), 'top items share of revenue in the selection window (percent x 100)'
ORDER BY step;

-- daily view for the last {{recent_daily_weeks}} complete weeks, levels 1-3 (total, state, store)
CREATE OR REPLACE TABLE fact_recent_daily AS
WITH last_w AS (SELECT max(w) AS lw FROM dim_week),
d AS (
  SELECT l.d_num, m.date, m.w, s.state_id, s.store_id, l.units, s.series_id, m.wm_yr_wk
  FROM stg_sales_long l JOIN map_day_week m USING (d_num) JOIN dim_series s USING (series_id), last_w
  WHERE m.w > last_w.lw - {{recent_daily_weeks}}),
r AS (
  SELECT d.date, d.w, d.state_id, d.store_id, d.units, d.units * coalesce(p.sell_price, 0) AS revenue
  FROM d JOIN dim_series s USING (series_id)
  LEFT JOIN stg_prices p ON p.store_id = s.store_id AND p.item_id = s.item_id AND p.wm_yr_wk = d.wm_yr_wk)
SELECT 1 AS level_id, 'T' AS node_id, date, w, CAST(sum(units) AS BIGINT) AS units, CAST(sum(revenue) AS DECIMAL(18, 2)) AS revenue FROM r GROUP BY date, w
UNION ALL SELECT 2, 'ST:' || state_id, date, w, sum(units), CAST(sum(revenue) AS DECIMAL(18, 2)) FROM r GROUP BY state_id, date, w
UNION ALL SELECT 3, 'S:' || store_id, date, w, sum(units), CAST(sum(revenue) AS DECIMAL(18, 2)) FROM r GROUP BY store_id, date, w
ORDER BY 1, 2, 3;
