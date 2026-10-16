-- 05_item_layer: bounded, reproducible bottom-of-the-pyramid sample (M5 levels 10 and 12).
-- Selection uses ONLY weeks 1..{{selection_end_week}} (the last week before the first backtest origin), so no future information
-- decides which series are modelled. Ties are broken by id. Excludes: all other items (levels 10-12 are not exhaustive), so the
-- item layer is NOT coherent with the aggregated levels and is evaluated separately (no reconciliation).

CREATE OR REPLACE TABLE sel_item_store AS
SELECT s.series_id, s.item_id, s.store_id, s.dept_id, s.cat_id, s.state_id, sum(f.revenue) AS selection_revenue
FROM fact_item_store_week f JOIN dim_series s USING (series_id)
WHERE f.w <= {{selection_end_week}}
GROUP BY ALL
ORDER BY sum(f.revenue) DESC, s.series_id
LIMIT {{top_item_stores}};

CREATE OR REPLACE TABLE sel_item AS
SELECT s.item_id, any_value(s.dept_id) AS dept_id, any_value(s.cat_id) AS cat_id, sum(f.revenue) AS selection_revenue
FROM fact_item_store_week f JOIN dim_series s USING (series_id)
WHERE f.w <= {{selection_end_week}}
GROUP BY s.item_id
ORDER BY sum(f.revenue) DESC, s.item_id
LIMIT {{top_items}};

CREATE OR REPLACE TABLE dim_item_node AS
SELECT 10 AS level_id, 'Item' AS level_name, 'I:' || item_id AS node_id, item_id AS node_label, cat_id, dept_id, NULL AS store_id, NULL AS state_id, selection_revenue
FROM sel_item
UNION ALL
SELECT 12, 'Item x Store', 'IS:' || item_id || '|' || store_id, item_id || ' | ' || store_id, cat_id, dept_id, store_id, state_id, selection_revenue
FROM sel_item_store
ORDER BY 1, 3;

CREATE OR REPLACE TABLE fact_item_node_week AS
SELECT 10 AS level_id, 'I:' || s.item_id AS node_id, f.w, sum(f.units) AS units, sum(f.revenue) AS revenue
FROM fact_item_store_week f JOIN dim_series s USING (series_id) JOIN sel_item i ON i.item_id = s.item_id
GROUP BY s.item_id, f.w
UNION ALL
SELECT 12, 'IS:' || s.item_id || '|' || s.store_id, f.w, f.units, f.revenue
FROM fact_item_store_week f JOIN dim_series s USING (series_id) JOIN sel_item_store i ON i.series_id = s.series_id
ORDER BY 1, 2, 3;
