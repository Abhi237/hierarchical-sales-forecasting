-- 03_item_store_week: weekly units and revenue for every item x store. Missing price = item not on sale that week -> revenue 0.
-- A price must exist whenever units > 0; violations are counted in the cleaning log (expected 0).

CREATE OR REPLACE TABLE fact_item_store_week AS
WITH wu AS (
  SELECT l.series_id, m.w, m.wm_yr_wk, sum(l.units) AS units
  FROM stg_sales_long l JOIN map_day_week m USING (d_num)
  GROUP BY l.series_id, m.w, m.wm_yr_wk)
SELECT wu.series_id, wu.w, CAST(wu.units AS INTEGER) AS units, p.sell_price,
       (p.sell_price IS NOT NULL) AS price_listed,
       CAST(wu.units * coalesce(p.sell_price, 0) AS DECIMAL(18, 2)) AS revenue
FROM wu
JOIN dim_series s USING (series_id)
LEFT JOIN stg_prices p ON p.store_id = s.store_id AND p.item_id = s.item_id AND p.wm_yr_wk = wu.wm_yr_wk
ORDER BY wu.series_id, wu.w;

-- bottom level of the modelled hierarchy: store x department
CREATE OR REPLACE TABLE dim_bottom AS
SELECT CAST(row_number() OVER (ORDER BY store_id, dept_id) AS INTEGER) AS bottom_idx,
       store_id || '|' || dept_id AS bottom_id, state_id, store_id, cat_id, dept_id,
       count(*) AS n_items_stores
FROM dim_series GROUP BY state_id, store_id, cat_id, dept_id
ORDER BY store_id, dept_id;

CREATE OR REPLACE TABLE fact_bottom_week AS
SELECT b.bottom_id, f.w, CAST(sum(f.units) AS BIGINT) AS units, sum(f.revenue) AS revenue
FROM fact_item_store_week f
JOIN dim_series s USING (series_id)
JOIN dim_bottom b ON b.store_id = s.store_id AND b.dept_id = s.dept_id
GROUP BY b.bottom_id, f.w
ORDER BY b.bottom_id, f.w;
