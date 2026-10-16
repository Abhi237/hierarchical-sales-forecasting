-- 01_stage: load the three raw M5 CSVs, build dimensions and a long daily fact. Placeholders {{...}} are filled by hier_forecast.db.
-- Revenue = units x weekly sell price. Prices are DECIMAL(9,2) so every sum below is exact (children sum to parents to the cent).

CREATE OR REPLACE TABLE stg_calendar AS
SELECT CAST(row_number() OVER (ORDER BY CAST(date AS DATE)) AS INTEGER) AS d_num,
       CAST(date AS DATE) AS date,
       CAST(wm_yr_wk AS INTEGER) AS wm_yr_wk,
       weekday, CAST(wday AS INTEGER) AS wday, CAST(month AS INTEGER) AS month, CAST(year AS INTEGER) AS year,
       event_name_1, event_type_1, event_name_2, event_type_2,
       CAST(snap_CA AS INTEGER) AS snap_CA, CAST(snap_TX AS INTEGER) AS snap_TX, CAST(snap_WI AS INTEGER) AS snap_WI
FROM read_csv('{{calendar_csv}}', header = true, nullstr = 'NA', sample_size = -1)
ORDER BY CAST(date AS DATE);

CREATE OR REPLACE TABLE stg_prices AS
SELECT store_id, item_id, CAST(wm_yr_wk AS INTEGER) AS wm_yr_wk, CAST(sell_price AS DECIMAL(9, 2)) AS sell_price
FROM read_csv('{{prices_csv}}', header = true, sample_size = -1);

CREATE OR REPLACE TABLE stg_sales_wide AS
SELECT item_id, dept_id, cat_id, store_id, state_id, COLUMNS('^d_[0-9]+$')
FROM read_csv('{{sales_csv}}', header = true, sample_size = -1);

-- series dimension: one row per item x store (30,490 in the full data), ids assigned in a fixed order
CREATE OR REPLACE TABLE dim_series AS
SELECT CAST(row_number() OVER (ORDER BY item_id, store_id) AS INTEGER) AS series_id,
       item_id, dept_id, cat_id, store_id, state_id
FROM stg_sales_wide
ORDER BY item_id, store_id;

-- long daily units, restricted to the first {{n_days}} day columns
CREATE OR REPLACE TABLE stg_sales_long AS
SELECT s.series_id, CAST(substr(u.d_col, 3) AS INTEGER) AS d_num, CAST(u.units AS INTEGER) AS units
FROM (UNPIVOT stg_sales_wide ON COLUMNS('^d_[0-9]+$') INTO NAME d_col VALUE units) u
JOIN dim_series s ON s.item_id = u.item_id AND s.store_id = u.store_id
WHERE CAST(substr(u.d_col, 3) AS INTEGER) <= {{n_days}};

DROP TABLE stg_sales_wide;
