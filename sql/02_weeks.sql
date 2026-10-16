-- 02_weeks: complete-week calendar with SNAP and event covariates. M5 weeks (wm_yr_wk) run Saturday to Friday.
-- Only weeks with all 7 days inside the sales window are modelled (the trailing partial week is dropped and logged).

CREATE OR REPLACE TABLE stg_calendar_scope AS
SELECT * FROM stg_calendar WHERE d_num <= {{n_days}};

CREATE OR REPLACE TABLE dim_week AS
WITH wk AS (
  SELECT wm_yr_wk, count(*) AS n_days, min(date) AS week_start, max(date) AS week_end,
         sum(snap_CA) AS snap_days_ca, sum(snap_TX) AS snap_days_tx, sum(snap_WI) AS snap_days_wi,
         sum(CASE WHEN event_type_1 = 'Sporting' THEN 1 ELSE 0 END + CASE WHEN event_type_2 = 'Sporting' THEN 1 ELSE 0 END) AS n_ev_sporting,
         sum(CASE WHEN event_type_1 = 'Cultural' THEN 1 ELSE 0 END + CASE WHEN event_type_2 = 'Cultural' THEN 1 ELSE 0 END) AS n_ev_cultural,
         sum(CASE WHEN event_type_1 = 'National' THEN 1 ELSE 0 END + CASE WHEN event_type_2 = 'National' THEN 1 ELSE 0 END) AS n_ev_national,
         sum(CASE WHEN event_type_1 = 'Religious' THEN 1 ELSE 0 END + CASE WHEN event_type_2 = 'Religious' THEN 1 ELSE 0 END) AS n_ev_religious,
         max(CASE WHEN event_name_1 = 'Christmas' OR event_name_2 = 'Christmas' THEN 1 ELSE 0 END) AS is_christmas_week,
         max(CASE WHEN event_name_1 = 'Thanksgiving' OR event_name_2 = 'Thanksgiving' THEN 1 ELSE 0 END) AS is_thanksgiving_week,
         max(CASE WHEN event_name_1 = 'SuperBowl' OR event_name_2 = 'SuperBowl' THEN 1 ELSE 0 END) AS is_superbowl_week,
         max(CASE WHEN event_name_1 = 'Easter' OR event_name_2 = 'Easter' THEN 1 ELSE 0 END) AS is_easter_week,
         string_agg(DISTINCT coalesce(event_name_1, ''), '; ' ORDER BY coalesce(event_name_1, '')) FILTER (WHERE event_name_1 IS NOT NULL) AS event_names
  FROM stg_calendar_scope GROUP BY wm_yr_wk)
SELECT CAST(row_number() OVER (ORDER BY wm_yr_wk) AS INTEGER) AS w, * FROM wk WHERE n_days = 7 ORDER BY wm_yr_wk;

-- day -> week index map (complete weeks only)
CREATE OR REPLACE TABLE map_day_week AS
SELECT c.d_num, c.date, c.wm_yr_wk, w.w
FROM stg_calendar_scope c JOIN dim_week w USING (wm_yr_wk)
ORDER BY c.d_num;
