-- 07_state_share: static state mix of every node, estimated on the selection window only (weeks 1..{{selection_end_week}}).
-- Used to turn the three state SNAP calendars into one node-specific SNAP-days covariate without peeking at the future.

CREATE OR REPLACE TABLE node_state_share AS
WITH hb AS (
  SELECT br.node_id, b.state_id, sum(f.revenue) AS rev
  FROM bridge_node_bottom br JOIN dim_bottom b USING (bottom_id) JOIN fact_bottom_week f USING (bottom_id)
  WHERE f.w <= {{selection_end_week}} GROUP BY br.node_id, b.state_id),
it AS (
  SELECT 'I:' || s.item_id AS node_id, s.state_id, sum(f.revenue) AS rev
  FROM fact_item_store_week f JOIN dim_series s USING (series_id) JOIN sel_item i ON i.item_id = s.item_id
  WHERE f.w <= {{selection_end_week}} GROUP BY s.item_id, s.state_id),
st AS (
  SELECT 'IS:' || item_id || '|' || store_id AS node_id, state_id, 1.0 AS rev FROM sel_item_store),
allr AS (SELECT * FROM hb UNION ALL SELECT * FROM it UNION ALL SELECT * FROM st)
SELECT node_id, state_id,
       CASE WHEN sum(rev) OVER (PARTITION BY node_id) > 0 THEN rev / sum(rev) OVER (PARTITION BY node_id) ELSE 0 END AS share
FROM allr
ORDER BY node_id, state_id;
