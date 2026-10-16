-- 04_hierarchy: the nine aggregated M5 levels (levels 1-9), node dimension, summing matrix as a bridge table, and two
-- independent aggregation paths (direct GROUP BY vs bridge) that the data-contract tests prove identical.
-- Level ids follow the M5 competition: 1 Total, 2 State, 3 Store, 4 Category, 5 Department, 6 State x Category,
-- 7 State x Department, 8 Store x Category, 9 Store x Department (bottom of the modelled hierarchy).

CREATE OR REPLACE TABLE dim_node AS
WITH n AS (
  SELECT 1 AS level_id, 'Total' AS level_name, 'T' AS node_id, 'Total' AS node_label, NULL AS state_id, NULL AS store_id, NULL AS cat_id, NULL AS dept_id
  UNION ALL SELECT 2, 'State', 'ST:' || state_id, state_id, state_id, NULL, NULL, NULL FROM (SELECT DISTINCT state_id FROM dim_bottom)
  UNION ALL SELECT 3, 'Store', 'S:' || store_id, store_id, state_id, store_id, NULL, NULL FROM (SELECT DISTINCT state_id, store_id FROM dim_bottom)
  UNION ALL SELECT 4, 'Category', 'C:' || cat_id, cat_id, NULL, NULL, cat_id, NULL FROM (SELECT DISTINCT cat_id FROM dim_bottom)
  UNION ALL SELECT 5, 'Department', 'D:' || dept_id, dept_id, NULL, NULL, cat_id, dept_id FROM (SELECT DISTINCT cat_id, dept_id FROM dim_bottom)
  UNION ALL SELECT 6, 'State x Category', 'ST:' || state_id || '|C:' || cat_id, state_id || ' | ' || cat_id, state_id, NULL, cat_id, NULL FROM (SELECT DISTINCT state_id, cat_id FROM dim_bottom)
  UNION ALL SELECT 7, 'State x Department', 'ST:' || state_id || '|D:' || dept_id, state_id || ' | ' || dept_id, state_id, NULL, cat_id, dept_id FROM (SELECT DISTINCT state_id, cat_id, dept_id FROM dim_bottom)
  UNION ALL SELECT 8, 'Store x Category', 'S:' || store_id || '|C:' || cat_id, store_id || ' | ' || cat_id, state_id, store_id, cat_id, NULL FROM (SELECT DISTINCT state_id, store_id, cat_id FROM dim_bottom)
  UNION ALL SELECT 9, 'Store x Department', 'S:' || store_id || '|D:' || dept_id, store_id || ' | ' || dept_id, state_id, store_id, cat_id, dept_id FROM (SELECT DISTINCT state_id, store_id, cat_id, dept_id FROM dim_bottom))
SELECT CAST(row_number() OVER (ORDER BY level_id, node_id) AS INTEGER) AS node_idx, * FROM n ORDER BY level_id, node_id;

-- summing matrix S in long form: node n aggregates bottom series b
CREATE OR REPLACE TABLE bridge_node_bottom AS
SELECT n.node_id, n.level_id, b.bottom_id
FROM dim_node n JOIN dim_bottom b
  ON (n.state_id IS NULL OR n.state_id = b.state_id)
 AND (n.store_id IS NULL OR n.store_id = b.store_id)
 AND (n.level_id NOT IN (4, 6, 8) OR n.cat_id = b.cat_id)
 AND (n.level_id NOT IN (5, 7, 9) OR n.dept_id = b.dept_id)
ORDER BY n.level_id, n.node_id, b.bottom_id;

-- path A: direct aggregation of the bottom fact by each level's keys
CREATE OR REPLACE TABLE fact_node_week AS
WITH bw AS (
  SELECT f.w, f.units, f.revenue, b.state_id, b.store_id, b.cat_id, b.dept_id
  FROM fact_bottom_week f JOIN dim_bottom b USING (bottom_id))
SELECT 1 AS level_id, 'T' AS node_id, w, sum(units) AS units, sum(revenue) AS revenue FROM bw GROUP BY w
UNION ALL SELECT 2, 'ST:' || state_id, w, sum(units), sum(revenue) FROM bw GROUP BY state_id, w
UNION ALL SELECT 3, 'S:' || store_id, w, sum(units), sum(revenue) FROM bw GROUP BY store_id, w
UNION ALL SELECT 4, 'C:' || cat_id, w, sum(units), sum(revenue) FROM bw GROUP BY cat_id, w
UNION ALL SELECT 5, 'D:' || dept_id, w, sum(units), sum(revenue) FROM bw GROUP BY dept_id, w
UNION ALL SELECT 6, 'ST:' || state_id || '|C:' || cat_id, w, sum(units), sum(revenue) FROM bw GROUP BY state_id, cat_id, w
UNION ALL SELECT 7, 'ST:' || state_id || '|D:' || dept_id, w, sum(units), sum(revenue) FROM bw GROUP BY state_id, dept_id, w
UNION ALL SELECT 8, 'S:' || store_id || '|C:' || cat_id, w, sum(units), sum(revenue) FROM bw GROUP BY store_id, cat_id, w
UNION ALL SELECT 9, 'S:' || store_id || '|D:' || dept_id, w, sum(units), sum(revenue) FROM bw GROUP BY store_id, dept_id, w
ORDER BY 1, 2, 3;

-- path B: summing matrix x bottom vector (what the reconciliation code uses)
CREATE OR REPLACE TABLE fact_node_week_bridge AS
SELECT br.level_id, br.node_id, f.w, sum(f.units) AS units, sum(f.revenue) AS revenue
FROM bridge_node_bottom br JOIN fact_bottom_week f USING (bottom_id)
GROUP BY br.level_id, br.node_id, f.w
ORDER BY 1, 2, 3;

-- partition edges used by the coherence contract. A (parent, child level) pair is an edge group when the child nodes at that level
-- that sit inside the parent cover the parent's bottom series exactly once, so the children MUST sum to the parent.
CREATE OR REPLACE TABLE dim_edge AS
WITH bs AS (SELECT node_id, level_id, count(*) AS n_bottom FROM bridge_node_bottom GROUP BY node_id, level_id),
shared AS (
  SELECT p.node_id AS parent_id, p.level_id AS parent_level, c.node_id AS child_id, c.level_id AS child_level, count(*) AS n_shared
  FROM bridge_node_bottom p JOIN bridge_node_bottom c ON p.bottom_id = c.bottom_id AND p.level_id < c.level_id
  GROUP BY 1, 2, 3, 4),
inside AS (
  SELECT sh.* , cb.n_bottom AS child_n, pb.n_bottom AS parent_n
  FROM shared sh JOIN bs cb ON cb.node_id = sh.child_id JOIN bs pb ON pb.node_id = sh.parent_id
  WHERE sh.n_shared = cb.n_bottom),
grp AS (SELECT parent_id, child_level, sum(child_n) AS cover, any_value(parent_n) AS parent_n FROM inside GROUP BY 1, 2)
SELECT i.parent_id, i.parent_level, i.child_id, i.child_level
FROM inside i JOIN grp g USING (parent_id, child_level)
WHERE g.cover = g.parent_n
ORDER BY 2, 4, 1, 3;
