"""Data-contract tests on the SQL layer, run on the small M5-format fixture (no raw data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def q(con, sql):
    return con.execute(sql).fetchall()


def test_series_and_day_counts(con):
    assert q(con, "SELECT count(*) FROM dim_series")[0][0] == 60
    assert q(con, "SELECT max(d_num) FROM stg_sales_long")[0][0] == 563
    assert q(con, "SELECT count(*) FROM stg_sales_long")[0][0] == 60 * 563


def test_no_null_or_negative_units(con):
    assert q(con, "SELECT count(*) FROM stg_sales_long WHERE units IS NULL OR units < 0")[0][0] == 0


def test_partial_trailing_week_dropped_and_logged(con):
    assert q(con, "SELECT max(w) FROM dim_week")[0][0] == 80
    log = dict(q(con, "SELECT metric, value FROM cleaning_log"))
    assert log["days_dropped_partial_week"] == 3
    assert log["complete_weeks_modelled"] == 80


def test_weeks_are_complete_and_contiguous(con):
    w = con.execute("SELECT w, n_days, wm_yr_wk FROM dim_week ORDER BY w").df()
    assert (w["n_days"] == 7).all()
    assert w["w"].tolist() == list(range(1, 81))
    assert (np.diff(w["wm_yr_wk"]) == 1).all()


def test_revenue_matches_independent_pandas_computation(con, fixture_dir):
    sales = pd.read_csv(fixture_dir / "sales_train_evaluation.csv")
    cal = pd.read_csv(fixture_dir / "calendar.csv")
    prices = pd.read_csv(fixture_dir / "sell_prices.csv")
    dcols = [f"d_{i}" for i in range(1, 564)]
    long = sales.melt(id_vars=["item_id", "store_id"], value_vars=dcols, var_name="d", value_name="units")
    long = long.merge(cal[["d", "wm_yr_wk"]], on="d").merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long["revenue"] = long["units"] * long["sell_price"].fillna(0.0)
    wk = long.groupby("wm_yr_wk")["revenue"].sum()
    complete = long.groupby("wm_yr_wk")["d"].nunique()
    expected = float(wk[complete[complete == 7].index].sum())
    got = float(q(con, "SELECT CAST(sum(revenue) AS DOUBLE) FROM fact_item_store_week")[0][0])
    assert got == pytest.approx(expected, abs=0.01)


def test_missing_price_means_zero_revenue_and_zero_units(con):
    assert q(con, "SELECT count(*) FROM fact_item_store_week WHERE NOT price_listed AND units > 0")[0][0] == 0
    assert q(con, "SELECT count(*) FROM fact_item_store_week WHERE NOT price_listed AND revenue <> 0")[0][0] == 0
    assert q(con, "SELECT count(*) FROM fact_item_store_week WHERE NOT price_listed")[0][0] > 0   # fixture has late launches


def test_store_closed_days_counted(con):
    log = dict(q(con, "SELECT metric, value FROM cleaning_log"))
    assert log["store_days_all_zero"] >= 8   # 4 stores x 2 closed days in the fixture


def test_cleaning_log_has_every_metric_once(con):
    m = [r[0] for r in q(con, "SELECT metric FROM cleaning_log")]
    assert len(m) == len(set(m)) == 24


def test_node_counts_per_level(con):
    got = dict(q(con, "SELECT level_id, count(*) FROM dim_node GROUP BY level_id"))
    assert got == {1: 1, 2: 3, 3: 4, 4: 2, 5: 3, 6: 6, 7: 9, 8: 8, 9: 12}


def test_summing_matrix_bridge(con):
    br = con.execute("SELECT node_id, count(*) AS n FROM bridge_node_bottom GROUP BY node_id").df().set_index("node_id")["n"]
    assert br["T"] == 12
    assert (br.loc[[n for n in br.index if n.startswith("S:") and "|D:" in n]] == 1).all()   # store x dept nodes cover one bottom series
    assert q(con, "SELECT count(*) FROM bridge_node_bottom WHERE level_id = 1")[0][0] == 12


def test_every_level_partitions_the_total(con):
    df = con.execute("""SELECT level_id, w, sum(revenue) AS r FROM fact_node_week GROUP BY level_id, w""").df()
    total = df[df["level_id"] == 1].set_index("w")["r"]
    for lvl in range(2, 10):
        x = df[df["level_id"] == lvl].set_index("w")["r"]
        assert (x - total).abs().max() == 0, f"level {lvl} does not sum to the total"


def test_children_sum_to_parent_exactly_for_every_edge(con):
    bad = q(con, """
        WITH s AS (SELECT e.parent_id, e.child_level, c.w, sum(c.revenue) AS rev, sum(c.units) AS units
                   FROM dim_edge e JOIN fact_node_week c ON c.node_id = e.child_id GROUP BY 1, 2, 3)
        SELECT count(*) FROM s JOIN fact_node_week p ON p.node_id = s.parent_id AND p.w = s.w
        WHERE s.rev <> p.revenue OR s.units <> p.units""")[0][0]
    assert bad == 0


def test_edges_cover_all_parent_child_level_pairs(con):
    pairs = {(a, b) for a, b in q(con, "SELECT DISTINCT parent_level, child_level FROM dim_edge")}
    assert (1, 2) in pairs and (2, 3) in pairs and (3, 9) in pairs and (4, 5) in pairs and (6, 8) in pairs and (8, 9) in pairs


def test_direct_and_bridge_aggregation_are_identical(con):
    n = q(con, "SELECT count(*) FROM (SELECT * FROM fact_node_week EXCEPT SELECT * FROM fact_node_week_bridge)")[0][0]
    m = q(con, "SELECT count(*) FROM (SELECT * FROM fact_node_week_bridge EXCEPT SELECT * FROM fact_node_week)")[0][0]
    assert n == 0 and m == 0
    assert q(con, "SELECT count(*) FROM fact_node_week")[0][0] == 48 * 80


def test_total_revenue_equals_item_level_sum(con):
    a = q(con, "SELECT sum(revenue) FROM fact_node_week WHERE level_id = 1")[0][0]
    b = q(con, "SELECT sum(revenue) FROM fact_item_store_week")[0][0]
    assert a == b


def test_node_state_share_sums_to_one(con):
    s = con.execute("SELECT node_id, sum(share) AS s FROM node_state_share GROUP BY node_id").df()
    assert np.allclose(s["s"], 1.0)


def test_item_selection_uses_only_the_selection_window(con):
    sel_end = q(con, "SELECT min(origin_w) FROM (SELECT 56 AS origin_w)")[0][0]
    df = con.execute(f"""SELECT s.item_id, s.store_id, s.series_id, sum(f.revenue) AS r FROM fact_item_store_week f JOIN dim_series s USING (series_id)
                         WHERE f.w <= {sel_end} GROUP BY ALL ORDER BY r DESC, s.series_id LIMIT 10""").df()
    chosen = con.execute("SELECT item_id, store_id FROM sel_item_store").df()
    assert set(map(tuple, df[["item_id", "store_id"]].values)) == set(map(tuple, chosen.values))


def test_item_layer_node_counts(con):
    got = dict(q(con, "SELECT level_id, count(*) FROM dim_item_node GROUP BY level_id"))
    assert got == {10: 5, 12: 10}
    assert q(con, "SELECT count(*) FROM fact_item_node_week")[0][0] == 15 * 80


def test_item_store_series_revenue_not_negative(con):
    assert q(con, "SELECT count(*) FROM fact_item_store_week WHERE revenue < 0")[0][0] == 0


def test_recent_daily_ties_to_weekly(con):
    d = con.execute("SELECT w, sum(revenue) AS r FROM fact_recent_daily WHERE level_id = 1 GROUP BY w ORDER BY w").df()
    wk = con.execute("SELECT w, revenue AS r FROM fact_node_week WHERE node_id = 'T' AND w >= (SELECT min(w) FROM fact_recent_daily) ORDER BY w").df()
    assert len(d) == 4
    assert np.allclose(d["r"].astype(float), wk["r"].astype(float))


def test_dim_week_covariates_present(con):
    w = con.execute("SELECT * FROM dim_week").df()
    assert w["is_christmas_week"].sum() == 2 and w["is_superbowl_week"].sum() == 2
    assert w[["snap_days_ca", "snap_days_tx", "snap_days_wi"]].max().max() <= 7
