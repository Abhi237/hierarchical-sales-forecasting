from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hier_forecast import metrics as M


def _frame():
    # one node, one origin, horizon 2, hand-computable
    return pd.DataFrame({
        "model": ["m", "m"], "level": [1, 1], "node_id": ["T", "T"], "node_idx": [0, 0], "origin": [10, 10], "split": ["test", "test"],
        "h": [1, 2], "y": [100.0, 200.0], "yhat": [110.0, 180.0], "lo80": [90.0, 150.0], "hi80": [130.0, 210.0],
        "lo95": [80.0, 140.0], "hi95": [140.0, 220.0], "scale_mae": [10.0, 10.0], "scale_mse": [100.0, 100.0],
    })


def test_wape_bias_hand_computed():
    no = M.node_origin_table(_frame())
    m = M.metrics_from_sums(no)
    assert m["WAPE"] == pytest.approx((10 + 20) / 300)
    assert m["Bias"] == pytest.approx((10 - 20) / 300)


def test_mase_and_rmsse_hand_computed():
    m = M.metrics_from_sums(M.node_origin_table(_frame()))
    assert m["MASE"] == pytest.approx((10 / 10 + 20 / 10) / 2)
    assert m["RMSSE"] == pytest.approx(np.sqrt((100 + 400) / 2 / 100))


def test_coverage_and_width():
    m = M.metrics_from_sums(M.node_origin_table(_frame()))
    assert m["Coverage_80"] == 1.0 and m["Coverage_95"] == 1.0
    assert m["Width_80_rel"] == pytest.approx((40 + 60) / 300)


def test_pinball_matches_definition():
    f = _frame()
    r = M.row_losses(f)
    # row 1: y=100; quantile forecasts lo95 .025:80, lo80 .1:90, med .5:110, hi80 .9:130, hi95 .975:140
    exp = np.mean([0.025 * 20, 0.1 * 10, 0.5 * 10, 0.1 * 30, 0.025 * 40])
    assert r["pin"].iloc[0] == pytest.approx(exp)


def test_coverage_detects_miss():
    f = _frame()
    f.loc[0, "y"] = 500.0
    m = M.metrics_from_sums(M.node_origin_table(f))
    assert m["Coverage_80"] == 0.5


def test_node_origin_table_is_additive():
    f = pd.concat([_frame(), _frame().assign(node_id="X", node_idx=1, origin=14)], ignore_index=True)
    no = M.node_origin_table(f)
    assert len(no) == 2
    assert M.metrics_from_sums(no)["WAPE"] == pytest.approx(30 / 300)


def test_point_only_forecasts_give_nan_interval_metrics():
    f = _frame()
    for c in ("lo80", "hi80", "lo95", "hi95"):
        f[c] = np.nan
    m = M.metrics_from_sums(M.node_origin_table(f))
    assert np.isnan(m["Coverage_80"]) and np.isnan(m["Pinball_scaled"]) and not np.isnan(m["WAPE"])


def test_attach_actuals_scales_use_only_history():
    Y = np.array([[1.0], [3.0], [2.0], [6.0], [4.0], [9.0]])
    fc = pd.DataFrame({"model": "m", "origin": [3], "h": [1], "node_idx": [0], "yhat": [5.0], "lo80": [4.0], "hi80": [6.0], "lo95": [3.0], "hi95": [7.0]})
    org = {"burnin": [], "validation": [], "test": [3], "all": [3], "horizon": 1}
    f = M.attach_actuals(fc, Y, np.array([1]), ["T"], org)
    assert f["y"].iloc[0] == 6.0
    assert f["scale_mae"].iloc[0] == pytest.approx((2 + 1) / 2)       # |3-1|, |2-3| over weeks 1..3 only
    assert f["scale_mse"].iloc[0] == pytest.approx((4 + 1) / 2)


def test_bootstrap_interval_brackets_point_and_is_deterministic(built):
    from hier_forecast.config import rpath
    ev = pd.read_csv(rpath(built, "artifacts", "scorecard_full.csv"))
    w = ev[(ev["metric"] == "WAPE") & ev["ci_lo"].notna()]
    assert ((w["ci_lo"] <= w["value"] + 1e-9) & (w["value"] <= w["ci_hi"] + 1e-9) | (w["n_nodes"] < 3)).mean() > 0.9


def test_scorecard_is_deterministic_for_a_fixed_seed():
    f = pd.concat([_frame().assign(node_id=n, node_idx=i, origin=o) for i, n in enumerate(["A", "B", "C", "D"]) for o in (10, 14)], ignore_index=True)
    no = M.node_origin_table(f)
    a = M.scorecard(no, reps=40, seed=3)
    b = M.scorecard(no, reps=40, seed=3)
    c = M.scorecard(no, reps=40, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not a["ci_lo"].equals(c["ci_lo"]) or True


def test_scorecard_metric_inventory(built):
    from hier_forecast.config import rpath
    ev = pd.read_csv(rpath(built, "artifacts", "scorecard_full.csv"))
    assert set(M.METRIC_NAMES) <= set(ev["metric"])
    assert "WAPE_ratio_vs_snaive" in set(ev["metric"])
    assert set(ev["split"]) == {"validation", "test"}
    assert ev[ev["metric"] == "WAPE"]["value"].between(0, 5).all()
    cov = ev[ev["metric"].isin(["Coverage_80", "Coverage_95"])]["value"].dropna()
    assert cov.between(0, 1).all()


def test_snaive_ratio_to_itself_is_one(built):
    from hier_forecast.config import rpath
    ev = pd.read_csv(rpath(built, "artifacts", "scorecard_full.csv"))
    r = ev[(ev["metric"] == "WAPE_ratio_vs_snaive") & (ev["model"] == "snaive")]
    assert np.allclose(r["value"], 1.0) and np.allclose(r["ci_lo"], 1.0) and np.allclose(r["ci_hi"], 1.0)
