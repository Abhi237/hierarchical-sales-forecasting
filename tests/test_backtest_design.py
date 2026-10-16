"""Rolling-origin design and leakage tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from hier_forecast.baselines import baseline_forecast
from hier_forecast.config import load_config, origins, split_of
from hier_forecast.data import load_panel
from hier_forecast.features import build_table
from hier_forecast.models_lgbm import conformal_frame


def test_origin_design_real_config():
    cfg = load_config()
    o = origins(cfg, 277)
    assert o["all"][-1] == 273 and o["all"][0] == 197 and len(o["all"]) == 20
    assert o["test"] == [253, 257, 261, 265, 269, 273]
    assert o["validation"] == [229, 233, 237, 241, 245, 249]
    assert len(o["burnin"]) == 8


def test_origins_ordered_nonoverlapping_and_test_is_latest(built, con):
    panel = load_panel(con, "hier")
    o = origins(built, panel.T)
    allo = o["all"]
    assert allo == sorted(allo) and len(set(allo)) == len(allo)
    assert max(o["burnin"]) < min(o["validation"]) < max(o["validation"]) < min(o["test"])
    assert all(b - a == built["backtest"]["step"] for a, b in zip(allo, allo[1:], strict=False))
    assert allo[-1] + built["backtest"]["horizon"] == panel.T   # the final origin forecasts the last observed weeks


def test_no_random_split_anywhere():
    import pathlib
    src = "\n".join(p.read_text() for p in pathlib.Path("src").rglob("*.py"))
    assert "train_test_split" not in src and "shuffle" not in src.lower() or "shuffle=False" in src


def test_split_of_raises_for_unknown_origin(built, con):
    panel = load_panel(con, "hier")
    o = origins(built, panel.T)
    try:
        split_of(1, o)
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_baseline_forecast_ignores_future_values(con):
    panel = load_panel(con, "hier")
    Y = panel.Y.copy()
    o = 56
    a = baseline_forecast(Y, o, 4, "snaive", 13)
    Y2 = Y.copy()
    Y2[o:] = Y2[o:] * 50 + 1e6       # destroy everything after the origin
    b = baseline_forecast(Y2, o, 4, "snaive", 13)
    assert np.allclose(a["point"], b["point"]) and np.allclose(a["sd"], b["sd"], equal_nan=True)


def test_features_do_not_depend_on_data_after_origin(con):
    panel = load_panel(con, "hier")
    t1 = build_table(panel, 4, 13)
    p2 = load_panel(con, "hier")
    cut = 50
    p2.Y[cut:] = p2.Y[cut:] * 3 + 77.0
    p2.U[cut:] = p2.U[cut:] * 3 + 5.0
    t2 = build_table(p2, 4, 13)
    m = (t1["s"] <= cut) & (t1["s"] + t1["h"] <= cut)       # rows whose origin and target are both before the change
    assert m.sum() > 100
    assert np.allclose(t1["X"][m], t2["X"][m], equal_nan=True)
    m2 = (t1["s"] <= cut) & (t1["s"] + t1["h"] > cut)       # rows with a future target: features still identical, target differs
    assert np.allclose(t1["X"][m2], t2["X"][m2], equal_nan=True)
    assert not np.allclose(t1["y"][m2], t2["y"][m2])


def test_feature_table_shapes_and_scaling(con):
    panel = load_panel(con, "hier")
    t = build_table(panel, 4, 13)
    assert t["X"].shape[0] == len(t["y"]) == len(t["s"])
    assert set(np.unique(t["h"])) == {1, 2, 3, 4}
    assert (t["scale"] >= 1.0).all()
    assert t["X"].shape[1] == len(t["names"])


def test_forecast_frame_structure(built):
    from hier_forecast.config import rpath
    f = pd.read_parquet(rpath(built, built["data"]["processed_dir"], "forecasts_hier.parquet"))
    assert {"naive", "snaive", "ma4", "ets", "sarima_111_010", "sarima_airline", "lgbm_q", "lgbm_conformal"} <= set(f["model"])
    assert set(f["h"]) == {1, 2, 3, 4}
    assert f[["yhat", "lo80", "hi80", "lo95", "hi95"]].notna().all().all()


def test_intervals_are_ordered_and_nonnegative(built):
    from hier_forecast.config import rpath
    f = pd.read_parquet(rpath(built, built["data"]["processed_dir"], "forecasts_hier.parquet"))
    assert (f["lo95"] <= f["lo80"] + 1e-6).all() and (f["lo80"] <= f["yhat"] + 1e-6).all()
    assert (f["yhat"] <= f["hi80"] + 1e-6).all() and (f["hi80"] <= f["hi95"] + 1e-6).all()
    assert (f[["yhat", "lo80", "hi80", "lo95", "hi95"]] >= 0).all().all()


def test_every_model_forecasts_every_node_at_every_origin(built, con):
    from hier_forecast.config import rpath
    panel = load_panel(con, "hier")
    f = pd.read_parquet(rpath(built, built["data"]["processed_dir"], "forecasts_hier.parquet"))
    o = origins(built, panel.T)
    for m, g in f.groupby("model"):
        want = set(o["all"]) if m != "lgbm_conformal" else set(o["all"][1:])
        assert set(g["origin"]) == want, m
        assert g.groupby(["origin", "h"]).size().eq(panel.N).all(), m


def test_conformal_uses_only_earlier_origins(built, con):
    """Changing the actuals of the LAST origin's target weeks must not change that origin's conformal intervals."""
    from hier_forecast.config import rpath
    from hier_forecast.models_lgbm import lgbm_quantile_frame
    panel = load_panel(con, "hier")
    o = origins(built, panel.T)
    base = lgbm_quantile_frame(panel, o["all"], built, "hier")
    c1 = conformal_frame(base, panel, o["all"], built, "hier")
    p2 = load_panel(con, "hier")
    last = o["all"][-1]
    p2.Y[last:] = p2.Y[last:] * 10 + 1000
    c2 = conformal_frame(base, p2, o["all"], built, "hier")
    a = c1[c1["origin"] == last].sort_values(["h", "node_idx"])
    b = c2[c2["origin"] == last].sort_values(["h", "node_idx"])
    assert np.allclose(a[["lo80", "hi80", "lo95", "hi95"]].to_numpy(), b[["lo80", "hi80", "lo95", "hi95"]].to_numpy())
    assert rpath(built, ".").exists()


def test_lgbm_training_rows_are_fully_observed_at_origin(con):
    panel = load_panel(con, "hier")
    t = build_table(panel, 4, 13)
    origin = 60
    tr = (t["s"] + t["h"] <= origin)
    assert (t["s"][tr] + t["h"][tr]).max() <= origin
    assert tr.sum() > 0
