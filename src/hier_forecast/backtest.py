from __future__ import annotations

import time

import pandas as pd

from .baselines import baselines_frame
from .config import origins, rpath
from .data import load_panel
from .db import connect
from .models_lgbm import conformal_frame, lgbm_quantile_frame
from .models_stats import stats_frame

HIER_STAT_MODELS = ["ets", "sarima_111_010", "sarima_airline"]


def run_layer(cfg: dict, layer: str, verbose: bool = True) -> pd.DataFrame:
    """Rolling-origin backtest for one layer. Every model is refit at each origin with data up to that origin only."""
    con = connect(cfg, read_only=True)
    try:
        panel = load_panel(con, layer)
    finally:
        con.close()
    b = cfg["backtest"]
    H, m = b["horizon"], b["season"]
    org = origins(cfg, panel.T)
    allo = org["all"]
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"[{layer}] {msg} ({time.time() - t0:.0f}s)", flush=True)

    frames = [baselines_frame(panel.Y, allo, H, m, b["ma_window"], layer)]
    log("baselines done")
    wanted = b["item_layer_models"] if layer == "item" else None
    stat_names = HIER_STAT_MODELS if layer == "hier" else ["ets"]
    if wanted is not None:
        stat_names = [n for n in stat_names if n in wanted]
    if stat_names:
        frames.append(stats_frame(panel.Y, allo, H, b["step"], m, stat_names, layer, n_jobs=cfg.get("stats_n_jobs", 4)))
        log("statsforecast done")
    base = lgbm_quantile_frame(panel, allo, cfg, layer)
    log("lightgbm quantiles done")
    conf = conformal_frame(base, panel, allo, cfg, layer)
    frames.append(base.drop(columns=["_scale"]))
    frames.append(conf)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["model", "origin", "h", "node_idx"], kind="mergesort").reset_index(drop=True)
    for c in ["yhat", "lo80", "hi80", "lo95", "hi95"]:
        out[c] = out[c].clip(lower=0.0)
    return out


def run_backtest(cfg: dict, layers: list[str] | None = None) -> None:
    out_dir = rpath(cfg, cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer in layers or ["hier", "item"]:
        df = run_layer(cfg, layer)
        df.to_parquet(out_dir / f"forecasts_{layer}.parquet", index=False)
