"""Global LightGBM with quantile objectives, plus split-conformal intervals around the median forecast."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_table

Q_COLS = {0.025: "lo95", 0.1: "lo80", 0.5: "yhat", 0.9: "hi80", 0.975: "hi95"}


def _params(cfg: dict, alpha: float, seed: int) -> dict:
    p = dict(cfg["lgbm"])
    p.pop("max_train_weeks", None)
    return dict(objective="quantile", alpha=alpha, random_state=seed, deterministic=True, force_row_wise=True,
                verbose=-1, **p)


def lgbm_quantile_frame(panel, origins: list[int], cfg: dict, layer: str) -> pd.DataFrame:
    import lightgbm as lgb

    b = cfg["backtest"]
    H, m = b["horizon"], b["season"]
    tab = build_table(panel, H, m)
    X, y, scale, s, h, node = tab["X"], tab["y"], tab["scale"], tab["s"], tab["h"], tab["node"]
    maxw = cfg["lgbm"]["max_train_weeks"]
    recs = []
    for o in origins:
        tr = (s + h <= o) & (s >= o - H - maxw + 1)
        te = (s == o)
        preds = {}
        for q in b["quantiles"]:
            mdl = lgb.LGBMRegressor(**_params(cfg, q, cfg["seed"]))
            mdl.fit(X[tr], y[tr])
            preds[q] = mdl.predict(X[te])
        P = np.column_stack([preds[q] for q in b["quantiles"]])
        P = np.sort(P, axis=1)                         # remove quantile crossing (rearrangement)
        P = np.maximum(P, 0.0) * scale[te][:, None]    # back to revenue dollars, revenue cannot be negative
        d = pd.DataFrame({"layer": layer, "model": "lgbm_q", "origin": o, "h": h[te], "node_idx": node[te]})
        for j, q in enumerate(b["quantiles"]):
            d[Q_COLS[q]] = P[:, j]
        d["_scale"] = scale[te]
        recs.append(d)
    out = pd.concat(recs, ignore_index=True)
    return out


def conformal_frame(base: pd.DataFrame, panel, origins: list[int], cfg: dict, layer: str,
                    min_samples: int = 20) -> pd.DataFrame:
    """Split-conformal intervals around the LightGBM median. For origin t, nonconformity scores are the absolute scaled
    residuals |y - median| / scale of every EARLIER origin whose target weeks are all <= t, pooled over nodes of the same level
    and horizon (pooled over horizons if fewer than `min_samples`). The first origin has no history and is left out."""
    H = cfg["backtest"]["horizon"]
    Y = panel.Y
    f = base.copy()
    f["y"] = Y[(f["origin"] + f["h"] - 1).to_numpy(), f["node_idx"].to_numpy()]
    f["level"] = panel.level_ids[f["node_idx"].to_numpy()]
    f["r"] = np.abs(f["y"] - f["yhat"]) / f["_scale"]
    recs = []
    for o in origins:
        prev = f[(f["origin"] < o) & (f["origin"] + H <= o)]
        cur = f[f["origin"] == o].copy()
        if prev.empty:
            continue
        for alpha, (lo, hi) in {0.2: ("lo80", "hi80"), 0.05: ("lo95", "hi95")}.items():
            w = np.full(len(cur), np.nan)
            for (lvl, hh), idx in cur.groupby(["level", "h"]).groups.items():
                samp = prev[(prev["level"] == lvl) & (prev["h"] == hh)]["r"].to_numpy()
                if len(samp) < min_samples:
                    samp = prev[prev["level"] == lvl]["r"].to_numpy()
                n = len(samp)
                lvl_q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
                w[cur.index.get_indexer(idx)] = np.quantile(samp, lvl_q, method="higher")
            cur[lo] = np.maximum(cur["yhat"] - w * cur["_scale"], 0.0)
            cur[hi] = cur["yhat"] + w * cur["_scale"]
        recs.append(cur)
    out = pd.concat(recs, ignore_index=True)
    out["model"] = "lgbm_conformal"
    return out[["layer", "model", "origin", "h", "node_idx", "yhat", "lo80", "hi80", "lo95", "hi95"]]
