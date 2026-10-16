"""Transparent baselines: naive, seasonal naive, moving average. Intervals are Gaussian from the in-sample h-step residuals
of the same method, so every model in the comparison has 80 and 95 percent bounds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

Z = {80: float(norm.ppf(0.9)), 95: float(norm.ppf(0.975))}


def _point(Y: np.ndarray, t: int, k: int, method: str, m: int, win: int) -> np.ndarray:
    """Forecast of week t+k made with data up to week t (1-based t). Y is (T, N)."""
    if method == "naive":
        return Y[t - 1]
    if method == "snaive":
        return Y[t + k - m - 1]
    if method == "ma":
        return Y[t - win:t].mean(axis=0)
    raise ValueError(method)


def baseline_forecast(Y: np.ndarray, origin: int, horizon: int, method: str, m: int, win: int = 4) -> dict:
    """Return point and sd arrays of shape (h, N) for one origin. Residuals use only weeks <= origin."""
    N = Y.shape[1]
    pt = np.zeros((horizon, N))
    sd = np.zeros((horizon, N))
    for k in range(1, horizon + 1):
        pt[k - 1] = _point(Y, origin, k, method, m, win)
        # in-sample h-step errors: forecasts made at s, evaluated at s+k <= origin
        first = {"naive": 1, "snaive": m + 1 - k + 0, "ma": win}[method]
        first = max(first, 1)
        s_idx = np.arange(first, origin - k + 1)
        if len(s_idx) < 3:
            sd[k - 1] = np.nan
            continue
        errs = np.empty((len(s_idx), N))
        for j, s in enumerate(s_idx):
            errs[j] = Y[s + k - 1] - _point(Y, s, k, method, m, win)
        sd[k - 1] = np.sqrt(np.mean(errs**2, axis=0))
    return {"point": pt, "sd": sd}


def baselines_frame(Y: np.ndarray, origins: list[int], horizon: int, m: int, win: int, layer: str) -> pd.DataFrame:
    recs = []
    names = {"naive": "naive", "snaive": "snaive", "ma": f"ma{win}"}
    for method in ("naive", "snaive", "ma"):
        for o in origins:
            r = baseline_forecast(Y, o, horizon, method, m, win)
            for k in range(1, horizon + 1):
                p, s = r["point"][k - 1], r["sd"][k - 1]
                recs.append(pd.DataFrame({
                    "layer": layer, "model": names[method], "origin": o, "h": k, "node_idx": np.arange(Y.shape[1]),
                    "yhat": p, "lo80": p - Z[80] * s, "hi80": p + Z[80] * s, "lo95": p - Z[95] * s, "hi95": p + Z[95] * s,
                }))
    return pd.concat(recs, ignore_index=True)
