"""ETS and SARIMA via statsforecast, refit at every rolling origin (expanding window)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _long(Y: np.ndarray) -> pd.DataFrame:
    T, N = Y.shape
    return pd.DataFrame({
        "unique_id": np.repeat(np.arange(N), T).astype(str),
        "ds": np.tile(np.arange(1, T + 1), N),
        "y": Y.T.reshape(-1),
    })


def stat_models(names: list[str], season: int) -> dict:
    from statsforecast.models import ARIMA, AutoETS

    avail = {
        # AutoETS picks error/trend/seasonal form by AICc; seasonal period is the annual week count
        "ets": lambda: AutoETS(season_length=season, alias="ets"),
        # fixed-order seasonal ARIMA: auto-selection with a 52-week season costs ~14 s per fit (tested), so orders are fixed a priori
        "sarima_111_010": lambda: ARIMA(order=(1, 1, 1), seasonal_order=(0, 1, 0), season_length=season, alias="sarima_111_010"),
        "sarima_airline": lambda: ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1), season_length=season, alias="sarima_airline"),
    }
    return {n: avail[n]() for n in names}


def stats_frame(Y: np.ndarray, origins: list[int], horizon: int, step: int, season: int, names: list[str], layer: str,
                n_jobs: int = 1) -> pd.DataFrame:
    from statsforecast import StatsForecast

    df = _long(Y)
    out = []
    for name, model in stat_models(names, season).items():
        sf = StatsForecast(models=[model], freq=1, n_jobs=n_jobs)
        cv = sf.cross_validation(df=df, h=horizon, step_size=step, n_windows=len(origins), level=[80, 95], refit=True)
        cv = cv.reset_index() if "unique_id" not in cv.columns else cv
        cv["node_idx"] = cv["unique_id"].astype(int)
        cv["origin"] = cv["cutoff"].astype(int)
        cv["h"] = cv["ds"].astype(int) - cv["origin"]
        res = pd.DataFrame({
            "layer": layer, "model": name, "origin": cv["origin"], "h": cv["h"], "node_idx": cv["node_idx"],
            "yhat": cv[name], "lo80": cv[f"{name}-lo-80"], "hi80": cv[f"{name}-hi-80"],
            "lo95": cv[f"{name}-lo-95"], "hi95": cv[f"{name}-hi-95"],
        })
        assert set(res["origin"].unique()) == set(origins), (sorted(res["origin"].unique()), origins)
        out.append(res)
    return pd.concat(out, ignore_index=True)
