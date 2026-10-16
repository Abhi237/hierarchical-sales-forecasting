"""Forecast reconciliation for point forecasts (numpy). Bottom-up, OLS, WLS (structural and variance scaling) and MinT with
Schaefer-Strimmer shrinkage. The base-forecast error covariance is estimated from the model's OWN earlier backtest errors
(origins whose targets are fully observed at the current origin), never from the future."""
from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ["bu", "ols", "wls_struct", "wls_var", "mint"]


def bottom_index(S: np.ndarray) -> np.ndarray:
    """Row index in S of each bottom series (rows of S that are unit vectors), ordered by bottom column."""
    m = S.shape[1]
    idx = np.full(m, -1)
    for j in range(m):
        rows = np.where((S[:, j] == 1) & (S.sum(axis=1) == 1))[0]
        idx[j] = rows[0]
    return idx


def shrink_cov(E: np.ndarray) -> np.ndarray:
    """Schaefer-Strimmer shrinkage of the (uncentred) covariance of residual rows E (n x p) towards its diagonal."""
    n, p = E.shape
    cov = E.T @ E / n
    d = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    Wn = E / d
    R = (Wn.T @ Wn) / n
    off = ~np.eye(p, dtype=bool)
    # variance of each sample correlation: n/((n-1)^3) * sum_k (w_ki w_kj - mean)^2
    prod_mean = R                                           # mean of w_ki*w_kj over k
    sq = (Wn**2).T @ (Wn**2) / n                            # mean of (w_ki w_kj)^2
    var_r = n / max((n - 1) ** 3, 1) * n * (sq - prod_mean**2)
    num = var_r[off].sum()
    den = (R[off] ** 2).sum()
    lam = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 1.0
    Rs = R.copy()
    Rs[off] *= (1 - lam)
    return Rs * np.outer(d, d)


def g_matrix(S: np.ndarray, method: str, W: np.ndarray | None = None) -> np.ndarray:
    n, m = S.shape
    if method == "bu":
        G = np.zeros((m, n))
        G[np.arange(m), bottom_index(S)] = 1.0
        return G
    if method == "ols":
        return np.linalg.solve(S.T @ S, S.T)
    if method in ("wls_struct", "wls_var", "mint"):
        if method == "wls_struct":
            Winv = 1.0 / S.sum(axis=1)
            SW = S.T * Winv
            return np.linalg.solve(SW @ S, SW)
        if method == "wls_var":
            Winv = 1.0 / np.maximum(np.diag(W), 1e-12)
            SW = S.T * Winv
            return np.linalg.solve(SW @ S, SW)
        Wi = np.linalg.inv(W)
        SW = S.T @ Wi
        return np.linalg.solve(SW @ S, SW)
    raise ValueError(method)


def reconcile_matrix(F: np.ndarray, S: np.ndarray, method: str, W: np.ndarray | None = None) -> np.ndarray:
    """F: (n_nodes, k) base forecasts -> coherent (n_nodes, k)."""
    G = g_matrix(S, method, W)
    return S @ (G @ F)


def reconcile_frame(fc: pd.DataFrame, Y: np.ndarray, S: np.ndarray, origins_eval: list[int], horizon: int, model: str,
                    methods: list[str] = METHODS) -> pd.DataFrame:
    """Reconcile one model's point forecasts at each evaluation origin. Returns rows with yhat only (intervals NaN)."""
    f = fc[fc["model"] == model]
    n = S.shape[0]
    recs = []
    errs = {}
    for o in sorted(f["origin"].unique()):
        sub = f[f["origin"] == o]
        ys = np.vstack([Y[o + k - 1] for k in range(1, horizon + 1)])           # (h, n) actual weeks o+1..o+h
        pt = np.zeros((horizon, n))
        for k in range(1, horizon + 1):
            r = sub[sub["h"] == k].sort_values("node_idx")
            pt[k - 1] = r["yhat"].to_numpy()
        errs[o] = ys - pt                                                       # (h, n)
    for o in origins_eval:
        prev = [errs[p] for p in sorted(errs) if p + horizon <= o]
        if not prev:
            continue
        E = np.vstack(prev)                                                     # rows = (origin, h) pairs, columns = nodes
        W = shrink_cov(E)
        sub = f[f["origin"] == o]
        base = np.zeros((n, horizon))
        for k in range(1, horizon + 1):
            r = sub[sub["h"] == k].sort_values("node_idx")
            base[:, k - 1] = r["yhat"].to_numpy()
        for meth in methods:
            rec = reconcile_matrix(base, S, meth, W)
            for k in range(1, horizon + 1):
                recs.append(pd.DataFrame({"layer": "hier", "model": f"py_{model}_{meth}", "origin": o, "h": k,
                                          "node_idx": np.arange(n), "yhat": rec[:, k - 1],
                                          "lo80": np.nan, "hi80": np.nan, "lo95": np.nan, "hi95": np.nan}))
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def coherence_gap(fc: pd.DataFrame, S: np.ndarray, level_ids: np.ndarray) -> pd.DataFrame:
    """Mean absolute relative incoherence of base forecasts: |F_node - S_node * F_bottom| / |F_node| per level."""
    bi = bottom_index(S)
    recs = []
    for (model, origin, h), g in fc.groupby(["model", "origin", "h"], sort=True):
        v = g.sort_values("node_idx")["yhat"].to_numpy()
        if len(v) != S.shape[0]:
            continue
        agg = S @ v[bi]
        gap = np.abs(v - agg) / np.maximum(np.abs(v), 1e-9)
        for lvl in np.unique(level_ids):
            recs.append((model, origin, h, int(lvl), float(gap[level_ids == lvl].mean())))
    return pd.DataFrame(recs, columns=["model", "origin", "h", "level", "mean_abs_rel_gap"])
