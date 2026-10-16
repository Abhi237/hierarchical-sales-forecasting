"""Leakage-safe feature table for the global LightGBM models (direct multi-horizon, horizon as a feature).

A row is (node, origin s, horizon h): features use only weeks <= s plus calendar facts of the TARGET week s+h (SNAP, events,
week of year), which are known in advance. The target is revenue(s+h) divided by the node's trailing 52-week mean revenue
at s, so one model can serve series of very different scale."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel

LAGS = [0, 1, 2, 3, 4, 5, 6, 7, 12]


def feature_names(season: int) -> list[str]:
    names = [f"lag{k}" for k in LAGS] + ["seas0", "seasm1", "seasp1", "mean4", "mean13", "std13", "zero13", "price_idx", "mom4_13"]
    names += ["h", "level", "woy", "month", "snap", "n_ev_sporting", "n_ev_cultural", "n_ev_national", "n_ev_religious",
              "is_christmas_week", "is_thanksgiving_week", "is_superbowl_week", "is_easter_week"]
    return names


def build_table(panel: Panel, horizon: int, season: int) -> dict:
    """Return dict with X (float32 matrix), y (target ratio), scale, and meta arrays (s, h, node)."""
    Y, U = panel.Y, panel.U
    T, N = Y.shape
    wk = panel.weeks
    week_end = pd.to_datetime(wk["week_end"])
    woy = week_end.dt.isocalendar().week.to_numpy().astype(float)
    month = week_end.dt.month.to_numpy().astype(float)
    snap_state = wk[["snap_days_ca", "snap_days_tx", "snap_days_wi"]].to_numpy(dtype=float)  # (T, 3)
    snap_node = snap_state @ panel.state_share.T                                             # (T, N)
    cal = {c: wk[c].to_numpy(dtype=float) for c in ["n_ev_sporting", "n_ev_cultural", "n_ev_national", "n_ev_religious",
                                                     "is_christmas_week", "is_thanksgiving_week", "is_superbowl_week", "is_easter_week"]}
    price = np.where(U > 0, Y / np.maximum(U, 1e-9), np.nan)
    s_min = season + 1
    blocks, ys, scales, ss, hs, nodes = [], [], [], [], [], []
    names = feature_names(season)
    for s in range(s_min, T):                   # origin week s (1-based); needs s + 1 <= T for at least one horizon
        win = Y[max(0, s - 52):s]               # weeks s-51 .. s
        scale = np.maximum(win.mean(axis=0), 1.0)
        lag = {k: Y[s - k - 1] / scale for k in LAGS}
        m4, m13 = Y[s - 4:s].mean(0) / scale, Y[s - 13:s].mean(0) / scale
        sd13 = Y[s - 13:s].std(0) / scale
        z13 = (Y[s - 13:s] == 0).mean(0)
        pw = price[max(0, s - 52):s]
        cnt = (~np.isnan(pw)).sum(axis=0)
        pmean = np.where(cnt > 0, np.nansum(pw, axis=0) / np.maximum(cnt, 1), np.nan)
        pnow = price[s - 1]
        pidx = np.where(np.isnan(pnow) | np.isnan(pmean) | (pmean == 0), 1.0, pnow / pmean)
        mom = np.where(m13 > 0, m4 / np.maximum(m13, 1e-9), 1.0)
        for h in range(1, horizon + 1):
            tgt = s + h
            if tgt > T:
                continue
            feats = [lag[k] for k in LAGS]
            feats += [Y[tgt - season - 1] / scale, Y[tgt - season - 2] / scale, Y[tgt - season] / scale]
            feats += [m4, m13, sd13, z13, pidx, mom]
            feats += [np.full(N, h), panel.level_ids.astype(float), np.full(N, woy[tgt - 1]), np.full(N, month[tgt - 1]), snap_node[tgt - 1]]
            feats += [np.full(N, cal[c][tgt - 1]) for c in ["n_ev_sporting", "n_ev_cultural", "n_ev_national", "n_ev_religious",
                                                             "is_christmas_week", "is_thanksgiving_week", "is_superbowl_week", "is_easter_week"]]
            blocks.append(np.column_stack(feats).astype(np.float32))
            ys.append(Y[tgt - 1] / scale)
            scales.append(scale)
            ss.append(np.full(N, s))
            hs.append(np.full(N, h))
            nodes.append(np.arange(N))
    X = np.vstack(blocks)
    assert X.shape[1] == len(names)
    return {"X": X, "y": np.concatenate(ys), "scale": np.concatenate(scales), "s": np.concatenate(ss),
            "h": np.concatenate(hs), "node": np.concatenate(nodes), "names": names}
