"""Accuracy, probabilistic and calibration metrics, with bootstrap uncertainty. All aggregation orders are explicit."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import split_of

QS = {0.025: "lo95", 0.1: "lo80", 0.5: "yhat", 0.9: "hi80", 0.975: "hi95"}
SUMS = ["abs", "y", "err", "pin", "c80", "c95", "w80", "w95", "mase", "n", "rmsse", "n_no"]


def attach_actuals(fc: pd.DataFrame, Y: np.ndarray, level_ids: np.ndarray, node_ids: list[str], org: dict) -> pd.DataFrame:
    f = fc.copy()
    ni = f["node_idx"].to_numpy()
    f["target_w"] = f["origin"] + f["h"]
    f["y"] = Y[f["target_w"].to_numpy() - 1, ni]
    f["level"] = level_ids[ni]
    f["node_id"] = np.asarray(node_ids)[ni]
    f["split"] = [split_of(o, org) for o in f["origin"]]
    # scale of the in-sample one-step naive error at the origin (weeks 1..origin only)
    d = np.abs(np.diff(Y, axis=0))
    cum_abs = np.vstack([np.zeros((1, Y.shape[1])), np.cumsum(d, axis=0)])           # cum_abs[t-1] = sum of |diff| up to week t
    cum_sq = np.vstack([np.zeros((1, Y.shape[1])), np.cumsum(np.diff(Y, axis=0) ** 2, axis=0)])
    o = f["origin"].to_numpy()
    f["scale_mae"] = np.maximum(cum_abs[o - 1, ni] / (o - 1), 1e-9)
    f["scale_mse"] = np.maximum(cum_sq[o - 1, ni] / (o - 1), 1e-9)
    return f


def row_losses(f: pd.DataFrame) -> pd.DataFrame:
    e = f["yhat"] - f["y"]
    r = pd.DataFrame(index=f.index)
    r["abs"] = e.abs()
    r["sq"] = e**2
    r["err"] = e
    pin = 0.0
    for q, col in QS.items():
        u = f["y"] - f[col]
        pin = pin + np.maximum(q * u, (q - 1) * u)
    r["pin"] = pin / len(QS)
    r["c80"] = ((f["y"] >= f["lo80"]) & (f["y"] <= f["hi80"])).astype(float).where(f["lo80"].notna())
    r["c95"] = ((f["y"] >= f["lo95"]) & (f["y"] <= f["hi95"])).astype(float).where(f["lo95"].notna())
    r["w80"] = f["hi80"] - f["lo80"]
    r["w95"] = f["hi95"] - f["lo95"]
    r["mase"] = r["abs"] / f["scale_mae"]
    return r


def node_origin_table(f: pd.DataFrame) -> pd.DataFrame:
    """One row per model x node x origin with additive sums (so any group metric is a ratio of sums)."""
    r = row_losses(f)
    d = pd.concat([f[["model", "level", "node_id", "node_idx", "origin", "split", "y", "scale_mse"]], r], axis=1)
    d["rmsse_inner"] = d["sq"] / d["scale_mse"]
    keys = ["model", "level", "node_id", "node_idx", "origin", "split"]
    grp = d.groupby(keys, sort=True)
    # min_count=1 keeps interval-based sums NaN for point-only (reconciled Python) forecasts instead of silently 0
    g = grp[["abs", "y", "err", "pin", "c80", "c95", "w80", "w95", "mase"]].sum(min_count=1)
    g["n"] = grp["abs"].size().astype(float)
    g["rmsse_inner"] = grp["rmsse_inner"].mean()
    g = g.reset_index()
    g["rmsse"] = np.sqrt(g.pop("rmsse_inner"))
    g["n_no"] = 1.0
    return g


def metrics_from_sums(s: pd.DataFrame | dict) -> dict:
    def g(k):
        return float(np.asarray(s[k], dtype=float).sum())

    y = g("y")
    return {
        "WAPE": g("abs") / y if y > 0 else np.nan,
        "MASE": g("mase") / g("n"),
        "RMSSE": g("rmsse") / g("n_no"),
        "Bias": g("err") / y if y > 0 else np.nan,
        "Pinball_scaled": g("pin") / y if y > 0 else np.nan,
        "Coverage_80": g("c80") / g("n"),
        "Coverage_95": g("c95") / g("n"),
        "Width_80_rel": g("w80") / y if y > 0 else np.nan,
        "Width_95_rel": g("w95") / y if y > 0 else np.nan,
    }


METRIC_NAMES = ["WAPE", "MASE", "RMSSE", "Bias", "Pinball_scaled", "Coverage_80", "Coverage_95", "Width_80_rel", "Width_95_rel"]


def _vec_metrics(M: np.ndarray) -> np.ndarray:
    """M: (B, len(SUMS)) sums per bootstrap rep -> (B, 9) metrics in METRIC_NAMES order."""
    c = {k: M[:, i] for i, k in enumerate(SUMS)}
    y = np.where(c["y"] > 0, c["y"], np.nan)
    return np.column_stack([
        c["abs"] / y, c["mase"] / c["n"], c["rmsse"] / c["n_no"], c["err"] / y, c["pin"] / y,
        c["c80"] / c["n"], c["c95"] / c["n"], c["w80"] / y, c["w95"] / y,
    ])


def scorecard(no: pd.DataFrame, reps: int = 1000, seed: int = 7, ref_model: str = "snaive") -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slices: point-only forecasts have no interval metrics
        return _scorecard(no, reps, seed, ref_model)


def _scorecard(no: pd.DataFrame, reps: int, seed: int, ref_model: str) -> pd.DataFrame:
    """Metrics per model x level x split with 95 percent bootstrap intervals.
    Resampling unit: nodes when the level has >= 3 nodes (resample series), otherwise origins (single-node levels).
    The same resample is applied to every model so the ratio to the reference model is paired."""
    rng = np.random.default_rng(seed)
    out = []
    for (level, split), grp in no.groupby(["level", "split"], sort=True):
        nodes = sorted(grp["node_id"].unique())
        origins_ = sorted(grp["origin"].unique())
        by_node = len(nodes) >= 3
        units = nodes if by_node else origins_
        key = "node_id" if by_node else "origin"
        W = rng.multinomial(len(units), np.full(len(units), 1.0 / len(units)), size=reps).astype(float)
        boot = {}
        for model, gm in grp.groupby("model", sort=True):
            agg = gm.groupby(key, sort=True)[SUMS].sum(min_count=1).reindex(units)
            boot[model] = _vec_metrics(W @ agg.to_numpy())
            pt = metrics_from_sums(gm)
            lo = np.nanpercentile(boot[model], 2.5, axis=0)
            hi = np.nanpercentile(boot[model], 97.5, axis=0)
            # per-origin spread of WAPE
            po = gm.groupby("origin", sort=True)[["abs", "y"]].sum()
            wape_o = (po["abs"] / po["y"]).to_numpy()
            for j, name in enumerate(METRIC_NAMES):
                out.append({
                    "model": model, "level": level, "split": split, "metric": name, "value": pt[name], "ci_lo": lo[j], "ci_hi": hi[j],
                    "ci_method": "bootstrap_nodes" if by_node else "bootstrap_origins", "n_nodes": len(nodes), "n_origins": len(origins_),
                    "origin_min": wape_o.min() if name == "WAPE" else np.nan,
                    "origin_max": wape_o.max() if name == "WAPE" else np.nan,
                })
        if ref_model in boot:
            ref = boot[ref_model][:, 0]
            for model, B in boot.items():
                rel = B[:, 0] / ref
                gm = grp[grp["model"] == model]
                gr = grp[grp["model"] == ref_model]
                pt = metrics_from_sums(gm)["WAPE"] / metrics_from_sums(gr)["WAPE"]
                out.append({"model": model, "level": level, "split": split, "metric": f"WAPE_ratio_vs_{ref_model}", "value": pt,
                            "ci_lo": np.nanpercentile(rel, 2.5), "ci_hi": np.nanpercentile(rel, 97.5),
                            "ci_method": "bootstrap_nodes" if by_node else "bootstrap_origins", "n_nodes": len(nodes),
                            "n_origins": len(origins_), "origin_min": np.nan, "origin_max": np.nan})
    return pd.DataFrame(out)


def per_origin(no: pd.DataFrame) -> pd.DataFrame:
    g = no.groupby(["model", "level", "split", "origin"], sort=True)[SUMS].sum().reset_index()
    rows = []
    for _, r in g.iterrows():
        m = metrics_from_sums(r)
        rows.append({"model": r["model"], "level": r["level"], "split": r["split"], "origin": r["origin"], **m})
    return pd.DataFrame(rows)


def coverage_by_horizon(f: pd.DataFrame) -> pd.DataFrame:
    r = row_losses(f)
    d = pd.concat([f[["model", "level", "split", "h", "y"]], r[["c80", "c95", "w80", "w95"]]], axis=1)
    g = d.groupby(["model", "level", "split", "h"], sort=True).agg(
        n=("c80", "size"), c80=("c80", "mean"), c95=("c95", "mean"), w80=("w80", "sum"), w95=("w95", "sum"), y=("y", "sum")).reset_index()
    g["w80_rel"] = g["w80"] / g["y"]
    g["w95_rel"] = g["w95"] / g["y"]
    return g.drop(columns=["w80", "w95", "y"])
