"""Assemble every forecast (Python base models, R fable models, Python-reconciled models), score them with one metric code
path, and write the evaluation artifacts used by the Tableau exports, figures and memo."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import metrics as M
from .config import origins, rpath
from .data import load_panel, summing_matrix
from .db import connect
from .reconcile import METHODS, coherence_gap, reconcile_frame

PY_RECON_MODELS = ["naive", "snaive", "ma4", "ets", "sarima_111_010", "sarima_airline", "lgbm_q"]
EVAL_SPLITS = ["validation", "test"]


def family(model: str) -> str:
    if model in ("naive", "snaive", "ma4"):
        return "baseline"
    if model in ("ets", "sarima_111_010", "sarima_airline"):
        return "statistical (statsforecast)"
    if model.startswith("lgbm"):
        return "machine learning (LightGBM)"
    if model.startswith("py_"):
        return "reconciled (Python, point only)"
    if model.startswith("geo_r_"):
        return "reconciled (R, geographic sub-hierarchy)"
    if model.startswith("r_") and model.endswith("_base"):
        return "statistical (R fable, base)"
    if model.startswith("r_"):
        return "reconciled (R fable)"
    return "other"


def read_r_forecasts(path, node_ids: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    r = pd.read_csv(path)
    ni = {n: i for i, n in enumerate(node_ids)}
    r["node_idx"] = r["node_id"].map(ni)
    if r["node_idx"].isna().any():
        raise ValueError("R forecasts contain unknown node ids")
    r["node_idx"] = r["node_idx"].astype(int)
    name = "r_" + r["base_model"] + "_" + r["recon"]
    r["model"] = np.where(r["structure"] == "geo84", "geo_" + name, name)
    r = r[~((r["structure"] == "geo84") & (r["recon"] == "base"))]
    r["layer"] = "hier"
    cols = ["layer", "model", "origin", "h", "node_idx", "yhat", "lo80", "hi80", "lo95", "hi95"]
    out = r[cols].copy()
    return out.sort_values(["model", "origin", "h", "node_idx"], kind="mergesort").reset_index(drop=True)


def recon_groups(models: list[str]) -> list[dict]:
    groups = []
    ms = set(models)
    for base in ("ets", "arima"):
        ref = f"r_{base}_base"
        mem = [ref] + [f"r_{base}_{m}" for m in METHODS if f"r_{base}_{m}" in ms]
        if ref in ms and len(mem) > 1:
            groups.append({"engine": "R fable", "structure": "full 154 nodes", "base_model": f"R {base.upper()}", "ref": ref, "members": mem})
        geo = [m for m in models if m.startswith(f"geo_r_{base}_")]
        if ref in ms and geo:
            groups.append({"engine": "R fable", "structure": "geographic 84 nodes", "base_model": f"R {base.upper()}", "ref": ref,
                           "members": [ref] + geo, "levels": [1, 2, 3, 9]})
    for base in PY_RECON_MODELS:
        mem = [base] + [f"py_{base}_{m}" for m in METHODS if f"py_{base}_{m}" in ms]
        if len(mem) > 1:
            groups.append({"engine": "Python", "structure": "full 154 nodes", "base_model": base, "ref": base, "members": mem})
    return groups


def method_of(model: str, ref: str) -> str:
    if model == ref:
        return "base (unreconciled)"
    for m in ("td_fp", "td_ap", "wls_struct", "wls_var", "mint", "ols", "bu"):
        if model.endswith("_" + m):
            return {"bu": "bottom-up", "ols": "OLS", "wls_struct": "WLS structural", "wls_var": "WLS variance",
                    "mint": "MinT shrink", "td_fp": "top-down (forecast proportions)", "td_ap": "top-down (average proportions)"}[m]
    return model


def run_evaluate(cfg: dict) -> dict:
    proc = rpath(cfg, cfg["data"]["processed_dir"])
    art = rpath(cfg, "artifacts")
    art.mkdir(exist_ok=True)
    con = connect(cfg, read_only=True)
    try:
        hier = load_panel(con, "hier")
        item = load_panel(con, "item")
        S, _ = summing_matrix(con, hier.node_ids)
    finally:
        con.close()
    b = cfg["backtest"]
    H = b["horizon"]
    org = origins(cfg, hier.T)
    reps = b["bootstrap_reps"]

    fh = pd.read_parquet(proc / "forecasts_hier.parquet")
    fr = read_r_forecasts(proc / "r_forecasts.csv", hier.node_ids)
    eval_o = org["validation"] + org["test"]
    pyrec = [reconcile_frame(fh, hier.Y, S, eval_o, H, m) for m in PY_RECON_MODELS if m in set(fh["model"])]
    frames = [fh] + ([fr] if len(fr) else []) + [p for p in pyrec if len(p)]
    FH = pd.concat(frames, ignore_index=True)
    FH = FH.sort_values(["model", "origin", "h", "node_idx"], kind="mergesort").reset_index(drop=True)
    FH.to_parquet(proc / "forecasts_hier_all.parquet", index=False)

    fa = M.attach_actuals(FH, hier.Y, hier.level_ids, hier.node_ids, org)
    fa = fa[fa["split"].isin(EVAL_SPLITS)].reset_index(drop=True)
    no_h = M.node_origin_table(fa)
    sc_h = M.scorecard(no_h, reps=reps)
    sc_h["layer"] = "hierarchy (levels 1-9)"
    po_h = M.per_origin(no_h)
    cov_h = M.coverage_by_horizon(fa)

    # item layer
    fi = pd.read_parquet(proc / "forecasts_item.parquet")
    ia = M.attach_actuals(fi, item.Y, item.level_ids, item.node_ids, org)
    ia = ia[ia["split"].isin(EVAL_SPLITS)].reset_index(drop=True)
    no_i = M.node_origin_table(ia)
    sc_i = M.scorecard(no_i, reps=reps)
    sc_i["layer"] = "item sample (levels 10 and 12)"
    po_i = M.per_origin(no_i)
    cov_i = M.coverage_by_horizon(ia)

    sc = pd.concat([sc_h, sc_i], ignore_index=True)
    sc["model_family"] = sc["model"].map(family)
    sc.to_csv(art / "scorecard_full.csv", index=False, float_format="%.6g")
    pd.concat([po_h.assign(layer="hierarchy (levels 1-9)"), po_i.assign(layer="item sample (levels 10 and 12)")]).to_csv(
        art / "per_origin_metrics.csv", index=False, float_format="%.6g")
    pd.concat([cov_h, cov_i]).to_csv(art / "coverage_by_horizon.csv", index=False, float_format="%.6g")

    # reconciliation comparison: paired against the unreconciled base model of each family
    rows = []
    models = sorted(set(FH["model"]))
    for g in recon_groups(models):
        sub = no_h[no_h["model"].isin(g["members"])]
        if "levels" in g:
            sub = sub[sub["level"].isin(g["levels"])]
        s = M.scorecard(sub, reps=reps, ref_model=g["ref"])
        s["metric"] = s["metric"].str.replace(f"WAPE_ratio_vs_{g['ref']}", "WAPE_ratio_vs_base", regex=False)
        s["engine"], s["structure"], s["base_model"] = g["engine"], g["structure"], g["base_model"]
        s["method"] = s["model"].map(lambda m, r=g["ref"]: method_of(m, r))
        rows.append(s)
    rc = pd.concat(rows, ignore_index=True)
    rc.to_csv(art / "reconciliation_comparison_full.csv", index=False, float_format="%.6g")

    # incoherence of base forecasts
    base_models = [m for m in models if not m.startswith(("py_", "geo_")) and not m.startswith("lgbm_conformal")]
    cg = coherence_gap(FH[FH["model"].isin(base_models)].merge(
        fa[["model", "origin", "h", "node_idx"]].drop_duplicates(), on=["model", "origin", "h", "node_idx"]), S, hier.level_ids)
    cg = cg.groupby(["model", "level"], as_index=False)["mean_abs_rel_gap"].mean()
    cg.to_csv(art / "coherence_gap.csv", index=False, float_format="%.6g")

    # residual tests (R)
    rt_path = proc / "r_residual_tests.csv"
    if rt_path.exists():
        rt = pd.read_csv(rt_path)
        lvl = dict(zip(hier.node_ids, hier.level_ids, strict=True))
        rt["level"] = rt["node_id"].map(lvl)
        rt.to_csv(art / "residual_tests_by_node.csv", index=False, float_format="%.6g")
    info = {"n_forecast_rows_hier": len(FH), "models_hier": models, "n_forecast_rows_item": len(fi)}
    (art / "evaluation_summary.json").write_text(json.dumps(info, indent=2, sort_keys=True))
    return info
