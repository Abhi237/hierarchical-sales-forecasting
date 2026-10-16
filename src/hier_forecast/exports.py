"""Tidy, flat, documented CSVs for the Tableau Public dashboard (tableau/data/). Every file is derived aggregates only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .config import origins, rpath
from .data import load_panel
from .db import connect
from .evaluate import EVAL_SPLITS, family

FVA_MODELS = ["naive", "snaive", "ma4", "ets", "sarima_airline", "lgbm_q", "lgbm_conformal",
              "r_ets_base", "r_ets_bu", "r_ets_ols", "r_ets_mint", "r_arima_base", "r_arima_bu", "r_arima_ols", "r_arima_mint"]
DEC_MODELS = FVA_MODELS
SENS_MARGIN = [0.15, 0.25, 0.35]
SENS_CARRY = [0.02, 0.05, 0.10]

MODEL_LABELS = {
    "naive": "Naive", "snaive": "Seasonal naive (52 wk)", "ma4": "Moving average (4 wk)", "ets": "ETS (statsforecast AutoETS)",
    "sarima_111_010": "SARIMA(1,1,1)(0,1,0)[52]", "sarima_airline": "SARIMA(0,1,1)(0,1,1)[52]",
    "lgbm_q": "LightGBM quantile", "lgbm_conformal": "LightGBM + conformal",
}


def model_label(m: str) -> str:
    if m in MODEL_LABELS:
        return MODEL_LABELS[m]
    meth = {"base": "base", "bu": "bottom-up", "ols": "OLS", "wls_struct": "WLS struct", "wls_var": "WLS var", "mint": "MinT shrink",
            "td_fp": "top-down FP", "td_ap": "top-down AP"}
    if m.startswith("geo_r_"):
        parts = m.split("_")
        return f"R fable {parts[2].upper()} {meth.get('_'.join(parts[3:]), '_'.join(parts[3:]))} (geo 84)"
    if m.startswith("r_"):
        parts = m.split("_")
        return f"R fable {parts[1].upper()} {meth.get('_'.join(parts[2:]), '_'.join(parts[2:]))}"
    if m.startswith("py_"):
        rest = m[3:]
        for k in ("wls_struct", "wls_var", "mint", "ols", "bu"):
            if rest.endswith("_" + k):
                base = rest[: -len(k) - 1]
                return f"{MODEL_LABELS.get(base, base)} + {meth[k]} (Python)"
    return m


def _levels(con, cfg) -> dict:
    d = dict(con.execute("SELECT DISTINCT level_id, level_name FROM dim_node ORDER BY level_id").fetchall())
    d[10] = f"Item (top {cfg['scope']['item_layer']['top_items']} sample)"
    d[12] = f"Item x Store (top {cfg['scope']['item_layer']['top_item_stores']} sample)"
    return d


def _wr(df: pd.DataFrame, path, **kw) -> dict:
    df.to_csv(path, index=False, float_format=kw.pop("float_format", "%.4f"), **kw)
    return {"file": path.name, "rows": len(df), "bytes": path.stat().st_size}


def run_exports(cfg: dict) -> list[dict]:
    root = rpath(cfg, "tableau", "data")
    root.mkdir(parents=True, exist_ok=True)
    proc = rpath(cfg, cfg["data"]["processed_dir"])
    art = rpath(cfg, "artifacts")
    con = connect(cfg, read_only=True)
    try:
        hier = load_panel(con, "hier")
        item = load_panel(con, "item")
        lvl = _levels(con, cfg)
        nodes = con.execute("""
            SELECT n.node_id, n.node_label, n.level_id, n.level_name, n.state_id AS state, n.store_id AS store, n.cat_id AS category,
                   n.dept_id AS department, count(b.bottom_id) AS n_bottom_series
            FROM dim_node n LEFT JOIN bridge_node_bottom b USING (node_id) GROUP BY ALL ORDER BY n.level_id, n.node_id""").df()
        item_nodes = con.execute("""
            SELECT node_id, node_label, level_id, level_name, state_id AS state, store_id AS store, cat_id AS category, dept_id AS department,
                   CAST(selection_revenue AS DOUBLE) AS selection_window_revenue FROM dim_item_node ORDER BY level_id, node_id""").df()
        daily = con.execute("""SELECT level_id, node_id, CAST(date AS DATE) AS date, w AS week_idx, units, CAST(revenue AS DOUBLE) AS revenue
                               FROM fact_recent_daily ORDER BY level_id, node_id, date""").df()
        cleaning = con.execute("SELECT step, metric, value, note FROM cleaning_log ORDER BY step").df()
    finally:
        con.close()
    org = origins(cfg, hier.T)
    weeks = hier.weeks
    wk_start = dict(zip(weeks["w"], weeks["week_start"].astype(str), strict=True))
    wk_end = dict(zip(weeks["w"], weeks["week_end"].astype(str), strict=True))
    out = []

    # 1. hierarchy nodes ---------------------------------------------------------------------------------------------------
    tot = hier.Y.sum()
    rev = {n: hier.Y[:, i].sum() for i, n in enumerate(hier.node_ids)}
    nodes["layer"] = "hierarchy (levels 1-9)"
    nodes["total_revenue_usd"] = nodes["node_id"].map(rev)
    nodes["avg_weekly_revenue_usd"] = nodes["total_revenue_usd"] / hier.T
    nodes["share_of_total_revenue_pct"] = 100 * nodes["total_revenue_usd"] / tot
    nodes["selection_window_revenue_usd"] = np.nan
    inodes = item_nodes.rename(columns={"selection_window_revenue": "selection_window_revenue_usd"})
    inodes["layer"] = "item sample (levels 10 and 12)"
    inodes["n_bottom_series"] = np.nan
    ir = {n: item.Y[:, i].sum() for i, n in enumerate(item.node_ids)}
    inodes["total_revenue_usd"] = inodes["node_id"].map(ir)
    inodes["avg_weekly_revenue_usd"] = inodes["total_revenue_usd"] / item.T
    inodes["share_of_total_revenue_pct"] = 100 * inodes["total_revenue_usd"] / tot
    hn = pd.concat([nodes, inodes], ignore_index=True)
    hn["level_name"] = hn["level_id"].map(lvl)
    out.append(_wr(hn, root / "hierarchy_nodes.csv"))

    # 2. weekly actuals ----------------------------------------------------------------------------------------------------
    rows = []
    for i, n in enumerate(hier.node_ids):
        rows.append(pd.DataFrame({"node_id": n, "level_id": hier.level_ids[i], "week_idx": weeks["w"], "week_start": weeks["week_start"].astype(str),
                                  "revenue_usd": hier.Y[:, i], "units": hier.U[:, i]}))
    act = pd.concat(rows, ignore_index=True)
    out.append(_wr(act, root / "actuals_weekly.csv", float_format="%.2f"))
    cal = weeks[["w", "week_start", "week_end", "snap_days_ca", "snap_days_tx", "snap_days_wi", "event_names"]].rename(columns={"w": "week_idx"}).copy()
    cal["week_start"] = cal["week_start"].astype(str)
    cal["week_end"] = cal["week_end"].astype(str)
    cal["is_christmas_week"] = weeks["is_christmas_week"]
    cal["role"] = "train/backtest history"
    for o in org["test"]:
        cal.loc[(cal["week_idx"] > o) & (cal["week_idx"] <= o + org["horizon"]), "role"] = "test target week"
    out.append(_wr(cal, root / "calendar_weeks.csv"))
    daily["node_label"] = daily["node_id"].map(dict(zip(nodes["node_id"], nodes["node_label"], strict=True)))
    daily["level_name"] = daily["level_id"].map(lvl)
    out.append(_wr(daily, root / "actuals_daily_recent.csv", float_format="%.2f"))

    # 3. forecast vs actual (test origins, curated models) -------------------------------------------------------------------
    FH = pd.read_parquet(proc / "forecasts_hier_all.parquet")
    fa_all = M.attach_actuals(FH, hier.Y, hier.level_ids, hier.node_ids, org)
    fa_all = fa_all[fa_all["split"].isin(EVAL_SPLITS)].reset_index(drop=True)
    sel = fa_all[(fa_all["split"] == "test") & fa_all["model"].isin(FVA_MODELS)].copy()
    lab = dict(zip(nodes["node_id"], nodes["node_label"], strict=True))
    fva = pd.DataFrame({
        "model": sel["model"], "model_label": sel["model"].map(model_label), "model_family": sel["model"].map(family),
        "level_id": sel["level"], "level_name": sel["level"].map(lvl), "node_id": sel["node_id"], "node_label": sel["node_id"].map(lab),
        "origin_week_idx": sel["origin"], "forecast_made_after": sel["origin"].map(wk_end), "horizon_weeks": sel["h"],
        "target_week_idx": sel["target_w"], "target_week_start": sel["target_w"].map(wk_start),
        "forecast_usd": sel["yhat"], "lo80_usd": sel["lo80"], "hi80_usd": sel["hi80"], "lo95_usd": sel["lo95"], "hi95_usd": sel["hi95"],
        "actual_usd": sel["y"],
    })
    fva["error_usd"] = fva["forecast_usd"] - fva["actual_usd"]
    fva["abs_error_usd"] = fva["error_usd"].abs()
    fva["in_80_interval"] = ((fva["actual_usd"] >= fva["lo80_usd"]) & (fva["actual_usd"] <= fva["hi80_usd"])).astype(int)
    fva["in_95_interval"] = ((fva["actual_usd"] >= fva["lo95_usd"]) & (fva["actual_usd"] <= fva["hi95_usd"])).astype(int)
    fva = fva.sort_values(["model", "level_id", "node_id", "origin_week_idx", "horizon_weeks"], kind="mergesort")
    out.append(_wr(fva, root / "forecast_vs_actual.csv", float_format="%.2f"))

    # 4. scorecard ---------------------------------------------------------------------------------------------------------
    sc = pd.read_csv(art / "scorecard_full.csv")
    sc["level_name"] = sc["level"].map(lvl)
    sc["model_label"] = sc["model"].map(model_label)
    sc = sc.rename(columns={"level": "level_id", "value": "value", "origin_min": "wape_origin_min", "origin_max": "wape_origin_max"})
    cols = ["layer", "model", "model_label", "model_family", "level_id", "level_name", "split", "metric", "value", "ci_lo", "ci_hi", "ci_method",
            "n_nodes", "n_origins", "wape_origin_min", "wape_origin_max"]
    out.append(_wr(sc[cols], root / "metric_scorecard.csv", float_format="%.5f"))

    # 5. reconciliation comparison -----------------------------------------------------------------------------------------
    rc = pd.read_csv(art / "reconciliation_comparison_full.csv")
    rc["level_name"] = rc["level"].map(lvl)
    rc["model_label"] = rc["model"].map(model_label)
    rc = rc.rename(columns={"level": "level_id"})
    keep = ["engine", "structure", "base_model", "method", "model", "model_label", "level_id", "level_name", "split", "metric", "value", "ci_lo", "ci_hi",
            "ci_method"]
    out.append(_wr(rc[keep], root / "reconciliation_comparison.csv", float_format="%.5f"))
    cg = pd.read_csv(art / "coherence_gap.csv")
    cg["level_name"] = cg["level"].map(lvl)
    cg["model_label"] = cg["model"].map(model_label)
    cg = cg.rename(columns={"level": "level_id"})
    out.append(_wr(cg[["model", "model_label", "level_id", "level_name", "mean_abs_rel_gap"]], root / "coherence_gap.csv", float_format="%.6f"))

    # 6. interval coverage -------------------------------------------------------------------------------------------------
    cov = sc[sc["metric"].isin(["Coverage_80", "Coverage_95", "Width_80_rel", "Width_95_rel"])].copy()
    cov["nominal"] = cov["metric"].str.extract(r"(\d+)").astype(int)
    cov["kind"] = np.where(cov["metric"].str.startswith("Coverage"), "coverage", "rel_width")
    p = cov.pivot_table(index=["layer", "model", "model_label", "model_family", "level_id", "level_name", "split", "nominal"], columns="kind",
                        values="value", aggfunc="first")
    lo = cov[cov["kind"] == "coverage"].set_index(["layer", "model", "model_label", "model_family", "level_id", "level_name", "split", "nominal"])
    ic = p.reset_index()
    ic = ic.merge(lo[["ci_lo", "ci_hi"]].reset_index(), on=["layer", "model", "model_label", "model_family", "level_id", "level_name", "split", "nominal"], how="left")
    ic = ic.dropna(subset=["coverage"])
    ic["nominal_coverage"] = ic["nominal"] / 100.0
    ic["miscalibration"] = ic["coverage"] - ic["nominal_coverage"]
    ic = ic.rename(columns={"ci_lo": "coverage_ci_lo", "ci_hi": "coverage_ci_hi", "rel_width": "mean_width_over_revenue"})
    out.append(_wr(ic[["layer", "model", "model_label", "model_family", "level_id", "level_name", "split", "nominal", "nominal_coverage", "coverage",
                       "coverage_ci_lo", "coverage_ci_hi", "miscalibration", "mean_width_over_revenue"]], root / "interval_coverage.csv", float_format="%.5f"))
    ch = pd.read_csv(art / "coverage_by_horizon.csv")
    ch["level_name"] = ch["level"].map(lvl)
    ch["model_label"] = ch["model"].map(model_label)
    ch = ch.rename(columns={"level": "level_id", "c80": "coverage_80", "c95": "coverage_95", "w80_rel": "width_80_over_revenue",
                            "w95_rel": "width_95_over_revenue", "h": "horizon_weeks"})
    out.append(_wr(ch[["model", "model_label", "level_id", "level_name", "split", "horizon_weeks", "n", "coverage_80", "coverage_95",
                       "width_80_over_revenue", "width_95_over_revenue"]], root / "interval_coverage_by_horizon.csv", float_format="%.5f"))
    po = pd.read_csv(art / "per_origin_metrics.csv")
    po["level_name"] = po["level"].map(lvl)
    po["model_label"] = po["model"].map(model_label)
    po["origin_week_end"] = po["origin"].map(wk_end)
    po = po.rename(columns={"level": "level_id", "origin": "origin_week_idx"})
    out.append(_wr(po[["layer", "model", "model_label", "level_id", "level_name", "split", "origin_week_idx", "origin_week_end", "WAPE", "MASE", "RMSSE",
                       "Bias", "Pinball_scaled", "Coverage_80", "Coverage_95", "Width_80_rel", "Width_95_rel"]],
                   root / "metrics_by_origin.csv", float_format="%.5f"))

    # 7. residual diagnostics ----------------------------------------------------------------------------------------------
    rt_path = art / "residual_tests_by_node.csv"
    if rt_path.exists():
        rt = pd.read_csv(rt_path)
        rt["level_name"] = rt["level"].map(lvl)
        rt["node_label"] = rt["node_id"].map(lab)
        rt["lb_reject_5pct"] = (rt["lb_pvalue"] < 0.05).astype(int)
        rt["shapiro_reject_5pct"] = (rt["shapiro_p"] < 0.05).astype(int)
        rt = rt.rename(columns={".model": "model", "level": "level_id"})
        out.append(_wr(rt[["model", "node_id", "node_label", "level_id", "level_name", "n", "lb_stat", "lb_pvalue", "lb_reject_5pct", "lag1_acf",
                           "shapiro_p", "shapiro_reject_5pct", "resid_mean_over_mean_revenue"]], root / "residual_diagnostics_nodes.csv", float_format="%.5f"))
        sm = rt.groupby(["model", "level_id", "level_name"], sort=True).agg(
            n_nodes=("node_id", "size"), share_ljung_box_rejected=("lb_reject_5pct", "mean"), median_ljung_box_p=("lb_pvalue", "median"),
            median_lag1_acf=("lag1_acf", "median"), share_shapiro_rejected=("shapiro_reject_5pct", "mean"),
            mean_resid_over_mean_revenue=("resid_mean_over_mean_revenue", "mean")).reset_index()
        out.append(_wr(sm, root / "residual_diagnostics_summary.csv", float_format="%.5f"))

    # 8. decision / planning (ASSUMED parameters) --------------------------------------------------------------------------
    dsel = fa_all[(fa_all["split"] == "test") & fa_all["model"].isin(DEC_MODELS)].copy()
    d = cfg["decision"]
    cu, co = d["assumed_gross_margin"], d["assumed_carrying_cost"]

    def costs(df: pd.DataFrame, plan_col: str) -> tuple[pd.Series, pd.Series]:
        under = (df["y"] - df[plan_col]).clip(lower=0)
        over = (df[plan_col] - df["y"]).clip(lower=0)
        return under, over
    dsel["under_pt"], dsel["over_pt"] = costs(dsel, "yhat")
    dsel["under_q"], dsel["over_q"] = costs(dsel, "hi80")
    dsel["hit_q"] = (dsel["y"] <= dsel["hi80"]).astype(float)
    g = dsel.groupby(["model", "level", "node_id"], sort=True).agg(
        actual=("y", "sum"), plan_point=("yhat", "sum"), plan_q90=("hi80", "sum"), shortfall_point=("under_pt", "sum"), excess_point=("over_pt", "sum"),
        shortfall_q90=("under_q", "sum"), excess_q90=("over_q", "sum"), service_level_q90=("hit_q", "mean"), n_weeks=("y", "size")).reset_index()
    g["safety_buffer_usd"] = g["plan_q90"] - g["plan_point"]
    g["safety_buffer_pct_of_actual"] = 100 * g["safety_buffer_usd"] / g["actual"].replace(0, np.nan)
    g["ASSUMED_cost_point_plan_usd"] = cu * g["shortfall_point"] + co * g["excess_point"]
    g["ASSUMED_cost_q90_plan_usd"] = cu * g["shortfall_q90"] + co * g["excess_q90"]
    g["ASSUMED_cost_change_pct_q90_vs_point"] = 100 * (g["ASSUMED_cost_q90_plan_usd"] / g["ASSUMED_cost_point_plan_usd"].replace(0, np.nan) - 1)
    g["ASSUMED_gross_margin_on_unmet_revenue"] = cu
    g["ASSUMED_carrying_cost_on_excess_plan"] = co
    g["ASSUMED_plan_quantile"] = d["plan_quantile"]
    g["model_label"] = g["model"].map(model_label)
    g["level_name"] = g["level"].map(lvl)
    g["node_label"] = g["node_id"].map(lab)
    g = g.rename(columns={"level": "level_id", "actual": "actual_revenue_usd", "plan_point": "plan_at_point_forecast_usd", "plan_q90": "plan_at_q90_usd",
                          "shortfall_point": "unmet_revenue_point_plan_usd", "excess_point": "excess_plan_point_plan_usd",
                          "shortfall_q90": "unmet_revenue_q90_plan_usd", "excess_q90": "excess_plan_q90_plan_usd", "service_level_q90": "share_weeks_actual_within_q90"})
    out.append(_wr(g[["model", "model_label", "level_id", "level_name", "node_id", "node_label", "n_weeks", "actual_revenue_usd", "plan_at_point_forecast_usd",
                      "plan_at_q90_usd", "safety_buffer_usd", "safety_buffer_pct_of_actual", "unmet_revenue_point_plan_usd", "excess_plan_point_plan_usd",
                      "unmet_revenue_q90_plan_usd", "excess_plan_q90_plan_usd", "share_weeks_actual_within_q90", "ASSUMED_cost_point_plan_usd",
                      "ASSUMED_cost_q90_plan_usd", "ASSUMED_cost_change_pct_q90_vs_point", "ASSUMED_gross_margin_on_unmet_revenue",
                      "ASSUMED_carrying_cost_on_excess_plan", "ASSUMED_plan_quantile"]], root / "decision_planning.csv", float_format="%.4f"))
    # sensitivity: plan at point / q90 / q97.5 under a grid of ASSUMED cost parameters, aggregated per level
    srows = []
    for plan_name, col in (("point forecast (q50)", "yhat"), ("upper 80% bound (q90)", "hi80"), ("upper 95% bound (q97.5)", "hi95")):
        under = (dsel["y"] - dsel[col]).clip(lower=0)
        over = (dsel[col] - dsel["y"]).clip(lower=0)
        t = pd.DataFrame({"model": dsel["model"], "level": dsel["level"], "y": dsel["y"], "under": under, "over": over}).groupby(["model", "level"]).sum().reset_index()
        for m_ in SENS_MARGIN:
            for c_ in SENS_CARRY:
                x = t.copy()
                x["plan_rule"] = plan_name
                x["ASSUMED_gross_margin"] = m_
                x["ASSUMED_carrying_cost"] = c_
                x["ASSUMED_cost_pct_of_revenue"] = 100 * (m_ * x["under"] + c_ * x["over"]) / x["y"]
                x["is_default_assumption"] = int(m_ == cu and c_ == co)
                srows.append(x)
    sens = pd.concat(srows, ignore_index=True)
    sens["model_label"] = sens["model"].map(model_label)
    sens["level_name"] = sens["level"].map(lvl)
    sens = sens.rename(columns={"level": "level_id", "y": "actual_revenue_usd", "under": "unmet_revenue_usd", "over": "excess_plan_usd"})
    out.append(_wr(sens[["model", "model_label", "level_id", "level_name", "plan_rule", "ASSUMED_gross_margin", "ASSUMED_carrying_cost", "actual_revenue_usd",
                         "unmet_revenue_usd", "excess_plan_usd", "ASSUMED_cost_pct_of_revenue", "is_default_assumption"]],
                   root / "decision_sensitivity.csv", float_format="%.4f"))

    # 9. design and data quality ---------------------------------------------------------------------------------------------
    od = []
    for split in ("burnin", "validation", "test"):
        for o in org[split]:
            od.append({"origin_week_idx": o, "split": {"burnin": "burn-in (training and conformal calibration only)", "validation": "validation",
                                                      "test": "test (final held-out)"}[split],
                       "train_through": wk_end[o], "forecast_first_week_start": wk_start[o + 1], "forecast_last_week_start": wk_start[o + org["horizon"]],
                       "horizon_weeks": org["horizon"], "n_training_weeks": o})
    out.append(_wr(pd.DataFrame(od), root / "origins_design.csv"))
    out.append(_wr(cleaning, root / "cleaning_log.csv"))
    return out
