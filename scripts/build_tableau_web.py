"""Build the five flattened, pre-calculated CSVs in tableau/web/ for Tableau Public web authoring.

Every value is read from the existing exports in tableau/data/ (no modelling, no new numbers). Rows are filtered, pivoted, ranked or
aggregated (sums of dollars before ratios), never re-estimated. Fractions (0.0256 = 2.56%) are used for every rate so one Tableau
percent format applies everywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tableau" / "data"
OUT = ROOT / "tableau" / "web"

# 11 models shown in the scorecard: baselines, statsforecast, LightGBM, R fable base and MinT. lgbm_conformal has the same point forecast as
# lgbm_q (duplicate WAPE), the 30 py_* point-only reconciled variants and the geo_* sub-hierarchy variants are excluded.
SCORECARD_MODELS = ["naive", "snaive", "ma4", "ets", "sarima_111_010", "sarima_airline", "lgbm_q", "r_ets_base", "r_arima_base", "r_ets_mint", "r_arima_mint"]
# calibration additionally keeps the conformal intervals (different from lgbm_q) and every R reconciliation on the full 154 nodes
CALIBRATION_MODELS = SCORECARD_MODELS + ["lgbm_conformal"] + [f"r_{b}_{m}" for b in ("ets", "arima") for m in ("bu", "ols", "wls_struct", "wls_var")]
EXPLORER_LABELS = ["Seasonal naive (52 wk)", "Moving average (4 wk)", "LightGBM quantile", "R fable ARIMA MinT shrink", "R fable ETS MinT shrink"]
EXPLORER_LEVELS = [1, 2, 3]
EXPLORER_HORIZONS = [1, 4]
METHOD_ORDER = ["base (unreconciled)", "bottom-up", "OLS", "WLS structural", "WLS variance", "MinT shrink"]
HIER = "hierarchy (levels 1-9)"
BAND_SEP = " | "
UNDER_COVER_TOL = 0.05   # a nominal interval under-covers when coverage < nominal - 0.05


def colour_group(family: pd.Series, reconciled: pd.Series | None = None) -> pd.Series:
    """Colour semantics of the dashboard: baseline grey, statistical blue, machine learning orange, reconciled teal."""
    f = family.astype(str)
    g = pd.Series("Statistical", index=f.index)
    g[f == "baseline"] = "Baseline"
    g[f.str.startswith("machine learning")] = "Machine learning"
    g[f.str.startswith("reconciled")] = "Reconciled"
    if reconciled is not None:
        g[reconciled] = "Reconciled"
    return g


def _yn(s: pd.Series) -> pd.Series:
    return np.where(s, "Yes", "No")


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(SRC / name)


def _wide(df: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame:
    d = df[df["metric"] == metric]
    return d[[*keys, "value", "ci_lo", "ci_hi"]]


def scorecard() -> pd.DataFrame:
    s = _read("metric_scorecard.csv")
    s = s[(s["split"] == "test") & (s["layer"] == HIER) & s["model"].isin(SCORECARD_MODELS)]
    keys = ["model", "model_label", "model_family", "level_id", "level_name"]
    w = _wide(s, keys, "WAPE").rename(columns={"value": "wape_pct", "ci_lo": "wape_ci_lo", "ci_hi": "wape_ci_hi"})
    r = _wide(s, keys, "WAPE_ratio_vs_snaive").rename(columns={"value": "wape_ratio_vs_snaive", "ci_lo": "wape_ratio_ci_lo", "ci_hi": "wape_ratio_ci_hi"})
    m = _wide(s, keys, "MASE")[[*keys, "value"]].rename(columns={"value": "mase"})
    d = w.merge(r, on=keys, validate="1:1").merge(m, on=keys, validate="1:1")
    d = d.rename(columns={"level_id": "level_order"})
    d["is_baseline"] = _yn(d["model_family"] == "baseline")
    d["rank_in_level"] = d.groupby("level_order")["wape_pct"].rank(method="min").astype(int)
    d["is_best_in_level"] = _yn(d["rank_in_level"] == 1)
    d["better_than_snaive"] = _yn(d["wape_ratio_vs_snaive"] < 1.0)
    d["colour_group"] = colour_group(d["model_family"])
    w_ = d.pivot(index="level_order", columns="model", values="wape_pct")
    d["ma4_beats_ets_in_level"] = d["level_order"].map(dict(zip(w_.index, _yn(w_["ma4"] < w_["ets"]), strict=True)))
    d["ma4_minus_ets_wape"] = d["level_order"].map(dict(zip(w_.index, (w_["ma4"] - w_["ets"]).round(6), strict=True)))
    cols = ["model_label", "model", "model_family", "colour_group", "is_baseline", "level_order", "level_name", "wape_pct", "wape_ci_lo", "wape_ci_hi",
            "wape_ratio_vs_snaive", "wape_ratio_ci_lo", "wape_ratio_ci_hi", "mase", "rank_in_level", "is_best_in_level", "better_than_snaive",
            "ma4_beats_ets_in_level", "ma4_minus_ets_wape"]
    return d[cols].sort_values(["level_order", "rank_in_level", "model"]).reset_index(drop=True)


def best_sarima() -> str:
    """The statsforecast SARIMA variant with the lowest mean test WAPE over the explorer levels (display choice only)."""
    s = _read("metric_scorecard.csv")
    s = s[(s["split"] == "test") & (s["layer"] == HIER) & (s["metric"] == "WAPE")]
    s = s[s["model"].isin(["sarima_111_010", "sarima_airline"]) & s["level_id"].isin(EXPLORER_LEVELS)]
    best = s.groupby(["model", "model_label"])["value"].mean().sort_values().index[0]
    return best[1]


def forecast_explorer() -> pd.DataFrame:
    f = _read("forecast_vs_actual.csv")
    labels = [*EXPLORER_LABELS, best_sarima()]
    f = f[f["model_label"].isin(labels) & f["level_id"].isin(EXPLORER_LEVELS) & f["horizon_weeks"].isin(EXPLORER_HORIZONS)]
    # forecast_vs_actual.csv holds only the 6 final-test origins (weeks 254-277); assert it so the "test only" claim is checked
    o = _read("origins_design.csv")
    test_origins = set(o.loc[o["split"].str.startswith("test"), "origin_week_idx"])
    assert set(f["origin_week_idx"]) <= test_origins
    d = f.rename(columns={"level_id": "level_order", "in_80_interval": "in_80", "in_95_interval": "in_95"})
    d["model_order"] = d["model_label"].map({m: i + 1 for i, m in enumerate(labels)})
    d["band_width_usd"] = (d["hi80_usd"] - d["lo80_usd"]).round(2)
    d["series_label"] = d["model_label"] + BAND_SEP + d["node_label"]
    d["colour_group"] = colour_group(d["model_family"])
    cols = ["level_order", "level_name", "node_id", "node_label", "series_label", "model_order", "model_label", "model_family", "colour_group", "origin_week_idx",
            "forecast_made_after", "target_week_idx", "target_week_start", "horizon_weeks", "forecast_usd", "actual_usd",
            "lo80_usd", "hi80_usd", "band_width_usd", "lo95_usd", "hi95_usd", "in_80", "in_95"]
    return d[cols].sort_values(["level_order", "node_id", "model_order", "horizon_weeks", "target_week_idx"]).reset_index(drop=True)


def reconciliation() -> pd.DataFrame:
    r = _read("reconciliation_comparison.csv")
    r = r[(r["split"] == "test") & (r["engine"] == "R fable") & (r["structure"] == "full 154 nodes")]
    keys = ["base_model", "method", "level_id", "level_name"]
    w = _wide(r, keys, "WAPE").rename(columns={"value": "wape_pct", "ci_lo": "wape_ci_lo", "ci_hi": "wape_ci_hi"})
    q = _wide(r, keys, "WAPE_ratio_vs_base").rename(columns={"value": "wape_ratio_vs_base", "ci_lo": "wape_ratio_ci_lo", "ci_hi": "wape_ratio_ci_hi"})
    d = w.merge(q, on=keys, validate="1:1").rename(columns={"level_id": "level_order"})
    d["base_model"] = d["base_model"].str.replace("R ", "", regex=False)
    d["method_order"] = d["method"].map({m: i + 1 for i, m in enumerate(METHOD_ORDER)})
    d["is_reconciled"] = _yn(d["method"] != "base (unreconciled)")
    d["better_than_base"] = _yn(d["wape_ratio_vs_base"] < 1.0)
    d["colour_group"] = np.where(d["method"] == "base (unreconciled)", "Statistical", "Reconciled")
    cols = ["base_model", "method_order", "method", "colour_group", "is_reconciled", "level_order", "level_name", "wape_pct", "wape_ci_lo", "wape_ci_hi",
            "wape_ratio_vs_base", "wape_ratio_ci_lo", "wape_ratio_ci_hi", "better_than_base"]
    return d[cols].sort_values(["base_model", "level_order", "method_order"]).reset_index(drop=True)


def calibration() -> pd.DataFrame:
    c = _read("interval_coverage.csv")
    c = c[(c["split"] == "test") & (c["layer"] == HIER) & c["model"].isin(CALIBRATION_MODELS)]
    d = c.rename(columns={"level_id": "level_order", "nominal": "nominal_pct"})
    d["is_reconciled"] = _yn(d["model_family"].str.startswith("reconciled"))
    d["colour_group"] = colour_group(d["model_family"])
    d["abs_miscalibration"] = (d["coverage"] - d["nominal_coverage"]).abs()
    best = d.groupby(["level_order", "nominal_pct"])["abs_miscalibration"].transform("min")
    d["is_best_interval_in_level"] = _yn(d["abs_miscalibration"] <= best + 1e-12)
    d["under_covers"] = _yn(d["coverage"] < d["nominal_coverage"] - UNDER_COVER_TOL)
    cols = ["model_label", "model", "model_family", "colour_group", "is_reconciled", "level_order", "level_name", "nominal_pct", "nominal_coverage", "coverage",
            "coverage_ci_lo", "coverage_ci_hi", "miscalibration", "abs_miscalibration", "is_best_interval_in_level", "under_covers", "mean_width_over_revenue"]
    return d[cols].sort_values(["level_order", "nominal_pct", "model"]).reset_index(drop=True)


def planning_buffer() -> pd.DataFrame:
    p = _read("decision_planning.csv")
    fam = _read("metric_scorecard.csv")[["model", "model_family"]].drop_duplicates()
    g = p.groupby(["model", "model_label", "level_id", "level_name"], as_index=False).agg(
        n_nodes=("node_id", "nunique"), actual=("actual_revenue_usd", "sum"), buffer=("safety_buffer_usd", "sum"),
        cost_point=("ASSUMED_cost_point_plan_usd", "sum"), cost_q90=("ASSUMED_cost_q90_plan_usd", "sum"),
        within=("share_weeks_actual_within_q90", "mean"), margin=("ASSUMED_gross_margin_on_unmet_revenue", "first"),
        carry=("ASSUMED_carrying_cost_on_excess_plan", "first"), q=("ASSUMED_plan_quantile", "first"))
    g = g.merge(fam, on="model", how="left", validate="m:1")
    g["buffer_pct_of_revenue"] = g["buffer"] / g["actual"]
    g["ASSUMED_cost_point_plan_pct_of_revenue"] = g["cost_point"] / g["actual"]
    g["ASSUMED_cost_q90_plan_pct_of_revenue"] = g["cost_q90"] / g["actual"]
    g["ASSUMED_cost_reduction_pct"] = 1 - g["cost_q90"] / g["cost_point"]
    d = g.rename(columns={"level_id": "level_order", "within": "realised_share_within_q90", "actual": "actual_revenue_usd", "buffer": "buffer_usd",
                          "cost_point": "ASSUMED_cost_point_plan_usd", "cost_q90": "ASSUMED_cost_q90_plan_usd", "margin": "ASSUMED_gross_margin",
                          "carry": "ASSUMED_carrying_cost", "q": "ASSUMED_plan_quantile"})
    d["q90_service_gap"] = d["realised_share_within_q90"] - d["ASSUMED_plan_quantile"]
    d["colour_group"] = colour_group(d["model_family"])
    cols = ["model_label", "model", "model_family", "colour_group", "level_order", "level_name", "n_nodes", "actual_revenue_usd", "buffer_usd", "buffer_pct_of_revenue",
            "realised_share_within_q90", "q90_service_gap", "ASSUMED_gross_margin", "ASSUMED_carrying_cost", "ASSUMED_plan_quantile",
            "ASSUMED_cost_point_plan_usd", "ASSUMED_cost_q90_plan_usd", "ASSUMED_cost_point_plan_pct_of_revenue", "ASSUMED_cost_q90_plan_pct_of_revenue",
            "ASSUMED_cost_reduction_pct"]
    return d[cols].sort_values(["level_order", "model"]).reset_index(drop=True)


BUILDERS = {"scorecard.csv": scorecard, "forecast_explorer.csv": forecast_explorer, "reconciliation.csv": reconciliation,
            "calibration.csv": calibration, "planning_buffer.csv": planning_buffer}
SNAIVE = "Seasonal naive (52 wk)"


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def pts(x: float) -> str:
    return f"{100 * abs(x):.1f} points"


def _test_design() -> tuple[int, int]:
    """(number of test origins, number of test forecast weeks) from origins_design.csv."""
    o = _read("origins_design.csv")
    o = o[o["split"].str.startswith("test")]
    return len(o), int(o["horizon_weeks"].sum())


def _store_share(best_label: str) -> tuple[int, int]:
    """Stores (level 3 nodes) where best_label has lower pooled WAPE than seasonal naive, over all horizons and test origins."""
    f = _read("forecast_vs_actual.csv")
    f = f[f["level_id"] == 3]
    w = f.groupby(["node_id", "model_label"]).agg(err=("abs_error_usd", "sum"), act=("actual_usd", "sum"))
    w["wape"] = w["err"] / w["act"]
    w = w["wape"].unstack("model_label")
    return int((w[best_label] < w[SNAIVE]).sum()), len(w)


def kpis(fr: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sc, cal, rec = fr["scorecard.csv"], fr["calibration.csv"], fr["reconciliation.csv"]
    best = sc[sc["is_best_in_level"] == "Yes"].set_index("level_order")
    b1 = best.loc[1]
    sn1 = sc[(sc["model"] == "snaive") & (sc["level_order"] == 1)].iloc[0]
    won = int((best["model"] == b1["model"]).sum())
    top_model = best["model_label"].value_counts().index[0]
    top_n = int((best["model_label"] == top_model).sum())
    b3 = best.loc[3]
    k, n = _store_share(b3["model_label"])
    c1 = cal[(cal["level_order"] == 1) & (cal["nominal_pct"] == 80) & (cal["is_best_interval_in_level"] == "Yes")].iloc[0]
    n_fc = int(_read("forecast_vs_actual.csv").query("level_id == 1 and model_label == @c1.model_label").shape[0])
    mint = rec[(rec["base_model"] == "ARIMA") & (rec["method"] == "MinT shrink") & (rec["level_order"] == 1)].iloc[0]
    n_or, n_wk = _test_design()
    rows = [
        ("Best WAPE at Total", b1["wape_pct"], pct(b1["wape_pct"]),
         f"{b1['model_label']}, the lowest-WAPE model at Total on the final test window. WAPE is the sum of absolute errors divided by the sum of actual revenue, 4-week horizon."),
        ("Error cut vs seasonal naive at Total", 1 - b1["wape_ratio_vs_snaive"], pct(1 - b1["wape_ratio_vs_snaive"]),
         f"Relative WAPE reduction of {b1['model_label']} versus seasonal naive ({pct(sn1['wape_pct'])}) at Total."),
        ("Levels won by the best Total model", won, f"{won} of {len(best)}",
         f"Levels where {b1['model_label']} has the lowest WAPE. {top_model} wins {top_n} of {len(best)}."),
        ("Stores where the best store model beats seasonal naive", k / n, f"{k} of {n}",
         f"Stores where {b3['model_label']} (best at Store level) has lower WAPE than seasonal naive, computed per store over horizons 1 to 4 and {n_or} test origins."),
        ("80% interval coverage, best interval model at Total", c1["coverage"], pct(c1["coverage"]),
         f"{c1['model_label']} comes closest to the 80% target at Total. Based on only {n_fc} forecasts, so it is noisy; it is the best case, not the typical one."),
        ("MinT shrink WAPE ratio vs ARIMA base at Total", mint["wape_ratio_vs_base"], f"{mint['wape_ratio_vs_base']:.3f}",
         "WAPE of fable ARIMA with MinT shrink divided by the unreconciled fable ARIMA at Total. Below 1.00 means reconciliation helped."),
        ("Test origins", n_or, f"{n_or} origins",
         f"Final held-out window: {n_or} rolling origins of 4 weeks each ({n_wk} forecast weeks). Rankings among close models are tentative."),
    ]
    d = pd.DataFrame(rows, columns=["kpi_label", "kpi_value", "kpi_display", "kpi_note"])
    d.insert(0, "kpi_order", range(1, len(d) + 1))
    return d


def findings(fr: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sc, cal, rec, pl = fr["scorecard.csv"], fr["calibration.csv"], fr["reconciliation.csv"], fr["planning_buffer.csv"]
    lv = dict(zip(sc["level_order"], sc["level_name"], strict=True))
    nl = len(lv)
    best = sc[sc["is_best_in_level"] == "Yes"].set_index("level_order")
    n_or, n_wk = _test_design()
    hn = _read("hierarchy_nodes.csv")
    n_stores, n_states = int((hn["level_id"] == 3).sum()), int((hn["level_id"] == 2).sum())
    rows: list[dict] = []

    def add(section: int, name: str, ftype: str, head: str, detail: str, model: str = "", level: str = "", assumed: str = "No", src: str = "") -> None:
        rows.append({"section": section, "section_name": name, "finding_type": ftype, "finding_headline": head, "finding_detail": detail,
                     "model_label": model, "level_name": level, "is_assumed": assumed, "evidence_file": src})

    # section 2: which model wins by level
    wins = best["model_label"].value_counts()
    top = wins.index[0]
    others = [f"{m} wins {', '.join(lv[i] for i in best.index[best['model_label'] == m])}" for m in wins.index[1:]]
    tw = sc[(sc["model_label"] == top) & sc["level_order"].isin(best.index[best["model_label"] == top])]
    sn = sc[(sc["model"] == "snaive") & sc["level_order"].isin(tw["level_order"])]
    head = f"{top} is the most accurate model at {wins.iloc[0]} of {nl} levels"
    if others:
        head += "; " + "; ".join(others)
    add(2, "Which model wins by level", "headline", head,
        f"Where it wins, its WAPE runs from {pct(tw['wape_pct'].min())} to {pct(tw['wape_pct'].max())}, against {pct(sn['wape_pct'].min())} to "
        f"{pct(sn['wape_pct'].max())} for seasonal naive at the same levels ({n_or} test origins).", top, "", "No", "scorecard.csv")
    e = sc.drop_duplicates("level_order").set_index("level_order")
    nm = int((e["ma4_beats_ets_in_level"] == "Yes").sum())
    gap = e["ma4_minus_ets_wape"].idxmin()
    w = sc[sc["level_order"] == gap].set_index("model")["wape_pct"]
    add(2, "Which model wins by level", "callout", f"A plain 4-week moving average beats ETS at {nm} of {nl} levels",
        f"The gap is widest at {lv[gap]}: the moving average has WAPE {pct(w['ma4'])} against {pct(w['ets'])} for ETS, so the more complex model adds nothing here.",
        "Moving average (4 wk)", lv[gap], "No", "scorecard.csv")

    # section 3: forecast vs actual with the 80% band (all horizons, Total, LightGBM quantile)
    f = _read("forecast_vs_actual.csv")
    f = f[(f["level_id"] == 1) & (f["model_label"] == "LightGBM quantile")]
    hit, tot = int(f["in_80_interval"].sum()), len(f)
    width = float(((f["hi80_usd"] - f["lo80_usd"]) / f["actual_usd"]).mean())
    cov = hit / tot
    word = "below" if cov < 0.75 else ("above" if cov > 0.85 else "close to")
    add(3, "Forecast vs actual", "headline", f"LightGBM's 80% band held the actual in {hit} of {tot} Total forecasts ({pct(cov)}), {word} the 80% target",
        f"The average 80% band is {pct(width)} of weekly revenue wide. The explorer shows horizons 1 and 4 for 14 nodes; the count here uses all 4 horizons of the {n_or} test origins.",
        "LightGBM quantile", "Total", "No", "forecast_vs_actual.csv")

    # section 4: does reconciliation help
    ar = rec[rec["base_model"] == "ARIMA"]
    imp = ar.groupby("method")["better_than_base"].apply(lambda s: int((s == "Yes").sum())).drop("base (unreconciled)")
    full = list(imp.index[imp == nl])
    bu_total = ar[(ar["method"] == "bottom-up") & (ar["level_order"] == 1)]["wape_ratio_vs_base"].iloc[0]
    mint = ar[ar["method"] == "MinT shrink"]["wape_ratio_vs_base"]
    es = rec[(rec["base_model"] == "ETS") & (rec["method"] == "MinT shrink")]
    es_tot = es[es["level_order"] == 1]["wape_ratio_vs_base"].iloc[0]
    lead = f"Only {full[0]} improves ARIMA at all {nl} levels" if len(full) == 1 else f"{', '.join(full) or 'No method'} improves ARIMA at all {nl} levels"
    bu_dir = "worse" if bu_total > 1 else "better"
    add(4, "Does reconciliation help", "headline", f"{lead}; bottom-up is {bu_dir} by {pct(abs(bu_total - 1))} at Total (it beats the base at {imp['bottom-up']} of {nl} levels)",
        f"MinT shrink cuts ARIMA WAPE by {pct(1 - mint.max())} to {pct(1 - mint.min())} depending on level. For ETS it helps at "
        f"{int((es['better_than_base'] == 'Yes').sum())} of {nl} levels and is {pct(abs(es_tot - 1))} {'worse' if es_tot > 1 else 'better'} at Total.",
        "R fable ARIMA MinT shrink", "", "No", "reconciliation.csv")

    # section 5: can we trust the intervals
    c80 = cal[cal["nominal_pct"] == 80]
    nu, nc = int((c80["under_covers"] == "Yes").sum()), len(c80)
    t = c80[c80["level_order"] == 1]
    bad = t[t["model"].str.match(r"r_(ets|arima)_(bu|ols|wls_struct|wls_var)$")]
    bt = t[t["is_best_interval_in_level"] == "Yes"].iloc[0]
    add(5, "Can we trust the intervals", "headline", f"{nu} of {nc} model and level cases under-cover the 80% interval by more than 5 points",
        f"At Total the closest model is {bt['model_label']} at {pct(bt['coverage'])}, from only {tot} forecasts. The fable bottom-up, OLS and WLS intervals cover only "
        f"{pct(bad['coverage'].min())} to {pct(bad['coverage'].max())} of actuals at Total.", bt["model_label"], "Total", "No", "calibration.csv")

    # section 6: recommendations
    b3, b9, b1 = best.loc[3], best.loc[nl], best.loc[1]
    mint_w = sc[(sc["model"] == "r_arima_mint") & (sc["level_order"] == 1)]["wape_pct"].iloc[0]
    add(6, "Recommendations", "recommendation", f"Plan store and department volumes from MinT-reconciled ARIMA: WAPE {pct(b3['wape_pct'])} by store, {pct(b9['wape_pct'])} by store and department",
        f"It has the lowest WAPE at {int((best['model'] == 'r_arima_mint').sum())} of {nl} levels. At Total the best model, {b1['model_label']}, is ahead by "
        f"{pts(mint_w - b1['wape_pct'])}, too small to matter given {n_or} test origins.", "R fable ARIMA MinT shrink", "", "No", "scorecard.csv")
    m4 = sc[sc["model"] == "ma4"].set_index("level_order")["wape_pct"]
    s4 = sc[sc["model"] == "snaive"].set_index("level_order")["wape_pct"]
    add(6, "Recommendations", "recommendation", f"Keep the 4-week moving average as the benchmark to beat, and do not use seasonal naive (WAPE {pct(s4.min())} to {pct(s4.max())})",
        f"The moving average's WAPE is {pct(m4.min())} to {pct(m4.max())} across levels and beats ETS at {nm} of {nl} levels. Any new model should be tested against it first.",
        "Moving average (4 wk)", "", "No", "scorecard.csv")
    p = pl[pl["model"] == "r_arima_mint"].set_index("level_order")
    a = p.iloc[0]
    add(6, "Recommendations", "recommendation",
        f"ASSUMED costs: planning at the 90th percentile adds a buffer of {pct(p.loc[1, 'buffer_pct_of_revenue'])} of revenue at Total and {pct(p.loc[nl, 'buffer_pct_of_revenue'])} at "
        f"{lv[nl]}, and cuts the ASSUMED cost by {pct(p.loc[1, 'ASSUMED_cost_reduction_pct'], 0)} at Total",
        f"ASSUMED parameters: {pct(a['ASSUMED_gross_margin'], 0)} margin lost on unmet revenue, {pct(a['ASSUMED_carrying_cost'], 0)} carrying cost on excess, plan at q{int(round(100 * a['ASSUMED_plan_quantile']))}. "
        f"These are scenario inputs, not estimates from the data. Figures are for MinT-reconciled ARIMA.", "R fable ARIMA MinT shrink", "Total", "Yes", "planning_buffer.csv")
    add(6, "Recommendations", "recommendation", f"Treat the buffer as a floor: only {pct(p.loc[1, 'realised_share_within_q90'], 0)} of Total weeks fell within the 90th percentile plan, and intervals under-cover",
        f"Results come from {n_or} test origins ({n_wk} weeks) and {n_stores} stores in {n_states} states, so re-test on your own data before committing inventory.",
        "R fable ARIMA MinT shrink", "Total", "No", "planning_buffer.csv")
    d = pd.DataFrame(rows)
    d.insert(0, "finding_order", range(1, len(d) + 1))
    return d


DERIVED = {"kpis.csv": kpis, "findings.csv": findings}


def build_all(out: Path = OUT) -> dict[str, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for name, fn in BUILDERS.items():
        frames[name] = fn()
    for name, fn in DERIVED.items():
        frames[name] = fn(frames)
    for name, df in frames.items():
        df.to_csv(out / name, index=False, encoding="utf-8", float_format="%.6f", lineterminator="\n")
    return frames


def main() -> int:
    for name, df in build_all().items():
        p = OUT / name
        print(f"{name}: {len(df):,} rows, {p.stat().st_size / 1e6:.3f} MB, {len(df.columns)} columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
