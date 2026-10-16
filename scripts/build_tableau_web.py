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
    cols = ["model_label", "model", "model_family", "is_baseline", "level_order", "level_name", "wape_pct", "wape_ci_lo", "wape_ci_hi",
            "wape_ratio_vs_snaive", "wape_ratio_ci_lo", "wape_ratio_ci_hi", "mase", "rank_in_level", "is_best_in_level", "better_than_snaive"]
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
    cols = ["level_order", "level_name", "node_id", "node_label", "model_order", "model_label", "model_family", "origin_week_idx",
            "forecast_made_after", "target_week_idx", "target_week_start", "horizon_weeks", "forecast_usd", "actual_usd",
            "lo80_usd", "hi80_usd", "lo95_usd", "hi95_usd", "in_80", "in_95"]
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
    cols = ["base_model", "method_order", "method", "is_reconciled", "level_order", "level_name", "wape_pct", "wape_ci_lo", "wape_ci_hi",
            "wape_ratio_vs_base", "wape_ratio_ci_lo", "wape_ratio_ci_hi", "better_than_base"]
    return d[cols].sort_values(["base_model", "level_order", "method_order"]).reset_index(drop=True)


def calibration() -> pd.DataFrame:
    c = _read("interval_coverage.csv")
    c = c[(c["split"] == "test") & (c["layer"] == HIER) & c["model"].isin(CALIBRATION_MODELS)]
    d = c.rename(columns={"level_id": "level_order", "nominal": "nominal_pct"})
    d["is_reconciled"] = _yn(d["model_family"].str.startswith("reconciled"))
    cols = ["model_label", "model", "model_family", "is_reconciled", "level_order", "level_name", "nominal_pct", "nominal_coverage", "coverage",
            "coverage_ci_lo", "coverage_ci_hi", "miscalibration", "mean_width_over_revenue"]
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
    cols = ["model_label", "model", "model_family", "level_order", "level_name", "n_nodes", "actual_revenue_usd", "buffer_usd", "buffer_pct_of_revenue",
            "realised_share_within_q90", "q90_service_gap", "ASSUMED_gross_margin", "ASSUMED_carrying_cost", "ASSUMED_plan_quantile",
            "ASSUMED_cost_point_plan_usd", "ASSUMED_cost_q90_plan_usd", "ASSUMED_cost_point_plan_pct_of_revenue", "ASSUMED_cost_q90_plan_pct_of_revenue",
            "ASSUMED_cost_reduction_pct"]
    return d[cols].sort_values(["level_order", "model"]).reset_index(drop=True)


BUILDERS = {"scorecard.csv": scorecard, "forecast_explorer.csv": forecast_explorer, "reconciliation.csv": reconciliation,
            "calibration.csv": calibration, "planning_buffer.csv": planning_buffer}


def build_all(out: Path = OUT) -> dict[str, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name, fn in BUILDERS.items():
        df = fn()
        df.to_csv(out / name, index=False, encoding="utf-8", float_format="%.6f", lineterminator="\n")
        frames[name] = df
    return frames


def main() -> int:
    for name, df in build_all().items():
        p = OUT / name
        print(f"{name}: {len(df):,} rows, {p.stat().st_size / 1e6:.3f} MB, {len(df.columns)} columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
