"""Static matplotlib figures from the exported (derived) tables. Colour palette is fixed and colour-blind safe."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import rpath  # noqa: E402

PALETTE = {
    "naive": "#9AA0A6", "snaive": "#5F6368", "ma4": "#BDC1C6", "ets": "#E69F00", "sarima_airline": "#009E73", "sarima_111_010": "#66C2A5",
    "lgbm_q": "#0072B2", "lgbm_conformal": "#56B4E9", "r_ets_base": "#D55E00", "r_ets_mint": "#CC79A7", "r_arima_base": "#8C510A",
    "r_arima_mint": "#B2182B",
}
LEVEL_SHORT = {1: "Total", 2: "State", 3: "Store", 4: "Category", 5: "Dept", 6: "State x Cat", 7: "State x Dept", 8: "Store x Cat", 9: "Store x Dept",
               10: "Item", 12: "Item x Store"}


def _style():
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25})


def make_figures(cfg: dict) -> list[str]:
    _style()
    td = rpath(cfg, "tableau", "data")
    fd = rpath(cfg, "docs", "figures")
    fd.mkdir(parents=True, exist_ok=True)
    sc = pd.read_csv(td / "metric_scorecard.csv")
    done = []

    # 1. WAPE by level for the main base models (test window) with bootstrap CI
    models = ["naive", "snaive", "ma4", "ets", "sarima_airline", "lgbm_q"]
    d = sc[(sc["split"] == "test") & (sc["metric"] == "WAPE") & sc["model"].isin(models) & sc["level_id"].between(1, 9)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.13
    for i, m in enumerate(models):
        x = d[d["model"] == m].sort_values("level_id")
        ax.bar(x["level_id"] + (i - 2.5) * w, 100 * x["value"], w, color=PALETTE[m], label=x["model_label"].iloc[0] if len(x) else m,
               yerr=[100 * (x["value"] - x["ci_lo"]), 100 * (x["ci_hi"] - x["value"])], error_kw={"lw": 0.6})
    ax.set_xticks(range(1, 10))
    ax.set_xticklabels([LEVEL_SHORT[i] for i in range(1, 10)], rotation=20)
    ax.set_ylabel("WAPE (%), 4-week horizon")
    ax.set_title("Test-window WAPE by hierarchy level (95% bootstrap intervals)")
    ax.legend(ncol=3, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(fd / "wape_by_level.png")
    plt.close(fig)
    done.append("wape_by_level.png")

    # 2. total revenue: forecast vs actual with intervals for several models over the test window (h=1 and h=4)
    fva = pd.read_csv(td / "forecast_vs_actual.csv")
    act = pd.read_csv(td / "actuals_weekly.csv")
    tot = act[act["node_id"] == "T"].copy()
    tot["week_start"] = pd.to_datetime(tot["week_start"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, m in zip(axes, ["snaive", "sarima_airline", "lgbm_q"], strict=True):
        z = fva[(fva["model"] == m) & (fva["node_id"] == "T")].copy()
        z["target_week_start"] = pd.to_datetime(z["target_week_start"])
        hist = tot[tot["week_idx"] >= z["target_week_idx"].min() - 20]
        ax.plot(hist["week_start"], hist["revenue_usd"] / 1e3, color="black", lw=1, label="Actual")
        for hh, a in ((1, 0.35), (4, 0.2)):
            zz = z[z["horizon_weeks"] == hh].sort_values("target_week_start")
            ax.fill_between(zz["target_week_start"], zz["lo95_usd"] / 1e3, zz["hi95_usd"] / 1e3, color=PALETTE[m], alpha=a * 0.6)
            ax.plot(zz["target_week_start"], zz["forecast_usd"] / 1e3, color=PALETTE[m], lw=1.2, ls="-" if hh == 1 else "--", label=f"h={hh}")
        ax.set_title(z["model_label"].iloc[0])
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Weekly revenue, thousand USD (total)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Total weekly revenue: forecasts at 1 and 4 weeks ahead with 95% intervals (test window)")
    fig.tight_layout()
    fig.savefig(fd / "forecast_vs_actual_total.png")
    plt.close(fig)
    done.append("forecast_vs_actual_total.png")

    # 3. reconciliation effect (R fable, full hierarchy): WAPE ratio vs base
    rc = pd.read_csv(td / "reconciliation_comparison.csv")
    r = rc[(rc["engine"] == "R fable") & (rc["structure"] == "full 154 nodes") & (rc["split"] == "test") & (rc["metric"] == "WAPE_ratio_vs_base")]
    if len(r):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        meths = ["bottom-up", "OLS", "WLS structural", "WLS variance", "MinT shrink"]
        cols = ["#D55E00", "#E69F00", "#009E73", "#56B4E9", "#0072B2"]
        for ax, b in zip(axes, ["R ETS", "R ARIMA"], strict=True):
            for i, (mth, c) in enumerate(zip(meths, cols, strict=True)):
                x = r[(r["base_model"] == b) & (r["method"] == mth)].sort_values("level_id")
                ax.bar(x["level_id"] + (i - 2) * 0.16, x["value"], 0.16, color=c, label=mth)
            ax.axhline(1.0, color="black", lw=0.8)
            ax.set_ylim(0.7, 1.35)
            ax.set_title(f"{b}: WAPE relative to unreconciled base (test)")
            ax.set_xticks(range(1, 10))
            ax.set_xticklabels([LEVEL_SHORT[i] for i in range(1, 10)], rotation=25)
        axes[0].set_ylabel("WAPE ratio (below 1 = reconciliation helps)")
        axes[0].legend(fontsize=7, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(fd / "reconciliation_effect.png")
        plt.close(fig)
        done.append("reconciliation_effect.png")

    # 4. calibration: empirical vs nominal coverage by model (pooled over hierarchy levels, test)
    ic = pd.read_csv(td / "interval_coverage.csv")
    z = ic[(ic["split"] == "test") & ic["level_id"].between(1, 9) & ic["model"].isin(list(PALETTE))]
    z = z.groupby(["model", "nominal"], as_index=False)["coverage"].mean()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0.5, 1], [0.5, 1], color="black", lw=0.8, ls=":")
    for m in sorted(z["model"].unique()):
        zz = z[z["model"] == m].sort_values("nominal")
        ax.plot(zz["nominal"] / 100, zz["coverage"], marker="o", color=PALETTE[m], label=m)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage (mean over 9 levels, test)")
    ax.set_xlim(0.75, 1.0)
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Interval calibration")
    ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(fd / "calibration.png")
    plt.close(fig)
    done.append("calibration.png")

    # 5. per-origin spread of WAPE at store x department (bottom level) and total
    po = pd.read_csv(td / "metrics_by_origin.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, lv in zip(axes, [1, 9], strict=True):
        for m in ["naive", "snaive", "ets", "sarima_airline", "lgbm_q"]:
            q = po[(po["model"] == m) & (po["level_id"] == lv) & (po["layer"].str.startswith("hier"))].sort_values("origin_week_idx")
            ax.plot(q["origin_week_idx"], 100 * q["WAPE"], marker="o", ms=3, color=PALETTE[m], label=m)
        split_line = po[po["split"] == "test"]["origin_week_idx"].min() - 2
        ax.axvline(split_line, color="black", ls=":", lw=0.8)
        ax.set_title(f"WAPE by forecast origin: {LEVEL_SHORT[lv]} (dotted line: start of test window)")
        ax.set_xlabel("Origin week index")
        ax.set_ylabel("WAPE (%)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(fd / "wape_by_origin.png")
    plt.close(fig)
    done.append("wape_by_origin.png")

    # 6. Ljung-Box rejection share by level
    rp = td / "residual_diagnostics_summary.csv"
    if rp.exists():
        rs = pd.read_csv(rp)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for i, (m, c) in enumerate([("ets", "#D55E00"), ("arima", "#8C510A")]):
            x = rs[rs["model"] == m].sort_values("level_id")
            ax.bar(x["level_id"] + (i - 0.5) * 0.35, 100 * x["share_ljung_box_rejected"], 0.35, color=c, label=f"R {m.upper()}")
        ax.axhline(5, color="black", lw=0.8, ls=":")
        ax.set_xticks(range(1, 10))
        ax.set_xticklabels([LEVEL_SHORT[i] for i in range(1, 10)], rotation=25)
        ax.set_ylabel("Nodes with Ljung-Box p < 0.05 (%)")
        ax.set_title("Residual autocorrelation (lag 52) by level, fable fit at the last test origin")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fd / "ljung_box_by_level.png")
        plt.close(fig)
        done.append("ljung_box_by_level.png")

    # 7. design timeline
    od = pd.read_csv(td / "origins_design.csv")
    fig, ax = plt.subplots(figsize=(9, 2.6))
    colors = {"burn-in (training and conformal calibration only)": "#BDC1C6", "validation": "#E69F00", "test (final held-out)": "#0072B2"}
    for _, r_ in od.iterrows():
        ax.barh(0, 4, left=r_["origin_week_idx"], color=colors[r_["split"]], edgecolor="white", height=0.5)
    ax.set_yticks([])
    ax.set_xlabel("Week index (each block = one 4-week forecast window, one origin every 4 weeks)")
    ax.set_title("Rolling-origin design: burn-in (grey), validation (orange), test (blue)")
    fig.tight_layout()
    fig.savefig(fd / "backtest_design.png")
    plt.close(fig)
    done.append("backtest_design.png")
    return done


__all__ = ["make_figures", "np"]
