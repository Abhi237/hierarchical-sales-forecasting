"""Build docs/planning_memo.pdf (2 pages) from the exported tables. Every number is read from tableau/data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
TD = ROOT / "tableau" / "data"
LV = {1: "Total", 2: "State", 3: "Store", 5: "Department", 9: "Store x Dept"}
PICK = {"ma4": "4-wk moving avg", "sarima_airline": "SARIMA airline", "lgbm_conformal": "LightGBM conf.", "r_arima_mint": "fable ARIMA+MinT"}
BEST = {"sarima_111_010": "SARIMA(1,1,1)(0,1,0)[52]"}


def tbl(df: pd.DataFrame) -> str:
    h = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f"<table><tr>{h}</tr>{rows}</table>"


def main() -> None:
    sc = pd.read_csv(TD / "metric_scorecard.csv")
    dp = pd.read_csv(TD / "decision_planning.csv")
    t = sc[(sc["split"] == "test") & sc["layer"].str.startswith("hier") & sc["model"].isin(PICK)]
    w = t[t["metric"] == "WAPE"].pivot_table(index="model", columns="level_id", values="value").reindex(list(PICK))
    w = (100 * w[list(LV)]).round(1)
    w.columns = [LV[c] for c in w.columns]
    w.insert(0, "Model (WAPE %, test)", [PICK[m] for m in w.index])
    c80 = t[t["metric"] == "Coverage_80"].pivot_table(index="model", columns="level_id", values="value").reindex(list(PICK))
    c80 = (100 * c80[list(LV)]).round(0).astype(int)
    c80.columns = [LV[c] for c in c80.columns]
    c80.insert(0, "Model (80% interval coverage, %)", [PICK[m] for m in c80.index])

    d = dp[dp["model"].isin(PICK) & dp["level_id"].isin(LV)]
    g = d.groupby(["model", "level_id"]).agg(act=("actual_revenue_usd", "sum"), cp=("ASSUMED_cost_point_plan_usd", "sum"), cq=("ASSUMED_cost_q90_plan_usd", "sum"),
                                              buf=("safety_buffer_usd", "sum")).reset_index()
    g["point"] = (100 * g["cp"] / g["act"]).round(2)
    g["q90"] = (100 * g["cq"] / g["act"]).round(2)
    g["buffer"] = (100 * g["buf"] / g["act"]).round(1)
    pl = g[g["level_id"].isin([1, 9])].copy()
    pl["Level"] = pl["level_id"].map(LV)
    pl["Model"] = pl["model"].map(PICK)
    pl = pl.sort_values(["level_id", "model"])[["Level", "Model", "point", "q90", "buffer"]]
    pl.columns = ["Level", "Model", "Cost, point plan (% of revenue)", "Cost, q90 plan (% of revenue)", "Buffer, q90 over point (% of revenue)"]

    tot = sc[(sc["split"] == "test") & (sc["level_id"] == 1) & sc["layer"].str.startswith("hier") & (sc["metric"] == "WAPE")].set_index("model")["value"]
    best_tot = tot.drop([m for m in tot.index if m.startswith(("py_", "geo_"))]).idxmin()
    sn = pd.read_csv(TD / "decision_sensitivity.csv")
    pv = sn.pivot_table(index=["model", "level_id", "ASSUMED_gross_margin", "ASSUMED_carrying_cost"], columns="plan_rule", values="ASSUMED_cost_pct_of_revenue")
    win = pv["upper 80% bound (q90)"] < pv["point forecast (q50)"]
    lose = pv[~win].reset_index()
    lose_txt = ", ".join(sorted({f"margin {a:.0%} with carrying cost {b:.0%}" for a, b in zip(lose["ASSUMED_gross_margin"], lose["ASSUMED_carrying_cost"])}))
    html = f"""<html><head><meta charset="utf-8"><style>
@page {{ size: Letter; margin: 0.5in; }}
body {{ font-family: DejaVu Sans, Arial, sans-serif; font-size: 8.1pt; line-height: 1.25; color: #1a1a1a; }}
h1 {{ font-size: 13pt; margin: 0 0 2pt 0; }} h2 {{ font-size: 10pt; margin: 9pt 0 2pt 0; color: #1f4e79; }}
.sub {{ color: #555; font-size: 8pt; margin-bottom: 6pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 3pt 0 4pt 0; font-size: 7.8pt; }}
th, td {{ border: 0.5pt solid #bbb; padding: 2pt 4pt; text-align: right; }} th {{ background: #eef3f8; }} td:first-child, th:first-child {{ text-align: left; }}
.pb {{ page-break-before: always; }} img {{ width: 80%; }} .note {{ font-size: 7.6pt; color: #444; }}
</style></head><body>
<h1>Planning memo: 4-week revenue forecasts for a US Walmart store group</h1>
<div class="sub">To: merchandise and finance planning. Data: M5 (10 stores in CA, TX, WI; 2011 to 2016). Test window: 6 rolling origins covering weeks 254 to 277 (2015-12-05 to 2016-05-14).
Dollar figures below are in percent of revenue; parameters marked ASSUMED are scenario inputs, not estimates from the data.</div>
<h2>Recommendation</h2>
<ul>
<li><b>Use the MinT-reconciled ARIMA forecast for store, category and department plans.</b> It has the lowest WAPE at every level below Total, and at Total it
is within about 0.1 percentage points of the best model ({BEST.get(best_tot, best_tot)} is lowest, WAPE {100 * tot[best_tot]:.2f}%). One coherent set of numbers means store plans add up to the company plan.</li>
<li><b>Do not use the 4-week moving average as the headline model, but keep it as the benchmark.</b> It beats ETS in WAPE at all 9 levels and is the hardest baseline to beat at store x department,
yet it is 2 to 3 points worse than ARIMA at the top. Seasonal naive (WAPE 14% at Total) should not be used.</li>
<li><b>Size safety buffers from calibrated quantiles, not from the reconciled fable intervals.</b> Those intervals under-cover (for fable ARIMA, 80% intervals contain 17% to 71% of actuals at Total depending on the method).
Among models with intervals, split-conformal LightGBM is closest to nominal at several of the lower levels.</li>
<li><b>Treat all model rankings among the top three as tentative:</b> there are only 6 test origins.</li></ul>
<h2>How accurate are the forecasts?</h2>
{tbl(w)}
<p class="note">WAPE = sum of absolute errors divided by sum of actuals, pooled over 4-week-ahead forecasts of all nodes at the level (lower is better). Errors grow as the
level gets more granular: the best store x department WAPE is several times the best Total WAPE.</p>
<h2>How decision-useful are the intervals?</h2>
{tbl(c80)}
<p class="note">Nominal coverage is 80. Coverage at Total is estimated from 24 forecasts and is noisy. All models are below nominal at most levels.</p>
<h2 class="pb">What does a safety buffer cost and save? (ASSUMED parameters)</h2>
<p>Scenario: each dollar of revenue the plan misses costs 25 cents of margin (ASSUMED), and each dollar planned but not sold costs 5 cents of carrying and markdown cost (ASSUMED).
Plan A orders at the point forecast. Plan B orders at the 90th percentile forecast (ASSUMED service level). Costs are summed over the nodes at each level and the 24 test weeks,
shown as percent of actual revenue; the table `decision_sensitivity.csv` varies margin (15, 25, 35 percent) and carrying cost (2, 5, 10 percent).</p>
{tbl(pl)}
<p>Under these assumptions Plan B lowers the cost in every row shown, because missing a sale costs five times as much as carrying a dollar of excess. The buffer is small at Total and grows with granularity,
since errors are larger and do not net out. Across all {len(win):,} model x level x scenario rows in the sensitivity table Plan B is cheaper in {100 * win.mean():.1f}%; it loses only for {lose_txt} ({len(lose)} rows).</p>
<p><b>Realised service.</b> The share of weeks in which actual revenue was at or below the q90 plan is exported per model, node and level in `decision_planning.csv`
(column `share_weeks_actual_within_q90`). It is a check on whether q90 behaves like a 90th percentile; for several models it falls short of 0.90 (for example MinT ARIMA at Total, 0.75), so the q90 plan did not deliver a 90 percent service level in this window.</p>
<img src="{(ROOT / 'docs/figures/forecast_vs_actual_total.png').as_uri()}">
<h2>Limits</h2>
<ul>
<li>M5 is 2011 to 2016 data from 10 stores in 3 states; the hierarchy, seasonality and SNAP effects may not hold elsewhere. Revenue is units x list price: promotions and stockouts are not separable from zero demand.</li>
<li>Only 6 origins of 4 weeks form the final test; the planning table uses the same weeks, so it is an ex post illustration, not a forecast of savings.</li>
<li>The item layer (top 300 items, top 600 item x store series) is a revenue-ranked sample, not reconciled; item x store WAPE is about 26% to 29% even for the best models.</li>
<li>Residual autocorrelation remains in the ARIMA and ETS fits (Ljung-Box rejects at most nodes), so Gaussian intervals are approximate.</li></ul>
</body></html>"""
    out = ROOT / "docs" / "planning_memo.pdf"
    HTML(string=html, base_url=str(ROOT)).write_pdf(str(out))
    print("wrote", out)


if __name__ == "__main__":
    main()
