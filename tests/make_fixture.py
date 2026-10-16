"""Generate the tiny M5-format fixture (committed under tests/fixtures/m5_mini). Deterministic: seed 5.

World: 3 states (CA, TX, WI), 4 stores, 2 categories, 3 departments, 5 items per department (60 item-store series),
563 daily columns (80 complete Saturday-to-Friday weeks + 3 trailing days), annual cycle replaced by a 13-week cycle so a short
history still has seasonality. Includes late-launch items (price missing, units 0), a stockout run, a closed-store day, SNAP and events.
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures" / "m5_mini"
N_DAYS = 563
CAL_ROWS = N_DAYS + 28
STORES = [("CA_1", "CA"), ("CA_2", "CA"), ("TX_1", "TX"), ("WI_1", "WI")]
DEPTS = [("FOODS_1", "FOODS"), ("FOODS_2", "FOODS"), ("HOBBIES_1", "HOBBIES")]
ITEMS_PER_DEPT = 5


def main() -> None:
    rng = np.random.default_rng(5)
    OUT.mkdir(parents=True, exist_ok=True)
    d0 = date(2013, 1, 5)  # a Saturday, like M5's d_1
    days = [d0 + timedelta(days=i) for i in range(CAL_ROWS)]
    wk0 = 11301
    events = {40: ("SuperBowl", "Sporting"), 100: ("Easter", "Religious"), 150: ("Thanksgiving", "National"), 200: ("Christmas", "National"),
              260: ("SuperBowl", "Sporting"), 330: ("Easter", "Religious"), 400: ("Thanksgiving", "National"), 450: ("Christmas", "National")}
    with open(OUT / "calendar.csv", "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        w.writerow(["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d", "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                    "snap_CA", "snap_TX", "snap_WI"])
        for i, dt in enumerate(days):
            ev = events.get(i, ("NA", "NA"))
            snap = [1 if (dt.day <= 10 and (i % 3 != 0)) else 0, 1 if (dt.day in (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)) else 0, 1 if dt.day % 2 == 0 and dt.day <= 12 else 0]
            row = [dt.isoformat(), wk0 + i // 7, dt.strftime("%A"), (i % 7) + 1, dt.month, dt.year, f"d_{i + 1}", ev[0], ev[1], "NA", "NA", *snap]
            w.writerow([x if x != "NA" else "NA" for x in row])
    # raw text: NA must be unquoted for pandas/duckdb nullstr; rewrite quoting of NA tokens
    txt = (OUT / "calendar.csv").read_text().replace('"NA"', "NA")
    (OUT / "calendar.csv").write_text(txt)

    series = []
    for st, state in STORES:
        for dp, cat in DEPTS:
            for k in range(1, ITEMS_PER_DEPT + 1):
                series.append((f"{dp}_{k:03d}", dp, cat, st, state))
    series.sort(key=lambda s: (s[0], s[3]))
    t = np.arange(N_DAYS)
    dow = np.array([1.25, 1.2, 0.85, 0.8, 0.85, 0.95, 1.1])[t % 7]
    prices = []
    rows = []
    for idx, (item, dp, cat, st, state) in enumerate(series):
        base = rng.uniform(0.5, 4.0) * (1 + 0.5 * (cat == "HOBBIES"))
        level = rng.uniform(1.0, 8.0) * (2.0 if st == "CA_1" else 1.0) * (1.5 if dp == "FOODS_1" else 1.0)
        season = 1 + 0.25 * np.sin(2 * np.pi * (t / 7.0) / 13.0 + rng.uniform(0, 6)) + 0.002 * t
        lam = level * season * dow
        units = rng.poisson(lam).astype(int)
        launch = 0 if idx % 9 else int(rng.integers(35, 120))      # some items launch late (price missing before)
        units[:launch] = 0
        if idx % 11 == 3:                                            # a stockout run of 6 weeks
            units[300:342] = 0
        for ev_day in (200, 450):                                    # store closed on the Christmas-like day
            units[ev_day] = 0
        rows.append((item, dp, cat, st, state, units, launch))
        week_price = {}
        p = round(base, 2)
        n_w = CAL_ROWS // 7 + 1
        for wk in range(n_w):
            if (wk + 1) * 7 > launch:
                if rng.random() < 0.08:
                    p = round(max(0.3, p * rng.choice([0.9, 1.0, 1.1])), 2)
                week_price[wk] = p
        for wk, pr in week_price.items():
            prices.append((st, item, wk0 + wk, pr))
    with open(OUT / "sales_train_evaluation.csv", "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        w.writerow(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + [f"d_{i}" for i in range(1, N_DAYS + 1)])
        for item, dp, cat, st, state, units, _ in rows:
            w.writerow([f"{item}_{st}_evaluation", item, dp, cat, st, state, *[int(u) for u in units]])
    prices.sort(key=lambda r: (r[0], r[1], r[2]))
    with open(OUT / "sell_prices.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["store_id", "item_id", "wm_yr_wk", "sell_price"])
        for st, item, wk, pr in prices:
            w.writerow([st, item, wk, f"{pr:.2f}"])
    print(len(rows), "series;", len(prices), "price rows;", N_DAYS, "days;", CAL_ROWS, "calendar rows")


if __name__ == "__main__":
    main()
