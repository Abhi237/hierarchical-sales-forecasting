"""Load modelling panels (weeks x nodes) from the DuckDB database with explicit, fixed orderings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STATES = ["CA", "TX", "WI"]
WEEK_COVARIATES = [
    "snap_days_ca", "snap_days_tx", "snap_days_wi", "n_ev_sporting", "n_ev_cultural", "n_ev_national", "n_ev_religious",
    "is_christmas_week", "is_thanksgiving_week", "is_superbowl_week", "is_easter_week",
]


@dataclass
class Panel:
    layer: str                 # 'hier' (levels 1-9) or 'item' (levels 10 and 12)
    node_ids: list[str]
    level_ids: np.ndarray      # (N,)
    labels: list[str]
    Y: np.ndarray              # (T, N) weekly revenue, float64, week index w = row + 1
    U: np.ndarray              # (T, N) weekly units
    weeks: pd.DataFrame        # T rows: w, week_start, week_end, covariates
    state_share: np.ndarray    # (N, 3) static state mix (CA, TX, WI)

    @property
    def T(self) -> int:
        return self.Y.shape[0]

    @property
    def N(self) -> int:
        return self.Y.shape[1]


def load_weeks(con) -> pd.DataFrame:
    cols = ", ".join(WEEK_COVARIATES)
    return con.execute(
        f"SELECT w, wm_yr_wk, CAST(week_start AS DATE) AS week_start, CAST(week_end AS DATE) AS week_end, {cols}, "
        "coalesce(event_names, '') AS event_names FROM dim_week ORDER BY w"
    ).df()


def _pivot(df: pd.DataFrame, node_ids: list[str], value: str, T: int) -> np.ndarray:
    p = df.pivot(index="w", columns="node_id", values=value).reindex(index=range(1, T + 1), columns=node_ids)
    return p.fillna(0.0).to_numpy(dtype=float, copy=True)


def load_panel(con, layer: str) -> Panel:
    weeks = load_weeks(con)
    T = len(weeks)
    if layer == "hier":
        nodes = con.execute("SELECT node_id, level_id, node_label FROM dim_node ORDER BY node_idx").df()
        facts = con.execute(
            "SELECT node_id, w, CAST(revenue AS DOUBLE) AS revenue, CAST(units AS DOUBLE) AS units FROM fact_node_week ORDER BY node_id, w"
        ).df()
    elif layer == "item":
        nodes = con.execute("SELECT node_id, level_id, node_label FROM dim_item_node ORDER BY level_id, node_id").df()
        facts = con.execute(
            "SELECT node_id, w, CAST(revenue AS DOUBLE) AS revenue, CAST(units AS DOUBLE) AS units FROM fact_item_node_week ORDER BY node_id, w"
        ).df()
    else:
        raise ValueError(layer)
    node_ids = nodes["node_id"].tolist()
    ss = con.execute("SELECT node_id, state_id, share FROM node_state_share").df()
    share = (
        ss.pivot(index="node_id", columns="state_id", values="share").reindex(index=node_ids, columns=STATES).fillna(0.0).to_numpy(dtype=float, copy=True)
    )
    return Panel(
        layer=layer,
        node_ids=node_ids,
        level_ids=nodes["level_id"].to_numpy(dtype=int),
        labels=nodes["node_label"].tolist(),
        Y=_pivot(facts, node_ids, "revenue", T),
        U=_pivot(facts, node_ids, "units", T),
        weeks=weeks,
        state_share=share,
    )


def summing_matrix(con, node_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """S (n_nodes x n_bottom) in the order of node_ids and dim_bottom.bottom_idx."""
    bottoms = con.execute("SELECT bottom_id FROM dim_bottom ORDER BY bottom_idx").df()["bottom_id"].tolist()
    br = con.execute("SELECT node_id, bottom_id FROM bridge_node_bottom").df()
    bi = {b: i for i, b in enumerate(bottoms)}
    ni = {n: i for i, n in enumerate(node_ids)}
    S = np.zeros((len(node_ids), len(bottoms)))
    S[br["node_id"].map(ni).to_numpy(), br["bottom_id"].map(bi).to_numpy()] = 1.0
    return S, bottoms
