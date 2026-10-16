from __future__ import annotations

import re
from pathlib import Path

import duckdb

from .config import rpath

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


def connect(cfg: dict, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    db = rpath(cfg, cfg["data"]["db_path"])
    db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db), read_only=read_only)
    con.execute(f"SET threads TO {int(cfg['duckdb']['threads'])}")
    con.execute(f"SET memory_limit = '{cfg['duckdb']['memory_limit']}'")
    con.execute("SET preserve_insertion_order = true")
    return con


def render(sql: str, params: dict) -> str:
    def sub(m: re.Match) -> str:
        k = m.group(1)
        if k not in params:
            raise KeyError(f"missing SQL parameter {k}")
        return str(params[k])
    return re.sub(r"\{\{(\w+)\}\}", sub, sql)


def sql_params(cfg: dict, selection_end_week: int | None = None) -> dict:
    d = cfg["data"]
    raw = rpath(cfg, d["raw_dir"])
    return {
        "calendar_csv": (raw / d["calendar_csv"]).as_posix(),
        "sales_csv": (raw / d["sales_csv"]).as_posix(),
        "prices_csv": (raw / d["prices_csv"]).as_posix(),
        "n_days": d["n_days"],
        "recent_daily_weeks": cfg["scope"]["recent_daily_weeks"],
        "top_items": cfg["scope"]["item_layer"]["top_items"],
        "top_item_stores": cfg["scope"]["item_layer"]["top_item_stores"],
        "selection_end_week": selection_end_week if selection_end_week is not None else 0,
    }


def run_sql_files(con, cfg: dict, files: list[str], params: dict) -> None:
    for f in files:
        txt = render((SQL_DIR / f).read_text(), params)
        con.execute(txt)


def build_database(cfg: dict) -> None:
    """Run all numbered SQL scripts. The item-layer selection window ends at the week before the first backtest origin's
    last training week, i.e. the first origin itself (training weeks 1..first_origin)."""
    from .config import origins
    con = connect(cfg)
    try:
        p = sql_params(cfg)
        run_sql_files(con, cfg, ["01_stage.sql", "02_weeks.sql"], p)
        n_weeks = con.execute("SELECT max(w) FROM dim_week").fetchone()[0]
        p["selection_end_week"] = origins(cfg, n_weeks)["all"][0]
        run_sql_files(con, cfg, ["03_item_store_week.sql", "04_hierarchy.sql", "05_item_layer.sql", "06_quality_and_daily.sql", "07_state_share.sql"], p)
        con.execute("CHECKPOINT")
    finally:
        con.close()
