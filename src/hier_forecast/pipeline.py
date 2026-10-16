from __future__ import annotations

import json
import os
import shutil
import subprocess

from .backtest import run_backtest
from .config import origins, rpath
from .db import build_database, connect
from .evaluate import run_evaluate
from .exports import run_exports


def run_sql(cfg: dict) -> dict:
    """Build the DuckDB database, then write the R inputs and the cleaning log."""
    build_database(cfg)
    proc = rpath(cfg, cfg["data"]["processed_dir"])
    art = rpath(cfg, "artifacts")
    art.mkdir(exist_ok=True)
    con = connect(cfg, read_only=True)
    try:
        n_weeks = con.execute("SELECT max(w) FROM dim_week").fetchone()[0]
        con.execute(f"""COPY (SELECT b.state_id AS state, b.store_id AS store, b.cat_id AS cat, b.dept_id AS dept, f.w,
            CAST(w.week_start AS VARCHAR) AS week_start, CAST(f.revenue AS DOUBLE) AS revenue, f.units
            FROM fact_bottom_week f JOIN dim_bottom b USING (bottom_id) JOIN dim_week w USING (w)
            ORDER BY b.store_id, b.dept_id, f.w) TO '{(proc / 'bottom_weekly.csv').as_posix()}' (HEADER, DELIMITER ',')""")
        con.execute(f"COPY (SELECT * FROM cleaning_log ORDER BY step) TO '{(art / 'cleaning_log.csv').as_posix()}' (HEADER, DELIMITER ',')")
        org = origins(cfg, n_weeks)
        design = {**org, "season": cfg["backtest"]["season"], "n_weeks": n_weeks}
        (proc / "origins.json").write_text(json.dumps(design, indent=2, sort_keys=True))
        return {"n_weeks": n_weeks, "origins": org}
    finally:
        con.close()


def run_r(cfg: dict) -> None:
    exe = shutil.which("Rscript")
    if exe is None:
        raise RuntimeError("Rscript not found; install R and run `Rscript -e 'renv::restore()'`")
    proc = rpath(cfg, cfg["data"]["processed_dir"])
    subprocess.run([exe, str(rpath(cfg, "R", "forecast_reconciliation.R")), str(proc)], check=True, cwd=rpath(cfg, "."),
                   env={**os.environ, "HIER_R_WORKERS": os.environ.get("HIER_R_WORKERS", "4")})


__all__ = ["run_sql", "run_r", "run_backtest", "run_evaluate", "run_exports"]
