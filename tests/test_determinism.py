"""Two independent runs of the whole pipeline on the fixture must produce byte-identical Tableau CSVs and forecasts."""
from __future__ import annotations

import hashlib
import pathlib

from conftest import make_cfg


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def run_all(root):
    from hier_forecast.pipeline import run_backtest, run_evaluate, run_exports, run_sql
    cfg = make_cfg(root)
    run_sql(cfg)
    run_backtest(cfg)
    run_evaluate(cfg)
    run_exports(cfg)
    return cfg


def test_two_runs_are_byte_identical(tmp_path):
    a = run_all(tmp_path / "a")
    b = run_all(tmp_path / "b")
    fa = sorted(pathlib.Path(a["_root"], "tableau", "data").glob("*.csv"))
    fb = sorted(pathlib.Path(b["_root"], "tableau", "data").glob("*.csv"))
    assert [p.name for p in fa] == [p.name for p in fb] and len(fa) >= 15
    for x, y in zip(fa, fb, strict=True):
        assert sha(x) == sha(y), x.name
    for name in ("artifacts/scorecard_full.csv", "artifacts/reconciliation_comparison_full.csv", "artifacts/cleaning_log.csv",
                 "data/processed/bottom_weekly.csv", "data/processed/origins.json"):
        assert sha(pathlib.Path(a["_root"]) / name) == sha(pathlib.Path(b["_root"]) / name), name
