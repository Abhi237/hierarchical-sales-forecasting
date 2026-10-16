from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hier_forecast.config import load_config

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "m5_mini"
REPO = Path(__file__).resolve().parents[1]

FIXTURE_OVERRIDES = {
    "data": {"n_days": 563, "raw_dir": str(FIXTURE), "expected": {"sales_rows": 60, "calendar_rows": 591, "price_rows": 5024}},
    "backtest": {"season": 13, "n_origins_burnin": 2, "n_origins_validation": 2, "n_origins_test": 2, "bootstrap_reps": 60},
    "scope": {"recent_daily_weeks": 4, "item_layer": {"top_items": 5, "top_item_stores": 10}},
    "lgbm": {"n_estimators": 25, "num_leaves": 7, "min_child_samples": 5, "n_jobs": 1, "max_train_weeks": 60},
    "stats_n_jobs": 1,
}


def make_cfg(root: Path) -> dict:
    cfg = load_config(overrides=FIXTURE_OVERRIDES, root=root)
    cfg["data"]["db_path"] = "data/processed/m5.duckdb"
    return cfg


@pytest.fixture(scope="session")
def fixture_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    return make_cfg(root)


@pytest.fixture(scope="session")
def built(fixture_cfg):
    """Run SQL, backtest, evaluate and exports once on the fixture; returns cfg."""
    from hier_forecast.pipeline import run_backtest, run_evaluate, run_exports, run_sql
    cfg = fixture_cfg
    run_sql(cfg)
    run_backtest(cfg)
    run_evaluate(cfg)
    run_exports(cfg)
    return cfg


@pytest.fixture(scope="session")
def con(built):
    from hier_forecast.db import connect
    c = connect(built, read_only=True)
    yield c
    c.close()


@pytest.fixture(scope="session")
def fixture_dir():
    return FIXTURE


@pytest.fixture()
def tmp_copy(tmp_path):
    d = tmp_path / "raw"
    shutil.copytree(FIXTURE, d)
    return d
