"""Repository hygiene: no raw data committed, no emoji or em dash, required documents present, R script sanity."""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".sql", ".yaml", ".yml", ".R", ".qmd", ".toml", ".txt", ".csv", ".json"}
BAD = re.compile("[\u2014\u2013\U0001F300-\U0001FAFF\u2600-\u27BF\u2190-\u21FF]")


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
        files = [ROOT / f for f in out if f]
        if files:
            return files
    except Exception:  # noqa: BLE001
        pass
    skip = {".venv", ".git", "data", ".pytest_cache", ".ruff_cache", "__pycache__", "egg-info"}
    return [p for p in ROOT.rglob("*") if p.is_file() and not (set(p.parts) & skip) and "egg-info" not in str(p)]


def test_gitignore_blocks_raw_data():
    gi = (ROOT / ".gitignore").read_text()
    for pat in ("data/raw/", "data/processed/", ".venv/"):
        assert pat in gi


def test_no_raw_m5_file_is_tracked():
    for p in tracked_files():
        if p.exists():
            assert p.name not in {"sales_train_validation.csv", "sell_prices.csv"} or "fixtures" in str(p), p
            assert p.stat().st_size < 30e6 or p.suffix in {".pdf"}, p


def test_no_emoji_or_em_dash_in_text_files():
    for p in tracked_files():
        if p.suffix in TEXT_SUFFIXES and p.exists() and "renv.lock" not in p.name:
            assert not BAD.search(p.read_text(encoding="utf-8", errors="ignore")), p


def test_required_files_present():
    for f in ["README.md", "Makefile", "pyproject.toml", ".github/workflows/ci.yml", "data/README.md", "config/config.yaml",
              "R/forecast_reconciliation.R", "reports/forecast_report.qmd", "docs/data_dictionary.md", "tableau/DASHBOARD_SPEC.md"]:
        assert (ROOT / f).exists(), f


def test_sql_files_are_numbered_and_ordered():
    names = sorted(p.name for p in (ROOT / "sql").glob("*.sql"))
    assert names == [f"{i:02d}_{n}.sql" for i, n in enumerate(["stage", "weeks", "item_store_week", "hierarchy", "item_layer", "quality_and_daily", "state_share"], 1)]


def test_r_script_declares_required_methods():
    r = (ROOT / "R" / "forecast_reconciliation.R").read_text()
    for token in ("min_trace", "bottom_up", "top_down", "mint_shrink", "wls_struct", "wls_var", "ols", "Box.test", "aggregate_key", "ETS(", "ARIMA("):
        assert token in r, token


def test_r_script_parses():
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not installed")
    res = subprocess.run(["Rscript", "-e", f"invisible(parse('{ROOT / 'R' / 'forecast_reconciliation.R'}'))"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_readme_mentions_tableau_placeholder_and_licence():
    txt = (ROOT / "README.md").read_text()
    assert "TABLEAU_PUBLIC_URL_TODO" in txt
    assert "licence" in txt.lower() or "license" in txt.lower()


def test_config_values_are_sane():
    import yaml
    c = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    assert c["duckdb"]["threads"] == 1
    assert c["backtest"]["horizon"] == 4 and c["backtest"]["season"] == 52
    assert c["decision"]["plan_quantile"] == 0.9
