from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dl():
    spec = importlib.util.spec_from_file_location("download_data", ROOT / "scripts" / "download_data.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_expected_shapes_are_the_documented_m5_shapes(dl):
    assert dl.EXPECT == {"calendar.csv": 1969, "sales_train_evaluation.csv": 30490, "sell_prices.csv": 6841121}
    assert dl.SALES_DAYS == 1941


def test_verify_rejects_fixture_with_wrong_shape(dl, fixture_dir):
    with pytest.raises(ValueError, match="rows, expected"):
        dl.verify(fixture_dir)


def test_verify_rejects_missing_file(dl, tmp_path):
    with pytest.raises(ValueError, match="missing"):
        dl.verify(tmp_path)


def test_verify_accepts_matching_shape(dl, tmp_copy, monkeypatch):
    monkeypatch.setattr(dl, "EXPECT", {"calendar.csv": 591, "sales_train_evaluation.csv": 60, "sell_prices.csv": 5024})
    monkeypatch.setattr(dl, "SALES_DAYS", 563)
    out = dl.verify(tmp_copy)
    assert out["sales_train_evaluation.csv"]["rows"] == 60 and len(out["calendar.csv"]["sha256"]) == 64


def test_count_rows_handles_quoted_header(dl, tmp_path):
    p = tmp_path / "x.csv"
    p.write_text('"a","b"\n1,2\n3,4\n')
    assert dl.count_rows(p) == (2, 2)


def test_source_urls_are_public_and_credential_free(dl):
    assert dl.NIXTLA_URL.startswith("https://github.com/Nixtla/m5-forecasts/")
    assert "kaggle" not in dl.NIXTLA_URL.lower() and "kaggle" not in dl.HF_BASE.lower()
    assert len(dl.NIXTLA_SHA256) == 64


def test_from_dir_with_wrong_shape_writes_nothing(tmp_path, fixture_dir, monkeypatch, dl):
    monkeypatch.setattr(dl, "RAW", tmp_path / "raw")
    monkeypatch.setattr(dl.sys, "argv", ["download_data.py", "--from-dir", str(fixture_dir)])
    with pytest.raises(ValueError):
        dl.main()
    assert not (tmp_path / "raw" / "manifest.json").exists()


def test_manifest_json_format(tmp_path, dl):
    m = {"source_url": "x", "files": {}}
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m))
    assert json.loads(p.read_text())["source_url"] == "x"
