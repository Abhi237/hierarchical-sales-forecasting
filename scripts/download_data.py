"""Fetch the M5 Forecasting (Walmart) files WITHOUT any credentials, verify shape, write data/raw/ + manifest.json.

Sources tried in order (first one that downloads AND parses to the documented shape wins):
  1. Nixtla public GitHub copy: https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip (50,219,189 bytes)
  2. Hugging Face mirror denephew/M5_Forecasting (original Kaggle file layout, with an extra `id` column)
Kaggle itself needs a login and is never contacted. A local copy can be used with --from-dir.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
NEEDED = ["calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv"]
EXPECT = {"calendar.csv": 1969, "sales_train_evaluation.csv": 30490, "sell_prices.csv": 6841121}
SALES_DAYS = 1941
NIXTLA_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
NIXTLA_SHA256 = "cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774"
HF_BASE = "https://huggingface.co/datasets/denephew/M5_Forecasting/resolve/main/"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path: Path) -> tuple[int, int]:
    """data rows (excluding header) and number of header columns."""
    with open(path, newline="") as f:
        header = next(csv.reader(f))
        n = sum(1 for _ in f)
    return n, len(header)


def verify(directory: Path) -> dict:
    out = {}
    for name in NEEDED:
        p = directory / name
        if not p.exists():
            raise ValueError(f"missing {name}")
        rows, cols = count_rows(p)
        if rows != EXPECT[name]:
            raise ValueError(f"{name}: {rows} rows, expected {EXPECT[name]}")
        if name == "sales_train_evaluation.csv":
            header = next(csv.reader(open(p, newline="")))
            day_cols = [c for c in header if c.startswith("d_")]
            if len(day_cols) != SALES_DAYS or day_cols[-1] != f"d_{SALES_DAYS}":
                raise ValueError(f"{name}: {len(day_cols)} day columns")
        out[name] = {"rows": rows, "columns": cols, "bytes": p.stat().st_size, "sha256": sha256(p)}
    return out


def fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "hier-forecast-download/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def from_nixtla(tmp: Path) -> tuple[str, dict]:
    zpath = tmp / "m5.zip"
    fetch(NIXTLA_URL, zpath)
    digest = sha256(zpath)
    with zipfile.ZipFile(zpath) as z:
        for name in NEEDED:
            with z.open(name) as src, open(tmp / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return NIXTLA_URL, {"archive_sha256": digest, "archive_bytes": zpath.stat().st_size,
                        "archive_sha256_matches_pinned": digest == NIXTLA_SHA256}


def from_hf(tmp: Path) -> tuple[str, dict]:
    for name in NEEDED:
        fetch(HF_BASE + name, tmp / name)
    return HF_BASE, {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", help="directory that already holds the three CSVs")
    ap.add_argument("--source", choices=["nixtla", "hf"], help="force one source")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = RAW / "_tmp"
    tmp.mkdir(exist_ok=True)
    attempts, used, extra, info = [], None, {}, None
    if a.from_dir:
        for n in NEEDED:
            shutil.copy(Path(a.from_dir) / n, tmp / n)
        info = verify(tmp)
        used = f"local:{a.from_dir}"
    else:
        for label, fn in (("nixtla", from_nixtla), ("hf", from_hf)):
            if a.source and a.source != label:
                continue
            try:
                used, extra = fn(tmp)
                info = verify(tmp)
                attempts.append({"source": label, "result": "used"})
                break
            except Exception as e:  # noqa: BLE001
                attempts.append({"source": label, "result": f"failed: {e}"})
                used = None
    if used is None:
        print(json.dumps(attempts, indent=2))
        shutil.rmtree(tmp, ignore_errors=True)
        print("No source produced the documented shape; nothing written.", file=sys.stderr)
        return 1
    for n in NEEDED:
        shutil.move(str(tmp / n), RAW / n)
    shutil.rmtree(tmp, ignore_errors=True)
    manifest = {"source_url": used, "attempts": attempts, **extra, "files": info,
                "note": "Raw files are git-ignored. Only derived aggregates are published."}
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
