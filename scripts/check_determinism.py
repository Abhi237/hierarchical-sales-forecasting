"""Hash every pipeline output so two independent runs can be compared byte for byte.

  python scripts/check_determinism.py snapshot artifacts/output_hashes.json
  python scripts/check_determinism.py compare run_a.json run_b.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ["tableau/data/*.csv", "artifacts/*.csv", "data/processed/r_*.csv", "data/processed/bottom_weekly.csv", "data/processed/origins.json"]
PARQUETS = ["data/processed/forecasts_hier.parquet", "data/processed/forecasts_item.parquet", "data/processed/forecasts_hier_all.parquet"]


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def snapshot() -> dict:
    out = {}
    for pat in PATTERNS:
        for p in sorted(ROOT.glob(pat)):
            out[str(p.relative_to(ROOT))] = sha_bytes(p.read_bytes())
    for rel in PARQUETS:
        p = ROOT / rel
        if p.exists():
            out[rel + " (as csv)"] = sha_bytes(pd.read_parquet(p).to_csv(index=False, float_format="%.10g").encode())
    return out


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "snapshot":
        snap = snapshot()
        Path(sys.argv[2]).write_text(json.dumps(snap, indent=2, sort_keys=True))
        print(f"{len(snap)} files hashed -> {sys.argv[2]}")
        return 0
    if len(sys.argv) >= 4 and sys.argv[1] == "compare":
        a, b = (json.loads(Path(x).read_text()) for x in sys.argv[2:4])
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        print(f"{len(a)} vs {len(b)} files; {len(diff)} differ")
        for k in diff:
            print("  DIFF", k)
        return 1 if diff else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
