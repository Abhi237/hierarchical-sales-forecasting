from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None, overrides: dict | None = None, root: Path | None = None) -> dict:
    """Load config.yaml, apply nested overrides, and resolve paths against `root` (default: repo root)."""
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    cfg = yaml.safe_load(path.read_text())
    if overrides:
        cfg = _merge(cfg, overrides)
    cfg["_root"] = str(root or ROOT)
    return cfg


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def rpath(cfg: dict, *parts: str) -> Path:
    return Path(cfg["_root"]).joinpath(*parts)


def origins(cfg: dict, n_weeks: int) -> dict:
    """Rolling-origin design. An origin o means: train on weeks 1..o, forecast weeks o+1..o+h.
    The last origin is n_weeks - h. Origins step back by `step`. Splits are fixed by counts, test = latest."""
    b = cfg["backtest"]
    h, step = b["horizon"], b["step"]
    n_tot = b["n_origins_burnin"] + b["n_origins_validation"] + b["n_origins_test"]
    last = n_weeks - h
    allo = [last - step * k for k in range(n_tot)][::-1]
    nb, nv = b["n_origins_burnin"], b["n_origins_validation"]
    return {
        "all": allo,
        "burnin": allo[:nb],
        "validation": allo[nb:nb + nv],
        "test": allo[nb + nv:],
        "horizon": h,
    }


def split_of(origin: int, org: dict) -> str:
    for s in ("burnin", "validation", "test"):
        if origin in org[s]:
            return s
    raise KeyError(origin)
