"""File-based run store: one JSON file per ticker run, grouped by day.

    runs/2026-09-19/GOOGL_20260919T101500Z.json

Each file holds the report, usage, model name and the raw tool outputs the
report was based on, so runs can be re-scored or re-analysed later.
"""
import json
import os
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path(os.environ.get("RUNS_DIR", "runs"))


def save_run(result: dict, runs_dir: Path = RUNS_DIR) -> Path:
    """Write one successful run; result['timestamp'] is its UTC ISO timestamp."""
    ts = datetime.fromisoformat(result["timestamp"])
    day_dir = runs_dir / ts.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{result['ticker']}_{ts.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def load_runs(runs_dir: Path = RUNS_DIR) -> list[dict]:
    """All stored runs, oldest first."""
    runs = [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*/*.json"))]
    return sorted(runs, key=lambda r: r["timestamp"])
