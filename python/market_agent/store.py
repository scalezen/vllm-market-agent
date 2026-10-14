"""File-based run store: one JSON file per run, grouped by day.

    runs/2026-09-19/GOOGL_sentiment_20260919T101500Z.json

Each file holds the report, usage, model name and the raw tool outputs the
report was based on, so runs can be re-scored or re-analysed later.
"""
import json
import os
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path(os.environ.get("RUNS_DIR", "runs"))

# Keys agent.py puts at the top level of a run; anything else belongs to the report.
_TOP_LEVEL_KEYS = {"ticker", "task", "user_prompt", "prompt_hash", "report", "model", "tool_outputs", "usage", "timestamp"}


def save_run(result: dict, runs_dir: Path = RUNS_DIR) -> Path:
    """Write one successful run; result['timestamp'] is its UTC ISO timestamp."""
    ts = datetime.fromisoformat(result["timestamp"])
    day_dir = runs_dir / ts.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    task = result.get("task", "sentiment")
    path = day_dir / f"{result['ticker']}_{task}_{ts.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def _normalise(run: dict) -> dict:
    """Upgrade a run saved before the report was nested under "report" to the current shape.

    Early runs (before tasks existed) had the report fields splatted into the top
    level, no "task" key, and "justification" instead of "summary".
    """
    if "report" in run and "task" in run:
        return run
    report = {k: v for k, v in run.items() if k not in _TOP_LEVEL_KEYS}
    if "justification" in report:
        report["summary"] = report.pop("justification")
    normalised = {k: v for k, v in run.items() if k in _TOP_LEVEL_KEYS}
    normalised["task"] = run.get("task", "sentiment")
    normalised["report"] = report
    return normalised


def load_runs(runs_dir: Path = RUNS_DIR, task: str | None = None) -> list[dict]:
    """All stored runs, oldest first, upgraded to the current shape. `task` filters by task name."""
    runs = [_normalise(json.loads(p.read_text())) for p in sorted(runs_dir.glob("*/*.json"))]
    if task is not None:
        runs = [r for r in runs if r["task"] == task]
    return sorted(runs, key=lambda r: r["timestamp"])
