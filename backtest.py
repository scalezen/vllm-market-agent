"""Score stored runs of a directional task against what the stock did afterwards.

    python backtest.py                              # sentiment task, 1/5/20 trading-day horizons
    python backtest.py --task sentiment --horizons 5 10 --csv scored.csv --runs-dir runs

Only tasks whose report has Bullish/Bearish-style fields can be scored (see
reports.signal_fields); a task with no such field exits with an explanation
instead of a table.

For each stored run the entry price is the last final close the agent saw: the
`as_of` date recorded by the technicals tool (the previous close if that tool saw
a live intraday price). Runs without it fall back to the run time: the previous
session's close, or the same day's if it ran after 17:00 ET. The forward return
is measured N trading days after that.
Returns are also taken relative to a benchmark (SPY), since a rising market
makes "Bullish" look right by default. A directional call counts as a hit when
Bullish beat the benchmark or Bearish trailed it; Neutral/Unavailable calls are
reported (n, mean return) but have no hit rate.

Only runs old enough to have N trading days of history are scored, so this
needs runs collected over time (e.g. a daily watchlist run).
"""
import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from reports import signal_fields
from store import RUNS_DIR, load_runs
from tasks import TASKS

BENCHMARK = "SPY"
NY = ZoneInfo("America/New_York")
MARKET_CLOSE_HOUR = 17  # runs at/after 17:00 ET can see that day's close


def price_cutoff(timestamp: str) -> date:
    """Latest date whose close was already final when the run happened."""
    ny = datetime.fromisoformat(timestamp).astimezone(NY)
    return ny.date() if ny.hour >= MARKET_CLOSE_HOUR else ny.date() - timedelta(days=1)


def entry_cutoff(run: dict) -> date:
    """Latest date whose close counts as the entry price for this run.

    Prefers the `as_of` the technicals tool recorded: that date's final close, or,
    if the tool saw a live intraday price, the close before it. Runs without it
    fall back to the run-time rule in price_cutoff.
    """
    try:
        tech = json.loads(run["tool_outputs"]["get_technical_indicators"])
        as_of = date.fromisoformat(tech["as_of"])
    except (KeyError, TypeError, ValueError):  # no tool output, an error string, or an old run
        return price_cutoff(run["timestamp"])
    return as_of - timedelta(days=1) if tech.get("intraday") else as_of


def fetch_closes(ticker: str, start: date) -> pd.Series:
    close = yf.Ticker(ticker).history(start=start.isoformat())["Close"].dropna()
    close.index = close.index.tz_localize(None).normalize()
    return close


def forward_return(close: pd.Series, cutoff: date, horizon: int) -> float | None:
    """Return from the last close on/before `cutoff` to `horizon` trading days later."""
    pos = close.index.searchsorted(pd.Timestamp(cutoff), side="right") - 1
    if pos < 0 or pos + horizon >= len(close):
        return None  # no price before the run, or the horizon has not elapsed yet
    return float(close.iloc[pos + horizon] / close.iloc[pos] - 1)


def build_table(runs: list[dict], horizons: list[int], signals: list[str]) -> pd.DataFrame:
    """One row per (run, horizon) that has matured."""
    cutoffs = {id(r): entry_cutoff(r) for r in runs}
    start = min(cutoffs.values()) - timedelta(days=10)
    closes = {t: fetch_closes(t, start) for t in {r["ticker"] for r in runs} | {BENCHMARK}}

    rows = []
    for run in runs:
        cutoff = cutoffs[id(run)]
        for h in horizons:
            ret = forward_return(closes[run["ticker"]], cutoff, h)
            bench = forward_return(closes[BENCHMARK], cutoff, h)
            if ret is None or bench is None:
                continue
            rows.append(
                {
                    "ticker": run["ticker"],
                    "run_time": run["timestamp"],
                    "horizon": h,
                    "ret": ret,
                    "excess": ret - bench,
                    **{s: run["report"].get(s) for s in signals},
                }
            )
    return pd.DataFrame(rows)


def score(table: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    """Per signal, horizon and label: count, mean return, mean excess return, hit rate."""
    long = table.melt(
        id_vars=["ticker", "run_time", "horizon", "ret", "excess"],
        value_vars=signals,
        var_name="signal",
        value_name="label",
    )
    long = long[long["label"].notna() & (long["label"] != "Unavailable")]
    long["hit"] = np.select(
        [long["label"] == "Bullish", long["label"] == "Bearish"],
        [(long["excess"] > 0).astype(float), (long["excess"] < 0).astype(float)],
        default=np.nan,
    )
    out = long.groupby(["signal", "horizon", "label"]).agg(
        n=("ret", "size"),
        mean_ret_pct=("ret", lambda s: s.mean() * 100),
        mean_excess_pct=("excess", lambda s: s.mean() * 100),
        hit_rate=("hit", "mean"),
    )
    out["signal"] = pd.Categorical(out.index.get_level_values("signal"), signals)
    return out.drop(columns="signal").round(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="sentiment", help="which task's stored runs to score")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20], help="trading days ahead")
    parser.add_argument("--csv", help="also write the per-run table to this CSV")
    args = parser.parse_args()

    task = TASKS.get(args.task)
    if task is None:
        raise SystemExit(f"Unknown task {args.task!r}. Available: {list(TASKS)}")
    signals = signal_fields(task.report_model)
    if not signals:
        raise SystemExit(
            f"Task {args.task!r} has no Bullish/Bearish-style field, so there is nothing to backtest."
        )

    runs = load_runs(args.runs_dir, task=args.task)
    if not runs:
        raise SystemExit(f"No stored runs for task {args.task!r} in {args.runs_dir}/. Run sentiment_agent.py first.")
    table = build_table(runs, args.horizons, signals)
    if table.empty:
        raise SystemExit(
            f"{len(runs)} runs found, but none is old enough for horizons {args.horizons}. "
            "Try a shorter --horizons or wait for more trading days."
        )
    if args.csv:
        table.to_csv(args.csv, index=False)

    per_horizon = table.groupby("horizon")["run_time"].size().to_dict()
    logger.info("{} runs stored; scored observations per horizon: {}", len(runs), per_horizon)
    if min(per_horizon.values()) < 30:
        logger.warning("Fewer than 30 observations at some horizons; treat these numbers as anecdotes.")
    print(score(table, signals).to_string())


if __name__ == "__main__":
    main()
