"""Score stored sentiment runs against what the stock did afterwards.

    python backtest.py                         # 1, 5 and 20 trading-day horizons
    python backtest.py --horizons 5 10 --csv scored.csv --runs-dir runs

For each stored run the entry price is the last close the agent could have
seen when it ran (the previous session's close, or the same day's if it ran
after the close). The forward return is measured N trading days after that.
Returns are also taken relative to a benchmark (SPY), since a rising market
makes "Bullish" look right by default. A directional call counts as a hit when
Bullish beat the benchmark or Bearish trailed it; Neutral/Unavailable calls are
reported (n, mean return) but have no hit rate.

Only runs old enough to have N trading days of history are scored, so this
needs runs collected over time (e.g. a daily watchlist run).
"""
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from store import RUNS_DIR, load_runs

SIGNALS = ["sentiment", "news_signal", "technical_signal", "analyst_signal"]
BENCHMARK = "SPY"
NY = ZoneInfo("America/New_York")
MARKET_CLOSE_HOUR = 17  # runs at/after 17:00 ET can see that day's close


def price_cutoff(timestamp: str) -> date:
    """Latest date whose close was already final when the run happened."""
    ny = datetime.fromisoformat(timestamp).astimezone(NY)
    return ny.date() if ny.hour >= MARKET_CLOSE_HOUR else ny.date() - timedelta(days=1)


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


def build_table(runs: list[dict], horizons: list[int]) -> pd.DataFrame:
    """One row per (run, horizon) that has matured."""
    cutoffs = {id(r): price_cutoff(r["timestamp"]) for r in runs}
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
                    **{s: run.get(s) for s in SIGNALS},
                }
            )
    return pd.DataFrame(rows)


def score(table: pd.DataFrame) -> pd.DataFrame:
    """Per signal, horizon and label: count, mean return, mean excess return, hit rate."""
    long = table.melt(
        id_vars=["ticker", "run_time", "horizon", "ret", "excess"],
        value_vars=SIGNALS,
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
    out["signal"] = pd.Categorical(out.index.get_level_values("signal"), SIGNALS)
    return out.drop(columns="signal").round(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20], help="trading days ahead")
    parser.add_argument("--csv", help="also write the per-run table to this CSV")
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"No runs found in {args.runs_dir}/. Run sentiment_agent.py first.")
    table = build_table(runs, args.horizons)
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
    print(score(table).to_string())


if __name__ == "__main__":
    main()
