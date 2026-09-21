"""Stock sentiment agent backed by a local vLLM-MLX server.

Start the server first (see serve.sh), then run:
    python sentiment_agent.py                      # default ticker
    python sentiment_agent.py AAPL MSFT NVDA       # batch, run concurrently
    python sentiment_agent.py --watchlist watchlist.txt --out results.jsonl

Per ticker the model calls three tools (news, technical indicators, analyst
recommendations), then writes a JSON report constrained to SentimentReport.

Every successful run is saved under runs/<date>/ (see store.py); score them
against later price moves with backtest.py.

Layout: config.py (settings, clients), usage.py (token accounting),
tools.py (tools the model calls), agent.py (agent loop), store.py (run store),
backtest.py (scoring); this file is the CLI.
"""
import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger

from agent import check_server, run_watchlist
from config import DEFAULT_CONCURRENCY
from tasks import TASKS
from store import RUNS_DIR, save_run


def load_tickers(args: argparse.Namespace) -> list[str]:
    tickers = list(args.tickers)
    if args.watchlist:
        with open(args.watchlist) as f:
            # one ticker per line (or comma/space separated); '#' starts a comment
            for line in f:
                tickers += line.split("#")[0].replace(",", " ").split()
    tickers = list(dict.fromkeys(t.upper() for t in tickers))  # dedupe, keep order
    return tickers or ["GOOGL"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="*", help="ticker symbols (default: GOOGL)")
    parser.add_argument("--watchlist", help="file with one ticker per line")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="parallel tickers")
    parser.add_argument("--out", help="append results to this JSONL file")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="where each run is stored (for backtest.py)")
    parser.add_argument("--no-save", action="store_true", help="do not write runs to the run store")
    args = parser.parse_args()

    tickers = load_tickers(args)
    check_server()
    results = asyncio.run(run_watchlist(tickers, TASKS["sentiment"], args.concurrency))

    if not args.no_save:
        for r in results:
            if "usage" in r:  # skip failed tickers
                save_run(r, args.runs_dir)
    # tool_outputs are kept in the run store, not printed
    results = [{k: v for k, v in r.items() if k != "tool_outputs"} for r in results]

    print(json.dumps(results[0] if len(results) == 1 else results, indent=2))
    usages = [r["usage"] for r in results if "usage" in r]
    if usages:
        logger.info(
            "Batch total: {} tokens ({} thinking) across {} tickers",
            sum(u["total_tokens"] for u in usages),
            sum(u["thinking_tokens"] for u in usages),
            len(usages),
        )
    if args.out:
        with open(args.out, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in results)


if __name__ == "__main__":
    main()
