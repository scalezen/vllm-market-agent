"""Generic agent for free-form questions about a ticker, backed by a local vLLM-MLX server.

Uses the same three tools as the sentiment agent (news, technical indicators,
analyst recommendations), but the system and user prompts are yours to set, and
the report is a GenericReport (summary + confidence) rather than a SentimentReport.

Start the server first (see serve.sh), then run:
    python generic_agent.py AAPL --user "What regulatory risks does {ticker} face?"
    python generic_agent.py AAPL MSFT --system "Answer like a skeptical short-seller." \\
        --user "What would make you avoid {ticker}?"
    python generic_agent.py --watchlist watchlist.txt --user "Summarise {ticker} in one line." --out results.jsonl

--system is appended to the generic task's base system prompt (which tells the
model to call the tools); --user replaces the task's default user prompt and
may reference "{ticker}", filled in per ticker just like the sentiment agent.
Omit either flag to fall back to the generic task's own defaults (tasks/generic.py).

Every successful run is saved under runs/<date>/ (see store.py). Not
backtestable (see backtest.py): GenericReport has no Bullish/Bearish field.

Layout: config.py (settings, clients), usage.py (token accounting),
tools/ (tools the model calls), agent.py (agent loop), store.py (run store);
this file is the CLI.
"""
import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from loguru import logger

from agent import check_server, run_watchlist
from config import DEFAULT_CONCURRENCY
from store import RUNS_DIR, save_run
from tasks import Task, TASKS


def load_tickers(args: argparse.Namespace) -> list[str]:
    tickers = list(args.tickers)
    if args.watchlist:
        with open(args.watchlist) as f:
            # one ticker per line (or comma/space separated); '#' starts a comment
            for line in f:
                tickers += line.split("#")[0].replace(",", " ").split()
    tickers = list(dict.fromkeys(t.upper() for t in tickers))  # dedupe, keep order
    return tickers or ["GOOGL"]


def build_task(args: argparse.Namespace) -> Task:
    """The generic task, with --system appended and --user substituted, if given."""
    task = TASKS["generic"]
    if args.system:
        task = replace(task, system_prompt=task.system_prompt + " " + args.system)
    if args.user:
        task = replace(task, user_prompt=args.user)
    return task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="*", help="ticker symbols (default: GOOGL)")
    parser.add_argument("--watchlist", help="file with one ticker per line")
    parser.add_argument("--system", help="appended to the generic task's system prompt")
    parser.add_argument("--user", help='the question to ask; may contain "{ticker}" (default: the task\'s own prompt)')
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="parallel tickers")
    parser.add_argument("--out", help="append results to this JSONL file")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="where each run is stored (for backtest.py)")
    parser.add_argument("--no-save", action="store_true", help="do not write runs to the run store")
    args = parser.parse_args()

    tickers = load_tickers(args)
    task = build_task(args)
    check_server()
    results = asyncio.run(run_watchlist(tickers, task, args.concurrency))

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
