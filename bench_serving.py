"""Benchmark: server throughput and latency vs. concurrent request count.

Exercises the inference client (config.aclient) directly against one frozen,
realistic prompt -- not the agent, and not live tool data -- so results are
reproducible run to run and comparable across server configs. Deliberately
independent of agent.py/tasks/reports so it keeps working across changes there.

Compare two server configs by starting each with serve.sh, then running this
against each with a --label naming the config, e.g.:

    ./serve.sh                                          # config A: baseline
    python bench_serving.py --label A-baseline --csv bench_results.csv

    CONTINUOUS_BATCHING=true MAX_NUM_SEQS=8 ./serve.sh   # config B
    python bench_serving.py --label B-continuous-batching-8 --csv bench_results.csv

Record hardware, model, quantisation and the exact server flags alongside
whatever --csv this produces -- see docs/decisions.md.
"""
import argparse
import asyncio
import csv
import time
from dataclasses import dataclass
from pathlib import Path

from config import MODEL_NAME, aclient

# A frozen prompt shaped like the sentiment task's final call: system + user +
# one real captured tool round-trip (AAPL, 2026-09-22, ~1250 prompt tokens),
# then the report request. Hardcoded (not imported) so this script does not
# depend on tasks/reports/agent and stays usable even if those change.
_SYSTEM_PROMPT = (
    "You are a quantitative financial assistant. For the requested ticker, call ALL of "
    "get_stock_news, get_technical_indicators and get_analyst_recommendations, then "
    "assess overall sentiment (Bullish, Bearish, or Neutral) from the results."
)
_NEWS = (
    '[{"published": "2026-09-22T19:32Z", "title": "Apple Takes Aim at Nvidia’s AI Economics: '
    'New Macs Have ‘No Cost per Token,’ Hardware Chief Says"}, '
    '{"published": "2026-09-22T19:25Z", "title": "Stock Market Today: Dow Skids As Trump Threatens '
    'Iran; This Medical Play Hits A Buy Zone (Live Coverage)"}, '
    '{"published": "2026-09-22T19:07Z", "title": "Rising Memory Costs Dampen Consumer PC Demand, UBS Says"}, '
    '{"published": "2026-09-22T18:48Z", "title": "Apple’s New CEO Is Already Being Pressured to Drop '
    'the Chinese Chip Deal Tim Cook Fought to Keep"}, '
    '{"published": "2026-09-22T18:39Z", "title": "Netflix Stock Downgraded As YouTube Swipes Viewers"}, '
    '{"published": "2026-09-22T18:04Z", "title": "AAPL Stock Hits Record Highs — Apple Reportedly '
    'Explores Screenless Health Trackers And Other Wearables Following iPhone Duo Launch"}, '
    '{"published": "2026-09-22T17:54Z", "title": "Apple doesn\'t need to be in first place for AI. '
    'Here\'s why."}, '
    '{"published": "2026-09-22T17:40Z", "title": "Dan Ives Calls Meta Muse A ‘Game Changer,’ Sees '
    'Apple As AI ‘Toll Collector’ In AI Arms Race"}, '
    '{"published": "2026-09-22T14:03Z", "title": "The Magnificent 7 are back: Meta’s AI bet leads '
    'the charge"}, '
    '{"published": "2026-09-21T19:52Z", "title": "Meta stock jumps as Wells Fargo raises price target"}]'
)
_TECHNICALS = (
    '{"as_of": "2026-09-22", "intraday": true, "price": 340.04, "rsi_14": 67.1, "rsi_reading": "neutral", '
    '"sma_50": 320.96, "sma_200": 286.49, "price_vs_sma_50": "above", "price_vs_sma_200": "above", '
    '"macd": 6.294, "macd_signal": 4.74, "macd_reading": "bullish", "change_1m_pct": 9.9}'
)
_ANALYSTS = (
    '{"as_of": "2026-09-22", "ratings_current_month": {"strongBuy": 6, "buy": 19, "hold": 13, "sell": 3, '
    '"strongSell": 3}, "price_targets": {"current": 340.04, "high": 405.0, "low": 215.0, "mean": 328.22, '
    '"median": 340.0}, "recent_changes": [{"date": "2026-09-18", "firm": "Evercore ISI Group", '
    '"action": "main", "from": "Outperform", "to": "Outperform"}, {"date": "2026-09-17", '
    '"firm": "B of A Securities", "action": "reit", "from": "Buy", "to": "Buy"}]}'
)
_TOOL_CALLS = [
    {"id": "call_0", "type": "function", "function": {"name": "get_stock_news", "arguments": '{"ticker": "AAPL"}'}},
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_technical_indicators", "arguments": '{"ticker": "AAPL"}'},
    },
    {
        "id": "call_2",
        "type": "function",
        "function": {"name": "get_analyst_recommendations", "arguments": '{"ticker": "AAPL"}'},
    },
]
MESSAGES = [
    {"role": "system", "content": _SYSTEM_PROMPT},
    {"role": "user", "content": "Can you check the current sentiment for AAPL?"},
    {"role": "assistant", "content": None, "tool_calls": _TOOL_CALLS},
    {"role": "tool", "tool_call_id": "call_0", "content": _NEWS},
    {"role": "tool", "tool_call_id": "call_1", "content": _TECHNICALS},
    {"role": "tool", "tool_call_id": "call_2", "content": _ANALYSTS},
    {
        "role": "user",
        "content": "Using only the tool results above, output the sentiment as JSON with a one-sentence summary.",
    },
]
# Minimal inline schema (not reports.SentimentReport) to keep this script decoupled.
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "bench_report",
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string", "enum": ["Bullish", "Bearish", "Neutral"]},
                "summary": {"type": "string"},
            },
            "required": ["sentiment", "summary"],
        },
        "strict": True,
    },
}


@dataclass
class RequestResult:
    ok: bool
    latency_s: float
    completion_tokens: int = 0
    error: str = ""


async def one_request() -> RequestResult:
    start = time.perf_counter()
    try:
        response = await aclient.chat.completions.create(
            model=MODEL_NAME, messages=MESSAGES, response_format=RESPONSE_FORMAT
        )
        latency = time.perf_counter() - start
        tokens = response.usage.completion_tokens if response.usage else 0
        return RequestResult(True, latency, tokens)
    except Exception as exc:  # 503s under load, timeouts, etc. are the point of the benchmark
        return RequestResult(False, time.perf_counter() - start, error=str(exc)[:150])


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def run_at_concurrency(n: int) -> dict:
    """Fire `n` requests at once; latency/throughput are computed over the ones that succeeded."""
    start = time.perf_counter()
    results = await asyncio.gather(*(one_request() for _ in range(n)))
    wall = time.perf_counter() - start
    ok = [r for r in results if r.ok]
    latencies = [r.latency_s for r in ok]
    total_tokens = sum(r.completion_tokens for r in ok)
    return {
        "concurrency": n,
        "n_ok": len(ok),
        "n_error": len(results) - len(ok),
        "wall_seconds": round(wall, 2),
        "total_completion_tokens": total_tokens,
        "throughput_tok_s": round(total_tokens / wall, 1) if wall and ok else None,
        "latency_p50_s": round(p, 2) if (p := _percentile(latencies, 0.5)) is not None else None,
        "latency_p90_s": round(p, 2) if (p := _percentile(latencies, 0.9)) is not None else None,
        "latency_p99_s": round(p, 2) if (p := _percentile(latencies, 0.99)) is not None else None,
        "sample_error": next((r.error for r in results if not r.ok), ""),
    }


async def run_all(concurrencies: list[int], reps: int, label: str) -> list[dict]:
    rows = []
    for n in concurrencies:
        for rep in range(reps):
            row = {"label": label, "rep": rep, **await run_at_concurrency(n)}
            print(row)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--reps", type=int, default=3, help="repeats per concurrency level, for stability")
    parser.add_argument("--label", required=True, help="name for the server config under test, e.g. A-baseline")
    parser.add_argument("--csv", help="append rows to this CSV (written even if it exists)")
    args = parser.parse_args()

    rows = asyncio.run(run_all(args.concurrencies, args.reps, args.label))

    if args.csv:
        path = Path(args.csv)
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            if write_header:
                writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
