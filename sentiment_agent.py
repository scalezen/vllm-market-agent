"""Stock sentiment agent backed by a local vLLM-MLX server.

Start the server first (see serve.sh), then run:
    python sentiment_agent.py                      # default ticker
    python sentiment_agent.py AAPL MSFT NVDA       # batch, run concurrently
    python sentiment_agent.py --watchlist watchlist.txt --out results.jsonl

Per ticker the model calls three tools (news, technical indicators, analyst
recommendations), then writes a JSON report constrained to REPORT_SCHEMA.
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from openai import APIConnectionError, AsyncOpenAI, OpenAI
from pydantic import BaseModel

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8010/v1")
MODEL_NAME = os.environ.get("VLLM_MODEL", "mlx-community/Qwen3-4B-4bit")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "mlx-community/embeddinggemma-300m-6bit")
# Headlines with cosine similarity at or above this are treated as the same story.
DEDUP_THRESHOLD = float(os.environ.get("DEDUP_THRESHOLD", "0.85"))
MAX_STEPS = 6
DEFAULT_CONCURRENCY = 4
# vllm-mlx's default engine runs one generation at a time and answers 503 to the
# rest, so LLM calls are serialized by default while tool fetches still overlap.
# Raise this if you start the server with continuous batching.
LLM_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "1"))

client = OpenAI(base_url=BASE_URL, api_key="local-dev")  # sync: used by tools (run in threads)
aclient = AsyncOpenAI(base_url=BASE_URL, api_key="local-dev")  # vLLM ignores the key
llm_slots = asyncio.Semaphore(LLM_CONCURRENCY)


async def chat(**kwargs):
    async with llm_slots:
        return await aclient.chat.completions.create(model=MODEL_NAME, **kwargs)


# --------------------------------------------------------------------------
# Structured output
# --------------------------------------------------------------------------
Signal = Literal["Bullish", "Bearish", "Neutral", "Unavailable"]


class SentimentReport(BaseModel):
    sentiment: Literal["Bullish", "Bearish", "Neutral"]
    confidence: float  # 0.0-1.0; clamped after parsing (bounds are not enforced by decoding)
    news_signal: Signal
    technical_signal: Signal
    analyst_signal: Signal
    justification: str


# NOTE: vllm-mlx rejects `tools` together with `response_format`, so this is only
# used on the final, tool-free call.
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_report",
        "schema": SentimentReport.model_json_schema(),
        "strict": True,
    },
}


# --------------------------------------------------------------------------
# Embeddings + dedup
# --------------------------------------------------------------------------
def dedup_headlines(headlines: list[str], threshold: float = DEDUP_THRESHOLD) -> list[str]:
    """Drop near-duplicate headlines, keeping the first of each cluster.

    Uses the server's embeddings endpoint; falls back to exact (case-insensitive)
    matching if the server has no embedding model loaded.
    """
    if len(headlines) < 2:
        return headlines
    try:
        data = client.embeddings.create(model=EMBED_MODEL, input=headlines).data
        vecs = np.array([d.embedding for d in sorted(data, key=lambda d: d.index)], dtype=float)
    except Exception as exc:  # no embedding model, connection error, ...
        logger.warning("Embedding dedup unavailable ({}); using exact-match dedup", exc)
        seen: dict[str, str] = {}
        for h in headlines:
            seen.setdefault(h.strip().lower(), h)
        return list(seen.values())

    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
    kept: list[int] = []
    for i in range(len(headlines)):
        if all(float(vecs[i] @ vecs[j]) < threshold for j in kept):
            kept.append(i)
    logger.info("Dedup: {} headlines -> {} unique", len(headlines), len(kept))
    return [headlines[i] for i in kept]


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
def get_stock_news(ticker: str, n: int = 10) -> str:
    """Fetch the latest news headlines for a ticker, with near-duplicates removed."""
    logger.info("[tool] get_stock_news({})", ticker)
    try:
        news = yf.Ticker(ticker).news or []
    except Exception as exc:  # network errors, Yahoo rate limits, etc.
        return f"Error fetching news for {ticker}: {exc}"

    headlines = []
    for item in news[:n]:
        # yfinance >= 0.2.50 nests fields under "content"; older versions are flat.
        title = item.get("content", item).get("title")
        if title:
            headlines.append(title)
    headlines = dedup_headlines(headlines)
    return json.dumps(headlines) if headlines else "No news found."


def _rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    last_gain, last_loss = gain.iloc[-1], loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    return float(100 - 100 / (1 + last_gain / last_loss))


def get_technical_indicators(ticker: str) -> str:
    """RSI(14), 50/200-day SMAs and MACD(12,26,9) computed from a year of daily closes."""
    logger.info("[tool] get_technical_indicators({})", ticker)
    try:
        close = yf.Ticker(ticker).history(period="1y")["Close"].dropna()
    except Exception as exc:
        return f"Error fetching price history for {ticker}: {exc}"
    if len(close) < 35:  # MACD needs at least the slow EMA window plus signal
        return f"Not enough price history for {ticker} to compute indicators."

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    price = float(close.iloc[-1])
    rsi = _rsi(close)
    sma50 = float(close.tail(50).mean())
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else None

    result = {
        "price": round(price, 2),
        "rsi_14": round(rsi, 1),
        "rsi_reading": "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral",
        "sma_50": round(sma50, 2),
        "sma_200": round(sma200, 2) if sma200 else None,
        "price_vs_sma_50": "above" if price > sma50 else "below",
        "price_vs_sma_200": None if sma200 is None else ("above" if price > sma200 else "below"),
        "macd": round(float(macd.iloc[-1]), 3),
        "macd_signal": round(float(macd_signal.iloc[-1]), 3),
        "macd_reading": "bullish" if macd.iloc[-1] > macd_signal.iloc[-1] else "bearish",
        "change_1m_pct": round(float(price / close.iloc[-22] - 1) * 100, 1) if len(close) > 22 else None,
    }
    return json.dumps(result)


def get_analyst_recommendations(ticker: str) -> str:
    """Analyst rating breakdown, price targets and the most recent upgrades/downgrades."""
    logger.info("[tool] get_analyst_recommendations({})", ticker)
    t = yf.Ticker(ticker)
    result: dict = {}
    try:
        summary = t.recommendations
        if summary is not None and not summary.empty:
            row = summary.iloc[0]  # current month
            result["ratings_current_month"] = {
                k: int(row[k]) for k in ("strongBuy", "buy", "hold", "sell", "strongSell") if k in row
            }
    except Exception as exc:
        logger.warning("recommendations failed for {}: {}", ticker, exc)
    try:
        targets = t.analyst_price_targets
        if targets:
            result["price_targets"] = {k: round(float(v), 2) for k, v in targets.items() if v is not None}
    except Exception as exc:
        logger.warning("price targets failed for {}: {}", ticker, exc)
    try:
        changes = t.upgrades_downgrades
        if changes is not None and not changes.empty:
            result["recent_changes"] = [
                {
                    "date": str(idx.date()),
                    "firm": row.get("Firm"),
                    "action": row.get("Action"),
                    "from": row.get("FromGrade"),
                    "to": row.get("ToGrade"),
                }
                for idx, row in changes.head(5).iterrows()
            ]
    except Exception as exc:
        logger.warning("upgrades/downgrades failed for {}: {}", ticker, exc)
    return json.dumps(result) if result else f"No analyst data found for {ticker}."


def _ticker_tool(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker symbol (e.g., AAPL, GOOGL)"}
                },
                "required": ["ticker"],
            },
        },
    }


TOOLS = [
    _ticker_tool("get_stock_news", "Fetch the latest (de-duplicated) news headlines for a stock ticker."),
    _ticker_tool(
        "get_technical_indicators",
        "Get RSI, 50/200-day moving averages, MACD and 1-month price change for a stock ticker.",
    ),
    _ticker_tool(
        "get_analyst_recommendations",
        "Get analyst buy/hold/sell counts, price targets and recent upgrades/downgrades for a stock ticker.",
    ),
]
TOOL_REGISTRY = {
    "get_stock_news": get_stock_news,
    "get_technical_indicators": get_technical_indicators,
    "get_analyst_recommendations": get_analyst_recommendations,
}

SYSTEM_PROMPT = (
    "You are a quantitative financial assistant. For the requested ticker, call ALL of "
    "get_stock_news, get_technical_indicators and get_analyst_recommendations, then "
    "assess overall sentiment (Bullish, Bearish, or Neutral) from the results."
)
REPORT_PROMPT = (
    "Using only the tool results above, output the sentiment report as JSON. "
    "Give each of news_signal, technical_signal and analyst_signal (use 'Unavailable' if that "
    "tool returned no usable data), the overall sentiment, a confidence between 0 and 1, "
    "and a one-sentence justification."
)


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
def check_server() -> None:
    """Fail fast with a useful message if the vLLM server is not reachable."""
    try:
        served = [m.id for m in client.models.list().data]
    except APIConnectionError:
        sys.exit(
            f"Cannot reach the vLLM server at {BASE_URL}.\n"
            "Start it first (./serve.sh) or set VLLM_BASE_URL to the right host/port."
        )
    if MODEL_NAME not in served:
        sys.exit(f"Model '{MODEL_NAME}' is not served. Server has: {served}")


async def run_tool(call) -> str:
    fn = TOOL_REGISTRY.get(call.function.name)
    if fn is None:
        return f"Unknown tool: {call.function.name}"
    try:
        args = json.loads(call.function.arguments or "{}")
        # yfinance is blocking, so keep it off the event loop.
        return await asyncio.to_thread(fn, **args)
    except (json.JSONDecodeError, TypeError) as exc:
        return f"Bad tool arguments: {exc}"


async def run_agent(ticker: str) -> dict:
    """Run the tool loop, then return the structured report as a dict."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Can you check the current sentiment for {ticker}?"},
    ]
    for step in range(MAX_STEPS):
        # A small model often answers from memory instead of calling a tool, so
        # force a tool call on the first step; afterwards let the model decide.
        response = await chat(
            messages=messages,
            tools=TOOLS,
            tool_choice="required" if step == 0 else "auto",
        )
        message = response.choices[0].message
        if not message.tool_calls:
            if step == 0:
                raise RuntimeError(f"No tool call parsed for {ticker}: {message.content!r}")
            break

        logger.info("{} step {}: {}", ticker, step, [c.function.name for c in message.tool_calls])
        messages.append(message.model_dump(exclude_none=True))
        observations = await asyncio.gather(*(run_tool(c) for c in message.tool_calls))
        for call, observation in zip(message.tool_calls, observations):
            logger.debug("{} <- {}", call.function.name, observation)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
    else:
        logger.warning("{}: hit MAX_STEPS; requesting the report with what was gathered", ticker)

    response = await chat(
        messages=messages + [{"role": "user", "content": REPORT_PROMPT}],
        response_format=RESPONSE_FORMAT,
    )
    content = response.choices[0].message.content or ""
    report = SentimentReport.model_validate_json(content)
    report.confidence = min(max(report.confidence, 0.0), 1.0)
    return {
        "ticker": ticker,
        **report.model_dump(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


async def run_watchlist(tickers: list[str], concurrency: int = DEFAULT_CONCURRENCY) -> list[dict]:
    """Analyse many tickers concurrently; a failure on one does not stop the rest."""
    sem = asyncio.Semaphore(concurrency)

    async def one(ticker: str) -> dict:
        async with sem:
            try:
                return await run_agent(ticker)
            except Exception as exc:
                logger.error("{} failed: {}", ticker, exc)
                return {"ticker": ticker, "error": str(exc)}

    return await asyncio.gather(*(one(t) for t in tickers))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
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
    args = parser.parse_args()

    tickers = load_tickers(args)
    check_server()
    results = asyncio.run(run_watchlist(tickers, args.concurrency))

    print(json.dumps(results[0] if len(results) == 1 else results, indent=2))
    if args.out:
        with open(args.out, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in results)


if __name__ == "__main__":
    main()
