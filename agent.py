"""The sentiment agent: tool-calling loop, then a schema-constrained JSON report."""
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from typing import Literal

from loguru import logger
from openai import APIConnectionError
from pydantic import BaseModel

from config import BASE_URL, DEFAULT_CONCURRENCY, MAX_STEPS, MODEL_NAME, aclient, client, llm_slots
from tools import TOOL_REGISTRY, TOOLS
from usage import Usage, current_usage

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

SYSTEM_PROMPT = (
    "You are a quantitative financial assistant. For the requested ticker, call ALL of "
    "get_stock_news, get_technical_indicators and get_analyst_recommendations, then "
    "assess overall sentiment (Bullish, Bearish, or Neutral) from the results."
)
# Appended to SYSTEM_PROMPT per run; tool results carry their own dates.
DATE_NOTE = (
    " Today is {today} (UTC). Tool results are dated: weight recent news and analyst "
    "actions more heavily, and treat anything more than a few days old as background."
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


async def chat(**kwargs):
    async with llm_slots:
        start = time.perf_counter()
        response = await aclient.chat.completions.create(model=MODEL_NAME, **kwargs)
        if usage := current_usage.get():
            usage.add_chat(response, time.perf_counter() - start)
        return response


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
    usage = Usage()
    current_usage.set(usage)
    started = time.perf_counter()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + DATE_NOTE.format(today=datetime.now(timezone.utc).date())},
        {"role": "user", "content": f"Can you check the current sentiment for {ticker}?"},
    ]
    tool_outputs: dict[str, str] = {}
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
            tool_outputs[call.function.name] = observation
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
        "model": MODEL_NAME,
        "tool_outputs": tool_outputs,
        "usage": usage.as_dict(time.perf_counter() - started),
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
