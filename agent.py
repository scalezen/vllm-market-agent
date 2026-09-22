"""The agent: a tool-calling loop, then a schema-constrained JSON report. What to ask,
which tools to offer and what the report looks like all come from a Task (see tasks/)."""
import asyncio
import json
import sys
import time
from datetime import datetime, timezone

from loguru import logger
from openai import APIConnectionError

from config import BASE_URL, DEFAULT_CONCURRENCY, MAX_STEPS, MODEL_NAME, aclient, client, llm_slots
from tasks import Task
from tools import get_tool_functions, get_tools
from usage import Usage, current_usage

# Appended to the task's system prompt per run; tool results carry their own dates.
DATE_NOTE = (
    " Today is {today} (UTC). Tool results are dated: weight recent news and analyst "
    "actions more heavily, and treat anything more than a few days old as background."
)


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


async def run_tool(call, functions: dict) -> str:
    fn = functions.get(call.function.name)
    if fn is None:
        return f"Unknown tool: {call.function.name}"
    try:
        args = json.loads(call.function.arguments or "{}")
        # yfinance is blocking, so keep it off the event loop.
        return await asyncio.to_thread(fn, **args)
    except (json.JSONDecodeError, TypeError) as exc:
        return f"Bad tool arguments: {exc}"


async def run_agent(ticker: str, task: Task) -> dict:
    """Run the tool loop for `task`, then return the structured report as a dict."""
    usage = Usage()
    current_usage.set(usage)
    started = time.perf_counter()
    tools = get_tools(task.tools)
    functions = get_tool_functions(task.tools)
    user_prompt = task.user_prompt.format(ticker=ticker)
    messages = [
        {"role": "system", "content": task.system_prompt + DATE_NOTE.format(today=datetime.now(timezone.utc).date())},
        {"role": "user", "content": user_prompt},
    ]
    tool_outputs: dict[str, str] = {}
    for step in range(MAX_STEPS):
        # A small model often answers from memory instead of calling a tool, so
        # force a tool call on the first step; afterwards let the model decide.
        response = await chat(
            messages=messages,
            tools=tools,
            tool_choice="required" if step == 0 else "auto",
        )
        message = response.choices[0].message
        if not message.tool_calls:
            if step == 0:
                raise RuntimeError(f"No tool call parsed for {ticker}: {message.content!r}")
            break

        logger.info("{} step {}: {}", ticker, step, [c.function.name for c in message.tool_calls])
        messages.append(message.model_dump(exclude_none=True))
        observations = await asyncio.gather(*(run_tool(c, functions) for c in message.tool_calls))
        for call, observation in zip(message.tool_calls, observations):
            logger.debug("{} <- {}", call.function.name, observation)
            tool_outputs[call.function.name] = observation
            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
    else:
        logger.warning("{}: hit MAX_STEPS; requesting the report with what was gathered", ticker)

    # NOTE: vllm-mlx rejects `tools` together with `response_format`, so the schema is
    # only sent on this final, tool-free call.
    response = await chat(
        messages=messages + [{"role": "user", "content": task.report_prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": f"{task.name}_report",
                "schema": task.report_model.response_schema(),
                "strict": True,
            },
        },
    )
    content = response.choices[0].message.content or ""
    report = task.report_model.model_validate_json(content)
    # The report is nested (rather than splatted into the top level) so a run's shape
    # does not depend on which task produced it; store.py/backtest.py key off "report".
    return {
        "ticker": ticker,
        "task": task.name,
        "user_prompt": user_prompt,
        "prompt_hash": task.prompt_hash,
        "report": report.model_dump(),
        "model": MODEL_NAME,
        "tool_outputs": tool_outputs,
        "usage": usage.as_dict(time.perf_counter() - started),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


async def run_watchlist(tickers: list[str], task: Task, concurrency: int = DEFAULT_CONCURRENCY) -> list[dict]:
    """Analyse many tickers concurrently; a failure on one does not stop the rest."""
    sem = asyncio.Semaphore(concurrency)

    async def one(ticker: str) -> dict:
        async with sem:
            try:
                return await run_agent(ticker, task)
            except Exception as exc:
                logger.error("{} failed: {}", ticker, exc)
                return {"ticker": ticker, "error": str(exc)}

    return await asyncio.gather(*(one(t) for t in tickers))
