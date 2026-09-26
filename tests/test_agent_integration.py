"""End-to-end checks against a live vLLM server: real tool calls, real model,
real structured report. Slow (each run is a full tool loop) and dependent on
model behaviour, unlike test_mcp_server.py's fast, mocked unit tests.

Each test only confirms the run *succeeded* and its report is populated --
not that the content is correct, since that's exactly what varies run to run.

Requires the server (./serve.sh); skips cleanly if it's not reachable:
    python -m pytest tests/test_agent_integration.py -v
"""
import asyncio
from dataclasses import replace

import pytest
from openai import APIConnectionError

from agent import run_agent
from config import client
from tasks import TASKS


def _server_up() -> bool:
    try:
        client.models.list()
        return True
    except APIConnectionError:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _server_up(), reason="vLLM server not reachable on VLLM_BASE_URL; run ./serve.sh"),
]


@pytest.fixture(scope="module")
def loop():
    """One event loop shared by every test here.

    config.aclient is a module-level singleton meant for one process/one event
    loop, same as sentiment_agent.py's single asyncio.run() call. A separate
    asyncio.run() per test would each spin up and tear down its own loop, and
    reusing aclient's pooled connections across a dead loop crashes with
    "Event loop is closed" on the second test onward.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _run(loop: asyncio.AbstractEventLoop, ticker: str, task) -> dict:
    return loop.run_until_complete(run_agent(ticker, task))


def _assert_ran_successfully(result: dict, expected_fields: set[str]) -> None:
    """The one thing every case checks: the report exists and is populated."""
    assert "report" in result, f"run failed, no report: {result}"
    report = result["report"]
    assert expected_fields <= report.keys(), f"missing fields {expected_fields - report.keys()}"
    assert isinstance(report["summary"], str) and report["summary"].strip(), "empty summary"
    assert 0.0 <= report["confidence"] <= 1.0


SENTIMENT_FIELDS = {"sentiment", "confidence", "news_signal", "technical_signal", "analyst_signal", "summary"}
GENERIC_FIELDS = {"confidence", "summary"}


def _assert_min_tools_called(result: dict, minimum: int) -> None:
    """Confirms at least `minimum` distinct tools were called.

    tool_outputs is keyed by tool name (agent.py:90), so a repeated call to the
    same tool wouldn't show up twice here -- fine for these tools, which take
    no argument that would make a second call meaningfully different.
    """
    called = set(result["tool_outputs"])
    assert len(called) >= minimum, f"only called {sorted(called)}, expected at least {minimum}"

# Reinforces tool coverage explicitly, for the two generic-task cases below
# whose base system prompt (tasks/generic.py) no longer states this itself.
ALL_TOOLS_INSTRUCTION = (
    "You must use all the tools available to you "
    "before answering."
)


def _consumer_fan_task(ticker: str):
    """A generic-task variant with a persona; told explicitly to use every tool.

    agent.py only formats {ticker} into the *user* prompt, never the system
    prompt, so the ticker is substituted here by hand rather than left as a
    literal "{ticker}" for the framework to fill in.
    """
    return replace(
        TASKS["generic"],
        system_prompt=(
            f"You are a quantitative financial assistant. {ALL_TOOLS_INSTRUCTION} "
            f"Assume you are a fan of consumer products from {ticker}."
        ),
        user_prompt="How will this business do in the age of AI?",
    )


def test_sentiment_agent_googl(loop):
    # tasks/sentiment.py's own system prompt already mandates using all three tools.
    task = replace(TASKS["sentiment"], user_prompt="Find the sentiment for {ticker}.")
    result = _run(loop, "GOOGL", task)
    _assert_ran_successfully(result, SENTIMENT_FIELDS)
    _assert_min_tools_called(result, len(task.tools))
    assert result["report"]["sentiment"] in {"Bullish", "Bearish", "Neutral"}


def test_generic_agent_googl(loop):
    task = _consumer_fan_task("GOOGL")
    result = _run(loop, "GOOGL", task)
    _assert_ran_successfully(result, GENERIC_FIELDS)
    _assert_min_tools_called(result, len(task.tools))


def test_generic_agent_aapl(loop):
    task = _consumer_fan_task("AAPL")
    result = _run(loop, "AAPL", task)
    _assert_ran_successfully(result, GENERIC_FIELDS)
    _assert_min_tools_called(result, len(task.tools))


def test_generic_agent_rsi_prompt(loop):
    # The narrower question from our earlier discussion: does a specific ask
    # still get a populated report, regardless of how many tools it calls --
    # no ALL_TOOLS_INSTRUCTION and no minimum-tools assertion here on purpose.
    task = replace(
        TASKS["generic"],
        system_prompt="You are a quantitative financial assistant. Respond as if you are a technical analyst.",
        user_prompt="What is {ticker}'s current RSI?",
    )
    result = _run(loop, "AAPL", task)
    _assert_ran_successfully(result, GENERIC_FIELDS)
