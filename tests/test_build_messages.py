"""build_messages: Tests that messages are built correctly and cache is busted for benchmarks,"""

import json
import re

from benchmark_impl import build_messages
from schemas.run_config import PromptConfig

PROMPT = PromptConfig(
    ticker="WMT",
    signal_file="signal.parquet",
    noise_file="noise.parquet",
    noise_levels=[3],
    system="You are a financial assistant.",
    user="Will the price target for {ticker} go up?",
    report="Answer as JSON.",
)
SIGNAL = ["signal headline 1", "signal headline 2"]
NOISE = ["noise headline 1", "noise headline 2", "noise headline 3"]
BUST_ID = re.compile(r"^\[request [0-9a-f]{8}\] ")


def test_structure():
    messages = build_messages(PROMPT, SIGNAL, NOISE, bust_prefix_cache=False)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user"]
    system, user, assistant, tool, report = messages

    assert system["content"] == PROMPT.system
    assert user["content"] == f"Will the price target for {PROMPT.ticker} go up?"

    # One replayed tool call, answered by the tool message with the same id.
    assert assistant["content"] is None
    (call,) = assistant["tool_calls"]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_stock_news"
    assert json.loads(call["function"]["arguments"]) == {"ticker": PROMPT.ticker}
    assert tool["tool_call_id"] == call["id"]

    # Noise first and signal last, one headline per line, so signal sits next to the question.
    assert tool["content"].split("\n") == NOISE + SIGNAL
    assert report["content"] == PROMPT.report


def test_inputs_not_modified():
    signal, noise = list(SIGNAL), list(NOISE)
    build_messages(PROMPT, signal, noise, bust_prefix_cache=True)
    assert (signal, noise) == (SIGNAL, NOISE)  # PROMPT is frozen


def test_bust_puts_an_id_at_the_very_front():
    system, *rest = build_messages(PROMPT, SIGNAL, NOISE, bust_prefix_cache=True)
    assert BUST_ID.match(system["content"])
    assert BUST_ID.sub("", system["content"]) == PROMPT.system
    # Only the system message changes; everything after it is the same as unbusted.
    assert rest == build_messages(PROMPT, SIGNAL, NOISE, bust_prefix_cache=False)[1:]


def test_bust_gives_every_request_a_fresh_id():
    n = 100
    systems = [build_messages(PROMPT, SIGNAL, NOISE, True)[0]["content"] for _ in range(n)]
    assert len(set(systems)) == n
    # The build_messages method prepends system_prompt with [request
    assert all(s.startswith("[request ") for s in systems)


def test_no_bust_gives_identical_prompts():
    # bust_prefix_cache=False: no request id is prepended, so repeated calls give identical messages.
    first = build_messages(PROMPT, SIGNAL, NOISE, bust_prefix_cache=False)
    assert all(build_messages(PROMPT, SIGNAL, NOISE, False) == first for _ in range(10))
