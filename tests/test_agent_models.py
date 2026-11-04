"""Runs the generic agent with Qwen and Gemma.

Each test starts its own server (benchmarks/serve.sh via ManagedServer) with the
model and parsers below, and stops it afterwards. Stop any server on port 8010
first. A model not yet in ~/.cache/huggingface is downloaded on first start,
which counts against READY_TIMEOUT_S; pre-fetch with `hf download <model>` if
the connection is slow.

    python -m pytest tests/test_agent_models.py -v -s
    python -m pytest tests/test_agent_models.py::test_qwen -v -s
    python -m pytest tests/test_agent_models.py::test_gemma -v -s
"""
import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from managed_server import ManagedServer, _server_reachable
from market_agent import agent
from market_agent.store import save_run
from market_agent.tasks import TASKS

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVE_SCRIPT = REPO_ROOT / "benchmarks" / "serve.sh"
READY_TIMEOUT_S = 300
TICKER = "GOOGL"
#TICKER = "AAPL"
TASK = replace(
    TASKS["generic"],
    #system_prompt="You are a quantitative financial assistant. For the requested ticker, call get_stock_news then your response.",
    system_prompt="You are a quantitative financial assistant. For the requested ticker, call all the available tools, then generate your response.",
    user_prompt="Is this {ticker} going to go up or down?",
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(_server_reachable(), reason="a server is already on port 8010; this test starts its own"),
]


@pytest.fixture(scope="module")
def loop():
    # One loop for all cases: config.aclient is a module-level client (see test_agent_integration.py).
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _run_with(model: str, tool_parser: str, reasoning_parser: str, loop, monkeypatch, tmp_path) -> None:
    server = {
        "model": model,
        "tool_call_parser": tool_parser,
        "reasoning_parser": reasoning_parser,
        "ready_timeout_s": READY_TIMEOUT_S,
        "continuous_batching": False,
        "max_num_seqs": None,
        "prefix_cache": True,
        "kv_cache_quantization": False,
        "kv_cache_quantization_bits": None,
    }
    # agent.py binds MODEL_NAME at import; requests and the saved run must name the served model.
    monkeypatch.setattr(agent, "MODEL_NAME", model)
    with ManagedServer(server, SERVE_SCRIPT, tmp_path / "serve.log"):
        result = loop.run_until_complete(agent.run_agent(TICKER, TASK))

    assert "report" in result, f"run failed: {result}"
    path = save_run(result, REPO_ROOT / "runs")
    print(f"\n[{model}] tools: {sorted(result['tool_outputs'])}")
    print(f"confidence: {result['report']['confidence']}")
    print(f"summary: {result['report']['summary']}")
    print(f"saved: {path}")
    assert result["report"]["summary"].strip()


# Both models: 4-bit weights, plus fp16 quantisation scales and biases (one each per group of 64 weights).
#   Qwen3-4B:    ~4.0B parameters, ~4B used per token, 2.1 GB on disk.
#   Gemma 4 E2B: ~4.6B parameters, ~2B used per token ("effective 2B"; the rest are
#                per-layer embedding tables, looked up rather than multiplied), 2.5 GB on disk.
def test_qwen(loop, monkeypatch, tmp_path):
    _run_with("mlx-community/Qwen3-4B-4bit", "qwen", "qwen3", loop, monkeypatch, tmp_path)


def test_gemma(loop, monkeypatch, tmp_path):
    _run_with("mlx-community/Gemma4-E2B-IT-Text-int4", "gemma4", "gemma4", loop, monkeypatch, tmp_path)
