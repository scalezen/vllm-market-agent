"""Tests for the FastMCP wrapper (mcp_server.py), not for the tools' own logic
(no network/yfinance calls here -- see tools/ for that).
"""
import asyncio

import pytest
from fastmcp import Client

from mcp_server import build_server
from tools import all_tool_names, get_tools


def _run(coro):
    return asyncio.run(coro)


# --- real registration: every tool in tools/ shows up correctly -----------------


def test_registers_every_tool_from_the_registry():
    server = build_server()
    async def list_names():
        async with Client(server) as client:
            return {t.name for t in await client.list_tools()}
    assert _run(list_names()) == set(all_tool_names())


def test_descriptions_and_required_args_match_the_tool_schemas():
    server = build_server()
    schemas = {s["function"]["name"]: s["function"] for s in get_tools()}

    async def list_tools():
        async with Client(server) as client:
            return await client.list_tools()

    for tool in _run(list_tools()):
        schema = schemas[tool.name]
        assert tool.description == schema["description"]
        assert tool.input_schema["required"] == schema["parameters"]["required"]


# --- wrapper dispatch: a call reaches the right stub with the right args --------


@pytest.fixture
def stub_server():
    calls = []

    def stub_news(ticker: str) -> str:
        calls.append(("get_stock_news", ticker))
        return "stubbed news for " + ticker

    def stub_technicals(ticker: str) -> str:
        calls.append(("get_technical_indicators", ticker))
        return "stubbed technicals for " + ticker

    def stub_analysts(ticker: str) -> str:
        calls.append(("get_analyst_recommendations", ticker))
        return "stubbed analysts for " + ticker

    functions = {
        "get_stock_news": stub_news,
        "get_technical_indicators": stub_technicals,
        "get_analyst_recommendations": stub_analysts,
    }
    descriptions = {name: f"stub description for {name}" for name in functions}
    return build_server(functions=functions, descriptions=descriptions), calls


def test_call_tool_dispatches_to_the_underlying_function_and_returns_its_result(stub_server):
    server, calls = stub_server

    async def call():
        async with Client(server) as client:
            return await client.call_tool("get_technical_indicators", {"ticker": "AAPL"})

    result = _run(call())
    assert result.content[0].text == "stubbed technicals for AAPL"
    assert calls == [("get_technical_indicators", "AAPL")]


def test_each_stub_tool_is_independently_reachable(stub_server):
    server, calls = stub_server

    async def call_all():
        async with Client(server) as client:
            for name in ("get_stock_news", "get_technical_indicators", "get_analyst_recommendations"):
                await client.call_tool(name, {"ticker": "MSFT"})

    _run(call_all())
    assert calls == [
        ("get_stock_news", "MSFT"),
        ("get_technical_indicators", "MSFT"),
        ("get_analyst_recommendations", "MSFT"),
    ]


def test_missing_required_argument_is_rejected(stub_server):
    server, calls = stub_server

    async def call_without_ticker():
        async with Client(server) as client:
            return await client.call_tool("get_stock_news", {})

    with pytest.raises(Exception):  # fastmcp raises a tool/validation error, not a bare crash
        _run(call_without_ticker())
    assert calls == []  # rejected before the stub ran
