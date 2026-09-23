"""MCP server exposing the agent's tools (tools/) via FastMCP.

    python mcp_server.py                 # stdio transport, for an MCP client (e.g. Claude Desktop) config
    python mcp_server.py --http 8020     # HTTP transport, for a quick manual check with an MCP client

Thin protocol adapter, nothing else: no new tool logic and no new data
source, just the existing functions in tools/ registered as MCP tools under
their existing names/descriptions (see tools/registry.py), so the schema is
not duplicated by hand.
"""
import argparse

from typing import Callable

from fastmcp import FastMCP

from tools import get_tool_functions, get_tools


def build_server(
    name: str = "vllm-market-agent-tools",
    functions: dict[str, Callable] | None = None,
    descriptions: dict[str, str] | None = None,
) -> FastMCP:
    """Register `functions` (default: every tool in tools/) as MCP tools on a fresh server.

    `functions`/`descriptions` are injectable so tests can register stubs instead of
    the real (network-calling) tools; production code always uses the defaults.
    """
    functions = get_tool_functions() if functions is None else functions
    if descriptions is None:
        descriptions = {s["function"]["name"]: s["function"]["description"] for s in get_tools()}
    server = FastMCP(name)
    for tool_name, fn in functions.items():
        server.tool(fn, name=tool_name, description=descriptions[tool_name])
    return server


mcp = build_server()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--http", type=int, metavar="PORT", help="serve over HTTP on this port instead of stdio")
    args = parser.parse_args()
    mcp.run(transport="http", port=args.http) if args.http else mcp.run()


if __name__ == "__main__":
    main()
