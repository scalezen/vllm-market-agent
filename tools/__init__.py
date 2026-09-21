"""Tools the model can call. One module per tool; importing them registers them.

To add a tool: create a module here with a @register_tool function and import
it below.
"""
from .registry import all_tool_names, get_tool_functions, get_tools

# Registration order is the order the model sees the tools in; keep it deliberate, not alphabetical.
from . import news, technicals, analysts  # isort: skip  # noqa: E402,F401

# Every registered tool; agent.py uses these until tasks pick tools by name.
TOOLS = get_tools()
TOOL_REGISTRY = get_tool_functions()

__all__ = ["TOOLS", "TOOL_REGISTRY", "all_tool_names", "get_tool_functions", "get_tools"]
