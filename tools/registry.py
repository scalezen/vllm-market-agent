"""Tool registry: tools register themselves with @register_tool; tasks pick them by name.

    @register_tool("What the model reads to decide when to call it.", params=TICKER_PARAM)
    def get_something(ticker: str, n: int = 10) -> str: ...

The function-calling schema is built from the signature. Parameters with a
default (like `n`) are left out of the schema, so the model only supplies the
required ones.
"""
import inspect
from typing import Callable

TICKER_PARAM = {"ticker": "The stock ticker symbol (e.g., AAPL, GOOGL)"}

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}
_FUNCTIONS: dict[str, Callable] = {}
_SCHEMAS: dict[str, dict] = {}


def _schema(fn: Callable, description: str, params: dict[str, str]) -> dict:
    properties, required = {}, []
    for name, p in inspect.signature(fn).parameters.items():
        if p.default is not inspect.Parameter.empty:
            continue  # optional: not offered to the model
        prop = {"type": _JSON_TYPES.get(p.annotation, "string")}
        if name in params:
            prop["description"] = params[name]
        properties[name] = prop
        required.append(name)
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def register_tool(description: str, params: dict[str, str] | None = None) -> Callable:
    """Register a tool under its function name; `params` maps parameter name -> description."""

    def decorator(fn: Callable) -> Callable:
        _FUNCTIONS[fn.__name__] = fn
        _SCHEMAS[fn.__name__] = _schema(fn, description, params or {})
        return fn

    return decorator


def all_tool_names() -> list[str]:
    return list(_FUNCTIONS)


def _check(names: list[str]) -> None:
    unknown = [n for n in names if n not in _FUNCTIONS]
    if unknown:
        raise KeyError(f"Unknown tool(s) {unknown}; available: {all_tool_names()}")


def get_tools(names: list[str] | None = None) -> list[dict]:
    """Function-calling schemas for `names` (default: all tools), in the order given."""
    names = all_tool_names() if names is None else names
    _check(names)
    return [_SCHEMAS[n] for n in names]


def get_tool_functions(names: list[str] | None = None) -> dict[str, Callable]:
    """name -> function for `names` (default: all tools)."""
    names = all_tool_names() if names is None else names
    _check(names)
    return {n: _FUNCTIONS[n] for n in names}
