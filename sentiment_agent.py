"""Stock-news sentiment agent backed by a local vLLM-MLX server.

Start the server first (see serve.sh), then run:
    python sentiment_agent.py [TICKER]
"""
import json
import os
import sys

from loguru import logger
import yfinance as yf
from openai import APIConnectionError, OpenAI

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8010/v1")
MODEL_NAME = os.environ.get("VLLM_MODEL", "mlx-community/Qwen3-4B-4bit")
MAX_STEPS = 5

client = OpenAI(base_url=BASE_URL, api_key="local-dev")  # vLLM ignores the key


def get_stock_news(ticker: str, n: int = 4) -> str:
    """Fetch the latest news headlines for a given stock ticker."""
    print(f"\n[Agent Action] Fetching news for {ticker}...")
    try:
        news = yf.Ticker(ticker).news or []
    except Exception as exc:  # network errors, Yahoo rate limits, etc.
        return f"Error fetching news for {ticker}: {exc}"

    headlines = []
    for item in news[:n]:
        # yfinance >= 0.2.50 nests fields under "content"; older versions are flat.
        content = item.get("content", item)
        title = content.get("title")
        if title:
            headlines.append(title)
    print(f"[Observation] {headlines}")
    return json.dumps(headlines) if headlines else "No news found."


TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_stock_news",
        "description": "Fetch the latest news headlines for a given stock ticker to analyze sentiment.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The stock ticker symbol (e.g., AAPL, GOOGL)"}
            },
            "required": ["ticker"],
        },
    },
}]
TOOL_REGISTRY = {"get_stock_news": get_stock_news}

SYSTEM_PROMPT = (
    "You are a quantitative financial assistant. Use tools to fetch recent news. "
    "Then provide a brief sentiment analysis (Bullish, Bearish, or Neutral) "
    "with a one-sentence justification."
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


def final_answer(messages: list) -> str:
    """Request the written analysis with tools switched off."""
    logger.warning("Empty answer after tool use; retrying without tools")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
        + [{
            "role": "user",
            "content": "Using only the headlines returned above, give the sentiment "
            "(Bullish, Bearish, or Neutral) and a one-sentence justification.",
        }],
    )
    content = response.choices[0].message.content or ""
    logger.info(
        "Final retry: finish_reason={} content={!r}",
        response.choices[0].finish_reason,
        content,
    )
    return content


def run_agent(ticker: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Can you check the current sentiment for {ticker}?"},
    ]
    for step in range(MAX_STEPS):
        print("Thinking...")
        # A 3B model often answers from memory instead of calling a tool, so
        # force a tool call on the first step; afterwards let the model decide.
        tool_choice = "required" if step == 0 else "auto"
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=TOOLS, tool_choice=tool_choice
        )
        message = response.choices[0].message

        if not message.tool_calls:
            if step == 0:
                # The server could not parse a tool call from the model output.
                logger.warning(
                    "Step 0: no tool call parsed. finish_reason={} raw content={!r}",
                    response.choices[0].finish_reason,
                    message.content,
                )
            else:
                # Normal exit: the model has the observations and gives its answer.
                logger.info(
                    "Step {}: final answer, finish_reason={} content={!r}",
                    step,
                    response.choices[0].finish_reason,
                    message.content,
                )
            if message.content and message.content.strip():
                return message.content
            if step > 0:
                # Small Llama models often emit an empty turn after a tool
                # result while tools are still offered. Ask once more, with no
                # tools, so the model must answer in plain text.
                return final_answer(messages)
            return ""

        logger.info(
            "Step {}: tool calls {}",
            step,
            [(c.function.name, c.function.arguments) for c in message.tool_calls],
        )

        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            fn = TOOL_REGISTRY.get(call.function.name)
            try:
                args = json.loads(call.function.arguments or "{}")
                observation = fn(**args) if fn else f"Unknown tool: {call.function.name}"
                logger.info(f"Observation is : {observation}")
            except (json.JSONDecodeError, TypeError) as exc:
                observation = f"Bad tool arguments: {exc}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})

    return "Stopped: reached the maximum number of agent steps."


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "GOOGL"
    check_server()
    print(f"\n[Final Analysis]\n{run_agent(ticker)}")


if __name__ == "__main__":
    main()
