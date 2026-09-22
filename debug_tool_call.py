"""Show exactly what the vLLM server returns for the first agent step.

Usage: python debug_tool_call.py
Prints the full raw response for three requests so you can see whether the
model produced text, a tool call, or nothing at all.
"""
import json

from config import MODEL_NAME, client
from tasks import TASKS
from tools import get_tools

TASK = TASKS["sentiment"]
TOOLS = get_tools(TASK.tools)  # the task's own tool subset, not every registered tool

MESSAGES = [
    {"role": "system", "content": TASK.system_prompt},
    {"role": "user", "content": TASK.user_prompt.format(ticker="GOOGL")},
]

CASES = {
    "1. no tools (does the model generate anything?)": {},
    "2. tools, tool_choice=auto": {"tools": TOOLS, "tool_choice": "auto"},
    "3. tools, tool_choice=required": {"tools": TOOLS, "tool_choice": "required"},
}

for label, extra in CASES.items():
    print(f"\n===== {label} =====")
    resp = client.chat.completions.create(model=MODEL_NAME, messages=MESSAGES, **extra)
    choice = resp.choices[0]
    print("finish_reason     :", choice.finish_reason)
    print("completion_tokens :", resp.usage.completion_tokens if resp.usage else "n/a")
    print("content           :", repr(choice.message.content))
    print("tool_calls        :", choice.message.tool_calls)
    print("full message      :", json.dumps(choice.message.model_dump(), indent=2))
