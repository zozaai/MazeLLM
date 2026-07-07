"""LFM2.5-specific formatting: neutral expert turns -> LFM2 native chat examples.

Verified against LFM2.5's `chat_template.jinja` and LiquidAI's tool-use docs:
its template renders only `message["content"]` (it never reads `tool_calls`), so
the tool call must live in the assistant *content* as
`<|tool_call_start|>[name(args)]<|tool_call_end|>`. Tools are passed to the chat
template in a **flat** schema (`{name, description, parameters}`), which the
template appends to the system turn as `List of tools: [...]`.
"""
from __future__ import annotations

from backend.agent.tools import TOOL_SCHEMAS

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"


def _argstr(args: dict) -> str:
    return ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in args.items())


def format_assistant(name: str, args: dict, rationale: str, include_rationale: bool = True) -> str:
    call = f"{TOOL_CALL_START}[{name}({_argstr(args)})]{TOOL_CALL_END}"
    return f"{call}\n{rationale}" if (include_rationale and rationale) else call


def format_example(turn: dict, include_rationale: bool = True) -> dict:
    return {
        "messages": [
            {"role": "system", "content": turn["system"]},
            {"role": "user", "content": turn["memory"]},
            {"role": "assistant",
             "content": format_assistant(turn["name"], turn["args"], turn["rationale"], include_rationale)},
        ],
        "meta": turn["meta"],
    }


def tool_schemas() -> list[dict]:
    """TOOL_SCHEMAS converted from OpenAI ({type, function}) to LFM2's flat shape."""
    return [
        {"name": t["function"]["name"],
         "description": t["function"]["description"],
         "parameters": t["function"]["parameters"]}
        for t in TOOL_SCHEMAS
    ]
