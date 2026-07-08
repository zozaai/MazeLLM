"""Qwen3.5-specific formatting: neutral expert turns -> Qwen chat examples.

Verified against `unsloth/Qwen3.5-4B`'s chat template. Qwen renders **structured**
`tool_calls` into its XML function-call format
(`<tool_call><function=NAME><parameter=…>…</parameter></function></tool_call>`),
so — unlike LFM2 — we keep the OpenAI-style structured `tool_calls`. Two gotchas:

  * `arguments` MUST be a **dict**, not a JSON string (a string makes the
    template silently drop the parameters).
  * Tools are passed in OpenAI `{type, function}` form (Qwen's native shape).

The template auto-prepends an (empty) `<think></think>` block; we leave it empty
and put the rationale as natural-language reasoning *before* the call, which the
template explicitly permits.
"""
from __future__ import annotations

from backend.agent.tools import TOOL_SCHEMAS


def format_example(turn: dict, include_rationale: bool = True) -> dict:
    return {
        "messages": [
            {"role": "system", "content": turn["system"]},
            {"role": "user", "content": turn["memory"]},
            {
                "role": "assistant",
                "content": turn["rationale"] if include_rationale else "",
                "tool_calls": [
                    {
                        "id": f"call_{turn['meta']['turn']}",
                        "type": "function",
                        "function": {"name": turn["name"], "arguments": turn["args"]},  # dict!
                    }
                ],
            },
        ],
        "meta": turn["meta"],
    }


def tool_schemas() -> list[dict]:
    """Qwen uses the OpenAI {type, function} schema verbatim."""
    return TOOL_SCHEMAS
