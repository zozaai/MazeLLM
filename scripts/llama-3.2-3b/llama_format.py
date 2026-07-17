"""Llama-3.2-3B-Instruct-specific formatting: neutral expert turns -> chat examples.

Verified against `unsloth/Llama-3.2-3B-Instruct`'s actual `chat_template.jinja`
(fetched from the model repo, not assumed):

  * Tools are rendered with `t | tojson` verbatim, so the existing OpenAI-style
    `{"type": "function", "function": {...}}` schema in `backend/agent/tools.py`
    works unmodified — no reshaping needed.
  * An assistant `tool_calls` entry renders as a single-line JSON object,
    `{"name": ..., "parameters": ...}`, reading `tool_calls[0].function.name`
    and `.arguments`. `arguments` MUST be a **dict**, not a JSON string (same
    gotcha as Qwen) — a string renders as an escaped blob instead of an object.
  * The template only supports **one** tool call per assistant turn (it raises
    if `len(tool_calls) != 1`), which matches this project's one-tool-call-per-
    turn design already.
  * Gotcha specific to Llama (unlike Qwen): whenever `tool_calls` is present,
    the template **drops `message.content` entirely** — there is no rendered
    slot for a rationale/chain-of-thought next to the call. Passing content
    anyway would be silently discarded at render time, so we don't: the
    rationale is kept only in `meta` for dataset inspection/debugging, never
    in the trained message. `include_rationale` therefore doesn't affect the
    rendered prompt for this model — see the README for why the empty-content
    turn is the correct/deliberate choice here.
  * `tools_in_user_message` defaults to True, which injects the tool listing
    into the *first* user message rather than the system message. Since each
    training example is a single independent turn (system + one memory-grid
    user message + one assistant call), that first user message is always the
    memory grid — exactly matching how the real serving loop's conversation
    looks too (every later turn's tool result uses `role: "tool"`, never
    `role: "user"`, so the memory message is *always* the one and only "first
    user message" the template ever sees, turn after turn).
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
                "content": None,  # dropped by Llama's template when tool_calls is set — see module docstring
                "tool_calls": [
                    {
                        "id": f"call_{turn['meta']['turn']}",
                        "type": "function",
                        "function": {"name": turn["name"], "arguments": turn["args"]},  # dict!
                    }
                ],
            },
        ],
        "meta": {**turn["meta"], "rationale": turn["rationale"]} if include_rationale else turn["meta"],
    }


def tool_schemas() -> list[dict]:
    """Llama's template serializes each tool with `tojson` verbatim, so the
    OpenAI `{type, function}` schema already used by the real agent is used
    as-is — same shape passed to `tokenizer.apply_chat_template(..., tools=...)`
    at both training and inference time."""
    return TOOL_SCHEMAS
