"""Thin wrapper over any OpenAI-compatible chat.completions endpoint.

Because vLLM and Ollama both serve an OpenAI-compatible API, swapping
providers is purely a matter of changing LLM_BASE_URL / LLM_API_KEY /
LLM_MODEL in .env — this class never needs to change.

Uses AsyncOpenAI, not the sync client: this is called from inside an async
WebSocket route, and a blocking HTTP call there would freeze the entire
event loop (no other connections, sends, or timeouts could proceed) for the
duration of every LLM turn.
"""
from __future__ import annotations

from openai import AsyncOpenAI

from ..config import Settings

REQUEST_TIMEOUT_SECONDS = 60.0


class LLMClient:
    def __init__(self, settings: Settings):
        default_headers = {}
        if settings.llm_site_url:
            default_headers["HTTP-Referer"] = settings.llm_site_url
        if settings.llm_site_name:
            default_headers["X-Title"] = settings.llm_site_name

        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "not-needed",
            timeout=REQUEST_TIMEOUT_SECONDS,
            default_headers=default_headers or None,
        )
        self._model = settings.llm_model

    async def chat(self, messages: list[dict], tools: list[dict]):
        """Send one chat turn with tool definitions attached.

        TODO: decide on streaming vs non-streaming, retry/backoff policy,
        and how tool_choice is set (auto vs forced) per step.
        """
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
        )
