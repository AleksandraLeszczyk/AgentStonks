"""Unified access to Gemini, OpenAI, and Anthropic chat models.

Gemini and OpenAI both speak the OpenAI chat-completions API (Gemini via its
OpenAI-compat endpoint), so they share a single `openai.OpenAI` client.
Anthropic's Messages API differs (system prompt is a separate field, content
is a list of typed blocks instead of `tool_calls`), so `AnthropicChatClient`
adapts it to expose the same `.chat.completions.create(...)` shape that
`agent.py`'s tool-calling loop and the tests' `FakeClient` already use —
callers don't need to know which provider they're talking to.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Optional

from pydantic import BaseModel

from . import observability as obs

PROVIDERS: tuple[str, ...] = ("gemini", "openai", "anthropic")

DEFAULT_AGENT_MODELS: dict[str, str] = {
    "gemini": "gemini-3.5-flash",
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-sonnet-4-6",
}

DEFAULT_NEWS_MODELS: dict[str, str] = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-5.4-nano",
    "anthropic": "claude-haiku-4-5-20251001",
}

ENV_KEYS: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _openai_tools_to_anthropic(tools: Optional[list[dict]]) -> list[dict]:
    if not tools:
        return []
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}},
        }
        for t in tools
    ]


def _openai_messages_to_anthropic(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
    system: Optional[str] = None
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system = m["content"]
        elif role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            blocks: list[dict] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls") or []:
                args = tc["function"]["arguments"]
                try:
                    parsed = json.loads(args) if isinstance(args, str) else (args or {})
                except json.JSONDecodeError:
                    parsed = {}
                blocks.append(
                    {"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"], "input": parsed}
                )
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        elif role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                    ],
                }
            )
    return system, out


class _AnthropicCompletions:
    def __init__(self, anthropic_client: Any) -> None:
        self._client = anthropic_client

    def create(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
        max_tokens: int = 4096,
    ) -> SimpleNamespace:
        system, anthropic_messages = _openai_messages_to_anthropic(messages)
        kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": anthropic_messages}
        if system:
            kwargs["system"] = system
        anthropic_tools = _openai_tools_to_anthropic(tools)
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        with obs.anthropic_generation(
            name="anthropic-messages", model=model, input=anthropic_messages
        ) as generation:
            response = self._client.messages.create(**kwargs)

            content_text: Optional[str] = None
            tool_calls: list[SimpleNamespace] = []
            for block in response.content:
                if block.type == "text":
                    content_text = (content_text or "") + block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        SimpleNamespace(
                            id=block.id,
                            function=SimpleNamespace(name=block.name, arguments=json.dumps(block.input)),
                        )
                    )

            obs.record_anthropic_usage(
                generation,
                response,
                {"content": content_text, "tool_calls": [tc.function.name for tc in tool_calls]},
            )

        message = SimpleNamespace(content=content_text, tool_calls=tool_calls or None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class AnthropicChatClient:
    """Adapts the Anthropic SDK to the OpenAI `client.chat.completions.create(...)` shape."""

    def __init__(self, api_key: str) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.chat = SimpleNamespace(completions=_AnthropicCompletions(self._client))


def get_agent_client(provider: str, api_key: str) -> Any:
    """Return a chat client exposing `.chat.completions.create(...)` for the given provider."""
    if provider == "gemini":
        OpenAI = obs.import_openai_class()
        return OpenAI(api_key=api_key, base_url=_GEMINI_BASE_URL)
    if provider == "openai":
        OpenAI = obs.import_openai_class()
        return OpenAI(api_key=api_key)
    if provider == "anthropic":
        return AnthropicChatClient(api_key)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def parse_structured(
    provider: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    response_model: type[BaseModel],
) -> Optional[BaseModel]:
    """Get a structured (schema-validated) response from any of the three providers."""
    if provider in ("gemini", "openai"):
        OpenAI = obs.import_openai_class()
        base_url = _GEMINI_BASE_URL if provider == "gemini" else None
        client = OpenAI(api_key=api_key, base_url=base_url)
        completion = client.chat.completions.parse(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=response_model,
        )
        return completion.choices[0].message.parsed

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        tool_name = "emit_result"
        with obs.anthropic_generation(
            name="anthropic-structured", model=model, input=user
        ) as generation:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[
                    {
                        "name": tool_name,
                        "description": "Emit the structured result.",
                        "input_schema": response_model.model_json_schema(),
                    }
                ],
                tool_choice={"type": "tool", "name": tool_name},
            )
            result = None
            for block in response.content:
                if block.type == "tool_use":
                    result = response_model.model_validate(block.input)
                    break
            obs.record_anthropic_usage(
                generation, response, result.model_dump() if result is not None else None
            )
            return result

    raise ValueError(f"Unknown LLM provider: {provider!r}")
