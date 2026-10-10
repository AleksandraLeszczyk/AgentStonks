"""Unified access to Gemini, OpenAI, and Anthropic chat models.

Gemini and OpenAI both speak the OpenAI chat-completions API (Gemini via its
OpenAI-compat endpoint), so they share a single `openai.OpenAI` client.
Anthropic's Messages API differs (system prompt is a separate field, content
is a list of typed blocks instead of `tool_calls`), so `AnthropicChatClient`
adapts it to expose the same `.chat.completions.create(...)` shape that
`agent.py`'s tool-calling loop and the tests' `FakeClient` already use —
callers don't need to know which provider they're talking to.

Some OpenAI models (gpt-6-astra, gpt-6.1-sol) take function tools only on
`/v1/responses`, so `_OpenAIResponsesCompletions` adapts that endpoint to the
same shape and their tool calls are routed there.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, Optional

from pydantic import BaseModel

from . import observability as obs

logger = logging.getLogger(__name__)

PROVIDERS: tuple[str, ...] = ("gemini", "openai", "anthropic")

DEFAULT_AGENT_MODELS: dict[str, str] = {
    "gemini": "gemini-3.8-flash",
    "openai": "gpt-6-luna",
    "anthropic": "claude-sonnet-5-5",
}

DEFAULT_NEWS_MODELS: dict[str, str] = {
    "gemini": "gemini-3.8-flash",
    "openai": "gpt-6-luna",
    "anthropic": "claude-sonnet-5-5",
}

# Curated per-provider model menus for the UI's model pickers. Ordered roughly
# most- to least-capable; the UI defaults the selection to the relevant
# DEFAULT_*_MODELS entry. Keep every model referenced by a DEFAULT_* dict listed
# here so those defaults are always selectable (see `models_for`).
SUPPORTED_MODELS: dict[str, tuple[str, ...]] = {
    "gemini": ("gemini-3.1-pro-preview", "gemini-3.8-flash"),
    "openai": ("gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna"),
    "anthropic": (
        "claude-fable-5-1",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
    ),
}


def models_for(provider: str, *, default: Optional[str] = None) -> list[str]:
    """Return the selectable models for `provider`, guaranteeing `default` is present.

    `default` (e.g. the agent/news/premarket default for this provider) is moved
    to the front if listed and prepended if the catalog somehow omits it, so a
    picker seeded with it never lands on a value outside its own options.
    """
    models = list(SUPPORTED_MODELS.get(provider, ()))
    if default:
        if default in models:
            models.remove(default)
        models.insert(0, default)
    return models

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


# Models that reject function tools on /v1/chat/completions at every
# reasoning_effort (they refuse "none", and any other value disallows tools
# there). The known ones go straight to /v1/responses: probing them costs a
# wasted 400 per process (and per Streamlit module reload), which Langfuse logs
# as a WARNING. Any other model that refuses is learned on its first rejection.
_KNOWN_RESPONSES_API_MODELS = frozenset({"gpt-6-astra", "gpt-6.1-sol"})
_RESPONSES_API_MODELS: set[str] = set()


def _chat_tools_to_responses(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "parameters": t["function"].get("parameters") or {"type": "object", "properties": {}},
            # Chat-completions function tools are non-strict; strict would
            # demand every property be required, which ours aren't.
            "strict": False,
        }
        for t in tools
    ]


def _chat_messages_to_responses(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
    instructions: Optional[str] = None
    items: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            instructions = m["content"]
        elif role == "user":
            items.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            if m.get("content"):
                items.append({"role": "assistant", "content": m["content"]})
            # Replayed without the item id or the reasoning item that preceded
            # it -- the endpoint accepts a bare call_id, and with store=False
            # there is no server-side copy to point at anyway.
            for tc in m.get("tool_calls") or []:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                )
        elif role == "tool":
            items.append(
                {"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]}
            )
    return instructions, items


class _OpenAIResponsesCompletions:
    """Serves a chat-completions tool call from `/v1/responses`, in the same shape."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def create(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
    ) -> SimpleNamespace:
        instructions, items = _chat_messages_to_responses(messages)
        request: dict[str, Any] = {"model": model, "input": items, "store": False}
        if instructions:
            request["instructions"] = instructions
        if tools:
            request["tools"] = _chat_tools_to_responses(tools)
            request["tool_choice"] = tool_choice
        response = self._client.responses.create(**request)

        tool_calls = [
            SimpleNamespace(
                id=item.call_id,
                function=SimpleNamespace(name=item.name, arguments=item.arguments),
            )
            for item in response.output
            if item.type == "function_call"
        ]
        message = SimpleNamespace(content=response.output_text or None, tool_calls=tool_calls or None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _rejects_chat_tools(exc: Exception) -> bool:
    """True for the 400 a model returns when chat-completions can't carry its tools."""
    return getattr(exc, "status_code", None) == 400 and getattr(exc, "param", None) == "reasoning_effort"


class _OpenAIToolCompletions:
    """Wraps OpenAI `chat.completions` to make function tools work with reasoning models.

    Newer OpenAI models (e.g. gpt-5.6, gpt-6-luna) apply a server-side default
    reasoning effort, and `/v1/chat/completions` rejects function tools whenever
    reasoning is active with: "Function tools with reasoning_effort are not
    supported ... set reasoning_effort to 'none'." We forward that documented
    workaround by defaulting `reasoning_effort="none"` on any tool-carrying call
    (unless the caller set it explicitly). Models that refuse "none" as well
    (gpt-6-astra, gpt-6.1-sol, routed up front; others after their first 400)
    are switched to `/v1/responses`, where their tools work at the server's
    default effort. Tool-less calls are passed through untouched.
    """

    def __init__(self, completions: Any, responses: _OpenAIResponsesCompletions) -> None:
        self._completions = completions
        self._responses = responses

    def create(self, *args: Any, **kwargs: Any) -> Any:
        if not kwargs.get("tools") or "reasoning_effort" in kwargs:
            return self._completions.create(*args, **kwargs)
        model = kwargs.get("model")
        if model in _KNOWN_RESPONSES_API_MODELS or model in _RESPONSES_API_MODELS:
            return self._responses.create(**kwargs)
        try:
            return self._completions.create(*args, **kwargs, reasoning_effort="none")
        except Exception as exc:
            if not _rejects_chat_tools(exc):
                raise
        logger.info("%s takes function tools only on /v1/responses; routing its tool calls there", model)
        _RESPONSES_API_MODELS.add(model)
        return self._responses.create(**kwargs)

    def __getattr__(self, name: str) -> Any:  # e.g. `.parse`, delegated unchanged
        return getattr(self._completions, name)


class _OpenAIToolChatClient:
    """Proxies an OpenAI client, swapping in reasoning-safe tool completions."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.chat = SimpleNamespace(
            completions=_OpenAIToolCompletions(client.chat.completions, _OpenAIResponsesCompletions(client))
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def get_agent_client(provider: str, api_key: str) -> Any:
    """Return a chat client exposing `.chat.completions.create(...)` for the given provider."""
    if provider == "gemini":
        OpenAI = obs.import_openai_class()
        return OpenAI(api_key=api_key, base_url=_GEMINI_BASE_URL)
    if provider == "openai":
        OpenAI = obs.import_openai_class()
        return _OpenAIToolChatClient(OpenAI(api_key=api_key))
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
        with obs.anthropic_generation(
            name="anthropic-structured", model=model, input=user
        ) as generation:
            # Structured outputs, not a forced tool call: Sonnet 5.5, Opus 5.5
            # and Fable 5.1 reject tool_choice "tool"/"any" with a 400.
            # transform_schema drops what the API can't enforce (ge/le) into the
            # descriptions; model_validate_json below still enforces it. The
            # headroom in max_tokens is for thinking, which those models run.
            response = client.messages.create(
                model=model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "format": {"type": "json_schema", "schema": anthropic.transform_schema(response_model)}
                },
            )
            result = None
            # A refusal or a cut-off answer is not schema-valid JSON; report it
            # as no result, as callers already expect.
            if response.stop_reason not in ("refusal", "max_tokens"):
                for block in response.content:
                    if block.type == "text":
                        result = response_model.model_validate_json(block.text)
                        break
            obs.record_anthropic_usage(
                generation, response, result.model_dump() if result is not None else None
            )
            return result

    raise ValueError(f"Unknown LLM provider: {provider!r}")
