"""Optional Langfuse observability for the LLM pipeline.

Every helper here degrades to a no-op when Langfuse is either not installed or
not configured (no ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` in the
env). The rest of the app -- and the test suite -- therefore never depends on
Langfuse being present. Langfuse Cloud vs. a self-hosted instance is chosen
purely via env vars (``LANGFUSE_HOST``), so switching backends needs no code
change.

Instrumentation lives at the single provider boundary in ``llm.py``:
  - OpenAI / Gemini (OpenAI-compat) go through Langfuse's drop-in OpenAI class,
    which auto-captures model, messages, token usage, and latency.
  - Anthropic calls -- which don't have an equivalent drop-in -- are wrapped
    manually via :func:`anthropic_generation`.
The agent groups a whole analyze-then-decide cycle into one trace via
:func:`observe` (see ``agent.run_agent_cycle``); the per-turn generations above
nest under it automatically.

The ``langfuse`` package is imported lazily inside each function so importing
this module is cheap and safe even when Langfuse isn't installed.
"""
from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def is_enabled() -> bool:
    """True only when Langfuse is both configured (creds present) and installed."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return False
    try:
        import langfuse  # noqa: F401
    except ImportError:
        return False
    return True


def _client() -> Optional[Any]:
    if not is_enabled():
        return None
    from langfuse import get_client

    return get_client()


def import_openai_class() -> type:
    """Return the OpenAI client class, preferring Langfuse's drop-in when enabled.

    The drop-in is API-compatible with ``openai.OpenAI`` but reports each
    chat-completions call to Langfuse. Used for both the ``openai`` provider and
    ``gemini`` (which talks to the OpenAI-compat endpoint via the same client).
    """
    if is_enabled():
        from langfuse.openai import OpenAI

        return OpenAI
    from openai import OpenAI

    return OpenAI


def observe(*, name: Optional[str] = None) -> Callable[[F], F]:
    """``@observe()``-style decorator that is a no-op while Langfuse is disabled.

    The enabled-check happens at call time (not decoration time) so it doesn't
    matter whether ``load_dotenv()`` has run before the decorated module is
    imported.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_enabled():
                return func(*args, **kwargs)
            from langfuse import observe as _observe

            return _observe(name=name)(func)(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def update_trace(**kwargs: Any) -> None:
    """Attach name/metadata/input/output to the current trace, if any."""
    client = _client()
    if client is not None:
        client.update_current_trace(**kwargs)


@contextmanager
def anthropic_generation(
    name: str, model: str, input: Any, **metadata: Any
) -> Iterator[Optional[Any]]:
    """Wrap an Anthropic ``messages.create`` as a Langfuse generation.

    Yields the generation handle (or ``None`` when disabled) so the caller can
    record the output and token usage via :func:`record_anthropic_usage` once
    the response is in hand.
    """
    client = _client()
    if client is None:
        yield None
        return
    with client.start_as_current_generation(
        name=name, model=model, input=input, metadata=metadata or None
    ) as generation:
        yield generation


def record_anthropic_usage(generation: Optional[Any], response: Any, output: Any) -> None:
    """Record output + token usage on an Anthropic generation handle.

    Token counts feed Langfuse's cost computation (cost = usage x the model's
    price-table entry), so newer model IDs need custom prices defined in the
    Langfuse project for the dollar figures to be non-zero.
    """
    if generation is None:
        return
    usage = getattr(response, "usage", None)
    usage_details = None
    if usage is not None:
        usage_details = {
            "input": getattr(usage, "input_tokens", 0),
            "output": getattr(usage, "output_tokens", 0),
        }
    generation.update(output=output, usage_details=usage_details)


def flush() -> None:
    """Flush buffered events to Langfuse (call on shutdown so none are lost)."""
    client = _client()
    if client is not None:
        client.flush()
