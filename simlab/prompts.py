"""Editable per-personality prompt overrides.

The SimLab Agents tab lets you edit a personality's base system prompt and
test the edited version in simulation. Overrides live as plain text files
(``data/simlab/prompts/{personality}.txt``) -- the file is the single source
of truth, easy to diff and version. The live app never reads them; only a
simulation launched from SimLab passes the override into ``run_agent_cycle``.

Langfuse prompt management could hold these instead (versioning + UI editing),
but a local file keeps the loop offline-capable and dependency-free; syncing
edited prompts to Langfuse is deliberately left as an explicit user action,
not an implicit side effect of saving.
"""
from __future__ import annotations

from pathlib import Path

from agent_stonks.agent import AGENT_PERSONALITIES

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "simlab" / "prompts"


def default_prompt(personality: str) -> str:
    return AGENT_PERSONALITIES[personality]["system_prompt"]


def _override_path(personality: str) -> Path:
    return PROMPTS_DIR / f"{personality}.txt"


def has_override(personality: str) -> bool:
    return _override_path(personality).exists()


def get_prompt(personality: str) -> str:
    """The effective base prompt: the saved override, else the built-in."""
    path = _override_path(personality)
    if path.exists():
        text = path.read_text()
        if text.strip():
            return text
    return default_prompt(personality)


def get_override(personality: str) -> "str | None":
    """The saved override text, or None when the built-in prompt is in use."""
    return get_prompt(personality) if has_override(personality) else None


def save_override(personality: str, text: str) -> None:
    if personality not in AGENT_PERSONALITIES:
        raise KeyError(f"unknown personality {personality!r}")
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    _override_path(personality).write_text(text)


def reset_override(personality: str) -> None:
    path = _override_path(personality)
    if path.exists():
        path.unlink()
