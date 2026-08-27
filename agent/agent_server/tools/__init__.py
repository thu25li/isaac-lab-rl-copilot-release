"""Tool registry — all 7 tools the agent can call."""
from __future__ import annotations

from typing import Any, Callable

from tools import (
    reward_synthesizer,
    reward_validator,
    config_validator,
    log_analyzer,
    diagnosis_engine,
    dr_advisor,
    curriculum_designer,
)

TOOL_MODULES = [
    reward_synthesizer,
    reward_validator,
    config_validator,
    log_analyzer,
    diagnosis_engine,
    dr_advisor,
    curriculum_designer,
]


def all_tool_defs() -> list[dict]:
    """OpenAI-format tool definitions for the LLM."""
    return [m.TOOL_DEF for m in TOOL_MODULES]


def get_executor(name: str) -> Callable[..., dict]:
    """Look up a tool's executor by name. Raises KeyError if unknown."""
    for m in TOOL_MODULES:
        if m.TOOL_DEF["function"]["name"] == name:
            return m.execute
    raise KeyError(f"unknown tool: {name}")


def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name with the given args. Returns the structured result."""
    fn = get_executor(name)
    return fn(**args)
