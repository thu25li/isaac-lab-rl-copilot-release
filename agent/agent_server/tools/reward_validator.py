"""Tool: reward_validator — AST static validation of reward code."""
from __future__ import annotations

from typing import Any

from scripts.reward_validator import RewardValidator

_validator = RewardValidator()

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "reward_validator",
        "description": (
            "AST-statically validate Isaac Lab reward code (without executing it). "
            "Use this when the user has reward code (their own or generated) and wants it checked for bugs. "
            "Performs 7 checks: syntax, imports, RewardsCfg class structure, RewTerm calls, "
            "func references, weight ranges, common mistakes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The reward code source (Python source text) to validate.",
                },
            },
            "required": ["code"],
        },
    },
}


def execute(code: str) -> dict[str, Any]:
    report = _validator.validate_code(code)
    return {
        "valid": report["valid"],
        "errors": report["errors"],
        "warnings": report["warnings"],
        "checks_run": report["checks"],
        "summary": (
            f"valid={report['valid']}, "
            f"{len(report['errors'])} errors, "
            f"{len(report['warnings'])} warnings"
        ),
    }
