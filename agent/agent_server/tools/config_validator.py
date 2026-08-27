"""Tool: config_validator — AST static validation of Isaac Lab env config files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.config_validator import ConfigValidator

_validator = ConfigValidator()

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "config_validator",
        "description": (
            "AST-statically validate an Isaac Lab env config file (env.py). "
            "Use this when the user shares an env config file path or content and wants it checked. "
            "Performs 8 checks: syntax, imports, env_cfg_class, required_fields, "
            "observation_groups, action_terms, rewards_reference, common_mistakes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the env.py file to validate.",
                },
            },
            "required": ["file_path"],
        },
    },
}


def execute(file_path: str) -> dict[str, Any]:
    p = Path(file_path)
    if not p.is_absolute():
        # Try resolving relative to cwd first, then to skill root
        from core.config import SKILL_ROOT
        candidates = [Path.cwd() / file_path, SKILL_ROOT / file_path]
        for c in candidates:
            if c.exists():
                p = c
                break
        else:
            return {"error": f"file not found: {file_path}", "valid": False}
    report = _validator.validate_file(p)
    return {
        "valid": report["valid"],
        "errors": report["errors"],
        "warnings": report["warnings"],
        "checks_run": report["checks"],
        "file": str(p),
    }
