"""Tool: reward_synthesizer — generate Isaac Lab reward code from NL task description."""
from __future__ import annotations

from typing import Any

from scripts.reward_synthesizer import RewardSynthesizer
from core.artifacts import make_attachment

_synthesizer = RewardSynthesizer()

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "reward_synthesizer",
        "description": (
            "Generate Isaac Lab reward code from a natural-language task description. "
            "Use this when the user describes a robot task (e.g., 'train quadruped to walk at 1 m/s') "
            "and wants reward code. Returns the rendered reward.py source + selected patterns + validation report."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Natural-language task description, e.g. 'train quadruped to walk forward at 1 m/s, keep body level stable, minimize energy'",
                },
                "include_optional": {
                    "type": "boolean",
                    "description": "Whether to include optional reward terms. Default false.",
                    "default": False,
                },
            },
            "required": ["task_description"],
        },
    },
}


def execute(task_description: str, include_optional: bool = False) -> dict[str, Any]:
    """Run synthesizer. Returns a dict the LLM can read to compose its answer.

    The reward code is written to logs/files/<uuid>.py and surfaced as a
    清小搭-spec attachment (fileUrl + fileName, NOT inline content). The LLM
    still sees a `code` field so it can explain the code in its reply.

    Also renders reward_weights.png showing the weight composition of the
    synthesized reward, returned as an additional attachment.
    """
    result = _synthesizer.synthesize(
        task_description=task_description,
        include_optional=include_optional,
        validate=True,
    )
    code_attachment = make_attachment(
        content=result.code,
        file_name="reward_generated.py",
        file_type="text",
        mime_type="text/x-python",
    )

    # Weight composition chart
    weights_attachment = None
    try:
        from scripts.utils.plotting import plot_reward_weights
        from core.artifacts import make_attachment as _make
        from core.config import FILES_DIR
        png_path = plot_reward_weights(
            result.patterns, result.config,
            FILES_DIR / f"reward_weights_{code_attachment['fileName'].split('.')[0]}.png",
        )
        with open(png_path, "rb") as f:
            weights_attachment = _make(
                content=f.read(),
                file_name="reward_weights.png",
                file_type="image",
                mime_type="image/png",
            )
    except Exception as e:
        # Plotting must never break the tool — degrade to no chart
        import logging
        logging.getLogger("agent_server.tools").warning(
            "reward_weights plot failed: %s: %s", type(e).__name__, e,
        )

    out = {
        "task_type": result.task_type,
        "patterns": result.patterns,
        "config": result.config,
        "validation": result.validation,
        "code": result.code,
        "explanation": result.explanation,
        "code_artifact": code_attachment,
    }
    if weights_attachment is not None:
        out["weights_chart"] = weights_attachment
    return out
