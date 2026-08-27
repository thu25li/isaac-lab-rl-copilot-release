"""Tool: dr_advisor — recommend Domain Randomization parameters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.dr_advisor import DRAdvisor

_advisor = DRAdvisor()

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "dr_advisor",
        "description": (
            "Recommend Domain Randomization parameters for a given robot + task. "
            "Use this when the user wants to add DR (e.g., '我要做 sim-to-real' / '帮我配置 DR' / "
            "'训练出来的 policy 在真机不 work'). Returns essential/recommended/optional DR terms "
            "with parameter ranges. Backed by 9 DR terms + 5 robot profiles."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "robot_type": {
                    "type": "string",
                    "enum": [
                        "quadruped_small", "quadruped_medium", "quadruped_large",
                        "biped_small", "manipulator_arm",
                    ],
                    "description": "Robot class. quadruped_small=A1/Go1; quadruped_medium=Go2; quadruped_large=ANYmal B/Spot; biped_small=Cassie; manipulator_arm=UR5e/Franka.",
                },
                "task_type": {
                    "type": "string",
                    "enum": ["locomotion_velocity", "locomotion_rough_terrain", "manipulation_reach"],
                    "description": "Task category.",
                },
            },
            "required": ["robot_type", "task_type"],
        },
    },
}


def execute(robot_type: str, task_type: str) -> dict[str, Any]:
    rec = _advisor.recommend(robot_type=robot_type, task_type=task_type)
    essential = list(rec.essential_terms)
    recommended = list(rec.recommended_terms)
    optional = list(rec.optional_terms)
    param_ranges = _to_jsonable(rec.parameter_ranges)
    term_details = _to_jsonable(rec.term_details)

    out = {
        "robot_type": rec.robot_type,
        "task_type": rec.task_type,
        "robot_mass_kg": rec.robot_mass_kg,
        "essential_terms": essential,
        "recommended_terms": recommended,
        "optional_terms": optional,
        "parameter_ranges": param_ranges,
        "term_details": term_details,
    }

    # DR parameter ranges chart (PNG attachment)
    if essential or recommended or optional:
        try:
            from scripts.utils.plotting import plot_dr_ranges
            from core.artifacts import make_attachment
            from core.config import FILES_DIR
            import uuid
            png_path = plot_dr_ranges(
                essential, recommended, optional, param_ranges,
                FILES_DIR / f"dr_ranges_{uuid.uuid4().hex[:8]}.png",
                title="Domain Randomization — Parameter Ranges",
            )
            with open(png_path, "rb") as f:
                out["dr_chart"] = make_attachment(
                    content=f.read(),
                    file_name="dr_ranges.png",
                    file_type="image",
                    mime_type="image/png",
                )
        except Exception as e:
            import logging
            logging.getLogger("agent_server.tools").warning(
                "dr_ranges plot failed: %s: %s", type(e).__name__, e,
            )

    return out


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert tuples and Path objects to JSON-friendly types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
