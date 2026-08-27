"""Tool: curriculum_designer — recommend curriculum terms."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.curriculum_designer import CurriculumDesigner

_designer = CurriculumDesigner()

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "curriculum_designer",
        "description": (
            "Recommend curriculum terms for a given task. Use this when the user says "
            "'训练不收敛' / 'agent 学不会' / '任务太难' and wants curriculum design. "
            "Returns essential/recommended curriculum terms with default params and pitfalls. "
            "Backed by 7 curriculum terms (terrain_levels, command_curriculum, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": ["locomotion_velocity", "locomotion_rough_terrain", "manipulation_reach", "navigation"],
                    "description": "Task category.",
                },
            },
            "required": ["task_type"],
        },
    },
}


def execute(task_type: str) -> dict[str, Any]:
    rec = _designer.recommend(task_type=task_type)
    essential = list(rec.essential_terms)
    recommended = list(rec.recommended_terms)
    optional = list(rec.optional_terms)
    term_details = _to_jsonable(rec.term_details)

    out = {
        "task_type": rec.task_type,
        "essential_terms": essential,
        "recommended_terms": recommended,
        "optional_terms": optional,
        "term_details": term_details,
    }

    # Curriculum progression chart (PNG attachment)
    if essential or recommended or optional:
        try:
            from scripts.utils.plotting import plot_curriculum_progression
            from core.artifacts import make_attachment
            from core.config import FILES_DIR
            import uuid
            png_path = plot_curriculum_progression(
                essential, recommended, optional, term_details,
                FILES_DIR / f"curriculum_{uuid.uuid4().hex[:8]}.png",
                title="Curriculum — Conservative Growth",
            )
            with open(png_path, "rb") as f:
                out["curriculum_chart"] = make_attachment(
                    content=f.read(),
                    file_name="curriculum.png",
                    file_type="image",
                    mime_type="image/png",
                )
        except Exception as e:
            import logging
            logging.getLogger("agent_server.tools").warning(
                "curriculum plot failed: %s: %s", type(e).__name__, e,
            )

    return out


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# Need Path imported for _to_jsonable
from pathlib import Path
