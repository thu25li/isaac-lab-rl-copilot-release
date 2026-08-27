"""Tool: diagnosis_engine — symptoms → ranked failure mode candidates."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from scripts.diagnosis_engine import DiagnosisEngine
from scripts.log_analyzer import Symptom

_engine = DiagnosisEngine()

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "diagnosis_engine",
        "description": (
            "Diagnose training failure based on symptoms. Returns ranked failure mode candidates "
            "with confidence scores, root causes, and prioritized fixes. Use this after log_analyzer "
            "has detected symptoms — feed those symptoms here to get a diagnosis. "
            "Backed by 17 failure modes across 9 categories (reward_design, training_instability, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symptoms": {
                    "type": "array",
                    "description": "Symptoms detected by log_analyzer (pass the symptoms field directly).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "metric": {"type": "string"},
                            "pattern": {"type": "string"},
                            "severity": {"type": "string", "enum": ["warning", "error"]},
                            "evidence": {"type": "string"},
                            "step_range": {"type": "array", "items": {"type": "number"}},
                            "tag": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["symptoms"],
        },
    },
}


def execute(symptoms: list[dict[str, Any]]) -> dict[str, Any]:
    sym_objs: list[Symptom] = []
    for s in symptoms:
        sr = s.get("step_range")
        sym_objs.append(Symptom(
            metric=s["metric"],
            tag=s.get("tag", s["metric"]),
            pattern=s["pattern"],
            severity=s.get("severity", "warning"),
            evidence=s.get("evidence", ""),
            step_range=tuple(sr) if sr else (0, 0),
        ))
    result = _engine.diagnose(sym_objs)

    candidates = [
        {
            "failure_mode_id": c.failure_mode_id,
            "confidence": c.confidence,
            "matched_count": len(c.matched),
            "matched": [
                {
                    "expected_metric": m.expected_metric,
                    "expected_pattern": m.expected_pattern,
                    "actual_metric": m.actual.metric,
                    "actual_pattern": m.actual.pattern,
                }
                for m in c.matched
            ],
            "root_causes": c.root_causes,
            "fixes": c.fixes,
            "verification": c.verification,
            "total_expected": c.total_expected,
        }
        for c in result.top_candidates(n=3)
    ]

    out = {
        "total_symptoms": result.total_symptoms,
        "top_candidates": candidates,
        "unmatched_symptoms": [
            {"metric": s.metric, "pattern": s.pattern}
            for s in result.unmatched_symptoms
        ],
    }

    # Diagnosis confidence chart (PNG attachment)
    if candidates:
        try:
            from scripts.utils.plotting import plot_diagnosis_confidence
            from core.artifacts import make_attachment
            from core.config import FILES_DIR
            import uuid
            png_path = plot_diagnosis_confidence(
                candidates,
                FILES_DIR / f"diagnosis_{uuid.uuid4().hex[:8]}.png",
                title="Failure Mode Confidence",
            )
            with open(png_path, "rb") as f:
                out["diagnosis_chart"] = make_attachment(
                    content=f.read(),
                    file_name="diagnosis.png",
                    file_type="image",
                    mime_type="image/png",
                )
        except Exception as e:
            import logging
            logging.getLogger("agent_server.tools").warning(
                "diagnosis plot failed: %s: %s", type(e).__name__, e,
            )

    return out
