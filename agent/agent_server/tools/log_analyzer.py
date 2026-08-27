"""Tool: log_analyzer — parse tensorboard metrics + detect anomalies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.log_analyzer import LogAnalyzer

_analyzer = LogAnalyzer()

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "log_analyzer",
        "description": (
            "Analyze training metrics to detect anomalies. Detects 8 patterns: "
            "sudden_drop, plateau, oscillation, spike, explosion, collapse, nan_or_inf, out_of_range. "
            "Use this when the user shares training metrics (reward, KL, entropy, grad_norm, etc.) "
            "and wants anomalies identified. Accepts either a path to a metrics.json file "
            "(format: {tag: [{step, value}, ...]}) or an inline metrics dict."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metrics_file_path": {
                    "type": "string",
                    "description": "Path to a JSON file containing metrics (tag -> [{step, value}]). Optional if metrics_inline is given.",
                },
                "metrics_inline": {
                    "type": "string",
                    "description": "Inline JSON string of metrics (same format as metrics_file_path). Use when the user pastes metrics in the chat.",
                },
            },
        },
    },
}


def execute(
    metrics_file_path: str | None = None,
    metrics_inline: str | None = None,
) -> dict[str, Any]:
    if not metrics_file_path and not metrics_inline:
        return {"error": "provide either metrics_file_path or metrics_inline"}
    if metrics_file_path:
        p = Path(metrics_file_path)
        if not p.is_absolute():
            from core.config import SKILL_ROOT
            for c in [Path.cwd() / metrics_file_path, SKILL_ROOT / metrics_file_path]:
                if c.exists():
                    p = c
                    break
        raw = json.loads(p.read_text(encoding="utf-8"))
    else:
        raw = json.loads(metrics_inline)
    # Convert {step, value} → (step, value) tuples
    metrics = {
        tag: [(int(p["step"]), float(p["value"])) for p in series]
        for tag, series in raw.items()
    }
    analysis = _analyzer.analyze(metrics)

    symptoms_dicts = [
        {
            "metric": s.metric,
            "pattern": s.pattern,
            "severity": s.severity,
            "evidence": s.evidence,
            "step_range": list(s.step_range) if s.step_range else None,
            "tag": s.tag,
            "value_summary": s.value_summary,
        }
        for s in analysis.symptoms
    ]

    out = {
        "metrics_analyzed": analysis.metrics_analyzed,
        "metrics_skipped": analysis.metrics_skipped,
        "warning_count": analysis.warning_count,
        "error_count": analysis.error_count,
        "symptoms": symptoms_dicts,
    }

    # Anomaly visualization (PNG attachment)
    if analysis.symptoms:
        try:
            from scripts.utils.plotting import plot_metrics_with_symptoms
            from core.artifacts import make_attachment
            from core.config import FILES_DIR
            import uuid
            png_path = plot_metrics_with_symptoms(
                metrics, symptoms_dicts,
                FILES_DIR / f"anomalies_{uuid.uuid4().hex[:8]}.png",
                title="Anomaly Detection",
            )
            with open(png_path, "rb") as f:
                out["anomaly_chart"] = make_attachment(
                    content=f.read(),
                    file_name="anomalies.png",
                    file_type="image",
                    mime_type="image/png",
                )
        except Exception as e:
            import logging
            logging.getLogger("agent_server.tools").warning(
                "anomaly plot failed: %s: %s", type(e).__name__, e,
            )

    return out
