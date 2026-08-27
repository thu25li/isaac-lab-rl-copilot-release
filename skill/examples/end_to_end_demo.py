"""End-to-end demonstration of Isaac Lab RL Co-pilot.

Walks through all 7 modules in sequence using a synthetic policy_collapse
scenario. Produces a complete troubleshooting report that a human (or LLM)
could hand to an Isaac Lab developer.

Run:
    python examples/end_to_end_demo.py

Output:
    examples/end_to_end_demo_outputs/report.md   (human-readable report)
    examples/end_to_end_demo_outputs/reward_generated.py  (synthesized reward)
    examples/end_to_end_demo_outputs/raw.json    (structured data for LLM)

No GPU, no Isaac Lab installation required — pure Python.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reward_synthesizer import RewardSynthesizer
from scripts.config_validator import ConfigValidator
from scripts.log_analyzer import LogAnalyzer
from scripts.diagnosis_engine import DiagnosisEngine
from scripts.dr_advisor import DRAdvisor
from scripts.curriculum_designer import CurriculumDesigner


TASK_DESCRIPTION = (
    "train quadruped to walk forward at 1 m/s, keep body level stable, minimize energy"
)
ROBOT_TYPE = "quadruped_medium"
TASK_TYPE = "locomotion_velocity"
ENV_FILE = ROOT / "examples" / "quadruped_locomotion" / "env.py"
SYNTHETIC_LOG = ROOT / "tests" / "test_data" / "synthetic_policy_collapse" / "metrics.json"


def section(title: str) -> str:
    return f"\n{'=' * 70}\n{title}\n{'=' * 70}\n"


def run_pipeline() -> dict:
    out: dict = {}
    ts_start = datetime.now().isoformat(timespec="seconds")

    # ---------- 1. Reward synthesis ----------
    print(section("1. Reward synthesis (NL -> code)"))
    synth = RewardSynthesizer()
    s_result = synth.synthesize(task_description=TASK_DESCRIPTION, validate=True)
    print(f"task: {TASK_DESCRIPTION}")
    print(f"detected task_type: {s_result.task_type}")
    print(f"patterns selected ({len(s_result.patterns)}): {s_result.patterns}")
    print(f"validation: valid={s_result.validation['valid']}, "
          f"errors={len(s_result.validation['errors'])}, "
          f"warnings={len(s_result.validation['warnings'])}")
    out["reward_synthesis"] = {
        "task_description": TASK_DESCRIPTION,
        "task_type": s_result.task_type,
        "patterns": s_result.patterns,
        "config": s_result.config,
        "validation": s_result.validation,
        "code": s_result.code,
    }

    # ---------- 2. Config validation ----------
    print(section("2. Env config validation (AST static analysis)"))
    validator = ConfigValidator()
    c_report = validator.validate_file(ENV_FILE)
    print(f"file: {ENV_FILE.relative_to(ROOT)}")
    print(f"valid={c_report['valid']}, "
          f"errors={len(c_report['errors'])}, "
          f"warnings={len(c_report['warnings'])}")
    print(f"checks run: {', '.join(c_report['checks'])}")
    out["config_validation"] = {
        "file": str(ENV_FILE.relative_to(ROOT)),
        "report": c_report,
    }

    # ---------- 3. Log analysis ----------
    print(section("3. Tensorboard log analysis (anomaly detection)"))
    raw_metrics = json.loads(SYNTHETIC_LOG.read_text(encoding="utf-8"))
    metrics = {tag: [(p["step"], p["value"]) for p in series]
               for tag, series in raw_metrics.items()}
    print(f"loaded {len(metrics)} metrics from synthetic log (policy_collapse scenario)")
    analyzer = LogAnalyzer()
    a_result = analyzer.analyze(metrics)
    print(f"symptoms detected: warning={a_result.warning_count}, error={a_result.error_count}")
    for s in a_result.symptoms:
        print(f"  [{s.severity.upper():7}] {s.metric:25} pattern={s.pattern:18} "
              f"step_range={s.step_range}")
    out["log_analysis"] = {
        "metrics_analyzed": a_result.metrics_analyzed,
        "metrics_skipped": a_result.metrics_skipped,
        "warning_count": a_result.warning_count,
        "error_count": a_result.error_count,
        "symptoms": [asdict(s) for s in a_result.symptoms],
    }

    # ---------- 4. Diagnosis ----------
    print(section("4. Diagnosis engine (symptoms -> failure modes -> fixes)"))
    engine = DiagnosisEngine()
    d_result = engine.diagnose(a_result.symptoms)
    print(f"top candidates ({len(d_result.candidates)} total):")
    for i, c in enumerate(d_result.top_candidates(n=3), 1):
        print(f"  {i}. [{c.confidence:.0%}] {c.failure_mode_id} "
              f"(matched {c.matched_count}/{d_result.total_symptoms} symptoms)")
        for m in c.matched:
            print(f"       - expected {m.expected_metric}/{m.expected_pattern} "
                  f"<-> saw {m.actual.metric}/{m.actual.pattern}")
        if c.fixes:
            top_fix = c.fixes[0]
            print(f"     top fix (priority {top_fix.get('priority', '?')}): "
                  f"{top_fix['action']}")
    out["diagnosis"] = {
        "total_symptoms": d_result.total_symptoms,
        "candidates": [asdict(c) for c in d_result.candidates[:3]],
        "unmatched_symptoms": [asdict(s) for s in d_result.unmatched_symptoms],
    }

    # ---------- 5. DR recommendation ----------
    print(section("5. Domain randomization advisor"))
    dr = DRAdvisor()
    dr_rec = dr.recommend(robot_type=ROBOT_TYPE, task_type=TASK_TYPE)
    print(f"robot: {ROBOT_TYPE} (~{dr_rec.robot_mass_kg} kg)")
    print(f"essential terms ({len(dr_rec.essential_terms)}): {dr_rec.essential_terms}")
    print(f"recommended terms ({len(dr_rec.recommended_terms)}): {dr_rec.recommended_terms}")
    out["dr_recommendation"] = {
        "robot_type": ROBOT_TYPE,
        "task_type": TASK_TYPE,
        "essential_terms": dr_rec.essential_terms,
        "recommended_terms": dr_rec.recommended_terms,
        "parameter_ranges": dr_rec.parameter_ranges,
        "term_details": dr_rec.term_details,
    }

    # ---------- 6. Curriculum recommendation ----------
    print(section("6. Curriculum designer"))
    cd = CurriculumDesigner()
    cd_rec = cd.recommend(task_type=TASK_TYPE)
    print(f"essential terms ({len(cd_rec.essential_terms)}): {cd_rec.essential_terms}")
    print(f"recommended terms ({len(cd_rec.recommended_terms)}): {cd_rec.recommended_terms}")
    out["curriculum_recommendation"] = {
        "task_type": TASK_TYPE,
        "essential_terms": cd_rec.essential_terms,
        "recommended_terms": cd_rec.recommended_terms,
        "term_details": cd_rec.term_details,
    }

    out["_meta"] = {
        "timestamp": ts_start,
        "duration_below": "see logs above",
        "skill_version": "0.1.0",
        "modules_run": [
            "reward_synthesizer", "config_validator", "log_analyzer",
            "diagnosis_engine", "dr_advisor", "curriculum_designer",
        ],
    }
    return out


def write_report(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. raw JSON
    (out_dir / "raw.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # 2. synthesized reward code
    (out_dir / "reward_generated.py").write_text(
        data["reward_synthesis"]["code"], encoding="utf-8",
    )

    # 3. anomaly-detection visualization (matplotlib PNG)
    viz_state = {}
    try:
        from scripts.utils.plotting import (
            plot_metrics_with_symptoms, plot_reward_weights,
            plot_diagnosis_confidence, plot_dr_ranges,
            plot_curriculum_progression,
        )
        # Reload metrics from the source JSON (write_report is a separate scope)
        viz_raw = json.loads(SYNTHETIC_LOG.read_text(encoding="utf-8"))
        viz_metrics = {
            tag: [(p["step"], p["value"]) for p in series]
            for tag, series in viz_raw.items()
        }
        symptom_dicts = [asdict(s) if hasattr(s, "__dataclass_fields__") else s
                         for s in data["log_analysis"]["symptoms"]]
        anomalies_path = plot_metrics_with_symptoms(
            viz_metrics, symptom_dicts, out_dir / "anomalies.png",
            title="synthetic_policy_collapse — Anomaly Detection",
        )
        viz_state["anomalies_png"] = str(anomalies_path.relative_to(out_dir))

        # Reward weight composition chart
        weights_path = plot_reward_weights(
            data["reward_synthesis"]["patterns"],
            data["reward_synthesis"]["config"],
            out_dir / "reward_weights.png",
            title="Synthesized Reward — Weight Composition",
        )
        viz_state["reward_weights_png"] = str(weights_path.relative_to(out_dir))

        # Diagnosis confidence chart
        diag_cands = data["diagnosis"]["candidates"]
        diag_path = plot_diagnosis_confidence(
            diag_cands, out_dir / "diagnosis.png",
            title="Failure Mode Confidence",
        )
        viz_state["diagnosis_png"] = str(diag_path.relative_to(out_dir))

        # DR parameter ranges chart
        dr = data["dr_recommendation"]
        dr_path = plot_dr_ranges(
            dr.get("essential_terms", []),
            dr.get("recommended_terms", []),
            dr.get("optional_terms", []),
            dr["parameter_ranges"], out_dir / "dr_ranges.png",
            title="Domain Randomization — Parameter Ranges",
        )
        viz_state["dr_ranges_png"] = str(dr_path.relative_to(out_dir))

        # Curriculum progression chart
        curr = data["curriculum_recommendation"]
        curr_path = plot_curriculum_progression(
            curr.get("essential_terms", []),
            curr.get("recommended_terms", []),
            curr.get("optional_terms", []),
            curr["term_details"], out_dir / "curriculum.png",
            title="Curriculum — Conservative Growth",
        )
        viz_state["curriculum_png"] = str(curr_path.relative_to(out_dir))
    except Exception as e:
        # Plotting is an enhancement; never let it crash the demo
        print(f"  (plotting skipped: {type(e).__name__}: {e})")
    data["_viz"] = viz_state or None

    # 4. human-readable markdown report
    r = data
    md = []
    md.append("# Isaac Lab RL Co-pilot — End-to-End Demo Report\n")
    md.append(f"Generated: {r['_meta']['timestamp']}\n")
    md.append(f"Skill version: {r['_meta']['skill_version']}\n\n")

    md.append("## Pipeline summary\n")
    md.append("| Step | Module | Result |\n|------|--------|--------|\n")
    md.append(f"| 1 | reward_synthesizer | {len(r['reward_synthesis']['patterns'])} patterns selected, "
              f"validation valid={r['reward_synthesis']['validation']['valid']} |\n")
    md.append(f"| 2 | config_validator | "
              f"errors={len(r['config_validation']['report']['errors'])}, "
              f"warnings={len(r['config_validation']['report']['warnings'])} |\n")
    md.append(f"| 3 | log_analyzer | "
              f"{r['log_analysis']['warning_count']} warnings, "
              f"{r['log_analysis']['error_count']} errors |\n")
    md.append(f"| 4 | diagnosis_engine | top: "
              f"{r['diagnosis']['candidates'][0]['failure_mode_id']} "
              f"@ {r['diagnosis']['candidates'][0]['confidence']:.0%} |\n")
    md.append(f"| 5 | dr_advisor | "
              f"{len(r['dr_recommendation']['essential_terms'])} essential DR terms |\n")
    md.append(f"| 6 | curriculum_designer | "
              f"{len(r['curriculum_recommendation']['essential_terms'])} essential curriculum terms |\n\n")

    # Embed anomaly visualization if generated
    if r.get("_viz") and r["_viz"].get("anomalies_png"):
        md.append("## Anomaly visualization\n\n")
        md.append(f"![anomaly detection]({r['_viz']['anomalies_png']})\n\n")
        md.append("Red shading = error-severity symptoms, yellow = warning. "
                  "Marked points show where each detector fired.\n\n")

    if r.get("_viz") and r["_viz"].get("reward_weights_png"):
        md.append("## Reward weight composition\n\n")
        md.append(f"![reward weights]({r['_viz']['reward_weights_png']})\n\n")
        md.append("Each bar = one reward term. Length = |weight|, color = category. "
                  "Positive (right) = reward shaping, negative (left) = penalty. "
                  "Tracking terms dominate (weight ~1.0), penalties sit at smaller magnitudes.\n\n")

    if r.get("_viz") and r["_viz"].get("diagnosis_png"):
        md.append("## Diagnosis confidence\n\n")
        md.append(f"![diagnosis]({r['_viz']['diagnosis_png']})\n\n")
        md.append("Each bar = a candidate failure mode, length = confidence %. "
                  "Star marks the top-1 diagnosis. Color shows symptom-match ratio "
                  "(green ≥60%, yellow 30-60%, gray <30%).\n\n")

    if r.get("_viz") and r["_viz"].get("dr_ranges_png"):
        md.append("## DR parameter ranges\n\n")
        md.append(f"![dr ranges]({r['_viz']['dr_ranges_png']})\n\n")
        md.append("Each bar = one Domain Randomization term's parameter range "
                  "(min → max). Color = tier: green essential, orange recommended, "
                  "purple optional.\n\n")

    if r.get("_viz") and r["_viz"].get("curriculum_png"):
        md.append("## Curriculum progression\n\n")
        md.append(f"![curriculum]({r['_viz']['curriculum_png']})\n\n")
        md.append("Trumpet shape shows how each curriculum term's parameter range "
                  "expands from the initial level (left) to the graduate level (right). "
                  "Conservative growth: start narrow, widen as the agent learns.\n\n")

    md.append("## Diagnosis (top 3 candidates)\n\n")
    for i, c in enumerate(r["diagnosis"]["candidates"], 1):
        md.append(f"### {i}. {c['failure_mode_id']} (confidence: {c['confidence']:.0%})\n")
        md.append(f"- matched: {len(c['matched'])}/{r['diagnosis']['total_symptoms']} symptoms\n")
        md.append("- matched symptoms:\n")
        for m in c["matched"]:
            md.append(f"  - expected `{m['expected_metric']}/{m['expected_pattern']}`, "
                      f"saw `{m['actual']['metric']}/{m['actual']['pattern']}`\n")
        md.append("- top fixes:\n")
        for f in c["fixes"][:3]:
            md.append(f"  {f.get('priority', '?')}. {f['action']}\n")
        md.append(f"- verification: {c['verification']}\n\n")

    md.append("## Recommended DR parameters (quadruped_medium)\n\n")
    md.append("| Term | Parameters |\n|------|------------|\n")
    for term_id, params in r["dr_recommendation"]["parameter_ranges"].items():
        md.append(f"| {term_id} | `{params}` |\n")
    md.append("\n")

    md.append("## Recommended curriculum\n\n")
    for term_id in r["curriculum_recommendation"]["essential_terms"] + \
            r["curriculum_recommendation"]["recommended_terms"]:
        detail = r["curriculum_recommendation"]["term_details"].get(term_id, {})
        md.append(f"### {term_id}\n")
        md.append(f"- func: `{detail.get('isaac_lab_func', 'unknown')}`\n")
        md.append(f"- purpose: {detail.get('purpose', 'n/a')}\n")
        md.append(f"- default params: `{detail.get('param_defaults', {})}`\n\n")

    md.append("## Synthesized reward\n\n")
    md.append("Full code: see `reward_generated.py` in this directory.\n\n")
    md.append("Pattern composition:\n")
    for p in r["reward_synthesis"]["patterns"]:
        w = r["reward_synthesis"]["config"].get(p, {}).get("weight", "?")
        md.append(f"- `{p}` (weight: {w})\n")

    (out_dir / "report.md").write_text("".join(md), encoding="utf-8")
    print(section(f"Outputs written to {out_dir}"))
    print(f"  - {out_dir / 'report.md'}    (human-readable)")
    print(f"  - {out_dir / 'reward_generated.py'}  (synthesized reward)")
    print(f"  - {out_dir / 'raw.json'}    (structured data for LLM)")


if __name__ == "__main__":
    data = run_pipeline()
    out_dir = Path(__file__).parent / "end_to_end_demo_outputs"
    write_report(data, out_dir)
