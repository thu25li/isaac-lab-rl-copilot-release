#!/usr/bin/env python3
"""Diagnosis engine — maps training symptoms to failure modes and fixes.

This is the reasoning layer of module 2. It takes symptoms (from
LogAnalyzer) and matches them against the failure mode database
(resources/failure_modes.json), producing ranked candidate diagnoses
with root causes and priority-ordered fixes.

Matching strategy:
- Each failure mode declares a set of expected symptoms (metric + pattern).
- For each mode, count how many expected symptoms are present in the
  actual symptom list. Confidence = matched / total, weighted by
  severity and reinforced when multiple symptoms of the same mode fire.
- Pattern matching is fuzzy: "not_increasing_or_decreasing" matches
  "not_increasing"; "spike_or_collapse" matches both "spike" and "collapse".

Design principle (from failure_modes.json note):
    Different tasks have different metric scales, so we rely on the
    LogAnalyzer's *relative* anomaly detection (z-scores, ratios) rather
    than absolute thresholds. The diagnosis engine then matches patterns
    qualitatively — if a symptom was detected, it's evidence; the engine
    doesn't second-guess the threshold.

Usage (library):
    from scripts.diagnosis_engine import DiagnosisEngine
    from scripts.log_analyzer import LogAnalyzer
    engine = DiagnosisEngine()
    result = engine.diagnose(log_analyzer_result.symptoms)
    for c in result.top_candidates(3):
        print(c.failure_mode_name, c.confidence)

Usage (CLI):
    python scripts/diagnosis_engine.py --symptoms symptoms.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow direct script execution
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from scripts.log_analyzer import Symptom


# ------------------------------------------------------------------
# Pattern compatibility — fuzzy matching between expected and actual patterns.
# Maps an expected pattern (from failure_modes.json) to a set of actual
# patterns (from LogAnalyzer) that should count as a match.
#
# Numerical instability patterns (spike, explosion, nan_or_inf) are treated
# as partially interchangeable: a spike often precedes an explosion, which
# often precedes NaN. If any one is detected, it's evidence for failure
# modes that expect any of the others.
# ------------------------------------------------------------------
_NUMERICAL_INSTABILITY = ["spike", "explosion", "nan_or_inf"]

PATTERN_COMPATIBILITY: Dict[str, List[str]] = {
    # reward_hacking
    "increasing": [],  # no direct detector (we detect the absence of increase)
    "abnormal": ["sudden_drop", "oscillation", "explosion"],
    "not_increasing_or_decreasing": ["not_increasing", "plateau"],
    "dominant": [],  # requires per-term contribution data, not yet supported
    # policy_collapse
    "sudden_drop": ["sudden_drop"],
    "spike_or_collapse": ["spike", "collapse"],
    # gradient_explosion — spike, explosion, nan_or_inf all indicate numerical blowup
    "spike": _NUMERICAL_INSTABILITY,
    "explosion": _NUMERICAL_INSTABILITY,
    "nan_or_inf": _NUMERICAL_INSTABILITY,
    # entropy_collapse
    "collapse": ["collapse"],
    # value_function_divergence
    "divergence": ["explosion", "not_increasing", "nan_or_inf"],
    # local_optimum
    "flat": ["plateau"],
    # reward_signal_too_sparse
    "near_zero": ["collapse"],
    # observation_normalization_issues
    "drift": ["not_increasing", "oscillation"],
    # action_rate_jittering
    "jittering": ["oscillation", "spike"],
    # catastrophic_forgetting
    "performance_drop": ["sudden_drop", "not_increasing"],
    # training_divergence
    "diverging": ["explosion", "nan_or_inf", "sudden_drop"],
    # sample_inefficiency
    "slow_progress": ["not_increasing"],
    # reward_scale_imbalance
    "imbalance": [],  # requires per-term data
    # out_of_range (surrogate ratio)
    "out_of_range": ["out_of_range"],
}


def patterns_match(expected: str, actual: str) -> bool:
    """Check if an actual detected pattern satisfies an expected pattern."""
    if expected == actual:
        return True
    compatible = PATTERN_COMPATIBILITY.get(expected, [])
    return actual in compatible


def metrics_match(expected_metric: str, actual_metric: str) -> bool:
    """Check if metrics match. Handles aliases loosely.

    The failure_modes.json uses a mix of canonical names ("total_reward",
    "mean_episode_reward") and descriptive names. We normalize a few common
    aliases here.

    "loss" is treated as a wildcard matching any *_loss metric (value_loss,
    policy_loss), since the failure_modes.json uses "loss" generically.
    """
    aliases = {
        "total_reward": {"reward"},
        "mean_episode_reward": {"reward"},
        "reward": {"reward"},
        "episode_length": {"episode_length"},
        "task_metric": set(),  # we don't have a canonical task_metric
        "reward_term_contribution": set(),
        "kl_divergence": {"kl_divergence"},
        "surrogate_ratio": {"surrogate_ratio"},
        "grad_norm": {"grad_norm"},
        "value_loss": {"value_loss"},
        "policy_loss": {"policy_loss"},
        "entropy": {"entropy"},
        "learning_rate": {"learning_rate"},
    }
    expected_set = aliases.get(expected_metric, {expected_metric})

    # "loss" is a generic match for any specific *_loss metric
    if expected_metric == "loss" and actual_metric in {"value_loss", "policy_loss"}:
        return True

    return actual_metric in expected_set


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------
@dataclass
class MatchedSymptom:
    """A pairing of an expected symptom (from failure mode) with an actual detected symptom."""
    expected_metric: str
    expected_pattern: str
    expected_note: str
    actual: Optional[Symptom]


@dataclass
class DiagnosisCandidate:
    """A candidate failure mode diagnosis.

    Attributes:
        failure_mode_id: ID from failure_modes.json (e.g., "reward_hacking").
        failure_mode_name: Human-readable name.
        category: Category (reward_design, training_instability, etc.).
        summary: One-line description of the failure mode.
        confidence: 0.0 to 1.0 — how well symptoms match.
        matched: List of MatchedSymptom (expected symptoms that were detected).
        unmatched: List of expected symptom dicts that weren't detected.
        root_causes: List of possible root causes.
        fixes: Priority-ordered list of fix actions.
        verification: How to verify the fix worked.
        sources: References for the thresholds/logic.
    """
    failure_mode_id: str
    failure_mode_name: str
    category: str
    summary: str
    confidence: float
    matched: List[MatchedSymptom] = field(default_factory=list)
    unmatched: List[Dict] = field(default_factory=list)
    root_causes: List[str] = field(default_factory=list)
    fixes: List[Dict] = field(default_factory=list)
    verification: str = ""
    sources: List[str] = field(default_factory=list)

    @property
    def matched_count(self) -> int:
        return len(self.matched)

    @property
    def total_expected(self) -> int:
        return len(self.matched) + len(self.unmatched)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_mode_id": self.failure_mode_id,
            "failure_mode_name": self.failure_mode_name,
            "category": self.category,
            "summary": self.summary,
            "confidence": round(self.confidence, 3),
            "matched_count": self.matched_count,
            "total_expected": self.total_expected,
            "matched": [
                {
                    "expected_metric": m.expected_metric,
                    "expected_pattern": m.expected_pattern,
                    "expected_note": m.expected_note,
                    "actual": m.actual.to_dict() if m.actual else None,
                }
                for m in self.matched
            ],
            "unmatched": self.unmatched,
            "root_causes": self.root_causes,
            "fixes": self.fixes,
            "verification": self.verification,
            "sources": self.sources,
        }


@dataclass
class DiagnosisResult:
    """Result of diagnosis.

    Attributes:
        candidates: Ranked list of DiagnosisCandidate (highest confidence first).
        unmatched_symptoms: Detected symptoms that didn't contribute to any candidate.
        total_symptoms: Total symptoms provided as input.
    """
    candidates: List[DiagnosisCandidate] = field(default_factory=list)
    unmatched_symptoms: List[Symptom] = field(default_factory=list)
    total_symptoms: int = 0

    def top_candidates(self, n: int = 3, min_confidence: float = 0.0) -> List[DiagnosisCandidate]:
        """Return top-n candidates above min_confidence."""
        return [c for c in self.candidates[:n] if c.confidence >= min_confidence]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "unmatched_symptoms": [s.to_dict() for s in self.unmatched_symptoms],
            "total_symptoms": self.total_symptoms,
        }


# ------------------------------------------------------------------
# DiagnosisEngine
# ------------------------------------------------------------------
class DiagnosisEngine:
    """Maps detected symptoms to ranked failure mode candidates.

    The engine is stateless — all state lives in the result.

    Confidence scoring:
        base = matched / total_expected
        If 2+ symptoms matched, boost by 0.15 (reinforcing evidence)
        If an error-severity symptom matched, boost by 0.1
        Cap at 1.0
    """

    MIN_CONFIDENCE_TO_REPORT = 0.25  # below this, don't include in candidates

    def __init__(self, failure_modes_path: Optional[Path] = None) -> None:
        """Initialize engine with failure mode database.

        Args:
            failure_modes_path: Path to failure_modes.json. Defaults to
                the package's resources/failure_modes.json.

        Raises:
            FileNotFoundError: If database file does not exist.
        """
        if failure_modes_path is None:
            failure_modes_path = _pkg_root / "resources" / "failure_modes.json"
        self.db_path = Path(failure_modes_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Failure modes database not found: {self.db_path}")

        with open(self.db_path, encoding="utf-8") as f:
            self._db: Dict[str, Any] = json.load(f)

        self._modes: List[Dict[str, Any]] = self._db.get("failure_modes", [])

    @property
    def version(self) -> str:
        return self._db.get("version", "unknown")

    def diagnose(
        self,
        symptoms: List[Symptom],
        config: Optional[Dict[str, Any]] = None,
    ) -> DiagnosisResult:
        """Diagnose failure mode(s) from detected symptoms.

        Args:
            symptoms: List of Symptom objects from LogAnalyzer.
            config: Optional env config dict for config-based hints
                (currently unused; reserved for future config_validator integration).

        Returns:
            DiagnosisResult with ranked candidates.

        Raises:
            TypeError: If symptoms is None or not a list.
        """
        if symptoms is None:
            raise TypeError("symptoms must be a list, got NoneType")
        if not isinstance(symptoms, list):
            raise TypeError(
                f"symptoms must be a list, got {type(symptoms).__name__}"
            )

        result = DiagnosisResult(total_symptoms=len(symptoms))

        if not symptoms:
            return result

        used_symptom_ids: set[int] = set()

        for mode in self._modes:
            candidate = self._score_mode(mode, symptoms)
            if candidate is not None and candidate.confidence >= self.MIN_CONFIDENCE_TO_REPORT:
                result.candidates.append(candidate)
                for m in candidate.matched:
                    if m.actual is not None:
                        used_symptom_ids.add(id(m.actual))

        # Sort by confidence descending
        result.candidates.sort(key=lambda c: c.confidence, reverse=True)

        # Collect unmatched symptoms
        result.unmatched_symptoms = [
            s for s in symptoms if id(s) not in used_symptom_ids
        ]

        return result

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score_mode(
        self, mode: Dict[str, Any], symptoms: List[Symptom]
    ) -> Optional[DiagnosisCandidate]:
        """Score a single failure mode against the symptom list."""
        expected_symptoms = mode.get("symptoms", [])
        if not expected_symptoms:
            return None

        matched: List[MatchedSymptom] = []
        unmatched: List[Dict] = []

        for expected in expected_symptoms:
            actual = self._find_matching_symptom(expected, symptoms)
            if actual is not None:
                matched.append(MatchedSymptom(
                    expected_metric=expected.get("metric", ""),
                    expected_pattern=expected.get("pattern", ""),
                    expected_note=expected.get("note", ""),
                    actual=actual,
                ))
            else:
                unmatched.append(expected)

        if not matched:
            return None

        total = len(expected_symptoms)
        base_confidence = len(matched) / total

        # Boost for reinforcing evidence (multiple symptoms of same mode)
        if len(matched) >= 2:
            base_confidence += 0.15

        # Boost if any matched actual symptom is error-severity
        if any(m.actual and m.actual.severity == "error" for m in matched):
            base_confidence += 0.10

        # Note: we deliberately do NOT penalize for unmatched expected symptoms.
        # Many expected patterns in failure_modes.json (e.g., "medium_stable",
        # "task_metric/below_benchmark") are not detectable by our LogAnalyzer,
        # and penalizing for them would suppress valid single-signal diagnoses.
        # Confidence is meant to reflect "this is a plausible candidate", not
        # "this is definitely the root cause".

        base_confidence = max(0.0, min(1.0, base_confidence))

        return DiagnosisCandidate(
            failure_mode_id=mode["id"],
            failure_mode_name=mode["name"],
            category=mode["category"],
            summary=mode["summary"],
            confidence=base_confidence,
            matched=matched,
            unmatched=unmatched,
            root_causes=list(mode.get("root_causes", [])),
            fixes=list(mode.get("fixes", [])),
            verification=mode.get("verification", ""),
            sources=list(mode.get("sources", [])),
        )

    def _find_matching_symptom(
        self, expected: Dict, symptoms: List[Symptom]
    ) -> Optional[Symptom]:
        """Find the best actual symptom matching an expected symptom spec."""
        expected_metric = expected.get("metric", "")
        expected_pattern = expected.get("pattern", "")

        # Special case: metrics we don't monitor can never match
        if not self._is_metric_monitorable(expected_metric):
            return None

        # Special case: patterns we don't have a detector for
        compatible_actuals = PATTERN_COMPATIBILITY.get(expected_pattern)
        if compatible_actuals is not None and not compatible_actuals and expected_pattern not in {
            "sudden_drop", "spike", "explosion", "nan_or_inf", "collapse",
            "not_increasing", "plateau", "oscillation", "out_of_range"
        }:
            return None

        # Find best match (prefer error severity, then more recent)
        best: Optional[Symptom] = None
        for s in symptoms:
            if not metrics_match(expected_metric, s.metric):
                continue
            if not patterns_match(expected_pattern, s.pattern):
                continue
            if best is None:
                best = s
            elif s.severity == "error" and best.severity != "error":
                best = s

        return best

    @staticmethod
    def _is_metric_monitorable(metric: str) -> bool:
        """Check if our LogAnalyzer can detect symptoms for this metric."""
        monitorable = {
            "total_reward", "mean_episode_reward", "reward", "episode_length",
            "grad_norm", "value_loss", "policy_loss", "loss",
            "entropy", "kl_divergence", "surrogate_ratio", "learning_rate",
        }
        return metric in monitorable


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose RL training failure modes from a symptom list (JSON). "
            "Symptoms are typically produced by log_analyzer.py."
        ),
    )
    parser.add_argument(
        "--symptoms", "-s", required=True,
        help="Path to JSON file of symptoms (from log_analyzer --json).",
    )
    parser.add_argument(
        "--top", type=int, default=3,
        help="Show top N candidates (default 3).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON (default: human-readable).",
    )

    args = parser.parse_args()

    symptoms_path = Path(args.symptoms)
    if not symptoms_path.exists():
        print(f"Error: symptoms file not found: {symptoms_path}", file=sys.stderr)
        sys.exit(1)

    with open(symptoms_path, encoding="utf-8") as f:
        symptoms_data = json.load(f)

    # Reconstruct Symptom objects from JSON
    symptoms: List[Symptom] = []
    for s_dict in symptoms_data.get("symptoms", symptoms_data if isinstance(symptoms_data, list) else []):
        step_range = tuple(s_dict.get("step_range", [0, 0]))
        symptoms.append(Symptom(
            metric=s_dict["metric"],
            tag=s_dict.get("tag", s_dict["metric"]),
            pattern=s_dict["pattern"],
            severity=s_dict["severity"],
            evidence=s_dict["evidence"],
            step_range=step_range,  # type: ignore
            value_summary=s_dict.get("value_summary", {}),
            threshold=s_dict.get("threshold"),
        ))

    engine = DiagnosisEngine()
    result = engine.diagnose(symptoms)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Diagnosis from {result.total_symptoms} symptoms:")
        if not result.candidates:
            print("  No failure modes matched with sufficient confidence.")
        else:
            print(f"\nTop candidates (showing {args.top}):")
            for c in result.top_candidates(args.top):
                print(f"\n  [{c.confidence:.0%}] {c.failure_mode_name} ({c.category})")
                print(f"    {c.summary}")
                print(f"    Matched: {c.matched_count}/{c.total_expected} expected symptoms")
                for m in c.matched:
                    if m.actual:
                        print(f"      ✓ {m.expected_metric}/{m.expected_pattern}: "
                              f"{m.actual.evidence[:100]}")
                for u in c.unmatched:
                    print(f"      ✗ {u.get('metric', '?')}/{u.get('pattern', '?')}: "
                          f"{u.get('note', '')[:80]}")
                if c.fixes:
                    print(f"    Suggested fixes (priority order):")
                    for fix in sorted(c.fixes, key=lambda f: f.get("priority", 99)):
                        print(f"      {fix['priority']}. {fix['action'][:100]}")
                if c.verification:
                    print(f"    Verification: {c.verification[:120]}")

        if result.unmatched_symptoms:
            print(f"\nUnmatched symptoms ({len(result.unmatched_symptoms)}):")
            for s in result.unmatched_symptoms:
                print(f"  [{s.severity}] {s.metric}/{s.pattern}: {s.evidence[:80]}")


if __name__ == "__main__":
    main()
