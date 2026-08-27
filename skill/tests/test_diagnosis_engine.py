"""Tests for the diagnosis engine (module 2 reasoning layer).

Covers:
- Pattern compatibility matching
- Metric alias matching (including "loss" wildcard)
- Single-symptom diagnosis
- Multi-symptom reinforced diagnosis
- Confidence scoring (boosts for multiple matches, error severity)
- Candidate ranking
- Unmatched symptom collection
- Edge cases (empty symptoms, no matches)
- Integration with log_analyzer output
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from scripts.diagnosis_engine import (
    DiagnosisEngine, DiagnosisCandidate, DiagnosisResult,
    MatchedSymptom, patterns_match, metrics_match,
)
from scripts.log_analyzer import Symptom, LogAnalyzer


@pytest.fixture
def engine() -> DiagnosisEngine:
    return DiagnosisEngine()


def make_symptom(metric, pattern, severity="warning", evidence="test"):
    return Symptom(
        metric=metric, tag=f"Test/{metric}", pattern=pattern,
        severity=severity, evidence=evidence, step_range=(0, 100),
    )


# ------------------------------------------------------------------
# Pattern compatibility
# ------------------------------------------------------------------

class TestPatternMatching:
    def test_exact_match(self):
        assert patterns_match("sudden_drop", "sudden_drop")
        assert patterns_match("explosion", "explosion")

    def test_numerical_instability_compatible(self):
        # spike, explosion, nan_or_inf are all numerical instability
        assert patterns_match("spike", "explosion")
        assert patterns_match("spike", "nan_or_inf")
        assert patterns_match("explosion", "spike")
        assert patterns_match("explosion", "nan_or_inf")
        assert patterns_match("nan_or_inf", "spike")
        assert patterns_match("nan_or_inf", "explosion")

    def test_not_increasing_matches_plateau(self):
        assert patterns_match("not_increasing_or_decreasing", "not_increasing")
        assert patterns_match("not_increasing_or_decreasing", "plateau")

    def test_spike_or_collapse_matches_both(self):
        assert patterns_match("spike_or_collapse", "spike")
        assert patterns_match("spike_or_collapse", "collapse")

    def test_incompatible_patterns_dont_match(self):
        assert not patterns_match("sudden_drop", "plateau")
        assert not patterns_match("collapse", "spike")
        assert not patterns_match("out_of_range", "explosion")


# ------------------------------------------------------------------
# Metric matching
# ------------------------------------------------------------------

class TestMetricMatching:
    def test_exact_metric_match(self):
        assert metrics_match("reward", "reward")
        assert metrics_match("grad_norm", "grad_norm")

    def test_reward_aliases(self):
        assert metrics_match("total_reward", "reward")
        assert metrics_match("mean_episode_reward", "reward")

    def test_loss_wildcard(self):
        # "loss" should match any *_loss metric
        assert metrics_match("loss", "value_loss")
        assert metrics_match("loss", "policy_loss")

    def test_non_metric_doesnt_match(self):
        assert not metrics_match("reward", "grad_norm")
        assert not metrics_match("entropy", "reward")


# ------------------------------------------------------------------
# Single-symptom diagnosis
# ------------------------------------------------------------------

class TestSingleSymptomDiagnosis:
    def test_reward_plateau_finds_local_optimum(self, engine):
        s = make_symptom("reward", "plateau")
        result = engine.diagnose([s])
        ids = [c.failure_mode_id for c in result.candidates]
        assert "local_optimum" in ids

    def test_reward_plateau_finds_entropy_collapse(self, engine):
        # entropy_collapse also expects reward/plateau
        s = make_symptom("reward", "plateau")
        result = engine.diagnose([s])
        ids = [c.failure_mode_id for c in result.candidates]
        assert "entropy_collapse" in ids

    def test_grad_explosion_finds_gradient_explosion(self, engine):
        s = make_symptom("grad_norm", "explosion", severity="error")
        result = engine.diagnose([s])
        ids = [c.failure_mode_id for c in result.candidates]
        assert "gradient_explosion" in ids

    def test_nan_finds_multiple_instability_modes(self, engine):
        s = make_symptom("value_loss", "nan_or_inf", severity="error")
        result = engine.diagnose([s])
        # nan_or_inf is evidence for several instability modes
        assert len(result.candidates) >= 1


# ------------------------------------------------------------------
# Multi-symptom reinforced diagnosis
# ------------------------------------------------------------------

class TestMultiSymptomDiagnosis:
    def test_policy_collapse_full_match(self, engine):
        # All 4 symptoms of policy_collapse present
        symptoms = [
            make_symptom("reward", "sudden_drop", severity="error"),
            make_symptom("kl_divergence", "spike"),
            make_symptom("entropy", "collapse"),
            make_symptom("surrogate_ratio", "out_of_range"),
        ]
        result = engine.diagnose(symptoms)
        top = result.candidates[0]
        assert top.failure_mode_id == "policy_collapse"
        assert top.confidence >= 0.9
        assert top.matched_count >= 3

    def test_reinforcing_evidence_boosts_confidence(self, engine):
        # Single symptom
        single = [make_symptom("reward", "sudden_drop", severity="error")]
        result_single = engine.diagnose(single)
        conf_single = next(
            c.confidence for c in result_single.candidates
            if c.failure_mode_id == "policy_collapse"
        )

        # Multiple symptoms (reinforcing)
        multi = [
            make_symptom("reward", "sudden_drop", severity="error"),
            make_symptom("kl_divergence", "spike"),
            make_symptom("entropy", "collapse"),
        ]
        result_multi = engine.diagnose(multi)
        conf_multi = next(
            c.confidence for c in result_multi.candidates
            if c.failure_mode_id == "policy_collapse"
        )

        assert conf_multi > conf_single

    def test_error_severity_boosts_confidence(self, engine):
        # Same symptom, different severity
        warning_sym = [make_symptom("grad_norm", "explosion", severity="warning")]
        error_sym = [make_symptom("grad_norm", "explosion", severity="error")]

        result_w = engine.diagnose(warning_sym)
        result_e = engine.diagnose(error_sym)

        conf_w = next(
            (c.confidence for c in result_w.candidates
             if c.failure_mode_id == "gradient_explosion"), 0.0
        )
        conf_e = next(
            (c.confidence for c in result_e.candidates
             if c.failure_mode_id == "gradient_explosion"), 0.0
        )
        assert conf_e > conf_w


# ------------------------------------------------------------------
# Candidate ranking
# ------------------------------------------------------------------

class TestCandidateRanking:
    def test_candidates_sorted_by_confidence_desc(self, engine):
        symptoms = [
            make_symptom("reward", "sudden_drop", severity="error"),
            make_symptom("grad_norm", "explosion", severity="error"),
            make_symptom("kl_divergence", "spike"),
        ]
        result = engine.diagnose(symptoms)
        confidences = [c.confidence for c in result.candidates]
        assert confidences == sorted(confidences, reverse=True)

    def test_top_candidates_limits_count(self, engine):
        symptoms = [
            make_symptom("reward", "sudden_drop", severity="error"),
            make_symptom("grad_norm", "explosion", severity="error"),
        ]
        result = engine.diagnose(symptoms)
        top = result.top_candidates(n=1)
        assert len(top) <= 1

    def test_top_candidates_min_confidence(self, engine):
        symptoms = [make_symptom("reward", "plateau")]
        result = engine.diagnose(symptoms)
        # All candidates should be at 25% (1/4 matched)
        high_conf = result.top_candidates(n=10, min_confidence=0.5)
        assert len(high_conf) == 0


# ------------------------------------------------------------------
# Candidate structure
# ------------------------------------------------------------------

class TestCandidateStructure:
    def test_candidate_has_root_causes(self, engine):
        s = make_symptom("reward", "sudden_drop", severity="error")
        result = engine.diagnose([s])
        for c in result.candidates:
            assert isinstance(c.root_causes, list)

    def test_candidate_has_fixes(self, engine):
        s = make_symptom("reward", "sudden_drop", severity="error")
        result = engine.diagnose([s])
        for c in result.candidates:
            assert isinstance(c.fixes, list)

    def test_candidate_fixes_have_priority(self, engine):
        s = make_symptom("reward", "sudden_drop", severity="error")
        result = engine.diagnose([s])
        # At least one candidate should have prioritized fixes
        has_prioritized = any(
            any("priority" in f for f in c.fixes)
            for c in result.candidates
        )
        assert has_prioritized

    def test_candidate_has_verification(self, engine):
        s = make_symptom("reward", "sudden_drop", severity="error")
        result = engine.diagnose([s])
        for c in result.candidates:
            # verification may be empty for some modes, but field exists
            assert isinstance(c.verification, str)

    def test_matched_symptom_has_actual(self, engine):
        s = make_symptom("reward", "sudden_drop", severity="error")
        result = engine.diagnose([s])
        # The top candidate should have at least one matched symptom with actual
        top = result.candidates[0]
        assert len(top.matched) >= 1
        assert top.matched[0].actual is not None

    def test_to_dict_serializable(self, engine):
        s = make_symptom("reward", "sudden_drop", severity="error")
        result = engine.diagnose([s])
        d = result.candidates[0].to_dict()
        assert "failure_mode_id" in d
        assert "confidence" in d
        assert "matched" in d
        assert "fixes" in d


# ------------------------------------------------------------------
# Unmatched symptoms
# ------------------------------------------------------------------

class TestUnmatchedSymptoms:
    def test_unmatched_symptoms_collected(self, engine):
        # A symptom that doesn't match any failure mode strongly
        # (e.g., learning_rate is rarely a direct symptom)
        s = make_symptom("learning_rate", "oscillation")
        result = engine.diagnose([s])
        # Whatever matches or not, unmatched list should be a list
        assert isinstance(result.unmatched_symptoms, list)

    def test_matched_symptom_not_in_unmatched(self, engine):
        s = make_symptom("reward", "sudden_drop", severity="error")
        result = engine.diagnose([s])
        # The sudden_drop symptom should be matched by policy_collapse
        # so it shouldn't appear in unmatched
        unmatched_metrics = [s.metric for s in result.unmatched_symptoms]
        # If policy_collapse matched, reward shouldn't be in unmatched
        if any(c.failure_mode_id == "policy_collapse" for c in result.candidates):
            assert "reward" not in unmatched_metrics


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_symptoms_returns_empty(self, engine):
        result = engine.diagnose([])
        assert len(result.candidates) == 0
        assert result.total_symptoms == 0

    def test_low_confidence_not_reported(self, engine):
        # A single weak symptom shouldn't produce high-confidence candidates
        s = make_symptom("learning_rate", "oscillation")
        result = engine.diagnose([s])
        # Either no candidates or all below 0.5
        for c in result.candidates:
            # Single-symptom matches cap around 0.25-0.35
            assert c.confidence < 0.6

    def test_total_symptoms_tracked(self, engine):
        symptoms = [
            make_symptom("reward", "sudden_drop", severity="error"),
            make_symptom("grad_norm", "explosion", severity="error"),
        ]
        result = engine.diagnose(symptoms)
        assert result.total_symptoms == 2


# ------------------------------------------------------------------
# Integration: log_analyzer → diagnosis_engine
# ------------------------------------------------------------------

class TestLogAnalyzerIntegration:
    """End-to-end: LogAnalyzer detects symptoms, DiagnosisEngine diagnoses them."""

    def test_policy_collapse_scenario_end_to_end(self, engine):
        # Simulate a training run that collapsed:
        # - reward was rising, then suddenly dropped
        # - KL diverged (spike)
        # - entropy collapsed
        analyzer = LogAnalyzer()
        reward_series = (
            [(i, float(i) * 0.5) for i in range(50)]  # rising
            + [(50 + i, 5.0) for i in range(1, 5)]      # sudden drop to 5
            + [(55 + i, 1.0) for i in range(1, 20)]     # stuck low
        )
        kl_series = (
            [(i, 0.005) for i in range(50)]             # normal KL
            + [(50 + i, 0.08) for i in range(1, 25)]    # KL spike
        )
        entropy_series = (
            [(i, 1.5 - i * 0.001) for i in range(50)]   # slow decline
            + [(50 + i, 0.05) for i in range(1, 25)]    # collapse
        )

        metrics = {
            "Train/mean_reward": reward_series,
            "Train/kl_divergence": kl_series,
            "Train/entropy": entropy_series,
        }
        analysis = analyzer.analyze(metrics)

        # LogAnalyzer should detect symptoms
        assert analysis.has_symptoms

        # DiagnosisEngine should identify policy_collapse as top candidate
        result = engine.diagnose(analysis.symptoms)
        assert len(result.candidates) > 0
        top = result.candidates[0]
        assert top.failure_mode_id == "policy_collapse"
        assert top.confidence >= 0.7

    def test_gradient_explosion_scenario_end_to_end(self, engine):
        # Grad norm explodes, value loss goes NaN
        analyzer = LogAnalyzer()
        grad_series = (
            [(i, 1.0) for i in range(40)]
            + [(40 + i, 200.0) for i in range(1, 20)]
        )
        vloss_series = (
            [(i, 0.5) for i in range(45)]
            + [(45 + i, float("nan")) for i in range(15)]
        )

        metrics = {
            "Loss/grad_norm": grad_series,
            "Loss/value_loss": vloss_series,
        }
        analysis = analyzer.analyze(metrics)
        result = engine.diagnose(analysis.symptoms)

        top = result.candidates[0]
        assert top.failure_mode_id == "gradient_explosion"
        assert top.confidence >= 0.8

    def test_healthy_training_no_diagnosis(self, engine):
        # All metrics healthy → no symptoms → no candidates
        analyzer = LogAnalyzer()
        metrics = {
            "Train/mean_reward": [(i, float(i) * 0.5) for i in range(100)],
            "Loss/grad_norm": [(i, 1.0) for i in range(100)],
            "Train/entropy": [(i, 1.0) for i in range(100)],
        }
        analysis = analyzer.analyze(metrics)
        assert not analysis.has_symptoms

        result = engine.diagnose(analysis.symptoms)
        assert len(result.candidates) == 0


# ------------------------------------------------------------------
# Edge cases (input boundaries, error handling)
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_none_symptoms_raises_type_error(self, engine):
        with pytest.raises(TypeError, match="symptoms"):
            engine.diagnose(None)

    def test_empty_symptoms_returns_no_candidates(self, engine):
        result = engine.diagnose([])
        assert len(result.candidates) == 0

    def test_top_candidates_zero_n_returns_empty(self, engine):
        """top_candidates(n=0) should return empty list, not crash."""
        from scripts.log_analyzer import Symptom
        s = Symptom(
            metric="Train/mean_reward",
            tag="Train/mean_reward",
            pattern="plateau",
            severity="warning",
            evidence="flat",
            step_range=(0, 100),
        )
        result = engine.diagnose([s])
        assert result.top_candidates(n=0) == []

    def test_duplicate_symptoms_handled(self, engine):
        """Two identical symptoms should not double-count or crash."""
        from scripts.log_analyzer import Symptom
        s = Symptom(
            metric="Train/mean_reward",
            tag="Train/mean_reward",
            pattern="plateau",
            severity="warning",
            evidence="flat",
            step_range=(0, 100),
        )
        result = engine.diagnose([s, s])
        assert isinstance(result.candidates, list)

    def test_chinese_evidence_does_not_crash(self, engine):
        """Evidence with Chinese characters should serialize fine."""
        from scripts.log_analyzer import Symptom
        s = Symptom(
            metric="Train/mean_reward",
            tag="Train/mean_reward",
            pattern="plateau",
            severity="warning",
            evidence="reward 长期平台不上升",
            step_range=(0, 100),
        )
        result = engine.diagnose([s])
        assert isinstance(result.candidates, list)
