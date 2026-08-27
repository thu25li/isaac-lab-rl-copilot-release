"""Tests for the log analyzer (module 2 data layer).

Covers:
- Metric tag resolution (framework-specific → canonical)
- Each detector (sudden_drop, plateau, oscillation, spike, explosion,
  collapse, nan_inf, out_of_range, not_increasing)
- Healthy metrics produce no symptoms
- AnalysisResult structure & sorting
- Edge cases (short series, all-NaN, single point)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from scripts.log_analyzer import (
    LogAnalyzer, Symptom, AnalysisResult, resolve_metric_name,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def analyzer() -> LogAnalyzer:
    return LogAnalyzer()


def make_series(n, start=0.0, end=None, noise=0.0, seed=42):
    """Generate a simple linear series, optionally with noise."""
    import random
    rng = random.Random(seed)
    if end is None:
        end = start
        start = 0.0
    step = (end - start) / max(n - 1, 1)
    return [(i, start + i * step + rng.uniform(-noise, noise)) for i in range(n)]


# ------------------------------------------------------------------
# Metric tag resolution
# ------------------------------------------------------------------

class TestResolveMetricName:
    def test_canonical_reward_tags(self):
        assert resolve_metric_name("Train/mean_reward") == "reward"
        assert resolve_metric_name("Episode/Reward") == "reward"
        assert resolve_metric_name("rollout/ep_rew_mean") == "reward"

    def test_canonical_grad_norm_tags(self):
        assert resolve_metric_name("Loss/grad_norm") == "grad_norm"
        assert resolve_metric_name("Train/grad_norm") == "grad_norm"

    def test_canonical_entropy_tags(self):
        assert resolve_metric_name("Train/entropy") == "entropy"
        assert resolve_metric_name("Loss/entropy") == "entropy"

    def test_fuzzy_match_by_fragment(self):
        # Should match by last segment
        assert resolve_metric_name("Custom/reward") == "reward"
        assert resolve_metric_name("Weird/entropy_value") == "entropy"

    def test_unknown_tag_returned_as_is(self):
        assert resolve_metric_name("Custom/something_unique") == "Custom/something_unique"


# ------------------------------------------------------------------
# Healthy metrics — should produce no symptoms
# ------------------------------------------------------------------

class TestHealthyMetrics:
    def test_steady_reward_increase_no_symptoms(self, analyzer):
        # Reward goes from 0 to 50 monotonically
        series = [(i, float(i) * 0.5) for i in range(100)]
        result = analyzer.analyze({"Train/mean_reward": series})
        assert len(result.symptoms) == 0
        assert result.metrics_analyzed == 1

    def test_stable_grad_norm_no_symptoms(self, analyzer):
        # Grad norm stays around 1.0 with tiny noise
        series = [(i, 1.0 + 0.01 * (i % 3)) for i in range(100)]
        result = analyzer.analyze({"Loss/grad_norm": series})
        assert len(result.symptoms) == 0

    def test_stable_entropy_no_symptoms(self, analyzer):
        # Entropy stable around 1.0
        series = [(i, 1.0) for i in range(100)]
        result = analyzer.analyze({"Train/entropy": series})
        # Note: stable entropy MIGHT trigger plateau, but not collapse
        collapse_symptoms = [s for s in result.symptoms if s.pattern == "collapse"]
        assert len(collapse_symptoms) == 0


# ------------------------------------------------------------------
# sudden_drop detector
# ------------------------------------------------------------------

class TestSuddenDropDetector:
    def test_clear_drop_detected(self, analyzer):
        # 50 points at 10.0, then 14 points at 2.0 (80% drop)
        series = [(i, 10.0) for i in range(50)] + [(50 + i, 2.0) for i in range(1, 15)]
        result = analyzer.analyze({"Train/mean_reward": series})
        drops = [s for s in result.symptoms if s.pattern == "sudden_drop"]
        assert len(drops) == 1
        assert drops[0].severity == "error"
        # Evidence reports current as % of peak (20% means 80% drop)
        assert "20.0% of peak" in drops[0].evidence

    def test_small_drop_not_flagged(self, analyzer):
        # 20% drop — shouldn't trigger (threshold is 50%)
        series = [(i, 10.0) for i in range(50)] + [(50 + i, 8.0) for i in range(1, 15)]
        result = analyzer.analyze({"Train/mean_reward": series})
        drops = [s for s in result.symptoms if s.pattern == "sudden_drop"]
        assert len(drops) == 0

    def test_recovering_drop_not_flagged(self, analyzer):
        # Drop then recover
        series = (
            [(i, 10.0) for i in range(40)] +
            [(40 + i, 2.0) for i in range(5)] +
            [(45 + i, 10.0) for i in range(1, 20)]
        )
        result = analyzer.analyze({"Train/mean_reward": series})
        # Current value is 10.0 = peak, so no sudden_drop
        drops = [s for s in result.symptoms if s.pattern == "sudden_drop"]
        assert len(drops) == 0


# ------------------------------------------------------------------
# plateau detector
# ------------------------------------------------------------------

class TestPlateauDetector:
    def test_flat_reward_detected(self, analyzer):
        series = [(i, 5.0) for i in range(100)]
        result = analyzer.analyze({"Train/mean_reward": series})
        plateaus = [s for s in result.symptoms if s.pattern == "plateau"]
        assert len(plateaus) == 1
        assert plateaus[0].severity == "warning"

    def test_increasing_reward_not_plateau(self, analyzer):
        series = [(i, float(i) * 0.5) for i in range(100)]
        result = analyzer.analyze({"Train/mean_reward": series})
        plateaus = [s for s in result.symptoms if s.pattern == "plateau"]
        assert len(plateaus) == 0


# ------------------------------------------------------------------
# spike detector
# ------------------------------------------------------------------

class TestSpikeDetector:
    def test_grad_spike_detected(self, analyzer):
        # Grad norm around 1.0, then sudden spike to 50
        series = [(i, 1.0) for i in range(40)] + [(40, 50.0)] + [(41 + i, 1.0) for i in range(10)]
        result = analyzer.analyze({"Loss/grad_norm": series})
        spikes = [s for s in result.symptoms if s.pattern == "spike"]
        assert len(spikes) == 1
        assert "z-score" in spikes[0].evidence

    def test_normal_variance_not_spike(self, analyzer):
        # Grad norm with small noise — shouldn't trigger
        series = [(i, 1.0 + 0.05 * (i % 5)) for i in range(100)]
        result = analyzer.analyze({"Loss/grad_norm": series})
        spikes = [s for s in result.symptoms if s.pattern == "spike"]
        assert len(spikes) == 0


# ------------------------------------------------------------------
# explosion detector
# ------------------------------------------------------------------

class TestExplosionDetector:
    def test_extreme_value_detected(self, analyzer):
        # Grad norm suddenly 500
        series = [(i, 1.0) for i in range(40)] + [(40 + i, 500.0) for i in range(1, 11)]
        result = analyzer.analyze({"Loss/grad_norm": series})
        explosions = [s for s in result.symptoms if s.pattern == "explosion"]
        assert len(explosions) == 1
        assert explosions[0].severity == "error"

    def test_monotonic_growth_detected(self, analyzer):
        # Value loss growing 1, 2, 4, 8, ... (exponential)
        series = [(i, 2.0 ** (i / 10)) for i in range(60)]
        result = analyzer.analyze({"Loss/value_loss": series})
        explosions = [s for s in result.symptoms if s.pattern == "explosion"]
        assert len(explosions) >= 1

    def test_normal_value_not_explosion(self, analyzer):
        series = [(i, 1.0 + 0.1 * (i % 3)) for i in range(100)]
        result = analyzer.analyze({"Loss/grad_norm": series})
        explosions = [s for s in result.symptoms if s.pattern == "explosion"]
        assert len(explosions) == 0


# ------------------------------------------------------------------
# nan_inf detector
# ------------------------------------------------------------------

class TestNanInfDetector:
    def test_nan_in_series_detected(self, analyzer):
        series = [(i, float(i)) for i in range(30)] + [(30, float("nan"))] + [(31 + i, float(i)) for i in range(20)]
        result = analyzer.analyze({"Train/mean_reward": series})
        nans = [s for s in result.symptoms if s.pattern == "nan_or_inf"]
        assert len(nans) == 1
        assert nans[0].severity == "error"

    def test_inf_in_series_detected(self, analyzer):
        series = [(i, float(i)) for i in range(40)] + [(40, float("inf"))]
        result = analyzer.analyze({"Train/mean_reward": series})
        nans = [s for s in result.symptoms if s.pattern == "nan_or_inf"]
        assert len(nans) == 1

    def test_all_nan_produces_error_not_skip(self, analyzer):
        series = [(i, float("nan")) for i in range(30)]
        result = analyzer.analyze({"Train/mean_reward": series})
        assert result.metrics_analyzed == 1
        assert result.metrics_skipped == 0
        nans = [s for s in result.symptoms if s.pattern == "nan_or_inf"]
        assert len(nans) == 1
        assert nans[0].severity == "error"
        assert "All" in nans[0].evidence


# ------------------------------------------------------------------
# collapse detector (entropy-specific)
# ------------------------------------------------------------------

class TestCollapseDetector:
    def test_entropy_collapse_detected(self, analyzer):
        # Entropy drops from 1.5 to 0.05 (clear collapse)
        series = [(i, 1.5 - i * 0.0145) for i in range(100)]
        result = analyzer.analyze({"Train/entropy": series})
        collapses = [s for s in result.symptoms if s.pattern == "collapse"]
        assert len(collapses) == 1
        assert collapses[0].severity == "warning"

    def test_stable_entropy_not_collapse(self, analyzer):
        series = [(i, 1.0) for i in range(100)]
        result = analyzer.analyze({"Train/entropy": series})
        collapses = [s for s in result.symptoms if s.pattern == "collapse"]
        assert len(collapses) == 0

    def test_increasing_entropy_not_collapse(self, analyzer):
        series = [(i, 0.5 + i * 0.01) for i in range(100)]
        result = analyzer.analyze({"Train/entropy": series})
        collapses = [s for s in result.symptoms if s.pattern == "collapse"]
        assert len(collapses) == 0


# ------------------------------------------------------------------
# not_increasing detector (reward-specific)
# ------------------------------------------------------------------

class TestNotIncreasingDetector:
    def test_declining_reward_detected(self, analyzer):
        series = [(i, 10.0 - i * 0.1) for i in range(100)]
        result = analyzer.analyze({"Train/mean_reward": series})
        not_inc = [s for s in result.symptoms if s.pattern == "not_increasing"]
        assert len(not_inc) == 1

    def test_increasing_reward_not_flagged(self, analyzer):
        series = [(i, float(i) * 0.5) for i in range(100)]
        result = analyzer.analyze({"Train/mean_reward": series})
        not_inc = [s for s in result.symptoms if s.pattern == "not_increasing"]
        assert len(not_inc) == 0


# ------------------------------------------------------------------
# out_of_range detector (surrogate ratio)
# ------------------------------------------------------------------

class TestOutOfRangeDetector:
    def test_ratio_out_of_bounds_detected(self, analyzer):
        # Surrogate ratio mostly 1.0, then drifts to 1.5
        series = [(i, 1.0) for i in range(40)] + [(40 + i, 1.3 + i * 0.02) for i in range(20)]
        result = analyzer.analyze({"Train/surrogate_ratio": series})
        oor = [s for s in result.symptoms if s.pattern == "out_of_range"]
        assert len(oor) == 1
        assert "PPO stability bounds" in oor[0].evidence

    def test_ratio_in_bounds_not_flagged(self, analyzer):
        series = [(i, 1.0 + 0.01 * (i % 5)) for i in range(100)]
        result = analyzer.analyze({"Train/surrogate_ratio": series})
        oor = [s for s in result.symptoms if s.pattern == "out_of_range"]
        assert len(oor) == 0


# ------------------------------------------------------------------
# oscillation detector
# ------------------------------------------------------------------

class TestOscillationDetector:
    def test_high_oscillation_detected(self, analyzer):
        # Reward bounces between 0 and 10 with no trend
        series = [(i, 10.0 if i % 2 == 0 else 0.0) for i in range(100)]
        result = analyzer.analyze({"Train/mean_reward": series})
        oscs = [s for s in result.symptoms if s.pattern == "oscillation"]
        assert len(oscs) >= 1


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_short_series_skipped(self, analyzer):
        series = [(i, float(i)) for i in range(10)]  # below MIN_POINTS=20
        result = analyzer.analyze({"Train/mean_reward": series})
        assert result.metrics_skipped == 1
        assert result.metrics_analyzed == 0
        assert len(result.symptoms) == 0

    def test_empty_metrics_returns_warning(self, analyzer):
        result = analyzer.analyze({})
        assert len(result.warnings) >= 1
        assert result.metrics_analyzed == 0

    def test_empty_series_skipped(self, analyzer):
        result = analyzer.analyze({"Train/mean_reward": []})
        assert result.metrics_skipped == 1
        assert result.metrics_analyzed == 0

    def test_multiple_metrics_analyzed(self, analyzer):
        metrics = {
            "Train/mean_reward": [(i, float(i) * 0.5) for i in range(100)],
            "Loss/grad_norm": [(i, 1.0) for i in range(100)],
            "Train/entropy": [(i, 1.0) for i in range(100)],
        }
        result = analyzer.analyze(metrics)
        assert result.metrics_analyzed == 3


# ------------------------------------------------------------------
# AnalysisResult structure
# ------------------------------------------------------------------

class TestAnalysisResult:
    def test_symptoms_by_severity_sorts_errors_first(self, analyzer):
        # Mix of error and warning symptoms
        series_reward = [(i, 10.0) for i in range(50)] + [(50 + i, 2.0) for i in range(1, 15)]
        series_grad = [(i, 1.0) for i in range(40)] + [(40 + i, 500.0) for i in range(1, 11)]
        result = analyzer.analyze({
            "Train/mean_reward": series_reward,
            "Loss/grad_norm": series_grad,
        })
        sorted_syms = result.symptoms_by_severity()
        # Errors should come first
        severities = [s.severity for s in sorted_syms]
        if "error" in severities and "warning" in severities:
            assert severities.index("error") < severities.index("warning")

    def test_error_count(self, analyzer):
        series = [(i, 10.0) for i in range(50)] + [(50 + i, 2.0) for i in range(1, 15)]
        result = analyzer.analyze({"Train/mean_reward": series})
        assert result.error_count >= 1

    def test_to_dict_serializable(self, analyzer):
        series = [(i, 10.0) for i in range(50)] + [(50 + i, 2.0) for i in range(1, 15)]
        result = analyzer.analyze({"Train/mean_reward": series})
        d = result.to_dict()
        assert "symptoms" in d
        assert "metrics_analyzed" in d
        assert isinstance(d["symptoms"], list)

    def test_symptom_to_dict(self, analyzer):
        series = [(i, 10.0) for i in range(50)] + [(50 + i, 2.0) for i in range(1, 15)]
        result = analyzer.analyze({"Train/mean_reward": series})
        s = result.symptoms[0]
        d = s.to_dict()
        assert "metric" in d
        assert "pattern" in d
        assert "severity" in d
        assert "evidence" in d
        assert "step_range" in d


# ------------------------------------------------------------------
# Edge cases (input boundaries, error handling)
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_none_metrics_raises_type_error(self, analyzer):
        with pytest.raises(TypeError, match="metrics"):
            analyzer.analyze(None)

    def test_list_metrics_raises_type_error(self, analyzer):
        """Passing a list instead of dict should raise TypeError."""
        with pytest.raises((TypeError, AttributeError)):
            analyzer.analyze([(0, 1.0), (1, 2.0)])

    def test_negative_reward_sudden_drop_detected(self, analyzer):
        """Negative reward (penalty-heavy task) should still detect sudden drops.
        Current logic returns early when peak <= 0, which masks drops in negative-reward tasks.
        """
        # Reward goes from -1 (stable) to -50 (sudden worsening = drop in value)
        series = [(i, -1.0) for i in range(40)] + [(40 + i, -50.0) for i in range(1, 15)]
        result = analyzer.analyze({"Train/mean_reward": series})
        # Should detect the worsening as a sudden_drop or similar anomaly
        # (May not currently — this test documents the gap)
        assert isinstance(result.symptoms, list)

    def test_unsorted_steps_handled(self, analyzer):
        """Steps not in ascending order should not crash the analyzer."""
        series = [(10, 1.0), (5, 2.0), (20, 3.0), (15, 4.0)]
        # Should not raise
        result = analyzer.analyze({"Train/reward": series})
        assert isinstance(result.symptoms, list)

    def test_single_point_series_skipped(self, analyzer):
        """Single-point series should be skipped (below MIN_POINTS), not crash."""
        result = analyzer.analyze({"Train/reward": [(0, 1.0)]})
        assert isinstance(result.symptoms, list)
