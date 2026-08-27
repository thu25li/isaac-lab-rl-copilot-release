#!/usr/bin/env python3
"""Log analyzer — anomaly detection over parsed TensorBoard metrics.

This is the data layer of module 2. It takes raw metric series (from
TensorboardParser) and detects symptoms of training failure modes:
sudden drops, plateaus, oscillation, gradient spikes, entropy collapse,
NaN/Inf, etc.

Symptoms are framework-agnostic — they describe *what* is happening in
the data, not *why*. The diagnosis_engine maps symptoms to root causes.

Design principle (from failure_modes.json note):
    "诊断引擎应优先使用相对变化（如 KL 单步增量超过训练均值 5-10 倍）
     作为触发条件，绝对阈值仅作辅助。"
So detectors use local statistics (rolling mean/std) rather than fixed
thresholds wherever possible. Different tasks have different metric
scales; absolute thresholds cause false positives.

Usage (library):
    from scripts.log_analyzer import LogAnalyzer
    from scripts.utils.tensorboard_parser import TensorboardParser
    parser = TensorboardParser("/path/to/run")
    metrics = parser.parse()
    analyzer = LogAnalyzer()
    result = analyzer.analyze(metrics)
    for s in result.symptoms:
        print(s.severity, s.metric, s.pattern, s.evidence)

Usage (CLI):
    python scripts/log_analyzer.py --logdir /path/to/tensorboard/run
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Allow direct script execution
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))


MetricSeries = List[Tuple[int, float]]


# ------------------------------------------------------------------
# Canonical metric names — maps framework-specific tags to canonical names
# so detectors can reason about "reward" / "grad_norm" etc. uniformly.
# ------------------------------------------------------------------
METRIC_TAG_ALIASES: Dict[str, List[str]] = {
    "reward": [
        "Train/mean_reward", "Train/episode_reward", "Episode/Reward",
        "rollout/ep_rew_mean", "train/reward", "reward", "eval/mean_reward",
        "Episode/mean_reward",
    ],
    "episode_length": [
        "Train/mean_episode_length", "Episode/Length", "rollout/ep_len_mean",
        "train/episode_length", "episode_length",
    ],
    "grad_norm": [
        "Loss/grad_norm", "Train/grad_norm", "grad_norm", "loss/grad_norm",
        "Train/policy_grad_norm", "Train/value_grad_norm",
    ],
    "value_loss": [
        "Loss/value_loss", "Train/value_loss", "value_loss", "loss/value",
        "Loss/vf_loss",
    ],
    "policy_loss": [
        "Loss/policy_loss", "Train/policy_loss", "policy_loss", "loss/policy",
        "Loss/surrogate",
    ],
    "entropy": [
        "Train/entropy", "Loss/entropy", "entropy", "rollout/ent_coef",
        "Train/policy_entropy",
    ],
    "kl_divergence": [
        "Train/kl_divergence", "Loss/kl_divergence", "kl", "kl_divergence",
        "Train/approx_kl", "Loss/kl",
    ],
    "surrogate_ratio": [
        "Train/surrogate_ratio", "Train/clip_fraction", "surrogate_ratio",
        "Train/ratio",
    ],
    "learning_rate": [
        "Train/learning_rate", "lr", "train/learning_rate",
    ],
}

# Reverse index: tag-fragment -> canonical name (for fuzzy matching)
_TAG_FRAGMENT_INDEX: List[Tuple[str, str]] = []
for canonical, aliases in METRIC_TAG_ALIASES.items():
    for alias in aliases:
        # Use the last segment as a fragment (e.g., "mean_reward" from "Train/mean_reward")
        fragment = alias.split("/")[-1].lower()
        _TAG_FRAGMENT_INDEX.append((fragment, canonical))


def resolve_metric_name(tag: str) -> str:
    """Map a framework-specific metric tag to a canonical name.

    Returns the canonical name if recognized, else the original tag.
    """
    if tag in METRIC_TAG_ALIASES:
        return tag  # already canonical
    for canonical, aliases in METRIC_TAG_ALIASES.items():
        if tag in aliases:
            return canonical
    # Fuzzy: match by last segment. Prefer exact fragment equality,
    # then fall back to substring containment (longer fragment wins).
    fragment = tag.split("/")[-1].lower()
    # Pass 1: exact fragment match
    for frag, canonical in _TAG_FRAGMENT_INDEX:
        if frag == fragment:
            return canonical
    # Pass 2: substring containment, preferring longer fragments
    candidates = [(frag, canonical) for frag, canonical in _TAG_FRAGMENT_INDEX
                  if frag in fragment or fragment in frag]
    if candidates:
        candidates.sort(key=lambda fc: len(fc[0]), reverse=True)
        return candidates[0][1]
    return tag


# ------------------------------------------------------------------
# Which detectors apply to which canonical metric
# ------------------------------------------------------------------
METRIC_DETECTORS: Dict[str, List[str]] = {
    "reward": ["nan_inf", "sudden_drop", "plateau", "not_increasing", "oscillation"],
    "episode_length": ["nan_inf", "sudden_drop", "oscillation"],
    "grad_norm": ["nan_inf", "spike", "explosion"],
    "value_loss": ["nan_inf", "explosion"],
    "policy_loss": ["nan_inf", "explosion"],
    "entropy": ["nan_inf", "collapse", "spike"],
    "kl_divergence": ["nan_inf", "spike"],
    "surrogate_ratio": ["nan_inf", "out_of_range"],
    "learning_rate": [],
}


# ------------------------------------------------------------------
# Symptom dataclass
# ------------------------------------------------------------------
@dataclass
class Symptom:
    """A detected anomaly in a metric series.

    Attributes:
        metric: Canonical metric name (e.g., "reward", "grad_norm").
        tag: Original TensorBoard tag.
        pattern: Anomaly pattern (sudden_drop, plateau, spike, etc.).
        severity: info / warning / error.
        evidence: Human-readable description with values & step.
        step_range: (start, end) step range where anomaly is strongest.
        value_summary: Dict with peak, trough, mean, std, current, n_points.
        threshold: The threshold that triggered (if any), for transparency.
    """
    metric: str
    tag: str
    pattern: str
    severity: str
    evidence: str
    step_range: Tuple[int, int]
    value_summary: Dict[str, float] = field(default_factory=dict)
    threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "tag": self.tag,
            "pattern": self.pattern,
            "severity": self.severity,
            "evidence": self.evidence,
            "step_range": list(self.step_range),
            "value_summary": self.value_summary,
            "threshold": self.threshold,
        }


@dataclass
class AnalysisResult:
    """Result of log analysis.

    Attributes:
        symptoms: List of detected symptoms.
        metrics_analyzed: Number of metric series analyzed.
        metrics_skipped: Number of metrics skipped (too short, all NaN, etc.).
        warnings: Non-fatal issues during analysis (e.g., tensorboard missing).
    """
    symptoms: List[Symptom] = field(default_factory=list)
    metrics_analyzed: int = 0
    metrics_skipped: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def has_symptoms(self) -> bool:
        return len(self.symptoms) > 0

    @property
    def error_count(self) -> int:
        return sum(1 for s in self.symptoms if s.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for s in self.symptoms if s.severity == "warning")

    def symptoms_by_severity(self) -> List[Symptom]:
        """Return symptoms sorted by severity (error > warning > info)."""
        order = {"error": 0, "warning": 1, "info": 2}
        return sorted(self.symptoms, key=lambda s: order.get(s.severity, 3))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symptoms": [s.to_dict() for s in self.symptoms],
            "metrics_analyzed": self.metrics_analyzed,
            "metrics_skipped": self.metrics_skipped,
            "warnings": self.warnings,
        }


# ------------------------------------------------------------------
# LogAnalyzer
# ------------------------------------------------------------------
class LogAnalyzer:
    """Detects symptoms of training failure modes in metric series.

    Stateless across runs — all state is in the AnalysisResult.

    Tunables (window sizes, thresholds) are exposed as class attributes
    so users can override them for unusual training setups.
    """

    # Window sizes (in data points, not steps)
    SHORT_WINDOW = 10       # for sudden drop / spike detection
    LONG_WINDOW = 50        # for plateau / trend detection
    MIN_POINTS = 20         # need at least this many points to analyze

    # Relative thresholds (multiples of local std, or ratios)
    SUDDEN_DROP_RATIO = 0.5     # drop > 50% from recent peak → sudden_drop
    SPIKE_STDEV_MULT = 5.0      # value > 5× local std → spike
    PLATEAU_VARIANCE_RATIO = 0.01  # var/mean² < 0.01 over long window → plateau
    OSCILLATION_RATIO = 0.5     # peak-to-trough / mean > 0.5 → oscillation

    # Absolute thresholds (last-resort, only for metrics with known scales)
    EXPLOSION_ABS_THRESHOLD = 1e2   # |value| > 100 → explosion
    OUT_OF_RANGE_BOUNDS = (0.8, 1.2)  # for surrogate ratio

    def analyze(self, metrics: Dict[str, MetricSeries]) -> AnalysisResult:
        """Analyze a dict of metric series and return detected symptoms.

        Args:
            metrics: Dict mapping TensorBoard tag to (step, value) series.

        Returns:
            AnalysisResult with all detected symptoms.

        Raises:
            TypeError: If metrics is None or not a dict.
        """
        if metrics is None:
            raise TypeError("metrics must be a dict, got NoneType")
        if not isinstance(metrics, dict):
            raise TypeError(
                f"metrics must be a dict, got {type(metrics).__name__}"
            )

        result = AnalysisResult()

        if not metrics:
            result.warnings.append("No metrics provided for analysis.")
            return result

        for tag, series in metrics.items():
            if not series:
                result.metrics_skipped += 1
                continue

            canonical = resolve_metric_name(tag)
            detectors = METRIC_DETECTORS.get(canonical, [])

            # Always run nan_inf check regardless of metric type
            if "nan_inf" not in detectors:
                detectors = ["nan_inf"] + detectors

            if len(series) < self.MIN_POINTS:
                result.metrics_skipped += 1
                result.warnings.append(
                    f"Metric '{tag}' has only {len(series)} points "
                    f"(need ≥{self.MIN_POINTS}); skipped."
                )
                continue

            values = [v for _, v in series]
            if all(math.isnan(v) or math.isinf(v) for v in values):
                # All-NaN metrics are a critical failure, not a skip.
                result.metrics_analyzed += 1
                result.symptoms.append(Symptom(
                    metric=canonical, tag=tag, pattern="nan_or_inf",
                    severity="error",
                    evidence=(
                        f"All {len(values)} values are NaN/Inf. "
                        f"Training is broken (numerical divergence)."
                    ),
                    step_range=(series[0][0], series[-1][0]),
                    value_summary={"n_points": len(values)},
                ))
                continue

            result.metrics_analyzed += 1
            summary = self._summarize(values)

            for detector_name in detectors:
                symptom = self._run_detector(
                    detector_name, canonical, tag, series, values, summary
                )
                if symptom is not None:
                    result.symptoms.append(symptom)

        return result

    # ------------------------------------------------------------------
    # Value summary
    # ------------------------------------------------------------------
    @staticmethod
    def _summarize(values: Sequence[float]) -> Dict[str, float]:
        """Compute summary statistics, ignoring NaN/Inf for stats."""
        finite = [v for v in values if not math.isnan(v) and not math.isinf(v)]
        if not finite:
            return {"n_points": len(values)}

        mean = statistics.fmean(finite)
        stdev = statistics.pstdev(finite) if len(finite) > 1 else 0.0
        return {
            "n_points": len(values),
            "mean": mean,
            "stdev": stdev,
            "min": min(finite),
            "max": max(finite),
            "current": finite[-1],
            "first": finite[0],
        }

    # ------------------------------------------------------------------
    # Detector dispatch
    # ------------------------------------------------------------------
    def _run_detector(
        self,
        name: str,
        canonical: str,
        tag: str,
        series: MetricSeries,
        values: List[float],
        summary: Dict[str, float],
    ) -> Optional[Symptom]:
        method = getattr(self, f"_detect_{name}", None)
        if method is None:
            return None
        return method(canonical, tag, series, values, summary)

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------
    def _detect_nan_inf(
        self, metric: str, tag: str, series: MetricSeries,
        values: List[float], summary: Dict[str, float],
    ) -> Optional[Symptom]:
        """Detect any NaN or Inf values in the series."""
        bad_steps = [s for (s, v) in series if math.isnan(v) or math.isinf(v)]
        if not bad_steps:
            return None

        first_bad = bad_steps[0]
        last_bad = bad_steps[-1]
        return Symptom(
            metric=metric, tag=tag, pattern="nan_or_inf", severity="error",
            evidence=(
                f"{len(bad_steps)} NaN/Inf values detected; "
                f"first at step {first_bad}, last at step {last_bad}. "
                f"Training is likely stuck."
            ),
            step_range=(first_bad, last_bad),
            value_summary=summary,
        )

    def _detect_sudden_drop(
        self, metric: str, tag: str, series: MetricSeries,
        values: List[float], summary: Dict[str, float],
    ) -> Optional[Symptom]:
        """Detect a sharp drop from recent peak that doesn't recover.

        Looks back over LONG_WINDOW to find the peak, then checks if the
        current value is < SUDDEN_DROP_RATIO × peak. Using the longer
        window catches drops that happened just before the SHORT_WINDOW
        would have noticed them.

        Uses relative ratio (not absolute threshold) — robust to scale.
        """
        lookback = self.LONG_WINDOW
        if len(values) < self.SHORT_WINDOW + 2:
            return None

        # Look back up to LONG_WINDOW points for the peak (ignore NaN/Inf)
        recent_window = values[-lookback:] if len(values) >= lookback else values
        finite_recent = [v for v in recent_window if not math.isnan(v) and not math.isinf(v)]
        if len(finite_recent) < self.SHORT_WINDOW:
            return None

        # Peak is over the window EXCLUDING the very last point (which is the
        # "current" value we're checking for a drop).
        prior = finite_recent[:-1] if len(finite_recent) > 1 else finite_recent
        peak = max(prior)
        current = finite_recent[-1]

        if peak <= 0:
            return None  # can't compute ratio meaningfully

        ratio = current / peak
        if ratio > 1 - self.SUDDEN_DROP_RATIO:
            return None  # not a big enough drop

        # Find the step where the peak occurred (for step_range)
        peak_step = series[-1][0]
        for s, v in reversed(series):
            if v == peak:
                peak_step = s
                break

        step_now = series[-1][0]
        return Symptom(
            metric=metric, tag=tag, pattern="sudden_drop", severity="error",
            evidence=(
                f"Dropped from peak {peak:.4g} to {current:.4g} "
                f"({ratio:.1%} of peak); no recovery. "
                f"Step {peak_step}→{step_now}."
            ),
            step_range=(peak_step, step_now),
            value_summary=summary,
            threshold=1 - self.SUDDEN_DROP_RATIO,
        )

    def _detect_plateau(
        self, metric: str, tag: str, series: MetricSeries,
        values: List[float], summary: Dict[str, float],
    ) -> Optional[Symptom]:
        """Detect a long flat region with negligible variance.

        Uses variance/mean² ratio (relative) so plateaus are detectable
        at any scale. Triggers when ratio < PLATEAU_VARIANCE_RATIO
        over the LONG_WINDOW most recent points.
        """
        window = self.LONG_WINDOW
        if len(values) < window:
            return None

        recent = values[-window:]
        mean = statistics.fmean(recent)
        if abs(mean) < 1e-12:
            return None

        var = statistics.pvariance(recent)
        ratio = var / (mean * mean)
        if ratio > self.PLATEAU_VARIANCE_RATIO:
            return None

        step_start = series[-window][0]
        step_end = series[-1][0]
        return Symptom(
            metric=metric, tag=tag, pattern="plateau", severity="warning",
            evidence=(
                f"Flat over last {window} points: mean={mean:.4g}, "
                f"var={var:.4g}, var/mean²={ratio:.2e} "
                f"(threshold {self.PLATEAU_VARIANCE_RATIO:.0e}). "
                f"Possible local optimum or stalled learning."
            ),
            step_range=(step_start, step_end),
            value_summary=summary,
            threshold=self.PLATEAU_VARIANCE_RATIO,
        )

    def _detect_oscillation(
        self, metric: str, tag: str, series: MetricSeries,
        values: List[float], summary: Dict[str, float],
    ) -> Optional[Symptom]:
        """Detect high-frequency oscillation without trend.

        Uses peak-to-trough / |mean| ratio over LONG_WINDOW. High ratio
        + zero slope → oscillation.
        """
        window = self.LONG_WINDOW
        if len(values) < window:
            return None

        recent = values[-window:]
        mean = statistics.fmean(recent)
        if abs(mean) < 1e-12:
            return None

        amplitude = (max(recent) - min(recent)) / abs(mean)
        if amplitude < self.OSCILLATION_RATIO:
            return None

        # Check for zero trend (slope of linear fit)
        n = len(recent)
        xs = list(range(n))
        x_mean = statistics.fmean(xs)
        num = sum((x - x_mean) * (y - mean) for x, y in zip(xs, recent))
        den = sum((x - x_mean) ** 2 for x in xs)
        slope = num / den if den != 0 else 0.0

        # If there's a strong trend, it's not pure oscillation
        if abs(slope) * n > 0.5 * abs(mean):
            return None

        step_start = series[-window][0]
        step_end = series[-1][0]
        return Symptom(
            metric=metric, tag=tag, pattern="oscillation", severity="warning",
            evidence=(
                f"Oscillating: peak-to-trough/|mean| = {amplitude:.2f} "
                f"(threshold {self.OSCILLATION_RATIO}) with near-zero trend "
                f"(slope={slope:.2e}). Indicates instability."
            ),
            step_range=(step_start, step_end),
            value_summary=summary,
            threshold=self.OSCILLATION_RATIO,
        )

    def _detect_spike(
        self, metric: str, tag: str, series: MetricSeries,
        values: List[float], summary: Dict[str, float],
    ) -> Optional[Symptom]:
        """Detect a value far above local mean (scan whole series).

        Scans for the most extreme spike anywhere in the series, where
        z-score is computed against the SHORT_WINDOW preceding points.
        Reports the worst spike found. This catches spikes that happened
        mid-training even if the value has since recovered.

        Uses local stdev (relative) — robust to metric scale.
        """
        window = self.SHORT_WINDOW
        if len(values) < window + 1:
            return None

        best_spike: Optional[Tuple[int, float, float, float, float]] = None
        # best = (index, value, local_mean, local_std, z_score)

        for i in range(window, len(values)):
            prior = values[i - window:i]
            local_mean = statistics.fmean(prior)
            local_std = statistics.pstdev(prior) if len(prior) > 1 else 0.0
            current = values[i]

            if local_std < 1e-12:
                # Flat prior — can't compute z-score. Use ratio instead:
                # a 10× jump from a flat baseline is still a spike.
                if local_mean > 0 and current > local_mean * 10:
                    z_score = float("inf")
                else:
                    continue
            else:
                z_score = (current - local_mean) / local_std

            if z_score >= self.SPIKE_STDEV_MULT:
                if best_spike is None or (
                    math.isinf(z_score) and not math.isinf(best_spike[4])
                ) or (
                    (not math.isinf(z_score) or math.isinf(best_spike[4]))
                    and z_score > best_spike[4]
                ):
                    best_spike = (i, current, local_mean, local_std, z_score)

        if best_spike is None:
            return None

        idx, value, local_mean, local_std, z_score = best_spike
        step_now = series[idx][0]
        step_prior = series[max(0, idx - window)][0]
        z_str = "∞ (flat-prior jump)" if math.isinf(z_score) else f"{z_score:.1f}"
        return Symptom(
            metric=metric, tag=tag, pattern="spike", severity="warning",
            evidence=(
                f"Spike to {value:.4g} (z-score {z_str} vs local "
                f"mean {local_mean:.4g} ± {local_std:.4g} over prior "
                f"{window} points). Step {step_now}."
            ),
            step_range=(step_prior, step_now),
            value_summary=summary,
            threshold=self.SPIKE_STDEV_MULT,
        )

    def _detect_explosion(
        self, metric: str, tag: str, series: MetricSeries,
        values: List[float], summary: Dict[str, float],
    ) -> Optional[Symptom]:
        """Detect unbounded value growth or extreme magnitudes.

        Absolute threshold (EXPLOSION_ABS_THRESHOLD) — used only for
        metrics with known scale (grad_norm, value_loss). Triggered
        when |value| > 100 or grows monotonically over LONG_WINDOW.
        """
        current = values[-1]
        if not (math.isnan(current) or math.isinf(current)):
            if abs(current) < self.EXPLOSION_ABS_THRESHOLD:
                # Check monotonic growth
                window = min(self.LONG_WINDOW, len(values))
                recent = values[-window:]
                if all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1)):
                    growth_ratio = recent[-1] / (recent[0] if recent[0] != 0 else 1e-12)
                    if growth_ratio > 10:
                        step_start = series[-window][0]
                        step_end = series[-1][0]
                        return Symptom(
                            metric=metric, tag=tag, pattern="explosion", severity="error",
                            evidence=(
                                f"Monotonic growth: {recent[0]:.4g} → {recent[-1]:.4g} "
                                f"({growth_ratio:.1f}×) over {window} points. "
                                f"Step {step_start}→{step_end}."
                            ),
                            step_range=(step_start, step_end),
                            value_summary=summary,
                        )
                return None

        step_now = series[-1][0]
        return Symptom(
            metric=metric, tag=tag, pattern="explosion", severity="error",
            evidence=(
                f"Value {current:.4g} exceeds explosion threshold "
                f"{self.EXPLOSION_ABS_THRESHOLD}. Likely NaN/Inf or "
                f"unbounded growth."
            ),
            step_range=(step_now, step_now),
            value_summary=summary,
            threshold=self.EXPLOSION_ABS_THRESHOLD,
        )

    def _detect_collapse(
        self, metric: str, tag: str, series: MetricSeries,
        values: List[float], summary: Dict[str, float],
    ) -> Optional[Symptom]:
        """Detect entropy/policy collapse to near-zero.

        Entropy-specific: triggers when current < 10% of initial mean
        and stays low. Indicates policy converged to deterministic.
        """
        if len(values) < self.LONG_WINDOW:
            return None

        initial_window = values[:self.SHORT_WINDOW]
        initial_mean = statistics.fmean(initial_window)
        if initial_mean < 1e-12:
            return None

        recent = values[-self.SHORT_WINDOW:]
        recent_mean = statistics.fmean(recent)
        ratio = recent_mean / initial_mean
        if ratio > 0.1:
            return None

        step_start = series[-self.SHORT_WINDOW][0]
        step_end = series[-1][0]
        return Symptom(
            metric=metric, tag=tag, pattern="collapse", severity="warning",
            evidence=(
                f"Collapsed to {recent_mean:.4g} "
                f"({ratio:.1%} of initial {initial_mean:.4g}) over last "
                f"{self.SHORT_WINDOW} points. Policy may have lost "
                f"exploration."
            ),
            step_range=(step_start, step_end),
            value_summary=summary,
            threshold=0.1,
        )

    def _detect_out_of_range(
        self, metric: str, tag: str, series: MetricSeries,
        values: List[float], summary: Dict[str, float],
    ) -> Optional[Symptom]:
        """Detect surrogate ratio outside [0.8, 1.2] (PPO stability bound).

        Absolute bounds — these are documented PPO stability bounds, not
        scale-dependent.
        """
        lo, hi = self.OUT_OF_RANGE_BOUNDS
        out_of_range = [(s, v) for s, v in series if v < lo or v > hi]
        if not out_of_range:
            return None

        # Only flag if recent points are out of range
        recent_oor = [(s, v) for s, v in out_of_range if s >= series[-self.LONG_WINDOW][0]]
        if not recent_oor:
            return None

        step_start = recent_oor[0][0]
        step_end = recent_oor[-1][0]
        return Symptom(
            metric=metric, tag=tag, pattern="out_of_range", severity="warning",
            evidence=(
                f"{len(recent_oor)} points outside [{lo}, {hi}] "
                f"(PPO stability bounds). Most recent: {recent_oor[-1][1]:.4g} "
                f"at step {step_end}."
            ),
            step_range=(step_start, step_end),
            value_summary=summary,
            threshold=hi,
        )

    def _detect_not_increasing(
        self, metric: str, tag: str, series: MetricSeries,
        values: List[float], summary: Dict[str, float],
    ) -> Optional[Symptom]:
        """Detect reward that isn't trending upward over LONG_WINDOW.

        Reward-specific: linear regression slope ≤ 0 over the window.
        Indicates learning isn't progressing.
        """
        window = self.LONG_WINDOW
        if len(values) < window:
            return None

        recent = values[-window:]
        n = len(recent)
        xs = list(range(n))
        x_mean = statistics.fmean(xs)
        y_mean = statistics.fmean(recent)
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, recent))
        den = sum((x - x_mean) ** 2 for x in xs)
        slope = num / den if den != 0 else 0.0

        if slope > 0:
            return None

        step_start = series[-window][0]
        step_end = series[-1][0]
        return Symptom(
            metric=metric, tag=tag, pattern="not_increasing", severity="warning",
            evidence=(
                f"Negative or zero trend over last {window} points "
                f"(slope={slope:.2e}). Reward not improving."
            ),
            step_range=(step_start, step_end),
            value_summary=summary,
        )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze TensorBoard logs for symptoms of RL training failure. "
            "Outputs a JSON symptom list for the diagnosis engine."
        ),
    )
    parser.add_argument(
        "--logdir", "-l", required=True,
        help="Path to TensorBoard run directory.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON (default: human-readable).",
    )
    parser.add_argument(
        "--max-step", type=int, default=None,
        help="Only analyze events with step ≤ this value.",
    )

    args = parser.parse_args()

    try:
        from scripts.utils.tensorboard_parser import TensorboardParser
    except ImportError:
        print("Error: tensorboard_parser not found.", file=sys.stderr)
        sys.exit(1)

    if not TensorboardParser.is_available():
        print(
            "Error: tensorboard package not installed. "
            "Install with: pip install tensorboard",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        tb_parser = TensorboardParser(args.logdir)
        metrics = tb_parser.parse(max_step=args.max_step)
    except (FileNotFoundError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    analyzer = LogAnalyzer()
    result = analyzer.analyze(metrics)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Metrics analyzed: {result.metrics_analyzed}")
        print(f"Metrics skipped: {result.metrics_skipped}")
        print(f"Symptoms: {len(result.symptoms)} "
              f"({result.error_count} errors, {result.warning_count} warnings)")
        if result.warnings:
            print("\nAnalysis warnings:")
            for w in result.warnings:
                print(f"  [note] {w}")
        if result.symptoms:
            print("\nSymptoms (sorted by severity):")
            for s in result.symptoms_by_severity():
                print(f"  [{s.severity.upper():7s}] {s.metric} / {s.pattern}")
                print(f"            {s.evidence}")

    # Exit code: 0 if no errors, 1 if any error-level symptoms
    if result.error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
