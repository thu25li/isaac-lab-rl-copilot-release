"""Tests for the plotting module.

We verify:
- plot_metrics_with_symptoms writes a non-empty PNG
- plot_reward_weights renders synthesized reward composition
- output dir is created if missing
- empty inputs raise ValueError
- end-to-end: LogAnalyzer output → plot renders with anomalies overlaid
- end-to-end: RewardSynthesizer output → weight bar chart renders
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.log_analyzer import LogAnalyzer
from scripts.reward_synthesizer import RewardSynthesizer
from scripts.diagnosis_engine import DiagnosisEngine
from scripts.dr_advisor import DRAdvisor
from scripts.utils.plotting import (
    plot_metrics_with_symptoms,
    plot_reward_weights,
    plot_diagnosis_confidence,
    plot_dr_ranges,
    plot_curriculum_progression,
)


@pytest.fixture
def simple_metrics():
    return {
        "Train/reward": [(i, float(i) * 0.1) for i in range(0, 100, 2)],
        "Train/loss": [(i, 1.0 / (i + 1)) for i in range(0, 100, 2)],
    }


@pytest.fixture
def simple_symptoms():
    return [
        {
            "metric": "reward", "tag": "Train/reward",
            "pattern": "sudden_drop", "severity": "error",
            "evidence": "reward dropped >50%", "step_range": [80, 100],
            "value_summary": {"trough": 1.0, "trough_step": 90},
        },
    ]


class TestPlottingBasics:
    def test_renders_png(self, tmp_path: Path, simple_metrics, simple_symptoms):
        out = plot_metrics_with_symptoms(
            simple_metrics, simple_symptoms, tmp_path / "out.png"
        )
        assert out.exists()
        assert out.stat().st_size > 1000  # PNG should be substantial

    def test_creates_parent_dir(self, tmp_path: Path, simple_metrics, simple_symptoms):
        nested = tmp_path / "nested" / "deeper" / "out.png"
        out = plot_metrics_with_symptoms(simple_metrics, simple_symptoms, nested)
        assert out.exists()

    def test_empty_metrics_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="empty"):
            plot_metrics_with_symptoms([], [], tmp_path / "x.png")

    def test_no_symptoms_still_renders(self, tmp_path: Path, simple_metrics):
        out = plot_metrics_with_symptoms(simple_metrics, [], tmp_path / "ok.png")
        assert out.exists()

    def test_returns_output_path(self, tmp_path: Path, simple_metrics, simple_symptoms):
        out_path = tmp_path / "r.png"
        result = plot_metrics_with_symptoms(simple_metrics, simple_symptoms, out_path)
        assert result == out_path
        assert isinstance(result, Path)


class TestPlottingWithRealLogAnalyzer:
    def test_end_to_end_policy_collapse(self, tmp_path: Path):
        """LogAnalyzer on synthetic_policy_collapse → plot shows the crash."""
        log_path = (
            Path(__file__).parent / "test_data" / "synthetic_policy_collapse" / "metrics.json"
        )
        metrics = json.loads(log_path.read_text(encoding="utf-8"))
        # Convert {tag: [{step, value}, ...]} → {tag: [(step, value), ...]}
        series = {
            tag: [(p["step"], p["value"]) for p in points]
            for tag, points in metrics.items()
        }

        analyzer = LogAnalyzer()
        result = analyzer.analyze(series)
        symptoms = [s.to_dict() for s in result.symptoms]
        assert len(symptoms) >= 3, "expected several symptoms from policy_collapse log"

        out = plot_metrics_with_symptoms(
            series, symptoms, tmp_path / "diagnosis.png",
            title="synthetic_policy_collapse",
        )
        assert out.exists()
        assert out.stat().st_size > 5000


class TestRewardWeightPlot:
    def test_renders_png_with_weights(self, tmp_path: Path):
        patterns = ["linear_velocity_tracking", "flat_orientation_l2", "energy_penalty"]
        config = {
            "linear_velocity_tracking": {"weight": 1.0},
            "flat_orientation_l2": {"weight": -2.5},
            "energy_penalty": {"weight": -0.001},
        }
        out = plot_reward_weights(patterns, config, tmp_path / "weights.png")
        assert out.exists()
        assert out.stat().st_size > 2000

    def test_empty_patterns_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="empty"):
            plot_reward_weights([], {}, tmp_path / "x.png")

    def test_all_missing_weights_raises(self, tmp_path: Path):
        # patterns listed but none have weight in config
        with pytest.raises(ValueError, match="no pattern has a numeric weight"):
            plot_reward_weights(["foo"], {"foo": {"std": 0.1}}, tmp_path / "x.png")

    def test_end_to_end_with_synthesizer(self, tmp_path: Path):
        """Synthesizer output → bar chart renders."""
        result = RewardSynthesizer().synthesize(
            "train quadruped to walk forward at 1 m/s, keep stable",
            validate=True,
        )
        out = plot_reward_weights(
            result.patterns, result.config, tmp_path / "weights.png",
        )
        assert out.exists()
        assert out.stat().st_size > 3000
        # Multiple terms should produce a multi-row chart
        assert len(result.patterns) >= 5


class TestDiagnosisConfidencePlot:
    def test_renders_png(self, tmp_path: Path):
        cands = [
            {"failure_mode_id": "policy_collapse", "confidence": 1.0,
             "matched": ["a", "b", "c"], "total_expected": 5},
            {"failure_mode_id": "grad_explosion", "confidence": 0.33,
             "matched": ["d"], "total_expected": 5},
            {"failure_mode_id": "entropy_collapse", "confidence": 0.25,
             "matched": ["e"], "total_expected": 5},
        ]
        out = plot_diagnosis_confidence(cands, tmp_path / "diag.png")
        assert out.exists()
        assert out.stat().st_size > 2000

    def test_empty_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="empty"):
            plot_diagnosis_confidence([], tmp_path / "x.png")

    def test_min_confidence_filter_raises_when_all_filtered(self, tmp_path: Path):
        cands = [{"failure_mode_id": "x", "confidence": 0.1,
                  "matched": [], "total_expected": 1}]
        with pytest.raises(ValueError, match="no candidates above"):
            plot_diagnosis_confidence(cands, tmp_path / "x.png", min_confidence=0.5)

    def test_end_to_end_with_engine(self, tmp_path: Path):
        """DiagnosisEngine on policy_collapse → chart renders."""
        import json
        log_path = (
            Path(__file__).parent / "test_data" / "synthetic_policy_collapse" / "metrics.json"
        )
        raw = json.loads(log_path.read_text(encoding="utf-8"))
        series = {tag: [(p["step"], p["value"]) for p in pts] for tag, pts in raw.items()}
        analysis = LogAnalyzer().analyze(series)
        diag = DiagnosisEngine().diagnose(analysis.symptoms)
        cands = [c.to_dict() for c in diag.top_candidates(5)]
        out = plot_diagnosis_confidence(cands, tmp_path / "diag.png")
        assert out.exists()
        assert out.stat().st_size > 3000
        # Top candidate should be policy_collapse with high confidence
        assert cands[0]["failure_mode_id"] == "policy_collapse"
        assert cands[0]["confidence"] >= 0.9


class TestDRRangePlot:
    def test_renders_png(self, tmp_path: Path):
        essential = ["robot_mass", "ground_friction"]
        recommended = ["motor_strength"]
        optional = ["obs_noise"]
        ranges = {
            "robot_mass": {"mass_range": (-3.0, 3.0)},
            "ground_friction": {"friction_range": (0.3, 1.1)},
            "motor_strength": {"strength_range": (-0.2, 0.2)},
            "obs_noise": {"noise_std": (0.0, 0.05)},
        }
        out = plot_dr_ranges(essential, recommended, optional, ranges,
                             tmp_path / "dr.png")
        assert out.exists()
        assert out.stat().st_size > 2000

    def test_empty_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="no DR terms"):
            plot_dr_ranges([], [], [], {}, tmp_path / "x.png")

    def test_no_numeric_ranges_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="no term.*numeric"):
            plot_dr_ranges(["foo"], [], [], {"foo": {"note": "no range"}}, tmp_path / "x.png")

    def test_end_to_end_with_advisor(self, tmp_path: Path):
        rec = DRAdvisor().recommend(
            robot_type="quadruped_medium", task_type="locomotion_velocity",
        )
        out = plot_dr_ranges(
            rec.essential_terms, rec.recommended_terms, rec.optional_terms,
            rec.parameter_ranges, tmp_path / "dr.png",
        )
        assert out.exists()
        assert out.stat().st_size > 3000


class TestCurriculumProgressionPlot:
    def test_renders_png(self, tmp_path: Path):
        essential = ["command_curriculum"]
        recommended = ["gait_curriculum"]
        optional = []
        details = {
            "command_curriculum": {
                "name": "Command Curriculum",
                "isaac_lab_func": "mdp.command_curriculum",
                "param_defaults": {
                    "initial_range": [-0.5, 0.5],
                    "final_range": [-1.5, 1.5],
                },
            },
            "gait_curriculum": {
                "name": "Gait Curriculum",
                "isaac_lab_func": "mdp.gait_curriculum",
                "param_defaults": {
                    "initial_range": [0.0, 0.2],
                    "final_range": [0.0, 1.0],
                },
            },
        }
        out = plot_curriculum_progression(
            essential, recommended, optional, details, tmp_path / "curr.png"
        )
        assert out.exists()
        assert out.stat().st_size > 2000

    def test_empty_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="no curriculum terms"):
            plot_curriculum_progression([], [], [], {}, tmp_path / "x.png")

    def test_no_ranges_falls_back_to_term_list(self, tmp_path: Path):
        """When no term has initial/final_range, plot falls back to term-list view."""
        out = plot_curriculum_progression(
            ["foo"], [], [],
            {"foo": {"name": "Foo Term", "isaac_lab_func": "mdp.foo",
                     "param_defaults": {}}},
            tmp_path / "fallback.png",
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_end_to_end_with_designer(self, tmp_path: Path):
        from scripts.curriculum_designer import CurriculumDesigner
        # locomotion_velocity has command_curriculum with initial/final_range
        rec = CurriculumDesigner().recommend(task_type="locomotion_velocity")
        out = plot_curriculum_progression(
            rec.essential_terms, rec.recommended_terms, rec.optional_terms,
            rec.term_details, tmp_path / "curr.png",
        )
        assert out.exists()
        assert out.stat().st_size > 2000
