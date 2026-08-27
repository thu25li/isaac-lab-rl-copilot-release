"""Tests for the reward synthesizer (module 1 core).

Covers:
- Basic synthesis from NL task description
- Task type detection (velocity vs rough_terrain)
- Pattern selection (required/recommended/optional)
- Config overrides
- File output
- Validation integration
- Explanation generation
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# Ensure package root is on sys.path so `from scripts...` works
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from scripts.reward_synthesizer import RewardSynthesizer, SynthesisResult
from scripts.utils.pattern_matcher import PatternMatcher
from scripts.reward_library import RewardLibrary


@pytest.fixture
def synthesizer() -> RewardSynthesizer:
    return RewardSynthesizer()


@pytest.fixture
def velocity_task() -> str:
    return "train quadruped to walk forward at 1 m/s, keep stable"


@pytest.fixture
def rough_terrain_task() -> str:
    return "train quadruped to walk over rough terrain with stairs"


@pytest.fixture
def chinese_task() -> str:
    return "训练四足以 1 m/s 前进，保持身体水平稳定"


# ------------------------------------------------------------------
# Task type detection
# ------------------------------------------------------------------

class TestTaskTypeDetection:
    def test_velocity_task_detected(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert result.task_type == "locomotion_velocity"

    def test_rough_terrain_task_detected(self, synthesizer, rough_terrain_task):
        result = synthesizer.synthesize(rough_terrain_task)
        assert result.task_type == "locomotion_rough_terrain"

    def test_chinese_velocity_task_detected(self, synthesizer, chinese_task):
        result = synthesizer.synthesize(chinese_task)
        assert result.task_type == "locomotion_velocity"

    def test_chinese_rough_terrain_detected(self, synthesizer):
        result = synthesizer.synthesize("训练四足爬楼梯、走过崎岖地形")
        assert result.task_type == "locomotion_rough_terrain"

    def test_unknown_task_defaults_to_velocity(self, synthesizer):
        result = synthesizer.synthesize("make the robot do things")
        assert result.task_type == "locomotion_velocity"


# ------------------------------------------------------------------
# Pattern selection
# ------------------------------------------------------------------

class TestPatternSelection:
    def test_velocity_includes_required_patterns(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert "linear_velocity_tracking" in result.patterns
        assert "angular_velocity_tracking" in result.patterns

    def test_velocity_includes_recommended_patterns(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        # Standard stability penalties should be in recommended
        assert "lin_vel_z_l2" in result.patterns
        assert "ang_vel_xy_l2" in result.patterns
        assert "joint_torques_l2" in result.patterns
        assert "action_rate_l2" in result.patterns
        assert "feet_air_time" in result.patterns

    def test_velocity_excludes_optional_by_default(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert "joint_pos_limits" not in result.patterns
        assert "is_terminated" not in result.patterns

    def test_optional_included_when_flag_set(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task, include_optional=True)
        assert "joint_pos_limits" in result.patterns
        assert "is_terminated" in result.patterns

    def test_rough_terrain_includes_flat_orientation(self, synthesizer, rough_terrain_task):
        result = synthesizer.synthesize(rough_terrain_task)
        # Rough terrain recommends flat_orientation_l2 (always on, not optional)
        assert "flat_orientation_l2" in result.patterns

    def test_all_patterns_exist_in_library(self, synthesizer, velocity_task):
        lib = RewardLibrary()
        result = synthesizer.synthesize(velocity_task)
        for pid in result.patterns:
            assert lib.validate_pattern_id(pid), f"Pattern {pid} not in library"


# ------------------------------------------------------------------
# Config & overrides
# ------------------------------------------------------------------

class TestConfig:
    def test_default_weights_assigned(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        cfg = result.config
        assert cfg["linear_velocity_tracking"]["weight"] == 1.0
        assert cfg["angular_velocity_tracking"]["weight"] == 0.5
        assert cfg["lin_vel_z_l2"]["weight"] == -2.0
        assert cfg["joint_torques_l2"]["weight"] == -1.0e-5

    def test_default_params_assigned(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        cfg = result.config["linear_velocity_tracking"]
        assert cfg["std"] == 0.25
        assert cfg["command_name"] == "base_velocity"

    def test_weight_override_applied(self, synthesizer, velocity_task):
        overrides = {"linear_velocity_tracking": {"weight": 2.0}}
        result = synthesizer.synthesize(velocity_task, config_overrides=overrides)
        assert result.config["linear_velocity_tracking"]["weight"] == 2.0

    def test_param_override_applied(self, synthesizer, velocity_task):
        overrides = {"linear_velocity_tracking": {"std": 0.5}}
        result = synthesizer.synthesize(velocity_task, config_overrides=overrides)
        assert result.config["linear_velocity_tracking"]["std"] == 0.5

    def test_override_does_not_mutate_defaults(self, synthesizer, velocity_task):
        overrides = {"linear_velocity_tracking": {"weight": 99.0}}
        synthesizer.synthesize(velocity_task, config_overrides=overrides)
        # Re-run without overrides; weight should be back to default
        result = synthesizer.synthesize(velocity_task)
        assert result.config["linear_velocity_tracking"]["weight"] == 1.0


# ------------------------------------------------------------------
# Code generation
# ------------------------------------------------------------------

class TestCodeGeneration:
    def test_code_is_nonempty_string(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert isinstance(result.code, str)
        assert len(result.code) > 100

    def test_code_contains_required_imports(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert "configclass" in result.code
        assert "RewTerm" in result.code
        assert "import isaaclab.envs.mdp as mdp" in result.code

    def test_code_contains_rewards_cfg_class(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert "@configclass" in result.code
        assert "class RewardsCfg:" in result.code

    def test_code_contains_velocity_tracking_terms(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert "track_lin_vel_xy_exp" in result.code
        assert "track_ang_vel_z_exp" in result.code
        assert "mdp.track_lin_vel_xy_exp" in result.code

    def test_code_contains_task_mdp_import_when_feet_air_time_present(
        self, synthesizer, velocity_task
    ):
        result = synthesizer.synthesize(velocity_task)
        # feet_air_time is in recommended for velocity task
        assert "import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp_loc" in result.code
        assert "mdp_loc.feet_air_time" in result.code

    def test_code_contains_timestamp(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert "Generated:" in result.code
        assert result.timestamp  # non-empty

    def test_code_contains_task_description(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert velocity_task in result.code

    def test_code_is_valid_python_syntax(self, synthesizer, velocity_task):
        # Critical: generated code must at least parse
        result = synthesizer.synthesize(velocity_task)
        compile(result.code, "<test>", "exec")


# ------------------------------------------------------------------
# File output
# ------------------------------------------------------------------

class TestFileOutput:
    def test_writes_to_file(self, synthesizer, velocity_task):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "reward.py"
            result = synthesizer.synthesize(velocity_task, output_path=out_path)
            assert out_path.exists()
            assert result.output_path == out_path
            written = out_path.read_text(encoding="utf-8")
            assert written == result.code

    def test_no_file_when_output_path_none(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert result.output_path is None


# ------------------------------------------------------------------
# Validation integration
# ------------------------------------------------------------------

class TestValidationIntegration:
    def test_validation_runs_when_requested(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task, validate=True)
        assert result.validation is not None
        assert "valid" in result.validation

    def test_no_validation_when_not_requested(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert result.validation is None

    def test_generated_code_passes_validation(self, synthesizer, velocity_task):
        # The synthesizer's own output must pass the static validator.
        # This is the end-to-end correctness check for module 1.
        result = synthesizer.synthesize(velocity_task, validate=True)
        assert result.validation is not None
        assert result.validation["valid"], (
            f"Generated code failed validation:\n"
            f"Errors: {result.validation.get('errors', [])}\n"
            f"Code:\n{result.code}"
        )


# ------------------------------------------------------------------
# Explanation
# ------------------------------------------------------------------

class TestExplanation:
    def test_explanation_contains_task_type(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert "locomotion_velocity" in result.explanation

    def test_explanation_lists_patterns(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert "linear_velocity_tracking" in result.explanation
        assert "Required patterns" in result.explanation

    def test_explanation_shows_weights(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert "weight=" in result.explanation


# ------------------------------------------------------------------
# SynthesisResult dataclass
# ------------------------------------------------------------------

class TestSynthesisResult:
    def test_summary_string(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task, validate=True)
        s = result.summary()
        assert "SynthesisResult" in s
        assert "locomotion_velocity" in s

    def test_patterns_is_list(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert isinstance(result.patterns, list)
        assert len(result.patterns) >= 2

    def test_config_is_dict(self, synthesizer, velocity_task):
        result = synthesizer.synthesize(velocity_task)
        assert isinstance(result.config, dict)
        for pid in result.patterns:
            assert pid in result.config


# ------------------------------------------------------------------
# Edge cases (input boundaries, error handling)
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_none_task_raises_type_error(self, synthesizer):
        with pytest.raises(TypeError, match="task_description"):
            synthesizer.synthesize(None)

    def test_empty_string_task_defaults_to_velocity(self, synthesizer):
        """Empty string should not crash — falls back to default task type."""
        result = synthesizer.synthesize("")
        assert result.task_type == "locomotion_velocity"
        assert len(result.patterns) > 0

    def test_task_with_special_chars_generates_valid_code(self, synthesizer):
        """Task description with quotes/newlines should not break generated code syntax."""
        tricky = 'train robot to "walk" with\nnewline and `backtick`'
        result = synthesizer.synthesize(tricky, validate=True)
        # Generated code must still compile
        compile(result.code, "<tricky>", "exec")
        assert result.validation["valid"] is True

    def test_override_with_unknown_pattern_id_is_ignored(self, synthesizer, velocity_task):
        """Override for a pattern not in selection should be silently ignored."""
        overrides = {"nonexistent_pattern": {"weight": 5.0}}
        result = synthesizer.synthesize(velocity_task, config_overrides=overrides)
        # Should not crash, should still produce valid output
        assert len(result.patterns) > 0
        assert "nonexistent_pattern" not in result.patterns

    def test_output_to_directory_path_raises(self, synthesizer, velocity_task, tmp_path):
        """output_path pointing to a directory should raise a clear error."""
        with pytest.raises((IsADirectoryError, ValueError, OSError)):
            synthesizer.synthesize(velocity_task, output_path=tmp_path)
