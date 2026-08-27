"""Tests for tool registry and execution.

Verifies:
- All 7 tools are registered with proper schemas
- Each tool's execute() runs end-to-end against real scripts
- Args are forwarded correctly
"""
from __future__ import annotations

import pytest

from tools import all_tool_defs, execute_tool, get_executor


class TestToolRegistry:
    def test_all_seven_tools_registered(self):
        defs = all_tool_defs()
        names = [d["function"]["name"] for d in defs]
        expected = [
            "reward_synthesizer", "reward_validator", "config_validator",
            "log_analyzer", "diagnosis_engine", "dr_advisor", "curriculum_designer",
        ]
        assert sorted(names) == sorted(expected)

    def test_each_tool_has_required_schema_fields(self):
        for d in all_tool_defs():
            assert d["type"] == "function"
            assert "name" in d["function"]
            assert "description" in d["function"]
            assert "parameters" in d["function"]
            assert d["function"]["parameters"]["type"] == "object"

    def test_get_executor_returns_callable_for_known_name(self):
        fn = get_executor("reward_synthesizer")
        assert callable(fn)

    def test_get_executor_raises_for_unknown_name(self):
        with pytest.raises(KeyError):
            get_executor("nonexistent_tool")


class TestRewardSynthesizerTool:
    def test_execute_returns_complete_result(self):
        result = execute_tool("reward_synthesizer", {
            "task_description": "train quadruped to walk forward at 1 m/s, keep stable",
        })
        assert "task_type" in result
        assert result["task_type"] == "locomotion_velocity"
        assert isinstance(result["patterns"], list)
        assert len(result["patterns"]) >= 5
        assert "code" in result
        assert "validation" in result
        assert result["validation"]["valid"] is True
        # code_artifact follows 清小搭 §1 spec: fileUrl + fileName (no inline content)
        att = result["code_artifact"]
        assert att["fileName"] == "reward_generated.py"
        assert att["fileType"] == "text"
        assert att["mimeType"] == "text/x-python"
        assert att["fileUrl"].startswith("http")
        assert att["fileUrl"].endswith(".py")
        assert att["fileSize"] > 0
        assert "content" not in att


class TestRewardValidatorTool:
    def test_execute_validates_valid_code(self):
        # First generate valid code
        synth_result = execute_tool("reward_synthesizer", {
            "task_description": "quadruped walk forward",
        })
        # Then validate it
        result = execute_tool("reward_validator", {
            "code": synth_result["code"],
        })
        assert result["valid"] is True
        assert len(result["errors"]) == 0


class TestConfigValidatorTool:
    def test_execute_validates_example_env(self):
        from core.config import SKILL_ROOT
        env_path = SKILL_ROOT / "examples" / "quadruped_locomotion" / "env.py"
        result = execute_tool("config_validator", {
            "file_path": str(env_path),
        })
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_execute_returns_error_for_missing_file(self):
        result = execute_tool("config_validator", {
            "file_path": "/nonexistent/path/env.py",
        })
        assert result["valid"] is False
        assert "not found" in result["error"].lower() or "error" in result


class TestLogAnalyzerTool:
    def test_execute_on_synthetic_log(self):
        from core.config import SKILL_ROOT
        log_path = SKILL_ROOT / "tests" / "test_data" / "synthetic_policy_collapse" / "metrics.json"
        result = execute_tool("log_analyzer", {
            "metrics_file_path": str(log_path),
        })
        assert result["metrics_analyzed"] >= 3
        assert result["warning_count"] + result["error_count"] >= 3
        # Should detect sudden_drop on reward
        symptoms = result["symptoms"]
        assert any(s["pattern"] == "sudden_drop" and s["metric"] == "reward" for s in symptoms)

    def test_execute_with_inline_metrics(self):
        inline = json.dumps({
            "Train/reward": [{"step": i, "value": float(i) * 0.1} for i in range(60)],
        })
        result = execute_tool("log_analyzer", {"metrics_inline": inline})
        assert result["metrics_analyzed"] == 1


class TestDiagnosisEngineTool:
    def test_execute_with_policy_collapse_symptoms(self):
        # First run log_analyzer on synthetic log
        from core.config import SKILL_ROOT
        log_path = SKILL_ROOT / "tests" / "test_data" / "synthetic_policy_collapse" / "metrics.json"
        la_result = execute_tool("log_analyzer", {
            "metrics_file_path": str(log_path),
        })
        # Then run diagnosis on those symptoms
        result = execute_tool("diagnosis_engine", {
            "symptoms": la_result["symptoms"],
        })
        assert result["total_symptoms"] >= 3
        assert len(result["top_candidates"]) >= 1
        # Top candidate should be policy_collapse
        assert result["top_candidates"][0]["failure_mode_id"] == "policy_collapse"
        assert result["top_candidates"][0]["confidence"] >= 0.9


class TestDRAdvisorTool:
    def test_execute_returns_full_recommendation(self):
        result = execute_tool("dr_advisor", {
            "robot_type": "quadruped_medium",
            "task_type": "locomotion_velocity",
        })
        assert result["robot_type"] == "quadruped_medium"
        assert result["robot_mass_kg"] == 15.0
        assert len(result["essential_terms"]) >= 2
        assert "robot_mass" in result["essential_terms"]
        # Parameter ranges should be populated
        assert "robot_mass" in result["parameter_ranges"]
        assert "mass_range" in result["parameter_ranges"]["robot_mass"]


class TestCurriculumDesignerTool:
    def test_execute_returns_full_recommendation(self):
        result = execute_tool("curriculum_designer", {
            "task_type": "locomotion_velocity",
        })
        assert result["task_type"] == "locomotion_velocity"
        assert len(result["essential_terms"]) >= 1
        assert "command_curriculum" in result["essential_terms"]
        # term_details should have isaac_lab_func
        first_term = result["essential_terms"][0]
        assert "isaac_lab_func" in result["term_details"][first_term]


# Need json import for inline metrics test
import json
