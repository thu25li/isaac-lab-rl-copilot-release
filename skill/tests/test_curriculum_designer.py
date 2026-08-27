"""Tests for the curriculum designer.

Covers:
- CurriculumLibrary: loading and querying the curriculum pattern database
- CurriculumDesigner: task → recommended curriculum terms + parameters
- Code generation: rendered CurriculumCfg is syntactically valid
- Explanations: human-readable rationale for each recommendation
- Edge cases: unknown task, parameter overrides, empty essential list
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from scripts.curriculum_designer import (
    CurriculumLibrary,
    CurriculumDesigner,
    CurriculumRecommendation,
)


# ------------------------------------------------------------------
# CurriculumLibrary
# ------------------------------------------------------------------

class TestCurriculumLibrary:
    def test_loads_database(self):
        lib = CurriculumLibrary()
        assert lib.version == "0.1.0"

    def test_lists_curriculum_terms(self):
        lib = CurriculumLibrary()
        term_ids = lib.list_term_ids()
        assert "terrain_levels" in term_ids
        assert "command_curriculum" in term_ids
        assert "gait_curriculum" in term_ids
        assert "obstacle_density" in term_ids

    def test_get_term(self):
        lib = CurriculumLibrary()
        term = lib.get_term("terrain_levels")
        assert term["category"] == "locomotion"
        assert "isaac_lab_func" in term
        assert "pitfalls" in term

    def test_unknown_term_raises(self):
        lib = CurriculumLibrary()
        with pytest.raises(KeyError):
            lib.get_term("nonexistent")

    def test_get_task_recommendation(self):
        lib = CurriculumLibrary()
        rec = lib.get_task_recommendation("locomotion_velocity")
        assert "essential" in rec
        assert "recommended" in rec
        assert "optional" in rec
        assert "command_curriculum" in rec["essential"]

    def test_unknown_task_raises(self):
        lib = CurriculumLibrary()
        with pytest.raises(KeyError):
            lib.get_task_recommendation("unknown_task")

    def test_list_terms_by_category(self):
        lib = CurriculumLibrary()
        loc_terms = lib.list_terms(category="locomotion")
        assert len(loc_terms) >= 3
        assert all(t["category"] == "locomotion" for t in loc_terms)

    def test_list_categories(self):
        lib = CurriculumLibrary()
        cats = lib.list_categories()
        assert "locomotion" in cats
        assert "manipulation" in cats

    def test_list_task_types(self):
        lib = CurriculumLibrary()
        tasks = lib.list_task_types()
        assert "locomotion_velocity" in tasks
        assert "locomotion_rough_terrain" in tasks
        assert "manipulation_reach" in tasks

    def test_get_design_principles(self):
        lib = CurriculumLibrary()
        principles = lib.get_design_principles()
        assert isinstance(principles, dict)
        assert len(principles) > 0


# ------------------------------------------------------------------
# CurriculumDesigner: recommendation logic
# ------------------------------------------------------------------

@pytest.fixture
def designer() -> CurriculumDesigner:
    return CurriculumDesigner()


class TestDesignerRecommendation:
    def test_recommend_for_locomotion_velocity(self, designer):
        rec = designer.recommend(task_type="locomotion_velocity")
        assert rec.task_type == "locomotion_velocity"
        # Command curriculum is essential for velocity task
        assert "command_curriculum" in rec.essential_terms

    def test_recommend_for_rough_terrain(self, designer):
        rec = designer.recommend(task_type="locomotion_rough_terrain")
        # Terrain levels is essential for rough terrain
        assert "terrain_levels" in rec.essential_terms

    def test_recommend_for_manipulation(self, designer):
        rec = designer.recommend(task_type="manipulation_reach")
        # Manipulation has no essential curriculum, obstacle_density is recommended
        assert rec.essential_terms == []
        assert "obstacle_density" in rec.recommended_terms

    def test_optional_terms_excluded_by_default(self, designer):
        rec = designer.recommend(task_type="locomotion_velocity")
        # terrain_levels is optional for velocity task
        assert "terrain_levels" not in rec.essential_terms
        assert "terrain_levels" not in rec.recommended_terms

    def test_optional_terms_included_when_requested(self, designer):
        rec = designer.recommend(
            task_type="locomotion_velocity",
            include_optional=True,
        )
        assert "terrain_levels" in rec.optional_terms

    def test_parameter_overrides(self, designer):
        rec = designer.recommend(
            task_type="locomotion_velocity",
            overrides={
                "command_curriculum": {"final_range": [-2.0, 2.0]},
            },
        )
        assert rec.parameter_ranges["command_curriculum"]["final_range"] == (-2.0, 2.0)

    def test_unknown_task_raises(self, designer):
        with pytest.raises(KeyError):
            designer.recommend(task_type="unknown_task")

    def test_recommendation_has_explanation(self, designer):
        rec = designer.recommend(task_type="locomotion_velocity")
        assert isinstance(rec.explanation, str)
        assert len(rec.explanation) > 0
        assert "locomotion_velocity" in rec.explanation

    def test_recommendation_has_pitfalls(self, designer):
        rec = designer.recommend(task_type="locomotion_velocity")
        for term_id in rec.all_term_ids():
            term_info = rec.term_details.get(term_id, {})
            assert "pitfalls" in term_info
            assert len(term_info["pitfalls"]) > 0

    def test_term_details_include_isaac_lab_func(self, designer):
        rec = designer.recommend(task_type="locomotion_rough_terrain")
        for term_id in rec.essential_terms:
            details = rec.term_details[term_id]
            assert "isaac_lab_func" in details
            assert len(details["isaac_lab_func"]) > 0

    def test_parameter_ranges_use_defaults(self, designer):
        rec = designer.recommend(task_type="locomotion_rough_terrain")
        # terrain_levels has param_defaults: distance_threshold=0.5, max_level=10
        params = rec.parameter_ranges["terrain_levels"]
        assert params["distance_threshold"] == 0.5
        assert params["max_level"] == 10

    def test_uses_param_defaults_for_command_curriculum(self, designer):
        rec = designer.recommend(task_type="locomotion_velocity")
        params = rec.parameter_ranges["command_curriculum"]
        assert "initial_range" in params
        assert "final_range" in params
        assert params["initial_range"] == (-0.5, 0.5)
        assert params["final_range"] == (-1.5, 1.5)

    def test_recommendation_includes_design_principles(self, designer):
        rec = designer.recommend(task_type="locomotion_velocity")
        # Should include design principles in the explanation or as a separate field
        assert rec.design_principles is not None
        assert isinstance(rec.design_principles, dict)
        assert len(rec.design_principles) > 0

    def test_summary(self, designer):
        rec = designer.recommend(task_type="locomotion_velocity")
        summary = rec.summary()
        assert isinstance(summary, str)
        assert "locomotion_velocity" in summary


# ------------------------------------------------------------------
# Code generation
# ------------------------------------------------------------------

class TestCodeGeneration:
    def test_generates_valid_python(self, designer):
        rec = designer.recommend(task_type="locomotion_velocity")
        code = designer.generate_code(rec)
        compile(code, "<curr_code>", "exec")

    def test_generated_code_has_curriculum_cfg_class(self, designer):
        rec = designer.recommend(task_type="locomotion_velocity")
        code = designer.generate_code(rec)
        assert "class CurriculumCfg" in code
        assert "@configclass" in code

    def test_generated_code_includes_essential_terms(self, designer):
        rec = designer.recommend(task_type="locomotion_rough_terrain")
        code = designer.generate_code(rec)
        for term_id in rec.essential_terms:
            assert term_id in code, f"Term {term_id} not in generated code"

    def test_generated_code_includes_curriculum_term_import(self, designer):
        rec = designer.recommend(task_type="locomotion_velocity")
        code = designer.generate_code(rec)
        assert "CurriculumTerm" in code
        assert "import" in code

    def test_generated_code_includes_mdp_import(self, designer):
        rec = designer.recommend(task_type="locomotion_velocity")
        code = designer.generate_code(rec)
        assert "isaaclab.envs.mdp" in code

    def test_generated_code_uses_correct_funcs(self, designer):
        rec = designer.recommend(task_type="locomotion_rough_terrain")
        code = designer.generate_code(rec)
        assert "mdp.terrain_levels" in code

    def test_generated_code_has_parameter_values(self, designer):
        rec = designer.recommend(task_type="locomotion_rough_terrain")
        code = designer.generate_code(rec)
        # max_level should appear
        assert "max_level" in code

    def test_generated_code_has_header(self, designer):
        rec = designer.recommend(task_type="locomotion_velocity")
        code = designer.generate_code(rec)
        assert "Auto-generated" in code or "auto-generated" in code.lower()
        assert rec.task_type in code

    def test_generated_code_includes_comments(self, designer):
        rec = designer.recommend(task_type="locomotion_rough_terrain")
        code = designer.generate_code(rec)
        # Each term should have its purpose mentioned
        for term_id in rec.essential_terms:
            assert term_id in code

    def test_write_to_file(self, designer, tmp_path):
        rec = designer.recommend(task_type="locomotion_velocity")
        out = tmp_path / "curriculum.py"
        designer.write_code(rec, out)
        assert out.exists()
        code = out.read_text(encoding="utf-8")
        compile(code, "<curr_file>", "exec")
        assert "class CurriculumCfg" in code

    def test_generated_code_handles_empty_essential(self, designer):
        """Manipulation task has empty essential list — code should still be valid."""
        rec = designer.recommend(task_type="manipulation_reach")
        code = designer.generate_code(rec)
        compile(code, "<empty_ess>", "exec")
        assert "class CurriculumCfg" in code

    def test_generated_code_includes_design_principles(self, designer):
        rec = designer.recommend(task_type="locomotion_velocity")
        code = designer.generate_code(rec)
        # Design principles should be in the comments
        assert "consecutive" in code.lower() or "降级" in code or "downgrade" in code.lower()


# ------------------------------------------------------------------
# Explanation
# ------------------------------------------------------------------

class TestExplanation:
    def test_explanation_lists_essential_terms(self, designer):
        rec = designer.recommend(task_type="locomotion_rough_terrain")
        for term_id in rec.essential_terms:
            assert term_id in rec.explanation

    def test_explanation_mentions_task_type(self, designer):
        rec = designer.recommend(task_type="locomotion_velocity")
        assert "locomotion_velocity" in rec.explanation

    def test_explanation_includes_pitfalls(self, designer):
        rec = designer.recommend(task_type="locomotion_rough_terrain")
        assert "pitfall" in rec.explanation.lower() or "陷阱" in rec.explanation

    def test_explanation_includes_design_principles(self, designer):
        rec = designer.recommend(task_type="locomotion_velocity")
        # Design principles should be mentioned
        assert "principle" in rec.explanation.lower() or "原则" in rec.explanation \
            or "consecutive" in rec.explanation.lower() or "window" in rec.explanation.lower()


# ------------------------------------------------------------------
# Integration
# ------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_velocity(self, designer):
        rec = designer.recommend(
            task_type="locomotion_velocity",
            include_optional=True,
        )
        code = designer.generate_code(rec)
        compile(code, "<integ>", "exec")
        for term_id in rec.all_term_ids():
            assert term_id in code

    def test_full_pipeline_rough_terrain(self, designer):
        rec = designer.recommend(task_type="locomotion_rough_terrain")
        code = designer.generate_code(rec)
        compile(code, "<integ_rt>", "exec")
        assert "terrain_levels" in code

    def test_velocity_and_rough_terrain_differ(self, designer):
        rec_vel = designer.recommend(task_type="locomotion_velocity")
        rec_rt = designer.recommend(task_type="locomotion_rough_terrain")
        # Velocity's essential is command_curriculum; rough terrain's is terrain_levels
        assert "command_curriculum" in rec_vel.essential_terms
        assert "terrain_levels" in rec_rt.essential_terms
        # Their generated code should differ
        code_vel = designer.generate_code(rec_vel)
        code_rt = designer.generate_code(rec_rt)
        assert code_vel != code_rt

    def test_override_propagates_to_code(self, designer):
        rec = designer.recommend(
            task_type="locomotion_rough_terrain",
            overrides={"terrain_levels": {"max_level": 15}},
        )
        code = designer.generate_code(rec)
        assert "15" in code


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_overrides(self, designer):
        rec = designer.recommend(
            task_type="locomotion_velocity",
            overrides={},
        )
        assert "command_curriculum" in rec.essential_terms

    def test_none_overrides(self, designer):
        rec = designer.recommend(
            task_type="locomotion_velocity",
            overrides=None,
        )
        assert "command_curriculum" in rec.essential_terms

    def test_override_unknown_term_ignored(self, designer):
        rec = designer.recommend(
            task_type="locomotion_velocity",
            overrides={"nonexistent_term": {"foo": 1}},
        )
        assert "command_curriculum" in rec.essential_terms

    def test_manipulation_has_no_essential(self, designer):
        rec = designer.recommend(task_type="manipulation_reach")
        assert rec.essential_terms == []
        # But should still have recommended terms
        assert len(rec.recommended_terms) > 0

    def test_rough_terrain_has_more_essential_than_manipulation(self, designer):
        rec_rt = designer.recommend(task_type="locomotion_rough_terrain")
        rec_man = designer.recommend(task_type="manipulation_reach")
        assert len(rec_rt.essential_terms) > len(rec_man.essential_terms)

    def test_navigation_includes_terrain_levels(self, designer):
        rec = designer.recommend(task_type="navigation")
        assert "terrain_levels" in rec.essential_terms


# ------------------------------------------------------------------
# Edge cases (input boundaries, error handling)
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_reversed_range_override_raises(self, designer):
        """final_range=[2.0, -2.0] (lo > hi) should raise ValueError."""
        with pytest.raises(ValueError, match="range|lo.*hi"):
            designer.recommend(
                task_type="locomotion_velocity",
                overrides={"command_curriculum": {"final_range": [2.0, -2.0]}},
            )

    def test_reversed_initial_range_raises(self, designer):
        with pytest.raises(ValueError, match="range|lo.*hi"):
            designer.recommend(
                task_type="locomotion_velocity",
                overrides={"command_curriculum": {"initial_range": [0.5, -0.5]}},
            )

    def test_none_overrides_works(self, designer):
        """None overrides should be treated as empty — no crash."""
        rec = designer.recommend(
            task_type="locomotion_velocity",
            overrides=None,
        )
        assert "command_curriculum" in rec.essential_terms

    def test_write_code_to_directory_raises(self, designer, tmp_path):
        """Writing to a directory path should raise a clear error."""
        rec = designer.recommend(task_type="locomotion_velocity")
        with pytest.raises((IsADirectoryError, ValueError, OSError)):
            designer.write_code(rec, tmp_path)

    def test_manipulation_with_include_optional_still_valid(self, designer):
        """Manipulation has empty essential — include_optional should still produce valid code."""
        rec = designer.recommend(
            task_type="manipulation_reach",
            include_optional=True,
        )
        code = designer.generate_code(rec)
        compile(code, "<manip_opt>", "exec")
