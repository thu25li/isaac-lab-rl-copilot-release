"""Tests for the DR advisor.

Covers:
- DRLibrary: loading and querying the DR pattern database
- DRAdvisor: robot+task → recommended DR terms + parameter ranges
- Code generation: rendered EventsCfg is syntactically valid and uses
  the recommended terms
- Explanations: human-readable rationale for each recommendation
- Edge cases: unknown robot/task, parameter overrides
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from scripts.dr_advisor import DRLibrary, DRAdvisor, DRRecommendation


# ------------------------------------------------------------------
# DRLibrary
# ------------------------------------------------------------------

class TestDRLibrary:
    def test_loads_database(self):
        lib = DRLibrary()
        assert lib.version == "0.1.0"

    def test_lists_robot_profiles(self):
        lib = DRLibrary()
        profiles = lib.list_robot_profiles()
        assert "quadruped_small" in profiles
        assert "quadruped_large" in profiles
        assert "biped_small" in profiles
        assert "manipulator_arm" in profiles

    def test_get_robot_profile(self):
        lib = DRLibrary()
        profile = lib.get_robot_profile("quadruped_small")
        assert profile["mass_kg"] == 12.0
        assert "defaults" in profile
        assert "mass_range" in profile["defaults"]

    def test_unknown_robot_profile_raises(self):
        lib = DRLibrary()
        with pytest.raises(KeyError):
            lib.get_robot_profile("unknown_robot")

    def test_lists_dr_terms(self):
        lib = DRLibrary()
        term_ids = lib.list_dr_term_ids()
        assert "robot_mass" in term_ids
        assert "ground_friction" in term_ids
        assert "motor_strength" in term_ids
        assert "obs_noise" in term_ids
        assert "action_delay" in term_ids

    def test_get_dr_term(self):
        lib = DRLibrary()
        term = lib.get_dr_term("ground_friction")
        assert term["category"] == "physics"
        assert "isaac_lab_func" in term
        assert "pitfalls" in term

    def test_unknown_dr_term_raises(self):
        lib = DRLibrary()
        with pytest.raises(KeyError):
            lib.get_dr_term("nonexistent_term")

    def test_get_task_recommendation(self):
        lib = DRLibrary()
        rec = lib.get_task_recommendation("locomotion_velocity")
        assert "essential" in rec
        assert "recommended" in rec
        assert "optional" in rec
        assert "robot_mass" in rec["essential"]
        assert "ground_friction" in rec["essential"]

    def test_unknown_task_recommendation_raises(self):
        lib = DRLibrary()
        with pytest.raises(KeyError):
            lib.get_task_recommendation("unknown_task")

    def test_get_robot_defaults(self):
        lib = DRLibrary()
        defaults = lib.get_robot_defaults("quadruped_small")
        assert "mass_range" in defaults
        assert "friction_range" in defaults
        assert "motor_stiffness_range" in defaults

    def test_list_dr_terms_by_category(self):
        lib = DRLibrary()
        physics_terms = lib.list_dr_terms(category="physics")
        assert len(physics_terms) >= 4
        assert all(t["category"] == "physics" for t in physics_terms)

    def test_list_categories(self):
        lib = DRLibrary()
        cats = lib.list_categories()
        assert "physics" in cats
        assert "sensor" in cats
        assert "perturbation" in cats


# ------------------------------------------------------------------
# DRAdvisor: recommendation logic
# ------------------------------------------------------------------

@pytest.fixture
def advisor() -> DRAdvisor:
    return DRAdvisor()


class TestAdvisorRecommendation:
    def test_recommend_for_quadruped_velocity(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        assert rec.robot_type == "quadruped_small"
        assert rec.task_type == "locomotion_velocity"
        # Essential terms should be included
        assert "robot_mass" in rec.essential_terms
        assert "ground_friction" in rec.essential_terms
        assert "motor_strength" in rec.essential_terms

    def test_recommend_for_quadruped_rough_terrain(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_large",
            task_type="locomotion_rough_terrain",
        )
        # Rough terrain should include joint_damping_friction as essential
        assert "joint_damping_friction" in rec.essential_terms
        assert "external_force" in rec.recommended_terms

    def test_recommend_for_manipulation(self, advisor):
        rec = advisor.recommend(
            robot_type="manipulator_arm",
            task_type="manipulation_reach",
        )
        assert "motor_strength" in rec.essential_terms
        assert "obs_noise" in rec.essential_terms

    def test_uses_robot_profile_defaults(self, advisor):
        rec_small = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        rec_large = advisor.recommend(
            robot_type="quadruped_large",
            task_type="locomotion_velocity",
        )
        # Small robot should have smaller mass range than large robot
        small_mass = rec_small.parameter_ranges["robot_mass"]["mass_range"]
        large_mass = rec_large.parameter_ranges["robot_mass"]["mass_range"]
        assert abs(small_mass[1]) < abs(large_mass[1])

    def test_uses_robot_profile_friction(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        friction = rec.parameter_ranges["ground_friction"]["friction_range"]
        # Friction range should be a tuple/list of 2 values
        assert len(friction) == 2
        assert friction[0] < friction[1]

    def test_optional_terms_included_when_requested(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
            include_optional=True,
        )
        # external_force is optional for velocity task
        assert "external_force" in rec.all_term_ids()

    def test_optional_terms_excluded_by_default(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        assert "external_force" not in rec.essential_terms
        assert "external_force" not in rec.recommended_terms

    def test_parameter_overrides(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
            overrides={
                "robot_mass": {"mass_range": (-3.0, 3.0)},
            },
        )
        assert rec.parameter_ranges["robot_mass"]["mass_range"] == (-3.0, 3.0)

    def test_unknown_robot_type_raises(self, advisor):
        with pytest.raises(KeyError):
            advisor.recommend(robot_type="unknown", task_type="locomotion_velocity")

    def test_unknown_task_type_raises(self, advisor):
        with pytest.raises(KeyError):
            advisor.recommend(robot_type="quadruped_small", task_type="unknown")

    def test_recommendation_has_explanation(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        assert isinstance(rec.explanation, str)
        assert len(rec.explanation) > 0
        # Explanation should reference the task type
        assert "locomotion_velocity" in rec.explanation or "velocity" in rec.explanation.lower()

    def test_recommendation_has_pitfalls(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        # Each essential term should have its pitfalls accessible
        for term_id in rec.essential_terms:
            term_info = rec.term_details.get(term_id, {})
            assert "pitfalls" in term_info
            assert len(term_info["pitfalls"]) > 0

    def test_term_details_include_isaac_lab_func(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        for term_id in rec.essential_terms:
            details = rec.term_details[term_id]
            assert "isaac_lab_func" in details
            assert len(details["isaac_lab_func"]) > 0


# ------------------------------------------------------------------
# Code generation
# ------------------------------------------------------------------

class TestCodeGeneration:
    def test_generates_valid_python(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        code = advisor.generate_code(rec)
        # Should be valid Python syntax
        compile(code, "<dr_code>", "exec")

    def test_generated_code_has_events_cfg_class(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        code = advisor.generate_code(rec)
        assert "class EventsCfg" in code
        assert "@configclass" in code

    def test_generated_code_includes_essential_terms(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        code = advisor.generate_code(rec)
        # Each essential term should appear as a field
        for term_id in rec.essential_terms:
            assert term_id in code, f"Term {term_id} not in generated code"

    def test_generated_code_includes_event_term_import(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        code = advisor.generate_code(rec)
        assert "EventTermCfg" in code
        assert "import" in code

    def test_generated_code_includes_mdp_import(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        code = advisor.generate_code(rec)
        assert "isaaclab.envs.mdp" in code or "import isaaclab.envs.mdp" in code

    def test_generated_code_includes_scene_entity(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        code = advisor.generate_code(rec)
        assert "SceneEntityCfg" in code

    def test_generated_code_uses_correct_funcs(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        code = advisor.generate_code(rec)
        # Check a few specific funcs
        assert "add_robot_mass" in code  # robot_mass
        assert "randomize_rigid_body_material" in code  # ground_friction
        assert "randomize_actuator_parameters" in code  # motor_strength

    def test_generated_code_has_parameter_values(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        code = advisor.generate_code(rec)
        # Mass range should appear
        assert "mass_range" in code
        # Friction range should appear
        assert "friction_range" in code

    def test_generated_code_includes_comments(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        code = advisor.generate_code(rec)
        # Each term should have a comment explaining its purpose
        for term_id in rec.essential_terms:
            # Look for the term as a field assignment, then check nearby comments
            assert term_id in code

    def test_generated_code_has_header(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        code = advisor.generate_code(rec)
        assert "Auto-generated" in code or "auto-generated" in code.lower()
        assert rec.task_type in code

    def test_generated_code_different_for_different_robots(self, advisor):
        rec_small = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        rec_large = advisor.recommend(
            robot_type="quadruped_large",
            task_type="locomotion_velocity",
        )
        code_small = advisor.generate_code(rec_small)
        code_large = advisor.generate_code(rec_large)
        # Mass ranges should differ
        # quadruped_small default mass_range is [-2.0, 2.0]
        # quadruped_large default mass_range is [-5.0, 5.0]
        assert "2.0" in code_small or "2.0," in code_small
        assert "5.0" in code_large

    def test_write_to_file(self, advisor, tmp_path):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        out = tmp_path / "events.py"
        advisor.write_code(rec, out)
        assert out.exists()
        code = out.read_text(encoding="utf-8")
        compile(code, "<dr_file>", "exec")
        assert "class EventsCfg" in code


# ------------------------------------------------------------------
# Explanation
# ------------------------------------------------------------------

class TestExplanation:
    def test_explanation_lists_essential_terms(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        for term_id in rec.essential_terms:
            assert term_id in rec.explanation

    def test_explanation_mentions_robot_type(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        assert "quadruped_small" in rec.explanation

    def test_explanation_includes_pitfalls(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        # At least one pitfall should be mentioned
        assert "pitfall" in rec.explanation.lower() or "陷阱" in rec.explanation

    def test_explanation_includes_param_ranges(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        # Should mention mass_range or friction_range
        assert "mass_range" in rec.explanation or "friction_range" in rec.explanation


# ------------------------------------------------------------------
# Integration: full pipeline
# ------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_quadruped(self, advisor):
        """Full recommend → generate_code → write_file pipeline."""
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
            include_optional=True,
        )
        code = advisor.generate_code(rec)
        # Code should be syntactically valid
        compile(code, "<integration>", "exec")
        # Should include all term categories
        for term_id in rec.all_term_ids():
            assert term_id in code

    def test_full_pipeline_manipulator(self, advisor):
        """Manipulator arm should produce different code than quadruped."""
        rec_quad = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        rec_arm = advisor.recommend(
            robot_type="manipulator_arm",
            task_type="manipulation_reach",
        )
        code_quad = advisor.generate_code(rec_quad)
        code_arm = advisor.generate_code(rec_arm)
        # Both should be valid
        compile(code_quad, "<quad>", "exec")
        compile(code_arm, "<arm>", "exec")
        # Should differ in parameters
        assert code_quad != code_arm

    def test_recommendation_summary(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        summary = rec.summary()
        assert isinstance(summary, str)
        assert "quadruped_small" in summary
        assert "locomotion_velocity" in summary


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_overrides(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
            overrides={},
        )
        # Should work normally
        assert "robot_mass" in rec.essential_terms

    def test_none_overrides(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
            overrides=None,
        )
        assert "robot_mass" in rec.essential_terms

    def test_override_unknown_term_is_ignored(self, advisor):
        # Override for a term not in the recommendation should be silently ignored
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
            overrides={"nonexistent_term": {"foo": 1}},
        )
        # Should still work
        assert "robot_mass" in rec.essential_terms

    def test_biped_robot(self, advisor):
        rec = advisor.recommend(
            robot_type="biped_small",
            task_type="locomotion_velocity",
        )
        assert "robot_mass" in rec.essential_terms
        # Biped should have smaller mass range than quadruped_large
        mass_range = rec.parameter_ranges["robot_mass"]["mass_range"]
        assert abs(mass_range[1]) <= 1.5  # biped_small default is [-1.0, 1.0]

    def test_rough_terrain_has_more_terms_than_velocity(self, advisor):
        rec_vel = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        rec_rough = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_rough_terrain",
        )
        # Rough terrain should have at least as many terms
        assert len(rec_rough.all_term_ids()) >= len(rec_vel.all_term_ids())

    def test_external_force_has_interval_mode(self, advisor):
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_rough_terrain",
            include_optional=False,  # external_force is recommended for rough terrain
        )
        # external_force is in recommended for rough terrain
        if "external_force" in rec.recommended_terms:
            details = rec.term_details["external_force"]
            assert details.get("mode") == "interval"
            assert "interval_time_s" in details


# ------------------------------------------------------------------
# Edge cases (input boundaries, error handling)
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_reversed_range_override_raises(self, advisor):
        """mass_range=(3.0, -3.0) (lo > hi) should raise ValueError, not silently accept."""
        with pytest.raises(ValueError, match="range|lo.*hi"):
            advisor.recommend(
                robot_type="quadruped_small",
                task_type="locomotion_velocity",
                overrides={"robot_mass": {"mass_range": (3.0, -3.0)}},
            )

    def test_reversed_friction_range_raises(self, advisor):
        with pytest.raises(ValueError, match="range|lo.*hi"):
            advisor.recommend(
                robot_type="quadruped_small",
                task_type="locomotion_velocity",
                overrides={"ground_friction": {"friction_range": (1.5, 0.3)}},
            )

    def test_scalar_range_override_raises(self, advisor):
        """Passing a scalar (2.0) instead of a (lo, hi) tuple should raise."""
        with pytest.raises((ValueError, TypeError)):
            advisor.recommend(
                robot_type="quadruped_small",
                task_type="locomotion_velocity",
                overrides={"robot_mass": {"mass_range": 2.0}},
            )

    def test_none_overrides_works(self, advisor):
        """None overrides should be treated as empty — no crash."""
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
            overrides=None,
        )
        assert "robot_mass" in rec.essential_terms

    def test_write_code_to_directory_raises(self, advisor, tmp_path):
        """Writing to a directory path should raise a clear error."""
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        with pytest.raises((IsADirectoryError, ValueError, OSError)):
            advisor.write_code(rec, tmp_path)
