#!/usr/bin/env python3
"""DR advisor — domain randomization recommendation engine.

Given a robot type and task type, recommends a curated set of Isaac Lab
EventTermCfg-based DR patterns with parameter ranges tailored to the
robot's mass class and the task's robustness requirements.

Pipeline:
1. Look up robot profile → default parameter ranges (mass, friction, etc.)
2. Look up task recommendation → which DR terms (essential/recommended/optional)
3. For each recommended term, fetch its definition (isaac_lab_func, params, pitfalls)
4. Apply user overrides (e.g., widen mass_range for harsher sim-to-real)
5. Render events_cfg.py.tmpl via CodeEmitter

Usage (CLI):
    python scripts/dr_advisor.py \\
        --robot quadruped_small \\
        --task locomotion_velocity \\
        --output events.py --include-optional --explain

Usage (library):
    from scripts.dr_advisor import DRAdvisor
    advisor = DRAdvisor()
    rec = advisor.recommend(robot_type="quadruped_small", task_type="locomotion_velocity")
    print(advisor.generate_code(rec))
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow direct script execution
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from scripts.utils.code_emitter import CodeEmitter


# ------------------------------------------------------------------
# DRLibrary: programmatic access to dr_patterns.json
# ------------------------------------------------------------------


class DRLibrary:
    """Provides access to the curated DR pattern database.

    The database is stored as JSON at resources/dr_patterns.json. It
    contains:
    - robot_profiles: per-robot default parameter ranges
    - dr_terms: DR term definitions (isaac_lab_func, params, pitfalls)
    - task_recommendations: which DR terms to use per task type

    Example:
        lib = DRLibrary()
        profile = lib.get_robot_profile("quadruped_small")
        term = lib.get_dr_term("ground_friction")
        rec = lib.get_task_recommendation("locomotion_velocity")
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize library with DR pattern database path.

        Args:
            db_path: Path to dr_patterns.json. If None, resolves to
                the package's resources/dr_patterns.json.

        Raises:
            FileNotFoundError: If database file does not exist.
        """
        import json

        if db_path is None:
            package_root = Path(__file__).resolve().parent.parent
            db_path = package_root / "resources" / "dr_patterns.json"

        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"DR pattern database not found: {self.db_path}. "
                f"Ensure the skill package structure is intact."
            )

        with open(self.db_path, encoding="utf-8") as f:
            self._db: Dict[str, Any] = json.load(f)

        # Index terms by id for O(1) lookup
        self._terms_by_id: Dict[str, Dict[str, Any]] = {
            t["id"]: t for t in self._db.get("dr_terms", [])
        }

        # Index robot profiles by id
        self._profiles_by_id: Dict[str, Dict[str, Any]] = {
            pid: profile for pid, profile in self._db.get("robot_profiles", {}).items()
        }

    @property
    def version(self) -> str:
        """Database version string."""
        return self._db.get("version", "unknown")

    @property
    def source(self) -> str:
        """Database source attribution."""
        return self._db.get("source", "unknown")

    # --- Robot profiles ---

    def list_robot_profiles(self) -> List[str]:
        """List all robot profile IDs."""
        return list(self._profiles_by_id.keys())

    def get_robot_profile(self, robot_id: str) -> Dict[str, Any]:
        """Get a robot profile by ID.

        Args:
            robot_id: e.g., "quadruped_small", "manipulator_arm".

        Returns:
            Profile dict with: description, mass_kg, examples, defaults.

        Raises:
            KeyError: If robot_id not in database.
        """
        if robot_id not in self._profiles_by_id:
            available = list(self._profiles_by_id.keys())
            raise KeyError(
                f"Robot profile '{robot_id}' not found. Available: {available}"
            )
        return self._profiles_by_id[robot_id]

    def get_robot_defaults(self, robot_id: str) -> Dict[str, Any]:
        """Get the default parameter ranges for a robot profile.

        Args:
            robot_id: e.g., "quadruped_small".

        Returns:
            Dict of parameter name → default range/value.
            e.g., {"mass_range": [-2.0, 2.0], "friction_range": [0.3, 1.2], ...}
        """
        profile = self.get_robot_profile(robot_id)
        return dict(profile.get("defaults", {}))

    def get_robot_mass_kg(self, robot_id: str) -> float:
        """Get the nominal robot mass in kg."""
        profile = self.get_robot_profile(robot_id)
        return float(profile.get("mass_kg", 0.0))

    # --- DR terms ---

    def list_dr_term_ids(self) -> List[str]:
        """List all DR term IDs."""
        return list(self._terms_by_id.keys())

    def list_dr_terms(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List DR terms, optionally filtered by category.

        Args:
            category: If provided, only return terms with this category
                (e.g., "physics", "sensor", "perturbation").

        Returns:
            List of term dicts.
        """
        if category is None:
            return list(self._db.get("dr_terms", []))
        return [t for t in self._db.get("dr_terms", []) if t.get("category") == category]

    def get_dr_term(self, term_id: str) -> Dict[str, Any]:
        """Get a DR term by ID.

        Args:
            term_id: e.g., "robot_mass", "ground_friction".

        Returns:
            Term dict with: id, name, category, purpose, isaac_lab_func,
            mode, params, typical_range_note, pitfalls, sources.

        Raises:
            KeyError: If term_id not in database.
        """
        if term_id not in self._terms_by_id:
            available = list(self._terms_by_id.keys())
            raise KeyError(
                f"DR term '{term_id}' not found. Available: {available}"
            )
        return self._terms_by_id[term_id]

    def list_categories(self) -> List[str]:
        """List all unique DR term categories."""
        return list({t.get("category") for t in self._db.get("dr_terms", [])})

    # --- Task recommendations ---

    def get_task_recommendation(self, task_id: str) -> Dict[str, List[str]]:
        """Get DR term recommendations for a task type.

        Args:
            task_id: e.g., "locomotion_velocity", "manipulation_reach".

        Returns:
            Dict with keys: essential, recommended, optional. Each maps to
            a list of DR term IDs.

        Raises:
            KeyError: If task_id not in database.
        """
        recs = self._db.get("task_recommendations", {})
        if task_id not in recs:
            available = list(recs.keys())
            raise KeyError(
                f"Task '{task_id}' not found. Available: {available}"
            )
        return dict(recs[task_id])

    def list_task_types(self) -> List[str]:
        """List all task types with recommendations."""
        return list(self._db.get("task_recommendations", {}).keys())


# ------------------------------------------------------------------
# DRRecommendation: result of advisor.recommend()
# ------------------------------------------------------------------


@dataclass
class DRRecommendation:
    """Result of a DR recommendation.

    Attributes:
        robot_type: The robot profile ID used.
        task_type: The task type ID used.
        robot_mass_kg: Nominal robot mass.
        essential_terms: List of essential DR term IDs.
        recommended_terms: List of recommended DR term IDs.
        optional_terms: List of optional DR term IDs (only if include_optional).
        parameter_ranges: Dict mapping term_id → {param_name: value}.
        term_details: Dict mapping term_id → full term dict (pitfalls, isaac_lab_func, etc.).
        explanation: Human-readable explanation of the recommendation.
        timestamp: When the recommendation was generated.
    """

    robot_type: str = ""
    task_type: str = ""
    robot_mass_kg: float = 0.0
    essential_terms: List[str] = field(default_factory=list)
    recommended_terms: List[str] = field(default_factory=list)
    optional_terms: List[str] = field(default_factory=list)
    parameter_ranges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    term_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    explanation: str = ""
    timestamp: str = ""

    def all_term_ids(self) -> List[str]:
        """Return all recommended term IDs in priority order."""
        return self.essential_terms + self.recommended_terms + self.optional_terms

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"DRRecommendation(robot={self.robot_type}, task={self.task_type}, "
            f"terms={len(self.all_term_ids())} "
            f"[{len(self.essential_terms)}e/{len(self.recommended_terms)}r/"
            f"{len(self.optional_terms)}o])"
        )


# ------------------------------------------------------------------
# DRAdvisor: main recommendation engine
# ------------------------------------------------------------------


class DRAdvisor:
    """Recommends DR configuration for a given robot + task.

    Combines:
    - DRLibrary: access to curated DR pattern database
    - CodeEmitter: Jinja2 template rendering for events_cfg.py

    The output is a complete EventsCfg class that can be mounted on a
    ManagerBasedRLEnvCfg.

    Example:
        advisor = DRAdvisor()
        rec = advisor.recommend(
            robot_type="quadruped_small",
            task_type="locomotion_velocity",
        )
        code = advisor.generate_code(rec)
    """

    DEFAULT_TEMPLATE = "envs/events_cfg.py.tmpl"

    def __init__(
        self,
        library: Optional[DRLibrary] = None,
        emitter: Optional[CodeEmitter] = None,
    ) -> None:
        """Initialize advisor with optional dependency injection.

        Args:
            library: DRLibrary instance (created if None).
            emitter: CodeEmitter instance (created if None).
        """
        self.library = library or DRLibrary()
        self.emitter = emitter or CodeEmitter()

    def recommend(
        self,
        robot_type: str,
        task_type: str,
        include_optional: bool = False,
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> DRRecommendation:
        """Recommend DR configuration for a robot + task.

        Args:
            robot_type: Robot profile ID (e.g., "quadruped_small").
            task_type: Task type ID (e.g., "locomotion_velocity").
            include_optional: If True, include optional terms in the
                recommendation. Defaults to False.
            overrides: Optional dict mapping term_id → {param_name: value}
                to override defaults. Example:
                {"robot_mass": {"mass_range": (-3.0, 3.0)}}

        Returns:
            DRRecommendation with terms, parameter ranges, and explanation.

        Raises:
            KeyError: If robot_type or task_type not in database.
        """
        # Validate inputs early
        robot_profile = self.library.get_robot_profile(robot_type)
        task_rec = self.library.get_task_recommendation(task_type)
        robot_defaults = self.library.get_robot_defaults(robot_type)
        robot_mass_kg = float(robot_profile.get("mass_kg", 0.0))

        # Determine which terms to include
        essential = list(task_rec.get("essential", []))
        recommended = list(task_rec.get("recommended", []))
        optional = list(task_rec.get("optional", [])) if include_optional else []

        # Build parameter ranges and term details for each included term
        all_term_ids = essential + recommended + optional
        parameter_ranges: Dict[str, Dict[str, Any]] = {}
        term_details: Dict[str, Dict[str, Any]] = {}

        for term_id in all_term_ids:
            term_def = self.library.get_dr_term(term_id)
            term_details[term_id] = dict(term_def)

            # Resolve parameters: start with term defaults, overlay robot defaults, overlay overrides
            params = self._resolve_params(
                term_id, term_def, robot_defaults, overrides or {}
            )
            parameter_ranges[term_id] = params

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        explanation = self._build_explanation(
            robot_type=robot_type,
            task_type=task_type,
            robot_mass_kg=robot_mass_kg,
            essential=essential,
            recommended=recommended,
            optional=optional,
            parameter_ranges=parameter_ranges,
            term_details=term_details,
        )

        return DRRecommendation(
            robot_type=robot_type,
            task_type=task_type,
            robot_mass_kg=robot_mass_kg,
            essential_terms=essential,
            recommended_terms=recommended,
            optional_terms=optional,
            parameter_ranges=parameter_ranges,
            term_details=term_details,
            explanation=explanation,
            timestamp=timestamp,
        )

    def _resolve_params(
        self,
        term_id: str,
        term_def: Dict[str, Any],
        robot_defaults: Dict[str, Any],
        overrides: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Resolve parameters for a DR term.

        Parameter sources, in priority order (highest wins):
        1. User overrides (overrides[term_id][param_name])
        2. Robot defaults — via param_mapping (term param name → robot_defaults key)
        3. Static params (term_def["static_params"]) — fixed values like num_buckets=16

        The param_mapping field in dr_patterns.json maps term param names
        (e.g., "stiffness_range") to robot_defaults keys (e.g., "motor_stiffness_range").
        This decouples term parameter naming from robot profile naming.
        """
        param_mapping = term_def.get("param_mapping", {})
        static_params = term_def.get("static_params", {})

        params: Dict[str, Any] = {}

        # 1. Apply robot defaults via param_mapping
        for term_param, robot_key in param_mapping.items():
            if robot_key in robot_defaults:
                params[term_param] = self._normalize_range(robot_defaults[robot_key])

        # 2. Apply static params (fixed values that don't depend on robot)
        for pname, pvalue in static_params.items():
            params[pname] = self._normalize_range(pvalue)

        # 3. Apply user overrides (highest priority)
        if term_id in overrides:
            for pname, pvalue in overrides[term_id].items():
                params[pname] = self._normalize_range(pvalue)

        # 4. Validate that *_range params are 2-element (lo, hi) with lo <= hi
        for pname, pvalue in params.items():
            if pname.endswith("_range"):
                if not isinstance(pvalue, (list, tuple)):
                    raise ValueError(
                        f"param '{pname}' must be a (lo, hi) range, got {type(pvalue).__name__}: {pvalue}"
                    )
                if len(pvalue) != 2:
                    raise ValueError(
                        f"param '{pname}' must have exactly 2 elements, got {len(pvalue)}: {pvalue}"
                    )
                lo, hi = pvalue[0], pvalue[1]
                if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo > hi:
                    raise ValueError(
                        f"param '{pname}' lo must be <= hi, got ({lo}, {hi})"
                    )

        return params

    @staticmethod
    def _normalize_range(value: Any) -> Any:
        """Normalize a list to a tuple for consistent rendering. Scalars pass through."""
        if isinstance(value, list):
            return tuple(value)
        return value

    def _build_explanation(
        self,
        robot_type: str,
        task_type: str,
        robot_mass_kg: float,
        essential: List[str],
        recommended: List[str],
        optional: List[str],
        parameter_ranges: Dict[str, Dict[str, Any]],
        term_details: Dict[str, Dict[str, Any]],
    ) -> str:
        """Build a human-readable explanation of the recommendation."""
        lines: List[str] = []
        lines.append(f"DR Recommendation for {robot_type} ({robot_mass_kg} kg) on {task_type}")
        lines.append("=" * 70)
        lines.append("")

        lines.append(f"Robot profile: {robot_type}")
        lines.append(f"  Nominal mass: {robot_mass_kg} kg")
        lines.append(f"Task type: {task_type}")
        lines.append("")

        lines.append(f"Essential terms ({len(essential)}):")
        for term_id in essential:
            details = term_details.get(term_id, {})
            purpose = details.get("purpose", "")
            func = details.get("isaac_lab_func", "")
            params = parameter_ranges.get(term_id, {})
            lines.append(f"  - {term_id}: {purpose}")
            lines.append(f"    Isaac Lab func: {func}")
            if params:
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                lines.append(f"    Params: {param_str}")
            pitfalls = details.get("pitfalls", [])
            if pitfalls:
                lines.append(f"    Pitfalls:")
                for pitfall in pitfalls:
                    lines.append(f"      - {pitfall}")
            lines.append("")

        if recommended:
            lines.append(f"Recommended terms ({len(recommended)}):")
            for term_id in recommended:
                details = term_details.get(term_id, {})
                purpose = details.get("purpose", "")
                lines.append(f"  - {term_id}: {purpose}")
            lines.append("")

        if optional:
            lines.append(f"Optional terms ({len(optional)}):")
            for term_id in optional:
                details = term_details.get(term_id, {})
                purpose = details.get("purpose", "")
                lines.append(f"  - {term_id}: {purpose}")
            lines.append("")

        lines.append("Tuning guide:")
        lines.append("  1. Start with essential terms only; verify training converges.")
        lines.append("  2. Add recommended terms once essential DR works.")
        lines.append("  3. Add optional terms for final sim-to-real hardening.")
        lines.append("  4. If training fails to converge, NARROW ranges (halve them).")
        lines.append("  5. If sim-to-real fails, WIDEN ranges gradually.")

        return "\n".join(lines)

    def generate_code(self, recommendation: DRRecommendation) -> str:
        """Render EventsCfg code for a recommendation.

        Args:
            recommendation: The DRRecommendation to render.

        Returns:
            Python source code implementing an EventsCfg class.
        """
        context = self._build_template_context(recommendation)
        return self.emitter.render(self.DEFAULT_TEMPLATE, context)

    def write_code(
        self,
        recommendation: DRRecommendation,
        output_path: Path | str,
    ) -> Path:
        """Render and write EventsCfg code to a file.

        Args:
            recommendation: The DRRecommendation to render.
            output_path: Path to write the code.

        Returns:
            Path to the written file.
        """
        context = self._build_template_context(recommendation)
        return self.emitter.emit(self.DEFAULT_TEMPLATE, context, output_path)

    def _build_template_context(self, rec: DRRecommendation) -> Dict[str, Any]:
        """Build the Jinja2 context dict for events_cfg.py.tmpl."""
        all_terms = rec.all_term_ids()

        # Per-term lookups for the template
        term_funcs: Dict[str, str] = {}
        term_modes: Dict[str, str] = {}
        term_intervals: Dict[str, Optional[float]] = {}
        term_categories: Dict[str, str] = {}
        term_purposes: Dict[str, str] = {}
        term_pitfalls: Dict[str, List[str]] = {}
        term_params: Dict[str, Dict[str, Any]] = {}
        term_render_types: Dict[str, str] = {}

        for term_id in all_terms:
            details = rec.term_details.get(term_id, {})
            term_funcs[term_id] = details.get("isaac_lab_func", "mdp.unknown_func")
            term_modes[term_id] = details.get("mode", "reset")
            term_intervals[term_id] = details.get("interval_time_s")
            term_categories[term_id] = details.get("category", "unknown")
            term_purposes[term_id] = details.get("purpose", "")
            term_pitfalls[term_id] = details.get("pitfalls", [])
            term_render_types[term_id] = details.get("render_type", "event_term")
            # Filter out asset_cfg from params (it's added by the template)
            params = {
                k: v for k, v in rec.parameter_ranges.get(term_id, {}).items()
                if k != "asset_cfg"
            }
            term_params[term_id] = params

        # Separate EventTerm-renderable terms from obs_cfg terms
        event_terms = [t for t in all_terms if term_render_types[t] == "event_term"]
        obs_cfg_terms = [t for t in all_terms if term_render_types[t] == "obs_cfg"]

        return {
            "robot_type": rec.robot_type,
            "robot_mass_kg": rec.robot_mass_kg,
            "task_type": rec.task_type,
            "timestamp": rec.timestamp,
            "essential_terms": rec.essential_terms,
            "recommended_terms": rec.recommended_terms,
            "optional_terms": rec.optional_terms,
            "all_terms": all_terms,
            "event_terms": event_terms,
            "obs_cfg_terms": obs_cfg_terms,
            "term_funcs": term_funcs,
            "term_modes": term_modes,
            "term_intervals": term_intervals,
            "term_categories": term_categories,
            "term_purposes": term_purposes,
            "term_pitfalls": term_pitfalls,
            "term_params": term_params,
            "term_render_types": term_render_types,
        }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> None:
    """CLI entry point for DR advisor."""
    parser = argparse.ArgumentParser(
        description=(
            "Recommend Isaac Lab Domain Randomization configuration for a "
            "given robot + task."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/dr_advisor.py \\\n"
            "    --robot quadruped_small \\\n"
            "    --task locomotion_velocity \\\n"
            "    --output events.py --include-optional --explain"
        ),
    )
    parser.add_argument(
        "--robot",
        "-r",
        required=True,
        help="Robot profile ID (e.g., quadruped_small, manipulator_arm).",
    )
    parser.add_argument(
        "--task",
        "-t",
        required=True,
        help="Task type ID (e.g., locomotion_velocity, manipulation_reach).",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional DR terms.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path. If omitted, code is printed to stdout.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print recommendation explanation to stderr.",
    )
    parser.add_argument(
        "--list-robots",
        action="store_true",
        help="List available robot profile IDs and exit.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available task type IDs and exit.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Override a parameter. Format: TERM_ID=PARAM=VALUE. "
            "Example: --override robot_mass=mass_range=(-3.0,3.0). "
            "Can be repeated."
        ),
    )

    args = parser.parse_args()

    lib = DRLibrary()

    if args.list_robots:
        print("Available robot profiles:")
        for rid in lib.list_robot_profiles():
            profile = lib.get_robot_profile(rid)
            print(f"  {rid}: {profile.get('description', '')}")
        return

    if args.list_tasks:
        print("Available task types:")
        for tid in lib.list_task_types():
            rec = lib.get_task_recommendation(tid)
            n = len(rec.get("essential", [])) + len(rec.get("recommended", []))
            print(f"  {tid}: {n} terms (essential+recommended)")
        return

    # Parse overrides
    overrides: Dict[str, Dict[str, Any]] = {}
    for override_str in args.override:
        if "=" not in override_str:
            print(
                f"Error: --override expects TERM_ID=PARAM=VALUE, got '{override_str}'",
                file=sys.stderr,
            )
            sys.exit(1)
        parts = override_str.split("=", 2)
        if len(parts) != 3:
            print(
                f"Error: --override expects TERM_ID=PARAM=VALUE, got '{override_str}'",
                file=sys.stderr,
            )
            sys.exit(1)
        term_id, param_name, value_str = parts
        try:
            value = eval(value_str, {"__builtins__": {}}, {})
        except Exception as e:
            print(
                f"Error: could not parse value '{value_str}': {e}",
                file=sys.stderr,
            )
            sys.exit(1)
        overrides.setdefault(term_id, {})[param_name] = value

    advisor = DRAdvisor(library=lib)
    rec = advisor.recommend(
        robot_type=args.robot,
        task_type=args.task,
        include_optional=args.include_optional,
        overrides=overrides if overrides else None,
    )

    if args.output:
        out_path = advisor.write_code(rec, args.output)
        print(f"Generated DR config: {out_path}", file=sys.stderr)
    else:
        print(advisor.generate_code(rec))

    if args.explain:
        print("\n" + "=" * 60, file=sys.stderr)
        print("DR RECOMMENDATION EXPLANATION", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(rec.explanation, file=sys.stderr)

    print(f"\n{rec.summary()}", file=sys.stderr)


if __name__ == "__main__":
    main()
