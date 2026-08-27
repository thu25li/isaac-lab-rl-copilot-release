#!/usr/bin/env python3
"""Curriculum designer — generates Isaac Lab CurriculumTerm configurations.

Given a task type, recommends a curated set of Isaac Lab CurriculumTerm
patterns with parameter defaults. Patterns are sourced from legged_gym,
IsaacLab, Walk These Ways, and IsaacGymEnvs.

Pipeline:
1. Look up task recommendation → which curriculum terms to include
2. For each term, fetch its definition (isaac_lab_func, params, pitfalls)
3. Apply user overrides
4. Render curriculum_cfg.py.tmpl via CodeEmitter

Usage (CLI):
    python scripts/curriculum_designer.py \\
        --task locomotion_velocity \\
        --output curriculum.py --include-optional --explain

Usage (library):
    from scripts.curriculum_designer import CurriculumDesigner
    designer = CurriculumDesigner()
    rec = designer.recommend(task_type="locomotion_velocity")
    print(designer.generate_code(rec))
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow direct script execution
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from scripts.utils.code_emitter import CodeEmitter


# ------------------------------------------------------------------
# CurriculumLibrary: programmatic access to curriculum_patterns.json
# ------------------------------------------------------------------


class CurriculumLibrary:
    """Provides access to the curated curriculum pattern database.

    The database is stored as JSON at resources/curriculum_patterns.json.
    It contains:
    - curriculum_terms: term definitions (isaac_lab_func, params, pitfalls)
    - task_recommendations: which terms to use per task type
    - design_principles: general curriculum design principles

    Example:
        lib = CurriculumLibrary()
        term = lib.get_term("terrain_levels")
        rec = lib.get_task_recommendation("locomotion_rough_terrain")
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize library with curriculum pattern database path.

        Args:
            db_path: Path to curriculum_patterns.json. If None, resolves
                to the package's resources/curriculum_patterns.json.

        Raises:
            FileNotFoundError: If database file does not exist.
        """
        import json

        if db_path is None:
            package_root = Path(__file__).resolve().parent.parent
            db_path = package_root / "resources" / "curriculum_patterns.json"

        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Curriculum pattern database not found: {self.db_path}. "
                f"Ensure the skill package structure is intact."
            )

        with open(self.db_path, encoding="utf-8") as f:
            self._db: Dict[str, Any] = json.load(f)

        self._terms_by_id: Dict[str, Dict[str, Any]] = {
            t["id"]: t for t in self._db.get("curriculum_terms", [])
        }

    @property
    def version(self) -> str:
        """Database version string."""
        return self._db.get("version", "unknown")

    @property
    def source(self) -> str:
        """Database source attribution."""
        return self._db.get("source", "unknown")

    def list_term_ids(self) -> List[str]:
        """List all curriculum term IDs."""
        return list(self._terms_by_id.keys())

    def list_terms(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List curriculum terms, optionally filtered by category.

        Args:
            category: If provided, only return terms with this category.

        Returns:
            List of term dicts.
        """
        if category is None:
            return list(self._db.get("curriculum_terms", []))
        return [
            t for t in self._db.get("curriculum_terms", [])
            if t.get("category") == category
        ]

    def get_term(self, term_id: str) -> Dict[str, Any]:
        """Get a curriculum term by ID.

        Args:
            term_id: e.g., "terrain_levels", "command_curriculum".

        Returns:
            Term dict with all fields.

        Raises:
            KeyError: If term_id not in database.
        """
        if term_id not in self._terms_by_id:
            available = list(self._terms_by_id.keys())
            raise KeyError(
                f"Curriculum term '{term_id}' not found. Available: {available}"
            )
        return self._terms_by_id[term_id]

    def list_categories(self) -> List[str]:
        """List all unique curriculum term categories."""
        return list({t.get("category") for t in self._db.get("curriculum_terms", [])})

    def get_task_recommendation(self, task_id: str) -> Dict[str, List[str]]:
        """Get curriculum term recommendations for a task type.

        Args:
            task_id: e.g., "locomotion_velocity", "manipulation_reach".

        Returns:
            Dict with keys: essential, recommended, optional.

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

    def get_design_principles(self) -> Dict[str, str]:
        """Get general curriculum design principles."""
        return dict(self._db.get("design_principles", {}))


# ------------------------------------------------------------------
# CurriculumRecommendation: result of designer.recommend()
# ------------------------------------------------------------------


@dataclass
class CurriculumRecommendation:
    """Result of a curriculum recommendation.

    Attributes:
        task_type: The task type ID used.
        essential_terms: List of essential curriculum term IDs.
        recommended_terms: List of recommended curriculum term IDs.
        optional_terms: List of optional curriculum term IDs (only if include_optional).
        parameter_ranges: Dict mapping term_id → {param_name: value}.
        term_details: Dict mapping term_id → full term dict.
        design_principles: General curriculum design principles.
        explanation: Human-readable explanation of the recommendation.
        timestamp: When the recommendation was generated.
    """

    task_type: str = ""
    essential_terms: List[str] = field(default_factory=list)
    recommended_terms: List[str] = field(default_factory=list)
    optional_terms: List[str] = field(default_factory=list)
    parameter_ranges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    term_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    design_principles: Dict[str, str] = field(default_factory=dict)
    explanation: str = ""
    timestamp: str = ""

    def all_term_ids(self) -> List[str]:
        """Return all recommended term IDs in priority order."""
        return self.essential_terms + self.recommended_terms + self.optional_terms

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"CurriculumRecommendation(task={self.task_type}, "
            f"terms={len(self.all_term_ids())} "
            f"[{len(self.essential_terms)}e/{len(self.recommended_terms)}r/"
            f"{len(self.optional_terms)}o])"
        )


# ------------------------------------------------------------------
# CurriculumDesigner: main recommendation engine
# ------------------------------------------------------------------


class CurriculumDesigner:
    """Recommends curriculum configuration for a given task.

    Combines:
    - CurriculumLibrary: access to curated curriculum pattern database
    - CodeEmitter: Jinja2 template rendering for curriculum_cfg.py

    Example:
        designer = CurriculumDesigner()
        rec = designer.recommend(task_type="locomotion_velocity")
        code = designer.generate_code(rec)
    """

    DEFAULT_TEMPLATE = "curricula/curriculum_cfg.py.tmpl"

    def __init__(
        self,
        library: Optional[CurriculumLibrary] = None,
        emitter: Optional[CodeEmitter] = None,
    ) -> None:
        """Initialize designer with optional dependency injection.

        Args:
            library: CurriculumLibrary instance (created if None).
            emitter: CodeEmitter instance (created if None).
        """
        self.library = library or CurriculumLibrary()
        self.emitter = emitter or CodeEmitter()

    def recommend(
        self,
        task_type: str,
        include_optional: bool = False,
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> CurriculumRecommendation:
        """Recommend curriculum configuration for a task.

        Args:
            task_type: Task type ID (e.g., "locomotion_velocity").
            include_optional: If True, include optional terms.
            overrides: Optional dict mapping term_id → {param_name: value}.

        Returns:
            CurriculumRecommendation with terms, parameters, and explanation.

        Raises:
            KeyError: If task_type not in database.
        """
        task_rec = self.library.get_task_recommendation(task_type)
        design_principles = self.library.get_design_principles()

        essential = list(task_rec.get("essential", []))
        recommended = list(task_rec.get("recommended", []))
        optional = list(task_rec.get("optional", [])) if include_optional else []

        all_term_ids = essential + recommended + optional
        parameter_ranges: Dict[str, Dict[str, Any]] = {}
        term_details: Dict[str, Dict[str, Any]] = {}

        for term_id in all_term_ids:
            term_def = self.library.get_term(term_id)
            term_details[term_id] = dict(term_def)
            params = self._resolve_params(term_id, term_def, overrides or {})
            parameter_ranges[term_id] = params

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        explanation = self._build_explanation(
            task_type=task_type,
            essential=essential,
            recommended=recommended,
            optional=optional,
            parameter_ranges=parameter_ranges,
            term_details=term_details,
            design_principles=design_principles,
        )

        return CurriculumRecommendation(
            task_type=task_type,
            essential_terms=essential,
            recommended_terms=recommended,
            optional_terms=optional,
            parameter_ranges=parameter_ranges,
            term_details=term_details,
            design_principles=design_principles,
            explanation=explanation,
            timestamp=timestamp,
        )

    def _resolve_params(
        self,
        term_id: str,
        term_def: Dict[str, Any],
        overrides: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Resolve parameters for a curriculum term.

        Parameter sources, in priority order (highest wins):
        1. User overrides (overrides[term_id][param_name])
        2. Term-specific defaults (term_def["param_defaults"])
        """
        param_defaults = term_def.get("param_defaults", {})
        params: Dict[str, Any] = {}

        for pname, pvalue in param_defaults.items():
            params[pname] = self._normalize_range(pvalue)

        if term_id in overrides:
            for pname, pvalue in overrides[term_id].items():
                params[pname] = self._normalize_range(pvalue)

        # Validate that *_range params are 2-element (lo, hi) with lo <= hi
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
        task_type: str,
        essential: List[str],
        recommended: List[str],
        optional: List[str],
        parameter_ranges: Dict[str, Dict[str, Any]],
        term_details: Dict[str, Dict[str, Any]],
        design_principles: Dict[str, str],
    ) -> str:
        """Build a human-readable explanation of the recommendation."""
        lines: List[str] = []
        lines.append(f"Curriculum Recommendation for {task_type}")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Task type: {task_type}")
        lines.append("")

        if essential:
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
        else:
            lines.append("No essential curriculum terms for this task.")
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

        lines.append("Design principles:")
        for principle, value in design_principles.items():
            lines.append(f"  - {principle}: {value}")
        lines.append("")

        lines.append("Tuning guide:")
        lines.append("  1. If curriculum doesn't progress, check success/failure thresholds.")
        lines.append("  2. If curriculum progresses too fast, require consecutive successes.")
        lines.append("  3. If curriculum oscillates, lengthen the evaluation window.")
        lines.append("  4. Always keep a downgrade path — agents need to drop back on failure.")

        return "\n".join(lines)

    def generate_code(self, recommendation: CurriculumRecommendation) -> str:
        """Render CurriculumCfg code for a recommendation."""
        context = self._build_template_context(recommendation)
        return self.emitter.render(self.DEFAULT_TEMPLATE, context)

    def write_code(
        self,
        recommendation: CurriculumRecommendation,
        output_path: Path | str,
    ) -> Path:
        """Render and write CurriculumCfg code to a file."""
        context = self._build_template_context(recommendation)
        return self.emitter.emit(self.DEFAULT_TEMPLATE, context, output_path)

    def _build_template_context(self, rec: CurriculumRecommendation) -> Dict[str, Any]:
        """Build the Jinja2 context dict for curriculum_cfg.py.tmpl."""
        all_terms = rec.all_term_ids()

        term_funcs: Dict[str, str] = {}
        term_categories: Dict[str, str] = {}
        term_priorities: Dict[str, str] = {}
        term_purposes: Dict[str, str] = {}
        term_pitfalls: Dict[str, List[str]] = {}
        term_params: Dict[str, Dict[str, Any]] = {}

        for term_id in all_terms:
            details = rec.term_details.get(term_id, {})
            term_funcs[term_id] = details.get("isaac_lab_func", "mdp.unknown_func")
            term_categories[term_id] = details.get("category", "unknown")
            term_priorities[term_id] = details.get("priority", "recommended")
            term_purposes[term_id] = details.get("purpose", "")
            term_pitfalls[term_id] = details.get("pitfalls", [])
            term_params[term_id] = rec.parameter_ranges.get(term_id, {})

        return {
            "task_type": rec.task_type,
            "timestamp": rec.timestamp,
            "essential_terms": rec.essential_terms,
            "recommended_terms": rec.recommended_terms,
            "optional_terms": rec.optional_terms,
            "all_terms": all_terms,
            "term_funcs": term_funcs,
            "term_categories": term_categories,
            "term_priorities": term_priorities,
            "term_purposes": term_purposes,
            "term_pitfalls": term_pitfalls,
            "term_params": term_params,
            "design_principles": rec.design_principles,
        }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> None:
    """CLI entry point for curriculum designer."""
    parser = argparse.ArgumentParser(
        description=(
            "Recommend Isaac Lab Curriculum configuration for a given task."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/curriculum_designer.py \\\n"
            "    --task locomotion_velocity \\\n"
            "    --output curriculum.py --include-optional --explain"
        ),
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
        help="Include optional curriculum terms.",
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
        "--list-tasks",
        action="store_true",
        help="List available task type IDs and exit.",
    )
    parser.add_argument(
        "--list-terms",
        action="store_true",
        help="List available curriculum term IDs and exit.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Override a parameter. Format: TERM_ID=PARAM=VALUE. "
            "Example: --override terrain_levels=max_level=15. "
            "Can be repeated."
        ),
    )

    args = parser.parse_args()

    lib = CurriculumLibrary()

    if args.list_tasks:
        print("Available task types:")
        for tid in lib.list_task_types():
            rec = lib.get_task_recommendation(tid)
            n_e = len(rec.get("essential", []))
            n_r = len(rec.get("recommended", []))
            print(f"  {tid}: {n_e} essential, {n_r} recommended")
        return

    if args.list_terms:
        print("Available curriculum terms:")
        for tid in lib.list_term_ids():
            term = lib.get_term(tid)
            print(f"  {tid}: {term.get('name', '')}")
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

    designer = CurriculumDesigner(library=lib)
    rec = designer.recommend(
        task_type=args.task,
        include_optional=args.include_optional,
        overrides=overrides if overrides else None,
    )

    if args.output:
        out_path = designer.write_code(rec, args.output)
        print(f"Generated curriculum config: {out_path}", file=sys.stderr)
    else:
        print(designer.generate_code(rec))

    if args.explain:
        print("\n" + "=" * 60, file=sys.stderr)
        print("CURRICULUM RECOMMENDATION EXPLANATION", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(rec.explanation, file=sys.stderr)

    print(f"\n{rec.summary()}", file=sys.stderr)


if __name__ == "__main__":
    main()
