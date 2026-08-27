#!/usr/bin/env python3
"""Reward synthesizer — the core of module 1.

Turns a natural language task description into an executable Isaac Lab
RewardsCfg by:
1. Selecting reward patterns via PatternMatcher (task-type-aware selection)
2. Rendering the locomotion_full_cfg.py.tmpl template via CodeEmitter
3. Optionally running static validation via RewardValidator

Usage (CLI):
    python scripts/reward_synthesizer.py \\
        --task "训练四足以 1 m/s 前进，保持身体稳定" \\
        --output reward.py --validate --explain

Usage (library):
    from scripts.reward_synthesizer import RewardSynthesizer
    synth = RewardSynthesizer()
    result = synth.synthesize("train quadruped to walk at 1 m/s")
    print(result.code)
    print(result.explanation)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow direct script execution: `python scripts/reward_synthesizer.py`
# Adds the package root to sys.path so absolute imports work.
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from scripts.reward_library import RewardLibrary
from scripts.utils.code_emitter import CodeEmitter
from scripts.utils.pattern_matcher import PatternMatcher


@dataclass
class SynthesisResult:
    """Result of reward synthesis.

    Attributes:
        code: The rendered reward.py source code.
        output_path: Path where code was written (None if not written).
        task_description: The input task description.
        task_type: Detected task type (e.g., "locomotion_velocity").
        patterns: List of selected pattern IDs.
        config: Dict mapping pattern_id to {weight, ...params}.
        explanation: Human-readable explanation of pattern selection.
        validation: Optional validation result dict.
        timestamp: When synthesis was run.
    """

    code: str
    output_path: Optional[Path] = None
    task_description: str = ""
    task_type: str = ""
    patterns: List[str] = field(default_factory=list)
    config: Dict[str, Dict] = field(default_factory=dict)
    explanation: str = ""
    validation: Optional[Dict[str, Any]] = None
    timestamp: str = ""

    def summary(self) -> str:
        """One-line summary for logging."""
        valid_str = ""
        if self.validation is not None:
            valid_str = f", valid={self.validation.get('valid', '?')}"
        return (
            f"SynthesisResult(task_type={self.task_type}, "
            f"patterns={len(self.patterns)} terms{valid_str})"
        )


class RewardSynthesizer:
    """Synthesizes executable Isaac Lab reward code from NL task descriptions.

    This is the core of module 1. It combines:
    - RewardLibrary: access to curated pattern database
    - PatternMatcher: task-type-aware pattern selection
    - CodeEmitter: Jinja2 template rendering

    The output is a complete RewardsCfg class that can be mounted on a
    ManagerBasedRLEnvCfg.

    Example:
        synth = RewardSynthesizer()
        result = synth.synthesize("train quadruped to walk forward at 1 m/s")
        print(result.code)
        print(result.explanation)
    """

    DEFAULT_TEMPLATE = "rewards/locomotion_full_cfg.py.tmpl"

    def __init__(
        self,
        library: Optional[RewardLibrary] = None,
        matcher: Optional[PatternMatcher] = None,
        emitter: Optional[CodeEmitter] = None,
    ) -> None:
        """Initialize synthesizer with optional dependency injection.

        Args:
            library: RewardLibrary instance (created if None).
            matcher: PatternMatcher instance (created if None).
            emitter: CodeEmitter instance (created if None).
        """
        self.library = library or RewardLibrary()
        self.matcher = matcher or PatternMatcher(self.library)
        self.emitter = emitter or CodeEmitter()

    def synthesize(
        self,
        task_description: str,
        include_optional: bool = False,
        config_overrides: Optional[Dict[str, Dict]] = None,
        output_path: Optional[Path] = None,
        validate: bool = False,
    ) -> SynthesisResult:
        """Synthesize reward code from a task description.

        Args:
            task_description: Natural language description of the task.
                Example: "训练四足以 1 m/s 前进，保持身体水平稳定"
            include_optional: Include optional penalty terms (joint_pos_limits,
                is_terminated, flat_orientation_l2). Defaults to False.
            config_overrides: Override default weights/params. Example:
                {"linear_velocity_tracking": {"weight": 1.5, "std": 0.5}}
            output_path: If provided, write generated code to this path.
            validate: If True, run static validation on generated code.

        Returns:
            SynthesisResult with code, metadata, and explanation.

        Raises:
            FileNotFoundError: If template or pattern database missing.
            TypeError: If task_description is None or not a string.
            ValueError: If pattern selection fails (internal inconsistency).
            jinja2.UndefinedError: If template variable is missing.
        """
        if task_description is None:
            raise TypeError("task_description must be a string, got NoneType")
        if not isinstance(task_description, str):
            raise TypeError(
                f"task_description must be a string, got {type(task_description).__name__}"
            )

        # Step 1: Select patterns based on task description
        selection = self.matcher.select_patterns(
            task_description,
            include_optional=include_optional,
            config_overrides=config_overrides,
        )

        # Step 2: Determine if task-level mdp import is needed
        # feet_air_time lives in isaaclab_tasks/.../velocity/mdp/, not core mdp
        use_task_mdp = "feet_air_time" in selection["patterns"]

        # Step 3: Sanitize task description for safe insertion into generated code.
        # Newlines would break comment lines; triple quotes would break docstrings.
        safe_task_description = task_description.replace("\n", " ").replace("\r", " ")
        safe_task_description = safe_task_description.replace('"""', '\\"\\"\\"')

        # Step 4: Build template context
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        context = {
            "task_description": safe_task_description,
            "timestamp": timestamp,
            "patterns": selection["patterns"],
            "config": selection["config"],
            "use_task_mdp": use_task_mdp,
            "include_optional": include_optional,
            "weight_source": "reward_patterns.json defaults + PATTERN_DEFAULTS",
        }

        # Step 4: Render template
        code = self.emitter.render(self.DEFAULT_TEMPLATE, context)

        # Step 5: Optional validation
        validation: Optional[Dict[str, Any]] = None
        if validate:
            validation = self._validate_code(code)

        # Step 6: Write to file if requested
        written_path: Optional[Path] = None
        if output_path is not None:
            written_path = self.emitter.emit(
                self.DEFAULT_TEMPLATE, context, output_path
            )

        # Step 7: Build human-readable explanation
        explanation = self.matcher.explain_selection(selection)

        return SynthesisResult(
            code=code,
            output_path=written_path,
            task_description=task_description,
            task_type=selection["task_type"],
            patterns=selection["patterns"],
            config=selection["config"],
            explanation=explanation,
            validation=validation,
            timestamp=timestamp,
        )

    def _validate_code(self, code: str) -> Dict[str, Any]:
        """Run static validation on generated code.

        Delegates to RewardValidator if available; falls back to basic
        syntax check via compile().

        Args:
            code: Generated Python source code.

        Returns:
            Dict with 'valid' (bool), 'checks' (list), 'errors' (list).
        """
        try:
            from scripts.reward_validator import RewardValidator

            validator = RewardValidator()
            return validator.validate_code(code)
        except ImportError:
            # Validator not yet implemented — fall back to syntax check
            try:
                compile(code, "<generated>", "exec")
                return {
                    "valid": True,
                    "checks": ["syntax (fallback; validator not available)"],
                    "errors": [],
                }
            except SyntaxError as e:
                return {
                    "valid": False,
                    "checks": ["syntax (fallback; validator not available)"],
                    "errors": [f"SyntaxError: {e}"],
                }


def main() -> None:
    """CLI entry point for reward synthesis."""
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize Isaac Lab reward code from a natural language "
            "task description."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/reward_synthesizer.py \\\n"
            "    --task 'train quadruped to walk forward at 1 m/s, keep stable' \\\n"
            "    --output reward.py --validate --explain"
        ),
    )
    parser.add_argument(
        "--task",
        "-t",
        required=True,
        help="Natural language task description.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path. If omitted, code is printed to stdout.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional penalty terms (joint_pos_limits, is_terminated).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run static validation on generated code.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print pattern selection explanation to stderr.",
    )
    parser.add_argument(
        "--weight-override",
        action="append",
        default=[],
        help=(
            "Override a pattern weight. Format: PATTERN_ID=WEIGHT. "
            "Example: --weight-override linear_velocity_tracking=1.5. "
            "Can be repeated."
        ),
    )

    args = parser.parse_args()

    # Parse weight overrides
    config_overrides: Dict[str, Dict] = {}
    for override in args.weight_override:
        if "=" not in override:
            print(
                f"Error: --weight-override expects PATTERN_ID=WEIGHT, got '{override}'",
                file=sys.stderr,
            )
            sys.exit(1)
        pid, weight_str = override.split("=", 1)
        try:
            weight = float(weight_str)
        except ValueError:
            print(
                f"Error: weight must be a number, got '{weight_str}'",
                file=sys.stderr,
            )
            sys.exit(1)
        config_overrides[pid] = {"weight": weight}

    # Run synthesis
    synth = RewardSynthesizer()
    result = synth.synthesize(
        task_description=args.task,
        include_optional=args.include_optional,
        config_overrides=config_overrides if config_overrides else None,
        output_path=Path(args.output) if args.output else None,
        validate=args.validate,
    )

    # Output
    if args.output:
        print(f"Generated reward code: {result.output_path}", file=sys.stderr)
    else:
        print(result.code)

    if args.explain:
        print("\n" + "=" * 60, file=sys.stderr)
        print("PATTERN SELECTION EXPLANATION", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(result.explanation, file=sys.stderr)

    if args.validate and result.validation:
        print("\n" + "=" * 60, file=sys.stderr)
        print("VALIDATION", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Valid: {result.validation.get('valid', False)}", file=sys.stderr)
        for check in result.validation.get("checks", []):
            print(f"  check: {check}", file=sys.stderr)
        for err in result.validation.get("errors", []):
            print(f"  ERROR: {err}", file=sys.stderr)

    print(f"\n{result.summary()}", file=sys.stderr)


if __name__ == "__main__":
    main()
