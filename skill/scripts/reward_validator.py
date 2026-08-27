#!/usr/bin/env python3
"""Reward validator — static validation of generated Isaac Lab reward code.

Performs the following checks (does NOT execute the code):
1. syntax: Python syntax valid (via ast.parse)
2. imports: Required imports present (configclass, RewTerm, mdp)
3. rewards_cfg_class: RewardsCfg class exists with @configclass decorator
4. rewterm_calls: Each RewTerm(...) has func, weight, params fields
5. func_references: Referenced mdp functions exist in isaac_lab_api.json
6. weight_ranges: Weights within documented ranges (warnings, not errors)
7. common_mistakes: Missing command_name, wrong weight sign, etc. (warnings)

Usage (CLI):
    python scripts/reward_validator.py --file path/to/reward.py

Usage (library):
    from scripts.reward_validator import RewardValidator
    validator = RewardValidator()
    result = validator.validate_code(code_string)
    if not result["valid"]:
        for err in result["errors"]:
            print(err)
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow direct script execution
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from scripts.reward_library import RewardLibrary


class RewardValidator:
    """Static validator for generated Isaac Lab reward code.

    Uses AST parsing for robust checks. Does not execute the validated code,
    so it is safe to run on untrusted input.

    Attributes:
        REQUIRED_IMPORTS: Names that must be imported for valid RewardsCfg.
    """

    REQUIRED_IMPORTS = ["configclass", "RewTerm", "mdp"]

    def __init__(
        self,
        library: Optional[RewardLibrary] = None,
        api_ref_path: Optional[Path] = None,
    ) -> None:
        """Initialize validator.

        Args:
            library: RewardLibrary for pattern weight-range lookups.
            api_ref_path: Path to isaac_lab_api.json for function reference checks.
        """
        self.library = library or RewardLibrary()

        if api_ref_path is None:
            api_ref_path = _pkg_root / "resources" / "isaac_lab_api.json"
        self.api_ref_path = Path(api_ref_path)

        self._api_ref: Dict[str, Any] = {}
        if self.api_ref_path.exists():
            with open(self.api_ref_path, encoding="utf-8") as f:
                self._api_ref = json.load(f)

    def validate_code(self, code: str) -> Dict[str, Any]:
        """Validate generated reward code.

        Args:
            code: Python source code to validate.

        Returns:
            Dict with:
                - valid: bool (True if no errors)
                - checks: list of check names performed
                - errors: list of blocking error messages
                - warnings: list of non-blocking warning messages

        Raises:
            TypeError: If code is None or not a string.
        """
        if code is None:
            raise TypeError("code must be a string, got NoneType")
        if not isinstance(code, str):
            raise TypeError(f"code must be a string, got {type(code).__name__}")

        errors: List[str] = []
        warnings: List[str] = []
        checks: List[str] = []

        # Check 1: Syntax
        checks.append("syntax")
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return {
                "valid": False,
                "checks": checks,
                "errors": errors,
                "warnings": warnings,
            }

        # Check 2: Required imports
        checks.append("imports")
        errors.extend(self._check_imports(tree))

        # Check 3: RewardsCfg class structure
        checks.append("rewards_cfg_class")
        cfg_issues, cfg_node = self._check_rewards_cfg_class(tree)
        errors.extend(cfg_issues)

        # Checks 4-7 only run if we have a valid RewardsCfg class
        if cfg_node is not None:
            # Check 4: RewTerm calls
            checks.append("rewterm_calls")
            term_issues, terms = self._check_rewterm_calls(cfg_node)
            errors.extend(term_issues)

            # Check 5: Function references
            checks.append("func_references")
            errors.extend(self._check_func_references(terms))

            # Check 6: Weight ranges (warnings)
            checks.append("weight_ranges")
            warnings.extend(self._check_weight_ranges(terms))

            # Check 7: Common mistakes (warnings)
            checks.append("common_mistakes")
            warnings.extend(self._check_common_mistakes(terms))

        return {
            "valid": len(errors) == 0,
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
        }

    def validate_file(self, path: Path | str) -> Dict[str, Any]:
        """Validate reward code from a file.

        Args:
            path: Path to the reward.py file.

        Returns:
            Same dict structure as validate_code.
        """
        path = Path(path)
        if not path.exists():
            return {
                "valid": False,
                "checks": [],
                "errors": [f"File not found: {path}"],
                "warnings": [],
            }
        code = path.read_text(encoding="utf-8")
        return self.validate_code(code)

    def _check_imports(self, tree: ast.AST) -> List[str]:
        """Check that required imports are present."""
        errors: List[str] = []
        imported_names: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)

        for required in self.REQUIRED_IMPORTS:
            if required not in imported_names:
                errors.append(f"Missing required import: {required}")

        return errors

    def _check_rewards_cfg_class(
        self, tree: ast.AST
    ) -> Tuple[List[str], Optional[ast.ClassDef]]:
        """Check that RewardsCfg class exists with @configclass decorator."""
        errors: List[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "RewardsCfg":
                has_configclass = False
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name) and dec.id == "configclass":
                        has_configclass = True
                    elif isinstance(dec, ast.Attribute) and dec.attr == "configclass":
                        has_configclass = True

                if not has_configclass:
                    errors.append(
                        "RewardsCfg class missing @configclass decorator"
                    )
                return errors, node

        errors.append("RewardsCfg class not found")
        return errors, None

    def _check_rewterm_calls(
        self, cfg_node: ast.ClassDef
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Check RewTerm calls and extract their data."""
        errors: List[str] = []
        terms: List[Dict[str, Any]] = []

        for stmt in cfg_node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and isinstance(
                        stmt.value, ast.Call
                    ):
                        call = stmt.value
                        func_name = self._get_call_name(call.func)
                        if func_name in ("RewTerm", "RewardTermCfg"):
                            term_data = self._extract_rewterm_data(target.id, call)
                            terms.append(term_data)

                            if "func" not in term_data:
                                errors.append(
                                    f"RewTerm '{target.id}' missing 'func' field"
                                )
                            if "weight" not in term_data:
                                errors.append(
                                    f"RewTerm '{target.id}' missing 'weight' field"
                                )

        return errors, terms

    @staticmethod
    def _get_call_name(func_node: ast.AST) -> str:
        """Extract function name from a Call's func node."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            return func_node.attr
        return ""

    def _extract_rewterm_data(
        self, term_name: str, call: ast.Call
    ) -> Dict[str, Any]:
        """Extract func, weight, params from a RewTerm call."""
        data: Dict[str, Any] = {"name": term_name}

        for keyword in call.keywords:
            if keyword.arg == "func":
                data["func"] = self._extract_func_ref(keyword.value)
            elif keyword.arg == "weight":
                data["weight"] = self._extract_value(keyword.value)
            elif keyword.arg == "params":
                data["params"] = self._extract_params(keyword.value)

        return data

    @staticmethod
    def _extract_func_ref(node: ast.AST) -> str:
        """Extract function reference like 'mdp.track_lin_vel_xy_exp'."""
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}.{node.attr}"
            return node.attr
        return ""

    def _extract_value(self, node: ast.AST) -> Any:
        """Extract a literal value from an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = self._extract_value(node.operand)
            if isinstance(val, (int, float)):
                return -val
        if isinstance(node, ast.Call):
            # Handle math.sqrt(0.25) etc.
            func_name = self._get_call_name(node.func)
            if func_name == "sqrt" and node.args:
                val = self._extract_value(node.args[0])
                if isinstance(val, (int, float)):
                    return math.sqrt(val)
        return None

    def _extract_params(self, node: ast.AST) -> Dict[str, Any]:
        """Extract params dict from AST."""
        params: Dict[str, Any] = {}
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant):
                    k = key.value
                    v = self._extract_value(value)
                    if v is None and isinstance(value, ast.Call):
                        v = f"<{self._get_call_name(value.func)}(...)>"
                    params[k] = v
        return params

    def _check_func_references(self, terms: List[Dict[str, Any]]) -> List[str]:
        """Check that referenced mdp functions exist in API reference.

        Custom functions (not in builtins) are allowed but not verified.
        Only flags potential typos in builtin function names.
        """
        errors: List[str] = []

        known_funcs: set[str] = set()
        if self._api_ref:
            for category_funcs in self._api_ref.get("builtin_rewards", {}).values():
                for f in category_funcs:
                    known_funcs.add(f["name"])

        if not known_funcs:
            return errors  # Can't check without API reference

        for term in terms:
            func_ref = term.get("func", "")
            func_name = (
                func_ref.split(".")[-1] if "." in func_ref else func_ref
            )

            # Only check if it looks like a builtin (mdp.xxx pattern)
            if func_ref.startswith("mdp."):
                if func_name not in known_funcs:
                    errors.append(
                        f"Term '{term['name']}': function '{func_ref}' not in "
                        f"isaaclab.envs.mdp builtins. If this is a custom "
                        f"function, ensure it is imported correctly."
                    )

        return errors

    def _check_weight_ranges(self, terms: List[Dict[str, Any]]) -> List[str]:
        """Warn if weights are outside documented ranges."""
        warnings: List[str] = []

        # Build pattern_id -> weight_range from library
        pattern_weights: Dict[str, str] = {}
        for p in self.library.list_patterns():
            pattern_weights[p["id"]] = p.get("weight_range", "")

        for term in terms:
            term_name = term.get("name", "")
            weight = term.get("weight")
            if weight is None:
                continue

            # Try to find matching pattern by term name
            for pid, weight_range in pattern_weights.items():
                if pid in term_name or term_name in pid:
                    if self._is_weight_suspicious(weight, weight_range):
                        warnings.append(
                            f"Term '{term_name}' weight={weight} may be outside "
                            f"documented range: {weight_range}"
                        )
                    break

        return warnings

    @staticmethod
    def _is_weight_suspicious(weight: float, weight_range: str) -> bool:
        """Heuristic check if weight is suspiciously outside documented range."""
        numbers = re.findall(r"-?\d+\.?\d*e?-?\d+", weight_range)
        if len(numbers) < 1:
            return False

        try:
            values = [float(n) for n in numbers]
            if len(values) == 1:
                if abs(weight) > 0 and abs(values[0]) > 0:
                    ratio = abs(weight) / abs(values[0])
                    if ratio > 10 or ratio < 0.1:
                        return True
            else:
                min_val = min(values)
                max_val = max(values)
                # Check if weight is >10x outside the documented range
                if weight < 0 and min_val < 0:
                    if weight < min_val * 10:
                        return True
                elif weight > 0 and max_val > 0:
                    if weight > max_val * 10:
                        return True
        except ValueError:
            return False

        return False

    def _check_common_mistakes(self, terms: List[Dict[str, Any]]) -> List[str]:
        """Check for common reward design mistakes."""
        warnings: List[str] = []

        for term in terms:
            func_ref = term.get("func", "")
            params = term.get("params", {}) or {}
            weight = term.get("weight")
            term_name = term.get("name", "?")

            # Velocity tracking must have command_name and std
            if "track_lin_vel" in func_ref or "track_ang_vel" in func_ref:
                if "command_name" not in params:
                    warnings.append(
                        f"Term '{term_name}' ({func_ref}): missing 'command_name' in params"
                    )
                if "std" not in params:
                    warnings.append(
                        f"Term '{term_name}' ({func_ref}): missing 'std' in params"
                    )

            # feet_air_time must have threshold and sensor_cfg
            if "feet_air_time" in func_ref:
                if "threshold" not in params:
                    warnings.append(
                        f"Term '{term_name}' ({func_ref}): missing 'threshold' in params"
                    )
                if "sensor_cfg" not in params:
                    warnings.append(
                        f"Term '{term_name}' ({func_ref}): missing 'sensor_cfg' in params"
                    )

            # Weight sign sanity
            if weight is not None:
                # NaN weights are always wrong — skip subsequent checks
                if isinstance(weight, float) and weight != weight:  # NaN check
                    warnings.append(
                        f"Term '{term_name}' ({func_ref}): weight is NaN — check for float('nan') or undefined computation"
                    )
                    continue
                func_name = (
                    func_ref.split(".")[-1] if "." in func_ref else func_ref
                )
                # Task rewards (tracking) should be positive
                if "track" in func_name and weight <= 0:
                    warnings.append(
                        f"Term '{term_name}' ({func_ref}): task reward has non-positive weight {weight}"
                    )
                # Penalties (l2, limits, rate, terminated) should be negative
                penalty_markers = ["l2", "limits", "rate", "terminated", "deviation"]
                if any(m in func_name for m in penalty_markers) and weight > 0:
                    warnings.append(
                        f"Term '{term_name}' ({func_ref}): penalty has positive weight {weight}"
                    )

        return warnings


def main() -> None:
    """CLI entry point for reward validation."""
    parser = argparse.ArgumentParser(
        description="Validate generated Isaac Lab reward code (static checks).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file",
        "-f",
        required=True,
        help="Path to the reward.py file to validate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (non-zero exit on warnings).",
    )

    args = parser.parse_args()

    validator = RewardValidator()
    result = validator.validate_file(args.file)

    print(f"Valid: {result['valid']}")
    print(f"Checks performed: {', '.join(result['checks'])}")

    if result["errors"]:
        print("\nErrors:")
        for err in result["errors"]:
            print(f"  [ERROR] {err}")

    if result["warnings"]:
        print("\nWarnings:")
        for warn in result["warnings"]:
            print(f"  [WARN]  {warn}")

    # Exit code: non-zero if errors (or warnings in strict mode)
    if not result["valid"]:
        sys.exit(1)
    if args.strict and result["warnings"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
