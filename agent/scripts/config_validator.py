#!/usr/bin/env python3
"""Config validator — static validation of Isaac Lab env config files.

Performs AST-based checks (does NOT execute the code, so it is safe on
untrusted input). Complements reward_validator.py: where reward_validator
checks a RewardsCfg in isolation, config_validator checks the full
ManagerBasedRLEnvCfg assembly.

Checks:
1. syntax: Python syntax valid
2. imports: Required imports present (configclass, ManagerBasedRLEnvCfg)
3. env_cfg_class: A @configclass class inheriting from ManagerBasedRLEnvCfg
4. required_fields: Env cfg has scene / actions / observations / rewards /
   terminations
5. observation_groups: ObservationsCfg has at least a policy group
6. action_terms: ActionsCfg has at least one action term
7. rewards_reference: Rewards field references a RewardsCfg class
8. common_mistakes: Missing num_envs, episode_length_s, decimation;
   out-of-range values; suspicious action scale; etc.

Usage (CLI):
    python scripts/config_validator.py --file path/to/env.py

Usage (library):
    from scripts.config_validator import ConfigValidator
    validator = ConfigValidator()
    result = validator.validate_code(code_string)
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow direct script execution
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Imports that must appear (by name) somewhere in the file
REQUIRED_IMPORT_NAMES = ["configclass"]

# Base class names that indicate an env cfg. We accept any of these —
# users may inherit from ManagerBasedRLEnvCfg directly or a subclass.
ENV_CFG_BASE_NAMES = {
    "ManagerBasedRLEnvCfg",
    "ManagerBasedEnvCfg",
    "LocomotionEnvCfg",
    "RLEnvCfg",
}

# Required fields on the env cfg class
REQUIRED_ENV_FIELDS = ["scene", "actions", "observations", "rewards"]

# Fields that are strongly recommended (warning if missing, not error)
RECOMMENDED_ENV_FIELDS = ["terminations", "num_envs", "episode_length_s", "decimation"]

# Valid observation group names (policy is required, critic is optional)
REQUIRED_OBS_GROUPS = ["policy"]

# Common action term class names
ACTION_TERM_NAMES = {"ActionTermCfg", "JointPositionActionCfg", "JointEffortActionCfg"}

# Reasonable ranges for env-level scalars (warnings, not errors)
NUM_ENVS_RANGE = (64, 16384)
EPISODE_LENGTH_S_RANGE = (1.0, 120.0)
DECIMATION_RANGE = (1, 20)
ACTION_SCALE_RANGE = (0.01, 1.0)


class ConfigValidator:
    """Static validator for Isaac Lab env config files.

    Uses AST parsing for robust checks. Does not execute the code, so
    safe on untrusted input.
    """

    def validate_code(self, code: str) -> Dict[str, Any]:
        """Validate env config source code.

        Args:
            code: Python source code of an env config file.

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
            return {"valid": False, "checks": checks, "errors": errors, "warnings": warnings}

        # Check 2: Imports
        checks.append("imports")
        errors.extend(self._check_imports(tree))

        # Check 3: Find env cfg class
        checks.append("env_cfg_class")
        env_cfg_node, env_cfg_base = self._find_env_cfg_class(tree)
        if env_cfg_node is None:
            errors.append(
                f"No @configclass class inheriting from one of "
                f"{sorted(ENV_CFG_BASE_NAMES)} found."
            )
            return {"valid": False, "checks": checks, "errors": errors, "warnings": warnings}
        if env_cfg_base is None:
            warnings.append(
                "Env cfg class found but could not confirm base class — "
                "make sure it inherits from ManagerBasedRLEnvCfg."
            )

        # Check 4: Required fields on env cfg
        checks.append("required_fields")
        field_issues, env_fields = self._check_env_fields(env_cfg_node)
        errors.extend(field_issues)

        # Check 5: Observation groups
        checks.append("observation_groups")
        if "observations" in env_fields:
            obs_issues = self._check_observation_groups(tree, env_fields["observations"])
            errors.extend(obs_issues)

        # Check 6: Action terms
        checks.append("action_terms")
        if "actions" in env_fields:
            action_issues = self._check_action_terms(tree, env_fields["actions"])
            errors.extend(action_issues)

        # Check 7: Rewards reference
        checks.append("rewards_reference")
        if "rewards" in env_fields:
            rew_issues = self._check_rewards_reference(tree, env_fields["rewards"])
            warnings.extend(rew_issues)

        # Check 8: Common mistakes (scalar fields)
        checks.append("common_mistakes")
        warnings.extend(self._check_common_mistakes(env_cfg_node, env_fields, tree))

        return {
            "valid": len(errors) == 0,
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
        }

    def validate_file(self, path: Path | str) -> Dict[str, Any]:
        """Validate env config from a file."""
        path = Path(path)
        if not path.exists():
            return {
                "valid": False, "checks": [], "errors": [f"File not found: {path}"],
                "warnings": [],
            }
        if path.is_dir():
            return {
                "valid": False, "checks": [],
                "errors": [f"Path is a directory, not a file: {path}"],
                "warnings": [],
            }
        code = path.read_text(encoding="utf-8")
        return self.validate_code(code)

    # ------------------------------------------------------------------
    # Check implementations
    # ------------------------------------------------------------------

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

        for required in REQUIRED_IMPORT_NAMES:
            if required not in imported_names:
                errors.append(f"Missing required import: {required}")

        return errors

    def _find_env_cfg_class(
        self, tree: ast.AST
    ) -> Tuple[Optional[ast.ClassDef], Optional[str]]:
        """Find the @configclass class inheriting from an env cfg base.

        Returns (class_node, base_name) or (None, None) if not found.
        """
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not self._has_configclass_decorator(node):
                continue
            for base in node.bases:
                base_name = self._extract_name(base)
                if base_name in ENV_CFG_BASE_NAMES:
                    return node, base_name
            # Also accept a class whose name ends in "EnvCfg" even if base
            # isn't recognized (warning, not error)
            if node.name.endswith("EnvCfg") or node.name.endswith("EnvironmentCfg"):
                return node, None

        return None, None

    @staticmethod
    def _has_configclass_decorator(node: ast.ClassDef) -> bool:
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == "configclass":
                return True
            if isinstance(dec, ast.Attribute) and dec.attr == "configclass":
                return True
        return False

    @staticmethod
    def _extract_name(node: ast.AST) -> str:
        """Extract a readable name from an AST node (for base class checks)."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _check_env_fields(
        self, env_cfg_node: ast.ClassDef
    ) -> Tuple[List[str], Dict[str, ast.AST]]:
        """Check required fields exist on env cfg; return field name → value node."""
        errors: List[str] = []
        fields: Dict[str, ast.AST] = {}

        for stmt in env_cfg_node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                fields[stmt.target.id] = stmt.value
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        fields[target.id] = stmt.value

        for required in REQUIRED_ENV_FIELDS:
            if required not in fields:
                errors.append(
                    f"Env cfg missing required field: '{required}'"
                )

        for recommended in RECOMMENDED_ENV_FIELDS:
            if recommended not in fields:
                # Only warn, don't error
                pass  # handled in common_mistakes

        return errors, fields

    def _check_observation_groups(
        self, tree: ast.AST, obs_field: ast.AST
    ) -> List[str]:
        """Check that ObservationsCfg has a 'policy' group."""
        errors: List[str] = []

        # The obs field might be assigned a class instance or a class reference.
        # Try to find the class definition.
        obs_class_name = self._extract_class_name_from_value(obs_field)
        if obs_class_name is None:
            # Inline assignment, check the class body directly
            return self._check_inline_obs_groups(obs_field)

        # Find the class definition in the tree
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == obs_class_name:
                return self._check_inline_obs_groups(node)

        # Class not found — might be imported; can't verify, just warn
        return []

    def _check_inline_obs_groups(self, node: ast.AST) -> List[str]:
        """Check obs groups on a class body or AnnAssign value."""
        errors: List[str] = []
        found_groups: List[str] = []

        # If it's a Call (e.g., SomeCfg()), we can't introspect — skip
        if not hasattr(node, "body"):
            return errors

        for stmt in node.body:  # type: ignore
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                name = stmt.target.id if isinstance(stmt, ast.AnnAssign) else (
                    stmt.targets[0].id if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name) else None
                )
                if name and not name.startswith("_"):
                    found_groups.append(name)

        for required in REQUIRED_OBS_GROUPS:
            if required not in found_groups:
                errors.append(
                    f"ObservationsCfg missing required group: '{required}'. "
                    f"Found: {found_groups}"
                )

        return errors

    def _check_action_terms(
        self, tree: ast.AST, action_field: ast.AST
    ) -> List[str]:
        """Check that ActionsCfg has at least one action term."""
        errors: List[str] = []

        action_class_name = self._extract_class_name_from_value(action_field)
        if action_class_name is None:
            return self._check_inline_action_terms(action_field)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == action_class_name:
                return self._check_inline_action_terms(node)

        return []

    def _check_inline_action_terms(self, node: ast.AST) -> List[str]:
        """Check action terms on a class body."""
        errors: List[str] = []
        if not hasattr(node, "body"):
            return errors

        term_count = 0
        for stmt in node.body:  # type: ignore
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Call):
                if self._is_action_term(stmt.value):
                    term_count += 1
            elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                if self._is_action_term(stmt.value):
                    term_count += 1

        if term_count == 0:
            errors.append(
                "ActionsCfg has no action terms. Add at least one "
                "(e.g., JointPositionActionCfg)."
            )

        return errors

    @staticmethod
    def _is_action_term(call: ast.Call) -> bool:
        """Check if a Call is an action term (ActionTermCfg or subclass)."""
        func_name = ""
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        return func_name in ACTION_TERM_NAMES or func_name.endswith("ActionCfg")

    def _check_rewards_reference(
        self, tree: ast.AST, reward_field: ast.AST
    ) -> List[str]:
        """Check that rewards field references a RewardsCfg class (warnings only).

        Recognizes the standard Isaac Lab patterns:
            rewards: RewardsCfg = RewardsCfg()        # instantiated
            rewards = RewardsCfg()                    # instantiated, no annotation
            rewards = some_other_RewardsCfg_class()   # subclass instance
        """
        warnings: List[str] = []

        # Case 1: direct Name reference (e.g., `rewards = SomeCfg`)
        if isinstance(reward_field, ast.Name):
            ref_name = reward_field.id
            if ref_name == "RewardsCfg":
                return []
            # Check if the referenced class is defined in this file
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == ref_name:
                    return []
            warnings.append(
                f"Rewards field references '{ref_name}' which is not defined "
                f"in this file. If imported, ensure it's a valid RewardsCfg subclass."
            )
            return []

        # Case 2: Call (e.g., `RewardsCfg()` or `MyRewardsCfg()`)
        if isinstance(reward_field, ast.Call):
            func_name = ""
            if isinstance(reward_field.func, ast.Name):
                func_name = reward_field.func.id
            elif isinstance(reward_field.func, ast.Attribute):
                func_name = reward_field.func.attr

            # Standard pattern: RewardsCfg() or *RewardsCfg()
            if func_name == "RewardsCfg" or func_name.endswith("RewardsCfg"):
                return []

            # Check if the called class is defined in this file
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == func_name:
                    return []

            warnings.append(
                f"Rewards field instantiates '{func_name}'. If this is a "
                f"RewardsCfg subclass (imported or defined elsewhere), fine. "
                f"Otherwise, use a RewardsCfg with @configclass."
            )
            return warnings

        return warnings

    @staticmethod
    def _extract_class_name_from_value(node: ast.AST) -> Optional[str]:
        """Extract class name from a field assignment.

        Handles: `field = ClassName()` → "ClassName"
                 `field = ClassName` → "ClassName"
                 `field: ClassName = ClassName()` → "ClassName"
        """
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
            if isinstance(node.func, ast.Attribute):
                return node.func.attr
        return None

    def _check_common_mistakes(
        self, env_cfg_node: ast.ClassDef, fields: Dict[str, ast.AST],
        tree: Optional[ast.AST] = None,
    ) -> List[str]:
        """Warn on common env config mistakes."""
        warnings: List[str] = []

        # Missing recommended fields
        for recommended in RECOMMENDED_ENV_FIELDS:
            if recommended not in fields and recommended not in REQUIRED_ENV_FIELDS:
                warnings.append(
                    f"Env cfg missing recommended field: '{recommended}'"
                )

        # num_envs range
        if "num_envs" in fields:
            num_envs = self._extract_literal(fields["num_envs"])
            if isinstance(num_envs, bool):
                # bool is subclass of int but not a valid num_envs
                warnings.append(
                    f"num_envs={num_envs} is a boolean, should be an int."
                )
            elif isinstance(num_envs, float):
                warnings.append(
                    f"num_envs={num_envs} is a float, should be an int. "
                    f"Typical: 1024-4096."
                )
            elif isinstance(num_envs, int):
                lo, hi = NUM_ENVS_RANGE
                if num_envs < lo:
                    warnings.append(
                        f"num_envs={num_envs} is very low (< {lo}); "
                        f"RL training usually needs >= 64 envs for variance reduction."
                    )
                elif num_envs > hi:
                    warnings.append(
                        f"num_envs={num_envs} is very high (> {hi}); "
                        f"may exceed GPU memory. Typical: 1024-4096."
                    )
            elif num_envs is None:
                # Value is a variable or expression — can't verify range
                warnings.append(
                    f"num_envs is assigned a non-literal value (variable or expression); "
                    f"cannot verify range. Ensure it resolves to 64-16384."
                )

        # episode_length_s range
        if "episode_length_s" in fields:
            ep_len = self._extract_literal(fields["episode_length_s"])
            if isinstance(ep_len, (int, float)):
                lo, hi = EPISODE_LENGTH_S_RANGE
                if ep_len < lo:
                    warnings.append(
                        f"episode_length_s={ep_len} is very short (< {lo}s); "
                        f"agent won't have time to demonstrate behavior."
                    )
                elif ep_len > hi:
                    warnings.append(
                        f"episode_length_s={ep_len} is very long (> {hi}s); "
                        f"may slow training. Typical: 10-30s."
                    )

        # decimation range
        if "decimation" in fields:
            dec = self._extract_literal(fields["decimation"])
            if isinstance(dec, int):
                lo, hi = DECIMATION_RANGE
                if dec < lo:
                    warnings.append(
                        f"decimation={dec} is too low (< {lo}); "
                        f"physics frequency would be too high."
                    )
                elif dec > hi:
                    warnings.append(
                        f"decimation={dec} is very high (> {hi}); "
                        f"control frequency would be too low for stable motion."
                    )

        # action scale (look inside ActionsCfg)
        if "actions" in fields:
            scale_warns = self._check_action_scale(fields["actions"], tree=tree)
            warnings.extend(scale_warns)

        return warnings

    def _check_action_scale(self, action_field: ast.AST, tree: Optional[ast.AST] = None) -> List[str]:
        """Warn if action scale is outside typical range.

        Looks for `scale=...` kwarg in action term calls. Searches both
        the action_field node itself and (if tree is provided) the class
        body referenced by the action_field.
        """
        warnings: List[str] = []
        nodes_to_check: List[ast.AST] = [action_field]

        # If action_field is a Call like ActionsCfg(), find the class body
        if tree is not None:
            class_name = self._extract_class_name_from_value(action_field)
            if class_name:
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        nodes_to_check.append(node)
                        break

        for root in nodes_to_check:
            for node in ast.walk(root):
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if kw.arg == "scale":
                            scale = self._extract_literal(kw.value)
                            if isinstance(scale, (int, float)):
                                lo, hi = ACTION_SCALE_RANGE
                                if scale < lo:
                                    warnings.append(
                                        f"action scale={scale} is very small (< {lo}); "
                                        f"agent actions will have minimal effect."
                                    )
                                elif scale > hi:
                                    warnings.append(
                                        f"action scale={scale} is large (> {hi}); "
                                        f"may cause instability."
                                    )

        return warnings

    @staticmethod
    def _extract_literal(node: ast.AST) -> Any:
        """Extract a literal value (int, float, str, bool) from an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = ConfigValidator._extract_literal(node.operand)
            if isinstance(val, (int, float)):
                return -val
        return None


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Isaac Lab env config files (static AST checks).",
    )
    parser.add_argument("--file", "-f", required=True, help="Path to env.py")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors.")
    args = parser.parse_args()

    validator = ConfigValidator()
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

    if not result["valid"]:
        sys.exit(1)
    if args.strict and result["warnings"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
