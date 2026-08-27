"""Pattern matcher — maps task descriptions to relevant reward patterns.

This is the "intelligence" layer of the synthesizer: given a natural language
task description, it decides which reward patterns to include and what default
weights/params to use.

Two-stage approach:
1. detect_task_type: NL → task type (keyword matching)
2. select_patterns: task type → pattern list (template-based selection)

Task type templates encode engineering knowledge about which patterns are
required/recommended/optional for each task type. Defaults are grounded in
legged_gym / IsaacLab configs (see resources/reward_patterns.json sources).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from scripts.reward_library import RewardLibrary


# Task type → pattern selection template
# required: must include (task won't work without these)
# recommended: standard engineering practice to include
# optional: include for specific scenarios (rough terrain, aggressive motion, etc.)
TASK_TYPE_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "locomotion_velocity": {
        "required": [
            "linear_velocity_tracking",
            "angular_velocity_tracking",
        ],
        "recommended": [
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "joint_torques_l2",
            "joint_acc_l2",
            "action_rate_l2",
            "feet_air_time",
            "undesired_contacts",
        ],
        "optional": [
            "flat_orientation_l2",
            "joint_pos_limits",
            "is_terminated",
        ],
    },
    "locomotion_rough_terrain": {
        "required": [
            "linear_velocity_tracking",
            "angular_velocity_tracking",
        ],
        "recommended": [
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "flat_orientation_l2",
            "joint_torques_l2",
            "joint_acc_l2",
            "action_rate_l2",
            "feet_air_time",
            "undesired_contacts",
            "joint_pos_limits",
            "is_terminated",
        ],
        "optional": [],
    },
}


# Keyword → task type detection (order matters: more specific types first)
TASK_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "locomotion_rough_terrain": [
        "崎岖", "地形", "rough terrain", "rough", "terrain",
        "爬", "climb", "stairs", "楼梯", "斜坡", "slope",
    ],
    "locomotion_velocity": [
        "走", "跑", "行走", "前进", "velocity", "walk", "run", "forward",
        "速度跟踪", "tracking", "locomotion", "四足", "quadruped",
        "机器狗", "biped", "双足",
    ],
}


# Default weights and params for each pattern (grounded in legged_gym/IsaacLab configs)
# These are the synthesizer's defaults; users can override via the config dict.
PATTERN_DEFAULTS: Dict[str, Dict[str, float]] = {
    "linear_velocity_tracking": {"weight": 1.0, "std": 0.25, "command_name": "base_velocity"},
    "angular_velocity_tracking": {"weight": 0.5, "std": 0.25, "command_name": "base_velocity"},
    "flat_orientation_l2": {"weight": 0.0},
    "joint_torques_l2": {"weight": -1.0e-5},
    "action_rate_l2": {"weight": -0.01},
    "feet_air_time": {"weight": 0.125, "threshold": 0.5, "command_name": "base_velocity"},
    "joint_pos_limits": {"weight": 0.0},
    "is_terminated": {"weight": 0.0},
    # Standard stability penalties (always included for locomotion)
    "lin_vel_z_l2": {"weight": -2.0},
    "ang_vel_xy_l2": {"weight": -0.05},
    "joint_acc_l2": {"weight": -2.5e-7},
    "undesired_contacts": {"weight": -1.0},
}


class PatternMatcher:
    """Maps task descriptions to relevant reward patterns.

    Example:
        lib = RewardLibrary()
        matcher = PatternMatcher(lib)
        selection = matcher.select_patterns("训练四足以 1 m/s 前进，保持稳定")
        # selection = {
        #     "task_type": "locomotion_velocity",
        #     "required": [...],
        #     "recommended": [...],
        #     "optional": [...],
        #     "patterns": [...],  # combined list
        #     "config": {...},    # default weights/params
        # }
    """

    def __init__(self, library: RewardLibrary) -> None:
        self.library = library

    def detect_task_type(self, task_description: str) -> str:
        """Detect task type from natural language description.

        Uses keyword matching. More specific types (rough_terrain) checked
        before generic types (velocity).

        Args:
            task_description: Natural language task description.

        Returns:
            Task type string (e.g., "locomotion_velocity").
        """
        desc_lower = task_description.lower()

        for task_type, keywords in TASK_TYPE_KEYWORDS.items():
            if any(kw.lower() in desc_lower for kw in keywords):
                return task_type

        # Default to velocity locomotion
        return "locomotion_velocity"

    def select_patterns(
        self,
        task_description: str,
        include_optional: bool = False,
        config_overrides: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, any]:
        """Select patterns for a task description and build default config.

        Args:
            task_description: Natural language task description.
            include_optional: Whether to include optional patterns.
            config_overrides: Dict mapping pattern_id to param overrides
                (e.g., {"linear_velocity_tracking": {"weight": 1.5}}).

        Returns:
            Dict with:
                - task_type: detected task type
                - required: list of required pattern IDs
                - recommended: list of recommended pattern IDs
                - optional: list of optional pattern IDs
                - patterns: combined list of all selected pattern IDs
                - config: dict mapping pattern_id to {weight, ...params}
        """
        task_type = self.detect_task_type(task_description)
        template = TASK_TYPE_PATTERNS.get(
            task_type, TASK_TYPE_PATTERNS["locomotion_velocity"]
        )

        required = list(template["required"])
        recommended = list(template["recommended"])
        optional = list(template["optional"]) if include_optional else []

        all_patterns = required + recommended + optional

        # Validate all pattern IDs exist in library
        for pid in all_patterns:
            if not self.library.validate_pattern_id(pid):
                raise ValueError(
                    f"Pattern '{pid}' in task type template but not in library. "
                    f"This is a skill-internal bug; the task type template and "
                    f"pattern database are out of sync."
                )

        # Build default config
        config = {}
        for pid in all_patterns:
            if pid in PATTERN_DEFAULTS:
                config[pid] = dict(PATTERN_DEFAULTS[pid])
            else:
                # Fallback: no defaults known, use neutral values
                config[pid] = {"weight": 0.0}

        # Apply user overrides
        if config_overrides:
            for pid, overrides in config_overrides.items():
                if pid in config:
                    config[pid].update(overrides)
                else:
                    config[pid] = dict(overrides)

        return {
            "task_type": task_type,
            "required": required,
            "recommended": recommended,
            "optional": optional,
            "patterns": all_patterns,
            "config": config,
        }

    def get_task_types(self) -> List[str]:
        """List all supported task types."""
        return list(TASK_TYPE_PATTERNS.keys())

    def explain_selection(self, selection: Dict) -> str:
        """Produce a human-readable explanation of a pattern selection.

        Useful for the synthesizer's output to help users understand why
        certain patterns were chosen.
        """
        lines = [
            f"Task type: {selection['task_type']}",
            f"",
            f"Required patterns ({len(selection['required'])}):",
        ]
        for pid in selection["required"]:
            pattern = self.library.get_pattern(pid)
            lines.append(f"  - {pid}: {pattern.get('purpose', '')}")

        lines.append(f"\nRecommended patterns ({len(selection['recommended'])}):")
        for pid in selection["recommended"]:
            pattern = self.library.get_pattern(pid)
            lines.append(f"  - {pid}: {pattern.get('purpose', '')}")

        if selection["optional"]:
            lines.append(f"\nOptional patterns ({len(selection['optional'])}):")
            for pid in selection["optional"]:
                pattern = self.library.get_pattern(pid)
                lines.append(f"  - {pid}: {pattern.get('purpose', '')}")

        lines.append(f"\nDefault config:")
        for pid, cfg in selection["config"].items():
            weight = cfg.get("weight", 0.0)
            lines.append(f"  - {pid}: weight={weight}")

        return "\n".join(lines)
