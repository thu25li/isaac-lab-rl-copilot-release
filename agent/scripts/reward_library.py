"""Reward pattern library — programmatic access to the curated pattern database.

Loads resources/reward_patterns.json and provides query methods for the
synthesizer and validator. This is the single source of truth for pattern
data; do not hardcode pattern info elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class RewardLibrary:
    """Provides access to the curated reward pattern database.

    The database is stored as JSON at resources/reward_patterns.json.
    Each pattern has: id, name, category, purpose, formula, isaac_lab_func,
    func_params, weight_range, when_to_use, pitfalls, template, sources.

    Example:
        lib = RewardLibrary()
        pattern = lib.get_pattern("linear_velocity_tracking")
        print(pattern["formula"])  # exp(-||v_cmd - v_base||^2 / std^2)
        locomotion_patterns = lib.list_patterns(category="task_reward")
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize library with pattern database path.

        Args:
            db_path: Path to reward_patterns.json. If None, resolves to
                the package's resources/reward_patterns.json.

        Raises:
            FileNotFoundError: If database file does not exist.
        """
        if db_path is None:
            package_root = Path(__file__).resolve().parent.parent
            db_path = package_root / "resources" / "reward_patterns.json"

        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Pattern database not found: {self.db_path}. "
                f"Ensure the skill package structure is intact."
            )

        with open(self.db_path, encoding="utf-8") as f:
            self._db: Dict[str, Any] = json.load(f)

        self._patterns_by_id: Dict[str, Dict[str, Any]] = {
            p["id"]: p for p in self._db.get("patterns", [])
        }

    @property
    def version(self) -> str:
        """Database version string."""
        return self._db.get("version", "unknown")

    @property
    def source(self) -> str:
        """Database source attribution."""
        return self._db.get("source", "unknown")

    def list_pattern_ids(self) -> List[str]:
        """List all pattern IDs in the database."""
        return list(self._patterns_by_id.keys())

    def list_patterns(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List patterns, optionally filtered by category.

        Args:
            category: If provided, only return patterns with this category.

        Returns:
            List of pattern dicts.
        """
        if category is None:
            return list(self._db.get("patterns", []))
        return [p for p in self._db.get("patterns", []) if p.get("category") == category]

    def get_pattern(self, pattern_id: str) -> Dict[str, Any]:
        """Get a pattern by ID.

        Args:
            pattern_id: The pattern's snake_case ID.

        Returns:
            Pattern dict with all fields.

        Raises:
            KeyError: If pattern_id not in database.
        """
        if pattern_id not in self._patterns_by_id:
            available = list(self._patterns_by_id.keys())
            raise KeyError(
                f"Pattern '{pattern_id}' not found. Available: {available}"
            )
        return self._patterns_by_id[pattern_id]

    def list_categories(self) -> List[str]:
        """List all unique pattern categories."""
        return list({p.get("category") for p in self._db.get("patterns", [])})

    def get_func_params(self, pattern_id: str) -> Dict[str, Any]:
        """Get the Isaac Lab func params for a pattern.

        These are the params passed to RewTerm(params={...}).
        """
        pattern = self.get_pattern(pattern_id)
        return dict(pattern.get("func_params", {}))

    def get_isaac_lab_func(self, pattern_id: str) -> str:
        """Get the Isaac Lab function reference (e.g., 'mdp.track_lin_vel_xy_exp')."""
        pattern = self.get_pattern(pattern_id)
        return pattern.get("isaac_lab_func", "")

    def get_composition_heuristics(self) -> Dict[str, Any]:
        """Get composition heuristic rules for combining patterns."""
        return self._db.get("composition_heuristics", {})

    def get_common_imports(self) -> List[str]:
        """Get common import statements for generated reward code."""
        return list(self._db.get("common_imports", []))

    def validate_pattern_id(self, pattern_id: str) -> bool:
        """Check if a pattern ID exists in the database."""
        return pattern_id in self._patterns_by_id
