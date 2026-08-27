"""Code emitter — Jinja2-based template rendering for Isaac Lab code generation.

Used by reward_synthesizer.py to render reward function templates into
executable Python code. Templates live in the package's templates/ directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined


class CodeEmitter:
    """Renders Jinja2 templates for code generation.

    Templates are loaded from the templates/ directory at the package root.
    Uses StrictUndefined to fail fast on missing context variables — this
    catches synthesis bugs during development rather than at runtime.

    Example:
        emitter = CodeEmitter()
        reward_code = emitter.render(
            "rewards/locomotion_velocity.py.tmpl",
            {"target_velocity": 1.0, "std": 0.5, "weight": 1.5},
        )
    """

    def __init__(self, templates_dir: Optional[Path] = None) -> None:
        """Initialize emitter with templates directory.

        Args:
            templates_dir: Path to templates directory. If None, resolves to
                the package root's templates/ folder (relative to this file).

        Raises:
            FileNotFoundError: If templates directory does not exist.
        """
        if templates_dir is None:
            package_root = Path(__file__).resolve().parent.parent.parent
            templates_dir = package_root / "templates"

        templates_dir = Path(templates_dir)
        if not templates_dir.exists():
            raise FileNotFoundError(
                f"Templates directory not found: {templates_dir}. "
                f"Ensure the skill package structure is intact."
            )

        self.templates_dir = templates_dir
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    def render(self, template_path: str, context: Dict[str, Any]) -> str:
        """Render a template with the given context.

        Args:
            template_path: Relative path to template, using forward slashes
                (e.g., "rewards/locomotion_velocity.py.tmpl").
            context: Dictionary of variables passed to the template.

        Returns:
            Rendered string.

        Raises:
            jinja2.TemplateNotFound: If template does not exist.
            jinja2.UndefinedError: If a variable used in template is missing
                from context.
        """
        template = self.env.get_template(template_path)
        return template.render(**context)

    def emit(
        self,
        template_path: str,
        context: Dict[str, Any],
        output_path: Path | str,
    ) -> Path:
        """Render template and write the result to a file.

        Args:
            template_path: Relative path to template.
            context: Dictionary of variables.
            output_path: Path to write rendered code. Parent directories are
                created if they do not exist.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rendered = self.render(template_path, context)
        output_path.write_text(rendered, encoding="utf-8")

        return output_path

    def list_templates(self, subdir: Optional[str] = None) -> list[str]:
        """List available templates, optionally filtered by subdirectory.

        Args:
            subdir: If provided, only list templates under this subdirectory.

        Returns:
            List of relative template paths (forward slashes).
        """
        search_root = self.templates_dir / subdir if subdir else self.templates_dir
        if not search_root.exists():
            return []

        return sorted(
            str(p.relative_to(self.templates_dir)).replace("\\", "/")
            for p in search_root.rglob("*")
            if p.is_file()
        )
