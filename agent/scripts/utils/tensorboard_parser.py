"""TensorBoard event file parser — extracts scalar metrics from training logs.

Used by log_analyzer.py to load training metrics before anomaly detection.
Falls back gracefully if the tensorboard package is not installed, so the
skill can still be loaded in environments without tensorboard (though log
analysis features will be unavailable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tensorboard.backend.event_processing import event_accumulator
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False


MetricSeries = List[Tuple[int, float]]


class TensorboardParser:
    """Parses TensorBoard event files to extract scalar metrics.

    Handles a single run directory (the typical Isaac Lab training output
    layout). For multi-run directories, instantiate one parser per run.

    Example:
        parser = TensorboardParser("/path/to/tensorboard/run1")
        metrics = parser.parse()
        reward_series = metrics.get("Train/mean_reward", [])
    """

    def __init__(self, logdir: Path | str) -> None:
        """Initialize parser with a tensorboard log directory.

        Args:
            logdir: Path to the tensorboard run directory (contains event files).

        Raises:
            FileNotFoundError: If logdir does not exist.
        """
        self.logdir = Path(logdir)
        if not self.logdir.exists():
            raise FileNotFoundError(f"Log directory not found: {self.logdir}")

    @staticmethod
    def is_available() -> bool:
        """Check whether the tensorboard package is installed."""
        return _TENSORBOARD_AVAILABLE

    def list_metrics(self) -> List[str]:
        """List available scalar metric names in the log directory.

        Returns:
            List of metric tag strings.

        Raises:
            ImportError: If tensorboard package is not installed.
        """
        self._require_tensorboard()

        ea = event_accumulator.EventAccumulator(str(self.logdir))
        ea.Reload()
        return list(ea.Tags().get("scalars", []))

    def parse(self, max_step: Optional[int] = None) -> Dict[str, MetricSeries]:
        """Parse all scalar metrics from the log directory.

        Args:
            max_step: If provided, only include events with step <= max_step.
                Useful for analyzing a specific training window.

        Returns:
            Dict mapping metric tag to a list of (step, value) tuples,
            ordered by step ascending.

        Raises:
            ImportError: If tensorboard package is not installed.
        """
        self._require_tensorboard()

        ea = event_accumulator.EventAccumulator(
            str(self.logdir),
            size_guidance={
                event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                event_accumulator.IMAGES: 4,
                event_accumulator.AUDIO: 4,
                event_accumulator.SCALARS: 0,
                event_accumulator.TENSORS: 0,
            },
        )
        ea.Reload()

        metrics: Dict[str, MetricSeries] = {}
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            series: MetricSeries = [
                (e.step, float(e.value))
                for e in events
                if max_step is None or e.step <= max_step
            ]
            metrics[tag] = series

        return metrics

    def parse_metric(self, tag: str, max_step: Optional[int] = None) -> MetricSeries:
        """Parse a single metric by tag.

        Args:
            tag: Metric tag (e.g., "Train/mean_reward").
            max_step: Optional step cutoff.

        Returns:
            List of (step, value) tuples.

        Raises:
            KeyError: If tag does not exist in the log.
            ImportError: If tensorboard is not installed.
        """
        metrics = self.parse(max_step=max_step)
        if tag not in metrics:
            available = list(metrics.keys())
            raise KeyError(
                f"Metric tag '{tag}' not found. Available: {available}"
            )
        return metrics[tag]

    @staticmethod
    def _require_tensorboard() -> None:
        if not _TENSORBOARD_AVAILABLE:
            raise ImportError(
                "tensorboard package is required for log parsing. "
                "Install with: pip install tensorboard"
            )
