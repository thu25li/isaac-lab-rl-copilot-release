"""Visualize metric series with detected symptoms overlaid.

Renders a multi-panel matplotlib figure:
- one subplot per metric (max 6 to keep readable)
- raw series in blue, smoothed trend in orange
- anomaly windows shaded red (error) / yellow (warning), with the strongest
  point marked
- title summarizing detection count

Used by:
- examples/end_to_end_demo.py to add a PNG to the report
- agent_server (optional) to surface as an attachment

Design choices:
- Headless-safe: uses Agg backend (no display required)
- No network/fonts deps beyond matplotlib
- Output: PNG at given path; caller handles cleanup
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Category → color (consistent across all reward plots)
_CATEGORY_COLORS = {
    "task_reward":         "#2ca02c",  # green — primary objectives
    "gait_reward":         "#17becf",  # cyan — gait shaping
    "stability_penalty":   "#1f77b4",  # blue — stability constraints
    "smoothness_penalty":  "#ff7f0e",  # orange — action smoothness
    "safety_penalty":      "#d62728",  # red — safety limits
    "energy_penalty":      "#9467bd",  # purple — energy / efficiency
}


def _color_for_category(category: str) -> str:
    """Category → matplotlib color. Unknown categories get a neutral gray."""
    return _CATEGORY_COLORS.get(category, "#7f7f7f")


def plot_metrics_with_symptoms(
    metrics: Dict[str, List[Tuple[int, float]]],
    symptoms: List[Dict[str, Any]],
    output_path: str | Path,
    *,
    title: str = "Training Metrics — Anomaly Detection",
    max_panels: int = 6,
) -> Path:
    """Render metrics + symptoms to a PNG file. Returns the output path.

    Args:
        metrics: {tag: [(step, value), ...]} — same shape LogAnalyzer consumes.
        symptoms: list of symptom dicts (Symptom.to_dict() output). Anomalies
            are overlaid on the corresponding metric panel.
        output_path: where to write the PNG (parent dir must exist).
        title: figure title.
        max_panels: render at most this many metric panels (most anomalies first).

    Raises:
        ValueError: if metrics is empty.
    """
    if not metrics:
        raise ValueError("metrics dict is empty — nothing to plot")

    # Lazy import — keeps module import cheap when plotting not needed
    import matplotlib
    matplotlib.use("Agg")  # headless backend, must be set before pyplot
    import matplotlib.pyplot as plt

    # Pick which metrics to plot: those with symptoms first, then by length desc
    tags_with_symptoms = {s.get("tag") for s in symptoms if s.get("tag")}
    ranked = sorted(
        metrics.items(),
        key=lambda kv: (kv[0] in tags_with_symptoms, -len(kv[1])),
    )
    ranked = ranked[:max_panels]

    n = len(ranked)
    # Layout: up to 2 columns, ceil rows
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 3 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    # Group symptoms by tag for shading
    symptoms_by_tag: Dict[str, List[Dict[str, Any]]] = {}
    for s in symptoms:
        tag = s.get("tag")
        if tag:
            symptoms_by_tag.setdefault(tag, []).append(s)

    for idx, (tag, series) in enumerate(ranked):
        ax = axes_flat[idx]
        if not series:
            ax.text(0.5, 0.5, f"{tag}\n(no data)", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        steps = [p[0] for p in series]
        values = [p[1] for p in series]
        ax.plot(steps, values, color="#1f77b4", alpha=0.6, linewidth=1, label="raw")

        # Simple moving-average smoothing if enough points
        if len(values) >= 10:
            window = max(3, len(values) // 20)
            smoothed: List[float] = []
            for i in range(len(values)):
                lo = max(0, i - window)
                hi = min(len(values), i + window + 1)
                smoothed.append(sum(values[lo:hi]) / (hi - lo))
            ax.plot(steps, smoothed, color="#ff7f0e", linewidth=2, label="smoothed")

        # Shade anomaly windows + mark strongest point
        for s in symptoms_by_tag.get(tag, []):
            sr = s.get("step_range", [None, None])
            sev = s.get("severity", "info")
            color = {"error": "#d62728", "warning": "#ffbb33", "info": "#7f7f7f"}.get(
                sev, "#7f7f7f"
            )
            if sr and sr[0] is not None and sr[1] is not None:
                ax.axvspan(sr[0], sr[1], alpha=0.18, color=color)
            # Mark the value_summary peak/trough location if present
            vs = s.get("value_summary", {})
            mark_step: Optional[int] = None
            mark_val: Optional[float] = None
            if s.get("pattern") in ("sudden_drop", "collapse", "plateau"):
                # mark trough
                if "trough" in vs and "trough_step" in vs:
                    mark_step, mark_val = vs["trough_step"], vs["trough"]
            elif s.get("pattern") in ("spike", "explosion"):
                if "peak" in vs and "peak_step" in vs:
                    mark_step, mark_val = vs["peak_step"], vs["peak"]
            if mark_step is not None and mark_val is not None:
                ax.scatter(
                    [mark_step], [mark_val], color=color, s=80, zorder=5,
                    edgecolors="black", linewidths=1,
                )
                ax.annotate(
                    s.get("pattern", ""),
                    xy=(mark_step, mark_val),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=9, color=color, weight="bold",
                )

        ax.set_title(f"{tag}", fontsize=10)
        ax.set_xlabel("step", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # Hide unused panels
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    error_count = sum(1 for s in symptoms if s.get("severity") == "error")
    warning_count = sum(1 for s in symptoms if s.get("severity") == "warning")
    fig.suptitle(
        f"{title}  ·  {error_count} errors, {warning_count} warnings",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_reward_weights(
    patterns: List[str],
    config: Dict[str, Dict[str, Any]],
    output_path: str | Path,
    *,
    title: str = "Synthesized Reward — Weight Composition",
) -> Path:
    """Render a horizontal bar chart of reward term weights, colored by category.

    Each bar = one reward term. Length = |weight|. Color = category
    (task_reward / stability_penalty / energy_penalty / ...). Negative weights
    shown as extending left from zero.

    Args:
        patterns: list of pattern ids in the synthesized reward (e.g. ['linear_velocity_tracking', ...]).
        config: {pattern_id: {"weight": float, ...}} from RewardSynthesizer output.
        output_path: PNG destination.
        title: figure title.

    Raises:
        ValueError: if patterns is empty or no pattern has a weight.
    """
    if not patterns:
        raise ValueError("patterns list is empty — nothing to plot")

    # Lazy import
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Resolve category per pattern via reward_library
    try:
        from scripts.reward_library import RewardLibrary
        lib = RewardLibrary()
        categories = {pid: lib.get_pattern(pid).get("category", "unknown") for pid in patterns}
        names = {pid: lib.get_pattern(pid).get("name", pid) for pid in patterns}
    except Exception:
        # Fallback: use the pid strings as-is if library isn't reachable
        categories = {pid: "unknown" for pid in patterns}
        names = {pid: pid for pid in patterns}

    # Pull weights from config; skip terms without a numeric weight
    rows = []
    for pid in patterns:
        cfg = config.get(pid, {})
        w = cfg.get("weight")
        if w is None or not isinstance(w, (int, float)):
            continue
        rows.append((pid, names.get(pid, pid), float(w), categories.get(pid, "unknown")))
    if not rows:
        raise ValueError("no pattern has a numeric weight in config")

    # Sort: rewards (positive) first by descending value, then penalties (negative) by ascending
    rows.sort(key=lambda r: (r[2] >= 0, -r[2] if r[2] >= 0 else r[2]))
    rows.reverse()  # so largest positive is at the top

    labels = [r[1] for r in rows]
    weights = [r[2] for r in rows]
    colors = [_color_for_category(r[3]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(rows) + 1.5)))

    bar_positions = range(len(rows))
    bars = ax.barh(bar_positions, weights, color=colors, edgecolor="black", linewidth=0.5)

    # Label each bar with the weight value
    for bar, w in zip(bars, weights):
        x = bar.get_width()
        x_text = x + (max(abs(min(weights)), abs(max(weights))) * 0.01) * (1 if x >= 0 else -1)
        ha = "left" if x >= 0 else "right"
        ax.text(x_text, bar.get_y() + bar.get_height() / 2,
                f"{w:+.4g}", va="center", ha=ha, fontsize=9, color="black")

    ax.set_yticks(bar_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("weight (signed)", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()  # top-down reading order

    # Legend for categories present in this figure
    from matplotlib.patches import Patch
    present_cats = []
    seen = set()
    for r in rows:
        c = r[3]
        if c not in seen:
            seen.add(c)
            present_cats.append(c)
    legend_handles = [
        Patch(facecolor=_color_for_category(c), edgecolor="black", linewidth=0.5, label=c)
        for c in present_cats
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    n_reward = sum(1 for w in weights if w > 0)
    n_penalty = sum(1 for w in weights if w < 0)
    fig.suptitle(
        f"{title}  ·  {n_reward} reward / {n_penalty} penalty terms",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_diagnosis_confidence(
    candidates: List[Dict[str, Any]],
    output_path: str | Path,
    *,
    title: str = "Diagnosis — Failure Mode Confidence",
    min_confidence: float = 0.0,
    max_to_show: int = 10,
) -> Path:
    """Horizontal bar chart of failure mode confidence scores.

    Each bar = one candidate failure mode. Length = confidence (0-1). Color =
    matched-ratio (high match = green, low = gray). Annotates the top-1 with
    a star marker.

    Args:
        candidates: list of DiagnosisCandidate.to_dict() outputs. Each must
            have failure_mode_id, confidence, matched (list), total_expected.
        output_path: PNG destination.
        title: figure title.
        min_confidence: filter out candidates below this threshold.
        max_to_show: cap number of candidates to keep chart readable.
    """
    if not candidates:
        raise ValueError("candidates list is empty — nothing to plot")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter + cap
    filtered = [c for c in candidates if c.get("confidence", 0) >= min_confidence]
    filtered = sorted(filtered, key=lambda c: c.get("confidence", 0), reverse=True)[:max_to_show]
    if not filtered:
        raise ValueError(f"no candidates above min_confidence={min_confidence}")

    # Build rows (top-first; we'll reverse for barh plotting so highest is on top)
    labels = []
    confidences = []
    colors = []
    matched_ratios = []
    for c in filtered:
        fm_id = c.get("failure_mode_id", "?")
        conf = float(c.get("confidence", 0))
        matched = c.get("matched", []) or []
        total = c.get("total_expected", 0) or max(len(matched), 1)
        ratio = len(matched) / max(total, 1)
        labels.append(fm_id.replace("_", " "))
        confidences.append(conf)
        matched_ratios.append(ratio)
        # Greener for higher match ratio
        if ratio >= 0.6:
            colors.append("#2ca02c")  # green
        elif ratio >= 0.3:
            colors.append("#ffbb33")  # yellow
        else:
            colors.append("#bdbdbd")  # gray

    # Reverse so top candidate appears at the top of the chart
    labels.reverse(); confidences.reverse(); colors.reverse(); matched_ratios.reverse()

    fig, ax = plt.subplots(figsize=(10, max(2, 0.5 * len(labels) + 1.5)))
    positions = list(range(len(labels)))
    bars = ax.barh(positions, confidences, color=colors,
                   edgecolor="black", linewidth=0.5)

    # Annotate each bar with confidence% and matched ratio
    for i, (bar, conf, ratio) in enumerate(zip(bars, confidences, matched_ratios)):
        x = bar.get_width()
        ax.text(min(x + 0.02, 1.0), bar.get_y() + bar.get_height() / 2,
                f"{conf:.0%}  ({matched_ratios[i]:.0%} symptoms)",
                va="center", ha="left", fontsize=9, color="black")

    # Star the top candidate (last in reversed order = highest)
    top_idx = len(labels) - 1
    ax.scatter([confidences[top_idx] / 2], [top_idx],
               marker="*", s=300, color="gold",
               edgecolors="black", linewidths=1, zorder=5)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("confidence", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()  # already reversed; this puts top at top

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2ca02c", edgecolor="black", linewidth=0.5,
              label="matched ≥60% symptoms"),
        Patch(facecolor="#ffbb33", edgecolor="black", linewidth=0.5,
              label="matched 30-60%"),
        Patch(facecolor="#bdbdbd", edgecolor="black", linewidth=0.5,
              label="matched <30%"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    top_id = filtered[0].get("failure_mode_id", "?").replace("_", " ")
    fig.suptitle(
        f"{title}  ·  top: {top_id} @ {confidences[-1]:.0%}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_dr_ranges(
    essential: List[str],
    recommended: List[str],
    optional: List[str],
    parameter_ranges: Dict[str, Dict[str, Any]],
    output_path: str | Path,
    *,
    title: str = "Domain Randomization — Parameter Ranges",
) -> Path:
    """Horizontal range chart of DR parameter ranges.

    Each row = one DR term. The bar shows the parameter range (min → max).
    Color = tier (essential green, recommended orange, optional gray).
    Numerical values annotated at both ends.

    Args:
        essential / recommended / optional: lists of term ids by tier.
        parameter_ranges: {term_id: {"<range_key>": (min, max), ...}}.
            If multiple range keys, the first numeric tuple is used.
        output_path: PNG destination.
        title: figure title.
    """
    all_terms = essential + recommended + optional
    if not all_terms:
        raise ValueError("no DR terms provided — nothing to plot")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Tier → color
    def tier_color(term_id: str) -> str:
        if term_id in essential:
            return "#2ca02c"  # green
        if term_id in recommended:
            return "#ff7f0e"  # orange
        return "#9467bd"  # purple

    # Build rows: (label, min, max, color)
    rows = []
    for term_id in all_terms:
        ranges = parameter_ranges.get(term_id, {})
        # Find first (min, max) tuple in the ranges dict
        picked = None
        range_key = None
        for k, v in ranges.items():
            if isinstance(v, (list, tuple)) and len(v) == 2 and all(
                isinstance(x, (int, float)) for x in v
            ):
                picked = (float(v[0]), float(v[1]))
                range_key = k
                break
        if picked is None:
            continue
        label = f"{term_id}  ({range_key})"
        rows.append((label, picked[0], picked[1], tier_color(term_id)))

    if not rows:
        raise ValueError("no term in parameter_ranges has a numeric (min, max) tuple")

    # Sort: by essential-first, then by range width descending
    def sort_key(r):
        label = r[0]
        if any(label.startswith(t) for t in essential):
            tier = 0
        elif any(label.startswith(t) for t in recommended):
            tier = 1
        else:
            tier = 2
        return (tier, -(r[2] - r[1]))

    rows.sort(key=sort_key)
    rows.reverse()  # for barh top-down order

    labels = [r[0] for r in rows]
    mins = [r[1] for r in rows]
    maxs = [r[2] for r in rows]
    colors = [r[3] for r in rows]
    widths = [mx - mn for mn, mx in zip(mins, maxs)]

    fig, ax = plt.subplots(figsize=(10, max(2, 0.5 * len(labels) + 1.5)))
    positions = list(range(len(labels)))
    bars = ax.barh(positions, widths, left=mins, color=colors,
                   edgecolor="black", linewidth=0.5, alpha=0.85)

    # Annotate min/max values
    for bar, mn, mx in zip(bars, mins, maxs):
        x_lo = bar.get_x()
        x_hi = x_lo + bar.get_width()
        span = abs(mx - mn) if mx != mn else 1.0
        # Decide decimal places
        def fmt(v):
            return f"{v:+.4g}" if v != 0 else "0"
        ax.text(x_lo - span * 0.02, bar.get_y() + bar.get_height() / 2, fmt(mn),
                va="center", ha="right", fontsize=8, color="black")
        ax.text(x_hi + span * 0.02, bar.get_y() + bar.get_height() / 2, fmt(mx),
                va="center", ha="left", fontsize=8, color="black")

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2ca02c", edgecolor="black", linewidth=0.5, label="essential"),
        Patch(facecolor="#ff7f0e", edgecolor="black", linewidth=0.5, label="recommended"),
        Patch(facecolor="#9467bd", edgecolor="black", linewidth=0.5, label="optional"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"{title}  ·  {len(essential)} essential / {len(recommended)} recommended / "
        f"{len(optional)} optional",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_curriculum_progression(
    essential: List[str],
    recommended: List[str],
    optional: List[str],
    term_details: Dict[str, Dict[str, Any]],
    output_path: str | Path,
    *,
    title: str = "Curriculum — Conservative Growth",
    n_steps: int = 100,
) -> Path:
    """Visualize curriculum progression: how parameter ranges expand over levels.

    For each term with param_defaults containing initial_range + final_range,
    render a "trumpet" shape showing range expanding from initial (level 0) to
    final (max level). The widening band visualizes the conservative-growth
    design principle.

    Args:
        essential / recommended / optional: term ids by tier.
        term_details: {term_id: term_metadata_dict}. Looks for
            param_defaults.{initial_range, final_range} keys.
        output_path: PNG destination.
        title: figure title.
        n_steps: resolution of level axis.
    """
    all_terms = essential + recommended + optional
    if not all_terms:
        raise ValueError("no curriculum terms provided — nothing to plot")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def tier_color(term_id: str) -> str:
        if term_id in essential:
            return "#2ca02c"
        if term_id in recommended:
            return "#ff7f0e"
        return "#9467bd"

    # Build progression rows
    rows = []
    for term_id in all_terms:
        td = term_details.get(term_id, {})
        params = td.get("param_defaults", {}) or {}
        initial = params.get("initial_range")
        final = params.get("final_range")
        if not (isinstance(initial, (list, tuple)) and isinstance(final, (list, tuple))
                and len(initial) == 2 and len(final) == 2):
            continue
        rows.append({
            "id": term_id,
            "name": td.get("name", term_id),
            "initial": (float(initial[0]), float(initial[1])),
            "final": (float(final[0]), float(final[1])),
            "color": tier_color(term_id),
            "isaac_lab_func": td.get("isaac_lab_func", ""),
        })

    if not rows:
        # Fallback: no term has initial/final ranges (common for rough_terrain
        # whose terrain_levels term uses different params). Render a tier-ranked
        # term list so the user still gets a visualization.
        return _render_curriculum_term_list(
            essential, recommended, optional, term_details, output_path, title,
        )

    fig, ax = plt.subplots(figsize=(10, max(2.5, 1.2 * len(rows) + 1.5)))

    levels = np.linspace(0, 1.0, n_steps)
    y_positions = list(range(len(rows)))
    bar_height = 0.6

    for i, row in enumerate(rows):
        # Linearly interpolate between initial and final ranges
        lo_initial, hi_initial = row["initial"]
        lo_final, hi_final = row["final"]
        los = lo_initial + (lo_final - lo_initial) * levels
        his = hi_initial + (hi_final - hi_initial) * levels

        # Fill the widening band
        ax.fill_betweenx(
            [i - bar_height / 2, i + bar_height / 2] * len(levels) if False else [i - bar_height / 2, i + bar_height / 2],
            [los[0], los[0]],
            [his[0], his[0]],
            color=row["color"], alpha=0.18,
        )
        # Actually fill_between on x=levels (level axis horizontal)
        ax.fill_between(levels, los + (i - bar_height / 2),
                        his + (i - bar_height / 2),
                        color=row["color"], alpha=0.35, edgecolor=row["color"],
                        linewidth=1.5)

        # Annotate initial and final values
        ax.text(-0.02, i, f"[{row['initial'][0]:+.2g}, {row['initial'][1]:+.2g}]",
                ha="right", va="center", fontsize=8, color=row["color"], weight="bold")
        ax.text(1.02, i, f"[{row['final'][0]:+.2g}, {row['final'][1]:+.2g}]",
                ha="left", va="center", fontsize=8, color=row["color"], weight="bold")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=10)
    ax.set_xlim(-0.15, 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["level 0\n(start)", "0.25", "0.5", "0.75", "max level\n(graduate)"])
    ax.set_xlabel("curriculum level", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2ca02c", edgecolor="black", linewidth=0.5, label="essential"),
        Patch(facecolor="#ff7f0e", edgecolor="black", linewidth=0.5, label="recommended"),
        Patch(facecolor="#9467bd", edgecolor="black", linewidth=0.5, label="optional"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"{title}  ·  {len(rows)} terms widen range as agent learns",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_curriculum_term_list(
    essential: List[str],
    recommended: List[str],
    optional: List[str],
    term_details: Dict[str, Dict[str, Any]],
    output_path: str | Path,
    title: str,
) -> Path:
    """Fallback renderer: tier-ranked horizontal bar chart when no term has
    initial_range/final_range. Shows each term as a labeled bar colored by tier
    (essential green, recommended orange, optional purple), with its
    isaac_lab_func annotated.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def tier_color(term_id: str) -> str:
        if term_id in essential:
            return "#2ca02c"
        if term_id in recommended:
            return "#ff7f0e"
        return "#9467bd"

    all_terms = list(reversed(essential + recommended + optional))  # top-down
    if not all_terms:
        raise ValueError("no curriculum terms provided")

    labels = []
    colors = []
    sublabels = []
    for term_id in all_terms:
        td = term_details.get(term_id, {})
        labels.append(td.get("name", term_id))
        sublabels.append(td.get("isaac_lab_func", ""))
        colors.append(tier_color(term_id))

    fig, ax = plt.subplots(figsize=(11, max(2.5, 0.55 * len(all_terms) + 1.5)))
    positions = list(range(len(all_terms)))
    # Bars all unit-width (just for color); the value isn't meaningful
    ax.barh(positions, [1.0] * len(all_terms), color=colors,
            edgecolor="black", linewidth=0.5, alpha=0.85)

    for i, sub in enumerate(sublabels):
        ax.text(0.02, i, sub, va="center", ha="left",
                fontsize=9, color="white", weight="bold")

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks([])
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2ca02c", edgecolor="black", linewidth=0.5, label="essential"),
        Patch(facecolor="#ff7f0e", edgecolor="black", linewidth=0.5, label="recommended"),
        Patch(facecolor="#9467bd", edgecolor="black", linewidth=0.5, label="optional"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    n_total = len(all_terms)
    fig.suptitle(
        f"{title}  ·  {len(essential)} essential / {len(recommended)} recommended / "
        f"{len(optional)} optional  (no ranged terms)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sim2real_radar(
    essential: List[str],
    recommended: List[str],
    optional: List[str],
    parameter_ranges: Dict[str, Dict[str, Any]],
    term_details: Dict[str, Dict[str, Any]],
    output_path: str | Path,
    *,
    robot_mass_kg: float = 15.0,
    title: str = "Sim-to-Real Readiness — DR Coverage",
) -> Path:
    """Radar chart of DR coverage by term. Each axis = one DR term; value =
    normalized coverage ratio (how wide the randomization range is relative
    to a sensible baseline).

    Coverage heuristic per term (higher = wider randomization):
    - robot_mass: |mass_range| / robot_mass_kg
    - ground_friction: (hi - lo) / 1.0
    - motor_strength: |strength_range| * 5  (typical ±20% → 1.0)
    - obs_noise: noise_std * 100  (typical 0.01 → 1.0)
    - center_of_mass_offset: |offset_range| * 10  (typical ±0.1m → 1.0)
    - joint_damping_friction: |range| * 5
    - action_delay: delay_range_s * 200  (typical 5-15ms → 1-3)
    fallback: take (hi-lo) of the first numeric range tuple, normalized.

    Values clipped to [0, 2] (1.0 = baseline; >1 = aggressive, <1 = conservative).
    """
    try:
        from scripts.utils.mode_card import _setup_cjk_font
        _setup_cjk_font()
    except Exception:
        pass
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def _normalize_coverage(term_id: str, ranges: dict) -> float:
        """Return a coverage score in [0, 2]. 1.0 = baseline."""
        # Pick first (min, max) tuple in ranges
        for k, v in ranges.items():
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                continue
            try:
                lo, hi = float(v[0]), float(v[1])
            except (TypeError, ValueError):
                continue
            span = abs(hi - lo)
            # Per-term heuristics
            if term_id == "robot_mass":
                return min(span / max(robot_mass_kg * 0.3, 1.0), 2.0)
            if term_id == "ground_friction":
                return min(span / 1.0, 2.0)
            if term_id == "motor_strength":
                return min(span * 5, 2.0)
            if term_id == "obs_noise":
                return min(span * 100, 2.0)
            if term_id == "center_of_mass_offset":
                return min(span * 10, 2.0)
            if term_id == "joint_damping_friction":
                return min(span * 5, 2.0)
            if term_id == "action_delay":
                return min(span * 200, 2.0)
            # Generic fallback: span (unitless)
            return min(span, 2.0)
        return 0.0

    all_terms = essential + recommended + optional
    if not all_terms:
        raise ValueError("no DR terms — radar needs at least 1 axis")

    # Take up to 8 axes for readability
    axes_terms = all_terms[:8]
    values = [_normalize_coverage(t, parameter_ranges.get(t, {})) for t in axes_terms]
    # Short labels
    short_names = []
    for t in axes_terms:
        td = term_details.get(t, {})
        # Use isaac_lab_func or just term_id
        short_names.append(td.get("name", t).replace("Randomization", "").strip()[:18])

    # Colors by tier
    tier_colors = []
    for t in axes_terms:
        if t in essential:
            tier_colors.append("#2ca02c")
        elif t in recommended:
            tier_colors.append("#ff7f0e")
        else:
            tier_colors.append("#9467bd")

    # Setup radar
    N = len(axes_terms)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values_closed = values + values[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw the baseline ring at 1.0
    ax.plot(angles, [1.0] * len(angles), color="#cccccc", linewidth=1, linestyle="--", alpha=0.7)
    ax.fill(angles, values_closed, color="#2ca02c", alpha=0.25)
    ax.plot(angles, values_closed, color="#2ca02c", linewidth=2)

    # Annotate each value
    for ang, val, color, label in zip(angles[:-1], values, tier_colors, short_names):
        ax.scatter(ang, val, color=color, s=80, zorder=5,
                   edgecolors="black", linewidths=1)
        # Place label slightly outside the value point
        ax.text(ang, min(val + 0.15, 2.2), f"{val:.2f}",
                ha="center", va="center", fontsize=9, weight="bold",
                color=color)

    # Axis labels (term names)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_names, fontsize=10)
    # Radial limits
    ax.set_ylim(0, 2.0)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(["0.5\nconservative", "1.0\nbaseline", "1.5\naggressive", "2.0\nvery aggressive"], fontsize=8)
    ax.yaxis.set_tick_params(pad=8)

    # Tier legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2ca02c", edgecolor="black", linewidth=0.5, label="essential"),
        Patch(facecolor="#ff7f0e", edgecolor="black", linewidth=0.5, label="recommended"),
        Patch(facecolor="#9467bd", edgecolor="black", linewidth=0.5, label="optional"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.25, 1.10),
              fontsize=9, framealpha=0.9)

    fig.suptitle(title, fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
