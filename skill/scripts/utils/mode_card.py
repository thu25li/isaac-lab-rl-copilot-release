"""Mode selection card + status icon generator.

Two outputs:
- make_mode_select_card(): large PNG shown in the agent's welcome message on
  清小搭 platform. Two button-style blocks side by side: 教学模式 (green) /
  辅助模式 (blue). User selects by typing the mode name in their next message.
- make_mode_icon(mode): small 64x64 PNG icon attached to every assistant
  response, indicating the active mode.

Color scheme:
- 教学模式 (teaching): green (#2ca02c) — growth, learning
- 辅助模式 (assist):   blue  (#1f77b4) — engineering, fast
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal


Mode = Literal["teaching", "assist"]

TEACHING_COLOR = "#2ca02c"  # green
ASSIST_COLOR = "#1f77b4"    # blue
NEUTRAL_COLOR = "#7f7f7f"   # gray for unknown


def _mode_color(mode: str) -> str:
    return {"teaching": TEACHING_COLOR, "assist": ASSIST_COLOR}.get(mode, NEUTRAL_COLOR)


def _mode_label(mode: str) -> str:
    return {"teaching": "教学模式", "assist": "辅助模式"}.get(mode, mode)


def _setup_cjk_font():
    """Configure matplotlib to render CJK. Win: Microsoft YaHei / SimHei."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm

    # Pick the first available CJK font on the system
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun",       # Windows
        "PingFang SC", "Hiragino Sans GB",            # macOS
        "WenQuanYi Zen Hei", "Noto Sans CJK SC",     # Linux
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen


def make_mode_select_card(output_path: str | Path) -> Path:
    """Render the welcome-card PNG: two large buttons side by side.

    The buttons are not actually clickable — they're a visual prompt. The user
    must type "教学模式" or "辅助模式" in their next message to select.
    """
    _setup_cjk_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Title
    ax.text(5, 3.5, "请选择对话模式", ha="center", va="center",
            fontsize=22, weight="bold", color="#222222")
    ax.text(5, 2.95, "回复下方模式名即可切换（可随时切换）",
            ha="center", va="center", fontsize=12, color="#666666")

    # Two button panels
    # Teaching (left, green)
    teach_box = FancyBboxPatch(
        (0.4, 0.5), 4.2, 2,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=2, edgecolor=TEACHING_COLOR, facecolor=TEACHING_COLOR,
        alpha=0.85,
    )
    ax.add_patch(teach_box)
    ax.text(2.5, 1.95, "教学模式", ha="center", va="center",
            fontsize=20, weight="bold", color="white")
    ax.text(2.5, 1.35, "详细讲解 · 引用文档 · 类比举例", ha="center", va="center",
            fontsize=10, color="white", alpha=0.95)
    ax.text(2.5, 1.0, "理解检查 · 学习路径菜单", ha="center", va="center",
            fontsize=10, color="white", alpha=0.95)
    ax.text(2.5, 0.65, "适合学生 · 想理解原理", ha="center", va="center",
            fontsize=10, color="white", alpha=0.95, style="italic")

    # Assist (right, blue)
    assist_box = FancyBboxPatch(
        (5.4, 0.5), 4.2, 2,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=2, edgecolor=ASSIST_COLOR, facecolor=ASSIST_COLOR,
        alpha=0.85,
    )
    ax.add_patch(assist_box)
    ax.text(7.5, 1.95, "辅助模式", ha="center", va="center",
            fontsize=20, weight="bold", color="white")
    ax.text(7.5, 1.35, "直接给方案 · 重点结论", ha="center", va="center",
            fontsize=10, color="white", alpha=0.95)
    ax.text(7.5, 1.0, "工程化代码 · 数值参数范围", ha="center", va="center",
            fontsize=10, color="white", alpha=0.95)
    ax.text(7.5, 0.65, "适合工程师 · 赶工", ha="center", va="center",
            fontsize=10, color="white", alpha=0.95, style="italic")

    # Footer hint
    ax.text(5, 0.15, "在下方输入框发送 \"教学模式\" 或 \"辅助模式\"",
            ha="center", va="center", fontsize=11, color="#444444", weight="bold")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def make_mode_icon(mode: str, output_path: str | Path) -> Path:
    """Render a small 64x64 status icon for the active mode.

    Shows a colored circle with a single letter (T for teaching, A for assist).
    """
    _setup_cjk_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = _mode_color(mode)
    letter = {"teaching": "T", "assist": "A"}.get(mode, "?")

    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_aspect("equal")

    # Filled circle — pass facecolor/edgecolor separately (color overrides both)
    ax.add_patch(plt.Circle((0.5, 0.5), 0.42, facecolor=color,
                            edgecolor="black", linewidth=1.5))
    # Letter
    ax.text(0.5, 0.5, letter, ha="center", va="center",
            fontsize=22, weight="bold", color="white")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=64, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
