"""Tests for mode detection + mode-aware responses."""
from __future__ import annotations

from core.llm_client import detect_mode, DEFAULT_MODE


class TestDetectMode:
    def test_default_when_no_signal(self):
        msgs = [{"role": "user", "content": "帮我生成 reward"}]
        assert detect_mode(msgs) == DEFAULT_MODE

    def test_teaching_keyword(self):
        msgs = [{"role": "user", "content": "教学模式"}]
        assert detect_mode(msgs) == "teaching"

    def test_assist_keyword(self):
        msgs = [{"role": "user", "content": "辅助模式"}]
        assert detect_mode(msgs) == "assist"

    def test_keyword_as_substring(self):
        """"我用教学模式" should still match teaching."""
        msgs = [{"role": "user", "content": "我用教学模式学 reward"}]
        assert detect_mode(msgs) == "teaching"

    def test_english_keyword(self):
        msgs = [{"role": "user", "content": "switch to teaching mode please"}]
        assert detect_mode(msgs) == "teaching"

    def test_latest_message_wins(self):
        """If user said both modes, the latest user message decides."""
        msgs = [
            {"role": "user", "content": "教学模式"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "改成辅助模式"},
        ]
        assert detect_mode(msgs) == "assist"

    def test_switch_back_forth(self):
        """Multiple switches — latest wins."""
        msgs = [
            {"role": "user", "content": "教学模式"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "辅助模式"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "教学模式"},
        ]
        assert detect_mode(msgs) == "teaching"

    def test_case_insensitive(self):
        msgs = [{"role": "user", "content": "TEACHING MODE"}]
        assert detect_mode(msgs) == "teaching"

    def test_non_user_messages_ignored(self):
        """Only user messages count — assistant mentioning 'mode' doesn't switch."""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "我可以切换到教学模式或辅助模式"},
        ]
        assert detect_mode(msgs) == DEFAULT_MODE


class TestPrompts:
    """Sanity checks on the two SYSTEM_PROMPTs."""

    def test_teaching_prompt_mentions_interactive_elements(self):
        from core.prompts import SYSTEM_PROMPT_TEACHING
        # Must mention key teaching-mode features
        assert "原理" in SYSTEM_PROMPT_TEACHING
        assert "类比" in SYSTEM_PROMPT_TEACHING
        assert "理解检查" in SYSTEM_PROMPT_TEACHING
        assert "学习路径菜单" in SYSTEM_PROMPT_TEACHING

    def test_assist_prompt_mentions_concise(self):
        from core.prompts import SYSTEM_PROMPT_ASSIST
        # Must emphasize conciseness
        assert "精简" in SYSTEM_PROMPT_ASSIST or "结论" in SYSTEM_PROMPT_ASSIST
        # Must NOT mandate teaching-mode interactive elements (e.g. "必问 1 个理解检查问题")
        assert "必问" not in SYSTEM_PROMPT_ASSIST
        assert "学习路径菜单" not in SYSTEM_PROMPT_ASSIST

    def test_get_prompt_falls_back_to_assist(self):
        from core.prompts import get_prompt
        assert get_prompt("teaching") != get_prompt("assist")
        assert get_prompt("unknown") == get_prompt("assist")
