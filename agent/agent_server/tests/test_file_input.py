"""Tests for file input handling (清小搭 §3 protocol).

Covers:
- extract_file_parts: strips file parts, keeps text, handles pure-text msgs
- process_uploaded_file routing:
    * metrics JSON → log_analyzer + diagnosis_engine (with attachments)
    * env-like .py → config_validator
    * reward-like .py → reward_validator
    * non-metrics JSON → json_preview fallback
    * unsupported type → supported=False
- end-to-end via /v1/chat/completions with a multimodal message
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from core.file_input import extract_file_parts, process_uploaded_file


def _synthetic_metrics() -> dict:
    p = Path(__file__).parent.parent.parent / "tests" / "test_data" / "synthetic_policy_collapse" / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))


class TestExtractFileParts:
    def test_strips_file_keeps_text(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "帮我诊断这个日志"},
                {"type": "file", "file": {"url": "https://oss.example/metrics.json", "filename": "metrics.json"}},
            ],
        }]
        cleaned, files = extract_file_parts(msgs)
        assert len(files) == 1
        assert files[0]["filename"] == "metrics.json"
        assert cleaned[0]["content"] == "帮我诊断这个日志"
        # content must be a plain string now (DeepSeek-safe)
        assert isinstance(cleaned[0]["content"], str)

    def test_plain_messages_untouched(self):
        msgs = [
            {"role": "user", "content": "教学模式\n你好"},
            {"role": "assistant", "content": "你好"},
        ]
        cleaned, files = extract_file_parts(msgs)
        assert files == []
        assert cleaned == msgs

    def test_file_only_message_gets_placeholder(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "file", "file": {"url": "https://x/y.json", "filename": "y.json"}},
            ],
        }]
        cleaned, files = extract_file_parts(msgs)
        assert len(files) == 1
        assert cleaned[0]["content"] == "（用户上传了文件）"

    def test_non_http_url_dropped(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "file", "file": {"url": "ftp://bad/y.json", "filename": "y.json"}},
            ],
        }]
        cleaned, files = extract_file_parts(msgs)
        assert files == []  # unsafe scheme rejected

    def test_image_parts_dropped_silently(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://x/img.png"}},
                {"type": "text", "text": "看这张图"},
            ],
        }]
        cleaned, files = extract_file_parts(msgs)
        assert files == []
        assert "看这张图" in cleaned[0]["content"]
        assert "image" not in cleaned[0]["content"].lower()


class TestProcessUploadedFile:
    async def _serve(self, monkeypatch, content: bytes, filename: str):
        """Monkeypatch download_file to return canned bytes."""
        async def fake_download(url: str) -> bytes:
            return content
        import core.file_input as fi
        monkeypatch.setattr(fi, "download_file", fake_download)
        return await process_uploaded_file("https://fake/" + filename, filename)

    async def test_metrics_json_routes_to_diagnosis(self, monkeypatch):
        result, atts = await self._serve(
            monkeypatch,
            json.dumps(_synthetic_metrics()).encode("utf-8"),
            "metrics.json",
        )
        assert result["kind"] == "metrics_analysis"
        la = result["log_analysis"]
        assert la["metrics_analyzed"] >= 3
        assert "diagnosis" in result
        assert result["diagnosis"]["top_candidates"][0]["failure_mode_id"] == "policy_collapse"
        # Visualizations surfaced as attachments (pulled out of tool result)
        names = [a["fileName"] for a in atts]
        assert "anomalies.png" in names
        assert "diagnosis.png" in names

    async def test_env_py_routes_to_config_validator(self, monkeypatch):
        env_src = (Path(__file__).parent.parent.parent / "examples" /
                   "quadruped_locomotion" / "env.py").read_text(encoding="utf-8")
        result, atts = await self._serve(monkeypatch, env_src.encode("utf-8"), "env.py")
        assert result["kind"] == "config_validation"
        assert result["report"]["valid"] is True

    async def test_reward_py_routes_to_reward_validator(self, monkeypatch):
        reward_src = (Path(__file__).parent.parent.parent / "examples" /
                      "quadruped_locomotion" / "reward.py").read_text(encoding="utf-8")
        result, atts = await self._serve(monkeypatch, reward_src.encode("utf-8"), "reward.py")
        assert result["kind"] == "reward_validation"

    async def test_non_metrics_json_falls_back(self, monkeypatch):
        result, atts = await self._serve(
            monkeypatch, b'{"foo": "bar"}', "config.json",
        )
        assert result["kind"] == "json_preview"
        # Should NOT have run the diagnosis pipeline
        assert "diagnosis" not in result
        assert "log_analysis" not in result

    async def test_unsupported_type(self, monkeypatch):
        result, atts = await self._serve(monkeypatch, b"%PDF-1.4 fake", "report.pdf")
        assert result["supported"] is False

    async def test_download_failure_graceful(self, monkeypatch):
        import core.file_input as fi
        async def boom(url: str) -> bytes:
            raise httpx.ConnectError("no route")
        monkeypatch.setattr(fi, "download_file", boom)
        result, atts = await process_uploaded_file("https://dead/x.json", "x.json")
        assert result["supported"] is False
        assert "下载失败" in result["error"]


class TestFileInputViaEndpoint:
    def test_multimodal_message_end_to_end(self, client, valid_auth_header, mock_llm_client):
        """A 清小搭-style multimodal message flows through /v1/chat/completions."""
        # The file download would fail (no real URL) but must degrade gracefully:
        # message gets cleaned, chat proceeds.
        mock_llm_client.run_agent_loop.return_value = (
            [{"role": "assistant", "content": "已收到文件并处理"}],
            [],
            "assist",
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
        resp = client.post("/v1/chat/completions", headers=valid_auth_header, json={
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "辅助模式\n帮我看看这个日志"},
                    {"type": "file", "file": {"url": "https://example.com/metrics.json",
                                              "filename": "metrics.json"}},
                ],
            }],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"].startswith("[辅助模式]")
        # The cleaned message (not the multimodal one) reached run_agent_loop
        called_msgs = mock_llm_client.run_agent_loop.call_args[0][0]
        assert isinstance(called_msgs[0]["content"], str)
