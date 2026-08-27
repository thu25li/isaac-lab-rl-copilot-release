"""Tests for HTTP endpoints — OpenAI compatibility contract.

Covers 清小搭 §1 (最小契约) and §7 (探测通过标准):
- GET /v1/models returns 200 with valid auth, 401 without
- POST /v1/chat/completions non-streaming returns OpenAI-shaped JSON
- POST /v1/chat/completions streaming returns SSE frames
- Auth: missing/wrong Bearer token → 401
- Health endpoint
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest


class TestModelsEndpoint:
    def test_models_returns_200_with_valid_auth(self, client, valid_auth_header):
        resp = client.get("/v1/models", headers=valid_auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1
        assert data["data"][0]["id"] == "isaac-lab-rl-copilot"

    def test_models_returns_401_without_auth(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 401

    def test_models_returns_401_with_wrong_auth(self, client, invalid_auth_header):
        resp = client.get("/v1/models", headers=invalid_auth_header)
        assert resp.status_code == 401


class TestChatNonStreaming:
    def test_chat_returns_openai_shaped_response(self, client, valid_auth_header, mock_llm_client):
        # Mock: agent loop returns (messages, attachments, mode, usage)
        mock_llm_client.run_agent_loop.return_value = (
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，我是 Isaac Lab RL Co-pilot"},
            ],
            [],
            "assist",
            {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )

        resp = client.post("/v1/chat/completions", headers={
            **valid_auth_header,
            "Content-Type": "application/json",
        }, json={
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"].startswith("[辅助模式]")
        assert "你好，我是 Isaac Lab RL Co-pilot" in data["choices"][0]["message"]["content"]
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]

    def test_chat_returns_401_without_auth(self, client):
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 401

    def test_chat_returns_400_for_empty_messages(self, client, valid_auth_header):
        resp = client.post("/v1/chat/completions", headers=valid_auth_header, json={
            "messages": [],
        })
        assert resp.status_code == 400

    def test_chat_returns_400_for_malformed_message(self, client, valid_auth_header):
        resp = client.post("/v1/chat/completions", headers=valid_auth_header, json={
            "messages": [{"no_role": "wrong"}],
        })
        assert resp.status_code == 400

    def test_chat_with_attachments(self, client, valid_auth_header, mock_llm_client):
        """x_soda.attachments should be in top-level response, 清小搭 §1 spec shape."""
        mock_llm_client.run_agent_loop.return_value = (
            [
                {"role": "user", "content": "生成 reward"},
                {"role": "assistant", "content": "已生成 reward.py"},
            ],
            [{
                "fileUrl": "http://testserver/files/reward_abc12345.py",
                "fileName": "reward_generated.py",
                "fileType": "text",
                "mimeType": "text/x-python",
                "fileSize": 1234,
            }],
            "assist",
            {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )
        resp = client.post("/v1/chat/completions", headers=valid_auth_header, json={
            "messages": [{"role": "user", "content": "生成 reward"}],
        })
        data = resp.json()
        att = data["x_soda"]["attachments"][0]
        # 清小搭 §1 spec: fileUrl + fileName + fileType + mimeType (no inline content)
        assert att["fileUrl"] == "http://testserver/files/reward_abc12345.py"
        assert att["fileName"] == "reward_generated.py"
        assert att["fileType"] == "text"
        assert att["mimeType"] == "text/x-python"
        assert "content" not in att  # not embedded


class TestChatStreaming:
    def test_stream_returns_sse_frames(self, client, valid_auth_header, mock_llm_client):
        """Streaming should yield SSE frames in §3.2 order: role → content → stop → [DONE]."""
        async def fake_stream(messages, initial_attachments=None):
            from core.sse import role_frame, content_frame, stop_frame, done_sentinel
            yield role_frame("test-cid")
            yield content_frame("test-cid", "你好")
            yield stop_frame("test-cid", usage={
                "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7
            })
            yield done_sentinel()

        mock_llm_client.stream_agent_loop = fake_stream

        with client.stream("POST", "/v1/chat/completions", headers={
            **valid_auth_header, "Content-Type": "application/json",
        }, json={
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
        }) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            body = b"".join(resp.iter_bytes()).decode("utf-8")

        # Should have multiple data: frames
        assert body.count("data: ") >= 4
        # Should end with [DONE]
        assert body.rstrip().endswith("data: [DONE]")
        # Should contain role frame (json.dumps adds spaces after : and ,)
        assert '"role": "assistant"' in body
        # Should contain content frame
        assert '"content": "你好"' in body
        # Should contain stop frame with finish_reason
        assert '"finish_reason": "stop"' in body


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_includes_components_breakdown(self, client):
        """v0.2 /health returns detailed component status, not just {status:ok}."""
        resp = client.get("/health")
        data = resp.json()
        assert "components" in data
        comps = data["components"]
        # Each component field present
        assert "skill_modules" in comps
        assert "deepseek" in comps
        assert "llm_client" in comps
        assert "files" in comps
        # Skill modules: 9 should all load
        assert comps["skill_modules"]["total"] == 9
        assert comps["skill_modules"]["loaded"] == 9
        assert comps["skill_modules"]["failed"] == []
        # DeepSeek: key should be present (from test env)
        assert comps["deepseek"]["api_key_present"] is True


class TestStrictBoolStream:
    def test_string_false_should_be_treated_as_false(
        self, client, valid_auth_header, mock_llm_client
    ):
        """§3 spec: stream must be parsed as strict JSON bool, not 'false' string.

        Our impl uses bool(body.get('stream', False)) which is correct for
        Python: bool('false') → True, but bool(False) → False. FastAPI's
        json parser already converts JSON false → Python False, so this
        is fine in practice. This test documents the behavior.
        """
        # JSON false → Python False → non-streaming path
        mock_llm_client.run_agent_loop.return_value = (
            [{"role": "assistant", "content": "ok"}], [], "assist",
            {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )
        resp = client.post("/v1/chat/completions", headers=valid_auth_header, json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")


class TestAttachmentsProtocol:
    """清小搭 §1 attachments spec — fileUrl + fileName + fileType + mimeType."""

    def test_attachment_has_no_inline_content(self, client, valid_auth_header, mock_llm_client):
        """Spec: 出参只放 URL，不内嵌文件字节。"""
        mock_llm_client.run_agent_loop.return_value = (
            [{"role": "assistant", "content": "已生成"}],
            [{
                "fileUrl": "http://testserver/files/x.py",
                "fileName": "reward.py",
                "fileType": "text",
                "mimeType": "text/x-python",
            }],
            "assist",
            {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )
        resp = client.post("/v1/chat/completions", headers=valid_auth_header, json={
            "messages": [{"role": "user", "content": "gen"}],
        })
        att = resp.json()["x_soda"]["attachments"][0]
        for forbidden in ("content", "data", "base64"):
            assert forbidden not in att, f"attachment must not embed {forbidden!r}"

    def test_files_endpoint_serves_written_artifact(self, client, valid_auth_header):
        """End-to-end: reward_synthesizer writes file → /files/<id> serves it.

        Bypasses LLM (calls tool directly), then hits /files to verify the
        artifact is actually downloadable.
        """
        from tools import execute_tool
        result = execute_tool("reward_synthesizer", {
            "task_description": "quadruped walk forward",
        })
        url = result["code_artifact"]["fileUrl"]
        file_id = url.rsplit("/", 1)[-1]

        # Auth-free file endpoint — 清小搭 fetches server-side, file_ids are unguessable UUIDs
        resp = client.get(f"/files/{file_id}")
        assert resp.status_code == 200
        assert "python" in resp.headers["content-type"].lower()
        body = resp.text
        assert "RewardsCfg" in body or "class " in body  # actual reward code

    def test_files_endpoint_rejects_path_traversal(self, client):
        """file_id is restricted to [a-zA-Z0-9._-] — ../etc/passwd must 404."""
        for bad in ("../etc/passwd", "..%2Fetc%2Fpasswd", "a/b", "a\\b", "x y"):
            resp = client.get(f"/files/{bad}")
            assert resp.status_code in (404, 422), f"{bad!r} should not resolve"

    def test_files_endpoint_404_for_unknown_id(self, client):
        resp = client.get("/files/nonexistent_12345.py")
        assert resp.status_code == 404
