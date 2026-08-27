"""Pytest fixtures for agent_server tests.

These tests don't hit the real DeepSeek API by default. They mock the LLM
client to return canned responses so tests are deterministic and free.

To run a real end-to-end test against DeepSeek, set:
    AGENT_RUN_REAL_LLM=1
in the environment before running pytest. This will spend real API tokens.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Ensure agent_server/ is on sys.path so `from core` / `from tools` works
_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

# CRITICAL: Add skill root to sys.path BEFORE any tool module is imported,
# because tool modules do `from scripts.X import Y` at the top.
# Skill root is the parent of agent_server/.
_SKILL_ROOT = _SERVER_DIR.parent.resolve()
if str(_SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(_SKILL_ROOT))
# Stash for config.py to pick up
os.environ.setdefault("SKILL_ROOT", str(_SKILL_ROOT))

# Force-load test env vars (avoid touching real .env) — but only when not
# running real-LLM tests. AGENT_RUN_REAL_LLM=1 means the user wants real
# DeepSeek tokens spent, so we must NOT override the real .env values.
if not os.environ.get("AGENT_RUN_REAL_LLM"):
    os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-key-for-pytest")
    os.environ.setdefault("AGENT_API_KEY", "sk-test-agent-key-for-pytest")
    os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-chat")
    os.environ.setdefault("PORT", "8765")


@pytest.fixture
def valid_auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer sk-test-agent-key-for-pytest"}


@pytest.fixture
def invalid_auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer wrong-key"}


@pytest.fixture
def mock_llm_client():
    """A mocked LLMClient that returns canned responses without hitting DeepSeek."""
    from core.llm_client import LLMClient

    client = MagicMock(spec=LLMClient)
    client.run_agent_loop = AsyncMock()
    client.stream_agent_loop = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def client(mock_llm_client, monkeypatch):
    """Starlette TestClient backed by in-process ASGI transport (no real HTTP/SSL).

    Starlette 0.52.x TestClient uses httpx with ASGITransport — requests go
    directly through the app without leaving the process, so no SSL CA bundle
    is needed. Avoids the Windows conda env CA bundle issue.
    """
    from fastapi.testclient import TestClient
    import main

    # Enter TestClient first so lifespan runs (it creates the real LLMClient),
    # THEN monkeypatch — otherwise lifespan overwrites our mock.
    with TestClient(main.app) as c:
        monkeypatch.setattr(main, "_llm", mock_llm_client)
        yield c
