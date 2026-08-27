"""End-to-end test against real DeepSeek API.

Skipped by default. To run:
    AGENT_RUN_REAL_LLM=1 pytest tests/test_real_llm.py -s

This will spend real API tokens. Use sparingly.
"""
from __future__ import annotations

import os
import asyncio
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("AGENT_RUN_REAL_LLM"),
    reason="set AGENT_RUN_REAL_LLM=1 to run real LLM e2e tests (spends tokens)",
)


@pytest.mark.asyncio
async def test_reward_synthesis_via_llm():
    """Full loop: user asks for reward → LLM calls synthesizer → returns code."""
    from core.llm_client import LLMClient

    client = LLMClient()
    try:
        messages, attachments = await client.run_agent_loop([
            {"role": "user", "content": "帮我生成四足机器人前进的 reward"},
        ])
        # Should have at least one assistant message with content
        assistant_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("content")]
        assert len(assistant_msgs) >= 1
        final = assistant_msgs[-1]["content"]
        # Should mention reward / reward.py / weight or similar
        assert any(kw in final.lower() for kw in ["reward", "pattern", "weight", "track"])
        # Should have collected a 清小搭-spec attachment (fileUrl + fileName)
        assert len(attachments) >= 1
        att = attachments[0]
        assert "reward" in att["fileName"].lower()
        assert att["fileUrl"].startswith("http")
        assert att["fileType"] == "text"
        # Verify the file actually exists on disk (download path is covered by
        # the mock-LLM test_files_endpoint_serves_written_artifact test; here
        # we only check the artifact was written).
        from core.config import FILES_DIR
        file_id = att["fileUrl"].rsplit("/", 1)[-1]
        disk_path = FILES_DIR / file_id
        assert disk_path.exists()
        assert "RewardsCfg" in disk_path.read_text(encoding="utf-8")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_diagnosis_via_llm():
    """Full loop: user describes training failure → LLM diagnoses."""
    from core.llm_client import LLMClient
    from core.config import SKILL_ROOT

    client = LLMClient()
    try:
        log_path = SKILL_ROOT / "tests" / "test_data" / "synthetic_policy_collapse" / "metrics.json"
        # Read and pass metrics inline
        import json
        metrics_data = json.loads(log_path.read_text(encoding="utf-8"))
        # Build a user message with the metrics inline
        user_msg = (
            "我的训练崩溃了，以下是 tensorboard 指标，帮我诊断：\n\n"
            + json.dumps(metrics_data)[:2000]  # truncate for token budget
        )
        messages, attachments = await client.run_agent_loop([
            {"role": "user", "content": user_msg},
        ])
        assistant_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("content")]
        assert len(assistant_msgs) >= 1
        final = assistant_msgs[-1]["content"]
        # Should mention policy_collapse or some failure mode
        assert any(
            kw in final.lower()
            for kw in ["policy_collapse", "policy collapse", "崩溃", "collapse", "失败模式", "诊断"]
        )
    finally:
        await client.close()
