"""File input handling — 清小搭 §3 protocol.

When the user uploads a document in the 清小搭 chat UI, the platform sends a
content part like:

    {"type": "file", "file": {"url": "https://...", "filename": "metrics.json"}}

DeepSeek (our text-only LLM) cannot consume such parts, so we:

1. EXTRACT file parts from user messages (and strip them, keeping text parts)
2. DOWNLOAD each file over http
3. ROUTE by filename/content to the right deterministic tool:
     - *.json / *metrics*  → log_analyzer → diagnosis_engine (full pipeline)
     - env*.py (EnvCfg)    → config_validator
     - reward*.py (RewardsCfg) → reward_validator
     - anything else       → graceful "not supported" note
4. Return (context_message, attachments) — the caller injects the context into
   the conversation and merges attachments into x_soda.

All routing reuses the existing tool layer, so visualizations and 清小搭 §1
attachments (anomalies.png, diagnosis.png, ...) are generated automatically.

Security: http/https only, 10 MB cap, 30s timeout, content is parsed (JSON/AST)
never executed.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

log = logging.getLogger("agent_server.file_input")

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
DOWNLOAD_TIMEOUT = 30.0


def extract_file_parts(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Strip file parts from user messages; keep text.

    Returns:
        (cleaned_messages, files)
        cleaned_messages: content normalized to plain text (DeepSeek-safe)
        files: [{"url": ..., "filename": ...}, ...]
    """
    cleaned: list[dict[str, Any]] = []
    files: list[dict[str, str]] = []

    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            cleaned.append(m)
            continue

        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "file":
                f = part.get("file") or {}
                url = f.get("url")
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    files.append({
                        "url": url,
                        "filename": str(f.get("filename") or "uploaded_file"),
                    })
                else:
                    log.warning("file part without usable url, skipped: %r", f)
            elif part.get("type") == "text" and part.get("text"):
                text_parts.append(str(part["text"]))
            # image/audio parts: 清小搭 strips them for us when unlisted;
            # if one slips through, drop it silently (text-only LLM).

        m2 = dict(m)
        m2["content"] = "\n".join(text_parts) if text_parts else "（用户上传了文件）"
        cleaned.append(m2)

    return cleaned, files


async def download_file(url: str) -> bytes:
    """Download the file with size + timeout guards. Raises on failure."""
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        if len(resp.content) > MAX_FILE_BYTES:
            raise ValueError(
                f"file too large ({len(resp.content)} bytes > {MAX_FILE_BYTES})"
            )
        return resp.content


def _looks_like_metrics(data: Any) -> bool:
    """True if the parsed JSON looks like {tag: [{step, value}, ...]}."""
    if not isinstance(data, dict) or not data:
        return False
    for v in data.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) \
                and "step" in v[0] and "value" in v[0]:
            return True
    return False


async def process_uploaded_file(
    url: str, filename: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Download + route one uploaded file. Never raises.

    Returns:
        (result, attachments)
        result: tool output dict (or {"supported": False, ...} on no route)
        attachments: 清小搭 §1 attachment dicts from the tool layer
    """
    try:
        content = await download_file(url)
    except Exception as e:
        log.warning("file download failed (%s): %s", filename, e)
        return {
            "supported": False,
            "error": f"文件下载失败：{type(e).__name__}",
            "filename": filename,
        }, []

    fname = filename.lower()

    # ---------- Route 1: metrics JSON → log_analyzer → diagnosis_engine ----------
    if fname.endswith(".json") or "metrics" in fname or "tensorboard" in fname:
        try:
            data = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = None
        if _looks_like_metrics(data):
            from tools import execute_tool
            la = execute_tool("log_analyzer", {
                "metrics_inline": json.dumps(data, ensure_ascii=False),
            })
            atts = _pull_attachments(la)
            out: dict[str, Any] = {"kind": "metrics_analysis", "log_analysis": _trim(la)}
            if la.get("symptoms"):
                dg = execute_tool("diagnosis_engine", {"symptoms": la["symptoms"]})
                atts.extend(_pull_attachments(dg))
                out["diagnosis"] = _trim(dg)
            return out, atts
        # JSON but not metrics-shaped → let LLM see a preview
        return {
            "kind": "json_preview",
            "filename": filename,
            "preview": content.decode("utf-8", errors="replace")[:1500],
            "note": "JSON 已收到，但不是标准 metrics 格式（需要 tag -> [{step, value}]）",
        }, []

    # ---------- Route 2: python source → config / reward validator ----------
    if fname.endswith(".py"):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return {"supported": False, "filename": filename,
                    "error": "非 UTF-8 的 Python 文件"}, []
        from tools import execute_tool
        # Route by CLASS DEFINITIONS (imports are unreliable: env.py also
        # references RewardsCfg when wiring the rewards field).
        #   env file   → defines "class XxxEnvCfg"
        #   reward file → defines "class RewardsCfg"
        if re.search(r"class\s+\w*EnvCfg", text):
            # config_validator takes a file path — persist to a temp file
            import tempfile, re as _re
            safe_name = _re.sub(r"[^a-zA-Z0-9_.-]", "_", filename) or "upload.py"
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", prefix="uploaded_", delete=False, encoding="utf-8",
            )
            tmp.write(text)
            tmp.close()
            rep = execute_tool("config_validator", {"file_path": tmp.name})
            import os as _os
            _os.unlink(tmp.name)
            return {"kind": "config_validation", "filename": filename,
                    "report": _trim(rep)}, _pull_attachments(rep)
        if re.search(r"class\s+RewardsCfg", text):
            rep = execute_tool("reward_validator", {"code": text})
            return {"kind": "reward_validation", "filename": filename,
                    "report": _trim(rep)}, _pull_attachments(rep)
        # Unknown python file — show a short preview so the LLM can react
        return {
            "kind": "python_preview",
            "filename": filename,
            "preview": text[:1500],
            "note": "Python 文件已收到，但未识别出 EnvCfg 或 RewardsCfg 结构",
        }, []

    # ---------- No route ----------
    return {
        "supported": False,
        "filename": filename,
        "size_bytes": len(content),
        "note": "暂不支持自动解析此文件类型（支持：metrics.json / env.py / reward.py）",
    }, []


def _pull_attachments(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Pop 清小搭 §1 attachment dicts (fileUrl+fileName) out of a tool result."""
    atts: list[dict[str, Any]] = []
    if isinstance(result, dict):
        for k, v in list(result.items()):
            if isinstance(v, dict) and "fileUrl" in v and "fileName" in v:
                atts.append(v)
                result.pop(k, None)
    return atts


def _trim(result: dict[str, Any], max_chars: int = 4000) -> dict[str, Any]:
    """Keep the context message within a sane size for the LLM."""
    s = json.dumps(result, ensure_ascii=False, default=str)
    if len(s) <= max_chars:
        return result
    return {
        "_truncated": True,
        "preview": s[:max_chars],
        "hint": "完整结果已作为附件输出，见文件卡片",
    }
