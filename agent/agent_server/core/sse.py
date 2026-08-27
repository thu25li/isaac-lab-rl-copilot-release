"""SSE frame helpers — strictly conforms to 清小搭 §3.2 spec."""
from __future__ import annotations

import json
import time
from typing import Any, Optional


def _frame(
    *,
    cid: str,
    delta: dict[str, Any],
    finish_reason: Optional[str] = None,
    usage: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> str:
    """Build one SSE data frame (without trailing \\n\\n)."""
    chunk: dict[str, Any] = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    if usage is not None:
        chunk["usage"] = usage
    if extra is not None:
        chunk.update(extra)
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def role_frame(cid: str) -> str:
    """First frame: declare assistant role."""
    return _frame(cid=cid, delta={"role": "assistant"})


def content_frame(cid: str, text: str) -> str:
    """Content increment frame."""
    return _frame(cid=cid, delta={"content": text})


def reasoning_frame(cid: str, text: str) -> str:
    """Optional L1 reasoning frame — rendered as 思考中 animation."""
    return _frame(cid=cid, delta={"reasoning": text})


def stop_frame(
    cid: str,
    *,
    usage: Optional[dict] = None,
    finish_reason: str = "stop",
    attachments: Optional[list[dict]] = None,
    error: Optional[dict] = None,
) -> str:
    """Stop frame with usage, optional attachments (x_soda) and error."""
    extra: dict[str, Any] = {}
    if attachments is not None:
        extra["x_soda"] = {"attachments": attachments}
    if error is not None:
        extra["error"] = error
    return _frame(
        cid=cid,
        delta={},
        finish_reason=finish_reason,
        usage=usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        extra=extra if extra else None,
    )


def done_sentinel() -> str:
    """Terminating sentinel — must be the last frame."""
    return "data: [DONE]\n\n"
