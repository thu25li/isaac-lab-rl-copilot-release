"""DeepSeek LLM client with function-calling loop.

DeepSeek exposes an OpenAI-compatible /chat/completions API. We use the
`tools` parameter to let the model pick one of our 7 tools. The loop:

    user message
        ↓
    LLM with tools → returns either content (final answer) or tool_call
        ↓ if tool_call
    execute the script (deterministic)
        ↓
    feed tool result back as role="tool"
        ↓
    LLM produces final answer

The loop terminates when LLM returns content without tool_calls, or
after MAX_ROUNDS to prevent infinite loops.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import httpx


# Alias to make intent clear in retry logic
_asyncio_sleep = asyncio.sleep

from core.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from tools import all_tool_defs, execute_tool

log = logging.getLogger("agent_server.llm")

MAX_ROUNDS = 5  # max tool-call rounds per request

# Default mode when conversation history has no mode signal yet. Per the
# product design, 清小搭 platform shows a mode-selection card on agent load,
# so the user's first message is always an explicit mode choice — but if for
# some reason it isn't, we default to assist (concise, solution-first).
DEFAULT_MODE = "assist"

# Keywords that signal a mode switch in user messages. Match case-insensitively
# as a substring. "教学" matches both "教学模式" and "我用教学模式".
TEACHING_KEYWORDS = ("教学模式", "teaching mode", "教学")
ASSIST_KEYWORDS = ("辅助模式", "assist mode", "辅助")


def detect_mode(user_messages: list[dict[str, Any]]) -> str:
    """Scan user_messages (newest first) and return the most recently selected mode.

    Looks for the latest user message containing a mode keyword. Falls back to
    DEFAULT_MODE if no signal is found.
    """
    for msg in reversed(user_messages):
        if msg.get("role") != "user":
            continue
        text = (msg.get("content") or "").lower() if isinstance(msg.get("content"), str) else ""
        if any(k in text for k in ASSIST_KEYWORDS):
            return "assist"
        if any(k in text for k in TEACHING_KEYWORDS):
            return "teaching"
    return DEFAULT_MODE


def _build_mode_icon_attachment(mode: str) -> dict[str, Any] | None:
    """Generate a mode status icon and return a 清小搭 §1 attachment dict, or
    None if generation fails.
    """
    try:
        from scripts.utils.mode_card import make_mode_icon
        from core.artifacts import make_attachment
        from core.config import FILES_DIR
        import uuid
        png_path = make_mode_icon(mode, FILES_DIR / f"mode_{mode}_{uuid.uuid4().hex[:6]}.png")
        with open(png_path, "rb") as f:
            return make_attachment(
                content=f.read(),
                file_name=f"mode_{mode}.png",
                file_type="image",
                mime_type="image/png",
            )
    except Exception as e:
        log.warning("mode icon generation failed: %s: %s", type(e).__name__, e)
        return None


def _build_teaching_extras(tool_name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Teaching-mode-only extra visualizations to deepen the learning experience.

    Different from the tool's base chart (which both modes get), these are
    additional pedagogical visualizations that help the user understand WHY
    the recommendation is what it is.

    Currently:
    - dr_advisor → sim2real_radar.png (DR coverage radar chart)

    Returns: list of 清小搭 §1 attachment dicts. Empty list if generation fails.
    """
    extras: list[dict[str, Any]] = []
    try:
        from scripts.utils.plotting import plot_sim2real_radar
        from core.artifacts import make_attachment
        from core.config import FILES_DIR
        import uuid

        if tool_name == "dr_advisor":
            radar_path = plot_sim2real_radar(
                essential=result.get("essential_terms", []),
                recommended=result.get("recommended_terms", []),
                optional=result.get("optional_terms", []),
                parameter_ranges=result.get("parameter_ranges", {}),
                term_details=result.get("term_details", {}),
                output_path=FILES_DIR / f"sim2real_radar_{uuid.uuid4().hex[:8]}.png",
                robot_mass_kg=result.get("robot_mass_kg", 15.0),
            )
            with open(radar_path, "rb") as f:
                extras.append(make_attachment(
                    content=f.read(),
                    file_name="sim2real_radar.png",
                    file_type="image",
                    mime_type="image/png",
                ))
                log.info("teaching-mode extra: sim2real_radar added")
    except Exception as e:
        log.warning("teaching extras for %s failed: %s: %s",
                    tool_name, type(e).__name__, e)
    return extras


class LLMClient:
    """Async wrapper around DeepSeek's OpenAI-compatible /chat/completions endpoint."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=DEEPSEEK_BASE_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def chat_round(
        self, messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Single round: send messages + tools, return assistant message + usage.

        Returns:
            (assistant_msg, usage_dict)
            assistant_msg has keys: 'content' (str|None), 'tool_calls' (list|None)
            usage_dict has keys: prompt_tokens, completion_tokens, total_tokens
                (DeepSeek returns real token counts; we accumulate across rounds)
        """
        # DeepSeek auto-caches the prompt prefix — same SYSTEM_PROMPT + tool defs
        # across requests will hit the cache (~10x cheaper, ~50% faster on prefix).
        # Reference: https://platform.deepseek.com (cached tokens reported separately
        # in usage.prompt_cache_hit_tokens when applicable).
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "tools": all_tool_defs(),
            "tool_choice": "auto",
        }

        # Exponential backoff retry on transient errors (429 / 5xx / network)
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                # Surface the provider's error body — it states the exact
                # cause (e.g. "messages[0].content is empty"). Without this,
                # debugging 400s is guesswork.
                log.error("DeepSeek %d error body: %s",
                          resp.status_code, resp.text[:500])
                if resp.status_code == 429 and attempt < max_retries:
                    wait = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
                    log.warning("DeepSeek 429 (attempt %d), retrying in %.1fs",
                                attempt + 1, wait)
                    await _asyncio_sleep(wait)
                    continue
                # For 5xx, retry once on the next attempt; otherwise propagate
                if 500 <= resp.status_code < 600 and attempt < max_retries:
                    wait = 0.5 * (2 ** attempt)
                    log.warning("DeepSeek %d (attempt %d), retrying in %.1fs",
                                resp.status_code, attempt + 1, wait)
                    await _asyncio_sleep(wait)
                    continue
                raise
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                if attempt < max_retries:
                    wait = 0.5 * (2 ** attempt)
                    log.warning("DeepSeek network error %s (attempt %d), retrying in %.1fs",
                                type(e).__name__, attempt + 1, wait)
                    await _asyncio_sleep(wait)
                    continue
                raise

        data = resp.json()
        usage = data.get("usage") or {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }
        # Surface cache hit info if DeepSeek returned it (helps with cost monitoring)
        cache_hit = usage.get("prompt_cache_hit_tokens", 0)
        if cache_hit:
            log.info("DeepSeek cache hit: %d tokens", cache_hit)
        return data["choices"][0]["message"], usage

    async def run_agent_loop(
        self, user_messages: list[dict[str, Any]],
        progress_cb: Any = None,
        initial_attachments: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, int]]:
        """Run the full tool-calling loop.

        Args:
            user_messages: conversation history (newest last).
            progress_cb: optional async callable(str_event_type, str_message) —
                when provided, gets real-time progress updates (used by streaming).
            initial_attachments: pre-seeded attachments (e.g. from file-input
                preprocessing) merged into the final x_soda payload.
                Event types: "reasoning" (status text), "tool" (tool name).

        Returns:
            (final_messages, attachments, mode, total_usage)
            total_usage: accumulated across all rounds {prompt_tokens, completion_tokens, total_tokens}
        """
        from core.prompts import get_prompt

        async def _emit(event: str, msg: str):
            if progress_cb is not None:
                try:
                    await progress_cb(event, msg)
                except Exception:
                    pass  # never let progress emission break the loop

        mode = detect_mode(user_messages)
        system_prompt = get_prompt(mode)
        log.info("detected mode: %s", mode)
        await _emit("reasoning", f"已进入{ '教学模式' if mode == 'teaching' else '辅助模式'}")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *user_messages,
        ]
        attachments: list[dict[str, Any]] = list(initial_attachments or [])
        # Accumulated token usage across all LLM rounds (real DeepSeek values)
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for round_idx in range(MAX_ROUNDS):
            log.info("LLM round %d: %d messages", round_idx + 1, len(messages))
            await _emit("reasoning", f"分析中（第 {round_idx + 1} 轮）…")
            assistant_msg, round_usage = await self.chat_round(messages)
            # DeepSeek returns real usage per round — accumulate it
            for k in total_usage:
                total_usage[k] += int(round_usage.get(k, 0))

            # If model wants to call tools, push the assistant message and execute
            tool_calls = assistant_msg.get("tool_calls")
            if tool_calls:
                # Strip content=None to keep OpenAI happy
                clean = {k: v for k, v in assistant_msg.items() if v is not None}
                messages.append(clean)

                # Parse + emit reasoning for all tool_calls first (fast, lets user
                # see all upcoming calls immediately)
                parsed_calls = []
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    raw_args = tc["function"].get("arguments")
                    if raw_args is None:
                        raw_args = tc["function"].get("args", "")
                    try:
                        args = json.loads(raw_args or "{}")
                    except json.JSONDecodeError as e:
                        args = {}
                        log.warning("malformed tool args: %s (%s)", raw_args, e)
                    parsed_calls.append((tc, tool_name, args))
                    await _emit("tool", tool_name)
                    await _emit(
                        "reasoning",
                        f"调用 {tool_name}（参数：{json.dumps(args, ensure_ascii=False)[:80]}）"
                    )

                # Execute tool_calls concurrently.
                # Most tools are sync + CPU-bound (deterministic scripts), so a
                # single threadpool can run them in parallel without blocking the
                # asyncio loop. For 1 tool this is just overhead — fine.
                import asyncio as _asyncio
                import functools

                async def _exec_one(name: str, args: dict) -> tuple[str, dict]:
                    def _run():
                        try:
                            return name, execute_tool(name, args)
                        except Exception as e:
                            log.exception("tool execution failed: %s", name)
                            return name, {"error": f"{type(e).__name__}: {e}"}
                    # Run sync tool in a thread — true parallelism for CPU work
                    return await _asyncio.get_event_loop().run_in_executor(
                        None, functools.partial(_run),
                    )

                results = await _asyncio.gather(*[
                    _exec_one(name, args) for (_, name, args) in parsed_calls
                ])
                results_by_name = {(name, i): r for i, ((name, r)) in enumerate(results)}
                # Re-map back to tool_call_id order (gather preserves input order)
                for (tc, tool_name, args), (_, result) in zip(parsed_calls, results):
                    log.info("tool result: %s", tool_name)
                    # Collect 清小搭 §1 spec attachments
                    if isinstance(result, dict):
                        for k, v in list(result.items()):
                            if (
                                isinstance(v, dict)
                                and "fileUrl" in v
                                and "fileName" in v
                            ):
                                attachments.append(v)
                                log.info(
                                    "collected attachment from %s.%s: %s (total=%d)",
                                    tool_name, k, v.get("fileName"), len(attachments),
                                )
                                result.pop(k, None)
                    # Teaching-mode enhancement: extra visualizations to deepen learning
                    if mode == "teaching" and isinstance(result, dict):
                        extra = _build_teaching_extras(tool_name, result)
                        if extra:
                            attachments.extend(extra)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })
                continue

            # No tool_calls → final answer
            content = assistant_msg.get("content") or ""
            messages.append({"role": "assistant", "content": content})
            return messages, attachments, mode, total_usage

        # Hit MAX_ROUNDS — return whatever we have with a notice
        log.warning("hit MAX_ROUNDS=%d, returning last state", MAX_ROUNDS)
        messages.append({
            "role": "assistant",
            "content": "（已达到最大工具调用轮数，无法继续。请简化您的问题后重试。）",
        })
        return messages, attachments, mode

    async def stream_agent_loop(
        self, user_messages: list[dict[str, Any]],
        initial_attachments: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the agent loop as SSE-formatted strings.

        Real streaming: emits reasoning frames DURING tool execution (not after),
        then streams the final answer character-by-character. The user sees
        "调用 log_analyzer..." within ~1s of sending the message, even though
        the LLM round itself takes 3-5s.

        Yields (清小搭 §3.2 frame order):
            role_frame → reasoning_frame(s) → content_frame(s) → stop_frame → [DONE]
        """
        from core.sse import (
            role_frame, content_frame, reasoning_frame, stop_frame, done_sentinel,
        )
        import uuid

        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        yield role_frame(cid)

        # Bridge: run_agent_loop calls progress_cb for each step. We use an
        # asyncio.Queue so the loop task can push progress while we yield SSE.
        import asyncio
        queue: asyncio.Queue = asyncio.Queue()

        async def progress_cb(event: str, msg: str):
            await queue.put((event, msg))

        # Start the agent loop as a background task
        loop_task = asyncio.create_task(
            self.run_agent_loop(user_messages, progress_cb, initial_attachments)
        )

        # Stream reasoning frames as they arrive
        try:
            while not loop_task.done():
                # Drain queue with a short timeout so we don't block on done loop
                try:
                    event, msg = await asyncio.wait_for(queue.get(), timeout=0.05)
                    if event == "reasoning":
                        yield reasoning_frame(cid, msg)
                except asyncio.TimeoutError:
                    continue
            # Drain any remaining items
            while not queue.empty():
                event, msg = queue.get_nowait()
                if event == "reasoning":
                    yield reasoning_frame(cid, msg)
        except Exception as e:
            log.exception("streaming progress failed")
            yield content_frame(cid, f"（流式进度出错：{type(e).__name__}: {e}）")

        # Get the final result
        try:
            messages, attachments, mode, total_usage = await loop_task
        except Exception as e:
            log.exception("agent loop failed")
            yield content_frame(cid, f"（出错了：{type(e).__name__}: {e}）")
            yield stop_frame(cid)
            yield done_sentinel()
            return

        # Dedupe by fileName (preprocessing + LLM tool-call may both produce
        # e.g. diagnosis.png) before emitting the stop frame
        from core.artifacts import dedupe_attachments
        attachments = dedupe_attachments(attachments)

        # The last assistant message is the final answer, with the mode tag
        # prepended in code (deterministic, same as the non-streaming path)
        final = ""
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("content"):
                final = m["content"]
                break
        mode_label = "[教学模式]" if mode == "teaching" else "[辅助模式]"
        final = f"{mode_label}\n{final}"

        # Stream final answer char-by-char (smooth UX)
        chunk_size = 6
        for i in range(0, len(final), chunk_size):
            yield content_frame(cid, final[i:i + chunk_size])

        usage = total_usage  # real DeepSeek token counts, accumulated across rounds
        yield stop_frame(
            cid,
            usage=usage,
            attachments=attachments if attachments else None,
        )
        yield done_sentinel()
