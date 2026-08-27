"""File artifact helper — writes generated files to disk and builds 清小搭 §1 spec attachments.

清小搭 attachments protocol requires fileUrl + fileName + fileType + mimeType.
We write files to disk under logs/files/, serve them via GET /files/{file_id},
and return the URL to the platform. 清小搭 then re-stores the file to its own
OSS, so we don't need to keep files long-term.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from core.config import FILES_DIR, PUBLIC_BASE_URL

# fileType 枚举（清小搭 §1.4）
_FILE_TYPE_BY_EXT = {
    ".py": "text", ".txt": "text", ".md": "text", ".rst": "text",
    ".pdf": "pdf",
    ".doc": "word", ".docx": "word",
    ".xls": "excel", ".xlsx": "excel",
    ".ppt": "ppt", ".pptx": "ppt",
    ".zip": "archive", ".rar": "archive", ".7z": "archive",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".webp": "image",
    ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".webm": "audio",
    ".mp4": "video", ".mov": "video",
}

# 扩展名 → MIME（覆盖常用类型，其他走 application/octet-stream）
_MIME_BY_EXT = {
    ".py": "text/x-python", ".txt": "text/plain", ".md": "text/markdown",
    ".rst": "text/x-rst",
    ".pdf": "application/pdf",
    ".doc": "application/msword", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel", ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint", ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".zip": "application/zip", ".rar": "application/vnd.rar", ".7z": "application/x-7z-compressed",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".webm": "audio/webm",
    ".mp4": "video/mp4", ".mov": "video/quicktime",
}


def make_attachment(
    *,
    content: str | bytes,
    file_name: str,
    file_type: str | None = None,
    mime_type: str | None = None,
) -> dict[str, Any]:
    """Write content to logs/files/<uuid><ext>, return 清小搭-spec attachment dict.

    Args:
        content: file bytes (str is encoded utf-8)
        file_name: display + download filename, e.g. "reward_generated.py"
        file_type: 清小搭 fileType enum (text/pdf/word/...). If None, inferred from extension.
        mime_type: HTTP Content-Type. If None, inferred from extension.

    Returns:
        {fileUrl, fileName, fileType, mimeType, fileSize} — no `content` field.
    """
    ext = Path(file_name).suffix.lower()
    # Use UUID + original extension as on-disk filename — avoids collisions,
    # preserves extension so FileResponse sets the right Content-Type.
    disk_name = f"{uuid.uuid4().hex}{ext}"
    disk_path = FILES_DIR / disk_name

    if isinstance(content, str):
        disk_path.write_text(content, encoding="utf-8")
    else:
        disk_path.write_bytes(content)

    if file_type is None:
        file_type = _FILE_TYPE_BY_EXT.get(ext, "file")
    if mime_type is None:
        mime_type = _MIME_BY_EXT.get(ext, "application/octet-stream")

    return {
        "fileUrl": f"{PUBLIC_BASE_URL}/files/{disk_name}",
        "fileName": file_name,
        "fileType": file_type,
        "mimeType": mime_type,
        "fileSize": disk_path.stat().st_size,
    }


# Path-traversal guard for /files/{file_id} route — only allow alnum + . _ -
_FILE_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


def resolve_file_id(file_id: str) -> Path | None:
    """Resolve a file_id to a safe disk path, or None if invalid/missing."""
    if not _FILE_ID_RE.match(file_id):
        return None
    path = FILES_DIR / file_id
    # Extra defense: ensure resolved path stays inside FILES_DIR
    try:
        path.resolve().relative_to(FILES_DIR.resolve())
    except ValueError:
        return None
    if not path.exists() or not path.is_file():
        return None
    return path


def dedupe_attachments(
    attachments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop duplicate attachments by fileName, keeping the LAST occurrence.

    Rationale: file preprocessing may run a tool (e.g. diagnosis_engine) whose
    chart is then re-generated if the LLM re-calls the same tool — producing
    two attachments with the same fileName. Deduping by fileName keeps the
    x_soda payload clean for the frontend.
    """
    by_name: dict[str, dict[str, Any]] = {}
    for att in attachments:
        name = att.get("fileName") or att.get("fileUrl", "")
        by_name[name] = att
    return list(by_name.values())
