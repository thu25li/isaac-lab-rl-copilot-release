"""Configuration loader for the agent server."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from agent_server/ directory (where this file's parent is)
_SERVER_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_SERVER_DIR / ".env")


def _required(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        raise RuntimeError(
            f"Missing required env var {key!r}. "
            f"Copy .env.example to .env and fill it in."
        )
    return val


DEEPSEEK_API_KEY: str = _required("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL: str = os.environ.get(
    "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
)
# DeepSeek uses alias naming — deepseek-chat auto-points to latest V3/V4-flash,
# deepseek-reasoner auto-points to latest R1. No version pinning needed.
# Verified 2026-08-02: deepseek-chat resolves to deepseek-v4-flash.
# If you want a specific snapshot, set it here.
# Available aliases as of 2026-08:
#   - deepseek-chat     (currently V4-flash, cheap, recommended for dev)
#   - deepseek-reasoner (R1, ~10x price, smarter reasoning)

AGENT_API_KEY: str = _required("AGENT_API_KEY")

HOST: str = os.environ.get("HOST", "0.0.0.0")
PORT: int = int(os.environ.get("PORT", "8765"))

# Public base URL for building file attachment URLs (清小搭 §1 spec).
# Local dev: http://localhost:8765  |  Public: https://<ngrok-or-cloudflare>.io
PUBLIC_BASE_URL: str = os.environ.get(
    "PUBLIC_BASE_URL", f"http://localhost:{PORT}"
).rstrip("/")

# Directory where generated file artifacts (reward.py, etc.) are written,
# served by GET /files/{file_id}. Created on import.
# Override via env var for cloud deployments (HF Spaces, Docker) where the
# writable path is different from the source tree.
_files_dir_env = os.environ.get("FILES_DIR")
if _files_dir_env:
    FILES_DIR: Path = Path(_files_dir_env)
else:
    FILES_DIR: Path = _SERVER_DIR / "logs" / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)

# Resolve skill root (parent of agent_server/) — absolute path
_skill_root_env = os.environ.get("SKILL_ROOT", "..")
SKILL_ROOT: Path = (_SERVER_DIR / _skill_root_env).resolve()
if not SKILL_ROOT.exists():
    raise RuntimeError(f"SKILL_ROOT does not exist: {SKILL_ROOT}")
# Sanity check — make sure it's actually the skill root
if not (SKILL_ROOT / "scripts" / "reward_synthesizer.py").exists():
    raise RuntimeError(
        f"SKILL_ROOT={SKILL_ROOT} does not look like the skill root "
        f"(scripts/reward_synthesizer.py not found)"
    )

# Add skill root to sys.path so we can import scripts.* from agent_server
import sys
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))


def check_auth(authorization: str | None) -> None:
    """Validate Bearer token. Raises HTTPException-style 401 on failure."""
    from fastapi import HTTPException
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    token = authorization[len("Bearer "):].strip()
    if token != AGENT_API_KEY:
        raise HTTPException(status_code=401, detail="invalid credential")
