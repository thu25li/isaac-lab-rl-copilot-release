"""Package the skill into a flat-structure ZIP.

Competition requirement (per user direction 2026-08): ZIP must contain skill
files FLAT at the archive root — no outer wrapper folder. Extracting the ZIP
yields SKILL.md, README.md, scripts/, etc. directly in the cwd.

Pre-2026-08 the script produced a single-rooted ZIP (isaac-lab-rl-copilot/...).
That was rejected by the platform — switched to flat.

Excludes:
- __pycache__/ directories and *.pyc files (Python bytecode, regenerable)
- .pytest_cache/ (pytest metadata, regenerable)
- .git/ (version control metadata)
- agent_server/ (belongs to the agent track, NOT the skill track — also
  contains agent_server/.env with a real DeepSeek API key that would leak)
- Any stray .env / .env.local files (defensive)

Run:
    python scripts/package_skill.py
    python scripts/package_skill.py --verify  # extract to temp + run tests
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG_NAME = "isaac-lab-rl-copilot"
EXCLUDED_DIRS = {
    "__pycache__", ".pytest_cache", ".git", ".idea", ".vscode",
    "agent_server",
    "end_to_end_demo_outputs",  # demo runtime output, regenerated on rerun
}
EXCLUDED_EXTS = {".pyc", ".pyo"}
EXCLUDED_FILES = {".DS_Store", "Thumbs.db"}
EXCLUDED_NAMES = {".env", ".env.local"}


def should_skip(path: Path) -> bool:
    if path.name in EXCLUDED_DIRS or path.name in EXCLUDED_FILES or path.name in EXCLUDED_NAMES:
        return True
    if path.suffix in EXCLUDED_EXTS:
        return True
    return False


def stage_to(staging: Path) -> None:
    """Copy skill contents flat into staging/ (no outer wrapper folder).

    Layout after staging:
        staging/
        ├── SKILL.md                  ← root entry (manual load)
        ├── scripts/, resources/...   ← skill body
    """
    for src in ROOT.iterdir():
        if should_skip(src):
            continue
        if src.name == PKG_NAME + ".zip":
            continue
        dst = staging / src.name
        if src.is_dir():
            shutil.copytree(
                src, dst,
                ignore=shutil.ignore_patterns(*EXCLUDED_DIRS, "*.pyc", "*.pyo"),
            )
        else:
            shutil.copy2(src, dst)


def make_zip(staging: Path, output: Path) -> None:
    """Zip the staging/ folder flat — arcnames are relative to staging/."""
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                arcname = path.relative_to(staging)
                zf.write(path, arcname)


def verify_zip(zip_path: Path) -> bool:
    """Extract ZIP to temp, verify FLAT structure, run tests + demo."""
    print(f"\n=== Verifying {zip_path.name} ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)

        # Check FLAT structure: SKILL.md must be at the extraction root,
        # not inside an outer wrapper folder.
        if not (tmp_path / "SKILL.md").exists():
            # Check if there's a single outer folder wrapping everything
            top_level = [p for p in tmp_path.iterdir()]
            if len(top_level) == 1 and top_level[0].is_dir():
                print(f"FAIL: ZIP has unwanted outer folder '{top_level[0].name}/' "
                      f"— competition requires FLAT structure")
            else:
                print(f"FAIL: SKILL.md not found at ZIP root")
            return False
        print("OK: FLAT structure (SKILL.md at ZIP root)")

        # Required files all at root
        required = ["SKILL.md", "README.md", "requirements.txt"]
        for name in required:
            if not (tmp_path / name).exists():
                print(f"FAIL: missing {name} at ZIP root")
                return False
            print(f"OK: {name} present at root")

        # Check no excluded artifacts
        bad = []
        for p in tmp_path.rglob("*"):
            if p.name in EXCLUDED_DIRS or p.suffix in EXCLUDED_EXTS:
                bad.append(str(p.relative_to(tmp_path)))
        if bad:
            print(f"FAIL: cache artifacts in package: {bad[:5]}...")
            return False
        print("OK: no cache artifacts")

        # Check agent_server truly absent (defensive — would leak API key)
        if (tmp_path / "agent_server").exists():
            print("FAIL: agent_server/ leaked into skill ZIP — would expose API key")
            return False
        print("OK: agent_server excluded")

        # Check no .env files leaked
        env_leaks = list(tmp_path.rglob(".env")) + list(tmp_path.rglob(".env.local"))
        if env_leaks:
            print(f"FAIL: .env files leaked: {[str(p.relative_to(tmp_path)) for p in env_leaks]}")
            return False
        print("OK: no .env files")

        # Run tests in extracted package
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q"],
            cwd=tmp_path, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print("FAIL: tests failed in extracted package")
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])
            return False
        last_line = [l for l in result.stdout.splitlines() if l.strip()][-1]
        print(f"OK: tests pass in extracted package ({last_line})")

        # Run end-to-end demo
        result = subprocess.run(
            [sys.executable, "examples/end_to_end_demo.py"],
            cwd=tmp_path, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print("FAIL: end-to-end demo failed")
            print(result.stderr[-2000:])
            return False
        print("OK: end-to-end demo runs in extracted package")

    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Extract ZIP to temp and run tests + demo")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or (ROOT / f"{PKG_NAME}.zip")
    if output.exists():
        output.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        print("Staging files (flat structure)...")
        stage_to(staging)
        file_count = sum(1 for _ in staging.rglob("*") if _.is_file())
        print(f"  staged {file_count} files")

        print(f"Creating {output.name}...")
        make_zip(staging, output)
        size_kb = output.stat().st_size // 1024
        print(f"  size: {size_kb} KB")

    if args.verify:
        ok = verify_zip(output)
        if not ok:
            print("\nVERIFICATION FAILED")
            sys.exit(1)
        print("\nVERIFICATION PASSED")


if __name__ == "__main__":
    main()
