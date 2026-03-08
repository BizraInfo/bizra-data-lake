"""
SAT Gate Helpers — shared utilities for gate functions.

Subprocess runners, path checkers, and output parsers.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


def _project_root() -> Path:
    """Return the BIZRA project root directory."""
    # Walk up from this file to find pyproject.toml
    p = Path(__file__).resolve().parent.parent.parent
    if (p / "pyproject.toml").exists():
        return p
    # Fallback to env var or cwd
    env = os.getenv("BIZRA_DATA_LAKE_ROOT")
    if env:
        return Path(env)
    return Path.cwd()


def _run(
    cmd: list[str],
    timeout: int = 120,
    cwd: Optional[Path] = None,
) -> Tuple[int, str]:
    """Run a subprocess and return (exit_code, combined_output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or _project_root(),
            env={**os.environ, "PYTHONPATH": str(_project_root())},
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        return result.returncode, output
    except FileNotFoundError:
        return 127, f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"Timeout after {timeout}s: {' '.join(cmd)}"
    except Exception as e:
        return 1, f"Error running {cmd[0]}: {e}"


def _has_tool(name: str) -> bool:
    """Check if a command-line tool is available."""
    return shutil.which(name) is not None


def path_exists(relative_path: str) -> bool:
    """Check if a path exists relative to project root."""
    return (_project_root() / relative_path).exists()


def last_line(output: str) -> str:
    """Extract the last non-empty line of output."""
    lines = [l for l in output.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def parse_test_count(output: str) -> int:
    """Parse pytest output for test count (e.g., '42 passed')."""
    import re

    match = re.search(r"(\d+)\s+passed", output)
    if match:
        return int(match.group(1))
    return 0


def prompt_human(question: str) -> bool:
    """Prompt for human attestation. Returns False in non-interactive mode."""
    try:
        if not sys.stdin.isatty():
            return False
        response = input(f"\n  [HUMAN ATTESTATION] {question} (y/n): ").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False
