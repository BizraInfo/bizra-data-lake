"""
BIZRA CLI Shared Utilities
============================

Constants, color helpers, and common I/O used across all command modules.
Extracted from the monolithic bizra_cli.py to eliminate duplication.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# ─── Constants ───────────────────────────────────────────────────
VERSION = "3.0.0-GENESIS"
CODENAME = "بذرة"
APP_NAME = "bizra"
DEMA_GREETING = "مرحبا! أنا ديما، مساعدتك الشخصية."

# Directories
HOME = Path.home()
BIZRA_HOME = Path(os.environ.get("BIZRA_HOME", HOME / ".bizra"))
BIZRA_STATE = BIZRA_HOME / "sovereign_state"
BIZRA_LOGS = BIZRA_HOME / "logs"
BIZRA_MODELS = BIZRA_HOME / "models"
BIZRA_PID = BIZRA_HOME / "bizra.pid"
BIZRA_IDENTITY = BIZRA_STATE / "identity.json"

# Ports
API_PORT = int(os.environ.get("BIZRA_API_PORT", "8010"))
WEB_PORT = int(os.environ.get("BIZRA_WEB_PORT", "3000"))
OLLAMA_PORT = 11434


# ─── Color helpers ───────────────────────────────────────────────
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[38;5;78m"
    TEAL = "\033[38;5;43m"
    GOLD = "\033[38;5;179m"
    RED = "\033[38;5;167m"
    BLUE = "\033[38;5;75m"
    PURPLE = "\033[38;5;141m"
    GRAY = "\033[38;5;245m"
    WHITE = "\033[38;5;255m"


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if platform.system() == "Windows":
        return os.environ.get("TERM") == "xterm" or "WT_SESSION" in os.environ
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# Disable colors if terminal doesn't support them
if not _supports_color():
    for attr in dir(C):
        if not attr.startswith("_"):
            setattr(C, attr, "")


# ─── Printing ────────────────────────────────────────────────────
def print_status(label: str, status: str, ok: bool = True) -> None:
    icon = f"{C.GREEN}✓{C.RESET}" if ok else f"{C.RED}✗{C.RESET}"
    print(f"  {icon} {C.WHITE}{label}{C.RESET}: {status}")


def print_warn(msg: str) -> None:
    print(f"  {C.GOLD}⚠ {msg}{C.RESET}")


def print_error(msg: str) -> None:
    print(f"  {C.RED}✗ {msg}{C.RESET}")


def print_info(msg: str) -> None:
    print(f"  {C.TEAL}• {msg}{C.RESET}")


# ─── Infrastructure ──────────────────────────────────────────────
def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def ensure_dirs() -> None:
    for d in [BIZRA_HOME, BIZRA_STATE, BIZRA_LOGS, BIZRA_MODELS]:
        d.mkdir(parents=True, exist_ok=True)


def find_python() -> str:
    for candidate in ["python3", "python"]:
        p = shutil.which(candidate)
        if p:
            return p
    return sys.executable


def find_bizra_root() -> Optional[Path]:
    env = os.environ.get("BIZRA_ROOT")
    if env:
        p = Path(env)
        if (p / "core" / "sovereign").exists():
            return p

    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parent.parent.parent,
        HOME / "BIZRA-DATA-LAKE",
        Path("/mnt/c/BIZRA-DATA-LAKE"),
    ]
    for c in candidates:
        if (c / "core" / "sovereign").exists():
            return c
    return None


def find_frontend_root() -> Optional[Path]:
    env = os.environ.get("BIZRA_FRONTEND")
    if env:
        p = Path(env)
        if (p / "package.json").exists():
            return p

    root = find_bizra_root()
    if root:
        for name in ["frontend", "award-winner-design"]:
            fp = root / name
            if (fp / "package.json").exists():
                return fp
    return None


def api_health() -> Optional[Dict[str, Any]]:
    """Check the sovereign API health endpoint."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:{API_PORT}/health",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def save_pid(pid: int) -> None:
    ensure_dirs()
    pids = load_pids()
    pids.append(pid)
    BIZRA_PID.write_text(json.dumps(pids))


def load_pids() -> list:
    if not BIZRA_PID.exists():
        return []
    try:
        return json.loads(BIZRA_PID.read_text())
    except (json.JSONDecodeError, ValueError):
        return []


def clear_pids() -> None:
    if BIZRA_PID.exists():
        BIZRA_PID.unlink()
