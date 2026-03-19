#!/usr/bin/env python3
"""
BIZRA — Sovereign Mission Operating System

Usage:
    bizra                  Launch the terminal (TUI + web)
    bizra start            Start the sovereign runtime (background)
    bizra stop             Stop all BIZRA services
    bizra status           Show node health and sovereignty score
    bizra mission "text"   Submit a mission directly from CLI
    bizra briefing         Show morning briefing from DEMA
    bizra wallet           Show SEED/BLOOM balance
    bizra identity         Show node identity (Ed25519 public key)
    bizra version          Show version info
    bizra doctor           Diagnose issues (check all dependencies)
    bizra reset            Reset to factory (keeps identity, clears cache)

Every human is a node. Every node is a seed.
Every seed has infinite potential.

"One mission, one proof, remembered forever."
"""

import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

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


# Colors (ANSI)
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
    """Check if terminal supports ANSI colors."""
    if os.environ.get("NO_COLOR"):
        return False
    if platform.system() == "Windows":
        return os.environ.get("TERM") == "xterm" or "WT_SESSION" in os.environ
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


if not _supports_color():
    # Disable all colors
    for attr in dir(C):
        if not attr.startswith("_"):
            setattr(C, attr, "")


# ─── Utility Functions ───────────────────────────────────────────


def _print_banner():
    """Print BIZRA startup banner."""
    print(f"""
{C.TEAL}╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   {C.WHITE}{C.BOLD}██████╗ ██╗███████╗██████╗  █████╗{C.RESET}{C.TEAL}                    ║
║   {C.WHITE}{C.BOLD}██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗{C.RESET}{C.TEAL}                   ║
║   {C.WHITE}{C.BOLD}██████╔╝██║  ███╔╝ ██████╔╝███████║{C.RESET}{C.TEAL}                   ║
║   {C.WHITE}{C.BOLD}██╔══██╗██║ ███╔╝  ██╔══██╗██╔══██║{C.RESET}{C.TEAL}                   ║
║   {C.WHITE}{C.BOLD}██████╔╝██║███████╗██║  ██║██║  ██║{C.RESET}{C.TEAL}                   ║
║   {C.WHITE}{C.BOLD}╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝{C.RESET}{C.TEAL}                   ║
║                                                          ║
║   {C.GOLD}Sovereign Mission Operating System{C.TEAL}                     ║
║   {C.GRAY}v{VERSION} · {CODENAME}{C.TEAL}                                   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝{C.RESET}
""")


def _print_status(label: str, status: str, ok: bool = True):
    """Print a status line."""
    icon = f"{C.GREEN}✓{C.RESET}" if ok else f"{C.RED}✗{C.RESET}"
    color = C.GREEN if ok else C.RED
    print(f"  {icon} {C.WHITE}{label:<30}{C.RESET} {color}{status}{C.RESET}")


def _print_warn(msg: str):
    print(f"  {C.GOLD}⚠ {msg}{C.RESET}")


def _print_error(msg: str):
    print(f"  {C.RED}✗ {msg}{C.RESET}")


def _print_info(msg: str):
    print(f"  {C.TEAL}→ {msg}{C.RESET}")


def _port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _ensure_dirs():
    """Create BIZRA directories if they don't exist."""
    for d in [BIZRA_HOME, BIZRA_STATE, BIZRA_LOGS, BIZRA_MODELS]:
        d.mkdir(parents=True, exist_ok=True)


def _find_python() -> str:
    """Find the best Python executable."""
    for py in ["python3.12", "python3.11", "python3.10", "python3", "python"]:
        path = shutil.which(py)
        if path:
            return path
    return "python3"


def _find_bizra_root() -> Optional[Path]:
    """Find the BIZRA source root (where core/ lives)."""
    # Check env var first
    env_root = os.environ.get("BIZRA_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root)

    # Check common locations
    candidates = [
        Path.home() / "BIZRA",
        Path.home() / "bizra",
        Path("/mnt/c/BIZRA-DATA-LAKE"),
        Path("C:/BIZRA-DATA-LAKE"),
        BIZRA_HOME / "source",
        Path.cwd(),
    ]

    for c in candidates:
        if (c / "core" / "sovereign" / "api.py").exists():
            return c

    return None


def _find_frontend_root() -> Optional[Path]:
    """Find the frontend root (where next.config exists)."""
    env_root = os.environ.get("BIZRA_FRONTEND")
    if env_root and Path(env_root).exists():
        return Path(env_root)

    candidates = [
        Path.home() / "award-winner-design",
        Path("/mnt/c/award-winner-design"),
        Path("C:/award-winner-design"),
        BIZRA_HOME / "frontend",
    ]

    for c in candidates:
        if (c / "next.config.mjs").exists() or (c / "next.config.js").exists():
            return c

    return None


def _api_health() -> Optional[dict]:
    """Check API health."""
    try:
        import urllib.request

        req = urllib.request.Request(
            f"http://127.0.0.1:{API_PORT}/v1/health",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _save_pid(pid: int):
    """Save process PID for later shutdown."""
    pids = _load_pids()
    pids.append(pid)
    BIZRA_PID.write_text(json.dumps(pids))


def _load_pids() -> list:
    """Load saved PIDs."""
    try:
        return json.loads(BIZRA_PID.read_text())
    except Exception:
        return []


def _clear_pids():
    """Clear PID file."""
    try:
        BIZRA_PID.unlink()
    except Exception:
        pass


# ─── Commands ────────────────────────────────────────────────────


def cmd_doctor():
    """Diagnose the BIZRA installation."""
    print(f"\n{C.BOLD}{C.WHITE}BIZRA Doctor{C.RESET}")
    print(f"{C.GRAY}{'─' * 50}{C.RESET}\n")

    issues = 0

    # Python
    py = _find_python()
    _print_status("Python", py, True)

    # BIZRA source
    root = _find_bizra_root()
    if root:
        _print_status("BIZRA source", str(root), True)
    else:
        _print_status("BIZRA source", "NOT FOUND", False)
        _print_warn("Set BIZRA_ROOT env var to your BIZRA-DATA-LAKE directory")
        issues += 1

    # Frontend
    frontend = _find_frontend_root()
    if frontend:
        _print_status("Frontend", str(frontend), True)
    else:
        _print_status("Frontend", "NOT FOUND", False)
        _print_warn("Set BIZRA_FRONTEND env var to your award-winner-design directory")
        issues += 1

    # Ollama
    ollama_path = shutil.which("ollama")
    if ollama_path:
        _print_status("Ollama", ollama_path, True)
    else:
        _print_status("Ollama", "NOT FOUND", False)
        _print_warn("Install Ollama: https://ollama.ai")
        issues += 1

    # Ollama running
    if _port_in_use(OLLAMA_PORT):
        _print_status("Ollama server", f"Running (:{OLLAMA_PORT})", True)
    else:
        _print_status("Ollama server", "Not running", False)
        issues += 1

    # API
    health = _api_health()
    if health:
        _print_status("Sovereign API", f"Healthy (:{API_PORT})", True)
    else:
        _print_status("Sovereign API", f"Not running (:{API_PORT})", False)
        issues += 1

    # Frontend dev server
    if _port_in_use(WEB_PORT):
        _print_status("Terminal UI", f"Running (:{WEB_PORT})", True)
    else:
        _print_status("Terminal UI", f"Not running (:{WEB_PORT})", False)
        issues += 1

    # Node.js
    node_path = shutil.which("node")
    if node_path:
        _print_status("Node.js", node_path, True)
    else:
        _print_status("Node.js", "NOT FOUND", False)
        issues += 1

    # BIZRA home
    _print_status("BIZRA home", str(BIZRA_HOME), BIZRA_HOME.exists())

    # Identity
    if BIZRA_IDENTITY.exists():
        _print_status("Node identity", "Exists", True)
    else:
        _print_status(
            "Node identity", "Not created (will generate on first run)", False
        )

    print(f"\n{C.GRAY}{'─' * 50}{C.RESET}")
    if issues == 0:
        print(f"  {C.GREEN}{C.BOLD}All systems operational.{C.RESET}")
    else:
        print(f"  {C.GOLD}{issues} issue(s) found.{C.RESET}")
    print()


def cmd_start(foreground: bool = False):
    """Start the BIZRA sovereign runtime."""
    _ensure_dirs()

    root = _find_bizra_root()
    if not root:
        _print_error("Cannot find BIZRA source. Set BIZRA_ROOT environment variable.")
        sys.exit(1)

    # Check if already running
    health = _api_health()
    if health:
        _print_info(f"Sovereign runtime already running on :{API_PORT}")
        return

    # Start Ollama if not running
    if not _port_in_use(OLLAMA_PORT) and shutil.which("ollama"):
        _print_info("Starting Ollama...")
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=open(BIZRA_LOGS / "ollama.log", "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _save_pid(proc.pid)
        time.sleep(2)

    # Start the sovereign API
    py = _find_python()
    venv_py = root / ".venv-linux" / "bin" / "python"
    venv_py_win = root / ".venv" / "Scripts" / "python.exe"

    if venv_py.exists():
        py = str(venv_py)
    elif venv_py_win.exists():
        py = str(venv_py_win)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["BIZRA_HOME"] = str(BIZRA_HOME)
    env["BIZRA_ENV"] = os.environ.get("BIZRA_ENV", "development")

    api_cmd = [
        py,
        "-m",
        "uvicorn",
        "core.sovereign.api:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(API_PORT),
        "--log-level",
        "info",
    ]

    if foreground:
        _print_info(f"Starting sovereign runtime on :{API_PORT} (foreground)...")
        subprocess.run(api_cmd, cwd=str(root), env=env)
    else:
        _print_info(f"Starting sovereign runtime on :{API_PORT}...")
        proc = subprocess.Popen(
            api_cmd,
            cwd=str(root),
            env=env,
            stdout=open(BIZRA_LOGS / "api.log", "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _save_pid(proc.pid)

        # Wait for healthy
        for i in range(15):
            time.sleep(1)
            if _api_health():
                _print_status("Sovereign runtime", f"Healthy (:{API_PORT})", True)
                return
            sys.stdout.write(".")
            sys.stdout.flush()

        print()
        _print_warn(
            "Runtime started but not yet healthy. Check logs: "
            + str(BIZRA_LOGS / "api.log")
        )


def cmd_stop():
    """Stop all BIZRA services."""
    pids = _load_pids()
    if not pids:
        _print_info("No BIZRA processes tracked.")
        return

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            _print_status(f"Process {pid}", "Stopped", True)
        except ProcessLookupError:
            _print_status(f"Process {pid}", "Already stopped", True)
        except PermissionError:
            _print_status(f"Process {pid}", "Permission denied", False)

    _clear_pids()
    _print_info("All BIZRA services stopped.")


def cmd_status():
    """Show node status."""
    health = _api_health()

    print(f"\n{C.BOLD}{C.WHITE}BIZRA Node Status{C.RESET}")
    print(f"{C.GRAY}{'─' * 50}{C.RESET}\n")

    if not health:
        _print_status("Runtime", "OFFLINE", False)
        _print_info(f"Start with: {C.WHITE}bizra start{C.RESET}")
        print()
        return

    _print_status("Runtime", "ONLINE", True)
    _print_status("Status", health.get("status", "unknown"), True)

    # Node identity
    if BIZRA_IDENTITY.exists():
        try:
            ident = json.loads(BIZRA_IDENTITY.read_text())
            pub_key = ident.get("public_key", "unknown")
            if len(pub_key) > 20:
                pub_key = pub_key[:10] + "..." + pub_key[-6:]
            _print_status("Node ID", pub_key, True)
        except Exception:
            pass

    # Try to get more data
    try:
        import urllib.request

        # Seed potential
        req = urllib.request.Request(f"http://127.0.0.1:{API_PORT}/v1/seed/potential")
        with urllib.request.urlopen(req, timeout=3) as resp:
            pot = json.loads(resp.read())
            score = pot.get("sovereignty_score", 0)
            tier = pot.get("tier", "UNKNOWN")
            _print_status("Sovereignty", f"{score:.2f}", score >= 0.5)
            _print_status("Tier", tier, True)
    except Exception:
        pass

    try:
        # Token balance
        req = urllib.request.Request(f"http://127.0.0.1:{API_PORT}/v1/token/balance")
        with urllib.request.urlopen(req, timeout=3) as resp:
            bal = json.loads(resp.read())
            balances = bal.get("balances", {})
            seed_bal = balances.get("SEED", {}).get("balance", 0)
            _print_status("SEED Balance", f"{seed_bal:.2f}", True)
    except Exception:
        pass

    print(f"\n{C.GRAY}{'─' * 50}{C.RESET}")
    print(f"  {C.TEAL}♥ Constitutional heartbeat: every 60s{C.RESET}")
    print(f"  {C.GRAY}Logs: {BIZRA_LOGS}{C.RESET}")
    print()


def cmd_mission(text: str):
    """Submit a mission from CLI."""
    health = _api_health()
    if not health:
        _print_error("Sovereign runtime not running. Start with: bizra start")
        sys.exit(1)

    print(f"\n{C.TEAL}🎯 Submitting mission...{C.RESET}")
    print(f'  {C.GRAY}"{text}"{C.RESET}\n')

    try:
        import urllib.request

        data = json.dumps(
            {
                "intent": text,
                "context": {},
            }
        ).encode()

        req = urllib.request.Request(
            f"http://127.0.0.1:{API_PORT}/v1/plan",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

            status = result.get("status", "unknown")
            ihsan = result.get("ihsan_score", 0)
            snr = result.get("snr_score", 0)
            synthesis = result.get("synthesis", "")
            seed = result.get("wallet_delta", {}).get("seed", 0)
            path = result.get("execution_path", "unknown")
            duration = result.get("duration_ms", 0)

            # Display receipt
            print(f"{C.GRAY}{'─' * 50}{C.RESET}")
            _print_status("Status", status, status == "COMPLETE")
            _print_status("Ihsān", f"{ihsan:.3f}", ihsan >= 0.95)
            _print_status("SNR", f"{snr:.3f}", snr >= 0.85)
            _print_status("Path", path, True)
            _print_status("Duration", f"{duration:.0f}ms", True)
            if seed > 0:
                _print_status("SEED Earned", f"+{seed:.2f}", True)
            print(f"{C.GRAY}{'─' * 50}{C.RESET}")

            if synthesis:
                print(f"\n  {C.WHITE}{synthesis}{C.RESET}")

            # Check for reflex compilation
            reflex = result.get("reflex_delta", {})
            if reflex.get("compiled"):
                print(
                    f"\n  {C.GOLD}⚡ REFLEX COMPILED — next execution will be System-1{C.RESET}"
                )
            elif reflex.get("near_compile"):
                count = reflex.get("compile_count", 0)
                threshold = reflex.get("threshold", 3)
                print(
                    f"\n  {C.GOLD}🔥 Near-compile: {count}/{threshold} toward reflex{C.RESET}"
                )

            print()

    except Exception as e:
        _print_error(f"Mission failed: {e}")
        sys.exit(1)


def cmd_briefing():
    """Show morning briefing from DEMA."""
    health = _api_health()
    if not health:
        _print_error("Sovereign runtime not running. Start with: bizra start")
        sys.exit(1)

    try:
        import urllib.request

        req = urllib.request.Request(
            f"http://127.0.0.1:{API_PORT}/v1/terminal/briefing"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            briefing = json.loads(resp.read())

            print(f"\n{C.PURPLE}💜 DEMA — Morning Briefing{C.RESET}")
            print(f"{C.GRAY}{'─' * 50}{C.RESET}\n")

            if isinstance(briefing, dict):
                for key, value in briefing.items():
                    if key != "timestamp":
                        print(f"  {C.TEAL}{key}:{C.RESET} {C.WHITE}{value}{C.RESET}")
            else:
                print(f"  {C.WHITE}{briefing}{C.RESET}")

            print()

    except Exception as e:
        _print_error(f"Could not fetch briefing: {e}")


def cmd_wallet():
    """Show wallet balance."""
    health = _api_health()
    if not health:
        _print_error("Sovereign runtime not running. Start with: bizra start")
        sys.exit(1)

    try:
        import urllib.request

        req = urllib.request.Request(f"http://127.0.0.1:{API_PORT}/v1/token/balance")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())

            print(f"\n{C.BOLD}{C.WHITE}BIZRA Wallet{C.RESET}")
            print(f"{C.GRAY}{'─' * 50}{C.RESET}\n")

            account = data.get("account", "unknown")
            _print_status("Account", account, True)

            balances = data.get("balances", {})
            for token, info in balances.items():
                bal = info.get("balance", 0) if isinstance(info, dict) else 0
                staked = info.get("staked", 0) if isinstance(info, dict) else 0
                color = C.GOLD if token == "SEED" else C.PURPLE
                print(
                    f"  {color}  {token}: {bal:.2f}{C.RESET}"
                    + (
                        f" {C.GRAY}(staked: {staked:.2f}){C.RESET}"
                        if staked > 0
                        else ""
                    )
                )

            print()

    except Exception as e:
        _print_error(f"Could not fetch wallet: {e}")


def cmd_identity():
    """Show node identity."""
    print(f"\n{C.BOLD}{C.WHITE}BIZRA Node Identity{C.RESET}")
    print(f"{C.GRAY}{'─' * 50}{C.RESET}\n")

    if BIZRA_IDENTITY.exists():
        try:
            ident = json.loads(BIZRA_IDENTITY.read_text())
            for key, value in ident.items():
                _print_status(key, str(value)[:60], True)
        except Exception as e:
            _print_error(f"Cannot read identity: {e}")
    else:
        _print_warn("No identity created yet.")
        _print_info("Identity is generated on first 'bizra start'")

    print()


def cmd_version():
    """Show version info."""
    print(f"\n  {C.BOLD}{C.WHITE}BIZRA{C.RESET} v{VERSION} ({CODENAME})")
    print(f"  {C.GRAY}Sovereign Mission Operating System{C.RESET}")
    print(f"  {C.GRAY}Dubai · BIZRA Foundation · 2026{C.RESET}")
    print(f'  {C.TEAL}"One mission, one proof, remembered forever."{C.RESET}')
    print()


def cmd_launch():
    """Main launch — start everything and open terminal."""
    _print_banner()

    # Ensure directories
    _ensure_dirs()

    # Step 1: Check/start runtime
    health = _api_health()
    if health:
        _print_status("Sovereign runtime", f"Already running (:{API_PORT})", True)
    else:
        _print_info("Starting sovereign runtime...")
        cmd_start()
        health = _api_health()
        if not health:
            _print_warn("Runtime starting — some features may be delayed")

    # Step 2: Check Ollama
    if _port_in_use(OLLAMA_PORT):
        _print_status("Ollama", f"Running (:{OLLAMA_PORT})", True)
    else:
        if shutil.which("ollama"):
            _print_info("Starting Ollama...")
            proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=open(BIZRA_LOGS / "ollama.log", "w"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _save_pid(proc.pid)
            _print_status("Ollama", "Starting...", True)
        else:
            _print_warn("Ollama not found — LLM features unavailable")

    # Step 3: Check/start frontend
    frontend = _find_frontend_root()
    if frontend and not _port_in_use(WEB_PORT):
        _print_info(f"Starting terminal UI on :{WEB_PORT}...")
        env = os.environ.copy()
        env["PORT"] = str(WEB_PORT)
        proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(frontend),
            env=env,
            stdout=open(BIZRA_LOGS / "frontend.log", "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _save_pid(proc.pid)
        _print_status("Terminal UI", f"Starting on :{WEB_PORT}", True)
    elif _port_in_use(WEB_PORT):
        _print_status("Terminal UI", f"Already running (:{WEB_PORT})", True)
    else:
        _print_warn("Frontend not found — CLI mode only")

    # Step 4: Show status
    print(f"\n{C.GRAY}{'─' * 50}{C.RESET}")

    if health:
        print(f"\n  {C.GREEN}{C.BOLD}BIZRA is alive.{C.RESET}")
        print(f"  {C.TEAL}♥ Constitutional heartbeat: every 60s{C.RESET}")
    else:
        print(f"\n  {C.GOLD}BIZRA is starting...{C.RESET}")

    print(f"\n  {C.WHITE}Terminal:{C.RESET}  http://localhost:{WEB_PORT}/terminal")
    print(f"  {C.WHITE}API:{C.RESET}       http://localhost:{API_PORT}/v1/health")
    print(f"  {C.WHITE}API Docs:{C.RESET}  http://localhost:{API_PORT}/docs")

    print(f"\n  {C.GRAY}Commands:{C.RESET}")
    print(f'    {C.TEAL}bizra mission "organize my files"{C.RESET}  — Submit a mission')
    print(
        f"    {C.TEAL}bizra briefing{C.RESET}                      — Morning briefing"
    )
    print(f"    {C.TEAL}bizra wallet{C.RESET}                        — Check balance")
    print(f"    {C.TEAL}bizra status{C.RESET}                        — Node status")
    print(
        f"    {C.TEAL}bizra stop{C.RESET}                          — Stop all services"
    )

    print(f"\n  {C.PURPLE}💜 {DEMA_GREETING}{C.RESET}")
    print(f'  {C.GOLD}"One mission, one proof, remembered forever."{C.RESET}')
    print()


# ─── CLI Entry Point ─────────────────────────────────────────────


def main():
    """
    CLI entry point — delegates to the modular CommandRegistry.

    The original monolithic if/elif chain is replaced by a registry
    that auto-discovers commands from core.cli.commands. All behavior
    is preserved. Bare `bizra` (no args) still launches the full terminal.
    """
    from core.cli.commands import ALL_COMMANDS
    from core.cli.hooks import CLIHooksManager
    from core.cli.registry import CommandRegistry

    # Build registry
    registry = CommandRegistry()
    for cmd_class in ALL_COMMANDS:
        registry.register(cmd_class())

    # Wire hooks (no EventBus at CLI level — hooks record locally)
    hooks = CLIHooksManager()
    registry.add_pre_hook(hooks.pre_command)
    registry.add_post_hook(hooks.post_command)

    args = sys.argv[1:]

    # Bare `bizra` → launch (default command)
    if not args:
        entry = registry.resolve("launch")
        if entry:
            entry.command.execute([])
        return

    cmd_name = args[0].lower().strip("-")

    # Help is handled directly (prints module docstring)
    if cmd_name in ("help", "h", "--help", "-h"):
        print(__doc__)
        return

    # Try registry dispatch
    result = registry.dispatch(args)

    # If unknown, treat the entire input as a mission (legacy behavior)
    if not result.success and "Unknown command" in result.message:
        result = registry.dispatch(["mission"] + args)

    if not result.success and result.exit_code != 0:
        sys.exit(result.exit_code)


if __name__ == "__main__":
    main()
