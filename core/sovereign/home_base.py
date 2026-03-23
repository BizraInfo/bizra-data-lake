"""
Home Base Awareness — the PAT's persistent knowledge of its sovereign space.

The PAT agents live inside the user's computer. This is their home.
They must know everything about it: hardware, software, data, changes.
A person in their home doesn't need to be told what's in the fridge.

This module maintains a living index of the entire local environment
and detects what changed since the last scan. The Ghost Panel draws
from this to make proactive suggestions.

Standing on: Maturana (autopoiesis — system knows itself),
Gibson (affordances — environment shapes action).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bizra.home_base")

HOME_BASE_PATH = Path.home() / ".bizra" / "home_base.json"


@dataclass
class HardwareProfile:
    """What the PAT knows about its physical home."""

    hostname: str = ""
    os_name: str = ""
    os_version: str = ""
    cpu: str = ""
    cpu_cores: int = 0
    ram_gb: float = 0.0
    gpu: str = ""
    disks: List[Dict[str, Any]] = field(default_factory=list)
    total_storage_gb: float = 0.0
    free_storage_gb: float = 0.0


@dataclass
class SoftwareProfile:
    """What the PAT knows about installed tools."""

    python_version: str = ""
    rust_version: str = ""
    node_version: str = ""
    ollama_models: List[str] = field(default_factory=list)
    docker_running: bool = False
    git_version: str = ""


@dataclass
class DataProfile:
    """What the PAT knows about the user's data."""

    watched_dirs: Dict[str, int] = field(default_factory=dict)  # dir → file count
    total_files: int = 0
    total_size_gb: float = 0.0
    recent_changes: List[Dict[str, Any]] = field(default_factory=list)
    unprocessed_items: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TaskState:
    """What the PAT knows about pending deliverables."""

    pending_actions: List[Dict[str, Any]] = field(default_factory=list)
    stale_deliverables: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class HomeBase:
    """Complete awareness of the sovereign local space."""

    last_scan: float = 0.0
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    software: SoftwareProfile = field(default_factory=SoftwareProfile)
    data: DataProfile = field(default_factory=DataProfile)
    tasks: TaskState = field(default_factory=TaskState)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def scan_hardware() -> HardwareProfile:
    """Discover hardware — the PAT knows its physical home."""
    hw = HardwareProfile()
    import subprocess

    hw.hostname = platform.node()
    hw.os_name = platform.system()
    hw.os_version = platform.version()

    # Detect real hardware via Windows (WSL2 underreports)
    def _ps(cmd: str) -> str:
        try:
            r = subprocess.run(
                ["powershell.exe", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return r.stdout.strip().replace("\r", "") if r.returncode == 0 else ""
        except Exception:
            return ""

    hw.cpu = (
        _ps("(Get-WmiObject Win32_Processor).Name") or platform.processor() or "unknown"
    )

    cores = _ps("(Get-WmiObject Win32_Processor).NumberOfLogicalProcessors")
    hw.cpu_cores = int(cores) if cores.isdigit() else (os.cpu_count() or 0)

    ram = _ps(
        "[math]::Round((Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory/1GB)"
    )
    if ram.isdigit():
        hw.ram_gb = float(ram)
    else:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        hw.ram_gb = round(int(line.split()[1]) / (1024**2), 1)
                        break
        except Exception:
            pass

    # GPU — nvidia-smi is fastest
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            hw.gpu = result.stdout.strip()
    except Exception:
        hw.gpu = _ps("(Get-WmiObject Win32_VideoController).Name") or ""

    # Disk space — the PAT knows every drive in its home
    for mount in ["/", "/mnt/c", "/mnt/b", "/mnt/d"]:
        try:
            usage = shutil.disk_usage(mount)
            hw.disks.append(
                {
                    "mount": mount,
                    "total_gb": round(usage.total / (1024**3), 1),
                    "free_gb": round(usage.free / (1024**3), 1),
                }
            )
            hw.total_storage_gb += round(usage.total / (1024**3), 1)
            hw.free_storage_gb += round(usage.free / (1024**3), 1)
        except Exception:
            pass

    return hw


def scan_software() -> SoftwareProfile:
    """Discover software — the PAT knows its tools."""
    sw = SoftwareProfile()
    sw.python_version = platform.python_version()

    import subprocess

    for cmd, attr in [
        (["rustc", "--version"], "rust_version"),
        (["node", "--version"], "node_version"),
        (["git", "--version"], "git_version"),
    ]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                setattr(sw, attr, result.stdout.strip())
        except Exception:
            pass

    # Ollama models
    try:
        import urllib.request

        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        data = json.loads(resp.read())
        sw.ollama_models = [m["name"] for m in data.get("models", [])]
    except Exception:
        pass

    # Docker
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        sw.docker_running = result.returncode == 0
    except Exception:
        pass

    return sw


def scan_data(watched_dirs: Optional[List[str]] = None) -> DataProfile:
    """Discover data — the PAT knows what's in every room."""
    dp = DataProfile()

    dirs_to_watch = watched_dirs or [
        "/mnt/c",  # Full C: partition — the PAT's primary home
    ]

    # B: drive — BIZRA Sovereign space (the PAT's full home)
    for sovereign_path in [
        "/mnt/b/BIZRA-SOVEREIGN",
        "/mnt/b/all files",
        "/mnt/b",
    ]:
        if os.path.isdir(sovereign_path) and sovereign_path not in dirs_to_watch:
            dirs_to_watch.append(sovereign_path)

    # Also check Windows paths
    for win_path in [
        "/mnt/c/Users/BIZRA-OS/Downloads",
        "/mnt/c/Users/BIZRA-OS/Desktop",
        "/mnt/c/Users/BIZRA-OS/Documents",
    ]:
        if os.path.isdir(win_path) and win_path not in dirs_to_watch:
            dirs_to_watch.append(win_path)

    for d in dirs_to_watch:
        if not os.path.isdir(d):
            continue
        try:
            # Top-level: count immediate children (files + dirs)
            items = list(Path(d).iterdir())
            files = [i for i in items if i.is_file()]
            dirs = [i for i in items if i.is_dir()]
            dp.watched_dirs[d] = len(files)
            dp.total_files += len(files)

            # Go one level deeper for key directories
            for subdir in dirs:
                name = subdir.name
                # Skip system/hidden dirs
                if name.startswith(".") or name in (
                    "$RECYCLE.BIN",
                    "System Volume Information",
                    "Windows",
                    "ProgramData",
                    "Recovery",
                ):
                    continue
                try:
                    sub_count = sum(1 for _ in subdir.iterdir())
                    dp.watched_dirs[str(subdir)] = sub_count
                    dp.total_files += sub_count
                except (PermissionError, OSError):
                    pass
        except (PermissionError, OSError):
            pass

    return dp


def scan_task_state() -> TaskState:
    """Discover pending deliverables — what needs attention."""
    ts = TaskState()

    # Check for papers not submitted
    papers_dir = Path("/mnt/c/BIZRA-DATA-LAKE/docs/papers")
    if papers_dir.exists():
        for p in papers_dir.glob("*.md"):
            age_hours = (time.time() - p.stat().st_mtime) / 3600
            if age_hours > 24:
                ts.stale_deliverables.append(
                    {
                        "type": "paper",
                        "path": str(p),
                        "name": p.name,
                        "age_hours": round(age_hours, 1),
                        "suggestion": "Convert to PDF and submit to ArXiv",
                    }
                )

    # Check for unprocessed Downloads
    downloads = Path("/mnt/c/Users/BIZRA-OS/Downloads")
    if downloads.exists():
        loose = sum(1 for f in downloads.iterdir() if f.is_file())
        if loose > 10:
            ts.pending_actions.append(
                {
                    "type": "organize",
                    "path": str(downloads),
                    "count": loose,
                    "suggestion": f"Downloads has {loose} loose files — organize?",
                }
            )

    # Check if heartbeat is running
    try:
        import urllib.request

        urllib.request.urlopen("http://127.0.0.1:9740/api/health", timeout=2)
    except Exception:
        ts.pending_actions.append(
            {
                "type": "service",
                "name": "kernel_daemon",
                "suggestion": "Kernel daemon offline — run: bizra start",
            }
        )

    # Check SEED balance
    try:
        from core.proof_engine.seed_ledger import balance

        b = balance()
        if b == 0:
            ts.pending_actions.append(
                {
                    "type": "onboarding",
                    "suggestion": "No SEED earned yet — run your first mission: bizra mission 'hello'",
                }
            )
    except Exception:
        pass

    return ts


def full_scan() -> HomeBase:
    """Complete home base scan — the PAT opens its eyes."""
    logger.info("Scanning home base...")
    t0 = time.time()

    hb = HomeBase(
        last_scan=time.time(),
        hardware=scan_hardware(),
        software=scan_software(),
        data=scan_data(),
        tasks=scan_task_state(),
    )

    elapsed = time.time() - t0
    logger.info("Home base scan: %.1fs", elapsed)
    return hb


def save_home_base(hb: HomeBase) -> Path:
    """Persist home base awareness to disk."""
    HOME_BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HOME_BASE_PATH, "w") as f:
        json.dump(hb.to_dict(), f, indent=2)
    return HOME_BASE_PATH


def load_home_base() -> Optional[HomeBase]:
    """Load last known home base state."""
    if not HOME_BASE_PATH.exists():
        return None
    try:
        with open(HOME_BASE_PATH) as f:
            data = json.load(f)
        hb = HomeBase(last_scan=data.get("last_scan", 0))
        # Reconstruct from dict (simplified)
        hb.hardware = HardwareProfile(**data.get("hardware", {}))
        hb.software = SoftwareProfile(**data.get("software", {}))
        return hb
    except Exception:
        return None


def detect_changes(previous: HomeBase, current: HomeBase) -> List[Dict[str, Any]]:
    """What changed since last scan — the PAT notices."""
    changes = []

    # New files in watched directories
    for d, count in current.data.watched_dirs.items():
        prev_count = previous.data.watched_dirs.get(d, 0)
        if count > prev_count:
            changes.append(
                {
                    "type": "new_files",
                    "directory": d,
                    "added": count - prev_count,
                    "suggestion": f"{count - prev_count} new files in {Path(d).name}",
                }
            )

    # New Ollama models
    prev_models = set(previous.software.ollama_models)
    curr_models = set(current.software.ollama_models)
    new_models = curr_models - prev_models
    if new_models:
        changes.append(
            {
                "type": "new_models",
                "models": list(new_models),
                "suggestion": f"New models available: {', '.join(new_models)}",
            }
        )

    return changes


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    hb = full_scan()
    path = save_home_base(hb)

    print(f"\nHome Base Awareness — {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}")
    print("\nHardware:")
    print(f"  Host:    {hb.hardware.hostname}")
    print(f"  OS:      {hb.hardware.os_name} {hb.hardware.os_version[:30]}")
    print(f"  CPU:     {hb.hardware.cpu} ({hb.hardware.cpu_cores} cores)")
    print(f"  RAM:     {hb.hardware.ram_gb} GB")
    print(f"  GPU:     {hb.hardware.gpu or 'none detected'}")
    print(
        f"  Storage: {hb.hardware.free_storage_gb:.0f} GB free / {hb.hardware.total_storage_gb:.0f} GB total"
    )

    print("\nSoftware:")
    print(f"  Python:  {hb.software.python_version}")
    print(f"  Rust:    {hb.software.rust_version}")
    print(f"  Git:     {hb.software.git_version}")
    print(f"  Ollama:  {len(hb.software.ollama_models)} models")
    for m in hb.software.ollama_models:
        print(f"    - {m}")

    print(f"\nData ({hb.data.total_files} files watched):")
    for d, count in hb.data.watched_dirs.items():
        print(f"  {Path(d).name}: {count} files")

    if hb.tasks.pending_actions:
        print("\nPending Actions:")
        for a in hb.tasks.pending_actions:
            print(f"  ! {a['suggestion']}")

    if hb.tasks.stale_deliverables:
        print("\nStale Deliverables:")
        for d in hb.tasks.stale_deliverables:
            print(f"  ! {d['name']} — {d['age_hours']:.0f}h old — {d['suggestion']}")

    print(f"\nSaved to: {path}")
