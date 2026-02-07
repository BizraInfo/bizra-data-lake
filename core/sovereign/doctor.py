"""
BIZRA Sovereign Engine — Health Check (Doctor Command)

Diagnoses the system state and checks all dependencies:
- LLM backends (LM Studio, Ollama, llama.cpp)
- Python version and dependencies
- GPU availability
- File system permissions
- Network connectivity

Usage:
    bizra doctor
    python -m core.sovereign doctor

Standing on Giants: Shannon • Lamport • Vaswani • Anthropic
"""

import asyncio
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List


class CheckStatus(str, Enum):
    """Status of a health check."""

    OK = "✅"
    WARN = "⚠️"
    FAIL = "❌"
    SKIP = "⏭️"


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    status: CheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DoctorReport:
    """Complete doctor report."""

    checks: List[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    @property
    def ok_count(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.OK)

    @property
    def warn_count(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.WARN)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

    @property
    def healthy(self) -> bool:
        return self.fail_count == 0


class BizraDoctor:
    """
    Health check system for BIZRA Sovereign Engine.

    Checks all components and provides actionable recommendations.
    """

    def __init__(self):
        self.report = DoctorReport()

    async def run_all_checks(self) -> DoctorReport:
        """Run all health checks."""
        self.report = DoctorReport()

        # Core checks
        await self.check_python()
        await self.check_dependencies()

        # Genesis identity
        await self.check_genesis()

        # LLM backends
        await self.check_lmstudio()
        await self.check_ollama()
        await self.check_llamacpp()

        # System
        await self.check_gpu()
        await self.check_filesystem()
        await self.check_constants()
        await self.check_memory_persistence()

        return self.report

    async def check_python(self) -> None:
        """Check Python version."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major >= 3 and version.minor >= 9:
            self.report.add(
                CheckResult(
                    name="Python Version",
                    status=CheckStatus.OK,
                    message=f"Python {version_str}",
                    details={"version": version_str, "required": ">=3.9"},
                )
            )
        else:
            self.report.add(
                CheckResult(
                    name="Python Version",
                    status=CheckStatus.FAIL,
                    message=f"Python {version_str} (requires >=3.9)",
                    details={"version": version_str, "required": ">=3.9"},
                )
            )

    async def check_dependencies(self) -> None:
        """Check required Python packages."""
        required = {
            "numpy": "numpy",
            "httpx": "httpx",
            "pydantic": "pydantic",
        }
        optional = {
            "torch": "torch",
            "networkx": "networkx",
            "faiss": "faiss",
        }

        missing = []
        available = []

        for name, module in required.items():
            try:
                __import__(module)
                available.append(name)
            except ImportError:
                missing.append(name)

        if not missing:
            self.report.add(
                CheckResult(
                    name="Core Dependencies",
                    status=CheckStatus.OK,
                    message=f"All core packages installed ({len(available)} packages)",
                    details={"available": available},
                )
            )
        else:
            self.report.add(
                CheckResult(
                    name="Core Dependencies",
                    status=CheckStatus.FAIL,
                    message=f"Missing: {', '.join(missing)}",
                    details={"missing": missing, "available": available},
                )
            )

        # Check optional
        opt_available = []
        opt_missing = []
        for name, module in optional.items():
            try:
                __import__(module)
                opt_available.append(name)
            except ImportError:
                opt_missing.append(name)

        if opt_missing:
            self.report.add(
                CheckResult(
                    name="Optional Dependencies",
                    status=CheckStatus.WARN,
                    message=f"Optional packages missing: {', '.join(opt_missing)}",
                    details={"available": opt_available, "missing": opt_missing},
                )
            )
        else:
            self.report.add(
                CheckResult(
                    name="Optional Dependencies",
                    status=CheckStatus.OK,
                    message="All optional packages installed",
                    details={"available": opt_available},
                )
            )

    async def check_genesis(self) -> None:
        """Check genesis identity state."""
        state_dir = Path("sovereign_state")
        genesis_file = state_dir / "node0_genesis.json"

        if not genesis_file.exists():
            self.report.add(
                CheckResult(
                    name="Genesis Identity",
                    status=CheckStatus.WARN,
                    message="No genesis file — running as ephemeral node",
                    details={"path": str(genesis_file)},
                )
            )
            return

        try:
            from .genesis_identity import load_and_validate_genesis

            genesis = load_and_validate_genesis(state_dir)
            if genesis is None:
                self.report.add(
                    CheckResult(
                        name="Genesis Identity",
                        status=CheckStatus.FAIL,
                        message="Genesis file exists but failed to load",
                        details={"path": str(genesis_file)},
                    )
                )
                return

            self.report.add(
                CheckResult(
                    name="Genesis Identity",
                    status=CheckStatus.OK,
                    message=(
                        f"{genesis.node_id} ({genesis.node_name}) — "
                        f"{len(genesis.pat_team)} PAT + {len(genesis.sat_team)} SAT"
                    ),
                    details={
                        "node_id": genesis.node_id,
                        "name": genesis.node_name,
                        "location": genesis.identity.location,
                        "pat_agents": len(genesis.pat_team),
                        "sat_agents": len(genesis.sat_team),
                        "genesis_hash": genesis.genesis_hash.hex()[:16] + "...",
                    },
                )
            )
        except ValueError as e:
            self.report.add(
                CheckResult(
                    name="Genesis Identity",
                    status=CheckStatus.FAIL,
                    message=f"Genesis corrupted: {e}",
                    details={"path": str(genesis_file), "error": str(e)},
                )
            )
        except ImportError:
            self.report.add(
                CheckResult(
                    name="Genesis Identity",
                    status=CheckStatus.SKIP,
                    message="Genesis loader not available",
                )
            )

    async def check_lmstudio(self) -> None:
        """Check LM Studio availability (v1 API with auth)."""
        import os

        host = os.getenv("LMSTUDIO_HOST", "192.168.56.1")
        port = os.getenv("LMSTUDIO_PORT", "1234")
        url = f"http://{host}:{port}/api/v1/models"

        api_key = (
            os.getenv("LM_API_TOKEN")
            or os.getenv("LMSTUDIO_API_KEY")
            or os.getenv("LM_STUDIO_API_KEY")
        )

        try:
            headers = {"Accept": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                models = data.get("models", data.get("data", []))
                loaded = [m for m in models if m.get("loaded_instances")]
                model_ids = [m.get("key", m.get("id", "unknown")) for m in models[:5]]

                self.report.add(
                    CheckResult(
                        name="LM Studio",
                        status=CheckStatus.OK,
                        message=f"Connected at {host}:{port} ({len(models)} models, {len(loaded)} loaded)",
                        details={"url": url, "models": model_ids, "count": len(models), "loaded": len(loaded)},
                    )
                )
        except urllib.error.HTTPError as e:
            if e.code == 401:
                self.report.add(
                    CheckResult(
                        name="LM Studio",
                        status=CheckStatus.WARN,
                        message=f"Auth required at {host}:{port} — set LM_API_TOKEN env var",
                        details={"url": url, "error": "401 Unauthorized"},
                    )
                )
            else:
                self.report.add(
                    CheckResult(
                        name="LM Studio",
                        status=CheckStatus.WARN,
                        message=f"HTTP {e.code} at {host}:{port}",
                        details={"url": url, "error": str(e)},
                    )
                )
        except urllib.error.URLError as e:
            self.report.add(
                CheckResult(
                    name="LM Studio",
                    status=CheckStatus.WARN,
                    message="Not running (optional, use Ollama fallback)",
                    details={"url": url, "error": str(e)},
                )
            )
        except Exception as e:
            self.report.add(
                CheckResult(
                    name="LM Studio",
                    status=CheckStatus.WARN,
                    message=f"Connection error: {e}",
                    details={"url": url, "error": str(e)},
                )
            )

    async def check_ollama(self) -> None:
        """Check Ollama availability."""
        url = "http://localhost:11434/api/tags"

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                models = data.get("models", [])
                model_names = [m.get("name", "unknown") for m in models[:3]]

                self.report.add(
                    CheckResult(
                        name="Ollama",
                        status=CheckStatus.OK,
                        message=f"Connected at localhost:11434 ({len(models)} models)",
                        details={
                            "url": url,
                            "models": model_names,
                            "count": len(models),
                        },
                    )
                )
        except urllib.error.URLError:
            self.report.add(
                CheckResult(
                    name="Ollama",
                    status=CheckStatus.WARN,
                    message="Not running (install: curl -fsSL https://ollama.com/install.sh | sh)",
                    details={"url": url},
                )
            )
        except Exception as e:
            self.report.add(
                CheckResult(
                    name="Ollama",
                    status=CheckStatus.WARN,
                    message=f"Connection error: {e}",
                    details={"url": url, "error": str(e)},
                )
            )

    async def check_llamacpp(self) -> None:
        """Check llama-cpp-python availability."""
        try:
            import llama_cpp

            self.report.add(
                CheckResult(
                    name="llama.cpp",
                    status=CheckStatus.OK,
                    message="llama-cpp-python installed (offline inference ready)",
                    details={"module": "llama_cpp"},
                )
            )
        except ImportError:
            self.report.add(
                CheckResult(
                    name="llama.cpp",
                    status=CheckStatus.SKIP,
                    message="Not installed (optional, for offline inference)",
                    details={"install": "pip install llama-cpp-python"},
                )
            )

    async def check_gpu(self) -> None:
        """Check GPU availability."""
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.report.add(
                    CheckResult(
                        name="GPU",
                        status=CheckStatus.OK,
                        message=f"{gpu_name} ({vram:.1f} GB VRAM)",
                        details={"name": gpu_name, "vram_gb": vram, "cuda": True},
                    )
                )
            else:
                self.report.add(
                    CheckResult(
                        name="GPU",
                        status=CheckStatus.WARN,
                        message="CUDA not available (using CPU inference)",
                        details={"cuda": False},
                    )
                )
        except ImportError:
            self.report.add(
                CheckResult(
                    name="GPU",
                    status=CheckStatus.SKIP,
                    message="PyTorch not installed (GPU check skipped)",
                    details={"torch": False},
                )
            )

    async def check_filesystem(self) -> None:
        """Check filesystem permissions."""
        paths_to_check = [
            Path.home() / ".bizra",
            Path("./sovereign_state"),
        ]

        writable = []
        not_writable = []

        for path in paths_to_check:
            try:
                path.mkdir(parents=True, exist_ok=True)
                test_file = path / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
                writable.append(str(path))
            except Exception:
                not_writable.append(str(path))

        if not not_writable:
            self.report.add(
                CheckResult(
                    name="Filesystem",
                    status=CheckStatus.OK,
                    message="Write permissions verified",
                    details={"writable": writable},
                )
            )
        else:
            self.report.add(
                CheckResult(
                    name="Filesystem",
                    status=CheckStatus.WARN,
                    message=f"Cannot write to: {', '.join(not_writable)}",
                    details={"writable": writable, "not_writable": not_writable},
                )
            )

    async def check_constants(self) -> None:
        """Check that constants are properly configured."""
        try:
            from core.integration.constants import (
                UNIFIED_IHSAN_THRESHOLD,
                UNIFIED_SNR_THRESHOLD,
            )

            self.report.add(
                CheckResult(
                    name="Constants",
                    status=CheckStatus.OK,
                    message=f"Ihsān={UNIFIED_IHSAN_THRESHOLD}, SNR={UNIFIED_SNR_THRESHOLD}",
                    details={
                        "ihsan_threshold": UNIFIED_IHSAN_THRESHOLD,
                        "snr_threshold": UNIFIED_SNR_THRESHOLD,
                    },
                )
            )
        except ImportError as e:
            self.report.add(
                CheckResult(
                    name="Constants",
                    status=CheckStatus.FAIL,
                    message=f"Cannot import constants: {e}",
                    details={"error": str(e)},
                )
            )


    async def check_memory_persistence(self) -> None:
        """Check memory persistence state."""
        state_dir = Path("sovereign_state")
        checkpoints_dir = state_dir / "checkpoints"
        living_memory_dir = state_dir / "living_memory"

        details: Dict[str, Any] = {}
        issues = []

        # Check checkpoint directory
        if checkpoints_dir.exists():
            cp_files = list(checkpoints_dir.glob("cp-*.json"))
            details["checkpoints"] = len(cp_files)
        else:
            details["checkpoints"] = 0
            issues.append("no checkpoints yet")

        # Check living memory
        memory_file = living_memory_dir / "memories.jsonl"
        if memory_file.exists():
            line_count = sum(1 for _ in open(memory_file))
            details["living_memory_entries"] = line_count
        else:
            details["living_memory_entries"] = 0

        # Check coordinator module
        try:
            from .memory_coordinator import MemoryCoordinator

            details["coordinator_available"] = True
        except ImportError:
            details["coordinator_available"] = False
            issues.append("MemoryCoordinator not importable")

        if issues:
            self.report.add(
                CheckResult(
                    name="Memory Persistence",
                    status=CheckStatus.WARN,
                    message=f"Issues: {', '.join(issues)}",
                    details=details,
                )
            )
        else:
            self.report.add(
                CheckResult(
                    name="Memory Persistence",
                    status=CheckStatus.OK,
                    message=f"{details['checkpoints']} checkpoints, {details['living_memory_entries']} memories",
                    details=details,
                )
            )


def print_report(report: DoctorReport, verbose: bool = False) -> None:
    """Print doctor report to console."""
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                    BIZRA SOVEREIGN DOCTOR                        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    for check in report.checks:
        print(f"  {check.status.value} {check.name}: {check.message}")
        if verbose and check.details:
            for key, value in check.details.items():
                print(f"       └─ {key}: {value}")

    print()
    print("─" * 60)
    print(
        f"  Summary: {report.ok_count} OK, {report.warn_count} warnings, {report.fail_count} failures"
    )
    print()

    if report.healthy:
        print("  ✅ System is ready. Run 'bizra start' to begin.")
    else:
        print("  ❌ System has issues. Please resolve failures above.")

    print()


async def run_doctor(verbose: bool = False, json_output: bool = False) -> int:
    """Run doctor and return exit code."""
    doctor = BizraDoctor()
    report = await doctor.run_all_checks()

    if json_output:
        import json

        output = {
            "healthy": report.healthy,
            "summary": {
                "ok": report.ok_count,
                "warn": report.warn_count,
                "fail": report.fail_count,
            },
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                }
                for c in report.checks
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report, verbose)

    return 0 if report.healthy else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA Sovereign Doctor")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    exit_code = asyncio.run(run_doctor(args.verbose, args.json))
    sys.exit(exit_code)
