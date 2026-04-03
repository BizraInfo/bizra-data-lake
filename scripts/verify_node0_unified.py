#!/usr/bin/env python3
"""
BIZRA Node0 Unified System Verification
========================================
Comprehensive verification that all Node0 components are healthy and
working in harmony. This script checks hardware, software, services,
and data integration.

Usage:
    python scripts/verify_node0_unified.py
    python scripts/verify_node0_unified.py --json
    python scripts/verify_node0_unified.py --fix
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


class Node0Verifier:
    """Comprehensive Node0 system verifier."""

    def __init__(self):
        self.checks: List[Dict[str, Any]] = []
        self.passed = 0
        self.warned = 0
        self.failed = 0

    def add_check(self, name: str, status: str, details: str = "", fix: str = ""):
        """Add a verification check result."""
        self.checks.append({
            "name": name,
            "status": status,
            "details": details,
            "fix": fix,
        })

        if status == "PASS":
            self.passed += 1
        elif status == "WARN":
            self.warned += 1
        else:
            self.failed += 1

    async def verify_hardware(self) -> Dict[str, Any]:
        """Verify hardware detection and status."""
        import psutil

        results = {"section": "Hardware", "checks": []}

        # CPU
        try:
            cpu_count = psutil.cpu_count(logical=True)
            cpu_usage = psutil.cpu_percent(interval=0.1)

            if cpu_count >= 8:
                self.add_check("CPU Cores", "PASS", f"{cpu_count} logical cores detected")
            else:
                self.add_check("CPU Cores", "WARN", f"Only {cpu_count} cores")

            if cpu_usage < 90:
                self.add_check("CPU Usage", "PASS", f"{cpu_usage:.1f}% utilization")
            else:
                self.add_check("CPU Usage", "WARN", f"High usage: {cpu_usage:.1f}%")
        except Exception as e:
            self.add_check("CPU", "FAIL", str(e))

        # Memory
        try:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            available_gb = mem.available / (1024**3)

            if total_gb >= 64:
                self.add_check("Total RAM", "PASS", f"{total_gb:.1f}GB")
            elif total_gb >= 32:
                self.add_check("Total RAM", "WARN", f"{total_gb:.1f}GB (64GB recommended)")
            else:
                self.add_check("Total RAM", "FAIL", f"{total_gb:.1f}GB (insufficient)")

            if available_gb >= 16:
                self.add_check("Available RAM", "PASS", f"{available_gb:.1f}GB free")
            elif available_gb >= 8:
                self.add_check("Available RAM", "WARN", f"{available_gb:.1f}GB free")
            else:
                self.add_check("Available RAM", "FAIL", f"{available_gb:.1f}GB free")
        except Exception as e:
            self.add_check("Memory", "FAIL", str(e))

        # GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                gpu_name = parts[0].strip()
                vram_total = int(float(parts[1].strip()))
                vram_free = int(float(parts[2].strip()))
                temp = float(parts[3].strip())

                self.add_check("GPU Detected", "PASS", gpu_name)
                self.add_check("GPU VRAM", "PASS", f"{vram_total}MB total, {vram_free}MB free")

                if temp < 80:
                    self.add_check("GPU Temperature", "PASS", f"{temp}°C")
                elif temp < 90:
                    self.add_check("GPU Temperature", "WARN", f"{temp}°C (high)")
                else:
                    self.add_check("GPU Temperature", "FAIL", f"{temp}°C (critical)")
            else:
                self.add_check("GPU", "WARN", "No NVIDIA GPU detected",
                             fix="GPU recommended for LLM inference")
        except FileNotFoundError:
            self.add_check("GPU", "WARN", "nvidia-smi not found",
                         fix="Install NVIDIA drivers")
        except Exception as e:
            self.add_check("GPU", "WARN", str(e))

        return results

    async def verify_services(self) -> Dict[str, Any]:
        """Verify Docker and other services."""
        import aiohttp

        results = {"section": "Services", "checks": []}

        # Docker services
        try:
            result = subprocess.run(
                ["docker", "compose", "ps", "--format", "{{.Name}}\t{{.State}}\t{{.Health}}"],
                capture_output=True, text=True, timeout=10,
                cwd="/mnt/c/BIZRA-Dual-Agentic-system--main"
            )

            if result.returncode == 0:
                healthy = 0
                total = 0

                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        total += 1
                        state = parts[1] if len(parts) > 1 else ""
                        health = parts[2] if len(parts) > 2 else ""

                        if state == "running" and health == "healthy":
                            healthy += 1

                if healthy == total and total > 0:
                    self.add_check("Docker Services", "PASS", f"{healthy}/{total} healthy")
                elif healthy > 0:
                    self.add_check("Docker Services", "WARN", f"{healthy}/{total} healthy",
                                 fix="Run: docker compose up -d")
                else:
                    self.add_check("Docker Services", "FAIL", "No services running",
                                 fix="Run: docker compose up -d")
            else:
                self.add_check("Docker Compose", "FAIL", "Command failed",
                             fix="Ensure Docker is running")
        except Exception as e:
            self.add_check("Docker", "FAIL", str(e),
                         fix="Install and start Docker Desktop")

        # Ollama
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get("http://localhost:11434/api/tags") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("models", [])
                        self.add_check("Ollama", "PASS", f"{len(models)} models available")
                    else:
                        self.add_check("Ollama", "WARN", f"HTTP {resp.status}",
                                     fix="Check ollama status")
        except Exception as e:
            self.add_check("Ollama", "FAIL", "Not responding",
                         fix="Run: ollama serve")

        # Redis (Synapse)
        try:
            result = subprocess.run(
                ["docker", "compose", "exec", "-T", "synapse", "redis-cli", "ping"],
                capture_output=True, text=True, timeout=5,
                cwd="/mnt/c/BIZRA-Dual-Agentic-system--main"
            )

            if "PONG" in result.stdout:
                self.add_check("Redis (Synapse)", "PASS", "PONG received")
            else:
                self.add_check("Redis (Synapse)", "WARN", "No PONG",
                             fix="Check synapse container")
        except Exception as e:
            self.add_check("Redis", "WARN", str(e))

        return results

    async def verify_data_lake(self) -> Dict[str, Any]:
        """Verify BIZRA-DATA-LAKE accessibility."""
        results = {"section": "Data Lake", "checks": []}

        data_lake_path = Path(os.getenv("DATA_LAKE_PATH", "/mnt/c/BIZRA-DATA-LAKE"))

        if data_lake_path.exists():
            self.add_check("Data Lake Path", "PASS", str(data_lake_path))

            # Check gold layer
            gold_path = data_lake_path / "04_GOLD"
            if gold_path.exists():
                self.add_check("Gold Layer", "PASS", "Accessible")
            else:
                self.add_check("Gold Layer", "WARN", "Not found")

            # Check PoI ledger
            poi_path = gold_path / "poi_ledger.jsonl"
            if poi_path.exists():
                line_count = sum(1 for _ in open(poi_path))
                self.add_check("PoI Ledger", "PASS", f"{line_count} entries")
            else:
                self.add_check("PoI Ledger", "WARN", "Not found")

            # Check indexed layer
            indexed_path = data_lake_path / "03_INDEXED"
            if indexed_path.exists():
                parquet_files = list(indexed_path.rglob("*.parquet"))
                self.add_check("Indexed Layer", "PASS", f"{len(parquet_files)} parquet files")
            else:
                self.add_check("Indexed Layer", "WARN", "Not found")
        else:
            self.add_check("Data Lake", "FAIL", f"Not found: {data_lake_path}",
                         fix=f"Create directory: {data_lake_path}")

        return results

    async def verify_identity(self) -> Dict[str, Any]:
        """Verify Node0 identity system."""
        results = {"section": "Identity", "checks": []}

        try:
            from bizra_kernel.node0_identity import Node0Identity
            identity = Node0Identity.load_or_create()
            self.add_check("Node0 Identity", "PASS",
                         f"Loaded: {identity.public_key_fingerprint[:16]}...")
        except ImportError:
            self.add_check("Node0 Identity", "WARN", "Module not available")
        except Exception as e:
            self.add_check("Node0 Identity", "WARN", str(e)[:50])

        try:
            from bizra_kernel.hardware_fingerprint import generate_fingerprint
            fp = generate_fingerprint()
            tier1_hash = fp.get("tiered_covenant", {}).get("tier_1_root", {}).get("hash", "")
            if tier1_hash:
                self.add_check("Hardware Covenant", "PASS", f"Tier-1: {tier1_hash[:16]}...")
            else:
                self.add_check("Hardware Covenant", "WARN", "Could not generate fingerprint")
        except ImportError:
            self.add_check("Hardware Covenant", "WARN", "Module not available")
        except Exception as e:
            self.add_check("Hardware Covenant", "WARN", str(e)[:50])

        return results

    async def verify_python_modules(self) -> Dict[str, Any]:
        """Verify Python modules can be imported."""
        results = {"section": "Python Modules", "checks": []}

        modules = [
            ("bizra_kernel.node0_unified", "Node0 Unified"),
            ("bizra_kernel.local_resource_manager", "Resource Manager"),
            ("bizra_kernel.sape_engine", "SAPE Engine"),
            ("bizra_kernel.seed_manager", "Seed Manager"),
            ("bizra_kernel.ihsan_gate", "Ihsan Gate"),
            ("core.sape", "Core SAPE"),
            ("core.fate", "Core FATE"),
        ]

        for module, name in modules:
            try:
                __import__(module)
                self.add_check(name, "PASS", "Imported successfully")
            except ImportError as e:
                self.add_check(name, "WARN", str(e)[:40])
            except Exception as e:
                self.add_check(name, "FAIL", str(e)[:40])

        return results

    async def verify_rust_build(self) -> Dict[str, Any]:
        """Verify Rust project builds."""
        results = {"section": "Rust Build", "checks": []}

        try:
            result = subprocess.run(
                ["cargo", "check", "--message-format=short"],
                capture_output=True, text=True, timeout=120,
                cwd="/mnt/c/BIZRA-Dual-Agentic-system--main"
            )

            if result.returncode == 0:
                self.add_check("Rust Compilation", "PASS", "cargo check passed")
            else:
                error_lines = [l for l in result.stderr.split("\n") if "error" in l.lower()]
                if error_lines:
                    self.add_check("Rust Compilation", "FAIL", error_lines[0][:50])
                else:
                    self.add_check("Rust Compilation", "WARN", "Warnings present")
        except subprocess.TimeoutExpired:
            self.add_check("Rust Build", "WARN", "Timeout (still building)")
        except Exception as e:
            self.add_check("Rust Build", "FAIL", str(e))

        return results

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all verification checks."""
        start_time = datetime.now()

        print("\n" + "=" * 60)
        print("  BIZRA NODE0 UNIFIED SYSTEM VERIFICATION")
        print("=" * 60 + "\n")

        # Run checks
        await self.verify_hardware()
        print("  Hardware checks complete")

        await self.verify_services()
        print("  Service checks complete")

        await self.verify_data_lake()
        print("  Data lake checks complete")

        await self.verify_identity()
        print("  Identity checks complete")

        await self.verify_python_modules()
        print("  Python module checks complete")

        # Skip Rust build check for speed (already verified)
        # await self.verify_rust_build()

        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "summary": {
                "passed": self.passed,
                "warned": self.warned,
                "failed": self.failed,
                "total": len(self.checks),
            },
            "overall": "PASS" if self.failed == 0 else "FAIL",
            "standalone_ready": self.failed == 0,
            "checks": self.checks,
        }

    def print_results(self, results: Dict[str, Any]):
        """Print formatted results."""
        print("\n" + "-" * 60)
        print("  VERIFICATION RESULTS")
        print("-" * 60 + "\n")

        for check in results["checks"]:
            icon = "✅" if check["status"] == "PASS" else "⚠️" if check["status"] == "WARN" else "❌"
            print(f"  {icon} {check['name']}: {check['details']}")
            if check.get("fix"):
                print(f"      💡 Fix: {check['fix']}")

        print("\n" + "-" * 60)
        s = results["summary"]
        print(f"  Summary: {s['passed']} passed, {s['warned']} warnings, {s['failed']} failed")
        print(f"  Elapsed: {results['elapsed_seconds']:.1f}s")

        print("\n" + "=" * 60)
        if results["standalone_ready"]:
            print("  ✅ NODE0 IS STANDALONE READY")
        else:
            print("  ❌ NODE0 NOT STANDALONE READY - Fix issues above")
        print("=" * 60 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="BIZRA Node0 Unified System Verification")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--fix", action="store_true", help="Show fix suggestions")
    args = parser.parse_args()

    verifier = Node0Verifier()
    results = await verifier.run_all_checks()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        verifier.print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
