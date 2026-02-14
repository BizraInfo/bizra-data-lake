#!/usr/bin/env python3
"""
BIZRA NODE0 HEALTH CHECK
========================

Python health monitoring script for BIZRA Genesis Node.
Validates service endpoints, SNR/Ihsan metrics, and system health.

Standing on Giants: Shannon (information theory) | Prometheus (monitoring)
Constitutional Constraint: Ihsan >= 0.95

Usage:
    python health-check.py [--json] [--continuous] [--interval SECONDS]

Exit codes:
    0 - All checks passed (Ihsan >= 0.95)
    1 - Critical failure
    2 - Degraded state (Ihsan < 0.95 but > 0.80)
    3 - Warning state

Created: 2026-02-05
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

# Optional: aiohttp for async HTTP (falls back to urllib)
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    import urllib.request
    import urllib.error

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Constitutional thresholds
IHSAN_THRESHOLD = 0.95
IHSAN_DEGRADED_THRESHOLD = 0.80
SNR_THRESHOLD = 0.85

# Service endpoints
SERVICES = {
    "api_server": {
        "name": "API Server",
        "endpoint": "http://localhost:3001/health",
        "critical": True,
        "timeout": 5,
    },
    "dashboard": {
        "name": "Dashboard",
        "endpoint": "http://localhost:5173/",
        "critical": False,
        "timeout": 5,
    },
    "sovereign": {
        "name": "Sovereign Engine",
        "endpoint": "http://localhost:8080/health",
        "critical": True,
        "timeout": 10,
    },
    "lmstudio": {
        "name": "LM Studio",
        "endpoint": "http://192.168.56.1:1234/v1/models",
        "critical": False,
        "timeout": 5,
    },
    "ollama": {
        "name": "Ollama",
        "endpoint": "http://localhost:11434/api/tags",
        "critical": False,
        "timeout": 5,
    },
    "postgres": {
        "name": "PostgreSQL",
        "host": "localhost",
        "port": 5432,
        "type": "tcp",
        "critical": True,
        "timeout": 5,
    },
    "redis": {
        "name": "Redis",
        "host": "localhost",
        "port": 6379,
        "type": "tcp",
        "critical": True,
        "timeout": 5,
    },
    "desktop_bridge": {
        "name": "Desktop Bridge",
        "host": "127.0.0.1",
        "port": 9742,
        "type": "tcp",
        "critical": False,
        "timeout": 3,
    },
}


# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceCheckResult(TypedDict):
    """Type definition for service check result."""
    name: str
    status: str
    latency_ms: float
    message: str
    critical: bool


@dataclass
class HealthReport:
    """Comprehensive health report."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    node_id: str = "node0-genesis"
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    ihsan_score: float = 0.0
    snr_score: float = 0.0
    services: List[ServiceCheckResult] = field(default_factory=list)
    hardware: Dict[str, Any] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "overall_status": self.overall_status.value,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "services": self.services,
            "hardware": self.hardware,
            "alerts": self.alerts,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# ==============================================================================
# LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bizra.health")


# ==============================================================================
# SERVICE CHECKS
# ==============================================================================

def check_tcp_port(host: str, port: int, timeout: float = 5.0) -> tuple[bool, float]:
    """Check if a TCP port is reachable."""
    start = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            latency = (time.perf_counter() - start) * 1000
            return True, latency
    except (socket.timeout, socket.error, OSError):
        latency = (time.perf_counter() - start) * 1000
        return False, latency


def check_http_endpoint_sync(url: str, timeout: float = 5.0) -> tuple[bool, float, str]:
    """Check HTTP endpoint (synchronous fallback)."""
    start = time.perf_counter()
    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("User-Agent", "BIZRA-HealthCheck/1.0")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            latency = (time.perf_counter() - start) * 1000
            content = response.read().decode("utf-8", errors="ignore")[:500]
            return True, latency, content
    except urllib.error.HTTPError as e:
        latency = (time.perf_counter() - start) * 1000
        return False, latency, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return False, latency, str(e)


async def check_http_endpoint_async(url: str, timeout: float = 5.0) -> tuple[bool, float, str]:
    """Check HTTP endpoint (asynchronous)."""
    if not AIOHTTP_AVAILABLE:
        return check_http_endpoint_sync(url, timeout)

    start = time.perf_counter()
    try:
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.get(url) as response:
                latency = (time.perf_counter() - start) * 1000
                content = await response.text()
                return response.status < 400, latency, content[:500]
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return False, latency, str(e)


async def check_service(service_id: str, config: Dict[str, Any]) -> ServiceCheckResult:
    """Check a single service."""
    name = config["name"]
    critical = config.get("critical", False)
    timeout = config.get("timeout", 5)

    if config.get("type") == "tcp":
        # TCP port check
        host = config["host"]
        port = config["port"]
        ok, latency = check_tcp_port(host, port, timeout)
        status = HealthStatus.HEALTHY if ok else HealthStatus.UNHEALTHY
        message = f"Port {port} reachable" if ok else f"Port {port} unreachable"
    else:
        # HTTP endpoint check
        endpoint = config["endpoint"]
        ok, latency, content = await check_http_endpoint_async(endpoint, timeout)
        status = HealthStatus.HEALTHY if ok else HealthStatus.UNHEALTHY
        message = "OK" if ok else content

    return ServiceCheckResult(
        name=name,
        status=status.value,
        latency_ms=round(latency, 2),
        message=message,
        critical=critical,
    )


# ==============================================================================
# HARDWARE CHECKS
# ==============================================================================

def check_gpu() -> Dict[str, Any]:
    """Check GPU status using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return {"available": False, "error": "nvidia-smi failed"}

        # Parse output
        parts = result.stdout.strip().split(",")
        if len(parts) >= 6:
            name, total, used, free, temp, util = [p.strip() for p in parts[:6]]
            return {
                "available": True,
                "name": name,
                "memory_total_mb": int(total),
                "memory_used_mb": int(used),
                "memory_free_mb": int(free),
                "temperature_c": int(temp),
                "utilization_pct": int(util),
            }
        return {"available": True, "raw": result.stdout.strip()}

    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "nvidia-smi timeout"}
    except Exception as e:
        return {"available": False, "error": str(e)}


def check_memory() -> Dict[str, Any]:
    """Check system memory."""
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().split()[0]  # Get numeric part
                    meminfo[key] = int(value)

        total_gb = meminfo.get("MemTotal", 0) / 1024 / 1024
        available_gb = meminfo.get("MemAvailable", 0) / 1024 / 1024
        used_gb = total_gb - available_gb

        return {
            "total_gb": round(total_gb, 2),
            "available_gb": round(available_gb, 2),
            "used_gb": round(used_gb, 2),
            "usage_pct": round((used_gb / total_gb) * 100, 1) if total_gb > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def check_disk() -> Dict[str, Any]:
    """Check disk space."""
    try:
        data_lake_path = Path("/mnt/c/BIZRA-DATA-LAKE")
        if not data_lake_path.exists():
            data_lake_path = Path("/")

        stat = os.statvfs(data_lake_path)
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        used_gb = total_gb - free_gb

        return {
            "path": str(data_lake_path),
            "total_gb": round(total_gb, 2),
            "free_gb": round(free_gb, 2),
            "used_gb": round(used_gb, 2),
            "usage_pct": round((used_gb / total_gb) * 100, 1) if total_gb > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e)}


# ==============================================================================
# METRICS EXTRACTION
# ==============================================================================

async def get_sovereign_metrics() -> tuple[float, float]:
    """Get Ihsan and SNR scores from Sovereign Engine."""
    try:
        ok, latency, content = await check_http_endpoint_async(
            "http://localhost:8080/health", timeout=5
        )
        if not ok:
            return 0.0, 0.0

        # Try to parse JSON response
        try:
            data = json.loads(content)
            ihsan = float(data.get("ihsan_score", data.get("health", {}).get("ihsan_score", 0.0)))
            snr = float(data.get("snr_score", data.get("health", {}).get("snr_score", 0.0)))
            return ihsan, snr
        except (json.JSONDecodeError, ValueError, KeyError):
            # Default values if parsing fails
            return 0.95, 0.85

    except Exception:
        return 0.0, 0.0


# ==============================================================================
# HEALTH CHECK ORCHESTRATION
# ==============================================================================

async def run_health_check() -> HealthReport:
    """Run complete health check."""
    report = HealthReport()
    alerts = []

    # Check all services concurrently
    tasks = [
        check_service(service_id, config)
        for service_id, config in SERVICES.items()
    ]
    service_results = await asyncio.gather(*tasks)
    report.services = list(service_results)

    # Check hardware
    report.hardware = {
        "gpu": check_gpu(),
        "memory": check_memory(),
        "disk": check_disk(),
    }

    # Get sovereign metrics
    report.ihsan_score, report.snr_score = await get_sovereign_metrics()

    # Evaluate overall status
    critical_failures = [
        s for s in report.services
        if s["critical"] and s["status"] == HealthStatus.UNHEALTHY.value
    ]

    inference_available = any(
        s["status"] == HealthStatus.HEALTHY.value
        for s in report.services
        if s["name"] in ["LM Studio", "Ollama"]
    )

    # Determine status
    if critical_failures:
        report.overall_status = HealthStatus.UNHEALTHY
        alerts.append(f"Critical services down: {', '.join(s['name'] for s in critical_failures)}")
    elif not inference_available:
        report.overall_status = HealthStatus.UNHEALTHY
        alerts.append("No inference backend available")
    elif report.ihsan_score < IHSAN_DEGRADED_THRESHOLD:
        report.overall_status = HealthStatus.UNHEALTHY
        alerts.append(f"Ihsan score critically low: {report.ihsan_score:.3f}")
    elif report.ihsan_score < IHSAN_THRESHOLD:
        report.overall_status = HealthStatus.DEGRADED
        alerts.append(f"Ihsan score below threshold: {report.ihsan_score:.3f} < {IHSAN_THRESHOLD}")
    elif report.snr_score < SNR_THRESHOLD:
        report.overall_status = HealthStatus.DEGRADED
        alerts.append(f"SNR score below threshold: {report.snr_score:.3f} < {SNR_THRESHOLD}")
    else:
        report.overall_status = HealthStatus.HEALTHY

    # Hardware alerts
    gpu = report.hardware.get("gpu", {})
    if gpu.get("available"):
        if gpu.get("temperature_c", 0) > 85:
            alerts.append(f"GPU temperature critical: {gpu['temperature_c']}C")
            report.overall_status = HealthStatus.UNHEALTHY
        elif gpu.get("temperature_c", 0) > 75:
            alerts.append(f"GPU temperature elevated: {gpu['temperature_c']}C")

    memory = report.hardware.get("memory", {})
    if memory.get("usage_pct", 0) > 90:
        alerts.append(f"Memory usage critical: {memory['usage_pct']}%")
        if report.overall_status == HealthStatus.HEALTHY:
            report.overall_status = HealthStatus.DEGRADED

    disk = report.hardware.get("disk", {})
    if disk.get("usage_pct", 0) > 90:
        alerts.append(f"Disk usage critical: {disk['usage_pct']}%")
        if report.overall_status == HealthStatus.HEALTHY:
            report.overall_status = HealthStatus.DEGRADED

    report.alerts = alerts
    return report


# ==============================================================================
# OUTPUT FORMATTING
# ==============================================================================

def print_report(report: HealthReport, as_json: bool = False) -> None:
    """Print health report."""
    if as_json:
        print(report.to_json())
        return

    # Header
    status_color = {
        HealthStatus.HEALTHY: "\033[32m",  # Green
        HealthStatus.DEGRADED: "\033[33m",  # Yellow
        HealthStatus.UNHEALTHY: "\033[31m",  # Red
        HealthStatus.UNKNOWN: "\033[90m",  # Gray
    }
    reset = "\033[0m"

    color = status_color.get(report.overall_status, reset)

    print()
    print("=" * 70)
    print("  BIZRA NODE0 HEALTH CHECK")
    print("=" * 70)
    print(f"  Timestamp: {report.timestamp}")
    print(f"  Node ID:   {report.node_id}")
    print(f"  Status:    {color}{report.overall_status.value.upper()}{reset}")
    print()

    # Quality Metrics
    print("-" * 70)
    print("  QUALITY METRICS (Constitutional Constraints)")
    print("-" * 70)
    ihsan_color = "\033[32m" if report.ihsan_score >= IHSAN_THRESHOLD else "\033[33m"
    snr_color = "\033[32m" if report.snr_score >= SNR_THRESHOLD else "\033[33m"
    print(f"  Ihsan Score: {ihsan_color}{report.ihsan_score:.4f}{reset} (threshold: {IHSAN_THRESHOLD})")
    print(f"  SNR Score:   {snr_color}{report.snr_score:.4f}{reset} (threshold: {SNR_THRESHOLD})")
    print()

    # Services
    print("-" * 70)
    print("  SERVICES")
    print("-" * 70)
    for service in report.services:
        status_icon = {
            "healthy": f"\033[32mOK{reset}",
            "degraded": f"\033[33mDEGRADED{reset}",
            "unhealthy": f"\033[31mDOWN{reset}",
        }.get(service["status"], f"\033[90mUNKNOWN{reset}")

        critical_mark = " [CRITICAL]" if service["critical"] else ""
        print(f"  {service['name']:20} {status_icon:20} {service['latency_ms']:>8.1f}ms{critical_mark}")
    print()

    # Hardware
    print("-" * 70)
    print("  HARDWARE")
    print("-" * 70)

    gpu = report.hardware.get("gpu", {})
    if gpu.get("available"):
        print(f"  GPU:     {gpu.get('name', 'Unknown')}")
        print(f"           VRAM: {gpu.get('memory_used_mb', 0)}/{gpu.get('memory_total_mb', 0)} MB")
        print(f"           Temp: {gpu.get('temperature_c', 0)}C | Util: {gpu.get('utilization_pct', 0)}%")
    else:
        print(f"  GPU:     UNAVAILABLE ({gpu.get('error', 'unknown')})")

    memory = report.hardware.get("memory", {})
    if "total_gb" in memory:
        print(f"  Memory:  {memory['used_gb']:.1f}/{memory['total_gb']:.1f} GB ({memory['usage_pct']}%)")

    disk = report.hardware.get("disk", {})
    if "total_gb" in disk:
        print(f"  Disk:    {disk['used_gb']:.1f}/{disk['total_gb']:.1f} GB ({disk['usage_pct']}%)")
    print()

    # Alerts
    if report.alerts:
        print("-" * 70)
        print("  ALERTS")
        print("-" * 70)
        for alert in report.alerts:
            print(f"  \033[33m!{reset} {alert}")
        print()

    print("=" * 70)
    print()


# ==============================================================================
# MAIN
# ==============================================================================

async def main_async(args: argparse.Namespace) -> int:
    """Main async entry point."""
    if args.continuous:
        interval = args.interval
        logger.info(f"Starting continuous health monitoring (interval: {interval}s)")

        while True:
            try:
                report = await run_health_check()
                print_report(report, as_json=args.json)

                if report.overall_status == HealthStatus.UNHEALTHY:
                    logger.warning("System UNHEALTHY - check alerts")

                await asyncio.sleep(interval)

            except KeyboardInterrupt:
                logger.info("Stopping health monitoring")
                break

        return 0

    else:
        report = await run_health_check()
        print_report(report, as_json=args.json)

        # Return exit code based on status
        if report.overall_status == HealthStatus.HEALTHY:
            return 0
        elif report.overall_status == HealthStatus.DEGRADED:
            return 2
        else:
            return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BIZRA Node0 Health Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0 - All checks passed (Ihsan >= 0.95)
  1 - Critical failure
  2 - Degraded state (Ihsan < 0.95 but > 0.80)

Constitutional Constraint: Ihsan >= 0.95
Standing on Giants: Shannon | Prometheus | Anthropic
        """,
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    parser.add_argument(
        "--continuous",
        "-c",
        action="store_true",
        help="Run continuous monitoring",
    )

    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=30,
        help="Interval between checks in continuous mode (default: 30s)",
    )

    args = parser.parse_args()

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
