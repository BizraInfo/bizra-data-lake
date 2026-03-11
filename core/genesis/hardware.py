"""
BIZRA Hardware Scanner — Cross-Platform Hardware Detection
============================================================

Thin wrapper around platform detection for genesis bootstrap.
In production, this delegates to HardwareCovenant from
scripts/genesis_identity.py for full 3-tier fingerprinting.

For the genesis CLI, we provide a lightweight cross-platform
summary of CPU, GPU, RAM, and VRAM.

Standing on Giants:
- Intel (1968): Hardware identification standards
- NVIDIA (1999): GPU compute (CUDA identification)
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Detected hardware summary."""

    cpu: str = "Unknown"
    gpu: str = "Unknown"
    ram_gb: int = 0
    vram_gb: int = 0
    platform_name: str = ""
    os_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu": self.cpu,
            "gpu": self.gpu,
            "ram_gb": self.ram_gb,
            "vram_gb": self.vram_gb,
            "platform": self.platform_name,
            "os_version": self.os_version,
        }

    def format_summary(self) -> str:
        """Format hardware info for CLI display."""
        parts = []
        if self.ram_gb > 0:
            parts.append(f"{self.ram_gb}GB RAM")
        if self.vram_gb > 0:
            parts.append(f"{self.vram_gb}GB VRAM")
        if self.gpu and self.gpu != "Unknown":
            parts.append(self.gpu)
        return " + ".join(parts) if parts else "Hardware detected"


class HardwareScanner:
    """
    Cross-platform hardware detection.

    Provides a lightweight hardware summary for genesis bootstrap.
    For full 3-tier verification, use HardwareCovenant directly.
    """

    def scan(self) -> HardwareInfo:
        """Scan and return hardware information."""
        info = HardwareInfo(
            platform_name=platform.system(),
            os_version=platform.version(),
        )

        # CPU detection
        try:
            info.cpu = platform.processor() or platform.machine()
        except Exception:  # noqa: BLE001 — boundary boundary
            info.cpu = platform.machine()

        # RAM detection
        try:
            import psutil

            info.ram_gb = round(psutil.virtual_memory().total / (1024**3))
        except ImportError:
            # Fallback: try platform-specific detection
            info.ram_gb = self._detect_ram_fallback()

        # GPU detection
        info.gpu = self._detect_gpu()
        info.vram_gb = self._detect_vram()

        logger.info(
            "Hardware scan: %s, %dGB RAM, %s (%dGB VRAM)",
            info.cpu,
            info.ram_gb,
            info.gpu,
            info.vram_gb,
        )
        return info

    @staticmethod
    def _detect_ram_fallback() -> int:
        """Fallback RAM detection without psutil."""
        try:

            if platform.system() == "Linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            kb = int(line.split()[1])
                            return round(kb / (1024**2))
            elif platform.system() == "Windows":
                import subprocess

                result = subprocess.run(
                    ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                lines = [
                    l.strip()
                    for l in result.stdout.split("\n")
                    if l.strip() and not l.strip().startswith("Total")
                ]
                if lines:
                    return round(int(lines[0]) / (1024**3))
        except (OSError, subprocess.SubprocessError):  # SEC-003 — subprocess boundary
            pass
        return 0

    @staticmethod
    def _detect_gpu() -> str:
        """Detect GPU name."""
        try:
            import subprocess

            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "path", "win32_videocontroller", "get", "name"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                lines = [
                    l.strip()
                    for l in result.stdout.split("\n")
                    if l.strip() and l.strip() != "Name"
                ]
                return lines[0] if lines else "Unknown"
            elif platform.system() == "Linux":
                result = subprocess.run(
                    ["lspci"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                for line in result.stdout.split("\n"):
                    if "VGA" in line or "3D" in line:
                        return line.split(": ", 1)[-1].strip()
        except (OSError, subprocess.SubprocessError):  # SEC-003 — subprocess boundary
            pass
        return "Unknown"

    @staticmethod
    def _detect_vram() -> int:
        """Detect GPU VRAM in GB."""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                mb = int(result.stdout.strip().split("\n")[0])
                return round(mb / 1024)
        except (OSError, subprocess.SubprocessError):  # SEC-003 — subprocess boundary
            pass
        return 0
