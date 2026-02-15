"""
Hardware Scanner — Cross-Platform Hardware Fingerprinting
==========================================================
Thin wrapper around platform-native hardware detection.
Produces a summary suitable for genesis identity and URP pledging.

Standing on Giants:
- Shannon (1948): Hardware entropy as identity signal
- Wiener (1948): Machine identity in cybernetic systems
"""

from __future__ import annotations

import hashlib
import platform
from typing import Any, Dict


class HardwareScanner:
    """
    Cross-platform hardware scanner for genesis identity.

    Produces cpu, gpu, ram, platform info without requiring
    external dependencies (WMIC, lshw, etc. are optional).
    """

    def scan(self) -> Dict[str, Any]:
        """
        Scan hardware and return summary dict.

        Returns:
            Dict with cpu, gpu, ram, platform, fingerprint keys.
        """
        cpu = platform.processor() or "unknown"
        machine = platform.machine() or "unknown"
        system = platform.system() or "unknown"
        node = platform.node() or "unknown"

        # GPU detection is platform-specific; stub for cross-platform
        gpu = self._detect_gpu()

        # RAM detection via platform-agnostic method
        ram_gb = self._detect_ram()

        # Deterministic fingerprint
        fp_input = f"{cpu}|{machine}|{system}|{node}"
        fingerprint = hashlib.sha256(fp_input.encode()).hexdigest()[:32]

        return {
            "cpu": cpu,
            "machine": machine,
            "gpu": gpu,
            "ram_gb": ram_gb,
            "platform": system,
            "node": node,
            "fingerprint": fingerprint,
        }

    def _detect_gpu(self) -> str:
        """Detect GPU name. Returns 'unknown' if not detectable."""
        try:
            import subprocess

            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "path", "win32_videocontroller", "get", "name"],
                    capture_output=True, text=True, timeout=10,
                )
                lines = [
                    line.strip()
                    for line in result.stdout.strip().split("\n")
                    if line.strip() and line.strip() != "Name"
                ]
                return lines[0] if lines else "unknown"
            elif platform.system() == "Linux":
                result = subprocess.run(
                    ["lspci"], capture_output=True, text=True, timeout=10,
                )
                for line in result.stdout.split("\n"):
                    if "VGA" in line or "3D" in line:
                        return line.split(":")[-1].strip()
        except Exception:
            pass
        return "unknown"

    def _detect_ram(self) -> float:
        """Detect RAM in GB. Returns 0.0 if not detectable."""
        try:
            import os

            if hasattr(os, "sysconf"):
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                if pages > 0 and page_size > 0:
                    return round(pages * page_size / (1024**3), 1)
        except Exception:
            pass

        try:
            import subprocess

            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "computersystem", "get", "totalphysicalmemory"],
                    capture_output=True, text=True, timeout=10,
                )
                lines = [
                    line.strip()
                    for line in result.stdout.strip().split("\n")
                    if line.strip() and not line.strip().startswith("Total")
                ]
                if lines:
                    return round(int(lines[0]) / (1024**3), 1)
        except Exception:
            pass

        return 0.0

    def format_summary(self, info: Dict[str, Any] | None = None) -> str:
        """Format hardware info for CLI display."""
        if info is None:
            info = self.scan()
        lines = [
            f"  CPU:        {info.get('cpu', 'unknown')}",
            f"  GPU:        {info.get('gpu', 'unknown')}",
            f"  RAM:        {info.get('ram_gb', 0.0)} GB",
            f"  Platform:   {info.get('platform', 'unknown')} ({info.get('machine', '')})",
            f"  Fingerprint: {info.get('fingerprint', 'unknown')[:16]}...",
        ]
        return "\n".join(lines)
