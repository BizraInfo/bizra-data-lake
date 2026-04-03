"""
bizra_kernel/hardware_fingerprint.py
=================================
Tiered Hardware Covenant fingerprinting for Node0.

Tier 1 (Root): CPU + GPU + Platform (hard fail)
Tier 2 (Mutable): RAM + MAC + Hostname (attestation required)
Tier 3 (Contextual): OS + WSL context (log only)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import uuid
from typing import Any, Dict, Optional

try:
    import psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False


def _canonical_json(data: Dict[str, Any]) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _hash_components(components: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(components)).hexdigest()


def _get_cpu_info() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "UNKNOWN_CPU"


def _get_gpu_info() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN_GPU"


def _get_mac_address() -> str:
    try:
        mac_num = uuid.getnode()
        return ":".join(("%012X" % mac_num)[i : i + 2] for i in range(0, 12, 2))
    except Exception:
        return "UNKNOWN_MAC"


def _get_ram_signature() -> str:
    if PSUTIL_AVAILABLE:
        try:
            total_ram = round(psutil.virtual_memory().total / (1024**3))
            return f"{total_ram}GB"
        except Exception:
            pass
    return "UNKNOWN_RAM"


def _detect_wsl_context() -> str:
    if os.getenv("WSL_DISTRO_NAME"):
        return "WSL"
    release = platform.release().lower()
    if "microsoft" in release or "wsl" in release:
        return "WSL"
    return "NATIVE"


def generate_fingerprint() -> Dict[str, Any]:
    """Generate tiered hardware fingerprint with hashes and components."""
    tier_1_components = {
        "cpu_fingerprint": _get_cpu_info(),
        "gpu_fingerprint": _get_gpu_info(),
        "platform_signature": platform.node() or socket.gethostname(),
    }
    tier_2_components = {
        "ram_signature": _get_ram_signature(),
        "mac_address": _get_mac_address(),
        "hostname": socket.gethostname(),
    }
    tier_3_components = {
        "os_fingerprint": f"{platform.system()} {platform.release()}",
        "wsl_context": _detect_wsl_context(),
    }

    tier_1_hash = _hash_components(tier_1_components)
    tier_2_hash = _hash_components(tier_2_components)
    tier_3_hash = _hash_components(tier_3_components)

    tiered_covenant = {
        "tier_1_root": {
            "hash": tier_1_hash,
            "components": tier_1_components,
        },
        "tier_2_mutable": {
            "hash": tier_2_hash,
            "components": tier_2_components,
        },
        "tier_3_contextual": {
            "hash": tier_3_hash,
            "components": tier_3_components,
        },
    }

    return {
        "fingerprint": tier_1_hash,
        "tiered_covenant": tiered_covenant,
    }


def verify_fingerprint(expected_fingerprint: str, previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Verify current fingerprint against expected and optional previous state."""
    current = generate_fingerprint()
    tier_results: Dict[str, Dict[str, Any]] = {}

    # Tier 1 — hard fail if mismatch
    tier_1_match = current["fingerprint"] == expected_fingerprint
    tier_results["tier_1_root"] = {
        "match": tier_1_match,
        "action": "PASS" if tier_1_match else "HARD_FAIL",
        "expected": expected_fingerprint,
        "current": current["fingerprint"],
    }

    # Tier 2 — warning if mismatch vs previous
    if previous:
        prev_t2 = previous.get("tiered_covenant", {}).get("tier_2_mutable", {}).get("hash")
        cur_t2 = current["tiered_covenant"]["tier_2_mutable"]["hash"]
        tier_results["tier_2_mutable"] = {
            "match": prev_t2 == cur_t2,
            "action": "WARN" if prev_t2 and prev_t2 != cur_t2 else "PASS",
            "previous": prev_t2,
            "current": cur_t2,
        }
    else:
        tier_results["tier_2_mutable"] = {
            "match": True,
            "action": "NOT_CHECKED",
        }

    # Tier 3 — informational only
    tier_results["tier_3_contextual"] = {
        "match": True,
        "action": "LOG_ONLY",
    }

    return {
        "verified": tier_1_match,
        "expected": expected_fingerprint,
        "current": current,
        "tier_results": tier_results,
    }
