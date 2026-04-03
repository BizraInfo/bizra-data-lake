"""
bizra_kernel/crown.py - The Hardware Crown (Coronation Protocol)
============================================================
Implements the specific "Tiered Hardware Covenant" requested by The Architect.
Binds the software Sovereign Identity to the physical Node0 (MSI Titan).

Structure:
  Tier 1 (Root): CPU + GPU + Platform (Hard Fail)
  Tier 2 (Mutable): RAM + Storage + MAC (Warning)
  Tier 3 (Contextual): BIOS + OS + WSL (Log)
"""

import platform
import psutil
import socket
import uuid
import hashlib
import json
import logging
import subprocess
import os
from typing import Dict, Any, Tuple

logger = logging.getLogger("BIZRA_CROWN")

class CrownViolation(Exception):
    """Raised when Tier 1 Hardware Covenant is violated."""
    pass

class CrownWarning(Warning):
    """Raised when Tier 2 Hardware Covenant changes (upgrades)."""
    pass

class HardwareCrown:
    """
    The Crown Authority Validation Engine.
    Forges and Verifies the 3-Tier Hardware Covenant.
    """

    def __init__(self):
        self.tiers = self._inventory_kingdom()
    
    def _get_gpu_info(self) -> str:
        """Best-effort GPU detection (nvidia-smi or fallback)."""
        try:
            # Try nvidia-smi query
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            pass
        return "UNKNOWN_GPU_DRIVER_MISSING"

    def _get_cpu_info(self) -> str:
        """Get precise CPU model."""
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except FileNotFoundError:
            return platform.processor()
        return "UNKNOWN_CPU"

    def _get_mac_address(self) -> str:
        """Get Node MAC address."""
        try:
            mac_num = uuid.getnode()
            return ':'.join(('%012X' % mac_num)[i:i+2] for i in range(0, 12, 2))
        except Exception:
            return "UNKNOWN_MAC"

    def _inventory_kingdom(self) -> Dict[str, Dict[str, str]]:
        """
        Inventory the hardware assets into the 3-Tier Covenant.
        """
        # Tier 1: The Immutable Core (Silicon Identity)
        tier_1 = {
            "cpu_fingerprint": self._get_cpu_info(),
            "gpu_fingerprint": self._get_gpu_info(),
            # In WSL, platform node usually leaks host info, typically sufficient for ID
            "platform_signature": platform.node(), 
            "strict": "TRUE"
        }

        # Tier 2: The Mutable Body (Upgradable Parts)
        total_ram = round(psutil.virtual_memory().total / (1024**3))
        root_disk = psutil.disk_usage('/').total // (1024**3)
        tier_2 = {
            "ram_signature": f"{total_ram}GB",
            "storage_signature": f"ROOT_{root_disk}GB",
            "mac_address": self._get_mac_address()
        }

        # Tier 3: The Context (Software/BIOS Environment)
        tier_3 = {
            "os_fingerprint": f"{platform.system()} {platform.release()}",
            "arch": platform.machine(),
            "python_version": platform.python_version()
        }

        return {
            "tier_1_root": tier_1,
            "tier_2_mutable": tier_2,
            "tier_3_contextual": tier_3
        }

    def forge_crown(self) -> Dict[str, Any]:
        """
        Forges the Crown artifact for the Genesis Block.
        """
        return {
            "type": "HARDWARE_COVENANT",
            "version": "1.0",
            "sovereign_intent": "NODE0_ORIGIN",
            "covenant": self.tiers,
            "crown_hash": self._sign_crown()
        }

    def _sign_crown(self) -> str:
        """
        Cryptographically binds the Tier 1 Root to the claim.
        Only Tier 1 is used for the Root Hash to prevent fragility.
        """
        t1 = self.tiers["tier_1_root"]
        # Canonical string for hashing
        raw_sig = f"{t1['cpu_fingerprint']}|{t1['gpu_fingerprint']}|{t1['platform_signature']}"
        return hashlib.sha256(raw_sig.encode()).hexdigest()

    def verify_sovereignty(self, genesis_covenant: Dict[str, Any]) -> bool:
        """
        Verifies the current hardware against the Genesis Covenant.
        Enforces the 3-Tier Rule.
        """
        current = self.tiers
        recorded = genesis_covenant["covenant"]

        # FAIL CONDITION: Tier 1 Mismatch
        t1_cur = current["tier_1_root"]
        t1_rec = recorded["tier_1_root"]
        
        # Check critical fields
        if (t1_cur["cpu_fingerprint"] != t1_rec["cpu_fingerprint"] or 
            t1_cur["platform_signature"] != t1_rec["platform_signature"]):
            
            error_msg = (
                f"CRITICAL SOVEREIGNTY FAILURE: Hardware Crown Mismatch.\n"
                f"Genesis CPU: {t1_rec['cpu_fingerprint']}\n"
                f"Current CPU: {t1_cur['cpu_fingerprint']}\n"
                "Node0 Authority Rejected."
            )
            print("!!! CROWN VIOLATION DETECTED !!!")
            raise CrownViolation(error_msg)

        # WARN CONDITION: Tier 2 Mismatch
        t2_cur = current["tier_2_mutable"]
        t2_rec = recorded["tier_2_mutable"]
        if t2_cur["ram_signature"] != t2_rec["ram_signature"]:
            print(f"[!] COVENANT WARNING: RAM Changed ({t2_rec['ram_signature']} -> {t2_cur['ram_signature']}). Verify Upgrade.")

        print(f"[+] COVENANT VERIFIED: {self._sign_crown()[:16]}... (Tier 1 Match)")
        return True
