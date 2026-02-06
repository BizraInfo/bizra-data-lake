"""
BIZRA GENESIS IDENTITY (v2.0)
"The Seed That Proves Seeds Can Grow"

This module defines Node0's identity and authority model.
It implements the THREE-TIER HARDWARE COVENANT and DELEGATION rules.

⚠️ THIS FILE IS SCRIPTURE. MODIFY WITH EXTREME CARE.
"""

import hashlib
import json
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# AUTHORITY MODEL — LOCKED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GENESIS_AUTHORITY_MODE = "ORIGIN_ONLY"  # Node0 is sovereign by origin
DELEGATION_ALLOWED = True                # Authority CAN be delegated to future nodes
TRANSFER_ALLOWED = False                 # Authority can NEVER be transferred/copied

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════

GENESIS_FILE = Path(r"C:\BIZRA-DATA-LAKE\04_GOLD\genesis.json")
DELEGATION_LEDGER = Path(r"C:\BIZRA-DATA-LAKE\04_GOLD\delegation_ledger.jsonl")

# ═══════════════════════════════════════════════════════════════════════════════
# TIERED HARDWARE COVENANT
# ═══════════════════════════════════════════════════════════════════════════════

class HardwareCovenant:
    """
    Three-tier hardware fingerprinting system.
    
    Tier 1 (ROOT): CPU + GPU + Platform — STRICT, hard fail on mismatch
    Tier 2 (MUTABLE): RAM + Storage + MAC — WARN, require attestation on mismatch
    Tier 3 (CONTEXTUAL): BIOS + OS + WSL — LOG ONLY, informational
    """
    
    @staticmethod
    def _run_wmic(query: str) -> str:
        """Run WMIC query and return result."""
        try:
            result = subprocess.run(
                ["wmic"] + query.split(),
                capture_output=True, text=True, timeout=10
            )
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            return lines[1] if len(lines) > 1 else "UNKNOWN"
        except Exception as e:
            return f"ERROR:{e}"
    
    @staticmethod
    def _get_gpu_name() -> str:
        """Get GPU name via WMIC."""
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True, text=True, timeout=10
            )
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip() and l.strip() != "Name"]
            return lines[0] if lines else "UNKNOWN"
        except:
            return "UNKNOWN"
    
    @classmethod
    def collect_tier_1_root(cls) -> Dict:
        """
        TIER 1: Root identity — STRICT verification.
        Mismatch = HARD FAIL. Node0 is invalid.
        """
        cpu_id = cls._run_wmic("cpu get processorid")
        gpu_name = cls._get_gpu_name()
        platform_name = platform.node()
        
        # Create deterministic fingerprint
        root_string = f"{cpu_id}|{gpu_name}|{platform_name}"
        fingerprint = hashlib.sha256(root_string.encode()).hexdigest()[:32]
        
        return {
            "cpu_fingerprint": hashlib.sha256(cpu_id.encode()).hexdigest()[:16],
            "gpu_fingerprint": hashlib.sha256(gpu_name.encode()).hexdigest()[:16],
            "platform_signature": platform_name,
            "combined_root_hash": fingerprint,
            "strict": True
        }
    
    @classmethod
    def collect_tier_2_mutable(cls) -> Dict:
        """
        TIER 2: Mutable components — WARN on mismatch.
        These can change (RAM upgrade, new SSD, etc.)
        Mismatch = ATTESTATION REQUIRED, not failure.
        """
        ram = cls._run_wmic("memorychip get capacity")
        # Get first network adapter MAC
        try:
            result = subprocess.run(
                ["getmac", "/fo", "csv", "/nh"],
                capture_output=True, text=True, timeout=10
            )
            mac = result.stdout.split(',')[0].strip('"') if result.stdout else "UNKNOWN"
        except:
            mac = "UNKNOWN"
        
        return {
            "ram_signature": hashlib.sha256(ram.encode()).hexdigest()[:16],
            "mac_address": mac,
            "strict": False,
            "action_on_mismatch": "WARN_REQUIRE_ATTESTATION"
        }
    
    @classmethod
    def collect_tier_3_contextual(cls) -> Dict:
        """
        TIER 3: Contextual info — LOG ONLY.
        These change frequently (OS updates, BIOS updates, etc.)
        Mismatch = Informational logging, no action.
        """
        os_info = f"{platform.system()} {platform.release()} {platform.version()}"
        
        # Check WSL presence
        wsl_present = Path("/mnt/c").exists() if platform.system() == "Linux" else "N/A (Windows Host)"
        
        return {
            "os_fingerprint": hashlib.sha256(os_info.encode()).hexdigest()[:16],
            "os_description": os_info,
            "wsl_context": str(wsl_present),
            "python_version": platform.python_version(),
            "strict": False,
            "action_on_mismatch": "LOG_ONLY"
        }
    
    @classmethod
    def generate_full_covenant(cls) -> Dict:
        """Generate complete 3-tier hardware covenant."""
        return {
            "tier_1_root": cls.collect_tier_1_root(),
            "tier_2_mutable": cls.collect_tier_2_mutable(),
            "tier_3_contextual": cls.collect_tier_3_contextual(),
            "generated_at": datetime.now().isoformat(),
            "covenant_version": "2.0"
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GENESIS VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class GenesisVerificationResult:
    """Result of genesis verification."""
    def __init__(self):
        self.tier_1_valid = False
        self.tier_2_valid = False
        self.tier_3_valid = False
        self.tier_2_warnings = []
        self.tier_3_logs = []
        self.fatal_error = None
    
    @property
    def is_valid(self) -> bool:
        """Node0 is valid if Tier 1 passes."""
        return self.tier_1_valid and self.fatal_error is None
    
    def __str__(self):
        if self.fatal_error:
            return f"❌ GENESIS INVALID: {self.fatal_error}"
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return f"{status} | T1:{self.tier_1_valid} T2:{self.tier_2_valid} T3:{self.tier_3_valid}"


def verify_genesis_hardware() -> GenesisVerificationResult:
    """
    Verify current hardware against stored Genesis covenant.
    
    Returns:
        GenesisVerificationResult with tier-by-tier status
    """
    result = GenesisVerificationResult()
    
    if not GENESIS_FILE.exists():
        result.fatal_error = "Genesis file not found. Node0 not initialized."
        return result
    
    with open(GENESIS_FILE, 'r') as f:
        genesis = json.load(f)
    
    stored_covenant = genesis.get("hardware_covenant", {})
    current_covenant = HardwareCovenant.generate_full_covenant()
    
    # TIER 1: STRICT CHECK
    stored_t1 = stored_covenant.get("tier_1_root", {})
    current_t1 = current_covenant["tier_1_root"]
    
    if stored_t1.get("combined_root_hash") == current_t1["combined_root_hash"]:
        result.tier_1_valid = True
    else:
        result.tier_1_valid = False
        result.fatal_error = (
            f"TIER 1 MISMATCH — Node0 identity compromised. "
            f"Expected: {stored_t1.get('combined_root_hash', 'N/A')[:8]}... "
            f"Got: {current_t1['combined_root_hash'][:8]}..."
        )
        return result  # Hard fail, no point checking further
    
    # TIER 2: WARN CHECK
    stored_t2 = stored_covenant.get("tier_2_mutable", {})
    current_t2 = current_covenant["tier_2_mutable"]
    
    if stored_t2.get("ram_signature") != current_t2["ram_signature"]:
        result.tier_2_warnings.append("RAM signature changed — hardware upgrade detected")
    if stored_t2.get("mac_address") != current_t2["mac_address"]:
        result.tier_2_warnings.append("MAC address changed — network adapter modified")
    
    result.tier_2_valid = len(result.tier_2_warnings) == 0
    
    # TIER 3: LOG ONLY
    stored_t3 = stored_covenant.get("tier_3_contextual", {})
    current_t3 = current_covenant["tier_3_contextual"]
    
    if stored_t3.get("os_fingerprint") != current_t3["os_fingerprint"]:
        result.tier_3_logs.append(f"OS updated: {current_t3['os_description']}")
    
    result.tier_3_valid = len(result.tier_3_logs) == 0
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# DELEGATION (Not Transfer)
# ═══════════════════════════════════════════════════════════════════════════════

class DelegationError(Exception):
    """Raised when delegation rules are violated."""
    pass


def delegate_authority(target_node_id: str, permissions: list, attestation: str) -> Dict:
    """
    Delegate specific permissions to a child node.
    
    Node0 remains the ORIGIN. Authority is DELEGATED, never TRANSFERRED.
    The child node operates under Node0's signature.
    
    Args:
        target_node_id: Unique identifier of the child node
        permissions: List of permissions to delegate (e.g., ["read_lake", "mint_poi"])
        attestation: Human-readable reason for delegation
    
    Returns:
        Delegation certificate
    """
    if not DELEGATION_ALLOWED:
        raise DelegationError("Delegation is disabled in this Node0 configuration")
    
    if not GENESIS_FILE.exists():
        raise DelegationError("Cannot delegate: Genesis not initialized")
    
    # Load genesis to get Node0's signature
    with open(GENESIS_FILE, 'r') as f:
        genesis = json.load(f)
    
    node0_hash = genesis.get("hardware_covenant", {}).get("tier_1_root", {}).get("combined_root_hash")
    
    delegation = {
        "type": "DELEGATION",
        "origin_node": "NODE0",
        "origin_hash": node0_hash[:16],
        "target_node": target_node_id,
        "permissions": permissions,
        "attestation": attestation,
        "delegated_at": datetime.now().isoformat(),
        "revocable": True,
        "transfer_prohibited": True  # This delegation cannot be re-delegated
    }
    
    # Sign the delegation
    delegation_string = json.dumps(delegation, sort_keys=True)
    delegation["signature"] = hashlib.sha256(delegation_string.encode()).hexdigest()
    
    # Append to delegation ledger
    with open(DELEGATION_LEDGER, 'a') as f:
        f.write(json.dumps(delegation) + "\n")
    
    return delegation


# ═══════════════════════════════════════════════════════════════════════════════
# GENESIS INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_genesis(author: str = "MoMo", force: bool = False) -> Dict:
    """
    Initialize Node0's Genesis identity.
    
    ⚠️ THIS CAN ONLY BE DONE ONCE unless force=True.
    Genesis is scripture — written once, never overwritten.
    """
    if GENESIS_FILE.exists() and not force:
        raise RuntimeError(
            "Genesis already exists. Node0 is already initialized. "
            "Set force=True only if you understand this will REPLACE the original genesis."
        )
    
    hardware_covenant = HardwareCovenant.generate_full_covenant()
    
    genesis = {
        "node_id": "NODE0",
        "node_type": "GENESIS_ORIGIN",
        "author": author,
        "created_at": datetime.now().isoformat(),
        "hardware_covenant": hardware_covenant,
        "authority": {
            "mode": GENESIS_AUTHORITY_MODE,
            "delegation_allowed": DELEGATION_ALLOWED,
            "transfer_allowed": TRANSFER_ALLOWED
        },
        "philosophy": {
            "name": "BIZRA",
            "meaning": "بذرة — Seed in Arabic",
            "principle": "Every human is a node. Every node is a seed. Every seed has infinite potential.",
            "mission": "Make good deeds profitable. Weaponize greed against itself."
        },
        "covenant_statement": (
            "Node0 is not immortal hardware. "
            "It is the first witness. "
            "The proof of origin. "
            "The seed that proves seeds can grow."
        )
    }
    
    # Write genesis
    GENESIS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GENESIS_FILE, 'w') as f:
        json.dump(genesis, f, indent=2)
    
    print(f"🌱 GENESIS INITIALIZED")
    print(f"   Node ID: {genesis['node_id']}")
    print(f"   Author: {genesis['author']}")
    print(f"   Root Hash: {hardware_covenant['tier_1_root']['combined_root_hash'][:16]}...")
    print(f"   Saved to: {GENESIS_FILE}")
    
    return genesis


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python genesis_identity.py [init|verify|show]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "init":
        initialize_genesis(author="MoMo")
    
    elif command == "verify":
        result = verify_genesis_hardware()
        print(f"\n{result}")
        if result.tier_2_warnings:
            print("  ⚠️ Tier 2 Warnings:")
            for w in result.tier_2_warnings:
                print(f"     - {w}")
        if result.tier_3_logs:
            print("  📝 Tier 3 Logs:")
            for l in result.tier_3_logs:
                print(f"     - {l}")
        sys.exit(0 if result.is_valid else 1)
    
    elif command == "show":
        if GENESIS_FILE.exists():
            with open(GENESIS_FILE, 'r') as f:
                print(json.dumps(json.load(f), indent=2))
        else:
            print("Genesis not initialized.")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
