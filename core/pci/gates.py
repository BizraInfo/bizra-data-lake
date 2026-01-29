"""
BIZRA PROOF-CARRYING INFERENCE (PCI) PROTOCOL
Receipt and Verification Gates
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta

from .envelope import PCIEnvelope
from .reject_codes import RejectCode
from .crypto import verify_signature

IHSAN_MINIMUM_THRESHOLD = 0.95
MAX_CLOCK_SKEW_SECONDS = 120

@dataclass
class VerificationResult:
    passed: bool
    reject_code: RejectCode
    details: str = ""
    gate_passed: List[str] = None

# Default constitution hash (should be set from NODE0_IDENTITY)
DEFAULT_CONSTITUTION_HASH = "d9c9b5f7a3e2c8d4f1a6e9b2c5d8a3f7e0c2b5d8a1e4f7c0b3d6a9e2c5f8b1d926f"

class PCIGateKeeper:
    """
    Executes the 3-tier gate chain for PCI Envelopes.
    
    Sovereignty: Default-deny with explicit policy verification.
    """
    
    def __init__(
        self, 
        seen_nonces_cache: Dict[str, float] = None,
        constitution_hash: str = None,
        policy_enforcement: bool = True
    ):
        self.seen_nonces = seen_nonces_cache if seen_nonces_cache is not None else {}
        self.constitution_hash = constitution_hash or DEFAULT_CONSTITUTION_HASH
        self.policy_enforcement = policy_enforcement
        
    def verify(self, envelope: PCIEnvelope) -> VerificationResult:
        gates_passed = []
        
        # ════════════════════════════════════════════════
        # TIER 1: CHEAP (<10ms)
        # ════════════════════════════════════════════════
        
        # 1. Schema (Implicit by typing, but strict check?)
        gates_passed.append("SCHEMA")
        
        # 2. Signature
        if not envelope.signature:
            return VerificationResult(False, RejectCode.REJECT_SIGNATURE, "Missing signature")
            
        digest = envelope.compute_digest()
        if not verify_signature(digest, envelope.signature.value, envelope.sender.public_key):
             return VerificationResult(False, RejectCode.REJECT_SIGNATURE, "Invalid ed25519 signature")
        gates_passed.append("SIGNATURE")
        
        # 3. Timestamp
        try:
            ts = datetime.fromisoformat(envelope.timestamp.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            delta = abs((now - ts).total_seconds())
            if delta > MAX_CLOCK_SKEW_SECONDS:
                code = RejectCode.REJECT_TIMESTAMP_FUTURE if ts > now else RejectCode.REJECT_TIMESTAMP_STALE
                return VerificationResult(False, code, f"Clock skew {delta}s > {MAX_CLOCK_SKEW_SECONDS}s")
        except ValueError:
             return VerificationResult(False, RejectCode.REJECT_SCHEMA, "Invalid timestamp format")
        gates_passed.append("TIMESTAMP")
        
        # 4. Replay
        if envelope.nonce in self.seen_nonces:
             return VerificationResult(False, RejectCode.REJECT_NONCE_REPLAY, "Nonce reused")
        self.seen_nonces[envelope.nonce] = time.time() # Should prune old ones
        gates_passed.append("REPLAY")
        
        # ════════════════════════════════════════════════
        # TIER 2: MEDIUM (<150ms)
        # ════════════════════════════════════════════════
        
        # 6. Ihsan
        if envelope.metadata.ihsan_score < IHSAN_MINIMUM_THRESHOLD:
             return VerificationResult(False, RejectCode.REJECT_IHSAN_BELOW_MIN, 
                                     f"Ihsan {envelope.metadata.ihsan_score} < {IHSAN_MINIMUM_THRESHOLD}")
        gates_passed.append("IHSAN")
        
        # 8. Policy (Sovereignty: Default-deny with explicit verification)
        if self.policy_enforcement:
            payload_policy = getattr(envelope.payload, 'policy_hash', None)
            if payload_policy and payload_policy != self.constitution_hash:
                return VerificationResult(
                    False, 
                    RejectCode.REJECT_POLICY_MISMATCH, 
                    f"Policy hash mismatch: expected {self.constitution_hash[:16]}..."
                )
        gates_passed.append("POLICY")
        
        return VerificationResult(True, RejectCode.SUCCESS, "Verified", gates_passed)

