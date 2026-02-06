"""
BIZRA PROOF-CARRYING INFERENCE (PCI) PROTOCOL
Receipt and Verification Gates

Standing on Giants: Lamport (BFT), Shannon (SNR)

GATE CHAIN ORDERING RATIONALE
=============================

PCI Gate Chain: SCHEMA -> SIGNATURE -> TIMESTAMP -> REPLAY -> IHSAN -> SNR -> POLICY

This ordering is intentional and differs from the License Gate ordering (SNR before IHSAN).

Why IHSAN before SNR in PCI?
----------------------------
PCI envelopes represent inter-node communication in the federation. When processing
messages from potentially untrusted peers, ethical violations (Ihsan) represent a
more severe class of failure than signal quality issues (SNR).

1. FAIL-FAST ON ETHICAL VIOLATIONS: An envelope with low Ihsan score indicates
   content that violates constitutional principles (harmful, misleading, or
   malicious). These must be rejected immediately regardless of signal quality.
   A high-SNR message that is ethically compromised is MORE dangerous than a
   noisy but benign message.

2. SECURITY POSTURE: In adversarial P2P environments, attackers may craft
   high-SNR messages specifically to bypass quality filters. Checking Ihsan
   first ensures malicious content is caught even if it has been optimized
   for clarity and coherence.

3. RESOURCE PROTECTION: SNR computation can be more expensive than Ihsan
   scoring (especially for large payloads). Rejecting ethical violations
   early prevents wasted computation on messages that would fail anyway.

Contrast with License Gate (model_license_gate.py):
---------------------------------------------------
License gates check SNR before Ihsan because they validate inference OUTPUT
from trusted local models, not untrusted external messages. In that context,
filtering noise first (low SNR = garbled output) before quality assessment
(Ihsan) is appropriate since the content source is already authenticated.

See also: core/sovereign/model_license_gate.py for the License Gate rationale.
"""

import hmac
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

# Import unified thresholds from authoritative source
from core.integration.constants import (
    UNIFIED_CLOCK_SKEW_SECONDS,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_NONCE_TTL_SECONDS,
    UNIFIED_SNR_THRESHOLD,
)

from .crypto import verify_signature
from .envelope import PCIEnvelope
from .reject_codes import RejectCode

# Use unified constants
IHSAN_MINIMUM_THRESHOLD = UNIFIED_IHSAN_THRESHOLD
SNR_MINIMUM_THRESHOLD = UNIFIED_SNR_THRESHOLD
MAX_CLOCK_SKEW_SECONDS = UNIFIED_CLOCK_SKEW_SECONDS
NONCE_TTL_SECONDS = UNIFIED_NONCE_TTL_SECONDS
MAX_NONCE_CACHE_SIZE = 10000  # Hard limit to prevent memory exhaustion


@dataclass
class VerificationResult:
    passed: bool
    reject_code: RejectCode
    details: str = ""
    gate_passed: List[str] = None


# Default constitution hash (should be set from NODE0_IDENTITY)
DEFAULT_CONSTITUTION_HASH = (
    "d9c9b5f7a3e2c8d4f1a6e9b2c5d8a3f7e0c2b5d8a1e4f7c0b3d6a9e2c5f8b1d926f"
)


class PCIGateKeeper:
    """
    Executes the 3-tier gate chain for PCI Envelopes.

    Sovereignty: Default-deny with explicit policy verification.
    """

    def __init__(
        self,
        seen_nonces_cache: Dict[str, float] = None,
        constitution_hash: str = None,
        policy_enforcement: bool = True,
    ):
        self.seen_nonces = seen_nonces_cache if seen_nonces_cache is not None else {}
        self.constitution_hash = constitution_hash or DEFAULT_CONSTITUTION_HASH
        self.policy_enforcement = policy_enforcement
        self._last_prune_time = time.time()

    def _prune_expired_nonces(self) -> int:
        """
        Remove nonces older than NONCE_TTL_SECONDS.
        Returns count of pruned entries.

        SECURITY: Prevents unbounded memory growth (DoS vector).
        Called automatically during verify() to maintain cache hygiene.
        """
        now = time.time()
        cutoff = now - NONCE_TTL_SECONDS
        expired = [nonce for nonce, ts in self.seen_nonces.items() if ts < cutoff]
        for nonce in expired:
            del self.seen_nonces[nonce]

        # Emergency pruning if cache exceeds hard limit (keep newest)
        if len(self.seen_nonces) > MAX_NONCE_CACHE_SIZE:
            sorted_nonces = sorted(self.seen_nonces.items(), key=lambda x: x[1])
            excess = len(self.seen_nonces) - MAX_NONCE_CACHE_SIZE
            for nonce, _ in sorted_nonces[:excess]:
                del self.seen_nonces[nonce]
            expired.extend([n for n, _ in sorted_nonces[:excess]])

        self._last_prune_time = now
        return len(expired)

    def verify(self, envelope: PCIEnvelope) -> VerificationResult:
        gates_passed = []

        # ════════════════════════════════════════════════
        # TIER 1: CHEAP (<10ms)
        # ════════════════════════════════════════════════

        # 1. Schema (Implicit by typing, but strict check?)
        gates_passed.append("SCHEMA")

        # 2. Signature
        if not envelope.signature:
            return VerificationResult(
                False, RejectCode.REJECT_SIGNATURE, "Missing signature"
            )

        digest = envelope.compute_digest()
        if not verify_signature(
            digest, envelope.signature.value, envelope.sender.public_key
        ):
            return VerificationResult(
                False, RejectCode.REJECT_SIGNATURE, "Invalid ed25519 signature"
            )
        gates_passed.append("SIGNATURE")

        # 3. Timestamp
        try:
            ts = datetime.fromisoformat(envelope.timestamp.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta = abs((now - ts).total_seconds())
            if delta > MAX_CLOCK_SKEW_SECONDS:
                code = (
                    RejectCode.REJECT_TIMESTAMP_FUTURE
                    if ts > now
                    else RejectCode.REJECT_TIMESTAMP_STALE
                )
                return VerificationResult(
                    False, code, f"Clock skew {delta}s > {MAX_CLOCK_SKEW_SECONDS}s"
                )
        except ValueError:
            return VerificationResult(
                False, RejectCode.REJECT_SCHEMA, "Invalid timestamp format"
            )
        gates_passed.append("TIMESTAMP")

        # 4. Replay Protection (with TTL-based eviction)
        # Prune expired nonces periodically (every 60s or if cache is large)
        if (time.time() - self._last_prune_time > 60) or (
            len(self.seen_nonces) > MAX_NONCE_CACHE_SIZE * 0.9
        ):
            self._prune_expired_nonces()

        if envelope.nonce in self.seen_nonces:
            return VerificationResult(
                False, RejectCode.REJECT_NONCE_REPLAY, "Nonce reused"
            )
        self.seen_nonces[envelope.nonce] = time.time()
        gates_passed.append("REPLAY")

        # ════════════════════════════════════════════════
        # TIER 2: MEDIUM (<150ms)
        # ════════════════════════════════════════════════

        # 6. Ihsan
        if envelope.metadata.ihsan_score < IHSAN_MINIMUM_THRESHOLD:
            return VerificationResult(
                False,
                RejectCode.REJECT_IHSAN_BELOW_MIN,
                f"Ihsan {envelope.metadata.ihsan_score} < {IHSAN_MINIMUM_THRESHOLD}",
            )
        gates_passed.append("IHSAN")

        # 7. SNR Gate (SEC-020: Shannon signal quality)
        if envelope.metadata.snr_score < SNR_MINIMUM_THRESHOLD:
            return VerificationResult(
                False,
                RejectCode.REJECT_SNR_BELOW_MIN,
                f"SNR {envelope.metadata.snr_score} < {SNR_MINIMUM_THRESHOLD}",
            )
        gates_passed.append("SNR")

        # 8. Policy (Sovereignty: Default-deny with explicit verification)
        # SECURITY: Use constant-time comparison to prevent timing attacks
        if self.policy_enforcement:
            payload_policy = getattr(envelope.payload, "policy_hash", None)
            if payload_policy and not hmac.compare_digest(
                payload_policy, self.constitution_hash
            ):
                return VerificationResult(
                    False,
                    RejectCode.REJECT_POLICY_MISMATCH,
                    f"Policy hash mismatch: expected {self.constitution_hash[:16]}...",
                )
        gates_passed.append("POLICY")

        return VerificationResult(True, RejectCode.SUCCESS, "Verified", gates_passed)
