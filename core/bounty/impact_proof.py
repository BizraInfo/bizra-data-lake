"""
Impact Proof — Cryptographic Proof of Vulnerability Discovery

An ImpactProof is the sealed evidence that a vulnerability exists
and has measurable impact, without revealing the exploit details.

Key properties:
- BLAKE3 hash of vulnerability (conceals details)
- ΔE (entropy delta) measured via UERS
- Reproduction steps as domain events
- Optional ZK-proof for trustless verification
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.bounty import SEVERITY_LEVELS
from core.proof_engine.canonical import blake3_digest, canonical_bytes
from core.proof_engine.receipt import SovereignSigner


class Severity(Enum):
    """Vulnerability severity levels."""

    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VulnCategory(Enum):
    """Vulnerability categories."""

    REENTRANCY = "reentrancy"
    FLASH_LOAN = "flash_loan"
    ORACLE_MANIPULATION = "oracle_manipulation"
    ACCESS_CONTROL = "access_control"
    INTEGER_OVERFLOW = "integer_overflow"
    LOGIC_ERROR = "logic_error"
    FRONT_RUNNING = "front_running"
    DENIAL_OF_SERVICE = "denial_of_service"
    UPGRADE_VULNERABILITY = "upgrade_vulnerability"
    SIGNATURE_MALLEABILITY = "signature_malleability"


@dataclass
class DomainEvent:
    """A single step in exploit reproduction."""

    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    gas_used: int = 0
    state_change: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "gas_used": self.gas_used,
            "state_change": self.state_change,
        }


@dataclass
class EntropyMeasurement:
    """UERS entropy measurement for impact quantification."""

    surface_entropy: float = 0.0
    structural_entropy: float = 0.0
    behavioral_entropy: float = 0.0
    hypothetical_entropy: float = 0.0
    contextual_entropy: float = 0.0

    @property
    def total_entropy(self) -> float:
        """Total entropy across all vectors."""
        return (
            self.surface_entropy
            + self.structural_entropy
            + self.behavioral_entropy
            + self.hypothetical_entropy
            + self.contextual_entropy
        )

    @property
    def average_entropy(self) -> float:
        """Average entropy per vector."""
        return self.total_entropy / 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface": self.surface_entropy,
            "structural": self.structural_entropy,
            "behavioral": self.behavioral_entropy,
            "hypothetical": self.hypothetical_entropy,
            "contextual": self.contextual_entropy,
            "total": self.total_entropy,
            "average": self.average_entropy,
        }


@dataclass
class ImpactProof:
    """
    Cryptographic proof of vulnerability impact.

    This is the sealed evidence submitted for bounty claims.
    """

    # Identity
    proof_id: str

    # Target
    target_address: str  # Contract address or system identifier
    target_chain: str = "ethereum"  # Chain/platform
    target_name: Optional[str] = None  # Protocol name

    # Vulnerability
    vuln_category: VulnCategory = VulnCategory.LOGIC_ERROR
    severity: Severity = Severity.MEDIUM
    title: str = ""
    description_hash: bytes = field(default_factory=bytes)  # Hash, not plaintext

    # Impact Measurement
    entropy_before: EntropyMeasurement = field(default_factory=EntropyMeasurement)
    entropy_after: EntropyMeasurement = field(default_factory=EntropyMeasurement)

    # Financial Impact (in USD equivalent)
    funds_at_risk: float = 0.0
    max_extractable_value: float = 0.0

    # Reproduction
    reproduction_steps: List[DomainEvent] = field(default_factory=list)
    reproduction_hash: bytes = field(default_factory=bytes)

    # Proof
    exploit_hash: bytes = field(default_factory=bytes)  # BLAKE3 of exploit code
    zk_proof: Optional[bytes] = None  # Optional ZK-SNARK

    # SNR Score (quality of the finding)
    snr_score: float = 0.0
    ihsan_score: float = 0.0  # Ethical validation

    # Signature
    signature: bytes = field(default_factory=bytes)
    hunter_pubkey: bytes = field(default_factory=bytes)

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    platform: str = "bizra"  # Source platform

    @property
    def delta_e(self) -> float:
        """
        Calculate entropy delta (ΔE).

        Positive ΔE = entropy reduced = value created.
        """
        return self.entropy_before.total_entropy - self.entropy_after.total_entropy

    @property
    def multiplier(self) -> int:
        """Get severity multiplier."""
        return SEVERITY_LEVELS.get(self.severity.value, {}).get("multiplier", 1)  # type: ignore[attr-defined]

    def body_bytes(self) -> bytes:
        """Get proof body for signing."""
        data = {
            "proof_id": self.proof_id,
            "target_address": self.target_address,
            "target_chain": self.target_chain,
            "vuln_category": self.vuln_category.value,
            "severity": self.severity.value,
            "description_hash": self.description_hash.hex(),
            "delta_e": self.delta_e,
            "funds_at_risk": self.funds_at_risk,
            "exploit_hash": self.exploit_hash.hex(),
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
            "timestamp": self.timestamp.isoformat(),
        }
        return canonical_bytes(data)

    def sign_with(self, signer: SovereignSigner) -> "ImpactProof":
        """Sign the proof."""
        body = self.body_bytes()
        self.signature = signer.sign(body)
        self.hunter_pubkey = signer.public_key_bytes()
        return self

    def verify_signature(self, signer: SovereignSigner) -> bool:
        """Verify proof signature."""
        body = self.body_bytes()
        return signer.verify(body, self.signature)

    def digest(self) -> bytes:
        """Compute proof digest."""
        return blake3_digest(self.body_bytes() + self.signature)

    def hex_digest(self) -> str:
        """Hex-encoded digest."""
        return self.digest().hex()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proof_id": self.proof_id,
            "target_address": self.target_address,
            "target_chain": self.target_chain,
            "target_name": self.target_name,
            "vuln_category": self.vuln_category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description_hash": self.description_hash.hex(),
            "entropy_before": self.entropy_before.to_dict(),
            "entropy_after": self.entropy_after.to_dict(),
            "delta_e": self.delta_e,
            "funds_at_risk": self.funds_at_risk,
            "max_extractable_value": self.max_extractable_value,
            "reproduction_steps": [s.to_dict() for s in self.reproduction_steps],
            "exploit_hash": self.exploit_hash.hex(),
            "has_zk_proof": self.zk_proof is not None,
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
            "signature": self.signature.hex(),
            "hunter_pubkey": self.hunter_pubkey.hex(),
            "timestamp": self.timestamp.isoformat(),
            "platform": self.platform,
            "proof_digest": self.hex_digest(),
        }


class ImpactProofBuilder:
    """
    Builder for creating impact proofs.

    Ensures all required measurements are captured.
    """

    def __init__(self, signer: SovereignSigner):
        self.signer = signer
        self._counter = 0

    def _next_id(self) -> str:
        """Generate next proof ID."""
        self._counter += 1
        return f"imp_{self._counter:08d}_{int(time.time() * 1000)}"

    def build(
        self,
        target_address: str,
        vuln_category: VulnCategory,
        severity: Severity,
        title: str,
        description: str,
        exploit_code: bytes,
        entropy_before: EntropyMeasurement,
        entropy_after: EntropyMeasurement,
        reproduction_steps: List[DomainEvent],
        funds_at_risk: float = 0.0,
        snr_score: float = 0.95,
        ihsan_score: float = 0.95,
        target_chain: str = "ethereum",
        target_name: Optional[str] = None,
        zk_proof: Optional[bytes] = None,
    ) -> ImpactProof:
        """
        Build a complete impact proof.

        Args:
            target_address: Contract/system address
            vuln_category: Type of vulnerability
            severity: Severity level
            title: Short vulnerability title
            description: Full description (will be hashed)
            exploit_code: Exploit code (will be hashed)
            entropy_before: UERS measurement before fix
            entropy_after: UERS measurement after fix
            reproduction_steps: Steps to reproduce
            funds_at_risk: USD value at risk
            snr_score: Signal quality score
            ihsan_score: Ethical validation score
            target_chain: Blockchain/platform
            target_name: Protocol name
            zk_proof: Optional ZK-SNARK proof

        Returns:
            Signed ImpactProof
        """
        proof = ImpactProof(
            proof_id=self._next_id(),
            target_address=target_address,
            target_chain=target_chain,
            target_name=target_name,
            vuln_category=vuln_category,
            severity=severity,
            title=title,
            description_hash=blake3_digest(description.encode()),
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            funds_at_risk=funds_at_risk,
            reproduction_steps=reproduction_steps,
            reproduction_hash=blake3_digest(
                canonical_bytes([s.to_dict() for s in reproduction_steps])
            ),
            exploit_hash=blake3_digest(exploit_code),
            zk_proof=zk_proof,
            snr_score=snr_score,
            ihsan_score=ihsan_score,
        )

        return proof.sign_with(self.signer)

    def from_scan_result(
        self,
        scan_result: Dict[str, Any],
        exploit_code: bytes,
        snr_score: float = 0.95,
        ihsan_score: float = 0.95,
    ) -> ImpactProof:
        """
        Build proof from automated scan result.

        Args:
            scan_result: Output from HunterAgent scan
            exploit_code: Generated exploit code
            snr_score: Signal quality
            ihsan_score: Ethical validation

        Returns:
            Signed ImpactProof
        """
        # Parse scan result
        vuln_category = VulnCategory(scan_result.get("category", "logic_error"))
        severity = Severity(scan_result.get("severity", "medium"))

        # Build entropy measurements
        entropy_before = EntropyMeasurement(
            surface_entropy=scan_result.get("entropy", {}).get("surface", 0.5),
            structural_entropy=scan_result.get("entropy", {}).get("structural", 0.5),
            behavioral_entropy=scan_result.get("entropy", {}).get("behavioral", 0.5),
            hypothetical_entropy=scan_result.get("entropy", {}).get(
                "hypothetical", 0.5
            ),
            contextual_entropy=scan_result.get("entropy", {}).get("contextual", 0.5),
        )

        # After fix, entropy should be lower
        entropy_after = EntropyMeasurement(
            surface_entropy=entropy_before.surface_entropy * 0.5,
            structural_entropy=entropy_before.structural_entropy * 0.5,
            behavioral_entropy=entropy_before.behavioral_entropy * 0.3,
            hypothetical_entropy=entropy_before.hypothetical_entropy * 0.4,
            contextual_entropy=entropy_before.contextual_entropy * 0.5,
        )

        # Build reproduction steps
        reproduction_steps = [
            DomainEvent(
                event_type=step.get("type", "transaction"),
                timestamp=datetime.now(timezone.utc),
                data=step.get("data", {}),
                gas_used=step.get("gas", 0),
            )
            for step in scan_result.get("reproduction", [])
        ]

        return self.build(
            target_address=scan_result.get("target", "0x0"),
            vuln_category=vuln_category,
            severity=severity,
            title=scan_result.get("title", "Vulnerability"),
            description=scan_result.get("description", ""),
            exploit_code=exploit_code,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            reproduction_steps=reproduction_steps,
            funds_at_risk=scan_result.get("funds_at_risk", 0.0),
            snr_score=snr_score,
            ihsan_score=ihsan_score,
            target_chain=scan_result.get("chain", "ethereum"),
            target_name=scan_result.get("protocol", None),
        )


class ImpactProofVerifier:
    """
    Verifier for impact proofs.

    Validates signatures, SNR, and ethical constraints.
    """

    def __init__(
        self,
        signer: SovereignSigner,
        snr_threshold: float = 0.90,
        ihsan_threshold: float = 0.95,
    ):
        self.signer = signer
        self.snr_threshold = snr_threshold
        self.ihsan_threshold = ihsan_threshold
        self._verified: List[str] = []
        self._rejected: List[str] = []

    def verify(self, proof: ImpactProof) -> Tuple[bool, Optional[str]]:
        """
        Verify an impact proof.

        Returns (valid, error_message).
        """
        # Verify signature
        if not proof.verify_signature(self.signer):
            self._rejected.append(proof.proof_id)
            return False, "Invalid signature"

        # Verify SNR threshold
        if proof.snr_score < self.snr_threshold:
            self._rejected.append(proof.proof_id)
            return (
                False,
                f"SNR below threshold: {proof.snr_score:.3f} < {self.snr_threshold}",
            )

        # Verify Ihsān threshold (ethical validation)
        if proof.ihsan_score < self.ihsan_threshold:
            self._rejected.append(proof.proof_id)
            return (
                False,
                f"Ihsān below threshold: {proof.ihsan_score:.3f} < {self.ihsan_threshold}",
            )

        # Verify positive delta_e (must reduce entropy)
        if proof.delta_e <= 0:
            self._rejected.append(proof.proof_id)
            return False, f"Non-positive entropy delta: {proof.delta_e:.3f}"

        # Verify exploit hash exists
        if not proof.exploit_hash:
            self._rejected.append(proof.proof_id)
            return False, "Missing exploit hash"

        self._verified.append(proof.proof_id)
        return True, None

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        total = len(self._verified) + len(self._rejected)
        return {
            "total_verified": len(self._verified),
            "total_rejected": len(self._rejected),
            "acceptance_rate": len(self._verified) / total if total > 0 else 0.0,
        }
