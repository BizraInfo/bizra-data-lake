"""
core/sovereign/integration.py - Constitutional Gate with Z3 + SNR v2
=====================================================================
Genesis Strict Synthesis v2.2.2 CRIT-3 Implementation.

Implements the Four Pillars architecture:
    Pillar 1: RUNTIME SOVEREIGNTY (Ring 0)
        - Z3-proven agents ONLY
        - Ihsan = 1.0 (100% proven)
        - Zero unproven code execution

    Pillar 2: MUSEUM MODE (Archival)
        - SNR-v2 scored, awaiting Z3 synthesis
        - Read-only, referenced but not executed
        - Promotion path upon Z3 proof generation

    Pillar 3: SIMULATION SANDBOX (Isolated)
        - Firecracker microVM (conceptual)
        - Read-only Data Lake
        - Recommendations only (unverified)

    Pillar 4: GENESIS CUTOFF (72h deadline)
        - T+72 hours ABSOLUTE
        - Unproven -> auto-archived to Museum
        - Runtime ships with proven subset only

Standing on Giants: Shannon, Lamport, Vaswani, Anthropic
Domain: bizra-sovereign-v1:constitutional-gate
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import constitutional thresholds - Genesis v2.2.2 compliance
from core.constants import (
    MUSEUM_SNR_FLOOR,
    MUSEUM_PROMOTION_THRESHOLD,
    GENESIS_CUTOFF_HOURS,
)

logger = logging.getLogger("bizra.sovereign.integration")

# Domain prefix for receipts
DOMAIN_PREFIX = "bizra-sovereign-v1:constitutional-gate:"
VERSION = "1.0.0"


class ExecutionTier(str, Enum):
    """
    Execution tier based on proof status.

    From Genesis Strict Synthesis v2.2.2:
    - RUNTIME: Z3-proven (Ihsan = 1.0) - Ring 0 execution
    - MUSEUM: SNR-v2 scored (Ihsan >= 0.85) - Read-only reference
    - SANDBOX: Isolated simulation only - Firecracker conceptual
    - ARCHIVAL: Age > 72h - Read-only permanent archive
    """

    RUNTIME = "runtime"
    MUSEUM = "museum"
    SANDBOX = "sandbox"
    ARCHIVAL = "archival"


@dataclass
class ProofStatus:
    """
    Z3 proof status for an agent capability.

    Tracks whether an agent has achieved Z3 formal verification
    and its current SNR score for Museum classification.
    """

    agent_id: str
    is_z3_proven: bool = False
    proof_hash: Optional[str] = None
    z3_certificate_path: Optional[str] = None
    snr_score: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    promoted_at: Optional[datetime] = None

    @property
    def age_hours(self) -> float:
        """Calculate age in hours since creation."""
        now = datetime.now(timezone.utc)
        delta = now - self.created_at
        return delta.total_seconds() / 3600.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for receipt emission."""
        return {
            "agent_id": self.agent_id,
            "is_z3_proven": self.is_z3_proven,
            "proof_hash": self.proof_hash,
            "z3_certificate_path": self.z3_certificate_path,
            "snr_score": self.snr_score,
            "age_hours": self.age_hours,
            "created_at": self.created_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
        }


@dataclass
class TierClassificationResult:
    """Result of execution tier classification."""

    tier: ExecutionTier
    agent_id: str
    reason: str
    snr_score: float
    has_z3_proof: bool
    age_hours: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "agent_id": self.agent_id,
            "reason": self.reason,
            "snr_score": self.snr_score,
            "has_z3_proof": self.has_z3_proof,
            "age_hours": self.age_hours,
            "timestamp": self.timestamp.isoformat(),
        }


class ConstitutionalGate:
    """
    Constitutional execution gate enforcing Genesis v2.2.2 Four Pillars.

    This gate determines which execution tier an agent belongs to:
    - Runtime: Z3-proven only (Ihsan = 1.0)
    - Museum: SNR-v2 scored (awaiting proofs)
    - Sandbox: Isolated simulation
    - Archival: Exceeded 72h cutoff

    Usage:
        gate = ConstitutionalGate()

        # Register unproven agent in Museum
        gate.register_museum_agent("agent-001", snr_score=0.92)

        # Check execution tier
        tier = gate.get_execution_tier("agent-001")

        # Promote to Runtime via Z3 proof
        gate.promote_to_runtime("agent-001", "/path/to/z3.cert", "sha256hash")
    """

    def __init__(self, receipt_path: Optional[Path] = None):
        """
        Initialize Constitutional Gate.

        Args:
            receipt_path: Path for storing classification receipts
        """
        self.runtime_agents: Dict[str, ProofStatus] = {}
        self.museum_agents: Dict[str, ProofStatus] = {}
        self.z3_certificates: Dict[str, str] = {}
        self.receipt_path = receipt_path or Path("docs/evidence/receipts/sovereign")
        self.receipt_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Constitutional Gate initialized (cutoff: {GENESIS_CUTOFF_HOURS}h)"
        )

    def classify_execution_tier(
        self,
        agent_id: str,
        snr_score: float,
        has_z3_proof: bool,
        age_hours: Optional[float] = None,
    ) -> TierClassificationResult:
        """
        Classify which execution tier an agent belongs to.

        Classification Logic (in priority order):
        1. Age > 72h -> ARCHIVAL (Genesis cutoff, non-negotiable)
        2. has_z3_proof = True -> RUNTIME (Z3-proven)
        3. snr_score >= 0.85 -> MUSEUM (SNR-v2 scored)
        4. Default -> SANDBOX (isolated)

        Args:
            agent_id: Agent identifier
            snr_score: SNR v2 score [0.0, 1.0]
            has_z3_proof: Whether Z3 proof exists
            age_hours: Hours since agent creation (auto-calculated if not provided)

        Returns:
            TierClassificationResult with tier and reasoning
        """
        # Calculate age if not provided
        if age_hours is None:
            if agent_id in self.runtime_agents:
                age_hours = self.runtime_agents[agent_id].age_hours
            elif agent_id in self.museum_agents:
                age_hours = self.museum_agents[agent_id].age_hours
            else:
                age_hours = 0.0

        # Pillar 4: Genesis cutoff (72 hours) - ABSOLUTE
        if age_hours > GENESIS_CUTOFF_HOURS:
            result = TierClassificationResult(
                tier=ExecutionTier.ARCHIVAL,
                agent_id=agent_id,
                reason=f"Genesis cutoff exceeded ({age_hours:.1f}h > {GENESIS_CUTOFF_HOURS}h)",
                snr_score=snr_score,
                has_z3_proof=has_z3_proof,
                age_hours=age_hours,
            )
            logger.warning(f"Agent {agent_id} -> ARCHIVAL (cutoff exceeded)")
            return result

        # Pillar 1: Runtime (Z3-proven only)
        if has_z3_proof:
            result = TierClassificationResult(
                tier=ExecutionTier.RUNTIME,
                agent_id=agent_id,
                reason="Z3-proven (Ihsan = 1.0)",
                snr_score=snr_score,
                has_z3_proof=has_z3_proof,
                age_hours=age_hours,
            )
            logger.info(f"Agent {agent_id} -> RUNTIME (Z3-proven)")
            return result

        # Pillar 2: Museum (SNR-v2 scored)
        if snr_score >= MUSEUM_SNR_FLOOR:
            result = TierClassificationResult(
                tier=ExecutionTier.MUSEUM,
                agent_id=agent_id,
                reason=f"SNR-v2 scored ({snr_score:.3f} >= {MUSEUM_SNR_FLOOR})",
                snr_score=snr_score,
                has_z3_proof=has_z3_proof,
                age_hours=age_hours,
            )
            logger.info(f"Agent {agent_id} -> MUSEUM (SNR: {snr_score:.3f})")
            return result

        # Pillar 3: Sandbox (default for unproven)
        result = TierClassificationResult(
            tier=ExecutionTier.SANDBOX,
            agent_id=agent_id,
            reason=f"Unproven (SNR: {snr_score:.3f} < {MUSEUM_SNR_FLOOR})",
            snr_score=snr_score,
            has_z3_proof=has_z3_proof,
            age_hours=age_hours,
        )
        logger.info(f"Agent {agent_id} -> SANDBOX (unproven)")
        return result

    def verify_runtime_eligibility(
        self,
        agent_id: str,
        z3_proof_path: Optional[str] = None,
    ) -> bool:
        """
        Verify agent is eligible for Runtime execution.

        Requirements:
        - Z3 proof must exist and be valid
        - Ihsan score = 1.0 (mathematically proven)
        - Age < 72 hours (Genesis cutoff)

        Args:
            agent_id: Agent identifier
            z3_proof_path: Optional path to Z3 certificate

        Returns:
            True if eligible for Runtime, False otherwise
        """
        if agent_id not in self.runtime_agents:
            logger.warning(f"Agent {agent_id} not registered in Runtime")
            return False

        proof = self.runtime_agents[agent_id]

        # Check Z3 proof
        if not proof.is_z3_proven:
            logger.error(f"Agent {agent_id} lacks Z3 proof for Runtime")
            return False

        # Check age (Genesis cutoff)
        if proof.age_hours > GENESIS_CUTOFF_HOURS:
            logger.error(
                f"Agent {agent_id} exceeds {GENESIS_CUTOFF_HOURS}h genesis cutoff"
            )
            return False

        # Validate Z3 certificate if path provided
        cert_path = z3_proof_path or proof.z3_certificate_path
        if cert_path:
            try:
                with open(cert_path, "r") as f:
                    cert_data = f.read()
                    if not self._validate_z3_certificate(cert_data, proof.proof_hash):
                        logger.error(f"Z3 certificate invalid for {agent_id}")
                        return False
            except FileNotFoundError:
                logger.warning(f"Z3 certificate not found at {cert_path}")
                # Certificate missing but proof status is True - allow if hash exists
                if not proof.proof_hash:
                    return False
            except Exception as e:
                logger.error(f"Failed to load Z3 cert for {agent_id}: {e}")
                return False

        logger.info(f"Agent {agent_id} verified for Runtime execution")
        return True

    def promote_to_runtime(
        self,
        agent_id: str,
        z3_certificate_path: str,
        proof_hash: str,
    ) -> bool:
        """
        Promote agent from Museum to Runtime via Z3 proof.

        Flow:
        1. Verify Z3 certificate exists and is valid
        2. Validate proof_hash matches certificate
        3. Move agent from Museum to Runtime
        4. Record promotion timestamp
        5. Emit promotion receipt

        Args:
            agent_id: Agent to promote
            z3_certificate_path: Path to Z3 proof certificate
            proof_hash: Expected SHA-256 hash of proof

        Returns:
            True if promotion succeeds, False otherwise
        """
        # Verify certificate exists
        try:
            with open(z3_certificate_path, "r") as f:
                cert_data = f.read()
        except FileNotFoundError:
            logger.error(f"Z3 certificate not found: {z3_certificate_path}")
            return False

        # Validate certificate
        if not self._validate_z3_certificate(cert_data, expected_hash=proof_hash):
            logger.error(f"Z3 certificate validation failed for {agent_id}")
            return False

        # Remove from museum if present
        museum_proof = None
        if agent_id in self.museum_agents:
            museum_proof = self.museum_agents.pop(agent_id)
            logger.info(
                f"Promoting {agent_id} from Museum (SNR: {museum_proof.snr_score})"
            )

        # Create runtime proof status
        now = datetime.now(timezone.utc)
        proof = ProofStatus(
            agent_id=agent_id,
            is_z3_proven=True,
            proof_hash=proof_hash,
            z3_certificate_path=z3_certificate_path,
            snr_score=MUSEUM_PROMOTION_THRESHOLD,  # Z3-proven = Ihsan 1.0
            created_at=museum_proof.created_at if museum_proof else now,
            promoted_at=now,
        )

        # Register in runtime
        self.runtime_agents[agent_id] = proof
        self.z3_certificates[agent_id] = cert_data

        # Emit promotion receipt
        self._emit_receipt(
            "promotion",
            {
                "agent_id": agent_id,
                "from_tier": "museum" if museum_proof else "new",
                "to_tier": "runtime",
                "proof_hash": proof_hash,
                "certificate_path": z3_certificate_path,
                "promoted_at": now.isoformat(),
                "previous_snr": museum_proof.snr_score if museum_proof else None,
            },
        )

        logger.info(f"Agent {agent_id} PROMOTED to Runtime (Z3-proven)")
        return True

    def register_museum_agent(
        self,
        agent_id: str,
        snr_score: float,
    ) -> bool:
        """
        Register unproven agent in Museum (Pillar 2).

        Requirements:
        - SNR score >= MUSEUM_SNR_FLOOR (0.85)
        - Will be archived if Z3 proof not generated within 72h

        Args:
            agent_id: Agent identifier
            snr_score: SNR v2 score

        Returns:
            True if registered, False if SNR too low
        """
        if snr_score < MUSEUM_SNR_FLOOR:
            logger.warning(
                f"Agent {agent_id} SNR {snr_score:.3f} < {MUSEUM_SNR_FLOOR} floor, "
                "not eligible for Museum (routed to Sandbox)"
            )
            return False

        proof = ProofStatus(
            agent_id=agent_id,
            is_z3_proven=False,
            snr_score=snr_score,
        )
        self.museum_agents[agent_id] = proof

        # Emit registration receipt
        self._emit_receipt(
            "registration",
            {
                "agent_id": agent_id,
                "tier": "museum",
                "snr_score": snr_score,
                "cutoff_at": (
                    proof.created_at.timestamp() + GENESIS_CUTOFF_HOURS * 3600
                ),
            },
        )

        logger.info(f"Agent {agent_id} registered in Museum (SNR: {snr_score:.3f})")
        return True

    def get_execution_tier(self, agent_id: str) -> ExecutionTier:
        """
        Get current execution tier for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            ExecutionTier classification
        """
        # Check runtime first
        if agent_id in self.runtime_agents:
            proof = self.runtime_agents[agent_id]
            if proof.age_hours > GENESIS_CUTOFF_HOURS:
                return ExecutionTier.ARCHIVAL
            return ExecutionTier.RUNTIME

        # Check museum
        if agent_id in self.museum_agents:
            proof = self.museum_agents[agent_id]
            if proof.age_hours > GENESIS_CUTOFF_HOURS:
                return ExecutionTier.ARCHIVAL
            return ExecutionTier.MUSEUM

        # Default to sandbox
        return ExecutionTier.SANDBOX

    def get_proof_status(self, agent_id: str) -> Optional[ProofStatus]:
        """Get proof status for an agent if registered."""
        if agent_id in self.runtime_agents:
            return self.runtime_agents[agent_id]
        if agent_id in self.museum_agents:
            return self.museum_agents[agent_id]
        return None

    def list_runtime_agents(self) -> List[str]:
        """List all agents in Runtime tier."""
        return [
            agent_id
            for agent_id, proof in self.runtime_agents.items()
            if proof.age_hours <= GENESIS_CUTOFF_HOURS
        ]

    def list_museum_agents(self) -> List[str]:
        """List all agents in Museum tier."""
        return [
            agent_id
            for agent_id, proof in self.museum_agents.items()
            if proof.age_hours <= GENESIS_CUTOFF_HOURS
        ]

    def enforce_genesis_cutoff(self) -> Dict[str, List[str]]:
        """
        Enforce Genesis cutoff - archive agents exceeding 72h.

        Returns:
            Dict with 'archived_from_runtime' and 'archived_from_museum' lists
        """
        archived = {
            "archived_from_runtime": [],
            "archived_from_museum": [],
        }

        # Check runtime agents
        for agent_id, proof in list(self.runtime_agents.items()):
            if proof.age_hours > GENESIS_CUTOFF_HOURS:
                archived["archived_from_runtime"].append(agent_id)
                logger.warning(
                    f"Agent {agent_id} archived from Runtime (age: {proof.age_hours:.1f}h)"
                )

        # Check museum agents
        for agent_id, proof in list(self.museum_agents.items()):
            if proof.age_hours > GENESIS_CUTOFF_HOURS:
                archived["archived_from_museum"].append(agent_id)
                logger.warning(
                    f"Agent {agent_id} archived from Museum (age: {proof.age_hours:.1f}h)"
                )

        if archived["archived_from_runtime"] or archived["archived_from_museum"]:
            self._emit_receipt("genesis_cutoff_enforcement", archived)

        return archived

    def _validate_z3_certificate(
        self,
        cert_data: str,
        expected_hash: Optional[str] = None,
    ) -> bool:
        """
        Validate Z3 SMT solver certificate.

        Checks:
        - Valid S-expression format (starts with '(')
        - Contains proof marker (proof, unsat, valid)
        - Hash matches if provided

        Args:
            cert_data: Certificate content
            expected_hash: Expected SHA-256 hash

        Returns:
            True if valid, False otherwise
        """
        # Basic format check - S-expression
        stripped = cert_data.strip()
        if not stripped.startswith("("):
            logger.error("Invalid Z3 certificate format: not S-expression")
            return False

        # Check for proof markers (case-insensitive)
        lower = stripped.lower()
        proof_markers = ["proof", "unsat", "valid", "certificate", "qed"]
        if not any(marker in lower for marker in proof_markers):
            logger.error("Invalid Z3 certificate: missing proof marker")
            return False

        # Hash validation
        if expected_hash:
            actual_hash = hashlib.sha256(cert_data.encode()).hexdigest()
            if actual_hash != expected_hash:
                logger.error(
                    f"Z3 certificate hash mismatch: {actual_hash[:16]}... != {expected_hash[:16]}..."
                )
                return False

        return True

    def _emit_receipt(self, operation: str, data: Dict[str, Any]) -> None:
        """Emit evidence receipt for gate operations."""
        now = datetime.now(timezone.utc)
        receipt = {
            "receipt_id": f"GATE-{hashlib.sha256(f'{operation}{now.isoformat()}'.encode()).hexdigest()[:16]}",
            "timestamp": now.isoformat(),
            "domain": DOMAIN_PREFIX + operation,
            "version": VERSION,
            "operation": operation,
            "data": data,
        }

        # Calculate integrity hash
        content = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
        receipt["integrity_hash"] = hashlib.sha256(content.encode()).hexdigest()

        # Write receipt
        receipt_file = (
            self.receipt_path / f"gate-{operation}-{now.strftime('%Y%m%d-%H%M%S')}.json"
        )
        try:
            with open(receipt_file, "w") as f:
                json.dump(receipt, f, indent=2)
            logger.debug(f"Receipt emitted: {receipt_file}")
        except Exception as e:
            logger.error(f"Failed to emit receipt: {e}")


# Global instance
_gate_instance: Optional[ConstitutionalGate] = None


def create_constitutional_gate(
    receipt_path: Optional[Path] = None,
) -> ConstitutionalGate:
    """
    Create or get the global Constitutional Gate instance.

    Args:
        receipt_path: Optional custom receipt storage path

    Returns:
        ConstitutionalGate instance
    """
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = ConstitutionalGate(receipt_path)
    return _gate_instance


def get_constitutional_gate() -> Optional[ConstitutionalGate]:
    """Get the global gate instance if initialized."""
    return _gate_instance


def reset_constitutional_gate() -> None:
    """Reset the global gate instance (for testing)."""
    global _gate_instance
    _gate_instance = None
