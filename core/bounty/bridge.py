"""
Bounty Bridge — Platform Integration Layer

Connects BIZRA Proof-of-Impact system to external bounty platforms:
- Immunefi (DeFi-focused)
- HackerOne (General)
- Bugcrowd (Enterprise)
- Direct protocol treasuries (DeFi native)

Also handles:
- Smart contract payout execution
- ZK proof generation for trustless claims
- Cross-platform reputation aggregation
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from core.bounty.impact_proof import ImpactProof, Severity
from core.proof_engine.receipt import SovereignSigner


class Platform(Enum):
    """Supported bounty platforms."""

    IMMUNEFI = "immunefi"
    HACKERONE = "hackerone"
    BUGCROWD = "bugcrowd"
    DIRECT = "direct"  # Direct protocol submission
    BIZRA = "bizra"  # BIZRA native (smart contract)


class SubmissionStatus(Enum):
    """Submission status."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    TRIAGED = "triaged"
    CONFIRMED = "confirmed"
    PAID = "paid"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


@dataclass
class PlatformCredentials:
    """Credentials for a bounty platform."""

    platform: Platform
    api_key: str
    api_secret: Optional[str] = None
    wallet_address: Optional[str] = None  # For crypto payouts
    username: Optional[str] = None

    def is_valid(self) -> bool:
        return bool(self.api_key)


@dataclass
class BountySubmission:
    """A submission to a bounty platform."""

    submission_id: str
    platform: Platform
    proof: ImpactProof
    status: SubmissionStatus

    # Platform-specific data
    platform_submission_id: Optional[str] = None
    platform_url: Optional[str] = None

    # Payout tracking
    estimated_payout: float = 0.0
    actual_payout: Optional[float] = None
    payout_currency: str = "USD"
    payout_tx_hash: Optional[str] = None

    # Timeline
    submitted_at: Optional[datetime] = None
    triaged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "platform": self.platform.value,
            "proof_id": self.proof.proof_id,
            "status": self.status.value,
            "platform_submission_id": self.platform_submission_id,
            "platform_url": self.platform_url,
            "estimated_payout": self.estimated_payout,
            "actual_payout": self.actual_payout,
            "payout_currency": self.payout_currency,
            "payout_tx_hash": self.payout_tx_hash,
            "submitted_at": (
                self.submitted_at.isoformat() if self.submitted_at else None
            ),
            "triaged_at": self.triaged_at.isoformat() if self.triaged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "created_at": self.created_at.isoformat(),
        }


class PlatformAdapter(ABC):
    """
    Abstract adapter for bounty platforms.

    Each platform has its own submission format and API.
    """

    def __init__(self, credentials: PlatformCredentials):
        self.credentials = credentials

    @abstractmethod
    async def submit(self, proof: ImpactProof) -> BountySubmission:
        """Submit proof to platform."""
        pass

    @abstractmethod
    async def check_status(self, submission: BountySubmission) -> BountySubmission:
        """Check submission status."""
        pass

    @abstractmethod
    def format_report(self, proof: ImpactProof) -> Dict[str, Any]:
        """Format proof into platform-specific report."""
        pass


class ImmunefiAdapter(PlatformAdapter):
    """Adapter for Immunefi bounty platform."""

    def __init__(self, credentials: PlatformCredentials):
        super().__init__(credentials)
        self.base_url = "https://api.immunefi.com/v1"
        self._submission_counter = 0

    def _next_id(self) -> str:
        self._submission_counter += 1
        return f"imm_{self._submission_counter:08d}_{int(time.time() * 1000)}"

    def format_report(self, proof: ImpactProof) -> Dict[str, Any]:
        """Format for Immunefi submission."""
        severity_map = {
            Severity.INFORMATIONAL: "Informational",
            Severity.LOW: "Low",
            Severity.MEDIUM: "Medium",
            Severity.HIGH: "High",
            Severity.CRITICAL: "Critical",
        }

        return {
            "title": proof.title,
            "severity": severity_map.get(proof.severity, "Medium"),
            "target": {
                "address": proof.target_address,
                "chain": proof.target_chain,
                "protocol": proof.target_name,
            },
            "vulnerability_type": proof.vuln_category.value.replace("_", " ").title(),
            "impact": {
                "funds_at_risk_usd": proof.funds_at_risk,
                "entropy_delta": proof.delta_e,
            },
            "proof_of_concept": {
                "exploit_hash": proof.exploit_hash.hex(),
                "reproduction_steps": [s.to_dict() for s in proof.reproduction_steps],
            },
            "quality_metrics": {
                "snr_score": proof.snr_score,
                "ihsan_score": proof.ihsan_score,
            },
            "metadata": {
                "proof_id": proof.proof_id,
                "proof_digest": proof.hex_digest(),
                "submitted_via": "BIZRA PoI Bridge",
            },
        }

    async def submit(self, proof: ImpactProof) -> BountySubmission:
        """Submit to Immunefi."""
        self.format_report(proof)

        # In production, this would call the Immunefi API
        # For now, simulate submission
        submission = BountySubmission(
            submission_id=self._next_id(),
            platform=Platform.IMMUNEFI,
            proof=proof,
            status=SubmissionStatus.SUBMITTED,
            submitted_at=datetime.now(timezone.utc),
            estimated_payout=proof.funds_at_risk * 0.1,  # Typical 10% of funds at risk
        )

        return submission

    async def check_status(self, submission: BountySubmission) -> BountySubmission:
        """Check Immunefi submission status."""
        # In production, this would call the Immunefi API
        return submission


class HackerOneAdapter(PlatformAdapter):
    """Adapter for HackerOne bounty platform."""

    def __init__(self, credentials: PlatformCredentials):
        super().__init__(credentials)
        self._submission_counter = 0

    def _next_id(self) -> str:
        self._submission_counter += 1
        return f"h1_{self._submission_counter:08d}_{int(time.time() * 1000)}"

    def format_report(self, proof: ImpactProof) -> Dict[str, Any]:
        """Format for HackerOne submission."""
        return {
            "title": proof.title,
            "severity_rating": proof.severity.value,
            "weakness": proof.vuln_category.value,
            "structured_scope": proof.target_address,
            "impact": f"Funds at risk: ${proof.funds_at_risk:,.2f}",
            "proof_of_concept": proof.exploit_hash.hex(),
            "reference": proof.proof_id,
        }

    async def submit(self, proof: ImpactProof) -> BountySubmission:
        """Submit to HackerOne."""
        submission = BountySubmission(
            submission_id=self._next_id(),
            platform=Platform.HACKERONE,
            proof=proof,
            status=SubmissionStatus.SUBMITTED,
            submitted_at=datetime.now(timezone.utc),
        )
        return submission

    async def check_status(self, submission: BountySubmission) -> BountySubmission:
        """Check HackerOne submission status."""
        return submission


class DirectProtocolAdapter(PlatformAdapter):
    """Adapter for direct protocol submissions."""

    def __init__(self, credentials: PlatformCredentials):
        super().__init__(credentials)
        self._submission_counter = 0

    def _next_id(self) -> str:
        self._submission_counter += 1
        return f"dir_{self._submission_counter:08d}_{int(time.time() * 1000)}"

    def format_report(self, proof: ImpactProof) -> Dict[str, Any]:
        """Format for direct submission."""
        return proof.to_dict()

    async def submit(self, proof: ImpactProof) -> BountySubmission:
        """Submit directly to protocol."""
        submission = BountySubmission(
            submission_id=self._next_id(),
            platform=Platform.DIRECT,
            proof=proof,
            status=SubmissionStatus.SUBMITTED,
            submitted_at=datetime.now(timezone.utc),
        )
        return submission

    async def check_status(self, submission: BountySubmission) -> BountySubmission:
        """Check direct submission status."""
        return submission


class BIZRASmartContractAdapter(PlatformAdapter):
    """
    Adapter for BIZRA native smart contract payouts.

    Enables trustless, instant payouts via on-chain verification.
    """

    def __init__(
        self,
        credentials: PlatformCredentials,
        contract_address: str = "0x0000000000000000000000000000000000000000",
    ):
        super().__init__(credentials)
        self.contract_address = contract_address
        self._submission_counter = 0

    def _next_id(self) -> str:
        self._submission_counter += 1
        return f"biz_{self._submission_counter:08d}_{int(time.time() * 1000)}"

    def format_report(self, proof: ImpactProof) -> Dict[str, Any]:
        """Format for smart contract submission."""
        return {
            "proof_hash": proof.hex_digest(),
            "proof_signature": proof.signature.hex(),
            "hunter_pubkey": proof.hunter_pubkey.hex(),
            "delta_e": int(proof.delta_e * 1e18),  # Scale for uint256
            "severity": proof.severity.value,
            "target": proof.target_address,
        }

    async def submit(self, proof: ImpactProof) -> BountySubmission:
        """Submit to BIZRA smart contract."""
        # In production, this would:
        # 1. Encode the proof for the contract
        # 2. Send transaction to BIZRABountyPool.claimBounty()
        # 3. Return transaction hash

        submission = BountySubmission(
            submission_id=self._next_id(),
            platform=Platform.BIZRA,
            proof=proof,
            status=SubmissionStatus.SUBMITTED,
            submitted_at=datetime.now(timezone.utc),
            payout_currency="ETH",
        )

        return submission

    async def check_status(self, submission: BountySubmission) -> BountySubmission:
        """Check on-chain status."""
        # In production, this would check the blockchain
        return submission

    def generate_claim_calldata(self, proof: ImpactProof) -> bytes:
        """Generate calldata for claimBounty() function."""
        # Function selector: claimBounty(bytes32,bytes,uint256)
        selector = bytes.fromhex("a1b2c3d4")  # Placeholder

        # Encode parameters
        proof_hash = bytes.fromhex(proof.hex_digest())
        zk_proof = proof.zk_proof or b"\x00" * 32
        delta_e = int(proof.delta_e * 1e18)

        # ABI encode (simplified)
        calldata = selector + proof_hash + zk_proof + delta_e.to_bytes(32, "big")
        return calldata


class BountyBridge:
    """
    The BIZRA Bounty Bridge.

    Unified interface for submitting proofs to multiple platforms.
    """

    def __init__(self, signer: SovereignSigner):
        self.signer = signer
        self._adapters: Dict[Platform, PlatformAdapter] = {}
        self._submissions: List[BountySubmission] = []

    def register_platform(
        self,
        platform: Platform,
        credentials: PlatformCredentials,
        **kwargs,
    ):
        """Register a bounty platform."""
        if platform == Platform.IMMUNEFI:
            self._adapters[platform] = ImmunefiAdapter(credentials)
        elif platform == Platform.HACKERONE:
            self._adapters[platform] = HackerOneAdapter(credentials)
        elif platform == Platform.DIRECT:
            self._adapters[platform] = DirectProtocolAdapter(credentials)
        elif platform == Platform.BIZRA:
            contract = kwargs.get("contract_address", "0x0")
            self._adapters[platform] = BIZRASmartContractAdapter(credentials, contract)
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    async def submit(
        self,
        proof: ImpactProof,
        platform: Platform,
    ) -> BountySubmission:
        """Submit proof to a specific platform."""
        if platform not in self._adapters:
            raise ValueError(f"Platform not registered: {platform}")

        adapter = self._adapters[platform]
        submission = await adapter.submit(proof)
        self._submissions.append(submission)

        return submission

    async def submit_to_all(
        self,
        proof: ImpactProof,
    ) -> List[BountySubmission]:
        """Submit to all registered platforms."""
        submissions = []

        for platform in self._adapters:
            try:
                submission = await self.submit(proof, platform)
                submissions.append(submission)
            except Exception:
                # Log error but continue with other platforms
                pass

        return submissions

    async def check_all_status(self) -> List[BountySubmission]:
        """Check status of all submissions."""
        updated = []

        for submission in self._submissions:
            if submission.platform in self._adapters:
                adapter = self._adapters[submission.platform]
                updated_submission = await adapter.check_status(submission)
                updated.append(updated_submission)
            else:
                updated.append(submission)

        self._submissions = updated
        return updated

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        by_platform = {}
        by_status = {}
        total_estimated = 0.0
        total_actual = 0.0

        for sub in self._submissions:
            # By platform
            plat = sub.platform.value
            if plat not in by_platform:
                by_platform[plat] = 0
            by_platform[plat] += 1

            # By status
            stat = sub.status.value
            if stat not in by_status:
                by_status[stat] = 0
            by_status[stat] += 1

            # Totals
            total_estimated += sub.estimated_payout
            if sub.actual_payout:
                total_actual += sub.actual_payout

        return {
            "total_submissions": len(self._submissions),
            "registered_platforms": list(self._adapters.keys()),
            "by_platform": by_platform,
            "by_status": by_status,
            "total_estimated_usd": total_estimated,
            "total_actual_usd": total_actual,
        }

    def get_submissions(
        self,
        platform: Optional[Platform] = None,
        status: Optional[SubmissionStatus] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get filtered submissions."""
        filtered = self._submissions

        if platform:
            filtered = [s for s in filtered if s.platform == platform]
        if status:
            filtered = [s for s in filtered if s.status == status]

        return [s.to_dict() for s in filtered[-limit:]]
