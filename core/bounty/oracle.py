"""
Bounty Oracle — Automatic Payout Calculation

The BountyOracle calculates fair payouts based on:
- ΔE (entropy reduced)
- Severity multiplier
- Funds at risk
- SNR quality score

Formula:
    payout = (base × ΔE + risk_factor) × severity_multiplier × quality_bonus
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.proof_engine.canonical import blake3_digest, canonical_bytes
from core.proof_engine.receipt import SovereignSigner
from core.bounty import (
    BASE_PAYOUT_PER_DELTA_E,
    SEVERITY_LEVELS,
    BOUNTY_SNR_THRESHOLD,
    BOUNTY_IHSAN_THRESHOLD,
)
from core.bounty.impact_proof import ImpactProof, Severity


@dataclass
class BountyCalculation:
    """
    Detailed breakdown of bounty calculation.

    Provides full transparency on how payout was determined.
    """
    calculation_id: str
    proof_id: str

    # Input factors
    delta_e: float
    severity: Severity
    funds_at_risk: float
    snr_score: float
    ihsan_score: float

    # Calculated components
    base_payout: float
    severity_multiplier: int
    risk_bonus: float
    quality_bonus: float

    # Final payout
    total_payout: float
    payout_currency: str = "USD"

    # Verification
    calculation_hash: bytes = field(default_factory=bytes)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calculation_id": self.calculation_id,
            "proof_id": self.proof_id,
            "delta_e": self.delta_e,
            "severity": self.severity.value,
            "funds_at_risk": self.funds_at_risk,
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
            "base_payout": self.base_payout,
            "severity_multiplier": self.severity_multiplier,
            "risk_bonus": self.risk_bonus,
            "quality_bonus": self.quality_bonus,
            "total_payout": self.total_payout,
            "payout_currency": self.payout_currency,
            "calculation_hash": self.calculation_hash.hex(),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BountyPayout:
    """
    Finalized bounty payout record.

    Signed and ready for smart contract execution.
    """
    payout_id: str
    calculation: BountyCalculation
    hunter_address: str  # Wallet/account to receive payout

    # Status
    status: str = "pending"  # pending, approved, paid, rejected

    # Smart contract data
    tx_hash: Optional[str] = None
    block_number: Optional[int] = None

    # Signature
    signature: bytes = field(default_factory=bytes)
    oracle_pubkey: bytes = field(default_factory=bytes)

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def body_bytes(self) -> bytes:
        """Get payout body for signing."""
        data = {
            "payout_id": self.payout_id,
            "calculation_id": self.calculation.calculation_id,
            "proof_id": self.calculation.proof_id,
            "hunter_address": self.hunter_address,
            "total_payout": self.calculation.total_payout,
            "payout_currency": self.calculation.payout_currency,
            "timestamp": self.timestamp.isoformat(),
        }
        return canonical_bytes(data)

    def sign_with(self, signer: SovereignSigner) -> "BountyPayout":
        """Sign the payout."""
        body = self.body_bytes()
        self.signature = signer.sign(body)
        self.oracle_pubkey = signer.public_key_bytes()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "payout_id": self.payout_id,
            "calculation": self.calculation.to_dict(),
            "hunter_address": self.hunter_address,
            "status": self.status,
            "tx_hash": self.tx_hash,
            "block_number": self.block_number,
            "signature": self.signature.hex(),
            "oracle_pubkey": self.oracle_pubkey.hex(),
            "timestamp": self.timestamp.isoformat(),
        }


class BountyOracle:
    """
    The BIZRA Bounty Oracle.

    Calculates and authorizes payouts based on verified impact proofs.
    Integrates with:
    - UERS framework (entropy measurement)
    - Proof Engine (verification)
    - Smart contracts (payment)
    """

    def __init__(
        self,
        signer: SovereignSigner,
        base_payout: float = BASE_PAYOUT_PER_DELTA_E,
        snr_threshold: float = BOUNTY_SNR_THRESHOLD,
        ihsan_threshold: float = BOUNTY_IHSAN_THRESHOLD,
        max_payout: float = 1_000_000,  # $1M cap per bounty
    ):
        self.signer = signer
        self.base_payout = base_payout
        self.snr_threshold = snr_threshold
        self.ihsan_threshold = ihsan_threshold
        self.max_payout = max_payout

        self._calculation_counter = 0
        self._payout_counter = 0
        self._calculations: List[BountyCalculation] = []
        self._payouts: List[BountyPayout] = []

    def _next_calculation_id(self) -> str:
        self._calculation_counter += 1
        return f"calc_{self._calculation_counter:08d}_{int(time.time() * 1000)}"

    def _next_payout_id(self) -> str:
        self._payout_counter += 1
        return f"pay_{self._payout_counter:08d}_{int(time.time() * 1000)}"

    def calculate_bounty(self, proof: ImpactProof) -> Tuple[BountyCalculation, Optional[str]]:
        """
        Calculate bounty payout for an impact proof.

        Returns (calculation, error_message).
        """
        # Validate thresholds
        if proof.snr_score < self.snr_threshold:
            return None, f"SNR below threshold: {proof.snr_score:.3f}"

        if proof.ihsan_score < self.ihsan_threshold:
            return None, f"Ihsān below threshold: {proof.ihsan_score:.3f}"

        if proof.delta_e <= 0:
            return None, f"Non-positive entropy delta: {proof.delta_e:.3f}"

        # Calculate components
        # 1. Base payout from entropy reduction
        base = self.base_payout * proof.delta_e

        # 2. Severity multiplier
        severity_config = SEVERITY_LEVELS.get(proof.severity.value, {})
        severity_mult = severity_config.get("multiplier", 1)
        min_payout = severity_config.get("min_payout", 0)

        # 3. Risk bonus (percentage of funds at risk, capped at 10%)
        risk_bonus = min(proof.funds_at_risk * 0.10, self.max_payout * 0.5)

        # 4. Quality bonus (SNR and Ihsān above threshold)
        snr_bonus = (proof.snr_score - self.snr_threshold) * 1000  # $1000 per 0.01 above threshold
        ihsan_bonus = (proof.ihsan_score - self.ihsan_threshold) * 500
        quality_bonus = max(0, snr_bonus + ihsan_bonus)

        # Calculate total
        total = (base + risk_bonus) * severity_mult + quality_bonus

        # Apply minimum and maximum
        total = max(total, min_payout)
        total = min(total, self.max_payout)

        # Create calculation record
        calculation = BountyCalculation(
            calculation_id=self._next_calculation_id(),
            proof_id=proof.proof_id,
            delta_e=proof.delta_e,
            severity=proof.severity,
            funds_at_risk=proof.funds_at_risk,
            snr_score=proof.snr_score,
            ihsan_score=proof.ihsan_score,
            base_payout=base,
            severity_multiplier=severity_mult,
            risk_bonus=risk_bonus,
            quality_bonus=quality_bonus,
            total_payout=round(total, 2),
        )

        # Compute calculation hash
        calculation.calculation_hash = blake3_digest(
            canonical_bytes(calculation.to_dict())
        )

        self._calculations.append(calculation)
        return calculation, None

    def create_payout(
        self,
        calculation: BountyCalculation,
        hunter_address: str,
    ) -> BountyPayout:
        """
        Create a signed payout authorization.

        This can be submitted to the smart contract for execution.
        """
        payout = BountyPayout(
            payout_id=self._next_payout_id(),
            calculation=calculation,
            hunter_address=hunter_address,
            status="approved",
        )

        payout.sign_with(self.signer)
        self._payouts.append(payout)

        return payout

    def process_proof(
        self,
        proof: ImpactProof,
        hunter_address: str,
    ) -> Tuple[Optional[BountyPayout], Optional[str]]:
        """
        Full pipeline: verify proof → calculate bounty → create payout.

        Returns (payout, error_message).
        """
        # Calculate bounty
        calculation, error = self.calculate_bounty(proof)
        if error:
            return None, error

        # Create payout
        payout = self.create_payout(calculation, hunter_address)
        return payout, None

    def get_stats(self) -> Dict[str, Any]:
        """Get oracle statistics."""
        if not self._calculations:
            return {
                "total_calculations": 0,
                "total_payouts": 0,
                "total_paid_usd": 0,
            }

        total_paid = sum(c.total_payout for c in self._calculations)
        avg_payout = total_paid / len(self._calculations)

        by_severity = {}
        for c in self._calculations:
            sev = c.severity.value
            if sev not in by_severity:
                by_severity[sev] = {"count": 0, "total": 0}
            by_severity[sev]["count"] += 1
            by_severity[sev]["total"] += c.total_payout

        return {
            "total_calculations": len(self._calculations),
            "total_payouts": len(self._payouts),
            "total_paid_usd": total_paid,
            "average_payout_usd": avg_payout,
            "by_severity": by_severity,
            "max_payout": self.max_payout,
            "base_per_delta_e": self.base_payout,
        }

    def get_recent_payouts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent payout records."""
        return [p.to_dict() for p in self._payouts[-limit:]]

    def estimate_payout(
        self,
        delta_e: float,
        severity: str,
        funds_at_risk: float = 0.0,
        snr_score: float = 0.95,
        ihsan_score: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Estimate payout without creating records.

        Useful for hunters to evaluate targets.
        """
        severity_config = SEVERITY_LEVELS.get(severity, {})
        severity_mult = severity_config.get("multiplier", 1)
        min_payout = severity_config.get("min_payout", 0)

        base = self.base_payout * delta_e
        risk_bonus = min(funds_at_risk * 0.10, self.max_payout * 0.5)
        quality_bonus = max(0, (snr_score - self.snr_threshold) * 1000)

        total = (base + risk_bonus) * severity_mult + quality_bonus
        total = max(total, min_payout)
        total = min(total, self.max_payout)

        return {
            "estimated_payout_usd": round(total, 2),
            "breakdown": {
                "base": base,
                "severity_multiplier": severity_mult,
                "risk_bonus": risk_bonus,
                "quality_bonus": quality_bonus,
            },
            "meets_thresholds": snr_score >= self.snr_threshold and ihsan_score >= self.ihsan_threshold,
        }
