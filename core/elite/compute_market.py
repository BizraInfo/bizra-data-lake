"""
Permission as Market — Harberger Tax + Gini Coefficient Enforcement

Creates a fair market mechanism for compute resource allocation using:
1. Harberger Tax: Self-assessed licensing with continuous taxation
2. Gini Coefficient: Inequality measurement and enforcement

Standing on Giants:
- Harberger (1965): Self-assessed property taxation
- Gini (1912): Statistical dispersion measure for inequality
- Vickrey (1961): Mechanism design and truthful revelation
- Ostrom (1990): Common pool resource governance
- Buterin (2018): Radical Markets for digital assets

Harberger Tax Mechanism:
- License holders self-assess the value of their compute allocation
- Pay a continuous tax based on that assessment (e.g., 5% per period)
- Anyone can buy the license at the self-assessed price
- This creates incentive for accurate valuation

Gini Coefficient Enforcement:
- Measures concentration of compute resources
- Gini = 0: Perfect equality (everyone has same resources)
- Gini = 1: Perfect inequality (one holder has everything)
- BIZRA target: Gini < 0.4 (moderate inequality)
- Enforcement redistributes when Gini exceeds threshold

Adl (Justice) Principle:
- Fair allocation reflects need and contribution
- No entity should monopolize shared resources
- Redistribution maintains system health

Created: 2026-02-03 | BIZRA Elite Integration v1.1.0
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Harberger Tax rate (5% per period)
DEFAULT_TAX_RATE = 0.05

# Tax period duration (1 hour)
TAX_PERIOD_SECONDS = 3600

# Gini coefficient threshold for redistribution
GINI_THRESHOLD = 0.40

# Maximum Gini before emergency redistribution
GINI_EMERGENCY = 0.60

# Minimum license value (prevents zero-valuation gaming)
MIN_LICENSE_VALUE = 1.0

# Maximum license duration without re-assessment
MAX_LICENSE_DURATION_HOURS = 24


# ============================================================================
# RESOURCE TYPES
# ============================================================================


class ResourceType(str, Enum):
    """Types of compute resources."""

    CPU = "cpu"  # CPU cycles
    GPU = "gpu"  # GPU compute units
    MEMORY = "memory"  # RAM allocation
    STORAGE = "storage"  # Disk space
    BANDWIDTH = "bandwidth"  # Network bandwidth
    INFERENCE = "inference"  # LLM inference slots
    CONSENSUS = "consensus"  # Consensus participation slots


@dataclass
class ResourceUnit:
    """
    A unit of compute resource.

    Represents a quantifiable resource with pricing.
    """

    resource_type: ResourceType
    quantity: float  # Amount in resource-specific units
    unit_name: str  # e.g., "cores", "GB", "TFLOPS"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize resource unit."""
        return {
            "type": self.resource_type.value,
            "quantity": self.quantity,
            "unit": self.unit_name,
        }


# ============================================================================
# LICENSE
# ============================================================================


class LicenseStatus(str, Enum):
    """License lifecycle status."""

    ACTIVE = "active"  # Currently held and valid
    EXPIRED = "expired"  # Tax period expired
    PURCHASED = "purchased"  # Just purchased, pending activation
    REVOKED = "revoked"  # Revoked due to non-payment
    TRANSFERRED = "transferred"  # Transferred to new holder


@dataclass
class ComputeLicense:
    """
    A Harberger-taxed compute license.

    The holder self-assesses value and pays continuous tax.
    Anyone can purchase at the self-assessed price.
    """

    # Identity
    license_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    # Resource
    resource: ResourceUnit = field(
        default_factory=lambda: ResourceUnit(ResourceType.INFERENCE, 1.0, "slots")
    )

    # Ownership
    holder_id: str = ""
    holder_ihsan: float = 0.95  # Holder's Ihsan score

    # Harberger valuation
    self_assessed_value: float = MIN_LICENSE_VALUE
    tax_rate: float = DEFAULT_TAX_RATE

    # Timing
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_tax_payment: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    expires_at: Optional[datetime] = None

    # Status
    status: LicenseStatus = LicenseStatus.ACTIVE

    # Accumulated tax debt
    tax_debt: float = 0.0

    # Transfer history
    transfer_count: int = 0

    def compute_tax_due(self, as_of: Optional[datetime] = None) -> float:
        """
        Compute tax due since last payment.

        Tax = value * rate * (periods_elapsed)
        """
        now = as_of or datetime.now(timezone.utc)
        elapsed = (now - self.last_tax_payment).total_seconds()
        periods = elapsed / TAX_PERIOD_SECONDS

        return self.self_assessed_value * self.tax_rate * periods

    def is_purchasable(self) -> bool:
        """Check if license can be purchased."""
        return self.status in (LicenseStatus.ACTIVE, LicenseStatus.EXPIRED)

    def time_until_expiry(self) -> Optional[timedelta]:
        """Get time until license expires."""
        if self.expires_at:
            return self.expires_at - datetime.now(timezone.utc)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize license."""
        return {
            "license_id": self.license_id,
            "resource": self.resource.to_dict(),
            "holder_id": self.holder_id,
            "holder_ihsan": self.holder_ihsan,
            "self_assessed_value": self.self_assessed_value,
            "tax_rate": self.tax_rate,
            "issued_at": self.issued_at.isoformat(),
            "last_tax_payment": self.last_tax_payment.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "tax_debt": self.tax_debt,
            "current_tax_due": self.compute_tax_due(),
            "transfer_count": self.transfer_count,
        }


# ============================================================================
# MARKET
# ============================================================================


@dataclass
class MarketTransaction:
    """A market transaction record."""

    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Parties
    seller_id: str = ""
    buyer_id: str = ""

    # License
    license_id: str = ""

    # Economics
    price: float = 0.0
    tax_collected: float = 0.0

    # Result
    success: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize transaction."""
        return {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp.isoformat(),
            "seller_id": self.seller_id,
            "buyer_id": self.buyer_id,
            "license_id": self.license_id,
            "price": self.price,
            "tax_collected": self.tax_collected,
            "success": self.success,
            "reason": self.reason,
        }


class ComputeMarket:
    """
    Harberger Tax-based compute market.

    Manages license issuance, taxation, and trading.
    Enforces Gini coefficient constraints for fair distribution.
    """

    def __init__(
        self,
        tax_rate: float = DEFAULT_TAX_RATE,
        gini_threshold: float = GINI_THRESHOLD,
        gini_emergency: float = GINI_EMERGENCY,
    ):
        self.tax_rate = tax_rate
        self.gini_threshold = gini_threshold
        self.gini_emergency = gini_emergency

        # License registry: license_id -> ComputeLicense
        self._licenses: Dict[str, ComputeLicense] = {}

        # Holder index: holder_id -> set of license_ids
        self._holder_licenses: Dict[str, Set[str]] = defaultdict(set)

        # Transaction history
        self._transactions: List[MarketTransaction] = []

        # Treasury (collected taxes)
        self._treasury: float = 0.0

        # Statistics
        self._total_issued = 0
        self._total_transfers = 0
        self._total_tax_collected = 0.0

    def issue_license(
        self,
        holder_id: str,
        resource: ResourceUnit,
        initial_value: float,
        holder_ihsan: float = UNIFIED_IHSAN_THRESHOLD,
    ) -> ComputeLicense:
        """
        Issue a new compute license.

        Args:
            holder_id: ID of the license holder
            resource: Resource being licensed
            initial_value: Initial self-assessed value
            holder_ihsan: Holder's Ihsan score

        Returns:
            The issued license
        """
        # Enforce minimum value
        value = max(MIN_LICENSE_VALUE, initial_value)

        # Create license
        license = ComputeLicense(
            resource=resource,
            holder_id=holder_id,
            holder_ihsan=holder_ihsan,
            self_assessed_value=value,
            tax_rate=self.tax_rate,
            expires_at=datetime.now(timezone.utc)
            + timedelta(hours=MAX_LICENSE_DURATION_HOURS),
        )

        # Register
        self._licenses[license.license_id] = license
        self._holder_licenses[holder_id].add(license.license_id)
        self._total_issued += 1

        logger.info(
            f"License issued: {license.license_id} to {holder_id} "
            f"({resource.resource_type.value}: {resource.quantity} {resource.unit_name}, "
            f"value={value})"
        )

        # Check Gini after issuance
        gini = self.compute_gini()
        if gini > self.gini_threshold:
            logger.warning(
                f"Gini coefficient {gini:.3f} exceeds threshold {self.gini_threshold}"
            )

        return license

    def get_license(self, license_id: str) -> Optional[ComputeLicense]:
        """Get license by ID."""
        return self._licenses.get(license_id)

    def get_holder_licenses(self, holder_id: str) -> List[ComputeLicense]:
        """Get all licenses held by an entity."""
        return [
            self._licenses[lid]
            for lid in self._holder_licenses.get(holder_id, set())
            if lid in self._licenses
        ]

    def reassess_value(
        self,
        license_id: str,
        new_value: float,
        holder_id: str,
    ) -> bool:
        """
        Reassess license value.

        Only the holder can reassess. New value must be >= MIN_LICENSE_VALUE.
        """
        license = self._licenses.get(license_id)
        if not license:
            logger.warning(f"License not found: {license_id}")
            return False

        if license.holder_id != holder_id:
            logger.warning(f"Not authorized to reassess: {holder_id}")
            return False

        old_value = license.self_assessed_value
        license.self_assessed_value = max(MIN_LICENSE_VALUE, new_value)

        logger.info(
            f"License {license_id} reassessed: {old_value} -> {license.self_assessed_value}"
        )

        return True

    def purchase_license(
        self,
        license_id: str,
        buyer_id: str,
        buyer_ihsan: float,
    ) -> MarketTransaction:
        """
        Purchase a license at its self-assessed value.

        The Harberger mechanism: anyone can buy at the declared price.

        Args:
            license_id: License to purchase
            buyer_id: ID of buyer
            buyer_ihsan: Buyer's Ihsan score

        Returns:
            Transaction record
        """
        license = self._licenses.get(license_id)
        if not license:
            return MarketTransaction(
                buyer_id=buyer_id,
                license_id=license_id,
                success=False,
                reason="License not found",
            )

        if not license.is_purchasable():
            return MarketTransaction(
                buyer_id=buyer_id,
                license_id=license_id,
                success=False,
                reason=f"License not purchasable (status: {license.status.value})",
            )

        # Cannot buy own license
        if license.holder_id == buyer_id:
            return MarketTransaction(
                buyer_id=buyer_id,
                license_id=license_id,
                success=False,
                reason="Cannot purchase own license",
            )

        # Ihsan check
        if buyer_ihsan < UNIFIED_IHSAN_THRESHOLD:
            return MarketTransaction(
                buyer_id=buyer_id,
                license_id=license_id,
                success=False,
                reason=f"Buyer Ihsan {buyer_ihsan} below threshold",
            )

        # Collect outstanding tax from seller
        tax_due = license.compute_tax_due()

        # Execute transfer
        old_holder = license.holder_id
        self._holder_licenses[old_holder].discard(license_id)

        license.holder_id = buyer_id
        license.holder_ihsan = buyer_ihsan
        license.status = LicenseStatus.TRANSFERRED
        license.transfer_count += 1
        license.last_tax_payment = datetime.now(timezone.utc)
        license.tax_debt = 0.0

        self._holder_licenses[buyer_id].add(license_id)

        # Collect tax
        self._treasury += tax_due
        self._total_tax_collected += tax_due
        self._total_transfers += 1

        transaction = MarketTransaction(
            seller_id=old_holder,
            buyer_id=buyer_id,
            license_id=license_id,
            price=license.self_assessed_value,
            tax_collected=tax_due,
            success=True,
            reason="Purchase completed",
        )

        self._transactions.append(transaction)

        logger.info(
            f"License {license_id} transferred: {old_holder} -> {buyer_id} "
            f"(price={license.self_assessed_value}, tax={tax_due:.2f})"
        )

        # Reactivate
        license.status = LicenseStatus.ACTIVE
        license.expires_at = datetime.now(timezone.utc) + timedelta(
            hours=MAX_LICENSE_DURATION_HOURS
        )

        return transaction

    def collect_taxes(self) -> float:
        """
        Collect taxes from all active licenses.

        Returns total tax collected.
        """
        total_collected = 0.0
        now = datetime.now(timezone.utc)

        for license in self._licenses.values():
            if license.status == LicenseStatus.ACTIVE:
                tax_due = license.compute_tax_due(now)
                license.tax_debt += tax_due
                license.last_tax_payment = now
                total_collected += tax_due

        self._treasury += total_collected
        self._total_tax_collected += total_collected

        logger.info(
            f"Tax collection: {total_collected:.2f} from {len(self._licenses)} licenses"
        )

        return total_collected

    def pay_tax(self, license_id: str, amount: float) -> bool:
        """
        Pay tax for a license.

        Returns True if payment clears debt.
        """
        license = self._licenses.get(license_id)
        if not license:
            return False

        payment = min(amount, license.tax_debt)
        license.tax_debt -= payment

        if license.tax_debt <= 0:
            license.tax_debt = 0
            license.status = LicenseStatus.ACTIVE
            return True

        return False

    def revoke_delinquent(self, debt_threshold: float = 0.0) -> List[str]:
        """
        Revoke licenses with excessive tax debt.

        Args:
            debt_threshold: Debt level that triggers revocation

        Returns:
            List of revoked license IDs
        """
        revoked = []

        for license in self._licenses.values():
            if (
                license.tax_debt > debt_threshold
                and license.status == LicenseStatus.ACTIVE
            ):
                license.status = LicenseStatus.REVOKED
                self._holder_licenses[license.holder_id].discard(license.license_id)
                revoked.append(license.license_id)

                logger.warning(
                    f"License {license.license_id} revoked: debt={license.tax_debt:.2f}"
                )

        return revoked

    # ========================================================================
    # GINI COEFFICIENT
    # ========================================================================

    def compute_gini(self) -> float:
        """
        Compute Gini coefficient of resource distribution.

        Gini = (sum of |x_i - x_j| for all pairs) / (2 * n * sum of x_i)

        Returns value in [0, 1]:
        - 0 = perfect equality
        - 1 = perfect inequality
        """
        # Aggregate value per holder
        holder_values: Dict[str, float] = defaultdict(float)

        for license in self._licenses.values():
            if license.status == LicenseStatus.ACTIVE:
                holder_values[license.holder_id] += license.self_assessed_value

        values = list(holder_values.values())

        if len(values) < 2:
            return 0.0  # Cannot compute with < 2 holders

        n = len(values)
        total = sum(values)

        if total == 0:
            return 0.0

        # Mean absolute difference
        sum_diff = sum(abs(values[i] - values[j]) for i in range(n) for j in range(n))

        gini = sum_diff / (2 * n * total)
        return gini

    def get_distribution(self) -> Dict[str, float]:
        """Get current resource distribution by holder."""
        distribution: Dict[str, float] = defaultdict(float)

        for license in self._licenses.values():
            if license.status == LicenseStatus.ACTIVE:
                distribution[license.holder_id] += license.self_assessed_value

        return dict(distribution)

    def enforce_gini(self) -> Dict[str, Any]:
        """
        Enforce Gini coefficient threshold.

        If Gini exceeds threshold, trigger redistribution:
        1. Identify over-concentrated holders
        2. Force sale of excess licenses
        3. Make licenses available at discounted rates

        Returns enforcement report.
        """
        gini = self.compute_gini()
        report: Dict[str, Any] = {
            "gini_before": gini,
            "threshold": self.gini_threshold,
            "action_taken": False,
            "forced_sales": [],
            "redistributed_value": 0.0,
        }

        if gini <= self.gini_threshold:
            return report

        report["action_taken"] = True
        logger.warning(
            f"Gini enforcement triggered: {gini:.3f} > {self.gini_threshold}"
        )

        # Get distribution
        distribution = self.get_distribution()
        total_value = sum(distribution.values())
        n_holders = len(distribution)

        if n_holders == 0:
            return report

        # Target: equal distribution
        target_per_holder = total_value / n_holders

        # Find over-allocated holders
        over_allocated = {
            h: v
            for h, v in distribution.items()
            if v > target_per_holder * 1.5  # 50% over fair share
        }

        for holder_id, excess in over_allocated.items():
            # Force sale of excess licenses
            holder_licenses = self.get_holder_licenses(holder_id)

            # Sort by value descending
            holder_licenses.sort(key=lambda l: l.self_assessed_value, reverse=True)

            current_value = excess
            for license in holder_licenses:
                if current_value <= target_per_holder:
                    break

                # Mark for forced sale (50% discount)
                license.self_assessed_value *= 0.5
                license.status = LicenseStatus.EXPIRED  # Available for purchase

                report["forced_sales"].append(
                    {
                        "license_id": license.license_id,
                        "holder_id": holder_id,
                        "new_value": license.self_assessed_value,
                    }
                )

                report["redistributed_value"] += license.self_assessed_value
                current_value -= license.self_assessed_value

        report["gini_after"] = self.compute_gini()

        logger.info(
            f"Gini enforcement complete: {report['gini_before']:.3f} -> {report['gini_after']:.3f}"
        )

        return report

    # ========================================================================
    # TREASURY
    # ========================================================================

    def get_treasury_balance(self) -> float:
        """Get current treasury balance."""
        return self._treasury

    def distribute_treasury(
        self,
        amount: float,
        recipients: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Distribute treasury funds.

        Can be used for:
        - UBI-style equal distribution
        - Rewards for high-Ihsan actors
        - Subsidizing under-resourced nodes

        Args:
            amount: Amount to distribute
            recipients: Specific recipients (None = all holders)

        Returns:
            Distribution by recipient
        """
        amount = min(amount, self._treasury)

        if recipients is None:
            recipients = list(self._holder_licenses.keys())

        if not recipients:
            return {}

        per_recipient = amount / len(recipients)
        distribution = {r: per_recipient for r in recipients}

        self._treasury -= amount

        logger.info(
            f"Treasury distribution: {amount:.2f} to {len(recipients)} recipients"
        )

        return distribution

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get market statistics."""
        active_licenses = [
            l for l in self._licenses.values() if l.status == LicenseStatus.ACTIVE
        ]
        total_value = sum(l.self_assessed_value for l in active_licenses)

        return {
            "total_licenses": len(self._licenses),
            "active_licenses": len(active_licenses),
            "total_holders": len(self._holder_licenses),
            "total_value": total_value,
            "treasury_balance": self._treasury,
            "gini_coefficient": self.compute_gini(),
            "gini_threshold": self.gini_threshold,
            "total_issued": self._total_issued,
            "total_transfers": self._total_transfers,
            "total_tax_collected": self._total_tax_collected,
            "avg_value": total_value / max(len(active_licenses), 1),
            "tax_rate": self.tax_rate,
        }

    def get_market_health(self) -> Dict[str, Any]:
        """
        Assess market health.

        Returns health indicators:
        - gini_status: healthy/warning/critical
        - liquidity: fraction of tradeable licenses
        - participation: active holders / all holders
        """
        gini = self.compute_gini()

        if gini <= self.gini_threshold:
            gini_status = "healthy"
        elif gini <= self.gini_emergency:
            gini_status = "warning"
        else:
            gini_status = "critical"

        active = len(
            [l for l in self._licenses.values() if l.status == LicenseStatus.ACTIVE]
        )
        purchasable = len([l for l in self._licenses.values() if l.is_purchasable()])

        return {
            "gini_status": gini_status,
            "gini_value": gini,
            "liquidity": purchasable / max(len(self._licenses), 1),
            "active_rate": active / max(len(self._licenses), 1),
            "treasury_health": "healthy" if self._treasury > 0 else "depleted",
            "adl_compliance": gini <= self.gini_threshold,  # Justice compliance
        }


# ============================================================================
# NTU INTEGRATION
# ============================================================================


class NTUMarketAdapter:
    """
    Adapts NTU temporal patterns to market decisions.

    Uses NTU state to inform:
    - License valuation recommendations
    - Gini threshold adjustment
    - Tax rate modulation
    """

    def __init__(self, market: ComputeMarket):
        self.market = market
        self._ntu = None

    @property
    def ntu(self):
        """Lazy-load NTU."""
        if self._ntu is None:
            try:
                from core.ntu import NTU, NTUConfig

                self._ntu = NTU(NTUConfig())  # type: ignore[assignment]
            except ImportError:
                logger.warning("NTU not available")
        return self._ntu

    def recommend_valuation(
        self,
        resource: ResourceUnit,
        holder_history: Optional[List[float]] = None,
    ) -> float:
        """
        Recommend license valuation based on NTU patterns.

        Uses temporal patterns to estimate fair market value.
        """
        # Base valuation from resource type
        base_values = {
            ResourceType.CPU: 10.0,
            ResourceType.GPU: 100.0,
            ResourceType.MEMORY: 5.0,
            ResourceType.STORAGE: 2.0,
            ResourceType.BANDWIDTH: 8.0,
            ResourceType.INFERENCE: 50.0,
            ResourceType.CONSENSUS: 25.0,
        }

        base = base_values.get(resource.resource_type, 10.0) * resource.quantity

        if self.ntu is None:
            return base

        # Observe holder history if available
        if holder_history:
            for value in holder_history[-5:]:  # Last 5 transactions
                self.ntu.observe(value / 100, {"source": "market_history"})

        # Adjust based on NTU state
        state = self.ntu.state
        confidence = state.belief * (1.0 - state.entropy)

        # High confidence = stable market = closer to base
        # Low confidence = volatile market = increase buffer
        adjustment = 1.0 + (0.5 - confidence) * 0.3

        return base * adjustment

    def observe_transaction(self, transaction: MarketTransaction) -> None:
        """Record transaction in NTU for pattern tracking."""
        if self.ntu is None:
            return

        # Quality signal: successful high-value transaction = positive
        if transaction.success:
            signal = min(1.0, transaction.price / 100)  # Normalize
        else:
            signal = 0.2  # Failed transaction = negative signal

        self.ntu.observe(
            signal,
            {
                "source": "market_transaction",
                "success": transaction.success,
                "value": transaction.price,
            },
        )


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_market(
    tax_rate: float = DEFAULT_TAX_RATE,
    gini_threshold: float = GINI_THRESHOLD,
) -> ComputeMarket:
    """
    Create a compute market.

    Example:
        market = create_market()
        license = market.issue_license(
            "node_001",
            ResourceUnit(ResourceType.INFERENCE, 10, "slots"),
            initial_value=500.0
        )
    """
    return ComputeMarket(tax_rate, gini_threshold)


def create_inference_license(
    market: ComputeMarket,
    holder_id: str,
    slots: int = 1,
    value: float = 50.0,
) -> ComputeLicense:
    """
    Create an inference slot license.

    Convenience function for the common case of licensing inference compute.
    """
    resource = ResourceUnit(
        resource_type=ResourceType.INFERENCE,
        quantity=float(slots),
        unit_name="slots",
    )

    return market.issue_license(holder_id, resource, value)
