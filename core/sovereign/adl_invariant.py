"""
BIZRA ADL (JUSTICE) INVARIANT - Protocol-Level Anti-Plutocracy Enforcement

Standing on Giants:
- Gini (1912): Statistical measure of inequality
- Harberger (1962): Self-assessed value with continuous taxation
- Rawls (1971): Veil of Ignorance - design as if position unknown
- Lamport (1982): Byzantine fault tolerance for distributed consensus

Constitutional Principle:
"Adl (عدل) - Justice is not optional. It is a hard constraint."

The Adl Invariant ensures that no transaction can push the network's
wealth distribution beyond the constitutional Gini threshold of 0.40.
This is not a warning - it is a HARD GATE that rejects plutocratic moves.

Mathematical Foundation:
- Gini coefficient G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
- Where x_i are sorted holdings and i is rank (1 to n)
- G = 0: Perfect equality (everyone holds the same)
- G = 1: Perfect inequality (one entity holds everything)
- Threshold = 0.40: Moderate inequality, prevents plutocracy

Harberger Tax Mechanism:
- All holdings are taxed at a continuous rate (default 5% annually)
- Tax proceeds flow to Universal Basic Compute (UBC) pool
- UBC distributes equally to all active nodes
- This creates a natural redistribution pressure toward equality
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Callable, Dict, Optional

# Import unified thresholds from authoritative source

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - CONSTITUTIONAL THRESHOLDS
# =============================================================================

# The Adl threshold - Gini coefficient must not exceed this value
# 0.40 represents moderate inequality - below most developed nations
# This is a HARD GATE, not a soft warning
ADL_GINI_THRESHOLD: float = 0.40

# Harberger tax rate (annual, applied continuously)
# 5% strikes balance between redistribution pressure and stability
HARBERGER_TAX_RATE: float = 0.05

# Minimum holding to be considered a participant
# Prevents dust attacks and ensures meaningful participation
MINIMUM_HOLDING: float = 1e-9

# Universal Basic Compute pool identifier
UBC_POOL_ID: str = "__UBC_POOL__"


# =============================================================================
# REJECTION CODES
# =============================================================================


class AdlRejectCode(IntEnum):
    """Rejection codes specific to Adl invariant violations."""

    SUCCESS = 0
    REJECT_GINI_EXCEEDED = 100  # Post-tx Gini would exceed threshold
    REJECT_CONSERVATION_VIOLATED = 101  # Total value changed
    REJECT_NEGATIVE_HOLDING = 102  # Would create negative balance
    REJECT_INVALID_TRANSACTION = 103  # Malformed transaction
    REJECT_DUST_AMOUNT = 104  # Amount below minimum holding


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class Transaction:
    """
    Represents a value transfer between nodes.

    Standing on Giants - Lamport:
    All transactions must be atomic and verifiable.
    """

    tx_id: str
    sender: str
    recipient: str
    amount: float
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
        if self.amount < 0:
            raise ValueError("Transaction amount cannot be negative")


@dataclass
class AdlValidationResult:
    """
    Result of Adl invariant validation.

    Contains detailed information about why a transaction was accepted or rejected,
    enabling transparent auditability of economic governance decisions.
    """

    passed: bool
    reject_code: AdlRejectCode
    message: str
    pre_gini: float = 0.0
    post_gini: float = 0.0
    gini_delta: float = 0.0
    threshold: float = ADL_GINI_THRESHOLD
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for PCI envelope inclusion."""
        return {
            "passed": self.passed,
            "reject_code": int(self.reject_code),
            "reject_code_name": self.reject_code.name,
            "message": self.message,
            "pre_gini": round(self.pre_gini, 6),
            "post_gini": round(self.post_gini, 6),
            "gini_delta": round(self.gini_delta, 6),
            "threshold": self.threshold,
            "details": self.details,
        }


@dataclass
class RedistributionResult:
    """
    Result of Harberger-style redistribution.

    Tracks the flow of value from holdings to the UBC pool
    and the equal distribution to all participants.
    """

    success: bool
    pre_gini: float
    post_gini: float
    total_tax_collected: float
    ubc_per_node: float
    nodes_affected: int
    holdings_before: Dict[str, float] = field(default_factory=dict)
    holdings_after: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# GINI COEFFICIENT CALCULATOR
# =============================================================================


def calculate_gini(holdings: Dict[str, float]) -> float:
    """
    Calculate the Gini coefficient for a distribution of holdings.

    Standing on Giants - Gini (1912):
    The Gini coefficient measures statistical dispersion intended to
    represent wealth inequality. It ranges from 0 (perfect equality)
    to 1 (perfect inequality).

    Complexity Analysis (P1-1 Optimization):
    ─────────────────────────────────────────
    Time:  O(n log n) dominated by sorting
    Space: O(n) for filtered values array

    The algorithm uses the direct formula with sorted values:
    G = (2 * Σ(i * x_i)) / (n * Σx_i) - (n + 1) / n

    This is optimal for the Gini coefficient calculation.
    Alternative O(n) approaches exist but require pre-sorted input.

    Formula: G = 1 - 2 * integral(L(p)) dp
           = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n

    Args:
        holdings: Dictionary mapping node_id to their holdings (must be >= 0)

    Returns:
        Gini coefficient in range [0, 1]

    Edge Cases:
        - Empty holdings: Returns 0 (no inequality in nothing)
        - Single holder: Returns 0 (no inequality with one party)
        - All equal: Returns 0 (perfect equality)
        - All zero: Returns 0 (no holdings = no inequality)
        - Negative values: Raises ValueError (invalid state)
    """
    # Check for negative values FIRST (invalid economic state)
    # This must happen before filtering to catch all negatives
    # Use generator for memory efficiency on large holdings
    all_values = [v for k, v in holdings.items() if k != UBC_POOL_ID]
    if any(v < 0 for v in all_values):
        raise ValueError("Holdings cannot be negative - invalid economic state")

    # Filter out nodes below minimum (dust)
    values = [v for v in all_values if v >= MINIMUM_HOLDING]

    # Edge cases - O(1)
    n = len(values)
    if n == 0:
        return 0.0  # No holdings = no inequality

    if n == 1:
        return 0.0  # Single holder = no inequality by definition

    total = sum(values)  # O(n)
    if total == 0:
        return 0.0  # All zeros = no inequality

    # Sort for O(n log n) calculation - DOMINANT COST
    sorted_values = sorted(values)

    # Calculate using the formula in single pass O(n):
    # G = (2 * sum((i+1) * x_i) / (n * sum(x_i))) - (n + 1) / n
    # where i is 0-indexed, so we use (i+1) for 1-indexed rank
    #
    # Optimized: compute weighted_sum in single enumeration
    weighted_sum = sum((i + 1) * x for i, x in enumerate(sorted_values))

    # Final calculation - O(1)
    gini = (2 * weighted_sum) / (n * total) - (n + 1) / n

    # Clamp to valid range [0, 1] to handle floating-point errors
    return max(0.0, min(1.0, gini))


def calculate_gini_components(holdings: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate Gini coefficient with detailed component breakdown.

    Useful for debugging and transparency in economic governance.

    Returns:
        Dictionary with gini coefficient and component details
    """
    values = [
        v for k, v in holdings.items() if k != UBC_POOL_ID and v >= MINIMUM_HOLDING
    ]

    if len(values) <= 1:
        return {
            "gini": 0.0,
            "n": len(values),
            "total": sum(values) if values else 0.0,
            "mean": values[0] if values else 0.0,
            "median": values[0] if values else 0.0,
            "min": values[0] if values else 0.0,
            "max": values[0] if values else 0.0,
            "top_10_pct_share": 1.0 if values else 0.0,
            "bottom_50_pct_share": 1.0 if values else 0.0,
        }

    sorted_values = sorted(values)
    n = len(sorted_values)
    total = sum(sorted_values)

    # Calculate Gini
    weighted_sum = sum((i + 1) * x for i, x in enumerate(sorted_values))
    gini = (2 * weighted_sum) / (n * total) - (n + 1) / n
    gini = max(0.0, min(1.0, gini))

    # Additional metrics for transparency
    median_idx = n // 2
    median = (
        (sorted_values[median_idx] + sorted_values[median_idx - 1]) / 2
        if n % 2 == 0
        else sorted_values[median_idx]
    )

    # Top 10% share
    top_10_count = max(1, n // 10)
    top_10_share = sum(sorted_values[-top_10_count:]) / total if total > 0 else 0.0

    # Bottom 50% share
    bottom_50_count = n // 2
    bottom_50_share = sum(sorted_values[:bottom_50_count]) / total if total > 0 else 0.0

    return {
        "gini": gini,
        "n": n,
        "total": total,
        "mean": total / n,
        "median": median,
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "top_10_pct_share": top_10_share,
        "bottom_50_pct_share": bottom_50_share,
    }


# =============================================================================
# ADL INVARIANT VALIDATOR
# =============================================================================


class AdlInvariant:
    """
    The Adl (Justice) Invariant - Protocol-level anti-plutocracy enforcement.

    Standing on Giants:
    - Rawls (1971): Design economic systems as if you don't know your position
    - Harberger (1962): Continuous taxation for efficient allocation
    - Lamport (1982): Byzantine fault tolerance for consensus

    This class enforces the constitutional constraint that no transaction
    may push the network's Gini coefficient above the threshold (0.40).

    Usage:
        invariant = AdlInvariant()

        # Validate a transaction
        result = invariant.validate_transaction(tx, current_holdings)
        if not result.passed:
            raise RejectionError(result.message)

        # Apply Harberger tax redistribution
        new_holdings = invariant.redistribute_soil_tax(holdings)
    """

    def __init__(
        self,
        gini_threshold: float = ADL_GINI_THRESHOLD,
        tax_rate: float = HARBERGER_TAX_RATE,
    ):
        """
        Initialize the Adl Invariant.

        Args:
            gini_threshold: Maximum allowed Gini coefficient (default 0.40)
            tax_rate: Annual Harberger tax rate (default 0.05 = 5%)
        """
        if not 0.0 <= gini_threshold <= 1.0:
            raise ValueError(f"Gini threshold must be in [0, 1], got {gini_threshold}")
        if not 0.0 <= tax_rate <= 1.0:
            raise ValueError(f"Tax rate must be in [0, 1], got {tax_rate}")

        self.gini_threshold = gini_threshold
        self.tax_rate = tax_rate
        self._validation_count = 0
        self._rejection_count = 0

    def validate_transaction(
        self,
        tx: Transaction,
        current_state: Dict[str, float],
    ) -> AdlValidationResult:
        """
        Validate a transaction against the Adl invariant.

        This is the HARD GATE. If the post-transaction Gini coefficient
        would exceed the threshold, the transaction is REJECTED.

        Standing on Giants - Rawls:
        "Justice is the first virtue of social institutions."

        Args:
            tx: The proposed transaction
            current_state: Current holdings (node_id -> amount)

        Returns:
            AdlValidationResult with pass/fail status and detailed metrics

        Invariants:
            - Post-transaction Gini <= threshold (0.40)
            - Conservation: sum(post_state) == sum(pre_state)
            - Non-negativity: all holdings >= 0
        """
        self._validation_count += 1

        # Validate transaction structure
        if tx.amount < MINIMUM_HOLDING:
            self._rejection_count += 1
            return AdlValidationResult(
                passed=False,
                reject_code=AdlRejectCode.REJECT_DUST_AMOUNT,
                message=f"Transaction amount {tx.amount} below minimum {MINIMUM_HOLDING}",
                details={"tx_id": tx.tx_id, "amount": tx.amount},
            )

        # Calculate pre-transaction Gini
        pre_gini = calculate_gini(current_state)

        # Compute post-transaction state
        post_state = current_state.copy()

        # Initialize sender if not present
        if tx.sender not in post_state:
            post_state[tx.sender] = 0.0

        # Check sender has sufficient balance
        if post_state[tx.sender] < tx.amount:
            self._rejection_count += 1
            return AdlValidationResult(
                passed=False,
                reject_code=AdlRejectCode.REJECT_NEGATIVE_HOLDING,
                message=f"Sender {tx.sender} has insufficient balance: {post_state[tx.sender]} < {tx.amount}",
                pre_gini=pre_gini,
                details={
                    "tx_id": tx.tx_id,
                    "sender_balance": post_state[tx.sender],
                    "amount": tx.amount,
                },
            )

        # Apply transaction
        post_state[tx.sender] -= tx.amount
        post_state[tx.recipient] = post_state.get(tx.recipient, 0.0) + tx.amount

        # Validate conservation law
        pre_total = sum(v for k, v in current_state.items() if k != UBC_POOL_ID)
        post_total = sum(v for k, v in post_state.items() if k != UBC_POOL_ID)

        if abs(pre_total - post_total) > 1e-9:
            self._rejection_count += 1
            return AdlValidationResult(
                passed=False,
                reject_code=AdlRejectCode.REJECT_CONSERVATION_VIOLATED,
                message=f"Conservation violated: pre={pre_total}, post={post_total}",
                pre_gini=pre_gini,
                details={
                    "tx_id": tx.tx_id,
                    "pre_total": pre_total,
                    "post_total": post_total,
                    "delta": post_total - pre_total,
                },
            )

        # Calculate post-transaction Gini
        post_gini = calculate_gini(post_state)
        gini_delta = post_gini - pre_gini

        # THE HARD GATE: Check Gini threshold
        if post_gini > self.gini_threshold:
            self._rejection_count += 1
            return AdlValidationResult(
                passed=False,
                reject_code=AdlRejectCode.REJECT_GINI_EXCEEDED,
                message=(
                    f"Adl violation: post-tx Gini {post_gini:.4f} > threshold {self.gini_threshold}. "
                    f"Transaction would increase inequality beyond constitutional limit."
                ),
                pre_gini=pre_gini,
                post_gini=post_gini,
                gini_delta=gini_delta,
                details={
                    "tx_id": tx.tx_id,
                    "sender": tx.sender,
                    "recipient": tx.recipient,
                    "amount": tx.amount,
                    "sender_post_balance": post_state[tx.sender],
                    "recipient_post_balance": post_state[tx.recipient],
                },
            )

        # Transaction passes Adl invariant
        return AdlValidationResult(
            passed=True,
            reject_code=AdlRejectCode.SUCCESS,
            message="Transaction passes Adl invariant",
            pre_gini=pre_gini,
            post_gini=post_gini,
            gini_delta=gini_delta,
            details={
                "tx_id": tx.tx_id,
                "gini_headroom": self.gini_threshold - post_gini,
            },
        )

    def redistribute_soil_tax(
        self,
        holdings: Dict[str, float],
        tax_rate: Optional[float] = None,
        time_fraction: float = 1.0,
    ) -> Dict[str, float]:
        """
        Apply Harberger-style soil tax redistribution.

        Standing on Giants - Harberger (1962):
        "A tax on the self-assessed value of assets creates efficient allocation
        by making hoarding expensive and encouraging productive use."

        The soil tax works as follows:
        1. Each holder pays tax_rate * holdings to the UBC pool
        2. The UBC pool is distributed equally to all active nodes
        3. Net effect: wealth flows from large holders to small holders

        Args:
            holdings: Current holdings (node_id -> amount)
            tax_rate: Override tax rate (default: self.tax_rate)
            time_fraction: Fraction of year elapsed (for continuous taxation)

        Returns:
            New holdings after tax redistribution

        Invariant:
            sum(new_holdings) == sum(old_holdings)  # Conservation
        """
        if tax_rate is None:
            tax_rate = self.tax_rate

        # Effective tax rate for the time period
        effective_rate = tax_rate * time_fraction

        # Get active participants (excluding UBC pool)
        participants = {
            k: v
            for k, v in holdings.items()
            if k != UBC_POOL_ID and v >= MINIMUM_HOLDING
        }

        if not participants:
            return holdings.copy()

        n_participants = len(participants)

        # Calculate tax collection
        total_tax = 0.0
        new_holdings = holdings.copy()

        for node_id, amount in participants.items():
            tax = amount * effective_rate
            new_holdings[node_id] = amount - tax
            total_tax += tax

        # Distribute equally to all participants (UBC)
        ubc_per_node = total_tax / n_participants

        for node_id in participants:
            new_holdings[node_id] += ubc_per_node

        # Store in UBC pool for transparency (net zero after distribution)
        new_holdings[UBC_POOL_ID] = new_holdings.get(UBC_POOL_ID, 0.0)

        return new_holdings

    def get_redistribution_impact(
        self,
        holdings: Dict[str, float],
        tax_rate: Optional[float] = None,
    ) -> RedistributionResult:
        """
        Calculate the impact of redistribution without applying it.

        Useful for previewing the effect of Harberger taxation.

        Args:
            holdings: Current holdings
            tax_rate: Override tax rate

        Returns:
            RedistributionResult with before/after metrics
        """
        pre_gini = calculate_gini(holdings)
        new_holdings = self.redistribute_soil_tax(holdings, tax_rate)
        post_gini = calculate_gini(new_holdings)

        participants = {
            k: v
            for k, v in holdings.items()
            if k != UBC_POOL_ID and v >= MINIMUM_HOLDING
        }

        rate = tax_rate if tax_rate is not None else self.tax_rate
        total_tax = sum(v * rate for v in participants.values())
        ubc_per_node = total_tax / len(participants) if participants else 0.0

        return RedistributionResult(
            success=True,
            pre_gini=pre_gini,
            post_gini=post_gini,
            total_tax_collected=total_tax,
            ubc_per_node=ubc_per_node,
            nodes_affected=len(participants),
            holdings_before=holdings.copy(),
            holdings_after=new_holdings,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "validations": self._validation_count,
            "rejections": self._rejection_count,
            "rejection_rate": self._rejection_count / max(1, self._validation_count),
            "gini_threshold": self.gini_threshold,
            "tax_rate": self.tax_rate,
        }


# =============================================================================
# ADL GATE - PCI INTEGRATION
# =============================================================================


class AdlGate:
    """
    Adl Gate for PCI envelope flow integration.

    This gate checks transactions within PCI envelopes against the
    Adl invariant and rejects any that would violate the constitutional
    Gini threshold.

    Gate Chain Position:
    SCHEMA -> SIGNATURE -> TIMESTAMP -> REPLAY -> IHSAN -> SNR -> POLICY -> ADL

    The Adl gate runs AFTER policy validation because it requires
    the transaction data to be authentic and policy-compliant before
    evaluating economic impact.

    Usage:
        gate = AdlGate(holdings_provider)
        result = gate.check(envelope)
        if not result.passed:
            return VerificationResult(False, result.reject_code, result.message)
    """

    def __init__(
        self,
        holdings_provider: Callable[..., Any],
        gini_threshold: float = ADL_GINI_THRESHOLD,
    ):
        """
        Initialize the Adl Gate.

        Args:
            holdings_provider: Callable that returns current holdings Dict[str, float]
            gini_threshold: Maximum allowed Gini coefficient
        """
        self.holdings_provider = holdings_provider
        self.invariant = AdlInvariant(gini_threshold=gini_threshold)

    def check(self, envelope: Any) -> AdlValidationResult:
        """
        Check a PCI envelope against the Adl invariant.

        Extracts transaction data from the envelope payload and validates
        against current holdings state.

        Args:
            envelope: PCIEnvelope with transaction payload

        Returns:
            AdlValidationResult
        """
        # Extract transaction from envelope payload
        try:
            payload_data = envelope.payload.data

            # Support both direct transaction and nested format
            if "transaction" in payload_data:
                tx_data = payload_data["transaction"]
            else:
                tx_data = payload_data

            tx = Transaction(
                tx_id=tx_data.get("tx_id", envelope.envelope_id),
                sender=tx_data.get("sender", envelope.sender.agent_id),
                recipient=tx_data.get("recipient", ""),
                amount=float(tx_data.get("amount", 0)),
                metadata=tx_data.get("metadata", {}),
            )

        except (AttributeError, KeyError, ValueError) as e:
            return AdlValidationResult(
                passed=False,
                reject_code=AdlRejectCode.REJECT_INVALID_TRANSACTION,
                message=f"Failed to parse transaction from envelope: {e}",
            )

        # Get current holdings
        try:
            current_holdings = self.holdings_provider()
        except Exception as e:
            return AdlValidationResult(
                passed=False,
                reject_code=AdlRejectCode.REJECT_INVALID_TRANSACTION,
                message=f"Failed to retrieve current holdings: {e}",
            )

        # Validate against Adl invariant
        return self.invariant.validate_transaction(tx, current_holdings)

    def get_current_gini(self) -> float:
        """Get the current Gini coefficient."""
        holdings = self.holdings_provider()
        return calculate_gini(holdings)

    def get_gini_headroom(self) -> float:
        """Get the headroom until Gini threshold is reached."""
        current = self.get_current_gini()
        return max(0.0, self.invariant.gini_threshold - current)


# =============================================================================
# EXTENDED PCI GATEKEEPER WITH ADL
# =============================================================================


def create_adl_extended_gatekeeper(
    base_gatekeeper: Any,
    holdings_provider: Callable[..., Any],
    gini_threshold: float = ADL_GINI_THRESHOLD,
) -> Any:
    """
    Create an extended PCIGateKeeper that includes Adl validation.

    This wraps the base gatekeeper and adds Adl gate to the chain.

    Args:
        base_gatekeeper: Existing PCIGateKeeper instance
        holdings_provider: Callable returning current holdings
        gini_threshold: Maximum allowed Gini coefficient

    Returns:
        Extended gatekeeper with Adl validation
    """
    from core.pci.gates import VerificationResult
    from core.pci.reject_codes import RejectCode

    adl_gate = AdlGate(holdings_provider, gini_threshold)

    # Store original verify method
    original_verify = base_gatekeeper.verify

    def extended_verify(envelope) -> VerificationResult:
        """Extended verification including Adl gate."""
        # Run base verification first
        base_result = original_verify(envelope)
        if not base_result.passed:
            return base_result

        # Check if this is a transaction envelope
        action = getattr(envelope.payload, "action", "")
        if action in ("transfer", "transaction", "value_transfer"):
            # Run Adl gate
            adl_result = adl_gate.check(envelope)
            if not adl_result.passed:
                return VerificationResult(
                    passed=False,
                    reject_code=RejectCode.REJECT_INVARIANT_FAILED,
                    details=f"Adl: {adl_result.message}",
                    gate_passed=base_result.gate_passed,
                )

            # Add ADL to passed gates
            if base_result.gate_passed:
                base_result.gate_passed.append("ADL")

        return base_result

    # Replace verify method
    base_gatekeeper.verify = extended_verify
    base_gatekeeper._adl_gate = adl_gate

    return base_gatekeeper


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def assert_adl_invariant(
    post_state: Dict[str, float],
    pre_state: Optional[Dict[str, float]] = None,
    threshold: float = ADL_GINI_THRESHOLD,
) -> None:
    """
    Assert that a state satisfies the Adl invariant.

    Raises AssertionError if violated - use for testing and runtime checks.

    Args:
        post_state: Holdings to validate
        pre_state: Optional pre-state for conservation check
        threshold: Gini threshold (default 0.40)
    """
    gini = calculate_gini(post_state)
    assert gini <= threshold, f"Adl violation: Gini {gini:.4f} > threshold {threshold}"

    if pre_state is not None:
        pre_total = sum(v for k, v in pre_state.items() if k != UBC_POOL_ID)
        post_total = sum(v for k, v in post_state.items() if k != UBC_POOL_ID)
        assert (
            abs(pre_total - post_total) < 1e-9
        ), f"Conservation violated: pre={pre_total}, post={post_total}"


def simulate_transaction_impact(
    tx: Transaction,
    current_state: Dict[str, float],
    threshold: float = ADL_GINI_THRESHOLD,
) -> Dict[str, Any]:
    """
    Simulate the impact of a transaction on Gini coefficient.

    Does not validate - just shows what would happen.

    Args:
        tx: Proposed transaction
        current_state: Current holdings
        threshold: Gini threshold for comparison

    Returns:
        Dictionary with pre/post metrics
    """
    pre_gini = calculate_gini(current_state)
    pre_components = calculate_gini_components(current_state)

    # Simulate post-state
    post_state = current_state.copy()
    post_state[tx.sender] = post_state.get(tx.sender, 0.0) - tx.amount
    post_state[tx.recipient] = post_state.get(tx.recipient, 0.0) + tx.amount

    post_gini = calculate_gini(post_state)
    post_components = calculate_gini_components(post_state)

    return {
        "transaction": {
            "sender": tx.sender,
            "recipient": tx.recipient,
            "amount": tx.amount,
        },
        "pre": pre_components,
        "post": post_components,
        "delta": {
            "gini": post_gini - pre_gini,
            "top_10_pct_share": post_components["top_10_pct_share"]
            - pre_components["top_10_pct_share"],
            "bottom_50_pct_share": post_components["bottom_50_pct_share"]
            - pre_components["bottom_50_pct_share"],
        },
        "would_pass": post_gini <= threshold,
        "headroom": threshold - post_gini,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "ADL_GINI_THRESHOLD",
    "HARBERGER_TAX_RATE",
    "MINIMUM_HOLDING",
    "UBC_POOL_ID",
    # Codes
    "AdlRejectCode",
    # Data structures
    "Transaction",
    "AdlValidationResult",
    "RedistributionResult",
    # Core functions
    "calculate_gini",
    "calculate_gini_components",
    # Main class
    "AdlInvariant",
    # PCI Integration
    "AdlGate",
    "create_adl_extended_gatekeeper",
    # Utilities
    "assert_adl_invariant",
    "simulate_transaction_impact",
]
