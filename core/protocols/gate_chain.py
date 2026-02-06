"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA GATE CHAIN PROTOCOL                                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Abstract interface for validation gate chains.                             ║
║                                                                              ║
║   Standardizes the gate chain implementations:                               ║
║   - core/pci/gates.py: SCHEMA -> SIGNATURE -> TIMESTAMP -> REPLAY -> ...     ║
║   - core/sovereign/model_license_gate.py: Different ordering                 ║
║                                                                              ║
║   Gate Chain Design Principles:                                              ║
║   1. Fail-fast: Cheap gates before expensive gates                           ║
║   2. Security-first: Signature verification early                            ║
║   3. Composable: Gates can be added/removed dynamically                      ║
║                                                                              ║
║   Standing on Giants:                                                        ║
║   - Chain of Responsibility Pattern (GoF, 1994)                              ║
║   - Railway Oriented Programming (Wlaschin, 2014)                            ║
║                                                                              ║
║   Constitutional: All gates must enforce Ihsan >= 0.95                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Created: 2026-02-05 | SAPE Elite Analysis Implementation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar


class GateStatus(Enum):
    """Result status of a gate validation."""

    PASS = auto()  # Validation passed
    FAIL = auto()  # Validation failed
    SKIP = auto()  # Gate skipped (not applicable)
    ERROR = auto()  # Gate encountered an error


@dataclass
class GateResult:
    """
    Result of a single gate validation.

    Provides detailed information about why a gate passed or failed.
    """

    status: GateStatus
    gate_name: str
    message: str = ""

    # Optional details
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # For failures
    reject_code: Optional[str] = None

    @property
    def passed(self) -> bool:
        """Check if the gate passed or was skipped."""
        return self.status in (GateStatus.PASS, GateStatus.SKIP)

    def __bool__(self) -> bool:
        """Allow using GateResult in boolean context."""
        return self.passed


@dataclass
class ChainResult:
    """
    Aggregate result of running a complete gate chain.
    """

    passed: bool
    gate_results: List[GateResult] = field(default_factory=list)

    # Summary
    total_gates: int = 0
    passed_gates: int = 0
    failed_gates: int = 0
    skipped_gates: int = 0

    # Performance
    total_duration_ms: float = 0.0

    # First failure (if any)
    first_failure: Optional[GateResult] = None

    def add_result(self, result: GateResult):
        """Add a gate result and update summary."""
        self.gate_results.append(result)
        self.total_gates += 1
        self.total_duration_ms += result.duration_ms

        if result.status == GateStatus.PASS:
            self.passed_gates += 1
        elif result.status == GateStatus.FAIL:
            self.failed_gates += 1
            if self.first_failure is None:
                self.first_failure = result
            self.passed = False
        elif result.status == GateStatus.SKIP:
            self.skipped_gates += 1
        elif result.status == GateStatus.ERROR:
            self.failed_gates += 1
            if self.first_failure is None:
                self.first_failure = result
            self.passed = False


# Generic type for items being validated
T = TypeVar("T")


class Gate(ABC, Generic[T]):
    """
    Abstract base class for a single validation gate.

    Gates are the building blocks of validation chains.
    Each gate performs a specific validation and returns a result.

    Example implementation:
    ```python
    class SignatureGate(Gate[PCIEnvelope]):
        @property
        def name(self) -> str:
            return "SIGNATURE"

        def validate(self, envelope: PCIEnvelope) -> GateResult:
            if envelope.verify_signature():
                return GateResult(GateStatus.PASS, self.name)
            return GateResult(
                GateStatus.FAIL, self.name,
                message="Invalid Ed25519 signature",
                reject_code="SIG_INVALID"
            )
    ```
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique gate identifier."""
        ...

    @property
    def tier(self) -> int:
        """
        Gate tier for ordering (lower = cheaper, run first).

        Tier 1: Cheap checks (schema, format)
        Tier 2: Medium checks (signature, timestamp)
        Tier 3: Expensive checks (Ihsan scoring, external calls)
        """
        return 2  # Default to medium tier

    @property
    def required(self) -> bool:
        """
        Whether this gate is required or optional.

        Required gates must pass for the chain to pass.
        Optional gates are informational only.
        """
        return True

    @abstractmethod
    def validate(self, item: T) -> GateResult:
        """
        Validate the item.

        Args:
            item: The item to validate

        Returns:
            Result indicating pass/fail with details
        """
        ...

    def should_run(self, item: T) -> bool:
        """
        Check if this gate should run for the given item.

        Override to conditionally skip gates based on item type.
        """
        return True

    def __repr__(self) -> str:
        return f"<Gate '{self.name}' tier={self.tier}>"


class GateChain(Generic[T]):
    """
    A chain of validation gates that run in sequence.

    Gates are automatically ordered by tier (cheap first).
    Chain execution stops at the first required gate failure.

    Example usage:
    ```python
    chain = GateChain[PCIEnvelope]()
    chain.add_gate(SchemaGate())      # Tier 1
    chain.add_gate(SignatureGate())   # Tier 1
    chain.add_gate(TimestampGate())   # Tier 1
    chain.add_gate(IhsanGate())       # Tier 3

    result = chain.validate(envelope)
    if result.passed:
        process(envelope)
    else:
        reject(result.first_failure.reject_code)
    ```
    """

    def __init__(self, fail_fast: bool = True):
        """
        Args:
            fail_fast: Stop at first required gate failure (default True)
        """
        self._gates: List[Gate[T]] = []
        self._fail_fast = fail_fast

    def add_gate(self, gate: Gate[T]) -> "GateChain[T]":
        """
        Add a gate to the chain.

        Gates are sorted by tier when the chain is validated.

        Returns:
            Self for method chaining
        """
        self._gates.append(gate)
        return self

    def remove_gate(self, name: str) -> bool:
        """
        Remove a gate by name.

        Returns:
            True if gate was found and removed
        """
        for i, gate in enumerate(self._gates):
            if gate.name == name:
                del self._gates[i]
                return True
        return False

    def validate(self, item: T) -> ChainResult:
        """
        Run all gates in the chain.

        Gates are sorted by tier (lowest first).
        Execution stops at first required gate failure if fail_fast=True.

        Args:
            item: The item to validate

        Returns:
            Aggregate result of all gates
        """
        result = ChainResult(passed=True)

        # Sort gates by tier for fail-fast optimization
        sorted_gates = sorted(self._gates, key=lambda g: g.tier)

        for gate in sorted_gates:
            # Check if gate should run
            if not gate.should_run(item):
                result.add_result(
                    GateResult(
                        GateStatus.SKIP,
                        gate.name,
                        message="Gate skipped (not applicable)",
                    )
                )
                continue

            # Run the gate
            import time

            start = time.perf_counter()
            try:
                gate_result = gate.validate(item)
            except Exception as e:
                gate_result = GateResult(
                    GateStatus.ERROR,
                    gate.name,
                    message=f"Gate error: {str(e)}",
                    reject_code="GATE_ERROR",
                )
            gate_result.duration_ms = (time.perf_counter() - start) * 1000

            result.add_result(gate_result)

            # Check fail-fast
            if self._fail_fast and not gate_result.passed and gate.required:
                break

        return result

    @property
    def gates(self) -> List[Gate[T]]:
        """Get list of gates in the chain."""
        return list(self._gates)

    def __len__(self) -> int:
        return len(self._gates)

    def __repr__(self) -> str:
        gate_names = [g.name for g in sorted(self._gates, key=lambda g: g.tier)]
        return f"<GateChain [{' -> '.join(gate_names)}]>"


# Utility function for creating simple gates
def simple_gate(
    name: str,
    validator: Callable[[T], bool],
    failure_message: str = "Validation failed",
    reject_code: Optional[str] = None,
    tier: int = 2,
) -> Gate[T]:
    """
    Create a simple gate from a validation function.

    Example:
    ```python
    not_empty = simple_gate(
        "NOT_EMPTY",
        lambda x: bool(x),
        "Item cannot be empty",
        "EMPTY_ITEM"
    )
    ```
    """

    class SimpleGate(Gate[T]):
        @property
        def name(self) -> str:
            return name

        @property
        def tier(self) -> int:
            return tier

        def validate(self, item: T) -> GateResult:
            if validator(item):
                return GateResult(GateStatus.PASS, self.name)
            return GateResult(
                GateStatus.FAIL,
                self.name,
                message=failure_message,
                reject_code=reject_code,
            )

    return SimpleGate()
