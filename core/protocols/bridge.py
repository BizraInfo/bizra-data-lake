"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA BRIDGE PROTOCOL                                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Abstract interface for cross-system bridges.                               ║
║                                                                              ║
║   Standardizes the 7+ bridge implementations:                                ║
║   - core/sovereign/bridge.py                                                 ║
║   - core/sovereign/dual_agentic_bridge.py                                    ║
║   - core/sovereign/swarm_knowledge_bridge.py                                 ║
║   - core/sovereign/local_inference_bridge.py                                 ║
║   - core/ntu/bridge.py                                                       ║
║   - core/pat/bridge.py                                                       ║
║   - core/bounty/bridge.py                                                    ║
║                                                                              ║
║   Standing on Giants: Gang of Four Bridge Pattern (1994)                     ║
║   Constitutional: All bridges must preserve Ihsan constraints                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Created: 2026-02-05 | SAPE Elite Analysis Implementation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Generic, Optional, TypeVar


class BridgeDirection(Enum):
    """Direction of data flow through the bridge."""

    UNIDIRECTIONAL = auto()  # One-way translation
    BIDIRECTIONAL = auto()  # Two-way translation
    BROADCAST = auto()  # One-to-many


class BridgeHealth(Enum):
    """Health status of a bridge connection."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISCONNECTED = "disconnected"


@dataclass
class BridgeMetrics:
    """Performance metrics for a bridge."""

    messages_sent: int = 0
    messages_received: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    last_activity: Optional[datetime] = None

    def record_message(self, latency_ms: float, direction: str = "sent"):
        """Record a message transmission."""
        if direction == "sent":
            self.messages_sent += 1
        else:
            self.messages_received += 1

        # Rolling average for latency
        total = self.messages_sent + self.messages_received
        self.avg_latency_ms = (self.avg_latency_ms * (total - 1) + latency_ms) / total
        self.last_activity = datetime.utcnow()

    def record_error(self):
        """Record an error."""
        self.errors += 1
        self.last_activity = datetime.utcnow()


# Generic type variables for bridge input/output
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class BridgeProtocol(ABC, Generic[TInput, TOutput]):
    """
    Abstract base class for system bridges.

    A bridge translates data between two systems or domains, maintaining
    semantic consistency and applying any necessary transformations.

    Type Parameters:
        TInput: The input type from the source system
        TOutput: The output type for the target system

    Example implementation:
    ```python
    class InferenceBridge(BridgeProtocol[QueryRequest, InferenceRequest]):
        async def translate(self, input: QueryRequest) -> InferenceRequest:
            return InferenceRequest(
                prompt=input.query,
                max_tokens=input.max_length,
                ...
            )
    ```
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable bridge identifier."""
        ...

    @property
    @abstractmethod
    def direction(self) -> BridgeDirection:
        """Data flow direction through this bridge."""
        ...

    @property
    def source_system(self) -> str:
        """Name of the source system."""
        return "unknown"

    @property
    def target_system(self) -> str:
        """Name of the target system."""
        return "unknown"

    @abstractmethod
    async def translate(self, input: TInput) -> TOutput:
        """
        Translate input from source format to target format.

        Args:
            input: Data in source system format

        Returns:
            Data transformed to target system format

        Raises:
            ValueError: Input cannot be translated
            ConnectionError: Target system unreachable
            RuntimeError: Translation failed
        """
        ...

    @abstractmethod
    async def health_check(self) -> BridgeHealth:
        """
        Check the health of both endpoints of the bridge.

        Returns:
            Current health status
        """
        ...

    # Optional methods with default implementations

    async def reverse_translate(self, output: TOutput) -> TInput:
        """
        Reverse translation for bidirectional bridges.

        Only required for BIDIRECTIONAL bridges.

        Raises:
            NotImplementedError: If bridge is unidirectional
        """
        if self.direction == BridgeDirection.UNIDIRECTIONAL:
            raise NotImplementedError(
                f"Bridge {self.name} is unidirectional, reverse translation not supported"
            )
        raise NotImplementedError("Subclass must implement reverse_translate()")

    def validate_input(self, input: TInput) -> bool:
        """
        Validate input before translation.

        Default implementation accepts all input.
        Override for input-specific validation.
        """
        return input is not None

    def validate_output(self, output: TOutput) -> bool:
        """
        Validate output after translation.

        Default implementation accepts all output.
        Override for output-specific validation.
        """
        return output is not None

    def get_metrics(self) -> BridgeMetrics:
        """
        Get bridge performance metrics.

        Returns:
            Current metrics snapshot
        """
        return BridgeMetrics()

    async def connect(self) -> bool:
        """
        Establish connection to both endpoints.

        Returns:
            True if connection successful
        """
        return await self.health_check() == BridgeHealth.HEALTHY

    async def disconnect(self) -> None:
        """
        Gracefully disconnect from endpoints.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self.name}' ({self.direction.name})>"


class CompositeBridge(BridgeProtocol[TInput, TOutput]):
    """
    A bridge composed of multiple sequential bridges.

    Useful for chaining transformations through intermediate formats.

    Example:
    ```python
    # Chain: Query -> Embedding -> VectorSearch -> Results
    pipeline = CompositeBridge([
        QueryToEmbeddingBridge(),
        EmbeddingToSearchBridge(),
        SearchToResultsBridge(),
    ])
    ```
    """

    def __init__(self, bridges: list[BridgeProtocol]):
        """
        Args:
            bridges: Ordered list of bridges to chain
        """
        if not bridges:
            raise ValueError("CompositeBridge requires at least one bridge")
        self._bridges = bridges

    @property
    def name(self) -> str:
        return f"Composite[{' -> '.join(b.name for b in self._bridges)}]"

    @property
    def direction(self) -> BridgeDirection:
        # Composite is bidirectional only if ALL bridges are bidirectional
        if all(b.direction == BridgeDirection.BIDIRECTIONAL for b in self._bridges):
            return BridgeDirection.BIDIRECTIONAL
        return BridgeDirection.UNIDIRECTIONAL

    async def translate(self, input: TInput) -> TOutput:
        """Chain translation through all bridges."""
        current = input
        for bridge in self._bridges:
            current = await bridge.translate(current)
        return current  # type: ignore[return-value]

    async def health_check(self) -> BridgeHealth:
        """Check health of all bridges in the chain."""
        healths = [await b.health_check() for b in self._bridges]

        if all(h == BridgeHealth.HEALTHY for h in healths):
            return BridgeHealth.HEALTHY
        if any(h == BridgeHealth.DISCONNECTED for h in healths):
            return BridgeHealth.DISCONNECTED
        if any(h == BridgeHealth.UNHEALTHY for h in healths):
            return BridgeHealth.UNHEALTHY
        return BridgeHealth.DEGRADED
