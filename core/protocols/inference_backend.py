"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA INFERENCE BACKEND PROTOCOL                                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Abstract interface for LLM inference backends (LMStudio, Ollama, LlamaCpp) ║
║                                                                              ║
║   Implementations:                                                           ║
║   - core/inference/lmstudio_backend.py                                       ║
║   - core/inference/ollama_backend.py (planned)                               ║
║   - core/inference/llamacpp_backend.py (planned)                             ║
║                                                                              ║
║   Standing on Giants: Liskov Substitution Principle (1987)                   ║
║   Constitutional: All responses validated against Ihsan >= 0.95              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Created: 2026-02-05 | SAPE Elite Analysis Implementation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, AsyncIterator, Dict, Optional, Set


class BackendCapability(Enum):
    """Capabilities that inference backends may support."""

    TEXT_COMPLETION = auto()
    CHAT_COMPLETION = auto()
    EMBEDDINGS = auto()
    VISION = auto()
    FUNCTION_CALLING = auto()
    STREAMING = auto()
    BATCHING = auto()
    STRUCTURED_OUTPUT = auto()
    CONTEXT_CACHING = auto()


@dataclass
class InferenceRequest:
    """
    Standardized inference request format.

    All backends must accept this format and transform it to their native API.
    """

    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    stop_sequences: list[str] = field(default_factory=list)

    # Optional fields
    system_prompt: Optional[str] = None
    chat_history: Optional[list[Dict[str, str]]] = None
    response_format: Optional[str] = None  # "json", "text", etc.

    # Metadata
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    timeout_ms: int = 30000

    def __post_init__(self):
        """Validate request parameters."""
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0.0, 2.0]")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be in [0.0, 1.0]")


@dataclass
class InferenceResponse:
    """
    Standardized inference response format.

    All backends must transform their native response to this format.
    """

    text: str
    tokens_used: int
    latency_ms: float

    # Quality metrics
    finish_reason: str = "stop"  # "stop", "length", "error"

    # Optional metadata
    model_id: Optional[str] = None
    request_id: Optional[str] = None

    # Performance tracking
    prompt_tokens: int = 0
    completion_tokens: int = 0
    queue_time_ms: float = 0.0

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def tokens_per_second(self) -> float:
        """Calculate generation speed."""
        if self.latency_ms <= 0:
            return 0.0
        return (self.completion_tokens / self.latency_ms) * 1000


class InferenceBackend(ABC):
    """
    Abstract base class for LLM inference backends.

    All implementations must:
    1. Implement the abstract methods
    2. Transform requests/responses to the standard format
    3. Handle errors gracefully with specific exception types
    4. Report health status accurately

    Example implementation:
    ```python
    class OllamaBackend(InferenceBackend):
        async def complete(self, request: InferenceRequest) -> InferenceResponse:
            # Transform to Ollama API format
            response = await self._client.generate(...)
            # Transform back to standard format
            return InferenceResponse(...)
    ```
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> Set[BackendCapability]:
        """Set of supported capabilities."""
        ...

    @abstractmethod
    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        """
        Execute a completion request.

        Args:
            request: Standardized inference request

        Returns:
            Standardized inference response

        Raises:
            ConnectionError: Backend unreachable
            TimeoutError: Request exceeded timeout
            ValueError: Invalid request parameters
            RuntimeError: Backend-specific errors
        """
        ...

    @abstractmethod
    async def is_healthy(self) -> bool:
        """
        Check if the backend is healthy and accepting requests.

        Returns:
            True if backend is operational, False otherwise
        """
        ...

    @abstractmethod
    async def get_loaded_models(self) -> list[str]:
        """
        Get list of currently loaded model identifiers.

        Returns:
            List of model IDs currently loaded in memory
        """
        ...

    # Optional methods with default implementations

    async def stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """
        Stream completion tokens as they're generated.

        Default implementation falls back to non-streaming complete().
        Override for backends that support native streaming.
        """
        if BackendCapability.STREAMING not in self.capabilities:
            response = await self.complete(request)
            yield response.text
            return

        # Subclasses should override this for native streaming
        raise NotImplementedError(
            f"{self.name} declares STREAMING capability but doesn't implement stream()"
        )

    async def batch_complete(
        self, requests: list[InferenceRequest]
    ) -> list[InferenceResponse]:
        """
        Execute multiple completion requests efficiently.

        Default implementation processes sequentially.
        Override for backends that support native batching.
        """
        if BackendCapability.BATCHING not in self.capabilities:
            return [await self.complete(req) for req in requests]

        # Subclasses should override this for native batching
        raise NotImplementedError(
            f"{self.name} declares BATCHING capability but doesn't implement batch_complete()"
        )

    async def warmup(self, model_id: Optional[str] = None) -> bool:
        """
        Pre-warm the backend by loading models into memory.

        Args:
            model_id: Specific model to warm up, or None for default

        Returns:
            True if warmup succeeded
        """
        return await self.is_healthy()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get backend-specific performance metrics.

        Returns:
            Dictionary of metric name -> value
        """
        return {}

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the backend, releasing resources.

        Called during application shutdown.
        """
        pass


# Type alias for backend factory functions
BackendFactory = type[InferenceBackend]
