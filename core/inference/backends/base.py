"""
BIZRA INFERENCE BACKEND BASE
═══════════════════════════════════════════════════════════════════════════════

Abstract base class for inference backends.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

# Import from parent module to avoid circular imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enum import Enum

class InferenceBackend(str, Enum):
    """Available inference backends."""
    LLAMACPP = "llamacpp"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    POOL = "pool"
    OFFLINE = "offline"


class InferenceBackendBase(ABC):
    """Abstract base class for inference backends."""
    
    @property
    @abstractmethod
    def backend_type(self) -> InferenceBackend:
        """Return the backend type."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the backend. Returns True if successful."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate a completion."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a completion with streaming."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        pass
    
    @abstractmethod
    def get_loaded_model(self) -> Optional[str]:
        """Return the currently loaded model name."""
        pass
