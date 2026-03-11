"""
BIZRA LLAMA.CPP BACKEND
═══════════════════════════════════════════════════════════════════════════════

Embedded inference via llama-cpp-python.
Primary backend for sovereign inference. No external dependencies, works offline.
"""

import threading
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Optional

from .base import InferenceBackend, InferenceBackendBase

if TYPE_CHECKING:
    from ..gateway import InferenceConfig  # type: ignore[attr-defined]


class LlamaCppBackend(InferenceBackendBase):
    """
    Embedded inference via llama-cpp-python.

    This is the primary backend for sovereign inference.
    No external dependencies, works offline.
    """

    def __init__(self, config: "InferenceConfig"):
        self.config = config
        self._model = None
        self._model_path: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def backend_type(self) -> InferenceBackend:
        return InferenceBackend.LLAMACPP

    async def initialize(self) -> bool:
        """Initialize llama.cpp with configured model."""
        try:
            from llama_cpp import Llama
        except ImportError:
            print("[LlamaCpp] llama-cpp-python not installed")
            return False

        model_path = self._resolve_model_path()
        if not model_path:
            print("[LlamaCpp] No model found")
            return False

        try:
            print(f"[LlamaCpp] Loading model: {model_path}")
            self._model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.context_length,
                n_gpu_layers=self.config.n_gpu_layers,
                n_threads=self.config.n_threads,
                n_batch=self.config.n_batch,
                verbose=False,
            )
            self._model_path = str(model_path)
            print("[LlamaCpp] Model loaded successfully")
            return True
        except Exception as e:
            print(f"[LlamaCpp] Failed to load model: {e}")
            return False

    async def generate(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> str:
        """Generate completion."""
        if not self._model:
            raise RuntimeError("Model not initialized")

        with self._lock:
            result = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
                **kwargs,
            )

        return result["choices"][0]["text"]

    async def generate_stream(  # type: ignore[override]
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> AsyncIterator[str]:
        """Generate completion with streaming."""
        if not self._model:
            raise RuntimeError("Model not initialized")

        with self._lock:
            for chunk in self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
                stream=True,
                **kwargs,
            ):
                if "choices" in chunk and chunk["choices"]:
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        yield text

    async def health_check(self) -> bool:
        """Check if model is loaded and responsive."""
        if not self._model:
            return False
        try:
            with self._lock:
                self._model("test", max_tokens=1)
            return True
        except Exception:
            return False

    def get_loaded_model(self) -> Optional[str]:
        """Return loaded model path."""
        return self._model_path

    def _resolve_model_path(self) -> Optional[Path]:
        """Resolve the model path."""
        # 1. Explicit path
        if self.config.model_path:
            path = Path(self.config.model_path)
            if path.exists():
                return path

        # 2. Look in model directory
        if self.config.model_dir.exists():
            gguf_files = list(self.config.model_dir.glob("*.gguf"))
            if gguf_files:
                return gguf_files[0]

        return None
