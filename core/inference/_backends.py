"""
BIZRA Inference Gateway — Backend Implementations
═══════════════════════════════════════════════════════════════════════════════

Abstract base + concrete backends: LlamaCpp, Ollama, LM Studio.

Each backend supports optional circuit breaker protection for resilience
against cascading failures when external services become unavailable.

Extracted from gateway.py for modularity; re-exported by gateway.py
for backward compatibility.

Created: 2026-02-09 | Refactor split from gateway.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from ._batching import BatchingInferenceQueue
from ._connection_pool import ConnectionPool, ConnectionPoolConfig
from ._resilience import CircuitBreaker, CircuitBreakerMetrics
from ._types import (
    CircuitBreakerConfig,
    CircuitMetrics,
    CircuitState,
    InferenceBackend,
    InferenceConfig,
)

# Import LM Studio backend (primary)
try:
    from .lmstudio_backend import ChatMessage
    from .lmstudio_backend import LMStudioBackend as LMStudioClient
    from .lmstudio_backend import LMStudioConfig

    LMSTUDIO_AVAILABLE = True
except ImportError:
    LMSTUDIO_AVAILABLE = False
    LMStudioClient = None  # type: ignore[assignment, misc]
    LMStudioConfig = None  # type: ignore[assignment, misc]
    ChatMessage = None  # type: ignore[assignment, misc]


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════


class InferenceBackendBase(ABC):
    """
    Abstract base class for inference backends with circuit breaker support.

    All backends support optional circuit breaker protection for resilience
    against cascading failures when external services become unavailable.
    """

    _circuit_breaker: Optional[CircuitBreaker] = None

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
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> str:
        """Generate a completion."""
        pass

    @abstractmethod
    async def generate_stream(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
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

    def get_circuit_breaker(self) -> Optional[CircuitBreaker]:
        """Get the circuit breaker for this backend (if configured)."""
        return self._circuit_breaker

    def get_circuit_state(self) -> Optional[CircuitState]:
        """Get current circuit breaker state."""
        if self._circuit_breaker:
            return self._circuit_breaker.state
        return None

    def get_circuit_metrics(self) -> Optional[CircuitMetrics]:
        """Get circuit breaker metrics as dict."""
        if not self._circuit_breaker:
            return None
        metrics: CircuitBreakerMetrics = self._circuit_breaker.get_metrics()
        return CircuitMetrics(
            state=metrics.state.value,
            failure_count=metrics.failure_count,
            success_count=metrics.success_count,
            last_failure_time=metrics.last_failure_time,
            last_success_time=metrics.last_success_time,
            last_state_change=metrics.last_state_change,
            total_calls=metrics.total_calls,
            total_failures=metrics.total_failures,
            total_successes=metrics.total_successes,
            total_rejections=metrics.total_rejections,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LLAMA.CPP BACKEND
# ═══════════════════════════════════════════════════════════════════════════════


class LlamaCppBackend(InferenceBackendBase):
    """
    Embedded inference via llama-cpp-python.

    This is the primary backend for sovereign inference.
    No external dependencies, works offline.

    P0-P1 Optimization: Request batching for 8x throughput improvement.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self._model = None
        self._model_path: Optional[str] = None
        self._lock = asyncio.Lock()

        # P0-P1: Batching queue (replaces serial lock)
        self._batch_queue: Optional[BatchingInferenceQueue] = None

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

            # P0-P1: Initialize batching queue if enabled
            if self.config.enable_batching:
                self._batch_queue = BatchingInferenceQueue(
                    backend_generate_fn=self._generate_direct,
                    max_batch_size=self.config.max_batch_size,
                    max_wait_ms=self.config.max_batch_wait_ms,
                )
                await self._batch_queue.start()
                print(
                    f"[LlamaCpp] Batching enabled (max_batch={self.config.max_batch_size})"
                )
            else:
                print("[LlamaCpp] Batching disabled (using serial lock)")

            return True
        except Exception as e:
            print(f"[LlamaCpp] Failed to load model: {e}")
            return False

    async def _generate_direct(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> str:
        """
        Direct generation (used by batch queue or fallback).

        NOTE: This is the actual inference call. The lock is acquired
        by the batch processor, not here.
        """
        if not self._model:
            raise RuntimeError("Model not initialized")

        # Run synchronous llama.cpp call in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
                **kwargs,
            ),
        )

        return result["choices"][0]["text"]

    async def generate(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> str:
        """
        Generate completion.

        P0-P1: Routes through batching queue if enabled, otherwise uses serial lock.
        """
        if not self._model:
            raise RuntimeError("Model not initialized")

        # P0-P1: Use batching queue if enabled
        if self._batch_queue:
            return await self._batch_queue.submit(prompt, max_tokens, temperature)

        # Fallback: Serial lock (original behavior)
        async with self._lock:
            return await self._generate_direct(
                prompt, max_tokens, temperature, **kwargs
            )

    async def generate_stream(  # type: ignore[override, misc]
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> AsyncIterator[str]:
        """Generate completion with streaming."""
        if not self._model:
            raise RuntimeError("Model not initialized")

        # For streaming, we yield chunks - acquire lock at start
        async with self._lock:
            loop = asyncio.get_event_loop()
            # Get all chunks synchronously then yield asynchronously
            chunks = await loop.run_in_executor(
                None,
                lambda: list(
                    self._model(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        echo=False,
                        stream=True,
                        **kwargs,
                    )
                ),
            )
            for chunk in chunks:
                if "choices" in chunk and chunk["choices"]:
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        yield text

    async def health_check(self) -> bool:
        """Check if model is loaded and responsive."""
        if not self._model:
            return False
        try:
            # Quick inference test with async lock
            async with self._lock:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, lambda: self._model("test", max_tokens=1)
                )
            return True
        except Exception:
            return False

    def get_loaded_model(self) -> Optional[str]:
        """Return loaded model path."""
        return self._model_path

    async def shutdown(self):
        """Shutdown backend and cleanup resources."""
        if self._batch_queue:
            await self._batch_queue.stop()
            self._batch_queue = None
        self._model = None

    def get_batching_metrics(self) -> Optional[Dict[str, Any]]:
        """Get batching metrics if batching is enabled."""
        if self._batch_queue:
            return self._batch_queue.get_metrics()  # type: ignore[return-value]
        return None

    def _resolve_model_path(self) -> Optional[Path]:
        """Resolve the model path."""
        # 1. Explicit path
        if self.config.model_path:
            path = Path(self.config.model_path)
            if path.exists():
                return path

        # 2. Look in model directory
        if self.config.model_dir.exists():
            # Find any .gguf file
            gguf_files = list(self.config.model_dir.glob("*.gguf"))
            if gguf_files:
                return gguf_files[0]

        # 3. Try to download from HuggingFace
        # This would be implemented in a separate model manager
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA BACKEND (FALLBACK)
# ═══════════════════════════════════════════════════════════════════════════════


class OllamaBackend(InferenceBackendBase):
    """
    Ollama backend for fallback inference with circuit breaker and connection pooling.

    Requires Ollama server running externally.
    Circuit breaker prevents cascading failures when Ollama becomes unavailable.
    Connection pool reduces latency overhead for high-frequency requests.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self._available_models: List[str] = []
        self._current_model: Optional[str] = None

        # Initialize circuit breaker for external service protection
        self._circuit_breaker = CircuitBreaker(
            name="ollama",
            config=config.circuit_breaker,
            on_state_change=self._on_circuit_state_change,
        )

        # Initialize connection pool if enabled
        self._connection_pool: Optional[ConnectionPool] = None
        if config.enable_connection_pool:
            self._connection_pool = ConnectionPool(
                backend_type="ollama",
                endpoint=config.ollama_url,
                config=config.connection_pool,
            )

    def _on_circuit_state_change(
        self, name: str, old_state: CircuitState, new_state: CircuitState
    ) -> None:
        """Handle circuit breaker state changes."""
        if new_state == CircuitState.OPEN:
            print("[Ollama] Circuit OPEN - backend unavailable, failing fast")
        elif new_state == CircuitState.HALF_OPEN:
            print("[Ollama] Circuit HALF_OPEN - testing recovery")
        elif new_state == CircuitState.CLOSED:
            print("[Ollama] Circuit CLOSED - backend recovered")

    @property
    def backend_type(self) -> InferenceBackend:
        return InferenceBackend.OLLAMA

    async def initialize(self) -> bool:
        """Check Ollama availability and list models."""
        import urllib.error
        import urllib.request

        try:
            req = urllib.request.Request(
                f"{self.config.ollama_url}/api/tags",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:  # nosec B310 — URL from trusted InferenceConfig (localhost Ollama)
                data = json.loads(resp.read().decode())
                self._available_models = [m["name"] for m in data.get("models", [])]

                if self._available_models:
                    # Prefer capable models over tiny ones
                    preferred = ["deepseek-r1", "llama3.1", "llama3", "mistral", "qwen"]
                    self._current_model = self._available_models[0]
                    for pref in preferred:
                        for model in self._available_models:
                            if pref in model and "embed" not in model:
                                self._current_model = model
                                break
                        if self._current_model != self._available_models[0]:
                            break
                    print(f"[Ollama] Available models: {self._available_models}")
                    print(f"[Ollama] Selected model: {self._current_model}")

                    # Start connection pool if enabled
                    if self._connection_pool:
                        await self._connection_pool.start()
                        print("[Ollama] Connection pool started")

                    return True
                else:
                    print("[Ollama] No models available")
                    return False

        except Exception as e:
            print(f"[Ollama] Not available: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown backend and cleanup resources."""
        if self._connection_pool:
            await self._connection_pool.stop()
            self._connection_pool = None

    def get_connection_pool_metrics(self) -> Optional[Dict[str, Any]]:
        """Get connection pool metrics if pooling is enabled."""
        if self._connection_pool:
            # Return synchronous metrics snapshot
            return {
                "active_connections": self._connection_pool.get_active_connections(),
                "available_connections": self._connection_pool.get_available_connections(),
                "total_connections": self._connection_pool.get_total_connections(),
            }
        return None

    async def _generate_internal(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Internal generate method (unprotected)."""
        import urllib.request

        # Separate system prompt from user query if delimiter present
        system_prompt = None
        user_prompt = prompt
        if "--- QUERY ---" in prompt:
            parts = prompt.split("--- QUERY ---", 1)
            system_prompt = parts[0].strip()
            user_prompt = parts[1].strip()

        payload_dict = {
            "model": self._current_model,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if system_prompt:
            payload_dict["system"] = system_prompt

        payload = json.dumps(payload_dict).encode()

        req = urllib.request.Request(
            f"{self.config.ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        # Run blocking call in executor
        loop = asyncio.get_event_loop()
        timeout = self.config.circuit_breaker.request_timeout

        def make_request():
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 — URL from trusted InferenceConfig (localhost Ollama)
                data = json.loads(resp.read().decode())
                return data.get("response", "")

        return await loop.run_in_executor(None, make_request)

    async def generate(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> str:
        """
        Generate via Ollama API with circuit breaker protection.

        Raises:
            CircuitBreakerError: If circuit is open
        """
        assert self._circuit_breaker is not None
        return await self._circuit_breaker.execute(
            self._generate_internal, prompt, max_tokens, temperature
        )

    async def generate_stream(  # type: ignore[override, misc]
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> AsyncIterator[str]:
        """Generate with streaming via Ollama API."""
        # For simplicity, just return full response
        # Full streaming implementation would use httpx or aiohttp
        response = await self.generate(prompt, max_tokens, temperature, **kwargs)
        yield response

    async def health_check(self) -> bool:
        """Check Ollama health (bypasses circuit breaker for health checks)."""
        import urllib.error
        import urllib.request

        try:
            req = urllib.request.Request(f"{self.config.ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:  # nosec B310 — URL from trusted InferenceConfig (localhost Ollama)
                return resp.status == 200
        except Exception:
            return False

    def get_loaded_model(self) -> Optional[str]:
        return self._current_model


# ═══════════════════════════════════════════════════════════════════════════════
# LM STUDIO BACKEND (PRIMARY - v2.2.1)
# ═══════════════════════════════════════════════════════════════════════════════


class LMStudioBackend(InferenceBackendBase):
    """
    LM Studio v1 API backend - PRIMARY backend for BIZRA inference.

    Connects to LM Studio at 192.168.56.1:1234 with native /api/v1/chat
    endpoint supporting stateful chats and MCP integration.

    Circuit breaker protection prevents cascading failures when LM Studio
    becomes unavailable or overloaded.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self._client: Any = None
        self._current_model: Optional[str] = None
        self._available = False

        # Initialize circuit breaker for external service protection
        self._circuit_breaker = CircuitBreaker(
            name="lmstudio",
            config=config.circuit_breaker,
            on_state_change=self._on_circuit_state_change,
        )

    def _on_circuit_state_change(
        self, name: str, old_state: CircuitState, new_state: CircuitState
    ) -> None:
        """Handle circuit breaker state changes."""
        if new_state == CircuitState.OPEN:
            print("[LMStudio] Circuit OPEN - backend unavailable, failing fast")
        elif new_state == CircuitState.HALF_OPEN:
            print("[LMStudio] Circuit HALF_OPEN - testing recovery")
        elif new_state == CircuitState.CLOSED:
            print("[LMStudio] Circuit CLOSED - backend recovered")

    @property
    def backend_type(self) -> InferenceBackend:
        return InferenceBackend.LMSTUDIO

    async def initialize(self) -> bool:
        """
        Initialize LM Studio connection.

        REQUIREMENT: A model must be loaded in LM Studio for initialization
        to succeed. If LM Studio is reachable but no model is loaded,
        initialization fails to prevent runtime errors in generate().
        """
        if not LMSTUDIO_AVAILABLE:
            print("[LMStudio] lmstudio_backend module not available")
            return False

        assert LMStudioConfig is not None  # guarded by LMSTUDIO_AVAILABLE
        assert LMStudioClient is not None  # guarded by LMSTUDIO_AVAILABLE

        try:
            lms_config = LMStudioConfig(
                host=self.config.lmstudio_url.replace("http://", "").split(":")[0],
                port=int(self.config.lmstudio_url.split(":")[-1]),
                api_key=os.getenv("LM_API_TOKEN")
                or os.getenv("LMSTUDIO_API_KEY")
                or os.getenv("LM_STUDIO_API_KEY"),
                use_native_api=True,
                enable_mcp=True,
            )
            self._client = LMStudioClient(lms_config)
            assert self._client is not None

            if await self._client.connect():
                # Check for loaded models - REQUIRED for successful initialization
                models = await self._client.list_models()
                loaded = [m for m in models if m.loaded]
                if loaded:
                    self._current_model = loaded[0].id
                    self._available = True
                    print(f"[LMStudio] Connected with model: {self._current_model}")
                    return True
                else:
                    # FAIL: No model loaded - generate() would fail
                    print(
                        f"[LMStudio] Connected but NO MODEL LOADED ({len(models)} available)"
                    )
                    print("[LMStudio] Load a model in LM Studio to enable this backend")
                    self._available = False
                    return False
            else:
                print("[LMStudio] Connection failed")
                return False
        except Exception as e:
            print(f"[LMStudio] Initialization error: {e}")
            return False

    async def _generate_internal(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Internal generate method (unprotected)."""
        if not self._client or not self._available:
            raise RuntimeError("LM Studio not initialized")

        # Separate system prompt from user query if delimiter present
        messages = []
        if "--- QUERY ---" in prompt:
            parts = prompt.split("--- QUERY ---", 1)
            messages.append(ChatMessage(role="system", content=parts[0].strip()))
            messages.append(ChatMessage(role="user", content=parts[1].strip()))
        else:
            messages.append(ChatMessage(role="user", content=prompt))

        response = await self._client.chat(
            messages=messages,
            model=self._current_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.content

    async def generate(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> str:
        """
        Generate via LM Studio API with circuit breaker protection.

        Raises:
            CircuitBreakerError: If circuit is open
        """
        assert self._circuit_breaker is not None
        return await self._circuit_breaker.execute(
            self._generate_internal, prompt, max_tokens, temperature
        )

    async def generate_stream(  # type: ignore[override, misc]
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> AsyncIterator[str]:
        """Generate with streaming."""
        # Simplified: return full response
        response = await self.generate(prompt, max_tokens, temperature, **kwargs)
        yield response

    async def health_check(self) -> bool:
        """Check LM Studio health (bypasses circuit breaker for health checks)."""
        if not self._client:
            return False
        try:
            models = await self._client.list_models()
            return len(models) > 0
        except Exception:
            return False

    def get_loaded_model(self) -> Optional[str]:
        return self._current_model
