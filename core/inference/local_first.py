"""
Local-First Backend Detection & Selection

Detects available local inference backends (LM Studio, Ollama, llama.cpp)
and returns the best available option. Enables zero-token operation on Node0.

Architecture:
  Priority 1: LM Studio (192.168.56.1:1234) - RTX 4090 native
  Priority 2: Ollama (localhost:11434) - Fallback
  Priority 3: llama.cpp (embedded) - Offline-capable edge

Standing on Giants:
  - Amdahl's Law: Local = zero latency overhead
  - Shannon: Maximize local SNR, minimize noise from remote calls
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


class LocalBackend(str, Enum):
    """Available local backends (zero-token operation)."""

    LMSTUDIO = "lmstudio"
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"
    NONE = "none"


@dataclass
class BackendStatus:
    """Status of a local backend."""

    backend: LocalBackend
    available: bool
    latency_ms: float
    reason: str = ""


class LocalFirstDetector:
    """Detects and selects best available local backend."""

    # Configuration (env vars override defaults)
    LM_STUDIO_HOST = os.getenv("LMSTUDIO_HOST", "192.168.56.1")
    LM_STUDIO_PORT = int(os.getenv("LMSTUDIO_PORT", "1234"))
    OLLAMA_HOST = os.getenv("OLLAMA_HOST_ADDR", "localhost")
    OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))

    # Health check timeouts (ms)
    PROBE_TIMEOUT = 2.0  # 2s timeout per backend

    @classmethod
    async def detect_available(cls) -> List[BackendStatus]:
        """
        Probe all local backends in parallel.

        Returns:
            List of BackendStatus, ordered by availability & latency
        """
        probes = [
            cls._probe_lmstudio(),
            cls._probe_ollama(),
            cls._probe_llamacpp(),
        ]

        results = await asyncio.gather(*probes, return_exceptions=True)
        statuses = []

        for result in results:
            if isinstance(result, BackendStatus):
                statuses.append(result)

        # Sort by availability (available first), then by latency
        return sorted(statuses, key=lambda s: (not s.available, s.latency_ms))

    @classmethod
    async def select_best(cls) -> LocalBackend:
        """
        Select the best available backend.

        Returns:
            LocalBackend enum (LMSTUDIO, OLLAMA, LLAMACPP, or NONE)
        """
        statuses = await cls.detect_available()

        for status in statuses:
            if status.available:
                logger.info(
                    f"Local-first: Selected {status.backend.value} "
                    f"({status.latency_ms:.1f}ms)"
                )
                return status.backend

        logger.warning("Local-first: No local backends available")
        return LocalBackend.NONE

    @classmethod
    async def _probe_lmstudio(cls) -> BackendStatus:
        """Check LM Studio availability (192.168.56.1:1234)."""
        import time

        start = time.perf_counter()
        try:
            import httpx

            async with httpx.AsyncClient(timeout=cls.PROBE_TIMEOUT) as client:
                url = f"http://{cls.LM_STUDIO_HOST}:{cls.LM_STUDIO_PORT}/api/v1/models"
                resp = await client.get(url)
                latency = (time.perf_counter() - start) * 1000

                if resp.status_code == 200:
                    return BackendStatus(
                        backend=LocalBackend.LMSTUDIO,
                        available=True,
                        latency_ms=latency,
                        reason="LM Studio v1 API responsive",
                    )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000

        return BackendStatus(
            backend=LocalBackend.LMSTUDIO,
            available=False,
            latency_ms=latency,
            reason=f"LM Studio unreachable ({str(exc)[:40]})",
        )

    @classmethod
    async def _probe_ollama(cls) -> BackendStatus:
        """Check Ollama availability (localhost:11434)."""
        import time

        start = time.perf_counter()
        try:
            import httpx

            async with httpx.AsyncClient(timeout=cls.PROBE_TIMEOUT) as client:
                url = f"http://{cls.OLLAMA_HOST}:{cls.OLLAMA_PORT}/api/tags"
                resp = await client.get(url)
                latency = (time.perf_counter() - start) * 1000

                if resp.status_code == 200:
                    return BackendStatus(
                        backend=LocalBackend.OLLAMA,
                        available=True,
                        latency_ms=latency,
                        reason="Ollama API responsive",
                    )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000

        return BackendStatus(
            backend=LocalBackend.OLLAMA,
            available=False,
            latency_ms=latency,
            reason=f"Ollama unreachable ({str(exc)[:40]})",
        )

    @classmethod
    async def _probe_llamacpp(cls) -> BackendStatus:
        """Check llama.cpp availability (embedded)."""
        import time

        start = time.perf_counter()
        try:
            # Try to import and initialize llama.cpp
            from .backends.llamacpp import LlamaCppBackend
            from .gateway import InferenceConfig

            config = InferenceConfig()
            backend = LlamaCppBackend(config)
            available = await backend.health_check()

            latency = (time.perf_counter() - start) * 1000

            return BackendStatus(
                backend=LocalBackend.LLAMACPP,
                available=available,
                latency_ms=latency,
                reason=(
                    "llama.cpp embedded backend"
                    if available
                    else "llama.cpp unavailable"
                ),
            )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000

        return BackendStatus(
            backend=LocalBackend.LLAMACPP,
            available=False,
            latency_ms=latency,
            reason=f"llama.cpp check failed ({str(exc)[:40]})",
        )


async def get_local_first_backend() -> LocalBackend:
    """
    Convenience function: Auto-detect best local backend.

    Usage:
        backend = await get_local_first_backend()
        if backend != LocalBackend.NONE:
            # Use backend for zero-token inference
    """
    return await LocalFirstDetector.select_best()
