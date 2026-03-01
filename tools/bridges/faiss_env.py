"""FAISS environment detection and GPU validation for BIZRA platform.

Standing on Giants: Meta FAISS, Google DeepMind XTR, Stanford ColBERTv2

This module provides a centralized, import-safe mechanism for detecting FAISS
installation variant (cpu vs gpu), GPU availability, and version metadata.
Used by warp_bridge, warp_retriever, hypergraph_engine, and collection_indexer.

Architecture:
    Import this module once at startup.  The module-level constants are
    populated eagerly so downstream consumers can branch cheaply::

        from faiss_env import FAISS_AVAILABLE, is_gpu_ready, faiss_summary

        if is_gpu_ready():
            # GPU-accelerated path
        else:
            # CPU fallback
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass

log = logging.getLogger("BIZRA.faiss_env")

# ---------------------------------------------------------------------------
# FAISS import
# ---------------------------------------------------------------------------

_faiss = None
try:
    import faiss as _faiss  # type: ignore[import-untyped]
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Torch / CUDA detection (optional — only used for GPU validation)
# ---------------------------------------------------------------------------

_torch = None
_cuda_available = False
try:
    import torch as _torch

    _cuda_available = _torch.cuda.is_available()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FAISSEnvironment:
    """Immutable snapshot of the FAISS runtime environment."""

    available: bool
    version: str
    has_gpu_api: bool
    cuda_available: bool
    gpu_functional: bool
    num_gpus: int
    os_name: str
    error: str | None = None

    @property
    def summary(self) -> str:
        """Human-readable one-liner."""
        if not self.available:
            return "FAISS not installed"
        variant = "GPU" if self.gpu_functional else "CPU"
        return (
            f"FAISS {self.version} ({variant}) | "
            f"GPUs: {self.num_gpus} | CUDA: {self.cuda_available} | "
            f"OS: {self.os_name}"
        )


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------


def _detect() -> FAISSEnvironment:
    """Probe the runtime and return an immutable environment snapshot."""
    os_name = platform.system()

    if _faiss is None:
        return FAISSEnvironment(
            available=False,
            version="N/A",
            has_gpu_api=False,
            cuda_available=_cuda_available,
            gpu_functional=False,
            num_gpus=0,
            os_name=os_name,
            error="faiss not importable",
        )

    version = getattr(_faiss, "__version__", "unknown")
    has_gpu_api = hasattr(_faiss, "StandardGpuResources")
    num_gpus = 0
    gpu_functional = False
    error = None

    if has_gpu_api and _cuda_available:
        try:
            res = _faiss.StandardGpuResources()
            num_gpus = _faiss.get_num_gpus()
            gpu_functional = num_gpus > 0
            del res
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            log.warning("FAISS GPU API present but non-functional: %s", exc)

    return FAISSEnvironment(
        available=True,
        version=version,
        has_gpu_api=has_gpu_api,
        cuda_available=_cuda_available,
        gpu_functional=gpu_functional,
        num_gpus=num_gpus,
        os_name=os_name,
        error=error,
    )


# ---------------------------------------------------------------------------
# Lazy detection — probes GPU only on first access, not at import time.
# This avoids startup side effects in modules that import faiss_env but
# may never need GPU information (e.g. search-only, CPU-only paths).
# ---------------------------------------------------------------------------

_env_cache: FAISSEnvironment | None = None


def _get_env() -> FAISSEnvironment:
    """Return the cached environment snapshot, detecting on first call."""
    global _env_cache
    if _env_cache is None:
        _env_cache = _detect()
        log.info("FAISS env: %s", _env_cache.summary)
    return _env_cache


# Public API — thin wrappers that trigger lazy detection on first use.


def get_env() -> FAISSEnvironment:
    """Return the (lazily detected) FAISS environment snapshot."""
    return _get_env()


# Backward-compatible module-level constants.
# These are evaluated eagerly *only* for the lightweight faiss import check
# (no GPU probe).  GPU readiness is deferred to get_env().
FAISS_AVAILABLE: bool = _faiss is not None
"""True if ``import faiss`` succeeded (lightweight, no GPU probe)."""


def is_gpu_ready() -> bool:
    """Check if FAISS GPU resources can be allocated (lazy, cached)."""
    return _get_env().gpu_functional


# Keep backward compat — but as a function so the GPU probe is lazy.
FAISS_GPU_READY: bool = False  # Sentinel; use is_gpu_ready() for accurate result.

FAISS_VERSION: str = getattr(_faiss, "__version__", "N/A") if _faiss else "N/A"
"""Installed FAISS version string, or ``'N/A'``."""


def faiss_summary() -> str:
    """Return a human-readable one-liner about the FAISS environment."""
    return _get_env().summary
