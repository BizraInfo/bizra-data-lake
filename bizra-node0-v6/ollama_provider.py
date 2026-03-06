"""
BIZRA Ollama Provider — Local-First LLM Inference
══════════════════════════════════════════════════

Zero cloud dependency. Every inference runs on NODE0's hardware.
This is the economic moat: C_LLM ≈ $0 (Corollary 4.2).

Features:
  - Model fallback chain: primary → secondary → tertiary
  - Circuit breaker: auto-disable failing models
  - Health monitoring: latency tracking, success rate
  - Structured output: JSON mode for agent responses
  - Timeout enforcement: per-tier latency budgets

Architecture:
  MissionPipeline → Coder agent → OllamaProvider → local model
  All inference stays on-device. Privacy by construction.

Constitution reference: §7 [hhmm.complexity_tiers] for latency budgets
"""

from __future__ import annotations

import json
import time
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("bizra.ollama")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_BASE_URL = "http://localhost:11434"

# Model fallback chain: try in order until one succeeds
DEFAULT_MODEL_CHAIN = [
    "phi3:mini",        # Primary: fast, small, good quality
    "llama3.2:3b",      # Secondary: larger, better reasoning
    "mistral:7b",       # Tertiary: strongest local model
    "qwen2.5:3b",       # Quaternary: alternative family
]

DEFAULT_TIMEOUT_S = 30.0
CIRCUIT_BREAKER_THRESHOLD = 3      # Failures before tripping
CIRCUIT_BREAKER_RESET_S = 60.0     # Seconds before retry after trip


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════════


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, blocked
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitBreaker:
    """Per-model circuit breaker. Prevents hammering failing models."""
    model: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_at: float = 0.0
    threshold: int = CIRCUIT_BREAKER_THRESHOLD
    reset_timeout_s: float = CIRCUIT_BREAKER_RESET_S

    def record_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_at = time.time()
        if self.failure_count >= self.threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPEN for model '{self.model}' "
                f"after {self.failure_count} failures"
            )

    def is_available(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_at
            if elapsed > self.reset_timeout_s:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN: allow one attempt
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class InferenceResult:
    """Result of a single LLM inference call."""
    text: str
    model: str
    latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    is_fallback: bool           # True if not the primary model
    fallback_chain: list[str]   # Models attempted before success
    success: bool
    error: str | None = None

    def as_evidence(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "latency_ms": self.latency_ms,
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": self.tokens_per_second,
            "is_fallback": self.is_fallback,
            "fallback_chain": self.fallback_chain,
            "success": self.success,
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA PROVIDER
# ═══════════════════════════════════════════════════════════════════════════════


class OllamaProvider:
    """
    Local-first LLM inference provider with circuit breaker.

    Tries models in the fallback chain until one succeeds.
    Failed models are circuit-broken to avoid repeated timeouts.
    Health metrics tracked per model.

    Usage:
        provider = OllamaProvider()
        result = provider.generate("What is BIZRA?")
        if result.success:
            print(result.text)
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model_chain: list[str] | None = None,
        default_timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_chain = model_chain or DEFAULT_MODEL_CHAIN
        self.default_timeout_s = default_timeout_s

        # Per-model circuit breakers
        self._breakers: dict[str, CircuitBreaker] = {
            model: CircuitBreaker(model=model)
            for model in self.model_chain
        }

        # Per-model health metrics
        self._metrics: dict[str, ModelMetrics] = {
            model: ModelMetrics(model=model)
            for model in self.model_chain
        }

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        timeout_s: float | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> InferenceResult:
        """
        Generate a response using the model fallback chain.

        Tries each model in the chain until one succeeds.
        Circuit-broken models are skipped automatically.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.
            timeout_s: Timeout per model attempt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            json_mode: If True, request JSON output format.

        Returns:
            InferenceResult with the generated text and metadata.
        """
        timeout = timeout_s or self.default_timeout_s
        attempted = []

        for i, model in enumerate(self.model_chain):
            breaker = self._breakers.get(model)
            if breaker and not breaker.is_available():
                attempted.append(f"{model}(circuit-open)")
                continue

            attempted.append(model)
            result = self._try_model(
                model=model,
                prompt=prompt,
                system=system,
                timeout_s=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
                is_fallback=(i > 0),
                attempted=list(attempted),
            )

            if result.success:
                if breaker:
                    breaker.record_success()
                self._metrics[model].record_success(result.latency_ms)
                return result
            else:
                if breaker:
                    breaker.record_failure()
                self._metrics[model].record_failure()

        # All models failed
        return InferenceResult(
            text="",
            model="none",
            latency_ms=0,
            tokens_generated=0,
            tokens_per_second=0,
            is_fallback=True,
            fallback_chain=attempted,
            success=False,
            error=f"All {len(attempted)} models failed: {attempted}",
        )

    def _try_model(
        self,
        model: str,
        prompt: str,
        system: str | None,
        timeout_s: float,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        is_fallback: bool,
        attempted: list[str],
    ) -> InferenceResult:
        """Attempt inference with a single model."""
        start = time.monotonic()

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        if json_mode:
            payload["format"] = "json"

        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode())

            elapsed_ms = (time.monotonic() - start) * 1000
            response_text = body.get("response", "")
            eval_count = body.get("eval_count", len(response_text.split()))
            eval_duration_ns = body.get("eval_duration", elapsed_ms * 1e6)
            tps = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0

            return InferenceResult(
                text=response_text,
                model=model,
                latency_ms=round(elapsed_ms, 2),
                tokens_generated=eval_count,
                tokens_per_second=round(tps, 2),
                is_fallback=is_fallback,
                fallback_chain=attempted,
                success=True,
            )

        except urllib.error.URLError as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.warning(f"Model '{model}' connection failed: {e}")
            return InferenceResult(
                text="", model=model, latency_ms=round(elapsed_ms, 2),
                tokens_generated=0, tokens_per_second=0,
                is_fallback=is_fallback, fallback_chain=attempted,
                success=False, error=f"Connection error: {e}",
            )
        except TimeoutError:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.warning(f"Model '{model}' timed out after {timeout_s}s")
            return InferenceResult(
                text="", model=model, latency_ms=round(elapsed_ms, 2),
                tokens_generated=0, tokens_per_second=0,
                is_fallback=is_fallback, fallback_chain=attempted,
                success=False, error=f"Timeout after {timeout_s}s",
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.warning(f"Model '{model}' failed: {e}")
            return InferenceResult(
                text="", model=model, latency_ms=round(elapsed_ms, 2),
                tokens_generated=0, tokens_per_second=0,
                is_fallback=is_fallback, fallback_chain=attempted,
                success=False, error=str(e),
            )

    # ── Chat API (for multi-turn) ──

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        timeout_s: float | None = None,
        temperature: float = 0.7,
    ) -> InferenceResult:
        """
        Multi-turn chat inference using Ollama's /api/chat endpoint.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
            model: Specific model to use (skips fallback chain).
            timeout_s: Timeout for the request.
            temperature: Sampling temperature.
        """
        timeout = timeout_s or self.default_timeout_s
        target_model = model or self.model_chain[0]

        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        start = time.monotonic()
        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode())

            elapsed_ms = (time.monotonic() - start) * 1000
            msg = body.get("message", {})
            response_text = msg.get("content", "")
            eval_count = body.get("eval_count", len(response_text.split()))

            return InferenceResult(
                text=response_text,
                model=target_model,
                latency_ms=round(elapsed_ms, 2),
                tokens_generated=eval_count,
                tokens_per_second=0,
                is_fallback=False,
                fallback_chain=[target_model],
                success=True,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            return InferenceResult(
                text="", model=target_model,
                latency_ms=round(elapsed_ms, 2),
                tokens_generated=0, tokens_per_second=0,
                is_fallback=False, fallback_chain=[target_model],
                success=False, error=str(e),
            )

    # ── Health & Introspection ──

    def list_models(self) -> list[str]:
        """Query Ollama for available models."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode())
            return [m["name"] for m in body.get("models", [])]
        except Exception:
            return []

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def health(self) -> dict[str, Any]:
        """Complete provider health report."""
        available = self.is_available()
        models = self.list_models() if available else []
        return {
            "server_available": available,
            "base_url": self.base_url,
            "available_models": models,
            "model_chain": self.model_chain,
            "circuit_breakers": {
                model: {
                    "state": b.state.value,
                    "failures": b.failure_count,
                }
                for model, b in self._breakers.items()
            },
            "metrics": {
                model: m.as_dict()
                for model, m in self._metrics.items()
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PER-MODEL METRICS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelMetrics:
    """Runtime health metrics for a single model."""
    model: str
    total_requests: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0

    def record_success(self, latency_ms: float):
        self.total_requests += 1
        self.successes += 1
        self.total_latency_ms += latency_ms

    def record_failure(self):
        self.total_requests += 1
        self.failures += 1

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successes / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        if self.successes == 0:
            return 0.0
        return self.total_latency_ms / self.successes

    def as_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "failures": self.failures,
        }
