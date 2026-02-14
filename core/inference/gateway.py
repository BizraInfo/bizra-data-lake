"""
BIZRA INFERENCE GATEWAY (PR1 IMPLEMENTATION)
═══════════════════════════════════════════════════════════════════════════════

Embedded LLM inference with fail-closed semantics and circuit breaker resilience.

Priority order (v2.2.1 - LM Studio as primary):
1. LM Studio v1 (192.168.56.1:1234) - RTX 4090 optimized, PRIMARY
2. Ollama (localhost:11434) - fallback
3. llama.cpp (embedded) - offline/edge
4. DENY (fail-closed)

This is the core of thermodynamic entropy reduction.
Local inference = local world model = sovereignty.

Standing on Giants:
- Nygard (2007): Release It! - Circuit breaker pattern for resilient systems
- Netflix Hystrix: Latency and fault tolerance library
- Fowler (2014): CircuitBreaker pattern documentation

Created: 2026-01-29 | BIZRA Sovereignty
Updated: 2026-02-01 | LM Studio v1 API as primary backend
Updated: 2026-02-04 | Circuit breaker pattern for backend resilience
Updated: 2026-02-04 | Rate limiting with token bucket algorithm (RFC 6585)
Updated: 2026-02-09 | Refactored into submodules (_types, _connection_pool,
                     | _resilience, _batching, _backends) for maintainability.
                     | This file is now a thin façade + InferenceGateway class.
Principle: لا نفترض — We do not assume.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Optional, Union

from ._backends import (  # noqa: F401 — re-exports
    LMSTUDIO_AVAILABLE,
    InferenceBackendBase,
    LlamaCppBackend,
    LMStudioBackend,
    OllamaBackend,
)
from ._batching import (  # noqa: F401 — re-exports
    BatchingInferenceQueue,
    PendingRequest,
)
from ._connection_pool import (  # noqa: F401 — re-exports
    ConnectionPool,
    ConnectionPoolConfig,
    ConnectionPoolMetrics,
    PooledConnection,
    PooledHttpClient,
)
from ._resilience import (  # noqa: F401 — re-exports
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerMetrics,
    RateLimiter,
    RateLimitError,
)
from ._types import (  # noqa: F401 — re-exports; TypedDicts; Protocol; Enums; Config dataclasses; Core dataclasses
    CACHE_DIR,
    DEFAULT_MODEL_DIR,
    TIER_CONFIGS,
    BackendGenerateFn,
    BatchingMetrics,
    CircuitBreakerConfig,
    CircuitMetrics,
    CircuitState,
    ComputeTier,
    GatewayStats,
    HealthData,
    InferenceBackend,
    InferenceConfig,
    InferenceResult,
    InferenceStatus,
    RateLimiterConfig,
    RateLimiterMetrics,
    TaskComplexity,
)

# ─── Re-exports from submodules (backward-compat) ────────────────────────────
# Every symbol that was historically importable from ``core.inference.gateway``
# is re-exported here so that no external ``from core.inference.gateway import X``
# statement breaks.


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE GATEWAY
# ═══════════════════════════════════════════════════════════════════════════════


class InferenceGateway:
    """
    Tiered inference gateway with fail-closed semantics, circuit breaker resilience,
    and rate limiting protection.

    Routes requests to appropriate compute tier based on complexity.
    Provides fallback chain when primary backend unavailable.
    Circuit breakers protect against cascading failures from external services.
    Rate limiting prevents request overload and ensures fair resource allocation.

    Fail-closed: If no backend available and require_local=True, deny request.

    Standing on Giants:
    - Nygard (2007): Release It! - Circuit breaker pattern
    - Netflix Hystrix: Fallback and resilience patterns
    - RFC 6585: HTTP 429 Too Many Requests
    - Token Bucket Algorithm: Rate limiting
    """

    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        self.config: InferenceConfig = config or InferenceConfig()
        self.status: InferenceStatus = InferenceStatus.COLD

        # Backends by tier
        self._backends: dict[ComputeTier, InferenceBackendBase] = {}
        self._active_backend: Optional[InferenceBackendBase] = None

        # Fallback chain (ordered list of backends to try)
        self._fallback_backends: list[InferenceBackendBase] = []

        # Rate limiter (RFC 6585 / Token Bucket)
        self._rate_limiter: Optional[RateLimiter] = None
        if self.config.enable_rate_limiting:
            self._rate_limiter = RateLimiter(
                config=self.config.rate_limiter, name="gateway"
            )
            print(
                f"[Gateway] Rate limiting enabled "
                f"(rate={self.config.rate_limiter.tokens_per_second}/s, "
                f"max={self.config.rate_limiter.max_tokens}, "
                f"burst={self.config.rate_limiter.burst_size})"
            )

        # Metrics
        self._total_requests: int = 0
        self._total_tokens: int = 0
        self._total_latency_ms: float = 0.0
        self._circuit_breaker_trips: int = 0
        self._fallback_invocations: int = 0
        self._rate_limit_rejections: int = 0

    async def initialize(self) -> bool:
        """
        Initialize the gateway and backends.

        Priority order (v2.2.1):
        - When require_local=True (fail-closed sovereign mode):
          1. llama.cpp ONLY (embedded, offline-capable)
          2. DENY (fail-closed)

        - When require_local=False (fallback-enabled mode):
          1. LM Studio v1 (192.168.56.1:1234) - PRIMARY, RTX 4090 optimized
          2. Configured fallbacks (ollama, lmstudio) in order
          3. llama.cpp (embedded) - offline/edge sovereign
          4. DENY (fail-closed)

        Returns True if at least one backend is available.
        """
        self.status = InferenceStatus.WARMING

        # SECURITY: When require_local=True, ONLY try llama.cpp (embedded)
        # External services (LM Studio, Ollama) are NOT considered "local"
        # as they require network connectivity and are not self-contained.
        if self.config.require_local:
            print("[Gateway] require_local=True: Only trying embedded backends")
            llamacpp = LlamaCppBackend(self.config)
            if await llamacpp.initialize():
                self._backends[ComputeTier.LOCAL] = llamacpp
                self._backends[ComputeTier.EDGE] = llamacpp
                self._active_backend = llamacpp
                self.status = InferenceStatus.READY
                print("[Gateway] llama.cpp backend ready (SOVEREIGN MODE)")
                return True

            # Fail-closed: No fallbacks allowed when require_local=True
            self.status = InferenceStatus.OFFLINE
            print(
                "[Gateway] No embedded backend available (OFFLINE MODE - FAIL-CLOSED)"
            )
            return False

        # --- Fallback-enabled mode (require_local=False) ---

        # 1. Try LM Studio first (PRIMARY - RTX 4090 optimized)
        if LMSTUDIO_AVAILABLE:
            lmstudio = LMStudioBackend(self.config)
            if await lmstudio.initialize():
                self._backends[ComputeTier.LOCAL] = lmstudio
                self._active_backend = lmstudio
                self.status = InferenceStatus.READY
                print("[Gateway] LM Studio v1 backend ready (PRIMARY MODE)")
                return True

        # 2. Try configured fallbacks in order
        for fallback in self.config.fallbacks:
            if fallback == "ollama":
                ollama = OllamaBackend(self.config)
                if await ollama.initialize():
                    self._backends[ComputeTier.LOCAL] = ollama
                    self._active_backend = ollama
                    self.status = InferenceStatus.DEGRADED
                    print("[Gateway] Ollama fallback ready (DEGRADED MODE)")
                    return True

            elif fallback == "lmstudio" and LMSTUDIO_AVAILABLE:
                # LM Studio as fallback (different config or retry)
                lmstudio = LMStudioBackend(self.config)
                if await lmstudio.initialize():
                    self._backends[ComputeTier.LOCAL] = lmstudio
                    self._active_backend = lmstudio
                    self.status = InferenceStatus.DEGRADED
                    print("[Gateway] LM Studio fallback ready (DEGRADED MODE)")
                    return True

        # 3. Try llama.cpp (offline/edge - embedded, sovereign)
        llamacpp = LlamaCppBackend(self.config)
        if await llamacpp.initialize():
            self._backends[ComputeTier.LOCAL] = llamacpp
            self._backends[ComputeTier.EDGE] = llamacpp
            self._active_backend = llamacpp
            self.status = InferenceStatus.READY
            print("[Gateway] llama.cpp backend ready (SOVEREIGN MODE)")
            return True

        # 4. Fail-closed
        self.status = InferenceStatus.OFFLINE
        print("[Gateway] No backend available (OFFLINE MODE)")
        return False

    def estimate_complexity(self, prompt: str) -> TaskComplexity:
        """
        Estimate task complexity for routing decisions.

        Simple heuristics for now. Could be replaced with classifier.
        """
        words = prompt.split()
        input_tokens = len(words) * 1.3  # Rough estimate

        # Heuristics for reasoning depth
        reasoning_keywords = ["why", "how", "explain", "analyze", "compare", "prove"]
        reasoning_depth = sum(
            1 for w in words if w.lower() in reasoning_keywords
        ) / max(len(words), 1)

        # Heuristics for domain specificity
        technical_keywords = [
            "algorithm",
            "equation",
            "theorem",
            "protocol",
            "architecture",
        ]
        domain_specificity = sum(
            1 for w in words if w.lower() in technical_keywords
        ) / max(len(words), 1)

        return TaskComplexity(
            input_tokens=int(input_tokens),
            estimated_output_tokens=min(int(input_tokens * 2), 2048),
            reasoning_depth=min(reasoning_depth * 5, 1.0),
            domain_specificity=min(domain_specificity * 5, 1.0),
        )

    def route(self, complexity: TaskComplexity) -> ComputeTier:
        """
        Route task to appropriate compute tier.

        EDGE: complexity < 0.3
        LOCAL: 0.3 <= complexity < 0.8
        POOL: complexity >= 0.8
        """
        score = complexity.score

        if score < 0.3:
            return ComputeTier.EDGE
        elif score < 0.8:
            return ComputeTier.LOCAL
        else:
            return ComputeTier.POOL

    async def infer(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        tier: Optional[ComputeTier] = None,
        stream: bool = False,
        client_id: Optional[str] = None,
    ) -> Union[InferenceResult, AsyncIterator[str]]:
        """
        Run inference on prompt with rate limiting, circuit breaker protection,
        and automatic failover.

        Rate limiting is applied first (RFC 6585). If rate limit exceeded,
        raises RateLimitError with HTTP 429 status and retry-after hint.

        When a backend's circuit breaker is open, the gateway automatically attempts
        fallback backends in order. This implements the Netflix Hystrix fallback pattern.

        Fail-closed: Raises RuntimeError if no backend available (all circuits open
        or no backends configured).

        Args:
            prompt: The prompt to run inference on
            max_tokens: Maximum tokens to generate (uses config default if None)
            temperature: Sampling temperature (0.0-1.0)
            tier: Force a specific compute tier (auto-routes if None)
            stream: If True, return AsyncIterator instead of InferenceResult
            client_id: Optional client identifier for per-client rate limiting

        Returns:
            InferenceResult or AsyncIterator[str] if stream=True

        Raises:
            RateLimitError: If rate limit exceeded (HTTP 429)
            RuntimeError: If no backend available or all circuits open

        Standing on Giants:
        - Nygard (2007): Release It! - Fail fast pattern
        - Netflix Hystrix: Fallback and bulkhead patterns
        - RFC 6585: HTTP 429 Too Many Requests
        """
        # Rate limiting check (RFC 6585)
        if self._rate_limiter:
            try:
                await self._rate_limiter.acquire(client_id=client_id)
            except RateLimitError as e:
                self._rate_limit_rejections += 1
                print(
                    f"[Gateway] Rate limit exceeded for client '{e.client_id}' "
                    f"(tokens={e.current_tokens:.2f}, retry_after={e.retry_after:.2f}s)"
                )
                raise

        # Check availability
        if self.status == InferenceStatus.OFFLINE:
            raise RuntimeError("Inference denied: no backend available (fail-closed)")

        if not self._active_backend:
            raise RuntimeError("Inference denied: no active backend")

        # Estimate complexity and route
        complexity = self.estimate_complexity(prompt)
        target_tier = tier or self.route(complexity)

        # Get backend for tier (fallback to active)
        primary_backend = self._backends.get(target_tier, self._active_backend)

        # Build fallback chain: primary -> other tier backends -> fallback backends
        backends_to_try = [primary_backend]
        for tier_backend in self._backends.values():
            if tier_backend not in backends_to_try:
                backends_to_try.append(tier_backend)
        for fallback_backend in self._fallback_backends:
            if fallback_backend not in backends_to_try:
                backends_to_try.append(fallback_backend)

        # Run inference with circuit breaker failover
        start_time = time.time()
        max_tokens = max_tokens or self.config.max_tokens

        if stream:
            # For streaming, we cannot easily failover mid-stream
            # Return generator from primary backend (circuit breaker will raise on open)
            return primary_backend.generate_stream(prompt, max_tokens, temperature)  # type: ignore[return-value]

        # Try each backend in order until one succeeds
        last_error: Optional[Exception] = None
        used_backend: Optional[InferenceBackendBase] = None

        for backend in backends_to_try:
            try:
                response = await backend.generate(prompt, max_tokens, temperature)
                used_backend = backend

                # Track if we used a fallback
                if backend != primary_backend:
                    self._fallback_invocations += 1
                    print(
                        f"[Gateway] Fallback to {backend.backend_type.value} succeeded"
                    )

                break

            except CircuitBreakerError as e:
                # Circuit is open - try next backend
                self._circuit_breaker_trips += 1
                print(
                    f"[Gateway] Circuit open for {e.backend_name}, trying fallback..."
                )
                last_error = e
                continue

            except Exception as e:
                # Other error - backend failure, try next
                print(f"[Gateway] Backend {backend.backend_type.value} failed: {e}")
                last_error = e
                continue

        if used_backend is None:
            # All backends failed
            raise RuntimeError(
                f"Inference denied: all backends unavailable. "
                f"Last error: {last_error}"
            )

        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        tokens_generated = len(response.split())  # Rough estimate
        tokens_per_second = (
            tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0
        )

        # Update stats
        self._total_requests += 1
        self._total_tokens += tokens_generated
        self._total_latency_ms += latency_ms

        return InferenceResult(
            content=response,
            model=used_backend.get_loaded_model() or "unknown",
            backend=used_backend.backend_type,
            tier=target_tier,
            tokens_generated=tokens_generated,
            tokens_per_second=round(tokens_per_second, 2),
            latency_ms=round(latency_ms, 2),
        )

    async def health(self) -> dict[str, Any]:
        """
        Get gateway health status including circuit breaker and rate limiter metrics.

        Returns comprehensive health information:
        - Gateway status and active backend
        - Per-backend health checks
        - Circuit breaker states and metrics
        - Rate limiter metrics (if enabled)
        - Batching metrics (if enabled)
        - Request/token statistics
        """
        backends_health: dict[str, bool] = {}
        circuit_breakers: dict[str, dict[str, Any]] = {}

        for tier, backend in self._backends.items():
            tier_name = tier.value
            backends_health[tier_name] = await backend.health_check()

            # Collect circuit breaker metrics
            cb_metrics = backend.get_circuit_metrics()
            if cb_metrics:
                circuit_breakers[f"{tier_name}_{backend.backend_type.value}"] = dict(
                    cb_metrics
                )

        # P0-P1: Include batching metrics if available
        batching_metrics: Optional[dict[str, Any]] = None
        if self._active_backend and hasattr(
            self._active_backend, "get_batching_metrics"
        ):
            batching_metrics = self._active_backend.get_batching_metrics()  # type: ignore[union-attr]

        health_data: dict[str, Any] = {
            "status": self.status.value,
            "active_backend": (
                self._active_backend.backend_type.value
                if self._active_backend
                else None
            ),
            "active_model": (
                self._active_backend.get_loaded_model()
                if self._active_backend
                else None
            ),
            "backends": backends_health,
            "stats": {
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "avg_latency_ms": (
                    self._total_latency_ms / self._total_requests
                    if self._total_requests > 0
                    else 0.0
                ),
                "circuit_breaker_trips": self._circuit_breaker_trips,
                "fallback_invocations": self._fallback_invocations,
                "rate_limit_rejections": self._rate_limit_rejections,
            },
        }

        if circuit_breakers:
            health_data["circuit_breakers"] = circuit_breakers

        # Rate limiter metrics (RFC 6585)
        if self._rate_limiter:
            health_data["rate_limiter"] = self._rate_limiter.get_metrics()

        if batching_metrics:
            health_data["batching"] = batching_metrics

        # P1: Include connection pool metrics if available
        connection_pool_metrics: dict[str, Any] = {}
        for tier, backend in self._backends.items():
            if hasattr(backend, "get_connection_pool_metrics"):
                pool_metrics = backend.get_connection_pool_metrics()
                if pool_metrics:
                    connection_pool_metrics[
                        f"{tier.value}_{backend.backend_type.value}"
                    ] = pool_metrics

        if connection_pool_metrics:
            health_data["connection_pools"] = connection_pool_metrics

        return health_data

    def get_circuit_breaker_summary(self) -> dict[str, dict[str, Any]]:
        """
        Get a summary of all circuit breaker states.

        Returns dict mapping backend names to their circuit breaker states.
        Useful for monitoring dashboards and alerting.
        """
        summary: dict[str, dict[str, Any]] = {}
        for tier, backend in self._backends.items():
            cb = backend.get_circuit_breaker()
            if cb:
                metrics = cb.get_metrics()
                summary[f"{tier.value}_{backend.backend_type.value}"] = {
                    "state": metrics.state.value,
                    "failure_count": metrics.failure_count,
                    "last_failure": metrics.last_failure_time,
                    "total_rejections": metrics.total_rejections,
                }
        return summary

    def reset_circuit_breaker(self, backend_type: str) -> bool:
        """
        Manually reset a circuit breaker to CLOSED state.

        Args:
            backend_type: The backend type to reset (e.g., "ollama", "lmstudio")

        Returns:
            True if reset successful, False if backend not found
        """
        for backend in self._backends.values():
            if backend.backend_type.value == backend_type:
                cb = backend.get_circuit_breaker()
                if cb:
                    cb.reset()
                    return True
        return False

    def get_rate_limiter(self) -> Optional[RateLimiter]:
        """Get the gateway's rate limiter instance."""
        return self._rate_limiter

    def get_rate_limiter_metrics(self) -> Optional[RateLimiterMetrics]:
        """
        Get rate limiter metrics.

        Returns:
            RateLimiterMetrics if rate limiting is enabled, None otherwise
        """
        if self._rate_limiter:
            return self._rate_limiter.get_metrics()
        return None

    async def get_client_rate_status(
        self, client_id: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """
        Get rate limit status for a specific client.

        Args:
            client_id: Client identifier (uses default if None)

        Returns:
            dict with client's current token count and limits
        """
        if self._rate_limiter:
            return await self._rate_limiter.get_client_metrics(client_id)
        return None

    def reset_rate_limiter(self, client_id: Optional[str] = None) -> bool:
        """
        Reset rate limiter state.

        Args:
            client_id: If provided, reset only this client. Otherwise reset all.

        Returns:
            True if rate limiter was reset, False if not enabled
        """
        if self._rate_limiter:
            self._rate_limiter.reset(client_id)
            return True
        return False

    async def shutdown(self) -> None:
        """Shutdown gateway and all backends."""
        for backend in self._backends.values():
            if hasattr(backend, "shutdown"):
                await backend.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_gateway_instance: Optional[InferenceGateway] = None


def get_inference_gateway() -> InferenceGateway:
    """Get the singleton inference gateway."""
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = InferenceGateway()
    return _gateway_instance


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA Inference Gateway")
    parser.add_argument("command", choices=["init", "infer", "health"])
    parser.add_argument("--prompt", help="Prompt for inference")
    parser.add_argument("--model", help="Model path")
    parser.add_argument("--tier", choices=["edge", "local", "pool"])
    args = parser.parse_args()

    gateway = get_inference_gateway()

    if args.model:
        gateway.config.model_path = args.model

    if args.command == "init":
        success = await gateway.initialize()
        print(f"Initialization: {'SUCCESS' if success else 'FAILED'}")
        print(f"Status: {gateway.status.value}")

    elif args.command == "infer":
        if not args.prompt:
            print("Error: --prompt required")
            return

        await gateway.initialize()
        tier = ComputeTier(args.tier) if args.tier else None

        result = await gateway.infer(args.prompt, tier=tier)
        assert isinstance(
            result, InferenceResult
        ), "Expected InferenceResult, not stream"
        print(f"\n{'='*60}")
        print(f"Model: {result.model}")
        print(f"Backend: {result.backend.value}")
        print(f"Tier: {result.tier.value}")
        print(f"Tokens: {result.tokens_generated} @ {result.tokens_per_second} tok/s")
        print(f"Latency: {result.latency_ms}ms")
        print(f"{'='*60}")
        print(result.content)

    elif args.command == "health":
        await gateway.initialize()
        health = await gateway.health()
        print(json.dumps(health, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
