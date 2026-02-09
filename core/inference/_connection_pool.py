"""
BIZRA Inference Gateway — Connection Pool
═══════════════════════════════════════════════════════════════════════════════

High-performance connection pooling for inference backends.

Standing on Giants:
- Apache Commons Pool (2001): Connection lifecycle management
- Amdahl (1967): Connection overhead reduction for parallel scaling
- HikariCP (2014): High-performance JDBC connection pooling

Extracted from gateway.py for modularity; re-exported by gateway.py
for backward compatibility.

Created: 2026-02-09 | Refactor split from gateway.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION POOLING (P1 OPTIMIZATION)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PooledConnection:
    """
    A pooled connection to an inference backend.

    Standing on Giants:
    - Apache Commons Pool (2001): Connection lifecycle management
    - Amdahl (1967): Connection overhead reduction for parallel scaling
    """

    id: str
    backend_type: str
    endpoint: str
    created_at: float
    last_used_at: float
    last_health_check: float
    is_healthy: bool = True
    in_use: bool = False
    use_count: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency for this connection."""
        return self.total_latency_ms / self.use_count if self.use_count > 0 else 0.0

    @property
    def age_seconds(self) -> float:
        """Age of connection in seconds."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Time since last use in seconds."""
        return time.time() - self.last_used_at


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pool."""

    # Pool sizing
    min_size: int = 2
    max_size: int = 10

    # Connection lifecycle
    max_idle_seconds: float = 300.0  # 5 minutes
    max_age_seconds: float = 3600.0  # 1 hour

    # Health checking
    health_check_interval_seconds: float = 30.0
    health_check_timeout_seconds: float = 5.0

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Connection acquisition
    acquisition_timeout_seconds: float = 30.0


class ConnectionPoolMetrics:
    """
    Thread-safe metrics for connection pool performance.

    Tracks pool hits, misses, and connection health for optimization.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._pool_hits = 0
        self._pool_misses = 0
        self._connections_created = 0
        self._connections_destroyed = 0
        self._health_checks_passed = 0
        self._health_checks_failed = 0
        self._total_wait_time_ms = 0.0
        self._acquisition_timeouts = 0

    async def record_hit(self) -> None:
        async with self._lock:
            self._pool_hits += 1

    async def record_miss(self) -> None:
        async with self._lock:
            self._pool_misses += 1

    async def record_connection_created(self) -> None:
        async with self._lock:
            self._connections_created += 1

    async def record_connection_destroyed(self) -> None:
        async with self._lock:
            self._connections_destroyed += 1

    async def record_health_check(self, passed: bool) -> None:
        async with self._lock:
            if passed:
                self._health_checks_passed += 1
            else:
                self._health_checks_failed += 1

    async def record_wait_time(self, wait_ms: float) -> None:
        async with self._lock:
            self._total_wait_time_ms += wait_ms

    async def record_acquisition_timeout(self) -> None:
        async with self._lock:
            self._acquisition_timeouts += 1

    async def get_metrics(self) -> Dict[str, Any]:
        async with self._lock:
            total_acquisitions = self._pool_hits + self._pool_misses
            return {
                "pool_hits": self._pool_hits,
                "pool_misses": self._pool_misses,
                "hit_rate": (
                    self._pool_hits / total_acquisitions
                    if total_acquisitions > 0
                    else 0.0
                ),
                "connections_created": self._connections_created,
                "connections_destroyed": self._connections_destroyed,
                "health_checks_passed": self._health_checks_passed,
                "health_checks_failed": self._health_checks_failed,
                "avg_wait_time_ms": (
                    self._total_wait_time_ms / total_acquisitions
                    if total_acquisitions > 0
                    else 0.0
                ),
                "acquisition_timeouts": self._acquisition_timeouts,
            }


class ConnectionPool:
    """
    High-performance connection pool for inference backends.

    PROBLEM: Creating new HTTP connections for each inference request adds
    significant latency overhead (TCP handshake, TLS negotiation).

    SOLUTION: Pool and reuse connections with:
    - LRU eviction for memory efficiency
    - Health checking for reliability
    - Automatic reconnection for fault tolerance

    Standing on Giants:
    - Apache Commons Pool (2001): Object pooling patterns
    - Amdahl (1967): Minimizing serial overhead in parallel systems
    - HikariCP (2014): High-performance JDBC connection pooling

    Expected improvement: 3-5x latency reduction for high-frequency requests.
    """

    def __init__(
        self,
        backend_type: str,
        endpoint: str,
        config: Optional[ConnectionPoolConfig] = None,
        connection_factory: Optional[Callable[[], Awaitable[Any]]] = None,
        health_checker: Optional[Callable[[Any], Awaitable[bool]]] = None,
    ) -> None:
        """
        Initialize connection pool.

        Args:
            backend_type: Type of backend (ollama, lmstudio)
            endpoint: Backend endpoint URL
            config: Pool configuration
            connection_factory: Async callable that creates a new connection
            health_checker: Async callable that checks connection health
        """
        self.backend_type = backend_type
        self.endpoint = endpoint
        self.config = config or ConnectionPoolConfig()
        self._connection_factory = connection_factory
        self._health_checker = health_checker

        # LRU-ordered pool (newest at end)
        self._pool: OrderedDict[str, PooledConnection] = OrderedDict()
        self._connections: Dict[str, Any] = {}  # Actual connection objects

        # Synchronization
        self._lock = asyncio.Lock()
        self._available = asyncio.Semaphore(self.config.max_size)

        # Background tasks
        self._health_check_task: Optional[asyncio.Task[None]] = None
        self._running = False

        # Metrics
        self.metrics = ConnectionPoolMetrics()

    async def start(self) -> None:
        """Start the connection pool and background health checker."""
        if self._running:
            return

        self._running = True

        # Pre-warm pool with minimum connections
        for _ in range(self.config.min_size):
            try:
                await self._create_connection()
            except Exception as e:
                print(f"[ConnectionPool] Pre-warm failed: {e}")
                break

        # Start health checker
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        print(
            f"[ConnectionPool] Started for {self.backend_type} "
            f"(min={self.config.min_size}, max={self.config.max_size})"
        )

    async def stop(self) -> None:
        """Stop the connection pool and cleanup resources."""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

        # Close all connections
        async with self._lock:
            for conn_id in list(self._pool.keys()):
                await self._destroy_connection(conn_id)

        print(f"[ConnectionPool] Stopped for {self.backend_type}")

    async def _create_connection(self) -> PooledConnection:
        """Create a new pooled connection."""
        conn_id = hashlib.sha256(
            f"{self.backend_type}:{self.endpoint}:{time.time()}".encode()
        ).hexdigest()[:16]

        now = time.time()
        pooled_conn = PooledConnection(
            id=conn_id,
            backend_type=self.backend_type,
            endpoint=self.endpoint,
            created_at=now,
            last_used_at=now,
            last_health_check=now,
            is_healthy=True,
            in_use=False,
        )

        # Create actual connection if factory provided
        if self._connection_factory:
            conn = await self._connection_factory()
            self._connections[conn_id] = conn

        async with self._lock:
            self._pool[conn_id] = pooled_conn
            # Move to end (LRU)
            self._pool.move_to_end(conn_id)

        await self.metrics.record_connection_created()
        return pooled_conn

    async def _destroy_connection(self, conn_id: str) -> None:
        """Destroy a pooled connection."""
        if conn_id in self._pool:
            del self._pool[conn_id]

        if conn_id in self._connections:
            conn = self._connections.pop(conn_id)
            # Close connection if it has a close method
            if hasattr(conn, "close"):
                try:
                    if asyncio.iscoroutinefunction(conn.close):
                        await conn.close()
                    else:
                        conn.close()
                except (OSError, ConnectionError, RuntimeError):
                    # Best-effort cleanup - connection may already be closed
                    pass

        await self.metrics.record_connection_destroyed()

    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a connection from the pool.

        Yields:
            Tuple of (PooledConnection, actual_connection_object)

        Usage:
            async with pool.acquire() as (pooled, conn):
                result = await use_connection(conn)
        """
        start_time = time.time()
        pooled_conn: Optional[PooledConnection] = None
        actual_conn: Any = None

        try:
            # Wait for available slot with timeout
            try:
                await asyncio.wait_for(
                    self._available.acquire(),
                    timeout=self.config.acquisition_timeout_seconds,
                )
            except asyncio.TimeoutError:
                await self.metrics.record_acquisition_timeout()
                raise RuntimeError(
                    f"Connection acquisition timeout after "
                    f"{self.config.acquisition_timeout_seconds}s"
                )

            # Find or create connection
            async with self._lock:
                # Find available healthy connection (LRU order - oldest first)
                for conn_id, conn in self._pool.items():
                    if not conn.in_use and conn.is_healthy:
                        # Check if connection is too old or idle
                        if (
                            conn.age_seconds > self.config.max_age_seconds
                            or conn.idle_seconds > self.config.max_idle_seconds
                        ):
                            # Destroy old connection
                            await self._destroy_connection(conn_id)
                            continue

                        # Found usable connection
                        conn.in_use = True
                        conn.last_used_at = time.time()
                        pooled_conn = conn
                        await self.metrics.record_hit()
                        break

            # No available connection - create new one if under limit
            if pooled_conn is None:
                await self.metrics.record_miss()
                async with self._lock:
                    if len(self._pool) < self.config.max_size:
                        pooled_conn = await self._create_connection()
                        pooled_conn.in_use = True

            if pooled_conn is None:
                raise RuntimeError("Failed to acquire connection")

            # Get actual connection object
            actual_conn = self._connections.get(pooled_conn.id)

            # Record wait time
            wait_ms = (time.time() - start_time) * 1000
            await self.metrics.record_wait_time(wait_ms)

            yield (pooled_conn, actual_conn)

        finally:
            # Release connection back to pool
            if pooled_conn is not None:
                async with self._lock:
                    if pooled_conn.id in self._pool:
                        pooled_conn.in_use = False
                        pooled_conn.use_count += 1
                        # Move to end (most recently used)
                        self._pool.move_to_end(pooled_conn.id)

            self._available.release()

    async def _health_check_loop(self) -> None:
        """Background task to check connection health."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)

                if not self._running:
                    break

                # Check all connections
                async with self._lock:
                    conn_ids = list(self._pool.keys())

                for conn_id in conn_ids:
                    if not self._running:
                        break

                    async with self._lock:
                        if conn_id not in self._pool:
                            continue
                        conn = self._pool[conn_id]
                        if conn.in_use:
                            continue

                    # Run health check
                    is_healthy = await self._check_connection_health(conn_id)

                    async with self._lock:
                        if conn_id in self._pool:
                            self._pool[conn_id].is_healthy = is_healthy
                            self._pool[conn_id].last_health_check = time.time()

                    await self.metrics.record_health_check(is_healthy)

                    # Destroy unhealthy connections
                    if not is_healthy:
                        async with self._lock:
                            await self._destroy_connection(conn_id)

                # Ensure minimum connections
                async with self._lock:
                    healthy_count = sum(
                        1 for c in self._pool.values() if c.is_healthy and not c.in_use
                    )

                while healthy_count < self.config.min_size and self._running:
                    try:
                        await self._create_connection()
                        healthy_count += 1
                    except Exception:
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ConnectionPool] Health check error: {e}")
                await asyncio.sleep(5)  # Backoff on error

    async def _check_connection_health(self, conn_id: str) -> bool:
        """Check if a specific connection is healthy."""
        if self._health_checker is None:
            return True

        conn = self._connections.get(conn_id)
        if conn is None:
            return False

        try:
            result = await asyncio.wait_for(
                self._health_checker(conn),
                timeout=self.config.health_check_timeout_seconds,
            )
            return bool(result)
        except asyncio.TimeoutError:
            return False
        except (ConnectionError, OSError, RuntimeError):
            # Health check failed - connection unhealthy
            return False

    def get_active_connections(self) -> int:
        """Get number of connections currently in use."""
        return sum(1 for c in self._pool.values() if c.in_use)

    def get_available_connections(self) -> int:
        """Get number of healthy connections available."""
        return sum(1 for c in self._pool.values() if c.is_healthy and not c.in_use)

    def get_total_connections(self) -> int:
        """Get total number of connections in pool."""
        return len(self._pool)

    async def get_status(self) -> Dict[str, Any]:
        """Get pool status including metrics."""
        metrics = await self.metrics.get_metrics()
        return {
            "backend_type": self.backend_type,
            "endpoint": self.endpoint,
            "total_connections": self.get_total_connections(),
            "active_connections": self.get_active_connections(),
            "available_connections": self.get_available_connections(),
            "config": {
                "min_size": self.config.min_size,
                "max_size": self.config.max_size,
                "max_idle_seconds": self.config.max_idle_seconds,
                "max_age_seconds": self.config.max_age_seconds,
            },
            "metrics": metrics,
        }


class PooledHttpClient:
    """
    HTTP client wrapper for connection pooling.

    Provides a simple interface for making HTTP requests through the pool.
    """

    def __init__(self, pool: ConnectionPool, base_url: str) -> None:
        self.pool = pool
        self.base_url = base_url.rstrip("/")

    async def request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Make an HTTP request through the pool."""
        import urllib.error
        import urllib.request

        url = f"{self.base_url}{path}"

        async with self.pool.acquire() as (pooled_conn, _):
            start_time = time.time()
            try:
                payload = json.dumps(data).encode() if data else None
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method=method,
                )

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: urllib.request.urlopen(req, timeout=timeout)  # nosec B310 — URL from trusted connection pool config
                )
                with response as resp:
                    result: Dict[str, Any] = json.loads(resp.read().decode())

                # Record latency
                latency_ms = (time.time() - start_time) * 1000
                pooled_conn.total_latency_ms += latency_ms

                return result

            except Exception:
                # Mark connection as unhealthy on error
                pooled_conn.is_healthy = False
                raise
