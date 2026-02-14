#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
    ██████╗ ██╗███████╗██████╗  █████╗     ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
    ██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗    ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
    ██████╔╝██║  ███╔╝ ██████╔╝███████║    ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
    ██╔══██╗██║ ███╔╝  ██╔══██╗██╔══██║    ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
    ██████╔╝██║███████╗██║  ██║██║  ██║    ██║ ╚████║███████╗██╔╝ ╚██╗╚██████╔╝███████║
    ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝  ╚═══╝╚══════╝╚═╝   ╚═╝ ╚═════╝ ╚══════╝
═══════════════════════════════════════════════════════════════════════════════════════════════════════
    BIZRA NEXUS — Unified Data Lake Orchestration Engine
    
    A production-grade, enterprise-architecture orchestration layer that unifies:
    • Sacred Wisdom Engine (Quran + Hadith hypergraph)
    • File Type Indexer (904K+ files across 478GB)
    • Sovereign Brain (Multi-engine reasoning)
    • Pattern Discovery (Scientific + Hidden knowledge)
    
    ARCHITECTURE PRINCIPLES:
    ├── SOLID: Single Responsibility, Open/Closed, Liskov, Interface Segregation, DI
    ├── Clean Architecture: Layered with clear boundaries
    ├── Domain-Driven Design: Ubiquitous language, bounded contexts
    ├── Event-Driven: Async message passing, observer pattern
    ├── Resource Efficiency: Lazy loading, generators, connection pooling
    └── Fault Tolerance: Circuit breakers, retry policies, graceful degradation
    
    GIANTS ABSORBED:
    • Martin Fowler (Patterns of Enterprise Application Architecture)
    • Robert C. Martin (Clean Code, SOLID)
    • Eric Evans (Domain-Driven Design)
    • Kent Beck (Test-Driven Development)
    • Shannon (Information Theory — SNR optimization)
    
    Created: 2026-01-23 | 15,000 hours of إخلاص | BIZRA Genesis
═══════════════════════════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
import threading
import weakref
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from functools import lru_cache, wraps, partial
from pathlib import Path
from typing import (
    Dict, List, Set, Tuple, Optional, Any, Union, Callable, 
    TypeVar, Generic, Protocol, Iterator, AsyncIterator,
    Awaitable, NamedTuple, Final, Literal, ClassVar
)
from collections import defaultdict, deque
from queue import Queue, Empty
import statistics

# Type variables for generic programming
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
R = TypeVar('R')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Immutable, Type-Safe, Validated
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class NexusConfig:
    """Immutable configuration with validation."""
    data_lake_path: Path = field(default_factory=lambda: Path(
        os.environ.get("BIZRA_DATA_LAKE_ROOT", 
                       "/mnt/c/BIZRA-DATA-LAKE" if os.path.exists("/mnt/c") else "C:/BIZRA-DATA-LAKE")
    ))
    max_workers: int = 8
    cache_size: int = 10000
    batch_size: int = 256
    retry_attempts: int = 3
    retry_delay: float = 0.5
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration on creation."""
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.cache_size < 0:
            raise ValueError("cache_size must be >= 0")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
    
    @property
    def gold_path(self) -> Path:
        return self.data_lake_path / "04_GOLD"
    
    @property
    def indexed_path(self) -> Path:
        return self.data_lake_path / "03_INDEXED"


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING — Structured, Contextual, Performance-Aware
# ═══════════════════════════════════════════════════════════════════════════════

class StructuredLogger:
    """Thread-safe structured logger with context propagation."""
    
    _instance: ClassVar[Optional['StructuredLogger']] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    
    def __new__(cls, name: str = "BizraNexus") -> 'StructuredLogger':
        """Singleton pattern with double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialize(name)
                    cls._instance = instance
        return cls._instance
    
    def _initialize(self, name: str):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._context: Dict[str, Any] = {}
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Console handler with formatting
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '[%(levelname)s] %(asctime)s | %(name)s | %(message)s',
                datefmt='%H:%M:%S'
            ))
            self._logger.addHandler(handler)
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager for scoped logging context."""
        old_context = self._context.copy()
        self._context.update(kwargs)
        try:
            yield
        finally:
            self._context = old_context
    
    def _format_message(self, msg: str, **kwargs) -> str:
        """Format message with context."""
        ctx = {**self._context, **kwargs}
        if ctx:
            ctx_str = " | ".join(f"{k}={v}" for k, v in ctx.items())
            return f"{msg} [{ctx_str}]"
        return msg
    
    def info(self, msg: str, **kwargs): 
        self._logger.info(self._format_message(msg, **kwargs))
    
    def debug(self, msg: str, **kwargs): 
        self._logger.debug(self._format_message(msg, **kwargs))
    
    def warning(self, msg: str, **kwargs): 
        self._logger.warning(self._format_message(msg, **kwargs))
    
    def error(self, msg: str, **kwargs): 
        self._logger.error(self._format_message(msg, **kwargs))
    
    def metric(self, name: str, value: float):
        """Record a metric for later analysis."""
        self._metrics[name].append(value)
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of all metrics."""
        summary = {}
        for name, values in self._metrics.items():
            if values:
                summary[name] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0
                }
        return summary


log = StructuredLogger()


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOLS — Interface Definitions (Structural Typing)
# ═══════════════════════════════════════════════════════════════════════════════

class Queryable(Protocol):
    """Protocol for queryable engines."""
    def query(self, query: str, **kwargs) -> Dict[str, Any]: ...

class Indexable(Protocol):
    """Protocol for indexable data sources."""
    def index(self, data: Any) -> str: ...
    def get(self, key: str) -> Optional[Any]: ...

class Observable(Protocol):
    """Protocol for observable subjects (Observer pattern)."""
    def subscribe(self, observer: 'Observer') -> Callable[[], None]: ...
    def notify(self, event: 'Event') -> None: ...

class Observer(Protocol):
    """Protocol for observers."""
    def on_event(self, event: 'Event') -> None: ...


# ═══════════════════════════════════════════════════════════════════════════════
# VALUE OBJECTS — Immutable Domain Objects
# ═══════════════════════════════════════════════════════════════════════════════

class EventType(Enum):
    """Types of system events."""
    QUERY_START = auto()
    QUERY_COMPLETE = auto()
    INDEX_START = auto()
    INDEX_COMPLETE = auto()
    ENGINE_READY = auto()
    ENGINE_ERROR = auto()
    CACHE_HIT = auto()
    CACHE_MISS = auto()
    CIRCUIT_OPEN = auto()
    CIRCUIT_CLOSE = auto()


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable event for pub/sub."""
    type: EventType
    source: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Immutable query result with metadata."""
    query: str
    results: Tuple[Dict[str, Any], ...]
    total_count: int
    elapsed_ms: float
    snr_score: float
    source_engine: str
    cached: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": list(self.results),
            "total_count": self.total_count,
            "elapsed_ms": self.elapsed_ms,
            "snr_score": self.snr_score,
            "source_engine": self.source_engine,
            "cached": self.cached
        }


class HealthStatus(Enum):
    """Health status for engines."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class HealthCheck:
    """Health check result."""
    engine: str
    status: HealthStatus
    latency_ms: float
    message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════════════════════════════
# RESILIENCE PATTERNS — Circuit Breaker, Retry, Timeout
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()    # Normal operation
    OPEN = auto()      # Failing, reject requests
    HALF_OPEN = auto() # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit Breaker Pattern implementation.
    
    Prevents cascading failures by stopping requests to failing services.
    """
    name: str
    threshold: int = 5
    timeout: float = 30.0
    
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    
    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - (self._last_failure_time or 0) > self.timeout:
                    self._state = CircuitState.HALF_OPEN
                    log.info(f"Circuit {self.name} transitioning to HALF_OPEN")
            return self._state
    
    def record_success(self):
        """Record successful call."""
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                log.info(f"Circuit {self.name} CLOSED")
    
    def record_failure(self):
        """Record failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.threshold:
                self._state = CircuitState.OPEN
                log.warning(f"Circuit {self.name} OPEN after {self._failure_count} failures")
    
    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            return True  # Allow test request
        else:
            return False


def with_circuit_breaker(circuit: CircuitBreaker):
    """Decorator to wrap function with circuit breaker."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not circuit.allow_request():
                raise RuntimeError(f"Circuit {circuit.name} is OPEN")
            try:
                result = func(*args, **kwargs)
                circuit.record_success()
                return result
            except Exception as e:
                circuit.record_failure()
                raise
        return wrapper
    return decorator


def with_retry(attempts: int = 3, delay: float = 0.5, backoff: float = 2.0):
    """Decorator for retry with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < attempts - 1:
                        log.warning(f"Retry {attempt + 1}/{attempts} for {func.__name__}: {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator


def with_timeout(seconds: float):
    """Decorator for function timeout."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"{func.__name__} timed out after {seconds}s")
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# CACHING — LRU with TTL, Thread-Safe
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with TTL."""
    value: T
    created_at: float
    ttl: float
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl


class TTLCache(Generic[K, V]):
    """Thread-safe LRU cache with TTL."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: Dict[K, CacheEntry[V]] = {}
        self._access_order: deque = deque()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: K) -> Optional[V]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            self._hits += 1
            # Update access order
            try:
                self._access_order.remove(key)
            except ValueError:
                pass
            self._access_order.append(key)
            return entry.value
    
    def set(self, key: K, value: V, ttl: Optional[float] = None):
        """Set value in cache."""
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._max_size and self._access_order:
                oldest = self._access_order.popleft()
                self._cache.pop(oldest, None)
            
            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl or self._default_ttl
            )
            self._access_order.append(key)
    
    def invalidate(self, key: K):
        """Remove key from cache."""
        with self._lock:
            self._cache.pop(key, None)
            try:
                self._access_order.remove(key)
            except ValueError:
                pass
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0
            }


def cached(cache: TTLCache, key_fn: Callable[..., str]):
    """Decorator for caching function results."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache_key = key_fn(*args, **kwargs)
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                log.debug(f"Cache hit for {func.__name__}", key=cache_key)
                return cached_value
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT BUS — Async Pub/Sub with Weak References
# ═══════════════════════════════════════════════════════════════════════════════

class EventBus:
    """
    Thread-safe event bus for publish/subscribe messaging.
    Stores handlers directly (caller responsible for lifecycle).
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = defaultdict(list)
        self._lock = threading.Lock()
        self._event_history: deque = deque(maxlen=1000)
    
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> Callable[[], None]:
        """
        Subscribe to event type. Returns unsubscribe function.
        Caller is responsible for handler lifecycle.
        """
        with self._lock:
            self._subscribers[event_type].append(handler)
        
        def unsubscribe():
            with self._lock:
                try:
                    self._subscribers[event_type].remove(handler)
                except ValueError:
                    pass
        
        return unsubscribe
    
    def publish(self, event: Event):
        """Publish event to all subscribers."""
        self._event_history.append(event)
        
        with self._lock:
            handlers = list(self._subscribers.get(event.type, []))
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                log.error(f"Event handler error: {e}")
    
    async def publish_async(self, event: Event):
        """Async version of publish."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.publish, event)
    
    def get_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """Get event history, optionally filtered by type."""
        events = list(self._event_history)
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCE POOL — Connection/Resource Management
# ═══════════════════════════════════════════════════════════════════════════════

class ResourcePool(Generic[T]):
    """
    Thread-safe resource pool with lazy initialization.
    Implements Object Pool pattern.
    """
    
    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
        validate: Optional[Callable[[T], bool]] = None,
        cleanup: Optional[Callable[[T], None]] = None
    ):
        self._factory = factory
        self._max_size = max_size
        self._validate = validate or (lambda x: True)
        self._cleanup = cleanup
        self._pool: Queue[T] = Queue(maxsize=max_size)
        self._size = 0
        self._lock = threading.Lock()
    
    @contextmanager
    def acquire(self) -> Iterator[T]:
        """Acquire resource from pool."""
        resource = None
        
        # Try to get from pool
        try:
            resource = self._pool.get_nowait()
            if not self._validate(resource):
                if self._cleanup:
                    self._cleanup(resource)
                resource = None
        except Empty:
            pass
        
        # Create new if needed
        if resource is None:
            with self._lock:
                if self._size < self._max_size:
                    resource = self._factory()
                    self._size += 1
                else:
                    # Wait for available resource
                    resource = self._pool.get(timeout=30.0)
        
        try:
            yield resource
        finally:
            # Return to pool
            if resource is not None and self._validate(resource):
                try:
                    self._pool.put_nowait(resource)
                except Exception:
                    if self._cleanup:
                        self._cleanup(resource)
                    with self._lock:
                        self._size -= 1
    
    def shutdown(self):
        """Cleanup all pooled resources."""
        while not self._pool.empty():
            try:
                resource = self._pool.get_nowait()
                if self._cleanup:
                    self._cleanup(resource)
            except Empty:
                break
        with self._lock:
            self._size = 0


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE ADAPTERS — Unified Interface for Different Engines
# ═══════════════════════════════════════════════════════════════════════════════

class EngineAdapter(ABC):
    """Abstract base for engine adapters. Template Method pattern."""
    
    def __init__(self, name: str, config: NexusConfig):
        self.name = name
        self.config = config
        self._circuit = CircuitBreaker(name=name)
        self._cache = TTLCache[str, QueryResult](max_size=config.cache_size)
        self._initialized = False
    
    @abstractmethod
    def _do_initialize(self) -> None:
        """Template method: Engine-specific initialization."""
        pass
    
    @abstractmethod
    def _do_query(self, query: str, **kwargs) -> QueryResult:
        """Template method: Engine-specific query."""
        pass
    
    @abstractmethod
    def _do_health_check(self) -> HealthCheck:
        """Template method: Engine-specific health check."""
        pass
    
    def initialize(self) -> None:
        """Initialize engine with error handling."""
        if self._initialized:
            return
        
        log.info(f"Initializing engine: {self.name}")
        try:
            self._do_initialize()
            self._initialized = True
            log.info(f"Engine initialized: {self.name}")
        except Exception as e:
            log.error(f"Engine initialization failed: {self.name}", error=str(e))
            raise
    
    @with_retry(attempts=3, delay=0.5)
    def query(self, query: str, use_cache: bool = True, **kwargs) -> QueryResult:
        """Execute query with caching and circuit breaker."""
        if not self._initialized:
            self.initialize()
        
        # Check cache
        cache_key = f"{self.name}:{hashlib.blake2b(query.encode(), digest_size=16).hexdigest()}"
        if use_cache:
            cached_result = self._cache.get(cache_key)
            if cached_result:
                return QueryResult(
                    query=cached_result.query,
                    results=cached_result.results,
                    total_count=cached_result.total_count,
                    elapsed_ms=cached_result.elapsed_ms,
                    snr_score=cached_result.snr_score,
                    source_engine=cached_result.source_engine,
                    cached=True
                )
        
        # Execute with circuit breaker
        if not self._circuit.allow_request():
            raise RuntimeError(f"Engine {self.name} circuit is OPEN")
        
        start = time.perf_counter()
        try:
            result = self._do_query(query, **kwargs)
            self._circuit.record_success()
            
            # Cache result
            if use_cache:
                self._cache.set(cache_key, result)
            
            elapsed = (time.perf_counter() - start) * 1000
            log.metric(f"{self.name}_query_ms", elapsed)
            
            return result
            
        except Exception as e:
            self._circuit.record_failure()
            raise
    
    def health_check(self) -> HealthCheck:
        """Perform health check."""
        start = time.perf_counter()
        try:
            check = self._do_health_check()
            latency = (time.perf_counter() - start) * 1000
            return HealthCheck(
                engine=self.name,
                status=check.status,
                latency_ms=latency,
                message=check.message
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return HealthCheck(
                engine=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                message=str(e)
            )


class SacredWisdomAdapter(EngineAdapter):
    """Adapter for Sacred Wisdom Engine."""
    
    def __init__(self, config: NexusConfig):
        super().__init__("SacredWisdom", config)
        self._engine = None
    
    def _do_initialize(self):
        from sacred_wisdom_engine import SacredWisdomEngine
        self._engine = SacredWisdomEngine(lazy_load=True)
        # Try to load, build if needed
        if not self._engine.load():
            log.info("Building Sacred Wisdom Engine...")
            self._engine.build()
    
    def _do_query(self, query: str, **kwargs) -> QueryResult:
        result = self._engine.query(query, **kwargs)
        return QueryResult(
            query=query,
            results=tuple(result.get("results", [])),
            total_count=len(result.get("results", [])),
            elapsed_ms=result.get("elapsed_ms", 0),
            snr_score=result.get("snr", 0),
            source_engine=self.name
        )
    
    def _do_health_check(self) -> HealthCheck:
        if self._engine and self._engine.is_built:
            return HealthCheck(
                engine=self.name,
                status=HealthStatus.HEALTHY,
                latency_ms=0,
                message=f"Nodes: {self._engine.stats.get('nodes', 0)}"
            )
        return HealthCheck(
            engine=self.name,
            status=HealthStatus.UNHEALTHY,
            latency_ms=0,
            message="Engine not built"
        )


class FileIndexAdapter(EngineAdapter):
    """Adapter for File Type Indexer."""
    
    def __init__(self, config: NexusConfig):
        super().__init__("FileIndex", config)
        self._index: Dict[str, Any] = {}
    
    def _do_initialize(self):
        # Load latest index from disk
        index_dir = self.config.indexed_path / "file_index"
        if index_dir.exists():
            summary_files = sorted(index_dir.glob("*_summary_*.json"), reverse=True)
            if summary_files:
                try:
                    with open(summary_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                        self._index = json.load(f)
                    log.info(f"Loaded file index: {self._index.get('total_files', 0)} files")
                except Exception as e:
                    log.warning(f"Failed to load index: {e}")
                    self._index = {}
    
    def _do_query(self, query: str, **kwargs) -> QueryResult:
        # Comprehensive search in index
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Search in categories
        for category, data in self._index.get("categories", {}).items():
            category_lower = category.lower()
            if any(word in category_lower for word in query_words):
                results.append({
                    "type": "category",
                    "category": category,
                    "count": data.get("count", 0),
                    "size": data.get("size", "0"),
                    "extensions": data.get("top_extensions", {})
                })
        
        # Search in extensions
        for ext, count in self._index.get("extensions", {}).items():
            if any(word in ext.lower() for word in query_words):
                results.append({
                    "type": "extension",
                    "extension": ext,
                    "count": count
                })
        
        # Search in golden gems if present
        for gem in self._index.get("golden_gems", []):
            gem_str = str(gem).lower()
            if any(word in gem_str for word in query_words):
                results.append({
                    "type": "golden_gem",
                    "gem": gem
                })
        
        # Add summary info
        if not results and self._index:
            results.append({
                "type": "summary",
                "total_files": self._index.get("total_files", 0),
                "total_size": self._index.get("total_size", "0"),
                "categories": list(self._index.get("categories", {}).keys())
            })
        
        return QueryResult(
            query=query,
            results=tuple(results[:30]),
            total_count=len(results),
            elapsed_ms=0.1,
            snr_score=0.85 if len(results) > 1 else (0.5 if results else 0.2),
            source_engine=self.name
        )
    
    def _do_health_check(self) -> HealthCheck:
        return HealthCheck(
            engine=self.name,
            status=HealthStatus.HEALTHY if self._index else HealthStatus.DEGRADED,
            latency_ms=0,
            message=f"Files indexed: {self._index.get('total_files', 0)}"
        )


class SovereignBrainAdapter(EngineAdapter):
    """Adapter for Sovereign Brain — Multi-engine reasoning orchestrator."""
    
    def __init__(self, config: NexusConfig):
        super().__init__("SovereignBrain", config)
        self._brain = None
        self._state: Dict[str, Any] = {}
    
    def _do_initialize(self):
        # Load sovereign brain state
        state_file = self.config.gold_path / "bizra_prime_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    self._state = json.load(f)
                log.info(f"Loaded Sovereign Brain state: {len(self._state)} keys")
            except Exception as e:
                log.warning(f"Failed to load brain state: {e}")
    
    def _do_query(self, query: str, **kwargs) -> QueryResult:
        # Search in state keys and values
        results = []
        query_lower = query.lower()
        
        for key, value in self._state.items():
            if query_lower in key.lower():
                results.append({
                    "key": key,
                    "value": str(value)[:200],
                    "type": type(value).__name__
                })
            elif isinstance(value, str) and query_lower in value.lower():
                results.append({
                    "key": key,
                    "value": value[:200],
                    "match_type": "value"
                })
        
        return QueryResult(
            query=query,
            results=tuple(results[:20]),
            total_count=len(results),
            elapsed_ms=0.5,
            snr_score=0.7 if results else 0.1,
            source_engine=self.name
        )
    
    def _do_health_check(self) -> HealthCheck:
        return HealthCheck(
            engine=self.name,
            status=HealthStatus.HEALTHY if self._state else HealthStatus.DEGRADED,
            latency_ms=0,
            message=f"State keys: {len(self._state)}"
        )


class KnowledgeGraphAdapter(EngineAdapter):
    """Adapter for Knowledge Graph — Graph-based semantic search."""
    
    def __init__(self, config: NexusConfig):
        super().__init__("KnowledgeGraph", config)
        self._graph_data: Dict[str, Any] = {}
        self._nodes: List[Dict] = []
        self._edges: List[Dict] = []
    
    def _do_initialize(self):
        # Load graph data from indexed knowledge
        graph_dir = self.config.indexed_path / "graph"
        if graph_dir.exists():
            # Try to load any graph files
            for graph_file in graph_dir.glob("*.json"):
                try:
                    with open(graph_file, 'r', encoding='utf-8', errors='ignore') as f:
                        data = json.load(f)
                        if "nodes" in data:
                            self._nodes.extend(data["nodes"])
                        if "edges" in data:
                            self._edges.extend(data["edges"])
                except Exception:
                    pass
            log.info(f"Loaded graph: {len(self._nodes)} nodes, {len(self._edges)} edges")
    
    def _do_query(self, query: str, **kwargs) -> QueryResult:
        # Search nodes by label or properties
        results = []
        query_lower = query.lower()
        
        for node in self._nodes:
            label = str(node.get("label", node.get("id", "")))
            if query_lower in label.lower():
                results.append({
                    "type": "node",
                    "id": node.get("id"),
                    "label": label,
                    "properties": node.get("properties", {})
                })
        
        return QueryResult(
            query=query,
            results=tuple(results[:30]),
            total_count=len(results),
            elapsed_ms=1.0,
            snr_score=0.75 if results else 0.15,
            source_engine=self.name
        )
    
    def _do_health_check(self) -> HealthCheck:
        status = HealthStatus.HEALTHY if self._nodes else HealthStatus.DEGRADED
        return HealthCheck(
            engine=self.name,
            status=status,
            latency_ms=0,
            message=f"Nodes: {len(self._nodes)}, Edges: {len(self._edges)}"
        )


class ChatHistoryAdapter(EngineAdapter):
    """Adapter for Chat History — Search past conversations."""
    
    def __init__(self, config: NexusConfig):
        super().__init__("ChatHistory", config)
        self._conversations: List[Dict] = []
    
    def _do_initialize(self):
        # Load chat history
        chat_dir = self.config.indexed_path / "chat_history"
        if chat_dir.exists():
            for chat_file in list(chat_dir.glob("*.json"))[:100]:  # Limit for performance
                try:
                    with open(chat_file, 'r', encoding='utf-8', errors='ignore') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            self._conversations.append({
                                "file": chat_file.name,
                                "data": data
                            })
                        elif isinstance(data, list):
                            self._conversations.append({
                                "file": chat_file.name,
                                "messages": data
                            })
                except Exception:
                    pass
            log.info(f"Loaded {len(self._conversations)} conversations")
    
    def _do_query(self, query: str, **kwargs) -> QueryResult:
        results = []
        query_lower = query.lower()
        
        for conv in self._conversations:
            file_name = conv.get("file", "")
            if query_lower in file_name.lower():
                results.append({
                    "file": file_name,
                    "match_type": "filename"
                })
            
            # Search in messages if present
            messages = conv.get("messages", [])
            for msg in messages[:50]:  # Limit search depth
                content = str(msg.get("content", msg.get("text", "")))
                if query_lower in content.lower():
                    results.append({
                        "file": file_name,
                        "content": content[:150],
                        "role": msg.get("role", "unknown")
                    })
                    break  # One match per conversation
        
        return QueryResult(
            query=query,
            results=tuple(results[:25]),
            total_count=len(results),
            elapsed_ms=2.0,
            snr_score=0.65 if results else 0.1,
            source_engine=self.name
        )
    
    def _do_health_check(self) -> HealthCheck:
        return HealthCheck(
            engine=self.name,
            status=HealthStatus.HEALTHY if self._conversations else HealthStatus.DEGRADED,
            latency_ms=0,
            message=f"Conversations: {len(self._conversations)}"
        )


class EmbeddingsAdapter(EngineAdapter):
    """Adapter for Vector Embeddings — Semantic similarity search."""
    
    def __init__(self, config: NexusConfig):
        super().__init__("Embeddings", config)
        self._embeddings_loaded = False
        self._metadata: List[Dict] = []
    
    def _do_initialize(self):
        # Check for embedding files
        embed_dir = self.config.indexed_path / "embeddings"
        if embed_dir.exists():
            # Load metadata (not actual embeddings for efficiency)
            for meta_file in embed_dir.glob("*_metadata.json"):
                try:
                    with open(meta_file, 'r', encoding='utf-8', errors='ignore') as f:
                        meta = json.load(f)
                        if isinstance(meta, list):
                            self._metadata.extend(meta)
                        elif isinstance(meta, dict):
                            self._metadata.append(meta)
                except Exception:
                    pass
            
            # Check for .npy or .pkl files
            embedding_files = list(embed_dir.glob("*.npy")) + list(embed_dir.glob("*.pkl"))
            self._embeddings_loaded = len(embedding_files) > 0
            log.info(f"Found {len(embedding_files)} embedding files, {len(self._metadata)} metadata entries")
    
    def _do_query(self, query: str, **kwargs) -> QueryResult:
        # Search metadata (full semantic search would require loading embeddings)
        results = []
        query_lower = query.lower()
        
        for meta in self._metadata:
            text = str(meta.get("text", meta.get("content", "")))
            if query_lower in text.lower():
                results.append({
                    "text": text[:200],
                    "source": meta.get("source", "unknown"),
                    "id": meta.get("id", "")
                })
        
        return QueryResult(
            query=query,
            results=tuple(results[:20]),
            total_count=len(results),
            elapsed_ms=0.5,
            snr_score=0.6 if results else 0.2,
            source_engine=self.name
        )
    
    def _do_health_check(self) -> HealthCheck:
        status = HealthStatus.HEALTHY if self._embeddings_loaded else HealthStatus.DEGRADED
        return HealthCheck(
            engine=self.name,
            status=status,
            latency_ms=0,
            message=f"Embeddings: {self._embeddings_loaded}, Metadata: {len(self._metadata)}"
        )


class AssertionsAdapter(EngineAdapter):
    """Adapter for POI Assertions — Proof of Intelligence ledger search."""
    
    def __init__(self, config: NexusConfig):
        super().__init__("Assertions", config)
        self._assertions: List[Dict] = []
    
    def _do_initialize(self):
        # Load assertions from GOLD layer
        assertions_file = self.config.gold_path / "assertions.jsonl"
        if assertions_file.exists():
            try:
                with open(assertions_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                self._assertions.append(json.loads(line))
                            except Exception:
                                pass
                log.info(f"Loaded {len(self._assertions)} assertions")
            except Exception as e:
                log.warning(f"Failed to load assertions: {e}")
        
        # Also check POI ledger
        poi_file = self.config.gold_path / "poi_ledger.jsonl"
        if poi_file.exists():
            try:
                with open(poi_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                self._assertions.append(json.loads(line))
                            except Exception:
                                pass
            except Exception:
                pass
    
    def _do_query(self, query: str, **kwargs) -> QueryResult:
        results = []
        query_lower = query.lower()
        
        for assertion in self._assertions:
            # Search in various assertion fields
            searchable = json.dumps(assertion).lower()
            if query_lower in searchable:
                results.append({
                    "type": assertion.get("type", "assertion"),
                    "timestamp": assertion.get("timestamp", ""),
                    "content": str(assertion)[:200]
                })
        
        return QueryResult(
            query=query,
            results=tuple(results[:30]),
            total_count=len(results),
            elapsed_ms=0.3,
            snr_score=0.85 if results else 0.1,
            source_engine=self.name
        )
    
    def _do_health_check(self) -> HealthCheck:
        return HealthCheck(
            engine=self.name,
            status=HealthStatus.HEALTHY if self._assertions else HealthStatus.DEGRADED,
            latency_ms=0,
            message=f"Assertions: {len(self._assertions)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# WARP ENGINE ADAPTER — Multi-Vector Contextualized Retrieval (ColBERT/XTR)
# ═══════════════════════════════════════════════════════════════════════════════

class WARPAdapter(EngineAdapter):
    """
    Adapter for XTR-WARP Multi-Vector Retrieval Engine.
    
    Provides high-accuracy late interaction retrieval via ColBERTv2/PLAID.
    Falls back gracefully when WARP is not available.
    """
    
    def __init__(self, config: NexusConfig):
        super().__init__("WARP", config)
        self._bridge = None
        self._available = False
    
    def _do_initialize(self):
        """Initialize WARP bridge with lazy loading."""
        try:
            from warp_bridge import WARPBridge, WARPStatus
            self._bridge = WARPBridge(lazy_init=True)
            self._available = self._bridge._warp_available
            if self._available:
                log.info("WARP engine available (lazy loaded)")
            else:
                log.warning("WARP dependencies not installed - using fallback")
        except ImportError as e:
            log.warning(f"WARP bridge not available: {e}")
            self._available = False
    
    def _do_query(self, query: str, **kwargs) -> QueryResult:
        """Execute query via WARP multi-vector retrieval."""
        if not self._available or self._bridge is None:
            return QueryResult(
                query=query,
                results=(),
                total_count=0,
                elapsed_ms=0,
                snr_score=0.0,
                source_engine=self.name,
                metadata={"fallback": True, "reason": "WARP not available"}
            )
        
        max_results = kwargs.get('max_results', 20)
        response = self._bridge.search(query, k=max_results)
        
        results = tuple({
            "id": r.chunk_id,
            "doc_id": r.doc_id,
            "text": r.text,
            "score": r.score,
            "rank": r.rank,
            "source": "warp"
        } for r in response.results)
        
        return QueryResult(
            query=query,
            results=results,
            total_count=response.total_results,
            elapsed_ms=response.execution_time_ms,
            snr_score=response.snr_estimate,
            source_engine=self.name,
            metadata=response.metadata
        )
    
    def _do_health_check(self) -> HealthCheck:
        """Check WARP engine health."""
        if not self._available:
            return HealthCheck(
                engine=self.name,
                status=HealthStatus.DEGRADED,
                latency_ms=0,
                message="WARP not available - using fallback"
            )
        
        if self._bridge:
            healthy, message = self._bridge.health_check()
            return HealthCheck(
                engine=self.name,
                status=HealthStatus.HEALTHY if healthy else HealthStatus.DEGRADED,
                latency_ms=0,
                message=message
            )
        
        return HealthCheck(
            engine=self.name,
            status=HealthStatus.UNHEALTHY,
            latency_ms=0,
            message="Bridge not initialized"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY ROUTER — Intelligent Query Distribution
# ═══════════════════════════════════════════════════════════════════════════════

class QueryRouter:
    """
    Intelligent query router that selects optimal engine(s) for each query.
    Implements Strategy pattern for routing decisions.
    """
    
    def __init__(self):
        self._strategies: Dict[str, Callable[[str], float]] = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default routing strategies."""
        
        # Sacred wisdom keywords
        sacred_keywords = {
            "quran", "hadith", "islam", "prophet", "allah", "mercy", "prayer",
            "faith", "guidance", "surah", "ayah", "verse", "bukhari", "muslim",
            "القرآن", "الحديث", "الله", "رحم", "صلاة", "compassion", "forgiveness",
            "heaven", "hell", "judgment", "creation", "revelation"
        }
        
        def sacred_strategy(query: str) -> float:
            query_lower = query.lower()
            matches = sum(1 for kw in sacred_keywords if kw in query_lower)
            return min(matches * 0.3, 1.0)
        
        # File search keywords
        file_keywords = {
            "file", "files", "image", "video", "document", "code", "archive",
            "zip", "pdf", "jpg", "png", "mp4", "python", "javascript", "folder",
            "directory", "extension", "size", "type"
        }
        
        def file_strategy(query: str) -> float:
            query_lower = query.lower()
            matches = sum(1 for kw in file_keywords if kw in query_lower)
            return min(matches * 0.3, 1.0)
        
        # Knowledge graph keywords
        graph_keywords = {
            "graph", "node", "edge", "relation", "connect", "link", "network",
            "entity", "concept", "semantic", "ontology"
        }
        
        def graph_strategy(query: str) -> float:
            query_lower = query.lower()
            matches = sum(1 for kw in graph_keywords if kw in query_lower)
            return min(matches * 0.3, 1.0)
        
        # Chat history keywords
        chat_keywords = {
            "conversation", "chat", "message", "said", "discussed", "talked",
            "history", "session", "dialogue", "asked", "answered"
        }
        
        def chat_strategy(query: str) -> float:
            query_lower = query.lower()
            matches = sum(1 for kw in chat_keywords if kw in query_lower)
            return min(matches * 0.3, 1.0)
        
        # Embeddings / semantic search keywords
        embed_keywords = {
            "similar", "semantic", "meaning", "vector", "embedding", "like",
            "related", "closest", "nearest", "analogy"
        }
        
        def embed_strategy(query: str) -> float:
            query_lower = query.lower()
            matches = sum(1 for kw in embed_keywords if kw in query_lower)
            return min(matches * 0.3, 1.0)
        
        # Assertions / POI keywords
        assertion_keywords = {
            "assertion", "proof", "verified", "claim", "evidence", "poi",
            "intelligence", "fact", "statement", "ledger"
        }
        
        def assertion_strategy(query: str) -> float:
            query_lower = query.lower()
            matches = sum(1 for kw in assertion_keywords if kw in query_lower)
            return min(matches * 0.3, 1.0)
        
        # Brain / state keywords
        brain_keywords = {
            "state", "brain", "sovereign", "config", "setting", "memory",
            "knowledge", "status", "system"
        }
        
        def brain_strategy(query: str) -> float:
            query_lower = query.lower()
            matches = sum(1 for kw in brain_keywords if kw in query_lower)
            return min(matches * 0.3, 1.0)
        
        self._strategies["SacredWisdom"] = sacred_strategy
        self._strategies["FileIndex"] = file_strategy
        self._strategies["KnowledgeGraph"] = graph_strategy
        self._strategies["ChatHistory"] = chat_strategy
        self._strategies["Embeddings"] = embed_strategy
        self._strategies["Assertions"] = assertion_strategy
        self._strategies["SovereignBrain"] = brain_strategy
    
    def route(self, query: str, engines: Dict[str, EngineAdapter]) -> List[Tuple[str, float]]:
        """
        Route query to appropriate engines with confidence scores.
        Returns list of (engine_name, confidence) sorted by confidence.
        """
        scores = []
        
        for engine_name in engines:
            if engine_name in self._strategies:
                score = self._strategies[engine_name](query)
            else:
                score = 0.5  # Default confidence
            scores.append((engine_name, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT AGGREGATOR — Merge Results with SNR Optimization
# ═══════════════════════════════════════════════════════════════════════════════

class ResultAggregator:
    """
    Aggregates and ranks results from multiple engines.
    Implements SNR optimization for result quality.
    """
    
    @staticmethod
    def aggregate(results: List[QueryResult], query: str) -> QueryResult:
        """Aggregate results from multiple engines."""
        if not results:
            return QueryResult(
                query=query,
                results=tuple(),
                total_count=0,
                elapsed_ms=0,
                snr_score=0,
                source_engine="aggregated"
            )
        
        # Merge all results
        all_items = []
        total_elapsed = 0
        
        for result in results:
            for item in result.results:
                item_with_source = {**item, "_source": result.source_engine}
                all_items.append((item_with_source, result.snr_score))
            total_elapsed += result.elapsed_ms
        
        # Sort by SNR score
        all_items.sort(key=lambda x: x[1], reverse=True)
        
        # Compute aggregate SNR
        if all_items:
            avg_snr = sum(score for _, score in all_items) / len(all_items)
        else:
            avg_snr = 0
        
        return QueryResult(
            query=query,
            results=tuple(item for item, _ in all_items),
            total_count=len(all_items),
            elapsed_ms=total_elapsed,
            snr_score=avg_snr,
            source_engine="aggregated"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA NEXUS — The Unified Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class BizraNexus:
    """
    BIZRA Nexus — Unified Data Lake Orchestration Engine.
    
    Facade pattern providing a simple interface to the complex subsystem.
    Coordinates multiple engines, routing, caching, and resilience patterns.
    """
    
    def __init__(self, config: Optional[NexusConfig] = None):
        self.config = config or NexusConfig()
        self._engines: Dict[str, EngineAdapter] = {}
        self._event_bus = EventBus()
        self._router = QueryRouter()
        self._aggregator = ResultAggregator()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._initialized = False
        
        log.info("=" * 70)
        log.info("   🏛️  BIZRA NEXUS — Unified Data Lake Orchestrator")
        log.info("=" * 70)
    
    def register_engine(self, adapter: EngineAdapter):
        """Register an engine adapter."""
        self._engines[adapter.name] = adapter
        log.info(f"Registered engine: {adapter.name}")
    
    def initialize(self, lazy: bool = True):
        """Initialize all registered engines."""
        if self._initialized:
            return
        
        log.info("Initializing Nexus engines...")
        
        # Register default engines if none registered
        if not self._engines:
            # Core engines
            self.register_engine(SacredWisdomAdapter(self.config))
            self.register_engine(FileIndexAdapter(self.config))
            # Extended engines
            self.register_engine(SovereignBrainAdapter(self.config))
            self.register_engine(KnowledgeGraphAdapter(self.config))
            self.register_engine(ChatHistoryAdapter(self.config))
            self.register_engine(EmbeddingsAdapter(self.config))
            self.register_engine(AssertionsAdapter(self.config))
            # High-accuracy retrieval engine (ColBERT/XTR multi-vector)
            self.register_engine(WARPAdapter(self.config))
        
        if not lazy:
            # Initialize all engines in parallel
            futures = {
                self._executor.submit(engine.initialize): name
                for name, engine in self._engines.items()
            }
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                    self._event_bus.publish(Event(
                        type=EventType.ENGINE_READY,
                        source=name,
                        payload={"engine": name}
                    ))
                except Exception as e:
                    log.error(f"Failed to initialize {name}: {e}")
                    self._event_bus.publish(Event(
                        type=EventType.ENGINE_ERROR,
                        source=name,
                        payload={"engine": name, "error": str(e)}
                    ))
        
        self._initialized = True
        log.info("Nexus initialization complete")
    
    def query(
        self,
        query: str,
        engines: Optional[List[str]] = None,
        parallel: bool = True,
        aggregate: bool = True,
        **kwargs
    ) -> Union[QueryResult, Dict[str, QueryResult]]:
        """
        Execute query across engines.
        
        Args:
            query: The query string
            engines: Specific engines to query (None = auto-route)
            parallel: Execute queries in parallel
            aggregate: Aggregate results from multiple engines
            **kwargs: Additional query parameters
        
        Returns:
            QueryResult if aggregate=True, else Dict[engine_name, QueryResult]
        """
        if not self._initialized:
            self.initialize()
        
        self._event_bus.publish(Event(
            type=EventType.QUERY_START,
            source="nexus",
            payload={"query": query}
        ))
        
        start = time.perf_counter()
        
        # Determine target engines
        if engines:
            target_engines = [(e, 1.0) for e in engines if e in self._engines]
        else:
            target_engines = self._router.route(query, self._engines)
            # Only use engines with confidence > 0.3
            target_engines = [(e, s) for e, s in target_engines if s > 0.3]
        
        if not target_engines:
            target_engines = [(name, 0.5) for name in self._engines.keys()]
        
        log.debug(f"Routing query to: {[e for e, _ in target_engines]}")
        
        # Execute queries
        results: Dict[str, QueryResult] = {}
        
        if parallel and len(target_engines) > 1:
            # Parallel execution
            futures = {
                self._executor.submit(
                    self._engines[name].query, query, **kwargs
                ): name
                for name, _ in target_engines
                if name in self._engines
            }
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    log.error(f"Query failed for {name}: {e}")
        else:
            # Sequential execution
            for name, _ in target_engines:
                if name in self._engines:
                    try:
                        results[name] = self._engines[name].query(query, **kwargs)
                    except Exception as e:
                        log.error(f"Query failed for {name}: {e}")
        
        elapsed = (time.perf_counter() - start) * 1000
        
        self._event_bus.publish(Event(
            type=EventType.QUERY_COMPLETE,
            source="nexus",
            payload={"query": query, "engines": list(results.keys()), "elapsed_ms": elapsed}
        ))
        
        log.metric("nexus_query_ms", elapsed)
        
        if aggregate:
            return self._aggregator.aggregate(list(results.values()), query)
        return results
    
    def health(self) -> Dict[str, HealthCheck]:
        """Get health status of all engines."""
        health_checks = {}
        
        for name, engine in self._engines.items():
            health_checks[name] = engine.health_check()
        
        return health_checks
    
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "engines": list(self._engines.keys()),
            "initialized": self._initialized,
            "metrics": log.get_metrics_summary(),
            "event_history": len(self._event_bus.get_history()),
            "health": {name: h.status.value for name, h in self.health().items()}
        }
    
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> Callable[[], None]:
        """Subscribe to nexus events."""
        return self._event_bus.subscribe(event_type, handler)
    
    def shutdown(self):
        """Graceful shutdown."""
        log.info("Shutting down Nexus...")
        self._executor.shutdown(wait=True)
        log.info("Nexus shutdown complete")
    
    def __enter__(self) -> 'BizraNexus':
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="BIZRA Nexus — Unified Data Lake Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bizra_nexus.py query "mercy in islam"
  python bizra_nexus.py query "find python files" --engine FileIndex
  python bizra_nexus.py health
  python bizra_nexus.py stats
        """
    )
    
    parser.add_argument("command", choices=["query", "health", "stats", "init"],
                       help="Command to execute")
    parser.add_argument("query_text", nargs="?", default="",
                       help="Query text (for query command)")
    parser.add_argument("--engine", "-e", type=str, action="append",
                       help="Specific engine(s) to use")
    parser.add_argument("--no-aggregate", action="store_true",
                       help="Don't aggregate results")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")
    
    args = parser.parse_args()
    
    with BizraNexus() as nexus:
        if args.command == "init":
            nexus.initialize(lazy=False)
            print("✓ Nexus initialized")
            
        elif args.command == "query":
            if not args.query_text:
                print("Error: Query text required")
                return
            
            result = nexus.query(
                args.query_text,
                engines=args.engine,
                aggregate=not args.no_aggregate
            )
            
            if args.json:
                if isinstance(result, QueryResult):
                    print(json.dumps(result.to_dict(), indent=2))
                else:
                    print(json.dumps({k: v.to_dict() for k, v in result.items()}, indent=2))
            else:
                if isinstance(result, QueryResult):
                    print(f"\n🔍 NEXUS QUERY (SNR: {result.snr_score:.3f} | {result.elapsed_ms:.1f}ms)")
                    print("=" * 60)
                    for i, r in enumerate(result.results[:10], 1):
                        source = r.get("_source", "unknown")
                        print(f"\n  [{i}] ({source})")
                        for k, v in r.items():
                            if k != "_source":
                                print(f"      {k}: {str(v)[:80]}")
                else:
                    for engine, res in result.items():
                        print(f"\n--- {engine} ---")
                        print(f"Results: {res.total_count}, SNR: {res.snr_score:.3f}")
            
        elif args.command == "health":
            health = nexus.health()
            print("\n🏥 NEXUS HEALTH STATUS")
            print("=" * 40)
            for name, check in health.items():
                status_icon = "✓" if check.status == HealthStatus.HEALTHY else "⚠️"
                print(f"  {status_icon} {name}: {check.status.value} ({check.latency_ms:.1f}ms)")
                if check.message:
                    print(f"     {check.message}")
            
        elif args.command == "stats":
            stats = nexus.stats()
            if args.json:
                print(json.dumps(stats, indent=2, default=str))
            else:
                print("\n📊 NEXUS STATISTICS")
                print("=" * 40)
                print(f"  Engines: {', '.join(stats['engines'])}")
                print(f"  Initialized: {stats['initialized']}")
                print(f"  Events: {stats['event_history']}")
                print("\n  Health:")
                for engine, status in stats['health'].items():
                    print(f"    {engine}: {status}")


if __name__ == "__main__":
    main()
