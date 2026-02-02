#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      SOVEREIGN ENGINE BRIDGE v1.0                            ║
║             Integration Layer for BIZRA Orchestrator & Prime                 ║
║                                                                              ║
║  Bridges the high-performance SovereignEngine into the existing ecosystem:  ║
║  • BIZRAOrchestrator (v3.0) — Query processing pipeline                     ║
║  • BizraPrime — Agentic core with PoI tracking                              ║
║  • HypergraphRAGEngine — Context retrieval enhancement                       ║
║  • ARTEEngine — Symbolic-neural tension caching                              ║
║                                                                              ║
║  Author: BIZRA Node-0 | إحسان Integration                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import hashlib

# Import the Sovereign Engine
from sovereign_engine import (
    SovereignEngine, EngineConfig, Event, EventType,
    BloomFilter, BPlusTree, SkipList, LRUCache,
    MetricsCollector, Priority
)

# Import BIZRA configuration
from bizra_config import (
    DATA_LAKE_ROOT, GOLD_PATH, INDEXED_PATH,
    SNR_THRESHOLD, IHSAN_CONSTRAINT
)

# Configure logging
logger = logging.getLogger("sovereign_bridge")


# ═══════════════════════════════════════════════════════════════════════════════
# BRIDGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BridgeConfig:
    """Configuration for the Sovereign Bridge integration."""
    
    # Cache layer configuration
    query_cache_size: int = 50_000
    query_cache_ttl: float = 600.0  # 10 minutes
    
    # Context cache for RAG
    context_cache_size: int = 10_000
    context_cache_ttl: float = 300.0  # 5 minutes
    
    # Embedding cache
    embedding_cache_size: int = 100_000
    embedding_cache_ttl: float = 3600.0  # 1 hour
    
    # Bloom filter for fast negative lookups
    bloom_expected_items: int = 1_000_000
    bloom_false_positive_rate: float = 0.01
    
    # Event sourcing for audit trail
    enable_event_sourcing: bool = True
    snapshot_interval: int = 1000
    
    # Metrics collection
    enable_metrics: bool = True
    
    # Persistence paths
    state_path: Path = field(default_factory=lambda: GOLD_PATH / "sovereign_bridge_state.json")
    events_path: Path = field(default_factory=lambda: GOLD_PATH / "sovereign_events.jsonl")


DEFAULT_BRIDGE_CONFIG = BridgeConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE LAYERS - Specialized Caches for BIZRA Components
# ═══════════════════════════════════════════════════════════════════════════════

class QueryResultCache:
    """
    High-performance cache for query results.
    Integrates with SovereignEngine for persistence and metrics.
    """
    
    __slots__ = ('_engine', '_prefix', '_metrics')
    
    def __init__(self, engine: SovereignEngine, prefix: str = "query"):
        self._engine = engine
        self._prefix = prefix
        self._metrics = engine._metrics
    
    def _make_key(self, query: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from query and parameters."""
        hasher = hashlib.sha256()
        hasher.update(query.encode('utf-8'))
        if params:
            hasher.update(json.dumps(params, sort_keys=True).encode('utf-8'))
        return f"{self._prefix}:{hasher.hexdigest()[:16]}"
    
    def get(self, query: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Get cached query result (with TTL validation)."""
        key = self._make_key(query, params)
        result = self._engine.get(key)

        if result is not None:
            # TTL validation (cache_ttl_seconds enforced at engine level, double-check here)
            cached_at = result.get('cached_at') if isinstance(result, dict) else None
            if cached_at and (time.time() - cached_at) > self._engine._config.cache_ttl_seconds:
                try:
                    self._engine.delete(key)
                except Exception:
                    pass
                self._metrics.counter('query_cache_expired', labels={'prefix': self._prefix})
                return None
            self._metrics.counter('query_cache_hits', labels={'prefix': self._prefix})
        else:
            self._metrics.counter('query_cache_misses', labels={'prefix': self._prefix})

        return result
    
    def put(self, query: str, result: Any, params: Optional[Dict] = None) -> None:
        """Cache a query result."""
        key = self._make_key(query, params)
        self._engine.put(key, {
            'result': result,
            'query': query,
            'params': params,
            'cached_at': time.time()
        })
        self._metrics.counter('query_cache_puts', labels={'prefix': self._prefix})
    
    def invalidate(self, query: str, params: Optional[Dict] = None) -> bool:
        """Invalidate a cached query."""
        key = self._make_key(query, params)
        return self._engine.delete(key)


class EmbeddingCache:
    """
    Cache for document/query embeddings.
    Uses Bloom filter for fast negative lookups.
    """
    
    __slots__ = ('_engine', '_bloom', '_metrics')
    
    def __init__(self, engine: SovereignEngine, expected_items: int = 100_000):
        self._engine = engine
        self._bloom: BloomFilter[str] = BloomFilter(
            expected_items=expected_items,
            false_positive_rate=0.01
        )
        self._metrics = engine._metrics
    
    def _make_key(self, text: str) -> str:
        """Generate cache key for text."""
        return f"emb:{hashlib.sha256(text.encode()).hexdigest()[:24]}"
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding."""
        key = self._make_key(text)
        
        # Fast path: Bloom filter check
        if key not in self._bloom:
            self._metrics.counter('embedding_bloom_rejections')
            return None
        
        result = self._engine.get(key)
        if result is not None:
            self._metrics.counter('embedding_cache_hits')
            return result.get('embedding')
        
        self._metrics.counter('embedding_cache_misses')
        return None
    
    def put(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding."""
        key = self._make_key(text)
        self._bloom.add(key)
        self._engine.put(key, {
            'embedding': embedding,
            'text_hash': hashlib.sha256(text.encode()).hexdigest(),
            'dim': len(embedding),
            'cached_at': time.time()
        })
        self._metrics.counter('embedding_cache_puts')
    
    def batch_get(self, texts: List[str]) -> Dict[str, Optional[List[float]]]:
        """Get multiple embeddings at once."""
        return {text: self.get(text) for text in texts}


class ContextCache:
    """
    Cache for RAG context retrieval results.
    Supports range queries for similar contexts.
    """
    
    __slots__ = ('_engine', '_index', '_metrics')
    
    def __init__(self, engine: SovereignEngine):
        self._engine = engine
        self._index: SkipList[str, str] = SkipList()  # query_hash -> cache_key
        self._metrics = engine._metrics
    
    def _make_key(self, query: str, mode: str) -> str:
        """Generate cache key."""
        hasher = hashlib.sha256()
        hasher.update(query.encode('utf-8'))
        hasher.update(mode.encode('utf-8'))
        return f"ctx:{hasher.hexdigest()[:20]}"
    
    def get(self, query: str, mode: str = "default") -> Optional[Dict]:
        """Get cached context."""
        key = self._make_key(query, mode)
        result = self._engine.get(key)
        
        if result is not None:
            self._metrics.counter('context_cache_hits')
        else:
            self._metrics.counter('context_cache_misses')
        
        return result
    
    def put(
        self, 
        query: str, 
        context: Dict, 
        mode: str = "default",
        sources: Optional[List[Dict]] = None
    ) -> None:
        """Cache a context retrieval result."""
        key = self._make_key(query, mode)
        
        self._engine.put(key, {
            'context': context,
            'sources': sources or [],
            'query': query,
            'mode': mode,
            'cached_at': time.time()
        })
        
        # Index for range queries
        self._index.insert(key, query)
        self._metrics.counter('context_cache_puts')


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT HANDLERS - Integration with BIZRA Events
# ═══════════════════════════════════════════════════════════════════════════════

class BridgeEventType(Enum):
    """Event types for the bridge layer."""
    QUERY_RECEIVED = auto()
    QUERY_COMPLETED = auto()
    CACHE_HIT = auto()
    CACHE_MISS = auto()
    CONTEXT_RETRIEVED = auto()
    RESPONSE_GENERATED = auto()
    AGENT_DISPATCHED = auto()
    POI_RECORDED = auto()
    ERROR_OCCURRED = auto()


@dataclass(frozen=True)
class BridgeEvent:
    """Event in the bridge layer."""
    id: str
    type: BridgeEventType
    timestamp: float
    data: Dict[str, Any]
    source: str  # Component that generated the event
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.name,
            'timestamp': self.timestamp,
            'data': self.data,
            'source': self.source
        }


class EventBus:
    """
    Pub/sub event bus for BIZRA component integration.
    """
    
    __slots__ = ('_subscribers', '_history', '_max_history', '_lock')
    
    def __init__(self, max_history: int = 10000):
        self._subscribers: Dict[BridgeEventType, List[Callable]] = {}
        self._history: List[BridgeEvent] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()
    
    def subscribe(
        self, 
        event_type: BridgeEventType, 
        handler: Callable[[BridgeEvent], None]
    ) -> Callable[[], None]:
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        
        def unsubscribe():
            if handler in self._subscribers.get(event_type, []):
                self._subscribers[event_type].remove(handler)
        
        return unsubscribe
    
    async def publish(self, event: BridgeEvent) -> None:
        """Publish an event to all subscribers."""
        async with self._lock:
            # Store in history
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        
        # Notify subscribers
        handlers = self._subscribers.get(event.type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    def get_history(
        self, 
        event_type: Optional[BridgeEventType] = None,
        since: Optional[float] = None,
        limit: int = 100
    ) -> List[BridgeEvent]:
        """Get event history with optional filtering."""
        events = self._history
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        return events[-limit:]


# ═══════════════════════════════════════════════════════════════════════════════
# SOVEREIGN BRIDGE - Main Integration Class
# ═══════════════════════════════════════════════════════════════════════════════

class SovereignBridge:
    """
    Integration bridge between SovereignEngine and BIZRA ecosystem.
    
    Provides:
    - Unified caching layer for all BIZRA components
    - Event bus for cross-component communication
    - Metrics aggregation and export
    - State persistence and recovery
    - Performance optimization hints
    
    Usage:
        bridge = SovereignBridge()
        await bridge.initialize()
        
        # Cache query results
        bridge.query_cache.put("What is BIZRA?", response)
        
        # Cache embeddings
        bridge.embedding_cache.put("document text", [0.1, 0.2, ...])
        
        # Publish events
        await bridge.publish_event(BridgeEventType.QUERY_COMPLETED, {...})
    """
    
    __slots__ = (
        '_config', '_engine', '_query_cache', '_embedding_cache',
        '_context_cache', '_event_bus', '_metrics', '_initialized',
        '_start_time', '_state'
    )
    
    def __init__(self, config: Optional[BridgeConfig] = None):
        """Initialize the Sovereign Bridge."""
        self._config = config or DEFAULT_BRIDGE_CONFIG
        
        # Create engine with optimized config
        engine_config = EngineConfig(
            cache_max_size=self._config.query_cache_size,
            cache_ttl_seconds=int(self._config.query_cache_ttl),
            bloom_size=self._config.bloom_expected_items,
            bloom_false_positive_rate=self._config.bloom_false_positive_rate,
            event_snapshot_interval=self._config.snapshot_interval,
            metrics_enabled=self._config.enable_metrics
        )
        
        self._engine = SovereignEngine(config=engine_config)
        
        # Initialize cache layers
        self._query_cache = QueryResultCache(self._engine, prefix="qry")
        self._embedding_cache = EmbeddingCache(
            self._engine, 
            expected_items=self._config.embedding_cache_size
        )
        self._context_cache = ContextCache(self._engine)
        
        # Initialize event bus
        self._event_bus = EventBus()
        
        # Reference to engine metrics
        self._metrics = self._engine._metrics
        
        self._initialized = False
        self._start_time = time.time()
        self._state: Dict[str, Any] = {}
        
        logger.info("SovereignBridge created")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and load persisted state."""
        if self._initialized:
            return True
        
        try:
            # Load persisted state
            if self._config.state_path.exists():
                with open(self._config.state_path, 'r') as f:
                    self._state = json.load(f)
                logger.info(f"Loaded state: {len(self._state)} keys")
            
            # Subscribe to system events for persistence
            self._event_bus.subscribe(
                BridgeEventType.QUERY_COMPLETED,
                self._on_query_completed
            )
            
            self._initialized = True
            logger.info("SovereignBridge initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Bridge initialization failed: {e}")
            return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # PROPERTY ACCESSORS
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def engine(self) -> SovereignEngine:
        """Access the underlying SovereignEngine."""
        return self._engine
    
    @property
    def query_cache(self) -> QueryResultCache:
        """Access the query result cache."""
        return self._query_cache
    
    @property
    def embedding_cache(self) -> EmbeddingCache:
        """Access the embedding cache."""
        return self._embedding_cache
    
    @property
    def context_cache(self) -> ContextCache:
        """Access the context cache."""
        return self._context_cache
    
    @property
    def event_bus(self) -> EventBus:
        """Access the event bus."""
        return self._event_bus
    
    # ─────────────────────────────────────────────────────────────────────────
    # EVENT PUBLISHING
    # ─────────────────────────────────────────────────────────────────────────
    
    async def publish_event(
        self,
        event_type: BridgeEventType,
        data: Dict[str, Any],
        source: str = "bridge"
    ) -> None:
        """Publish an event to the event bus."""
        import uuid
        
        event = BridgeEvent(
            id=str(uuid.uuid4()),
            type=event_type,
            timestamp=time.time(),
            data=data,
            source=source
        )
        
        await self._event_bus.publish(event)
        
        # Also log to SovereignEngine event store
        if self._config.enable_event_sourcing:
            engine_event = Event.create(
                event_type=EventType.COMMAND,
                aggregate_id=f"bridge:{event_type.name}",
                data=data
            )
            self._engine._event_store.append(engine_event)
    
    # ─────────────────────────────────────────────────────────────────────────
    # EVENT HANDLERS
    # ─────────────────────────────────────────────────────────────────────────
    
    def _on_query_completed(self, event: BridgeEvent) -> None:
        """Handle query completion for metrics."""
        if 'execution_time' in event.data:
            self._metrics.histogram(
                'query_execution_time',
                event.data['execution_time']
            )
        
        if 'snr_score' in event.data:
            self._metrics.gauge(
                'query_snr_score',
                event.data['snr_score']
            )
    
    # ─────────────────────────────────────────────────────────────────────────
    # ORCHESTRATOR INTEGRATION
    # ─────────────────────────────────────────────────────────────────────────
    
    async def enhance_query(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Enhance a query with cached context and embeddings.
        Called by BIZRAOrchestrator before retrieval.
        """
        result = {
            'original_query': query,
            'cached_context': None,
            'cached_embedding': None,
            'suggestions': []
        }
        
        # Check context cache
        cached_ctx = self._context_cache.get(query)
        if cached_ctx:
            result['cached_context'] = cached_ctx
            await self.publish_event(
                BridgeEventType.CACHE_HIT,
                {'type': 'context', 'query': query[:50]},
                source='orchestrator'
            )
        
        # Check embedding cache
        cached_emb = self._embedding_cache.get(query)
        if cached_emb:
            result['cached_embedding'] = cached_emb
        
        return result
    
    async def cache_orchestrator_result(
        self,
        query: str,
        response: Dict[str, Any],
        context: Optional[Dict] = None
    ) -> None:
        """
        Cache an orchestrator result.
        Called by BIZRAOrchestrator after response generation.
        """
        # Cache the full response
        self._query_cache.put(query, response)
        
        # Cache context if provided
        if context:
            self._context_cache.put(
                query,
                context,
                sources=response.get('sources', [])
            )
        
        # Publish completion event
        await self.publish_event(
            BridgeEventType.QUERY_COMPLETED,
            {
                'query': query[:100],
                'snr_score': response.get('snr_score', 0),
                'execution_time': response.get('execution_time', 0),
                'source_count': len(response.get('sources', []))
            },
            source='orchestrator'
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # PRIME INTEGRATION
    # ─────────────────────────────────────────────────────────────────────────
    
    async def record_agent_dispatch(
        self,
        agent_role: str,
        task: str,
        result: Dict[str, Any]
    ) -> None:
        """
        Record an agent dispatch from BizraPrime.
        """
        # Cache the result
        cache_key = f"agent:{agent_role}:{hashlib.sha256(task.encode()).hexdigest()[:12]}"
        self._engine.put(cache_key, {
            'role': agent_role,
            'task': task,
            'result': result,
            'timestamp': time.time()
        })
        
        # Publish event
        await self.publish_event(
            BridgeEventType.AGENT_DISPATCHED,
            {
                'agent': agent_role,
                'task': task[:100],
                'status': result.get('status', 'unknown')
            },
            source='prime'
        )
    
    async def record_poi(
        self,
        action: str,
        benchmarks: Dict[str, float],
        attestation_hash: str
    ) -> None:
        """
        Record a Proof-of-Impact from BizraPrime.
        """
        # Store in engine
        poi_key = f"poi:{attestation_hash[:16]}"
        self._engine.put(poi_key, {
            'action': action,
            'benchmarks': benchmarks,
            'hash': attestation_hash,
            'timestamp': time.time()
        })
        
        # Publish event
        await self.publish_event(
            BridgeEventType.POI_RECORDED,
            {
                'action': action[:50],
                'total_impact': sum(benchmarks.values())
            },
            source='prime'
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # HYPERGRAPH RAG INTEGRATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_cached_retrieval(
        self,
        query: str,
        mode: str = "default"
    ) -> Optional[Dict]:
        """Get cached RAG retrieval result."""
        return self._context_cache.get(query, mode)
    
    def cache_retrieval(
        self,
        query: str,
        context: Dict,
        mode: str = "default",
        sources: Optional[List[Dict]] = None
    ) -> None:
        """Cache a RAG retrieval result."""
        self._context_cache.put(query, context, mode, sources)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ARTE ENGINE INTEGRATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def cache_tension_analysis(
        self,
        context: str,
        analysis: Dict[str, Any]
    ) -> None:
        """Cache ARTE tension analysis result."""
        key = f"arte:{hashlib.sha256(context.encode()).hexdigest()[:16]}"
        self._engine.put(key, {
            'analysis': analysis,
            'context_hash': hashlib.sha256(context.encode()).hexdigest(),
            'cached_at': time.time()
        })
    
    def get_cached_tension(self, context: str) -> Optional[Dict]:
        """Get cached ARTE tension analysis."""
        key = f"arte:{hashlib.sha256(context.encode()).hexdigest()[:16]}"
        result = self._engine.get(key)
        return result.get('analysis') if result else None
    
    # ─────────────────────────────────────────────────────────────────────────
    # STATISTICS & HEALTH
    # ─────────────────────────────────────────────────────────────────────────
    
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive bridge statistics."""
        engine_stats = self._engine.stats()
        
        return {
            'bridge': {
                'uptime_seconds': time.time() - self._start_time,
                'initialized': self._initialized,
                'event_history_size': len(self._event_bus._history)
            },
            'engine': engine_stats,
            'caches': {
                'query': {
                    'type': 'QueryResultCache',
                    'prefix': self._query_cache._prefix
                },
                'embedding': {
                    'type': 'EmbeddingCache',
                    'bloom_count': self._embedding_cache._bloom.count
                },
                'context': {
                    'type': 'ContextCache',
                    'index_size': len(self._context_cache._index)
                }
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        engine_health = self._engine.health_check()
        
        return {
            'bridge_initialized': self._initialized,
            'engine_status': engine_health['status'],
            'ihsan_score': engine_health['ihsan_score'],
            'uptime_seconds': time.time() - self._start_time,
            'status': 'healthy' if self._initialized and engine_health['status'] == 'healthy' else 'degraded'
        }
    
    def export_metrics(self) -> str:
        """Export Prometheus-format metrics."""
        return self._engine.export_metrics()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────
    
    async def save_state(self) -> None:
        """Persist bridge state."""
        self._state['last_saved'] = datetime.now().isoformat()
        self._state['stats'] = self.stats()
        
        with open(self._config.state_path, 'w') as f:
            json.dump(self._state, f, indent=2, default=str)
        
        logger.info(f"State saved to {self._config.state_path}")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the bridge."""
        logger.info("Shutting down SovereignBridge...")
        
        await self.save_state()
        
        # Export final metrics
        if self._config.enable_metrics:
            metrics_path = GOLD_PATH / "sovereign_metrics_final.txt"
            with open(metrics_path, 'w') as f:
                f.write(self.export_metrics())
        
        logger.info("SovereignBridge shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_bridge_instance: Optional[SovereignBridge] = None


def get_bridge() -> SovereignBridge:
    """Get or create the singleton SovereignBridge instance."""
    global _bridge_instance
    
    if _bridge_instance is None:
        _bridge_instance = SovereignBridge()
    
    return _bridge_instance


async def initialize_bridge() -> SovereignBridge:
    """Initialize and return the bridge instance."""
    bridge = get_bridge()
    await bridge.initialize()
    return bridge


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    """CLI entry point for testing the bridge."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sovereign Bridge CLI')
    parser.add_argument('command', choices=['test', 'health', 'stats', 'demo'])
    args = parser.parse_args()
    
    bridge = await initialize_bridge()
    
    if args.command == 'health':
        health = bridge.health_check()
        print(json.dumps(health, indent=2))
    
    elif args.command == 'stats':
        stats = bridge.stats()
        print(json.dumps(stats, indent=2, default=str))
    
    elif args.command == 'test':
        print("Running bridge tests...")
        
        # Test query cache
        bridge.query_cache.put("test query", {"answer": "test answer"})
        result = bridge.query_cache.get("test query")
        assert result is not None, "Query cache failed"
        print("  ✓ Query cache working")
        
        # Test embedding cache
        bridge.embedding_cache.put("test text", [0.1, 0.2, 0.3])
        emb = bridge.embedding_cache.get("test text")
        assert emb is not None, "Embedding cache failed"
        print("  ✓ Embedding cache working")
        
        # Test context cache
        bridge.context_cache.put("test query", {"context": "test"})
        ctx = bridge.context_cache.get("test query")
        assert ctx is not None, "Context cache failed"
        print("  ✓ Context cache working")
        
        # Test event publishing
        await bridge.publish_event(
            BridgeEventType.QUERY_COMPLETED,
            {'test': True},
            source='test'
        )
        print("  ✓ Event publishing working")
        
        print("\nAll tests passed!")
    
    elif args.command == 'demo':
        print("Running bridge demo...")
        
        # Simulate orchestrator workflow
        query = "What is the BIZRA architecture?"
        
        # Check cache (miss expected)
        cached = bridge.query_cache.get(query)
        print(f"  Cache check: {'HIT' if cached else 'MISS'}")
        
        # Simulate response
        response = {
            'answer': 'BIZRA is a sovereign AI architecture...',
            'snr_score': 0.95,
            'sources': [{'doc': 'architecture.md'}],
            'execution_time': 0.5
        }
        
        # Cache result
        await bridge.cache_orchestrator_result(query, response)
        print("  Result cached")
        
        # Verify cache hit
        cached = bridge.query_cache.get(query)
        print(f"  Cache verify: {'HIT' if cached else 'MISS'}")
        
        # Print stats
        print("\nBridge Statistics:")
        print(json.dumps(bridge.stats(), indent=2, default=str))
    
    await bridge.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
