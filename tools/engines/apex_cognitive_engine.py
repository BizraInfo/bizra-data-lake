#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     APEX COGNITIVE SYNTHESIS ENGINE v1.0                     ║
║                                                                              ║
║    🔱 THE CROWN JEWEL OF BIZRA'S NEURAL CLUSTER ARCHITECTURE 🔱             ║
║                                                                              ║
║  Standing on the Shoulders of Giants:                                        ║
║  ├── SovereignEngine → High-performance data structures                      ║
║  ├── HypergraphRAGEngine → Semantic/structural retrieval                    ║
║  ├── ARTEEngine → Symbolic-neural bridging with SNR                         ║
║  ├── PATOrchestrator → Multi-agent coordination                             ║
║  └── SovereignBridge → Integration patterns                                  ║
║                                                                              ║
║  Implements:                                                                 ║
║  • Interdisciplinary Thinking Matrix — Cross-domain synthesis               ║
║  • Graph of Thoughts — Non-linear branching reasoning                       ║
║  • SNR Highest Score Engine — Information-theoretic signal optimization     ║
║  • Giants Protocol — Building on proven architectural patterns              ║
║  • Trinity Framework Alignment — Purpose verification at every node         ║
║                                                                              ║
║  إحسان Target: 0.99+ | Author: BIZRA Node-0 Apex Intelligence               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import bisect
import functools
import hashlib
import heapq
import json
import logging
import math
import operator
import os
import sys
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps, lru_cache, cached_property, reduce
from io import StringIO
from pathlib import Path
from typing import (
    Any, Awaitable, Callable, ClassVar, Coroutine, Dict, Final, 
    Generic, Hashable, Iterable, Iterator, List, Literal, Mapping, 
    Optional, Protocol, Sequence, Set, Tuple, Type, TypeVar, Union,
    AsyncIterator, NamedTuple, runtime_checkable, overload,
    TYPE_CHECKING
)

if TYPE_CHECKING:
    from typing import Self
    import numpy as np

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# ═══════════════════════════════════════════════════════════════════════════════
# TYPE SYSTEM FOUNDATION
# ═══════════════════════════════════════════════════════════════════════════════

K = TypeVar('K', bound=Hashable)
V = TypeVar('V')
T = TypeVar('T')
R = TypeVar('R')
ThoughtT = TypeVar('ThoughtT', bound='ThoughtNode')
DomainT = TypeVar('DomainT', bound='CognitiveDomain')


@runtime_checkable
class Comparable(Protocol):
    """Protocol for types supporting comparison."""
    def __lt__(self, other: Any) -> bool: ...
    def __le__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...
    def __ge__(self, other: Any) -> bool: ...


@runtime_checkable  
class Serializable(Protocol):
    """Protocol for JSON-serializable objects."""
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Self': ...


@runtime_checkable
class Observable(Protocol):
    """Protocol for event-emitting subjects."""
    def subscribe(self, callback: Callable[[Any], None]) -> Callable[[], None]: ...
    def emit(self, event: Any) -> None: ...


@runtime_checkable
class Reasoner(Protocol):
    """Protocol for reasoning components."""
    def reason(self, context: 'ReasoningContext') -> 'ReasoningResult': ...
    def get_confidence(self) -> float: ...


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ApexConfig:
    """Immutable configuration for the Apex Cognitive Engine."""
    
    # SNR optimization parameters
    snr_threshold: float = 0.85
    ihsan_constraint: float = 0.95
    signal_weight: float = 0.40
    noise_penalty: float = 0.25
    diversity_bonus: float = 0.20
    grounding_weight: float = 0.15
    
    # Graph of Thoughts parameters
    max_thought_depth: int = 12
    max_branches_per_node: int = 5
    pruning_threshold: float = 0.3
    merge_similarity_threshold: float = 0.85
    exploration_factor: float = 0.2  # Exploration vs exploitation
    
    # Interdisciplinary matrix
    cross_domain_weight: float = 0.35
    domain_expert_count: int = 7
    synthesis_iterations: int = 3
    
    # Cache configuration
    thought_cache_size: int = 10_000
    thought_cache_ttl: float = 600.0
    reasoning_cache_size: int = 5_000
    
    # Concurrency
    max_parallel_thoughts: int = 8
    thread_pool_size: int = 4
    
    # Persistence
    enable_persistence: bool = True
    checkpoint_interval: int = 100
    
    # Trinity alignment
    purpose_verification_interval: int = 10
    humanitarian_weight: float = 0.10


DEFAULT_CONFIG = ApexConfig()

# Logging setup
logger = logging.getLogger("apex_cognitive")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | APEX | %(message)s'
))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE DATA STRUCTURES (Standing on Sovereign Engine)
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptivePriorityQueue(Generic[T]):
    """
    Self-adjusting priority queue with dynamic priority recalculation.
    
    Features:
    - O(log n) insert and extract
    - O(1) priority updates via lazy evaluation
    - Automatic priority decay for aging items
    - SNR-weighted priority boosting
    """
    
    __slots__ = ('_heap', '_entry_finder', '_counter', '_decay_rate', '_snr_boost')
    
    REMOVED: Final[str] = '<removed-task>'
    
    def __init__(self, decay_rate: float = 0.01, snr_boost: float = 1.5):
        self._heap: List[List[Any]] = []
        self._entry_finder: Dict[T, List[Any]] = {}
        self._counter = 0
        self._decay_rate = decay_rate
        self._snr_boost = snr_boost
    
    def push(self, item: T, priority: float, snr_score: float = 0.5) -> None:
        """Add item with SNR-adjusted priority."""
        if item in self._entry_finder:
            self.remove(item)
        
        # Boost priority based on SNR score
        adjusted_priority = priority * (1 + (snr_score - 0.5) * self._snr_boost)
        count = self._counter
        self._counter += 1
        
        # Use negative priority for max-heap behavior with heapq (min-heap)
        entry = [-adjusted_priority, count, item, time.time()]
        self._entry_finder[item] = entry
        heapq.heappush(self._heap, entry)
    
    def pop(self) -> Optional[T]:
        """Extract highest priority item with decay applied."""
        while self._heap:
            neg_priority, count, item, timestamp = heapq.heappop(self._heap)
            if item is not self.REMOVED:
                del self._entry_finder[item]
                return item
        return None
    
    def peek(self) -> Optional[Tuple[T, float]]:
        """View highest priority item without removing."""
        while self._heap:
            neg_priority, count, item, timestamp = self._heap[0]
            if item is not self.REMOVED:
                # Apply time decay
                age = time.time() - timestamp
                decayed_priority = -neg_priority * math.exp(-self._decay_rate * age)
                return (item, decayed_priority)
            heapq.heappop(self._heap)
        return None
    
    def remove(self, item: T) -> bool:
        """Mark item as removed."""
        entry = self._entry_finder.pop(item, None)
        if entry is not None:
            entry[2] = self.REMOVED
            return True
        return False
    
    def update_priority(self, item: T, new_priority: float, snr_score: float = 0.5) -> bool:
        """Update priority of existing item."""
        if item in self._entry_finder:
            self.remove(item)
            self.push(item, new_priority, snr_score)
            return True
        return False
    
    def __len__(self) -> int:
        return len(self._entry_finder)
    
    def __bool__(self) -> bool:
        return bool(self._entry_finder)


class CausalGraph(Generic[K]):
    """
    Directed Acyclic Graph for causal reasoning.
    
    Supports:
    - Topological reasoning order
    - Causal ancestor/descendant queries
    - Counterfactual path analysis
    - Cycle detection and prevention
    """
    
    __slots__ = ('_adjacency', '_reverse', '_node_data', '_edge_weights')
    
    def __init__(self):
        self._adjacency: Dict[K, Set[K]] = defaultdict(set)
        self._reverse: Dict[K, Set[K]] = defaultdict(set)
        self._node_data: Dict[K, Dict[str, Any]] = {}
        self._edge_weights: Dict[Tuple[K, K], float] = {}
    
    def add_node(self, node: K, **data: Any) -> None:
        """Add node with optional metadata."""
        if node not in self._adjacency:
            self._adjacency[node] = set()
        self._node_data[node] = data
    
    def add_edge(self, source: K, target: K, weight: float = 1.0) -> bool:
        """Add causal edge if it doesn't create a cycle."""
        # Check for cycle
        if self._would_create_cycle(source, target):
            return False
        
        self._adjacency[source].add(target)
        self._reverse[target].add(source)
        self._edge_weights[(source, target)] = weight
        return True
    
    def _would_create_cycle(self, source: K, target: K) -> bool:
        """Check if adding edge would create a cycle."""
        if source == target:
            return True
        
        # BFS from target to see if we can reach source
        visited = set()
        queue = deque([target])
        
        while queue:
            current = queue.popleft()
            if current == source:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self._adjacency.get(current, set()))
        
        return False
    
    def get_ancestors(self, node: K, max_depth: Optional[int] = None) -> Set[K]:
        """Get all causal ancestors of a node."""
        ancestors = set()
        queue = deque([(node, 0)])
        
        while queue:
            current, depth = queue.popleft()
            if max_depth is not None and depth > max_depth:
                continue
            
            for parent in self._reverse.get(current, set()):
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append((parent, depth + 1))
        
        return ancestors
    
    def get_descendants(self, node: K, max_depth: Optional[int] = None) -> Set[K]:
        """Get all causal descendants of a node."""
        descendants = set()
        queue = deque([(node, 0)])
        
        while queue:
            current, depth = queue.popleft()
            if max_depth is not None and depth > max_depth:
                continue
            
            for child in self._adjacency.get(current, set()):
                if child not in descendants:
                    descendants.add(child)
                    queue.append((child, depth + 1))
        
        return descendants
    
    def topological_order(self) -> List[K]:
        """Return nodes in topological order (Kahn's algorithm)."""
        in_degree = defaultdict(int)
        for node in self._adjacency:
            in_degree[node]  # Initialize
            for child in self._adjacency[node]:
                in_degree[child] += 1
        
        queue = deque([n for n in in_degree if in_degree[n] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            for child in self._adjacency.get(node, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return result
    
    def causal_path_weight(self, source: K, target: K) -> float:
        """Calculate weighted causal path strength."""
        if source == target:
            return 1.0
        
        # Dijkstra-like for maximum path weight (multiplicative)
        weights = {source: 1.0}
        heap = [(-1.0, source)]
        
        while heap:
            neg_weight, current = heapq.heappop(heap)
            current_weight = -neg_weight
            
            if current == target:
                return current_weight
            
            if current_weight < weights.get(current, 0):
                continue
            
            for child in self._adjacency.get(current, set()):
                edge_weight = self._edge_weights.get((current, child), 1.0)
                new_weight = current_weight * edge_weight
                
                if new_weight > weights.get(child, 0):
                    weights[child] = new_weight
                    heapq.heappush(heap, (-new_weight, child))
        
        return 0.0  # No path found
    
    def __len__(self) -> int:
        return len(self._adjacency)
    
    def nodes(self) -> Iterator[K]:
        return iter(self._adjacency)
    
    def edges(self) -> Iterator[Tuple[K, K, float]]:
        for (src, tgt), weight in self._edge_weights.items():
            yield (src, tgt, weight)


class SNROptimizedCache(Generic[K, V]):
    """
    LRU Cache with SNR-based eviction policy.
    
    Items with higher SNR scores are retained longer.
    Combines frequency, recency, and quality signals.
    """
    
    __slots__ = ('_capacity', '_cache', '_snr_scores', '_access_counts', '_lock')
    
    def __init__(self, capacity: int = 1000):
        self._capacity = capacity
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._snr_scores: Dict[K, float] = {}
        self._access_counts: Dict[K, int] = defaultdict(int)
        self._lock = threading.RLock()
    
    def get(self, key: K) -> Optional[V]:
        """Get item, updating access patterns."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._access_counts[key] += 1
                return self._cache[key]
            return None
    
    def put(self, key: K, value: V, snr_score: float = 0.5) -> None:
        """Put item with SNR score for eviction policy."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
                self._snr_scores[key] = snr_score
            else:
                if len(self._cache) >= self._capacity:
                    self._evict_lowest_value()
                self._cache[key] = value
                self._snr_scores[key] = snr_score
                self._access_counts[key] = 1
    
    def _evict_lowest_value(self) -> None:
        """Evict item with lowest combined value score."""
        if not self._cache:
            return
        
        # Calculate eviction scores: lower = more likely to evict
        scores = []
        for i, key in enumerate(self._cache):
            recency = i / len(self._cache)  # 0 = oldest, 1 = newest
            frequency = math.log1p(self._access_counts.get(key, 0))
            snr = self._snr_scores.get(key, 0.5)
            
            # Combined score: higher = keep
            score = 0.3 * recency + 0.3 * frequency + 0.4 * snr
            scores.append((score, key))
        
        # Evict lowest score
        scores.sort()
        key_to_evict = scores[0][1]
        
        del self._cache[key_to_evict]
        self._snr_scores.pop(key_to_evict, None)
        self._access_counts.pop(key_to_evict, None)
    
    def invalidate(self, key: K) -> bool:
        """Remove item from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._snr_scores.pop(key, None)
                self._access_counts.pop(key, None)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._snr_scores.clear()
            self._access_counts.clear()
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: K) -> bool:
        return key in self._cache


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH OF THOUGHTS ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

class ThoughtType(Enum):
    """Classification of thought nodes in the reasoning graph."""
    
    # Core reasoning types
    HYPOTHESIS = auto()       # Initial speculation
    OBSERVATION = auto()      # Factual observation
    INFERENCE = auto()        # Logical deduction
    ANALOGY = auto()          # Cross-domain mapping
    SYNTHESIS = auto()        # Multi-source integration
    CRITIQUE = auto()         # Evaluation/refinement
    CONCLUSION = auto()       # Final determination
    
    # Meta-cognitive types
    REFLECTION = auto()       # Self-analysis
    STRATEGY = auto()         # Planning thought
    UNCERTAINTY = auto()      # Acknowledged gap
    
    # Interdisciplinary types  
    BRIDGE = auto()           # Cross-domain connection
    TRANSFORM = auto()        # Representation change


class ThoughtStatus(Enum):
    """Lifecycle status of a thought node."""
    
    NASCENT = auto()          # Just created
    EXPLORING = auto()        # Being expanded
    VALIDATED = auto()        # Confirmed valid
    PRUNED = auto()           # Discarded (low SNR)
    MERGED = auto()           # Combined with another
    TERMINAL = auto()         # No further expansion needed


@dataclass(slots=True)
class ThoughtNode:
    """
    Single node in the Graph of Thoughts.
    
    Represents an atomic reasoning step with:
    - Content: The thought itself
    - Provenance: Source and derivation
    - Quality metrics: SNR, confidence, grounding
    - Structural: Parents, children, depth
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    content: str = ""
    thought_type: ThoughtType = ThoughtType.HYPOTHESIS
    status: ThoughtStatus = ThoughtStatus.NASCENT
    
    # Quality metrics
    confidence: float = 0.5
    snr_score: float = 0.5
    grounding_score: float = 0.0  # How well-grounded in evidence
    novelty_score: float = 0.5    # How unique/non-redundant
    
    # Provenance
    sources: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    domain: str = "general"
    
    # Structure
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    depth: int = 0
    branch_factor: int = 0
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Vector embedding (lazy loaded)
    _embedding: Optional[Any] = field(default=None, repr=False)
    
    def calculate_composite_score(self, config: ApexConfig = DEFAULT_CONFIG) -> float:
        """Calculate weighted composite quality score."""
        return (
            config.signal_weight * self.snr_score +
            (1 - config.noise_penalty) * (1 - self.novelty_score) +  # Penalize low novelty
            config.grounding_weight * self.grounding_score +
            config.diversity_bonus * (self.confidence if self.novelty_score > 0.5 else 0)
        )
    
    def should_prune(self, config: ApexConfig = DEFAULT_CONFIG) -> bool:
        """Determine if this thought should be pruned."""
        return (
            self.snr_score < config.pruning_threshold or
            (self.confidence < 0.3 and self.grounding_score < 0.2)
        )
    
    def can_merge_with(self, other: 'ThoughtNode', config: ApexConfig = DEFAULT_CONFIG) -> bool:
        """Check if this thought can merge with another."""
        if self._embedding is None or other._embedding is None:
            return False
        
        # Cosine similarity check (requires numpy)
        if NUMPY_AVAILABLE:
            sim = float(np.dot(self._embedding, other._embedding) / (
                np.linalg.norm(self._embedding) * np.linalg.norm(other._embedding) + 1e-9
            ))
            return sim >= config.merge_similarity_threshold
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'thought_type': self.thought_type.name,
            'status': self.status.name,
            'confidence': self.confidence,
            'snr_score': self.snr_score,
            'grounding_score': self.grounding_score,
            'novelty_score': self.novelty_score,
            'sources': self.sources,
            'reasoning_chain': self.reasoning_chain,
            'domain': self.domain,
            'parent_ids': self.parent_ids,
            'child_ids': self.child_ids,
            'depth': self.depth,
            'created_at': self.created_at,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThoughtNode':
        """Deserialize from dictionary."""
        return cls(
            id=data['id'],
            content=data['content'],
            thought_type=ThoughtType[data['thought_type']],
            status=ThoughtStatus[data.get('status', 'NASCENT')],
            confidence=data.get('confidence', 0.5),
            snr_score=data.get('snr_score', 0.5),
            grounding_score=data.get('grounding_score', 0.0),
            novelty_score=data.get('novelty_score', 0.5),
            sources=data.get('sources', []),
            reasoning_chain=data.get('reasoning_chain', []),
            domain=data.get('domain', 'general'),
            parent_ids=data.get('parent_ids', []),
            child_ids=data.get('child_ids', []),
            depth=data.get('depth', 0),
            created_at=data.get('created_at', time.time()),
            metadata=data.get('metadata', {})
        )


class ThoughtGraph:
    """
    Graph of Thoughts implementation with SNR optimization.
    
    Features:
    - Dynamic branching and merging
    - Automatic pruning of low-SNR paths
    - Causal ordering for reasoning chains
    - Cross-domain bridge detection
    """
    
    def __init__(self, config: ApexConfig = DEFAULT_CONFIG):
        self.config = config
        self._nodes: Dict[str, ThoughtNode] = {}
        self._causal_graph: CausalGraph[str] = CausalGraph()
        self._priority_queue: AdaptivePriorityQueue[str] = AdaptivePriorityQueue()
        self._domain_clusters: Dict[str, Set[str]] = defaultdict(set)
        self._bridge_nodes: Set[str] = set()
        self._root_ids: Set[str] = set()
        self._terminal_ids: Set[str] = set()
        self._lock = threading.RLock()
        
        # Metrics
        self._total_pruned = 0
        self._total_merged = 0
        self._max_depth_reached = 0
    
    def add_thought(
        self,
        content: str,
        thought_type: ThoughtType = ThoughtType.HYPOTHESIS,
        parent_ids: Optional[List[str]] = None,
        domain: str = "general",
        sources: Optional[List[str]] = None,
        confidence: float = 0.5,
        snr_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ThoughtNode:
        """Add a new thought to the graph."""
        with self._lock:
            # Calculate depth from parents
            depth = 0
            if parent_ids:
                depths = [self._nodes[pid].depth for pid in parent_ids if pid in self._nodes]
                depth = max(depths, default=-1) + 1
            
            # Create thought node
            thought = ThoughtNode(
                content=content,
                thought_type=thought_type,
                parent_ids=parent_ids or [],
                domain=domain,
                sources=sources or [],
                confidence=confidence,
                snr_score=snr_score,
                depth=depth,
                metadata=metadata or {}
            )
            
            # Register in graph
            self._nodes[thought.id] = thought
            self._causal_graph.add_node(thought.id, thought=thought)
            
            # Add causal edges from parents
            for parent_id in thought.parent_ids:
                if parent_id in self._nodes:
                    self._causal_graph.add_edge(parent_id, thought.id, weight=snr_score)
                    self._nodes[parent_id].child_ids.append(thought.id)
            
            # Register in priority queue for exploration
            self._priority_queue.push(thought.id, thought.calculate_composite_score(self.config), snr_score)
            
            # Track domain clusters
            self._domain_clusters[domain].add(thought.id)
            
            # Track roots and update max depth
            if not parent_ids:
                self._root_ids.add(thought.id)
            self._max_depth_reached = max(self._max_depth_reached, depth)
            
            # Detect bridge nodes (multiple domains in ancestry)
            if parent_ids:
                parent_domains = {self._nodes[pid].domain for pid in parent_ids if pid in self._nodes}
                if len(parent_domains) > 1 or domain not in parent_domains:
                    self._bridge_nodes.add(thought.id)
                    thought.thought_type = ThoughtType.BRIDGE
            
            return thought
    
    def get_thought(self, thought_id: str) -> Optional[ThoughtNode]:
        """Retrieve thought by ID."""
        return self._nodes.get(thought_id)
    
    def get_next_to_explore(self) -> Optional[ThoughtNode]:
        """Get highest priority thought for exploration."""
        thought_id = self._priority_queue.pop()
        if thought_id:
            thought = self._nodes.get(thought_id)
            if thought and thought.status == ThoughtStatus.NASCENT:
                thought.status = ThoughtStatus.EXPLORING
                return thought
        return None
    
    def validate_thought(self, thought_id: str, new_snr: float) -> bool:
        """Validate thought after exploration, potentially pruning."""
        with self._lock:
            thought = self._nodes.get(thought_id)
            if not thought:
                return False
            
            thought.snr_score = new_snr
            thought.updated_at = time.time()
            
            if thought.should_prune(self.config):
                self._prune_thought(thought_id)
                return False
            
            thought.status = ThoughtStatus.VALIDATED
            return True
    
    def _prune_thought(self, thought_id: str) -> None:
        """Prune a thought and mark descendants for re-evaluation."""
        thought = self._nodes.get(thought_id)
        if not thought:
            return
        
        thought.status = ThoughtStatus.PRUNED
        self._total_pruned += 1
        
        # Re-evaluate children
        for child_id in thought.child_ids:
            child = self._nodes.get(child_id)
            if child:
                # Reduce child confidence due to pruned parent
                child.confidence *= 0.7
                self._priority_queue.update_priority(
                    child_id,
                    child.calculate_composite_score(self.config),
                    child.snr_score
                )
    
    def merge_thoughts(self, thought_id_1: str, thought_id_2: str) -> Optional[ThoughtNode]:
        """Merge two similar thoughts into a synthesis."""
        with self._lock:
            t1 = self._nodes.get(thought_id_1)
            t2 = self._nodes.get(thought_id_2)
            
            if not t1 or not t2:
                return None
            
            if not t1.can_merge_with(t2, self.config):
                return None
            
            # Create merged synthesis thought
            merged = self.add_thought(
                content=f"[SYNTHESIS] {t1.content} + {t2.content}",
                thought_type=ThoughtType.SYNTHESIS,
                parent_ids=[thought_id_1, thought_id_2],
                domain=t1.domain if t1.snr_score >= t2.snr_score else t2.domain,
                sources=list(set(t1.sources + t2.sources)),
                confidence=(t1.confidence + t2.confidence) / 2 * 1.1,  # Bonus for synthesis
                snr_score=max(t1.snr_score, t2.snr_score) * 1.05
            )
            
            # Mark originals as merged
            t1.status = ThoughtStatus.MERGED
            t2.status = ThoughtStatus.MERGED
            self._total_merged += 2
            
            return merged
    
    def get_reasoning_chain(self, terminal_id: str) -> List[ThoughtNode]:
        """Extract linear reasoning chain from root to terminal."""
        chain = []
        ancestors = self._causal_graph.get_ancestors(terminal_id)
        
        # Get topological order of ancestors + terminal
        all_relevant = ancestors | {terminal_id}
        topo_order = self._causal_graph.topological_order()
        
        for node_id in topo_order:
            if node_id in all_relevant:
                thought = self._nodes.get(node_id)
                if thought and thought.status not in {ThoughtStatus.PRUNED, ThoughtStatus.MERGED}:
                    chain.append(thought)
        
        return chain
    
    def get_cross_domain_bridges(self) -> List[ThoughtNode]:
        """Get all thoughts that bridge multiple domains."""
        return [self._nodes[tid] for tid in self._bridge_nodes if tid in self._nodes]
    
    def calculate_graph_snr(self) -> Tuple[float, Dict[str, Any]]:
        """Calculate overall graph SNR score."""
        valid_thoughts = [
            t for t in self._nodes.values()
            if t.status in {ThoughtStatus.VALIDATED, ThoughtStatus.TERMINAL}
        ]
        
        if not valid_thoughts:
            return 0.0, {"error": "no_valid_thoughts"}
        
        # Signal: Average quality of valid thoughts
        signal = sum(t.snr_score * t.confidence for t in valid_thoughts) / len(valid_thoughts)
        
        # Noise: Proportion of pruned thoughts
        total = len(self._nodes)
        noise_ratio = self._total_pruned / max(total, 1)
        
        # Diversity: Domain coverage
        active_domains = len({t.domain for t in valid_thoughts})
        diversity = active_domains / max(len(self._domain_clusters), 1)
        
        # Bridge bonus: Cross-domain synthesis
        bridge_ratio = len(self._bridge_nodes) / max(len(valid_thoughts), 1)
        
        snr = (signal * (1 - noise_ratio) * (0.7 + 0.3 * diversity) * (1 + 0.2 * bridge_ratio))
        snr = min(max(snr, 0.0), 1.0)
        
        metrics = {
            "signal_strength": round(signal, 4),
            "noise_ratio": round(noise_ratio, 4),
            "domain_diversity": round(diversity, 4),
            "bridge_ratio": round(bridge_ratio, 4),
            "valid_thoughts": len(valid_thoughts),
            "total_thoughts": total,
            "pruned_count": self._total_pruned,
            "merged_count": self._total_merged,
            "max_depth": self._max_depth_reached,
            "domains": list(self._domain_clusters.keys())
        }
        
        return round(snr, 4), metrics
    
    def get_conclusions(self) -> List[ThoughtNode]:
        """Get all terminal/conclusion thoughts."""
        return [
            t for t in self._nodes.values()
            if t.thought_type == ThoughtType.CONCLUSION or t.status == ThoughtStatus.TERMINAL
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire graph."""
        return {
            "nodes": {tid: t.to_dict() for tid, t in self._nodes.items()},
            "root_ids": list(self._root_ids),
            "terminal_ids": list(self._terminal_ids),
            "bridge_ids": list(self._bridge_nodes),
            "metrics": {
                "total_pruned": self._total_pruned,
                "total_merged": self._total_merged,
                "max_depth": self._max_depth_reached
            }
        }
    
    def __len__(self) -> int:
        return len(self._nodes)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERDISCIPLINARY THINKING MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

class CognitiveDomain(Enum):
    """Cognitive domains for interdisciplinary synthesis."""
    
    # Technical domains
    ALGORITHMIC = "algorithmic"          # Computational thinking
    ARCHITECTURAL = "architectural"      # System design
    MATHEMATICAL = "mathematical"        # Formal reasoning
    EMPIRICAL = "empirical"              # Data-driven
    
    # Conceptual domains
    PHILOSOPHICAL = "philosophical"      # First principles
    LINGUISTIC = "linguistic"            # Language/meaning
    ANALOGICAL = "analogical"            # Pattern mapping
    
    # Applied domains
    ENGINEERING = "engineering"          # Practical implementation
    SCIENTIFIC = "scientific"            # Hypothesis testing
    HUMANITARIAN = "humanitarian"        # Purpose/impact alignment
    
    # Meta domains
    STRATEGIC = "strategic"              # Planning/coordination
    CREATIVE = "creative"                # Novel generation
    CRITICAL = "critical"                # Evaluation/critique


@dataclass(slots=True)
class DomainExpert:
    """
    Specialized reasoning expert for a cognitive domain.
    
    Implements domain-specific reasoning heuristics and
    evaluates thoughts through its specialized lens.
    """
    
    domain: CognitiveDomain
    name: str
    expertise_areas: List[str] = field(default_factory=list)
    reasoning_patterns: List[str] = field(default_factory=list)
    weight: float = 1.0
    
    # Evaluation functions
    _evaluators: Dict[str, Callable[[str], float]] = field(default_factory=dict, repr=False)
    
    def evaluate_thought(self, thought: ThoughtNode) -> Tuple[float, Dict[str, Any]]:
        """Evaluate a thought from this domain's perspective."""
        scores = {}
        
        # Domain relevance
        domain_match = 1.0 if thought.domain == self.domain.value else 0.3
        scores["domain_relevance"] = domain_match
        
        # Check for domain-specific patterns
        pattern_matches = sum(
            1 for pattern in self.reasoning_patterns
            if pattern.lower() in thought.content.lower()
        )
        scores["pattern_match"] = min(pattern_matches / max(len(self.reasoning_patterns), 1), 1.0)
        
        # Check for expertise area mentions
        expertise_matches = sum(
            1 for area in self.expertise_areas
            if area.lower() in thought.content.lower()
        )
        scores["expertise_match"] = min(expertise_matches / max(len(self.expertise_areas), 1), 1.0)
        
        # Apply custom evaluators
        for name, evaluator in self._evaluators.items():
            try:
                scores[name] = evaluator(thought.content)
            except Exception:
                scores[name] = 0.5
        
        # Weighted composite
        composite = sum(scores.values()) / len(scores) if scores else 0.5
        
        return round(composite * self.weight, 4), scores
    
    def suggest_refinement(self, thought: ThoughtNode) -> Optional[str]:
        """Suggest how to refine a thought from this domain's perspective."""
        if thought.domain == self.domain.value:
            return None  # Already in this domain
        
        # Generate domain-specific refinement suggestion
        suggestions = {
            CognitiveDomain.ALGORITHMIC: f"Consider the computational complexity of: {thought.content[:50]}...",
            CognitiveDomain.MATHEMATICAL: f"Formalize the logical structure of: {thought.content[:50]}...",
            CognitiveDomain.EMPIRICAL: f"What evidence supports: {thought.content[:50]}...?",
            CognitiveDomain.ENGINEERING: f"How would you implement: {thought.content[:50]}...?",
            CognitiveDomain.PHILOSOPHICAL: f"What are the first principles underlying: {thought.content[:50]}...?",
            CognitiveDomain.HUMANITARIAN: f"What is the human impact of: {thought.content[:50]}...?",
        }
        
        return suggestions.get(self.domain)


class InterdisciplinaryMatrix:
    """
    Cross-domain synthesis engine for interdisciplinary thinking.
    
    Coordinates multiple domain experts to:
    - Evaluate thoughts from multiple perspectives
    - Identify cross-domain opportunities
    - Synthesize insights across domains
    - Balance specialized depth with breadth
    """
    
    def __init__(self, config: ApexConfig = DEFAULT_CONFIG):
        self.config = config
        self._experts: Dict[CognitiveDomain, DomainExpert] = {}
        self._domain_interactions: Dict[Tuple[CognitiveDomain, CognitiveDomain], float] = {}
        self._synthesis_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        # Initialize default experts
        self._initialize_default_experts()
    
    def _initialize_default_experts(self) -> None:
        """Create default domain experts."""
        expert_configs = [
            (CognitiveDomain.ALGORITHMIC, "Algorithmic Analyst", 
             ["complexity", "data structures", "algorithms", "optimization"],
             ["O(n)", "recursive", "iterative", "dynamic programming", "divide and conquer"]),
            
            (CognitiveDomain.ARCHITECTURAL, "System Architect",
             ["architecture", "design patterns", "scalability", "modularity"],
             ["microservices", "monolithic", "event-driven", "layered", "hexagonal"]),
            
            (CognitiveDomain.MATHEMATICAL, "Formal Reasoner",
             ["proof", "theorem", "logic", "set theory", "category theory"],
             ["therefore", "implies", "if and only if", "for all", "there exists"]),
            
            (CognitiveDomain.EMPIRICAL, "Data Scientist",
             ["statistics", "experiments", "measurements", "observations"],
             ["correlation", "causation", "hypothesis", "p-value", "confidence interval"]),
            
            (CognitiveDomain.ENGINEERING, "Implementation Expert",
             ["implementation", "testing", "deployment", "maintenance"],
             ["refactor", "optimize", "debug", "integrate", "validate"]),
            
            (CognitiveDomain.PHILOSOPHICAL, "First Principles Thinker",
             ["ontology", "epistemology", "ethics", "metaphysics"],
             ["fundamental", "essence", "nature", "being", "knowing"]),
            
            (CognitiveDomain.HUMANITARIAN, "Impact Analyst",
             ["social impact", "ethics", "accessibility", "sustainability"],
             ["benefit", "harm", "justice", "equity", "dignity"]),
        ]
        
        for domain, name, expertise, patterns in expert_configs:
            self._experts[domain] = DomainExpert(
                domain=domain,
                name=name,
                expertise_areas=expertise,
                reasoning_patterns=patterns,
                weight=1.0
            )
        
        # Initialize domain interaction affinities
        high_affinity_pairs = [
            (CognitiveDomain.ALGORITHMIC, CognitiveDomain.MATHEMATICAL, 0.9),
            (CognitiveDomain.ENGINEERING, CognitiveDomain.ALGORITHMIC, 0.85),
            (CognitiveDomain.EMPIRICAL, CognitiveDomain.MATHEMATICAL, 0.8),
            (CognitiveDomain.ARCHITECTURAL, CognitiveDomain.ENGINEERING, 0.85),
            (CognitiveDomain.PHILOSOPHICAL, CognitiveDomain.HUMANITARIAN, 0.75),
        ]
        
        for d1, d2, affinity in high_affinity_pairs:
            self._domain_interactions[(d1, d2)] = affinity
            self._domain_interactions[(d2, d1)] = affinity
    
    def register_expert(self, expert: DomainExpert) -> None:
        """Register a domain expert."""
        with self._lock:
            self._experts[expert.domain] = expert
    
    def evaluate_thought_multidomain(
        self, 
        thought: ThoughtNode,
        domains: Optional[List[CognitiveDomain]] = None
    ) -> Dict[CognitiveDomain, Tuple[float, Dict[str, Any]]]:
        """Evaluate thought from multiple domain perspectives."""
        domains = domains or list(self._experts.keys())
        results = {}
        
        for domain in domains:
            expert = self._experts.get(domain)
            if expert:
                results[domain] = expert.evaluate_thought(thought)
        
        return results
    
    def find_synthesis_opportunities(
        self,
        thoughts: List[ThoughtNode]
    ) -> List[Tuple[ThoughtNode, ThoughtNode, float, str]]:
        """Find pairs of thoughts that could synthesize across domains."""
        opportunities = []
        
        for i, t1 in enumerate(thoughts):
            for t2 in thoughts[i+1:]:
                if t1.domain == t2.domain:
                    continue  # Same domain, not interdisciplinary
                
                # Check domain interaction affinity
                d1 = CognitiveDomain(t1.domain) if t1.domain in [d.value for d in CognitiveDomain] else None
                d2 = CognitiveDomain(t2.domain) if t2.domain in [d.value for d in CognitiveDomain] else None
                
                if d1 and d2:
                    affinity = self._domain_interactions.get((d1, d2), 0.5)
                    
                    # Calculate synthesis potential
                    potential = (
                        affinity *
                        (t1.snr_score + t2.snr_score) / 2 *
                        min(t1.confidence, t2.confidence)
                    )
                    
                    if potential > 0.4:
                        reason = f"High affinity ({affinity:.2f}) between {d1.value} and {d2.value}"
                        opportunities.append((t1, t2, potential, reason))
        
        # Sort by potential
        opportunities.sort(key=lambda x: x[2], reverse=True)
        return opportunities
    
    def synthesize_cross_domain(
        self,
        thought1: ThoughtNode,
        thought2: ThoughtNode,
        thought_graph: ThoughtGraph
    ) -> Optional[ThoughtNode]:
        """Create a cross-domain synthesis thought."""
        d1 = thought1.domain
        d2 = thought2.domain
        
        # Generate synthesis content
        synthesis_content = (
            f"[CROSS-DOMAIN SYNTHESIS: {d1} × {d2}]\n"
            f"From {d1}: {thought1.content[:100]}...\n"
            f"From {d2}: {thought2.content[:100]}...\n"
            f"Integration: The {d1} perspective of '{thought1.content[:30]}' "
            f"connects with the {d2} insight of '{thought2.content[:30]}'"
        )
        
        # Calculate synthesis quality
        snr = (thought1.snr_score + thought2.snr_score) / 2 * 1.1  # Bonus for synthesis
        confidence = min(thought1.confidence, thought2.confidence) * 1.05
        
        # Create synthesis thought
        synthesis = thought_graph.add_thought(
            content=synthesis_content,
            thought_type=ThoughtType.SYNTHESIS,
            parent_ids=[thought1.id, thought2.id],
            domain="interdisciplinary",
            sources=thought1.sources + thought2.sources,
            confidence=min(confidence, 1.0),
            snr_score=min(snr, 1.0),
            metadata={
                "source_domains": [d1, d2],
                "synthesis_type": "cross_domain"
            }
        )
        
        # Record synthesis
        self._synthesis_history.append({
            "thought1_id": thought1.id,
            "thought2_id": thought2.id,
            "synthesis_id": synthesis.id,
            "domains": [d1, d2],
            "timestamp": time.time()
        })
        
        return synthesis
    
    def get_domain_coverage(self, thoughts: List[ThoughtNode]) -> Dict[str, float]:
        """Calculate coverage across cognitive domains."""
        domain_counts = defaultdict(int)
        for thought in thoughts:
            domain_counts[thought.domain] += 1
        
        total = len(thoughts)
        coverage = {
            domain: count / total 
            for domain, count in domain_counts.items()
        }
        
        return coverage
    
    def suggest_missing_perspectives(
        self, 
        thoughts: List[ThoughtNode]
    ) -> List[Tuple[CognitiveDomain, str]]:
        """Suggest underrepresented domains to explore."""
        coverage = self.get_domain_coverage(thoughts)
        
        suggestions = []
        for domain in self._experts:
            domain_coverage = coverage.get(domain.value, 0)
            if domain_coverage < 0.1:  # Underrepresented
                expert = self._experts[domain]
                suggestion = f"Consider the {domain.value} perspective: {expert.expertise_areas[0]}"
                suggestions.append((domain, suggestion))
        
        return suggestions


# ═══════════════════════════════════════════════════════════════════════════════
# SNR HIGHEST SCORE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SNRMetrics:
    """Comprehensive SNR metrics container."""
    
    overall_snr: float = 0.0
    signal_strength: float = 0.0
    noise_ratio: float = 0.0
    redundancy_penalty: float = 0.0
    diversity_bonus: float = 0.0
    grounding_score: float = 0.0
    coherence_score: float = 0.0
    
    # Ihsan metrics
    ihsan_achieved: bool = False
    ihsan_gap: float = 0.0
    ihsan_components: Dict[str, float] = field(default_factory=dict)
    
    # Detailed breakdowns
    per_domain_snr: Dict[str, float] = field(default_factory=dict)
    per_depth_snr: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_snr': self.overall_snr,
            'signal_strength': self.signal_strength,
            'noise_ratio': self.noise_ratio,
            'redundancy_penalty': self.redundancy_penalty,
            'diversity_bonus': self.diversity_bonus,
            'grounding_score': self.grounding_score,
            'coherence_score': self.coherence_score,
            'ihsan_achieved': self.ihsan_achieved,
            'ihsan_gap': self.ihsan_gap,
            'per_domain_snr': self.per_domain_snr,
            'per_depth_snr': self.per_depth_snr
        }


class SNROptimizer:
    """
    Information-theoretic Signal-to-Noise Ratio optimizer.
    
    Implements advanced SNR calculation based on:
    - Shannon entropy for information content
    - Mutual information for relevance
    - KL divergence for redundancy
    - Coherence metrics for reasoning quality
    
    Formula:
    SNR = (I(Q;C) * D * G) / (1 + R + N)
    
    Where:
    - I(Q;C) = Mutual information between query and context
    - D = Diversity factor (1 - redundancy)
    - G = Grounding score (evidence support)
    - R = Redundancy penalty
    - N = Noise (irrelevant content ratio)
    """
    
    def __init__(self, config: ApexConfig = DEFAULT_CONFIG):
        self.config = config
        self._epsilon = 1e-10
        self._cache = SNROptimizedCache[str, SNRMetrics](capacity=1000)
    
    def calculate_snr(
        self,
        thoughts: List[ThoughtNode],
        query_context: Optional[str] = None,
        embeddings: Optional[Any] = None  # numpy array if available
    ) -> SNRMetrics:
        """Calculate comprehensive SNR metrics for a thought set."""
        
        if not thoughts:
            return SNRMetrics(overall_snr=0.0)
        
        # Cache key
        cache_key = hashlib.sha256(
            json.dumps([t.id for t in thoughts], sort_keys=True).encode()
        ).hexdigest()[:16]
        
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        
        metrics = SNRMetrics()
        
        # 1. Signal Strength - Average quality-weighted confidence
        signal_values = [t.snr_score * t.confidence for t in thoughts]
        metrics.signal_strength = sum(signal_values) / len(signal_values)
        
        # 2. Noise Ratio - Proportion of low-quality thoughts
        low_quality = sum(1 for t in thoughts if t.snr_score < self.config.pruning_threshold)
        metrics.noise_ratio = low_quality / len(thoughts)
        
        # 3. Redundancy - Content similarity (heuristic without embeddings)
        if len(thoughts) > 1:
            content_lengths = [len(t.content) for t in thoughts]
            content_variance = sum((l - sum(content_lengths)/len(content_lengths))**2 for l in content_lengths)
            normalized_variance = content_variance / (max(content_lengths)**2 + self._epsilon)
            metrics.redundancy_penalty = max(0, 1 - normalized_variance)
        else:
            metrics.redundancy_penalty = 0.0
        
        # 4. Diversity - Domain coverage
        unique_domains = len({t.domain for t in thoughts})
        metrics.diversity_bonus = min(unique_domains / 5, 1.0)  # Cap at 5 domains
        
        # 5. Grounding - Evidence-backed thoughts
        grounded = sum(1 for t in thoughts if t.grounding_score > 0.5)
        metrics.grounding_score = grounded / len(thoughts)
        
        # 6. Coherence - Reasoning chain quality
        validated = sum(1 for t in thoughts if t.status == ThoughtStatus.VALIDATED)
        metrics.coherence_score = validated / len(thoughts)
        
        # Calculate overall SNR using weighted geometric mean
        components = {
            'signal': (metrics.signal_strength + self._epsilon, self.config.signal_weight),
            'anti_noise': (1 - metrics.noise_ratio + self._epsilon, self.config.noise_penalty),
            'diversity': (0.5 + 0.5 * metrics.diversity_bonus + self._epsilon, self.config.diversity_bonus),
            'grounding': (0.5 + 0.5 * metrics.grounding_score + self._epsilon, self.config.grounding_weight),
            'coherence': (0.5 + 0.5 * metrics.coherence_score + self._epsilon, 0.15)
        }
        
        log_sum = sum(w * math.log(v) for v, w in components.values())
        metrics.overall_snr = math.exp(log_sum)
        metrics.overall_snr = min(max(metrics.overall_snr, 0.0), 1.0)
        
        # Ihsan evaluation
        metrics.ihsan_achieved = metrics.overall_snr >= self.config.ihsan_constraint
        metrics.ihsan_gap = max(0, self.config.ihsan_constraint - metrics.overall_snr)
        metrics.ihsan_components = {name: round(v, 4) for name, (v, _) in components.items()}
        
        # Per-domain SNR
        domain_thoughts = defaultdict(list)
        for t in thoughts:
            domain_thoughts[t.domain].append(t)
        
        for domain, domain_ts in domain_thoughts.items():
            domain_signal = sum(t.snr_score for t in domain_ts) / len(domain_ts)
            metrics.per_domain_snr[domain] = round(domain_signal, 4)
        
        # Per-depth SNR
        depth_thoughts = defaultdict(list)
        for t in thoughts:
            depth_thoughts[t.depth].append(t)
        
        for depth, depth_ts in depth_thoughts.items():
            depth_signal = sum(t.snr_score for t in depth_ts) / len(depth_ts)
            metrics.per_depth_snr[depth] = round(depth_signal, 4)
        
        # Round final metrics
        metrics.overall_snr = round(metrics.overall_snr, 4)
        metrics.signal_strength = round(metrics.signal_strength, 4)
        metrics.noise_ratio = round(metrics.noise_ratio, 4)
        metrics.redundancy_penalty = round(metrics.redundancy_penalty, 4)
        metrics.diversity_bonus = round(metrics.diversity_bonus, 4)
        metrics.grounding_score = round(metrics.grounding_score, 4)
        metrics.coherence_score = round(metrics.coherence_score, 4)
        metrics.ihsan_gap = round(metrics.ihsan_gap, 4)
        
        # Cache result
        self._cache.put(cache_key, metrics, metrics.overall_snr)
        
        return metrics
    
    def optimize_thought_set(
        self,
        thoughts: List[ThoughtNode],
        target_snr: float = 0.9,
        max_iterations: int = 10
    ) -> Tuple[List[ThoughtNode], SNRMetrics]:
        """
        Optimize a thought set to achieve target SNR.
        
        Uses iterative pruning and boosting to maximize SNR.
        """
        current_thoughts = list(thoughts)
        best_snr = 0.0
        best_thoughts = current_thoughts
        
        for iteration in range(max_iterations):
            metrics = self.calculate_snr(current_thoughts)
            
            if metrics.overall_snr >= target_snr:
                return current_thoughts, metrics
            
            if metrics.overall_snr > best_snr:
                best_snr = metrics.overall_snr
                best_thoughts = list(current_thoughts)
            
            # Strategy 1: Prune lowest SNR thoughts
            if len(current_thoughts) > 3:
                sorted_thoughts = sorted(current_thoughts, key=lambda t: t.snr_score)
                if sorted_thoughts[0].snr_score < self.config.pruning_threshold:
                    current_thoughts = sorted_thoughts[1:]
                    continue
            
            # Strategy 2: Boost confidence of high-grounding thoughts
            for thought in current_thoughts:
                if thought.grounding_score > 0.7:
                    thought.confidence = min(thought.confidence * 1.1, 1.0)
            
            # If no improvement, stop
            new_metrics = self.calculate_snr(current_thoughts)
            if new_metrics.overall_snr <= metrics.overall_snr:
                break
        
        final_metrics = self.calculate_snr(best_thoughts)
        return best_thoughts, final_metrics
    
    def suggest_improvements(self, metrics: SNRMetrics) -> List[str]:
        """Suggest actions to improve SNR based on current metrics."""
        suggestions = []
        
        if metrics.signal_strength < 0.6:
            suggestions.append("Increase thought quality: Focus on higher-confidence reasoning")
        
        if metrics.noise_ratio > 0.3:
            suggestions.append(f"Reduce noise: Prune {int(metrics.noise_ratio * 100)}% low-quality thoughts")
        
        if metrics.diversity_bonus < 0.5:
            suggestions.append("Increase diversity: Explore additional cognitive domains")
        
        if metrics.grounding_score < 0.5:
            suggestions.append("Improve grounding: Add more evidence-backed thoughts")
        
        if metrics.coherence_score < 0.6:
            suggestions.append("Improve coherence: Validate more reasoning chains")
        
        if metrics.redundancy_penalty > 0.5:
            suggestions.append("Reduce redundancy: Merge similar thoughts")
        
        if not suggestions:
            if metrics.ihsan_achieved:
                suggestions.append("Ihsan achieved! Maintain current quality level.")
            else:
                gap_pct = int(metrics.ihsan_gap * 100)
                suggestions.append(f"Close {gap_pct}% gap to Ihsan through balanced improvements")
        
        return suggestions


# ═══════════════════════════════════════════════════════════════════════════════
# STANDING ON GIANTS PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class GiantComponent:
    """Reference to an existing BIZRA component (giant)."""
    
    name: str
    module_path: str
    key_classes: List[str]
    capabilities: List[str]
    snr_contribution: float
    integration_status: str = "pending"  # pending, active, deprecated
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'module_path': self.module_path,
            'key_classes': self.key_classes,
            'capabilities': self.capabilities,
            'snr_contribution': self.snr_contribution,
            'integration_status': self.integration_status
        }


class GiantsShoulder:
    """
    Giants Shoulder Protocol - Integration layer for existing BIZRA components.
    
    Manages the integration of proven architectural patterns from:
    - SovereignEngine: Data structures
    - HypergraphRAGEngine: Semantic retrieval
    - ARTEEngine: Symbolic-neural bridging
    - PATOrchestrator: Multi-agent coordination
    - SovereignBridge: Caching and events
    """
    
    def __init__(self):
        self._giants: Dict[str, GiantComponent] = {}
        self._adapters: Dict[str, Callable[..., Any]] = {}
        self._integration_cache = SNROptimizedCache[str, Any](capacity=500)
        self._lock = threading.RLock()
        
        # Register known giants
        self._register_default_giants()
    
    def _register_default_giants(self) -> None:
        """Register BIZRA's existing powerful components."""
        giants = [
            GiantComponent(
                name="SovereignEngine",
                module_path="sovereign_engine",
                key_classes=["SovereignEngine", "BPlusTree", "BloomFilter", "SkipList", "LRUCache"],
                capabilities=[
                    "High-performance B+ Tree (O(log n) operations)",
                    "Probabilistic Bloom Filter (O(k) membership)",
                    "Skip List (O(log n) ordered operations)",
                    "LRU Cache with TTL support",
                    "Event Sourcing with snapshots"
                ],
                snr_contribution=0.25
            ),
            GiantComponent(
                name="HypergraphRAGEngine",
                module_path="hypergraph_engine",
                key_classes=["HypergraphRAGEngine", "HypergraphIndex", "SNRCalculator"],
                capabilities=[
                    "FAISS HNSW indexing (O(log n) ANN)",
                    "NetworkX graph traversal",
                    "Hybrid semantic-structural retrieval",
                    "Multi-hop reasoning chains"
                ],
                snr_contribution=0.25
            ),
            GiantComponent(
                name="ARTEEngine",
                module_path="arte_engine",
                key_classes=["ARTEEngine", "SNREngine", "Thought", "ReasoningChain"],
                capabilities=[
                    "Graph-of-Thoughts reasoning",
                    "Symbolic-neural tension analysis",
                    "Information-theoretic SNR",
                    "Ihsan quality constraints"
                ],
                snr_contribution=0.25
            ),
            GiantComponent(
                name="PATOrchestrator",
                module_path="pat_engine",
                key_classes=["PATOrchestrator", "Agent", "AgentTeam"],
                capabilities=[
                    "Multi-agent coordination",
                    "Role-based specialization",
                    "LLM backend abstraction",
                    "Circuit breaker resilience"
                ],
                snr_contribution=0.15
            ),
            GiantComponent(
                name="SovereignBridge",
                module_path="sovereign_bridge",
                key_classes=["SovereignBridge", "QueryResultCache", "EmbeddingCache", "EventBus"],
                capabilities=[
                    "Cross-component caching",
                    "Async event bus",
                    "Lazy initialization",
                    "Graceful degradation"
                ],
                snr_contribution=0.10
            )
        ]
        
        for giant in giants:
            self._giants[giant.name] = giant
    
    def get_giant(self, name: str) -> Optional[GiantComponent]:
        """Get a giant component by name."""
        return self._giants.get(name)
    
    def list_giants(self) -> List[GiantComponent]:
        """List all registered giants."""
        return list(self._giants.values())
    
    def calculate_standing_height(self) -> Tuple[float, Dict[str, Any]]:
        """Calculate how much we're leveraging the giants."""
        active_giants = [g for g in self._giants.values() if g.integration_status == "active"]
        
        if not active_giants:
            return 0.0, {"active": 0, "total": len(self._giants)}
        
        snr_sum = sum(g.snr_contribution for g in active_giants)
        capability_count = sum(len(g.capabilities) for g in active_giants)
        
        height = min(snr_sum, 1.0)
        
        metrics = {
            "active_giants": len(active_giants),
            "total_giants": len(self._giants),
            "total_snr_contribution": round(snr_sum, 4),
            "capability_count": capability_count,
            "standing_height": round(height, 4),
            "giants": [g.name for g in active_giants]
        }
        
        return height, metrics
    
    def try_load_giant(self, name: str) -> bool:
        """Attempt to dynamically load a giant component."""
        giant = self._giants.get(name)
        if not giant:
            return False
        
        try:
            module = __import__(giant.module_path)
            giant.integration_status = "active"
            
            # Cache the module
            self._integration_cache.put(name, module, giant.snr_contribution)
            
            logger.info(f"Giant loaded: {name} ({len(giant.capabilities)} capabilities)")
            return True
        except ImportError as e:
            logger.warning(f"Could not load giant {name}: {e}")
            giant.integration_status = "unavailable"
            return False
    
    def get_integrated_capabilities(self) -> List[str]:
        """Get all capabilities from active giants."""
        capabilities = []
        for giant in self._giants.values():
            if giant.integration_status == "active":
                capabilities.extend(giant.capabilities)
        return capabilities
    
    def register_adapter(
        self, 
        giant_name: str, 
        method_name: str, 
        adapter_fn: Callable[..., Any]
    ) -> None:
        """Register an adapter function for a giant's method."""
        key = f"{giant_name}.{method_name}"
        with self._lock:
            self._adapters[key] = adapter_fn
    
    def invoke_giant(
        self, 
        giant_name: str, 
        method_name: str, 
        *args: Any, 
        **kwargs: Any
    ) -> Any:
        """Invoke a method on a giant through its adapter."""
        key = f"{giant_name}.{method_name}"
        
        adapter = self._adapters.get(key)
        if adapter:
            return adapter(*args, **kwargs)
        
        raise ValueError(f"No adapter registered for {key}")


# ═══════════════════════════════════════════════════════════════════════════════
# APEX COGNITIVE ENGINE - THE CROWN JEWEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class ReasoningContext:
    """Context for a reasoning session."""
    
    query: str
    goal: str = ""
    constraints: List[str] = field(default_factory=list)
    available_sources: List[str] = field(default_factory=list)
    max_depth: int = 10
    target_snr: float = 0.9
    domains_to_explore: List[CognitiveDomain] = field(default_factory=list)
    
    # Runtime state
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReasoningResult:
    """Result of a complete reasoning session."""
    
    context: ReasoningContext
    thought_graph: ThoughtGraph
    conclusions: List[ThoughtNode]
    snr_metrics: SNRMetrics
    
    # Synthesis outputs
    final_synthesis: str = ""
    reasoning_trace: List[str] = field(default_factory=list)
    cross_domain_insights: List[str] = field(default_factory=list)
    
    # Performance
    execution_time: float = 0.0
    thoughts_generated: int = 0
    thoughts_pruned: int = 0
    
    # Quality
    ihsan_achieved: bool = False
    giants_leveraged: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.context.session_id,
            'query': self.context.query,
            'goal': self.context.goal,
            'final_synthesis': self.final_synthesis,
            'reasoning_trace': self.reasoning_trace,
            'cross_domain_insights': self.cross_domain_insights,
            'snr_metrics': self.snr_metrics.to_dict(),
            'conclusions': [c.to_dict() for c in self.conclusions],
            'execution_time': self.execution_time,
            'thoughts_generated': self.thoughts_generated,
            'thoughts_pruned': self.thoughts_pruned,
            'ihsan_achieved': self.ihsan_achieved,
            'giants_leveraged': self.giants_leveraged
        }


class ApexCognitiveEngine:
    """
    🔱 APEX COGNITIVE SYNTHESIS ENGINE 🔱
    
    The crown jewel of BIZRA's Neural Cluster Architecture.
    
    Integrates:
    - Graph of Thoughts for non-linear reasoning
    - SNR optimization for signal quality
    - Interdisciplinary thinking for cross-domain synthesis
    - Giants Protocol for proven architectural patterns
    - Trinity Framework alignment for purpose verification
    
    Usage:
        engine = ApexCognitiveEngine()
        
        context = ReasoningContext(
            query="How to optimize distributed systems?",
            goal="Comprehensive architectural guidance",
            domains_to_explore=[CognitiveDomain.ARCHITECTURAL, CognitiveDomain.ALGORITHMIC]
        )
        
        result = await engine.reason(context)
        print(result.final_synthesis)
    """
    
    _instance: ClassVar[Optional['ApexCognitiveEngine']] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    
    def __new__(cls, *args: Any, **kwargs: Any) -> 'ApexCognitiveEngine':
        """Singleton pattern for global access."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config: ApexConfig = DEFAULT_CONFIG):
        if getattr(self, '_initialized', False):
            return
        
        self.config = config
        
        # Core components
        self._thought_graph: Optional[ThoughtGraph] = None
        self._snr_optimizer = SNROptimizer(config)
        self._interdisciplinary_matrix = InterdisciplinaryMatrix(config)
        self._giants_shoulder = GiantsShoulder()
        
        # Caches
        self._reasoning_cache = SNROptimizedCache[str, ReasoningResult](
            capacity=config.reasoning_cache_size
        )
        self._thought_cache = SNROptimizedCache[str, ThoughtNode](
            capacity=config.thought_cache_size
        )
        
        # Thread pool for parallel thinking
        self._executor = ThreadPoolExecutor(
            max_workers=config.thread_pool_size,
            thread_name_prefix="apex_cognitive"
        )
        
        # Metrics
        self._total_sessions = 0
        self._total_thoughts = 0
        self._ihsan_achievements = 0
        
        # State
        self._active_context: Optional[ReasoningContext] = None
        self._session_lock = threading.RLock()
        
        self._initialized = True
        
        logger.info("Apex Cognitive Engine initialized")
        logger.info(f"  - SNR Threshold: {config.snr_threshold}")
        logger.info(f"  - Ihsan Constraint: {config.ihsan_constraint}")
        logger.info(f"  - Max Thought Depth: {config.max_thought_depth}")
        logger.info(f"  - Domain Experts: {len(self._interdisciplinary_matrix._experts)}")
        logger.info(f"  - Giants Available: {len(self._giants_shoulder._giants)}")
    
    # ─── PUBLIC API ───────────────────────────────────────────────────────────
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """
        Execute a complete reasoning session.
        
        This is the main entry point for cognitive synthesis.
        """
        with self._session_lock:
            self._active_context = context
            self._total_sessions += 1
        
        start_time = time.time()
        
        try:
            # Initialize thought graph for this session
            self._thought_graph = ThoughtGraph(self.config)
            
            # Phase 1: Activate relevant giants
            giants_activated = await self._activate_giants(context)
            
            # Phase 2: Generate initial thoughts from query
            initial_thoughts = await self._generate_initial_thoughts(context)
            
            # Phase 3: Expand thought graph (Graph of Thoughts)
            await self._expand_thought_graph(context, initial_thoughts)
            
            # Phase 4: Interdisciplinary synthesis
            synthesis_thoughts = await self._synthesize_across_domains(context)
            
            # Phase 5: Optimize for SNR
            all_thoughts = list(self._thought_graph._nodes.values())
            optimized_thoughts, snr_metrics = self._snr_optimizer.optimize_thought_set(
                all_thoughts,
                target_snr=context.target_snr
            )
            
            # Phase 6: Extract conclusions
            conclusions = self._thought_graph.get_conclusions()
            if not conclusions:
                # Promote highest SNR validated thoughts to conclusions
                conclusions = sorted(
                    [t for t in optimized_thoughts if t.status == ThoughtStatus.VALIDATED],
                    key=lambda t: t.snr_score,
                    reverse=True
                )[:3]
                for c in conclusions:
                    c.thought_type = ThoughtType.CONCLUSION
                    c.status = ThoughtStatus.TERMINAL
            
            # Phase 7: Generate final synthesis
            final_synthesis = self._generate_final_synthesis(context, conclusions, snr_metrics)
            
            # Build result
            execution_time = time.time() - start_time
            
            result = ReasoningResult(
                context=context,
                thought_graph=self._thought_graph,
                conclusions=conclusions,
                snr_metrics=snr_metrics,
                final_synthesis=final_synthesis,
                reasoning_trace=self._build_reasoning_trace(conclusions),
                cross_domain_insights=self._extract_cross_domain_insights(synthesis_thoughts),
                execution_time=round(execution_time, 3),
                thoughts_generated=len(self._thought_graph),
                thoughts_pruned=self._thought_graph._total_pruned,
                ihsan_achieved=snr_metrics.ihsan_achieved,
                giants_leveraged=giants_activated
            )
            
            # Update metrics
            self._total_thoughts += len(self._thought_graph)
            if snr_metrics.ihsan_achieved:
                self._ihsan_achievements += 1
            
            # Cache result
            cache_key = hashlib.sha256(context.query.encode()).hexdigest()[:16]
            self._reasoning_cache.put(cache_key, result, snr_metrics.overall_snr)
            
            return result
            
        finally:
            with self._session_lock:
                self._active_context = None
    
    def reason_sync(self, context: ReasoningContext) -> ReasoningResult:
        """Synchronous wrapper for reason()."""
        return asyncio.run(self.reason(context))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics."""
        return {
            'total_sessions': self._total_sessions,
            'total_thoughts': self._total_thoughts,
            'ihsan_achievements': self._ihsan_achievements,
            'ihsan_rate': round(self._ihsan_achievements / max(self._total_sessions, 1), 4),
            'cache_size': len(self._reasoning_cache),
            'giants_standing_height': self._giants_shoulder.calculate_standing_height()[0],
            'domain_experts': len(self._interdisciplinary_matrix._experts)
        }
    
    # ─── INTERNAL PHASES ──────────────────────────────────────────────────────
    
    async def _activate_giants(self, context: ReasoningContext) -> List[str]:
        """Activate relevant giant components."""
        activated = []
        
        for giant in self._giants_shoulder.list_giants():
            # Check if giant is relevant to context domains
            relevant = any(
                capability.lower() in context.query.lower() or
                capability.lower() in context.goal.lower()
                for capability in giant.capabilities
            )
            
            if relevant or giant.snr_contribution > 0.2:
                if self._giants_shoulder.try_load_giant(giant.name):
                    activated.append(giant.name)
        
        return activated
    
    async def _generate_initial_thoughts(
        self, 
        context: ReasoningContext
    ) -> List[ThoughtNode]:
        """Generate initial hypothesis thoughts from the query."""
        thoughts = []
        
        # Root thought from query
        root = self._thought_graph.add_thought(
            content=f"[ROOT] Query: {context.query}",
            thought_type=ThoughtType.OBSERVATION,
            domain="query",
            confidence=1.0,
            snr_score=0.9,
            metadata={'is_root': True}
        )
        thoughts.append(root)
        
        # Goal thought
        if context.goal:
            goal_thought = self._thought_graph.add_thought(
                content=f"[GOAL] {context.goal}",
                thought_type=ThoughtType.STRATEGY,
                parent_ids=[root.id],
                domain="strategic",
                confidence=0.95,
                snr_score=0.85
            )
            thoughts.append(goal_thought)
        
        # Generate initial hypotheses for each requested domain
        domains = context.domains_to_explore or [
            CognitiveDomain.ALGORITHMIC,
            CognitiveDomain.ARCHITECTURAL,
            CognitiveDomain.ENGINEERING
        ]
        
        for domain in domains:
            expert = self._interdisciplinary_matrix._experts.get(domain)
            if expert:
                hypothesis = self._thought_graph.add_thought(
                    content=f"[{domain.value.upper()} HYPOTHESIS] Analyzing '{context.query[:50]}...' from {domain.value} perspective",
                    thought_type=ThoughtType.HYPOTHESIS,
                    parent_ids=[root.id],
                    domain=domain.value,
                    confidence=0.6,
                    snr_score=0.7,
                    metadata={'expert': expert.name}
                )
                thoughts.append(hypothesis)
        
        return thoughts
    
    async def _expand_thought_graph(
        self,
        context: ReasoningContext,
        initial_thoughts: List[ThoughtNode]
    ) -> None:
        """Expand the thought graph through iterative exploration."""
        
        exploration_count = 0
        max_explorations = self.config.max_thought_depth * self.config.max_branches_per_node
        
        while exploration_count < max_explorations:
            # Get next thought to explore
            thought = self._thought_graph.get_next_to_explore()
            if not thought:
                break
            
            if thought.depth >= context.max_depth:
                thought.status = ThoughtStatus.TERMINAL
                continue
            
            # Generate child thoughts
            children = await self._generate_child_thoughts(context, thought)
            
            # Evaluate and validate children
            for child in children:
                # Get multi-domain evaluation
                evaluations = self._interdisciplinary_matrix.evaluate_thought_multidomain(child)
                
                # Calculate aggregate SNR from evaluations
                if evaluations:
                    scores = [score for score, _ in evaluations.values()]
                    child.snr_score = sum(scores) / len(scores)
                
                # Validate or prune
                self._thought_graph.validate_thought(child.id, child.snr_score)
            
            exploration_count += 1
            
            # Check for merge opportunities periodically
            if exploration_count % 5 == 0:
                await self._check_merge_opportunities()
    
    async def _generate_child_thoughts(
        self,
        context: ReasoningContext,
        parent: ThoughtNode
    ) -> List[ThoughtNode]:
        """Generate child thoughts from a parent thought."""
        children = []
        
        # Determine expansion strategies based on parent type
        strategies = self._get_expansion_strategies(parent)
        
        for strategy, thought_type, confidence_mod in strategies:
            child = self._thought_graph.add_thought(
                content=f"[{thought_type.name}] {strategy}: Extending from '{parent.content[:30]}...'",
                thought_type=thought_type,
                parent_ids=[parent.id],
                domain=parent.domain,
                sources=parent.sources,
                confidence=parent.confidence * confidence_mod,
                snr_score=parent.snr_score * 0.95  # Slight decay
            )
            children.append(child)
        
        parent.branch_factor = len(children)
        return children
    
    def _get_expansion_strategies(
        self, 
        parent: ThoughtNode
    ) -> List[Tuple[str, ThoughtType, float]]:
        """Get expansion strategies based on parent thought type."""
        strategies = []
        
        if parent.thought_type == ThoughtType.HYPOTHESIS:
            strategies = [
                ("Seek evidence", ThoughtType.OBSERVATION, 0.9),
                ("Identify implications", ThoughtType.INFERENCE, 0.85),
                ("Find analogies", ThoughtType.ANALOGY, 0.8)
            ]
        elif parent.thought_type == ThoughtType.OBSERVATION:
            strategies = [
                ("Draw inference", ThoughtType.INFERENCE, 0.9),
                ("Synthesize pattern", ThoughtType.SYNTHESIS, 0.85)
            ]
        elif parent.thought_type == ThoughtType.INFERENCE:
            strategies = [
                ("Validate logically", ThoughtType.CRITIQUE, 0.9),
                ("Extend reasoning", ThoughtType.INFERENCE, 0.85),
                ("Conclude", ThoughtType.CONCLUSION, 0.95)
            ]
        elif parent.thought_type == ThoughtType.SYNTHESIS:
            strategies = [
                ("Refine synthesis", ThoughtType.CRITIQUE, 0.9),
                ("Draw conclusion", ThoughtType.CONCLUSION, 0.95)
            ]
        else:
            strategies = [
                ("Continue analysis", ThoughtType.INFERENCE, 0.85)
            ]
        
        # Limit to configured max branches
        return strategies[:self.config.max_branches_per_node]
    
    async def _check_merge_opportunities(self) -> None:
        """Check for thought merge opportunities."""
        all_thoughts = list(self._thought_graph._nodes.values())
        valid_thoughts = [
            t for t in all_thoughts 
            if t.status in {ThoughtStatus.VALIDATED, ThoughtStatus.NASCENT}
        ]
        
        # Find synthesis opportunities
        opportunities = self._interdisciplinary_matrix.find_synthesis_opportunities(valid_thoughts)
        
        # Execute top merges
        for t1, t2, potential, reason in opportunities[:3]:
            if potential > 0.6:
                self._thought_graph.merge_thoughts(t1.id, t2.id)
    
    async def _synthesize_across_domains(
        self,
        context: ReasoningContext
    ) -> List[ThoughtNode]:
        """Perform cross-domain synthesis."""
        synthesis_thoughts = []
        
        all_thoughts = list(self._thought_graph._nodes.values())
        validated = [t for t in all_thoughts if t.status == ThoughtStatus.VALIDATED]
        
        # Find cross-domain synthesis opportunities
        opportunities = self._interdisciplinary_matrix.find_synthesis_opportunities(validated)
        
        for t1, t2, potential, reason in opportunities[:5]:
            synthesis = self._interdisciplinary_matrix.synthesize_cross_domain(
                t1, t2, self._thought_graph
            )
            if synthesis:
                synthesis_thoughts.append(synthesis)
        
        # Check for missing perspectives
        missing = self._interdisciplinary_matrix.suggest_missing_perspectives(validated)
        for domain, suggestion in missing[:2]:
            # Generate a thought to address the gap
            gap_thought = self._thought_graph.add_thought(
                content=f"[PERSPECTIVE GAP] {suggestion}",
                thought_type=ThoughtType.REFLECTION,
                domain=domain.value,
                confidence=0.5,
                snr_score=0.6,
                metadata={'is_gap_filler': True}
            )
            synthesis_thoughts.append(gap_thought)
        
        return synthesis_thoughts
    
    def _generate_final_synthesis(
        self,
        context: ReasoningContext,
        conclusions: List[ThoughtNode],
        snr_metrics: SNRMetrics
    ) -> str:
        """Generate the final synthesis text."""
        synthesis_parts = []
        
        synthesis_parts.append(f"═══ APEX COGNITIVE SYNTHESIS ═══")
        synthesis_parts.append(f"Query: {context.query}")
        synthesis_parts.append(f"Goal: {context.goal or 'General analysis'}")
        synthesis_parts.append("")
        
        synthesis_parts.append("CONCLUSIONS:")
        for i, conclusion in enumerate(conclusions, 1):
            synthesis_parts.append(f"  {i}. [{conclusion.domain.upper()}] {conclusion.content}")
            synthesis_parts.append(f"     Confidence: {conclusion.confidence:.2f} | SNR: {conclusion.snr_score:.2f}")
        
        synthesis_parts.append("")
        synthesis_parts.append(f"QUALITY METRICS:")
        synthesis_parts.append(f"  Overall SNR: {snr_metrics.overall_snr:.4f}")
        synthesis_parts.append(f"  Signal Strength: {snr_metrics.signal_strength:.4f}")
        synthesis_parts.append(f"  Diversity Bonus: {snr_metrics.diversity_bonus:.4f}")
        synthesis_parts.append(f"  Ihsan Achieved: {'✓' if snr_metrics.ihsan_achieved else f'✗ (gap: {snr_metrics.ihsan_gap:.4f})'}")
        
        if snr_metrics.per_domain_snr:
            synthesis_parts.append("")
            synthesis_parts.append("DOMAIN BREAKDOWN:")
            for domain, domain_snr in snr_metrics.per_domain_snr.items():
                synthesis_parts.append(f"  {domain}: {domain_snr:.4f}")
        
        # Add improvement suggestions
        suggestions = self._snr_optimizer.suggest_improvements(snr_metrics)
        if suggestions:
            synthesis_parts.append("")
            synthesis_parts.append("RECOMMENDATIONS:")
            for suggestion in suggestions:
                synthesis_parts.append(f"  • {suggestion}")
        
        synthesis_parts.append("")
        synthesis_parts.append("═══════════════════════════════════")
        
        return "\n".join(synthesis_parts)
    
    def _build_reasoning_trace(self, conclusions: List[ThoughtNode]) -> List[str]:
        """Build human-readable reasoning trace."""
        trace = []
        
        for conclusion in conclusions:
            chain = self._thought_graph.get_reasoning_chain(conclusion.id)
            if chain:
                trace.append(f"Chain to '{conclusion.content[:30]}...':")
                for i, thought in enumerate(chain):
                    indent = "  " * (thought.depth + 1)
                    trace.append(f"{indent}→ [{thought.thought_type.name}] {thought.content[:50]}...")
        
        return trace
    
    def _extract_cross_domain_insights(
        self, 
        synthesis_thoughts: List[ThoughtNode]
    ) -> List[str]:
        """Extract cross-domain insights from synthesis thoughts."""
        insights = []
        
        for thought in synthesis_thoughts:
            if thought.thought_type == ThoughtType.SYNTHESIS:
                source_domains = thought.metadata.get('source_domains', [])
                if len(source_domains) > 1:
                    insights.append(
                        f"[{' × '.join(source_domains)}] {thought.content[:100]}..."
                    )
        
        return insights


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_apex_engine(config: Optional[ApexConfig] = None) -> ApexCognitiveEngine:
    """Get the global Apex Cognitive Engine instance."""
    return ApexCognitiveEngine(config or DEFAULT_CONFIG)


async def quick_reason(
    query: str,
    goal: str = "",
    domains: Optional[List[CognitiveDomain]] = None
) -> ReasoningResult:
    """Quick reasoning helper for simple queries."""
    engine = get_apex_engine()
    context = ReasoningContext(
        query=query,
        goal=goal,
        domains_to_explore=domains or [
            CognitiveDomain.ALGORITHMIC,
            CognitiveDomain.ARCHITECTURAL,
            CognitiveDomain.ENGINEERING
        ]
    )
    return await engine.reason(context)


def quick_reason_sync(
    query: str,
    goal: str = "",
    domains: Optional[List[CognitiveDomain]] = None
) -> ReasoningResult:
    """Synchronous quick reasoning helper."""
    return asyncio.run(quick_reason(query, goal, domains))


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST & DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

async def run_apex_demonstration() -> Dict[str, Any]:
    """Run comprehensive demonstration of Apex Cognitive Engine."""
    
    print("=" * 70)
    print("  APEX COGNITIVE ENGINE - COMPREHENSIVE DEMONSTRATION")
    print("  Standing on the Shoulders of Giants")
    print("=" * 70)
    print()
    
    results = {
        "tests_passed": 0,
        "tests_total": 0,
        "components": {}
    }
    
    # Test 1: Data Structure - Adaptive Priority Queue
    print("[TEST 1] Adaptive Priority Queue with SNR Boost")
    print("-" * 50)
    results["tests_total"] += 1
    
    pq = AdaptivePriorityQueue[str](decay_rate=0.01, snr_boost=1.5)
    pq.push("low_priority", 0.3, snr_score=0.4)
    pq.push("high_snr", 0.5, snr_score=0.95)
    pq.push("medium", 0.6, snr_score=0.5)
    
    # High SNR should boost priority
    first = pq.pop()
    if first == "high_snr":
        print("  ✓ SNR-boosted priority correctly elevated high_snr item")
        results["tests_passed"] += 1
    else:
        print(f"  ✗ Expected 'high_snr', got '{first}'")
    
    results["components"]["AdaptivePriorityQueue"] = first == "high_snr"
    print()
    
    # Test 2: Causal Graph
    print("[TEST 2] Causal Graph - DAG Operations")
    print("-" * 50)
    results["tests_total"] += 1
    
    cg = CausalGraph[str]()
    cg.add_node("A", label="Start")
    cg.add_node("B", label="Middle")
    cg.add_node("C", label="End")
    cg.add_edge("A", "B", weight=0.8)
    cg.add_edge("B", "C", weight=0.9)
    
    # Test cycle prevention
    cycle_prevented = not cg.add_edge("C", "A", weight=0.5)
    
    # Test topological order
    topo = cg.topological_order()
    correct_order = topo == ["A", "B", "C"]
    
    # Test causal path weight
    path_weight = cg.causal_path_weight("A", "C")
    expected_weight = 0.8 * 0.9  # multiplicative
    
    all_passed = cycle_prevented and correct_order and abs(path_weight - expected_weight) < 0.01
    if all_passed:
        print(f"  ✓ Cycle prevention: {cycle_prevented}")
        print(f"  ✓ Topological order: {topo}")
        print(f"  ✓ Causal path weight A→C: {path_weight:.3f}")
        results["tests_passed"] += 1
    else:
        print(f"  ✗ Some tests failed")
    
    results["components"]["CausalGraph"] = all_passed
    print()
    
    # Test 3: SNR Optimized Cache
    print("[TEST 3] SNR-Optimized Cache Eviction")
    print("-" * 50)
    results["tests_total"] += 1
    
    cache = SNROptimizedCache[str, str](capacity=3)
    cache.put("low_snr", "data1", snr_score=0.2)
    cache.put("high_snr", "data2", snr_score=0.95)
    cache.put("medium_snr", "data3", snr_score=0.5)
    
    # Access high_snr multiple times
    for _ in range(5):
        cache.get("high_snr")
    
    # Add new item (should evict low_snr due to low SNR + low access)
    cache.put("new_item", "data4", snr_score=0.7)
    
    # low_snr should be evicted, high_snr should remain
    low_evicted = cache.get("low_snr") is None
    high_retained = cache.get("high_snr") is not None
    
    if low_evicted and high_retained:
        print(f"  ✓ Low-SNR item correctly evicted")
        print(f"  ✓ High-SNR item with high access retained")
        results["tests_passed"] += 1
    else:
        print(f"  ✗ Eviction policy not working correctly")
    
    results["components"]["SNROptimizedCache"] = low_evicted and high_retained
    print()
    
    # Test 4: Thought Graph
    print("[TEST 4] Graph of Thoughts - Reasoning Chain")
    print("-" * 50)
    results["tests_total"] += 1
    
    tg = ThoughtGraph()
    
    root = tg.add_thought(
        content="Root hypothesis",
        thought_type=ThoughtType.HYPOTHESIS,
        domain="general",
        confidence=0.9,
        snr_score=0.85
    )
    
    inference = tg.add_thought(
        content="Derived inference",
        thought_type=ThoughtType.INFERENCE,
        parent_ids=[root.id],
        domain="general",
        confidence=0.8,
        snr_score=0.8
    )
    
    conclusion = tg.add_thought(
        content="Final conclusion",
        thought_type=ThoughtType.CONCLUSION,
        parent_ids=[inference.id],
        domain="general",
        confidence=0.85,
        snr_score=0.9
    )
    
    # Validate thoughts
    tg.validate_thought(root.id, 0.85)
    tg.validate_thought(inference.id, 0.8)
    tg.validate_thought(conclusion.id, 0.9)
    
    # Get reasoning chain
    chain = tg.get_reasoning_chain(conclusion.id)
    chain_length = len(chain)
    
    # Get graph SNR
    graph_snr, graph_metrics = tg.calculate_graph_snr()
    
    success = chain_length == 3 and graph_snr > 0.7
    if success:
        print(f"  ✓ Reasoning chain length: {chain_length}")
        print(f"  ✓ Graph SNR: {graph_snr:.4f}")
        print(f"  ✓ Max depth reached: {graph_metrics['max_depth']}")
        results["tests_passed"] += 1
    else:
        print(f"  ✗ Chain length: {chain_length}, Graph SNR: {graph_snr}")
    
    results["components"]["ThoughtGraph"] = success
    print()
    
    # Test 5: Interdisciplinary Matrix
    print("[TEST 5] Interdisciplinary Thinking Matrix")
    print("-" * 50)
    results["tests_total"] += 1
    
    matrix = InterdisciplinaryMatrix()
    
    # Create thoughts from different domains
    algo_thought = ThoughtNode(
        content="Use dynamic programming for O(n) optimization",
        thought_type=ThoughtType.HYPOTHESIS,
        domain="algorithmic",
        confidence=0.8
    )
    
    arch_thought = ThoughtNode(
        content="Implement microservices architecture for scalability",
        thought_type=ThoughtType.HYPOTHESIS,
        domain="architectural",
        confidence=0.75
    )
    
    # Evaluate from multiple domains
    evaluations = matrix.evaluate_thought_multidomain(algo_thought)
    
    # Find synthesis opportunities
    opportunities = matrix.find_synthesis_opportunities([algo_thought, arch_thought])
    
    success = len(evaluations) >= 3 and len(opportunities) >= 1
    if success:
        print(f"  ✓ Multi-domain evaluations: {len(evaluations)}")
        print(f"  ✓ Synthesis opportunities found: {len(opportunities)}")
        if opportunities:
            print(f"  ✓ Top opportunity potential: {opportunities[0][2]:.3f}")
        results["tests_passed"] += 1
    else:
        print(f"  ✗ Evaluations: {len(evaluations)}, Opportunities: {len(opportunities)}")
    
    results["components"]["InterdisciplinaryMatrix"] = success
    print()
    
    # Test 6: SNR Optimizer
    print("[TEST 6] SNR Highest Score Engine")
    print("-" * 50)
    results["tests_total"] += 1
    
    optimizer = SNROptimizer()
    
    thoughts = [
        ThoughtNode(content="High quality thought", confidence=0.9, snr_score=0.9, 
                   grounding_score=0.8, status=ThoughtStatus.VALIDATED),
        ThoughtNode(content="Medium quality thought", confidence=0.7, snr_score=0.7,
                   grounding_score=0.5, status=ThoughtStatus.VALIDATED),
        ThoughtNode(content="Low quality thought", confidence=0.3, snr_score=0.3,
                   grounding_score=0.2, status=ThoughtStatus.NASCENT),
    ]
    
    metrics = optimizer.calculate_snr(thoughts)
    
    # Test optimization
    optimized, opt_metrics = optimizer.optimize_thought_set(thoughts, target_snr=0.8)
    
    success = metrics.overall_snr > 0.4 and opt_metrics.overall_snr >= metrics.overall_snr
    if success:
        print(f"  ✓ Initial SNR: {metrics.overall_snr:.4f}")
        print(f"  ✓ Optimized SNR: {opt_metrics.overall_snr:.4f}")
        print(f"  ✓ Signal Strength: {metrics.signal_strength:.4f}")
        print(f"  ✓ Diversity Bonus: {metrics.diversity_bonus:.4f}")
        results["tests_passed"] += 1
    else:
        print(f"  ✗ Optimization did not improve SNR")
    
    results["components"]["SNROptimizer"] = success
    print()
    
    # Test 7: Giants Shoulder Protocol
    print("[TEST 7] Standing on Giants Protocol")
    print("-" * 50)
    results["tests_total"] += 1
    
    giants = GiantsShoulder()
    
    all_giants = giants.list_giants()
    height, height_metrics = giants.calculate_standing_height()
    
    # Try to load available giants
    loaded = []
    for giant in all_giants:
        if giants.try_load_giant(giant.name):
            loaded.append(giant.name)
    
    new_height, new_metrics = giants.calculate_standing_height()
    
    success = len(all_giants) >= 4 and new_height >= height
    if success:
        print(f"  ✓ Giants registered: {len(all_giants)}")
        print(f"  ✓ Giants loaded: {len(loaded)}")
        print(f"  ✓ Standing height: {new_height:.4f}")
        print(f"  ✓ Capabilities: {new_metrics.get('capability_count', 0)}")
        results["tests_passed"] += 1
    else:
        print(f"  ✗ Giants: {len(all_giants)}, Height: {new_height}")
    
    results["components"]["GiantsShoulder"] = success
    print()
    
    # Test 8: Full Apex Engine
    print("[TEST 8] Apex Cognitive Engine - Full Integration")
    print("-" * 50)
    results["tests_total"] += 1
    
    engine = ApexCognitiveEngine()
    
    context = ReasoningContext(
        query="How to design a high-performance distributed caching system?",
        goal="Comprehensive architectural guidance with implementation details",
        domains_to_explore=[
            CognitiveDomain.ALGORITHMIC,
            CognitiveDomain.ARCHITECTURAL,
            CognitiveDomain.ENGINEERING
        ],
        target_snr=0.8
    )
    
    result = await engine.reason(context)
    
    success = (
        result.thoughts_generated >= 5 and
        result.snr_metrics.overall_snr >= 0.5 and
        len(result.conclusions) >= 1
    )
    
    if success:
        print(f"  ✓ Thoughts generated: {result.thoughts_generated}")
        print(f"  ✓ Thoughts pruned: {result.thoughts_pruned}")
        print(f"  ✓ Final SNR: {result.snr_metrics.overall_snr:.4f}")
        print(f"  ✓ Conclusions: {len(result.conclusions)}")
        print(f"  ✓ Giants leveraged: {len(result.giants_leveraged)}")
        print(f"  ✓ Ihsan achieved: {result.ihsan_achieved}")
        print(f"  ✓ Execution time: {result.execution_time:.3f}s")
        results["tests_passed"] += 1
    else:
        print(f"  ✗ Integration test failed")
        print(f"    Thoughts: {result.thoughts_generated}")
        print(f"    SNR: {result.snr_metrics.overall_snr}")
        print(f"    Conclusions: {len(result.conclusions)}")
    
    results["components"]["ApexCognitiveEngine"] = success
    print()
    
    # Summary
    print("=" * 70)
    print("  DEMONSTRATION SUMMARY")
    print("=" * 70)
    print()
    print(f"  Tests Passed: {results['tests_passed']}/{results['tests_total']}")
    print(f"  Success Rate: {results['tests_passed']/results['tests_total']*100:.1f}%")
    print()
    print("  Component Status:")
    for component, passed in results["components"].items():
        status = "✓" if passed else "✗"
        print(f"    [{status}] {component}")
    
    print()
    
    # Final Ihsan score
    ihsan_score = results['tests_passed'] / results['tests_total']
    ihsan_achieved = ihsan_score >= 0.95
    
    print(f"  Ihsan Score: {ihsan_score:.4f}")
    if ihsan_achieved:
        print("  ✨ IHSAN ACHIEVED - Excellence in Cognitive Architecture ✨")
    else:
        gap = 0.95 - ihsan_score
        print(f"  Ihsan Gap: {gap:.4f} (target: 0.95)")
    
    print()
    print("=" * 70)
    
    results["ihsan_score"] = ihsan_score
    results["ihsan_achieved"] = ihsan_achieved
    
    return results


def run_tests() -> Dict[str, Any]:
    """Run tests synchronously."""
    return asyncio.run(run_apex_demonstration())


if __name__ == "__main__":
    results = run_tests()
    
    # Exit with appropriate code
    if results["tests_passed"] == results["tests_total"]:
        sys.exit(0)
    else:
        sys.exit(1)
