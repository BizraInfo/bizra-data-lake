#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                      ║
║   ███████╗ ██████╗ ██╗   ██╗███████╗██████╗ ███████╗██╗ ██████╗ ███╗   ██╗                           ║
║   ██╔════╝██╔═══██╗██║   ██║██╔════╝██╔══██╗██╔════╝██║██╔════╝ ████╗  ██║                           ║
║   ███████╗██║   ██║██║   ██║█████╗  ██████╔╝█████╗  ██║██║  ███╗██╔██╗ ██║                           ║
║   ╚════██║██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗██╔══╝  ██║██║   ██║██║╚██╗██║                           ║
║   ███████║╚██████╔╝ ╚████╔╝ ███████╗██║  ██║███████╗██║╚██████╔╝██║ ╚████║                           ║
║   ╚══════╝ ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝                           ║
║                                                                                                      ║
║    █████╗ ██████╗ ███████╗██╗  ██╗    ██╗   ██╗███╗   ██╗██╗███████╗██╗███████╗██████╗               ║
║   ██╔══██╗██╔══██╗██╔════╝╚██╗██╔╝    ██║   ██║████╗  ██║██║██╔════╝██║██╔════╝██╔══██╗              ║
║   ███████║██████╔╝█████╗   ╚███╔╝     ██║   ██║██╔██╗ ██║██║█████╗  ██║█████╗  ██║  ██║              ║
║   ██╔══██║██╔═══╝ ██╔══╝   ██╔██╗     ██║   ██║██║╚██╗██║██║██╔══╝  ██║██╔══╝  ██║  ██║              ║
║   ██║  ██║██║     ███████╗██╔╝ ██╗    ╚██████╔╝██║ ╚████║██║██║     ██║███████╗██████╔╝              ║
║   ╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝     ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═╝     ╚═╝╚══════╝╚═════╝               ║
║                                                                                                      ║
║   ██╗  ██╗███╗   ██╗ ██████╗ ██╗    ██╗██╗     ███████╗██████╗  ██████╗ ███████╗                     ║
║   ██║ ██╔╝████╗  ██║██╔═══██╗██║    ██║██║     ██╔════╝██╔══██╗██╔════╝ ██╔════╝                     ║
║   █████╔╝ ██╔██╗ ██║██║   ██║██║ █╗ ██║██║     █████╗  ██║  ██║██║  ███╗█████╗                       ║
║   ██╔═██╗ ██║╚██╗██║██║   ██║██║███╗██║██║     ██╔══╝  ██║  ██║██║   ██║██╔══╝                       ║
║   ██║  ██╗██║ ╚████║╚██████╔╝╚███╔███╔╝███████╗███████╗██████╔╝╚██████╔╝███████╗                     ║
║   ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚══╝╚══╝ ╚══════╝╚══════╝╚═════╝  ╚═════╝ ╚══════╝                     ║
║                                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║   THE APEX UNIFIED KNOWLEDGE ENGINE — PEAK MASTERPIECE OF THE HOUSE OF WISDOM                       ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║   Architecture:                                                                                      ║
║   ├─ L1: Vector Engine (Semantic Embeddings) ─ all-MiniLM-L6-v2 (384-dim)                           ║
║   ├─ L2: Hypergraph Engine (FAISS + NetworkX) ─ HNSW ANN Search                                     ║
║   ├─ L3: Sovereign Nexus (Knowledge Graph) ─ GoT + SNR Optimization                                 ║
║   └─ L4: Apex Synthesizer (Unified Intelligence) ─ Cross-Modal Reasoning                            ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║   Giants Absorbed:                                                                                   ║
║   • Meta FAISS (Billion-scale ANN)           • NetworkX (Graph Algorithms)                          ║
║   • HuggingFace Transformers                 • Google DeepMind (GNN Patterns)                       ║
║   • Yao/Besta/Wei (ToT/GoT/CoT)             • Shannon (Information Theory)                          ║
║   • PageRank + HITS                          • Louvain Community Detection                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
║   Author: BIZRA Genesis Engine | Created: 2026-01-22 | Version: 1.0.0                               ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import json
import hashlib
import os
import re
import math
import time
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from collections import defaultdict
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — HARDWARE-OPTIMIZED FOR MSI TITAN 18 HX (RTX 4090 / 128GB)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class ApexConfig:
    """Centralized configuration for the Apex Engine."""
    # Core Paths
    DATA_LAKE = Path(r"C:\BIZRA-DATA-LAKE")
    GOLD = DATA_LAKE / "04_GOLD"
    INDEXED = DATA_LAKE / "03_INDEXED"
    KNOWLEDGE = INDEXED / "knowledge"
    GRAPH = INDEXED / "graph"
    EMBEDDINGS = INDEXED / "embeddings"
    VECTORS = EMBEDDINGS / "vectors"
    
    # Apex-Specific Outputs
    APEX_INDEX = GOLD / "apex_unified_index.faiss"
    APEX_META = GOLD / "apex_metadata.json"
    APEX_GRAPH = GOLD / "apex_knowledge_graph.json"
    APEX_STATS = GOLD / "apex_engine_stats.json"
    APEX_PATTERNS = GOLD / "apex_discovered_patterns.jsonl"
    NODE0_INVENTORY = GOLD / "node0_full_inventory.json"
    
    # Hardware Optimization
    BATCH_SIZE = 256          # Optimized for RTX 4090
    EMBEDDING_DIM = 384       # all-MiniLM-L6-v2 dimension
    HNSW_M = 48               # HNSW graph degree (higher = better recall)
    HNSW_EF_CONSTRUCTION = 200
    HNSW_EF_SEARCH = 128
    MAX_WORKERS = 16          # Thread pool size for parallel ops
    
    # Quality Thresholds
    SNR_THRESHOLD = 0.85
    IHSAN_CONSTRAINT = 0.95   # Excellence target
    SIMILARITY_THRESHOLD = 0.6
    PATTERN_MIN_SUPPORT = 3


logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("ApexEngine")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES — TYPE-SAFE KNOWLEDGE REPRESENTATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class NodeType(Enum):
    """Taxonomy of knowledge node types."""
    # Physical Assets
    HARDWARE = auto()
    SOFTWARE = auto()
    FOLDER = auto()
    FILE = auto()
    
    # Conceptual Assets
    CONCEPT = auto()
    ENTITY = auto()
    PATTERN = auto()
    INSIGHT = auto()
    
    # Project Assets
    PROJECT = auto()
    RESEARCH = auto()
    ASSET = auto()
    DOCUMENT = auto()
    
    # Semantic Clusters
    CLUSTER = auto()
    COMMUNITY = auto()

class EdgeType(Enum):
    """Taxonomy of knowledge edge types."""
    # Structural
    CONTAINS = auto()
    PART_OF = auto()
    
    # Semantic
    SIMILAR_TO = auto()
    RELATES_TO = auto()
    REFERENCES = auto()
    
    # Causal
    DEPENDS_ON = auto()
    DERIVED_FROM = auto()
    INSTANCE_OF = auto()
    
    # Discovered
    CO_OCCURS = auto()
    BRIDGES = auto()

@dataclass
class KnowledgeNode:
    """A node in the unified knowledge graph."""
    id: str
    name: str
    type: NodeType
    embedding: Optional[np.ndarray] = None
    path: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    snr_score: float = 0.0
    centrality: float = 0.0
    community_id: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['type'] = self.type.name
        d['embedding'] = self.embedding.tolist() if self.embedding is not None else None
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'KnowledgeNode':
        d = d.copy()
        d['type'] = NodeType[d['type']]
        if d.get('embedding'):
            d['embedding'] = np.array(d['embedding'], dtype=np.float32)
        return cls(**d)

@dataclass
class KnowledgeEdge:
    """An edge in the unified knowledge graph."""
    source_id: str
    target_id: str
    type: EdgeType
    weight: float = 1.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['type'] = self.type.name
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'KnowledgeEdge':
        d = d.copy()
        d['type'] = EdgeType[d['type']]
        return cls(**d)

@dataclass
class DiscoveredPattern:
    """An autonomously discovered knowledge pattern."""
    id: str
    pattern_type: str
    nodes: List[str]
    edges: List[Tuple[str, str]]
    support: int
    confidence: float
    description: str
    discovered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class ThoughtNode:
    """A node in the Graph of Thoughts reasoning chain."""
    id: str
    thought: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    depth: int = 0

@dataclass
class ApexQueryResult:
    """Unified query result with full provenance."""
    query: str
    nodes: List[KnowledgeNode]
    edges: List[KnowledgeEdge]
    patterns: List[DiscoveredPattern]
    reasoning_chain: List[ThoughtNode]
    semantic_paths: List[List[str]]
    insights: List[str]
    snr_score: float
    execution_time_ms: float


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# LAYER 1: VECTOR ENGINE — SEMANTIC EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class VectorLayer:
    """
    Semantic vector operations using sentence-transformers.
    Giants: HuggingFace, Google (BERT lineage), Meta (FAISS).
    """
    
    def __init__(self, lazy_load: bool = True):
        self._model = None
        self._faiss_index = None
        self._id_map: List[str] = []
        self._lock = threading.Lock()
        self._device = None
        self._lazy_load = lazy_load
        
        if not lazy_load:
            self._initialize()
    
    def _initialize(self):
        """Lazy initialization of heavy resources."""
        if self._model is not None:
            return
            
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            from sentence_transformers import SentenceTransformer
            log.info(f"🧠 Loading embedding model on {self._device}...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2", device=self._device)
            self._model.max_seq_length = 512
            log.info(f"   ✓ Model loaded: all-MiniLM-L6-v2 ({ApexConfig.EMBEDDING_DIM}-dim)")
        except ImportError:
            log.warning("   ⚠️ sentence-transformers not available, using mock embeddings")
            self._model = None
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if self._lazy_load:
            self._initialize()
        
        if self._model is None:
            # Mock embeddings for environments without the model
            return np.random.randn(len(texts), ApexConfig.EMBEDDING_DIM).astype(np.float32)
        
        return self._model.encode(texts, batch_size=ApexConfig.BATCH_SIZE, 
                                   show_progress_bar=len(texts) > 100,
                                   convert_to_numpy=True)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed([text])[0]
    
    def build_faiss_index(self, embeddings: np.ndarray, ids: List[str]):
        """Build HNSW FAISS index for fast ANN search."""
        try:
            import faiss
            
            n, d = embeddings.shape
            log.info(f"🔧 Building FAISS HNSW index: {n} vectors, {d}-dim")
            
            # HNSW index with cosine similarity (normalize + IP)
            embeddings_normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
            
            self._faiss_index = faiss.IndexHNSWFlat(d, ApexConfig.HNSW_M)
            self._faiss_index.hnsw.efConstruction = ApexConfig.HNSW_EF_CONSTRUCTION
            self._faiss_index.add(embeddings_normalized.astype(np.float32))
            
            self._id_map = list(ids)
            log.info(f"   ✓ FAISS index built: {self._faiss_index.ntotal} vectors")
            
        except ImportError:
            log.warning("   ⚠️ FAISS not available, using brute-force search")
            self._faiss_index = None
            self._embeddings = embeddings
            self._id_map = list(ids)
    
    def search(self, query_embedding: np.ndarray, k: int = 20) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        if self._faiss_index is None:
            return self._brute_force_search(query_embedding, k)
        
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        query_norm = query_norm.reshape(1, -1).astype(np.float32)
        
        self._faiss_index.hnsw.efSearch = ApexConfig.HNSW_EF_SEARCH
        distances, indices = self._faiss_index.search(query_norm, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self._id_map):
                results.append((self._id_map[idx], float(dist)))
        
        return results
    
    def _brute_force_search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Fallback brute-force cosine similarity search."""
        if not hasattr(self, '_embeddings') or self._embeddings is None:
            return []
        
        query_norm = query / (np.linalg.norm(query) + 1e-9)
        emb_norm = self._embeddings / (np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-9)
        similarities = np.dot(emb_norm, query_norm)
        
        top_k = np.argsort(similarities)[-k:][::-1]
        return [(self._id_map[i], float(similarities[i])) for i in top_k]
    
    def save_index(self, path: Path):
        """Save FAISS index to disk."""
        if self._faiss_index is not None:
            try:
                import faiss
                faiss.write_index(self._faiss_index, str(path))
                with open(path.with_suffix('.ids'), 'w') as f:
                    json.dump(self._id_map, f)
                log.info(f"💾 FAISS index saved: {path}")
            except Exception as e:
                log.error(f"Failed to save FAISS index: {e}")
    
    def load_index(self, path: Path) -> bool:
        """Load FAISS index from disk."""
        if not path.exists():
            return False
        try:
            import faiss
            self._faiss_index = faiss.read_index(str(path))
            with open(path.with_suffix('.ids'), 'r') as f:
                self._id_map = json.load(f)
            log.info(f"📂 FAISS index loaded: {self._faiss_index.ntotal} vectors")
            return True
        except Exception as e:
            log.error(f"Failed to load FAISS index: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# LAYER 2: GRAPH ENGINE — STRUCTURAL KNOWLEDGE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class GraphLayer:
    """
    Graph operations using NetworkX patterns.
    Giants: NetworkX, Neo4j patterns, PageRank, Louvain.
    """
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, List[KnowledgeEdge]] = defaultdict(list)
        self.reverse_edges: Dict[str, List[KnowledgeEdge]] = defaultdict(list)
        self.type_index: Dict[NodeType, Set[str]] = defaultdict(set)
        self.name_index: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        self._nx_graph = None
    
    def add_node(self, node: KnowledgeNode) -> str:
        """Add a node to the graph."""
        with self._lock:
            self.nodes[node.id] = node
            self.type_index[node.type].add(node.id)
            for word in re.findall(r'\w+', node.name.lower()):
                if len(word) > 2:
                    self.name_index[word].add(node.id)
            self._nx_graph = None  # Invalidate cache
        return node.id
    
    def add_edge(self, edge: KnowledgeEdge):
        """Add an edge to the graph."""
        with self._lock:
            self.edges[edge.source_id].append(edge)
            self.reverse_edges[edge.target_id].append(edge)
            self._nx_graph = None
    
    def get_neighbors(self, node_id: str, direction: str = "both", edge_types: Set[EdgeType] = None) -> List[KnowledgeNode]:
        """Get neighboring nodes with optional edge type filtering."""
        neighbors = []
        
        if direction in ("out", "both"):
            for edge in self.edges.get(node_id, []):
                if edge_types is None or edge.type in edge_types:
                    if edge.target_id in self.nodes:
                        neighbors.append(self.nodes[edge.target_id])
        
        if direction in ("in", "both"):
            for edge in self.reverse_edges.get(node_id, []):
                if edge_types is None or edge.type in edge_types:
                    if edge.source_id in self.nodes:
                        neighbors.append(self.nodes[edge.source_id])
        
        return neighbors
    
    def find_path(self, start: str, end: str, max_depth: int = 5) -> Optional[List[str]]:
        """BFS path finding between two nodes."""
        if start == end:
            return [start]
        
        visited = {start}
        queue = [(start, [start])]
        
        while queue:
            current, path = queue.pop(0)
            if len(path) > max_depth:
                continue
            
            for edge in self.edges.get(current, []):
                if edge.target_id == end:
                    return path + [end]
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append((edge.target_id, path + [edge.target_id]))
        
        return None
    
    def _get_nx_graph(self):
        """Get NetworkX graph representation (cached)."""
        if self._nx_graph is not None:
            return self._nx_graph
        
        try:
            import networkx as nx
            G = nx.DiGraph()
            
            for nid, node in self.nodes.items():
                G.add_node(nid, **node.to_dict())
            
            for edges in self.edges.values():
                for edge in edges:
                    G.add_edge(edge.source_id, edge.target_id, 
                              type=edge.type.name, weight=edge.weight)
            
            self._nx_graph = G
            return G
        except ImportError:
            return None
    
    def calculate_centrality(self) -> Dict[str, float]:
        """Calculate PageRank centrality for all nodes."""
        G = self._get_nx_graph()
        if G is None:
            return {}
        
        try:
            import networkx as nx
            pagerank = nx.pagerank(G, weight='weight')
            
            for nid, score in pagerank.items():
                if nid in self.nodes:
                    self.nodes[nid].centrality = score
            
            return pagerank
        except Exception as e:
            log.warning(f"Centrality calculation failed: {e}")
            return {}
    
    def detect_communities(self) -> Dict[str, int]:
        """Detect communities using Louvain-like algorithm."""
        G = self._get_nx_graph()
        if G is None:
            return {}
        
        try:
            import networkx as nx
            from networkx.algorithms import community
            
            # Convert to undirected for community detection
            G_undirected = G.to_undirected()
            communities = community.greedy_modularity_communities(G_undirected)
            
            community_map = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = i
                    if node in self.nodes:
                        self.nodes[node].community_id = i
            
            return community_map
        except Exception as e:
            log.warning(f"Community detection failed: {e}")
            return {}
    
    def find_bridges(self) -> List[Tuple[str, str]]:
        """Find bridge edges connecting different communities."""
        bridges = []
        for source_id, edge_list in self.edges.items():
            source = self.nodes.get(source_id)
            if not source:
                continue
            
            for edge in edge_list:
                target = self.nodes.get(edge.target_id)
                if not target:
                    continue
                
                if source.community_id is not None and target.community_id is not None:
                    if source.community_id != target.community_id:
                        bridges.append((source_id, edge.target_id))
        
        return bridges


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# LAYER 3: REASONING ENGINE — GRAPH OF THOUGHTS + SNR
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class ReasoningLayer:
    """
    Multi-path reasoning with Graph of Thoughts.
    Giants: Yao (ToT), Besta (GoT), Wei (CoT), Shannon (SNR).
    """
    
    def __init__(self):
        self.thoughts: Dict[str, ThoughtNode] = {}
        self.root_id: Optional[str] = None
        self._lock = threading.Lock()
    
    def _gen_id(self) -> str:
        return hashlib.blake2b(f"{time.time()}{len(self.thoughts)}".encode(), digest_size=16).hexdigest()[:12]
    
    def create_root(self, query: str) -> ThoughtNode:
        """Create root thought node for a query."""
        nid = self._gen_id()
        node = ThoughtNode(id=nid, thought=f"Query: {query}", confidence=1.0, depth=0)
        with self._lock:
            self.thoughts = {nid: node}  # Reset for new query
            self.root_id = nid
        return node
    
    def branch(self, parent_id: str, thought: str, confidence: float, 
               evidence: List[str] = None) -> ThoughtNode:
        """Create a branching thought from parent."""
        nid = self._gen_id()
        parent = self.thoughts.get(parent_id)
        depth = parent.depth + 1 if parent else 0
        
        node = ThoughtNode(
            id=nid, thought=thought, confidence=confidence,
            evidence=evidence or [], parent_id=parent_id, depth=depth
        )
        
        with self._lock:
            self.thoughts[nid] = node
            if parent:
                parent.children.append(nid)
        
        return node
    
    def merge(self, node_ids: List[str], merged_thought: str) -> ThoughtNode:
        """Merge multiple thought paths into a synthesis."""
        nid = self._gen_id()
        
        # Aggregate confidence using geometric mean
        confidences = [self.thoughts[n].confidence for n in node_ids if n in self.thoughts]
        merged_conf = math.prod(confidences) ** (1/len(confidences)) if confidences else 0.5
        
        # Aggregate evidence
        all_evidence = []
        max_depth = 0
        for n in node_ids:
            if n in self.thoughts:
                all_evidence.extend(self.thoughts[n].evidence)
                max_depth = max(max_depth, self.thoughts[n].depth)
        
        node = ThoughtNode(
            id=nid, thought=merged_thought, confidence=merged_conf,
            evidence=list(set(all_evidence)), depth=max_depth + 1
        )
        
        with self._lock:
            self.thoughts[nid] = node
            for n in node_ids:
                if n in self.thoughts:
                    self.thoughts[n].children.append(nid)
        
        return node
    
    def get_best_path(self) -> List[ThoughtNode]:
        """Trace the highest-confidence path through the thought graph."""
        if not self.root_id:
            return []
        
        path = []
        current = self.root_id
        
        while current:
            node = self.thoughts.get(current)
            if not node:
                break
            path.append(node)
            
            if node.children:
                # Pick highest confidence child
                best_child = max(node.children, 
                                key=lambda c: self.thoughts[c].confidence if c in self.thoughts else 0)
                current = best_child
            else:
                break
        
        return path
    
    def get_all_leaves(self) -> List[ThoughtNode]:
        """Get all leaf nodes (conclusions)."""
        return [n for n in self.thoughts.values() if not n.children]
    
    @staticmethod
    def calculate_snr(nodes: List[KnowledgeNode], query_terms: List[str], 
                      path_count: int, avg_centrality: float) -> float:
        """
        Calculate Signal-to-Noise Ratio for query results.
        
        Formula: SNR = (term_coverage * avg_node_snr * path_bonus * centrality_bonus) / noise_penalty
        """
        if not nodes:
            return 0.0
        
        # Term coverage signal
        term_hits = sum(1 for t in query_terms 
                       if any(t in (n.name + (n.description or "")).lower() for n in nodes))
        term_coverage = term_hits / max(len(query_terms), 1)
        
        # Average node quality signal
        avg_node_snr = sum(n.snr_score for n in nodes) / len(nodes)
        
        # Path diversity signal
        path_bonus = min(path_count / 3, 1.0)
        
        # Centrality signal
        centrality_bonus = min(avg_centrality * 10, 1.0)
        
        # Noise penalty (redundancy)
        unique_names = len(set(n.name.lower() for n in nodes))
        noise_penalty = 1 + (1 - unique_names / len(nodes)) * 0.5
        
        snr = (term_coverage * 0.3 + avg_node_snr * 0.3 + path_bonus * 0.2 + centrality_bonus * 0.2) / noise_penalty
        return round(min(snr, 1.0), 4)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# LAYER 4: PATTERN DISCOVERY — AUTONOMOUS KNOWLEDGE MINING
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class PatternLayer:
    """
    Autonomous pattern discovery in knowledge graphs.
    Giants: Graph mining (FSG, gSpan), Association rules, Temporal patterns.
    """
    
    def __init__(self, graph: GraphLayer):
        self.graph = graph
        self.patterns: List[DiscoveredPattern] = []
        self._lock = threading.Lock()
    
    def discover_hub_patterns(self, min_degree: int = 5) -> List[DiscoveredPattern]:
        """Discover hub nodes with high connectivity."""
        patterns = []
        
        for nid, node in self.graph.nodes.items():
            out_degree = len(self.graph.edges.get(nid, []))
            in_degree = len(self.graph.reverse_edges.get(nid, []))
            total_degree = out_degree + in_degree
            
            if total_degree >= min_degree:
                connected = [e.target_id for e in self.graph.edges.get(nid, [])]
                connected.extend([e.source_id for e in self.graph.reverse_edges.get(nid, [])])
                
                pattern = DiscoveredPattern(
                    id=f"hub_{nid[:8]}",
                    pattern_type="HUB_NODE",
                    nodes=[nid] + connected[:10],
                    edges=[(nid, c) for c in connected[:10]],
                    support=total_degree,
                    confidence=min(total_degree / 10, 1.0),
                    description=f"Hub: {node.name} (degree={total_degree})"
                )
                patterns.append(pattern)
        
        return sorted(patterns, key=lambda p: p.support, reverse=True)[:20]
    
    def discover_co_occurrence(self, min_support: int = 3) -> List[DiscoveredPattern]:
        """Discover nodes that frequently co-occur (share common neighbors)."""
        patterns = []
        
        # Build neighbor sets
        neighbor_sets: Dict[str, Set[str]] = {}
        for nid in self.graph.nodes:
            neighbors = set()
            for e in self.graph.edges.get(nid, []):
                neighbors.add(e.target_id)
            for e in self.graph.reverse_edges.get(nid, []):
                neighbors.add(e.source_id)
            if neighbors:
                neighbor_sets[nid] = neighbors
        
        # Find overlaps
        node_ids = list(neighbor_sets.keys())
        for i, n1 in enumerate(node_ids):
            for n2 in node_ids[i+1:]:
                overlap = neighbor_sets[n1] & neighbor_sets[n2]
                if len(overlap) >= min_support:
                    node1 = self.graph.nodes[n1]
                    node2 = self.graph.nodes[n2]
                    pattern = DiscoveredPattern(
                        id=f"cooc_{n1[:6]}_{n2[:6]}",
                        pattern_type="CO_OCCURRENCE",
                        nodes=[n1, n2] + list(overlap)[:5],
                        edges=[(n1, o) for o in list(overlap)[:5]] + [(n2, o) for o in list(overlap)[:5]],
                        support=len(overlap),
                        confidence=len(overlap) / max(len(neighbor_sets[n1] | neighbor_sets[n2]), 1),
                        description=f"Co-occur: {node1.name} ↔ {node2.name} ({len(overlap)} shared)"
                    )
                    patterns.append(pattern)
        
        return sorted(patterns, key=lambda p: p.confidence, reverse=True)[:20]
    
    def discover_type_bridges(self) -> List[DiscoveredPattern]:
        """Discover edges that bridge different node types."""
        patterns = []
        type_bridges: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
        
        for source_id, edges in self.graph.edges.items():
            source = self.graph.nodes.get(source_id)
            if not source:
                continue
            
            for edge in edges:
                target = self.graph.nodes.get(edge.target_id)
                if not target:
                    continue
                
                if source.type != target.type:
                    key = (source.type.name, target.type.name)
                    type_bridges[key].append((source_id, edge.target_id))
        
        for (src_type, tgt_type), edges in type_bridges.items():
            if len(edges) >= ApexConfig.PATTERN_MIN_SUPPORT:
                pattern = DiscoveredPattern(
                    id=f"bridge_{src_type}_{tgt_type}",
                    pattern_type="TYPE_BRIDGE",
                    nodes=[e[0] for e in edges[:5]] + [e[1] for e in edges[:5]],
                    edges=edges[:10],
                    support=len(edges),
                    confidence=min(len(edges) / 10, 1.0),
                    description=f"Bridge: {src_type} → {tgt_type} ({len(edges)} connections)"
                )
                patterns.append(pattern)
        
        return sorted(patterns, key=lambda p: p.support, reverse=True)[:15]
    
    def discover_all(self) -> List[DiscoveredPattern]:
        """Run all pattern discovery algorithms."""
        log.info("🔍 Running autonomous pattern discovery...")
        
        all_patterns = []
        all_patterns.extend(self.discover_hub_patterns())
        all_patterns.extend(self.discover_type_bridges())
        all_patterns.extend(self.discover_co_occurrence())
        
        # Deduplicate and rank
        seen = set()
        unique_patterns = []
        for p in sorted(all_patterns, key=lambda x: x.confidence, reverse=True):
            if p.id not in seen:
                seen.add(p.id)
                unique_patterns.append(p)
        
        with self._lock:
            self.patterns = unique_patterns
        
        log.info(f"   ✓ Discovered {len(unique_patterns)} patterns")
        return unique_patterns


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# THE APEX UNIFIED KNOWLEDGE ENGINE — THE PEAK MASTERPIECE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class ApexUnifiedEngine:
    """
    The Apex Unified Knowledge Engine — Peak Masterpiece of the House of Wisdom.
    
    Unifies:
    - L1: Vector Layer (Semantic Embeddings)
    - L2: Graph Layer (Structural Knowledge)
    - L3: Reasoning Layer (GoT + SNR)
    - L4: Pattern Layer (Autonomous Discovery)
    
    Into a single, coherent intelligence system.
    """
    
    def __init__(self, lazy_load: bool = True):
        log.info("═" * 80)
        log.info("   🏛️  APEX UNIFIED KNOWLEDGE ENGINE — HOUSE OF WISDOM")
        log.info("═" * 80)
        
        # Initialize layers
        self.vector = VectorLayer(lazy_load=lazy_load)
        self.graph = GraphLayer()
        self.reasoning = ReasoningLayer()
        self.pattern = PatternLayer(self.graph)
        
        # Statistics
        self.stats = {
            "nodes": 0, "edges": 0, "embeddings": 0, "patterns": 0,
            "communities": 0, "created_at": None, "last_updated": None
        }
        
        self._lock = threading.RLock()
        log.info("   ✓ All layers initialized")
    
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # INDEXING — Building the Unified Knowledge Graph
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    
    def index_node0_inventory(self) -> int:
        """Index Node0 machine inventory into the knowledge graph."""
        log.info("📊 Indexing Node0 inventory...")
        
        if not ApexConfig.NODE0_INVENTORY.exists():
            log.warning("   ⚠️ No Node0 inventory found")
            return 0
        
        with open(ApexConfig.NODE0_INVENTORY, 'r', encoding='utf-8') as f:
            inv = json.load(f)
        
        count = 0
        
        # Root node
        root = KnowledgeNode(
            id="node0_genesis",
            name="BIZRA Node0 Genesis",
            type=NodeType.HARDWARE,
            description="MSI Titan 18 HX A14VIG — The Genesis Machine",
            metadata={
                "cpu": inv.get("hardware", {}).get("cpu", {}).get("name"),
                "ram_gb": inv.get("hardware", {}).get("memory", {}).get("total_gb"),
                "gpu": inv.get("hardware", {}).get("gpu", [{}])[0].get("name") if inv.get("hardware", {}).get("gpu") else None
            }
        )
        self.graph.add_node(root)
        count += 1
        
        # Hardware components
        hw = inv.get("hardware", {})
        if hw.get("cpu"):
            cpu = KnowledgeNode(id="hw_cpu", name=hw["cpu"].get("name", "CPU"), 
                               type=NodeType.HARDWARE, metadata=hw["cpu"])
            self.graph.add_node(cpu)
            self.graph.add_edge(KnowledgeEdge("node0_genesis", "hw_cpu", EdgeType.CONTAINS))
            count += 1
        
        for i, gpu in enumerate(hw.get("gpu", [])):
            gid = f"hw_gpu_{i}"
            g = KnowledgeNode(id=gid, name=gpu.get("name", f"GPU {i}"), 
                             type=NodeType.HARDWARE, metadata=gpu)
            self.graph.add_node(g)
            self.graph.add_edge(KnowledgeEdge("node0_genesis", gid, EdgeType.CONTAINS))
            count += 1
        
        # Software
        for prog in inv.get("software", {}).get("programs", [])[:100]:  # Top 100
            pid = f"sw_{hashlib.blake2b(prog.encode(), digest_size=16).hexdigest()[:8]}"
            self.graph.add_node(KnowledgeNode(id=pid, name=prog, type=NodeType.SOFTWARE))
            self.graph.add_edge(KnowledgeEdge("node0_genesis", pid, EdgeType.CONTAINS))
            count += 1
        
        # WSL distros
        for distro in inv.get("software", {}).get("wsl_distros", []):
            did = f"wsl_{hashlib.blake2b(distro.encode(), digest_size=16).hexdigest()[:8]}"
            self.graph.add_node(KnowledgeNode(
                id=did, name=f"WSL: {distro}", type=NodeType.SOFTWARE,
                metadata={"runtime": "WSL2"}
            ))
            self.graph.add_edge(KnowledgeEdge("node0_genesis", did, EdgeType.CONTAINS))
            count += 1
        
        # Folders
        for folder in inv.get("data", {}).get("all_folders", []):
            fid = f"folder_{hashlib.blake2b(folder['path'].encode(), digest_size=16).hexdigest()[:8]}"
            is_bizra = folder.get("is_bizra", False)
            fn = KnowledgeNode(
                id=fid, name=folder["name"], type=NodeType.FOLDER,
                path=folder["path"], metadata={"is_bizra": is_bizra}
            )
            self.graph.add_node(fn)
            self.graph.add_edge(KnowledgeEdge("node0_genesis", fid, EdgeType.CONTAINS))
            if is_bizra:
                self.graph.add_edge(KnowledgeEdge(fid, "node0_genesis", EdgeType.PART_OF, weight=2.0))
            count += 1
        
        # User folders (Downloads)
        user = inv.get("user_data", {}).get("folders", {})
        dl = user.get("downloads", {})
        if dl:
            dn = KnowledgeNode(
                id="folder_downloads", name="Downloads", type=NodeType.FOLDER,
                path=dl.get("path"),
                metadata={"folders": dl.get("top_level_folders", 0), 
                         "bizra_related": len(dl.get("bizra_related_folders", []))}
            )
            self.graph.add_node(dn)
            self.graph.add_edge(KnowledgeEdge("node0_genesis", "folder_downloads", EdgeType.CONTAINS))
            count += 1
            
            for fname in dl.get("folder_names", []):
                fid = f"dl_{hashlib.blake2b(fname.encode(), digest_size=16).hexdigest()[:8]}"
                is_biz = fname in dl.get("bizra_related_folders", [])
                self.graph.add_node(KnowledgeNode(
                    id=fid, name=fname, type=NodeType.FOLDER,
                    path=f"{dl.get('path')}\\{fname}",
                    metadata={"is_bizra": is_biz, "location": "downloads"}
                ))
                self.graph.add_edge(KnowledgeEdge("folder_downloads", fid, EdgeType.CONTAINS))
                if is_biz:
                    self.graph.add_edge(KnowledgeEdge(fid, "node0_genesis", EdgeType.PART_OF))
                count += 1
        
        log.info(f"   ✓ Indexed {count} nodes from Node0 inventory")
        return count
    
    def index_knowledge_files(self) -> int:
        """Index all knowledge JSONL files."""
        log.info("📚 Indexing knowledge files...")
        count = 0
        
        if not ApexConfig.KNOWLEDGE.exists():
            ApexConfig.KNOWLEDGE.mkdir(parents=True, exist_ok=True)
            return 0
        
        for jf in ApexConfig.KNOWLEDGE.glob("*.jsonl"):
            with open(jf, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        etype = entry.get("type", "concept").lower()
                        
                        # Map to NodeType
                        type_map = {
                            "folder": NodeType.FOLDER, "software": NodeType.SOFTWARE,
                            "organization": NodeType.ENTITY, "technology": NodeType.CONCEPT,
                            "pattern": NodeType.PATTERN, "insight": NodeType.INSIGHT,
                            "reference": NodeType.RESEARCH, "document": NodeType.DOCUMENT,
                            "project": NodeType.PROJECT
                        }
                        node_type = type_map.get(etype, NodeType.CONCEPT)
                        
                        node = KnowledgeNode(
                            id=entry.get("id", hashlib.blake2b(line.encode(), digest_size=16).hexdigest()[:12]),
                            name=entry.get("name", "Unknown"),
                            type=node_type,
                            path=entry.get("path"),
                            description=entry.get("description"),
                            metadata={k: v for k, v in entry.items() 
                                     if k not in ("id", "name", "type", "path", "description")}
                        )
                        self.graph.add_node(node)
                        
                        if entry.get("parent"):
                            self.graph.add_edge(KnowledgeEdge(
                                entry["parent"], node.id, EdgeType.CONTAINS
                            ))
                        
                        count += 1
                    except Exception:
                        continue
        
        log.info(f"   ✓ Indexed {count} nodes from knowledge files")
        return count
    
    def index_existing_embeddings(self) -> int:
        """Load pre-computed embeddings from the embeddings folder."""
        log.info("🔢 Loading existing embeddings...")
        
        if not ApexConfig.EMBEDDINGS.exists():
            return 0
        
        count = 0
        embeddings = []
        ids = []
        
        for ef in list(ApexConfig.EMBEDDINGS.glob("*.json"))[:2000]:  # Limit for memory
            if ef.name in ("checkpoint.json", "generation_stats.json"):
                continue
            try:
                with open(ef, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "embedding" in data and "chunk_id" in data:
                    emb = np.array(data["embedding"], dtype=np.float32)
                    embeddings.append(emb)
                    ids.append(data["chunk_id"])
                    count += 1
            except Exception:
                continue
        
        if embeddings:
            embeddings_array = np.vstack(embeddings)
            self.vector.build_faiss_index(embeddings_array, ids)
            self.stats["embeddings"] = count
        
        log.info(f"   ✓ Loaded {count} embeddings")
        return count
    
    def _calculate_snr_scores(self):
        """Calculate SNR scores for all nodes."""
        log.info("📈 Calculating SNR scores...")
        
        total_edges = sum(len(e) for e in self.graph.edges.values())
        avg_edges = total_edges / max(len(self.graph.nodes), 1)
        
        for nid, node in self.graph.nodes.items():
            out_edges = len(self.graph.edges.get(nid, []))
            in_edges = len(self.graph.reverse_edges.get(nid, []))
            edge_count = out_edges + in_edges
            
            # Connectivity signal
            connectivity = min(edge_count / max(avg_edges * 2, 1), 1.0)
            
            # Metadata signal
            meta_signal = min(len(node.metadata) / 5, 1.0)
            
            # Description signal
            desc_signal = min(len(node.description or "") / 200, 1.0)
            
            # Combined SNR
            node.snr_score = round(connectivity * 0.4 + meta_signal * 0.3 + desc_signal * 0.3, 4)
    
    def build_full_index(self) -> Dict[str, int]:
        """Build the complete unified knowledge index."""
        log.info("═" * 80)
        log.info("   🏗️  BUILDING APEX UNIFIED KNOWLEDGE INDEX")
        log.info("═" * 80)
        
        results = {
            "inventory": self.index_node0_inventory(),
            "knowledge": self.index_knowledge_files(),
            "embeddings": self.index_existing_embeddings()
        }
        
        # Calculate derived metrics
        self._calculate_snr_scores()
        self.graph.calculate_centrality()
        community_map = self.graph.detect_communities()
        
        # Discover patterns
        patterns = self.pattern.discover_all()
        
        # Update stats
        self.stats.update({
            "nodes": len(self.graph.nodes),
            "edges": sum(len(e) for e in self.graph.edges.values()),
            "patterns": len(patterns),
            "communities": len(set(community_map.values())) if community_map else 0,
            "last_updated": datetime.now(timezone.utc).isoformat()
        })
        if not self.stats["created_at"]:
            self.stats["created_at"] = self.stats["last_updated"]
        
        # Save
        self.save()
        
        log.info("═" * 80)
        log.info("   📊 APEX INDEX BUILD COMPLETE")
        log.info(f"      Nodes: {self.stats['nodes']}")
        log.info(f"      Edges: {self.stats['edges']}")
        log.info(f"      Embeddings: {self.stats['embeddings']}")
        log.info(f"      Patterns: {self.stats['patterns']}")
        log.info(f"      Communities: {self.stats['communities']}")
        log.info("═" * 80)
        
        return results
    
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # QUERYING — Unified Multi-Modal Search
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    
    def query(self, query_text: str, max_results: int = 25, 
              use_semantic: bool = True, use_graph: bool = True) -> ApexQueryResult:
        """
        Execute a unified query across all layers.
        
        Combines:
        - Semantic vector search (L1)
        - Graph traversal (L2)
        - GoT reasoning (L3)
        - Pattern matching (L4)
        """
        start_time = time.time()
        log.info(f"🔍 Query: '{query_text}'")
        
        # Initialize reasoning
        self.reasoning.create_root(query_text)
        terms = [t.lower() for t in re.findall(r'\w+', query_text) if len(t) > 2]
        
        matched_nodes: Set[str] = set()
        semantic_results: List[Tuple[str, float]] = []
        
        # L1: Semantic search
        if use_semantic and self.vector._faiss_index is not None:
            query_emb = self.vector.embed_single(query_text)
            semantic_results = self.vector.search(query_emb, k=max_results)
            
            thought = self.reasoning.branch(
                self.reasoning.root_id,
                f"Semantic search: {len(semantic_results)} vectors",
                confidence=0.8,
                evidence=[r[0] for r in semantic_results[:5]]
            )
        
        # L2: Graph name matching
        if use_graph:
            for term in terms:
                if term in self.graph.name_index:
                    matched_nodes.update(self.graph.name_index[term])
            
            self.reasoning.branch(
                self.reasoning.root_id,
                f"Graph match: {len(matched_nodes)} nodes",
                confidence=0.85,
                evidence=list(matched_nodes)[:5]
            )
            
            # Expand via graph traversal
            expanded = set(matched_nodes)
            for nid in list(matched_nodes)[:20]:
                neighbors = self.graph.get_neighbors(nid)
                for n in neighbors[:5]:
                    expanded.add(n.id)
            
            self.reasoning.branch(
                self.reasoning.root_id,
                f"Graph expansion: {len(expanded)} nodes",
                confidence=0.7
            )
            matched_nodes = expanded
        
        # Merge semantic and graph results
        all_candidates: Dict[str, float] = {}
        
        for nid in matched_nodes:
            if nid in self.graph.nodes:
                all_candidates[nid] = all_candidates.get(nid, 0) + 0.5
        
        for chunk_id, score in semantic_results:
            # Try to map chunk to node
            for nid in self.graph.nodes:
                if chunk_id in nid or nid in chunk_id:
                    all_candidates[nid] = all_candidates.get(nid, 0) + score
                    break
        
        # Rank by combined score + SNR + centrality
        ranked_nodes = []
        for nid, base_score in all_candidates.items():
            node = self.graph.nodes[nid]
            final_score = base_score + node.snr_score * 0.3 + node.centrality * 10
            ranked_nodes.append((node, final_score))
        
        ranked_nodes.sort(key=lambda x: x[1], reverse=True)
        result_nodes = [n for n, s in ranked_nodes[:max_results]]
        
        # Find paths between top results
        paths = []
        if len(result_nodes) >= 2:
            for i in range(min(3, len(result_nodes))):
                for j in range(i+1, min(4, len(result_nodes))):
                    path = self.graph.find_path(result_nodes[i].id, result_nodes[j].id)
                    if path:
                        paths.append(path)
        
        # L4: Match against discovered patterns
        matched_patterns = []
        result_ids = {n.id for n in result_nodes}
        for pattern in self.pattern.patterns[:20]:
            overlap = len(set(pattern.nodes) & result_ids)
            if overlap >= 2:
                matched_patterns.append(pattern)
        
        # Generate insights
        insights = self._generate_insights(result_nodes, terms, paths, matched_patterns)
        
        # Synthesize final thought
        leaves = self.reasoning.get_all_leaves()
        self.reasoning.merge(
            [l.id for l in leaves],
            f"Synthesized {len(result_nodes)} results across {len(paths)} paths"
        )
        
        # Calculate final SNR
        avg_centrality = sum(n.centrality for n in result_nodes) / max(len(result_nodes), 1)
        snr = self.reasoning.calculate_snr(result_nodes, terms, len(paths), avg_centrality)
        
        # Collect edges
        result_ids = {n.id for n in result_nodes}
        edges = []
        for node in result_nodes:
            for edge in self.graph.edges.get(node.id, []):
                if edge.target_id in result_ids:
                    edges.append(edge)
        
        execution_time = (time.time() - start_time) * 1000
        
        return ApexQueryResult(
            query=query_text,
            nodes=result_nodes,
            edges=edges,
            patterns=matched_patterns,
            reasoning_chain=self.reasoning.get_best_path(),
            semantic_paths=paths,
            insights=insights,
            snr_score=snr,
            execution_time_ms=round(execution_time, 2)
        )
    
    def _generate_insights(self, nodes: List[KnowledgeNode], terms: List[str],
                          paths: List[List[str]], patterns: List[DiscoveredPattern]) -> List[str]:
        """Generate human-readable insights from query results."""
        insights = []
        
        if not nodes:
            return ["No results found"]
        
        # Type distribution
        type_counts = defaultdict(int)
        for n in nodes:
            type_counts[n.type.name] += 1
        
        dominant = max(type_counts.items(), key=lambda x: x[1])
        insights.append(f"Dominated by {dominant[0]} ({dominant[1]}/{len(nodes)})")
        
        # BIZRA assets
        bizra_count = sum(1 for n in nodes if n.metadata.get("is_bizra") or "bizra" in n.name.lower())
        if bizra_count:
            insights.append(f"{bizra_count} BIZRA assets found")
        
        # High SNR nodes
        high_snr = [n for n in nodes if n.snr_score > 0.6]
        if high_snr:
            insights.append(f"{len(high_snr)} high-SNR nodes (>0.6)")
        
        # Paths
        if paths:
            insights.append(f"{len(paths)} knowledge paths discovered")
        
        # Patterns
        if patterns:
            insights.append(f"Matched {len(patterns)} patterns")
        
        # Communities
        communities = set(n.community_id for n in nodes if n.community_id is not None)
        if len(communities) > 1:
            insights.append(f"Spans {len(communities)} communities")
        
        return insights
    
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # PERSISTENCE — Save/Load Operations
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    
    def save(self):
        """Save the complete Apex index to disk."""
        ApexConfig.GOLD.mkdir(parents=True, exist_ok=True)
        
        # Save graph
        graph_data = {
            "nodes": [n.to_dict() for n in self.graph.nodes.values()],
            "edges": [e.to_dict() for edges in self.graph.edges.values() for e in edges]
        }
        with open(ApexConfig.APEX_GRAPH, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        # Save patterns
        with open(ApexConfig.APEX_PATTERNS, 'w', encoding='utf-8') as f:
            for p in self.pattern.patterns:
                f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
        
        # Save stats
        with open(ApexConfig.APEX_STATS, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        # Save FAISS index
        self.vector.save_index(ApexConfig.APEX_INDEX)
        
        log.info(f"💾 Apex index saved to {ApexConfig.GOLD}")
    
    def load(self) -> bool:
        """Load the Apex index from disk."""
        if not ApexConfig.APEX_GRAPH.exists():
            return False
        
        try:
            # Load graph
            with open(ApexConfig.APEX_GRAPH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for nd in data.get("nodes", []):
                # Skip embedding to save memory on load
                nd['embedding'] = None
                self.graph.add_node(KnowledgeNode.from_dict(nd))
            
            for ed in data.get("edges", []):
                self.graph.add_edge(KnowledgeEdge.from_dict(ed))
            
            # Load patterns
            if ApexConfig.APEX_PATTERNS.exists():
                with open(ApexConfig.APEX_PATTERNS, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.pattern.patterns.append(
                                DiscoveredPattern(**json.loads(line))
                            )
            
            # Load stats
            if ApexConfig.APEX_STATS.exists():
                with open(ApexConfig.APEX_STATS, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
            
            # Load FAISS
            self.vector.load_index(ApexConfig.APEX_INDEX)
            
            log.info(f"📂 Loaded {len(self.graph.nodes)} nodes, {len(self.pattern.patterns)} patterns")
            return True
            
        except Exception as e:
            log.error(f"Failed to load: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# CLI — COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

def main():
    import sys
    
    apex = ApexUnifiedEngine(lazy_load=True)
    
    if len(sys.argv) < 2:
        print("""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                 APEX UNIFIED KNOWLEDGE ENGINE — HOUSE OF WISDOM                       ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║   python sovereign_apex.py build           — Build the unified knowledge index       ║
║   python sovereign_apex.py query "text"    — Query the knowledge graph               ║
║   python sovereign_apex.py stats           — Show engine statistics                  ║
║   python sovereign_apex.py bizra           — Find all BIZRA assets                   ║
║   python sovereign_apex.py patterns        — Show discovered patterns                ║
║   python sovereign_apex.py communities     — Show detected communities               ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
        """)
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == "build":
        apex.build_full_index()
    
    elif cmd == "query" and len(sys.argv) > 2:
        if apex.load():
            q = " ".join(sys.argv[2:])
            result = apex.query(q)
            
            print(f"\n🔍 APEX QUERY RESULTS (SNR: {result.snr_score} | {result.execution_time_ms}ms)")
            print("═" * 70)
            
            for node in result.nodes[:20]:
                bar = "█" * int(node.snr_score * 10)
                cent = "●" * min(int(node.centrality * 100), 5)
                print(f"  [{node.type.name:10}] {node.name[:35]:35} {bar} {cent}")
            
            if result.insights:
                print(f"\n💡 INSIGHTS: {' | '.join(result.insights)}")
            
            if result.patterns:
                print(f"\n🧩 MATCHED PATTERNS:")
                for p in result.patterns[:3]:
                    print(f"   • {p.description}")
            
            if result.reasoning_chain:
                print(f"\n🧠 REASONING CHAIN:")
                for t in result.reasoning_chain[:5]:
                    indent = "  " * t.depth
                    print(f"   {indent}→ {t.thought[:60]} (conf: {t.confidence:.2f})")
            
            print("═" * 70)
        else:
            print("❌ Run 'build' first to create the index")
    
    elif cmd == "stats":
        if apex.load():
            print(f"\n📊 APEX ENGINE STATISTICS")
            print("═" * 50)
            print(f"  Nodes:       {apex.stats['nodes']}")
            print(f"  Edges:       {apex.stats['edges']}")
            print(f"  Embeddings:  {apex.stats['embeddings']}")
            print(f"  Patterns:    {apex.stats['patterns']}")
            print(f"  Communities: {apex.stats['communities']}")
            print(f"  Created:     {apex.stats.get('created_at', 'N/A')}")
            print(f"  Updated:     {apex.stats.get('last_updated', 'N/A')}")
            print("═" * 50)
        else:
            print("❌ Run 'build' first")
    
    elif cmd == "bizra":
        if apex.load():
            print(f"\n🌱 BIZRA ECOSYSTEM ASSETS")
            print("═" * 60)
            
            bizra = [n for n in apex.graph.nodes.values() 
                    if n.metadata.get("is_bizra") or "bizra" in n.name.lower()]
            bizra.sort(key=lambda x: x.snr_score, reverse=True)
            
            for node in bizra[:40]:
                bar = "█" * int(node.snr_score * 10)
                print(f"  [{node.type.name:8}] {node.name[:40]:40} {bar}")
            
            print(f"\n  TOTAL: {len(bizra)} BIZRA assets")
            print("═" * 60)
        else:
            print("❌ Run 'build' first")
    
    elif cmd == "patterns":
        if apex.load():
            print(f"\n🧩 DISCOVERED PATTERNS")
            print("═" * 70)
            
            for p in apex.pattern.patterns[:25]:
                conf_bar = "█" * int(p.confidence * 10)
                print(f"  [{p.pattern_type:15}] {p.description[:45]:45} {conf_bar} ({p.support})")
            
            print(f"\n  TOTAL: {len(apex.pattern.patterns)} patterns")
            print("═" * 70)
        else:
            print("❌ Run 'build' first")
    
    elif cmd == "communities":
        if apex.load():
            print(f"\n🏘️  DETECTED COMMUNITIES")
            print("═" * 60)
            
            comm_nodes = defaultdict(list)
            for node in apex.graph.nodes.values():
                if node.community_id is not None:
                    comm_nodes[node.community_id].append(node)
            
            for cid in sorted(comm_nodes.keys())[:10]:
                nodes = comm_nodes[cid]
                top_nodes = sorted(nodes, key=lambda x: x.centrality, reverse=True)[:3]
                names = ", ".join(n.name[:20] for n in top_nodes)
                print(f"  Community {cid}: {len(nodes)} nodes — {names}...")
            
            print(f"\n  TOTAL: {len(comm_nodes)} communities")
            print("═" * 60)
        else:
            print("❌ Run 'build' first")
    
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
