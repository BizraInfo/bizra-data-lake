# BIZRA Hypergraph RAG Engine v1.1
# Standing on Giants: FAISS (Meta), NetworkX, Sentence-Transformers, XTR-WARP (ColBERT)
# Implements: HNSW indexing, Graph traversal, SNR-validated retrieval
# v1.1: Added WARP multi-vector retrieval as high-accuracy alternative
# Architecture: Symbolic-Neural Bridge with measurable quality metrics

import numpy as np
import pandas as pd
import faiss
import networkx as nx
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import torch
from bizra_config import (
    CORPUS_TABLE_PATH, CHUNKS_TABLE_PATH, INDEXED_PATH, GRAPH_PATH,
    EMBEDDINGS_PATH, SNR_THRESHOLD, BATCH_SIZE, MAX_SEQ_LENGTH,
    WARP_ENABLED
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(INDEXED_PATH / "hypergraph_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HypergraphEngine")


class RetrievalMode(Enum):
    """Retrieval strategies for different query types."""
    SEMANTIC = "semantic"           # Pure vector similarity (FAISS)
    STRUCTURAL = "structural"       # Pure graph traversal
    HYBRID = "hybrid"               # Combined (default)
    MULTI_HOP = "multi_hop"         # Graph-guided semantic chains
    WARP = "warp"                   # ColBERT/XTR multi-vector (high accuracy)


@dataclass
class RetrievalResult:
    """Single retrieval result with full provenance."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    retrieval_path: List[str]       # How we found this (for explainability)
    graph_distance: int = 0         # Hops from query anchor
    semantic_rank: int = 0          # Position in vector search
    metadata: Dict = field(default_factory=dict)


@dataclass
class QueryContext:
    """Assembled context for RAG generation."""
    query: str
    query_embedding: np.ndarray
    results: List[RetrievalResult]
    snr_score: float
    reasoning_trace: List[str]      # Graph-of-Thoughts trace
    total_tokens_est: int
    retrieval_mode: RetrievalMode


class SNRCalculator:
    """
    Signal-to-Noise Ratio Calculator

    Implements information-theoretic SNR measurement:
    - Signal: Semantic relevance to query (cosine similarity)
    - Noise: Redundancy + irrelevance in retrieved set

    Formula: SNR = (mean_relevance * diversity_factor) / (1 + redundancy_penalty)
    """

    def __init__(self, relevance_threshold: float = 0.3):
        self.relevance_threshold = relevance_threshold

    def calculate(
        self,
        query_embedding: np.ndarray,
        result_embeddings: np.ndarray,
        results: List[RetrievalResult]
    ) -> Tuple[float, Dict]:
        """
        Calculate SNR for a retrieval set.

        Returns:
            Tuple of (snr_score, detailed_metrics)
        """
        if len(results) == 0:
            return 0.0, {"error": "empty_results"}

        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        results_norm = result_embeddings / (np.linalg.norm(result_embeddings, axis=1, keepdims=True) + 1e-9)

        # Signal: Mean cosine similarity to query
        similarities = np.dot(results_norm, query_norm)
        signal_strength = float(np.mean(similarities[similarities > self.relevance_threshold]))
        if np.isnan(signal_strength):
            signal_strength = float(np.mean(similarities))

        # Diversity: How different are the results from each other?
        pairwise_sim = np.dot(results_norm, results_norm.T)
        np.fill_diagonal(pairwise_sim, 0)  # Exclude self-similarity
        redundancy = float(np.mean(pairwise_sim))
        diversity_factor = 1.0 - redundancy

        # Coverage: How many unique documents represented?
        unique_docs = len(set(r.doc_id for r in results))
        coverage_factor = unique_docs / max(len(results), 1)

        # Final SNR calculation
        snr = (signal_strength * diversity_factor * coverage_factor) / (1 + redundancy)
        snr = min(max(snr, 0.0), 1.0)  # Clamp to [0, 1]

        metrics = {
            "signal_strength": round(signal_strength, 4),
            "diversity_factor": round(diversity_factor, 4),
            "redundancy": round(redundancy, 4),
            "coverage_factor": round(coverage_factor, 4),
            "unique_documents": unique_docs,
            "total_results": len(results),
            "above_threshold": int(np.sum(similarities > self.relevance_threshold))
        }

        return round(snr, 4), metrics


class HypergraphIndex:
    """
    HNSW Vector Index + Knowledge Graph Integration

    Combines:
    - FAISS HNSW index for O(log n) approximate nearest neighbor search
    - NetworkX MultiDiGraph for relationship traversal
    - Chunk-to-document mapping for provenance
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.IndexHNSWFlat] = None
        self.graph: Optional[nx.MultiDiGraph] = None
        self.chunk_ids: List[str] = []
        self.chunk_to_doc: Dict[str, str] = {}
        self.chunk_texts: Dict[str, str] = {}
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.doc_to_chunks: Dict[str, List[str]] = defaultdict(list)
        self.is_built = False

    def build_from_parquet(self, chunks_path: Path, graph_path: Optional[Path] = None) -> bool:
        """
        Build index from chunks.parquet and optional graph files.

        Args:
            chunks_path: Path to chunks.parquet with embeddings
            graph_path: Path to directory with nodes.jsonl and edges.jsonl

        Returns:
            True if successful
        """
        logger.info(f"Building hypergraph index from {chunks_path}")

        try:
            # Load chunks with embeddings
            df_chunks = pd.read_parquet(chunks_path)
            logger.info(f"Loaded {len(df_chunks)} chunks")

            # Extract embeddings (stored as lists in Parquet)
            embeddings_list = df_chunks['embedding'].tolist()
            self.chunk_embeddings = np.array(embeddings_list, dtype=np.float32)

            # Validate embedding dimensions
            if self.chunk_embeddings.shape[1] != self.embedding_dim:
                logger.warning(f"Embedding dim mismatch: expected {self.embedding_dim}, got {self.chunk_embeddings.shape[1]}")
                self.embedding_dim = self.chunk_embeddings.shape[1]

            # Build FAISS HNSW index
            # HNSW parameters: M=32 (connections per layer), efConstruction=200 (build quality)
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 64  # Search quality
            self.index.add(self.chunk_embeddings)
            logger.info(f"FAISS HNSW index built with {self.index.ntotal} vectors")

            # Build lookup tables
            self.chunk_ids = df_chunks['chunk_id'].tolist()
            self.chunk_to_doc = dict(zip(df_chunks['chunk_id'], df_chunks['doc_id']))
            self.chunk_texts = dict(zip(df_chunks['chunk_id'], df_chunks['chunk_text']))

            for chunk_id, doc_id in self.chunk_to_doc.items():
                self.doc_to_chunks[doc_id].append(chunk_id)

            # Load knowledge graph if available
            if graph_path and graph_path.exists():
                self._load_graph(graph_path)
            else:
                # Create minimal graph from document relationships
                self._build_minimal_graph(df_chunks)

            self.is_built = True
            logger.info("Hypergraph index build complete")
            return True

        except Exception as e:
            logger.error(f"Failed to build index: {e}", exc_info=True)
            return False

    def _load_graph(self, graph_path: Path):
        """Load pre-built knowledge graph from JSONL files."""
        self.graph = nx.MultiDiGraph()

        nodes_file = graph_path / "nodes.jsonl"
        edges_file = graph_path / "edges.jsonl"

        if nodes_file.exists():
            with open(nodes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    node = json.loads(line)
                    self.graph.add_node(node['id'], **node.get('attributes', {}))
            logger.info(f"Loaded {self.graph.number_of_nodes()} nodes")

        if edges_file.exists():
            with open(edges_file, 'r', encoding='utf-8') as f:
                for line in f:
                    edge = json.loads(line)
                    self.graph.add_edge(
                        edge['source'],
                        edge['target'],
                        relation=edge.get('relation', 'RELATED'),
                        **edge.get('attributes', {})
                    )
            logger.info(f"Loaded {self.graph.number_of_edges()} edges")

    def _build_minimal_graph(self, df_chunks: pd.DataFrame):
        """Build minimal graph from chunk-document relationships."""
        self.graph = nx.MultiDiGraph()

        # Add document nodes
        doc_ids = df_chunks['doc_id'].unique()
        for doc_id in doc_ids:
            self.graph.add_node(f"doc::{doc_id}", type="document")

        # Add chunk nodes and edges
        for _, row in df_chunks.iterrows():
            chunk_id = row['chunk_id']
            doc_id = row['doc_id']
            self.graph.add_node(f"chunk::{chunk_id}", type="chunk")
            self.graph.add_edge(f"chunk::{chunk_id}", f"doc::{doc_id}", relation="PART_OF")

        logger.info(f"Built minimal graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def search_semantic(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Pure semantic search using HNSW index.

        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_from_parquet first.")

        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # FAISS returns L2 distances; convert to similarity
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunk_ids):
                chunk_id = self.chunk_ids[idx]
                # Convert L2 distance to similarity score (1 / (1 + dist))
                similarity = 1.0 / (1.0 + dist)
                results.append((chunk_id, similarity))

        return results

    def search_graph_neighbors(
        self,
        anchor_ids: List[str],
        max_hops: int = 2,
        relation_filter: Optional[Set[str]] = None
    ) -> Dict[str, int]:
        """
        Find graph neighbors within max_hops of anchor nodes.

        Returns:
            Dict mapping node_id to minimum hop distance
        """
        if not self.graph:
            return {}

        neighbors = {}
        frontier = set()

        # Initialize with anchors
        for anchor in anchor_ids:
            # Try both chunk:: and doc:: prefixes
            for prefix in ["chunk::", "doc::", ""]:
                node_id = f"{prefix}{anchor}"
                if self.graph.has_node(node_id):
                    frontier.add(node_id)
                    neighbors[node_id] = 0

        # BFS expansion
        for hop in range(1, max_hops + 1):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in neighbors:
                        # Apply relation filter if specified
                        if relation_filter:
                            edges = self.graph.get_edge_data(node, neighbor)
                            if edges:
                                relations = {e.get('relation') for e in edges.values()}
                                if not relations.intersection(relation_filter):
                                    continue
                        neighbors[neighbor] = hop
                        next_frontier.add(neighbor)
            frontier = next_frontier

        return neighbors

    def get_chunk_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific chunk."""
        if chunk_id in self.chunk_ids:
            idx = self.chunk_ids.index(chunk_id)
            return self.chunk_embeddings[idx]
        return None


class HypergraphRAGEngine:
    """
    Complete Hypergraph RAG Engine

    Implements the full retrieval pipeline:
    1. Query embedding
    2. Hybrid retrieval (semantic + structural)
    3. Multi-hop expansion
    4. SNR-validated context assembly
    5. Graph-of-Thoughts reasoning trace
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Initializing HypergraphRAGEngine with model: {model_name}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.encoder.max_seq_length = MAX_SEQ_LENGTH

        self.index = HypergraphIndex(embedding_dim=384)
        self.snr_calculator = SNRCalculator()
        self.is_initialized = False

        logger.info(f"Using device: {self.device}")

    def initialize(self) -> bool:
        """Initialize the engine by building indices."""
        if not CHUNKS_TABLE_PATH.exists():
            logger.error(f"Chunks table not found: {CHUNKS_TABLE_PATH}")
            return False

        success = self.index.build_from_parquet(
            CHUNKS_TABLE_PATH,
            GRAPH_PATH if GRAPH_PATH.exists() else None
        )

        self.is_initialized = success
        return success

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        k: int = 10,
        max_hops: int = 2,
        snr_threshold: float = SNR_THRESHOLD,
        max_tokens: int = 4000
    ) -> QueryContext:
        """
        Execute retrieval with specified mode.

        Args:
            query: Natural language query
            mode: Retrieval strategy
            k: Number of initial results
            max_hops: Maximum graph traversal depth
            snr_threshold: Minimum SNR for quality gate
            max_tokens: Maximum tokens in assembled context

        Returns:
            QueryContext with results and metadata
        """
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize engine")

        reasoning_trace = [f"Query: {query}", f"Mode: {mode.value}"]

        # Step 1: Encode query (skip for WARP mode which has its own encoder)
        if mode != RetrievalMode.WARP:
            query_embedding = self.encoder.encode(query, convert_to_numpy=True)
            reasoning_trace.append(f"Encoded query to {len(query_embedding)}-dim vector")
        else:
            query_embedding = None

        # Step 2: Initial retrieval based on mode
        if mode == RetrievalMode.SEMANTIC:
            results = self._semantic_retrieval(query_embedding, k, reasoning_trace)
        elif mode == RetrievalMode.STRUCTURAL:
            results = self._structural_retrieval(query, k, reasoning_trace)
        elif mode == RetrievalMode.MULTI_HOP:
            results = self._multi_hop_retrieval(query_embedding, k, max_hops, reasoning_trace)
        elif mode == RetrievalMode.WARP:
            results = self._warp_retrieval(query, k, reasoning_trace)
        else:  # HYBRID (default)
            results = self._hybrid_retrieval(query_embedding, k, max_hops, reasoning_trace)

        # Step 3: Calculate SNR
        if results and query_embedding is not None:
            result_embeddings = np.array([
                self.index.get_chunk_embedding(r.chunk_id)
                for r in results
                if self.index.get_chunk_embedding(r.chunk_id) is not None
            ])

            if len(result_embeddings) > 0:
                snr_score, snr_metrics = self.snr_calculator.calculate(
                    query_embedding, result_embeddings, results
                )
                reasoning_trace.append(f"SNR Score: {snr_score} (threshold: {snr_threshold})")
                reasoning_trace.append(f"SNR Metrics: {snr_metrics}")
            else:
                snr_score = 0.0
                reasoning_trace.append("SNR: Unable to calculate (no valid embeddings)")
        else:
            snr_score = 0.0
            reasoning_trace.append("SNR: 0.0 (no results)")

        # Step 4: SNR-based refinement
        if snr_score < snr_threshold and mode != RetrievalMode.SEMANTIC:
            reasoning_trace.append(f"SNR below threshold, expanding search...")
            expanded_results = self._expand_low_snr(
                query_embedding, results, k * 2, reasoning_trace
            )
            if expanded_results:
                results = expanded_results
                # Recalculate SNR
                result_embeddings = np.array([
                    self.index.get_chunk_embedding(r.chunk_id)
                    for r in results
                    if self.index.get_chunk_embedding(r.chunk_id) is not None
                ])
                if len(result_embeddings) > 0:
                    snr_score, _ = self.snr_calculator.calculate(
                        query_embedding, result_embeddings, results
                    )
                    reasoning_trace.append(f"SNR after expansion: {snr_score}")

        # Step 5: Token-aware truncation
        total_tokens = 0
        truncated_results = []
        for result in results:
            token_est = len(result.text.split()) * 1.3
            if total_tokens + token_est <= max_tokens:
                truncated_results.append(result)
                total_tokens += token_est
            else:
                reasoning_trace.append(f"Token limit reached, keeping {len(truncated_results)} results")
                break

        return QueryContext(
            query=query,
            query_embedding=query_embedding,
            results=truncated_results,
            snr_score=snr_score,
            reasoning_trace=reasoning_trace,
            total_tokens_est=int(total_tokens),
            retrieval_mode=mode
        )

    def _semantic_retrieval(
        self,
        query_embedding: np.ndarray,
        k: int,
        trace: List[str]
    ) -> List[RetrievalResult]:
        """Pure semantic search."""
        trace.append(f"Semantic search: k={k}")

        search_results = self.index.search_semantic(query_embedding, k)

        results = []
        for rank, (chunk_id, score) in enumerate(search_results):
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                doc_id=self.index.chunk_to_doc.get(chunk_id, "unknown"),
                text=self.index.chunk_texts.get(chunk_id, ""),
                score=score,
                retrieval_path=["semantic"],
                semantic_rank=rank + 1
            ))

        trace.append(f"Found {len(results)} semantic matches")
        return results

    def _structural_retrieval(
        self,
        query: str,
        k: int,
        trace: List[str]
    ) -> List[RetrievalResult]:
        """Graph-based retrieval using entity matching."""
        trace.append(f"Structural search: k={k}")

        # Extract potential entities from query
        query_terms = set(query.lower().split())

        # Find matching nodes in graph
        matching_nodes = []
        if self.index.graph:
            for node in self.index.graph.nodes():
                node_lower = node.lower()
                for term in query_terms:
                    if term in node_lower and len(term) > 3:
                        matching_nodes.append(node)
                        break

        trace.append(f"Found {len(matching_nodes)} matching nodes")

        # Get chunks connected to matching nodes
        results = []
        seen_chunks = set()

        for node in matching_nodes[:k]:
            neighbors = self.index.search_graph_neighbors([node], max_hops=2)
            for neighbor, distance in neighbors.items():
                if neighbor.startswith("chunk::"):
                    chunk_id = neighbor.replace("chunk::", "")
                    if chunk_id not in seen_chunks and chunk_id in self.index.chunk_texts:
                        seen_chunks.add(chunk_id)
                        results.append(RetrievalResult(
                            chunk_id=chunk_id,
                            doc_id=self.index.chunk_to_doc.get(chunk_id, "unknown"),
                            text=self.index.chunk_texts.get(chunk_id, ""),
                            score=1.0 / (1 + distance),
                            retrieval_path=["structural", node],
                            graph_distance=distance
                        ))

        trace.append(f"Found {len(results)} structural matches")
        return results[:k]

    def _hybrid_retrieval(
        self,
        query_embedding: np.ndarray,
        k: int,
        max_hops: int,
        trace: List[str]
    ) -> List[RetrievalResult]:
        """Combined semantic + structural retrieval."""
        trace.append(f"Hybrid search: k={k}, max_hops={max_hops}")

        # Get semantic results
        semantic_results = self._semantic_retrieval(query_embedding, k, trace)

        # Use top semantic results as anchors for graph expansion
        anchor_chunks = [r.chunk_id for r in semantic_results[:k//2]]
        anchor_docs = [r.doc_id for r in semantic_results[:k//2]]

        # Find graph neighbors
        neighbors = self.index.search_graph_neighbors(
            anchor_chunks + anchor_docs,
            max_hops=max_hops
        )

        # Collect neighbor chunks
        neighbor_results = []
        seen_chunks = set(r.chunk_id for r in semantic_results)

        for node_id, distance in neighbors.items():
            if node_id.startswith("chunk::"):
                chunk_id = node_id.replace("chunk::", "")
                if chunk_id not in seen_chunks and chunk_id in self.index.chunk_texts:
                    # Get embedding and calculate similarity
                    chunk_emb = self.index.get_chunk_embedding(chunk_id)
                    if chunk_emb is not None:
                        similarity = float(np.dot(
                            query_embedding / np.linalg.norm(query_embedding),
                            chunk_emb / np.linalg.norm(chunk_emb)
                        ))
                        neighbor_results.append(RetrievalResult(
                            chunk_id=chunk_id,
                            doc_id=self.index.chunk_to_doc.get(chunk_id, "unknown"),
                            text=self.index.chunk_texts.get(chunk_id, ""),
                            score=similarity * (1.0 / (1 + distance)),  # Weighted by distance
                            retrieval_path=["hybrid", "graph_expansion"],
                            graph_distance=distance,
                            semantic_rank=0
                        ))

        # Merge and sort by score
        all_results = semantic_results + neighbor_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        trace.append(f"Hybrid total: {len(all_results)} ({len(semantic_results)} semantic + {len(neighbor_results)} graph)")
        return all_results[:k * 2]  # Return more for SNR filtering

    def _warp_retrieval(
        self,
        query: str,
        k: int,
        trace: List[str]
    ) -> List[RetrievalResult]:
        """
        Multi-vector retrieval using XTR-WARP (ColBERT late interaction).
        
        WARP provides higher accuracy than single-vector retrieval by using
        per-token embeddings and late interaction scoring.
        """
        trace.append(f"WARP multi-vector search: k={k}")
        
        try:
            from warp_bridge import WARPBridge
            
            # Use lazy-initialized WARP bridge
            if not hasattr(self, '_warp_bridge'):
                self._warp_bridge = WARPBridge(lazy_init=True)
            
            response = self._warp_bridge.search(query, k=k)
            
            if response.metadata.get('fallback'):
                trace.append(f"WARP fallback: {response.metadata.get('error', 'unknown')}")
                # Fall back to semantic retrieval
                query_embedding = self.encoder.encode(query, convert_to_numpy=True)
                return self._semantic_retrieval(query_embedding, k, trace)
            
            results = []
            for r in response.results:
                results.append(RetrievalResult(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    text=r.text,
                    score=r.score,
                    retrieval_path=["warp", "colbert_late_interaction"],
                    graph_distance=0,
                    semantic_rank=r.rank,
                    metadata={"warp_score": r.score}
                ))
            
            trace.append(f"WARP returned {len(results)} results (SNR estimate: {response.snr_estimate:.4f})")
            return results
            
        except ImportError:
            trace.append("WARP not available, falling back to semantic retrieval")
            query_embedding = self.encoder.encode(query, convert_to_numpy=True)
            return self._semantic_retrieval(query_embedding, k, trace)
        except Exception as e:
            trace.append(f"WARP error: {e}, falling back to semantic retrieval")
            query_embedding = self.encoder.encode(query, convert_to_numpy=True)
            return self._semantic_retrieval(query_embedding, k, trace)

    def _multi_hop_retrieval(
        self,
        query_embedding: np.ndarray,
        k: int,
        max_hops: int,
        trace: List[str]
    ) -> List[RetrievalResult]:
        """Multi-hop reasoning: iterative semantic + graph traversal."""
        trace.append(f"Multi-hop search: k={k}, max_hops={max_hops}")

        all_results = []
        seen_chunks = set()
        current_embedding = query_embedding

        for hop in range(max_hops):
            trace.append(f"Hop {hop + 1}")

            # Semantic search from current position
            hop_results = self.index.search_semantic(current_embedding, k)

            new_results = []
            for rank, (chunk_id, score) in enumerate(hop_results):
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    new_results.append(RetrievalResult(
                        chunk_id=chunk_id,
                        doc_id=self.index.chunk_to_doc.get(chunk_id, "unknown"),
                        text=self.index.chunk_texts.get(chunk_id, ""),
                        score=score * (0.9 ** hop),  # Decay by hop
                        retrieval_path=["multi_hop", f"hop_{hop + 1}"],
                        graph_distance=hop,
                        semantic_rank=rank + 1
                    ))

            all_results.extend(new_results)
            trace.append(f"Hop {hop + 1}: found {len(new_results)} new chunks")

            # Update embedding for next hop (centroid of new results)
            if new_results:
                embeddings = [
                    self.index.get_chunk_embedding(r.chunk_id)
                    for r in new_results[:3]
                    if self.index.get_chunk_embedding(r.chunk_id) is not None
                ]
                if embeddings:
                    current_embedding = np.mean(embeddings, axis=0)

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:k * 2]

    def _expand_low_snr(
        self,
        query_embedding: np.ndarray,
        current_results: List[RetrievalResult],
        k: int,
        trace: List[str]
    ) -> List[RetrievalResult]:
        """Expand search when SNR is below threshold."""
        trace.append("Expanding search for better diversity")

        # Get more candidates from semantic search
        all_candidates = self.index.search_semantic(query_embedding, k * 2)

        # Filter out already-seen chunks
        seen = set(r.chunk_id for r in current_results)
        new_candidates = [(c, s) for c, s in all_candidates if c not in seen]

        # Select diverse candidates (maximize doc coverage)
        doc_count = defaultdict(int)
        for r in current_results:
            doc_count[r.doc_id] += 1

        diverse_results = list(current_results)
        for chunk_id, score in new_candidates:
            doc_id = self.index.chunk_to_doc.get(chunk_id, "unknown")
            if doc_count[doc_id] < 2:  # Max 2 chunks per doc
                diverse_results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=self.index.chunk_texts.get(chunk_id, ""),
                    score=score,
                    retrieval_path=["expansion", "diversity"],
                    semantic_rank=len(diverse_results) + 1
                ))
                doc_count[doc_id] += 1

            if len(diverse_results) >= k:
                break

        trace.append(f"Expanded to {len(diverse_results)} results")
        return diverse_results


def main():
    """Demonstration of Hypergraph RAG Engine."""
    print("=" * 70)
    print("BIZRA HYPERGRAPH RAG ENGINE v1.0")
    print("=" * 70)

    engine = HypergraphRAGEngine()

    if not engine.initialize():
        print("Failed to initialize engine. Ensure chunks.parquet exists.")
        return

    # Test queries
    test_queries = [
        "How does the BIZRA system process incoming files?",
        "What is the architecture of the data lake?",
        "Explain the embedding generation process"
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print("=" * 70)

        context = engine.retrieve(
            query=query,
            mode=RetrievalMode.HYBRID,
            k=5,
            max_hops=2
        )

        print(f"\nSNR Score: {context.snr_score}")
        print(f"Total Results: {len(context.results)}")
        print(f"Estimated Tokens: {context.total_tokens_est}")
        print(f"\nReasoning Trace:")
        for step in context.reasoning_trace:
            print(f"  - {step}")

        print(f"\nTop Results:")
        for i, result in enumerate(context.results[:3]):
            print(f"\n  [{i+1}] Score: {result.score:.4f} | Doc: {result.doc_id}")
            print(f"      Path: {' -> '.join(result.retrieval_path)}")
            print(f"      Text: {result.text[:200]}...")


if __name__ == "__main__":
    main()
