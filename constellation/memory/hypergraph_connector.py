# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Constellation - HyperGraphRAG Integration v1.0
# ═══════════════════════════════════════════════════════════════════════════════
"""
Connects the Islamic Masterminds Constellation to HyperGraphRAG for:
- Knowledge retrieval across agent domains
- Semantic memory storage and recall
- Cross-agent knowledge sharing
- Evidence verification through graph traversal
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum

# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE NODE TYPES
# ─────────────────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    CONCEPT = "concept"
    AGENT = "agent"
    DOMAIN = "domain"
    CLAIM = "claim"
    EVIDENCE = "evidence"
    MEMORY = "memory"
    SKILL = "skill"
    TOOL = "tool"
    SESSION = "session"


class EdgeType(str, Enum):
    """Types of edges in the knowledge graph."""
    RELATES_TO = "relates_to"
    DERIVED_FROM = "derived_from"
    VERIFIED_BY = "verified_by"
    PRODUCED_BY = "produced_by"
    RECALLS = "recalls"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    SPECIALIZES = "specializes"
    DELEGATES_TO = "delegates_to"
    TRIGGERS = "triggers"


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeNode:
    """A node in the knowledge hypergraph."""
    id: str
    type: NodeType
    content: str
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: Optional[str] = None  # Agent slug
    snr_score: float = 0.0
    claim_tag: Optional[str] = None  # MEASURED, IMPLEMENTED, etc.
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "snr_score": self.snr_score,
            "claim_tag": self.claim_tag,
        }


@dataclass
class HyperEdge:
    """A hyperedge connecting multiple nodes."""
    id: str
    type: EdgeType
    source_nodes: list[str]  # Node IDs
    target_nodes: list[str]  # Node IDs
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "source_nodes": self.source_nodes,
            "target_nodes": self.target_nodes,
            "weight": self.weight,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class RetrievalResult:
    """Result from knowledge retrieval."""
    nodes: list[KnowledgeNode]
    edges: list[HyperEdge]
    query: str
    relevance_scores: list[float]
    reasoning_path: list[str]
    total_found: int


# ─────────────────────────────────────────────────────────────────────────────
# HYPERGRAPH RAG CONNECTOR
# ─────────────────────────────────────────────────────────────────────────────

class HyperGraphRAGConnector:
    """
    Connects BIZRA Constellation to HyperGraphRAG for knowledge operations.
    
    Provides:
    - Semantic retrieval with graph-aware ranking
    - Multi-hop reasoning through hyperedges
    - Agent-specific knowledge filtering
    - Evidence chain verification
    """
    
    def __init__(
        self,
        graph_path: Optional[Path] = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.graph_path = graph_path or Path("bizra_data_vault/knowledge_graph")
        self.embedding_model = embedding_model
        self.nodes: dict[str, KnowledgeNode] = {}
        self.edges: dict[str, HyperEdge] = {}
        self._embedding_cache: dict[str, list[float]] = {}
        
    def initialize(self) -> None:
        """Initialize the knowledge graph, loading existing data."""
        self.graph_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing nodes
        nodes_file = self.graph_path / "nodes.jsonl"
        if nodes_file.exists():
            with open(nodes_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    node = KnowledgeNode(
                        id=data["id"],
                        type=NodeType(data["type"]),
                        content=data["content"],
                        embedding=data.get("embedding"),
                        metadata=data.get("metadata", {}),
                        created_at=data.get("created_at", ""),
                        created_by=data.get("created_by"),
                        snr_score=data.get("snr_score", 0.0),
                        claim_tag=data.get("claim_tag"),
                    )
                    self.nodes[node.id] = node
                    
        # Load existing edges
        edges_file = self.graph_path / "edges.jsonl"
        if edges_file.exists():
            with open(edges_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    edge = HyperEdge(
                        id=data["id"],
                        type=EdgeType(data["type"]),
                        source_nodes=data["source_nodes"],
                        target_nodes=data["target_nodes"],
                        weight=data.get("weight", 1.0),
                        metadata=data.get("metadata", {}),
                        created_at=data.get("created_at", ""),
                    )
                    self.edges[edge.id] = edge
                    
    def add_node(self, node: KnowledgeNode) -> str:
        """Add a node to the knowledge graph."""
        if not node.id:
            node.id = self._generate_id(node.content)
            
        self.nodes[node.id] = node
        self._persist_node(node)
        return node.id
        
    def add_edge(self, edge: HyperEdge) -> str:
        """Add a hyperedge to the knowledge graph."""
        if not edge.id:
            edge.id = self._generate_id(
                f"{edge.source_nodes}->{edge.target_nodes}"
            )
            
        self.edges[edge.id] = edge
        self._persist_edge(edge)
        return edge.id
        
    def retrieve(
        self,
        query: str,
        agent_filter: Optional[str] = None,
        domain_filter: Optional[str] = None,
        min_snr: float = 0.0,
        max_hops: int = 3,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Retrieve relevant knowledge using hybrid vector + graph search.
        
        Args:
            query: Natural language query
            agent_filter: Limit to knowledge from specific agent
            domain_filter: Limit to specific domain
            min_snr: Minimum SNR score threshold
            max_hops: Maximum graph traversal depth
            top_k: Number of results to return
            
        Returns:
            RetrievalResult with nodes, edges, and reasoning path
        """
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Score all nodes by relevance
        scored_nodes = []
        for node in self.nodes.values():
            # Apply filters
            if agent_filter and node.created_by != agent_filter:
                continue
            if domain_filter and node.metadata.get("domain") != domain_filter:
                continue
            if node.snr_score < min_snr:
                continue
                
            # Calculate relevance score
            score = self._calculate_relevance(query_embedding, node)
            scored_nodes.append((node, score))
            
        # Sort by relevance
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k seed nodes
        seed_nodes = [n for n, _ in scored_nodes[:top_k]]
        seed_scores = [s for _, s in scored_nodes[:top_k]]
        
        # Expand via graph traversal
        expanded_nodes, expanded_edges, path = self._graph_expand(
            seed_nodes, max_hops
        )
        
        return RetrievalResult(
            nodes=seed_nodes + expanded_nodes,
            edges=expanded_edges,
            query=query,
            relevance_scores=seed_scores,
            reasoning_path=path,
            total_found=len(self.nodes),
        )
        
    def store_agent_output(
        self,
        agent_slug: str,
        content: str,
        claims: list[dict],
        confidence: float,
        session_id: Optional[str] = None,
    ) -> list[str]:
        """
        Store agent output as knowledge nodes.
        
        Creates nodes for:
        - The main output content
        - Each individual claim with its tag
        - Links to the agent and session
        """
        node_ids = []
        
        # Create main content node
        content_node = KnowledgeNode(
            id=self._generate_id(f"{agent_slug}:{content[:100]}"),
            type=NodeType.MEMORY,
            content=content,
            created_by=agent_slug,
            snr_score=confidence,
            metadata={
                "session_id": session_id,
                "claim_count": len(claims),
            },
        )
        self.add_node(content_node)
        node_ids.append(content_node.id)
        
        # Create claim nodes
        claim_node_ids = []
        for claim in claims:
            claim_node = KnowledgeNode(
                id=self._generate_id(f"claim:{claim.get('text', '')[:50]}"),
                type=NodeType.CLAIM,
                content=claim.get("text", ""),
                created_by=agent_slug,
                snr_score=confidence,
                claim_tag=claim.get("tag"),
                metadata={
                    "evidence": claim.get("evidence"),
                },
            )
            self.add_node(claim_node)
            claim_node_ids.append(claim_node.id)
            node_ids.append(claim_node.id)
            
        # Create edges linking content to claims
        if claim_node_ids:
            edge = HyperEdge(
                id=self._generate_id(f"produced:{content_node.id}"),
                type=EdgeType.PRODUCED_BY,
                source_nodes=[content_node.id],
                target_nodes=claim_node_ids,
                metadata={"agent": agent_slug},
            )
            self.add_edge(edge)
            
        return node_ids
        
    def verify_claim(
        self,
        claim_text: str,
        verifier_agent: str,
    ) -> tuple[bool, list[KnowledgeNode]]:
        """
        Verify a claim against existing knowledge.
        
        Returns:
            Tuple of (verified, supporting_evidence)
        """
        # Find related claims
        results = self.retrieve(
            query=claim_text,
            min_snr=0.90,
            top_k=5,
        )
        
        supporting = []
        contradicting = []
        
        for node in results.nodes:
            if node.type == NodeType.CLAIM:
                # Check for contradiction or support
                # (In production, use semantic similarity)
                if node.snr_score >= 0.93:
                    supporting.append(node)
                    
        verified = len(supporting) > 0 and len(contradicting) == 0
        return verified, supporting
        
    def get_agent_knowledge(
        self,
        agent_slug: str,
        limit: int = 100,
    ) -> list[KnowledgeNode]:
        """Get all knowledge produced by a specific agent."""
        return [
            node for node in self.nodes.values()
            if node.created_by == agent_slug
        ][:limit]
        
    def get_domain_knowledge(
        self,
        domain: str,
        min_snr: float = 0.0,
    ) -> list[KnowledgeNode]:
        """Get all knowledge in a specific domain."""
        return [
            node for node in self.nodes.values()
            if node.metadata.get("domain") == domain
            and node.snr_score >= min_snr
        ]
        
    def find_contradictions(
        self,
        node_id: str,
    ) -> list[tuple[KnowledgeNode, HyperEdge]]:
        """Find nodes that contradict the given node."""
        contradictions = []
        
        for edge in self.edges.values():
            if edge.type == EdgeType.CONTRADICTS:
                if node_id in edge.source_nodes:
                    for target_id in edge.target_nodes:
                        if target_id in self.nodes:
                            contradictions.append(
                                (self.nodes[target_id], edge)
                            )
                            
        return contradictions
        
    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text (cached)."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
            
        # Mock embedding for now - replace with actual embedding call
        # In production: openai.embeddings.create(model=self.embedding_model, input=text)
        import random
        embedding = [random.random() for _ in range(1536)]
        
        self._embedding_cache[cache_key] = embedding
        return embedding
        
    def _calculate_relevance(
        self,
        query_embedding: list[float],
        node: KnowledgeNode,
    ) -> float:
        """Calculate relevance score between query and node."""
        if node.embedding is None:
            node.embedding = self._get_embedding(node.content)
            
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(query_embedding, node.embedding))
        norm_q = sum(a * a for a in query_embedding) ** 0.5
        norm_n = sum(a * a for a in node.embedding) ** 0.5
        
        if norm_q == 0 or norm_n == 0:
            return 0.0
            
        similarity = dot_product / (norm_q * norm_n)
        
        # Boost by SNR score
        boosted = similarity * (1 + 0.1 * node.snr_score)
        
        return boosted
        
    def _graph_expand(
        self,
        seed_nodes: list[KnowledgeNode],
        max_hops: int,
    ) -> tuple[list[KnowledgeNode], list[HyperEdge], list[str]]:
        """Expand from seed nodes via graph traversal."""
        expanded_nodes = []
        expanded_edges = []
        path = []
        
        visited = {n.id for n in seed_nodes}
        frontier = [n.id for n in seed_nodes]
        
        for hop in range(max_hops):
            if not frontier:
                break
                
            new_frontier = []
            
            for node_id in frontier:
                # Find edges connecting to this node
                for edge in self.edges.values():
                    if node_id in edge.source_nodes or node_id in edge.target_nodes:
                        expanded_edges.append(edge)
                        
                        # Get connected nodes
                        connected = edge.source_nodes + edge.target_nodes
                        for connected_id in connected:
                            if connected_id not in visited:
                                visited.add(connected_id)
                                if connected_id in self.nodes:
                                    expanded_nodes.append(self.nodes[connected_id])
                                    new_frontier.append(connected_id)
                                    path.append(
                                        f"hop_{hop}: {node_id} --{edge.type.value}--> {connected_id}"
                                    )
                                    
            frontier = new_frontier
            
        return expanded_nodes, expanded_edges, path
        
    def _generate_id(self, content: str) -> str:
        """Generate unique ID from content."""
        return hashlib.sha256(
            f"{content}:{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]
        
    def _persist_node(self, node: KnowledgeNode) -> None:
        """Persist node to storage."""
        nodes_file = self.graph_path / "nodes.jsonl"
        with open(nodes_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(node.to_dict()) + "\n")
            
    def _persist_edge(self, edge: HyperEdge) -> None:
        """Persist edge to storage."""
        edges_file = self.graph_path / "edges.jsonl"
        with open(edges_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(edge.to_dict()) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT KNOWLEDGE INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class AgentKnowledgeInterface:
    """
    Interface for agents to interact with the knowledge graph.
    
    Provides agent-specific methods for:
    - Retrieving relevant knowledge
    - Storing outputs
    - Verifying claims
    - Cross-agent knowledge sharing
    """
    
    def __init__(
        self,
        agent_slug: str,
        connector: HyperGraphRAGConnector,
    ):
        self.agent_slug = agent_slug
        self.connector = connector
        
    def recall(
        self,
        query: str,
        include_other_agents: bool = True,
        min_snr: float = 0.0,
    ) -> RetrievalResult:
        """Recall relevant knowledge for processing."""
        return self.connector.retrieve(
            query=query,
            agent_filter=None if include_other_agents else self.agent_slug,
            min_snr=min_snr,
        )
        
    def remember(
        self,
        content: str,
        claims: list[dict],
        confidence: float,
        session_id: Optional[str] = None,
    ) -> list[str]:
        """Store output as knowledge for future recall."""
        return self.connector.store_agent_output(
            agent_slug=self.agent_slug,
            content=content,
            claims=claims,
            confidence=confidence,
            session_id=session_id,
        )
        
    def verify(self, claim_text: str) -> tuple[bool, list[KnowledgeNode]]:
        """Verify a claim against existing knowledge."""
        return self.connector.verify_claim(claim_text, self.agent_slug)
        
    def get_my_knowledge(self, limit: int = 100) -> list[KnowledgeNode]:
        """Get knowledge I've previously produced."""
        return self.connector.get_agent_knowledge(self.agent_slug, limit)
        
    def share_with(
        self,
        target_agent: str,
        node_ids: list[str],
    ) -> None:
        """Share specific knowledge nodes with another agent."""
        for node_id in node_ids:
            if node_id in self.connector.nodes:
                edge = HyperEdge(
                    id=self.connector._generate_id(f"share:{node_id}:{target_agent}"),
                    type=EdgeType.DELEGATES_TO,
                    source_nodes=[node_id],
                    target_nodes=[],
                    metadata={
                        "from_agent": self.agent_slug,
                        "to_agent": target_agent,
                    },
                )
                self.connector.add_edge(edge)
