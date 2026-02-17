# Phase 33: HyperGraph RAG — N-ary Relations + Rust Crate + Retrieval Fusion

> **Status: COMPLETE** — See `phase_33_completion.md` for deviations from this spec. Python (13 tests) + Rust crate `bizra-hypergraph` (23 tests).

> Extends the pairwise semantic layer with hyperedge support, enabling concept clusters, causal chains, and cross-domain bridges. Adds a Rust crate for performance-critical graph traversal.

Standing on Giants: Berge (1973, hypergraph theory) + Vaswani (2017, attention as soft hyperedge) + Shannon (1948, information content of hyperedge membership) + Besta (2024, graph-of-thoughts as directed hypergraph)

## Context

`core/graph/semantic_layer.py` (908 lines) uses `DualOverlayGraph` with two `networkx.DiGraph` instances — all edges are binary (source, target). The Phase 31 spec identified hyperedges as the single largest structural gap in the knowledge graph layer. Real-world knowledge relationships are often n-ary: a research paper cites 5 sources (concept cluster), a causal chain links 4 events, a design pattern bridges 3 domains.

## Gaps Addressed

| Gap | Current State | Target State |
|-----|--------------|--------------|
| Edge cardinality | Binary only (source → target) | N-ary hyperedges (N >= 2 members) |
| Retrieval fusion | Vector-only in CognitiveFusion | Vector + keyword + graph (3-way) |
| Graph Rust crate | Absent | `bizra-hypergraph` with SIMD traversal |
| GoT reasoning | Implicit graph | Explicit hypergraph thought nodes |

## Package Structure

```
core/hypergraph/                          # NEW — HyperGraph extension
  __init__.py                  # 20 lines
  hyperedge.py                 # 180 lines — hyperedge types + operations
  store.py                     # 250 lines — in-memory hypergraph with query
  retrieval_fusion.py          # 200 lines — vector + keyword + graph fusion
  bridge.py                    # 100 lines — adapts DualOverlayGraph edges

bizra-omega/bizra-hypergraph/             # NEW — Rust crate
  Cargo.toml                   # Workspace member
  src/
    lib.rs                     # 50 lines — module exports
    hyperedge.rs               # 200 lines — typed hyperedge
    store.rs                   # 300 lines — adjacency-list hypergraph
    traversal.rs               # 250 lines — BFS/DFS/reachability with SIMD
```

Total new code: ~750 Python lines + ~800 Rust lines.

---

## 1. HyperEdge Types

```
ENUM HyperEdgeType:
  CONCEPT_CLUSTER      # N nodes share a concept (e.g., "autopoiesis" links 5 papers)
  CAUSAL_CHAIN         # Ordered sequence of cause→effect across N nodes
  CROSS_DOMAIN_BRIDGE  # N nodes from different domains share structural pattern
  TEMPORAL_WINDOW      # N events co-occur within a time window
  AGENT_COLLABORATION  # N agents participated in a joint decision

DATACLASS HyperEdge:
  """
  An n-ary relationship connecting 2+ nodes.

  Standing on Giants: Berge (hypergraph theory)
  Artifact: core/hypergraph/hyperedge.py
  """
  id: str                       # SHA-256 of sorted member IDs + edge_type
  members: FrozenSet[str]       # Node IDs (unordered for clusters, use rank for chains)
  edge_type: HyperEdgeType
  weight: float = 1.0           # Strength of the relationship
  metadata: dict = {}           # Type-specific attributes
  rank: Optional[List[str]] = None  # Ordered member list (for CAUSAL_CHAIN)
  created_at: datetime = NOW

  PROPERTY cardinality -> int:
    RETURN len(self.members)

  FUNCTION contains(self, node_id: str) -> bool:
    RETURN node_id IN self.members

  FUNCTION overlaps(self, other: HyperEdge) -> FrozenSet[str]:
    """Shared members between two hyperedges."""
    RETURN self.members & other.members
```

---

## 2. HyperGraph Store

```
CLASS HyperGraphStore:
  """
  In-memory hypergraph with incidence-list indexing.

  Data Structure:
    _edges: Dict[str, HyperEdge]           — edge_id → HyperEdge
    _incidence: Dict[str, Set[str]]         — node_id → Set[edge_id]
    _type_index: Dict[HyperEdgeType, Set[str]] — type → Set[edge_id]

  Standing on Giants: Berge (incidence structure)
  Artifact: core/hypergraph/store.py
  """

  FUNCTION add_edge(self, edge: HyperEdge) -> str:
    self._edges[edge.id] = edge
    FOR member IN edge.members:
      self._incidence.setdefault(member, set()).add(edge.id)
    self._type_index.setdefault(edge.edge_type, set()).add(edge.id)
    RETURN edge.id

  FUNCTION remove_edge(self, edge_id: str) -> None:
    edge = self._edges.pop(edge_id)
    FOR member IN edge.members:
      self._incidence[member].discard(edge_id)
    self._type_index[edge.edge_type].discard(edge_id)

  FUNCTION edges_of(self, node_id: str) -> List[HyperEdge]:
    """All hyperedges containing this node."""
    edge_ids = self._incidence.get(node_id, set())
    RETURN [self._edges[eid] for eid IN edge_ids]

  FUNCTION neighbors(self, node_id: str) -> Set[str]:
    """All nodes reachable via one hyperedge hop."""
    result = set()
    FOR edge IN self.edges_of(node_id):
      result |= edge.members
    result.discard(node_id)
    RETURN result

  FUNCTION query_by_type(self, edge_type: HyperEdgeType) -> List[HyperEdge]:
    edge_ids = self._type_index.get(edge_type, set())
    RETURN [self._edges[eid] for eid IN edge_ids]

  FUNCTION subgraph(self, node_ids: Set[str], max_hops: int = 1) -> HyperGraphStore:
    """Extract local neighborhood subgraph."""
    visited_nodes = set(node_ids)
    visited_edges = set()
    frontier = set(node_ids)

    FOR hop IN range(max_hops):
      next_frontier = set()
      FOR node IN frontier:
        FOR edge IN self.edges_of(node):
          IF edge.id NOT IN visited_edges:
            visited_edges.add(edge.id)
            next_frontier |= edge.members - visited_nodes
      visited_nodes |= next_frontier
      frontier = next_frontier

    sub = HyperGraphStore()
    FOR eid IN visited_edges:
      sub.add_edge(self._edges[eid])
    RETURN sub

  FUNCTION to_pairwise_projection(self) -> nx.DiGraph:
    """
    Project hypergraph to pairwise graph for compatibility with
    existing DualOverlayGraph consumers.

    Each hyperedge {A, B, C} generates edges: A→B, A→C, B→C
    (complete biclique of the member set).
    """
    G = nx.DiGraph()
    FOR edge IN self._edges.values():
      members = list(edge.members)
      FOR i IN range(len(members)):
        FOR j IN range(i+1, len(members)):
          G.add_edge(members[i], members[j],
                     hyperedge_id=edge.id,
                     edge_type=edge.edge_type.name,
                     weight=edge.weight)
    RETURN G

  PROPERTY stats -> dict:
    RETURN {
      "node_count": len(self._incidence),
      "edge_count": len(self._edges),
      "avg_cardinality": mean(e.cardinality for e in self._edges.values()) IF self._edges ELSE 0,
      "type_distribution": {t.name: len(ids) for t, ids in self._type_index.items()},
    }
```

---

## 3. Retrieval Fusion (Vector + Keyword + Graph)

```
CLASS RetrievalFusionEngine:
  """
  3-way retrieval combining vector similarity, keyword matching,
  and hypergraph neighborhood traversal.

  Standing on Giants: Shannon (information fusion) + Berge (graph retrieval)
  Artifact: core/hypergraph/retrieval_fusion.py
  """

  FUNCTION __init__(self, agent_db, hypergraph_store, keyword_index=None):
    self.agent_db = agent_db               # AgentDB with HNSW index
    self.hypergraph = hypergraph_store     # HyperGraphStore
    self.keyword_index = keyword_index     # Optional FTS5 index

  FUNCTION retrieve(self, query: str, query_embedding: List[float],
                    top_k: int = 10, weights: tuple = (0.5, 0.2, 0.3)
                    ) -> List[RetrievalResult]:
    """
    Fuse three retrieval signals:
      - Vector similarity (weight[0]): cosine distance via AgentDB
      - Keyword match (weight[1]): BM25/FTS5 term matching
      - Graph proximity (weight[2]): hyperedge co-membership

    Returns ranked list of RetrievalResult with fused score.
    """
    w_vec, w_kw, w_graph = weights

    # Signal 1: Vector retrieval
    vec_results = self.agent_db.search(query_embedding, top_k=top_k * 2)
    vec_scores = {r.id: r.score for r in vec_results}

    # Signal 2: Keyword retrieval
    kw_scores = {}
    IF self.keyword_index IS NOT None:
      kw_results = self.keyword_index.search(query, top_k=top_k * 2)
      kw_scores = {r.id: r.score for r in kw_results}

    # Signal 3: Graph proximity
    # Find nodes mentioned in top vector results, expand via hyperedges
    seed_nodes = set(list(vec_scores.keys())[:5])
    graph_neighbors = set()
    FOR node IN seed_nodes:
      graph_neighbors |= self.hypergraph.neighbors(node)
    graph_scores = {}
    FOR neighbor IN graph_neighbors:
      # Score = inverse of minimum hop distance (1-hop = 1.0, 2-hop = 0.5)
      graph_scores[neighbor] = 1.0  # All 1-hop neighbors get full graph score

    # Fuse: Reciprocal Rank Fusion variant
    all_ids = set(vec_scores) | set(kw_scores) | set(graph_scores)
    fused = []
    FOR doc_id IN all_ids:
      score = (
        w_vec * vec_scores.get(doc_id, 0.0) +
        w_kw * kw_scores.get(doc_id, 0.0) +
        w_graph * graph_scores.get(doc_id, 0.0)
      )
      fused.append(RetrievalResult(id=doc_id, score=score, signals={
        "vector": vec_scores.get(doc_id, 0.0),
        "keyword": kw_scores.get(doc_id, 0.0),
        "graph": graph_scores.get(doc_id, 0.0),
      }))

    fused.sort(key=lambda r: r.score, reverse=True)
    RETURN fused[:top_k]


DATACLASS RetrievalResult:
  id: str
  score: float
  signals: dict    # Per-signal breakdown
  content: Optional[str] = None
```

---

## 4. Rust Crate: bizra-hypergraph

```rust
// bizra-omega/bizra-hypergraph/src/lib.rs

pub mod hyperedge;
pub mod store;
pub mod traversal;

pub use hyperedge::{HyperEdge, HyperEdgeType};
pub use store::HyperGraphStore;
pub use traversal::{bfs_reachable, subgraph_extract};
```

```rust
// hyperedge.rs — Core types

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HyperEdgeType {
    ConceptCluster,
    CausalChain,
    CrossDomainBridge,
    TemporalWindow,
    AgentCollaboration,
}

#[derive(Clone, Debug)]
pub struct HyperEdge {
    pub id: [u8; 32],                    // BLAKE3 hash
    pub members: BTreeSet<NodeId>,       // Sorted for deterministic hashing
    pub edge_type: HyperEdgeType,
    pub weight: f64,
}

impl HyperEdge {
    pub fn new(members: impl IntoIterator<Item = NodeId>,
               edge_type: HyperEdgeType) -> Self {
        let members: BTreeSet<_> = members.into_iter().collect();
        let id = blake3_hash_members(&members, &edge_type);
        Self { id, members, edge_type, weight: 1.0 }
    }

    pub fn cardinality(&self) -> usize { self.members.len() }
    pub fn contains(&self, node: &NodeId) -> bool { self.members.contains(node) }
    pub fn overlap(&self, other: &Self) -> BTreeSet<NodeId> {
        &self.members & &other.members
    }
}
```

```rust
// traversal.rs — SIMD-friendly BFS

/// Collect all nodes reachable within max_hops via hyperedge traversal.
/// Uses bitset representation for visited tracking (SIMD-friendly).
pub fn bfs_reachable(
    store: &HyperGraphStore,
    seeds: &[NodeId],
    max_hops: usize,
) -> Vec<NodeId> {
    let mut visited = BitSet::with_capacity(store.node_count());
    let mut frontier: Vec<NodeId> = seeds.to_vec();

    for seed in seeds {
        if let Some(idx) = store.node_index(seed) {
            visited.insert(idx);
        }
    }

    for _hop in 0..max_hops {
        let mut next_frontier = Vec::new();
        for node in &frontier {
            for edge in store.edges_of(node) {
                for member in &edge.members {
                    if let Some(idx) = store.node_index(member) {
                        if !visited.contains(idx) {
                            visited.insert(idx);
                            next_frontier.push(member.clone());
                        }
                    }
                }
            }
        }
        if next_frontier.is_empty() { break; }
        frontier = next_frontier;
    }

    store.nodes_from_bitset(&visited)
}
```

---

## 5. TDD Anchors

```
TEST test_hyperedge_creation_deterministic_id:
  e1 = HyperEdge(members={"A", "B", "C"}, edge_type=CONCEPT_CLUSTER)
  e2 = HyperEdge(members={"C", "A", "B"}, edge_type=CONCEPT_CLUSTER)
  ASSERT e1.id == e2.id    # Order-independent

TEST test_hyperedge_cardinality:
  e = HyperEdge(members={"A", "B", "C", "D"}, edge_type=CAUSAL_CHAIN)
  ASSERT e.cardinality == 4

TEST test_store_add_and_query:
  store = HyperGraphStore()
  store.add_edge(HyperEdge({"A", "B", "C"}, CONCEPT_CLUSTER))
  ASSERT store.stats["edge_count"] == 1
  ASSERT store.stats["node_count"] == 3
  ASSERT "B" IN store.neighbors("A")

TEST test_store_edges_of:
  store = HyperGraphStore()
  store.add_edge(HyperEdge({"A", "B"}, CONCEPT_CLUSTER))
  store.add_edge(HyperEdge({"A", "C"}, CROSS_DOMAIN_BRIDGE))
  ASSERT len(store.edges_of("A")) == 2
  ASSERT len(store.edges_of("B")) == 1

TEST test_store_subgraph_1hop:
  store = HyperGraphStore()
  store.add_edge(HyperEdge({"A", "B", "C"}, CONCEPT_CLUSTER))
  store.add_edge(HyperEdge({"C", "D", "E"}, CAUSAL_CHAIN))
  sub = store.subgraph({"A"}, max_hops=1)
  ASSERT sub.stats["node_count"] == 3   # A, B, C (1-hop from A)

TEST test_pairwise_projection:
  store = HyperGraphStore()
  store.add_edge(HyperEdge({"A", "B", "C"}, CONCEPT_CLUSTER))
  G = store.to_pairwise_projection()
  ASSERT G.number_of_edges() == 3   # A-B, A-C, B-C

TEST test_retrieval_fusion_3way:
  fusion = RetrievalFusionEngine(mock_agentdb, mock_hypergraph, mock_fts5)
  results = fusion.retrieve("test query", [0.1]*768, top_k=5)
  ASSERT len(results) <= 5
  ASSERT all(r.score >= 0 for r in results)
  ASSERT "vector" IN results[0].signals

TEST test_retrieval_fusion_graph_boost:
  # Node in hyperedge with query-relevant node gets boosted
  ASSERT results_with_graph_signal[0].score > results_without[0].score

TEST test_rust_bfs_reachable:
  store = RustHyperGraphStore()
  store.add_edge(rust_edge(["A", "B", "C"]))
  store.add_edge(rust_edge(["C", "D"]))
  reachable = bfs_reachable(store, ["A"], max_hops=2)
  ASSERT "D" IN reachable

TEST test_rust_python_parity:
  # Same graph, same query — Rust and Python produce identical results
  ASSERT rust_neighbors("A") == python_neighbors("A")
```

## Success Criteria

| Metric | Target |
|--------|--------|
| Hyperedge support | N >= 2 members per edge, 5 edge types |
| 3-way retrieval fusion | Vector + keyword + graph signals combined |
| Rust crate | `cargo test` passing, SIMD BFS |
| Backward compatibility | `to_pairwise_projection()` feeds existing DualOverlayGraph |
| Test count | +10 Python, +5 Rust tests |
