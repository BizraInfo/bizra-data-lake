# Phase 31: Ecosystem Flywheel — MoE+HRM+HyperGraph Fusion + Memory Auto-Coder

> Wires the MoE router, HRM hierarchy, HyperGraph RAG, and Memory Auto-Coder into a unified cognitive fusion engine that drives the MMRPG-inspired ecosystem flywheel.

## Context

Phase 30 defined WHAT BIZRA is. Phase 31 specifies the three subsystems that remain partially implemented and wires them into the cognitive fusion that powers the ecosystem flywheel.

**Gaps identified in audit:**
1. HyperGraph RAG — current `core/graph/semantic_layer.py` uses pairwise edges only; needs hyperedge support
2. Memory Auto-Coder — current `core/autopoiesis/` does agent parameter evolution; needs code-aware memory synthesis
3. MoE+HRM fusion — MoE and HRM exist independently; need unified routing that feeds HRM level results into MoE expert selection

Standing on Giants: Berge (1973, hypergraph theory) + Vaswani (2017, attention is all you need) + Simon (1962, hierarchy) + Shannon (1948, entropy) + Kauffman (1993, adjacent possible) + Deming (1950, PDCA quality) + Al-Ghazali (1095, Ihsan)

## Package Structure

```
core/hypergraph/                          # NEW — HyperGraph RAG extension
  __init__.py                  # 30 lines
  hyperedge.py                 # 200 lines — hyperedge types + operations
  hypergraph_store.py          # 250 lines — in-memory hypergraph with query
  rag_fusion.py                # 200 lines — vector + keyword + graph retrieval fusion

core/cognitive_fusion/                    # NEW — MoE+HRM+HyperGraph unified engine
  __init__.py                  # 25 lines
  fusion_engine.py             # 300 lines — the unified cognitive router
  complexity_adapter.py        # 150 lines — adapts MoE complexity to HRM levels

core/memory_coder/                        # NEW — Memory-aware synthesis
  __init__.py                  # 25 lines
  memory_synthesizer.py        # 250 lines — distills memory into reusable patterns
  pattern_codebook.py          # 150 lines — codebook of learned patterns
```

Total new code: ~1,580 lines across 3 packages.

---

## 1. HyperGraph RAG

### HyperEdge Types

A hyperedge connects N >= 2 nodes simultaneously — unlike a pairwise edge which connects exactly 2.

```
ENUM HyperEdgeType:
  CONCEPT_CLUSTER      # N nodes share a concept (e.g., "autopoiesis" links 5 papers)
  CAUSAL_CHAIN         # Ordered sequence of cause→effect across N nodes
  CROSS_DOMAIN_BRIDGE  # N nodes from different domains share structural pattern
  TEMPORAL_COHORT      # N nodes active in same time window
  EVIDENCE_BUNDLE      # N evidence items supporting one claim

DATACLASS HyperEdge:
  edge_id: str                    # SHA-256 of sorted node_ids
  node_ids: FrozenSet[str]        # The connected nodes (N >= 2)
  edge_type: HyperEdgeType
  weight: float                   # Strength [0, 1]
  metadata: Dict[str, Any]        # Domain-specific attributes
  created_at: str                 # ISO 8601

  PROPERTY cardinality -> len(node_ids)
  PROPERTY is_pairwise -> cardinality == 2

DATACLASS HyperGraphNode:
  node_id: str
  label: str
  domain: str                     # e.g., "agriculture", "healthcare"
  embedding: Optional[List[float]]  # 768-dim vector for semantic search
  metadata: Dict[str, Any]
```

### HyperGraph Store

```
CLASS HyperGraphStore:
  """In-memory hypergraph with O(1) node lookup and O(k) edge traversal."""

  INIT():
    _nodes: Dict[str, HyperGraphNode] = {}
    _edges: Dict[str, HyperEdge] = {}
    _node_to_edges: Dict[str, Set[str]] = defaultdict(set)  # node_id → edge_ids

  METHOD add_node(node: HyperGraphNode):
    _nodes[node.node_id] = node

  METHOD add_hyperedge(node_ids: Set[str], edge_type, weight, metadata={}) -> HyperEdge:
    """Create a hyperedge connecting N nodes."""
    ASSERT len(node_ids) >= 2
    ASSERT all(nid IN _nodes FOR nid IN node_ids)
    edge_id = hex_digest(sorted(node_ids))
    edge = HyperEdge(edge_id, frozenset(node_ids), edge_type, weight, metadata)
    _edges[edge_id] = edge
    FOR nid IN node_ids:
      _node_to_edges[nid].add(edge_id)
    RETURN edge

  METHOD get_neighbors(node_id: str) -> Set[str]:
    """All nodes reachable via any hyperedge from node_id."""
    neighbors = set()
    FOR edge_id IN _node_to_edges[node_id]:
      edge = _edges[edge_id]
      neighbors.update(edge.node_ids - {node_id})
    RETURN neighbors

  METHOD get_hyperedges(node_id: str, edge_type: Optional = None) -> List[HyperEdge]:
    """All hyperedges containing node_id, optionally filtered by type."""
    edges = [_edges[eid] FOR eid IN _node_to_edges[node_id]]
    IF edge_type:
      edges = [e FOR e IN edges IF e.edge_type == edge_type]
    RETURN edges

  METHOD query_by_concept(concept_embedding: List[float], top_k: int = 10) -> List[HyperGraphNode]:
    """Semantic search over node embeddings (cosine similarity)."""
    # Uses HNSW if AgentDB available, else linear scan
    IF agent_db:
      RETURN agent_db.search(concept_embedding, top_k)
    ELSE:
      RETURN linear_cosine_scan(_nodes.values(), concept_embedding, top_k)

  METHOD get_cross_domain_bridges() -> List[HyperEdge]:
    """Find all CROSS_DOMAIN_BRIDGE hyperedges (highest SNR per NorthStar)."""
    RETURN [e FOR e IN _edges.values() IF e.edge_type == CROSS_DOMAIN_BRIDGE]

  PROPERTY node_count -> len(_nodes)
  PROPERTY edge_count -> len(_edges)
  PROPERTY mean_cardinality -> mean(e.cardinality FOR e IN _edges.values())
```

### RAG Fusion (Vector + Keyword + HyperGraph)

```
CLASS HyperGraphRAGFusion:
  """
  Triple-source retrieval fusion: vector similarity + keyword match + graph traversal.

  Standing on Giants: Shannon (dual representation) + Berge (hypergraph) + Vaswani (attention)
  """

  INIT(hypergraph: HyperGraphStore, agent_db: Optional[AgentDB]):
    WEIGHTS = {
      "vector": 0.40,       # Semantic similarity (HNSW)
      "keyword": 0.15,      # Lexical match (FTS5)
      "graph_hop": 0.25,    # Hyperedge traversal
      "recency": 0.10,      # Temporal decay
      "importance": 0.10,   # Node centrality
    }

  METHOD retrieve(query: str, query_embedding: List[float], top_k: int = 10) -> List[RetrievalResult]:
    """Fused retrieval across all three sources."""

    # Source 1: Vector similarity
    vector_results = agent_db.search(query_embedding, top_k * 2) IF agent_db ELSE []

    # Source 2: Keyword match
    keyword_results = agent_db.keyword_search(query, top_k * 2) IF agent_db ELSE []

    # Source 3: HyperGraph traversal
    # Find seed nodes from vector results, then expand via hyperedges
    seed_ids = [r.id FOR r IN vector_results[:5]]
    graph_results = []
    FOR seed_id IN seed_ids:
      neighbors = hypergraph.get_neighbors(seed_id)
      # Score neighbors by edge weight * edge cardinality bonus
      FOR neighbor_id IN neighbors:
        edges = hypergraph.get_hyperedges(neighbor_id)
        max_weight = max(e.weight FOR e IN edges) IF edges ELSE 0
        cardinality_bonus = log(mean(e.cardinality FOR e IN edges)) IF edges ELSE 0
        graph_results.append((neighbor_id, max_weight + cardinality_bonus * 0.1))

    # Fuse scores
    all_candidates = _merge_results(vector_results, keyword_results, graph_results)
    FOR candidate IN all_candidates:
      candidate.fused_score = (
        WEIGHTS["vector"] * candidate.vector_score +
        WEIGHTS["keyword"] * candidate.keyword_score +
        WEIGHTS["graph_hop"] * candidate.graph_score +
        WEIGHTS["recency"] * candidate.recency_score +
        WEIGHTS["importance"] * candidate.importance_score
      )

    RETURN sorted(all_candidates, key=fused_score, reverse=True)[:top_k]
```

---

## 2. Cognitive Fusion Engine (MoE + HRM + HyperGraph)

```
CLASS CognitiveFusionEngine:
  """
  Unified router: MoE complexity → HRM level → HyperGraph RAG → NorthStar gate.

  This is the cognitive core of Node0 — it decides HOW to think about a query,
  not just WHAT to retrieve.

  Standing on Giants: Vaswani (MoE) + Simon (hierarchy) + Shannon (SNR) + Besta (GoT)
  """

  INIT(
    moe_router: MoERouter,
    hrm_engine: HierarchicalReasoningModel,
    hypergraph_rag: HyperGraphRAGFusion,
    northstar_engine: NorthStarEngine,
  ):

  METHOD process(query: str, query_embedding: List[float], context: Dict = {}) -> FusionResult:
    """
    Full cognitive pipeline:
    1. MoE classifies complexity → selects expert tier
    2. HRM maps complexity to abstraction level → runs targeted cycle
    3. HyperGraph RAG retrieves context with graph expansion
    4. NorthStar gates the output quality
    """

    # Step 1: CLASSIFY (MoE)
    routing = moe_router.route(query, constraints=context.get("constraints", {}))
    complexity = routing.complexity_class    # TRIVIAL..FRONTIER
    expert_tier = routing.expert_tier        # NANO..FRONTIER

    # Step 2: MAP COMPLEXITY → HRM LEVEL
    level_map = {
      "TRIVIAL":  AbstractionLevel.PERCEPTUAL,     # L0
      "STANDARD": AbstractionLevel.OPERATIONAL,     # L1
      "COMPLEX":  AbstractionLevel.TACTICAL,        # L2
      "EXPERT":   AbstractionLevel.STRATEGIC,       # L3
      "FRONTIER": AbstractionLevel.META_COGNITIVE,  # LN
    }
    target_level = level_map[complexity]

    # Step 3: HRM CYCLE (focused on target level and above)
    observation = {
      "query": query,
      "complexity": complexity,
      "target_level": target_level.value,
      "context": context,
    }
    hrm_result = hrm_engine.run_cycle(observation)

    # Step 4: RETRIEVE (HyperGraph RAG with HRM-informed expansion)
    retrieval = hypergraph_rag.retrieve(
      query=query,
      query_embedding=query_embedding,
      top_k=_retrieval_depth(complexity),  # More retrieval for complex queries
    )

    # Step 5: GATE (NorthStar quality check)
    ns_observation = {
      "query_complexity": complexity,
      "hrm_compound_snr": hrm_result.compound_snr,
      "retrieval_count": len(retrieval),
      "observed_domains": list(set(r.domain FOR r IN retrieval)),
    }
    ns_report = northstar_engine.run_cycle(ns_observation)

    RETURN FusionResult(
      routing=routing,
      hrm_result=hrm_result,
      retrieval=retrieval,
      northstar_report=ns_report,
      target_level=target_level,
      snr_score=ns_report.unified_snr,
      ihsan_score=ns_report.ihsan_score,
      passes_gate=ns_report.passes_all_gates,
    )

  STATIC METHOD _retrieval_depth(complexity: str) -> int:
    """More complex queries need deeper retrieval."""
    RETURN {"TRIVIAL": 3, "STANDARD": 5, "COMPLEX": 10, "EXPERT": 20, "FRONTIER": 50}[complexity]


DATACLASS FusionResult:
  routing: RoutingDecision          # MoE output
  hrm_result: HRMCycleResult       # HRM cycle output
  retrieval: List[RetrievalResult]  # HyperGraph RAG results
  northstar_report: NorthStarReport # Quality gate
  target_level: AbstractionLevel    # Which HRM level was targeted
  snr_score: float
  ihsan_score: float
  passes_gate: bool

  PROPERTY is_elite -> snr_score >= 0.98 AND ihsan_score >= 0.99
  PROPERTY expert_tier -> routing.expert_tier
  PROPERTY compound_snr -> hrm_result.compound_snr
```

### Complexity Adapter

```
CLASS ComplexityAdapter:
  """
  Maps between MoE's 5 complexity tiers and HRM's 5 abstraction levels.

  The key insight (Golden Gem): complexity classification IS level selection.
  A TRIVIAL query only needs L0 (perceptual pattern match).
  A FRONTIER query needs LN (meta-cognitive self-reflection).

  Standing on Giants: Simon (bounded rationality) + Kauffman (adjacent possible)
  """

  LEVEL_TO_TIER = {
    AbstractionLevel.PERCEPTUAL:     ExpertTier.NANO,      # 0.5B params suffice
    AbstractionLevel.OPERATIONAL:    ExpertTier.EDGE,      # 1.5B
    AbstractionLevel.TACTICAL:       ExpertTier.LOCAL,     # 7B
    AbstractionLevel.STRATEGIC:      ExpertTier.POOL,      # 32B
    AbstractionLevel.META_COGNITIVE: ExpertTier.FRONTIER,  # 70B+
  }

  SNR_REQUIREMENTS = {
    AbstractionLevel.PERCEPTUAL:     0.85,  # UNIFIED_SNR_THRESHOLD
    AbstractionLevel.OPERATIONAL:    0.85,
    AbstractionLevel.TACTICAL:       0.90,  # SNR_THRESHOLD_T2_STANDARD
    AbstractionLevel.STRATEGIC:      0.95,  # SNR_THRESHOLD_T1_HIGH
    AbstractionLevel.META_COGNITIVE: 0.98,  # SNR_THRESHOLD_T0_ELITE
  }

  METHOD adapt(complexity: str) -> Tuple[AbstractionLevel, float]:
    """Returns (target_level, required_snr)."""
    level = COMPLEXITY_TO_LEVEL[complexity]
    required_snr = SNR_REQUIREMENTS[level]
    RETURN (level, required_snr)
```

---

## 3. Memory Auto-Coder (Pattern Synthesis)

The Memory Auto-Coder does NOT generate Python source code. It synthesizes **reusable cognitive patterns** from accumulated memory — distilling experience into a codebook of patterns that accelerate future reasoning.

```
CLASS MemorySynthesizer:
  """
  Distills raw memories into reusable cognitive patterns.

  Standing on Giants: Deming (PDCA) + Shannon (compression) + Kauffman (adjacent possible)

  The insight: repeated memory access patterns reveal underlying cognitive shortcuts.
  A pattern accessed 100 times with high SNR is worth codifying.
  """

  INIT(agent_db: AgentDB, codebook: PatternCodebook):

  METHOD synthesize_cycle(window_hours: int = 24) -> List[SynthesizedPattern]:
    """
    Run one synthesis cycle:
    1. Retrieve recent high-frequency memories
    2. Cluster by semantic similarity
    3. Extract common pattern from each cluster
    4. Validate pattern SNR against threshold
    5. Add to codebook if novel
    """
    # Step 1: Retrieve access-frequency-sorted memories
    recent = agent_db.query(
      time_window=window_hours,
      sort_by="access_count",
      min_access_count=5,     # Accessed at least 5 times
      min_snr=UNIFIED_SNR_THRESHOLD,
    )

    IF len(recent) < 3:
      RETURN []  # Not enough data to synthesize

    # Step 2: Cluster by semantic similarity
    clusters = _cluster_by_embedding(recent, min_cluster_size=3, threshold=0.85)

    new_patterns = []
    FOR cluster IN clusters:
      # Step 3: Extract pattern
      pattern = _extract_pattern(cluster)

      # Step 4: Validate SNR
      IF pattern.snr < UNIFIED_SNR_THRESHOLD:
        CONTINUE  # Below quality floor

      # Step 5: Check novelty against existing codebook
      IF NOT codebook.contains_similar(pattern, threshold=0.90):
        codebook.add(pattern)
        new_patterns.append(pattern)

    RETURN new_patterns

  METHOD _extract_pattern(cluster: List[MemoryRecord]) -> SynthesizedPattern:
    """Extract the common pattern from a cluster of similar memories."""
    # Use the centroid embedding as the pattern vector
    centroid = mean(m.embedding FOR m IN cluster)

    # Pattern label = most frequent keywords across cluster
    keywords = Counter(kw FOR m IN cluster FOR kw IN m.keywords).most_common(5)

    # Pattern SNR = mean SNR of cluster members
    snr = mean(m.snr FOR m IN cluster)

    RETURN SynthesizedPattern(
      pattern_id=hex_digest(centroid),
      embedding=centroid,
      keywords=[kw FOR kw, _ IN keywords],
      snr=snr,
      source_count=len(cluster),
      access_count=sum(m.access_count FOR m IN cluster),
    )


DATACLASS SynthesizedPattern:
  pattern_id: str
  embedding: List[float]
  keywords: List[str]
  snr: float
  source_count: int
  access_count: int
  created_at: str = now()

  PROPERTY is_strong -> snr >= 0.95 AND source_count >= 10


CLASS PatternCodebook:
  """
  A growing codebook of synthesized cognitive patterns.

  Patterns are indexed by embedding for fast retrieval.
  Strong patterns (SNR >= 0.95, 10+ sources) get priority in retrieval.
  """

  INIT(agent_db: Optional[AgentDB]):
    _patterns: Dict[str, SynthesizedPattern] = {}

  METHOD add(pattern: SynthesizedPattern):
    _patterns[pattern.pattern_id] = pattern
    # Also store in AgentDB for vector search
    IF agent_db:
      agent_db.store(
        content=json.dumps(pattern.keywords),
        embedding=pattern.embedding,
        metadata={"type": "codebook_pattern", "snr": pattern.snr},
      )

  METHOD lookup(query_embedding: List[float], top_k: int = 5) -> List[SynthesizedPattern]:
    """Find matching patterns by semantic similarity."""
    IF agent_db:
      results = agent_db.search(query_embedding, top_k, filter={"type": "codebook_pattern"})
      RETURN [_patterns[r.id] FOR r IN results IF r.id IN _patterns]
    ELSE:
      RETURN linear_cosine_scan(_patterns.values(), query_embedding, top_k)

  METHOD contains_similar(pattern: SynthesizedPattern, threshold: float) -> bool:
    """Check if a similar pattern already exists."""
    matches = lookup(pattern.embedding, top_k=1)
    RETURN len(matches) > 0 AND cosine_similarity(matches[0].embedding, pattern.embedding) >= threshold

  PROPERTY size -> len(_patterns)
  PROPERTY strong_patterns -> [p FOR p IN _patterns.values() IF p.is_strong]
```

---

## Integration: Cognitive Fusion → Ecosystem Flywheel

```
THE COMPLETE LOOP:

  User Query
    │
    ├──► CognitiveFusionEngine.process()
    │      ├── MoE classifies complexity
    │      ├── HRM runs targeted cognitive cycle
    │      ├── HyperGraph RAG retrieves + expands context
    │      └── NorthStar gates quality
    │
    ├──► SovereignRuntime.query()
    │      ├── GoT reasoning
    │      ├── Guardian Council review
    │      └── 6-Gate Chain validation
    │
    ├──► MemorySynthesizer (async, background)
    │      ├── Distills patterns from memory access
    │      └── Updates PatternCodebook
    │
    ├──► Proof of Impact
    │      ├── Contribution verification
    │      ├── Citation graph update
    │      └── Token distribution
    │
    └──► Autopoietic Loop (continuous)
           ├── Agent parameter evolution
           ├── Hypothesis testing
           └── Performance compounding
```

## TDD Anchors

```
TEST hyperedge_connects_n_nodes:
  store = HyperGraphStore()
  store.add_node(HyperGraphNode("a", "A", "domain1"))
  store.add_node(HyperGraphNode("b", "B", "domain1"))
  store.add_node(HyperGraphNode("c", "C", "domain2"))
  edge = store.add_hyperedge({"a", "b", "c"}, CONCEPT_CLUSTER, weight=0.9)
  ASSERT edge.cardinality == 3
  ASSERT store.get_neighbors("a") == {"b", "c"}

TEST hyperedge_requires_minimum_2_nodes:
  store = HyperGraphStore()
  store.add_node(HyperGraphNode("a", "A", "domain1"))
  WITH RAISES AssertionError:
    store.add_hyperedge({"a"}, CONCEPT_CLUSTER, weight=0.9)

TEST rag_fusion_triple_source:
  fusion = HyperGraphRAGFusion(store, agent_db)
  results = fusion.retrieve("autopoiesis", embedding=[0.1]*768, top_k=5)
  ASSERT len(results) <= 5
  ASSERT all(r.fused_score > 0 FOR r IN results)

TEST cognitive_fusion_maps_complexity_to_level:
  engine = CognitiveFusionEngine(moe, hrm, rag, northstar)
  result = engine.process("simple question", [0.1]*768)
  ASSERT result.target_level == AbstractionLevel.PERCEPTUAL  # TRIVIAL → L0

TEST cognitive_fusion_frontier_uses_ln:
  engine = CognitiveFusionEngine(moe, hrm, rag, northstar)
  result = engine.process("meta-cognitive self-reflection question", [0.5]*768)
  # Complex query should target higher levels
  ASSERT result.target_level.value >= AbstractionLevel.TACTICAL.value

TEST memory_synthesizer_extracts_patterns:
  synthesizer = MemorySynthesizer(agent_db, codebook)
  # Pre-populate agent_db with 10 similar memories
  FOR i IN range(10):
    agent_db.store(f"autopoiesis concept {i}", embedding=[0.9]*768)
  patterns = synthesizer.synthesize_cycle()
  ASSERT len(patterns) >= 1
  ASSERT patterns[0].source_count >= 3

TEST pattern_codebook_dedup:
  codebook = PatternCodebook(agent_db)
  p1 = SynthesizedPattern("p1", [0.9]*768, ["auto"], snr=0.95, source_count=5, access_count=50)
  codebook.add(p1)
  # Adding near-duplicate should be detected
  ASSERT codebook.contains_similar(p1, threshold=0.90) == True

TEST complexity_adapter_snr_gradient:
  adapter = ComplexityAdapter()
  level_l0, snr_l0 = adapter.adapt("TRIVIAL")
  level_ln, snr_ln = adapter.adapt("FRONTIER")
  ASSERT snr_l0 < snr_ln  # Higher complexity requires higher SNR
  ASSERT snr_l0 == 0.85    # UNIFIED_SNR_THRESHOLD
  ASSERT snr_ln == 0.98    # SNR_THRESHOLD_T0_ELITE
```

## Architectural Invariants

1. Hyperedges require >= 2 nodes — no self-loops
2. HyperGraph store is in-memory for v1; future persistence via AgentDB
3. RAG fusion weights sum to 1.0 — monoid property preserved
4. Cognitive fusion maps are bidirectional — MoE tier ↔ HRM level
5. Memory synthesizer only creates patterns from memories with SNR >= 0.85
6. Pattern codebook deduplicates at 0.90 cosine similarity threshold
7. ALL SNR thresholds imported from `core/integration/constants.py`
8. Complexity → retrieval depth scales: TRIVIAL=3, FRONTIER=50
