# Phase 08 — Intelligence Pipeline: CognitiveFusion, HyperGraphRAG, MoE

> Source: Atlas v5.0 — Diagrams D17 (CognitiveFusion Pipeline), D18 (HyperGraphRAG), D22 (MoE + HRM)
> Status: SPECIFICATION SEALED | SNR: 0.95

---

## 1. Functional Requirements

### FR-080: CognitiveFusion Pipeline

End-to-end knowledge pipeline transforming a 1.3 TB heterogeneous corpus into a
grounded, query-ready knowledge substrate. Source material: sacred texts, 3000+
conversation transcripts, 150+ research documents, 144 code repositories, and
continuously-ingested user data. Pipeline is async, batch-oriented, and
fail-closed at every quality gate.

**Stage 1 -- Intelligent Chunking.** Adaptive segmentation using
sentence-boundary detection + semantic coherence scoring. Chunk sizes
dynamically adjust: 128-512 tokens for dense factual material, 512-2048 tokens
for narrative/conversational content. Overlap: 10% of chunk size. Metadata
preserved: source document ID, page/paragraph offset, creation timestamp,
privacy class (`PRIVACY_CLASSES`).

**Stage 2 -- Embedding.** `sentence-transformers/all-MiniLM-L6-v2` produces
384-dimensional dense vectors (`FAISS_EMBEDDING_DIM`). Batch size: 256 chunks.
Async pipeline: producer thread reads chunks, consumer thread encodes via GPU
(RTX 4090) or CPU fallback. Throughput target: 10,000 chunks/minute on GPU.
Output: `(chunk_id, embedding_vector, metadata)` triples written to
`04_GOLD/chunks.parquet`.

**Stage 3 -- Shannon Quality Gate.** Every chunk passes an entropy filter
before indexing. Shannon entropy `H(chunk) = -SUM(p_i * log2(p_i))` computed
over token frequency distribution. Gate: `H > 0.85` normalized (raw bits /
theoretical maximum). Chunks below threshold are quarantined to
`99_QUARANTINE/low_entropy/` with a diagnostic record. This eliminates
boilerplate, repeated disclaimers, and near-duplicate content that would pollute
retrieval.

**Stage 4 -- NTU Temporal Beliefs.** Non-monotonic Temporal Update layer
maintains a belief store where each fact is a triple `(claim, evidence_set,
confidence)`. Confidence incorporates:
- *Evidence weight:* number and quality of supporting chunks.
- *Temporal decay:* exponential decay `lambda = 0.02/day` (slower than
  HyperGraphRAG edge decay to preserve long-term knowledge).
- *Revision:* contradictory evidence reduces confidence via Bayesian update;
  when confidence falls below `CONFIDENCE_MINIMUM` (0.50), the belief is
  retracted. Retraction is logged to the evidence ledger for auditability.

**Stage 5 -- Knowledge Graph Construction.** Four node types and three
relationship classes:
- *Entities:* named objects extracted via NER (spaCy + LLM fallback).
- *Relations:* binary edges `(subject, predicate, object)` with provenance.
- *Hyperedges:* N-ary shared-context groupings (see FR-081).
- *Communities:* Louvain-detected clusters with LLM-generated summaries.

Entity dedup: `BLAKE3(normalized_name + type)` + fuzzy matching (Levenshtein
<= 2). Graph: `04_GOLD/knowledge_graph.jsonl` with Merkle root integrity.

**Stage 6 -- Vector Index.** Dual-index: FAISS IVF-PQ (`nlist=1024`,
`nprobe=32`, `m=48`) at `FAISS_INDEX_PATH` for sub-ms ANN; Chroma for
metadata-filtered retrieval. Hybrid search: cosine similarity + BM25 via
reciprocal rank fusion (k=60). Top-K (`FAISS_DEFAULT_TOP_K` = 10) reranked
by cross-encoder (MiniLM-L12-v2-msmarco). Context assembly: concatenate up
to model context limit (4096 tokens), respecting chunk boundaries.

**Stage 7 -- Agent Query Interface.** `CognitiveFusion.query(prompt)` merges
retrieval, NTU beliefs, and graph context into a grounded response with
citations `[source_id:chunk_offset]`. Ihsan feedback loop (FR-025) feeds back
into NTU confidence and retrieval ranking. Acceptance: `UNIFIED_IHSAN_THRESHOLD`
(0.95).

### FR-081: HyperGraphRAG

Extension of the knowledge graph (FR-080 Stage 5) beyond binary relations to
N-ary hyperedges, enabling retrieval over shared contexts that traditional
vector search cannot represent.

**Node Types.**
- *Entity nodes:* people, organizations, technologies, concepts.
- *Document nodes:* source artifacts with metadata.
- *Concept nodes:* abstract topics derived from LLM extraction.

**Edge Types.**
- *Binary edges:* standard `(subject, predicate, object)`.
- *Hyperedges:* N-ary, connecting 3+ nodes that share a context (e.g., a
  research paper references 5 concepts in a single argument). Stored as edge
  sets with a shared context embedding.
- *Edge weights:* `w = frequency * recency * impact`. Frequency: co-occurrence.
  Recency: decay `lambda = 0.05/day`. Impact: citation/usage from evidence ledger.

**Community Detection.** Louvain (resolution 1.0). Each community gets an LLM
summary (max 500 tokens). Hierarchical: recursive Louvain at 0.5 (coarse) and
2.0 (fine). Bridge nodes (top 5% betweenness centrality) tagged for cross-community
retrieval.

**4 Retrieval Strategies:**

| Strategy   | Mechanism                                  | Use Case                          |
|------------|-------------------------------------------|-----------------------------------|
| Local      | Entity walk: BFS from seed entity, depth 2 | "Tell me about X"               |
| Global     | Community summaries, top-K by relevance    | "What themes emerge across..."   |
| Drift      | Local seed, expand to global via bridges   | Exploratory, "start here, go wide"|
| Multi-Hop  | Relation chain traversal, max 4 hops       | "How does A connect to B?"       |

**Temporal Layer.** Every edge carries `created_at`. Decay:
`w_eff = w_base * exp(-0.05 * age_days)`. High-access edges (3+ queries in
7 days) refresh decay. Monthly prune: `w_eff < 0.01` archived to
`99_ARCHIVE/graph_edges/`. Pruned edges remain in Merkle tree for provenance.

### FR-082: Mixture of Experts + Hierarchical Resource Manager

**5 Ollama Expert Roster:**

| Expert        | Model              | Specialty           | VRAM   | Latency Target |
|---------------|--------------------|---------------------|--------|----------------|
| CodeExpert    | DeepSeek-Coder     | Code gen/review     | 8 GB   | 2000 ms        |
| ReasonExpert  | Llama-3 70B (q4)   | Multi-step logic    | 40 GB  | 5000 ms        |
| CreativeExpert| Mistral-Large      | Writing, ideation   | 24 GB  | 3000 ms        |
| FastExpert    | Phi-3 / Gemma 2B   | Trivial / triage    | 2 GB   | 500 ms         |
| VisionExpert  | LLaVA              | Image understanding | 8 GB   | 3000 ms        |

Router classifies tasks by type (code, reasoning, creative, fast, multimodal)
and complexity via entropy router (FR-020) + keyword heuristics. Ties: lowest load.

**Hierarchical Resource Manager (HRM).**
- *SEED Budget:* cost = `SEED_COMPUTE_HOUR_PEG` (1.0) / 3600 * vram_gb * secs.
- *Priority Queue:* urgency (user=1.0, proactive=0.5, bg=0.2) * impact.
  Depth: `ACTION_BUS_MAX_CONCURRENT` (10) active, 100 waiting.
- *GPU Slicing:* round-robin 200ms slices. Preempt low-priority within 50ms.
- *Throttle:* `ACTION_BUS_MAX_PER_HOUR` (100). Burst: 20 in 10s window.
  Exponential backoff on VRAM exhaustion.

**Confidence Cascade.** (1) Route to FastExpert (500ms). (2) Score SNR.
(3) If >= `CONFIDENCE_HIGH` (0.95): return. (4) If >= `CONFIDENCE_MEDIUM`
(0.85): return tagged. (5) If below: escalate to domain expert. (6) If domain
expert also below: ensemble vote (2+ experts), majority wins. (7) If ensemble
fails: fail closed, error receipt.

**KV-Cache Management.** Shared prefix cache (system prompts, ~2000 tokens
saved/call). Per-user context: last 3 turns, keyed by
`BLAKE3(user_id + session_id)`. LRU eviction: capacity = VRAM - 4 GB headroom;
high-Ihsan entries persist longer. Pre-warm: `AutoModelRouter.preload_mission_fleet()`
loads predicted experts into VRAM before first call.

---

## 2. Edge Cases

**EC-080: Embedding Service Down.** Sentence-transformers model fails to load
(CUDA OOM, corrupted weights). Mitigation: (1) CPU fallback with reduced batch
size (32). (2) If CPU also fails, queue chunks to `99_QUARANTINE/pending_embed/`
with retry-after metadata. (3) Alert via event bus (`bizra-hooks`). No data
loss: chunks are idempotently re-processable via `chunk_id`.

**EC-081: All Experts Busy.** Every slot in the priority queue is occupied and
a new high-priority task arrives. Mitigation: (1) Preempt the lowest-priority
background task (save KV-cache state to disk). (2) If no preemptable tasks,
return `503 SERVICE_BUSY` with estimated wait time. (3) Exponential backoff on
caller side (200ms, 400ms, 800ms, max 5s). Never silently drop a request.

**EC-082: Hyperedge Cycle.** Circular hyperedge references during graph
traversal cause infinite loops. Mitigation: visited-set tracking in all
traversal functions (Local, Drift, Multi-Hop). Max traversal depth: 4 hops
(Multi-Hop) or 2 hops (Local). Cycle detection via `BLAKE3(path_node_ids)`;
if hash repeats, terminate branch and return partial results with
`cycle_detected=true` flag.

**EC-083: Stale Community Summaries.** Entities change but summaries lag.
Mitigation: (1) Dirty-flag on member add/remove/weight change > 20%.
(2) Background regeneration every `TIMESCALE_T3_CONSOLIDATION_HOURS` (24h).
(3) Query-time: if `last_summarized < newest_member_edge`, attach
`summary_stale=true` warning.

**EC-084: NTU Belief Contradiction.** Two high-confidence claims conflict.
Mitigation: (1) Bayesian revision adjusts posteriors by evidence strength.
(2) If both remain above `CONFIDENCE_MEDIUM` (0.85), flag `CONTESTED` and
present both with citations. (3) User resolution creates high-weight evidence
to break tie. (4) Never silently discard a well-evidenced belief.

---

## 3. Pseudocode

### 3.1 cognitive_fusion_query(query)

```
FUNCTION cognitive_fusion_query(query: str, user_id: str, config: FusionConfig) -> FusionResult:
    # Stage A: Embed the query
    q_embedding = embed(query, model="all-MiniLM-L6-v2")  # 384-dim
    IF q_embedding IS None:
        RETURN FusionResult(ERROR, "embedding_service_unavailable")

    # Stage B: Hybrid retrieval (vector + keyword)
    vector_hits = faiss_index.search(q_embedding, top_k=config.top_k * 2)  # Over-retrieve for rerank
    bm25_hits = bm25_index.search(query, top_k=config.top_k * 2)
    fused = reciprocal_rank_fusion(vector_hits, bm25_hits, k=60)

    # Stage C: Cross-encoder rerank
    reranked = cross_encoder.rerank(query, fused[:config.top_k * 3])
    top_chunks = reranked[:config.top_k]
    IF len(top_chunks) == 0:
        RETURN FusionResult(NO_RESULTS, "no_relevant_chunks")

    # Stage D: Similarity floor
    top_chunks = [c FOR c IN top_chunks IF c.score >= FAISS_SIMILARITY_FLOOR]
    IF len(top_chunks) == 0:
        RETURN FusionResult(NO_RESULTS, "all_below_similarity_floor")

    # Stage E: NTU belief augmentation
    beliefs = ntu_store.query_relevant(query, top_k=5)
    active_beliefs = [b FOR b IN beliefs IF b.confidence >= CONFIDENCE_MINIMUM]
    contested = [b FOR b IN beliefs IF b.status == CONTESTED]

    # Stage F: Knowledge graph context
    seed_entities = ner_extract(query)
    graph_context = hypergraph.local_retrieve(seed_entities, depth=2)

    # Stage G: Context window assembly
    context_budget = config.context_limit  # default 4096 tokens
    assembled = assemble_context(top_chunks, active_beliefs, graph_context, budget=context_budget)

    # Stage H: Grounded generation via MoE
    task = InferenceTask(query=query, context=assembled, require_citations=True)
    response = moe_route(task)
    IF response.ihsan < UNIFIED_IHSAN_THRESHOLD:
        RETURN FusionResult(REJECTED, "ihsan_below_threshold", ihsan=response.ihsan)

    # Stage I: Citation verification
    FOR citation IN response.citations:
        IF NOT verify_citation(citation, top_chunks, active_beliefs):
            response.citations.remove(citation)
            response.confidence -= 0.05

    # Stage J: Feedback registration
    receipt = create_receipt(query, response, user_id)
    evidence_ledger.append(receipt)
    RETURN FusionResult(SUCCESS, response=response, citations=response.citations,
                        contested_beliefs=contested, receipt_id=receipt.id)
```

### 3.2 hypergraph_retrieve(query, strategy)

```
FUNCTION hypergraph_retrieve(query: str, strategy: RetrievalStrategy, config: HyperGraphConfig) -> RetrievalResult:
    MATCH strategy:
        CASE Local:
            seed_entities = ner_extract(query)
            IF len(seed_entities) == 0:
                RETURN RetrievalResult(EMPTY, "no_entities_extracted")
            visited = SET()
            results = []
            FOR entity IN seed_entities:
                node = graph.find_node(entity)
                IF node IS None: CONTINUE
                neighbors = bfs(node, depth=2, visited=visited, max_results=config.top_k)
                results.extend(neighbors)
            RETURN RetrievalResult(SUCCESS, nodes=deduplicate(results)[:config.top_k])

        CASE Global:
            community_summaries = graph.get_community_summaries(level=1)
            scored = [(c, semantic_similarity(query, c.summary)) FOR c IN community_summaries]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_communities = scored[:config.top_k]
            results = []
            FOR community, score IN top_communities:
                IF community.summary_stale:
                    community.summary = regenerate_summary(community)
                results.append(CommunityResult(community, score))
            RETURN RetrievalResult(SUCCESS, communities=results)

        CASE Drift:
            # Start local, expand to global via bridge nodes
            local_result = hypergraph_retrieve(query, Local, config)
            IF len(local_result.nodes) == 0:
                RETURN hypergraph_retrieve(query, Global, config)
            bridge_nodes = [n FOR n IN local_result.nodes IF n.is_bridge]
            IF len(bridge_nodes) == 0:
                bridge_nodes = graph.find_nearest_bridges(local_result.nodes, k=3)
            expanded = []
            FOR bridge IN bridge_nodes:
                cross_communities = graph.communities_of(bridge)
                FOR comm IN cross_communities:
                    expanded.extend(comm.top_entities(k=5))
            RETURN RetrievalResult(SUCCESS, nodes=local_result.nodes,
                                   expanded=deduplicate(expanded)[:config.top_k])

        CASE MultiHop:
            entities = ner_extract(query)
            IF len(entities) < 2:
                RETURN RetrievalResult(EMPTY, "multi_hop_requires_two_entities")
            source = graph.find_node(entities[0])
            target = graph.find_node(entities[-1])
            IF source IS None OR target IS None:
                RETURN RetrievalResult(EMPTY, "entity_not_found")
            paths = bfs_paths(source, target, max_depth=4, visited=SET())
            IF len(paths) == 0:
                RETURN RetrievalResult(EMPTY, "no_path_found")
            scored_paths = [(p, path_weight(p)) FOR p IN paths]
            scored_paths.sort(key=lambda x: x[1], reverse=True)
            RETURN RetrievalResult(SUCCESS, paths=scored_paths[:config.max_paths])
```

### 3.3 moe_route(task)

```
FUNCTION moe_route(task: InferenceTask, hrm: HierarchicalResourceManager) -> InferenceResult:
    # Classify task type and complexity
    task_type = classify_task_type(task.query)  # code | reasoning | creative | fast | multimodal
    complexity = entropy_route(task.query).tier  # from FR-020

    # Select primary expert
    expert_map = {
        "code":       CodeExpert,
        "reasoning":  ReasonExpert,
        "creative":   CreativeExpert,
        "fast":       FastExpert,
        "multimodal": VisionExpert,
    }
    primary = expert_map.get(task_type, FastExpert)

    # Check SEED budget
    estimated_cost = hrm.estimate_cost(primary.vram_gb, primary.latency_target_ms)
    IF NOT hrm.can_afford(task.user_id, estimated_cost):
        RETURN InferenceResult(REJECTED, "seed_budget_exhausted")

    # Enqueue with priority
    priority = task.urgency * task.impact
    slot = hrm.enqueue(task, primary, priority)
    IF slot IS None:
        # EC-081: all busy — attempt preemption
        preempted = hrm.preempt_lowest(priority)
        IF preempted IS None:
            RETURN InferenceResult(SERVICE_BUSY, estimated_wait=hrm.estimated_wait_ms())
        slot = hrm.enqueue(task, primary, priority)

    # Execute via confidence cascade
    result = confidence_cascade(task, primary, hrm)

    # Debit SEED
    actual_cost = hrm.compute_actual_cost(primary.vram_gb, result.latency_ms)
    hrm.debit_seed(task.user_id, actual_cost)

    RETURN result
```

### 3.4 confidence_cascade(task, initial_expert)

```
FUNCTION confidence_cascade(task: InferenceTask, initial_expert: Expert,
                            hrm: HierarchicalResourceManager) -> InferenceResult:
    # Step 1: Try fast expert first (unless already routed to fast)
    IF initial_expert != FastExpert AND task.complexity IN (TRIVIAL, SIMPLE):
        fast_result = FastExpert.generate(task, timeout_ms=500)
        IF fast_result IS NOT None:
            fast_snr = snr_engine.calculate(task.query, fast_result.content)
            IF fast_snr >= CONFIDENCE_HIGH:
                RETURN InferenceResult(SUCCESS, fast_result, confidence="high", snr=fast_snr)
            IF fast_snr >= CONFIDENCE_MEDIUM:
                RETURN InferenceResult(SUCCESS, fast_result, confidence="medium", snr=fast_snr)

    # Step 2: Domain expert
    domain_result = initial_expert.generate(task, timeout_ms=initial_expert.latency_target_ms)
    IF domain_result IS None: RETURN InferenceResult(ERROR, "expert_timeout")
    domain_snr = snr_engine.calculate(task.query, domain_result.content)
    IF domain_snr >= CONFIDENCE_MEDIUM:
        tag = "high" IF domain_snr >= CONFIDENCE_HIGH ELSE "medium"
        RETURN InferenceResult(SUCCESS, domain_result, confidence=tag, snr=domain_snr)

    # Step 3: Ensemble vote (2+ experts)
    candidates = [domain_result]
    FOR peer IN select_ensemble_peers(initial_expert, exclude=[FastExpert], max=2):
        IF hrm.can_afford_concurrent(peer):
            peer_result = peer.generate(task, timeout_ms=peer.latency_target_ms)
            IF peer_result IS NOT None: candidates.append(peer_result)

    IF len(candidates) < 2:
        RETURN InferenceResult(SUCCESS, domain_result, confidence="low", snr=domain_snr)

    scored = [(c, snr_engine.calculate(task.query, c.content)) FOR c IN candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0]
    IF best[1] >= CONFIDENCE_LOW:
        RETURN InferenceResult(SUCCESS, best[0], confidence="ensemble", snr=best[1])

    # Step 4: Fail closed
    RETURN InferenceResult(REJECTED, "ensemble_confidence_insufficient", best_snr=best[1])
```

---

## 4. TDD Anchors

```
TEST fusion_query_returns_grounded_citations:
    index = build_test_index([chunk("Python is great", src="doc1"),
                              chunk("Rust is fast", src="doc2")])
    result = cognitive_fusion_query("Compare Python and Rust", "user-1", default_config)
    ASSERT result.status == SUCCESS
    ASSERT len(result.citations) >= 1
    ASSERT ALL(c.source_id IN ("doc1", "doc2") FOR c IN result.citations)

TEST fusion_query_rejects_below_ihsan:
    mock_moe_response(ihsan=0.88)  # Below UNIFIED_IHSAN_THRESHOLD (0.95)
    result = cognitive_fusion_query("Tell me about BIZRA", "user-1", default_config)
    ASSERT result.status == REJECTED AND "ihsan" IN result.reason

TEST hypergraph_local_retrieves_entity_neighborhood:
    graph = build_test_graph(entities=["BIZRA", "Rust", "Python"],
                             edges=[("BIZRA", "uses", "Rust"), ("BIZRA", "uses", "Python")])
    result = hypergraph_retrieve("What does BIZRA use?", Local, default_config)
    ASSERT result.status == SUCCESS
    ASSERT "Rust" IN [n.name FOR n IN result.nodes]
    ASSERT "Python" IN [n.name FOR n IN result.nodes]

TEST hypergraph_multihop_finds_path:
    graph = build_test_graph(entities=["A", "B", "C"],
                             edges=[("A", "relates_to", "B"), ("B", "relates_to", "C")])
    result = hypergraph_retrieve("How does A connect to C?", MultiHop, default_config)
    ASSERT result.status == SUCCESS AND len(result.paths) >= 1
    ASSERT result.paths[0].hops <= 4

TEST hypergraph_cycle_detection_terminates:
    graph = build_test_graph(entities=["X", "Y", "Z"],
                             edges=[("X", "r", "Y"), ("Y", "r", "Z"), ("Z", "r", "X")])
    result = hypergraph_retrieve("Trace from X", Local, HyperGraphConfig(top_k=10))
    ASSERT result.status == SUCCESS  # Must not hang
    ASSERT len(result.nodes) <= 10

TEST moe_routes_code_to_code_expert:
    task = InferenceTask(query="Write a Python function to sort a list", type_hint="code")
    result = moe_route(task, test_hrm)
    ASSERT result.expert == "CodeExpert" OR result.expert == "fast"
    ASSERT result.status == SUCCESS

TEST confidence_cascade_escalates_low_confidence:
    mock_fast_expert(snr=0.60)   # Below CONFIDENCE_MEDIUM (0.85)
    mock_reason_expert(snr=0.92) # Above CONFIDENCE_MEDIUM
    task = InferenceTask(query="Explain quantum entanglement in detail")
    result = confidence_cascade(task, FastExpert, test_hrm)
    # Should have escalated past fast expert
    ASSERT result.confidence IN ("medium", "high")
    ASSERT result.snr >= CONFIDENCE_MEDIUM

TEST confidence_cascade_fails_closed_on_all_low:
    mock_all_experts(snr=0.40)  # All below CONFIDENCE_LOW (0.70)
    task = InferenceTask(query="Obscure untestable question")
    result = confidence_cascade(task, FastExpert, test_hrm)
    ASSERT result.status == REJECTED AND "ensemble_confidence_insufficient" IN result.reason
```

---

## 5. Cross-References

### Python Modules

- `core/iaas/snr_v2_adapter.py` -- `SNRv2Adapter`. Shannon quality gate (FR-080) and confidence cascade (FR-082).
- `core/iaas/snr_v2.py` -- Core SNR engine. `core/iaas/renyi_entropy.py` -- Renyi entropy scoring.
- `core/inference/gateway.py` -- `InferenceGateway`, circuit breaker, rate limiter. All expert calls route here.
- `core/inference/model_routing.py` -- `DEFAULT_MODEL_ROUTING`, `resolve_model_for_agent()`. Task-to-model mapping.
- `core/inference/auto_model_router.py` -- `AutoModelRouter`, VRAM pre-loading. KV-cache pre-warm (FR-082).
- `core/inference/multi_model_manager.py` -- Multi-model orchestration for ensemble routing.
- `core/reasoning/entropy_router.py` -- `EntropyRouter`, `RoutingDecision`. MoE router input (FR-082).
- `core/reasoning/graph_core.py` -- `GraphOfThoughts`. `graph_search.py` -- traversal. `graph_types.py` -- Node/Edge/HyperEdge.
- `core/living_memory/core.py` -- `MemoryType`, `MemoryState`, HHMM promotion. NTU maps to memory lifecycle.
- `core/proof_engine/evidence_ledger.py` -- `EvidenceLedger`. NTU belief revision audit trail.
- `core/integration/constants.py` -- All thresholds: `UNIFIED_IHSAN_THRESHOLD` (0.95), `CONFIDENCE_*` (0.95/0.85/0.70/0.50), `FAISS_EMBEDDING_DIM` (384), `FAISS_DEFAULT_TOP_K` (10), `FAISS_SIMILARITY_FLOOR` (0.35), `ACTION_BUS_MAX_*`, `SEED_COMPUTE_HOUR_PEG` (1.0), `PRIVACY_CLASSES`.

### Rust Crates

- `bizra-omega/bizra-inference/` -- `InferenceGateway`, `ModelSelector`, `ModelTier`, `TaskComplexity`. LMStudio/Ollama backends. Rust-side MoE routing.
- `bizra-omega/bizra-memory/` -- `BizraMemory`, `MemoryPipeline`, `SynthesisEngine`, `InMemoryStore`. `bridge.rs` -- `RuleExtractor`, `Searcher` (FFI to Python vector search).
- `bizra-omega/bizra-hooks/` -- Event bus (8 shards, FNV-1a). Embedding failure (EC-080) and saturation alerts (EC-081).
- `bizra-omega/bizra-core/` -- `IHSAN_THRESHOLD` (0.95), `SNR_THRESHOLD` (0.85). Constitutional constants.
- `bizra-omega/bizra-agent/src/reflex_cache.rs` -- `ReflexCache`. High-Ihsan fusion results precipitate via G.R.A.S.P. (Phase 02).

### Atlas v5 Phases

- Phase 00 -- FR-001/003: System architecture, data pipeline (00_INTAKE through 04_GOLD)
- Phase 01 -- FR-010/013: Node identity, BLAKE3 hashing for graph integrity
- Phase 02 -- FR-020: Entropy Router (MoE input); FR-021: Diffusion Cognition (GoT candidates); FR-023: G.R.A.S.P. (reflex precipitation); FR-025: Ihsan Feedback Loop
- Phase 03 -- FR-030: PAT-7 agents (expert-to-agent mapping via model_routing.py)
- Phase 05 -- FR-050/051: BlockGraph + PoI (NTU belief provenance and evidence weight)
- Phase 06 -- FR-062: FATE Gate (all MoE responses gated); FR-065: Governance (community summaries)
- Phase 07 -- FR-074: Federated Learning (expert gradients); FR-076: Reflex Diffusion (fusion capsules)

### Standing on Giants

- Shannon (1948): Information entropy -- quality gate, SNR, confidence scoring
- Bordes et al. (2013): TransE -- knowledge graph embeddings
- Blondel et al. (2008): Louvain -- community detection algorithm
- Shazeer et al. (2017): Mixture of Experts -- sparse expert routing
- Robertson & Zaragoza (2009): BM25 -- probabilistic keyword retrieval
- Johnson et al. (2021): FAISS -- billion-scale similarity search
- Nogueira & Cho (2019): Cross-encoder reranking
- Reimers & Gurevych (2019): Sentence-BERT -- all-MiniLM-L6-v2
- Al-Ghazali (1095): Ihsan -- excellence as the quality floor
- Kahneman (2011): System 1/2 -- fast/slow inference cascade
- Nygard (2007): Release It! -- circuit breaker resilience
