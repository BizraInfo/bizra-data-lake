# Phase 5: Post-Closure Optimization Track
## Goal: Gemma 4 routing, TurboQuant, warm/hot reflex promotion
### References: 00_master_spec.md §8, BYOB moat (Golden Gem #7)

---

## 1. Precondition

**This phase does NOT begin until all closure gates pass.**

```
ASSERT Phase 1 PERMIT achieved (SNR >= 0.85, Ihsan >= 0.95)
ASSERT Phase 2 proof bundle validates
ASSERT Phase 3 CI all-green
ASSERT Phase 4 shell consuming kernel truth
THEN unlock Phase 5
```

## 2. Track A: Gemma 4 Integration

### 2.1 Model Selection (RTX 4090, 16GB VRAM)

| Model | Params | VRAM | Multimodal | Fit |
|-------|--------|------|------------|-----|
| gemma-4-E2B-it | 5B | ~4GB | Any-to-Any | Agent tasks, fast |
| gemma-4-E4B-it | 8B | ~8GB | Any-to-Any | GoT synthesis, quality |
| gemma-4-26B-A4B-it | 27B (4B active) | ~6GB | Vision+Text | MoE, best quality/VRAM |
| gemma-4-31B-it | 33B | ~20GB | Vision+Text | Too large for 4090 alone |

**Recommendation**: `gemma-4-26B-A4B-it` (Mixture of Experts, 4B active params,
fits in 16GB VRAM) for GoT synthesis. `gemma-4-E2B-it` for fast agent tasks.

### 2.2 Pseudocode: BYOB Model Router

```
FUNCTION route_inference(task_type, complexity) -> ModelSelection:
    # BYOB: Bring Your Own Brain — model is NOT authority
    # Model choice is L3 (Experiments), not L0 (Law)

    available = discover_local_models()  # Ollama, LM Studio, GGUF

    IF task_type == "got_synthesis":
        # Quality-first: pick largest capable model
        prefer = [
            "gemma-4-26B-A4B",   # MoE, best quality/VRAM
            "gemma-4-E4B",       # 8B, solid
            "llama3.1:8b",       # Proven baseline
            "qwen2.5:3b",        # Minimum viable
        ]
    ELIF task_type == "agent_execution":
        # Speed-first: pick fastest adequate model
        prefer = [
            "gemma-4-E2B",       # 5B, fast, any-to-any
            "qwen2.5:3b",        # Known fast
            "phi3:mini",         # Lightweight
        ]
    ELIF task_type == "embedding":
        # Specialized: embedding models only
        prefer = [
            "nomic-embed-text",  # 768-dim, proven
            "all-MiniLM-L6-v2",  # SentenceTransformer
        ]
    ELIF task_type == "vision":
        # Multimodal: Gemma 4 excels here
        prefer = [
            "gemma-4-E4B",       # Any-to-Any (vision+audio+text)
            "gemma-4-26B-A4B",   # MoE multimodal
            "moondream:1.8b",    # Lightweight vision
        ]

    FOR candidate IN prefer:
        IF candidate IN available:
            RETURN ModelSelection(model=candidate, reason="preferred match")

    # Fallback: any available non-embedding model
    RETURN ModelSelection(model=available[0], reason="fallback")
```

### 2.3 Pseudocode: Ollama Model Management

```
PROCEDURE ensure_gemma4_available():
    # Check if Gemma 4 models are pulled
    available = ollama_list_models()

    targets = [
        "gemma4:26b-a4b-it",   # MoE for quality
        "gemma4:e2b-it",       # Fast agent tasks
    ]

    FOR target IN targets:
        IF target NOT IN available:
            ollama_pull(target)
            # Verify after pull
            ASSERT target IN ollama_list_models()

    # Warm the primary model
    ollama_generate(model=targets[0], prompt="warmup", max_tokens=1)
```

## 3. Track B: TurboQuant Compression

### 3.1 Application Areas

```
1. KV Cache Compression
   - Reduce VRAM usage during long GoT reasoning chains
   - Target: 4x compression with < 1% quality loss
   - Method: Mixed-precision quantization of attention KV pairs

2. Vector Store Compression
   - FAISS index: 102K vectors × 384 dims = ~150MB
   - Target: 4x compression → ~37MB (fits in L2 cache)
   - Method: Product quantization (PQ) or scalar quantization (SQ)

3. Retrieval-Quality Preservation
   - Compressed embeddings must maintain recall@10 >= 95%
   - Benchmark: query "Ihsan principle" against full vs compressed index
   - Metric: recall@10, MRR, latency_ms
```

### 3.2 Pseudocode: FAISS Compression Benchmark

```
FUNCTION benchmark_faiss_compression():
    # Load full-precision index
    index_fp32 = faiss.read_index("04_GOLD/faiss_index.bin")
    vectors = index_fp32.reconstruct_n(0, index_fp32.ntotal)

    # Test queries
    queries = load_benchmark_queries(k=100)

    # Baseline: full precision
    baseline_results = search(index_fp32, queries, k=10)
    baseline_latency = measure_latency(index_fp32, queries)

    # Variant 1: Scalar Quantization (SQ8)
    index_sq8 = faiss.IndexScalarQuantizer(384, faiss.ScalarQuantizer.QT_8bit)
    index_sq8.train(vectors)
    index_sq8.add(vectors)
    sq8_results = search(index_sq8, queries, k=10)
    sq8_recall = compute_recall(baseline_results, sq8_results, k=10)
    sq8_size = get_index_size(index_sq8)

    # Variant 2: Product Quantization (PQ32)
    index_pq = faiss.IndexPQ(384, 32, 8)
    index_pq.train(vectors)
    index_pq.add(vectors)
    pq_results = search(index_pq, queries, k=10)
    pq_recall = compute_recall(baseline_results, pq_results, k=10)
    pq_size = get_index_size(index_pq)

    # Report
    PRINT table:
        | Method | Size | Compression | Recall@10 | Latency |
        | FP32   | baseline_size | 1x | 1.000 | baseline_latency |
        | SQ8    | sq8_size | 4x | sq8_recall | sq8_latency |
        | PQ32   | pq_size | 12x | pq_recall | pq_latency |

    # Constitutional gate
    ASSERT sq8_recall >= 0.95, "SQ8 recall below 95% — reject compression"

    RETURN best_variant  # highest compression where recall >= 0.95
```

## 4. Track C: Warm/Hot Reflex Promotion

### 4.1 Path Stratification

```
COLD PATH (current — proof-heavy):
    Mission → Full PAT-7 → GoT synthesis → SNR/Ihsan scoring
    → Receipt → Evidence chain → Token economy
    Latency: 60-120s
    When: First time, novel queries, high-stakes

WARM PATH (target — bounded review):
    Mission → Cache check → Partial PAT (2-3 agents) → Quick score
    → Receipt (linked to cold-path lineage)
    Latency: 5-15s
    When: Similar query to a PERMIT-ed cold-path mission

HOT PATH (future — reflex only):
    Mission → Reflex cache hit → Instant response
    → Receipt (linked to warm-path lineage)
    Latency: <1s
    When: Exact match to receipted warm-path query
```

### 4.2 Pseudocode: Reflex Promotion

```
FUNCTION execute_with_path_selection(mission):
    query_hash = blake3(mission.query)

    # Check hot path (reflex)
    reflex = reflex_cache.get(query_hash)
    IF reflex AND reflex.lineage_valid():
        RETURN hot_path_response(reflex)

    # Check warm path (similar prior PERMIT)
    similar = find_similar_permit(mission.query, threshold=0.92)
    IF similar:
        RETURN warm_path_response(mission, similar)

    # Fall through to cold path
    RETURN cold_path_response(mission)


FUNCTION promote_to_reflex(mission_result):
    """After a warm-path PERMIT, promote to hot-path reflex."""
    IF mission_result.verdict == "PERMIT":
        IF mission_result.path == "warm":
            query_hash = blake3(mission_result.query)
            reflex_cache.store(
                key=query_hash,
                response=mission_result.response,
                lineage={
                    "cold_receipt": mission_result.cold_ancestor_hash,
                    "warm_receipt": mission_result.receipt_hash,
                    "promoted_at": utc_now(),
                    "ttl_hours": 24,  # reflexes expire
                },
            )
            # Constitutional constraint: reflex MUST have receipted lineage
            ASSERT reflex_cache.get(query_hash).lineage.cold_receipt IS NOT NONE
```

## 5. TDD Anchors

```python
# tests/test_post_closure.py

def test_model_router_prefers_gemma4_for_synthesis():
    """GoT synthesis routes to Gemma 4 MoE when available."""
    available = ["gemma-4-26B-A4B-it", "llama3.1:8b", "qwen2.5:3b"]
    selection = route_inference("got_synthesis", available)
    assert "gemma-4" in selection.model

def test_model_router_prefers_small_for_agents():
    """Agent execution routes to fast model."""
    available = ["gemma-4-E2B-it", "gemma-4-26B-A4B-it"]
    selection = route_inference("agent_execution", available)
    assert "E2B" in selection.model

def test_faiss_sq8_recall_above_95():
    """Scalar-quantized FAISS maintains 95%+ recall."""
    recall = benchmark_faiss_compression("sq8")
    assert recall >= 0.95

def test_reflex_requires_lineage():
    """Hot-path reflex cannot exist without cold-path receipt."""
    reflex = reflex_cache.get(query_hash)
    assert reflex.lineage.cold_receipt is not None

def test_warm_path_links_to_cold():
    """Warm-path receipt references its cold-path ancestor."""
    result = warm_path_response(mission, similar_permit)
    assert result.cold_ancestor_hash == similar_permit.receipt_hash

def test_reflex_expires():
    """Reflexes expire after TTL (no stale responses)."""
    store_reflex(query_hash, response, ttl_hours=0)  # expired
    assert reflex_cache.get(query_hash) is None
```

## 6. Gemma 4 Specific Notes

From the HuggingFace collection (2026-04-04):

- **Any-to-Any**: E2B and E4B handle vision, audio, AND text — enables multimodal missions
- **MoE architecture**: 26B-A4B has 27B total but only 4B active — fits 16GB VRAM
- **Instruction-tuned**: `-it` variants optimized for chat/instruction — ideal for GoT prompts
- **Open weights**: Apache 2.0 — compatible with BIZRA's sovereignty requirement (BYOB)

Integration path: Ollama will likely add Gemma 4 GGUF quantizations within days of release.
Monitor `ollama.com/library` for `gemma4` availability.

## 7. Validation Gates

```
Track A (Gemma 4):
  [ ] Model pulled and generates on Ollama
  [ ] GoT synthesis uses Gemma 4 for hypotheses
  [ ] SNR score with Gemma 4 >= score with llama3.1:8b

Track B (TurboQuant):
  [ ] FAISS SQ8 recall@10 >= 0.95
  [ ] Index size reduction >= 3x
  [ ] Search latency unchanged or improved

Track C (Warm/Hot):
  [ ] Warm path produces PERMIT with < 15s latency
  [ ] Hot path reflex response < 1s
  [ ] All reflexes have valid cold-path lineage
  [ ] Reflex TTL enforced (stale entries purged)
```

---

*Intelligence is replaceable; state, law, and receipts are sovereign.*
*BYOB means the system's value transcends any single model.*
*Gemma, Llama, Qwen, Mistral — all are welcome under the constitution.*
