# Phase 32: Live Cognition — Real Embeddings + NTU Wiring + Quality Gates

> **Status: COMPLETED** — Committed `d4347d4` + `9902f87` (2026-02-17). See `phase_32_completion.md` for deviations from this spec.

> Replaces dummy embeddings with live sentence-transformer inference, wires the NeuroTemporal Unit into the cognitive fusion pipeline, and adds embedding-quality gates to prevent silent degradation.

Standing on Giants: Reimers & Gurevych (2019, sentence-BERT) + Shannon (1948, information entropy as quality signal) + Takens (1981, embedding theorem — NTU temporal patterns) + Deming (1950, quality gates at every stage)

## Context

Phase 31.1 wired CognitiveFusionEngine as Stage 1.5 in the Sovereign query pipeline. However, `SovereignRuntime._run_cognitive_fusion()` passes `dummy_embedding = [0.0] * 768` — the RAG stage silently returns empty results. The NTU module (`core/ntu/`, 1,146 lines) computes temporal belief/entropy/potential but is not connected to the fusion pipeline. This phase closes both gaps.

## Gaps Addressed

| Gap | Current State | Target State |
|-----|--------------|--------------|
| Embedding generation | `[0.0] * 768` dummy vector | Live sentence-transformer or Ollama embedding |
| Embedding quality gate | None — zero vector accepted silently | SNR gate rejects low-norm embeddings |
| NTU integration | Standalone module, not wired | NTU feeds temporal context into fusion |
| AgentDB vector search | Functional but never receives real queries | Real embeddings drive semantic retrieval |

## Package Changes

```
core/embedding/                          # NEW — Embedding service facade
  __init__.py                  # 15 lines — re-exports EmbeddingService
  service.py                   # 200 lines — tiered embedding with fallback
  quality_gate.py              # 80 lines — embedding norm + entropy validation

core/ntu/bridge.py                       # MODIFY — add CognitiveFusion adapter
core/sovereign/runtime_core.py           # MODIFY — replace dummy_embedding
core/cognitive_fusion/fusion_engine.py   # MODIFY — add embedding quality check
```

Total new code: ~295 lines. Total modified: ~60 lines across 3 files.

---

## 1. Embedding Service (Tiered Fallback)

```
CLASS EmbeddingService:
  """
  Tiered embedding generation with local-first fallback.

  Standing on Giants: Reimers & Gurevych (sentence-BERT, 2019)
  Artifact: core/embedding/service.py
  """

  FUNCTION __init__(self, config: Optional[EmbeddingConfig] = None):
    self.config = config OR EmbeddingConfig.from_env()
    self._model = None          # Lazy-loaded sentence-transformer
    self._dimension = 768       # Default; overridden by model metadata
    self._fallback_url = None   # Ollama /api/embeddings endpoint

  FUNCTION embed(self, text: str) -> List[float]:
    """
    Generate embedding vector for text input.

    Tier 1: sentence-transformers (local GPU/CPU)
    Tier 2: Ollama /api/embeddings (local inference)
    Tier 3: Raise EmbeddingUnavailableError (no silent fallback to zeros)

    INVARIANT: Never returns a zero vector.
    INVARIANT: Output dimension matches self._dimension.
    """
    TRY:
      RETURN self._embed_local(text)
    EXCEPT ModelNotLoadedError:
      TRY:
        RETURN self._embed_ollama(text)
      EXCEPT ConnectionError:
        RAISE EmbeddingUnavailableError(
          "No embedding backend available. "
          "Install sentence-transformers or start Ollama."
        )

  FUNCTION _embed_local(self, text: str) -> List[float]:
    IF self._model IS None:
      FROM sentence_transformers IMPORT SentenceTransformer
      self._model = SentenceTransformer(self.config.model_name)
      self._dimension = self._model.get_sentence_embedding_dimension()
    vector = self._model.encode(text, normalize_embeddings=True)
    RETURN vector.tolist()

  FUNCTION _embed_ollama(self, text: str) -> List[float]:
    response = httpx.post(
      f"{self.config.ollama_url}/api/embeddings",
      json={"model": self.config.ollama_model, "prompt": text},
      timeout=10.0
    )
    response.raise_for_status()
    RETURN response.json()["embedding"]

  PROPERTY dimension -> int:
    RETURN self._dimension


DATACLASS EmbeddingConfig:
  model_name: str = "all-MiniLM-L6-v2"       # 384-dim, fast
  ollama_url: str = "http://localhost:11434"
  ollama_model: str = "nomic-embed-text"      # 768-dim
  max_text_length: int = 512                  # Truncate beyond this

  CLASSMETHOD from_env(cls) -> EmbeddingConfig:
    """Load from BIZRA_EMBED_MODEL, OLLAMA_URL env vars."""
    RETURN cls(
      model_name=os.environ.get("BIZRA_EMBED_MODEL", cls.model_name),
      ollama_url=os.environ.get("OLLAMA_URL", cls.ollama_url),
      ollama_model=os.environ.get("BIZRA_OLLAMA_EMBED", cls.ollama_model),
    )
```

---

## 2. Embedding Quality Gate

```
CLASS EmbeddingQualityGate:
  """
  Validates embedding vectors before they enter the retrieval pipeline.

  Standing on Giants: Shannon (entropy as quality signal)
  Artifact: core/embedding/quality_gate.py
  """

  FUNCTION __init__(self, min_norm: float = 0.1, max_entropy_ratio: float = 0.98):  # Calibrated from 0.95
    self.min_norm = min_norm
    self.max_entropy_ratio = max_entropy_ratio

  FUNCTION validate(self, embedding: List[float]) -> GateResult:
    norm = sqrt(sum(x*x for x in embedding))

    IF norm < self.min_norm:
      RETURN GateResult(passed=False, reason="embedding_norm_too_low",
                        score=norm)

    # Shannon entropy of normalized distribution
    abs_values = [abs(x) for x in embedding]
    total = sum(abs_values) OR 1e-10
    probs = [v / total for v in abs_values]
    entropy = -sum(p * log2(p) for p in probs IF p > 0)
    max_entropy = log2(len(embedding))
    entropy_ratio = entropy / max_entropy

    IF entropy_ratio > self.max_entropy_ratio:
      RETURN GateResult(passed=False, reason="embedding_too_uniform",
                        score=entropy_ratio)

    RETURN GateResult(passed=True, reason="ok",
                      score=1.0 - entropy_ratio)

DATACLASS GateResult:
  passed: bool
  reason: str
  score: float
```

---

## 3. NTU → CognitiveFusion Bridge

```
# MODIFY: core/ntu/bridge.py — add CognitiveFusionAdapter

CLASS NTUFusionAdapter:
  """
  Feeds NTU temporal state into CognitiveFusion context.

  Standing on Giants: Takens (embedding theorem, temporal patterns)
  Artifact: core/ntu/bridge.py
  """

  FUNCTION __init__(self, ntu_instance):
    self.ntu = ntu_instance

  FUNCTION enrich_context(self, context: dict) -> dict:
    """
    Inject NTU temporal signals into fusion context dict.

    The CognitiveFusionEngine reads context["ntu_state"] to:
    - Adjust retrieval depth based on entropy (high entropy = more retrieval)
    - Weight HRM level selection by belief strength
    - Feed potential score into NorthStar alignment
    """
    state = self.ntu.state()
    context["ntu_state"] = {
      "belief": state.belief,
      "entropy": state.entropy,
      "potential": state.potential,
      "iteration": state.iteration,
      "pattern": self.ntu.detect_pattern().name IF self.ntu.detect_pattern() ELSE None,
    }

    # Entropy-driven retrieval depth: high uncertainty -> more sources
    IF state.entropy > 0.7:
      context.setdefault("retrieval_depth_multiplier", 2.0)
    ELIF state.entropy > 0.4:
      context.setdefault("retrieval_depth_multiplier", 1.5)

    RETURN context
```

---

## 4. Runtime Integration

```
# MODIFY: core/sovereign/runtime_core.py — _run_cognitive_fusion()

FUNCTION _run_cognitive_fusion(self, query, thought_prompt) -> Optional[FusionResult]:
  """
  Replace dummy_embedding with live embedding service.
  Add NTU temporal context enrichment.
  """
  TRY:
    # Step 1: Generate real embedding
    IF self._embedding_service IS NOT None:
      embedding = self._embedding_service.embed(query.text)

      # Step 1a: Quality gate
      IF self._embedding_gate IS NOT None:
        gate_result = self._embedding_gate.validate(embedding)
        IF NOT gate_result.passed:
          self.logger.warning(f"Embedding quality gate failed: {gate_result.reason}")
          RETURN None
    ELSE:
      self.logger.info("Embedding service unavailable — skipping cognitive fusion")
      RETURN None

    # Step 2: Enrich context with NTU temporal state
    context = dict(query.context)
    IF self._ntu_adapter IS NOT None:
      context = self._ntu_adapter.enrich_context(context)

    # Step 3: Run fusion pipeline
    RETURN self._cognitive_fusion.process(
      query=query.text,
      query_embedding=embedding,
      context=context,
    )

  EXCEPT EmbeddingUnavailableError AS e:
    self.logger.warning(f"Cognitive fusion skipped (no embedding): {e}")
    RETURN None
  EXCEPT Exception AS e:
    self.logger.warning(f"Cognitive fusion skipped: {e}")
    RETURN None
```

---

## 5. TDD Anchors

```
TEST test_embedding_service_local_produces_nonzero_vector:
  service = EmbeddingService(EmbeddingConfig(model_name="all-MiniLM-L6-v2"))
  vec = service.embed("test query")
  ASSERT len(vec) == 384
  ASSERT sum(x*x for x in vec) > 0.01

TEST test_embedding_service_ollama_fallback:
  # Mock: sentence-transformers not installed, Ollama returns 768-dim vector
  service = EmbeddingService()
  service._model = RAISE_ON_ACCESS(ImportError)
  mock_ollama(return_value={"embedding": [0.1]*768})
  vec = service.embed("test")
  ASSERT len(vec) == 768

TEST test_embedding_service_raises_when_both_unavailable:
  service = EmbeddingService()
  service._model = RAISE_ON_ACCESS(ImportError)
  mock_ollama(side_effect=ConnectionError)
  ASSERT_RAISES(EmbeddingUnavailableError, service.embed, "test")

TEST test_quality_gate_rejects_zero_vector:
  gate = EmbeddingQualityGate()
  result = gate.validate([0.0] * 768)
  ASSERT result.passed IS False
  ASSERT result.reason == "embedding_norm_too_low"

TEST test_quality_gate_accepts_normal_embedding:
  gate = EmbeddingQualityGate()
  vec = normalize(random_vector(768))
  result = gate.validate(vec)
  ASSERT result.passed IS True

TEST test_quality_gate_rejects_uniform_embedding:
  gate = EmbeddingQualityGate()
  result = gate.validate([1.0 / 768] * 768)
  ASSERT result.passed IS False
  ASSERT result.reason == "embedding_too_uniform"

TEST test_ntu_fusion_adapter_enriches_context:
  ntu = create_ntu(belief=0.8, entropy=0.3, potential=0.9)
  adapter = NTUFusionAdapter(ntu)
  ctx = adapter.enrich_context({})
  ASSERT "ntu_state" IN ctx
  ASSERT ctx["ntu_state"]["belief"] == 0.8

TEST test_ntu_high_entropy_increases_retrieval_depth:
  ntu = create_ntu(entropy=0.8)
  adapter = NTUFusionAdapter(ntu)
  ctx = adapter.enrich_context({})
  ASSERT ctx["retrieval_depth_multiplier"] == 2.0

TEST test_runtime_fusion_uses_real_embedding:
  # Integration: verify runtime calls embedding_service.embed() not dummy
  runtime = create_runtime(embedding_service=MockEmbeddingService())
  result = await runtime.query("test query")
  ASSERT result.fusion_report IS NOT None
  ASSERT result.fusion_report["retrieval_count"] >= 0

TEST test_runtime_graceful_when_embedding_unavailable:
  runtime = create_runtime(embedding_service=None)
  result = await runtime.query("test query")
  # Should complete without fusion, not crash
  ASSERT result.answer IS NOT None
  ASSERT result.fusion_report IS None
```

---

## Dependencies

| Dependency | Required For | Install |
|-----------|-------------|---------|
| `sentence-transformers>=2.2.0` | Tier 1 local embeddings | `pip install sentence-transformers` |
| `httpx>=0.24.0` | Tier 2 Ollama embeddings | Already in pyproject.toml |

**Note:** `sentence-transformers` is already listed in `pyproject.toml` under `[full]` extras. This phase makes it the preferred embedding backend but does not hard-require it — Ollama fallback works without it.

## Success Criteria

| Metric | Target |
|--------|--------|
| Zero dummy embeddings in pipeline | `grep -r "0.0.*768" core/sovereign/` returns 0 hits |
| Embedding quality gate | Rejects norm < 0.1 and entropy_ratio > 0.98 (calibrated from 0.95 — see `phase_32_completion.md` §1) |
| NTU state in fusion context | `fusion_report["ntu_belief"]` present when NTU available |
| Graceful degradation | No crashes when sentence-transformers and Ollama both unavailable |
| Test count | +10 new tests, all passing |
