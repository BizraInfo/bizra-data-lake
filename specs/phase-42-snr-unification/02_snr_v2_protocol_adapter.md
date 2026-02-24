# 02: Wire SNR v2 into SNRFacade via Protocol Adapter

## Standing on Giants
Shannon (1948): Channel capacity · Renyi (1961): Generalized entropy · PEP 544: Structural subtyping

## Problem

`SNRCalculatorV2` is the most rigorous SNR engine (Shannon-enhanced with Renyi-2 entropy, real CUDA embeddings, production-grade). But:

1. It is NOT registered in `SNRFacade` — the facade only knows about `arte_engine` (embedding) and `snr_maximizer` (text)
2. It has no `calculate_snr_normalized()` method, so it doesn't conform to `SNRProtocol`
3. Most call sites pass `iaas_score=0.8` as a constant instead of computing real IaaS dimensions

## Design Decision

`SNRCalculatorV2` should become the **primary embedding engine** in `SNRFacade`, replacing or supplementing `arte_engine.SNREngine`. Rationale:

- `snr_v2` uses real embeddings + Shannon + Renyi-2 — the most complete measurement
- `arte_engine` requires a populated FAISS index that doesn't exist in most environments
- `snr_v2` is already initialized in `node0_activate.py` at kernel startup (line 82-84)

## Pseudocode

### 2A: Protocol Adapter for SNRCalculatorV2

```
CLASS SNRv2Adapter:
    """Adapts SNRCalculatorV2 to conform to SNRProtocol."""

    INIT(calculator: SNRCalculatorV2, ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD):
        self.calculator = calculator
        self.ihsan_threshold = ihsan_threshold

    FUNCTION calculate_snr_normalized(**kwargs) -> SNRResult:
        # Extract inputs — accept both naming conventions
        query = kwargs.get("query") OR kwargs.get("text", "")
        texts = kwargs.get("texts") OR [kwargs.get("text", "")]
        query_embedding = kwargs.get("query_embedding")
        text_embeddings = kwargs.get("text_embeddings") OR kwargs.get("context_embeddings")

        # Call the real engine
        IF query_embedding is not None AND text_embeddings is not None:
            components = calculator.compute_snr(
                query=query,
                texts=texts,
                query_embedding=query_embedding,
                text_embeddings=text_embeddings,
            )
        ELSE:
            # Lexical-only fallback (no embeddings available)
            components = calculator.calculate_simple(query=query, texts=texts)

        # Map SNRComponentsV2 → SNRResult
        RETURN SNRResult(
            score = components.snr,           # Already [0,1] weighted geometric mean
            ihsan_achieved = components.ihsan_achieved,
            engine = "snr_v2",
            metrics = {
                "signal_strength": components.signal_strength,
                "diversity": components.diversity,
                "grounding": components.grounding,
                "iaas_score": components.iaas_score,
                "semantic_relevance": components.semantic_relevance,
                "channel_efficiency": components.channel_efficiency,
                "quality_tier": components.quality_tier,
                "redundancy": components.redundancy,
                "entropy": components.entropy,
            },
            recommendations = _build_recommendations(components),
        )

    FUNCTION _build_recommendations(components: SNRComponentsV2) -> list[str]:
        recs = []
        IF components.signal_strength < 0.6:
            recs.append("Improve semantic alignment with query")
        IF components.diversity < 0.5:
            recs.append("Increase source diversity (Renyi-2 detected concentration)")
        IF components.redundancy > 0.4:
            recs.append("Reduce redundant content")
        IF components.grounding < 0.5:
            recs.append("Add grounding evidence or citations")
        RETURN recs
```

### 2B: Register in SNRFacade

```
# Update SNRFacade.__init__ to accept snr_v2 as primary embedding engine

CLASS SNRFacade:
    INIT(
        embedding_engine = None,       # arte_engine (legacy)
        text_engine = None,            # snr_maximizer
        v2_engine = None,              # SNRv2Adapter (NEW — preferred embedding path)
        ihsan_threshold = 0.95,
    ):
        self.embedding_engine = embedding_engine
        self.text_engine = text_engine
        self.v2_engine = v2_engine
        self.ihsan_threshold = ihsan_threshold

    FUNCTION calculate(**kwargs) -> SNRResult:
        has_v2 = self.v2_engine is not None
        has_embeddings = (query_embedding and context_embeddings and embedding_engine)
        has_text = (text and text_engine)

        # Priority: v2 > arte > text-only
        IF has_v2 AND has_text:
            v2_result = v2_engine.calculate_snr_normalized(**kwargs)
            txt_result = _from_text_engine(text, query, sources)
            RETURN _ensemble_v2(v2_result, txt_result)
        ELIF has_v2:
            RETURN v2_engine.calculate_snr_normalized(**kwargs)
        ELIF has_embeddings AND has_text:
            RETURN _ensemble(...)          # Legacy path
        ELIF has_embeddings:
            RETURN _from_embedding_engine(...)
        ELIF has_text:
            RETURN _from_text_engine(...)
        ELSE:
            RETURN SNRResult(score=0.0, ...)

    FUNCTION _ensemble_v2(v2_result, txt_result) -> SNRResult:
        """Geometric mean of v2 (embedding-grade) and text (heuristic) scores."""
        score = geometric_mean(v2_result.score, txt_result.score)
        RETURN SNRResult(
            score = score,
            ihsan_achieved = score >= ihsan_threshold,
            engine = "ensemble_v2",
            metrics = {
                "v2_snr": v2_result.score,
                "v2_tier": v2_result.metrics.get("quality_tier"),
                "text_snr": txt_result.score,
                "ensemble_method": "geometric_mean",
            },
        )
```

## Where to Put the Adapter

New file: `core/iaas/snr_v2_adapter.py` (~60 lines)

This keeps the adapter separate from the engine itself, following the existing pattern where `core/snr_protocol.py` defines the protocol and `arte_engine.py` has its own adapter method.

## TDD Anchors

```python
# test_snr_v2_adapter.py

def test_adapter_conforms_to_protocol():
    """SNRv2Adapter satisfies SNRProtocol structural typing."""
    from core.snr_protocol import SNRProtocol
    adapter = SNRv2Adapter(SNRCalculatorV2())
    assert isinstance(adapter, SNRProtocol)

def test_adapter_with_real_embeddings():
    """Adapter produces valid SNRResult from real embedding inputs."""
    adapter = SNRv2Adapter(SNRCalculatorV2())
    result = adapter.calculate_snr_normalized(
        query="What is signal processing?",
        texts=["Signal processing is the analysis of signals."],
        query_embedding=np.random.randn(384),
        text_embeddings=np.random.randn(1, 384),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.engine == "snr_v2"
    assert "quality_tier" in result.metrics

def test_adapter_lexical_fallback():
    """Without embeddings, adapter uses calculate_simple."""
    adapter = SNRv2Adapter(SNRCalculatorV2())
    result = adapter.calculate_snr_normalized(
        query="test query",
        texts=["test response text"],
    )
    assert 0.0 <= result.score <= 1.0
    assert result.engine == "snr_v2"

def test_facade_with_v2_engine():
    """SNRFacade routes to v2 when v2_engine is provided."""
    facade = SNRFacade(
        v2_engine=SNRv2Adapter(SNRCalculatorV2()),
        text_engine=SNRMaximizer(),
    )
    result = facade.calculate(
        text="Signal processing fundamentals.",
        query="signal processing",
    )
    assert result.engine in ("ensemble_v2", "text")
    assert 0.0 < result.score < 1.0
```

## Files Modified

- `core/iaas/snr_v2_adapter.py` — NEW (~60 lines)
- `core/snr_protocol.py` — Add `v2_engine` parameter to `SNRFacade.__init__` and routing logic
- `core/iaas/__init__.py` — Export `SNRv2Adapter`

## Files NOT Modified

- `core/iaas/snr_v2.py` — Adapter wraps it; no changes needed to the engine itself
- `arte_engine.py` — Kept as legacy embedding path; v2 takes priority when available

## Risk Assessment

- **Blast radius**: Medium — changes to SNRFacade affect all consumers (runtime_core, bridges)
- **Backward compatibility**: Preserved — `v2_engine=None` defaults to existing behavior
- **Performance**: `snr_v2.compute_snr()` with CUDA embeddings takes ~50ms. Acceptable for mission receipts.
