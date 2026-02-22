# 01: Fix SNRFacade Normalization Bug

## Standing on Giants
Shannon (1948): [0,1] normalized information metrics · PEP 544 (2017): Structural subtyping

## Problem

`SNRFacade._from_text_engine()` at `core/snr_protocol.py:200` uses `analysis.snr_linear` (unbounded ratio, range 0 to 10^10) and clamps it to [0,1]:

```python
# CURRENT (BUG): snr_linear is a ratio, not a score
normalized_score = min(max(analysis.snr_linear, 0.0), 1.0)
```

This always produces `1.0` for any text with nonzero signal and zero noise (since `signal / (0 + 1e-10)` is astronomically large). The facade was written before `snr_normalized` existed.

## Fix

Replace with the bounded `snr_normalized` field added in Phase 41 Fix 2:

```python
# FIXED: snr_normalized is already bounded [0,1]
normalized_score = analysis.snr_normalized
```

## Pseudocode

```
FUNCTION _from_text_engine(text, query, sources) -> SNRResult:
    IF text_engine is None:
        RETURN SNRResult(score=0, ihsan=False, engine="text")

    analysis = text_engine.analyze(text, query, sources)

    # Use the bounded [0,1] snr_normalized (Phase 41 Fix 2)
    # NOT snr_linear which is an unbounded diagnostic ratio
    normalized_score = analysis.snr_normalized

    RETURN SNRResult(
        score = normalized_score,
        ihsan_achieved = analysis.ihsan_achieved,
        engine = "text",
        metrics = analysis.to_dict(),
        recommendations = analysis.recommendations,
    )
```

## TDD Anchors

```python
# test_snr_protocol.py

def test_facade_text_engine_uses_normalized_score():
    """SNRFacade._from_text_engine uses snr_normalized, not snr_linear."""
    facade = SNRFacade(text_engine=SNRMaximizer())
    result = facade.calculate(text="Clean text with good content.", query="content")
    # snr_normalized is bounded [0,1], never artificially 1.0
    assert 0.0 < result.score < 1.0
    assert result.engine == "text"

def test_facade_text_score_matches_maximizer_normalized():
    """SNRFacade score equals snr_maximizer's snr_normalized."""
    maximizer = SNRMaximizer()
    analysis = maximizer.analyze("Test content for validation.")
    facade = SNRFacade(text_engine=maximizer)
    result = facade.calculate(text="Test content for validation.")
    assert abs(result.score - analysis.snr_normalized) < 1e-6

def test_facade_text_zero_noise_not_always_one():
    """With zero noise, score should be signal quality — NOT 1.0."""
    facade = SNRFacade(text_engine=SNRMaximizer())
    result = facade.calculate(text="Simple sentence.")
    # A simple sentence doesn't deserve score=1.0
    assert result.score < 0.95
```

## Files Modified

- `core/snr_protocol.py` — Line 200: replace `analysis.snr_linear` with `analysis.snr_normalized`

## Risk Assessment

- **Blast radius**: Low — SNRFacade is used by runtime_core.py and bridge layers
- **Reversibility**: Single-line change, easily reverted
- **Breaking change**: Yes — scores will drop for text-only paths (from always-1.0 to honest 0.3-0.8). This is correct behavior — the old score was a facade.
