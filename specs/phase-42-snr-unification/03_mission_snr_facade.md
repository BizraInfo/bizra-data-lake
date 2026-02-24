# 03: Replace Mission SNR with SNRFacade

## Standing on Giants
Shannon (1948): Unified measurement · Deming (1950): Single source of truth · Lamport (1978): Consistent state

## Problem

`scripts/node0_activate.py` uses `SNRCalculatorV2` directly (imported at line 82-84). The Phase 41 VIP wiring calls `_SNR_CALCULATOR.compute_snr()` on agent output text with real CUDA embeddings. This works correctly but:

1. Only uses the embedding engine — no text-space noise analysis (7 dimensions from snr_maximizer)
2. No ensemble scoring — misses the heuristic signal (verbosity, ambiguity, bias detection)
3. Not routed through `SNRFacade` — so the unified protocol is bypassed

## Current Mission SNR Path (Phase 41)

```python
# In _execute_mission() — approximate location
from core.iaas.snr_v2 import SNRCalculatorV2
_SNR_CALCULATOR = SNRCalculatorV2()

# After agents complete:
combined_text = "\n\n".join(agent_outputs)
components = _SNR_CALCULATOR.compute_snr(
    query=mission_description,
    texts=[combined_text],
    # CUDA embeddings generated internally by compute_snr
)
snr_score = components.snr
ihsan_achieved = components.ihsan_achieved
```

## Target Mission SNR Path

```python
from core.snr_protocol import SNRFacade, SNRResult
from core.iaas.snr_v2 import SNRCalculatorV2
from core.iaas.snr_v2_adapter import SNRv2Adapter
from core.sovereign.snr_maximizer import SNRMaximizer

# At kernel init:
_SNR_FACADE = SNRFacade(
    v2_engine=SNRv2Adapter(SNRCalculatorV2()),
    text_engine=SNRMaximizer(),
)

# After agents complete:
result: SNRResult = _SNR_FACADE.calculate(
    text=combined_text,
    query=mission_description,
)
snr_score = result.score           # Ensemble of v2 + maximizer
ihsan_achieved = result.ihsan_achieved
engine_used = result.engine        # "ensemble_v2" or fallback
```

## Pseudocode

### 3A: Modify Kernel Initialization

```
# In node0_activate.py, at module level (near line 82):

# OLD:
# _SNR_CALCULATOR = SNRCalculatorV2()

# NEW:
FUNCTION _init_snr_facade() -> SNRFacade:
    TRY:
        calculator = SNRCalculatorV2()
        v2_adapter = SNRv2Adapter(calculator)
        logger.info("  SNR v2 adapter: initialized (Shannon + Renyi-2)")
    EXCEPT ImportError:
        v2_adapter = None
        logger.warning("  SNR v2 adapter: unavailable")

    TRY:
        maximizer = SNRMaximizer()
        logger.info("  SNR maximizer: initialized (7 noise dimensions)")
    EXCEPT ImportError:
        maximizer = None
        logger.warning("  SNR maximizer: unavailable")

    facade = SNRFacade(
        v2_engine = v2_adapter,
        text_engine = maximizer,
        ihsan_threshold = UNIFIED_IHSAN_THRESHOLD,
    )
    logger.info(f"  SNR facade: ready (engines: v2={v2_adapter is not None}, text={maximizer is not None})")
    RETURN facade

_SNR_FACADE = _init_snr_facade()
```

### 3B: Modify Mission Execution

```
# In _execute_mission(), replace direct SNR call:

FUNCTION _compute_mission_snr(mission_desc: str, agent_outputs: list[str]) -> dict:
    """Compute unified SNR for mission output via facade."""
    combined_text = "\n\n".join(agent_outputs)

    result = _SNR_FACADE.calculate(
        text = combined_text,
        query = mission_desc,
    )

    RETURN {
        "snr_score": result.score,
        "ihsan_achieved": result.ihsan_achieved,
        "engine": result.engine,
        "quality_tier": result.metrics.get("quality_tier", "unknown"),
        "v2_snr": result.metrics.get("v2_snr"),
        "text_snr": result.metrics.get("text_snr"),
        "recommendations": result.recommendations,
    }
```

### 3C: Enrich Mission Receipt

```
# In the receipt/evidence recording section:

snr_data = _compute_mission_snr(mission["description"], agent_texts)

receipt_payload = {
    "mission_id": mission_id,
    "agents": agent_results,
    "snr": {
        "score": snr_data["snr_score"],
        "engine": snr_data["engine"],
        "ihsan_achieved": snr_data["ihsan_achieved"],
        "quality_tier": snr_data["quality_tier"],
        # Ensemble breakdown (when available)
        "v2_snr": snr_data.get("v2_snr"),
        "text_snr": snr_data.get("text_snr"),
    },
    "recommendations": snr_data["recommendations"],
    "status": _classify_status(snr_data),
}
```

### 3D: Status Classification

```
FUNCTION _classify_status(snr_data: dict) -> str:
    """Constitutional gate: classify mission quality status."""
    score = snr_data["snr_score"]

    IF score >= SNR_THRESHOLD_T0_ELITE:           # 0.98
        RETURN "green_elite"
    ELIF score >= SNR_THRESHOLD_T1_HIGH:          # 0.95
        RETURN "green_production"
    ELIF score >= UNIFIED_SNR_THRESHOLD:          # 0.85
        RETURN "amber_restricted"
    ELSE:
        RETURN "red_rejected"
```

## TDD Anchors

```python
# test_mission_snr_facade.py

def test_mission_uses_facade_not_direct():
    """Mission SNR computation routes through SNRFacade."""
    # Verify _SNR_FACADE is initialized (not _SNR_CALCULATOR)
    from scripts.node0_activate import _SNR_FACADE
    assert isinstance(_SNR_FACADE, SNRFacade)

def test_mission_snr_returns_ensemble():
    """When both engines available, mission gets ensemble score."""
    result = _compute_mission_snr(
        "Explain signal processing",
        ["Signal processing involves analyzing..."]
    )
    assert "engine" in result
    assert result["snr_score"] > 0.0
    assert result["engine"] in ("ensemble_v2", "snr_v2", "text")

def test_mission_receipt_contains_breakdown():
    """Mission receipt includes both v2 and text SNR when available."""
    result = _compute_mission_snr("test", ["test response"])
    if result["engine"] == "ensemble_v2":
        assert result["v2_snr"] is not None
        assert result["text_snr"] is not None

def test_status_classification_tiers():
    """Status classification follows constitutional thresholds."""
    assert _classify_status({"snr_score": 0.99}) == "green_elite"
    assert _classify_status({"snr_score": 0.96}) == "green_production"
    assert _classify_status({"snr_score": 0.90}) == "amber_restricted"
    assert _classify_status({"snr_score": 0.50}) == "red_rejected"
```

## Files Modified

- `scripts/node0_activate.py` — Replace `_SNR_CALCULATOR` with `_SNR_FACADE`; update `_execute_mission()`
- No new files (uses existing SNRFacade + new adapter from spec 02)

## Backward Compatibility

- The `--verify` CLI command still works (reads evidence ledger, not live SNR)
- Old receipts with `snr_method: "snr_v2_embeddings"` remain valid
- New receipts will show `snr_method: "ensemble_v2"` when both engines available

## Performance

- Additional cost: `SNRMaximizer.analyze()` ~5ms for text heuristics
- Total mission SNR: ~55ms (50ms v2 embeddings + 5ms text heuristics)
- Acceptable: missions take 2-5 minutes for LLM calls; 55ms SNR is negligible
