# 05: Validation Plan

## Standing on Giants
Deming (PDCA, 1950) · Lamport (correctness, 1978)

## Validation Sequence

Each spec is validated independently, then the full stack is validated end-to-end.

### Stage 1: Unit Tests (After Each Spec)

```
Spec 01 (Facade Fix):
  pytest tests/core/test_snr_protocol.py -q
  ASSERT: facade text score < 1.0 for normal text
  ASSERT: facade text score matches snr_normalized

Spec 02 (v2 Adapter):
  pytest tests/core/iaas/test_snr_v2_adapter.py -q
  ASSERT: adapter conforms to SNRProtocol
  ASSERT: adapter with embeddings produces valid SNRResult
  ASSERT: adapter without embeddings uses lexical fallback
  ASSERT: facade with v2_engine routes correctly

Spec 03 (Mission Facade):
  pytest tests/integration/test_mission_snr.py -q
  ASSERT: _SNR_FACADE is SNRFacade instance
  ASSERT: mission SNR returns ensemble when both engines available
  ASSERT: status classification matches tier thresholds

Spec 04 (GoT Wiring):
  pytest tests/integration/test_got_mission.py -q
  ASSERT: GoT synthesis produces non-empty conclusion
  ASSERT: GoT failure falls back to concatenation
  ASSERT: receipt includes thought_chain metadata
```

### Stage 2: Regression Suite (After All Specs)

```
# Existing suites — all must remain green
pytest tests/core/sovereign/test_snr_maximizer.py -q    # 91 tests
pytest tests/core/iaas/ -q                               # 32 tests
pytest tests/test_snr_engine.py -q                       # 10 tests
pytest tests/integration/test_autonomous_pilot.py -q     # 15 tests
```

### Stage 3: Live Mission Validation

```
# Run a live mission with the unified stack
python scripts/node0_activate.py mission "What is the Sovereign Empowerment Loop?"

VERIFY:
  1. Receipt shows engine: "ensemble_v2" (not "snr_v2_embeddings")
  2. Receipt includes v2_snr and text_snr breakdown
  3. Receipt includes got.thought_chain with >= 1 thought
  4. Receipt includes got.active == true
  5. SNR score is honest (0.4-0.8 range, not 0.98)
  6. Status is amber_restricted or green_production (depends on LLM quality)
  7. Evidence ledger hash chain is intact

# Verify with CLI
python scripts/node0_activate.py verify
VERIFY: Latest receipt has valid chain link
```

### Stage 4: Negative Validation

```
# Test that bad content is rejected

# 1. Garbage mission
python scripts/node0_activate.py mission "asdf jkl;"
VERIFY: status == "red_rejected", snr < 0.85

# 2. Mission with LM Studio down
# Stop LM Studio, then:
python scripts/node0_activate.py mission "Test without LLM"
VERIFY: GoT falls back to templates, mission still completes
VERIFY: got.active == false or got.thought_count == 0
```

## Acceptance Criteria (Phase 42 Complete)

| # | Criterion | Verification Method |
|---|-----------|-------------------|
| 1 | SNRFacade uses `snr_normalized` not `snr_linear` | Unit test |
| 2 | SNRv2Adapter conforms to SNRProtocol | `isinstance()` check |
| 3 | SNRFacade routes to v2 when v2_engine provided | Unit test |
| 4 | Mission receipts show ensemble engine | Live mission |
| 5 | Mission receipts include v2 + text SNR breakdown | Live mission |
| 6 | GoT produces thought chain with LLM | Live mission |
| 7 | GoT falls back gracefully without LLM | Negative test |
| 8 | 91/91 snr_maximizer tests pass | Regression |
| 9 | 15/15 smoke tests pass | Regression |
| 10 | Evidence ledger hash chain intact | `--verify` CLI |

## Build Sequence

```
Spec 01  →  Spec 02  →  Spec 03  →  Spec 04  →  Stage 2  →  Stage 3
(5 min)     (20 min)    (30 min)    (45 min)    (5 min)     (10 min)

Dependencies:
  02 depends on 01 (facade must be fixed before adding v2_engine)
  03 depends on 02 (mission needs v2 adapter registered in facade)
  04 depends on 03 (GoT output feeds into facade measurement)
  Stages 2-4 depend on all specs
```

## Rollback Plan

Each spec is independently revertable:

- **Spec 01**: Revert line 200 in `snr_protocol.py` — one-line change
- **Spec 02**: Delete `snr_v2_adapter.py`, remove `v2_engine` param — backward compat preserved
- **Spec 03**: Restore `_SNR_CALCULATOR` in `node0_activate.py` — Phase 41 behavior
- **Spec 04**: Set `_GOT_ENGINE = None` — disables GoT synthesis entirely

Full rollback to Phase 41 state: revert all 4 changes. No data migration needed — evidence ledger is append-only.
