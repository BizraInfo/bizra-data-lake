# Spec 05: Validation Plan

Standing on Giants:
- Deming (1950): Measure before and after — no unverified improvement claims
- Shannon (1948): SNR is the definitive quality measure
- Lamport (1978): Verify invariants at every state transition

## Validation Stages

### Stage 1: Unit Tests — FounderContext + Router + Covenant

```bash
pytest tests/core/sovereign/test_founder_context.py -v
```

Expected: 15+ tests covering:
- FounderContext load/graceful-fail (4 tests)
- Three-tier formatting and token budgets (7 tests)
- ContextTierRouter keyword/role routing (7 tests)
- GenesisCovenant load/format/invariants (5 tests)

### Stage 2: Regression — All Previous Phases Still Pass

```bash
# Phase 41: SNR Maximizer (91 tests)
pytest tests/core/sovereign/test_snr_maximizer.py -q

# Phase 42: IaaS + Protocol + Adapter (43 + 13 + 11 tests)
pytest tests/core/iaas/ -q
pytest tests/core/test_snr_protocol.py -q

# Legacy SNR engine (10 tests)
pytest tests/test_snr_engine.py -q

# Smoke pillars (15 tests)
pytest tests/integration/test_autonomous_pilot.py -q
```

Expected: 0 regressions. Every previous test passes unchanged.

### Stage 3: Integration — Enriched Prompt Reaches LLM

Manual verification (requires LM Studio running):

```bash
python scripts/node0_activate.py mission "What should I prioritize this week?"
```

Verify in output:
- [ ] Agent results mention MoMo's weekly goals (not generic advice)
- [ ] Guardian output references covenant invariants
- [ ] Context tier logged in mission result metadata
- [ ] SNR score is computed via ensemble (Phase 42 pipeline intact)
- [ ] Evidence receipt is emitted (Phase 41 pipeline intact)

### Stage 4: SNR Comparison — Before vs After

Run the same mission with and without founder context:

1. **Baseline** (current Phase 42, no founder context):
   - SNR score for "How should I prioritize this week?"
   - Note: agent output relevance (subjective)

2. **Phase 43** (with founder context):
   - Same mission, same agents, same models
   - SNR score comparison
   - Agent output relevance comparison

Expected: Higher SNR because founder context provides the query-relevant
signal that agents otherwise lack.

## Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | FounderContext loads from sovereign_state/ | Unit test |
| 2 | Three tiers (full/standard/minimal) format correctly | Unit test |
| 3 | ContextTierRouter selects appropriate tier | Unit test |
| 4 | GenesisCovenant contains Three Invariants | Unit test |
| 5 | Guardian agent gets covenant + full founder context | Integration test |
| 6 | Executor agent gets minimal context | Integration test |
| 7 | Missing identity files degrade gracefully | Unit test |
| 8 | Token overhead < 200 per agent (full tier) | Unit test |
| 9 | All 91 SNR maximizer tests pass | Regression |
| 10 | All 43 IaaS tests pass | Regression |
| 11 | All 13 protocol tests pass | Regression |
| 12 | All 15 smoke tests pass | Regression |
| 13 | Context tier logged in mission result | Integration |
| 14 | No hardcoded identity values in code | Code review |

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Identity files corrupted | Graceful degradation: empty context, original prompt |
| Token budget exceeded | Tier system caps at ~180 tokens (full) |
| Stale identity data | Data comes from persistent sovereign_state/ files |
| Agent confused by extra context | Structured format with clear delimiters |
| Import failure on FounderContext | try/except with fallback to original prompt |

## What This Phase Proves

Phase 43 proves that Node0 is not just a runtime — it's MoMo's node.
When the PAT team knows WHO they serve, WHAT assets exist, and WHY
this work matters, their output shifts from generic AI advice to
personalized sovereign intelligence.

This is the first step toward the Sovereign Empowerment Loop:
**Perceive → Think → Plan → Act → Sense → Learn → Remember → Share**

Phase 43 gives the PAT team the "Remember" — the persistent identity
that carries across missions. Phase 44 (SMA) will add episodic memory.
Phase 45 (HMM) will add prediction. But identity comes first.

"Before 8B humans, prove 1 node → 1 human."
Node0 = MoMo. The seed knows who planted it.
