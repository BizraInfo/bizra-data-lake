# BIZRA Economic Constitution v1.0

**Status:** first executable economic invariant slice  
**Truth label:** `[ENFORCEMENT: WIRED]` for local deterministic checks; cryptographic signing remains `[OPTIMIZATION: PLANNED]`.

## Purpose

This slice turns the Semantic Transducer policy stubs (`zann_zero`, `riba_zero`,
`gini_threshold`) into a deterministic, testable economic gate without granting
runtime action power.

The trusted path is:

```text
RawParsedClaim -> Claim -> semantic fate_gate -> economic_fate_gate -> GateDecision
```

No transfer is executed by this slice. `LedgerEntry` values are inert records
that can be hashed and later signed by a dedicated signing slice.

## Enforced invariants

1. **Integer money:** all amounts use nanocaps (`amount_nc`) to avoid float drift.
2. **Immutable state:** `LedgerState.apply(...)` returns a new state and conserves `total_issued_nc`.
3. **Riba-zero:** deterministic pattern detection rejects fixed-interest, compounding, discount-distortion, and leverage evidence.
4. **Zakat arithmetic:** obligation is computed by integer basis points.
5. **Gini containment:** proposed transfers are simulated before admission; inequality-worsening transfers above policy threshold escalate.
6. **Semantic-first gating:** `economic_fate_gate(...)` never bypasses the PR #83 semantic gate.
7. **Unsigned honesty:** entries default to `LOCAL_UNSIGNED_DEV`; placeholder signatures are forbidden.

## Explicit non-goals

- No daemon start or stop.
- No Node1 start.
- No mission executor wiring.
- No memory ingestion.
- No Third Fact publication.
- No real transfer execution.
- No claim that cryptographic signing is complete.

## Next frontier

The next bounded slice should add signed economic receipts or proof-of-health
status visibility, not broad autonomous economic actions.
