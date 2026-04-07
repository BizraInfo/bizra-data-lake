# P0 Gate Acceptance Standard

**Status:** CANONICAL — applies to all D5 P0 gate verifications
**Origin:** Day 2 substrate transition audit (2026-04-08)
**Derived from:** Three-model convergence (Claude Opus 4.6 + GPT-5.4 + user synthesis)

## The Four-Condition Rule

A P0 gate is CLOSED if and only if all four conditions are met.
Any single condition unchecked → OPEN until fixed.

### Condition 1: Gate exists in CI or equivalent enforced workflow

A dedicated mechanism that runs automatically on every push/PR.

- Workflow file path must be named
- Job or step name must be identified
- "It runs locally" is insufficient — CI enforcement is required

### Condition 2: Gate checks the intended contracts

The mechanism must verify the specific thing the P0 is about.

- Constants checked must be enumerated
- Both language/system sources must be verified (if cross-language)
- "We have tests" is insufficient — the tests must cover the P0's specific concern

### Condition 3: Failure is observable and blocks correctly

When the gate fails, something visible happens that prevents regression.

- On drift/failure: build fails, PR blocked, or alert raised
- The failure mode must be demonstrated or a test must assert it
- Silent failures do not count as gates

### Condition 4: A receipt or artifact proves current passage

Evidence that the gate is currently passing, not just that it once passed.

- Last successful run URL or timestamp
- Or local run output with date
- Or test file path that asserts the gate's behavior
- Stale evidence (> 7 days without re-verification) should be refreshed

## Closure Rule

Binary. No third option.

- **All four boxes checked** → CLOSED WITH RECEIPT
- **Any single box unchecked** → OPEN, with the gap named precisely

No "probably." No "covered by another mechanism." If the four conditions
cannot be ticked off concretely, the answer is OPEN until they can.

## Application History

| P0 | Date | Result | Evidence |
|----|------|--------|----------|
| P0-IHSAN | 2026-03-19 | CLOSED | 0115016b — gate corrected 0.85→0.95 |
| P0-REDIS | 2026-04-07 | CLOSED | e9d700f3 — requirepass + auth URL |
| P0-DEPBOT | 2026-04-08 | CLOSED | a4e5a2b1 — vite 6.4.2, dependency tree proof |
| P0-CROSSLANG | 2026-04-08 | CLOSED | 6714c8cf — audit script, 4 constants aligned |
| P0-DILITHIUM | 2026-04-08 | CLOSED | 22308654 — 20 fate-binding tests, ML-DSA-87 + Z3 |
| P0-REFLEX-FLAG | 2026-04-08 | CLOSED | 22308654 — DEFAULT-LIVE, MOE-001 gate |

## Generalization

This standard is not specific to P0-CROSSLANG or any individual P0.
It is the universal acceptance rule for any P0 enforced by automation:

1. Gate exists in CI
2. Gate checks the right thing
3. Gate fails observably on regression
4. Receipt proves it currently passes

Any future P0 added to the registry must satisfy all four conditions
before it can be marked CLOSED.
