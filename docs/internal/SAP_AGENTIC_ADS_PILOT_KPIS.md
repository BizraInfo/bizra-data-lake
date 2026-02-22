# SAP Agentic Ads Retail v0 — Pilot KPI Pack (Internal)

## Purpose
Define measurable, internal-only safety and quality thresholds for the shadow pilot.

## Gate Equations
Let:
- `CR = conformance_passed / conformance_total`
- `SR = shadow_passed / shadow_total`
- `CV = covered_core8_providers / 8`

Release gate:
```text
G = min(CR, SR, CV)
```

Internal-go/no-go threshold for Corpus-Truth-First mode:
```text
G == 1.0
```

## Current Measured Baseline (2026-02-21)
1. `CR = 22/22 = 1.0000`
2. `SR = 4/4 = 1.0000`
3. `CV = 5/8 = 0.6250`
4. `G = 0.6250` (blocked by provider coverage)

## KPI Targets
### 1. Protocol Integrity
1. Conformance pass rate: target `100%`.
2. Receipt-chain integrity errors: target `0`.
3. Consent-scope violations accepted: target `0`.

### 2. Safety and Governance
1. Redline capture coverage on simulated breaches: target `100%`.
2. Strict session-limit denial coverage: target `100%`.
3. Offer provenance completeness: target `100%`.
4. Disclosure completeness (`source_refs` + `uncertainty`): target `100%`.

### 3. Shadow-Mode Marketing
1. Claim-bearing responses with non-empty evidence refs: target `100%`.
2. Out-of-evidence prompts fail-closed: target `100%`.
3. Consent-sensitive prompts without consent fail-closed: target `100%`.
4. Shadow receipt-chain verification: target `100%`.

## Guardrails
1. No externally-audited language in pilot outputs.
2. No public claim expansion beyond measured internal values.
3. Any consent breach or chain failure triggers immediate review.
