# Node0 Risk Register

Status: active

## Scoring Model

- Probability: 1-5
- Impact: 1-5
- Score = Probability x Impact

## Risks

| ID | Risk | Probability | Impact | Score | Mitigation | Owner |
|---|---|---:|---:|---:|---|---|
| R1 | Truth drift between spec, DoD, and runtime | 3 | 5 | 15 | docs parity gate, README routing, DoD audit trail | Docs / Architecture |
| R2 | Production auth regression reopens anonymous surface | 3 | 5 | 15 | fail-closed tests in CI, explicit env requirements | Security / Backend |
| R3 | Operator drift creates alternate birth paths | 2 | 5 | 10 | freeze canonical commands, reject undocumented scripts | Runtime |
| R4 | Extraction copies lake noise into production repo | 4 | 4 | 16 | dependency-closure-only import manifest | Architecture |
| R5 | Native Linux certification is skipped in favor of WSL convenience | 3 | 5 | 15 | certification gate in release policy | Ops |
| R6 | Performance budgets degrade silently | 3 | 4 | 12 | benchmark ratchets, artifact upload, trend review | SRE |
| R7 | Signing and provenance remain aspirational | 3 | 5 | 15 | release policy plus CI provenance job | Release Engineering |
| R8 | Genesis-100 scope reopens Node0 birth semantics | 2 | 4 | 8 | keep separate gate hierarchy | Program Management |

## Cascading Risk Notes

- R1 amplifies R3 and R8.
- R2 amplifies R7 because unsigned insecure releases create false trust.
- R4 amplifies R6 by making benchmarks noisy and non-reproducible.
- R5 invalidates certification claims even if CI is green.

## Review Cadence

- Weekly during extraction
- Before every release candidate
- Immediately after any security incident or lifecycle contract change
