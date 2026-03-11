# Node0 Program Blueprint

Status: active execution blueprint
Scope: production canon for Node0

## Purpose

This document turns Node0 from an evidence-backed birth into a managed
production program. It integrates:

- PMBOK delivery discipline
- DevOps and CI/CD automation
- performance and quality ratchets
- security fail-closed rules
- SAPE as the reasoning and promotion method

## Executive Thesis

The correct order is:

```text
birth truth
  -> production canon
  -> native Linux certification
  -> signed release surface
  -> Genesis-100 preflight
```

## Standing on the Shoulders of Giants

| Source | Rule for Node0 |
|---|---|
| Lamport | one runtime truth object, explicit invariants |
| Dijkstra | keep the trusted core small |
| Deming | quality is designed into the pipeline |
| Shannon | prioritize high-SNR work over speculative breadth |
| Harel | state and transition truth before behavior claims |
| OWASP | fail closed, least privilege, explicit auth |
| SRE | latency, reliability, recovery, and rollback are release gates |
| Ihsan / Adl / Amanah | no false readiness, no silent weakening, no ambiguous custody |

## Program Workstreams

### WS1 - Truth and Governance

Deliverables:
- aligned spec / DoD / audit trail
- one visible documentation path
- no contradictory gate counts

Exit:
- one authoritative reading path from `README.md`

### WS2 - Security and Trust Boundaries

Deliverables:
- production startup fails without required auth
- stable JWT secret handling
- Ghost bridge disabled by default in production
- websocket auth contract is type-consistent

Exit:
- zero anonymous production surfaces

### WS3 - Canonical Operator Surface

Deliverables:
- frozen command set
- ceremony remains verification-only
- no alternate birth path

Exit:
- docs, scripts, and CI all reference the same commands

### WS4 - Production Repo Extraction

Deliverables:
- dependency-closure import set
- signed release policy
- protected main
- import manifest pinned to upstream commit

Exit:
- Node0 boots, tasks, and ceremonies inside this repo without the lake

### WS5 - Certification and Release

Deliverables:
- native Linux certification
- performance budgets
- provenance artifacts
- release signing

Exit:
- one certified native Linux run and one compatibility WSL2 smoke

## PMBOK x DevOps x SAPE Mapping

| Dimension | PMBOK focus | DevOps focus | SAPE focus | Node0 rule |
|---|---|---|---|---|
| Integration | one program board | one release surface | symbolic truth convergence | no competing truth |
| Scope | explicit boundaries | no hidden release creep | abstraction discipline | Node0 separate from Genesis-100 |
| Schedule | phased exits | automation-driven flow | probe before promotion | each wave has evidence |
| Quality | DoD and audits | CI/CD gates | elevation only after proof | no threshold weakening |
| Risk | tracked mitigations | rollback and recovery | fail-closed probes | security before scale |
| Communications | one reading path | operator-first docs | high SNR | README must route in one hop |

## SAPE Execution Loop

1. Symbolic
   - define the invariant, schema, or threshold
2. Abstraction
   - place it in the correct plane and interface
3. Probe
   - test, benchmark, or certify it
4. Elevation
   - promote it to a hard gate, keep it informational, or reject it

## Immediate Priority Order

1. production repo extraction
2. native Linux certification
3. provenance and signed release policy execution
4. benchmark ratchets
5. Genesis-100 preflight

## Acceptance

This blueprint is satisfied only when `bizra-node0` can act as the only
production release surface with documented, tested, and certified behavior.
