# Sprint 3 Scope Lock — 7 Days (20-27 Mar 2026)
# ═══════════════════════════════════════════════

## Context
OpenAI agent analysis + Claude multi-session synthesis converge on the same conclusion:
**BIZRA is past architecture discovery. It is in operationalization.**

The correct move is NOT another framework, audit, or synthesis.
The correct move is: green CI → staged release → live cockpit → signed evidence → next sprint.

## Three Epics (and nothing else)

### Epic 1: Release Hardening (v0.87.1)
- [ ] All CI jobs green (Test Python + Test Rust are the remaining gates)
- [ ] SEC-003b baseline stabilized at 140
- [ ] CVE flatted patched (done: 267e7535)
- [ ] Dependabot: only 2 low remaining
- [ ] `verify_genesis.py` 6/6 green (done: 61d68358)
- [ ] Tag v0.87.1 when CI is fully green
- [ ] Release notes generated from git log

### Epic 2: Operator Cockpit (vertical slice)
- [ ] React component: BIZRA_Operator_Cockpit.jsx (done: 334 lines)
- [ ] Wire to live Python EventBus via WebSocket (kernel_daemon.py:9740)
- [ ] Show real-time: mission → guardian → execution → receipt → chain → evidence
- [ ] Ihsan meter (circular gauge, live score)
- [ ] 12-agent grid (PAT gold / SAT teal, consulted agents highlighted)
- [ ] Constitutional receipt card (BLAKE3 hashes, SNR, model, latency)
- [ ] Proof chain block viewer (genesis → current)
- [ ] Constitutional triangle: Ihsan / Amanah / Adl status
- [ ] Daughter Test: would أبوك وأمك understand this in 5 seconds?

### Epic 3: Economic Exactness
- [ ] Wire bizra-sippar through ALL treasury paths (done for resourcepool: cd84ac21)
- [ ] Verify Harberger tax computation uses Decimal (done)
- [ ] Verify Zakat distribution uses Decimal (done)
- [ ] Verify Gini coefficient uses Decimal (done: zero f64 drift)
- [ ] Add ExactAmount to token mint path
- [ ] Extend Sippar tests to cover marketplace split scenarios

## What is NOT in Sprint 3
- No new crates
- No new agent types
- No federation/multi-node
- No WASM/embedded targets
- No speculative architecture (Lean 4, HHMM formalism, DRA as runtime)

## Definition of Done (per internal enterprise standard)
- Tests pass (unit + integration)
- CI green on all gates
- Docs updated
- Benchmarks show no regression
- Security review passed (SEC-003b clean)
- Staging smoke test passes
- Monitoring/alerting configured
- Evidence receipt generated

## Sprint 3 Velocity Target
- v0.87.1 tagged by Day 2
- Cockpit live-wired by Day 5
- Economic exactness extended by Day 7
- Sprint review + v0.88.0 scope by Day 7
