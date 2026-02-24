# Peak Hidden Thoughts Flow v1.0 (Canonical)

Status: Canonical  
Version: 1.0  
Date: 2026-02-22  
Audience: Architecture, security, runtime, and verification engineers

## Purpose

This document defines the verified "hidden thoughts flow pattern" for BIZRA using strict signal discipline:

- Signal: actionable architectural behavior implemented in code.
- Noise: speculative detail, motivational framing, and unmeasured claims.

Every major claim below is mapped to concrete source evidence.

## Claim Discipline

Rules used in this artifact:

1. No claim without repository evidence.
2. No hard performance numbers unless backed by reproducible benchmark artifacts.
3. "Implemented" means executable code path exists.
4. "Verified" means implemented and surfaced in status/reporting or tests.

## Canonical Flow Pattern (Verified)

```text
Constitutional Invariants
-> Threat-Aware Gate Chain
-> Entropy Routing (S1/S2)
-> Graph-of-Thought Deliberation
-> System2->System1 Skill Compilation
-> Cryptographic Receipt Chain
-> Runtime/API Verification Surface
-> Atlas Verification Report
```

## Evidence Map (Signal Only)

| Layer | Verified Signal | Evidence | Status |
|---|---|---|---|
| Constitutional gate ordering | PCI gate chain enforces `IHSAN` before `SNR` for untrusted inter-node messages | `core/pci/gates.py:10`, `core/pci/gates.py:14`, `core/pci/gates.py:35` | Implemented |
| Entropy routing | Complexity scoring combines entropy + structural markers and routes to explicit tiers | `core/reasoning/entropy_router.py:101`, `core/reasoning/entropy_router.py:130`, `core/reasoning/entropy_router.py:173` | Implemented |
| Deliberative graph | Graph-of-Thought engine supports generate/aggregate/refine/validate/prune and computes a content hash | `core/sovereign/graph_core.py:67`, `core/sovereign/graph_core.py:74`, `core/sovereign/graph_core.py:156` | Implemented |
| Skill compilation cache | Structural hash caching with HHMM-style TTL policy and floor-based eviction | `core/hashtable/skill_cache.py:47`, `core/hashtable/skill_cache.py:68`, `core/hashtable/skill_cache.py:139`, `core/hashtable/skill_cache.py:173` | Implemented |
| Hash verification primitives | Bloom filter (double hash + merge), Merkle tree (domain separation + O(log n) proofs) | `core/hashtable/bloom_filter.py:92`, `core/hashtable/bloom_filter.py:139`, `core/hashtable/merkle_tree.py:24`, `core/hashtable/merkle_tree.py:167` | Implemented |
| Negotiation receipts | PAT+SAT co-signed negotiation receipt with digest/signature verification and evidence ledger append | `core/bridges/dual_agentic_bridge.py:101`, `core/bridges/dual_agentic_bridge.py:136`, `core/bridges/dual_agentic_bridge.py:351`, `core/bridges/dual_agentic_bridge.py:419` | Implemented |
| Runtime verification surface | Runtime status exposes PAT<->SAT receipt chain telemetry and verification state | `core/sovereign/runtime_core.py:2844`, `core/sovereign/runtime_core.py:2938`, `core/sovereign/runtime_core.py:3049` | Implemented |
| API verification surface | `/v1/health` includes receipt-chain verification fields | `core/sovereign/api.py:1051`, `core/sovereign/api.py:1055` | Implemented |
| Atlas integration | Atlas gap report consumes runtime status and marks PAT-SAT protocol verified/unverified | `scripts/atlas/atlas_gap_report.py:73`, `scripts/atlas/atlas_gap_report.py:117`, `scripts/atlas/atlas_gap_report.py:141` | Implemented |

## Hidden Golden Gems (Grounded)

### Gem 1: Threat-model-dependent gate ordering

Signal:
- Gate order is threat-aware, not arbitrary. In PCI paths, ethics (`IHSAN`) is checked before signal quality (`SNR`).

Why it matters:
- Prevents high-clarity but malicious payloads from bypassing safety intent.

Evidence:
- `core/pci/gates.py:10`
- `core/pci/gates.py:14`
- `core/pci/gates.py:26`

### Gem 2: System2 -> System1 compression is explicit

Signal:
- Deliberative outputs are converted into structural hashes and stored in an LRU cache.
- Cache entries can be TTL-scoped by hierarchy layer.

Why it matters:
- Reduces repeated deliberation cost while preserving policy-controlled quality floors.

Evidence:
- `core/hashtable/skill_cache.py:139`
- `core/hashtable/skill_cache.py:147`
- `core/hashtable/skill_cache.py:238`

### Gem 3: Hash-table stack has complementary roles

Signal:
- Bloom filter: probabilistic membership and OR-based merge.
- Merkle tree: inclusion proofs for tamper-evident integrity.
- Skill cache: constant-time retrieval of compiled behavior.

Why it matters:
- Together they provide fast lookup, federation merge semantics, and cryptographic verifiability.

Evidence:
- `core/hashtable/bloom_filter.py:94`
- `core/hashtable/bloom_filter.py:141`
- `core/hashtable/merkle_tree.py:37`
- `core/hashtable/merkle_tree.py:169`

### Gem 4: PAT-SAT agreements are cryptographically typed events

Signal:
- Negotiation receipts carry proposer/sat identities and both signatures over the same payload digest.
- Receipts are appended to an evidence ledger with stable origin metadata.

Why it matters:
- Enables audit replay, dispute resolution, and runtime health assertions from first principles.

Evidence:
- `core/bridges/dual_agentic_bridge.py:104`
- `core/bridges/dual_agentic_bridge.py:137`
- `core/bridges/dual_agentic_bridge.py:462`

### Gem 5: Verification is runtime-visible, not just internal

Signal:
- Runtime status and API health both surface receipt-chain verification fields.
- Atlas report pipeline consumes these fields for capability verification.

Why it matters:
- Converts "security architecture" from documentation into operational truth signals.

Evidence:
- `core/sovereign/runtime_core.py:3049`
- `core/sovereign/api.py:1051`
- `scripts/atlas/atlas_gap_report.py:141`

## Deferred / Not Yet Implemented (Explicit)

These are intentionally excluded from "implemented" claims:

1. HMM Baum-Welch learning is deferred.
   - Evidence: `core/prediction/hmm_engine.py:560`, `core/prediction/hmm_engine.py:568`
2. Any hard claims about benchmarked latency, throughput, or SNR percentages without reproducible benchmark artifacts.

## Operational Verification Commands

Run these commands to re-validate critical claims:

```bash
python -m pytest tests/core/sovereign/test_runtime_core.py -q
python -m pytest tests/core/sovereign/test_api_metrics.py -q
python scripts/atlas/atlas_gap_report.py --matrix docs/atlas_alignment/atlas_capability_matrix.yaml --json
```

## Immediate Next Steps (Decision-Complete)

1. Extend PAT-SAT chain status with typed signature fields:
   - Add `pat_signature_valid` and `sat_signature_valid` into runtime status payload.
   - Acceptance: values exposed in `runtime.status()`, `/v1/health`, and atlas report.

2. Add DRA convergence gate as policy-bound output filter:
   - Gate condition: high-quality + low-variance + constitutional pass.
   - Acceptance: filtered outputs produce quarantine reason codes when rejected.

3. Add measurable trajectory artifact:
   - Generate `ratio_by_phase.json` for doc:code evolution.
   - Acceptance: artifact included in evidence package and referenced from Atlas report.

