# Peak Hidden Thoughts Flow v1.1 (SNR/HHMM/Hash/Diffusion)

Status: Canonical  
Version: 1.1  
Date: 2026-03-01

## Scope

This pass uses strict SNR discipline:

- Signal: executable architecture behavior with concrete evidence.
- Noise: speculative framing, deferred logic, or unbounded claims.

Primary graph artifact:

- `artifacts/atlas/cross_module_flow_report.json`

## Graph Baseline (Current Run)

From `cross_module_flow_report.json`:

- Python files scanned: 922
- Rust files scanned: 67
- Internal Python module edges: 1,758
- Top chokepoint: `core.integration.constants` (degree 182)
- Top cross-domain edge: `tests -> core` (678)

Evidence:

- `artifacts/atlas/cross_module_flow_report.json:12`
- `artifacts/atlas/cross_module_flow_report.json:16`
- `artifacts/atlas/cross_module_flow_report.json:20`
- `artifacts/atlas/cross_module_flow_report.json:500`

## Verified Hidden Flow Pattern

```text
Constitution/Threshold Hub
-> Multi-Engine SNR Dispatch (Rust > v2 > text > fallback)
-> T1 HMM Forecasting (micro-state prediction)
-> T2 Diffusion/GoT Deliberation
-> System2->System1 Skill Compilation (HHMM TTL + structural hash)
-> Hash-Table Integrity Layer (Bloom + Merkle)
-> Evidence Ledger / Receipt Surface
-> Runtime/API visibility + test pressure loop
```

Evidence:

- Threshold hub: `core/integration/constants.py:2`, `core/integration/constants.py:99`, `core/integration/constants.py:135`
- Tri-temporal + diffusion stage model: `core/integration/constants.py:271`, `core/integration/constants.py:289`, `core/integration/constants.py:302`
- Unified SNR facade priority chain: `core/snr_protocol.py:92`, `core/snr_protocol.py:150`
- Node0 VIP pipeline wiring: `scripts/node0_activate.py:127`, `scripts/node0_activate.py:153`, `scripts/node0_activate.py:192`
- T1 HMM purpose and T2 feed: `core/prediction/hmm_engine.py:2`, `core/prediction/hmm_engine.py:8`
- HHMM skill-cache TTL + floor-gated retrieval: `core/hashtable/skill_cache.py:47`, `core/hashtable/skill_cache.py:135`, `core/hashtable/skill_cache.py:173`
- Hash primitives (Bloom/Merkle): `core/hashtable/bloom_filter.py:2`, `core/hashtable/bloom_filter.py:139`, `core/hashtable/merkle_tree.py:2`, `core/hashtable/merkle_tree.py:167`
- Living-memory HHMM promotion: `core/living_memory/core.py:62`, `core/living_memory/core.py:613`, `core/living_memory/core.py:775`
- Runtime noise/signal pressure context: `artifacts/atlas/workspace_masterpiece_report.md:7`, `artifacts/atlas/workspace_masterpiece_report.md:141`

## HHMM Map (Actionable)

### H0 Fast (Reactive)

- T1 cycle and proactive loop constants are explicit.
- HMM micro-state engine predicts next cognitive state.

Evidence:

- `core/integration/constants.py:286`
- `core/integration/constants.py:287`
- `core/prediction/hmm_engine.py:82`

### H1 Deliberative

- SEL "THINK" stage is diffusion reasoning with GoT hypotheses.
- Unified SNR facade routes scoring to strongest available engine.

Evidence:

- `core/integration/constants.py:310`
- `core/snr_protocol.py:150`

### H2 Consolidation

- Skill cache applies HHMM-layer TTL resolution and enforces Ihsan floor.
- Living memory promotes entries by reinforcement thresholds.

Evidence:

- `core/hashtable/skill_cache.py:238`
- `core/hashtable/skill_cache.py:212`
- `core/living_memory/core.py:613`
- `core/living_memory/core.py:628`

### H3 Glacial / Governance

- Canonical threshold source fans out into most critical domains.
- Worktree SNR policy marks governance/security deltas as keep-track.

Evidence:

- `artifacts/atlas/cross_module_flow_report.json:20`
- `artifacts/ops/worktree_snr_report.md:26`

## Hidden Golden Gems (Highest SNR)

1. **Single constitutional control plane**  
   `core.integration.constants` is the top structural dependency chokepoint. This gives tight policy coherence (good) and high blast radius (risk).  
   Evidence: `artifacts/atlas/cross_module_flow_report.json:20`, `core/integration/constants.py:2`

2. **Fail-closed multi-engine SNR routing**  
   Runtime scoring preference is explicit and graceful-fallback aware (`rust > v2+text > v2 > emb+text > emb > text > none`).  
   Evidence: `core/snr_protocol.py:150`, `core/snr_protocol.py:183`, `scripts/node0_activate.py:180`

3. **System2->System1 compression is real, not conceptual**  
   Structural hashes and HHMM TTL are implemented in cache mechanics, with floor-based eviction/rejection.  
   Evidence: `core/hashtable/skill_cache.py:139`, `core/hashtable/skill_cache.py:215`, `core/hashtable/skill_cache.py:173`

4. **Hash-table layer is compositional, not redundant**  
   Bloom handles probabilistic membership/merge; Merkle handles tamper-evident inclusion proofs; both share canonical hashing substrate.  
   Evidence: `core/hashtable/bloom_filter.py:92`, `core/hashtable/bloom_filter.py:139`, `core/hashtable/merkle_tree.py:24`, `core/hashtable/merkle_tree.py:48`

5. **Diffusion reasoning is codified at architecture-constant layer**  
   SEL stages explicitly encode diffusion-based thinking before action and receipt.  
   Evidence: `core/integration/constants.py:302`, `core/integration/constants.py:308`

6. **Test harness is the dominant external pressure on core behavior**  
   Largest inter-domain edge is `tests -> core`, indicating behavior lock-in via regression surfaces.  
   Evidence: `artifacts/atlas/cross_module_flow_report.json:500`

## Noise / Risk (Filtered Out of "Peak Claims")

1. **Global workspace noise still high**  
   Atlas global SNR is 0.719347 due to runtime/build volume dominance.
   Evidence: `artifacts/atlas/workspace_masterpiece_report.md:7`, `artifacts/atlas/workspace_masterpiece_report.md:19`

2. **Diffusion "amplification" currently stronger as policy language than mechanized runtime primitive**  
   Runtime signal amplification exists in `snr_maximizer`, but diffusion-reasoning amplifier is not a single consolidated module boundary yet.
   Evidence: `core/sovereign/snr_maximizer.py:771`, `core/integration/constants.py:310`

3. **HMM training remains deferred**  
   Forecasting exists; full Baum-Welch learning remains explicitly deferred.
   Evidence: `core/prediction/hmm_engine.py:5`, `core/prediction/hmm_engine.py:279`

## Professional Next Moves (SNR-Optimal)

1. Extract a dedicated `diffusion_reasoning_amplifier` runtime interface (single policy boundary).
2. Split top-noise domains (`bizra-omega`, `filedfs`, `personaplex`, `core`) into source/runtime manifests consumed by Atlas.
3. Add cross-engine SNR parity tests (Rust vs v2 vs text) on identical fixtures.
4. Wire HMM learning mode behind explicit feature flag + offline trainer artifact.
