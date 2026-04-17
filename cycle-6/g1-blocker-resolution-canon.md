# BIZRA Agent Instruction — Cycle-6 G1 Blocker Resolution Canon

بسم الله الرحمن الرحيم

**Filed:** 2026-04-17 (Friday), Dubai GST
**Status:** ACTIVE — blocker-resolution window for Cycle-6 G1
**Authority:** Founder decision `/@ C`, 2026-04-17 14:31 GST

## Status

Cycle-6 G1 code is ACTIVE but currently HALTED on an evidence-found blocker.

## Authority

Derived from:
- `cycle-6/execution-canon.md`
- `cycle-6/g1-authority-adr.md`
- `cycle-6/g1-execution-brief.md`
- Founder decision for the current blocker resolution path: `/@ C`

## Current blocker truth

Do not implement file-hash verification from guesswork.

**Empirical result already established:**
- 4-entry × 3-algorithm verification matrix (SHA-256 raw, BLAKE2b-32 raw, SHA-256 canonical JSON)
- 12 tests, 0 matches

Therefore:
- ADR §Bootstrap flow step 4 ("verify referenced receipt file hash") is not implementable yet
- ADR §Bootstrap flow step 5 ("assert computed chain head equals `block_zero.receipt_chain.chain_hash`") is also not valid as currently assumed
- `block_zero` and live `activation_chain_*.json` are not the same chain surface (zero hash overlap between their respective receipt lists)

## Mandatory next action

Read the Python writer that produces `sovereign_state/` receipts and chain envelopes.

## Files to inspect first

1. `core/sovereign/`
2. `core/sovereign/atomic_io*`
3. `core/sovereign/organism.py`
4. `core/sovereign/mission.py`
5. `core/sovereign/cel.py`
6. Any writer touching:
   - `sovereign_state/receipts/`
   - `sovereign_state/block_zero/`
   - `activation_chain_*.json`

## Goal of the read

Determine, with evidence:

1. What exact hash algorithm is used
2. What exact content is hashed
3. Whether content is raw file bytes, canonical JSON, subset fields, or domain-specific serialization
4. How live activation-chain envelopes relate to `block_zero` genealogy
5. Whether Rust can reproduce this exactly in G1 Phase 1

## Hard rules

- No speculative Rust hash implementation
- No "likely patched / likely same" style inference for cryptographic semantics
- No widening scope beyond G1
- No signer audit in this pass
- No threshold drift check in this pass
- Fail closed if semantics cannot be reproduced confidently

## Decision rule after Python read

### If the Python writer is simple and reproducible

- Implement compatible Rust verification in G1 Phase 1
- Preserve `CognitionRuntime::from_sovereign_state(&Path)` path
- Keep read-only projection model

### If the Python writer is complex / domain-specific / under-documented

- Do NOT fake equivalence
- Amend G1 Phase 1 to envelope-internal linkage only
- Defer full file-hash verification and `block_zero` reconciliation to Cycle-6.5c

## Deliverable format

Return with:
1. Writer files identified
2. Exact hash/content rule found, or explicit failure to derive it
3. Whether G1 can proceed with full verification or must narrow
4. Exact proposed code scope for the next commit

## Constitutional filter

All work must preserve:
- **IHSAN_FLOOR**
- **ZANN_ZERO**
- **RIBA_ZERO**
- **CLAIM_MUST_BIND**
- **NO_SHADOW_STATE**

## Default implementation posture

Evidence first.
Then interface.
Then code.
Never the reverse.

---

الحمد لله.
