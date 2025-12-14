# ADR-0002: Validator Safety Parameters (N=5, f=1) — Genesis Lock

**Status:** Proposed → (Seal when evidence artifacts exist)  
**Context:** BIZRA Genesis Node requires a mathematically correct, simple BFT envelope for early phases.  

## Decision

We fix **Genesis BFT parameters** to:

- **Total validators:** `N = 5`
- **Fault tolerance:** `f = 1`
- **Safety requirement:** `N >= 3f + 1`  →  `5 >= 4` ✅
- **Quorum for commit/decision:** `Q = 2f + 1 = 3` (i.e., **3-of-5**)  

This aligns with classical PBFT-family thresholds for safety and liveness under partial synchrony, while keeping operational complexity low for Phase 0–1.

## Rationale

- **Sovereign bootstrap:** 5 validators are feasible for a single founder + early council.
- **Byzantine bound clarity:** 1 malicious node is tolerated; ≥2 faults exceed Genesis assumptions and must trigger safety mode.
- **Auditability:** 3-of-5 signatures are easy to inspect, store, and verify.

## Consequences

- If the system detects behaviors consistent with `f >= 2` (e.g., repeated equivocations, double-signing), it must:
  1) enter **safe mode**,  
  2) freeze non-essential state transitions, and  
  3) require a governance-level incident response and re-keying / validator rotation.

- Scaling beyond `N=5` requires a new ADR that re-derives `f`, quorum rules, and network assumptions.

## Evidence Needed to Seal

- A checked-in derivation doc (or PDF) with its own hash
- `evidence/audit-results-node0.json` showing the actual validator keys used for genesis
- A signed tag containing the hashes of the above artifacts
