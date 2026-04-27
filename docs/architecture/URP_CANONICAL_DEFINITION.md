# URP Canonical Definition

**Status:** canonical documentation guardrail  
**Date:** 2026-04-26 GST  
**Scope:** architecture, audit, public-claim discipline, and future runbooks.

---

## Canonical Expansion

**URP = Universal Resource Pool.**

URP is the shared constitutional resource substrate where sovereign nodes
contribute, discover, allocate, verify, and reconcile resources without turning
Node0 into a central authority server.

Canonical sentence:

> Node0 proves a sovereign seed can live alone; the Universal Resource Pool lets
> sovereign seeds coordinate resources without surrendering authority.

---

## Boundary Rules

1. **Use "Universal Resource Pool"** in architecture audits, public-proof docs,
   cross-node resource coordination, reconciliation, and claim discipline.
2. **Use "receipt plane" descriptively, not as the URP expansion.** Receipts are
   how resource events are made verifiable inside the broader Universal Resource Pool.
3. **Do not use URP for unrelated protocol names.** "Universal Resource
   Protocol", "Universal Rights Protocol", and "Universal Reasoning Protocol"
   are historical aliases unless a future ADR explicitly reassigns them.
4. **When quoting old docs, preserve history but add a redirect.** The redirect
   should say: "historical URP wording; canonical URP is Universal Resource
   Pool."

---

## Proof-of-Truth Role

| Lane | URP role |
|---|---|
| Formal | Defines the cross-node state boundary: `AwaitingReconciliation -> UrpValidating -> Complete`. |
| Cryptographic | Makes contributed resources and cross-node events verifiable through receipt hashes, signatures, previous-receipt links, and verifier identity. |
| Empirical | Gives audits a stable target for DOM captures, chain checks, federation tests, and reconciliation runs. |
| Economic | Prevents public claims from selling "network scale" before receipt exchange and reconciliation are measured. |

---

## Historical Aliases

| Historical wording | Disposition | Use now |
|---|---|---|
| Universal Receipt Plane | Historical alias | Say "receipt layer inside the Universal Resource Pool" unless quoting old material. |
| Universal Resource Protocol | Historical alias | Redirect to Universal Resource Pool unless quoting old material. |
| Universal Rights Protocol | Historical alias | Do not use without a new ADR. |
| Universal Reasoning Protocol | Historical alias | Do not use without a new ADR. |

---

## Code And Evidence Anchors

- `bizra-omega/bizra-mission/src/state.rs` defines reconciliation states.
- `bizra-omega/bizra-mission/src/receipt.rs` defines mission receipt chain and signature verification.
- `bizra-omega/bizra-resourcepool/` implements the resource-pool subsystem.
- `docs/audits/omnidirectional_hyperdimensional_audit_v0_1/PROOF_OF_TRUTH_CONVERGENCE_MAP.md` defines the current convergence map.
- `tools/audit/omni_audit/urp_canonicality.py` detects drift in URP expansion.

---

## Maintenance Rule

If a document introduces a new URP expansion, it must either:

1. update this file through an ADR-backed canon change, or
2. mark the wording as a historical alias and point back here.
