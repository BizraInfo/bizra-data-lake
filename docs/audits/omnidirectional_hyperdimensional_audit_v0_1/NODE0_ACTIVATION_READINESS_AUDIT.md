# Node0 Activation Readiness Audit — BIZRA v0.1

**Gate mapping:** Findings mapped to 5-tier Definition of Done.

- **Tier A — Birth:** Node0 identity sealed.
- **Tier B — Breath:** Node0 emits signed receipts on every effect.
- **Tier C — Body:** Node0 face surfaces authoritative chain head; UX is whole.
- **Tier D — Standing Alone:** Node0 can be used by an external human without operator intervention.
- **Tier E — Future Forest:** Node0 → Genesis 100 cohort path is operational.

---

## Tier A — Birth (sealed identity)

| Gate | Status | Evidence |
|---|---|---|
| Genesis seal exists | ✅ **PASS** | `bizra-omega/bizra-core/src/genesis_seal.rs`; memory `project_node0_sovereign_origin_sealed.md` (BIZRA-641A1D00) |
| Ed25519 keypair bound to seal | ✅ **PASS** | `canonical_receipt.rs` signs with Ed25519 |
| BLAKE3 chaining primitive | ✅ **PASS** | `blake3+rayon` dependency; `previous_receipt_hash` field |
| Cross-language parity (Py/Rust) | ✅ **PASS** | 246 parity tests |

**Tier A verdict:** ✅ **PASS.**

## Tier B — Breath (receipt-native action)

| Gate | Status | Evidence |
|---|---|---|
| Every visible effect emits receipt | ✅ **PASS** | `advance!` macro + state-machine fail-closed transition law |
| Full-body signature | ✅ **PASS** | PR #50 closed weak-signature vuln (per memory `project_mission_receipt_full_payload_signature_2026_04_23.md`) — verify merge |
| Reflex persistence across restart | ✅ **PASS** | `bizra-agent/src/persistence.rs` content-addressed store |
| Receipt chain replay (canonical spearpoint) | ✅ **PASS** | PR #49 (Node0 closure row 4); 38/38 tests green |
| Panic surface on hot path | ⚠️ **WATCHLIST** | 806 `.unwrap()` sites; hot-path audit pending |

**Tier B verdict:** ✅ **PASS** with a watchlist item (panic audit). Not a blocker for activation.

## Tier C — Body (visible surface is whole)

| Gate | Status | Evidence |
|---|---|---|
| Dema face reads authoritative chain head | ✅ **PASS** | `/v1/chain` endpoint + trust-surface binding (memory `project_node0_closure_row6_trust_surface.md`) |
| Honest 503 on gateway down (no shadow state) | ✅ **PASS** | Trust surface binding |
| PAT/SAT canonical topology | ⚠️ **WIRED_PARTIAL** | Canon defined; gateway wiring partial (memory `project_pat_sat_canonical_topology.md`) |
| P2 cockpit receipts surface | ✅ **PASS** | Multiple supporting artifacts |

**Tier C verdict:** ✅ **PASS** with one partial-wiring item (PAT/SAT gateway). Not a blocker for activation if we're not surfacing PAT/SAT in UX.

## Tier D — Standing Alone (external human can use it)

| Gate | Status | Evidence |
|---|---|---|
| Public claim discipline on bizra.ai | ❌ **FAIL** | C4/C5/C7/C9 live without receipts — see `WEBSITE_PUBLIC_CLAIMS_AUDIT.md` |
| Privacy policy published | ❌ **NOT_TESTED** | Not located; needed to back "no telemetry" / "no cloud" claims |
| Node-onboarding runbook | ❌ **MISSING** | No doc covering install → seal → join URP for a new human |
| Minimum-hardware profile | ❌ **MISSING** | — |
| Canon Store Ingestion Gate spec | ❌ **FAIL** | Pack on disk; gate not designed |
| Secret-pattern scanner current state | ✅ **PASS** | `secret_findings.json` has 0 current matches; continuous scanner gate remains security hardening |
| Operator kill-switch documented | ❌ **MISSING** | Needed for paid-ad lane, optional for Tier D |

**Tier D verdict:** ❌ **NO-GO.** Six blockers remain; the current secret-pattern queue is not one of them. Most are short-effort (hours, not days).

## Tier E — Future Forest (Genesis 100 path)

| Gate | Status | Evidence |
|---|---|---|
| Genesis-100 activation plan | ❌ **BLOCKED** | Not authored; GTM directory absent (`docs/gtm/node0_activation_go_to_market_v0_1/` not on disk) |
| Multi-peer federation benchmark | ❌ **NOT_TESTED** | No N=10/100/1000 results published |
| Cost-model receipt | ❌ **NOT_TESTED** | No methodology published |
| SBOM per release | ❌ **MISSING** | Supply-chain attestation absent |

**Tier E verdict:** ❌ **BLOCKED.** Several distinct planning + measurement gaps. None are hard-blocked by code; all are docs + instrumentation.

---

## Critical-path summary

**To reach Tier D GO (external-human-usable public surface):**

1. Close public-claim drift on bizra.ai (C4/C5/C7/C9) — highest leverage.
2. Publish privacy policy OR soften C1/C2.
3. Maintain zero secret-pattern findings; add scanner gate as security hardening.
4. Author node-onboarding runbook + min-hardware profile.
5. Draft Canon Store Ingestion Gate spec (separate lane).
6. Document kill-switch path (for paid ads and security incidents).

**Estimated effort:** 1–3 days of operator time across those six items. No code change required to reach Tier D except possibly the ingestion-gate spec (spec-only, not code).

**To reach Tier E (Genesis 100):**

7. Author GTM Node0 activation plan.
8. Publish at least one benchmark receipt (test-count is easiest first win).
9. Add SBOM generation to release pipeline.
10. Run multi-peer federation benchmark.

Tier E is post-organic-launch territory. Don't block on it.

## Per-tier GO / NO-GO

| Tier | Verdict |
|---|---|
| A — Birth | ✅ GO |
| B — Breath | ✅ GO (watchlist: panic audit) |
| C — Body | ✅ GO (watchlist: PAT/SAT gateway wiring) |
| D — Standing Alone | ❌ NO-GO (6 blockers; secrets currently clean) |
| E — Future Forest | ❌ BLOCKED (planning + measurement gaps) |
