# Scalability Audit — BIZRA v0.1

**Scope:** Node0 → Genesis 100 path; local-node replication; resource assumptions; multi-node future; cost-model claims; queue/concurrency strategy; failure-domain boundaries.

---

## 1. Node0 → Genesis 100 path

**Doctrine:** Node0 is an archetype, not an authority server. Genesis 100 is the first cohort of sovereign nodes — each is its own Node0, not a client of a central Node0.

**Architecture enabling this:**
- `bizra-omega/bizra-node/src/substrate/` — cross-platform resource discovery means any machine can host a sovereign node.
- `bizra-mission` — `AwaitingReconciliation` state permits offline operation with eventual URP reconciliation.
- Receipt chains are per-node; URP is the shared cross-node substrate.

**Assessment:** ✅ **Architecturally scalable by design.** Genesis 100 is a cohort concept, not a hardware requirement.

**Gap:** The *operational* path — how a human installs Node0 on their machine, gets a sovereign identity sealed, joins URP — is not documented end-to-end. This is the onboarding-runbook gap (see `DOCUMENTATION_AUDIT.md §3`).

## 2. Local-node replication

**Observable:**
- Rust binary `bizra-omega/bizra-node/` is the sovereign binary.
- Substrate layer is `#[cfg(target_os)]` gated for Linux / Windows / Android.
- Offline reconciliation is a first-class state in the mission state machine.

**Assessment:** ✅ Model is sound.

**Risk:** public claim "local-only" vs. architecture "cloud-optional" (see `SECURITY_AUDIT.md §6`, `ARCHITECTURE_AUDIT.md §7`). Either reframe or lose honesty.

## 3. Resource assumptions

**Observable from substrate discovery:** CPU, RAM, GPU, disks, local LLM runtimes. No minimum-hardware manifest published.

**Action:** Document minimum-hardware profile for Node0 (CPU cores, RAM, disk). Needed for onboarding runbook + for honest cost claims.

## 4. Multi-node future

**Architecture:**
- `bizra-omega/bizra-federation/` — gossip + signed messages.
- `bizra-omega/iceoryx-bridge/` — zero-copy IPC (intra-host, not multi-node).
- `services/node_gateway/` — REST gateway for multi-node API.

**Assessment:** ✅ Primitives exist. Testing under high-peer-count is not visible in this audit — recommend microbench with N=10, 100, 1 000 peers once topology is settled.

## 5. Cost-model claims

**Public claim:** "cost per action from $0.10 toward $0.008" on bizra.ai.

**Verification state:** NEEDS_REWRITE. No published methodology receipt in this repo.

**Assessment:** Cost-scaling under Genesis-100 growth cannot be credibly claimed until:
1. Per-action compute profile is measured.
2. Local-LLM cost (electricity / opportunity) is estimated.
3. Optional cloud-sync cost is separately itemized.

**Action:** Separate lane — cost-model receipt — required before the $ claim returns.

## 6. Queue / concurrency strategy

**Observable:**
- `bizra-omega/bizra-hooks` — sharded EventBus (8 FNV-1a shards).
- `bizra-omega/bizra-agent` — two-phase OmniKernel (`try_cache_hit` + `complete_cache_hit`).
- Python side has `async` / `asyncio` usage; CLAUDE.md pins `asyncio.run()` as the only correct sync→async bridge.

**Assessment:** ✅ Sharding and two-phase completion are good primitives. Hot-path benchmarks not published.

**Risk:** 806 Rust `.unwrap()` means concurrency-error handling is fragile in some places. Audit hot-paths.

## 7. Failure-domain boundaries

**Observable:**
- Node0 binary is the single failure domain for the sovereign identity.
- Gateway services are optional; failure there does not kill sovereign identity.
- Foundry pipeline is offline from runtime; its failure cannot corrupt runtime canon.
- Canon-pack promotion is idempotent (content-hash deterministic).

**Assessment:** ✅ Good compartmentalization. Single risk: if the ingestion-gate tool is built with write access to MEMORY.md without the human gate, failure-domain leaks — see `ARCHITECTURE_AUDIT.md §9`.

## 8. Scalability debts

| # | Debt | Severity | Action |
|---|---|---|---|
| ScD1 | Node-onboarding runbook (Genesis-100 path) | HIGH | Author end-to-end install → seal → join doc |
| ScD2 | Minimum-hardware profile | MEDIUM | Publish alongside onboarding runbook |
| ScD3 | Multi-peer federation benchmark at N=10/100/1000 | MEDIUM | Dedicated lane |
| ScD4 | Cost-model receipt publication | MEDIUM | Cost-model lane (separate) |
| ScD5 | Hot-path `.unwrap()` audit in concurrency-sensitive crates | MEDIUM | Tech-debt ticket |
| ScD6 | Canon Store Ingestion Gate enforcement (CI guard) | LOW-MEDIUM | CODEOWNERS + diff-check |

---

**Scalability verdict:** architecture permits Genesis-100 cleanly; operational runbooks and honest cost claims are the missing layer. This is docs-debt, not architecture-debt.
