# Production Readiness And GTM Closure Sprint

**Date:** 2026-04-26 GST
**Sprint type:** controlled closure sprint
**Scope:** Node0 health, private-pilot readiness, production stop lines, and GTM execution sequence.
**Non-authorization:** this document does not authorize runtime changes, paid ads, public production federation claims, or Canon Store ingestion.

---

## 1. Executive Call

Node0 is **healthy for single-node operation** and **ready for controlled private-pilot preparation**.

Node0 is **not yet ready for public production federation, paid ads, Genesis-100 scale claims, or open self-serve onboarding**.

The honest next milestone is:

> Run a two-device private pilot where Node0 and one trusted user device exchange signed handshake artifacts, verify them independently, archive both artifacts, and update GTM claims only from that evidence.

---

## 2. Health Evidence Captured This Sprint

| Check | Result | Meaning |
|---|---|---|
| `python3 --version` | Python 3.12.3 | Current interpreter baseline recorded. |
| `python3 -m py_compile scripts/node0_standalone.py tests/scripts/test_node0_standalone.py` | PASS | Pilot CLI and tests parse successfully. |
| `python3 scripts/node0_standalone.py health` | `status=ready`, `ready=true` | All 11 status-determining Node0 lifecycle gates are green. |
| `python3 scripts/node0_standalone.py pilot-doctor` | `status=ready`, `blocking=[]` | Private-pilot readiness docs and local Node0 gates are present. |
| Readiness docs existence check | PASS | GTM, onboarding, hardware, kill-switch, claim discipline, and K4 gate design docs exist. |

Health highlights from `health`:

- Node ID: `node0_ce5af35c848ce889`
- Genesis authority source: `canonical_genesis`
- MVSA: `genesis_hash_valid=true`, `bootstrap_ok=true`, `self_validation_ok=true`
- Mission: last status `COMPLETE`, Ihsan `0.95`, SNR `0.9776676111790307`
- Restart recovery: `restart_recovery_ready=true`

Current local runtime caveats:

- Proactive runtime is not running.
- Desktop bridge is not reachable.
- LM Studio and Ollama are not reachable.
- These are not blockers for private-pilot handshake evidence, but they remain operator-readiness items before a polished live demo.

---

## 3. Production Readiness Gates

| Gate | Status | Evidence / blocker | Current call |
|---|---|---|---|
| Single-node Node0 lifecycle | MEASURED | `health` reports 11/11 gates ready | GO |
| Private-pilot readiness | MEASURED_LOCAL | `pilot-doctor` reports `blocking=[]` | GO for controlled pilot prep |
| Two-device signed handshake | NOT YET MEASURED | Needs real user device reciprocal artifact | NEXT GATE |
| Production URP transport | PLANNED | Heartbeat/prototype exists; bootnode not proven | NO-GO |
| SAT-5 multi-node coordination | PLANNED | Topology canon exists; operational multi-node SAT not proven | NO-GO |
| Public website claim safety | PARTIAL | Paid-readiness checklist still blocks C4/C5/C7/C9 unless removed or receipted | BLOCKER for paid ads |
| Paid ads | NO-GO | Needs claim sign-off, visual QA, landing-page coherence, policy setup | NO-GO |
| Canon Store ingestion | DESIGN_ONLY | K4 design exists; dry-run validator not implemented | NO runtime ingestion |

---

## 4. Sprint Backlog To Close What Remains

### Sprint 1 — Private-Pilot Evidence Closure

Goal: upgrade from `MEASURED_LOCAL_ARTIFACT` to `MEASURED_TWO_DEVICE_PILOT`.

Required actions:

1. Select one trusted user device and record its hardware profile.
2. Run `pilot-doctor` on Node0 immediately before the pilot.
3. Generate a Node0 handshake artifact for the user device.
4. Verify the Node0 artifact on the user device.
5. Generate the reciprocal user-device artifact.
6. Verify the user artifact on Node0.
7. Archive both artifacts, public keys, device profile, operator notes, and restart-recovery notes.
8. Update `PILOT_EVIDENCE_REGISTER.md` with measured results only.

Exit criteria:

- Two artifacts verified independently.
- Tamper rejection still passes.
- No public claim upgraded beyond the measured result.

### Sprint 2 — Public Surface Claim Closure

Goal: make organic public launch safe without paid ads.

Required actions:

1. Compare live website copy against `CLAIM_SAFE_LAUNCH_COPY.md`.
2. Remove, soften, or receipt-link any C4/C5/C7/C9 claims.
3. Publish or defer privacy-sensitive claims such as "no telemetry" or "local-only".
4. Confirm Arabic and English copy carry the same truth label.
5. Record operator sign-off for final launch copy.

Exit criteria:

- Organic launch copy is claim-safe.
- Paid ads remain disabled unless the separate ads checklist is fully green.

### Sprint 3 — Canon Store Gate Dry-Run

Goal: make K4 executable without allowing runtime canon mutation.

Required actions:

1. Implement dry-run-only request validation.
2. Emit dry-run and rejection receipts.
3. Reject one tampered candidate.
4. Keep all runtime targets paused.
5. Add CI-safe invocation once stable.

Exit criteria:

- One valid candidate passes dry-run with no mutation.
- One tampered candidate fails with a rejection receipt.
- No write occurs to `MEMORY.md`, runtime canon, or the Origin Kernel.

### Sprint 4 — GTM Launch Pack Closure

Goal: convert private-pilot evidence into usable operator/investor assets.

Required actions:

1. Update investor handover with two-device evidence once measured.
2. Prepare private pilot invite copy.
3. Prepare first-look video outline from claim-safe copy.
4. Build a pilot scorecard with onboarding time, support issues, and receipt verification pass rate.
5. Keep paid-readiness as a later lane until every paid checklist gate is green.

Exit criteria:

- Private pilot narrative has measured proof.
- Investor/operator handover separates measured, planned, and directional claims.
- GTM materials point to evidence artifacts rather than aspirational numbers.

---

## 5. Stop Lines

Stop immediately if any of the following occur:

- `health` no longer reports `ready=true`.
- `pilot-doctor` reports any blocker.
- A handshake artifact verifies without matching the expected payload digest.
- Any public copy introduces unsupported production-federation, AGI, world-first, financial-return, or security-certification claims.
- Any workflow attempts to ingest `docs/canon/BIZRA_ORIGIN_KERNEL.md` into runtime canon before the Canon Store Ingestion Gate has passed dry-run and ADR promotion.
- Any paid ad is prepared before claim sign-off, visual QA, landing-page coherence, and platform policy review are complete.

---

## 6. GTM Roadmap From Here

| Window | Work | Launch posture |
|---|---|---|
| Now | Private-pilot device selection, Node0 health, K4 design hardening | Internal only |
| Next 48 hours | Two-device signed handshake and evidence archive | Controlled private pilot |
| Next 7 days | Claim-safe organic surface, founder outreach, pilot scorecard | Organic launch only |
| Next 30 days | 2-5 trusted devices, repeatable onboarding, investor update | Private beta |
| After evidence | Paid-readiness checklist, visual QA, privacy copy, platform policy | Paid ads only if all gates green |

---

## 7. Operator Decision

Recommended next typed GO:

```text
GO — two-device private pilot evidence pack
```

That is the highest-signal next move because it upgrades the system from local proof to real cross-device proof without overstating production federation.
