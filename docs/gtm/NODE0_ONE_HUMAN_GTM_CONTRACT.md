# Node0 One-Human GTM Contract v0.1

**Date:** 2026-05-02 GST
**Status:** `PRIVATE_PILOT_CONTRACT_READY / EXECUTION_GATED`
**Scope:** one human, one Node0 machine, one bounded proof path, one evidence pack.
**Non-authorization:** this contract does not authorize daemon start, Node1 launch, public
federation claims, paid ads, real token operations, autonomous posting, or canon mutation.

---

## 1. Purpose

This contract defines the minimum runnable path for proving BIZRA Node0 to one
human without overstating production readiness.

The GTM claim is intentionally narrow:

> One human can install, inspect, run a bounded local proof path, export evidence,
> and replay verification for a Node0-owned receipt-backed action or refusal.

Anything beyond that is `PLANNED` until separately measured.

---

## 2. Truth Labels

| Label | Meaning | Allowed public language |
|---|---|---|
| `MEASURED_LOCAL` | Reproduced on Node0 or a clean local verification tree. | "Verified locally on Node0." |
| `MEASURED_PRIVATE_PILOT` | Reproduced with one approved human/device and archived evidence. | "Verified in a private pilot." |
| `DERIVED` | Inferred from measured artifacts with explicit reasoning. | "Derived from local/private-pilot evidence." |
| `PLANNED` | Designed but not exercised. | "Planned" or "roadmap." |
| `FORBIDDEN` | Not supported by evidence or policy. | Do not publish. |

Current Node0 GTM truth label: `MEASURED_LOCAL / PRIVATE_PILOT_NEXT`.

---

## 3. Current Evidence Baseline

Validated release base:

| Evidence | Result |
|---|---|
| PR #85 | merged identity-bound Receipt v1 and ADK evidence-preserving refusal |
| PR #86 | merged CLI smoke, CI key-marker hygiene, memory persistence, runtime lifecycle cleanup |
| PR #87 | merged executable scanner policy for secret-shaped `BIZRA_*_HEX` literals |
| `/bizra-verify` | `PASS @ TIER 0.95 PRODUCTION` |
| Spearpoint seal | `b08f2208` reachable |
| Cross-language constants | `ALIGNED` |
| Node0 status | LM Studio connected, token set, 7 PAT agents configured, `proactive_partner` mode |
| Node0 hardware floor | compliant; 32 CPU cores, 125.5 GB RAM, 936.8 GB disk observed |

Important current caveats:

- Rust Bus reported `Not built (PyO3 unavailable)` in the observed status output.
- Evidence ledger reported no runtime evidence yet for the proactive path.
- This contract does not treat those caveats as public-launch blockers, but they
  remain private-pilot evidence items.

---

## 4. GTM Stop Lines

Stop the GTM flow immediately if any condition occurs:

1. `/bizra-verify` does not pass at `TIER 0.95 PRODUCTION`.
2. `python scripts/node0_activate.py status` cannot connect to LM Studio.
3. Required model/token state is missing for the intended demo path.
4. Any command prints a private key, token value, or raw secret.
5. A receipt verifies with an unexpected signer or mismatched public key.
6. A refusal drops evidence references or fabricates content.
7. Any public copy claims production federation, guaranteed safety, investment
   return, AGI, world-first status, or autonomous posting.
8. Any step requires starting Node1, public network discovery, or paid ads.
9. Any operator wants to run a mutating command without an explicit typed GO.

---

## 5. One-Human Proof Path

### Step 0 — Release and Workspace Check

**Goal:** prove the operator is on a verified release base.

```bash
git status --short
git rev-parse --short HEAD
git merge-base --is-ancestor b08f2208 HEAD && echo "SPEARPOINT_OK"
```

Success criteria:

- Working tree is clean or only intended documentation/evidence artifacts are
  staged.
- Spearpoint ancestry is reachable.
- Operator records the commit SHA in the evidence pack.

Truth label: `MEASURED_LOCAL`.

### Step 1 — Environment Check

**Goal:** prove local prerequisites without starting the runtime.

```bash
source .venv/bin/activate
python --version
python scripts/node0_activate.py status
pytest tests/integration/test_autonomous_pilot.py -q
```

Success criteria:

- Python is 3.11+.
- Node0 status reports LM Studio connected or a documented diagnostic blocker.
- Smoke suite passes all 8 pillars:
  `RuntimeBoot`, `TokenSystem`, `EvidenceChain`, `SNR`, `SpearPoint`,
  `OpportunityPipeline`, `CLI`, and `FullStack`.

Truth label: `MEASURED_LOCAL`.

### Step 2 — Identity and Receipt Key Check

**Goal:** prove the demo uses identity-bound receipts, not self-declared authority.

Required evidence:

- Registry lookup for the intended signer succeeds.
- Unknown signer is rejected.
- Revoked signer is rejected.
- Mismatched signer/key pair is rejected.
- Tampered receipt hash and tampered signature are rejected.

Validation command:

```bash
pytest tests/core/mission_kernel tests/core/proof_engine/test_receipt.py -q
```

Success criteria:

- Identity-bound receipt tests pass.
- Direct bootstrap-only receipt creation is not used as production authority.

Truth label: `MEASURED_LOCAL`.

### Step 3 — DEMA Semantic Boundary Check

**Goal:** prove semantic inputs are validated before any action/refusal path.

```bash
pytest tests/core/dema tests/core/adk/test_researcher.py -q
```

Success criteria:

- Raw parsed claims remain untrusted.
- Validated claims are deterministic inputs to FATE/receipt paths.
- Backend failure refusal preserves `evidence_refs`.
- Client-facing refusal reason is sanitized.
- No fabricated answer content is returned.

Truth label: `MEASURED_LOCAL`.

### Step 4 — Relief Pre-Start

**Goal:** show the human can inspect readiness without daemon activation.

Run only non-mutating diagnostics:

```bash
python scripts/node0_activate.py status
test -f sovereign_state/proactive.pid && cat sovereign_state/proactive.pid || true
ls -la logs/proactive 2>/dev/null || true
```

Success criteria:

- No stale PID is present, or stale PID is classified and removed only after
  explicit operator approval.
- Logs are present when daemon mode was previously used; absence is not a
  blocker for diagnostics-only mode.
- Operator can explain whether Node0 is `running`, `stopped`, or `degraded`.

Truth label: `MEASURED_LOCAL`.

### Step 5 — One Bounded Task Demo

**Goal:** produce one useful bounded output with proof discipline.

Allowed demo task shape:

```text
Summarize one local evidence artifact and return:
1. what it proves,
2. what it does not prove,
3. evidence references,
4. a refusal if proof is insufficient.
```

Execution path is intentionally confirmation-gated:

```bash
python scripts/node0_activate.py mission "<bounded local proof task>"
```

Success criteria:

- Operator gives explicit typed GO before this mutating command.
- Output is evidence-bound.
- If the backend fails, refusal preserves gathered evidence references.
- No raw transport error becomes the primary client-facing reason.
- Any action/refusal receipt is identity-bound.

Truth label after success: `MEASURED_LOCAL_DEMO`.

### Step 6 — Receipt Export

**Goal:** archive exactly what happened.

Evidence pack fields:

| Field | Required |
|---|---|
| `commit_sha` | yes |
| `operator_id` | yes |
| `node_id` | yes |
| `signer_id` | yes |
| `signer_public_key_fingerprint` | yes |
| `task_text` | yes |
| `result_type` | `action` or `refusal` |
| `evidence_refs` | yes, can be empty only for preflight-only runs |
| `receipt_id` | yes if a receipt was emitted |
| `receipt_hash` | yes if a receipt was emitted |
| `previous_hash` | yes when available |
| `verification_result` | yes |
| `operator_notes` | yes |

Recommended artifact path:

```text
artifacts/gtm/node0_one_human/<YYYYMMDD-HHMMSS>/
```

Do not commit private keys, tokens, local `.env` files, or raw secret material.

Truth label: `MEASURED_LOCAL`.

### Step 7 — Proof Replay

**Goal:** prove the evidence pack can be checked after the demo.

Replay checks:

```bash
pytest tests/core/mission_kernel tests/core/proof_engine/test_receipt.py -q
python scripts/ci_secret_scan.py
```

If a Proof Forge receipt is generated, verify the chain with the current
Proof Forge tooling and record the chain hash in the evidence pack.

Success criteria:

- Receipt verification passes.
- Tamper rejection still passes.
- Secret scan passes.
- Evidence pack can be explained without relying on hidden runtime state.

Truth label: `MEASURED_LOCAL_REPLAYED`.

### Step 8 — Public Demo Package

**Goal:** produce claim-safe materials, not a hype launch.

Required package:

1. One-page human explanation.
2. 3-minute demo script.
3. Evidence pack index.
4. Redacted receipt proof.
5. Known limitations.
6. Stop-line list.
7. Private-pilot invitation copy.

Forbidden package contents:

- "fully autonomous"
- "cannot fail"
- "production federation live"
- "guaranteed safety"
- "investment return"
- "AGI"
- "world first" unless externally certified

Truth label: `PUBLIC_DRAFT_READY`, not `PUBLIC_READY`.

---

## 6. Go / No-Go Matrix

| Gate | GO condition | NO-GO condition |
|---|---|---|
| Release base | `/bizra-verify` passes at TIER 0.95 | any verify gate fails |
| Runtime status | LM Studio reachable, token set, PAT configured | unreachable model gateway for demo path |
| Smoke suite | 8/8 pilot pillars pass | any deterministic smoke failure |
| Identity | registry-bound signer verifies | unknown/revoked/mismatched signer accepted |
| Refusal | evidence-preserving sanitized refusal | fabricated content or raw transport reason exposed |
| Receipt | receipt verifies and tamper rejection passes | receipt verifies under wrong key or tampered body |
| Evidence pack | complete, redacted, replayable | missing receipt/proof metadata or secret material present |
| Public copy | truth-labeled and limitation-aware | unsupported production/federation/financial claims |

The first `NO-GO` blocks the GTM flow.

---

## 7. Operator Runbook

### Diagnostics-only command set

These are safe to run without extra approval:

```bash
python scripts/node0_activate.py status
pytest tests/integration/test_autonomous_pilot.py -q
pytest tests/core/mission_kernel tests/core/proof_engine/test_receipt.py -q
python scripts/ci_secret_scan.py
```

### Confirmation-gated command set

These require explicit typed operator approval immediately before execution:

```bash
python scripts/node0_activate.py start
./scripts/start_proactive.sh --mode proactive_partner --config config/proactive_config.yaml
./scripts/stop_proactive.sh
python scripts/node0_activate.py mission "<bounded local proof task>"
```

Approval phrase:

```text
GO — Node0 one-human bounded proof demo
```

---

## 8. Interdisciplinary Lens Synthesis

| Lens | Current call |
|---|---|
| Systems | GO for diagnostics; mutating mission remains confirmation-gated. |
| Reliability | GO for local smoke and replay; daemon lifecycle not started by this contract. |
| Security | GO only with redacted evidence pack and `ci_secret_scan.py` pass. |
| Economics | GO for bounded local demo; no token/paid/financial claim. |
| Ethics / Ihsān | GO if `/bizra-verify` remains TIER 0.95 and Daughter Test passes. |
| Operations | GO for one operator runbook; no background daemon unless explicitly approved. |
| Product impact | GO for private-pilot narrative; NO-GO for public production claims. |

---

## 9. GoT Path Summary

Three GTM paths were considered:

| Path | Verdict | Reason |
|---|---|---|
| Public launch now | rejected | evidence is local/private-pilot prep, not production federation |
| Two-device pilot now | next milestone | requires an approved second device and reciprocal artifact |
| One-human proof contract now | selected | highest SNR: converts merged proof spine into a runnable demo without scope creep |

Converged path:

```text
Release-verified Node0
→ diagnostics-only readiness
→ identity-bound proof/refusal checks
→ confirmation-gated bounded task
→ receipt export
→ replay
→ claim-safe private-pilot package
```

---

## 10. Giants Provenance

Standing on Giants:

- Shannon — SNR discipline and signal/noise separation.
- Boyd — OODA-style observe/orient/decide/act runtime loop.
- Deming — PDCA quality gate and replay-first correction.
- Lamport — identity/order/verification discipline for distributed trust.
- Al-Ghazali — Ihsān ethics as an operating gate, not decoration.

Repo anchors:

- `scripts/node0_activate.py`
- `tests/integration/test_autonomous_pilot.py`
- `core/mission_kernel/bridge.py`
- `tests/core/mission_kernel/test_bridge.py`
- `core/adk/agents/researcher.py`
- `scripts/ci_secret_scan.py`
- `docs/gtm/node0_activation_go_to_market_v0_1/PRODUCTION_READINESS_AND_GTM_CLOSURE_SPRINT.md`

---

## 11. Final Contract Verdict

```text
NODE0 ONE-HUMAN GTM CONTRACT: READY FOR PRIVATE-PILOT EXECUTION
Truth label: MEASURED_LOCAL / EXECUTION_GATED
Next proof upgrade: MEASURED_PRIVATE_PILOT after one approved human/device replays the evidence pack
```

The next typed GO is:

```text
GO — Node0 one-human bounded proof demo
```

Until that GO is given, this contract authorizes diagnostics, documentation,
and evidence-pack preparation only.
