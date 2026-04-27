# Dema GTM Masterplan v0.1

**Date:** 2026-04-27 GST
**Status:** PLANNED
**Author:** Mumu (founder) + Claude Code (implementer)
**Scope:** Phased execution plan to ship BIZRA-Dema from current code-space
state to a private-pilot-ready GTM posture.
**Non-authorization:** This doc does NOT authorize public launch, paid ads,
real-token mint, public network exposure, autonomous social posting, or any
canon mutation. Each phase ships with its own scoped PR and explicit rules.

---

## §1 Dema Product Truth

Dema is **not an app the user opens**. Dema is the **always-on sovereign
operating layer of the human node**.

The killer model — applied uniformly across life, work, learning, finance,
mission execution, and node evolution — is the four-state universal control
loop named in the BIZRA Origin Manifest §9:

```
Current State → Ideal State → Gap → Next Admissible Action
```

Every meaningful action that flows through Dema must be:

1. **Consent-aware** — the human approves before Dema touches anything that
   crosses a boundary (network, file, identity, social).
2. **Receipt-backed** — every meaningful action emits a hash-chained,
   Ed25519-signed receipt with truth-label metadata.
3. **Truth-labeled** — `MEASURED`, `DERIVED`, `PLANNED`, or `SANDBOX`. No
   claim ships without a label.
4. **FATE-gated** — Ihsān ≥ 0.95, SNR ≥ 0.85, ZANN_ZERO, RIBA_ZERO. Failures
   reject the action; they do not silently pass.

Dema's voice: pragmatic-mystic. Companion / mirror / disciplined guide. **Not**
chatbot, hype engine, manipulative guru, cold auditor, or oracle.

Public-facing names: **BIZRA, Dema, node, sovereign intelligence, verified
action, Ihsān**. Runtime names (URP, PAT, SAT, Node0, MoMo, FATE, membrane)
stay internal. Brand canon §15 forbids exposing those on consumer surfaces.

---

## §2 Current Shipped Base (MEASURED)

Verified on `main @ cb7db266` (2026-04-27):

| Capability | Evidence | Truth label |
|---|---|---|
| Mission submit → signed receipt | `terminal-mission.tsx` + `/v1/plan` (live) | MEASURED |
| Chain head + receipt trust surface | `chain-trust-surface.test.tsx` 5/5 (PR #51) | MEASURED |
| Ihsān/SNR/Gini gate enforcement | `terminal-shell.tsx:129–142` + `/v1/health` | MEASURED |
| URP_LOCAL_ACTIVE proof artifacts | `artifacts/proofs/node0-local-urp/` (PR #59) | MEASURED |
| 9 URP contract tests + CI skip guard | `tests/scripts/test_node0_local_urp_proof.py` | MEASURED |
| Node0 11/11 lifecycle gates ready | `node0_standalone.py health` | MEASURED |
| Pilot-doctor `truth_label: MEASURED`, `blocking: []` | `node0_standalone.py pilot-doctor` | MEASURED |
| Coverage floor 60 (post-marker hygiene) | `pyproject.toml:151` | MEASURED |
| Trust-surface stale-cleanup behavior | `useFetch` ref guard (PR #51) | MEASURED |
| SAT canon evolution note | `docs/canon/SAT_CANON_NOTE_v0_1.md` (PR #59) | DERIVED |

**The base is stronger than expected.** GTM gap is in *surfacing* what's
already real, plus closing the two-device handshake.

---

## §3 GTM Phases (A → E)

### Phase A — Dema Founder OS (single-node, on Node0)

| Sub-phase | Deliverable | Key files |
|---|---|---|
| **A0** Ambient Kernel | onboarding, daily append-only logs, dream phase, mission state machine, local receipts | `scripts/dema/dema_*.py` + `core/dema/` |
| **A1** Goal Surface | the §9 four-state UI (Current/Ideal/Gap/Next) | `frontend/src/components/terminal/terminal-goal.tsx` |
| **A2–A5** Proof Surface | evidence-auditor visibility, claim+source flow, sovereignty reveal, signed receipt export | `frontend/src/components/terminal/` + receipt panel |
| **A6** Memory Import | private import of historical chats + Downloads with consent gates | `scripts/dema/dema_import_memory.py` |
| **A7** Desktop Actuator | tier-gated open_app/open_url with approval + receipts; AHK design doc | `scripts/dema/dema_desktop.py` + `docs/product/DEMA_AHK_*.md` |

**Truth label after A:** `MEASURED_LOCAL_DEMO`.

### Phase B — Two-Device Pilot (Node0 ↔ Node1)

Build the missing peer-side primitive. Node1 = friend's Asus VivoBook.

- `scripts/pilot/node1_kit.py` — Node1 keypair + hardware profile + reciprocal handshake
- `scripts/pilot/verify_peer_handshake_chain.py` — Node0-side chain verifier
- OOB pubkey ceremony (operator-mediated, never over the same channel)
- VivoBook hardware profile vs `MINIMUM_HARDWARE_PROFILE.md`
- Operator kill-switch dry-fire (test, not just doc)
- One row in `PILOT_EVIDENCE_REGISTER.md`

**Truth label after B:** `MEASURED_TWO_DEVICE_PILOT`.

### Phase C — Brand v0.2 Public Alignment (parallel, doc-only)

- Refactor public landing per brand canon §18
- Redact runtime names (URP/PAT/SAT/Node0) from public surface
- Replace deck slide 12 with canonical two-zone topology (per kit-review H1)
- Add truth-label badges to every public claim
- Fix kit-review H1–H4 (DDCAGI removal, Guardian disambiguation, etc.)

**Truth label after C:** `PUBLIC_SURFACE_CLAIM_SAFE`.

### Phase D — Public Launch Pack (drafts only, no posting)

- `docs/gtm/BIZRA_PUBLIC_PAPER_V0_1.md` (EN + AR)
- `docs/gtm/BIZRA_ONE_PAGE_SUMMARY_V0_1.md`
- `docs/gtm/BIZRA_SOCIAL_POSTS_V0_1.md` (drafts, **never auto-post**)
- `docs/gtm/BIZRA_FIRST_LOOK_VIDEO_SCRIPT_V0_1.md`
- `docs/gtm/BIZRA_GTM_CLAIM_REGISTER_V0_1.md` (every claim → label + evidence + allowed/forbidden wording)

**Truth label after D:** `PUBLIC_DRAFTS_READY` (still not published).

### Phase E — Release Gate

- `docs/gtm/DEMA_PRIVATE_PILOT_RELEASE_GATE_V0_1.md` checklist
- `artifacts/gtm/dema_private_pilot_gate_report.json` machine-readable
- Verdict: `BLOCKED | PRIVATE_PILOT_READY | PUBLIC_READY`
- `PUBLIC_READY` requires C + D complete + founder explicit sign-off

---

## §4 Non-Negotiable Rules (claim discipline)

### No-token-before-proof

- No real, tradable, or financialized token operations in any phase.
- SEED/BLOOM remain `POI_SANDBOX` with `monetary_value: none` until
  measured multi-node evidence + legal review exist.
- Reward language is `sandbox_xp | bloom_credit | future_token_eligibility`
  — never "earnings," "yield," "return," or "investment."

### Human-in-loop social

- Phase D produces social *drafts* only.
- No autonomous posting to X / LinkedIn / GitHub Discussions / Telegram /
  Discord without explicit founder approval per post.
- Founder approval recorded as a receipt with `action: social_publish`,
  `approval_required: true`, `approval_status: granted_by_<id>`.

### Proof / truth-label policy

Every public claim carries one of:
- `MEASURED` — observable, replayable, evidence pack exists
- `DERIVED` — inferred from MEASURED inputs; the inference is explicit
- `PLANNED` — design intent; not yet exercised
- `SANDBOX` — measured but valueless (PoI, sandbox tokens, simulation)

Forbidden in any public surface (per brand canon §15):
- "guaranteed," "AGI achieved," "risk-free," "cannot lie," "cannot fail"
- "first in the world" (unless externally certified)
- exact Ihsān / SNR / latency numbers (until verified by external party)
- token-profit framing of any kind

### Repo discipline

- One phase = one branch = one PR = wait for deterministic checks
- No mega-PRs
- No editing `MEMORY.md` without typed authorization
- No rewriting canon docs (`docs/canon/*`) without a separate review pass
- No `--force` push to `main` ever; only `--force-with-lease` on feature
  branches when explicitly authorized

---

## §5 Acceptance Criteria Per Phase

### A0 — Ambient Kernel

- [ ] `dema_onboarding.py --init` writes a profile receipt locally
- [ ] `dema_status.py --json` returns Current/Ideal/Gap/Next data shape
- [ ] `dema_dream.py --read-only --max-seconds 15` produces candidate
      memory notes without auto-promotion
- [ ] `dema_daemon.py --once` emits one daily-log line and one receipt
- [ ] All artifacts under local-only / gitignored paths
- [ ] `pytest tests/scripts/test_dema_ambient_kernel.py` passes 100%
- [ ] Ruff clean

### A1 — Goal Surface

- [ ] Goal view renders under existing Dema terminal navigation
- [ ] Real data from `/v1/plan` + `/v1/health` + `/v1/chain/latest` when
      available; truth-labeled placeholder otherwise
- [ ] No fake metrics
- [ ] Frontend typecheck + lint + vitest all green

### A2–A5 — Proof Surface

- [ ] Mission receipt detail shows evidence-auditor verdict
- [ ] Claim + source form blocks submission on missing source
- [ ] Sovereignty reveal animates only on a real receipt land
- [ ] Receipt export downloads valid signed JSON
- [ ] Replay with public key verifies the exported receipt

### A6 — Memory Import

- [ ] Read-only scan; never moves or deletes source files
- [ ] Outputs under `.sovereign/dema/imports/<run_id>/` (gitignored)
- [ ] No raw chat content in any committed artifact
- [ ] Approval boundary before promotion to long-term memory

### A7 — Desktop Actuator

- [ ] All actions tier-labeled (0–4)
- [ ] Tier ≥ 2 actions require explicit approval before execution
- [ ] Dry-run mode prints intended action without executing
- [ ] Every executed action emits a receipt
- [ ] No credentials, no destructive commands, no autonomous social

### B — Node1 Kit

- [ ] Node1 keypair generated **on the VivoBook**, not on Node0
- [ ] OOB pubkey exchange ceremony documented + executed
- [ ] Node0 verifies a Node1-signed reciprocal handshake referencing
      Node0's `receipt_hash` via `previous_receipt_hash`
- [ ] Tampered Node1 artifact rejected by Node0
- [ ] Both devices pass restart-recovery
- [ ] `PILOT_EVIDENCE_REGISTER.md` updated with one row, truth-labeled

### C — Brand v0.2 Public Alignment

- [ ] Public landing strings audited against canon §15
- [ ] Runtime names (URP/PAT/SAT/Node0) absent from public surface
- [ ] Deck slide 12 rebuilt against canonical topology
- [ ] Truth-label badges on all public claims

### D — Public Launch Pack

- [ ] All 6 launch-pack files committed to `docs/gtm/`
- [ ] Claim register row exists for every public statement
- [ ] Forbidden-phrase grep returns zero hits
- [ ] No file marked `PUBLIC_READY` without founder sign-off

### E — Release Gate

- [ ] All Phase A items merged or explicitly deferred
- [ ] Phase B merged or explicitly scheduled
- [ ] Phase C merged
- [ ] Phase D drafts ready
- [ ] Verdict written to `artifacts/gtm/dema_private_pilot_gate_report.json`

---

## §6 Risks + Kill Conditions

| Risk | Kill condition |
|---|---|
| Phase exceeds scope | If a phase grows beyond its named files / 3× initial estimate, **STOP**, reduce to smallest demonstrable slice. No architecture expansion. |
| Public claim slips past truth-label discipline | Block PR; rewrite per canon §15. |
| Token-economy language leaks into Phase D | Reject; rewrite as `sandbox` or `future eligibility`. |
| Memory import touches anything outside its sandbox folder | Halt; quarantine; investigate before resume. |
| Desktop actuator triggers without approval | Disable the actuator until policy hardening lands. |
| Node1 kit ships with private-key generation on Node0 | Revert; the kit is invalid if the key isn't local to Node1. |

---

## §7 Recommended Phase Order

```
Phase 0 (this PR)             ← GTM masterplan (you are here)
Phase A0  Ambient Kernel
Phase A1  Goal Surface
Phase A2-A5  Proof Surface
Phase A6  Memory Import
Phase A7  Desktop Actuator
Phase B   Node1 Kit
Phase C   Brand v0.2 (parallel to A6/A7/B if doc-only)
Phase D   Public Launch Pack (drafts)
Phase E   Release Gate
```

Each phase opens its own branch, its own PR, waits for deterministic CI
gates, and merges only on founder approval per the locked `merge PR #N`
discipline.

---

## §8 Definition of Done (GTM)

```
Dema starts with the device.
Dema remembers with consent.
Dema accepts a task from MuMu.
Dema frames Current/Ideal/Gap/Next.
Dema acts safely on the local machine.
Dema produces a receipt.
Dema prepares public content without unsupported claims.
Dema can onboard Node1 through the private pilot kit.
Dema turns internal gaps into Forge tasks with PoI sandbox scoring.
```

When all nine lines are MEASURED, the release gate may issue
`PRIVATE_PILOT_READY`. `PUBLIC_READY` requires the same plus founder
sign-off + claim-register completeness.

---

## §9 References

| Source | Role |
|---|---|
| `reference_bizra_topology_canon_frozen_2026_03_25` (memory) | Topology authority — PAT-7/SAT-5 names, ONE URP, membrane |
| `docs/canon/SAT_CANON_NOTE_v0_1.md` | SAT placement evolution note |
| `docs/gtm/node0_activation_go_to_market_v0_1/PRODUCTION_READINESS_AND_GTM_CLOSURE_SPRINT.md` | Pre-existing closure sprint |
| `docs/gtm/node0_activation_go_to_market_v0_1/USER_NODE_ONBOARDING_RUNBOOK.md` | Friend-side onboarding flow |
| `docs/gtm/node0_activation_go_to_market_v0_1/NODE0_PRIVATE_PILOT_PLAN.md` | Pilot scope and exit criteria |
| `docs/gtm/node0_activation_go_to_market_v0_1/PILOT_EVIDENCE_REGISTER.md` | Where measured pilot results land |
| `/home/bizra-operating-system/Downloads/BIZRA_Killer_Product_Strategy_and_Money_Shot.pdf` | Engineering positioning thesis |
| `/home/bizra-operating-system/Downloads/bizra_brand_identity_canon_v_0.md` (v0.2) | Brand canon authority |
| `/home/bizra-operating-system/Downloads/03-master-deck.md` | Outward narrative arc |
| `/home/bizra-operating-system/Downloads/06-kit-review.md` | Self-adversarial pass (4 high findings) |
| PR #51 (merged) | CI hygiene + trust-surface useFetch refs |
| PR #59 (merged) | Node0-local URP proof v0.1 (`URP_LOCAL_ACTIVE`) |
| PR #60 (open) | SAT canon note doc |

---

## §10 Bounds

This document carries **no AGI guarantee**, **no token-value claim**, **no
public production federation claim**, and **no public-launch authorization**.
Every claim above is either MEASURED on `main @ cb7db266`, DERIVED from
those MEASURED inputs, or PLANNED with a named sub-phase that produces its
own receipt before the claim escalates.

If any part of this masterplan conflicts with the BIZRA Topology Canon
(2026-03-25, Mumu-signed), the Origin Manifest, or the Brand Canon v0.2,
**those canonical sources win** and this masterplan must be amended.
