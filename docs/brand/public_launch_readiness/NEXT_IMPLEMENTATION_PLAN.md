# Next Implementation Plan — BIZRA Public Launch v0.1

**Plan date:** 2026-04-24 (GST)
**Scope:** Concrete, sequenced next steps to move from "media kit landed in repo + website claims audited" to "safe public launch." Every step is gated by typed operator authorization.

---

## Task list (sequenced)

### P0 — Bulletproofing today (highest ROI)

| # | Task | Effort | Blocker | Owner |
|---|---|---|---|---|
| P0.1 | Visual spot-check of 12 `rendered_concepts/` PNGs for small-text / Arabic-shaping / color-drift defects | 30 min | none | operator |
| P0.2 | Visual spot-check of 11 `ready_to_post/` raster exports at final post size | 20 min | none | operator |
| P0.3 | Human Arabic-speaker pass on all AR copy in `CLAIM_SAFE_LAUNCH_COPY.md` | 30 min | none | operator or Arabic-speaking reviewer |
| P0.4 | Operator sign-off on `CLAIM_SAFE_LAUNCH_COPY.md §1 / §2 / §3 / §4` | 15 min | P0.3 | operator |
| P0.5 | Headless-Chromium DOM capture of `bizra.ai` to replace pre-check findings with verified snapshot | 30 min | none | operator (or script) |

P0 is all low-effort, high-value, pre-publishing. Closes 4 of 5 open items in HANDOFF_NOTES.

### P1 — Website claim cleanup (blocking all paid ads)

| # | Task | Effort | Blocker | Owner |
|---|---|---|---|---|
| P1.1 | Locate the bizra.ai source repo (separate from bizra-data-lake) | 5 min | none | operator |
| P1.2 | Decide for each C-class claim (C4/C5/C7/C9): remove, rewrite with link, or publish receipt | 1 hour | P1.1, P0.5 | operator |
| P1.3 | If "publish receipt" on any claim: specify what receipt looks like, where it lives, how it's updated | 2 hours | P1.2 | operator + technical lead |
| P1.4 | Implement site copy changes in bizra.ai repo | 2–4 hours | P1.2 | web lead |
| P1.5 | If C9 is kept: wire live early-access counter backed by a source of truth | 4–8 hours | P1.2 | web lead |
| P1.6 | Set OG tags in bizra.ai shell HTML so social link previews render | 30 min | none | web lead |
| P1.7 | Publish or retire privacy policy (decides C1 / C2 classification) | 2–4 hours | P1.2 | operator + legal-adjacent review |
| P1.8 | Deploy + verify from a separate device (not cached) | 30 min | P1.4–P1.7 | web lead |

P1 is the biggest single blocker for paid advertising. Can't be completed in this repo because bizra.ai source is elsewhere.

### P2 — Organic rollout (Phases 1–3 from PROFILE_ROLLOUT_CHECKLIST)

| # | Task | Effort | Blocker | Owner |
|---|---|---|---|---|
| P2.1 | Render all SVG templates to PNGs at platform-specific sizes | 30 min | P0.1–P0.2 | creative lead |
| P2.2 | Claim/verify handles on X, LinkedIn, IG, Threads, YouTube | 1 hour | none | operator |
| P2.3 | Upload avatar/cover/header + bio to each platform (silent, no post yet) | 1 hour | P2.1, P2.2, P0.4 | operator |
| P2.4 | Phase 2 — launch post across platforms | 30 min coordinated | P2.3, P1.8 (ideally) | operator |
| P2.5 | Phase 3 — day 2–7 support posts | 1 hour/day | P2.4 | operator |

### P3 — Paid ads (gated by ADS_READINESS_CHECKLIST)

| # | Task | Effort | Blocker | Owner |
|---|---|---|---|---|
| P3.1 | All 6 gates in ADS_READINESS_CHECKLIST green | varies | P1 complete + P2.4 telemetry | operator |
| P3.2 | Platform ad-account creation / 2FA / billing / kill-switch | 2 hours | none | operator |
| P3.3 | First ad concept selection + budget envelope + geo/age targeting | 2 hours | P3.1 | operator |
| P3.4 | First ad launch + 24h monitoring | 1 day | P3.2, P3.3 | operator |

### P4 — Press / investor (independent lane)

| # | Task | Effort | Blocker | Owner |
|---|---|---|---|---|
| P4.1 | Press pitch draft using `CLAIM_SAFE_LAUNCH_COPY.md §1 long` + §7 founder quotes | 2 hours | P0.4 | operator |
| P4.2 | Target journalist list for AI + sovereignty + MENA tech beat | 2 hours | none | operator |
| P4.3 | Outreach (personalized, not blast) | 2 hours/round | P4.1, P4.2 | operator |

---

## Critical path

```
P0.1-P0.5   (bulletproofing) ──┐
                               ├──> P2.3 (silent profile foundation) ──> P2.4 (launch moment) ──> P2.5 (first week)
P1.1-P1.8   (website cleanup) ─┘                                                                   │
                                                                                                    │
                                                                              ──> P3.1-P3.4 (paid ads — after telemetry from P2.5)
```

**Critical path duration:** 1–3 days to launch (P2.4) if P1 runs in parallel. P1 may extend depending on benchmark-receipt decisions.

---

## Decisions requiring typed operator authorization

Every item below is a distinct decision that requires explicit typed approval, **not inferred from context**. None of these will be auto-started by me.

1. **Publish now or wait?** — if any P0 / P1 / P2 items are not done, should the launch wait?
2. **bizra.ai content policy** — remove C4/C5/C7/C9, or rewrite with receipts, or keep as-is and accept ad-ineligibility?
3. **Benchmark receipt publication** — commit to a public receipt chain for technical claims, or drop them from public marketing?
4. **Privacy policy publication** — publish a public privacy statement? (Enables C1/C2 claims to move from B to A.)
5. **100-node cohort decision** — real live counter, aspirational framing, or remove?
6. **Arabic reviewer** — who is the operator of record for Arabic copy sign-off?
7. **Paid-ad kill-switch owner** — who is on call to pause ads within 10 minutes if something goes wrong?
8. **Press strategy** — announce simultaneously with organic launch, or let organic run 1–2 weeks first?

---

## What this plan does NOT do

- Does NOT publish anything.
- Does NOT upload to any platform.
- Does NOT edit bizra.ai source code (that repo is not in scope).
- Does NOT touch Cognitive Foundry canon, Node0 runtime, MEMORY.md, receipt-lineage WIP, PR #49/#50.
- Does NOT start the Canon Store Ingestion Gate (that's a separate closed-today lane).
- Does NOT commit anything to git.
- Does NOT run `/review`.

---

## Stop-signals

- If operator types "land the plane" after any P0/P1/P2 phase completes, treat that phase as the stopping point for the day.
- If a claim-register rule is violated in any draft (by me or anyone else), halt and surface before publishing.
- If any platform flags an account for review, halt all rollout immediately and route to operator.

---

## Session status after this plan lands

- **Media kit workspace:** created, documented, QA'd.
- **Website audit:** created with known caveat that SPA content is observed via pre-check, not fresh DOM.
- **Claim register:** created with A/B/C/D classifications.
- **Claim-safe launch copy:** drafted in both EN and AR, awaiting Arabic reviewer + operator sign-off.
- **Ads readiness:** gated, not ready.
- **Profile rollout:** ready for Phase 1–3 after P0 bulletproofing and operator sign-off.
- **Git / publishing:** untouched. Nothing is public yet as a result of this session.
