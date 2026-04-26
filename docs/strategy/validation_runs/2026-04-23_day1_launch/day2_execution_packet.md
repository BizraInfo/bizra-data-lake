# Day 2 Execution Packet — 2026-04-24

**Operator-facing one-pager. Read this first thing on Day 2 morning.**

---

## Day 1 completed assets (ready to use)

- `README.md` — folder index
- `founder_prep_interview_template_filled_stub.md` — STUB; fill Day 1 morning BEFORE Day 2
- `recruitment_copy_A3_primary.md` — ready to paste
- `recruitment_copy_A1_secondary.md` — ready to paste
- `card1_publish_copy.md` — ready to post
- `evidence_log_seed.csv` — ready; fill with first external-interview rows
- `outreach_tracker_seed.csv` — ready; fill with Day 1 outreach
- `day2_execution_packet.md` — this file

All content is READY. Nothing has been published or executed.

---

## Exact Day 2 goal

By end of Day 2 (2026-04-24 23:59 GST):

1. **Two (2) external A3 interviews completed**, one morning and one afternoon.
2. **CARD_2** published on an A1/A8-appropriate channel (r/LocalLLaMA, HN, Rust community Discord, or dev-focused newsletter).
3. **Outreach tracker** updated with all Day 1 screening responses.
4. **Evidence log** updated with ≥ 4 `external_evidence` rows total across both interviews.

Nothing more. Do not publish CARD_3 today. Do not interview A1 or A8 today.

---

## Two A3 interview target — hard requirements

For each A3 candidate booked for Day 2:

| Check | Must be |
|---|---|
| Screening Q1 (client-facing) | YES |
| Screening Q2 (AI in delivery) | YES |
| Screening Q3 (recurring) | YES |
| Screening Q4 (S4 — 90-day incident) | **YES with specific dated example and named party** |
| Employer affiliation | NOT Anthropic / OpenAI / other AI-vendor |
| BIZRA contributor / friend-of-founder | NO |
| Consent to record | YES |
| Template 1 pre-prepared with `interview_type=external_evidence` | YES |

If ANY of these is NO — do NOT proceed. Rebook a different candidate.

---

## Tightened screening reminder (GUARDRAIL 2)

**The single biggest risk to the sprint is weak-S4 contamination.**

- Do NOT book a candidate whose Q4 is "I think so" / "probably" / "abstractly yes" / "clients sometimes ask me things like that."
- Q4 must be a specific incident, with a specific recent date (within 90 days), with a named party (by role, not necessarily name), with an articulable ask.
- If the candidate cannot produce that in screening, they cannot produce it in interview. Release them with thanks.

The interview guide assumes the candidate has a real story. Without one, the interview produces weak signal, contaminates H1/H2/H3 aggregates, and wastes a $50 gift card.

---

## Contamination control — present vs future state (added 2026-04-23 per founder-prep M4 / 12.5)

**A3 interviewees will sometimes describe a designed/intended/wished-for system rather than what they actually do today.** This most often happens at **Q7** (AI unavailable fallback), **Q8** (trust-decision ritual), and **Q10** (local/cloud ratio). When it does, interrupt the interviewee with the exact probe from the interview guide:

> *"Pause. Is that what you do today, or what you want the system to do eventually?"*

If the answer is future-state, redirect:

> *"Stay with today. What do you actually do now?"*

**Evidence rules:**
- Future-state architecture answers do NOT count as current-behavior signal.
- Do NOT aggregate future-state answers into H1/H2/H3 tallies, even when they reference the correct pain shape.
- If an interviewee cannot give a present-day answer after one redirect, note "no current behavior captured" in the Template 1 artifact for that question — do NOT synthesize one.
- An interview that passes the disconfirming-signal requirement but contains future-state drift at Q7/Q8/Q10 is still acceptable — just flag the drift, exclude those answers from the numeric aggregation, and score the rest normally.

**Why this rule exists:** the founder-prep session on Day 1 surfaced that Q7, Q8, and Q10 are structurally ambiguous between "what you do today" and "what your system will do eventually." External A3 candidates may also drift there, especially if they are AI-literate or have been thinking about governance. The interrupt probe is the contamination fence.

---

## CARD_2 publication reminder

**CARD_2:** "Run governed AI missions on your own hardware. No cloud required. A signed record for every step."

**Audience:** A1 + A8
**Channel today (pick ONE — diversify from Day 1's LinkedIn/X A3 channels):**
- r/LocalLLaMA (Reddit) — moderator-approved post
- HN — Show HN or relevant comment thread
- Rust community Discord (#tools or #ai channel with mod permission)
- A self-hosted / local-AI newsletter (Ollama newsletter, local-AI Substack)

**Channel rules:**
- Do NOT post CARD_2 on the same channel as CARD_1 within the same week (preserves matched-pair integrity for H13).
- Use the `signed_record` wording variant (CARD_2 is already in that family).
- Include a link back to the A1 screening form, NOT to a BIZRA landing page (no product landing page exists yet).
- Track impressions + replies in Template 4 Message Test Results.

**Do not rush to publish.** If the chosen channel is not ready, wait to Day 3 rather than rush-post on a weak channel.

---

## Day 2 success checks (end of day)

- [ ] 2 external A3 interviews completed (45 min each)
- [ ] Each interview produced a completed Template 1 artifact with `interview_type=external_evidence`
- [ ] Each Template 1 has the Q11 wording-preference field filled (H13 data)
- [ ] Each Template 1 has the Q12 workflow-change field explicitly filled (H3 dividing-line)
- [ ] Each interview contributed ≥ 1 disconfirming signal (NON-NEGOTIABLE — if not, interviewer was leading)
- [ ] Evidence log updated with ≥ 4 rows across both interviews
- [ ] Outreach tracker updated with all Day 1 screening responses (scheduled / disqualified / released / pending)
- [ ] CARD_2 published with tracking link/UTM
- [ ] Template 4 row opened for CARD_2 (impressions will fill over coming days)

---

## Day 2 failure checks (end of day)

If ANY of the following is true, act on it before Day 3:

- Fewer than 2 interviews completed → recruitment funnel is broken. **Do NOT loosen S4 to recover volume.** Instead, multiply outreach volume on Day 3 and accept a possibly-extended week-1 cadence.
- More than 1 interview without a disconfirming signal → interviewer was leading. Review both recordings with a second researcher BEFORE booking Day 3 interviews. Consider revising any specific probe that keeps producing enthusiasm-only answers.
- Evidence log shows 0 S-grade (strong) rows → expected after only 2 interviews; not alarming yet, but watch Day 3.
- CARD_1 (Day 1 publication) is showing only likes/RTs and no named-incident replies at 48h → start rotating to variant B or C of CARD_1 on Day 3. Do NOT switch to CARD_4 yet.
- Founder-prep artifact was never filled OR was accidentally counted as external evidence → stop everything, correct the tagging, and verify no evidence-log rows carry `interview_type=founder_prep` into aggregate views.

---

## What NOT to do on Day 2

1. **Do NOT pitch BIZRA in the interview.** Phase 4 artifact-reaction is the ONLY moment BIZRA is named, and even then not as a pitch.
2. **Do NOT explain the canon, the invariants, the constitutional gates, or the full product.** The interviewee's current reality is the data; contaminating it with future-BIZRA detail kills the interview's value.
3. **Do NOT count founder-prep output as evidence.** (GUARDRAIL 1.)
4. **Do NOT book a weak-S4 A3 candidate "to make volume."** (GUARDRAIL 2.)
5. **Do NOT publish a third message card today** — cards roll out at 1/day cadence per sprint plan to preserve channel pacing.
6. **Do NOT use "cryptographic receipt" in external language today.** "Signed record" is the default. (GUARDRAIL 3.)
7. **Do NOT treat post engagement (likes, RT, "interesting!") as signal without a named-incident reply.**
8. **Do NOT skip the interviewer self-check at the end of each session.** Eight items; all must be checked or flagged.
9. **Do NOT interview an A1 or A8 today** — Day 2 is A3-only per sprint plan. A1s start Day 3.
10. **Do NOT commit any of these research files to git.** Sprint package remains in `docs/strategy/validation_runs/` as uncommitted working state per session scope.
11. **Do NOT accept future-state architecture answers at Q7/Q8/Q10 as current-behavior evidence.** Use the interrupt probe: *"Pause. Is that what you do today, or what you want the system to do eventually?"* If future-state, redirect to present-day. Tag drifted answers as aspirational; they do NOT aggregate into H1/H2/H3. (Contamination control per founder-prep M4.)

---

## End-of-Day 2 handoff to Day 3

At 23:59 GST on 2026-04-24, produce (or update) these artifacts in this folder or a new sibling `2026-04-24_day2/`:

- `INT-2026-04-24-A3-01.md` — morning A3 interview (Template 1 filled)
- `INT-2026-04-24-A3-02.md` — afternoon A3 interview (Template 1 filled)
- Updated `evidence_log_seed.csv` (renamed to `EVIDENCE_LOG.csv` once real rows exist)
- Updated `outreach_tracker_seed.csv` (same pattern)
- Template 4 row for CARD_2 (initial)
- Day 3 execution packet (matching this pattern)

The Day-3 packet should target 2 A3 + 1 A1 external interviews, publish CARD_3, and note whether CARD_1's named-incident reply count has crossed the minimum threshold.

---

**End of Day 2 Execution Packet. Operator-ready.**
