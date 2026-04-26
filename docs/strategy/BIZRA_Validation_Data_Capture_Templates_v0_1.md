# BIZRA External Validation — Data Capture Templates v0.1

**Date:** 2026-04-23 (GST)
**Purpose:** Seven reusable templates for interview notes, evidence aggregation, objection tracking, message-test logging, pass/fail synthesis, wedge-revision, and recruitment funnel.
**Rule:** every interview and every message-test must fill at least one template. No ad-hoc note-taking.

Paste each template into the validation tool of choice (Notion, Airtable, Google Docs, Markdown in repo, etc.). Treat the templates as canonical shape.

---

## Template 1 — Interview Notes

File name convention: `INT-YYYY-MM-DD-<segment>-<sequence>.md`
(e.g. `INT-2026-04-29-A3-03.md`)

```markdown
# Interview — [Interviewee first name or pseudonym]

## Meta
- Date / Time (GST):
- Duration (min):
- Interviewer:
- **interview_type:** [ external_evidence | founder_prep ]  ← GUARDRAIL 1. `founder_prep` rows are EXCLUDED from all signal aggregation and all sprint thresholds.
- Segment: [A3 primary | A1 secondary | A8 stretch | N/A_founder_prep]
- Recruitment channel:
- A3 hard-filter S4 confirmed (client-facing + AI in delivery + recent 90-day incident): [ Y | N | N/A ] ← GUARDRAIL 2
- Consent recorded: [ ]
- Recording archived at:
- Notetaker/transcript source: [live / AI / post-call review]

## Screening pass
- [ ] S1  [ ] S2  [ ] S3  (A3 set) OR  [ ] S4  [ ] S5  [ ] S6  (A1)  OR  [ ] S7  [ ] S8  [ ] S9  (A8)
- Disqualifiers: none / [specify]

## Phase 1 — Current behavior

### Q1. Last important AI-assisted work
- Project:
- AI tool(s):
- Deliverable type:
- Where does the output live now?

### Q2. Observed tool opening
- Tool opened:
- Tabs / windows / bookmarks visible:
- Speed of access:

### Q3. Retention strategy
- How outputs are kept:
- Reproducibility test: could they find last week's output? [Y/N]

## Phase 2 — Pain & incidents

### Q4. Last wrong/unexpected AI incident
- Incident:
- Date (approx):
- Severity (interviewee's words):
- What they did:

### Q5. Client / collaborator asked for proof of AI work
- [ ] YES with incident  [ ] YES but vague  [ ] NO
- Incident (if YES):
- Asker role:
- Their ask:
- Interviewee's response:
- Time cost:

### Q6. Reproducibility probe
- Could they reproduce a specific recent output now? [Y / N / partial]
- What broke:

### Q7. AI-unavailable-backup behavior
- Fallback behavior:
- Named incident (A8 probe):
- Pain tolerance:

## Phase 3 — Trust & proof

### Q8. Trust-decision ritual
- How they currently decide to trust output:

### Q9. Proof probe
- Current proof mechanism (or "none"):
- What would be good enough for a client:

### Q10. Local / cloud ratio
- Cloud AI %:
- Local AI %:
- Reasons for split:

## Phase 4 — Artifact reaction

### Q11. Mock signed-record reaction (GUARDRAIL 3 — artifact framed as "signed record," NOT "cryptographic receipt")
- **Wording shown** (what the interviewer called it):
- **Wording understood** (what they think it is in their own words):
- **Wording preferred** (signed record / verifiable record / cryptographic receipt / other — capture exact phrase):
- First field they looked at:
- First question they asked:
- Readable? [Y / N / partial]
- Imagined-scenario answer:
- Would forward to whom:

## Phase 5 — WTP & close

### Q12. Value probe
- Value statement:
- Dollar figure (if any):
- Units (per-month / per-output / project / other):
- Threshold condition (e.g. "only if X"):
- **Workflow-change statement (CRITICAL — H3 dividing line):** [What would you stop doing? Start doing? Be specific.]
- **Workflow-change captured?** [ concrete_workflow_change | vague_preference | enthusiasm_only | no_change ] — if `enthusiasm_only` or `no_change`, downgrade Q12 evidence to WEAK regardless of enthusiasm level.

## Closing answers
- C1 (anything not asked):
- C2 (referrals):
- C3 (15-min follow-up OK?): [Y/N]
- C4 (anonymized quotes OK?): [Y/N]

## Evidence capture checklist
- [ ] ONE specific incident (Q4 or Q5) with date + recipient
- [ ] ONE observed behavior (Q2 or Q3)
- [ ] ONE named most-used AI tool
- [ ] ONE reproducibility result (Q6)
- [ ] ONE artifact-reaction note (Q11)
- [ ] ONE WTP anchor (Q12)
- [ ] ONE disconfirming signal

## Per-hypothesis tag (evidence grade S/M/W/D/N)
- H1 wedge: [ ]
- H2 user (primary vs secondary): [ ]
- H3 value prop: [ ]
- H4 trust/proof (crypto): [ ]
- H5 local-first: [ ]
- H6 Dema face (CLI): [ ]
- H7 business model (WTP): [ ]
- H8 receipt is marketing: [ ]
- H9 frontier independence: [ ]
- H10 replay specificity: [ ]
- H11 lifecycle visibility: [ ]

## Interviewer self-check
- [ ] Specific incident named (not hypothetical)
- [ ] Observed behavior via screen-share
- [ ] No BIZRA naming before Phase 4
- [ ] Captured disconfirming signal
- [ ] No feature-discussion loops
- [ ] Interviewee used their own language, not mine

## Raw quotes (direct, with minute-mark)
- [mm:ss] "..."
- [mm:ss] "..."

## Surprises / notes-to-self
-
```

---

## Template 2 — Evidence Log (aggregate across interviews)

File: `EVIDENCE_LOG.csv`

```
evidence_id,interview_id,interview_type,date,segment,hypothesis_id,grade,workflow_change,incident_summary,raw_quote_ref,tags,notes
E001,INT-2026-04-29-A3-03,external_evidence,2026-04-29,A3,H1,S,YES,"Client asked to prove AI report — 4 hrs to reconstruct; would drop screenshot step if signed record existed",mm:14:22,named_incident|client_pressure|workflow_change,"Accounting firm client"
E002,INT-2026-04-28-FOUNDER-00,founder_prep,2026-04-28,N/A,H12,N/A,N/A,"Founder prep calibration — captured 2 candidate probe biases",mm:03:00,founder_prep|interviewer_calibration,"NOT counted in aggregates per GUARDRAIL 1"
```

Columns:
- `evidence_id` — monotonic per log
- `interview_id` — points to Template 1 file
- **`interview_type`** — `external_evidence` | `founder_prep` (GUARDRAIL 1). **Only `external_evidence` rows count toward hypothesis aggregation and sprint thresholds. `founder_prep` rows are present for completeness and interviewer calibration only.**
- `date` — YYYY-MM-DD
- `segment` — A3 / A1 / A8 / N/A_founder_prep
- `hypothesis_id` — H1 / H2 / ... / H13
- `grade` — S / M / W / D / N (per sprint rubric)
- **`workflow_change`** — YES / NO / PARTIAL (H3 dividing-line indicator; required for S grade)
- `incident_summary` — one-line, factual, no interpretation
- `raw_quote_ref` — mm:ss in source recording or quote verbatim
- `tags` — pipe-separated; controlled vocabulary: `named_incident`, `client_pressure`, `observed_behavior`, `artifact_preference`, `wtp_anchor`, `forward_intent`, `disconfirming`, `workflow_change`, `wording_preference`, `founder_prep`, `interviewer_calibration`
- `notes` — interviewer flags, caveats, ambiguity

**Aggregation rule:** one row per distinct evidence point, not one row per interview. A single external interview can produce 5–10 rows. **Before any aggregation/threshold calculation: `FILTER WHERE interview_type = 'external_evidence'`.** Any script, spreadsheet view, or dashboard that aggregates without this filter is invalid and must be rejected.

---

## Template 3 — Objection Log

File: `OBJECTION_LOG.csv`

```
objection_id,interview_id,date,segment,raw_objection,root_hypothesis_challenged,interviewer_response,interviewer_confidence,follow_up_needed,notes
O001,INT-2026-04-29-A3-03,2026-04-29,A3,"'My clients don't ask about AI, they just want results'",H1,"Probed for specific recent incidents — none named",HIGH_CONFIDENCE_DISCONFIRM,N,"Strong D evidence for H1 in A3 subsegment"
```

Columns:
- `objection_id`
- `interview_id`
- `date`, `segment`
- `raw_objection` — interviewee's words, verbatim
- `root_hypothesis_challenged` — which hypothesis does this threaten?
- `interviewer_response` — what did the interviewer say/ask next?
- `interviewer_confidence` — `HIGH_CONFIDENCE_DISCONFIRM` / `MAYBE_DISCONFIRM` / `MAYBE_CONFIRM` / `UNCLEAR`
- `follow_up_needed` — Y/N
- `notes` — interviewer assessment

**Every objection must be logged.** Objections are the most valuable signal in the sprint and should not be paraphrased away.

---

## Template 4 — Message Test Results

File: `MESSAGE_TEST_RESULTS.csv`

```
test_id,card_id,wording_variant,channel,start_date,end_date,impressions,clicks_or_replies,named_incident_replies,workflow_change_replies,forward_requests,disconfirms,signal_score,card_verdict,notes
T001,CARD_1,signed_record,LinkedIn,2026-04-23,2026-04-30,812,34,4,3,2,1,STRONG,proceed,"Engagement from consultants; 3 workflow-change statements"
T002,CARD_4,cryptographic_receipt,LinkedIn,2026-04-27,2026-05-04,805,29,2,1,1,2,MODERATE,narrow,"Matched-pair test against CARD_1; signed_record variant outperforming on named-incident rate"
```

Columns:
- `test_id` — monotonic per card-channel combo
- `card_id` — CARD_1 / CARD_2 / ... / CARD_5 (per reordered cards doc; CARD_1 = signed record, CARD_4 = receipt)
- **`wording_variant`** — `signed_record` | `cryptographic_receipt` | `mixed` (GUARDRAIL 3 — required for H13 cross-read)
- `channel` — LinkedIn / X / HN / Reddit-subname / Newsletter-name / Email
- `start_date` / `end_date`
- `impressions` — best-effort count
- `clicks_or_replies` — total engagement
- `named_incident_replies` — replies that name a specific incident (S-grade in message-test context)
- **`workflow_change_replies`** — replies that explicitly describe what they would change in their workflow (CRITICAL — H3 dividing line)
- `forward_requests` — asks for demo / sample / follow-up
- `disconfirms` — replies that actively say the pain doesn't exist
- `signal_score` — `STRONG` (≥3 named-incident AND ≥2 workflow-change) / `MODERATE` (1–2 named-incident OR 1 workflow-change) / `WEAK` (vague positive; enthusiasm without workflow-change) / `NO_SIGNAL` (nothing) / `DISCONFIRM` (≥5 disconfirms)
- `card_verdict` — `proceed` / `narrow` / `reframe` / `kill`
- `notes`

**H13 matched-pair rule (GUARDRAIL 3):** CARD_1 vs CARD_4 must be compared on the same channel and comparable time-of-week. Compute `named_incident_replies / impressions` ratio per variant. If `signed_record` variant ratio ≥ 2× `cryptographic_receipt` variant → wording preference confirmed for "signed record"; launch surface language must lead with it. If inverse, flip. If equivalent, no wording change needed.

---

## Template 5 — Pass/Fail Summary (per hypothesis)

File: `HYPOTHESIS_PASS_FAIL.md`

Fill at Day 13 of sprint. One entry per hypothesis from the register.

```markdown
# Hypothesis H[N] — [short name]

**Claim:** [restate exact claim from register]

**Evidence count:**
- S (Strong): __
- M (Moderate): __
- W (Weak): __
- D (Disconfirming): __
- N (No-signal): __

**Net signal (S+M) vs (W+D):** __ vs __  → **[positive / neutral / negative]**

**Segment breakdown:**
- A3: [counts]
- A1: [counts]
- A8: [counts]

**Verdict:** `CONFIRMED` / `PARTIALLY_CONFIRMED` / `UNCLEAR` / `CONTRADICTED` / `FALSIFIED`

**Confidence after sprint:** `HIGH` / `MEDIUM` / `LOW`

**Key supporting quotes:**
1. [interview_id @ mm:ss] "..."
2. [interview_id @ mm:ss] "..."

**Key contradicting quotes:**
1. [interview_id @ mm:ss] "..."

**Implication for wedge:** [one sentence]

**Next test required (if any):** [one sentence]
```

Repeat for H1–H12.

At the end, write a 200-word synthesis naming which hypotheses survived, which fell, and what that means for wedge shape.

---

## Template 6 — Wedge Revision Log

File: `WEDGE_REVISION_LOG.md`

```markdown
# Wedge Revision Log

## Revision [N] — [date]

### Trigger
[Evidence that forced the revision — which hypothesis failed, which interview quote, which message-test result]

### Wedge language BEFORE
> [exact prior phrasing]

### Wedge language AFTER
> [exact new phrasing]

### What changed
- Primary segment: [before → after]
- Key missions: [before → after]
- Required proof moment: [before → after]
- Business model: [before → after]

### Evidence references
- [evidence_id E001]
- [objection_id O004]
- [message test T003]

### What this revision does NOT change
[anchors that remain — e.g. "Dema still the only face", "still local-first"]
```

Every wedge revision must be logged here with explicit before/after text. Revisions without evidence references are not allowed.

---

## Template 7 — Recruitment Funnel (per channel)

File: `RECRUITMENT_FUNNEL.csv`

```
channel,date,impressions,leads,screened,qualified,scheduled,completed,notes
LinkedIn-consultant-post-1,2026-04-24,1200,18,12,6,4,3,"High intent; low volume"
reddit-consulting,2026-04-25,400,5,3,1,1,0,"Low signal; disqualified 2"
```

Track conversion per channel. At end of sprint, compute:
- Cost-per-completed-interview (if paid ads used)
- Channel yield ranking
- Drop-off point identification

Use this to plan Sprint 2 channel strategy.

---

## Cross-template rules

- Every interview fills **Template 1** AND contributes rows to **Template 2** AND (if any objection) contributes rows to **Template 3**.
- Every message-test run fills one row in **Template 4**.
- **Template 5** is filled once at end-of-sprint with aggregated data.
- **Template 6** is filled only when a wedge-revision is triggered by evidence.
- **Template 7** is updated daily during the sprint.

### Founder-prep exclusion rule (GUARDRAIL 1 — applies to all templates)

- The founder self-interview on Day 1 produces a Template 1 artifact with `interview_type=founder_prep`.
- Its evidence rows in Template 2 carry `interview_type=founder_prep` and are **EXCLUDED** from every aggregation, every threshold check, every pass/fail computation, and every decision input.
- Any Template 5 pass/fail computation that references founder-prep rows is invalid.
- Any Template 6 wedge-revision triggered by founder-prep evidence is out of policy.
- Template 7 must not count founder-prep in interview-completion metrics.
- The sole purpose of founder-prep data is interviewer calibration: surfacing wording that feels natural, probes that don't land, and interviewer biases that must be watched during external interviews.

## Final cross-check before post-sprint decision

Before producing the PROCEED / NARROW / REFRAME / KILL recommendation:

- [ ] Template 2 has ≥ 1 `external_evidence` row per hypothesis from every external interview (15 external × 13 hypotheses ≠ required fill, but key hypotheses H1-H4 + H13 should have ≥ 10 external rows each)
- [ ] Template 3 has ≥ 15 rows (at least one objection per external interview — if not, interviewer was leading)
- [ ] Template 4 has ≥ 5 rows (one per card, at minimum) AND CARD_1 vs CARD_4 matched-pair row present for H13 cross-read
- [ ] Template 5 is complete for H1–H13
- [ ] Template 6 is either empty (wedge held) or has at least one revision entry (wedge moved)
- [ ] Template 7 shows at least 3 distinct recruitment channels contributed (excluding founder-prep)
- [ ] **GUARDRAIL 1 check:** all sprint-threshold aggregations filter `interview_type = 'external_evidence'`; no founder-prep leakage into decision inputs
- [ ] **GUARDRAIL 2 check:** all A3 rows in Template 1 show `S4 confirmed = Y`; no weak-S4 candidates in final cohort
- [ ] **GUARDRAIL 3 check:** Template 4 contains at least one `wording_variant=signed_record` AND one `wording_variant=cryptographic_receipt` row on the same channel for H13 cross-read; Template 1 Q11 wording-preference data captured for ≥ 10 of 15 external interviews

If any box is unchecked, the sprint is incomplete — do not produce the recommendation.

---

**End of Validation Data Capture Templates v0.1.** Pair with the other 4 validation docs for a complete external-validation sprint package.
