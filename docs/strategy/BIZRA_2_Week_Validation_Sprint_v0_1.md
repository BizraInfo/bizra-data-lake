# BIZRA External Validation — 2-Week Sprint Plan v0.1

**Date:** 2026-04-23 (GST)
**Sprint window:** Day 1 to Day 14 after operator greenlight
**Primary segment:** A3 (AI-native creator/consultant). **Secondary:** A1 (solo sovereign builder). **Stretch:** A8 (hyperscaler-dependency-anxious dev).
**Success output:** a short post-sprint memo with pass/fail per hypothesis in the register and a wedge-revision recommendation.

---

## Sprint goals (one screen)

| Goal | Metric | Minimum acceptable |
|---|---|---|
| **External** interview volume | Completed 45-min interviews (external only; founder self-interview excluded per GUARDRAIL 1) | **15 total** (10 A3 + 4 A1 + 1 A8) |
| Incident specificity | Interviews (external) with a named, dated, specific pain incident | **≥ 8 of 15** |
| Workflow-change specificity | Interviews (external) producing at least one concrete workflow-change statement at Q12 | **≥ 6 of 15** — workflow change is the dividing line |
| Message-test reach | Cumulative impressions across 5 cards | **≥ 1,500** (organic OK) |
| Message-test signal | Specific named-incident replies | **≥ 10 across all cards** |
| Disconfirming signals captured | Logged disconfirming evidence points (external interviews) | **≥ 15 (one per interview)** |
| Artifact-reaction data | Mock signed-record reactions captured on camera/transcript (external interviews) | **≥ 12 of 15** |
| WTP anchor data | Interviews (external) producing a specific dollar figure or threshold | **≥ 6 of 15** |
| Wording-preference data (H13) | External interviews producing a captured wording-preference answer at Q11 | **≥ 10 of 15** |

If any metric fails the minimum, do NOT proceed to build. Re-run or narrow.

**GUARDRAIL 1 discipline:** The founder self-interview produced on Day 1 is counted as `interview_type=founder_prep` and is EXCLUDED from every "of 15" denominator above, every threshold tally, and every proceed/narrow/reframe/kill decision. It serves solely as interviewer-bias / wording calibration. See Template 1 `interview_type` field.

---

## Daily plan

### Week 1 — Discovery

**Day 1 — Prep + founder calibration (NOT market evidence)**
- **Founder self-interview — PREP ONLY per GUARDRAIL 1.** Founder answers all 12 core questions about own BIZRA use. Archive as Template 1 artifact with `interview_type=founder_prep`. This is **wording debugging, probe rehearsal, and interviewer-bias surfacing**. It is NOT counted toward the 15 external interview total, NOT entered in the evidence log aggregate, NOT used in hypothesis confirmation/falsification, and NOT a proceed/narrow/reframe/kill input. Its outputs go into a separate "founder_prep" notes bucket.
- Finalize recruitment copy for each channel.
- Publish CARD_1 ("signed record for clients" — GUARDRAIL 3 leads with signed-record wording) to primary outbound channel.
- Recruitment target: 25 qualified leads in pipeline by end of day (applying tightened A3 filter — see Recruitment section).

**Day 2 — First external cohort**
- 2 A3 **external** interviews (morning + afternoon).
- Publish CARD_2 (local-first mission runtime with signed steps).
- Evidence log entries: 2 (external only).

**Day 3 — Volume**
- 2 A3 + 1 A1 **external** interviews.
- Publish CARD_3 (replay a prior AI decision).
- First interim review: are the tightened A3 screening questions (S1-S4) filtering correctly? Are interviews producing named incidents AND workflow-change statements?

**Day 4 — Calibration**
- 2 A3 **external** interviews.
- Interim synthesis: 6 external interviews done — what pattern is emerging? Name top 2 surprises. Tag any wording-preference signal for H13.
- Adjust probes if leading questions are sneaking in.

**Day 5 — Secondary reach + wording A/B begins**
- 1 A3 + 2 A1 **external** interviews.
- Publish CARD_4 (crypto-receipt-framed A3 variant) — the matched-pair test against CARD_1 for H13.
- End-of-week mini-review: progress vs minimums.

**Day 6 — Slack day**
- No new interviews scheduled.
- Re-listen to 3 interviews and reclassify evidence.
- Spend half-day on message-test reply threads — capture signal-vs-noise.
- Update the evidence log + objection log.

**Day 7 — Rest / synthesis**
- No active work. Rest + let reactions settle.

### Week 2 — Pressure testing

**Day 8 — Mid-sprint check**
- 2 A3 **external** interviews.
- Publish CARD_5 (verify without trusting vendor — final crypto-framed variant).
- Mid-sprint checkpoint: total external interviews so far vs target. If behind, accelerate recruitment under the tightened A3 filter (do NOT loosen S4 to make volume).

**Day 9 — Concierge demo option**
- 1 A3 + 1 A1 interview.
- First "concierge demo" attempt with an interviewee from Day 2 or 3: send them a mock receipt (Phase 4 artifact) over email. Capture reaction asynchronously.

**Day 10 — Stretch segment**
- 1 A8 interview.
- 1 A3 interview.
- Cross-segment comparison begins: does pain language differ between A3 and A8?

**Day 11 — Artifact test**
- 1 A3 interview with artifact reaction focus.
- Send three concierge-demo follow-ups from Week 1 interviews. Ask: *"Would you want to try this on a real task? Here's how (no commitment)."*

**Day 12 — Closeout interviews**
- Final interviews to hit minimum (up to 2).
- Start synthesis: populate `wedge_revision_log` template with the top 3 surprises and top 3 confirmations.

**Day 13 — Synthesis**
- No new interviews.
- Full-day synthesis. Fill pass/fail summary per hypothesis. Draft wedge-revision recommendation.

**Day 14 — Decision**
- Executive read-out: primary segment decision, biggest risk, proceed/narrow/reframe/kill.
- Archive all raw notes.
- Handoff document produced for operator decision.

---

## Recruitment

### Channels by segment

**A3 (primary, target 10) — TIGHTENED FILTER PER GUARDRAIL 2:**
- **Hard screening required before scheduling:** client-facing + uses AI in delivery + has experienced ≥1 credibility/proof/revision dispute in last 90 days. Loose "AI-native creator/consultant" does NOT qualify.
- Recruitment copy must explicitly ask: *"In the last 90 days, have you been challenged by a client or collaborator on something an AI contributed to your work?"* Use this as opt-in filter, not just screening-call filter.
- Consultant Twitter/LinkedIn (posts from consulting creators with engaged audiences)
- Consultant/freelancer Substack communities
- Reddit: r/consulting, r/freelance_forhire (DO NOT spam — offer as research, small payment if possible)
- Personal network: founder's consulting connections, 2nd-degree intros
- AI-consultant Slack / Discord communities
- **Do NOT accept A3 candidates who pass S1-S3 but fail S4 (no recent incident).** Redirect them to a later cohort or thank and release. Recruiting weak-S4 candidates contaminates H1/H2/H3 signal and is the single largest quality risk for this sprint.

**A1 (secondary, target 4):**
- GitHub: contributors to local-AI tools (Ollama, llama.cpp, LM Studio-adjacent)
- Reddit: r/LocalLLaMA, r/selfhosted
- Rust community Discord
- Indie-hacker newsletters

**A8 (stretch, target 1):**
- Comments on any recent public post about Claude/OpenAI rate-limit incidents
- Dev newsletters that cover cost / quota topics
- Anthropic / OpenAI API user forums

### Incentive

- No cash incentive for first pass (biases responses).
- Offer: $25-50 gift card post-interview as thank-you, sent after interview so it does not contaminate.
- Offer: early access to the tool when shipped.
- Honesty requirement: do NOT offer access to things that don't exist yet.

### Qualifying conversion funnel (track per channel)

```
Channel impressions → leads → screening responses → qualified → scheduled → completed
```

Target: 7–10% impressions → leads, 50% screening response rate, 40% of screened = qualified, 80% of qualified = scheduled, 85% of scheduled = completed.

Back-solving from 15 completed: need ~40 qualified, ~100 screened, ~200 leads, ~2,500 impressions.

---

## Evidence scoring rubric

Every piece of captured evidence is classified as one of:

| Grade | Definition | Example |
|---|---|---|
| **S (Strong)** | Specific incident + dated + named-recipient + **workflow-change statement** ("I would stop X, start Y") + follow-up behavior observed | "Last month my client at [firm] asked me to show how the AI produced the report — I screenshotted everything and spent 4 hours reconstructing. If a signed record existed I would drop the screenshot step and send this instead." |
| **M (Moderate)** | Specific incident without workflow-change statement OR behavior + partial workflow-change signal | "Clients sometimes ask me about AI — usually I just explain. I might attach a record if I had one." |
| **W (Weak)** | Interviewee opinion / hypothetical / vague / **enthusiasm without workflow-change statement** | "I think AI audit is going to be a big deal" OR "Yeah this sounds useful, I'd try it" |
| **D (Disconfirming)** | Interviewee actively contradicts hypothesis with specifics | "My clients have never asked and I doubt they ever will — they just want results" |
| **N (No-signal)** | Opinion-level "that sounds cool" with no incident, no behavior, no specifics | "Interesting idea" |

**Critical scoring rule (dividing line):** enthusiasm without a workflow-change statement is **WEAK** regardless of its energy level. The line between a cool feature and a real wedge is "would you change your workflow to get this?" — nothing else. A Strong grade REQUIRES a concrete workflow-change answer at Q12 in addition to an incident.

Interview is scored on S/M/W/D counts. A healthy interview has at least 1 S or 2 M and at least 1 D. Founder-prep interview outputs are scored separately and do NOT aggregate into these counts.

---

## Decision thresholds (end of sprint)

Compute these at Day 13. Use hypothesis register to aggregate. **Founder-prep evidence is excluded from every threshold below (GUARDRAIL 1).** All denominators are "of 15 external interviews."

| Outcome | Criteria | Action |
|---|---|---|
| **PROCEED** | ≥ 4 of top 5 hypotheses (H1, H2, H3, H4, H8) show net positive evidence (S+M > W+D) AND ≥ 8 external interviews produced specific incidents AND ≥ 6 external interviews produced concrete workflow-change statements (H3 dividing-line pass) | Build minimum CLI surface (M05+M03+M06+M11). Plan public wedge launch with "signed record" as primary launch language if H13 favored it. |
| **NARROW** | H1 confirmed but H2 suggests primary segment is wrong (A3 did not show more pain than A1) OR H13 suggests "signed record" clearly outperforms "cryptographic receipt" (or vice versa) — surface-language revision needed | Keep wedge shape; swap primary segment OR simplify artifact surface OR commit to the favored wording. One additional 1-week validation round targeting the narrowed cohort. |
| **REFRAME** | H1 confirmed (pain exists and is specific) but H3 falsified — fewer than 3 external interviews produced a workflow-change statement despite positive incident capture. Pain is real; receipt does not move behavior. | Wedge shape needs revision. Look at adjacent missions (M24, M12, M21). New hypothesis register, new sprint. |
| **KILL** | H1 falsified (fewer than 3 of 15 external interviewees describe any specific pain matching the wedge) | Wedge is wrong. Do not build the named CLI surface. Return to audit and pick a different wedge — likely reshaping around M02 (local-first runtime), M24 (citation validation), or M21 (quality gate). |

---

## What counts as strong signal vs weak signal

### Strong (in order of weight)

1. **Unprompted incident.** Interviewee describes H1-matching pain before you ask about it specifically.
2. **Observed-behavior match.** Screen-share shows interviewee already wrangling with proof-of-AI-work ad hoc (screenshot folder, custom logs, hand-written notes).
3. **Client pressure story.** Interviewee names a client/collaborator who asked for proof — with date.
4. **Artifact preference.** Interviewee prefers mock receipt over screenshot + timestamp AND can articulate why.
5. **Forward-intent.** "I would send this to X" with X named by role.
6. **WTP anchor.** A specific number paired with a trigger condition.

### Weak (do not treat as validation)

1. "That's interesting."
2. "Sounds useful."
3. "I could see that being valuable."
4. "Are you going to make it open source?"
5. "Does it integrate with [popular tool]?"
6. Feature-request responses unrelated to H1 pain.

### Traps (explicit)

- **Enthusiasm from A6 interviewees.** Ideological adopters will cheer regardless. Tag A6 responses separately and weight them at 0.3×.
- **Consultant politeness.** A3s are professionally trained to react positively. Discount "yes" replies without specifics.
- **Builder curiosity.** A1s ask technical questions that feel like engagement but don't validate mission-fit.

---

## Post-sprint deliverables (to be produced at end of Day 13)

1. **Pass/fail summary** per hypothesis (H1–H12 from register).
2. **Wedge revision log** — if wedge language changes, record exact before/after + the evidence that forced the change.
3. **Primary-segment confirmation or swap** with evidence.
4. **Top 3 surprises** — things that contradicted prior assumptions.
5. **Top 3 confirmations** — things the audit hypothesized that held up.
6. **Next-sprint recommendation** — validation round 2 OR build greenlight OR re-audit.

---

## Anti-pattern watchlist for the sprint

- Do not let a single enthusiastic A3 interviewee cause the sprint to declare PROCEED early.
- Do not let the recruitment funnel collapse into founder's own network only — at least 6 interviews must come from non-1st-degree introductions.
- Do not demo a prototype in interviews — Phase 4 mock receipt is the maximum artifact allowed.
- Do not extend the sprint to 3 weeks. If the evidence is not clear in 2 weeks, the frame is wrong — not the sample size.
- Do not write the wedge language first and then find evidence for it. Let the evidence revise the wedge.

---

## If the sprint fails

All four failure modes (NARROW, REFRAME, KILL) are valid outcomes of a disciplined validation. The point is not to confirm the wedge. The point is to find out.

A sprint that produces a KILL in Week 2 is a **win** — it saves the cost of building a CLI surface around a wedge that has no market.

---

**End of 2-Week Validation Sprint Plan v0.1.** Pair with hypothesis register + interview guide + message cards + capture templates.
