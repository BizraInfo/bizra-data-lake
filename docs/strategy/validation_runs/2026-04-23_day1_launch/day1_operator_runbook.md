# Day 1 Operator Runbook — 2026-04-23

**By-the-minute execution plan for a human operator.**

**Total time budget:** ~4.5 active hours, spreadable across one day.
**Assumed operator:** the founder OR a delegated human with access to budget, social accounts, and scheduling.
**Assumed tools:** LinkedIn + X account, a form platform (Google Forms / Typeform / Tally), a chat assistant (Claude / ChatGPT), a gift-card provider (Tremendous / Amazon gift cards / similar), email client.

Tag convention used below:

| Tag | Meaning |
|---|---|
| `[HUMAN-ONLY]` | Requires a real human to execute. No agent substitute possible. |
| `[CHAT-ASSISTED]` | Can be done faster with an LLM helper, but still operator-initiated. |
| `[OPTIONAL]` | Nice-to-have, not blocking Day 1 success. |

---

## T-minus preflight (5 min) — before anything else

`[HUMAN-ONLY]`

- [ ] Confirm you have logins for: LinkedIn, X, chosen form platform, email, gift-card provider.
- [ ] Confirm you have ~4.5 hours of focused time available today (not necessarily contiguous).
- [ ] Read through `card1_publish_copy.md` once. Get comfortable with the language.
- [ ] Read through `recruitment_copy_A3_primary.md` §5 disqualifier table once. You must be ruthless on Q4.
- [ ] Put this runbook open in a tab you can check throughout the day.

**If you cannot commit to end-of-day logging (Phase 5 below): do NOT start Day 1. Restart tomorrow with a contiguous window. A partial Day 1 is worse than a skipped day.**

---

## Slot 1 — Budget + form deploy (30 min, morning)

### Step 1.1 — Reserve incentive budget `[HUMAN-ONLY]` (10 min)

- [ ] Confirm a $750 line available for 15 × $50 gift cards.
- [ ] Create a tracking line-item in whatever expense/finance tool you use. Memo: `BIZRA Validation Sprint 2026-04 — 15 interviews × $50 gift`.
- [ ] If using Tremendous / similar: fund the account with $750 now OR verify card-on-file will cover 15 × $50 disbursements through 2026-05-07.
- [ ] **Do NOT send any gift cards until after an interview completes.** Pre-sending biases responses.

### Step 1.2 — Deploy intake form `[CHAT-ASSISTED]` (15-20 min)

- [ ] Open `intake_form_google_forms_ready.md`.
- [ ] Follow the 5-minute deploy checklist in §5.
- [ ] Test-submit the form yourself twice (one qualified, one disqualified test) and confirm both routings fire correctly.
- [ ] Delete your test responses from the response sheet.
- [ ] **Copy the public form URL into a notepad you'll keep open all day.**

### Step 1.3 — Record the form URL `[HUMAN-ONLY]` (5 min)

- [ ] In `docs/strategy/validation_runs/2026-04-23_day1_launch/README.md`, add a line under "Day 1 completed assets": `Intake form URL: <URL> (deployed [time] GST)`.
- [ ] In `recruitment_copy_A3_primary.md`, find every `[LINK]` placeholder and replace with the form URL.
- [ ] In `recruitment_copy_A1_secondary.md`, do the same (or decide: deploy a second form with A1-specific questions, OR reuse the same form with a tag in Q7 for segment).

**Slot 1 stop point:** form is live, budget is reserved, URLs are recorded. Before moving on, confirm you can still complete the rest of Day 1 today.

---

## Slot 2 — Peer selection + founder-prep interview (90 min, morning or mid-morning)

### Step 2.1 — Select peer interviewer `[HUMAN-ONLY]` (10 min)

Ideal criteria for the peer interviewer:

- Not involved in BIZRA; no prior knowledge of the wedge thesis
- Trustworthy with confidential business context
- Comfortable asking questions without interrupting
- 90 minutes of availability today or early tomorrow
- Willing to follow the interview guide exactly

- [ ] Make a short list of 3 candidates from your network.
- [ ] Reach out to the top candidate now. Offer a thank-you gesture (drink, meal, or $50 gift card — same denomination as external interviews for consistency).
- [ ] If no one confirms within 30 minutes → fall back to chat-assistant facilitation. See Step 2.2b.

### Step 2.2a — Human-peer founder-prep interview `[HUMAN-ONLY]` (75 min)

- [ ] Schedule a video call with the peer.
- [ ] Share the link to `BIZRA_Interview_Guide_v0_1.md` for them to read before the call (optional — 5 min read).
- [ ] Share the founder-prep stub path: `founder_prep_interview_template_filled_stub.md` — they use this to capture.
- [ ] Before the call: remind the peer "you are NOT pitching; you are running a research probe against me."
- [ ] Run the 12 questions in order (Phase 1 through Phase 5) per the interview guide.
- [ ] Run Section 12 meta-reflection (M1-M4) at the end.
- [ ] After the call: save the filled artifact as `founder_prep_interview_FILLED.md` in this folder. Tag `interview_type: founder_prep`.
- [ ] Apply any interview-guide edits identified in M4 before Day 2's first external interview.

### Step 2.2b — Chat-assistant fallback founder-prep `[CHAT-ASSISTED]` (75 min)

If no human peer is available:

- [ ] Open `founder_prep_facilitation_packet.md`.
- [ ] Paste the §3 Assistant System Prompt into a Claude or ChatGPT session.
- [ ] Answer the 12 questions the assistant asks, honestly, as if the assistant were a stranger.
- [ ] At the end, honestly fill M1-M4 — especially M3 (at least 2 biases the assistant exhibited).
- [ ] Copy the transcript into `founder_prep_interview_FILLED.md`.
- [ ] Mark interviewer as `chat_assistant_<model>_<date>`.

### Step 2.3 — Post-founder-prep hygiene `[HUMAN-ONLY]` (5 min)

- [ ] Verify the filled artifact has `interview_type: founder_prep` set (NOT `external_evidence`).
- [ ] Verify NO rows in `evidence_log_seed.csv` were added from this session.
- [ ] Note the ≥2 interviewer biases from Section 12.4 in a quick bullet list — apply them as self-discipline in Day 2's external interviews.
- [ ] Apply M4 guide changes to `BIZRA_Interview_Guide_v0_1.md` if any were identified (this is a permissible mid-sprint edit because it's pre-external).

**Slot 2 stop point:** founder-prep filled, tagged prep-only, biases captured, guide updated if needed.

---

## Slot 3 — Publish CARD_1 (30 min, midday)

### Step 3.1 — Final pre-publish check `[HUMAN-ONLY]` (10 min)

- [ ] Open `card1_publish_copy.md`.
- [ ] Open your LinkedIn and X draft composers in separate tabs.
- [ ] Paste the LinkedIn copy into a LinkedIn draft. Do NOT publish yet.
- [ ] Paste the X copy into an X draft. Confirm character count under 280. Do NOT publish yet.
- [ ] Run the §8 publication checklist end-to-end. All 8 items must be YES.

### Step 3.2 — Publish LinkedIn `[HUMAN-ONLY]` (5 min)

- [ ] Publish the LinkedIn post.
- [ ] **Immediately copy the post URL** — you will need it later.
- [ ] Note the exact timestamp (GST).

### Step 3.3 — Publish X `[HUMAN-ONLY]` (5 min)

- [ ] Publish the X post.
- [ ] **Immediately copy the post URL** — you will need it later.
- [ ] Note the exact timestamp.

### Step 3.4 — Record publications `[HUMAN-ONLY]` (10 min)

- [ ] Open `README.md` in the Day 1 folder.
- [ ] Under "Day 1 completed assets," add:
  - `CARD_1 LinkedIn URL: <URL> (published [time] GST, wording variant: signed_record)`
  - `CARD_1 X URL: <URL> (published [time] GST, wording variant: signed_record)`
- [ ] Open the Message-Test-Results file (create if doesn't exist) — you'll add row T001 with impressions=0 initially, updating over the next 7 days.

### Step 3.5 — Email/newsletter (OPTIONAL) `[OPTIONAL]` (5 min, skip if not ready)

- [ ] If an email list or newsletter channel is ready, paste the `card1_publish_copy.md` §3 blurb into it and send.
- [ ] If not ready, skip. Not a Day 1 blocker.

**Slot 3 stop point:** CARD_1 is live on LinkedIn and X. Two URLs are recorded. Timestamps noted.

---

## Slot 4 — A3 outreach wave 1 (60 min, afternoon)

### Step 4.1 — Build a 20-30 person target list `[HUMAN-ONLY] [CHAT-ASSISTED]` (20 min)

- [ ] Start a list in any format (Google Sheet / Notion / text file).
- [ ] Columns: `name` / `segment` / `channel` / `likely_fit_score (1-5)` / `contact_handle`
- [ ] Populate with 20-30 A3-likely consultants/creators you can reach today:
  - 1st-degree network consulting contacts
  - 2nd-degree referrals (ask 3-5 trusted peers "who in your consulting network uses AI in client work?")
  - LinkedIn searches (consultants with "AI" in bio, recent consultant post engagement)
  - Twitter/X searches (consulting creators with engaged audiences)
- [ ] Target mix: at least 12 non-1st-degree prospects (per sprint plan's "≥6 interviews from non-1st-degree intros" rule).

### Step 4.2 — Personalize and send `[HUMAN-ONLY]` (30 min)

- [ ] Open `recruitment_copy_A3_primary.md` §1 (short DM version).
- [ ] For each of the 10 highest-fit prospects (scoring 3+):
  - Personalize the first line ("I saw your post about X...", "I found you via Y...").
  - Send the DM. Include the intake form URL.
  - Log in the outreach tracker (see Step 4.3).
- [ ] Do NOT mass-blast. Personal DMs only. If you don't have time for 10, send 5 with full personalization — quality over quantity on Day 1.

### Step 4.3 — Log outreach `[HUMAN-ONLY]` (10 min)

- [ ] Open `outreach_tracker_seed.csv`.
- [ ] Delete the example row.
- [ ] Rename the file to `OUTREACH_TRACKER.csv` (or keep the seed name, just add real rows).
- [ ] For each DM sent, add a row:
  - `contact_id`: `C001`, `C002`, ... (your sequence)
  - `segment`: `A3`
  - `source_channel`: `LinkedIn_DM_2026-04-23` (or `X_DM_2026-04-23`, or `email_2026-04-23`)
  - `contacted_date`: `2026-04-23`
  - `response_state`: `pending`
  - `A3_S4_confirmed`: `pending`
  - `interview_scheduled`: (blank)
  - `notes`: why you picked them, 1st/2nd-degree, any relevant context

**Slot 4 stop point:** 10 A3 outreach DMs sent. All logged in tracker.

---

## Slot 5 — End-of-day logging (30 min, evening)

### Step 5.1 — Check early responses `[HUMAN-ONLY]` (15 min)

- [ ] Check intake form response sheet: any qualified responses?
  - For each STRONG_S4 / MODERATE_S4 response: reply within 1 hour offering Day 2 or Day 3 interview slot.
  - For each WEAK_S4 response: send disqualifier email B per `intake_form_google_forms_ready.md` §3.
  - Log each in outreach tracker.
- [ ] Check LinkedIn + X for replies on CARD_1 post.
  - For each reply: classify per the 5 reply types in `card1_publish_copy.md` §5.
  - Strong / Moderate / Weak / Disconfirming — the classification determines whether to invite to intake form.
- [ ] Do NOT convert any DM response into evidence-log row yet. Only completed interviews produce evidence log rows.

### Step 5.2 — Update Message-Test-Results `[HUMAN-ONLY]` (5 min)

- [ ] Check CARD_1 LinkedIn impressions (visible in LinkedIn post-author view).
- [ ] Check CARD_1 X impressions.
- [ ] Add or update T001 (CARD_1 LinkedIn) and T002 (CARD_1 X) in the Message Test Results CSV:
  - `wording_variant`: `signed_record`
  - `impressions`: current count
  - `named_incident_replies`: count from your reply inspection
  - `workflow_change_replies`: count
  - `forward_requests`: count
  - `disconfirms`: count

### Step 5.3 — Day 2 readiness check `[HUMAN-ONLY]` (10 min)

- [ ] Read `day2_execution_packet.md`.
- [ ] Confirm at least 2 A3 interviews are scheduled for Day 2 (from today's intake responses). If not, plan a Day 2 morning recruitment push.
- [ ] Confirm your Day 2 calendar has the interview slots blocked.
- [ ] Confirm CARD_2 publication channel is selected (r/LocalLLaMA, HN, Rust Discord, dev newsletter).
- [ ] Go to sleep. Day 2 starts with 2 real interviews.

**Slot 5 stop point:** all Day 1 activity logged. Day 2 is ready to execute.

---

## Exact stop points (if you can't finish)

- **Cannot finish Slot 1 today:** stop after Slot 1. Day 1 partial-state is: budget reserved + form live. No CARD_1 yet. Restart Day 1 Slots 2-5 tomorrow. The sprint clock extends by one day — this is fine. A half-baked Day 1 is worse.
- **Cannot finish Slot 2 today:** stop after Slot 2. No CARD_1 yet. Restart Slots 3-5 tomorrow. Clock extends one day.
- **Cannot finish Slot 3 today:** you published CARD_1 without logging the URL — go back and log immediately. If you cannot do Slot 4, stop; restart tomorrow. Clock extends.
- **Cannot finish Slot 4 today:** do a half wave (5 DMs instead of 10), log what you sent, and commit to complete the rest tomorrow morning. This is acceptable.
- **Cannot finish Slot 5 today:** do NOT skip Slot 5. If you ran out of time, at minimum log publish URLs and DMs sent. Save deeper reply-classification for tomorrow morning.

---

## Contamination risks to watch

| Risk | How to spot it | Mitigation |
|---|---|---|
| Founder-prep leaking into evidence | Any row in `EVIDENCE_LOG.csv` with `interview_type=founder_prep` | Re-check template; ensure field is set; delete leaking rows |
| Weak-S4 A3 booked because "seemed nice" | Q4 <40 chars or vague | Send disqualifier email B; do not book |
| "Cryptographic receipt" used in external copy | Review CARD_1 post copy once live | You're 30 min late if this slipped through — consider editing (LinkedIn allows edit; X limited) |
| Enthusiasm counted as strong signal | Reply classification in §5 of card1 says "sounds useful" → WEAK, not strong | Apply rule strictly; re-class any misclassified |
| 1st-degree intro dominance | Outreach tracker shows >50% 1st-degree contacts | Add non-1st-degree outreach in Slot 4 or tomorrow |
| Friend-of-founder contaminated response | Known BIZRA follower replied enthusiastically | Disqualify; do not book |
| **Future-state drift at Q7/Q8/Q10** (added 2026-04-23 per founder-prep M4) | Interviewee describes a designed/intended/wished-for system instead of what they do today; uses architecture terms ("fallback logic," "trust gate," "local-first policy") without naming concrete current actions | Interrupt with: *"Pause. Is that what you do today, or what you want the system to do eventually?"* If future-state: *"Stay with today. What do you actually do now?"* Tag drifted answers as aspirational; do NOT aggregate into H1/H2/H3 |

---

## Do-not list (Day 1)

1. Do NOT publish CARD_4 (cryptographic-receipt variant) — runs Day 5 per sprint plan for H13 matched-pair integrity.
2. Do NOT post CARD_1 multiple times to farm engagement.
3. Do NOT DM the same prospect twice today.
4. Do NOT pitch BIZRA in any reply — redirect to the intake form only.
5. Do NOT share the intake-form Q4 answers publicly — confidentiality is part of the compensation.
6. Do NOT add founder-prep evidence to `EVIDENCE_LOG.csv`.
7. Do NOT send the $50 gift card before the interview happens.
8. Do NOT commit this runbook or any Day 1 artifacts to git — uncommitted working state by design.
9. Do NOT book an A3 candidate who failed S4 under a "softer" framing.
10. Do NOT start Day 2 until Day 1 slots 1-5 are complete (with the partial-stop exceptions above).
11. Do NOT accept future-state architecture answers at Q7/Q8/Q10 as current-behavior signal. Use the interrupt probe per the interview guide's §0 rule 3. Future-state answers do NOT count toward H1/H2/H3 aggregation. (Contamination control per founder-prep M4, added 2026-04-23.)

---

## What's OK to skip without harm

- Email/newsletter blurb publication (Slot 3.5) — OPTIONAL
- Alternate headline variants B and C — keep in reserve for Day 3/4 if CARD_1 signal is weak
- Deploying a separate A1 intake form — Q7 tag is an acceptable lightweight substitute
- Personalizing the 15th+ outreach DM — 10 well-personalized beats 30 generic

---

## Handoff to Day 2

Once Slot 5 is complete, tomorrow starts with `day2_execution_packet.md`:

- 2 A3 external interviews (scheduled from today's intake)
- CARD_2 publication on an A1/A8 channel
- Updated trackers

---

**End of Day 1 Operator Runbook.** 4.5 active hours; partial-completion fallbacks defined. Operator-executable without agent-side action.
