# Intake Form — Google-Forms-Ready (A3 primary segment)

**Purpose:** Deploy the A3 screening form in under 5 minutes with zero ambiguity about wording, field types, validation, or routing.

**Source:** This is the exact Q1–Q4 screening from `recruitment_copy_A3_primary.md` §4, reformatted for direct paste into Google Forms (primary) or Typeform/Tally (fallback).

**Deploy order:** paste → validate → test-submit yourself → publish → record URL in `README.md` AND both recruitment copy files.

---

## 1. Google Forms — copy-paste version

### Form settings

- **Title:** `AI & client-facing work — 4-question intake`
- **Description (shown at top):**

```
Thanks for your interest in the research call.

Four quick questions before we book a time. Honest answers — if any of
them is "no" or "not really," the form will release you with thanks and
a later follow-up offer.

About 2 minutes to complete. $50 gift card goes out after the call, not
before.
```

- **Settings → General:** Collect email addresses = YES. Limit to 1 response = YES (discourages duplicate intakes).
- **Settings → Presentation:** Show progress bar = YES. Shuffle question order = NO.
- **Settings → Responses:** Accepting responses = YES. Response destination = Google Sheet titled `BIZRA Validation Sprint — Intake Responses 2026-04`.

### Question 1 — Client-facing work

- **Field type:** Multiple choice
- **Question text:** `Do you currently deliver work to paying clients or external stakeholders who receive specific deliverables from you?`
- **Help text:** `"Deliverables" = work someone pays you for or depends on you for — a report, a design, code, a strategy memo, a legal doc, etc. Not internal notes or experiments.`
- **Options:** `Yes` / `No`
- **Required:** YES

### Question 2 — AI in delivery

- **Field type:** Short answer
- **Question text:** `In the last 30 days, has an AI tool (Claude, ChatGPT, Cursor, Copilot, etc.) contributed directly to what you delivered — not just to your research or note-taking? If yes, which tool(s)?`
- **Help text:** `One sentence is fine. If no, write "no."`
- **Validation:** Response validation → Text → Length → Minimum number of characters: 2. (Rejects blanks.)
- **Required:** YES

### Question 3 — Recurring

- **Field type:** Multiple choice
- **Question text:** `Is AI-assisted work a recurring part of your workflow, weekly or more often?`
- **Options:** `Yes, weekly or more` / `Occasionally / monthly` / `No, one-offs only`
- **Required:** YES

### Question 4 — Recent incident (HARD FILTER)

- **Field type:** Paragraph
- **Question text:** `In the last 90 days, has a client, collaborator, or reviewer challenged you on what the AI did, whether it was correct, or whether it should have been used? If YES, please briefly describe: when, who asked, and what they were asking about. A sentence or two is fine — no sensitive details needed.`
- **Help text:** `This is the most important question. "Abstract yes" / "clients worry about AI" without a specific incident will not qualify — we are looking for a real recent moment.`
- **Validation:** Response validation → Text → Length → Minimum number of characters: 40. (Forces a real answer, not "yes" alone. 40 chars ≈ "Last month my client at X asked me to...")
- **Required:** YES

### Question 5 — Contact info (added for scheduling)

- **Field type:** Short answer
- **Question text:** `What's the best email to reach you for scheduling?`
- **Validation:** Response validation → Text → Regular expression → Matches → `^[^@\s]+@[^@\s]+\.[^@\s]+$` (basic email shape)
- **Required:** YES

### Question 6 — Time zone (added for scheduling)

- **Field type:** Short answer
- **Question text:** `What time zone and city are you in?`
- **Required:** YES

### Question 7 — Referral source (optional, track for recruitment funnel)

- **Field type:** Multiple choice with "Other" text option
- **Question text:** `Where did you hear about this study?`
- **Options:** `LinkedIn post` / `X / Twitter post` / `Direct message` / `Email` / `Referral from a friend` / `Reddit` / `Other`
- **Required:** NO

---

## 2. Disqualifier routing (Google Forms section-based)

Google Forms supports "Go to section based on answer." Set up like this:

- **Q1 = No** → go to section `Disqualified (No deliverables)` → thank-you page A
- **Q2 = "no" (contains only "no" or blank)** → go to section `Disqualified (No AI in delivery)` → thank-you page A
- **Q3 = "No, one-offs only"** → go to section `Disqualified (Not recurring)` → thank-you page A
- **Q4 contains fewer than 40 chars OR the text is "no" / "n/a" / "not really"** → **Cannot auto-route in Google Forms;** handle manually at the operator's triage step (see §3).
- **All other passes** → continue to Q5-Q7 → thank-you page B (qualified)

### Section — Disqualified (any)

- **Thank-you page A text:**

```
Thanks for taking the time. Based on your answers, you're not the target
cohort for this specific study — we're focused on people who've had a
recent (last 90 days), specific credibility/proof/revision dispute with
a client about AI-assisted work.

If that changes — a specific incident, not an abstract worry — feel free
to come back. We'll also follow up when the related tool is closer to
launch.
```

### Section — Qualified

- **Thank-you page B text:**

```
Thank you — that sounds like exactly the kind of story I'm trying to
understand. I'll review the answers within 24 hours and reach out at the
email you provided to find a 45-minute slot that works for you.

The $50 gift card goes out after the call.

Reminder: this is a research conversation, not a sales pitch. If at any
point the conversation feels like a pitch, you can stop it.
```

---

## 3. Manual triage (Q4 quality check — no auto-routing possible)

Even with 40-char minimum validation, Q4 answers can still be weak. After each response lands in the Google Sheet:

The operator reviews Q4 and tags each response as one of:

| Tag | Criteria | Action |
|---|---|---|
| `STRONG_S4` | Specific incident + date (or timeframe like "last month") + named party (client role, firm type, collaborator) + what they were asking | Book interview. Priority. |
| `MODERATE_S4` | Specific incident but vague on one of: date, party, or ask | Book interview. Flag in outreach_tracker notes. |
| `WEAK_S4` | Abstract worry / "clients sometimes..." / no specific incident / just "yes" repeated | **Release.** Send disqualifier email B. Do not book. |
| `ADJACENT` | Incident is about AI in a non-consultant context (employee + boss, student + professor, etc.) | Thank and flag; not A3 first-wedge — log as A5/A7 adjacent for possible later cohort |

**Do not soften triage to make volume.** Weak-S4 contamination is the single largest sprint quality risk per Guardrail 2.

### Disqualifier email B (for weak-S4 who submitted real emails):

```
Subject: Thanks for responding — follow-up

Hi [first name],

Thanks for taking the time to fill out the intake. After reading your
answer to question 4, I think our research is looking for something a
bit more specific than your current situation — a concrete recent
incident where a client or collaborator challenged you about AI work.

If that changes, or you realize you do have a specific example from the
last 90 days, please reply and I'll reopen your intake.

Either way, thank you.

[Your name]
```

---

## 4. Typeform / Tally shorter version (fallback)

Typeform and Tally handle conditional logic differently but benefit from shorter, conversational framing.

### Typeform / Tally intake — 4 cards

**Card 1 — welcome:**

> Hi — thanks for your interest. 4 quick questions, about 2 minutes.
>
> Honest answers only. If it's not the right fit, the form will tell you and we'll release you with thanks.

**Card 2 (Q1) — yes/no:**

> Do you deliver work to paying clients or external stakeholders?

**Card 3 (Q2) — short text:**

> In the last 30 days, has AI contributed directly to what you delivered? If yes, which tool?

**Card 4 (Q3) — multiple choice:**

> How often is AI part of your work?
> - Weekly or more
> - Occasionally
> - One-offs only

**Card 5 (Q4) — paragraph, required:**

> This is the critical one.
>
> In the last 90 days, has a client or collaborator pushed back on something an AI contributed to your work?
>
> If yes — who, when, and what were they asking about?

**Card 6 — email + tz:**

> Best email for scheduling + your time zone?

**End card — branch by Q4 answer length:**

- If Q4 < 40 chars OR Q3 = "one-offs" OR Q1 = no → release copy (thank-you A)
- Else → qualified copy (thank-you B)

---

## 5. 5-Minute Deploy Checklist

For the operator. Run through in order. Target: form live in under 5 minutes.

- [ ] **00:00** Open forms.google.com OR typeform.com OR tally.so (pick one, do not agonize)
- [ ] **00:30** Click "Blank form"
- [ ] **00:45** Paste title from §1: `AI & client-facing work — 4-question intake`
- [ ] **01:00** Paste description from §1
- [ ] **01:30** Create Q1 (multiple choice) — paste text, set required, set options
- [ ] **02:00** Create Q2 (short answer) — paste text, set required, set validation (min 2 chars)
- [ ] **02:30** Create Q3 (multiple choice) — paste text, set required, set 3 options
- [ ] **03:00** Create Q4 (paragraph) — paste text, set required, set validation (min 40 chars)
- [ ] **03:30** Create Q5, Q6, Q7 — email, time zone, referral source
- [ ] **04:00** Configure section routing (Q1=No / Q3=one-offs → disqualified section)
- [ ] **04:15** Paste thank-you-A and thank-you-B copy into respective sections
- [ ] **04:30** Click "Send" to preview the form URL
- [ ] **04:45** **Fill the form yourself as a test.** Submit a qualified test response. Submit a disqualified test response. Confirm routing fires correctly.
- [ ] **05:00** **Copy the public form URL.** This is the URL to paste into recruitment copy.

After the form is live:

- [ ] Paste URL into `docs/strategy/validation_runs/2026-04-23_day1_launch/README.md` under "Day 1 completed assets"
- [ ] Paste URL into `recruitment_copy_A3_primary.md` wherever `[LINK]` appears
- [ ] Paste URL into `recruitment_copy_A1_secondary.md` wherever `[LINK]` appears (form is A3-tuned but A1 variant — either deploy a second form with A1 questions, or reuse this form with a tag field)
- [ ] Delete the two test responses from the Google Sheet / form responses

---

## 6. Monitoring checklist (after publish)

Once the form is live:

- Check responses 3 times on Day 1 (midday, afternoon, end-of-day)
- Tag each Q4 answer with STRONG_S4 / MODERATE_S4 / WEAK_S4 / ADJACENT
- For STRONG_S4 and MODERATE_S4: reply within 24 hours to schedule
- For WEAK_S4: send disqualifier email B within 24 hours
- Log every form response in `outreach_tracker_seed.csv` (rename to live tracker) with appropriate tag

---

## 7. What this form does NOT do

- Does NOT schedule automatically — you still do scheduling manually (unless you integrate Calendly on thank-you page B, which is a good 3-minute addition but not blocking)
- Does NOT send the $50 gift card — that happens after the interview
- Does NOT track the source referral with UTM — Q7 (self-reported source) is the substitute
- Does NOT pre-screen for Anthropic/OpenAI employment — check manually from email domain
- Does NOT prevent signal contamination from BIZRA contributors / friends-of-founder — check manually against your known list

These are all intentional trade-offs for a 5-minute deploy.

---

**End of Intake Form packet. Ready for Google Forms / Typeform / Tally paste.**
