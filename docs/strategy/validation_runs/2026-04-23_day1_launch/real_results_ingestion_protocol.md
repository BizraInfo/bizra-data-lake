# Real-Results Ingestion Protocol

**Purpose:** When the operator returns from executing Day 1 (or any subsequent day) with real-world results, this protocol specifies exactly what to paste and how the chat assistant converts it into tracker rows, evidence rows, and classifications — **without fabricating anything**.

**Core rule:** the operator is the only source of truth for what happened. The assistant only transforms what the operator reports. No synthesis, no extrapolation, no invented replies.

---

## 0. Anti-fabrication discipline

Before ingesting any data, the assistant must:

- Treat all fields the operator did not explicitly report as BLANK, not as "inferred" or "plausible."
- Refuse to fill in names, dates, incidents, or timestamps the operator did not state.
- Refuse to classify a reply the operator did not paste verbatim.
- Flag any ingestion request that tries to add rows without source data ("add 5 more likely A3 replies" → refuse).
- Never rename `_seed.csv` files as live files unless real rows have been added.

If the operator asks "was that enough?" or "should I invent more?" — the answer is always **no**. The only path forward is more real execution.

---

## 1. What the operator pastes back (templates)

When the operator returns and says "Day 1 is done, let's ingest," paste the following templates pre-filled with the operator's real data. The assistant then transforms each into tracker/log rows.

### 1.A — Publication confirmation block

```
--- DAY 1 PUBLICATION RESULTS ---
Date: YYYY-MM-DD
Intake form URL: <full URL>
Intake form platform: <google_forms | typeform | tally>
Intake form deployed at: HH:MM GST

CARD_1 LinkedIn URL: <full post URL>
CARD_1 LinkedIn published at: HH:MM GST
CARD_1 LinkedIn wording variant: signed_record

CARD_1 X URL: <full post URL>
CARD_1 X published at: HH:MM GST
CARD_1 X wording variant: signed_record

Email/newsletter (optional): <URL or "not sent">

Peer interviewer for founder-prep: <name/role or "chat_assistant_<model>_<date>">
Founder-prep completed: YES / NO (if NO, stop ingestion — Day 1 is incomplete)
Founder-prep filename: <path to filled artifact>

Incentive budget reserved: YES / NO
```

### 1.B — Outreach list block

```
--- DAY 1 OUTREACH SENT ---
One row per contact. Do NOT list people you didn't actually message.

Format:
| sequence | segment | channel           | contact_handle_or_anon | degree   | notes                           |
|----------|---------|-------------------|------------------------|----------|---------------------------------|
| 1        | A3      | LinkedIn_DM       | @handle or anon_001    | 1st / 2nd/ cold | why you picked them       |
| 2        | A3      | email             | anon_002               | 2nd      | referred by X                   |
...

Also report the TOTAL COUNT sent today.
```

**Rule:** if the operator says "I sent 10 DMs" but only lists 7, the assistant asks for the missing 3 or logs only 7. Don't average. Don't guess.

### 1.C — Replies-received block

Only replies the operator has actually received. Pasted verbatim.

```
--- DAY 1 REPLIES RECEIVED ---
For each reply, paste the verbatim text (anonymize names/firms as needed).

Reply #1:
  Channel: [LinkedIn post comment | LinkedIn DM | X reply | X DM | email | form]
  From: [anon handle or role descriptor, NOT real name unless operator explicitly OKs]
  Timestamp: YYYY-MM-DD HH:MM GST
  Verbatim text:
  """
  [paste the reply exactly as received]
  """

Reply #2:
  ...
```

**If no replies received:** paste `--- NO REPLIES RECEIVED ON DAY 1 ---`. The assistant will NOT fill in hypothetical replies.

### 1.D — Intake-form submissions block

```
--- DAY 1 INTAKE FORM SUBMISSIONS ---
For each submission in the response sheet, paste:

Submission #1:
  Timestamp: YYYY-MM-DD HH:MM GST
  Referral source (Q7): [value]
  Q1 (client-facing): [Yes | No]
  Q2 (AI in delivery): "[verbatim]"
  Q3 (recurring): [value]
  Q4 (90-day incident) — VERBATIM:
  """
  [paste Q4 answer exactly]
  """
  Q4 operator classification: [STRONG_S4 | MODERATE_S4 | WEAK_S4 | ADJACENT]
  Interview scheduled: YES (date/time) | NO (reason)

Submission #2:
  ...
```

### 1.E — Interview bookings block

```
--- DAY 1 INTERVIEW BOOKINGS (for Day 2 or later) ---
Per scheduled interview:

Booking #1:
  Interview_id: INT-<date>-<segment>-<seq>
  Segment: A3
  Booked for: YYYY-MM-DD HH:MM GST
  From intake submission #: <N>
  A3 hard-filter S4 confirmed: YES (strong) | YES (moderate) | NO (disqualified)
  Calendar link / call link: [URL]

Booking #2:
  ...
```

### 1.F — Founder-prep output block

```
--- DAY 1 FOUNDER-PREP NOTES ---
(Filled artifact path): <path>
Section 12 outputs only (NOT the 12 Q&A — those stay in the filled artifact):

12.1 — Wording that felt natural:
- [bullet]
- [bullet]

12.2 — Wording that felt forced:
- [bullet]

12.3 — Probes that did not land:
- [bullet]

12.4 — Interviewer biases observed (min 2):
- [bias 1]
- [bias 2]

12.5 — Interview-guide changes to apply before Day 2:
- [change or "none"]
```

**Tag:** `interview_type: founder_prep` — excluded from evidence aggregation.

---

## 2. How the assistant transforms each block

### 2.A → updates `README.md`

- Append Day 1 completed-assets section with intake form URL, CARD_1 URLs, timestamps, peer interviewer.
- If founder-prep = NO, assistant flags: "Day 1 incomplete — founder-prep missing; halt ingestion, re-run slot 2."

### 2.B → adds rows to `OUTREACH_TRACKER.csv`

One row per listed outreach. No padding.

```csv
contact_id,segment,source_channel,contacted_date,response_state,A3_S4_confirmed,interview_scheduled,notes
C001,A3,LinkedIn_DM_2026-04-23,2026-04-23,pending,pending,,<notes from operator>
C002,A3,email_2026-04-23,2026-04-23,pending,pending,,<notes>
...
```

**Rule:** `response_state=pending` for all. Assistant does NOT mark any as "responded" until block 1.C shows matched replies.

### 2.C → updates response state on existing outreach rows + adds Message Test Results rows

For each verbatim reply in 1.C:
1. If the reply is from a prospect in the outreach tracker: update their `response_state` from `pending` → `responded`.
2. If the reply is a public comment on CARD_1: add a row to the inbound log (a new file `inbound_replies_day1.csv` — can be created during ingestion if needed).
3. **Classify each reply** against the 5 reply types in `card1_publish_copy.md` §5. Criteria:

| Operator-visible pattern | Class |
|---|---|
| Named incident (dated, named party role) + workflow-change statement | STRONG |
| Named incident (dated, named party) without workflow-change | MODERATE |
| Vague enthusiasm ("sounds useful", "I'd try it", "cool") with no specifics | WEAK |
| Feature request with no pain anchor | WEAK |
| Active disconfirm with specifics ("my clients don't care") | DISCONFIRMING |
| Silence, likes, RTs only | NO_SIGNAL (not a reply — do not log) |
| Adjacent-segment inbound (enterprise, legal, healthcare) | ADJACENT (log separately, not in A3 count) |

**Rule:** the assistant justifies each classification with a one-line rationale referencing the verbatim text. If the text does not clearly meet STRONG criteria, default to MODERATE or WEAK — never upgrade.

### 2.D → updates outreach tracker + scheduling

For each intake submission in 1.D:
1. Cross-reference timestamp and referral source against outreach tracker to find the matching contact_id (or assign a new one if the submitter was outbound-unsolicited).
2. Update `A3_S4_confirmed`: `Y` for STRONG_S4, `Y (moderate)` for MODERATE_S4, `N` for WEAK_S4, `adjacent` for ADJACENT.
3. Update `response_state`: `qualified` (if S4 passed) / `disqualified` / `released` / `adjacent_segment`.
4. Update `interview_scheduled` with the slot or blank.

### 2.E → creates Template 1 pre-stubs for booked interviews

For each booking in 1.E, the assistant creates a new file `INT-<date>-<segment>-<seq>.md` in the appropriate day folder, pre-filled with:
- Meta fields (date, interview_type=external_evidence, segment, recruitment channel, A3_S4_confirmed per booking)
- Template 1 skeleton ready for the operator to fill during/after the interview
- Reminders: "signed record" wording at Q11, workflow-change probe at Q12, disconfirming-evidence slot required.

### 2.F → writes/updates `founder_prep_interview_FILLED.md`

- Section 12 outputs go into the filled artifact.
- `interview_type: founder_prep` is enforced.
- NO rows added to `EVIDENCE_LOG.csv` from this block.
- Bias bullets (12.4) are flagged for the operator to watch on Day 2.

---

## 3. What gets written, when, where

| Block | Artifacts updated | New artifacts created |
|---|---|---|
| 1.A Publication | `README.md` (Day 1 completed-assets section) | — |
| 1.B Outreach | `OUTREACH_TRACKER.csv` (+ rows) | — (renaming seed → live) |
| 1.C Replies | `OUTREACH_TRACKER.csv` (response_state updates); `MESSAGE_TEST_RESULTS.csv` (CARD_1 row update with reply counts) | `inbound_replies_day1.csv` (verbatim log with classification) if not existing |
| 1.D Intake submissions | `OUTREACH_TRACKER.csv` (state + S4 updates) | — |
| 1.E Bookings | `OUTREACH_TRACKER.csv` (interview_scheduled) | `INT-<date>-<segment>-<seq>.md` pre-stubs |
| 1.F Founder-prep | — (EVIDENCE_LOG unchanged) | `founder_prep_interview_FILLED.md` |

**No evidence-log rows are created on Day 1.** Evidence-log rows only appear starting Day 2, after the first external interview is conducted and the operator has filled Template 1 with real quotes.

---

## 4. Evidence-log row creation (Day 2+)

Once the operator has conducted an external interview and filled a `Template 1` artifact (`INT-<date>-<segment>-<seq>.md`), they paste into the chat:

```
--- INTERVIEW COMPLETED ---
Interview_id: INT-<date>-<segment>-<seq>
interview_type: external_evidence
Segment: A3 | A1 | A8

Key evidence items (one per finding, with quote reference):
1. Hypothesis: H[N]
   Grade (S/M/W/D/N): [letter]
   Workflow-change captured: YES | NO | PARTIAL
   Incident summary (one factual line, no interpretation):
   Raw quote (verbatim, with mm:ss):

2. ...

Disconfirming evidence captured:
   Verbatim:

Objections captured:
   Verbatim with context:

Wording-preference (Q11 probe answer):
   [exact phrase chosen]

Interviewer self-check (all 8 items):
   [ ] specific-incident ...
   [ ] ...
```

The assistant converts this into:

- N `EVIDENCE_LOG.csv` rows (one per finding)
- 1+ `OBJECTION_LOG.csv` rows (per objection listed)
- Updates the interview's Template 1 file if needed

**Classification discipline during conversion:**

- `S` grade REQUIRES workflow-change=YES. If the operator claims S without workflow-change, the assistant downgrades to M and flags.
- Every interview must produce ≥1 disconfirming row. If the operator did not list one, the assistant flags: "no disconfirming evidence — interviewer may have been leading; review before Day 3."
- `tags` column filled per controlled vocabulary (`named_incident`, `client_pressure`, `workflow_change`, `wording_preference`, etc.).

---

## 5. Message-Test-Results row updates

When the operator reports CARD_1 metrics (or subsequent cards):

```
--- MESSAGE TEST UPDATE ---
card_id: CARD_1
wording_variant: signed_record
channel: LinkedIn | X | email
as-of: YYYY-MM-DD HH:MM GST
impressions: <number>
clicks_or_replies: <number>
named_incident_replies: <number — ONLY count verbatim replies that met STRONG classification>
workflow_change_replies: <number — replies with concrete workflow-change statement>
forward_requests: <number — asks for sample/demo>
disconfirms: <number>
notes: <operator notes>
```

Assistant updates or adds the appropriate row in `MESSAGE_TEST_RESULTS.csv`. Signal-score computed per the rule in the templates doc:

- `STRONG`: ≥3 named-incident AND ≥2 workflow-change
- `MODERATE`: 1–2 named-incident OR 1 workflow-change
- `WEAK`: vague positive; enthusiasm without workflow-change
- `NO_SIGNAL`: nothing
- `DISCONFIRM`: ≥5 disconfirms

---

## 6. Paste-once, accept-once protocol

To avoid double-counting or rewriting:

- Operator pastes each day's results ONCE.
- Assistant acknowledges receipt, writes the derived artifacts, and confirms what was written.
- If the operator has corrections, they paste a `--- CORRECTION BLOCK ---` with the specific line to change.
- Assistant does NOT rewrite ingestion artifacts from scratch — it edits the specific row/field.

Corrections block format:

```
--- CORRECTION BLOCK ---
File: <path>
Row: <contact_id or evidence_id>
Field: <column name>
Was: <old value>
Should be: <new value>
Reason: <why>
```

---

## 7. Edge cases

| Case | Handling |
|---|---|
| Operator pastes Day 1 results but founder-prep was skipped | Assistant flags Day 1 as incomplete and refuses to mark intake URL / CARD_1 URL as "Day 1 DONE" until founder-prep is filled. Partial artifacts are written; status reflects incomplete. |
| Operator pastes an outreach list with no replies yet | Fine. Outreach tracker gets rows with `response_state=pending`. No reply-related artifacts created. |
| Operator pastes a reply that looks strong but the verbatim text is vague | Assistant classifies per rubric (not operator claim). If the quote does not meet STRONG criteria, it's MODERATE or WEAK regardless of operator framing. Flag to operator: "Your grading seems elevated vs the verbatim — here's the class I applied and why." |
| Operator pastes contradictory data (e.g., "10 DMs sent" in 1.A summary, 7 rows in 1.B list) | Assistant uses the detailed block (1.B) as source of truth and flags the summary mismatch. |
| Operator asks assistant to predict who will reply | Refuse. Not a prediction task. Wait for real replies. |
| Operator asks to fabricate a "backup" interview if real interviews fall through | Refuse. A missed interview is a missed interview. Log it truthfully. |
| Operator pastes CARD_4 publication as Day 1 | Flag: CARD_4 is scheduled for Day 5 per sprint plan, not Day 1. Ask operator to confirm they meant to break the matched-pair sequence. Do not write until confirmed. |

---

## 8. What the assistant must refuse

- Filling in names, firms, or contact handles the operator did not state
- Synthesizing "likely" replies to a post the operator hasn't received replies to
- Upgrading a MODERATE/WEAK classification because the operator is optimistic
- Writing founder-prep data into the evidence log
- Renaming seed files to live files if no real rows were added
- Creating tracker rows for outreach not listed in block 1.B
- Generating Day 2 simulated results if the operator just asks "what would Day 2 look like?"

If any of these is attempted, the assistant refuses, cites the rule, and asks for the real data.

---

## 9. Acknowledgement pattern (every ingestion session)

At the end of every ingestion pass, the assistant produces:

```
--- INGESTION SUMMARY ---
Date/time ingested: YYYY-MM-DD HH:MM GST
Files updated:
- <path>: +N rows | N field updates
- ...

Files created:
- <path>
- ...

Flags raised:
- [any discipline flags]

Next expected input: <what the operator should bring next — e.g., "Day 2 interview results," or "more CARD_1 reply updates at 48h mark">
```

No commentary beyond this. No "great job!" No interpretation of what the results "mean." The operator does analysis after the sprint ends.

---

**End of Real-Results Ingestion Protocol.** Assistant-usable without ambiguity. Anti-fabrication rules enforced.
