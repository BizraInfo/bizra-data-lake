# Day 1 Launch Folder — BIZRA External Validation Sprint v0.1

**Date:** 2026-04-23 (GST)
**Folder purpose:** Working artifacts for Day 1 of the 2-week external validation sprint.
**Status:** **ALL ASSETS READY. NOTHING PUBLISHED. NOTHING INTERVIEWED.**
**Scope:** Research/validation working state. Uncommitted. Not to be mistaken for runtime code or product.

---

## ⚠️ Truth discipline

- **Nothing in this folder represents completed work until an operator explicitly executes it.**
- Template stubs are STUBS, not filled artifacts. They become filled when the operator runs the action.
- Copy drafts are DRAFTS, not published posts. They become published only when the operator clicks post.
- CSV seed files contain ONE labeled example row each. Delete the example before logging real data.
- The founder self-interview stub, when filled, is **PREP ONLY** — excluded from all evidence counts per GUARDRAIL 1.

If anyone reads this folder and assumes interviews happened or posts went live, they are wrong. Use the outreach tracker as the source of truth for what has actually been done.

---

## Files in this folder (8)

| # | File | Status | Who acts |
|---|---|---|---|
| 1 | `README.md` | ← you are here (index) | — |
| 2 | `founder_prep_interview_template_filled_stub.md` | STUB — ready to fill on Day 1 morning | Founder + a peer as interviewer |
| 3 | `recruitment_copy_A3_primary.md` | READY — paste into outreach channels | Operator (DMs, emails, posts) |
| 4 | `recruitment_copy_A1_secondary.md` | READY — paste into outreach channels | Operator |
| 5 | `card1_publish_copy.md` | READY — click-post decision by operator | Operator |
| 6 | `day2_execution_packet.md` | READY — one-page Day 2 instructions | Operator on Day 2 morning |
| 7 | `evidence_log_seed.csv` | SEED — contains 1 example row to delete | Operator (delete example, add real external-evidence rows) |
| 8 | `outreach_tracker_seed.csv` | SEED — contains 1 example row to delete | Operator (delete example, log all outreach responses) |

---

## Day 1 execution order (what the operator does today)

1. **Morning — Fill the founder prep stub** (`founder_prep_interview_template_filled_stub.md`).
   - Have a peer interview the founder with the 12 core questions.
   - Save the filled artifact as `founder_prep_interview_FILLED.md` in this folder.
   - Record ≥ 2 interviewer biases in Section 12.4.
   - This is PREP ONLY — excluded from evidence counts.

2. **Publish CARD_1** using `card1_publish_copy.md`.
   - Post on LinkedIn (mandatory).
   - Post on X (mandatory).
   - Optional: email/newsletter blurb.
   - Include tracking link/UTM pointing to the A3 intake form.

3. **Deploy A3 recruitment outreach** using `recruitment_copy_A3_primary.md`.
   - Send short DM to 20-30 first-wave A3 prospects.
   - Post the longer post variant on LinkedIn (overlap with CARD_1 is fine — they're two different angles on the same study).
   - Ensure the A3 screening form (Q1-Q4 in `recruitment_copy_A3_primary.md`) is live and linked.

4. **Deploy A1 recruitment outreach** using `recruitment_copy_A1_secondary.md`.
   - Lower-volume — aim for 8-12 prospects touched today.
   - Channels: GitHub contributors to Ollama/llama.cpp, r/LocalLLaMA, Rust Discord.

5. **Log everything in `outreach_tracker_seed.csv`** (rename to `OUTREACH_TRACKER.csv` once real rows exist).
   - Every DM, every email, every post impression tagged with a `contact_id`.
   - Update `response_state` as replies arrive.
   - Confirm `A3_S4_confirmed` only after the candidate answers the screening Q4 with a specific, dated, named-party incident.

6. **By end-of-day, schedule at minimum 2 A3 interviews for Day 2.**

7. **Read `day2_execution_packet.md`** before logging off for Day 1.

---

## Hard rules for this folder

- **Do NOT fabricate evidence** in any seed CSV. Delete the example row the moment real data starts.
- **Do NOT tag a founder-prep row as `external_evidence`** in the evidence log.
- **Do NOT book an A3 interview without a confirmed Q4 incident.**
- **Do NOT publish CARD_4 (cryptographic-receipt variant) on Day 1** — it runs on Day 5 per sprint plan.
- **Do NOT commit this folder to git.** It is uncommitted working state per session scope.
- **Do NOT substitute an A1 or A8 candidate for an unfilled A3 slot** — segment discipline.

---

## How this folder evolves

At end of Day 1, this folder retains all content (nothing deleted).
At end of Day 2, a sibling folder is created: `docs/strategy/validation_runs/2026-04-24_day2/`.
Each day gets its own sibling folder. The root `docs/strategy/` directory holds the permanent sprint canon (hypothesis register, interview guide, message cards, sprint plan, templates). The `validation_runs/` subtree holds the execution artifacts per day.

Pattern: `docs/strategy/validation_runs/YYYY-MM-DD_dayN_<tag>/`

At sprint end (Day 14), a post-sprint synthesis folder will contain:
- `EVIDENCE_LOG.csv` — final aggregate
- `OBJECTION_LOG.csv` — final aggregate
- `MESSAGE_TEST_RESULTS.csv` — final aggregate
- `HYPOTHESIS_PASS_FAIL.md` — per-hypothesis verdict
- `WEDGE_REVISION_LOG.md` — if wedge moved
- `post_sprint_decision_memo.md` — PROCEED / NARROW / REFRAME / KILL

---

## Cross-reference to sprint canon

All copy, screening forms, evidence classification, and decision thresholds in this folder derive from:

- `docs/strategy/BIZRA_External_Validation_Hypotheses_v0_1.csv`
- `docs/strategy/BIZRA_Interview_Guide_v0_1.md`
- `docs/strategy/BIZRA_Message_Test_Cards_v0_1.md`
- `docs/strategy/BIZRA_2_Week_Validation_Sprint_v0_1.md`
- `docs/strategy/BIZRA_Validation_Data_Capture_Templates_v0_1.md`

If any working artifact in this folder appears to contradict the sprint canon above, the canon wins. Adjust the working artifact.

---

**End of Day 1 launch folder README.** Operator-ready.
