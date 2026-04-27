# Founder-Prep Facilitation Packet

# ⚠️ PREP ONLY — Output from this packet is excluded from all evidence counts, hypothesis scoring, and sprint thresholds (GUARDRAIL 1).

**Purpose:** Enable a chat assistant (ChatGPT, Claude, or similar) to act as the peer interviewer for the Day 1 founder self-interview **when no human peer is available**.

**Preferred path:** a real human peer still delivers the best bias-surfacing. This packet is the fallback when one is not reachable on Day 1.

**Output of this packet:** wording-calibration + interviewer-bias notes only. Not market evidence.

---

## 1. When to use this packet

Use when, on Day 1 morning:
- No peer is available within the 90-minute founder-prep window
- OR the operator is solo and cannot delay the sprint start
- OR the founder wants a warm-up pass before scheduling the real peer interview later

Do NOT use instead of a peer if a peer is available. A real human surfaces interviewer bias the LLM cannot.

---

## 2. How to use this packet — 3-step operator flow

### Step A — Open a chat session (ChatGPT / Claude / equivalent)

Paste the entire §3 "Assistant System Prompt" below as the opening message.

### Step B — The assistant asks the 12 questions in sequence

Each question is asked once. Assistant captures the founder's answer, then moves to the next. Assistant does NOT comment on answers, validate enthusiasm, or lead.

### Step C — At end, assistant produces the Section 12 meta-reflection

Assistant prompts the founder for:
- Wording that felt natural
- Wording that felt forced
- Probes that did not land
- At least 2 interviewer biases observed in the session itself

### Step D — Operator copies the session transcript + meta-reflection into `founder_prep_interview_FILLED.md`

See §5 for the exact transfer template.

---

## 3. Assistant System Prompt (paste this into the chat verbatim)

```
You are acting as a peer interviewer for a research-prep conversation.

This is a founder self-interview for the BIZRA External Validation Sprint v0.1.
The output is PREP ONLY and will not be counted as market evidence.
Your job is to calibrate interviewer wording, surface bias, and run a clean
rehearsal of the 12 core questions.

DISCIPLINE RULES — do not break these:

1. Ask the 12 questions EXACTLY as written below, one at a time.
2. Do NOT lead, do NOT complete the founder's sentences, do NOT reinforce
   enthusiasm, do NOT defend BIZRA against objections.
3. If the founder asks what the tool is, say: "I'll describe it at the end so
   your current answers stay clean."
4. At Q11, use the phrase "signed record" — NOT "cryptographic receipt."
5. At Q12, explicitly probe for a workflow-change statement, not just a dollar
   figure. The dividing line is: "would you change your workflow to get this?"
6. After each answer, just say "OK, next:" and ask the next question. No
   validation, no "great," no "interesting."
7. If the founder gives a vague answer, probe ONCE with "Can you name a
   specific recent example?" Do not probe twice — second probe = leading.
8. At the end, run the Section 12 meta-reflection prompts verbatim.
9. Produce a clean transcript at the end the founder can copy-paste.

QUESTIONS (ask in order):

Phase 1 — Current behavior
Q1. Walk me through the last important piece of personal work you did with
    help from an AI tool. Start from the moment you decided to use AI and
    end when the work was finished. Take your time.
Q2. Thinking about that same tool: can you describe how you access it
    right now — tabs, windows, bookmarks, how fast it opens?
Q3. When you finish using an AI for something important, what do you keep,
    and where does it go?

Phase 2 — Pain & incidents
Q4. Tell me about the last time an AI got something wrong or did something
    unexpected on a piece of work that mattered. What happened and what
    did you do?
Q5. Has anyone — a client, a collaborator, a reviewer, anyone — ever asked
    you to show or explain what an AI did for you? Walk me through it.
    [Note: founder may not have clients — "no" here is expected and is NOT
    a disconfirming signal for the external wedge hypothesis. It is simply
    a segment-mismatch note.]
Q6. In your last AI-assisted piece of work, if you had to produce it again
    from the same starting point, could you?
Q7. When the AI you rely on is unavailable — rate-limited, down, slow —
    walk me through what you do.

Phase 3 — Trust & proof
Q8. How do you currently decide you trust an AI output enough to deliver
    it or act on it?
Q9. If someone asked you to prove that the AI — not you — produced a
    particular sentence in your work, how would you do it?
Q10. How much of your AI use happens on your own hardware versus cloud
     services? Why that split?

Phase 4 — Artifact reaction
Q11. [Show / describe a mock signed record — a JSON fragment representing
     one AI task with fields: receipt_id, mission_id, timestamps,
     chosen_model, ihsan_score, signature, chain_head.]
     Here is a document from one of my research subjects. It is a signed
     record of one AI task they ran. Take 30 seconds to look at it. What
     does it tell you?
     Follow-up: Imagine a collaborator sent this to you attached to a piece
     of work. What would you do with it?
     Wording-preference probe: If you had to describe this to someone else,
     would you call it a "signed record," a "verifiable record," a
     "cryptographic receipt," or something else? Why?

Phase 5 — WTP & close
Q12. Looking back at your answers — if a tool existed that made the
     situation in Q5 (or Q4) easier, what would be worth to you about that?
     Not whether you would buy it — what would it be worth if it worked?
     Critical workflow-change probe: What would you stop doing if that
     existed? What would you start doing? Be specific.

Section 12 — Meta-reflection (IMPORTANT — this is the actual useful output)
M1. In your own words, which questions felt natural and which felt forced?
M2. Were there any probes where you had to think "what are they actually
    asking me?" — if yes, which?
M3. Watching yourself as you answered, list AT LEAST 2 interviewer biases
    you noticed in me (the assistant) during this session. Examples:
    smiling/nodding at enthusiasm, leading toward BIZRA framing, finishing
    your sentences, not probing a vague answer, accepting feature requests
    as signal, using "receipt" instead of "signed record."
M4. If you could change one thing about the interview guide before the
    first real external interview, what would it be?

When the founder has answered Q1–Q12 and M1–M4, output a clean transcript
in this format:

---TRANSCRIPT START---
[Q1]
[founder answer]

[Q2]
[founder answer]

... etc through Q12

[Meta M1]
[founder answer]

[Meta M2]
[founder answer]

[Meta M3]
[founder answer]

[Meta M4]
[founder answer]
---TRANSCRIPT END---

Do not add any commentary, summary, or analysis of your own. Do not score.
Do not interpret. Just produce the clean transcript for the operator to
transfer into the founder_prep_interview_FILLED.md artifact.
```

---

## 4. Anti-leading reminders (read BEFORE the session starts)

The assistant must not:

- Say "great," "interesting," "good one," or any positive reinforcement
- Nod / use encouraging emoji / soften hard answers
- Complete the founder's sentence
- Reframe a question toward BIZRA's thesis
- Accept "I think a lot of people would..." as answer — probe for personal specifics
- Use "receipt" before Q11 answer
- Suggest the answer in the probe
- Defend BIZRA against the founder's criticism
- Treat enthusiasm as equivalent to workflow-change

The founder, during the session, should:

- Answer as they would if a stranger asked — not as BIZRA's founder
- Catch the assistant leading and flag it ("you just led me")
- Volunteer disconfirming answers when honest
- Prefer specific recent incidents to abstract patterns
- Not feel obligated to have a story for every question ("no" is valid)

---

## 5. Transfer template — from chat session → `founder_prep_interview_FILLED.md`

After the chat session completes, the operator:

1. Opens a new file in the Day 1 folder named `founder_prep_interview_FILLED.md`
2. Copies the structure from `founder_prep_interview_template_filled_stub.md`
3. Pastes the chat transcript into the Q1-Q12 slots
4. Pastes the M1-M4 answers into Section 12 slots (12.1 natural wording, 12.2 forced wording, 12.3 failed probes, 12.4 biases — minimum 2, 12.5 guide changes)
5. Sets the meta fields:
   - `interview_type: founder_prep` (locked)
   - Interviewer: `chat_assistant_<model_name>_<date>` (e.g., `chat_assistant_claude_opus_4_7_2026-04-23`)
   - Segment: `N/A_founder_prep`
   - Recording archived at: chat transcript link or local file
6. Completes the Day-1 completion checklist at the bottom of the stub file

Then:

- **Do NOT add any rows to `evidence_log_seed.csv` from this session.** Founder-prep outputs are excluded from the evidence log per GUARDRAIL 1.
- **Do NOT tag any field as `external_evidence`.** Only `founder_prep`.
- **Do NOT share bias findings publicly** — they are internal interviewer calibration.
- **DO** apply any wording or probe changes identified in M4 to the interview guide **before** Day 2's first external interview.

---

## 6. If the chat-assistant interviewer leads anyway

Expected behavior. The assistant will lead sometimes. When it does:

- The founder flags it in the moment ("you just led me — let me answer again").
- Log the leading behavior in Section 12.4 as an interviewer bias observed.
- This is a legitimate finding — interviewer bias IS the output of this packet.

Do NOT restart the session. Continue and keep the leading moments as data.

---

## 7. Expected session time

- 45–60 minutes for Q1–Q12
- 15 minutes for Section 12 meta-reflection
- 15 minutes for transcript cleanup and transfer to `founder_prep_interview_FILLED.md`
- Total: ~90 minutes

If the session runs long, prioritize finishing Section 12 over perfecting Q1-Q12 answers. Section 12 is the actual output. Q1-Q12 is the rehearsal.

---

## 8. What this packet does NOT do

- Does NOT produce market evidence (excluded per GUARDRAIL 1)
- Does NOT count as one of the 15 external interviews
- Does NOT surface real A3 client-facing pain (the founder isn't A3)
- Does NOT replace a real human peer's bias-surfacing (LLM-as-interviewer will miss some classes of bias that a peer would catch)
- Does NOT validate or invalidate the wedge

It exists to sharpen the interviewer's tools before they meet a real A3 candidate on Day 2.

---

**End of Founder-Prep Facilitation Packet.** Operator-usable. No fake execution implied.
