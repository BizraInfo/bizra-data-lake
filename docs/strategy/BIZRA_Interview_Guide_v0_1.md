# BIZRA External Validation — Interview Guide v0.1

**Date:** 2026-04-23 (GST)
**Scope:** primary = A3 (AI-native creator/consultant), secondary = A1 (solo sovereign builder), stretch = A8 (hyperscaler-dependency-anxious dev)
**Companion:** `BIZRA_External_Validation_Hypotheses_v0_1.csv`, `BIZRA_2_Week_Validation_Sprint_v0_1.md`

---

## 0. Interview Discipline

### Three rules that supersede everything else

1. **Never ask if the interviewee would like BIZRA.** BIZRA is not even named in the script. Ask about **current behavior, current pain, current proof needs**. The moment you pitch, the interview is dead.
2. **Never explain receipts until Phase 4.** Phases 0–3 observe reality as the interviewee already lives it. Explaining too early contaminates the signal.
3. **[CONTAMINATION CONTROL — present vs future state]** **Keep interviewees in present-day lived behavior.** If any answer drifts into what they are building, planning, designing, or wishing the tool would do, interrupt with:
   > *"Pause. Is that what you do today, or what you want the system to do eventually?"*
   If the answer is future-state, redirect:
   > *"Stay with today. What do you actually do now?"*
   **Present-day answers are evidence. Future-state answers are not.** Drift happens most often at Q7, Q8, and Q10 — see contamination-control callouts in those question blocks. Added 2026-04-23 from founder-prep M4 / 12.5 finding.

### Session shape

- 45 minutes, video call with screen-share encouraged
- Recorded with consent; transcript + notes archived in `evidence log` template
- Interviewer must not demo, pitch, or roleplay as BIZRA sales
- Every question has an **evidence capture prompt** — the interviewer must write down the specific artifact, incident, or example the interviewee names, not their opinion

### Anti-leading rules

- Do not use the words: *sovereign, constitution, proof, receipt, chain, governance, agent* in the first 30 minutes
- Do not complete the interviewee's sentences
- If the interviewee asks "what's the tool?" — deflect: *"I'll describe it at the end so your current answers stay clean."*
- If the interviewee offers a feature suggestion — capture but do not reinforce
- If the interviewee says "this is just like X" — follow up on X, do not defend BIZRA

---

## 1. Screening Questions (for recruitment channels)

Use these to qualify candidates. Disqualify is OK.

**External wording defaults (GUARDRAIL 3):** When Phase 4 artifact-reaction or Phase 5 value probe requires naming the thing BIZRA produces, default to "signed record," "verifiable record," or "proof of what the AI did." Do NOT lead with "cryptographic receipt" — that vocabulary is internal/technical. Introduce "cryptographic receipt" only if the interviewee spontaneously asks what mechanism underlies the record, or if wording-preference probe (end of Q11) requires a second variant for comparison.

### Primary segment (A3) — must meet all 4 (GUARDRAIL 2, tightened 2026-04-23)

- S1. **Client-facing:** Do you currently deliver work to paying clients, employer stakeholders, or external collaborators who receive specific deliverables from you (not internal sketches — actual deliverables)?
- S2. **AI in delivery:** In the last 30 days, has an AI tool (Claude, ChatGPT, Cursor, Copilot, etc.) contributed directly to what you delivered — not just to your research or note-taking?
- S3. **Recurring:** Is this a recurring part of your workflow (weekly or more), not a one-off experiment?
- S4. **Recent incident (HARD FILTER):** In the last 90 days, have you experienced at least one credibility, proof, or revision dispute with a client or collaborator tied to work that involved AI — where you were challenged on *what the AI did, whether it was correct, whether it should have been used, or whether its output could be trusted*?

Without S4, disqualify. Loose "I might have faced something like that" does not count — require a specific incident the interviewee can place in time and name the party who raised it.

### Secondary segment (A1) — must meet all 3

- S4. Have you installed a command-line tool (anything from `gh` to `ollama` to a custom Rust binary) in the past 90 days and used it more than twice?
- S5. Do you run any AI locally on your own machine (Ollama, LM Studio, llama.cpp, local Whisper, local embedding models, etc.)?
- S6. Do you read open-source codebases as part of evaluating whether to adopt a tool?

### Stretch segment (A8) — must meet at least 2

- S7. In the past 90 days, has a cloud AI provider's rate limit, quota, pricing change, or deprecation notice affected a project you were working on?
- S8. Do you currently pay for more than one frontier-model subscription or API?
- S9. Have you attempted to swap one AI backend for another in production or near-production code in the past year?

### Disqualifiers (do not interview)

**Universal:**
- Has never used AI in work
- Only uses GUI-only tools and refuses CLI (disqualifies A1, softer for A3)
- Is a BIZRA contributor / prior consultant to the founder (contaminates signal)
- Is employed at an AI-provider company (Anthropic, OpenAI, etc.) — risk of strategic interest coloring answers

**A3-specific (GUARDRAIL 2):**
- Does not currently deliver work to external clients / stakeholders — internal work only
- Uses AI only for research/personal productivity, never in actual deliverables
- Has no recent (last 90 days) credibility / proof / revision incident tied to AI work — even if S1-S3 pass
- Can only describe incidents in the abstract ("clients worry about AI") with no specific dated example

If S4 fails, thank the candidate and redirect to a later cohort once more A3 volume is available. Do not interview a weak-S4 candidate as a "gentler" A3 — it contaminates H1/H2/H3 signal.

---

## 2. Interview Structure (45 min)

| Phase | Minutes | Focus |
|---|---|---|
| 0. Context | 0:00–0:05 | Consent, recording, confidentiality |
| 1. Current behavior | 0:05–0:20 | How do you actually work today? |
| 2. Pain & incidents | 0:20–0:32 | Specific recent moments of friction |
| 3. Trust & proof | 0:32–0:40 | How do you decide to trust AI output? |
| 4. Artifact reaction | 0:40–0:43 | Show mock receipt (ONE only); capture response |
| 5. WTP & close | 0:43–0:45 | Price probes + what they would stop doing |

---

## 3. Core Interview Questions (12)

Each question carries an **evidence capture prompt** — the specific item the interviewer must write down. Follow-up probes are in *italics*.

### Phase 1 — Current behavior

**Q1.** *Walk me through the last important piece of work you did with help from an AI tool. Start from the moment you decided to use AI and end when the work was delivered. Take your time.*
→ Evidence: Tool used, project description, start/end time, number of AI calls, final deliverable type.
→ Probe: *Where is that output now? Can you find it?*

**Q2.** *Show me: in front of me right now, pull up the AI tool you used most in the last week. Open it.*
→ Evidence: Observe tool choice, UI familiarity, speed of opening, tab organization.
→ Probe: *Where do the outputs from this tool live after you close the window?*

**Q3.** *When you finish using an AI for something important, what do you keep, and where does it go?*
→ Evidence: Retention strategy — screenshots, copy-paste into docs, no retention, custom logs, etc.
→ Probe: *Can you find an output from last week? Show me.* Observe whether they actually can.

### Phase 2 — Pain & incidents

**Q4.** *Tell me about the last time an AI got something wrong or did something unexpected on a piece of work that mattered. What happened and what did you do?*
→ Evidence: Specific incident, severity, resolution behavior.
→ Probe: *How did you figure out it had gone wrong?*

**Q5.** *Has anyone — a client, a collaborator, a boss, a reader, a regulator, anyone — ever asked you to show or explain what an AI did for you? Walk me through that.*
→ Evidence: YES/NO + who asked + what the asker wanted + how the interviewee responded + how long it took.
→ Probe: *What would you do differently if the same thing happened tomorrow?*

**Q6.** *In your last AI-assisted project, if you had to produce it again from the same starting point, could you?*
→ Evidence: Confidence, actual reproducibility, what would break.
→ Probe: *Let's pick a specific output from last week. If I asked you to reproduce it right now, what happens?*

**Q7.** *When the AI you rely on is unavailable — rate-limited, down, slow — walk me through what you do.*
→ Evidence: Backup behavior, fallback tools, pain tolerance.
→ Probe (A8 stretch): *Has that happened recently? Tell me about the most recent time.*
→ **[CONTAMINATION CONTROL — present vs future state]** If the interviewee describes a designed/intended fallback system rather than what they actually do today, interrupt: *"Pause. Is that what you do today, or what you want the system to do eventually?"* If future-state: *"Stay with today. What do you actually do now?"* Future-state answers do NOT count as current-behavior evidence.

### Phase 3 — Trust & proof

**Q8.** *How do you currently decide you trust an AI output enough to deliver it or act on it?*
→ Evidence: Verification ritual, skepticism signals, sanity checks.
→ Probe: *What would make you distrust an output you'd initially accepted?*
→ **[CONTAMINATION CONTROL — present vs future state]** If the interviewee describes an ideal/designed trust architecture rather than how they currently decide, interrupt: *"Pause. Is that what you do today, or what you want the system to do eventually?"* If future-state: *"Stay with today. What do you actually do now?"* Future-state answers do NOT count as current-behavior evidence.

**Q9.** *If someone asked you to prove that the AI — not you — produced a particular sentence in your work, how would you do it?*
→ Evidence: Honest description of what is and isn't possible in their current setup. Screenshot? Git log? Nothing?
→ Probe: *What would be good enough for a client to accept?*

**Q10.** *How much of your AI use happens on your own hardware versus cloud services? Why that split?*
→ Evidence: Cloud/local ratio, reasons (privacy, cost, speed, quality).
→ Probe: *What would make you shift that split?*
→ **[CONTAMINATION CONTROL — present vs future state]** If the interviewee compresses a mixed reality into a clean ideological answer ("it all happens locally" / "it's all cloud"), interrupt: *"Pause. Is that what you do today, or what you want the system to do eventually?"* If future-state: *"Stay with today. What do you actually do now?"* Push for the concrete split across all AI tools they actually touch this week. Future-state answers do NOT count as current-behavior evidence.

### Phase 4 — Artifact reaction

*[At 40 min mark, ONLY, show the interviewee a single mock artifact. A JSON fragment on-screen OR printed. In spoken framing call it a "signed record of one AI task" — GUARDRAIL 3 wording default.]*

**Q11.** *Here's a document one of my research subjects produced recently. It represents a signed record of one AI task they ran. Take 30 seconds to look at it. What does it tell you?*
→ Evidence: Can they read it? Do they intuit what's in it? Which field do they look at first? Do they ask what anything means?
→ Probe: *Now imagine a collaborator sent this to you attached to a piece of work. What would you do with it?*
→ **Wording-preference probe (H13):** *If you had to describe this to a client, would you call it a "signed record," a "verifiable record," a "cryptographic receipt," or something else? Why?* Capture exact word choice.

### Phase 5 — WTP & close

**Q12.** *Looking back at your answers today — if a tool existed that made the situation you described in Q5 [or Q4 if no Q5 incident] easier, what would be worth to you about that? Not whether you'd buy it — what would it be worth if it worked.*
→ Evidence: Dollar figures, units (per-output, per-month, per-project), threshold language ("only if", "definitely if").
→ **CRITICAL workflow-change probe (H3):** *What would you stop doing if that existed? What would you start doing? Walk me through a specific thing that would change.*
→ **Scoring rule:** An answer to "what's it worth" without a concrete workflow-change answer is **WEAK SIGNAL only**. The dividing line is not "interesting" — it is "would you change workflow to get this." Enthusiasm without workflow-change implication = weak.

---

## 4. Segment-specific probes

### For A3 (primary)

- *Do you bill clients in a way that references AI work? How?*
- *Have you ever refused a refund or disputed a scope claim where AI output was central?*
- *Do you have clients who are themselves skeptical of AI? What do they want to see?*
- *If one of your AI outputs becomes the subject of a legal dispute, what do you have?*

### For A1 (secondary)

- *Show me your terminal. What's in your shell history?*
- *Which open-source AI infra have you installed and then uninstalled? Why?*
- *How do you personally decide whether to depend on a hosted tool vs self-host?*
- *What did you install most recently that surprised you with how well it worked?*

### For A8 (stretch)

- *Show me your billing dashboard for any AI provider. Talk me through the last 3 months.*
- *Have you written glue code to abstract over AI providers? Show me.*
- *When a provider changes pricing, how do you find out and what do you do?*
- *What would convince you to route all your AI through one tool instead of N?*

---

## 5. Evidence capture prompts (used throughout)

For every interview, the interviewer must fill:

- [ ] ONE specific incident from Q5 or Q4 with date + client/recipient + resolution
- [ ] ONE observed behavior from Q2 or Q3 (screen-share observation)
- [ ] ONE specific tool the interviewee names as their "most-used AI tool"
- [ ] ONE reproduced-output result (from Q6 — did it actually reproduce or not?)
- [ ] ONE artifact-reaction note from Q11 (what field did they look at first?)
- [ ] ONE WTP anchor from Q12 (a number or a threshold)
- [ ] ONE thing the interviewee said that DISCONFIRMS the wedge hypothesis (hardest and most important)

Interviews with no disconfirming evidence logged are suspected of interviewer leading — flag for peer review.

---

## 6. Trust/proof-sensitivity probes (interleaved)

Use sparingly within Phase 3. Not every interview needs all of these.

- *When a vendor provides "proof" of something, what do you actually check?*
- *Have you ever been burned by a tool that claimed to do X and didn't? What was the tell?*
- *If an AI claimed it had checked something, what would make you believe the claim?*
- *How much effort would you spend verifying an AI output that mattered?*

---

## 7. Local-first sensitivity probes

Use in Q10 and within A3/A1 follow-ups.

- *Is there work you would not send to a cloud AI? What kind?*
- *If a client asked "does this ever leave my machine?" — what's your honest answer?*
- *Do you run anything on-device today that you'd have been fine running in the cloud 3 years ago?*

---

## 8. Closing questions

**C1.** *Is there anything important about how you work with AI that I haven't asked about?*
**C2.** *Who else does work like yours that I should talk to?* (referral — critical for sample expansion)
**C3.** *Would you be open to a 15-minute follow-up in 2 weeks if I have a concrete thing to show you?* (gate for concierge demo phase)
**C4.** *Is it OK if I quote anonymized versions of what you said in internal strategy docs?*

---

## 9. Interviewer self-check (after each session)

Before closing notes, the interviewer answers:

- [ ] Did the interviewee name at least one specific recent incident? (Not hypothetical.)
- [ ] Did I observe at least one behavior via screen-share?
- [ ] Did I refrain from pitching or naming BIZRA until Phase 4?
- [ ] Did I capture at least one disconfirming signal?
- [ ] Did I avoid feature-discussion loops?
- [ ] Did the interviewee's language or examples come from their domain, not from mine?
- [ ] Did I default to "signed record" wording and capture the wording-preference probe at Q11? (GUARDRAIL 3)
- [ ] Did I score any enthusiasm-without-workflow-change answer as WEAK, not STRONG? (workflow-change dividing line)

If any box is NO, note it in the evidence log and review with a second researcher before scoring.

---

## 10. What NOT to do

- Do not show BIZRA documentation or a product screenshot
- Do not say "imagine" more than once per interview
- Do not explain the canon, the invariants, the chain, or the thesis
- Do not attempt to recruit the interviewee as an early user during the call
- Do not treat enthusiastic reactions as signal unless paired with a specific behavioral claim
- Do not treat "I would use that" as a confirmed hypothesis — only treat specific actions or incidents as evidence
- Do not skip the disconfirming-evidence slot in the template — every interview must produce one

---

**End of Interview Guide v0.1.** Pair with `BIZRA_Validation_Data_Capture_Templates_v0_1.md` for note templates.
