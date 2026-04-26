# Recruitment Copy — A1 Secondary Segment

**Segment:** A1 — Solo sovereign builder (secondary, target 4 interviews)
**Filter:** runs AI locally + CLI-comfortable + open-source-native
**Compensation offered:** $50 gift card, sent after interview
**Tone rule:** plain, builder-to-builder. No ideology sermons. No BIZRA marketing.

---

## 1. Short DM (X, LinkedIn direct message) — ~500 chars

> Hi [name] — I'm running a research study on people who run AI locally on their own hardware (Ollama, LM Studio, custom Rust/Python setups).
>
> Specifically curious how you think about provenance, reproducibility, and "proof of what your AI did" in your own workflow.
>
> 45 minutes, $50 gift card. No pitch — just listening to how you operate today.
>
> Interested? I'll send a 4-question intake.

---

## 2. Email version

**Subject line options:**
- `45 min · $50 · research on local-AI provenance & replay`
- `Research call — how do you prove what your local AI did?`
- `[GitHub handle] — quick research conversation?`

**Body:**

> Hi [first name],
>
> I'm doing external research on how solo developers and builders who run AI locally (Ollama, LM Studio, llama.cpp, local Whisper / embeddings, custom setups) handle provenance and reproducibility.
>
> I'm interested in how you currently:
>
> - decide which AI workload runs locally vs. cloud
> - keep a record of important outputs
> - reproduce or verify a past AI result when it matters
> - react when a cloud AI provider rate-limits you or changes pricing mid-project
>
> 45-minute call, $50 gift card sent after. This is pure research — I'm not pitching anything and I won't turn this into a product call.
>
> Four-question intake form: [LINK]
>
> Thanks,
> [Your name]

---

## 3. Post version (X, LinkedIn, dev newsletters, r/LocalLLaMA, r/selfhosted, Rust community Discord)

> Looking for 4 devs who run AI locally on their own hardware — Ollama, LM Studio, llama.cpp, custom setups.
>
> Specifically want to learn how you handle provenance, reproducibility, and "proof of what the AI did" when it matters.
>
> 45 min, $50 gift card. Not a sales call.
>
> DM or reply — I'll send a short intake.

**Variant for r/LocalLLaMA / r/selfhosted specifically (note community etiquette):**

> [Research] Looking for 4 local-AI practitioners for a paid research call.
>
> Not building anything yet — studying current workflows. Specifically interested in:
>
> - how you track what your local AI did on important tasks
> - whether you've needed to reproduce an output later and what happened
> - how you handle cloud-AI rate-limit / deprecation pain when it hits
>
> 45 min, $50 gift card after the call. Moderator-approved.
>
> Reply or DM if interested. I'll send a 4-question intake form.

---

## 4. Screening form text (4 items)

**Intake: local AI & provenance — 4 questions**

> Four quick questions before we book a time.

**Q1. Local AI.**
> Do you run AI models locally on your own hardware? If yes, which — Ollama / LM Studio / llama.cpp / custom setup / local Whisper / local embeddings / other?
> *[short answer]*

**Q2. CLI comfort.**
> Have you installed a command-line dev tool in the last 90 days and used it more than twice? If yes, which? (Examples: `gh`, `jq`, `ripgrep`, `ollama`, a custom Rust or Go binary.)
> *[short answer]*

**Q3. Open-source evaluation.**
> When you evaluate whether to adopt a tool, do you read its open-source codebase (at least skim) before committing? (Yes / No / Sometimes)

**Q4. Reproducibility / proof incident (softer filter than A3).**
> In the last 90 days, was there a specific moment when you needed to reproduce or prove an AI output and it was hard or impossible? Briefly describe if yes.
> *[optional — YES answer is preferred but not hard-required for A1]*

**End of form.** Candidate must answer Q1=Yes, Q2=Yes, Q3=Yes or Sometimes. Q4 YES with incident is high-priority; Q4 NO is acceptable but lower-priority scheduling.

---

## 5. Disqualifier notes (for operator)

| Condition | Action |
|---|---|
| Q1 = No (cloud-only AI user) | Thank and release. Not an A1. |
| Q2 = No (GUI-only, refuses CLI) | Disqualify — cannot exercise the Dema CLI surface that's under validation. |
| Q3 = No (never reads source) | Soft disqualify — flag, consider only if other signals strong. |
| Employed by Anthropic, OpenAI, Google DeepMind, Meta AI, xAI | Disqualify — strategic contamination. |
| Known BIZRA contributor / prior consultant to founder / friend-of-founder | Disqualify — signal contamination. |
| Q4 = YES with specific incident | High priority — schedule first. |
| Q4 = NO | Lower priority but acceptable — schedule if slots available after Q4-YES candidates booked. |
| Candidate explicitly ideology-first (crypto / sovereignty / Islamic finance enthusiasm without technical work) | This is A6, not A1. Thank and release for this sprint. A6 is not in validation scope. |

### Tagging inside outreach tracker

For every A1 candidate, tag:
- `q4_incident=Y/N` (prioritization signal)
- `primary_use` (Ollama / LM Studio / custom / mixed)
- `referral_source` (important for recruitment funnel analysis)

---

## 6. Operator notes

- A1 interviews are **secondary**. If the A3 recruitment pipeline is hitting shortfall, do NOT substitute A1s to make volume. Segment discipline matters.
- A1 interviewees will naturally engage with technical detail. This is fine — do NOT over-correct by becoming artificially technical; the interview guide is the same for all segments.
- A1s often overlap with A6 (ideological adopters). If the interviewee starts talking about sovereignty as first principle rather than as a workflow choice, tag the interview as A1-with-A6-overlap and reduce weight on their enthusiasm by 30% per the sprint evidence rubric.
- Wording default is still "signed record" (GUARDRAIL 3) — even with CLI-native A1 candidates. Their preference for "receipt" or other crypto vocabulary at Q11 is itself data (H13).
- $50 paid AFTER the call.

---

**End of A1 recruitment copy.** Ready to paste.
