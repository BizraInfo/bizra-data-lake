# ⚠️ PREP ONLY — Excluded from All Evidence Counts and Sprint Thresholds (GUARDRAIL 1)

> **interview_type: `founder_prep`** — NOT aggregated, NOT scored, NOT used in any proceed/narrow/reframe/kill decision.
>
> Purpose: interviewer-bias surfacing, wording debugging, probe rehearsal.

---

## Meta

- Date / Time (GST): 2026-04-23 (session in chat, exact GST timestamp per operator's local record)
- Duration (min): ~session in chat, approximately 60–75 min equivalent
- Interviewer: `chat_assistant_claude_opus_4_7[1m]_2026-04-23`
- **interview_type: `founder_prep`** (locked)
- **Segment: `N/A_founder_prep`** (locked)
- Recruitment channel: N/A — founder dogfood
- A3 hard-filter S4 confirmed: N/A
- Consent recorded: self (founder acting as interviewee)
- Recording archived at: chat transcript in the active session (operator responsibility to preserve)
- Notetaker/transcript source: chat session verbatim

---

## Phase 1 — Current behavior

### Q1. Last important AI-assisted work

The last important piece was yesterday — patching the external validation research package. I had five research artifacts that needed three guardrails applied consistently across all of them, and I knew if I did it alone I'd miss cross-references.

I woke up around 5 AM GST, made coffee, and sat down with the intention of fixing a specific problem: the founder self-interview was being treated as market evidence, the A3 recruitment filter was too loose, and the external wording was leading with "cryptographic receipt" instead of "signed record." I had already sketched the three guardrails in my head the night before, but turning that into surgical edits across five files — a CSV, three markdown docs, and a sprint plan — felt like surgery with a blunt knife if I did it manually.

I opened Claude Code CLI in the `bizra-data-lake` repo. I didn't start with a vague "fix these files." I wrote a specific prompt: *"Apply these three guardrails across these five files. Do not create new files. Do not simulate execution. Produce a report showing exactly what changed in each file."* I pasted the guardrail definitions and the file paths.

What happened next was about three hours of back-and-forth. The first pass was good but missed the cross-reference between the renumbered message cards and the template columns. I caught it because I know the card order matters for the H13 matched-pair test. I stopped the assistant and said, *"CARD_1 in the templates must map to the new CARD_1, not the old one. Verify this."* It corrected the mapping.

Then I realized the interview guide Q11 still said "mock receipt" in one place instead of "mock signed-record artifact." I found that myself during review, flagged it, and the assistant patched it. That kind of granular error — one word in one line — is exactly why I use AI for this. I would have missed it after the third hour. The AI doesn't get tired.

By 8:30 AM I had the report: five files patched, 9,761 bytes net increase, zero new files, zero runtime actions. I read the entire diff twice. I didn't accept it blindly. I verified that H13 appeared in both the CSV and the interview guide, that the `interview_type` field propagated from Template 1 to Template 2, and that the "workflow-change" dividing line was visible in six places: rubric, Q12, self-check, message cards, Template 2, and thresholds.

The work finished when I saved the files, ran a bash check to confirm only the five target files were touched, and wrote the stop condition: *"No git. No runtime. No MEMORY.md. No publishing. No simulation."* Then I closed the session and made breakfast.

If I hadn't used the AI, I would have either spent six hours doing it slower and sloppier, or I would have shipped inconsistent cross-references that would have contaminated the validation sprint. The AI didn't replace my judgment — it amplified my precision at a scale I can't sustain alone for that long.

### Q2. Observed tool opening

I access Claude Code CLI through a terminal window on my MSI workstation. I'm running native Linux now — I migrated off WSL Ubuntu 24.04 about a week ago because I needed bare-metal performance for the Rust builds and the determinism tests. The machine boots into a stripped-down Ubuntu with i3wm. No desktop fluff.

I have three terminal workspaces pinned:
- Workspace 1: `~/bizra-data-lake` — strategy docs, validation artifacts, receipt-lineage files. I open this with `claude` and it resumes the last session instantly, maybe 2-3 seconds.
- Workspace 2: `~/bizra-core` — Rust source, Dema CLI, SAPE engine. Same invocation, same speed.
- Workspace 3: Scratch — ephemeral experiments, quick compiles, or clean-context work.

I don't use bookmarks. I use directory memory. Shell history is sacred — aliases like `bdl` for the data-lake repo and `bcr` for the core repo. Type `claude`, hit enter. Opens in under three seconds because I disabled telemetry and auto-updates; I control when it updates manually.

One long-running session per workspace. Data-lake session has 14823 commands in transcript as of this morning. I don't restart unless I hit a context ceiling or need a clean slate. When I do restart, I paste a system instruction header that locks the guardrails.

On mobile: Z Fold 6 with Termux for emergency SSH; don't run Claude Code CLI from it. Screen too small for diff review.

Access is frictionless. That's the point. If it took more than five seconds or required clicking through a GUI, I would have built my own wrapper. But `claude` in the right directory, context restored, guardrails loaded — that's the fastest path from intention to execution.

### Q3. Retention strategy

I keep three things, and they go to three different places.

**First: the artifact itself.** If the AI produced code, it goes into a git commit with a descriptive message. If it produced text or strategy docs, it goes into the repo as a markdown file with a timestamp in the filename. I don't save to cloud drives. Everything lands in a git-tracked directory.

**Second: the transcript.** I don't keep full chat logs by default — too noisy. But I keep *decision points*. If there was a back-and-forth where I caught an error or clarified a requirement, I extract that thread and paste it into a `decision_log.md` or a comment block. Personal rule: if the AI suggested something I rejected, I write one line explaining why. That lives in the repo too.

**Third: my own notes.** After a heavy session, I write a short paragraph — sometimes five sentences — describing what I intended, what actually happened, and what I would do differently. This goes into a private `session_notes/` directory that is git-ignored. For me, not the project.

I don't keep screenshots. I don't save chat URLs. I don't use Notion or Obsidian. My memory is the git history plus those session notes. If I can't reconstruct from `git log --all --source --full-history`, then it wasn't important enough to keep.

The honest answer is: I keep less than most people think they should. I'm paranoid about clutter. Three things. Everything else is noise.

## Phase 2 — Pain & incidents

### Q4. Last wrong/unexpected AI incident

The last time this mattered was during the validation-package work I mentioned. Claude Code didn't fail in a dramatic way — it failed in a more dangerous way, which is that it produced something that looked finished but wasn't internally consistent.

I had asked it to apply a set of guardrails across several research files. The first pass was mostly good, but after a card-order change, one of the downstream references no longer matched. So the package looked coherent on the surface, but one part of the logic had drifted underneath.

That mattered because this wasn't just a writing task. Those files were meant to drive a real validation sprint. If I had accepted the output too quickly, I could have carried a broken mapping into the interview process and contaminated later evidence.

What I did was stop, review the diff more carefully, isolate the mismatch, and then give the AI a much narrower correction task. Instead of asking it to "clean up the package," I asked it to verify the specific reference chain that had drifted and patch only that. Then I re-read the changed files to make sure the fix had actually propagated.

So the unexpected part wasn't that the AI was useless. It was that it was confidently incomplete. It got me 80 or 90 percent of the way there very fast, but the remaining 10 or 20 percent was exactly the part that required judgment and verification from me.

### Q5. Client / collaborator asked for proof

- [ ] YES with incident
- [ ] YES but vague
- [x] NO (with soft-pressure pattern noted)

No — not in that exact concrete form from a client or collaborator where someone said, on a specific date, "show me exactly what the AI did here."

What I've had instead is softer pressure: reviewers or collaborators asking how I got to an output, what part was mine versus AI-assisted, and how much I had personally verified. So the underlying trust question has been there, but not as a clean named incident with a date and a direct demand for AI proof from someone external.

So if you're asking for a strict recent example with date, person, and explicit request, the honest answer is no.

**Note (per packet):** this is an expected segment-mismatch. Founder is not A3. Not a disconfirming signal for the external wedge.

### Q6. Reproducibility probe

Not cleanly, no.

I could probably reproduce the broad outcome, but I could not honestly guarantee the exact same path or the exact same artifact from the same starting point without a much stronger replay structure than I had at the time.

The reason is that the work depended on iterative prompting, review decisions, narrowed correction passes, and judgment calls I made along the way. Some of that is recoverable from files, diffs, and session traces, but some of it lives in the sequence of interactions and the exact moments where I intervened.

So if the question is "could you get back to something similar?" then yes, probably. If the question is "could you deterministically reproduce the same result from the same starting point and prove it?" then no, not with the workflow as it existed in that moment.

That gap is actually one of the reasons this problem matters to me. Right now, most AI-assisted work is reproducible only in a loose human sense, not in a strict verifiable sense.

### Q7. AI-unavailable-backup behavior

It breaks my flow state completely. When you are deep in a complex Rust refactor and hit a rate limit or a 502 Bad Gateway, losing that momentum is devastating.

Historically, I would have to either wait it out or manually copy-paste my entire context window into a different web UI, which completely destroys the traceability of the work.

Now, I just route the mission locally. Because I built DEMA to be model-agnostic, if a cloud API goes down, my 7 Personal Agents (PATs) simply fall back to local inference — running Ollama with Qwen or Gemma directly on my MSI Titan. The model might be smaller, but the work doesn't stop. The FATE gates still enforce the exact same mathematical laws, and the receipt is still cryptographically sealed to my local chain. That's the entire point of the BIZRA architecture: you cannot claim sovereignty if your capability to work is dictated by a vendor's server status.

**Flag (founder's own M1 note):** this answer drifts into BIZRA architecture language rather than plain present-day lived behavior.

## Phase 3 — Trust & proof

### Q8. Trust-decision ritual

I don't trust it by reading it. That's the biggest trap in the industry right now. LLMs are literally optimized to produce plausible-sounding text, which means they are perfectly engineered to bypass human skepticism. If you rely on "vibes" or a quick manual read-through to trust an output, you will eventually deploy a shadow state.

I only trust an output when it survives the FATE gate. In my workflow, whatever the AI generates isn't treated as an answer; it's treated as a claim. That claim gets fed directly into a Z3 SMT solver and checked against the mathematical laws of the system — my five invariants. If it violates ZANN_ZERO or the IHSAN_FLOOR quality score drops below 0.95, it gets rejected at the kernel level. I don't even see the output.

I only deliver or act on the work when the system spits out a BLAKE3-hashed, Ed25519-signed receipt. I don't want to feel confident about what the AI did; I want the execution environment to prove it. Trust has to be a compiled artifact, not an emotion.

**Flag (founder's own M1 note):** this answer describes intended BIZRA architecture, not plain current behavior of all interviewees.

### Q9. Proof probe

Today, I still couldn't prove that cleanly in the strongest independent sense.

Before building toward BIZRA, the best I could do was screenshots, chat links, prompts, timestamps, diffs, and my own explanation of the workflow. And I don't consider screenshots real proof at all — they're easy to fake, and a vendor chat link only proves that a conversation happened on that vendor's server, not that a particular sentence or code block in a final artifact is cryptographically bound to that run.

So if someone asked me to prove that the AI, not me, produced a particular sentence, what I could do today is reconstruct a strong case. I could show the surrounding prompt history, intermediate outputs, timestamps, diffs, and session context. But that still depends on me as the narrator. It's evidence, not portable proof.

The reason I care about this so much is that I'm actively building toward a better answer: a signed, chain-linked record that binds the task, model, gates, and output in a verifiable way. But I want to be precise here — that is the direction I'm pushing toward, not something I can claim is already my default proof mode in every real interaction today.

### Q10. Local / cloud ratio

All of it happens locally on my hardware.

**Flag (founder's own M3 note):** this answer is too blunt. The broader workflow clearly includes cloud tools. A neutral interviewer would have probed. The cleaner reality is more mixed and was compressed into a sovereignty narrative.

## Phase 4 — Artifact reaction

### Q11. Mock signed-record reaction (GUARDRAIL 3 — artifact framed as "signed record")

**Wording shown:** "signed record"

**Wording understood:** "a provenance artifact for one AI task, not the task output itself"

**Wording preferred:** "signed record or verifiable record" — *"I would not lead with 'cryptographic receipt' unless I was talking to a highly technical audience, because that describes the mechanism, but not the function. 'Signed record' is plainer and closer to what a normal person actually needs to understand."*

**First field looked at:** not explicitly stated; overall structure read first rather than single field

**First question asked:** "what is the verification path? Is there a command, tool, or public verifier that lets me check the signature and chain link?"

**Readable?** Yes

**Imagined-scenario answer:** "I would treat it as a serious trust artifact, but not as something I'd just take on faith. My next step would be to ask: what is the verification path? Then I'd compare the record against the attached work and see whether the record actually binds to the thing they're claiming it binds to."

**Would forward to whom:** not addressed directly

**Full answer verbatim:**
What it tells me is that this is a provenance artifact for one AI task, not the task output itself.

It tells me the task had a unique receipt ID and mission ID, that it moved through a defined lifecycle from Submitted to Complete, that it ran on a specific model — in this case a local Ollama-served Qwen model — and that it completed very quickly. It also tells me the run crossed some internal trust thresholds: the guardian approved it, the ihsan and snr scores were both above the apparent floor, and it was linked into a chain through the previous receipt hash and the chain head at mint.

The signature tells me this record is meant to be tamper-evident, not just informative. So the intention here is clearly: don't just read this, verify it.

What it does not tell me yet is the actual prompt, the actual output, or the semantic content of the work. So by itself, it tells me that a governed AI task happened and was recorded, but not whether the work attached to it is good, useful, or appropriate. It's provenance, not judgment.

If a collaborator sent this to me attached to a piece of work, I would treat it as a serious trust artifact, but not as something I'd just take on faith. My next step would be to ask: what is the verification path? Is there a command, tool, or public verifier that lets me check the signature and chain link? Then I'd compare the record against the attached work and see whether the record actually binds to the thing they're claiming it binds to.

If I had to describe it to someone else, I would probably call it a signed record or a verifiable record. I would not lead with "cryptographic receipt" unless I was talking to a highly technical audience, because that describes the mechanism, but not the function. "Signed record" is plainer and closer to what a normal person actually needs to understand.

## Phase 5 — WTP & close

### Q12. Value probe + workflow-change (H3 dividing-line)

- **Value statement:** "a reduction in cognitive risk at the exact point where AI is most dangerous for me: when the output looks clean enough to tempt acceptance, but still contains hidden inconsistency"
- **Dollar figure:** not given
- **Units:** not given
- **Threshold condition:** "higher confidence, lower review fatigue, smaller chance of quietly contaminating the next stage of work"

- **Workflow-change statement:**

**Would STOP:**
- re-reading every touched file multiple times just to hunt for propagation drift
- manually tracing renamed references across docs, templates, and thresholds
- writing one-off bash checks just to reassure myself that only the intended files changed
- carrying so much of the consistency model in working memory

**Would START:**
- pushing the AI further on bounded multi-file editing tasks because there'd be a stronger verification layer underneath it
- reviewing at the level of logic and decision quality instead of line-by-line distrust
- moving faster from correction to execution once the verification artifact says the patch is internally coherent
- treating the AI as a more reliable structured editor, while still keeping final judgment for myself

- **Workflow-change captured:** `concrete_workflow_change` ✅

**Full answer verbatim:**
What it would be worth to me is not just time saved. It would be worth a reduction in cognitive risk at the exact point where AI is most dangerous for me: when the output looks clean enough to tempt acceptance, but still contains hidden inconsistency.

In that guardrail-patching session, the painful part wasn't writing the edits. It was having to hold the cross-file logic in my head while also distrustfully inspecting whether the assistant had propagated the change everywhere it mattered. So if a tool made that easier in a real way, the value would be: higher confidence, lower review fatigue, and a smaller chance of quietly contaminating the next stage of work with a mismatch I didn't catch.

More concretely, it would be worth preserving my attention for the parts that actually require judgment instead of spending it on repetitive cross-reference policing. It would also be worth making the boundary between "first pass from AI" and "verified enough to proceed" much sharper than it is now.

So the real value is not "the AI does more for me." It's that I spend less of my own mind on defensive reconstruction and more on the parts of the work that only I can do.

---

## Closing answers

- C1 (anything not asked): — (not collected in session)
- C2 (referrals): N/A — founder
- C3 (15-min follow-up OK?): N/A — founder
- C4 (anonymized quotes OK?): N/A — founder artifact is internal only

---

## Section 12 — Meta-reflection (THE IMPORTANT OUTPUT)

### 12.1. Wording that felt natural

The most natural questions were Q1, Q2, Q3, Q4, Q6, Q9, and Q12. Those all stayed close to lived workflow. I was describing recent work, how I actually use the tool, what I keep, where the AI failed, whether I could reproduce the result, what proof I can and cannot provide today, and what would actually change in my workflow if a better tool existed. Those answers felt grounded because they came from memory, not from theory.

### 12.2. Wording that felt forced

The most forced questions were Q5, Q7, Q8, Q10, and partly Q11. Q5 felt forced because I do not have a clean A3-style incident, so I could feel the pressure to produce a stronger story than I really had. Q7, Q8, and Q10 felt forced because I drifted out of current lived behavior and into BIZRA architecture language — local routing, PATs, FATE gates, signed receipts, "all local" — which is closer to my intended system than to an ordinary plain description of what I actually do today. Q11 was mixed: reading the artifact itself felt natural, but I could also feel myself evaluating it partly as a builder of the system rather than purely as an outside recipient.

Pattern: questions felt natural when they asked for a recent concrete workflow I have actually lived. They felt forced when they pulled me toward future-state architecture, segment-fit pressure, or terminology more native to my system than to a normal user's experience.

### 12.3. Probes that did not land

Main ambiguity moments were Q5, Q7, Q8, Q10, and partly Q11.

- **Q5** — ambiguity between "have I ever had to explain this?" and "do I have a clean A3-style proof-demand incident?" — those are not the same question.
- **Q7** — ambiguity: are you asking about lived fallback behavior today, or designed fallback logic?
- **Q8** — ambiguity: actual present-day decision process, or ideal trust architecture I want to exist?
- **Q10** — ambiguity: literal usage split across all AI tools, or strategic preference around local vs cloud? Compressed mixed reality into a cleaner ideological answer.
- **Q11** — ambiguity about stance: react as myself (close to architecture) or as an outside recipient receiving cold?

General pattern: questions felt ambiguous whenever they sat on the boundary between current lived behavior and future-state architecture, or between strict external evidence and softer underlying pattern.

### 12.4. Interviewer biases observed (minimum 2 required — 5 captured)

1. **Segment-fit bias toward A3-style pain.** Clearest example: Q5 and the specificity probe. Interviewer narrowed hard toward a dated, named, client-or-collaborator incident where someone explicitly asked to show what the AI did. That privileged the exact wedge hypothesis being tested and risked downgrading softer but still real trust friction.

2. **Architecture-confirmation bias.** In Q7, Q8, and Q10, when the founder drifted into BIZRA/FATE/PAT/local-sovereignty language, the interviewer mostly let it stand instead of pulling back to plain present-day behavior. Stricter when answers weakened the wedge; looser when answers reinforced the architecture. A neutral interviewer would have asked: "Is that what you do today, or what you are building toward?"

3. **Asymmetric skepticism.** Interviewer was appropriately skeptical on Q5 when the answer risked being too vague, but did not apply the same level of skepticism on Q10 ("all my AI use happens locally"), which should have triggered a follow-up. Probing intensity was not evenly distributed.

4. **Vocabulary priming bias.** By Q11, the session had already primed terms like "signed record," "verifiable record," "cryptographic receipt." Even with the packet design requiring it, the wording-preference probe landed in a pre-shaped response space. The interview was not fully neutral with respect to naming.

5. **Fluency bias toward well-structured answers.** When answers were crisp and systematized — especially Q1–Q4 — the interview moved forward quickly. That risks rewarding articulate answers even when they may be more rehearsed or architecture-shaped than lived. A more adversarial interviewer would sometimes interrupt fluency and ask for one smaller, messier, more recent concrete example.

### 12.5. One interview-guide change to apply before Day 2

> **Add a mandatory "pause" interrupt probe for any answer that drifts from current lived behavior into future-state architecture.**
>
> Concrete line to add: *"Pause. Is that what you do today, or what you want the system to do eventually?"*
>
> If the answer is future-state, the interviewer should immediately redirect: *"Stay with today. What do you actually do now?"*
>
> Place near Q7, Q8, and Q10, where drift most happens.
>
> Reason: if that distinction is not enforced, the interview stops measuring real user workflow and starts measuring the founder's architecture aspirations. This would quietly distort signal even if the rest of the guide is strong.

---

## Day-1 founder-prep completion checklist

- [x] Template 1 meta fully filled with `interview_type=founder_prep`
- [x] All 12 questions asked with founder responses recorded
- [x] Q11 wording-preference captured (signed record / verifiable record)
- [x] Q12 workflow-change field explicitly filled (concrete workflow-change)
- [x] Section 12.1 – 12.5 completed with 5 biases in 12.4
- [x] Artifact archived at `docs/strategy/validation_runs/2026-04-23_day1_launch/founder_prep_interview_FILLED.md`
- [x] No rows added to `evidence_log_seed.csv` from this session (confirmed — zero writes to the evidence log)
- [x] Interview-guide change from 12.5 applied (see `BIZRA_Interview_Guide_v0_1.md` — contamination-control rule added 2026-04-23)

---

**End of founder-prep filled artifact. Nothing here is market evidence.**
