# BIZRA External Validation — Message Test Cards v0.1

**Date:** 2026-04-23 (GST) · patched 2026-04-23 for GUARDRAIL 3 (wording order) and workflow-change scoring rule
**Purpose:** Short frames that can be tested via landing-page copy, X / HN / LinkedIn posts, direct outreach, or interview-phase probe. Each card is standalone and runs as a discrete hypothesis test.
**Audit anchor:** tests the top 4 wedge missions (M01, M03, M06, M11) plus local-first framing (M02/M05).

---

## Rules of engagement

- Each card tests **one frame only**. Do not combine.
- **GUARDRAIL 3 — wording order:** cards leading with "signed record" / "verifiable record" / "proof of what the AI did" run FIRST (CARD_1 → CARD_3). Cards leading with "cryptographic receipt" language run LATER (CARD_4 → CARD_5) and are treated as technical-vocabulary variants for comparison. Naming itself is under validation (see hypothesis H13).
- Each card names the **target audience** before anyone drafts a landing page.
- **Workflow-change dividing line:** positive reaction = a specific measurable behavior OR a concrete workflow-change statement. Generic enthusiasm without a named workflow change counts as **WEAK** and is ignored in the proceed decision.
- Weak/no-fit reaction = silence, bounce, or vague enthusiasm without action OR without workflow-change implication.
- No card is allowed to overclaim. If the product does not yet do the thing promised, the card must be honest about that (e.g., "early access" not "shipped").
- BIZRA is mentioned, but never pitched in the headline. The frame is about the user's pain and proof moment.

### Frame order (tests these wording families in sequence)

1. "Signed record of AI work for clients/collaborators" — CARD_1 (leads)
2. "Local-first mission runtime with signed steps" — CARD_2
3. "Replay a prior AI decision later" — CARD_3
4. "Prove my AI actually did this (receipt framing)" — CARD_4 (crypto vocabulary begins here)
5. "Verify AI output without trusting the vendor (cryptographic receipt framing)" — CARD_5

Cards 1–3 exist to check whether non-crypto wording carries the pain. Cards 4–5 exist to check whether crypto-native vocabulary is *additive* (adds signal) or *subtractive* (costs signal) for A3 specifically.

---

## CARD_1 — "Signed record of AI work for clients & collaborators"

**Tests:** H1, H3, H7, H8, H13 · Missions M11, M23, M06

**Audience:** A3 explicitly (primary segment). Channels: consultant communities, freelancer newsletters, indie-consulting Substacks, targeted ads to "consultant" LinkedIn cohort.

**Headline (one sentence):**
> *"A signed record of every AI task — one per engagement, shareable with your client."*

**One-paragraph explanation:**
> If you use AI in client-facing work, you probably have a pile of screenshots, half-forgotten chat URLs, and an implicit agreement that you'll "do the right thing." BIZRA gives you one thing you can send to your client along with the deliverable: a compact signed record per task. They can verify it without installing anything from you or buying anything from us. Over the course of a project you can hand them a single chain of signed records that they can audit, replay, or forward to their own compliance team. Local tool. Free during early access.

**What the reader is expected to understand:**
- The signed record is a delivery artifact, not an admin burden
- Clients can act on it without learning BIZRA
- Early-access is free; no implied future price pressure
- This raises the floor of AI-assisted consulting practice

**Positive reaction (must include workflow-change):**
- Consultant replies with a concrete engagement type and a specific workflow change they would make ("I do audit prep for accounting firms — I would replace the screenshot-collage step with this")
- Questions about multi-record bundling per project
- Interest in sharing the record format with their own client (forwarding intent, with a named recipient role)
- Offers to be an early tester + refers a peer

**Weak/no-fit reaction:**
- "My clients don't ask about AI" — pain does not exist
- Enthusiasm about internal use with no client-facing angle OR no workflow-change statement
- Questions about how to "hide" AI use from clients — anti-fit, wrong ethics
- "Interesting" / "useful" / "I'd try it" without specifics = WEAK; do not count as signal

---

## CARD_2 — "Local-first mission runtime with signed steps"

**Tests:** H5, H9 · Missions M02, M05, M08

**Audience:** A1 + A8. Channels: r/LocalLLaMA, r/selfhosted, HN (Show HN or side-post), Rust-community Discord, indie-dev newsletters.

**Headline:**
> *"Run governed AI missions on your own hardware. No cloud required. A signed record for every step."*

**One-paragraph explanation:**
> You have Ollama. You have your own hardware. What's missing is a way to treat each AI task as a *mission* — something submitted, gated, executed, and recorded — without shipping your inputs or your outputs to a cloud service. BIZRA's Dema CLI sits on top of your local inference and adds a governance spine: admissibility checks, signed records, replayable chain. Single binary, open-source, MIT. No account, no quota, no telemetry.

**What the reader is expected to understand:**
- BIZRA does not replace their local model — it orchestrates missions over it
- This is additive to Ollama/LM Studio/etc., not competitive with them
- Zero cloud dependency
- Open-source, free local

**Positive reaction (must include workflow-change):**
- Clone, build, and run the CLI the same day AND describe what step it replaces in their current stack
- GitHub issues with specific use cases or feature requests that match the wedge
- "Finally, someone doing governance for local" paired with a concrete plan
- Replies from people already running their own infra setups who can name the piece BIZRA fits into

**Weak/no-fit reaction:**
- "Just use LangChain" — reader missed the signed-record axis
- "Why not a SaaS version?" — wrong audience
- "Does it include its own model?" — confused scope; might be misrecruited
- Installs, opens README, bounces
- Stars the repo, never engages = WEAK signal, not validation

---

## CARD_3 — "Replay a prior AI decision later"

**Tests:** H1, H3, H10 · Missions M03, M06, M27

**Audience:** A1 (coding-adjacent) + A3 subset doing research/technical work. Channels: r/MachineLearning, dev newsletters, targeted outreach to ML research solopreneurs.

**Headline:**
> *"The AI gave you an answer last month. Can you run it again and get the same one?"*

**One-paragraph explanation:**
> LLM outputs are non-deterministic. But your liability for them isn't. When a client or collaborator challenges something the AI gave you last month, there's usually no way to reconstruct the same outcome — different model version, different seed, different context. BIZRA turns every task into a canonical mission with a signed record, so you can replay it: same inputs, same gates, bound to the original record. The record tells you what model answered, what gates passed, what the output was, and whether a fresh run matches. Local, open-source, early access.

**What the reader is expected to understand:**
- Non-determinism is a real liability
- Replay is specific (not "ask the AI again" — bind to the original signed record)
- This is valuable when someone challenges a past AI output
- Works locally

**Positive reaction (must include workflow-change):**
- "I had this exact problem last year" + named specifics + statement about what workflow change would follow
- Technical follow-up on how replay handles temperature, seeding, model versioning
- References to audit-defense or legal-hold scenarios with named context
- Request to see an example replay output

**Weak/no-fit reaction:**
- "Just set temperature to 0" — wrong axis
- Confusion between replay and re-prompt
- "Why not just save the chat transcript?" — pain not felt
- Enthusiasm about the concept without a named use case = WEAK

---

## CARD_4 — "Prove my AI actually did this" (receipt framing — CRYPTO VOCABULARY TEST)

**Tests:** H1, H3, H4, H8, H13 · Missions M01, M03, M11

**Audience:** A3. Same channels as CARD_1 but delayed — run only AFTER CARD_1 has been live ≥ 3 days to allow clean wording-family comparison.

**Headline (one sentence):**
> *"The next time your client asks 'did AI write this?' — hand them a cryptographic receipt."*

**One-paragraph explanation:**
> When you deliver AI-assisted work, your client has no way to verify what the AI actually did. Screenshots can be faked; transcripts can be edited; your word is just your word. BIZRA is a local tool that signs and chains every AI action you take, so you can hand a client a portable cryptographic receipt that they can verify on their own machine without trusting you or your vendor. You keep control of your data. They get proof. Early access now.

**What the reader is expected to understand:**
- This is about delivery-time trust, not model quality
- The receipt is portable (they can share it)
- Verification does not require trusting BIZRA or the AI vendor
- This is a local tool, not a SaaS

**Positive reaction (must include workflow-change):**
- Click-through to scheduled-demo / email-signup form
- Replies that name a specific client incident ("last month my client...") AND a concrete workflow change
- Asks for a sample receipt to inspect
- Forwards to another consultant with a named role

**Weak/no-fit reaction:**
- Vague "interesting" / "cool" without any action or workflow-change statement
- "Why would a client ask that?" — signals pain does not exist for them
- "Can it also do X/Y/Z?" — feature-expansion requests with no anchor in the stated pain
- "Does this work with Notion?" — scope-creep, not fit

**H13 cross-read:** Compare CARD_4 named-incident reply rate to CARD_1's. If CARD_4 is ≥ 2× CARD_1 on A3 → crypto vocabulary adds signal; if CARD_1 is ≥ 2× CARD_4 → crypto vocabulary costs signal and launch language must lead with "signed record." Equivalent → no effect.

**Anti-patterns to avoid in follow-up:**
- Do not turn one reply into a long pitch; stay in pain-discovery mode
- Do not offer the tool as a solution until H1 pain is corroborated in interview

---

## CARD_5 — "Verify AI output without trusting the vendor" (receipt framing, dev/compliance angle)

**Tests:** H1, H4, H9, H13 · Missions M06, M23, M27

**Audience:** A3 + A8. Channels: HN comments on AI-vendor outage posts; X replies to AI-vendor-quota threads; dev-focused Substacks.

**Headline:**
> *"Verifying what an AI did shouldn't require trusting the AI vendor."*

**One-paragraph explanation:**
> Every "audit trail" shipped by a hosted AI provider is audited by that same provider. If Anthropic or OpenAI is the one certifying what Anthropic or OpenAI did, the certification adds nothing. BIZRA is a local tool that produces cryptographic receipts you can verify on a cold machine — no BIZRA account, no cloud round-trip, no vendor consultation. You, or your client, or a regulator, runs `dema verify <receipt>` and gets a YES/NO locally. Early access now.

**What the reader is expected to understand:**
- Vendor-provided audit is circular
- Independent local verification is possible and desirable
- This is about separation-of-concerns between the AI vendor and the auditor
- Replayability is the guarantee

**Positive reaction (must include workflow-change):**
- Detailed questions about how the verifier handles non-determinism
- Reference to EU AI Act, SOC2, HIPAA, or other audit frameworks with a named project
- "We had to prove X last quarter and it was a nightmare" — named incident + described workflow-change
- Engineer/developer forwarding to compliance/security peer with a named recipient

**Weak/no-fit reaction:**
- "I trust my vendor" — pain does not exist
- "What's the difference from vendor logs?" asked rhetorically (not curiously)
- Confusion about what "cold machine" means — wrong audience

---

## Cross-card analysis rules

When more than one card runs in the same channel (e.g., two X posts a week apart):

- **Compare click-to-signup rates** (if using a landing page).
- **Compare specificity of replies.** A card producing 10 specific replies beats a card producing 100 generic ones.
- **Cross-read segments:** do replies on card N align with other cards? If one cohort only responds to cards 1 and 4, that cohort is A3. If only to cards 2 and 3, that is A1/A8. This itself is validation of segment-narrowing.
- **H13 wording cross-read (GUARDRAIL 3):** explicitly compare CARD_1 vs CARD_4 as matched-pair test. Same audience (A3), different wording family ("signed record" vs "cryptographic receipt"). Use same channel, same time-of-week, same message length where possible. Winner on named-incident-reply rate informs launch language.

## Thresholds per card (30-day window; see Sprint doc for rationale)

| Outcome | Signal | Threshold for "proceed with this frame" |
|---|---|---|
| Specific named-incident replies | Count across all channels | ≥ 3 |
| Replies that include a named workflow-change statement | Count | ≥ 2 |
| Asked for a sample record / demo | Count | ≥ 2 |
| Disconfirming engagement ("pain does not exist for me") | Count | ≤ total of 5 (above this, drop the frame) |
| Generic positive engagement without specifics | Count | Ignore — not signal |

**Workflow-change scoring rule (reinforced):** a card that produces high engagement but zero workflow-change statements is a SIGNAL of interest, not a SIGNAL of wedge fit. Treat such engagement as WEAK.

---

## What this deliberately does NOT test

- Enterprise security/governance framing (A5/A7 — out of scope for first validation)
- Ideological / Islamic finance framing (A6 — explicitly deferred per audit anti-patterns)
- Token-economy / SEED / BLOOM framing (frozen per canon)
- PAT-7 / SAT-5 agent-spawn framing (rows 8-9 PLANNED)
- Decentralized-network framing (single-node reality)

Any card attempting these frames is out-of-policy for this sprint.

---

**End of Message Test Cards v0.1 (GUARDRAIL 3 patch applied 2026-04-23).**
