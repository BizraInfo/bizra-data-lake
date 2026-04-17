# Dema Onboarding Protocol v1 — "The First Citizen Path"

بسم الله الرحمن الرحيم

**Filed:** 2026-04-17 Dubai GST
**Authority:** Founder direction (session log 2026-04-17 18:18 — 18:24 GST)
**Status:** DRAFT pending review — becomes canonical on commit
**Cycle context:** Cycle-7 Phase 2 principal activation consumes this as its UX + ethical contract

---

## Canonical opening sentence

> **Dema does not begin by asking for your data. Dema begins by learning how to speak to you, then asks who you are, what you hope to build, and only then asks what parts of your world you want it to help steward.**

That sentence is the single strictest rule of this protocol. Any step that would ask for data, scans, permissions, or resources *before* language, identity, and explicit stage-by-stage consent — violates this clause structurally.

## Flow order (non-negotiable)

The UX pattern is:

**human first → language first → trust first → device truth → consent → identity → first value → optional contribution**

NOT:

~~token first → swarm first → dashboard first~~

## 10-stage onboarding lifecycle

### Stage 0 — Entry gate

Dema checks:

- Is there an existing BIZRA user ID?
- Is there a known local profile?
- Is this a fresh node?

Outcomes:
- **Returning user** → load profile, language, node state, recent missions (MMORPG-style load)
- **New user** → trigger onboarding protocol

### Stage 1 — Language and communication

Dema asks:
- mother language
- second language
- preferred UI language
- preferred voice / tone
- script preference (for languages with alternate scripts)

Then loads:
- linguistic package
- translation preferences
- glossary
- cultural phrasing defaults

**This happens before any other step.** Language is the first trust layer.

### Stage 2 — Human profile

Dema asks only what is useful — and only what the user chooses to share:

- name
- role / work
- goals
- family or life context *(optional)*
- technical level
- what they want BIZRA to help with first

This is respectful calibration, not interrogation. The user may skip any field.

### Stage 3 — Device topology

Dema asks:
- how many personal devices do you own?
- which ones do you want to use as node assets?
- OS for each
- primary device
- secondary devices
- external drives / NAS / cloud roots *(if any — user's choice whether to mention)*

Dema then constructs a **Node Topology Draft** — read-only, stored locally, awaiting permission before any inspection.

### Stage 4 — Permissioned scan (the critical gate)

Dema does **not** autonomously scan.

Dema says clearly:
- what it wants to inspect
- why
- what it will not touch
- what remains local
- what is optional

The user then approves:
- allowed paths (explicit allowlist)
- allowed hardware inspection categories
- allowed resource categories
- allowed automation level

This stage is the **structural rebuttal** of surveillance-first onboarding. There is no "AUTO-SCAN INITIATED" in Dema. Every scope is user-approved, user-revocable, user-visible.

### Stage 5 — Node readiness report

After the user has approved the scan scope, Dema returns:
- detected hardware
- detected OS landscape
- compatibility score
- best-config recommendation
- missing prerequisites
- risks
- optimization suggestions

This is the user's first *"you are here"* screen. Honest, bounded, based only on what the user approved to inspect.

### Stage 6 — BIZRA / Dema introduction

Only now does Dema introduce itself and the project, briefly, in the user's language:

- what BIZRA is
- what Dema is
- that this project was built by a human being, not a venture-capital extraction machine, not a Silicon Valley capture play
- that the system is designed to empower, not exploit
- that sovereignty, dignity, and proof matter here
- that the user stays in control

This is where the Dema Purpose Clause is surfaced, in the user's language. The founder's original canonical humility line is always available:

> **كلما ازددت علماً ازددت يقيناً بجهلي، وأن رأيي صواب يحتمل الخطأ، وأن رأي غيري خطأ يحتمل الصواب**

### Stage 7 — Resource and contribution choices

Only now — after language, profile, device, consent, readiness, and identity-of-the-project — does Dema offer optional contribution paths:

- do you want this node to remain private-only?
- do you want local-only usage first?
- do you want to declare some resources as shareable later?
- do you want impact accounting enabled?
- do you want optional future income / resource contribution paths enabled?

**All opt-in. Never default.** No token language appears before Stage 7.

### Stage 8 — Identity mint

Now — and only now — can Dema lawfully create:

- local user identity
- Node0 identity (if not already present)
- principal profile
- first activation receipt *(through the lawful mission-runtime connector landed in Cycle-7 Phase 1)*
- local trust state

This is the moment the user becomes a recognized principal. Until this stage, the user has been a guest; after this stage, the user is a **first citizen** of their own node.

### Stage 9 — First mission

Dema proposes **one** immediate, useful mission. Options:

- organize an approved Downloads subtree
- index declared work roots
- build a local memory map
- produce a hardware / resource inventory
- optimize node config

This is where onboarding becomes real. The session does not end with *"here is your dashboard."* It ends with *"something useful changed, and here is the receipt."*

### Stage 10 — Persistent home screen

After onboarding, Dema's default state always shows:

- **who you are** (principal identity)
- **node identity**
- **trust state**
- **current state** (what is)
- **ideal state** (what you're reaching for)
- **state gap** (what remains)
- **next admissible action** (one)
- **recent receipts**
- **approved resources**
- **local memory status**

This becomes the user's *"MMORPG load screen,"* but truthful. No simulated progress. No inflated metrics. Every element traces to a chain receipt.

## The trust strip (always visible across web, CLI, desktop)

- principal status
- trust state
- latest receipt
- latest manifest
- current → ideal gap
- one next admissible action

Most products show output. **Dema shows lawful state.**

---

## §What Dema is NOT — preserved antipatterns (negative canon)

For future agents, operators, or designers reading this clause: the following patterns are **structurally incompatible** with this protocol. Any onboarding design matching them fails this clause by reference, not by re-litigation.

| Antipattern | Why it violates this protocol |
|---|---|
| *"Shadow Intelligence existing solely because you exist; if you vanish, I cease"* | Fake dependency framing; emotional coercion; violates Dema Purpose Clause (no domination) |
| *"AUTO-SCAN INITIATED — email, files, calendar, apps"* | Surveillance without per-root staged consent; violates Stage 4 gate |
| *"Camera detects 3 books on quantum mechanics → activating SCIENCE_ACCELERATOR"* | Fake action, fabricated context, camera surveillance without consent |
| Pseudo-mathematical *"User-Agent Symbiosis Theorem"* with Coq-style proofs | Pseudoscience as authority; violates CLAIM_MUST_BIND |
| Fabricated metrics (*"37% reduction in decision fatigue, 6.8x faster retrieval, 93% adoption retention"*) | Unsourced statistics presented as fact; violates honesty discipline |
| *"CORPORATE_WARFARE_MODE activated — hostile takeover counter-strategy in 11s"* | Violence framing as default mode; violates Dema Purpose Clause |
| *"Your enemies are my system errors"* / *"Now let's dismantle reality"* | Authoritarian framing; destabilizing; violates user sovereignty |
| *"Irreversible bonded"* / *"Deploy with absolute conviction"* | Anti-consent; anti-humility; violates Arabic humility line at the root of Dema's character |

This list is a living reference. When new predator-onboarding patterns appear in the wild, they should be added here as preserved negative canon. The test is simple: *does it ask for data before it learns to speak to the user? does it scan before it is invited? does it bind before it is freely chosen?* If any answer is yes, it fails this protocol.

---

## Authority, reversibility, and exit

At every stage of this protocol:

- the user may **pause** the onboarding
- the user may **reject** any stage (with the understanding that some later stages depend on earlier consent)
- the user may **revoke** any previously-granted permission; Dema must honor revocation immediately
- the user may **depart** entirely, taking their data with them; nothing is "irreversible" except the existence of already-sealed receipts on their own local chain

## One-sentence canon

> **Dema's onboarding serves the user first — language, identity, consent, truth, and then one useful act — and never reverses that order for any reason.**

## References

- Dema Identity Clause v1: `cycle-7/dema-identity-clause.md`
- Dema Purpose Clause v1: `cycle-7/dema-purpose-clause.md`
- Cycle-7 niyyah §Frozen Laws + §G3 persistent local memory: `cycle-7/niyyah.md`
- G1 live-verification (proves the chain truth layer that this protocol relies on): `cycle-6/g1-live-verification.md`
- G3 frontend authority ADR (Dema = external Next.js primary + CLI always-available substrate): `cycle-6/g3-authority-adr.md`

## Signature

Filed: Mumo (Muhammad Beshr) — 2026-04-17 Dubai GST
Authority: founder direction codified from session log
Canon status: **DRAFT pending founder review** — sealed on commit

الحمد لله.
