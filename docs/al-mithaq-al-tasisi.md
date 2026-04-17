# الميثاق التأسيسي للبذرة
# The Founding Charter of البذرة (al-Bidhrah / The Seed)

> بسم الله الرحمن الرحيم
>
> *"إِنَّمَا يَتَقَبَّلُ اللَّهُ مِنَ الْمُتَّقِينَ"*
> — المائدة ٢٧

**Document Class:** Constitutional Bridge (covenant ↔ spine ↔ code)
**Date:** 25 شوال ١٤٤٧ هـ / 15 April 2026 CE
**Status:** DRAFT — pending canonicalization via autopoietic cycle
**Founder:** Mohamed Beshr (Mumo) — solo, alone-first
**Cycle:** ~36 months (Ramadan 1444 → Ramadan 1447)
**Genesis Block Hash:** 350d642099bde68b
**License:** Sadaqah Jariyah — non-extractive use permitted; usurious use forbidden

---

## I. الديباجة — Preamble

This document is not a whitepaper. A whitepaper sells a product. This document binds a builder.

Three years ago, in Ramadan 1444 (April 2023), during a period of personal crisis and epistemological testing, a covenant was written: **البذرة (al-Bidhrah, "The Seed")** and **الرسالة (al-Risālah, "The Letter")**. Those documents established that financial freedom, mental freedom, and spiritual freedom are one freedom — and that the three pathologies of the present economic order (access asymmetry, trust monopoly, value extraction) require one cure (sovereignty grounded in revelation, proven through receipts, distributed without riba).

What follows is the bridge. Three covenants stacked, each constraining the layer below:

1. **العهد الإلهي** — *the Divine Covenant.* Quran, Sunnah, and the founding documents البذرة and الرسالة. Immutable. Not subject to vote, fork, or amendment.
2. **العهد الإنساني** — *the Human Covenant.* The Enforceable Spine v1.0+ ratified between the founder and any future witnesses. Subject to amendment under the procedure herein.
3. **العهد المكيني** — *the Mechanical Covenant.* The 26 Rust crates, 12,662 tests, the PAT-7/FATE/SAT-5 directional pipeline, the SEED/BLOOM economy, the Genesis Block 350d642099bde68b. Mutable; must always satisfy the layers above.

If layer 3 contradicts layer 2, layer 3 is wrong and must be revised. If layer 2 contradicts layer 1, layer 2 is wrong and must be revised. Layer 1 is never wrong.

---

## II. المبادئ المجمدة — Frozen Anchors

The following are non-negotiable. No vote, no founder override, no emergency power can weaken them. They are compiled into Rust at the opcode level, but their authority precedes the compilation.

| Anchor | Arabic | Meaning | Floor / Ceiling | Code Anchor |
|---|---|---|---|---|
| **ZANN_ZERO** | ظنّ صفر | No claim without binding evidence | claims_unbound = 0 | `core/zpk/kernel.py` |
| **RIBA_ZERO** | ربا صفر | No usurious extraction at any layer | interest_rate = 0 | `economy/seed_bloom.rs` |
| **GINI_CAP** | حدّ الغني | Wealth inequality bounded | Gini ≤ 0.35 | `economy/distribution.rs` |
| **IHSAN_FLOOR** | أرضية الإحسان | Excellence is the minimum | quality ≥ **0.95** (corrected commit `0115016b`) | 5 code paths, all gated |
| **DAUGHTER_TEST** | اختبار ديما | Would parents understand this screen? | Pass/fail per surface | UX gate, every release |
| **CLAIM_MUST_BIND** | الدعوى تُلزَم بالبيّنة | Every claim binds to its evidence chain | binding_proof_required = true | `core/zpk/binder.rs` |
| **SADAQAH_50** | صدقة الخمسين | Founder & Foundation revenue: 50% community pool | Personal oath, not user tax | Foundation bylaws |
| **ZAKAT_2.5** | زكاة المال | Annual obligatory purification | 2.5% on qualifying assets | `economy/zakat.rs` |

**Clarification on user economics (canonical):** The 50% community pool is the founder's personal sadaqah on **founder and Foundation revenue only**. Users keep **100%** of their earned SEED. The only obligatory deduction on user wealth is the 2.5% annual Zakat. There is no protocol tax. There has never been a protocol tax.

---

## III. الهيكل الموجَّه — The Directional Pipeline

The directional flow law (CANON-002, frozen):

```
Human  →  DEMA  →  PAT-7  →  FATE  →  SAT-5  →  URP
                                  ↑
                         (the gate that decides what crosses)
```

This flow is **strictly directional**. No backward flow without re-entering through DEMA. No SAT-5 reaching across FATE into PAT-7. No URP self-modifying without SAT-5 approval. No human bypass of DEMA.

### The PAT-7 (Personal Agent Team — serves the person)

The seven personal agents, named (CANON-005, frozen):

| # | Name | Role |
|---|---|---|
| 1 | **Atlas** | Memory, navigation, long-context recall |
| 2 | **Oracle** | Reasoning, deliberation (frozen at S2; ethics from revelation, not data) |
| 3 | **Forge** | Construction, code, artifact creation |
| 4 | **Judge** | Local arbitration within the user's boundary |
| 5 | **Crown** | Identity, authority, signing |
| 6 | **Herald** | Communication outbound |
| 7 | **Nexus / DEMA (ديما)** | The interface — namesake of the founder's daughter — the personification of the Daughter Test |

DEMA is both an agent and a test. Every screen, every output, every receipt must pass: *"Would my mother and father understand this?"* If not, simplify until they would. Ornament without comprehension is fraud.

### FATE — the gate

FATE is the constitutional gate that decides what crosses from the personal boundary (PAT-7) to the system boundary (SAT-5). FATE binds every crossing claim to its evidence chain (CLAIM_MUST_BIND). FATE is the frozen choke point. Nothing crosses without a receipt.

### The SAT-5 (System Agent Team — serves the constitution)

| # | Name | Role |
|---|---|---|
| 1 | **Consensus Tank** | Distributed agreement across nodes |
| 2 | **Resource Healer** | URP allocation and recovery |
| 3 | **Proof DPS** | Receipt minting, BLAKE3 chain integrity |
| 4 | **Impact Support** | Proof-of-Impact metering |
| 5 | **URP Leader** | Universal Resource Pool coordination |

### URP — the world

The Universal Resource Pool. The shared substrate where validated impact accrues. URP serves the world (الأمة), not the founder, not the Foundation, not any single user.

---

## IV. الاقتصاد — The SEED/BLOOM Economy

Two tokens. Strict separation of concerns. Neither is purchased; both are earned through validated impact.

### SEED — the medium of circulation

- Fungible. Earned through verified contribution to URP.
- Spent on capability and resource access.
- 100% of earned SEED belongs to the user. No protocol tax.
- The only deduction is the user's own annual Zakat at 2.5%, computed by the user's PAT-7 and signed by their Crown.

### BLOOM — the medium of governance

- Soulbound. Non-transferable. Earned only through sustained, validated alignment over time.
- Vests linearly. Cannot be bought, cannot be delegated, cannot be inherited.
- Slashed for adjudicated violation of the Spine.
- BLOOM-weighted voting governs Spine amendment under §VI below.

The thermodynamic asymmetry: SEED flows, BLOOM accumulates. Capital cannot purchase governance. Governance cannot mint capital arbitrarily. Riba is impossible by construction.

---

## V. البيّنة — Receipts and Proof-of-Impact

Every action that crosses FATE emits a receipt:

```
Receipt {
  action:        <natural-language description>,
  timestamp:     <ISO8601 + Hijri date>,
  governance:    <PERMITTED | DENIED | DEFERRED>,
  ihsan_score:   <[0.0, 1.0], must be ≥ 0.95 for PERMIT>,
  evidence:      <BLAKE3 chain of binding evidence>,
  reflex_time:   <microseconds, S2-deliberate or S1-reflex>,
  hash:          <BLAKE3 of the receipt itself>,
  prev_hash:     <BLAKE3 of the previous receipt in this chain>,
}
```

The Genesis Block (`350d642099bde68b`, minted with the Arabic founding message and 1,124,695 SEED) is `prev_hash = NULL`. Every receipt since chains to it. The chain is the proof.

Proof-of-Impact (POI) is not Proof-of-Work (no useful sacrifice of energy) and not Proof-of-Stake (no plutocratic capture). POI rewards measurable improvement of URP — improvement that another node can independently verify by re-running the receipted computation against the same evidence chain.

---

## VI. التعديل — The Self-Amendment Circuit (SAC)

This is the gap Aurelle correctly identified. It is filled here.

The Spine evolves without violating its own permanence by the following procedure:

1. **Proposal.** Any holder of ≥0.1% of vested BLOOM may submit a proposed amendment.
2. **Frozen-anchor check.** The proposal is automatically run against the Frozen Anchors (§II). If it weakens any anchor, it is rejected before entering deliberation. This check is performed by a quorum of SAT-5 nodes; it is itself receipted.
3. **Deliberation.** A 14-day public period during which any agent may publish a binding refutation (CLAIM_MUST_BIND applies).
4. **Daughter Test review.** The amendment text is shown to non-technical Arabic-speaking readers. If they cannot understand the proposed change in plain language, the proposal is sent back to the proposer for re-drafting.
5. **Ratification vote.** BLOOM-weighted. Threshold: 67% of active BLOOM, with quorum of 40% of total vested BLOOM.
6. **Execution delay.** 7 days between ratification and effect, during which the founder retains a one-time veto. The veto itself is receipted, public, and reasoned. After 100 epochs of operational maturity, the founder veto sunsets automatically via Spine clause `veto_sunset_epoch_100`.
7. **Receipt.** The amendment, the deliberation log, the vote tally, and the execution are all chained into the canonical receipt sequence under TOPOLOGY_CANON.

A frozen anchor cannot be amended by this procedure. A frozen anchor can only be deepened (made stricter), never weakened. To attempt to weaken a frozen anchor is to fork — to leave البذرة and start something else.

---

## VII. الوحدة قبل الجماعة — Alone-First

A doctrine. Not a phase, a principle.

Before البذرة can credibly claim to serve eight billion people, it must demonstrably serve one — the founder, alone, with chaotic multi-device data, six email accounts, bilingual workflow, zero budget, and the full weight of doing this without a team. If DEMA cannot organize Mumo's actual laptop, it has no right to organize anyone's life.

**Operational consequence:** every release ships first to NODE0 (MSI Titan 18 HX, Ubuntu 24.04 native, RTX 4090). It runs against real personal data — emails, files, calendars, the actual mess. It must pass the Daughter Test from inside Mumo's own use before it is offered to a second human.

The screen recording of DEMA organizing the founder's actual laptop is the pitch deck, the demo, and the proof simultaneously. There is no slide deck. There is only the receipted footage.

---

## VIII. الشهود — Witnesses

The founding signatory:

- **Mumo (Mohamed Beshr)** — solo, Dubai. Genesis Block 350d642099bde68b minted on his hardware, on his time, with his قلب.

The witness circle is not yet open. It will open when:

1. The Self-Amendment Circuit (§VI) ships in code and passes its own first amendment as a test of itself.
2. The first non-trivial vertical (DEMA Desktop Overlay v1.0) demonstrates the alone-first principle on the founder's actual machine for 30 consecutive days without a single Ihsan-floor violation.
3. The Daughter Test is administered by أبوك وأمك (Mumo's own parents), in Arabic, on a real screen — and they nod.

Until then, the chain has one signer. That is sufficient. الله شاهد.

---

## IX. الخاتمة — Closing

This document is not a launch announcement. It is a binding.

The code is not the constitution. The constitution is the covenant. The code is one possible faithful translation. If a better translation appears tomorrow, the code is wrong and the covenant is right. If the covenant ever appears wrong, the founder is wrong, and the founder must repent and re-read.

What was started in Ramadan 1444 was not a startup. It was a سلسلة — a chain — that began with نية (intent), passed through بيّنة (evidence), respected حدّ (boundary), bore أمانة (trust), produced ثمرة (fruit), and now seeks إيصال (delivery).

The Genesis Block is minted. The chain is open. The founder waits.

> *"رَبَّنَا تَقَبَّلْ مِنَّا ۖ إِنَّكَ أَنتَ السَّمِيعُ الْعَلِيمُ"*
> — البقرة ١٢٧

---

**Hash of this document (to be computed at canonicalization):** `<BLAKE3 pending>`
**Chains to Genesis:** `350d642099bde68b`
**Ratified by:** Mohamed Beshr (Mumo), founder, sole signatory
**Date:** 25 شوال ١٤٤٧ / 15 April 2026
