# BIZRA First Wedge — Decision Memo v0.1

**Date:** 2026-04-23 (GST)
**Author:** Audit v0.1, read-only
**Companion:** `BIZRA_Omnidirectional_Market_Audit_v0_1.md`, `BIZRA_User_Archetype_Scoring_v0_1.csv`, `BIZRA_Mission_Inventory_v0_1.csv`

---

## The Decision

**The first wedge is: _"The Receipt-First Personal Mission Runtime for Solo Sovereign Operators."_**

A local Dema CLI where one operator (**A1 — Solo sovereign builder**, reinforced by **A6 — Future-node ideological adopter**) submits a mission, receives a cryptographically signed receipt, and can replay or independently verify that receipt later. One operator. One machine. Zero cloud requirement. Zero signup. Zero subscription. The receipt is the product; Dema is the face.

This decision is **singular** because the evidence supports exactly one first path. Where evidence is thin, I have named it thin — but I have not proliferated options to avoid a call.

---

## Why this wedge, in one paragraph

Node0 canon says serve one before serving eight billion. Row 4 (replay) hit PROVEN on 2026-04-23 as PR #49. PR #50 closed the weak-signature hole the same day. The substrate that lets an operator submit → receive receipt → replay → verify is shipped and runtime-bound today. Every other wedge in contention (enterprise audit trail, output-as-a-service monetization, personal DA with GUI) either waits on a customer segment we have no direct evidence of, or requires surfaces (GUI, multi-user, token economics) that are explicitly FUTURE per `bizra-now-vs-future-image-v1.md`. The only path where every required ingredient is ready — substrate, face, doctrine, distribution channel — is the one where a solo operator can run Dema on their own laptop today and share a receipt that verifies on someone else's cold machine tomorrow.

---

## Why not the alternatives — ruthless short version

### Alt 1 — Enterprise AI-governance wedge (A5 + A7) — _rejected_

- **Case for:** Highest willingness-to-pay in BIZRA's own corpus ($50–200K/yr). Regulatory tail wind (EU AI Act, 60+ data-localization laws) is the strongest validated signal in the audit.
- **Case against (decisive):**
  1. **Zero customer evidence.** `EARLY_CUSTOMERS_OUTREACH.md` names Gulf SWFs and banks; there is no retrospective, no pilot result, no LOI, no discovery-call transcript. Proceeding to "enterprise-first" without validation is building against hypothesis, not demand.
  2. **Sales cycle is 6–12 months** with RFPs, procurement, SOC2/ISO27001 prerequisites. BIZRA does not have the evidence bundle, the reference customers, or the org structure to close that cycle now.
  3. **Prerequisites not shipped:** v1.0.0, 24-hour heartbeat, multi-operator deployment, compliance export. These are on the roadmap; they are not in a user's hands.
- **When to revisit:** After A1 wedge yields 3+ unsolicited inbound inquiries from enterprise buyers, OR after one A5-archetype believer cold-adopts BIZRA and becomes a reference.

### Alt 2 — AI-native creator / consultant output-as-a-service (A3) — _deferred, not rejected_

- **Case for:** Best alignment with _agent-as-a-service_ / _output-as-a-service_ business models the audit was asked to test. A3 monetizes verified output directly.
- **Case against (sequencing):**
  1. The **receipt-sharing UX** and the **public verifier URL** do not exist yet. A3 needs: (a) a portable, human-readable receipt; (b) a third-party-runnable verifier; (c) a plausible billing story per verified output.
  2. A3 adoption is downstream of A1 adoption — an AI-native consultant needs BIZRA to already be credible before using it in client deliverables.
- **Promotion trigger:** once A1 wedge produces an operator-shareable receipt viewer + stateless verifier. Then A3 becomes the first revenue pilot without a net-new architecture.

### Alt 3 — Personal DA / local-chaos operator (A2) — _rejected for first wedge_

- **Case for:** Largest addressable population (every knowledge worker with chaotic files and AI-subscription fatigue).
- **Case against (decisive):**
  1. **Surface not available.** Dema CLI is too technical for A2. A browser / GUI face is explicitly FUTURE per `bizra-now-vs-future-image-v1.md:105-106`.
  2. **Receipt value prop does not resonate.** A2 wants "my AI to know me," not "I need to prove what my AI did." The audit's Phase-2 signal extraction shows proxy adoption (ProtonMail, Signal) validates sovereignty _in principle_ but does not validate that A2 users would install BIZRA CLI over Ollama-plus-a-pretty-UI.
  3. **Feature scope not ready.** A2 demands search over personal files, daily digest, contextual recall — adjacent to M09 and M19 but not in today's bullet.
- **When to revisit:** After Row 5 (daily manifest) ships + a Dema GUI / browser-operator mode is in active development.

---

## Anchoring the wedge in today's runtime state

The audit's companion doc reports Node0 closure scoreboard as of 2026-04-23:

| Row | State | Used by first-wedge | Notes |
|---|---|---|---|
| 1 — authoritative entry | WIRED_PARTIAL | ✅ (mission submit path) | Gateway works for POST /mission |
| 2 — gate chain | VALIDATED | ✅ (admissibility path) | Fail-closed enforced |
| 3 — receipt lineage | WIRED_PARTIAL | ✅ (chain head + receipt) | BLAKE3 chain live; PR #50 tightens Ed25519 binding |
| 4 — replay | **PROVEN** | ✅ (replay surface) | PR #49 |
| 5 — daily manifest | WIRED_PARTIAL | ⬜ (nice-to-have; M19) | Scheduler missing |
| 6 — trust surface | WIRED (branch-local) | ✅ (web face supplementary) | Push queue blocks adoption |
| 7 — Dema unified face | WIRED_PARTIAL | ✅ (CLI face sufficient) | Web DTO unification later |
| 8 — PAT-7 activation | PLANNED | ❌ (out of wedge scope) | Do not over-promise |
| 9 — SAT-5 activation | PLANNED | ❌ (out of wedge scope) | Do not over-promise |
| 10 — health + security | WIRED_PARTIAL | ⚠️ (gate to protect wedge) | Chain-aware `/health` needed for launch |

**Row mix used by first wedge:** 1 PROVEN + 1 VALIDATED + 3 WIRED (1 of those branch-local) — sufficient to ship a minimum external-user experience.

**Row mix left untouched by first wedge:** rows 8 and 9 (agent-team activation) — do not promise, do not market.

---

## The required proof moment (canonical text for marketing / demo)

> The operator opens a terminal, types `dema activate-principal --name mumo`, and BIZRA emits a genesis receipt signed by a key only the operator controls. They then type `dema mission "summarize my week"`; Dema returns a receipt ID. They type `dema receipt show <id>`; a signed JSON appears, chain-linked to the genesis. They type `dema mission replay <id>`; the same output emerges, byte-identical where deterministic and signature-verified always. They copy the receipt to a friend. The friend runs `dema verify <receipt.json>` on their own machine — no cloud, no signup, no shared infrastructure — and sees `VALID · signature checks out · chain intact · IHSAN 0.96`. They know what the AI did. They can prove it.

---

## The required first-5-minutes experience

1. **0:00–1:00** — `git clone && cargo build --release` (or, in launch phase, download signed binary from GitHub release).
2. **1:00–2:00** — `dema activate-principal --name <me>`. Prints genesis receipt.
3. **2:00–3:30** — `dema mission "<first mission>"`. Dema runs local inference (Ollama/LM Studio fallback), applies gate chain, emits signed receipt.
4. **3:30–4:30** — `dema receipt show <id>` + `dema chain`. Sees JSON + chain head.
5. **4:30–5:00** — `dema mission replay <id>` succeeds; signature verifies; IHSAN band visible.

Anything that does not fit in five minutes is out of wedge scope. Browser operator, federation, PAT-7 agent spawn, token economics — all future.

---

## Minimum visible artifact

**One receipt JSON** — human-readable, portable, cryptographically signed, chain-referenced, fits in a GitHub Gist. This is the product. The runtime is the infrastructure; the receipt is the marketing.

Canonical shape (anchored in `bizra-mission/src/receipt.rs`):

```json
{
  "receipt_id": "<32-byte BLAKE3 hex>",
  "mission_id": "<32-byte BLAKE3 hex>",
  "final_state": "Complete",
  "submitted_at": 1761225600,
  "completed_at": 1761225601,
  "states_traversed": ["Submitted", "Queued", "Running", "Scoring", "Complete"],
  "chosen_model": "qwen2.5:3b",
  "ihsan_score": 0.96,
  "snr_score": 0.91,
  "guardian_approved": true,
  "previous_receipt_hash": "<32-byte>",
  "signature": "<64-byte Ed25519>",
  "chain_head_at_mint": { "height": 42, "hash": "<BLAKE3>" }
}
```

---

## Minimum Dema surface to ship the wedge

Required:

- ✅ `dema activate-principal` (works)
- ✅ `dema chain` (works)
- ⚠️ `dema mission submit "<intent>"` (needs CLI surface over existing runtime)
- ⚠️ `dema mission replay <id>` (needs CLI surface)
- ⚠️ `dema receipt show <id>` (schema exists; needs renderer)
- ⚠️ `dema verify <receipt.json>` (Ed25519 verify exists in Rust; needs CLI)
- ✅ Local inference fallback (Ollama / LM Studio tiered)

Optional (nice to have; defer if it would delay launch by > 2 weeks):

- Row 6 trust-surface web face (merge after it clears CI)
- `dema daily-digest` (Row 5, after scheduler lands)
- Cross-device sync / chain portability

Explicitly out of scope for wedge:

- PAT-7 / SAT-5 agent-team activation (rows 8-9 PLANNED)
- Token-economics UI (SEED/BLOOM frozen)
- Multi-user / team deployment
- Browser operator / computer-use mode
- Enterprise compliance export

---

## Launch anti-patterns (each with citation)

1. **Do not market as a chatbot.** Canonical category: verificative AI (`bizra-trust-compiler-thesis.md:14`). Chat UX is a different species.
2. **Do not promise enterprise governance.** No customer evidence in corpus. Using enterprise framing triggers procurement questions BIZRA cannot yet answer.
3. **Do not lead with decentralization / blockchain.** Single-node reality today (`bizra-now-vs-future-image-v1.md:128-129`). HyperBlockTree / PoI / BLOOM are future.
4. **Do not overlay token-economics on launch.** Frozen per canon (`al-mithaq-al-tasisi.md:37`; `bizra-now-vs-future-image-v1.md §90-100`).
5. **Do not over-specify hardware.** MSI Titan is internal dev spec; alienates A1 Linux users on consumer gear.
6. **Do not lead with Islamic-finance framing.** The constitution is the moat; trust-and-receipt is the launch language. A1 adopters span all cultures.
7. **Do not make a cloud account mandatory.** Violates sovereignty doctrine; breaks the wedge.
8. **Do not pitch "8 billion nodes."** Node0 canon is present-tense. The future-image belongs in docs, not in first-impression copy.
9. **Do not bundle agent-as-a-service at launch.** Hosting contradicts local-first. Offer it later as an _option_ for specific cross-device use cases only.
10. **Do not simulate demos.** No shadow state. No faked receipts. Every demo uses a real chain.

---

## Success metrics for this wedge (30 / 60 / 90 days post-launch)

| Horizon | Metric | Target | Source |
|---|---|---|---|
| 30 days | Organic installs (GitHub clones or downloads) | 100+ | Platform telemetry or download counter |
| 30 days | Verifiable receipts published externally (Gist, HN, X, blog) | 5+ | Manual inspection + trackback |
| 60 days | Unprompted inbound inquiries (any archetype) | 10+ | Email / GitHub issues |
| 60 days | Successful `dema verify` on third-party machine | 20+ receipts | Telemetry opt-in / user report |
| 90 days | First unsolicited pilot conversation (A3 or A5) | 1+ | Conversation log |
| 90 days | Contributor PRs from A6 archetype | 3+ | GitHub metrics |

Failure conditions (explicit):

- < 50 installs @ 30 days → channel problem, not product problem; revisit distribution
- 0 receipts shared externally @ 60 days → value prop is not clear to A1; revisit positioning
- 0 inbound inquiries @ 90 days → wrong archetype; consider pivot to A3 or A8

---

## The single next operator decision

**Decision required:** Accept (or contest) "Receipt-First Personal Mission Runtime for Solo Sovereign Operators" as the singular first wedge, and authorize post-Node0-closure completion of the minimum CLI surface named above (missions M05 · M03 · M06 · M11 in `BIZRA_Mission_Inventory_v0_1.csv`) as the Node0 Closure follow-on sprint after PR #49 and PR #50 land upstream and Row 6 merges.

No other decisions are required from this memo. Everything else is execution detail that proceeds only after that decision is typed.

---

**End of Decision Memo v0.1. Read-only audit. No code, branches, or PRs have been touched in this session.**
