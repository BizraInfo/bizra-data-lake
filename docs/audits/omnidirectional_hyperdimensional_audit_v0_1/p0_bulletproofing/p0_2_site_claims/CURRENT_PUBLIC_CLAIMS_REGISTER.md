# Current Public Claims Register — bizra.ai / bizra.info

**Capture source:** `../../artifacts/website_claims.json` (operator-supplied pre-check in `--no-network` mode) + brand canon v0.2 §15 + existing `docs/brand/public_launch_readiness/PUBLIC_CLAIMS_REGISTER.md`.

**Verification state:** bizra.ai is an SPA; plain HTTP fetch returns shell only. Content items C1–C9 were observed via browser pre-check and are treated as authoritative until a fresh headless-Chromium capture replaces them.

---

## Classification legend

| Code | Meaning | Ad use | Hero use | Sub-page use |
|---|---|---|---|---|
| **SAFE_NOW** | Identity / mission / philosophy — no numeric promise | ✅ | ✅ | ✅ |
| **SAFE_WITH_RECEIPT** | Defensible but requires a published receipt / methodology | ⚠️ with citation | ⚠️ prefer sub-page | ✅ with receipt |
| **REWRITE_REQUIRED** | Over-promises, brittle, or ambiguous — swap wording | ❌ | ❌ | ❌ (until rewritten) |
| **INTERNAL_ONLY** | Fine for dev docs / investor deck — not consumer surface | ❌ | ❌ | ❌ |
| **REMOVE_NOW** | High liability — delete from live copy today | ❌ | ❌ | ❌ |
| **PROHIBITED** | Never publicly (AGI, first-in-world, financial return, unsub. cert.) | ❌ | ❌ | ❌ |

---

## Register (21 entries)

### Identity / mission / philosophy — SAFE_NOW (8)

| # | Claim | Source | Class | Notes |
|---|---|---|---|---|
| I1 | "BIZRA / بذرة" — brand name + Arabic root | shell title, brand canon | **SAFE_NOW** | Free. |
| I2 | "The Seed of Sovereign Intelligence" | kit tagline, brand canon §9 | **SAFE_NOW** | Primary EN tagline. |
| I3 | "بذرة الذكاء السيادي" | kit tagline | **SAFE_NOW** | Primary AR tagline. |
| I4 | "Build with meaning. Act with proof. Grow with Ihsan." | kit motto | **SAFE_NOW** | Primary EN motto. |
| I5 | "ابنِ بالمعنى. اعمل بالبرهان. وانمُ بالإحسان." | kit motto | **SAFE_NOW** | Primary AR motto. |
| I6 | "Every human is a node. Every node is a seed." | brand canon §9 | **SAFE_NOW** | Movement line. |
| I7 | "Not another chatbot. Not another platform that owns you." | kit launch copy | **SAFE_NOW** | Differentiator; softened "owns you" kept. |
| I8 | "One human. One node. One sovereign path." | kit launch copy | **SAFE_NOW** | — |

### Quantitative / technical — REMOVE_NOW (3)

| # | Claim | Source | Class | Action |
|---|---|---|---|---|
| C4 | "cost per action dropping from about $0.10 toward $0.008" | bizra.ai pre-check | **REMOVE_NOW** | Delete from live copy. No published methodology. Precise $ triggers ad-platform substantiation review. Replace via `CLAIM_SAFE_REWRITE_PACK.md §C4`. |
| C5 | "SNR 0.974" | bizra.ai pre-check | **REMOVE_NOW** | Delete. Exact SNR without published benchmark = regulator flag. Replace via `§C5`. |
| C9 | "73 of 100 nodes remaining" | bizra.ai pre-check | **REMOVE_NOW** | Delete OR wire live counter backed by source-of-truth. Manufactured-scarcity claim without live counter = "deceptive practices" risk. Replace via `§C9`. |

### Quantitative / technical — REWRITE_REQUIRED (3)

| # | Claim | Source | Class | Action |
|---|---|---|---|---|
| C1 | "local agents / no cloud dependency" | bizra.ai pre-check | **REWRITE_REQUIRED** | Architecturally local-first but design is cloud-*optional* (Postgres / URP reconciliation contemplate cloud-sync by user choice). Reframe as "your machine, your keys, your node." Replace via `§C1`. |
| C7 | "100% pass rate" | bizra.ai pre-check | **REWRITE_REQUIRED** | Brittle (any future CI red falsifies), compliance-adjacent. Replace with policy claim: "CI must pass before merge." See `§C7`. |
| K1 | "BIZRA is live." | kit launch copy | **REWRITE_REQUIRED** | "Live" implies production readiness beyond evidence. Replace with "The Seed is public." See `§K1`. |

### Quantitative / technical — SAFE_WITH_RECEIPT (5)

| # | Claim | Source | Class | Action |
|---|---|---|---|---|
| C2 | "no telemetry" | bizra.ai pre-check | **SAFE_WITH_RECEIPT** | Publish privacy policy + architecture-level attestation. Then retain claim. Otherwise soften via `§C2`. |
| C3 | "Ed25519 receipt signatures" | bizra.ai pre-check | **SAFE_WITH_RECEIPT** | Architecturally true (`canonical_receipt.rs`). Keep in dev / investor docs (**INTERNAL_ONLY for consumer hero**). For consumer surface: soften to direction, link receipt chain example. |
| C6 | "8,072 verified tests" | bizra.ai pre-check | **SAFE_WITH_RECEIPT** | Backed by `pytest --collect-only` + `cargo test -- --list` if timestamped + commit-hash-linked. See `RECEIPTIFICATION_REQUIREMENTS.md §C6`. |
| C8 | "Ihsan Gate >= 0.95" | bizra.ai pre-check | **SAFE_WITH_RECEIPT** | Accurate to `core/integration/constants.py`. Frame as "internal conscience gate" — keep with context. See `§C8`. |
| C6b | "Thousands of verified tests" (softer variant) | proposed | **SAFE_NOW** | Directional framing; always-safe fallback. |

### Internal-only technical terminology — INTERNAL_ONLY (2)

| # | Claim | Source | Class | Action |
|---|---|---|---|---|
| T1 | "Ed25519 receipt signatures" (in consumer hero) | bizra.ai pre-check | **INTERNAL_ONLY** | Remove from consumer hero. Keep in `bizra.ai/under-the-hood` / investor deck / GitHub readme. |
| T2 | "PAT / SAT / URP" internal-topology language | not visible today | **INTERNAL_ONLY** | Verify not leaked. Keep watchlist. |

### Live watchlist — PROHIBITED (0 today, maintain vigilance)

| # | Claim | Status | Notes |
|---|---|---|---|
| P1 | AGI claims | not currently live | Monitor — brand canon §15 forbids |
| P2 | "first-in-world" / "only-X" claims | not currently live | Monitor |
| P3 | Financial-return / guaranteed-savings claims | not currently live | Monitor |
| P4 | SOC2 / ISO certs (none obtained) | not currently live | Monitor |
| P5 | Benchmark-superiority ("beats GPT-X") | not currently live | Monitor |

---

## Summary

- **Most urgent (REMOVE_NOW, 3):** C4, C5, C9. These are ad-platform liability and should come off the live site before paid advertising or organic campaigns drive external attention.
- **Fastest wins (REWRITE_REQUIRED, 3):** C1, C7, K1. Small wording edits with ready replacements in `CLAIM_SAFE_REWRITE_PACK.md`.
- **Receipt-gated (SAFE_WITH_RECEIPT, 5):** C2, C3, C6, C8, C6b. Return to live copy once the receipt chain is published.
- **Identity (SAFE_NOW, 8):** I1-I8. Use freely — these are the safe core of any launch copy today.
