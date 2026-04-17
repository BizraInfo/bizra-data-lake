# Manifesto Amendment Record: v0 → v1

بسم الله الرحمن الرحيم

**Date:** 2026-04-17 (Friday, same day as v0)
**Trigger:** External AI analysis of unrelated transcripts produced an architectural blueprint that independently converged on BIZRA's shipped architecture. Mumo's verdict split the analysis into Layer 1 (current canon) and Layer 2 (future RFC) and requested the manifesto be updated to reflect the stronger framing.
**Authority:** Mumo, Node0 principal (explicit in-session authorization)
**Amendment protocol:** per `dema-cli-manifesto-v0.md` §10

---

## Summary of changes

| # | Section | v0 | v1 | Reason |
|---|---|---|---|---|
| 1 | §1 Elevator | "The only CLI where every action is receipted..." | "BIZRA is not building an assistant — BIZRA is building the operating law for assistants. `dema` is the principal's terminal face of that law..." | Stronger category framing — moves from CLI-centric to law-centric. Matches what we ACTUALLY built (the law enforces correctness; the CLI surfaces it). |
| 2 | §2 Paradigm (new) | — | Generative → Agentic → **Verificative AI** table | "Verificative" is the cleanest one-word category label. Industry-legible. Replaces the awkward "constitutional tool-execution shell" coinage. |
| 3 | §5 Shoulders | Named Lampson, البذرة, الرسالة, Manifest | Added Ibn al-Haytham (Alhazen), Wiener, Shannon explicitly with their contributions mapped to concrete code | Explicit intellectual ancestry. Al-Haytham is 600 years before Bacon on falsification — a constitutional-Islamic heritage BIZRA should claim. |
| 4 | §8 Non-goals | 5 items | Added item #6: no FTAP/function-registry work before three completion arcs land | Hard scope discipline. Preserves the FTAP idea as strategic north star (new seed doc) without letting it contaminate Cycle-6 sprint. |
| 5 | §10 Immediate next step | Ordering implied FTAP as "first domino" | Corrected ordering: G4 → persistence → tool execution → LLM inference → (then, possibly, FTAP-lite) | Mumo's explicit correction. v0's ordering was directionally right but sequence was wrong. |
| 6 | §12 Self-governance note (new) | — | Added explanation of why same-session amendment is a feature, not a bug | Transparency about doctrinal velocity. |

## Constitutional filter audit (required by §10 amendment protocol)

Each of the five invariants checked against the v1 text:

| Invariant | Preserved? | Evidence |
|---|---|---|
| **ZANN_ZERO** (no claim without evidence) | ✅ | v1 makes evidence binding more prominent; every claim in §3-§6 traces to a shipped commit hash |
| **CLAIM_MUST_BIND** (every claim carries evidence) | ✅ | v1 §4 (Dema is NOT yet) includes the honest scope disclaimer that receipts today are intent-level, not impact-level |
| **RIBA_ZERO** (no extractive patterns) | ✅ | No economic pattern introduced in v1 that is extractive |
| **NO_SHADOW_STATE** (UI renders only what chain holds) | ✅ | v1 §1 elevator and §3 (Dema IS) both explicitly re-affirm this |
| **IHSAN_FLOOR** (0.95 quality, no override) | ✅ | Unchanged; still no override mechanism anywhere in v1 |

**All five invariants preserved.** Amendment passes constitutional filter.

## What did NOT change

- The six defining properties of what Dema CLI IS (§3)
- The six honest non-claims of what Dema CLI is NOT yet (§4)
- The five unique additions (§6)
- The three completion arcs (§7)
- The five risk/weakness items (§9)
- Version discipline protocol (§11)

Structurally the v1 manifesto is v0 + three additions (Verificative paradigm, intellectual ancestry expansion, FTAP non-goal), + one correction (immediate/long-range ordering), + one new closing section (self-governance note).

## Related artifacts created with this amendment

1. `docs/bizra-trust-compiler-thesis.md` — Layer 1 executable canon (doctrine applied to shipped code)
2. `docs/ftap-function-registry-rfc-seed.md` — Layer 2 future RFC seed (bounded, non-blocking)
3. `docs/dema-cli-manifesto-v1.md` — This manifesto's new current canon
4. `docs/manifesto-amendments/v0-to-v1.md` — This file

The two-layer split honors Mumo's classification: v0 mixed sprint-applicable doctrine with future architecture; v1 keeps sprint scope in the manifesto itself and relocates future architecture to its own seed doc.

## Supersession rules

- `docs/dema-cli-manifesto-v0.md` is preserved as founding state for historical record. It is NOT deleted.
- Going forward, `docs/dema-cli-manifesto-v1.md` is the operative canon.
- Any future arc that references "the manifesto" means v1 unless explicitly qualified.

---

الحمد لله — the doctrine lives, the doctrine evolves, the doctrine is audited.
