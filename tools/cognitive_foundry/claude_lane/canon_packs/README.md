# Canon Packs — Disposition Notes

**Purpose:** staging area for canon packs produced by `promote.py`. **Nothing here has been ingested into any BIZRA canonical store.** Each pack sits on disk awaiting a separate, not-yet-implemented Canon Store Ingestion Gate.

## Current packs (as of 2026-04-24 — final review state)

All five packs below were generated from successive review states of the same reviewed workbook:

`tools/cognitive_foundry/claude_lane/output/20260424T000948Z_78a12953b97a1085/04_review_pack/review_workbook.csv`

Each is an honest snapshot of the **same origin run** at a different point in the human review cycle. The review progressed fact → decision → hypothesis; each milestone materialized a pack. The final 27-entry pack reflects the workbook in its fully-reviewed state (facts + decisions + hypotheses, `pending=0`) and supersedes all earlier snapshots for preferred-pack purposes.

| Pack | Tool | Hash model | promoted_at (UTC) | Entries | Disposition |
|---|---|---|---|---|---|
| `20260424T000948Z_..._promoted_20260424T021405Z_07bd7dfd7a76` | `promote.py` v0.1.0 | v1 (promoted_at mixed into canon_entry_id) | 2026-04-24T02:14:05Z | 12 (facts only) | **Historical / superseded.** Earliest snapshot; v1 hash model. Do NOT prefer for future ingestion. |
| `20260424T000948Z_..._promoted_20260424T021954Z_82816efc46b0` | `promote.py` v0.2.0 | v2 (split content/issuance) | 2026-04-24T02:19:54Z | 12 (facts only) | **Historical / superseded.** First v0.2.0 pack; was preferred during fact-only review phase. |
| `20260424T000948Z_..._promoted_20260424T021956Z_1273115ff550` | `promote.py` v0.2.0 | v2 (split content/issuance) | 2026-04-24T02:19:56Z | 12 (facts only) | **Historical / superseded.** Determinism-proof twin of the preceding v0.2.0 pack. Same content_hash, different issuance_hash. |
| `20260424T000948Z_..._promoted_20260424T031902Z_67d1f4c85271` | `promote.py` v0.2.0 | v2 (split content/issuance) | 2026-04-24T03:19:02Z | 25 (12 facts + 13 decisions) | **Historical / superseded.** Was preferred during the facts+decisions phase. |
| `20260424T000948Z_..._promoted_20260424T053225Z_cface302f993` | `promote.py` v0.2.0 | v2 (split content/issuance) | 2026-04-24T05:32:25Z | **27 (12 facts + 13 decisions + 2 hypotheses)** | **PREFERRED PACK.** Full-review final state. `pending=0` across the workbook. Future ingestion tool should point here. |

**Final preferred-pack content_hash:** `f76d8c1fde61f17e19d50a5a26004155fd6febf6ddd2146903f92eb9fbe8c770`
**Final preferred-pack issuance_hash:** `cface302f993108d0a414bca020162a0d97d4b6baa543fbb4330d7efff668385`

## Why earlier packs are superseded

Two orthogonal reasons apply across the set:

1. **Hash-model migration (v0.1.0 → v0.2.0).** The v0.1.0 promotion mixed `promoted_at` into the per-entry `canon_entry_id`, which meant every rerun of `promote.py` on the same reviewed content produced different entry IDs and different pack hashes. That made re-promotion detection unreliable for any future ingestion tool. The v0.2.0 fix (see `../README.md` → Promotion → Hash model) separates:
   - `content_hash_blake2b_32` — deterministic reviewed-content identity
   - `issuance_hash_blake2b_32` — promotion-event identity (unique per run)

2. **Review-completeness progression.** The human review cycle processed fact → decision → hypothesis in that order. Each phase completion materialized a pack from the workbook state at that moment. The 27-entry pack is the strict superset that includes all content from every earlier snapshot — every promotion in any earlier pack is also in the final pack, plus two additional hypothesis promotions (`R000305`, `R000313`).

Taken together: the final 27-entry pack is correct on hash model AND carries the complete reviewed content. Earlier packs are honest, valid snapshots of an earlier review state — but they should not be preferred for future ingestion.

## Why superseded packs stay on disk

- They are an honest historical record of the review trajectory.
- Nothing in them is wrong; they represent earlier review-completeness states (and for the v0.1.0 pack, an earlier hash model).
- No tool is ingesting from them, so they cannot contaminate canon.
- Deletion would lose the audit breadcrumb for the review cycle and the v0.1.0 → v0.2.0 migration.

If disk hygiene becomes a concern later, superseded packs can be safely deleted AFTER any downstream reference (if any) has confirmed it no longer points at them. The final preferred pack is sufficient for all forward use.

## What this preferred pack is — and is not

- **IS:** a candidate-for-canon Foundry output. Human-reviewed. Content-addressable via `content_hash_blake2b_32`. Deterministic across reruns on the same reviewed content.
- **IS NOT:** BIZRA runtime canon. NOT `MEMORY.md`. NOT a signed sovereign receipt. NOT cryptographically sealed. NOT yet wired to any BIZRA canonical store.

## Canon-separation discipline (reiterated)

- **None of these packs are BIZRA canon.** They are *candidates for* canon, pending a separate **Canon Store Ingestion Gate** — a human-gated ingestion tool that does not yet exist.
- Do NOT hand-copy any pack content into `MEMORY.md`, `constants.py`, `topology_canon.rs`, any runtime store, or any public surface without the ingestion gate + a human confirmation step.
- Packs here are content-addressable by `content_hash_blake2b_32` for v0.2.0 packs; the future ingestion tool should deduplicate by that hash.

## Concrete preferred-pack path

For the current reviewed workbook, the preferred canon pack is:

```
tools/cognitive_foundry/claude_lane/canon_packs/20260424T000948Z_78a12953b97a1085_promoted_20260424T053225Z_cface302f993/
```

When the Canon Store Ingestion Gate is built, point it at that directory.
