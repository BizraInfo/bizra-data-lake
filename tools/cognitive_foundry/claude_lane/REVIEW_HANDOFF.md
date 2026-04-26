# Claude Cognitive Foundry — Review Handoff

**Checkpoint date:** 2026-04-24 (GST) — **FULL REVIEW COMPLETE**
**Purpose:** Minimum context for a fresh Claude Code session to resume work without re-deriving state. Read this first.

---

## 1. Preferred reviewed workbook

```
tools/cognitive_foundry/claude_lane/output/20260424T000948Z_78a12953b97a1085/04_review_pack/review_workbook.csv
```

- Origin pipeline run id: `20260424T000948Z_78a12953b97a1085`
- Workbook schema: 15 columns per `schema.py → REVIEW_WORKBOOK_COLS`
- Total rows: **359** (71 facts + 131 decisions + 157 hypotheses + 0 obsolete)
- **Review state: COMPLETE. `pending=0` across all candidate types.**

### Safety backups on disk

| Backup | Phase |
|---|---|
| `review_workbook.csv.pre_review_backup` | Pre-fact-review (byte-identical to pipeline output) |
| `review_workbook.csv.pre_decision_review_backup` | Post-fact-review, pre-decision labeling |
| `review_workbook.csv.pre_hypothesis_apply_20260424T052705Z` | Post-decision-review, pre-hypothesis labeling |

---

## 2. Preferred canon pack (final full-review state)

```
tools/cognitive_foundry/claude_lane/canon_packs/20260424T000948Z_78a12953b97a1085_promoted_20260424T053225Z_cface302f993/
```

- 4 files per pack: `canon_pack.json`, `canon_pack.csv`, `canon_pack.manifest.json`, `promotion_report.md`
- `content_hash_blake2b_32`: `f76d8c1fde61f17e19d50a5a26004155fd6febf6ddd2146903f92eb9fbe8c770`
- `issuance_hash_blake2b_32`: `cface302f993108d0a414bca020162a0d97d4b6baa543fbb4330d7efff668385`
- `promoted_at_utc`: `2026-04-24T05:32:25Z`
- Tool version: `0.2.0`
- Hash model: `v2_split_content_and_issuance`
- Entries: **27** (12 facts + 13 decisions + 2 hypotheses)
- `human_gated`: true; `non_promotion_tool`: true

---

## 3. Final full-review result

| Metric | Value |
|---|---|
| Total rows reviewed | **359** of 359 (pending=0) |
| Fact rows reviewed | 71 of 71 (approved=14, needs_followup=32, rejected=25) |
| Decision rows reviewed | 131 of 131 (approved=30, needs_followup=46, rejected=55) |
| Hypothesis rows reviewed | 157 of 157 (approved=4, needs_followup=59, rejected=94) |
| promote_to_canon=yes total | 27 |
| &nbsp;&nbsp;facts promoted | 12 |
| &nbsp;&nbsp;decisions promoted | 13 |
| &nbsp;&nbsp;hypotheses promoted | 2 |
| Canon packs on disk | 5 (1 preferred + 4 historical-superseded) |

### Promoted row IDs (full inventory)

- **Facts (12):** R000003, R000013, R000019, R000029, R000033, R000034, R000042, R000043, R000044, R000046, R000048, R000063
- **Decisions (13):** R000103, R000117, R000128, R000137, R000141, R000142, R000148, R000152, R000164, R000171, R000179, R000183, R000188
- **Hypotheses (2):** R000305, R000313

Thematic clusters of promoted content (across all types): BIZRA-identity, Node0-doctrine, sovereignty/operating-principles, review-discipline, evidence-discipline, human-in-the-loop governance, formal-verification posture, compositional verification, canonical JSON invariants, reproducible builds (Nix-pinning), Observe→Reason→Act→Verify→Learn→Reuse cognitive loop, BZT/BZC terminology. See `canon_pack.json` and `promotion_report.md` in the preferred pack for exact content.

---

## 4. Hash model (v0.2.0 split identity)

| Hash | Role | Determinism |
|---|---|---|
| `content_hash_blake2b_32` | Deterministic reviewed-content identity. Excludes `promoted_at`, `promoter`. | Same workbook + same reviewed content → same hash every rerun. |
| `issuance_hash_blake2b_32` | Promotion-event identity. `blake2b("issuance\|v1\|" \| content_hash \| promoted_at \| promoter \| workbook_sha256)[:32]` | Unique per promotion event. |
| `entries_hash_blake2b_32` | Backward-compat alias. Under v0.2.0 == `content_hash_blake2b_32`. | Deterministic. |
| `canon_entry_id` (v2) | `blake2b("canon_entry\|v2\|" \| source_candidate_id \| content)[:16]` — does NOT include `promoted_at`. | Stable across reruns and re-promotions. |

Pack directory naming uses `issuance_hash` prefix so reruns never collide on disk.

See `tools/cognitive_foundry/claude_lane/README.md` → Promotion → "Hash model (v0.2.0, split identity)" for the full treatment.

---

## 5. Canon-pack disposition (5 packs on disk)

| Pack | Tool version | Entries | Disposition |
|---|---|---|---|
| `…_20260424T021405Z_07bd7dfd7a76` | v0.1.0 | 12 (facts) | **Historical / superseded.** v1 hash model; earliest snapshot. |
| `…_20260424T021954Z_82816efc46b0` | v0.2.0 | 12 (facts) | **Historical / superseded.** Was preferred during fact-only phase. |
| `…_20260424T021956Z_1273115ff550` | v0.2.0 | 12 (facts) | **Historical / superseded.** Determinism-proof twin. |
| `…_20260424T031902Z_67d1f4c85271` | v0.2.0 | 25 (facts + decisions) | **Historical / superseded.** Was preferred during facts+decisions phase. |
| `…_20260424T053225Z_cface302f993` | v0.2.0 | **27 (facts + decisions + hypotheses)** | **PREFERRED PACK.** Full-review final state. Point future ingestion tool here. |

All five packs share `origin_run_id = 20260424T000948Z_78a12953b97a1085`. They are honest snapshots of the same origin run at successive review-completeness milestones. The 27-entry pack is a strict superset of every earlier pack's promotions.

**No pack has been ingested into `MEMORY.md` or any BIZRA runtime canonical store.** A separate, not-yet-implemented **Canon Store Ingestion Gate** would perform that step.

See `tools/cognitive_foundry/claude_lane/canon_packs/README.md` for full disposition notes.

---

## 6. What the preferred pack is — and is not

- **IS:** Foundry canon-pack output. Candidate-for-canon. Human-reviewed. Content-addressable via `content_hash_blake2b_32`. Deterministic across reruns on the same reviewed content.
- **IS NOT:** BIZRA runtime canon. NOT `MEMORY.md`. NOT a signed sovereign receipt. NOT cryptographically sealed. NOT yet wired to any BIZRA canonical store.

Any forward use of the pack (ingestion, surface rendering, runtime consultation) **requires** building the Canon Store Ingestion Gate first — a separate, human-gated tool that has not been started and requires explicit typed authorization before it is started.

---

## 7. Next recommended work (options — pick one with explicit typed authorization)

The Foundry review cycle for origin run `20260424T000948Z_78a12953b97a1085` is **complete**. No further action is forced. Likely follow-ons:

- **A. Canon Store Ingestion Gate design** — specify the human-gated tool that ingests a preferred pack into a runtime-visible canonical store. Spec-first; brand-new lane. NOT to be started without explicit authorization.
- **B. New Foundry run** — point `inventory.py` at a different archive (different origin run). Same 4-stage pipeline; produces a new workbook + new review cycle.
- **C. Workbook annotation pass** — revisit `needs_followup` rows (137 across all types) for a second review pass. Would materialize a new pack if any rows flip to `approved + promote=yes`.
- **D. Stop / land the plane** — treat the 27-entry pack as the stable checkpoint for this origin run and close the loop for now.

Do NOT auto-start any of A–C. Each is a new lane.

---

## Guardrails carried into the next session (do NOT violate)

- **Do NOT touch Node0 runtime code** (`core/`, `bizra-omega/` outside `tools/cognitive_foundry/`).
- **Do NOT touch receipt-lineage WIP** — pre-existing dirty state on branch `prep/node0-closure-receipt-lineage`; leave as-is.
- **Do NOT touch PR #49 / #50** — in-flight merges.
- **Do NOT edit `MEMORY.md`** — canon discipline.
- **Do NOT auto-set `promote_to_canon=yes`** — human-only.
- **Do NOT ingest any canon pack** — no ingestion tool exists yet; design and build require explicit authorization.
- **Do NOT start LangExtract, GraphRAG, Hypergraph RAG, HRM, motif-pack, audit-protocol lanes** — deferred.
- **Do NOT start brand-lane work** (brand identity canon v0.2 is OUT of scope here — separate surface, separate discipline).
- **Do NOT launch new lanes** without explicit typed authorization from the operator.

---

## Open threads (not blocking)

- R000039 (founder-voice, supporting_count=8) was flagged during fact review for a potential separate "Founder Voice / Origin Canon" lane later — not mixed with architecture canon.
- R000035 (P2 Glass Cockpit) + R000060 (P2 cockpit/receipts layer) are marked for merger in canon-pack review — both describe P2.
- R000019 wording-care flag recorded: "zero-dependency" = sovereign bootstrap / non-gatekeeper, NOT literal no-software-dependencies. Future ingest tool should carry this annotation.
- R000028 polemic wording, R000029 vision-tightening — noted in reviewer_notes, will need editorial pass if/when ingestion occurs.
- Earlier test fixtures (`v2_pack_2`) can be safely deleted at any time without affecting the preferred pack.
- Superseded packs (4 of them) can be safely deleted if disk hygiene becomes a concern; no downstream tool references them.

---

**End of handoff note.** Full review of origin run `20260424T000948Z_78a12953b97a1085` is complete. The 27-entry preferred pack sits on disk awaiting a separate, human-gated Canon Store Ingestion Gate that has not yet been designed or authorized. Start the next session by choosing one of the options in §7 — or explicitly deciding to stop and land the plane.
