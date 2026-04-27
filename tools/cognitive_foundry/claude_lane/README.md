# Claude Cognitive Archive Pilot

Deterministic, repo-contained pipeline that processes a **Claude export zip** into reviewable knowledge candidates.

**Pilot version:** `0.1.0-pilot`
**Language:** Python 3.10+, stdlib only (no `pip install` required)
**Location:** `tools/cognitive_foundry/claude_lane/`
**Safety:** isolated from Node0 runtime. Never mutates `MEMORY.md`, never promotes to canon.

---

## Quick start

From the repo root:

```bash
python tools/cognitive_foundry/claude_lane/run_pipeline.py \
  --archive /absolute/path/to/claude-export.zip
```

Outputs land under:

```
tools/cognitive_foundry/claude_lane/output/<run_id>/
├── run_manifest.json
├── 01_inventory/
├── 02_distillation/
├── 03_adjudication/
└── 04_review_pack/
```

**The only file a human reviewer opens next** is `04_review_pack/review_workbook.csv`.

---

## Input

A standard Claude export zip. The pipeline expects these four JSON files to exist anywhere inside the archive:

- `users.json` — user profile (read for future lanes; currently unused)
- `projects.json` — project registry (inventory uses this)
- `memories.json` — user memory entries (read for future lanes; currently unused)
- `conversations.json` — full conversation + message log

If any of the four are missing, the pipeline exits with a clear error.

---

## Pipeline stages

### Stage 1 — Ingest / Inventory (`01_inventory/`)

Heuristic-only. No LLM. Reads the archive and emits:

| File | Rows | Purpose |
|---|---|---|
| `conversation_inventory.csv` | 1 per conversation | uuid, name, project, turn counts, topic buckets |
| `project_inventory.csv` | 1 per project | uuid, name, is_starred, conversation count |
| `topic_bucket_counts.csv` | 1 per bucket | conversation/message/char counts per bucket |
| `top_signal_sessions.csv` | top-K | rank + signal_score + turn counts |
| `run_manifest.json` | — | stage metadata |

Topic buckets come from `config.py`; defaults cover BIZRA-specific buckets plus an `uncategorized` catch-all.

### Stage 2 — Distillation (`02_distillation/`)

Heuristic-only. Extracts candidates via regex + keyword patterns. Defaults extract only from **user** messages (the assistant's voice is noise for canonization).

| File | Extraction |
|---|---|
| `fact_candidates.csv` | `I am …`, `My X is …`, `Entity is …`, `I live in …` |
| `decision_candidates.csv` | `I decided …`, `Let's …`, `The plan is …`, `Go with …` |
| `contradiction_candidates.csv` | Same (entity, predicate) with ≥2 distinct values across sessions |
| `reasoning_exemplars.csv` | Long turns (>= 1500 chars) with `because`/`therefore`/numbered lists |

Every row carries provenance:
`source_lane`, `source_conversation_uuid`, `source_message_uuid`, `source_created_at`.

Candidate IDs are deterministic:
`sha256(candidate_type | normalized_text | source_message_uuid)[:16]`.

### Stage 3 — Adjudication (`03_adjudication/`)

Heuristic clustering. No LLM.

| File | Content |
|---|---|
| `canonical_candidate_facts.csv` | Most-recent fact in each (entity, predicate) cluster |
| `canonical_candidate_decisions.csv` | Most-recent decision per normalized-text cluster |
| `hypothesis_candidates.csv` | Single-occurrence facts/decisions (default threshold: occurrences ≤ 1) |
| `obsolete_conflicted_candidates.csv` | Older cluster members whose normalized value differs from canonical + age delta ≥ 7 days |
| `cluster_registry.csv` | 1 row per cluster (type, member count, canonical candidate, member ids) |

### Stage 4 — Human Review Pack (`04_review_pack/`)

Consolidates Stage 3 outputs into a reviewer-ready form.

| File | Purpose |
|---|---|
| `review_workbook.csv` | **The primary review artifact.** One row per reviewable candidate (fact, decision, hypothesis, obsolete) with columns: `review_status`, `reviewer_notes`, `promote_to_canon`. |
| `facts_for_review.csv` | Facts subset (same schema) |
| `decisions_for_review.csv` | Decisions subset |
| `hypotheses_for_review.csv` | Hypotheses subset |
| `human_review_brief.md` | Readable instructions + counts + review heuristics |

**Every row starts with `review_status = pending_review` and `promote_to_canon` blank.** The pipeline NEVER sets `promote_to_canon = yes`. Promotion is human-only and requires the separate `promote.py` step (see **Promotion** section below).

---

## Configuration

Defaults in `config.py`. No CLI overrides in the pilot — edit the file if a knob needs changing. Key knobs:

- `DistillationThresholds` — fact/decision length bounds, reasoning-marker requirement
- `AdjudicationThresholds` — hypothesis threshold (default 1), obsolete age delta (default 7 days)
- `TopSignalConfig` — top-K (default 50), min turn count (default 4)
- `DEFAULT_TOPIC_BUCKETS` — list of `TopicBucket(name, keywords)` — add or modify

---

## Assumptions (explicit)

1. **No LLM calls.** The pilot is pure heuristics. An OpenAI / Gemini-powered distillation lane is out-of-scope; this is the foundation they will plug into.
2. **No fabrication.** If zero rows match the heuristics, the CSV is empty. Empty results are a valid outcome — they mean "nothing in this archive matched the patterns," not "analysis failed."
3. **Stdlib-only.** Works on a fresh Python 3.10+ install. No `pip install`, no `pyproject.toml` change.
4. **Deterministic.** Same input → same output. IDs are content-derived. CSVs use stable sort keys.
5. **Provenance preserved.** Every candidate carries its source conversation/message uuid and timestamp.
6. **No canon writes.** The pipeline does NOT touch `MEMORY.md`, any Node0 runtime file, or any PR-in-flight branch.
7. **No cross-lane contamination.** Rows are tagged `source_lane = "claude"`. Future OpenAI / Gemini lanes emit `"openai"` / `"gemini"` so adjudication across lanes is possible later.
8. **Archive layout flexibility.** Expected JSON files are matched by suffix; nested paths inside the zip are OK as long as the four filenames are present somewhere.

---

## What's deliberately NOT in the pilot

| Not done | Why | Future lane |
|---|---|---|
| LLM-powered distillation (semantic entity extraction, paraphrase clustering) | Adds API deps, cost, and non-determinism | Future `enrichment_lane/` |
| Cross-lane adjudication (Claude + OpenAI + Gemini together) | Needs all three lanes first | After at least two lanes exist |
| Auto-promotion to MEMORY.md | Explicit canon-separation rule | Separate `promote.py`, behind a human confirmation gate |
| Memory entries from `memories.json` | Claude's native memory format is lane-specific; deserves its own distillation path | Add to this lane next iteration |
| Attachment / file content processing | Pilot focuses on text turns | Attachment lane — later |
| Embedding-based clustering | Stdlib-only constraint + determinism | After embeddings infra exists |

---

## Running individual stages

Use `--stages` to run a subset:

```bash
# Inventory only
python tools/cognitive_foundry/claude_lane/run_pipeline.py --archive X.zip --stages 1

# Stages 2 and 3 only (assumes 1 already ran with same run-id)
python tools/cognitive_foundry/claude_lane/run_pipeline.py --archive X.zip --stages 2,3 --run-id <existing_run_id>
```

Stages read their inputs from the previous stage's CSVs on disk, so partial reruns work.

---

## Human review workflow (Stage 5, not in pipeline)

1. Open `04_review_pack/review_workbook.csv` in your spreadsheet tool of choice.
2. Read `human_review_brief.md` for review heuristics.
3. Annotate each row: set `review_status` (`approved` / `rejected` / `needs_followup`) and free-form `reviewer_notes`.
4. Only for rows you want promoted to canon, set `promote_to_canon = yes`.
5. Save the file. Then run `promote.py` (below) to produce a structured canon pack.

## Promotion (`promote.py`)

After you've annotated the review workbook, produce a hash-signed canon pack:

```bash
python tools/cognitive_foundry/claude_lane/promote.py \
  --workbook tools/cognitive_foundry/claude_lane/output/<run_id>/04_review_pack/review_workbook.csv \
  [--output-dir tools/cognitive_foundry/claude_lane/canon_packs] \
  [--dry-run]
```

**Produces** (under `canon_packs/<origin_run_id>_promoted_<ts>_<issuance_hash_prefix12>/`):

- `canon_pack.json` — structured entries, one per promoted row
- `canon_pack.csv` — same, human-readable
- `canon_pack.manifest.json` — pack metadata + split hashes + partition counts + workbook sha256
- `promotion_report.md` — partition summary + validation status

### Hash model (v0.2.0, split identity)

The v0.2.0 fix separates **content identity** from **promotion-event identity**:

| Field | What it is | Determinism |
|---|---|---|
| `content_hash_blake2b_32` | **Deterministic** over stable fields only (canon_entry_id, source_candidate_id, candidate_type, content, entity, predicate, provenance, review_status, reviewer_notes, origin_run_id). Explicitly excludes `promoted_at` and `promoter`. | Same workbook + same reviewed content → **same content_hash every time**, regardless of when the promotion ran. |
| `issuance_hash_blake2b_32` | Identity of **one specific promotion event**: `blake2b("issuance\|v1\|" \| content_hash \| promoted_at \| promoter \| workbook_sha256)[:32]`. | Unique per promotion event. Two reruns produce identical `content_hash` but different `issuance_hash`. |
| `entries_hash_blake2b_32` | **Backward-compatibility alias.** Under v0.2.0 this equals `content_hash_blake2b_32`. Old consumers reading this field get the stable content hash — which is the more useful of the two. | Deterministic. |
| `canon_entry_id` (per-entry, v2) | `blake2b("canon_entry\|v2\|" \| source_candidate_id \| content)[:16]`. `promoted_at` is recorded as entry provenance but NOT mixed into the id. | Stable across reruns and re-promotions. |
| Pack directory name | `<origin_run>_promoted_<ts>_<issuance_hash_prefix12>` — uses **issuance** hash so reruns never collide on disk. | Each promotion event gets its own directory. |

**Why this split matters:**

- **Re-promotion is safe.** If you re-run the reviewed workbook through `promote.py` later, the `content_hash` and every `canon_entry_id` stay identical — future ingestion tools can cheaply deduplicate and skip already-ingested content.
- **Audit still works.** Every promotion event has a unique `issuance_hash` that captures *this specific run at this specific time by this specific promoter*.
- **No directory collisions.** Two promotions of the same content at different times write to separate directories because the directory name embeds `issuance_hash`, not content hash.

### v0.1.0 → v0.2.0 compatibility

- **v0.1.0 packs are valid-but-superseded.** They use the old `entries_hash` formula which mixed `promoted_at` into the per-entry id. The promoted row set itself is fine; only the hash shape differs.
- **Future ingestion tools should prefer v0.2.0 packs.** The stable `content_hash` is the canonical identity; an ingest tool that consumes v0.2.0 can safely detect re-promotions and deduplicate.
- **Never ingest a v0.1.0 pack and a v0.2.0 pack for the same content both**; pick one. The default choice is the v0.2.0 pack.
- Tool version is recorded in each pack's `canon_pack.manifest.json` as `tool_version` (`0.1.0` or `0.2.0`) and as `hash_model` (`v2_split_content_and_issuance` for v0.2.0).

See `canon_packs/README.md` for concrete disposition of the packs currently on disk.

**Strict rules enforced in code:**

- Only rows with `review_status=approved` AND `promote_to_canon=yes` are promoted. Both must be set explicitly by a human.
- Contradictions (e.g., `promote_to_canon=yes` but `review_status=rejected`) → hard stop, exit 4, no pack written.
- Invalid `review_status` or `promote_to_canon` values → hard stop.
- Zero approved+promoted rows → clean no-op report, no pack, exit 0.
- Tool NEVER writes to `MEMORY.md` or any BIZRA runtime file.
- Tool NEVER mutates the input workbook.
- Same workbook + same content = same `content_hash` across all reruns (v0.2.0).

**What promote.py does NOT do (explicit):**

- Does NOT perform cryptographic signing (hash-only content-addressing). A future runtime tool can upgrade to Ed25519 if required.
- Does NOT ingest the pack into any canonical BIZRA store. The pack sits on disk awaiting a SEPARATE, human-gated ingestion tool (not yet implemented).
- Does NOT auto-set `promote_to_canon=yes` under any circumstance.

---

## Safety checklist (what the pipeline asserts on itself)

- [x] Writes only under `tools/cognitive_foundry/claude_lane/output/<run_id>/`.
- [x] Never writes to `MEMORY.md`.
- [x] Never sets `promote_to_canon = yes`.
- [x] Never modifies `core/`, `bizra-omega/`, `docs/strategy/`, or any other repo path.
- [x] Never makes network calls (stdlib-only, no HTTP client imports).
- [x] Produces deterministic output (content-derived IDs, stable sort keys).
- [x] Emits a top-level `run_manifest.json` that records what was done and when.

---

## Next step after running

Read `04_review_pack/human_review_brief.md`. Review. Annotate. Stop before promotion.
