# Human Review Brief — Run 20260423T234857Z_78a12953b97a1085

**Archive:** /home/bizra-operating-system/Downloads/data-43cc0344-1239-4174-867b-25beff52f00f-1776139825-1ff56815-batch-0000.zip
**Generated:** 2026-04-23 23:48Z
**Lane:** claude

## Counts

- Canonical candidate facts: **236**
- Canonical candidate decisions: **131**
- Hypothesis candidates (single-occurrence, not yet corroborated): **292**
- Obsolete/conflicted candidates (superseded by newer statements): **0**
- **Total rows in `review_workbook.csv`:** 659

## What this is

A deterministic, heuristic-extracted set of candidates from a Claude export. **Nothing in this pack is canon.** The pipeline cannot promote anything to canon. Only a human reading each row and typing `yes` in the `promote_to_canon` column can do that.

## How to review

1. Open `review_workbook.csv` in a spreadsheet tool (Excel, Numbers, LibreOffice, Google Sheets — any).
2. For each row:
   - Read `content`. Is the candidate a real, durable, non-trivial thing worth canonizing?
   - Check `provenance_conversation_uuids` + `provenance_earliest`/`provenance_most_recent` — is this from recent activity or stale?
   - Set `review_status` to one of: `approved`, `rejected`, `needs_followup`.
   - Write free-form `reviewer_notes` if the decision needs context.
   - **Only if approved AND you want it promoted:** set `promote_to_canon` to `yes`. Leave blank otherwise.
3. Save the workbook. Keep the file — a future promotion tool will read it and produce canon entries.

## Review heuristics (suggested, not rules)

| Candidate type | Default disposition | Watch out for |
|---|---|---|
| **fact** with supporting_count ≥ 3 | Likely approve | Personal / one-shot details that shouldn't be canonicalized |
| **fact** with supporting_count = 1 | Marked as hypothesis; verify before approving | The founder-prep M4 / current-vs-future drift pattern also applies here — many "facts" are intended designs, not today's truth |
| **decision** with supporting_count ≥ 2 | Likely approve | One-off experiments the founder didn't follow through on |
| **decision** with supporting_count = 1 | Marked as hypothesis; verify | Same as above |
| **hypothesis** | Needs verification — `approved` here means "worth tracking," not "canon-ready" | Promoting a hypothesis to canon = skipping verification; resist |
| **obsolete** | Usually `rejected` (retire) unless the reviewer believes the newer claim is wrong | Read the `reviewer_notes` field carefully; it names the superseding candidate |

## What the pipeline deliberately did NOT do

- Did NOT use an LLM to judge meaning or quality.
- Did NOT mutate any repo file outside `tools/cognitive_foundry/claude_lane/output/`.
- Did NOT write to MEMORY.md, Node0 runtime files, or any PR-in-flight branches.
- Did NOT set `promote_to_canon=yes` on any row.
- Did NOT assume one Claude conversation = one topic. Conversations can span buckets.

## What the next step should be

1. Operator does the spreadsheet review described above.
2. A separate `promote.py` tool (not yet implemented) reads the annotated workbook and produces structured canon entries ready for explicit inclusion in MEMORY.md or an equivalent persistent store — with a final confirmation gate.

Until both of those happen, the candidates stay candidates.
