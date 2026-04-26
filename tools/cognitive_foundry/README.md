# Cognitive Foundry

Isolated toolchain for processing cognitive archives (Claude, OpenAI, Gemini export dumps) into reviewable knowledge candidates.

**Isolation rules:**
- Does NOT touch Node0 runtime code under `core/` or `bizra-omega/`.
- Does NOT read or write `MEMORY.md`, `receipt-lineage` WIP, or any PR-in-flight files.
- Does NOT auto-promote anything to canon. Every promotion requires a human `promote_to_canon=yes` in a reviewed workbook.
- Outputs stay inside `tools/cognitive_foundry/<lane>/output/` and are gitignorable.

## Lanes

| Lane | Status | Scope |
|---|---|---|
| `claude_lane/` | Pilot implementation | Processes a Claude export zip (users.json, projects.json, memories.json, conversations.json) through 4 stages. |
| `openai_lane/` | Not yet implemented | Future — will mirror claude_lane's stage contract with OpenAI's export format. |
| `gemini_lane/` | Not yet implemented | Future — same stage contract, Gemini's export format. |

## Stage contract (shared across all future lanes)

Every lane must produce these stage outputs, in this order, under `output/<run_id>/`:

1. **Stage 1 — Inventory** (`01_inventory/`) — source registry + topic bucket counts + top-signal sessions.
2. **Stage 2 — Distillation** (`02_distillation/`) — fact / decision / contradiction / reasoning-exemplar candidates.
3. **Stage 3 — Adjudication** (`03_adjudication/`) — canonical candidates + hypothesis / obsolete candidates + cluster registry.
4. **Stage 4 — Review Pack** (`04_review_pack/`) — reviewer-ready CSV workbook + human review brief markdown.

All intermediate artifacts are CSV + JSON (no binaries). Provenance (source conversation uuid, source message uuid, timestamps) is preserved in every candidate row.

## Run (pilot — claude_lane)

```bash
python tools/cognitive_foundry/claude_lane/run_pipeline.py \
  --archive /absolute/path/to/claude-export.zip \
  --output-dir tools/cognitive_foundry/claude_lane/output
```

See `claude_lane/README.md` for details, schemas, assumptions, and review instructions.

## Canon separation

The foundry produces **candidates**. It does not produce **canon**. The only path from candidate to canon is:

1. Operator opens `04_review_pack/review_workbook.csv` in a spreadsheet tool.
2. Operator reads each candidate row, sets `review_status` = `approved` / `rejected` / `needs_followup`.
3. For approved rows, operator sets `promote_to_canon` = `yes`.
4. Operator runs a separate — not-yet-implemented — `promote.py` tool that takes an annotated workbook and writes MEMORY.md entries (or equivalent).

The pilot implements steps 1–2 preparation; steps 3–4 require human judgment and an explicit future tool.
