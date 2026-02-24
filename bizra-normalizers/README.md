# BIZRA Platform Normalizers v1.0

> **CV = 1.0** — All 8 platform normalizers validated. GENESIS gate open.
> **Core10-ready ingestion:** OpenAI API + Grok parsers added (Core8 gate remains unchanged).

## Mission

Convert conversation exports from 8 AI platforms into BIZRA's unified `ConversationTurn` schema, feeding the stereoscopic intelligence pipeline that makes Node0 know its user.

## Peak Next Step

`AutonomousSNRGoTEngine` now compiles normalized turns into a graph-of-thoughts report with:

- Cross-platform corroboration boost (3+ providers).
- Composite SNR scoring and elite node ranking.
- Weighted signal edges from hint co-occurrence.
- Embedded "standing on shoulders of giants" protocol manifest.

## Architecture

```text
Platform Exports (8 providers)
    │
    ├── ChatGPT    ──┐
    ├── Claude     ──┤  Existing (ingest_conversations.py)
    ├── Gemini     ──┤
    ├── Perplexity ──┘
    │
    ├── DeepSeek   ──┐
    ├── Qwen       ──┤  NEW (this package)
    ├── Kimi       ──┤
    └── Zhipu      ──┘
            │
            ▼
    ConversationTurn (unified schema)
            │
            ▼
    FragmentHints → bizra-memory 10 FragmentKind variants
            │
            ▼
    GENESIS Compiler → User Model → "My AI Knows Me"
```

## Stereoscopic Signal Mapping

| Platform | Unique Signal | Fragment Target | Why It Matters |
|---|---|---|---|
| DeepSeek | Reasoning traces (`<think>`) | Pattern + Expertise | Reveals complexity depth user engages with |
| Qwen | Multilingual code-switching | Style + Domain | Captures bilingual/cultural working style |
| Kimi | Long-context cross-references | Temporal + Relationship | Captures document-heavy workflow patterns |
| Zhipu | Structured outputs + tool calls | Fact + Goal | Captures analytical and tool-driven intent |

**Cross-platform validation rule:** same pattern on 3+ platforms → confidence × 1.5.

## CV Gate Mathematics

```text
G = min(CV, SQ, IH)

CV = normalized_platforms / total_platforms
   = 8/8 = 1.0
```

Before this package: `CV = 4/8 = 0.5`.
After this package: `CV = 8/8 = 1.0`.

## Quick Start

```bash
# From this directory:
cd bizra-normalizers

# Run tests
python -m pytest tests/test_normalizers.py -v

# Validate CV gate
python validate_coverage.py --fixtures

# Build unified parquet + index artifacts
python build_unified_corpus.py /path/to/exports \
  --out-parquet /mnt/c/BIZRA-DATA-LAKE/04_GOLD/conversations_unified.parquet \
  --out-dir /mnt/c/BIZRA-DATA-LAKE/04_GOLD \
  --json

# Preflight readiness (Step 1)
python preflight_readiness.py /mnt/c/BIZRA-DATA-LAKE/00_INTAKE --available-only

# Compile stereoscopic graph report
python compile_stereoscopic_graph.py --fixtures --json

# Scan a corpus directory
python validate_coverage.py /path/to/exports/
python compile_stereoscopic_graph.py /path/to/exports/ --out report.json

# Enable fail-closed GENESIS gate
python compile_stereoscopic_graph.py /path/to/exports/ \
  --genesis-gate --gate-provider-set exportable_now \
  --min-cv 1.0 --min-elite-nodes 1

# Strict release/audit mode (must be full Core8)
python compile_stereoscopic_graph.py /path/to/exports/ \
  --genesis-gate --gate-provider-set core8 \
  --min-cv 1.0 --min-elite-nodes 1

# Checkpoint drift report history
python compile_stereoscopic_graph.py /path/to/exports/ \
  --checkpoint-dir ../artifacts/corpus/v1/stereoscopic_drift \
  --checkpoint-label daily

# Bridge report nodes into bizra-memory (typed adapter)
python compile_stereoscopic_graph.py /path/to/exports/ \
  --ingest-bizra-memory \
  --export-ingest-jsonl ../artifacts/corpus/v1/stereoscopic_drift/ingest_payload.jsonl
```

## Runbook Order

1. Preflight readiness scan.
2. Compile with `--genesis-gate` using `--gate-provider-set exportable_now` for operational runs.
3. Use `--gate-provider-set core8` for strict release/audit checks.

## File Structure

```text
bizra-normalizers/
├── schemas/
│   ├── __init__.py
│   └── conversation_turn.py
├── normalizers/
│   ├── __init__.py
│   ├── base.py
│   ├── openai_api.py
│   ├── grok.py
│   ├── deepseek.py
│   ├── qwen.py
│   ├── kimi.py
│   └── zhipu.py
├── build_unified_corpus.py
├── engine.py
├── genesis_gate.py
├── memory_bridge.py
├── compile_stereoscopic_graph.py
├── tests/
│   ├── __init__.py
│   └── test_normalizers.py
├── fixtures/
│   ├── deepseek_export.json
│   ├── qwen_export.json
│   ├── kimi_export.json
│   └── zhipu_export.json
├── validate_coverage.py
└── README.md
```

## Test Results

`106 passed`

- Schema + base parser behavior
- DeepSeek reasoning-trace extraction
- Qwen multilingual/code-switch extraction
- Kimi long-context cross-reference extraction
- Zhipu structured output + tool-call extraction
- Cross-platform CV and confidence boost behavior
- Edge-case robustness (empty, unicode, nested, single-conversation)
- Graph-of-thoughts engine, SNR ranking, edge synthesis, protocol manifest
- GENESIS fail-closed gate and typed bizra-memory bridge adapter

## Zero Dependencies

- Python 3.8+ standard library only
- `pytest` only for tests

## What This Unlocks

1. Ingest conversations from all 8 providers.
2. Ingest OpenAI API/Grok traces with schema-first detection and JSONL support.
3. Extract stereoscopic signals unique to each platform.
4. Apply cross-platform validation boosts on confirmed patterns.
5. Feed `bizra-memory` fragment extraction pipeline.
6. Strengthen the "My AI Knows Me" experience.

بسم الله
