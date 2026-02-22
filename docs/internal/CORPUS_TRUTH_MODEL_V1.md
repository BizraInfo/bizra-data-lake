# Corpus Truth Model v1

## Objective
Provide a deterministic, attestable corpus inventory for User Zero with reproducible dedup and measurable provider coverage.

## Canonical Artifacts
1. `schemas/corpus/corpus_record.v1.schema.json`
2. `schemas/corpus/dedup_report.v1.schema.json`
3. `schemas/corpus/corpus_manifest.v1.schema.json`
4. `scripts/corpus/provider_normalizers.py`
5. `scripts/corpus/dedup_core8.py`
6. `scripts/corpus/build_corpus_manifest.py`
7. `artifacts/corpus/v1/core8_records.jsonl`
8. `artifacts/corpus/v1/dedup_report.v1.json`
9. `artifacts/corpus/v1/corpus_manifest.v1.json`

## Canonical Record Contract
`CorpusRecord` required fields:
1. `provider`
2. `account_scope`
3. `conversation_id`
4. `message_id`
5. `role`
6. `timestamp`
7. `content_hash`
8. `source_path`
9. `import_run_id`

## Dedup Logic (Deterministic)
1. Primary key: `(provider, account_scope, conversation_id, message_id)`
2. Fallback key: `(role, normalized_text_hash, timestamp_bucket)`
3. Retention rule: lexical min over `(timestamp, source_path, message_id)`
4. Idempotence invariant: rerun with same inputs produces same kept IDs and same manifest hash.

## Mathematical Metrics
Given manifest values:
- `C_raw`, `C_unique`, `M_raw`, `M_unique`, and `P` (Core-8 detected providers)

Compute:
```text
unique_conversation_ratio = C_unique / C_raw
unique_message_ratio = M_unique / M_raw
duplicate_message_rate = 1 - unique_message_ratio
duplication_factor = M_raw / M_unique
core8_coverage_ratio = P / 8
```

## Baseline Update Rule
`sovereign_state/node0_baseline.json` corpus counts must be refreshed from generated manifest outputs only. Manual count edits are disallowed.

## Current Baseline Snapshot (2026-02-21)
1. `core8_coverage_ratio = 0.6250`
2. `duplication_factor = 2.605423`
3. `unique_message_ratio = 0.383815`

This is an internal evidence model, not an external audit artifact.
