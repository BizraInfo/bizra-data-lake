# Corpus Attestation Report

- Generated at: `2026-02-21T04:51:05.511134+00:00`
- Manifest hash: `504145f781412a4103249f78f46d61609eb1d02f81a1c2fa2f051184b23c6e09`
- Providers covered: `chatgpt_openai, claude, gemini_google, deepseek, qwen, kimi, perplexity, zhipu`
- Raw conversations: `2153`
- Unique conversations: `1549`
- Raw messages: `83242`
- Unique messages: `31968`
- Duplication factor: `2.603916`

## Derived Metrics (Mathematical)

Let:
- `P = providers_covered_count`
- `C_raw = raw_conversations`
- `C_unique = unique_conversations`
- `M_raw = raw_messages`
- `M_unique = unique_messages`

Computed:
1. `core8_coverage_ratio = P/8 = 8/8 = 1.0000`
2. `unique_conversation_ratio = C_unique/C_raw = 1549/2153 = 0.719461`
3. `unique_message_ratio = M_unique/M_raw = 31968/83242 = 0.384037`
4. `duplicate_message_rate = 1 - unique_message_ratio = 0.615963`
5. `duplication_factor = M_raw/M_unique = 83242/31968 = 2.603916`

## Evidence

1. Dedup report: `/mnt/c/BIZRA-DATA-LAKE/artifacts/corpus/v1/dedup_report.v1.json`
2. Canonical records: `/mnt/c/BIZRA-DATA-LAKE/artifacts/corpus/v1/core8_records.jsonl`
3. Manifest: `/mnt/c/BIZRA-DATA-LAKE/artifacts/corpus/v1/corpus_manifest.v1.json`

## Uncertainty Notes

1. Provider parser coverage is Core-8 best effort and evolves as export formats change.
2. Files that fail JSON parsing are skipped and counted implicitly by reduced discovered volume.
3. This is an internal truth artifact and not an external audit statement.

## Duplicate Cluster Summary

- Duplicate clusters: `32804`
