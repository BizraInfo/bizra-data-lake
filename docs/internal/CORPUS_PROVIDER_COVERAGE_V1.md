# Corpus Provider Coverage v1

## Locked Core-8 Providers
1. `chatgpt_openai`
2. `claude`
3. `gemini_google`
4. `deepseek`
5. `qwen`
6. `kimi`
7. `perplexity`
8. `zhipu`

## Current Measured Coverage (2026-02-21)
Detected in current manifest:
1. `chatgpt_openai`
2. `claude`
3. `gemini_google`
4. `deepseek`
5. `perplexity`

Missing vs Core-8 target:
1. `qwen`
2. `kimi`
3. `zhipu`

Coverage equation:
```text
coverage_ratio = detected_core8 / 8 = 5/8 = 0.6250
```

## Normalizer Readiness (2026-02-21)

All 8 Core-8 normalizers are **implemented and tested** (31/31 tests pass).
Coverage gap is data availability, not parser capability.

| Provider | Detection | Parsing | Tested Formats | Status |
|----------|-----------|---------|----------------|--------|
| `chatgpt_openai` | path + signal | mapping + parts | mapping/author/parts | data present |
| `claude` | path + signal | chat_messages | sender/text | data present |
| `gemini_google` | path + signal | messages | role/content (model=assistant) | data present |
| `deepseek` | path + signal | mapping + fragments | REQUEST/RESPONSE/THINK/SEARCH | data present |
| `qwen` | path + signal | messages, history pairs, data envelope | messages, [[user,bot]], {data:{messages}} | **normalizer ready** |
| `kimi` | path + signal | messages, segments, items | messages, segments[], {items:[{messages}]} | **normalizer ready** |
| `perplexity` | path + signal | messages | role/content | data present |
| `zhipu` | path + signal | messages, choices, prompt/response, data envelope | messages, choices[].message, prompt+response, {data:{choices}} | **normalizer ready** |

## To Push Coverage to 8/8

Place conversation export JSON files in scan roots:
```
00_INTAKE/<provider-keyword>/    (e.g., 00_INTAKE/qwen-export/)
sovereign_state/chat_import/     (alternative root)
```

Then re-run the pipeline:
```bash
cd scripts/corpus
python dedup_core8.py
python build_corpus_manifest.py --attestation-out docs/internal/CORPUS_ATTESTATION_REPORT_2026-02-21.md
```

Each provider is auto-detected via:
1. Path keyword inference (directory/filename containing provider name).
2. JSON signal inference (`model`, `provider`, `platform` fields).
3. Structural shape inference (`mapping`, `segments`, `history`, `choices`).

## Coverage Policy
1. Core-8 remains mandatory for v1 completion.
2. Unknown exports may parse as `generic` but do not increase Core-8 coverage.
3. Coverage must be measured from generated manifest only.

## Detection Method
1. Path keyword inference.
2. JSON signal inference (model name, provider field, platform field).
3. JSON shape inference (`mapping`, `chat_messages`, `messages`, `segments`, `history`, `choices`, `items`, `prompt`/`response`).
4. Canonical record normalization into `CorpusRecord`.

## Known Limits
1. Export formats vary by provider version and region.
2. Some exports contain empty message arrays and produce zero records.
3. Non-JSON export formats are out of current parser scope.
4. Chinese field names (e.g., Qwen regional exports) require the standard JSON field fallbacks.
