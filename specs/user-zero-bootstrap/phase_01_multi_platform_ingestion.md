# Phase 01: Multi-Platform Conversation Ingestion

**Status:** SPEC | **Dependencies:** None (entry point)
**Goal:** Ingest conversation exports from 10+ AI platforms into a unified schema,
deduplicate cross-platform, enrich metadata, and produce a single Parquet table
ready for Phase 02 compilation fuel extraction.

---

## 1. Unified Conversation Schema

```pseudocode
ConversationTurn:
  id:              BLAKE3(platform + conversation_id + turn_index)
  platform:        enum { ChatGPT, OpenAI_API, Gemini, DeepSeek, Perplexity,
                          Qwen, Kimi, Zhipu, Claude, Grok }
  conversation_id: string
  turn_index:      u32
  role:            enum { User, Assistant, System, Tool }
  content:         string
  model:           string | null
  timestamp:       ISO8601 | null
  metadata:        Map<string, Value>
  token_count:     u32 | null
  embedding:       Vec<f32> [384]         # all-MiniLM-L6-v2, post-ingestion
```

ID hashing follows the domain-separated pattern in
`bizra-omega/bizra-agent/src/hash_namespace.rs` (lines 8-11: `TRIGGER_DOMAIN`,
`ACTION_DOMAIN`, etc.). Ingestion domain: `"genesis/conversation/v1"`.

```pseudocode
fn conversation_turn_id(platform: string, conv_id: string, index: u32) -> [u8; 32]:
  hasher = blake3::Hasher::new_derive_key("genesis/conversation/v1")
  hasher.update(platform.as_bytes())
  hasher.update(conv_id.as_bytes())
  hasher.update(index.to_le_bytes())
  return hasher.finalize()
```

---

## 2. PlatformParser Trait

```pseudocode
trait PlatformParser:
  fn platform_name() -> Platform
  fn detect(raw_bytes: &[u8]) -> bool           # first 4096 bytes, auto-detect
  fn parse(input: File | JSON) -> Vec<ConversationTurn>  # streaming for large exports
  fn normalize_timestamp(raw: string) -> ISO8601
  fn extract_model_name(raw: JSON) -> string | null
```

Parsers never panic; malformed records produce warnings and skip.
Reference: `ingest_conversations.py:extract_turns()` (line 76) for the
ChatGPT `mapping` traversal pattern.

---

## 3. Platform Format Specifications

| Platform | Format | Key Path | Timestamp | Model Field | Gotchas |
|----------|--------|----------|-----------|-------------|---------|
| **ChatGPT** | JSON | `mapping.{id}.message.content.parts[0]` | Unix epoch float | `metadata.model_slug` | Tree-structured mapping; linearize via `children` pointers. Some `parts` are dicts (images). Ref: `ingest_conversations.py` line 76-100. |
| **OpenAI API** | JSONL | `messages[i].role/content` | File mtime (no native) | Top-level `model` | Includes system prompts; keep as `role=System`. |
| **Gemini** | Google Takeout JSON | `chunks[i].parts[j].text` | RFC 3339 `createTime` | Infer or default `gemini-pro` | Author is numeric (0=user, 1=model). |
| **DeepSeek** | JSON | `messages[i].role/content` | ISO8601 or epoch `created_at` | `model` | Separate `reasoning_content` for CoT; capture as metadata. |
| **Perplexity** | Markdown + YAML | `## Query` / `## Answer` sections | YAML `date` (YYYY-MM-DD) | Default `perplexity` | Strip citation markers `[1]`; preserve in metadata. |
| **Qwen** | JSON | `messages[i].role/content` | Millisecond epoch | `model_id` | Some exports use GB2312 encoding; detect and convert. |
| **Kimi** | Moonshot JSON | `messages[i].role/content` | Unix epoch `created_at` | `model` | 128K context; chunk if content > 8000 chars. |
| **Zhipu** | ChatGLM JSON | `messages[i].role/content` | Unix epoch `created` | `model` | Tool-use responses have `tool_calls[]`. |
| **Claude** | JSON | `chat_messages[i].sender/text` | ISO8601 `created_at` | `model` | Map `"human"` to `User`. Attachments in metadata. |
| **Grok** | JSON/CSV | `messages[i].role/content` | ISO8601 `timestamp` | `model` | CSV loses threading; prefer JSON. Tweet context as system msgs. |

---

## 4. Deduplication Pipeline

```pseudocode
DEDUP_DOMAIN = "genesis/dedup/v1"

fn deduplicate(turns: Vec<ConversationTurn>) -> Vec<ConversationTurn>:
  seen: HashMap<[u8; 32], ConversationTurn> = {}

  for turn in turns:
    normalized = turn.content.trim().lowercase().collapse_whitespace()
    content_hash = blake3::derive_key(DEDUP_DOMAIN, normalized.as_bytes())

    if content_hash not in seen:
      seen[content_hash] = turn
    else:
      existing = seen[content_hash]
      existing.metadata["duplicate_platforms"].push(turn.platform)
      existing.metadata["duplicate_ids"].push(turn.id)
      log::info("duplicate: {} ({}) matches {} ({})",
                turn.id, turn.platform, existing.id, existing.platform)

  return seen.values().sorted_by(|a, b| a.timestamp.cmp(b.timestamp))
```

Reference: `hash_namespace.rs` lines 8-11 for domain-separation convention.
The `collapse_whitespace()` normalization ensures formatting differences across
platforms do not create false duplicates.

---

## 5. Metadata Enrichment

### Timestamp Normalization

```pseudocode
fn normalize_timestamp(raw: string, platform: Platform) -> ISO8601:
  match platform:
    ChatGPT     -> datetime.fromtimestamp(float(raw), UTC)
    OpenAI_API  -> file_mtime                          # no native timestamp
    Gemini      -> parse_rfc3339(raw)
    DeepSeek    -> parse_iso_or_epoch(raw)
    Perplexity  -> parse_date(raw) + "T00:00:00Z"      # date-only
    Qwen        -> datetime.fromtimestamp(int(raw)/1000, UTC)
    Kimi|Zhipu  -> datetime.fromtimestamp(int(raw), UTC)
    Claude|Grok -> parse_iso8601(raw)
```

### Token Counting

```pseudocode
fn estimate_tokens(content: string, model: string | null) -> u32:
  if model starts with "gpt" or "o1" or "o3":
    return tiktoken.encode(content, model).len()    # exact
  elif contains_cjk(content):
    return content.len() / 2                         # CJK heuristic
  else:
    return content.len() / 4                         # English heuristic
```

### Language Detection and Topic Extraction

- **Language:** fasttext `lid.176.bin` (production) or langdetect (dev). Returns `(ISO639, confidence)`.
- **Topics:** RAKE keyword extraction, max 5 per turn. Full LDA deferred to Phase 02.

---

## 6. Output Schema

```pseudocode
conversations_unified.parquet:
  columns: [id, platform, conversation_id, turn_index, role, content,
            model, timestamp, metadata_json, token_count, content_hash,
            language, language_conf, topics_json, embedding]
  partitioned_by: [platform]
  compression: zstd
  row_group_size: 65536

Index files:
  platform_index.json     # platform -> [conversation_ids]
  timeline_index.json     # month -> [turn_ids]
  dedup_manifest.json     # content_hash -> {canonical, duplicates}
  ingestion_report.json   # summary stats (counts, date range, languages)
```

Reference: `corpus_manager.py` (lines 9-17) for BLAKE3/SHA-256 content hashing
pattern. `ingest_conversations.py` (lines 45-48) for FAISS index output pattern.

---

## 7. Data Flow Diagram

```
  Platform Exports (10+ formats)
  [ChatGPT JSON] [Gemini Takeout] [Claude JSON] [DeepSeek JSON] ...
       |
       v
  +-------------------------+
  |   Format Detection      | <-- PlatformParser.detect() on first 4096 bytes
  +-------------------------+
       |
       v
  +-------------------------+
  |   Platform Parsers      | <-- PlatformParser.parse(), streaming
  |   (10 implementations)  |
  +-------------------------+
       |
       v
  +-------------------------+
  |   Unified Schema        | --> ConversationTurn (one shape)
  +-------------------------+
       |
       v
  +-------------------------+
  |   Deduplication         | <-- BLAKE3 content hash, genesis/dedup/v1
  +-------------------------+
       |
       v
  +-------------------------+
  |   Metadata Enrichment   | <-- timestamps, tokens, language, topics
  +-------------------------+
       |
       v
  +-------------------------+
  |   Embedding Generation  | <-- all-MiniLM-L6-v2 (384-dim), GPU batch
  +-------------------------+
       |
       v
  conversations_unified.parquet + index files
```

---

## 8. TDD Anchors

Tests marked [PROP] are property-based. Tests marked [INT] require filesystem access.

```pseudocode
# --- Parser Tests ---

test_chatgpt_parser_standard_export:
  turns = ChatGPTParser.parse(load("fixtures/chatgpt_standard.json"))
  assert turns.len() > 0
  assert all(t.platform == ChatGPT and t.role in {User, Assistant} for t in turns)

test_chatgpt_parser_nested_mapping:
  turns = ChatGPTParser.parse(load("fixtures/chatgpt_branched.json"))
  assert no_duplicate_turn_indices(turns)   # linearization picks final branch

test_openai_api_jsonl_parser:
  turns = OpenAIAPIParser.parse(load("fixtures/openai_api_log.jsonl"))
  assert all(t.model is not null for t in turns)

test_gemini_takeout_parser:
  turns = GeminiParser.parse(load("fixtures/gemini_takeout/conv_001.json"))
  assert all(t.timestamp is not null for t in turns)

test_claude_parser:
  turns = ClaudeParser.parse(load("fixtures/claude_export.json"))
  assert "human" not in [t.role for t in turns]   # mapped to User

test_platform_autodetect:
  for fixture in all_fixtures():
    assert detect_platform(fixture.bytes[:4096]) == fixture.expected_platform

# --- Deduplication Tests ---

test_dedup_identical_content:
  a = ConversationTurn(platform=ChatGPT, content="Hello world")
  b = ConversationTurn(platform=Claude,  content="Hello world")
  result = deduplicate([a, b])
  assert result.len() == 1
  assert "Claude" in result[0].metadata["duplicate_platforms"]

test_dedup_whitespace_normalization:
  a = ConversationTurn(content="Hello   world\n\n")
  b = ConversationTurn(content="Hello world")
  assert deduplicate([a, b]).len() == 1

[PROP] test_dedup_idempotent:
  forall turns: assert deduplicate(deduplicate(turns)) == deduplicate(turns)

# --- Metadata Tests ---

test_timestamp_normalization:
  assert normalize("1702646400.123", ChatGPT) == "2023-12-15T12:00:00.123Z"
  assert normalize("1702646400123", Qwen)     == "2023-12-15T12:00:00.123Z"
  assert normalize("2024-01-15T10:30:00Z", Claude) == "2024-01-15T10:30:00Z"

test_token_count_estimation:
  content = "The quick brown fox jumps over the lazy dog."
  exact = tiktoken.encode(content, "gpt-4").len()
  assert estimate_tokens(content, "gpt-4") == exact
  assert abs(estimate_tokens(content, "gemini-pro") - exact) / exact < 0.10

# --- Schema Tests ---

test_unified_parquet_schema:
  df = read_parquet(run_ingestion("fixtures/"))
  assert "embedding" in df.columns and df["embedding"][0].len() == 384

[INT] test_large_export_streaming:
  result = run_ingestion(generate_synthetic_export(100_000), max_memory_gb=8)
  assert result.turn_count == 100_000 and result.peak_memory_gb < 8

test_empty_conversation_handling:
  assert ChatGPTParser.parse({"mapping": {}}).len() == 0
```

---

## 9. Open Questions

1. **Archive vs live polling.** Phase 01 = static exports only. Live polling deferred.
2. **PII detection.** For User Zero: flag but do not block. Multi-user: hard gate.
3. **Storage budget.** ~3-5GB total (raw + Parquet + FAISS) for 7000+ conversations.
4. **Incremental ingestion.** Idempotent merge using content hashes; re-runs skip known turns.
5. **Platform priority.** ChatGPT (largest) > Claude > DeepSeek > Gemini > rest.
