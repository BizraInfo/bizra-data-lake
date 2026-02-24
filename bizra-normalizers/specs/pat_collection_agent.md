# PAT Data Collection Agent — Specification
# Status: PLANNED (future milestone)
# Priority: Required for CV=1.0 against full CORE8
# Created: 2026-02-22

## Problem

Not all AI platforms provide native data export functionality.
As of 2025-12, the following platforms have NO download/export option:

| Platform    | Export Status          | Collection Strategy           |
|-------------|------------------------|-------------------------------|
| Perplexity  | No export API/UI       | Browser scrape via MCP/PAT    |
| ChatGPT WS  | Team workspace blocked | API key or browser scrape     |
| Google Chat | Partial (Takeout gaps) | Takeout + browser supplement  |

Platforms with confirmed native export:
chatgpt (personal), claude, deepseek, gemini (Takeout), qwen, kimi, zhipu

## Architecture

The PAT (Personal Agent Tool) collection agent operates as a
BIZRA-native function that:

1. Authenticates to target platform via browser session (MCP web agent)
2. Navigates conversation history UI
3. Extracts turn-level data (role, content, timestamp, model)
4. Emits normalized JSONL in the same schema as manual exports
5. Deduplicates against existing 00_INTAKE corpus

## Output Contract

Each collected conversation MUST produce a JSON file matching the
platform's parser schema expectations:

```json
{
  "model": "perplexity-sonar-v2",
  "query": "<user_query>",
  "answer": "<assistant_response>",
  "citations": [{"url": "..."}],
  "created_at": "<ISO8601>"
}
```

Or for multi-turn:

```json
{
  "model": "perplexity-sonar-v2",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```


## Implementation Phases

### Phase 1: Perplexity Collector (highest priority)
- Use MCP browser agent to navigate perplexity.ai/library
- Extract all conversation threads with timestamps
- Parse inline citations and source URLs
- Emit one JSON file per thread into 00_INTAKE/perplexity-export/

### Phase 2: ChatGPT Workspace Collector
- Handle Team/Workspace accounts that block Settings > Export
- Navigate conversation list via sidebar DOM
- Extract mapping-style conversation data
- Merge with existing personal export (deduplicate by conversation_id)

### Phase 3: Google Chat Gap Filler
- Supplement Google Takeout with browser-scraped threads
- Focus on threads NOT present in existing Takeout archives
- Respect existing gemini parser schema

## Integration Points

- Output lands in 00_INTAKE/<platform>-export/ directories
- preflight_readiness.py validates new data is schema-detected
- compile_stereoscopic_graph.py picks up new providers automatically
- COLLECTION_GAP constant in normalizers/__init__.py updated as
  platforms are collected
- Gate advisory INFO:COLLECTION_GAP reasons clear automatically

## Success Criteria

- perplexity appears in provider_coverage after PAT collection
- CV reaches 1.0 against full CORE8
- GENESIS gate passes with target_providers == CORE8
- All collected data passes schema-first detection (no filename fallback)

## Dependencies

- MCP browser agent (Claude in Chrome or equivalent)
- Authenticated browser sessions for target platforms
- Rate limiting / politeness controls to avoid account flags
- Deduplication logic against existing corpus

## Notes

This is a User Zero personal data collection tool, not a general
scraping framework. It operates exclusively on the user's own
conversation history across platforms they have authenticated access to.
