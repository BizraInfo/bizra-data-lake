# Phase 58: First Heartbeat Optimization Sprint

> Standing on Giants: Amdahl (parallel speedup, 1967) · Shannon (information rate limits, 1948) · Boyd (OODA time compression, 1976)

## Premise

Level 2 First Heartbeat validated at 29.4s end-to-end. Target: < 5s.

## Measured Breakdown (2026-03-03T07:31 UTC)

```
Phase               Duration    % Total    Bottleneck?
─────────────────── ────────── ────────── ───────────
Auth + Boot         0.02s       0.1%      No
Memory Retrieve     0.03s       0.1%      No
Channel Decompose   0.01s       0.0%      No
Brave Search        1.30s       4.4%      No (network bound)
Ollama Synthesis   28.00s      95.2%      YES (cold model, CPU)
SNR Gate            0.01s       0.0%      No
Evidence Emit       0.02s       0.1%      No
Memory Store        0.01s       0.0%      No
Briefing Write      0.01s       0.0%      No
─────────────────── ────────── ──────────
TOTAL              29.41s      100%
```

Amdahl's Law: 95.2% of time is in LLM synthesis. Optimizing anything else
yields negligible improvement. The pipeline is not the bottleneck. The model is.

## Optimization Paths (ordered by expected impact)

### O1: GPU Inference via LM Studio (Target: 2-3s synthesis)

**Current**: Ollama phi3:mini on CPU → ~28s for 512 tokens
**Target**: LM Studio on RTX 4090 → 2-3s for 512 tokens

Action:
1. Load a quantized model in LM Studio (qwen2.5:7b-q4 or llama3.1:8b-q4)
2. Gateway already prefers LM Studio as primary → automatic routing
3. `use_native_api=False` fix already deployed → `/v1/chat/completions` works

Expected total: 1.3s (search) + 2.5s (GPU synthesis) + 0.1s (gates) = ~4s

Risk: LM Studio on Windows, Ollama in WSL — network hop adds ~50ms, negligible.

### O2: Prompt Compression (Target: 30-50% fewer input tokens)

Current synthesis prompt includes full search results (5 items with titles, URLs, snippets).
Most models can synthesize from 3 items with shorter snippets.

Action:
- Limit to top 3 search results in `_build_synthesis_prompt()`
- Truncate snippets to 100 chars
- Remove redundant instruction text

Expected impact: ~30% fewer input tokens → proportional speedup on generation.

### O3: Warm Model Pool (Target: eliminate cold start)

phi3:mini cold start: ~25s. Warm: ~5s.

Action:
- Send a 1-token warmup request on bridge startup
- Already partially done: `BIZRA_ENABLE_LLM=1` triggers gateway init
- Add explicit warmup in `MissionOrchestrator.initialize()`:
  ```
  await ollama_warmup("phi3:mini")  # Pre-load model weights
  ```

Expected impact: Eliminates 20s cold start penalty on first mission.

### O4: Streaming Synthesis (Target: perceived latency < 2s)

Even if synthesis takes 5s, streaming first tokens to the user within 500ms
changes perceived performance dramatically.

Action:
- Add `stream=True` to Ollama `/api/generate` call
- Yield first tokens to bridge immediately
- Bridge returns partial result with `status: STREAMING`
- AHK client shows incremental text in toast

Complexity: Medium. Requires bridge protocol change for streaming responses.

### O5: Parallel Channel Execution (Target: -1s)

Currently channels execute sequentially. Browser search (1.3s) blocks desktop context.

Action:
- `asyncio.gather()` for independent channels
- Already supported by ChannelDispatcher's dependency DAG

Expected impact: Browser and desktop execute in parallel, saving ~0.5-1s.

## Priority Order

| Priority | Optimization | Expected Gain | Effort |
|----------|-------------|---------------|--------|
| P0       | O1: GPU inference | 28s → 3s | Low (load model) |
| P1       | O3: Warm model | -20s cold start | Low (1 warmup call) |
| P2       | O2: Prompt compress | -30% tokens | Low (trim prompt) |
| P3       | O5: Parallel channels | -1s | Low (asyncio.gather) |
| P4       | O4: Streaming | Perceived < 2s | Medium (protocol) |

## Success Criteria

- Mission completes in < 5s (warm, GPU)
- Mission completes in < 15s (cold, CPU fallback)
- SNR remains ≥ 0.95 (no quality regression from prompt compression)
- All 96 existing tests remain green
- Briefing quality passes Daughter Test (human review)

## Non-Goals

- Cloud inference (sovereignty violation)
- Caching search results (freshness matters)
- Reducing evidence/memory overhead (already instant)
- Multi-model ensemble (premature complexity)
