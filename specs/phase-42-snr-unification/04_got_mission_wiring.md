# 04: Activate GoT Reasoning in Mission Execution

## Standing on Giants
Besta (Graph-of-Thoughts, 2024) · Vaswani (attention, 2017) · Boyd (OODA loop, 1976)

## Problem

The `TRUE SPEARPOINT` bridge exists (`core/sovereign/graph_reasoning.py`):
- `_llm_generate()` calls `InferenceGateway.infer()` for real LLM hypotheses
- `_generate_hypothesis_contents()` uses LLM when gateway wired, templates when not
- `_formulate_conclusion_via_llm()` synthesizes via LLM
- `SovereignRuntime` wires `InferenceGateway` into `GraphOfThoughts` post-hoc

But `node0_activate.py` bypasses all of this:
- Direct `httpx.AsyncClient.post()` to LM Studio `/v1/chat/completions`
- No hypothesis generation, no multi-path reasoning, no synthesis
- PAT agents called sequentially with independent prompts
- No thought graph, no convergence analysis

## Design Decision: Incremental Activation

Wire GoT into the **synthesis phase** of mission execution — after PAT agents generate their independent analyses, GoT synthesizes them into a coherent conclusion. This preserves the existing PAT agent architecture while adding genuine reasoning.

**NOT replacing** PAT agent calls with GoT (that would be a larger refactor). Instead, GoT operates on agent outputs as its input evidence.

## Architecture

```
Mission: "Explain the Sovereign Empowerment Loop"
    │
    ▼
┌─────────────────────────────────────┐
│ Phase 1: PAT Agent Execution        │  (EXISTING — unchanged)
│  strategist → LM Studio (direct)    │
│  guardian   → LM Studio (direct)    │
│  coordinator → LM Studio (direct)   │
└─────────────────┬───────────────────┘
                  │ 3 independent analyses
                  ▼
┌─────────────────────────────────────┐
│ Phase 2: GoT Synthesis (NEW)        │
│  Input: 3 agent outputs + mission   │
│  1. Generate hypotheses from agents │
│  2. Score each path (SNR)           │
│  3. Merge strongest paths           │
│  4. Formulate unified conclusion    │
│  Output: synthesized answer + graph │
└─────────────────┬───────────────────┘
                  │ thought chain + conclusion
                  ▼
┌─────────────────────────────────────┐
│ Phase 3: SNR Measurement (EXISTING) │
│  SNRFacade on GoT conclusion        │
│  Ensemble: v2 + maximizer           │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ Phase 4: Receipt (EXISTING)         │
│  Evidence ledger + hash chain       │
│  Includes thought_chain metadata    │
└─────────────────────────────────────┘
```

## Pseudocode

### 4A: Initialize GoT in Kernel

```
# In node0_activate.py, after SNR facade init:

FUNCTION _init_got_engine(base_url, token) -> Optional[GraphOfThoughts]:
    """Initialize GoT with InferenceGateway for LLM-backed reasoning."""
    TRY:
        from core.inference.gateway import InferenceGateway
        from core.sovereign.graph_core import GraphOfThoughts

        # Create gateway pointing to LM Studio
        gateway = InferenceGateway(
            endpoints = [
                {"url": base_url, "tier": "local", "token": token},
            ],
        )

        got = GraphOfThoughts(
            strategy = ReasoningStrategy.BEST_FIRST,
            max_depth = 5,
            beam_width = 3,
            inference_gateway = gateway,
        )

        logger.info(f"  GoT engine: initialized (LLM={got._has_llm})")
        RETURN got

    EXCEPT Exception as e:
        logger.warning(f"  GoT engine: unavailable ({e})")
        RETURN None

_GOT_ENGINE = _init_got_engine(base_url, token)
```

### 4B: GoT Synthesis After PAT Agents

```
ASYNC FUNCTION _synthesize_with_got(
    mission_desc: str,
    agent_results: list[dict],     # [{agent_id, content, model, tokens}]
    got_engine: GraphOfThoughts,
) -> dict:
    """Use Graph-of-Thoughts to synthesize PAT agent outputs.

    Returns:
        {
            "conclusion": str,           # Synthesized answer
            "thought_count": int,        # Nodes in graph
            "reasoning_paths": int,      # Branches explored
            "snr_score": float,          # GoT-internal SNR
            "thought_chain": list[dict], # For receipt metadata
        }
    """

    # Convert agent outputs to GoT-compatible facts
    facts = []
    FOR result IN agent_results:
        facts.append(f"[{result['agent_id']}]: {result['content']}")

    # Run GoT reasoning with LLM-backed synthesis
    TRY:
        reasoning_result = AWAIT got_engine.reason(
            query = mission_desc,
            facts = facts,
            domain = "mission_synthesis",
            max_depth = 3,          # Shallow — agents already did deep work
        )

        RETURN {
            "conclusion": reasoning_result.conclusion,
            "thought_count": reasoning_result.thought_count,
            "reasoning_paths": reasoning_result.paths_explored,
            "snr_score": reasoning_result.best_snr,
            "thought_chain": [
                {
                    "id": t.id,
                    "type": t.thought_type.value,
                    "snr": t.snr_score,
                    "ihsan": t.ihsan_score,
                    "depth": t.depth,
                }
                FOR t IN reasoning_result.thoughts
            ],
        }

    EXCEPT Exception as e:
        logger.warning(f"GoT synthesis failed: {e}")
        # Fallback: simple concatenation (current behavior)
        RETURN {
            "conclusion": "\n\n".join(r["content"] FOR r IN agent_results),
            "thought_count": 0,
            "reasoning_paths": 0,
            "snr_score": 0.0,
            "thought_chain": [],
        }
```

### 4C: Wire into _execute_mission()

```
ASYNC FUNCTION _execute_mission(mission, agents) -> dict:
    # --- Phase 1: PAT agent execution (UNCHANGED) ---
    agent_results = []
    FOR agent_id IN agents:
        result = AWAIT _call_agent(agent_id, mission)
        agent_results.append(result)

    # --- Phase 2: GoT synthesis (NEW) ---
    got_synthesis = None
    IF _GOT_ENGINE is not None:
        got_synthesis = AWAIT _synthesize_with_got(
            mission["description"],
            agent_results,
            _GOT_ENGINE,
        )
        logger.info(
            f"GoT synthesis: {got_synthesis['thought_count']} thoughts, "
            f"{got_synthesis['reasoning_paths']} paths, "
            f"SNR={got_synthesis['snr_score']:.3f}"
        )

    # --- Phase 3: SNR measurement (UPDATED — use GoT conclusion if available) ---
    IF got_synthesis AND got_synthesis["conclusion"]:
        snr_text = got_synthesis["conclusion"]
    ELSE:
        snr_text = "\n\n".join(r["content"] FOR r IN agent_results)

    snr_data = _compute_mission_snr(mission["description"], [snr_text])

    # --- Phase 4: Receipt (UPDATED — include thought chain) ---
    receipt = {
        "mission_id": mission_id,
        "agents": agent_results,
        "snr": snr_data,
        "got": {
            "active": got_synthesis is not None,
            "thought_count": got_synthesis["thought_count"] IF got_synthesis ELSE 0,
            "reasoning_paths": got_synthesis["reasoning_paths"] IF got_synthesis ELSE 0,
            "thought_chain": got_synthesis["thought_chain"] IF got_synthesis ELSE [],
        },
        "conclusion": snr_text,
    }

    RETURN receipt
```

## Edge Cases

1. **No InferenceGateway available**: GoT initializes with `_has_llm=False`. Synthesis uses template fallback. Templates produce lower SNR — this is correct (templates are noise, not signal).

2. **LLM call fails during synthesis**: `_synthesize_with_got()` catches exceptions and falls back to concatenation. Mission still completes — GoT is additive, not blocking.

3. **GoT takes too long**: Set `max_depth=3` (shallow) and `beam_width=3` (narrow). With 3 agent outputs as facts, GoT generates ~5-10 thoughts in ~10 seconds. Timeout at 60s with fallback.

4. **Agent outputs are garbage**: GoT synthesis on garbage inputs produces garbage output — but the SNR measurement on the GoT conclusion will correctly score it low. The gate still gates.

## TDD Anchors

```python
# test_got_mission_wiring.py

@pytest.mark.requires_ollama
async def test_got_synthesis_produces_conclusion():
    """GoT synthesis with LLM generates a non-template conclusion."""
    got = _init_got_engine("http://192.168.56.1:1234", token)
    if got is None or not got._has_llm:
        pytest.skip("No LLM available")
    result = await _synthesize_with_got(
        "Explain signal processing",
        [{"agent_id": "strategist", "content": "Signal processing is..."}],
        got,
    )
    assert result["thought_count"] > 0
    assert len(result["conclusion"]) > 50

def test_got_synthesis_fallback_without_llm():
    """Without LLM, GoT synthesis falls back to concatenation."""
    got = GraphOfThoughts()  # No inference_gateway
    result = asyncio.run(_synthesize_with_got(
        "test",
        [{"agent_id": "a", "content": "text A"},
         {"agent_id": "b", "content": "text B"}],
        got,
    ))
    assert "text A" in result["conclusion"]
    assert "text B" in result["conclusion"]

def test_got_failure_does_not_block_mission():
    """If GoT raises, mission still completes with concatenated output."""
    # Mock GoT that raises
    result = asyncio.run(_synthesize_with_got(
        "test", [{"agent_id": "a", "content": "text"}], None
    ))
    # Should return fallback, not raise
    assert result["thought_count"] == 0

def test_receipt_includes_thought_chain():
    """Mission receipt contains got.thought_chain metadata."""
    # After a mission with GoT active
    assert "got" in receipt
    assert "thought_chain" in receipt["got"]
    assert isinstance(receipt["got"]["thought_chain"], list)
```

## Files Modified

- `scripts/node0_activate.py` — Add `_init_got_engine()`, `_synthesize_with_got()`, wire into `_execute_mission()`

## Files NOT Modified

- `core/sovereign/graph_core.py` — Already accepts `inference_gateway`
- `core/sovereign/graph_reasoning.py` — TRUE SPEARPOINT bridge already works
- `core/inference/gateway.py` — Already production-ready

## Performance Budget

| Phase | Current | With GoT | Delta |
|-------|---------|----------|-------|
| PAT agents (3x) | ~6 min | ~6 min | 0 |
| GoT synthesis | 0 | ~15 sec | +15 sec |
| SNR measurement | ~50 ms | ~55 ms | +5 ms |
| Receipt | ~5 ms | ~5 ms | 0 |
| **Total** | ~6 min | ~6.25 min | **+2.5%** |

Acceptable trade-off for genuine multi-path reasoning.

## Risk Assessment

- **Blast radius**: Medium — changes to mission execution affect all missions
- **Reversibility**: High — `_GOT_ENGINE = None` disables GoT entirely
- **Failure mode**: Graceful — GoT failure falls back to existing behavior
- **LLM dependency**: GoT adds 1 additional LLM call (synthesis). If LM Studio is slow, this adds latency. Mitigated by `max_depth=3` and 60s timeout.
