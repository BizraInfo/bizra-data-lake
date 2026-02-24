# Phase 5: Validation Plan — End-to-End Verification

> ADR-004 | V3 Unified Swarm Coordination Engine
> Standing on Giants: Dijkstra (structured testing, 1972) · Hoare (pre/post conditions, 1969) · Deming (PDCA, 1950)

## 5.1 Test Pyramid

```
                    ┌─────────────┐
                    │  E2E Smoke  │  1 test — real LM Studio
                    │  (manual)   │
                    ├─────────────┤
                    │ Integration │  6 tests — cross-module
                    │  (mocked)   │
                    ├─────────────┤
                    │    Unit     │  25+ tests — fast, isolated
                    │             │
                    └─────────────┘
```

## 5.2 Unit Tests (Fast, No I/O)

All unit tests run without network, filesystem, or LM Studio access.

### `tests/core/swarm/test_types.py` (Phase 1)

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_agent_spec_from_pat_agent` | PAT_AGENTS dict → AgentSpec conversion |
| 2 | `test_agent_spec_from_pat_non_thinking` | Non-thinking model gets 30s timeout |
| 3 | `test_swarm_config_defaults` | Default values match constants.py |
| 4 | `test_swarm_event_immutable` | frozen=True prevents mutation |
| 5 | `test_agent_role_values` | All 7 roles are string enums |
| 6 | `test_swarm_topology_values` | 3 topologies are string enums |

### `tests/core/swarm/test_engine.py` (Phase 2)

| # | Test | Validates |
|---|------|-----------|
| 7 | `test_sequential_execution_order` | Agents run in order |
| 8 | `test_parallel_execution_concurrent` | Agents run concurrently |
| 9 | `test_parallel_bounded_by_semaphore` | Max concurrent respected |
| 10 | `test_hierarchical_mesh_coordinator_first` | Coordinator runs before workers |
| 11 | `test_hierarchical_mesh_synthesis_pass` | Coordinator gets second pass |
| 12 | `test_event_emission` | Phase transitions emit events |
| 13 | `test_preload_integration` | AutoModelRouter.preload called |
| 14 | `test_graceful_degradation_on_preload_failure` | Engine continues if pre-load fails |
| 15 | `test_agent_failure_does_not_halt_swarm` | One failure doesn't stop others |
| 16 | `test_result_flattening` | Exceptions from gather become dicts |
| 17 | `test_equalizer_integration` | Equalizer action included in result |

### `tests/core/swarm/test_event_bridge.py` (Phase 3)

| # | Test | Validates |
|---|------|-----------|
| 18 | `test_bridge_publishes_to_bus` | SwarmEvent → Event translation |
| 19 | `test_bridge_correlation_id` | swarm_id becomes correlation_id |
| 20 | `test_bridge_priority_mapping` | AGENT_FAILED → HIGH priority |
| 21 | `test_bridge_degrades_without_bus` | None bus → None bridge |
| 22 | `test_bridge_does_not_break_engine` | Bus failure doesn't crash engine |

### `tests/core/swarm/test_node0_migration.py` (Phase 4)

| # | Test | Validates |
|---|------|-----------|
| 23 | `test_legacy_path_when_disabled` | Feature flag off → old loop |
| 24 | `test_swarm_engine_path_when_enabled` | Feature flag on → SwarmEngine |
| 25 | `test_swarm_results_format_matches_legacy` | Output format compatibility |
| 26 | `test_topology_selection_from_config` | YAML config → topology |
| 27 | `test_topology_fallback_on_invalid` | Bad topology → SEQUENTIAL |
| 28 | `test_event_bus_wired_when_available` | Bridge created when bus exists |
| 29 | `test_call_fn_adapter` | AgentSpec → _call_agent mapping |

## 5.3 Integration Tests (Mocked HTTP)

These tests exercise multiple modules together but mock external HTTP calls.

```
# tests/integration/test_swarm_integration.py

TEST test_full_mission_through_swarm_engine():
    """Complete mission: types → engine → bridge → results."""
    # Setup
    bus = EventBus()
    config = SwarmConfig(topology=SwarmTopology.SEQUENTIAL)
    engine = SwarmEngine(config=config)
    bridge = wire_swarm_to_bus(engine, bus)

    bus_events = []
    bus.subscribe("swarm.*", LAMBDA evt: bus_events.append(evt))

    # Build agents from PAT_AGENTS format
    specs = [
        AgentSpec.from_pat_agent("strategist", PAT_AGENTS["strategist"]),
        AgentSpec.from_pat_agent("researcher", PAT_AGENTS["researcher"]),
    ]

    # Mock call function
    ASYNC FUNCTION mock_call(agent):
        RETURN {"agent": agent.id, "success": True, "text": "mock output"}

    # Execute
    result = AWAIT engine.execute_mission("m-test", specs, mock_call)

    # Verify results
    ASSERT len(result["results"]) == 2
    ASSERT all(r["success"] FOR r IN result["results"])
    ASSERT result["topology"] == "sequential"

    # Verify bus events were emitted
    AWAIT asyncio.sleep(0.05)  # Allow async bridge publish
    event_topics = [e.topic FOR e IN bus_events]
    ASSERT "swarm.swarm_created" IN event_topics
    ASSERT "swarm.mission_complete" IN event_topics


TEST test_parallel_topology_faster_than_sequential():
    """Parallel execution completes faster than sequential for slow agents."""
    ASYNC FUNCTION slow_call(agent):
        AWAIT asyncio.sleep(0.1)
        RETURN {"agent": agent.id, "success": True}

    agents = [make_spec(f"a{i}") FOR i IN range(5)]

    # Sequential
    seq_engine = SwarmEngine(config=SwarmConfig(topology=SEQUENTIAL))
    t0 = time.monotonic()
    AWAIT seq_engine.execute_mission("seq", agents, slow_call)
    seq_time = time.monotonic() - t0

    # Parallel
    par_engine = SwarmEngine(config=SwarmConfig(topology=PARALLEL, max_concurrent=5))
    t0 = time.monotonic()
    AWAIT par_engine.execute_mission("par", agents, slow_call)
    par_time = time.monotonic() - t0

    # Parallel should be significantly faster (5x agents × 0.1s = 0.5s vs ~0.1s)
    ASSERT par_time < seq_time * 0.5


TEST test_hierarchical_mesh_full_cycle():
    """Coordinator plans, workers execute, coordinator synthesizes."""
    call_log = []
    ASYNC FUNCTION tracked_call(agent):
        call_log.append(agent.id)
        RETURN {"agent": agent.id, "success": True}

    coordinator = AgentSpec(id="coord", role=AgentRole.COORDINATOR, ...)
    workers = [
        AgentSpec(id="w1", role=AgentRole.RESEARCHER, ...),
        AgentSpec(id="w2", role=AgentRole.ANALYST, ...),
    ]

    engine = SwarmEngine(config=SwarmConfig(topology=HIERARCHICAL_MESH))
    result = AWAIT engine.execute_mission("hm", [coordinator, *workers], tracked_call)

    # Coordinator should appear first and last (plan + synthesize)
    ASSERT call_log[0] == "coord"
    ASSERT call_log[-1] == "coord"
    # Workers should appear in between
    ASSERT set(call_log[1:-1]) == {"w1", "w2"}


TEST test_engine_with_auto_model_router():
    """SwarmEngine + AutoModelRouter pre-load integration."""
    mock_router = MockAutoModelRouter()
    engine = SwarmEngine(
        config=SwarmConfig(preload_models=True),
        model_router=mock_router,
    )

    AWAIT engine.execute_mission("m1", [agent], fake_call, model_config={})

    ASSERT mock_router.preload_called IS True
    ASSERT mock_router.preload_agent_ids IS NOT None


TEST test_swarm_engine_backward_compat_with_node0():
    """SwarmEngine result can be consumed by existing node0 scoring."""
    engine = SwarmEngine()
    result = AWAIT engine.execute_mission("m1", agents, fake_call)

    # node0_activate expects to iterate result["results"]
    # and access r.get("success"), r.get("text"), r.get("agent")
    FOR r IN result["results"]:
        ASSERT "success" IN r
        ASSERT isinstance(r["success"], bool)


TEST test_mixed_success_and_failure():
    """Some agents succeed, some fail — all results captured."""
    ASYNC FUNCTION mixed_call(agent):
        IF agent.id == "bad":
            RAISE Exception("model timeout")
        RETURN {"agent": agent.id, "success": True}

    agents = [
        AgentSpec(id="good1", ...),
        AgentSpec(id="bad", ...),
        AgentSpec(id="good2", ...),
    ]

    engine = SwarmEngine(config=SwarmConfig(topology=PARALLEL))
    result = AWAIT engine.execute_mission("mixed", agents, mixed_call)

    ASSERT len(result["results"]) == 3
    successes = [r FOR r IN result["results"] IF r.get("success")]
    failures = [r FOR r IN result["results"] IF NOT r.get("success")]
    ASSERT len(successes) == 2
    ASSERT len(failures) == 1
    ASSERT "model timeout" IN failures[0].get("error", "")
```

## 5.4 E2E Smoke Test (Manual, Requires LM Studio)

This test requires a running LM Studio instance with at least one loaded model.
It is NOT part of the CI suite — run manually during integration testing.

```bash
# Prerequisites:
# 1. LM Studio running at 192.168.56.1:1234
# 2. At least 1 model loaded
# 3. LM_API_TOKEN set

# Enable SwarmEngine
export SWARM_ENGINE_ENABLED=true

# Run a mission through the new path
python scripts/node0_activate.py mission "Test swarm engine: summarize BIZRA genesis"

# Expected log output includes:
#   Pre-loaded X/Y models into VRAM
#   [SWARM] Phase: PRELOADING → EXECUTING
#   [SWARM] Agent strategist started
#   [SWARM] Agent strategist completed
#   [SWARM] Phase: EXECUTING → SCORING
#   [SWARM] Phase: SCORING → EQUALIZING
#   Equalizer: <action or steady>
#   [SWARM] Phase: EQUALIZING → COMPLETE
```

## 5.5 Success Criteria

| Criterion | Metric | Phase |
|-----------|--------|-------|
| Types import without side effects | `python -c "from core.swarm.types import *"` exits 0 | 1 |
| All 7 PAT agents convert to AgentSpec | Unit test passes | 1 |
| Sequential topology preserves order | call_order matches input order | 2 |
| Parallel topology achieves speedup | wall_time < sequential * 0.5 | 2 |
| Hierarchical mesh runs coordinator first | call_order[0] == coordinator | 2 |
| Event bridge publishes to bus | Bus receives swarm.* events | 3 |
| Bridge does not break engine on failure | Engine completes despite bus errors | 3 |
| Feature flag controls path selection | Legacy runs when disabled | 4 |
| Result format backward-compatible | node0 scoring works on both paths | 4 |
| Zero test regressions | `pytest tests/ -x -q` passes | All |
| CI passes with flag off (default) | No SwarmEngine code runs in CI | All |

## 5.6 Monitoring (Post-Deploy)

Once enabled in production, monitor these signals:

| Signal | Source | Alert Threshold |
|--------|--------|-----------------|
| `swarm.agent_failed` events | EventBus | > 3 per mission |
| Mission wall-clock time | Logs | > 300s (regression) |
| Pre-load success rate | `swarm.model_preloaded` | < 50% |
| Equalizer action frequency | `swarm.equalizer_action` | > 1 HALT per hour |
| Memory/VRAM usage | System metrics | > 14GB VRAM |

## 5.7 Dependency Graph

```
Phase 1 (types.py)
    │
    ├──→ Phase 2 (engine.py) ──→ Phase 4 (node0 migration)
    │         │
    │         └──→ Phase 3 (event_bridge.py) ──→ Phase 4
    │
    └──→ Phase 5 (validation) — runs after each phase
```

Phases 1-2 are independent of Phases 3-4.
Phase 3 depends on Phase 2 (SwarmEngine) and the existing EventBus.
Phase 4 depends on all prior phases.
Phase 5 (this document) runs incrementally after each phase completes.
