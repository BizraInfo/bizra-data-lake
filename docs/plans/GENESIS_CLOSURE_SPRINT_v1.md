# Genesis Closure Sprint v1

Date: 2026-03-09
Status: Phase S COMPLETE (2026-03-14)
Scope: Close the gap between constitutional architecture and operational truth

## Phase S Closure (2026-03-14)

Nervous system gaps closed: 8 test fixes + 1 batching hang + config optimization.
- `apex_engine.py`: AttributeError catch in GoT bridge fallback
- `_batching.py`: Exception propagation to unblock callers (was infinite hang)
- `test_autonomy.py`, `test_integration_runtime.py`, `test_connection_pool.py`: exception type alignment
- `test_hmm_engine.py`: learn() contract truth (Phase 47 implementation)
- `test_moe_e2e.py`: MockReflex _hash_input for closed-loop pipeline
- `pyproject.toml`: timeout_method=signal, pytest-asyncio>=0.23, MyPy override ordering
- Identity Genesis: Ed25519 signatures, persona seeds, threshold registry (47 tests)
- Spearpoint campaign: 3/3 targets GREEN, all 12 gates PASS, run_id=23e385a2c870

## Executive Judgment

BIZRA does not need another conceptual expansion first.
It needs closure.

The highest-SNR move is to close the nervous system, make Ihsan authoritative
on the live path, prove the receipt chain end-to-end, and turn ReflexCompiler
from an available subsystem into a measured operational accelerator.

## Ground Truth Baseline

This sprint is grounded in the current repo state, not older audit language.

- `InferenceGateway` already exists in `core/inference/gateway.py`.
- `MissionOrchestrator` already initializes LLM access when `BIZRA_ENABLE_LLM=1`
  in `core/sovereign/mission.py`.
- `SovereignRuntime` already initializes and injects gateway state in
  `core/sovereign/runtime_core.py`.
- `POST /v1/plan` already exists in `core/sovereign/api.py` and already touches
  `ReflexCompiler`.
- `EventBus` and `ActionBus` exist, but the season archive subscriber layer is
  not yet integrated as `core/bus/subscribers.py`.

Therefore the real work is:

1. Wire the missing subscriber layer.
2. Make Ihsan computation authoritative on the live mission path.
3. Harden the inference path and prove its provenance in receipts.
4. Upgrade and operationalize ReflexCompiler on `/v1/plan`.
5. Produce one real First Heartbeat proof artifact.

Machine-readable execution board:

- `config/bizra_delivery_control_plane.json`
- `config/genesis_closure_program_board.json`

## Sprint Outcome

By the end of this sprint, one real request must pass through:

`request -> orchestration -> inference -> Ihsan/SNR -> receipt chain -> proof artifact`

without injected shortcuts, silent fallback ambiguity, or unverifiable quality claims.

## 12 Tasks

| # | Task | File / Component | Closure Criterion | Required Test | Ihsan / SNR Effect |
|---|------|------------------|-------------------|---------------|--------------------|
| 1 | Lock subscriber contract surface | `docs/schemas/action_schema_v1.json`, `docs/schemas/event_schema_v1.json`, `core/bus/topics.py` | Action/event topics are canonical, named, and versioned before wiring | New schema validation test under `tests/scripts/` or `tests/core/sovereign/` | Reduces semantic drift; raises SNR by eliminating topic ambiguity |
| 2 | Implement the 12-subscriber nervous system | `core/bus/subscribers.py` | All 12 subscribers exist with explicit topic bindings and side effects | New `tests/core/sovereign/test_bus_subscribers.py` | Turns EventBus from passive queue into operational nervous system |
| 3 | Wire subscribers into runtime boot | `core/bus/sovereign_wiring.py`, `core/sovereign/runtime_core.py` | Runtime boot registers subscribers automatically and fails closed on invalid wiring | Extend `tests/core/sovereign/test_runtime_core_pipeline.py` | Ensures signal propagation is runtime truth, not archive intent |
| 4 | Prove subscriber coverage end-to-end | `core/sovereign/event_bus.py`, `core/bus/subscribers.py` | One integration test emits representative events and proves all 12 handlers are reachable | Extend `tests/core/test_rust_bridge.py` and add integration path test | Increases SNR through observable event causality |
| 5 | Thin-slice inference connector hardening | `core/inference/gateway.py`, `core/sovereign/mission.py` | Mission path records which backend answered, latency, and fallback reason when degraded | Extend `tests/integration/test_mission_pipeline.py` and `tests/integration/test_full_reasoning_cycle.py` | Makes inference provenance auditable; blocks hidden low-signal fallbacks |
| 6 | Attach inference provenance to receipts | `core/sovereign/api.py`, `core/proof_engine/receipt.py`, `core/proof_engine/evidence_ledger.py` | `/v1/plan` result contains backend/model provenance in the proof surface | Extend `tests/core/sovereign/test_plan_receipt_contract.py` | Raises auditability dimension of Ihsan and boosts trust-weighted SNR |
| 7 | Make Ihsan content-derived on the mission path | `core/proof_engine/ihsan_computer.py`, `core/proof_engine/ihsan_gate.py`, `core/sovereign/mission.py`, `core/sovereign/runtime_core.py` | Live mission path computes Ihsan from content/evidence signals, not injected defaults on the authoritative branch | Extend `tests/core/sovereign/test_runtime_core_pipeline.py` and `tests/integration/test_fate_gate_pipeline.py` | Converts Ihsan from narrative to runtime law |
| 8 | Prove fail-closed degradation after low-Ihsan sequence | `core/proof_engine/ihsan_gate.py`, `core/sovereign/runtime_core.py` | Repeated sub-threshold outputs trigger explicit degraded behavior and reason codes | Extend `tests/integration/test_seven_layer_stack.py` | Protects SNR by preventing repeated low-quality operation |
| 9 | Upgrade ReflexCompiler with season capabilities | `core/sovereign/reflex_compiler.py` | Add HHMM/evidence/gossip/import/revalidation features without regressing current bounded-cache fixes | Extend `tests/core/sovereign/test_reflex_compiler.py` | Enables higher-SNR compiled reflexes and lower latency |
| 10 | Make ReflexCompiler first-class on `/v1/plan` | `core/sovereign/api.py`, `core/sovereign/reflex_compiler.py` | `/v1/plan` reports reflex hit/miss, hit latency, and precipitation observations in response/telemetry | Extend `tests/integration/test_plan_endpoint.py` | Converts deliberation into measured compiled excellence |
| 11 | Execute First Heartbeat proof run | `scripts/node0_live.sh`, `scripts/node0_standalone.py`, `scripts/genesis_heartbeat_live.py` or new `scripts/first_heartbeat.py` | One real seed request produces a signed proof artifact and receipt chain tail | Add smoke proof test under `tests/scripts/` | Turns architectural legitimacy into lived evidence |
| 12 | Add Closure Gate to CI | `.github/workflows/ci.yml`, `tests/scripts/test_node0_standalone.py`, targeted integration suite | CI enforces the closure pack: subscribers, Ihsan computation, plan receipt, standalone API smoke | New CI-target list in workflow + green run | Prevents regression back into injected or unwired behavior |

## Execution Waves

### Wave 1: Nervous System

Tasks: 1, 2, 3, 4

Goal:
- make event propagation real
- make subscriber count measurable
- convert "bus architecture" into an observable runtime mechanism

### Wave 2: Runtime Truth

Tasks: 5, 6, 7, 8

Goal:
- make inference provenance explicit
- make Ihsan authoritative on content
- prove degradation behavior when constitutional floor is violated

### Wave 3: Compiled Excellence

Tasks: 9, 10

Goal:
- upgrade ReflexCompiler to the richer season-grade version
- make `/v1/plan` visibly faster and more truthful on warm paths

### Wave 4: First Heartbeat

Tasks: 11, 12

Goal:
- run one real end-to-end request
- preserve the result as evidence
- prevent regression through CI

## Definition of Done

This sprint is not complete unless all five proofs exist:

1. Receipt chain from request to response is preserved and signed.
2. All 12 subscribers are wired and covered by integration tests.
3. Ihsan on the authoritative mission path is computed from content-derived signals.
4. Reflex hit path works on `/v1/plan` with explicit telemetry and target latency.
5. One real seed request produces a proof artifact that can be replayed and inspected.

## Non-Goals

These are intentionally out of scope for this sprint:

- new grand architecture documents
- multi-node federation proof
- full MOE expansion beyond what is needed for closure
- UI polish beyond what is needed to expose proof/telemetry
- speculative "crown" features before nervous-system closure

## Risks and Controls

| Risk | Control |
|------|---------|
| Subscriber wiring drifts from actual topic names | Lock schemas first, then wire |
| Ihsan remains mixed between computed and injected paths | Define one authoritative branch and test it directly |
| Reflex upgrade regresses bounded-cache safety | Preserve current eviction/time-bounded fixes and add regression tests first |
| Proof artifact exists but lacks backend provenance | Make provenance part of receipt contract, not optional metadata |
| Standalone demo works but CI never exercises it | Add a minimal closure gate to CI |

## Operational Commandment

Do not add another crown before closing the nervous system.

In Arabic:

لا تضف تاجًا جديدًا قبل أن تغلق الجهاز العصبي للنظام.

## Immediate Next Move

Start with Task 1 and Task 2 together:

- lock action/event contracts
- implement `core/bus/subscribers.py`

That is the shortest path from architectural intention to operational closure.



