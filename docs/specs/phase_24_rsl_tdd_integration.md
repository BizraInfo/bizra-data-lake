# Phase 24: RSL TDD Anchors, Integration Wiring & Structural Coupling

> Standing on Giants: Maturana-Varela (Autopoiesis, 1972) · Beck (TDD, 1999) · Shannon (Channel Capacity, 1948) · Deming (PDCA, 1950) · Al-Ghazali (Ihsan, 1095) · Lamport (State Machines, 1978)

## 1. Overview

This phase covers three critical deliverables:
1. **Structural Coupling Engine** — how the RSL self-evolves to fit the user's shape
2. **Integration Wiring** — how RSL connects to all existing subsystems
3. **Complete TDD Test Plan** — test-first anchors for the full RSL implementation

## 2. Structural Coupling Engine (`core/rsl/structural_coupling.py`)

```pseudocode
MODULE core.rsl.structural_coupling

IMPORT ShadowGraph FROM core.rsl.shadow_graph
IMPORT PersonaTransformer FROM core.rsl.persona_transformer
IMPORT AutopoieticLoop FROM core.autopoiesis.loop_engine
IMPORT UNIFIED_IHSAN_THRESHOLD FROM core.integration.constants

CLASS StructuralCoupling:
    """
    The RSL's self-evolution mechanism.

    Maturana-Varela Insight: A system is "autopoietic" when it
    continuously regenerates itself to maintain its organization.

    The StructuralCoupling engine does this for the RSL:
    - Analyzes interaction history to find persona mismatches
    - Adjusts voice templates based on user feedback signals
    - Evolves the Shadow Graph's weight parameters
    - Periodically "molts" — sheds outdated patterns

    The lobster metaphor: To grow, you must shed your shell.
    The RSL periodically evaluates whether its current persona
    still fits the user, and if not, evolves.
    """

    CONSTRUCTOR(shadow: ShadowGraph, config: RSLConfig = DEFAULT):
        self.shadow = shadow
        self.config = config
        self._active = False
        self._evolution_count: int = 0

    # ═══════════════════════════════════════════════════════
    # EVOLUTION CYCLE
    # ═══════════════════════════════════════════════════════

    ASYNC METHOD start_evolution_cycle():
        """
        Begin the periodic evolution assessment.
        Runs every config.coupling_evolution_hours (default: 24h).
        """
        self._active = True
        WHILE self._active:
            AWAIT sleep(self.config.coupling_evolution_hours * 3600)
            TRY:
                AWAIT self.evolve()
            EXCEPT Exception as e:
                LOG.error("Evolution cycle failed: %s", e)

    METHOD stop():
        self._active = False

    ASYNC METHOD evolve():
        """
        Single evolution step.

        PDCA Cycle (Deming):
        1. PLAN: Analyze recent interaction patterns
        2. DO: Propose persona adjustments
        3. CHECK: Validate against Ihsan gate
        4. ACT: Apply or discard adjustments
        """
        LOG "Evolution cycle %d starting", self._evolution_count

        # PLAN: Analyze
        analysis = self._analyze_recent_interactions()
        IF analysis.satisfaction_score > 0.9:
            LOG "High satisfaction (%.2f) — no evolution needed", analysis.satisfaction_score
            RETURN

        # DO: Propose
        proposals = self._propose_adjustments(analysis)

        # CHECK: Validate
        valid_proposals = []
        FOR proposal IN proposals:
            ihsan = self._validate_proposal(proposal)
            IF ihsan >= UNIFIED_IHSAN_THRESHOLD:
                valid_proposals.append(proposal)
            ELSE:
                LOG "Proposal rejected (ihsan=%.2f): %s", ihsan, proposal.description

        # ACT: Apply
        FOR proposal IN valid_proposals:
            self._apply_adjustment(proposal)
            LOG "Applied: %s", proposal.description

        self._evolution_count += 1
        LOG "Evolution cycle %d complete: %d adjustments applied",
            self._evolution_count, len(valid_proposals)

    METHOD _analyze_recent_interactions() -> CouplingAnalysis:
        """
        Analyze the last N interactions for persona fit.

        Metrics:
        - Satisfaction: Did the user accept/use the response?
        - Interruption count: Did the user cut off the response?
        - Correction count: Did the user rephrase/retry?
        - Trust trajectory: Is trust growing or declining?
        """
        recent = self.shadow.get_recent_interactions(limit=100)

        accepted = count(i FOR i IN recent IF i.was_accepted)
        interrupted = count(i FOR i IN recent IF i.was_interrupted)
        corrected = count(i FOR i IN recent IF i.was_corrected)

        RETURN CouplingAnalysis(
            satisfaction_score=accepted / max(len(recent), 1),
            interruption_rate=interrupted / max(len(recent), 1),
            correction_rate=corrected / max(len(recent), 1),
            trust_trajectory=self.shadow._trust_score,
            sample_size=len(recent),
        )

    METHOD _propose_adjustments(analysis: CouplingAnalysis) -> List[CouplingProposal]:
        """
        Generate adjustment proposals based on analysis.
        """
        proposals = []

        # High correction rate → voice tone mismatch
        IF analysis.correction_rate > 0.2:
            proposals.append(CouplingProposal(
                dimension="voice_tone",
                description="High correction rate — adjusting voice conciseness",
                adjustment={"preferred_voice_tone": "concise"},
            ))

        # High interruption rate → too verbose
        IF analysis.interruption_rate > 0.15:
            proposals.append(CouplingProposal(
                dimension="verbosity",
                description="High interruption rate — reducing output length",
                adjustment={"max_response_lines": max(3, current - 2)},
            ))

        # Declining trust → increase transparency
        IF analysis.trust_trajectory < 0.3:
            proposals.append(CouplingProposal(
                dimension="transparency",
                description="Low trust — increasing explanation depth",
                adjustment={"show_reasoning": True},
            ))

        RETURN proposals

    METHOD _validate_proposal(proposal: CouplingProposal) -> float:
        """
        Validate a proposal against the Ihsan constraint.

        Every evolution must pass the Daughter Test:
        "Would I be comfortable if my daughter experienced this change?"
        """
        # Simulate the adjustment on recent interactions
        simulated_score = self._simulate_adjustment(proposal)
        RETURN simulated_score

    METHOD _apply_adjustment(proposal: CouplingProposal):
        """Apply a validated adjustment to the Shadow Graph."""
        IF proposal.dimension == "voice_tone":
            self.shadow._rhythm.preferred_voice_tone = proposal.adjustment["preferred_voice_tone"]
        ELIF proposal.dimension == "verbosity":
            self.shadow._rhythm.max_response_lines = proposal.adjustment["max_response_lines"]
        ELIF proposal.dimension == "transparency":
            self.shadow._rhythm.show_reasoning = proposal.adjustment["show_reasoning"]

        self.shadow.save()

    # ═══════════════════════════════════════════════════════
    # STATE RECORDING (for evolution analysis)
    # ═══════════════════════════════════════════════════════

    METHOD record_state(state: CognitiveState):
        """Record cognitive state for trend analysis."""
        self.shadow.record_entropy_snapshot(
            self.shadow._entropy_history.latest_manifold()
            IF self.shadow._entropy_history
            ELSE default_manifold()
        )


DATACLASS CouplingAnalysis:
    satisfaction_score: float
    interruption_rate: float
    correction_rate: float
    trust_trajectory: float
    sample_size: int

DATACLASS CouplingProposal:
    dimension: str
    description: str
    adjustment: Dict[str, Any]
```

## 3. Integration Wiring

### 3.1 Files to Modify

| File | Change | Lines |
|------|--------|-------|
| `core/rsl/__init__.py` | NEW — Package exports | ~25 |
| `core/sovereign/runtime_core.py` | Init RSL in `_init_proactive()` | +15 |
| `core/sovereign/proactive_integration.py` | Wire RSL into OODA loop output | +20 |
| `core/bridges/desktop_bridge.py` | Add `rsl_status` RPC method | +10 |
| `core/sovereign/api.py` | Add `/rsl/status` and `/rsl/speak` endpoints | +30 |
| `core/elite/hooks.py` | Register RSL event hooks | +10 |

### 3.2 Package Init (`core/rsl/__init__.py`)

```pseudocode
MODULE core.rsl

# Public API
EXPORT SovereignSoul FROM core.rsl.soul
EXPORT ShadowGraph FROM core.rsl.shadow_graph
EXPORT PersonaTransformer FROM core.rsl.persona_transformer
EXPORT ProactiveCoworker FROM core.rsl.proactive_coworker
EXPORT StructuralCoupling FROM core.rsl.structural_coupling
EXPORT EntropySensor FROM core.rsl.entropy_sensor
EXPORT RSLConfig FROM core.rsl.config

# Types
EXPORT CognitiveState, SoulFaculty, SoulResponse FROM core.rsl.types
EXPORT ProactiveInsight, RelationalContext, UserRhythm FROM core.rsl.types
EXPORT FACULTY_MAP FROM core.rsl.types
```

### 3.3 Runtime Integration (`core/sovereign/runtime_core.py`)

```pseudocode
# In _init_proactive() method, after PAT engine init:

METHOD _init_rsl():
    """Initialize the Reflective Soul Layer."""
    FROM core.rsl import SovereignSoul, RSLConfig

    config = RSLConfig(
        shadow_graph_path=str(self.state_dir / "shadow_graph.db"),
        ihsan_threshold=UNIFIED_IHSAN_THRESHOLD,
        snr_threshold=UNIFIED_SNR_THRESHOLD,
    )

    self.soul = SovereignSoul(
        pat_engine=self.pat_engine,
        memory=self.agent_db,          # V3 AgentDB
        entropy_calc=self.entropy_calc, # UERS 5D
        config=config,
    )

    LOG "RSL initialized with AgentDB + UERS entropy"
```

### 3.4 Proactive Integration Wire

```pseudocode
# In ProactiveSovereignEntity._execute_ooda_cycle():

# After line ~350 (where raw OODA output is formatted):
IF self.soul IS NOT None:
    # Wrap OODA output through the Soul Layer
    soul_response = AWAIT self.soul.speak(ooda_output)
    formatted_output = soul_response.message
ELSE:
    # Fallback: raw output (backward compatible)
    formatted_output = ooda_output
```

### 3.5 API Endpoints (`core/sovereign/api.py`)

```pseudocode
# New endpoints:

@app.get("/rsl/status")
ASYNC FUNCTION rsl_status():
    """Get RSL status and current user model."""
    soul = get_runtime().soul
    IF soul IS None:
        RETURN {"status": "not_initialized"}

    state = AWAIT soul.perceive_user_state()
    context = soul.shadow.get_relational_context(state)

    RETURN {
        "status": "active",
        "cognitive_state": state.name,
        "trust_level": context.trust_level,
        "session_minutes": context.session_duration_minutes,
        "active_project": context.active_project,
        "evolution_count": soul.coupling._evolution_count,
    }

@app.post("/rsl/speak")
ASYNC FUNCTION rsl_speak(request: SpeakRequest):
    """Process intent through the Soul Layer."""
    soul = get_runtime().soul
    IF soul IS None:
        RAISE HTTPException(503, "RSL not initialized")

    response = AWAIT soul.speak(request.intent)
    RETURN {
        "message": response.message,
        "faculty": response.faculty_used.name,
        "state": response.cognitive_state.name,
        "proactive": response.proactive,
        "ihsan": response.ihsan_score,
        "snr": response.snr_score,
    }
```

### 3.6 Hook Registration

```pseudocode
# In core/elite/hooks.py — register RSL events:

HOOK_DEFINITIONS.extend([
    HookDefinition(
        name="rsl:proactive_insight",
        phase=HookPhase.POST_ANALYSIS,
        description="Triggered when RSL detects an unspoken need",
    ),
    HookDefinition(
        name="rsl:persona_evolved",
        phase=HookPhase.POST_ANALYSIS,
        description="Triggered when structural coupling adjusts persona",
    ),
    HookDefinition(
        name="rsl:cognitive_state_change",
        phase=HookPhase.POST_ANALYSIS,
        description="Triggered when user cognitive state transitions",
    ),
])
```

## 4. Complete TDD Test Plan

### 4.1 Test File Structure

```
tests/core/rsl/
├── conftest.py                    # Shared fixtures
├── test_types.py                  # Type classifications
├── test_soul.py                   # SovereignSoul orchestrator
├── test_shadow_graph.py           # Shadow Graph persistence
├── test_entropy_sensor.py         # Entropy bridge
├── test_persona_transformer.py    # Voice synthesis
├── test_proactive_coworker.py     # Anticipatory engine
├── test_structural_coupling.py    # Self-evolution
└── test_integration.py            # Cross-module wiring
```

### 4.2 Fixtures (`conftest.py`)

```pseudocode
FIXTURE mock_agentdb():
    """In-memory AgentDB for testing."""
    db = AgentDB(config=AgentDBConfig(path=":memory:"))
    YIELD db
    db.close()

FIXTURE mock_entropy_calc():
    """Configurable entropy calculator for testing."""
    calc = MockEntropyCalculator(default_total=3.0)
    YIELD calc

FIXTURE mock_pat_engine():
    """PAT engine that returns predictable outputs."""
    engine = MockPATEngine(
        default_output={"summary": "Task completed.", "details": "No issues found."},
    )
    YIELD engine

FIXTURE rsl_soul(mock_agentdb, mock_entropy_calc, mock_pat_engine):
    """Fully wired SovereignSoul for testing."""
    soul = SovereignSoul(
        pat_engine=mock_pat_engine,
        memory=mock_agentdb,
        entropy_calc=mock_entropy_calc,
    )
    YIELD soul

FUNCTION make_context(**overrides) -> RelationalContext:
    """Factory for test RelationalContext."""
    defaults = {
        "cognitive_state": CognitiveState.ENGAGED_WORK,
        "trust_level": 0.5,
        "session_duration_minutes": 30,
        "active_project": "test-project",
    }
    defaults.update(overrides)
    RETURN RelationalContext(**defaults)

FUNCTION make_manifold(total=3.0, **overrides) -> ManifoldState:
    """Factory for test ManifoldState."""
    # Distribute total across 5 dimensions
    per_dim = total / 5.0
    RETURN ManifoldState(
        surface=EntropyMeasurement("surface", per_dim, per_dim / 5.0, **overrides.get("surface", {})),
        structural=EntropyMeasurement("structural", per_dim, per_dim / 5.0),
        behavioral=EntropyMeasurement("behavioral", per_dim, per_dim / 5.0),
        hypothetical=EntropyMeasurement("hypothetical", per_dim, per_dim / 5.0),
        contextual=EntropyMeasurement("contextual", per_dim, per_dim / 5.0),
    )
```

### 4.3 Core Test Anchors

```pseudocode
# ─── test_types.py ────────────────────────────────────────

TEST "CognitiveState classification covers full range":
    ASSERT classify_state(0.0) == CognitiveState.DEEP_FLOW
    ASSERT classify_state(1.9) == CognitiveState.DEEP_FLOW
    ASSERT classify_state(2.0) == CognitiveState.ENGAGED_WORK
    ASSERT classify_state(3.4) == CognitiveState.ENGAGED_WORK
    ASSERT classify_state(3.5) == CognitiveState.MILD_SCATTER
    ASSERT classify_state(4.1) == CognitiveState.MILD_SCATTER
    ASSERT classify_state(4.2) == CognitiveState.HIGH_CHAOS
    ASSERT classify_state(4.7) == CognitiveState.HIGH_CHAOS
    ASSERT classify_state(4.8) == CognitiveState.OVERWHELMED
    ASSERT classify_state(10.0) == CognitiveState.OVERWHELMED

TEST "Faculty mapping covers all 7 PAT agent types":
    FOR agent_type IN AgentType:
        ASSERT agent_type IN FACULTY_MAP
    ASSERT len(FACULTY_MAP) == 7

TEST "SoulResponse dataclass validates ihsan range":
    response = SoulResponse(message="test", ihsan_score=0.97, ...)
    ASSERT 0.0 <= response.ihsan_score <= 1.0

# ─── test_soul.py ─────────────────────────────────────────

TEST "SovereignSoul.speak returns unified message":
    soul = rsl_soul()
    response = AWAIT soul.speak("fix the bug")
    ASSERT isinstance(response, SoulResponse)
    ASSERT response.message != ""
    ASSERT response.ihsan_score >= UNIFIED_IHSAN_THRESHOLD
    ASSERT response.proactive == False

TEST "SovereignSoul respects Ihsan gate":
    soul = rsl_soul()
    # Force low-quality output from PAT
    soul.faculties.set_output({"summary": ""})
    response = AWAIT soul.speak("test")
    # Even with empty input, Ihsan gate should elevate
    ASSERT response.ihsan_score >= UNIFIED_IHSAN_THRESHOLD

TEST "SovereignSoul.perceive_user_state returns valid state":
    soul = rsl_soul()
    state = AWAIT soul.perceive_user_state()
    ASSERT isinstance(state, CognitiveState)

TEST "find_the_unspoken returns None during DEEP_FLOW":
    soul = rsl_soul()
    soul.entropy._calc.set_total(1.0)  # DEEP_FLOW
    insight = soul.find_the_unspoken()
    ASSERT insight IS None  # Sacred silence

# ─── test_shadow_graph.py ─────────────────────────────────

TEST "Shadow Graph roundtrip save/load":
    graph = ShadowGraph(memory=mock_agentdb)
    graph._rhythm.peak_hours = [9, 10, 14]
    graph._trust_score = 0.75
    graph.save()

    graph2 = ShadowGraph(memory=mock_agentdb)
    graph2.load()
    ASSERT graph2._rhythm.peak_hours == [9, 10, 14]
    ASSERT abs(graph2._trust_score - 0.75) < 0.01

TEST "Trust decay over time":
    graph = ShadowGraph(memory=mock_agentdb)
    graph._trust_score = 1.0
    graph._last_interaction_time = 3.5_days_ago  # Half of 7-day decay
    graph.apply_trust_decay()
    ASSERT 0.4 < graph._trust_score < 0.6  # ~50% decay

TEST "Entropy trend computation":
    graph = ShadowGraph(memory=mock_agentdb)
    FOR i IN range(10):
        graph.record_entropy_snapshot(make_manifold(total=2.0 + i * 0.2))
    trend = graph.compute_entropy_trend(window_minutes=30)
    ASSERT trend > 0  # Rising entropy

# ─── test_structural_coupling.py ──────────────────────────

TEST "Evolution proposes voice change on high correction rate":
    coupling = StructuralCoupling(shadow=mock_shadow)
    analysis = CouplingAnalysis(
        satisfaction_score=0.6,
        interruption_rate=0.1,
        correction_rate=0.3,  # > 0.2 threshold
        trust_trajectory=0.5,
        sample_size=100,
    )
    proposals = coupling._propose_adjustments(analysis)
    ASSERT any(p.dimension == "voice_tone" FOR p IN proposals)

TEST "Evolution does not trigger on high satisfaction":
    coupling = StructuralCoupling(shadow=mock_shadow)
    coupling._analyze = LAMBDA: CouplingAnalysis(satisfaction_score=0.95, ...)
    AWAIT coupling.evolve()
    ASSERT coupling._evolution_count == 0  # Skipped — too happy to change

TEST "All proposals pass Ihsan gate":
    coupling = StructuralCoupling(shadow=mock_shadow)
    FOR proposal IN test_proposals:
        ihsan = coupling._validate_proposal(proposal)
        ASSERT ihsan >= UNIFIED_IHSAN_THRESHOLD OR proposal.was_rejected

# ─── test_integration.py ──────────────────────────────────

TEST "RSL imports succeed":
    FROM core.rsl import SovereignSoul, ShadowGraph, PersonaTransformer
    FROM core.rsl import ProactiveCoworker, StructuralCoupling, EntropySensor
    FROM core.rsl import CognitiveState, SoulFaculty, RSLConfig

TEST "RSL wires to AgentDB V3":
    FROM core.memory import AgentDB
    db = AgentDB(config=test_config)
    soul = SovereignSoul(pat_engine=mock_pat, memory=db, entropy_calc=mock_entropy)
    ASSERT soul.shadow.memory IS db

TEST "RSL wires to UERS entropy":
    FROM core.uers.entropy import EntropyCalculator
    calc = EntropyCalculator()
    sensor = EntropySensor(calculator=calc)
    state, manifold = AWAIT sensor.sense()
    ASSERT isinstance(state, CognitiveState)

TEST "Full lifecycle: start → speak → proactive → evolve → stop":
    soul = rsl_soul()
    AWAIT soul.start()

    # Speak
    response = AWAIT soul.speak("hello")
    ASSERT response.message != ""

    # Simulate proactive detection
    soul.entropy._calc.set_total(5.0)  # OVERWHELMED
    insight = soul.find_the_unspoken()
    ASSERT insight IS NOT None

    # Simulate evolution
    soul.coupling._evolution_count == 0
    AWAIT soul.coupling.evolve()

    AWAIT soul.stop()
```

## 5. Implementation Order

| Step | File | Depends On | Estimated Lines |
|------|------|-----------|-----------------|
| 1 | `core/rsl/types.py` | `core.uers.entropy`, `core.pat.agent`, `core.integration.constants` | ~120 |
| 2 | `core/rsl/config.py` | `core.integration.constants` | ~60 |
| 3 | `core/rsl/entropy_sensor.py` | `core.uers.entropy`, types | ~80 |
| 4 | `core/rsl/shadow_graph.py` | `core.memory.agent_db`, entropy_sensor, types | ~300 |
| 5 | `core/rsl/persona_transformer.py` | types, config | ~200 |
| 6 | `core/rsl/proactive_coworker.py` | shadow_graph, entropy_sensor, types | ~250 |
| 7 | `core/rsl/structural_coupling.py` | shadow_graph, persona_transformer | ~200 |
| 8 | `core/rsl/soul.py` | ALL above | ~250 |
| 9 | `core/rsl/__init__.py` | ALL above | ~25 |
| 10 | Integration wiring (runtime_core, api, hooks) | core/rsl package | ~75 |
| **Total** | | | **~1,560** |

## 6. Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Proactive insights become annoying | Rate limit (5/hr), urgency gate (0.7), sacred silence during DEEP_FLOW |
| Persona evolution drifts from user intent | Ihsan gate on ALL evolution proposals, PDCA cycle with rollback |
| Shadow Graph grows unbounded | RingBuffer (1000 entries), AgentDB handles HNSW cleanup |
| Entropy sensor adds latency | Sensor runs async on 30s cycle, never blocks user commands |
| Trust model games the system | Asymmetric trust (grows slowly, decays 3x faster) |
| Integration breaks existing OODA | RSL wraps output, never replaces; `IF soul IS NOT None` guard everywhere |

## 7. Constitutional Invariants

- **SEL remains read-only** — RSL never writes to the Experience Ledger
- **All thresholds from constants.py** — no hardcoded values in RSL
- **Daughter Test on evolution** — every persona change passes constitutional gate
- **Sacred silence** — DEEP_FLOW state suppresses ALL proactive messages
- **Trust earned, not given** — starts at 0.0, grows only through correct behavior
- **Transparent audit** — raw_outputs preserved in every SoulResponse for inspection
