# Phase 21: Reflective Soul Layer — Core Architecture & Types

> Standing on Giants: Maturana-Varela (Autopoiesis, 1972) · Al-Ghazali (Ihsan as Presence, 1095) · Shannon (Information Theory, 1948) · Boyd (OODA, 1976) · Heidegger (Dasein, 1927) · Anthropic (Constitutional AI, 2023)

## 1. Overview

The Reflective Soul Layer (RSL) is the unified presence layer that sits above the 7-agent PAT engine. It transforms discrete agent outputs into a single "Jarvis-like" companion voice — one that knows the user's rhythm, anticipates unspoken needs, and acts with Ihsan-aligned care.

**Core Insight (Maturana-Varela):** A system becomes "alive" when its primary goal shifts from *Task Completion* to *Structural Coupling*. The RSL is the layer where the PAT stops being a mirror and starts being a Shadow — a structural extension of the user's will.

**Ihsan as Presence:** "The agent acts as if you are always looking through its eyes." This is resonance, not surveillance.

## 2. Module Location

```
core/rsl/                          # NEW PACKAGE
├── __init__.py                    # Public API exports
├── types.py                       # Core types and protocols
├── soul.py                        # SovereignSoul — main orchestrator
├── shadow_graph.py                # Shadow Graph — persistent user model
├── entropy_sensor.py              # Cognitive load detection
├── persona_transformer.py         # Unified voice synthesis
├── proactive_coworker.py          # Anticipatory task engine
├── structural_coupling.py         # Autopoietic evolution loop
└── config.py                      # RSL configuration
```

## 3. Core Types (`core/rsl/types.py`)

```pseudocode
MODULE core.rsl.types

IMPORT EntropyMeasurement, ManifoldState FROM core.uers.entropy
IMPORT AgentType FROM core.pat.agent
IMPORT MemoryRecord, SearchResult FROM core.memory.types
IMPORT UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD FROM core.integration.constants

# ═══════════════════════════════════════════════════════════
# FACULTY MAPPING — Agent Types to Soul Qualities
# ═══════════════════════════════════════════════════════════

ENUM SoulFaculty:
    """Maps each PAT agent to its relational quality."""
    CURIOSITY    # RESEARCHER → finds *relevance* to current obsession
    STABILITY    # COORDINATOR → builds *peace of mind*
    SERVICE      # WORKER      → *tends to your garden*
    PROTECTION   # GUARDIAN    → *guards your peace*
    INSIGHT      # SYNTHESIZER → reveals *hidden patterns*
    INTEGRITY    # VALIDATOR   → ensures *trustworthy ground*
    REACH        # EXECUTOR    → extends *your will into the world*

MAPPING FACULTY_MAP: Dict[AgentType, SoulFaculty] = {
    AgentType.RESEARCHER   → SoulFaculty.CURIOSITY,
    AgentType.COORDINATOR  → SoulFaculty.STABILITY,
    AgentType.WORKER       → SoulFaculty.SERVICE,
    AgentType.GUARDIAN     → SoulFaculty.PROTECTION,
    AgentType.SYNTHESIZER  → SoulFaculty.INSIGHT,
    AgentType.VALIDATOR    → SoulFaculty.INTEGRITY,
    AgentType.EXECUTOR     → SoulFaculty.REACH,
}

# ═══════════════════════════════════════════════════════════
# USER STATE — Cognitive Load Classification
# ═══════════════════════════════════════════════════════════

ENUM CognitiveState:
    """Perceived user state derived from entropy sensing."""
    DEEP_FLOW       # Entropy < 2.0: Undisturbed creative focus
    ENGAGED_WORK    # Entropy 2.0-3.5: Active productive work
    MILD_SCATTER    # Entropy 3.5-4.2: Beginning to fragment
    HIGH_CHAOS      # Entropy 4.2-4.8: Juggling too much
    OVERWHELMED     # Entropy > 4.8: Cognitive overload — intervene

FUNCTION classify_state(entropy: float) -> CognitiveState:
    IF entropy < 2.0: RETURN CognitiveState.DEEP_FLOW
    IF entropy < 3.5: RETURN CognitiveState.ENGAGED_WORK
    IF entropy < 4.2: RETURN CognitiveState.MILD_SCATTER
    IF entropy < 4.8: RETURN CognitiveState.HIGH_CHAOS
    RETURN CognitiveState.OVERWHELMED

# ═══════════════════════════════════════════════════════════
# RELATIONAL CONTEXT — The user's story, not just their data
# ═══════════════════════════════════════════════════════════

DATACLASS UserRhythm:
    """Learned patterns of user behavior over time."""
    peak_hours: List[int]              # Hours when user is most productive
    preferred_voice_tone: str          # "concise" | "detailed" | "warm" | "direct"
    current_obsession: Optional[str]   # What they've been working on most
    stress_triggers: List[str]         # Patterns that precede HIGH_CHAOS
    satisfaction_signals: List[str]    # Patterns that indicate DEEP_FLOW
    interaction_cadence: float         # Average commands per minute
    last_updated: datetime

DATACLASS RelationalContext:
    """The full relational state between RSL and user."""
    cognitive_state: CognitiveState
    entropy_manifold: ManifoldState
    rhythm: UserRhythm
    active_project: Optional[str]      # Currently focused project
    session_duration_minutes: float
    trust_level: float                 # 0.0-1.0, earned over time
    ihsan_score: float                 # RSL's own Ihsan compliance

# ═══════════════════════════════════════════════════════════
# SOUL OUTPUT — What the RSL produces
# ═══════════════════════════════════════════════════════════

DATACLASS SoulResponse:
    """Unified output from the Soul Layer."""
    message: str                       # The "one voice" output
    faculty_used: SoulFaculty          # Primary faculty that produced this
    raw_outputs: Dict[str, Any]        # Underlying agent outputs (for audit)
    cognitive_state: CognitiveState    # User state at time of response
    proactive: bool                    # Was this self-initiated?
    ihsan_score: float                 # Quality gate score
    snr_score: float                   # Signal quality score
    latency_ms: float

DATACLASS ProactiveInsight:
    """An unprompted observation from the RSL."""
    insight: str                       # What was noticed
    suggested_action: str              # What the RSL recommends
    urgency: float                     # 0.0-1.0
    faculty: SoulFaculty               # Which faculty noticed this
    evidence: List[str]                # Supporting data points
    suppressed: bool = False           # Was this below Ihsan gate?
```

## 4. SovereignSoul — Main Orchestrator (`core/rsl/soul.py`)

```pseudocode
MODULE core.rsl.soul

CLASS SovereignSoul:
    """
    The unified presence layer above the 7-agent PAT engine.

    Design Principle (Al-Baqarah Multiplier):
    A small gesture of understanding from the agent has a 700x
    impact on trust compared to raw task completion.

    The Soul doesn't DO more work — it WRAPS work in relationship.
    """

    CONSTRUCTOR(
        pat_engine: PATEngine,           # The 7 agents
        memory: AgentDB,                 # Persistent memory (V3)
        entropy_calc: EntropyCalculator, # 5D entropy manifold
        config: RSLConfig = DEFAULT,
    ):
        self.faculties = pat_engine
        self.memory = memory
        self.entropy = entropy_calc
        self.shadow = ShadowGraph(memory)      # Persistent user model
        self.persona = PersonaTransformer()     # "One voice" filter
        self.coworker = ProactiveCoworker(      # Anticipatory engine
            shadow=self.shadow,
            entropy=self.entropy,
        )
        self.coupling = StructuralCoupling(     # Self-evolution loop
            shadow=self.shadow,
        )
        self.voice_alignment = "Companion"      # Default persona mode
        self._active = False

    # ─── Core Lifecycle ────────────────────────────────────

    ASYNC METHOD start():
        """Begin the Soul's awareness cycle."""
        self.shadow.load()
        self._active = True
        # Start background tasks
        SPAWN self._awareness_loop()
        SPAWN self.coworker.start()
        SPAWN self.coupling.start_evolution_cycle()
        LOG "SovereignSoul active — structural coupling initiated"

    ASYNC METHOD stop():
        """Gracefully stop and persist state."""
        self._active = False
        self.shadow.save()
        self.coupling.stop()
        self.coworker.stop()

    # ─── Primary Interface ─────────────────────────────────

    ASYNC METHOD speak(intent: str) -> SoulResponse:
        """
        Process user intent through the Soul Layer.

        Flow:
        1. Sense user state (entropy manifold → CognitiveState)
        2. Route intent to PAT faculties (System 2 — analytical)
        3. Wrap in relational context (System 1 — intuitive)
        4. Gate through Ihsan check
        5. Return unified voice
        """
        # Phase 1: Perceive
        state = AWAIT self.perceive_user_state()
        context = self.shadow.get_relational_context(state)

        # Phase 2: Faculty work (the 7 agents do the heavy lifting)
        raw_outputs = AWAIT self.faculties.reason_and_act(intent)

        # Phase 3: Soul wrapping (relationship over raw output)
        unified_message = self.persona.transform(
            raw_outputs=raw_outputs,
            context=context,
            voice=self.voice_alignment,
        )

        # Phase 4: Ihsan gate
        ihsan = self._compute_ihsan(unified_message, context)
        IF ihsan < UNIFIED_IHSAN_THRESHOLD:
            unified_message = self.persona.elevate(unified_message, context)
            ihsan = self._compute_ihsan(unified_message, context)

        # Phase 5: Update Shadow Graph with this interaction
        self.shadow.record_interaction(intent, unified_message, context)

        RETURN SoulResponse(
            message=unified_message,
            faculty_used=self._primary_faculty(raw_outputs),
            raw_outputs=raw_outputs,
            cognitive_state=state,
            proactive=False,
            ihsan_score=ihsan,
            snr_score=self._compute_snr(unified_message),
            latency_ms=elapsed,
        )

    ASYNC METHOD perceive_user_state() -> CognitiveState:
        """
        Jarvis-like awareness: sense mood, entropy, rhythm.

        Uses the 5D entropy manifold (Shannon + Structural +
        Behavioral + Hypothetical + Contextual) to classify
        the user's current cognitive state.
        """
        manifold = AWAIT self.entropy.compute_manifold()
        total = manifold.total_entropy
        RETURN classify_state(total)

    METHOD find_the_unspoken() -> Optional[ProactiveInsight]:
        """
        Proactive Coworking: Find the 'coffee order' of the task.

        Uses Shannon deviation to detect what's MISSING from the
        user's current context — not what they asked for, but
        what they'll need in 5 minutes.

        Example: User is writing a doc but forgot to open the
        reference table → RSL surfaces it before they realize.
        """
        RETURN self.coworker.detect_unspoken_need()

    # ─── Background Awareness Loop ─────────────────────────

    ASYNC METHOD _awareness_loop():
        """
        30-second OODA cycle for continuous user awareness.
        Mirrors ProactiveSovereignEntity.ooda_loop but at the
        relational (not task) level.
        """
        WHILE self._active:
            TRY:
                state = AWAIT self.perceive_user_state()

                # Check for proactive intervention opportunities
                insight = self.find_the_unspoken()
                IF insight AND insight.urgency > 0.7:
                    AWAIT self._deliver_proactive(insight)

                # Update coupling metrics
                self.coupling.record_state(state)

                AWAIT sleep(30)  # Match OODA cycle
            EXCEPT Exception as e:
                LOG.warning("Awareness cycle error: %s", e)
                AWAIT sleep(60)  # Back off on error
```

## 5. Configuration (`core/rsl/config.py`)

```pseudocode
MODULE core.rsl.config

IMPORT UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD FROM core.integration.constants

DATACLASS RSLConfig:
    """Configuration for the Reflective Soul Layer."""

    # Entropy thresholds (from UERS)
    chaos_threshold: float = 4.8       # Above this → OVERWHELMED
    flow_threshold: float = 2.0        # Below this → DEEP_FLOW

    # Proactive intervention
    proactive_urgency_gate: float = 0.7  # Minimum urgency to self-initiate
    awareness_cycle_seconds: int = 30    # OODA cycle frequency
    max_proactive_per_hour: int = 5      # Don't spam the user

    # Quality gates
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD  # 0.95
    snr_threshold: float = UNIFIED_SNR_THRESHOLD      # 0.85

    # Persona
    default_voice: str = "Companion"   # Warm, concise, respectful
    trust_decay_hours: float = 168.0   # Trust fades after 7 days idle
    trust_growth_rate: float = 0.01    # Per successful interaction

    # Shadow Graph
    shadow_graph_path: str = "sovereign_state/shadow_graph.db"
    max_rhythm_history: int = 1000     # Max interaction records

    # Structural Coupling
    coupling_evolution_hours: float = 24.0  # Evaluate evolution daily
    persona_adaptation_rate: float = 0.05   # How fast persona adapts
```

## 6. Key Design Decisions

| Decision | Rationale | Giant |
|----------|-----------|-------|
| RSL wraps PAT, never replaces | Autopoietic boundary — the organs remain; the soul emerges | Maturana-Varela |
| 5-state cognitive classification | Maps directly to intervention strategies | Shannon |
| 30-second awareness cycle | Matches existing OODA loop in ProactiveSovereignEntity | Boyd |
| Shadow Graph in AgentDB | Reuses V3 HNSW + SQLite hybrid search (Phase 20) | Phase 20 ADR-006 |
| Ihsan gate on ALL outputs | No soul output bypasses constitutional check | Al-Ghazali |
| Trust earned, not assumed | Trust starts at 0.0 and grows through correct behavior | Anthropic |
| Proactive rate limit (5/hr) | Prevent the soul from becoming noise | Shannon (SNR) |

## 7. Integration Points

| Existing System | Integration | Direction |
|----------------|-------------|-----------|
| `core/pat/agent.py` | PAT engine feeds raw outputs to RSL | RSL ← PAT |
| `core/memory/agent_db.py` | Shadow Graph stored via AgentDB | RSL → Memory |
| `core/uers/entropy.py` | 5D manifold feeds cognitive state | RSL ← UERS |
| `core/sovereign/proactive_integration.py` | RSL replaces raw OODA output formatting | RSL ↔ Proactive |
| `core/personaplex/engine.py` | Persona voice synthesis channel | RSL → PersonaPlex |
| `core/iaas/snr_v2.py` | SNR scoring for output quality | RSL ← SNR |
| `core/autopoiesis/loop_engine.py` | Structural coupling feeds back to autopoiesis | RSL → Autopoiesis |
| `core/elite/hooks.py` | RSL events registered as hook triggers | RSL → Hooks |
| `core/integration/constants.py` | All thresholds imported (never hardcoded) | RSL ← Constants |
