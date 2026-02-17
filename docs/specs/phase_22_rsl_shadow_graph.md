# Phase 22: Shadow Graph — Persistent User Modeling & Entropy Sensing

> Standing on Giants: Shannon (Entropy, 1948) · Jung (Shadow/Individuation, 1934) · Kahneman (Dual Process, 2003) · Maturana-Varela (Structural Coupling, 1972) · Besta (Graph-of-Thoughts, 2024)

## 1. Overview

The Shadow Graph is a persistent, entropy-weighted model of the user's cognitive patterns, rhythms, and relationship with the system. Unlike a database that stores *what the user did*, the Shadow Graph stores *who the user is becoming* — their evolving habits, stress patterns, creative rhythms, and trust trajectory.

**Naming (Jung):** The "Shadow" is not darkness — it is the unconscious partner. The Shadow Graph maps the user's unspoken patterns that the RSL uses to anticipate needs before they surface.

## 2. Module: `core/rsl/shadow_graph.py`

```pseudocode
MODULE core.rsl.shadow_graph

IMPORT AgentDB FROM core.memory.agent_db
IMPORT ManifoldState, EntropyMeasurement FROM core.uers.entropy
IMPORT CognitiveState, UserRhythm, RelationalContext FROM core.rsl.types

CLASS ShadowGraph:
    """
    Persistent user model — the 'unconscious' of the RSL.

    Three data layers:
    1. RHYTHM LAYER — temporal patterns (peak hours, cadence, session lengths)
    2. ENTROPY LAYER — cognitive load history (manifold snapshots over time)
    3. COUPLING LAYER — relationship evolution (trust, persona fit, satisfaction)

    Storage: AgentDB (HNSW vectors + SQLite FTS5 metadata)
    The Shadow Graph stores embeddings of *interaction patterns*, not
    raw text — it learns the *shape* of how you work.
    """

    CONSTRUCTOR(memory: AgentDB, config: RSLConfig = DEFAULT):
        self.memory = memory
        self.config = config
        self._rhythm = UserRhythm(defaults)
        self._entropy_history: RingBuffer[ManifoldSnapshot] = RingBuffer(1000)
        self._trust_score: float = 0.0
        self._interaction_count: int = 0
        self._loaded = False

    # ═══════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════

    METHOD load():
        """Load Shadow Graph from AgentDB."""
        rhythm_record = self.memory.retrieve("shadow:rhythm")
        IF rhythm_record:
            self._rhythm = deserialize_rhythm(rhythm_record.content)

        trust_record = self.memory.retrieve("shadow:trust")
        IF trust_record:
            self._trust_score = float(trust_record.content)

        # Load recent entropy history
        history = self.memory.search_by_tag("shadow:entropy", limit=1000)
        FOR record IN history:
            self._entropy_history.append(deserialize_snapshot(record))

        self._loaded = True
        LOG "Shadow Graph loaded: trust=%.2f, history=%d", self._trust_score, len(self._entropy_history)

    METHOD save():
        """Persist Shadow Graph to AgentDB."""
        self.memory.store(
            id="shadow:rhythm",
            content=serialize_rhythm(self._rhythm),
            tags=["shadow", "rhythm"],
        )
        self.memory.store(
            id="shadow:trust",
            content=str(self._trust_score),
            tags=["shadow", "trust"],
        )
        # Entropy history is stored incrementally (not bulk saved)
        LOG "Shadow Graph saved"

    # ═══════════════════════════════════════════════════════
    # RHYTHM LAYER — Temporal Pattern Learning
    # ═══════════════════════════════════════════════════════

    METHOD update_rhythm(interaction_time: datetime, session_context: Dict):
        """
        Learn the user's temporal patterns.

        Tracks: hour-of-day productivity, session length distribution,
        command cadence, project switching frequency.
        """
        hour = interaction_time.hour

        # Update peak hours distribution
        self._rhythm.hour_distribution[hour] += 1

        # Update interaction cadence (exponential moving average)
        IF self._last_interaction_time:
            gap_seconds = (interaction_time - self._last_interaction_time).total_seconds()
            IF gap_seconds > 0:
                cadence = 60.0 / gap_seconds  # commands per minute
                alpha = 0.1  # EMA smoothing factor
                self._rhythm.interaction_cadence = (
                    alpha * cadence + (1 - alpha) * self._rhythm.interaction_cadence
                )

        self._last_interaction_time = interaction_time
        self._interaction_count += 1

    METHOD detect_peak_hours() -> List[int]:
        """
        Identify the user's most productive hours.
        Uses Shannon entropy of the hour distribution to find
        concentrated vs. distributed work patterns.
        """
        total = sum(self._rhythm.hour_distribution.values())
        IF total < 10: RETURN []  # Not enough data

        probs = [count / total FOR count IN self._rhythm.hour_distribution.values()]
        mean_prob = 1.0 / 24

        peaks = []
        FOR hour, count IN self._rhythm.hour_distribution.items():
            IF (count / total) > mean_prob * 1.5:  # 50% above average
                peaks.append(hour)

        RETURN sorted(peaks)

    # ═══════════════════════════════════════════════════════
    # ENTROPY LAYER — Cognitive Load Tracking
    # ═══════════════════════════════════════════════════════

    METHOD record_entropy_snapshot(manifold: ManifoldState):
        """
        Store a point-in-time entropy measurement.

        The entropy history enables:
        - Trend detection (is the user getting MORE stressed over time?)
        - Pattern matching (does this entropy signature match a known stress trigger?)
        - Proactive timing (when entropy is rising, intervene BEFORE peak)
        """
        snapshot = ManifoldSnapshot(
            timestamp=now(),
            surface=manifold.surface.normalized,
            structural=manifold.structural.normalized,
            behavioral=manifold.behavioral.normalized,
            hypothetical=manifold.hypothetical.normalized,
            contextual=manifold.contextual.normalized,
            total=manifold.total_entropy,
            classified_state=classify_state(manifold.total_entropy),
        )

        self._entropy_history.append(snapshot)

        # Store in AgentDB for long-term pattern analysis
        self.memory.store(
            id=f"shadow:entropy:{snapshot.timestamp.isoformat()}",
            content=serialize_snapshot(snapshot),
            tags=["shadow", "entropy", snapshot.classified_state.name],
            importance=0.3,  # Low importance — bulk data
        )

    METHOD compute_entropy_trend(window_minutes: int = 30) -> float:
        """
        Compute the derivative of entropy over a time window.

        Returns:
            Positive = entropy rising (user getting more scattered)
            Negative = entropy falling (user entering flow)
            Near zero = stable state

        Standing on Giants: Shannon (1948) — entropy as a measure of
        disorder applied to cognitive state over time.
        """
        recent = self._entropy_history.get_window(window_minutes)
        IF len(recent) < 3: RETURN 0.0  # Not enough data

        # Simple linear regression slope
        x = [s.timestamp.timestamp() FOR s IN recent]
        y = [s.total FOR s IN recent]
        slope = linear_regression_slope(x, y)

        RETURN slope

    METHOD detect_stress_signature() -> Optional[str]:
        """
        Match current entropy pattern against known stress signatures.

        A "stress signature" is a characteristic entropy trajectory:
        - Rapid surface entropy increase (window switching)
        - Structural entropy spike (project switching)
        - Behavioral entropy jump (erratic command patterns)
        """
        trend = self.compute_entropy_trend(window_minutes=15)
        current = self._entropy_history.latest()

        IF current IS None: RETURN None

        # Signature: "Rapid Scatter" — surface + behavioral spike
        IF current.surface > 0.8 AND current.behavioral > 0.7 AND trend > 0.05:
            RETURN "rapid_scatter"

        # Signature: "Context Overload" — all dimensions elevated
        IF current.total > 4.0 AND all(d > 0.6 FOR d IN current.dimensions()):
            RETURN "context_overload"

        # Signature: "Deep Rabbit Hole" — hypothetical high, others low
        IF current.hypothetical > 0.8 AND current.surface < 0.3:
            RETURN "deep_rabbit_hole"

        RETURN None

    # ═══════════════════════════════════════════════════════
    # COUPLING LAYER — Trust & Relationship
    # ═══════════════════════════════════════════════════════

    METHOD update_trust(feedback_signal: float):
        """
        Update trust level based on interaction outcome.

        Trust grows through correct anticipation and shrinks through
        unwanted interruption or incorrect suggestions.

        Args:
            feedback_signal: +1.0 (accepted/helpful) to -1.0 (rejected/harmful)
        """
        # Trust grows slowly, decays quickly (asymmetric by design)
        IF feedback_signal > 0:
            delta = self.config.trust_growth_rate * feedback_signal
        ELSE:
            delta = self.config.trust_growth_rate * feedback_signal * 3  # 3x decay

        self._trust_score = clamp(self._trust_score + delta, 0.0, 1.0)

    METHOD apply_trust_decay():
        """
        Trust decays over inactivity.
        After config.trust_decay_hours (default: 168h = 7 days) of
        no interaction, trust returns to 0.
        """
        IF self._last_interaction_time IS None: RETURN

        hours_idle = hours_since(self._last_interaction_time)
        IF hours_idle > 0:
            decay_factor = max(0, 1.0 - (hours_idle / self.config.trust_decay_hours))
            self._trust_score *= decay_factor

    # ═══════════════════════════════════════════════════════
    # RELATIONAL CONTEXT ASSEMBLY
    # ═══════════════════════════════════════════════════════

    METHOD get_relational_context(state: CognitiveState) -> RelationalContext:
        """
        Assemble the full relational context for the Persona Transformer.

        This is the "System 1" input — the intuitive, relational frame
        that wraps around the "System 2" analytical output of the PAT agents.
        """
        self.apply_trust_decay()

        RETURN RelationalContext(
            cognitive_state=state,
            entropy_manifold=self._entropy_history.latest_manifold(),
            rhythm=self._rhythm,
            active_project=self._detect_active_project(),
            session_duration_minutes=self._session_duration(),
            trust_level=self._trust_score,
            ihsan_score=self._compute_coupling_ihsan(),
        )

    # ═══════════════════════════════════════════════════════
    # INTERACTION RECORDING
    # ═══════════════════════════════════════════════════════

    METHOD record_interaction(
        user_intent: str,
        soul_response: str,
        context: RelationalContext,
    ):
        """
        Record a complete interaction for pattern learning.

        Stores the interaction as a vector embedding in AgentDB for
        future similarity search (finding similar past interactions
        to inform current response).
        """
        self.memory.store(
            id=f"shadow:interaction:{now().isoformat()}",
            content=f"USER: {user_intent}\nRSL: {soul_response}",
            tags=["shadow", "interaction", context.cognitive_state.name],
            importance=0.5 + (0.3 * context.trust_level),  # Higher trust = more important
            metadata={
                "state": context.cognitive_state.name,
                "project": context.active_project,
                "trust": context.trust_level,
                "session_min": context.session_duration_minutes,
            },
        )

        self.update_rhythm(now(), {"intent": user_intent})
```

## 3. Entropy Sensor Bridge (`core/rsl/entropy_sensor.py`)

```pseudocode
MODULE core.rsl.entropy_sensor

IMPORT EntropyCalculator, ManifoldState FROM core.uers.entropy
IMPORT CognitiveState, classify_state FROM core.rsl.types

CLASS EntropySensor:
    """
    Bridge between the UERS 5D Entropy Manifold and the RSL.

    The UERS provides raw entropy measurements across 5 dimensions:
    - Surface (Shannon): Character/token entropy
    - Structural: Graph topology entropy
    - Behavioral: Action pattern entropy
    - Hypothetical: Path entropy (GoT branches)
    - Contextual: Context window entropy

    The EntropySensor translates these into cognitive load signals
    and provides real-time trend analysis for the Shadow Graph.
    """

    CONSTRUCTOR(calculator: EntropyCalculator):
        self._calc = calculator
        self._last_manifold: Optional[ManifoldState] = None
        self._manifold_history: RingBuffer[ManifoldState] = RingBuffer(100)

    ASYNC METHOD sense() -> Tuple[CognitiveState, ManifoldState]:
        """
        Perform a single entropy measurement and classify state.

        Returns:
            (classified_state, raw_manifold)
        """
        manifold = AWAIT self._calc.compute_manifold()
        self._last_manifold = manifold
        self._manifold_history.append(manifold)

        state = classify_state(manifold.total_entropy)
        RETURN (state, manifold)

    METHOD get_chaos_signature() -> Dict[str, float]:
        """
        Extract the chaos signature — a normalized vector of the
        5 entropy dimensions that characterizes the *type* of chaos.

        High surface + low structural = "scattered but single-project"
        High structural + high behavioral = "project-hopping panic"
        High hypothetical + low surface = "deep thought spiral"
        """
        IF self._last_manifold IS None:
            RETURN {"surface": 0, "structural": 0, "behavioral": 0,
                    "hypothetical": 0, "contextual": 0}

        m = self._last_manifold
        RETURN {
            "surface": m.surface.normalized,
            "structural": m.structural.normalized,
            "behavioral": m.behavioral.normalized,
            "hypothetical": m.hypothetical.normalized,
            "contextual": m.contextual.normalized,
        }

    METHOD is_entropy_rising(window: int = 5) -> bool:
        """Check if entropy is trending upward over last N measurements."""
        IF len(self._manifold_history) < window: RETURN False
        recent = self._manifold_history.last(window)
        values = [m.total_entropy FOR m IN recent]
        RETURN values[-1] > values[0] + 0.5  # Significant rise
```

## 4. Data Flow

```
User Interaction
       │
       ▼
┌──────────────┐     ┌───────────────┐
│ UERS 5D      │────▶│ EntropySensor │
│ Entropy Calc │     │ (Bridge)      │
└──────────────┘     └───────┬───────┘
                             │
                    ┌────────▼────────┐
                    │  Shadow Graph   │
                    │ ┌─────────────┐ │
                    │ │ RHYTHM      │ │ ← peak hours, cadence, obsession
                    │ │ LAYER       │ │
                    │ ├─────────────┤ │
                    │ │ ENTROPY     │ │ ← manifold snapshots, trends
                    │ │ LAYER       │ │
                    │ ├─────────────┤ │
                    │ │ COUPLING    │ │ ← trust score, satisfaction
                    │ │ LAYER       │ │
                    │ └─────────────┘ │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Relational      │
                    │ Context         │─────▶ Persona Transformer
                    └─────────────────┘
```

## 5. TDD Anchors

```pseudocode
TEST "Shadow Graph loads and saves rhythm data":
    graph = ShadowGraph(memory=mock_agentdb)
    graph._rhythm.peak_hours = [9, 10, 14, 15]
    graph.save()

    graph2 = ShadowGraph(memory=mock_agentdb)
    graph2.load()
    ASSERT graph2._rhythm.peak_hours == [9, 10, 14, 15]

TEST "Entropy trend detects rising chaos":
    graph = ShadowGraph(memory=mock_agentdb)
    # Inject rising entropy snapshots
    FOR i IN range(10):
        graph.record_entropy_snapshot(make_manifold(total=2.0 + i * 0.3))
    trend = graph.compute_entropy_trend(window_minutes=30)
    ASSERT trend > 0  # Rising

TEST "Trust decays over 7 days":
    graph = ShadowGraph(memory=mock_agentdb)
    graph._trust_score = 0.8
    graph._last_interaction_time = 7_days_ago
    graph.apply_trust_decay()
    ASSERT graph._trust_score < 0.1  # Nearly zero after 7 days

TEST "Stress signature detects rapid scatter":
    sensor = EntropySensor(calculator=mock_calc)
    # Inject high surface + high behavioral manifold
    manifold = make_manifold(surface=0.9, behavioral=0.8, total=4.5)
    sensor._last_manifold = manifold
    sensor._manifold_history.append(manifold)

    graph = ShadowGraph(memory=mock_agentdb)
    graph.record_entropy_snapshot(manifold)
    sig = graph.detect_stress_signature()
    ASSERT sig == "rapid_scatter"

TEST "Cognitive state classification":
    ASSERT classify_state(1.5) == CognitiveState.DEEP_FLOW
    ASSERT classify_state(3.0) == CognitiveState.ENGAGED_WORK
    ASSERT classify_state(4.0) == CognitiveState.MILD_SCATTER
    ASSERT classify_state(4.5) == CognitiveState.HIGH_CHAOS
    ASSERT classify_state(5.2) == CognitiveState.OVERWHELMED
```
