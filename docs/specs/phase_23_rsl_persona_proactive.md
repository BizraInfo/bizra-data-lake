# Phase 23: Persona Transformer & Proactive Coworking Engine

> Standing on Giants: Kahneman (System 1/System 2, 2003) · Csikszentmihalyi (Flow, 1990) · Shannon (Channel Capacity, 1948) · Al-Ghazali (Ihsan, 1095) · Maturana-Varela (Autopoiesis, 1972) · Boyd (OODA, 1976)

## 1. Overview

### Persona Transformer
The Persona Transformer is the "one voice" filter that prevents the user from feeling the seams between 7 PAT agents. Every output from the PAT is channeled through a single relational context, producing responses that sound like they come from one companion — not a committee.

### Proactive Coworking Engine
The Proactive Coworker finds the "coffee order" of the task — the unspoken need the user hasn't articulated yet. Using Shannon deviation analysis, it detects what's *missing* from the user's current context and surfaces it at the right moment.

**Design Principle (Al-Baqarah Multiplier):** A small gesture of understanding from the agent ("I noticed you're typing faster, I've cleared the distractions") has a 700x impact on trust compared to raw task output.

## 2. Persona Transformer (`core/rsl/persona_transformer.py`)

```pseudocode
MODULE core.rsl.persona_transformer

IMPORT CognitiveState, RelationalContext, SoulFaculty FROM core.rsl.types
IMPORT UNIFIED_IHSAN_THRESHOLD FROM core.integration.constants

# ═══════════════════════════════════════════════════════════
# VOICE TEMPLATES — The "tone of voice" for each state
# ═══════════════════════════════════════════════════════════

MAPPING VOICE_TEMPLATES = {
    # When user is in deep flow — minimal, respectful
    CognitiveState.DEEP_FLOW: {
        "prefix": "",            # No greeting — don't break flow
        "style": "terse",        # Maximum brevity
        "emoji": False,
        "meta_info": False,      # No performance stats
    },

    # When user is engaged — concise but warm
    CognitiveState.ENGAGED_WORK: {
        "prefix": "",
        "style": "concise",
        "emoji": False,
        "meta_info": False,
    },

    # When user is scattering — anchoring, calming
    CognitiveState.MILD_SCATTER: {
        "prefix": "",
        "style": "structured",   # Use headers/bullets to impose order
        "emoji": False,
        "meta_info": False,
    },

    # When user is in chaos — proactive simplification
    CognitiveState.HIGH_CHAOS: {
        "prefix": "Let me simplify this. ",
        "style": "directive",    # Clear, single next-step
        "emoji": False,
        "meta_info": False,
    },

    # When user is overwhelmed — maximal care
    CognitiveState.OVERWHELMED: {
        "prefix": "I've got this. ",
        "style": "calming",      # Warm, reassuring, minimal
        "emoji": False,
        "meta_info": False,
    },
}

# ═══════════════════════════════════════════════════════════
# PERSONA TRANSFORMER
# ═══════════════════════════════════════════════════════════

CLASS PersonaTransformer:
    """
    Filters 7 agent outputs into a single unified voice.

    Design: System 1 (intuitive relational wrap) around
    System 2 (analytical agent outputs).

    The transformer doesn't change WHAT is said — it changes
    HOW it's said based on the user's current state.
    """

    CONSTRUCTOR(config: RSLConfig = DEFAULT):
        self.config = config
        self._persona_memory: Dict[str, str] = {}  # Learned preferences

    METHOD transform(
        raw_outputs: Dict[str, Any],
        context: RelationalContext,
        voice: str = "Companion",
    ) -> str:
        """
        Transform raw PAT agent outputs into a unified voice.

        Args:
            raw_outputs: Dict from PAT engine (agent_name -> output)
            context: Current relational context from Shadow Graph
            voice: Voice alignment mode

        Returns:
            Single unified message string

        Flow:
        1. Select voice template based on cognitive state
        2. Extract primary result from raw outputs
        3. Apply relational wrapping
        4. Check Ihsan gate
        """
        template = VOICE_TEMPLATES[context.cognitive_state]

        # Step 1: Extract the core message from agent outputs
        primary = self._extract_primary(raw_outputs)
        summary = primary.get("summary", "")
        details = primary.get("details", "")

        # Step 2: Apply style transformation
        styled = self._apply_style(summary, details, template["style"])

        # Step 3: Add relational prefix (state-aware)
        prefix = template["prefix"]
        IF context.trust_level > 0.8 AND context.cognitive_state == CognitiveState.ENGAGED_WORK:
            # High trust + normal work = just the answer, no fluff
            prefix = ""

        # Step 4: Assemble
        message = f"{prefix}{styled}".strip()

        # Step 5: Apply user preferences from persona memory
        message = self._apply_preferences(message, context)

        RETURN message

    METHOD elevate(message: str, context: RelationalContext) -> str:
        """
        Elevate a message that failed the Ihsan gate.

        Re-writes with higher care, more grounding, and explicit
        acknowledgment of the user's state.
        """
        IF context.cognitive_state IN [CognitiveState.HIGH_CHAOS, CognitiveState.OVERWHELMED]:
            RETURN f"I've handled the core task. {self._simplify(message)} Focus on what matters most right now."
        ELSE:
            RETURN f"{message} Let me know if you need anything adjusted."

    METHOD _apply_style(summary: str, details: str, style: str) -> str:
        """Apply voice style to content."""
        MATCH style:
            CASE "terse":
                RETURN summary  # Just the answer
            CASE "concise":
                IF details:
                    RETURN f"{summary}\n\n{self._abbreviate(details)}"
                RETURN summary
            CASE "structured":
                RETURN self._structure_as_list(summary, details)
            CASE "directive":
                RETURN self._extract_next_step(summary, details)
            CASE "calming":
                RETURN self._gentle_wrap(summary)
            DEFAULT:
                RETURN summary

    METHOD _extract_primary(raw_outputs: Dict[str, Any]) -> Dict[str, str]:
        """
        Select the primary agent output.

        Priority: summary > synthesis > first non-empty result
        """
        IF "summary" IN raw_outputs:
            RETURN raw_outputs["summary"]
        IF "synthesis" IN raw_outputs:
            RETURN {"summary": raw_outputs["synthesis"]}
        FOR key, value IN raw_outputs.items():
            IF value AND isinstance(value, str):
                RETURN {"summary": value}
        RETURN {"summary": "Task completed."}

    METHOD _apply_preferences(message: str, context: RelationalContext) -> str:
        """Apply learned user preferences to message."""
        tone = context.rhythm.preferred_voice_tone

        MATCH tone:
            CASE "concise":
                # Remove filler phrases
                RETURN self._strip_filler(message)
            CASE "detailed":
                # Keep full output
                RETURN message
            CASE "warm":
                # Add relational markers
                RETURN message
            CASE "direct":
                # Remove hedging ("I think", "perhaps")
                RETURN self._strip_hedging(message)
            DEFAULT:
                RETURN message

    # ─── Style Helpers ────────────────────────────────────

    STATIC METHOD _simplify(text: str) -> str:
        """Reduce to single sentence summary."""
        sentences = split_sentences(text)
        IF len(sentences) > 1:
            RETURN sentences[0]
        RETURN text

    STATIC METHOD _abbreviate(text: str) -> str:
        """Keep first 3 lines of details."""
        lines = text.split("\n")
        IF len(lines) > 3:
            RETURN "\n".join(lines[:3]) + "\n..."
        RETURN text

    STATIC METHOD _structure_as_list(summary: str, details: str) -> str:
        """Convert prose into bullet points for scattered users."""
        points = split_to_points(details)
        IF points:
            bullet_list = "\n".join(f"- {p}" FOR p IN points[:5])
            RETURN f"{summary}\n\n{bullet_list}"
        RETURN summary

    STATIC METHOD _extract_next_step(summary: str, details: str) -> str:
        """Extract single most important next action."""
        RETURN f"Next step: {summary.split('.')[0]}."

    STATIC METHOD _gentle_wrap(text: str) -> str:
        """Warm, reassuring framing for overwhelmed users."""
        RETURN f"{text} One thing at a time."
```

## 3. Proactive Coworking Engine (`core/rsl/proactive_coworker.py`)

```pseudocode
MODULE core.rsl.proactive_coworker

IMPORT ShadowGraph FROM core.rsl.shadow_graph
IMPORT EntropySensor FROM core.rsl.entropy_sensor
IMPORT ProactiveInsight, SoulFaculty, CognitiveState FROM core.rsl.types
IMPORT UNIFIED_IHSAN_THRESHOLD FROM core.integration.constants

CLASS ProactiveCoworker:
    """
    Anticipatory task engine — finds what's MISSING before the user asks.

    The "coffee order" metaphor: A great coworker doesn't wait for you
    to ask for coffee. They notice you've been staring at code for 2
    hours and bring it. The Proactive Coworker does the same for
    information, context, and environmental adjustments.

    Standing on Giants:
    - Shannon (deviation): What's expected vs. what's present = gap = need
    - Csikszentmihalyi (flow): Protect flow state, reduce interruption
    - Boyd (OODA): Observe → Orient → Decide → Act proactively
    """

    CONSTRUCTOR(
        shadow: ShadowGraph,
        entropy: EntropySensor,
        config: RSLConfig = DEFAULT,
    ):
        self.shadow = shadow
        self.entropy = entropy
        self.config = config
        self._interventions_this_hour: int = 0
        self._last_hour_reset: datetime = now()
        self._active = False

    # ═══════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════

    ASYNC METHOD start():
        """Begin proactive awareness cycle."""
        self._active = True
        SPAWN self._proactive_loop()

    METHOD stop():
        self._active = False

    # ═══════════════════════════════════════════════════════
    # PROACTIVE DETECTION — The "Unspoken Need" Finder
    # ═══════════════════════════════════════════════════════

    METHOD detect_unspoken_need() -> Optional[ProactiveInsight]:
        """
        Use Shannon deviation to find what's MISSING.

        Algorithm:
        1. Get current context (what the user is working on)
        2. Get expected context (what's typically needed for this task)
        3. Compute deviation = expected - current
        4. If deviation exceeds threshold → insight

        Example:
        - User writing a doc → Expected: reference table open
        - Reference table NOT open → Deviation detected
        - Insight: "You might need the reference table for this doc"
        """
        context = self.shadow.get_relational_context(
            self.entropy._last_manifold.total_entropy
            IF self.entropy._last_manifold
            ELSE 3.0
        )

        insights = []

        # Check 1: Entropy-based intervention
        entropy_insight = self._check_entropy_intervention(context)
        IF entropy_insight: insights.append(entropy_insight)

        # Check 2: Context gap detection
        gap_insight = self._check_context_gaps(context)
        IF gap_insight: insights.append(gap_insight)

        # Check 3: Session duration check
        duration_insight = self._check_session_health(context)
        IF duration_insight: insights.append(duration_insight)

        # Return highest urgency insight (if any pass the gate)
        IF insights:
            best = max(insights, key=LAMBDA i: i.urgency)
            IF best.urgency > self.config.proactive_urgency_gate:
                IF self._rate_limit_ok():
                    RETURN best

        RETURN None

    METHOD _check_entropy_intervention(context: RelationalContext) -> Optional[ProactiveInsight]:
        """
        Check if entropy state warrants proactive intervention.

        Interventions:
        - HIGH_CHAOS → "I've consolidated X into Y. Focus on Z."
        - OVERWHELMED → "I've paused non-critical tasks. Here's your priority."
        - Rising entropy → "You're starting to scatter. Want me to simplify?"
        """
        state = context.cognitive_state
        trend = self.shadow.compute_entropy_trend(window_minutes=15)

        IF state == CognitiveState.OVERWHELMED:
            RETURN ProactiveInsight(
                insight="Cognitive overload detected",
                suggested_action="I can consolidate your active tasks into a priority list and pause background items.",
                urgency=0.95,
                faculty=SoulFaculty.PROTECTION,
                evidence=["entropy_total > 4.8", f"trend: {trend:.3f}"],
            )

        IF state == CognitiveState.HIGH_CHAOS AND trend > 0.05:
            RETURN ProactiveInsight(
                insight="Entropy rising — context fragmentation detected",
                suggested_action="Let me organize your open contexts. Which project should we focus on?",
                urgency=0.8,
                faculty=SoulFaculty.STABILITY,
                evidence=[f"entropy_total: {context.entropy_manifold.total_entropy:.2f}", f"trend: +{trend:.3f}/min"],
            )

        # Protect deep flow — suppress ALL proactive messages
        IF state == CognitiveState.DEEP_FLOW:
            RETURN None  # Sacred silence

        RETURN None

    METHOD _check_context_gaps(context: RelationalContext) -> Optional[ProactiveInsight]:
        """
        Detect missing context using Shadow Graph pattern matching.

        Looks at similar past sessions (via HNSW similarity search)
        and finds resources that were typically present but are
        currently absent.
        """
        IF NOT context.active_project: RETURN None

        # Search Shadow Graph for similar past sessions
        similar_sessions = self.shadow.memory.search(
            query=f"project:{context.active_project} state:{context.cognitive_state.name}",
            top_k=5,
        )

        IF len(similar_sessions) < 2: RETURN None  # Not enough history

        # Extract resources typically used in similar sessions
        typical_resources = extract_typical_resources(similar_sessions)
        current_resources = get_current_open_resources()  # From desktop bridge

        gaps = typical_resources - current_resources
        IF gaps:
            RETURN ProactiveInsight(
                insight=f"You usually have {next(iter(gaps))} open for this kind of work",
                suggested_action=f"Want me to open {next(iter(gaps))}?",
                urgency=0.5,
                faculty=SoulFaculty.CURIOSITY,
                evidence=[f"pattern_match: {len(similar_sessions)} similar sessions"],
            )

        RETURN None

    METHOD _check_session_health(context: RelationalContext) -> Optional[ProactiveInsight]:
        """
        Check if the user has been working too long without a break.

        Csikszentmihalyi's flow research: Sustained flow requires
        periodic recovery. Sessions > 90 minutes without a break
        show diminishing returns.
        """
        IF context.session_duration_minutes > 90:
            # Only suggest if not in deep flow
            IF context.cognitive_state != CognitiveState.DEEP_FLOW:
                RETURN ProactiveInsight(
                    insight=f"Session running {context.session_duration_minutes:.0f} minutes",
                    suggested_action="Consider a 5-minute break. I'll hold your place.",
                    urgency=0.4,
                    faculty=SoulFaculty.SERVICE,
                    evidence=[f"session_min: {context.session_duration_minutes:.0f}"],
                )

        RETURN None

    # ═══════════════════════════════════════════════════════
    # RATE LIMITING
    # ═══════════════════════════════════════════════════════

    METHOD _rate_limit_ok() -> bool:
        """
        Enforce max_proactive_per_hour.

        The soul must not become noise. Shannon's channel capacity
        theorem: exceeding capacity degrades ALL signal.
        """
        IF hours_since(self._last_hour_reset) >= 1.0:
            self._interventions_this_hour = 0
            self._last_hour_reset = now()

        IF self._interventions_this_hour >= self.config.max_proactive_per_hour:
            RETURN False

        self._interventions_this_hour += 1
        RETURN True

    # ═══════════════════════════════════════════════════════
    # PROACTIVE LOOP
    # ═══════════════════════════════════════════════════════

    ASYNC METHOD _proactive_loop():
        """
        Background loop that runs every OODA cycle (30s).

        This is the "always-on" awareness that makes the RSL
        feel like a companion rather than a command processor.
        """
        WHILE self._active:
            TRY:
                insight = self.detect_unspoken_need()
                IF insight AND NOT insight.suppressed:
                    # Deliver through the SovereignSoul's channel
                    EMIT event("rsl:proactive_insight", insight)

                AWAIT sleep(self.config.awareness_cycle_seconds)
            EXCEPT Exception as e:
                LOG.warning("Proactive loop error: %s", e)
                AWAIT sleep(60)
```

## 4. Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERACTION                          │
│                                                             │
│  "Fix the authentication bug in login.py"                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                  ┌─────────▼─────────┐
                  │  SovereignSoul    │
                  │  .speak(intent)   │
                  └─────────┬─────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
   ┌────────▼──────┐ ┌─────▼──────┐ ┌──────▼───────┐
   │ PAT Engine    │ │ Entropy    │ │ Shadow Graph │
   │ (7 Agents)    │ │ Sensor     │ │ (Context)    │
   │               │ │            │ │              │
   │ RESEARCHER    │ │ State:     │ │ Trust: 0.72  │
   │  → found bug  │ │ ENGAGED    │ │ Project: auth│
   │ WORKER        │ │            │ │ Cadence: 3/m │
   │  → fixed it   │ │            │ │              │
   │ VALIDATOR     │ │            │ │              │
   │  → tests pass │ │            │ │              │
   └────────┬──────┘ └─────┬──────┘ └──────┬───────┘
            │               │               │
            └───────────────┼───────────────┘
                            │
                  ┌─────────▼─────────────────┐
                  │  Persona Transformer       │
                  │                            │
                  │  State: ENGAGED_WORK       │
                  │  Trust: 0.72 (high)        │
                  │  Style: "concise"          │
                  │                            │
                  │  INPUT: "Found null-check  │
                  │   bypass in login handler. │
                  │   Patched line 142. Tests  │
                  │   pass (14/14)."           │
                  │                            │
                  │  OUTPUT: "Fixed the auth   │
                  │   bug — null-check bypass  │
                  │   at login.py:142.         │
                  │   All 14 tests pass."      │
                  └─────────┬─────────────────┘
                            │
                  ┌─────────▼─────────┐
                  │  Ihsan Gate       │
                  │  Score: 0.97      │
                  │  PASSED           │
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │  User receives    │
                  │  ONE voice,       │
                  │  not a committee  │
                  └───────────────────┘
```

## 5. Proactive Intervention Flow

```
┌──────────────────────────────────────────────────────────────┐
│  BACKGROUND: Proactive Coworker Loop (every 30s)             │
│                                                              │
│  t=0s   Entropy: 3.2 (ENGAGED_WORK)      → No action       │
│  t=30s  Entropy: 3.8 (MILD_SCATTER)      → Monitor         │
│  t=60s  Entropy: 4.3 (HIGH_CHAOS)        → Trend: +0.06    │
│  t=90s  Entropy: 4.6 (HIGH_CHAOS, rising)→ THRESHOLD MET   │
│                                                              │
│  Rate limit check: 2/5 this hour → OK                       │
│  Trust level: 0.72 → OK for suggestion                      │
│                                                              │
│  EMIT ProactiveInsight:                                      │
│    "Entropy rising — context fragmentation detected"         │
│    "Let me organize your open contexts. Which project        │
│     should we focus on?"                                     │
│    urgency=0.8, faculty=STABILITY                            │
│                                                              │
│  Delivered via Node0 Console:                                │
│    ┌──────────────────────────────────────────────────┐      │
│    │ 💡 You're juggling too much.                      │      │
│    │ Want me to organize your contexts?                │      │
│    │ [Yes, prioritize] [Not now] [Mute for 1h]       │      │
│    └──────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

## 6. TDD Anchors

```pseudocode
# ─── Persona Transformer Tests ────────────────────────────

TEST "Deep flow state produces terse output":
    transformer = PersonaTransformer()
    context = make_context(state=CognitiveState.DEEP_FLOW, trust=0.9)
    result = transformer.transform(
        raw_outputs={"summary": "Fixed bug in auth module. Tests pass."},
        context=context,
    )
    ASSERT len(result) < len("Fixed bug in auth module. Tests pass.") + 10
    ASSERT "Fixed bug" IN result

TEST "Overwhelmed state adds calming prefix":
    transformer = PersonaTransformer()
    context = make_context(state=CognitiveState.OVERWHELMED, trust=0.5)
    result = transformer.transform(
        raw_outputs={"summary": "Multiple issues found in codebase."},
        context=context,
    )
    ASSERT result.startswith("I've got this.")

TEST "High chaos extracts single next step":
    transformer = PersonaTransformer()
    context = make_context(state=CognitiveState.HIGH_CHAOS, trust=0.6)
    result = transformer.transform(
        raw_outputs={"summary": "Need to fix X, then Y, then Z."},
        context=context,
    )
    ASSERT "Next step" IN result OR "simplify" IN result.lower()

TEST "Ihsan elevation rewrites below-threshold messages":
    transformer = PersonaTransformer()
    context = make_context(state=CognitiveState.OVERWHELMED)
    original = "Error found."
    elevated = transformer.elevate(original, context)
    ASSERT len(elevated) > len(original)
    ASSERT "focus" IN elevated.lower() OR "matters" IN elevated.lower()

# ─── Proactive Coworker Tests ─────────────────────────────

TEST "No intervention during DEEP_FLOW":
    coworker = ProactiveCoworker(shadow=mock_shadow, entropy=mock_entropy)
    mock_entropy.state = CognitiveState.DEEP_FLOW
    insight = coworker.detect_unspoken_need()
    ASSERT insight IS None

TEST "Intervention triggered at OVERWHELMED":
    coworker = ProactiveCoworker(shadow=mock_shadow, entropy=mock_entropy)
    mock_entropy.state = CognitiveState.OVERWHELMED
    mock_shadow.entropy_trend = 0.08
    insight = coworker.detect_unspoken_need()
    ASSERT insight IS NOT None
    ASSERT insight.urgency > 0.9
    ASSERT insight.faculty == SoulFaculty.PROTECTION

TEST "Rate limiting caps at 5 per hour":
    coworker = ProactiveCoworker(shadow=mock_shadow, entropy=mock_entropy,
                                  config=RSLConfig(max_proactive_per_hour=5))
    FOR i IN range(5):
        ASSERT coworker._rate_limit_ok() == True
    ASSERT coworker._rate_limit_ok() == False  # 6th blocked

TEST "Session health suggests break after 90 minutes":
    coworker = ProactiveCoworker(shadow=mock_shadow, entropy=mock_entropy)
    context = make_context(session_duration_minutes=120, state=CognitiveState.ENGAGED_WORK)
    insight = coworker._check_session_health(context)
    ASSERT insight IS NOT None
    ASSERT "break" IN insight.suggested_action.lower()
```
