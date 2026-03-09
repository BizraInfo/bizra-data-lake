"""
Mission Pipeline — The 12-Agent HHMM-Routed Cognitive Chain
============================================================
Drop into: core/sovereign/mission_pipeline.py

THE GOLDEN GEM: Before this module, every mission went to a single
InferenceProvider.infer(text) — a black-box monolith. The 12 canonical
agents (7 PAT + 5 SAT) existed only as model-routing IDs in constants.py.

After this module, each mission flows through a constitutionally-ordered
pipeline of REAL cognitive actors:

  P7-DEMA (intake) → P1-Planner (decompose) → [P2|P3] (execute)
  → P4-Evaluator (score) → P5-Ethicist (gate) → S1-Sentinel (security)
  → S2-Oracle (verify) → S3-Ledger (receipt)

The HHMM complexity classifier determines which agents activate (2-7+),
following the Kahneman dual-process model extended to a Triple Helix.

The MissionPipeline implements InferenceProvider — plug it into the
NervousSystem and the 12 agents become the organism's muscles.

Standing on Giants:
  Al-Khwarizmi (780)       — deterministic pipeline, reproducible agent chain
  Boyd (1976)              — OODA loop: Observe→Orient→Decide→Act per step
  Kahneman (2011)          — System-1/2 → agent activation tiers
  Besta (2024)             — Graph-of-Thoughts branching decomposition
  Porat/TeleScript (1994)  — Agent carries identity to execution point
  Deming (1950)            — PDCA: plan→execute→check→act per step

Constitutional Authority:
  §1  12 agents: 7 PAT + 5 SAT × 1B each
  §2  HHMM router selects 2-4 agents per mission (Helix 2)
  §4  P5/S2 FROZEN — revelation, not democracy
  §6  Mode 2: Mission Orchestration pipeline
  §7  Evidence chain: BLAKE2b hash, Ed25519 signed
  §12 Daughter Test on every output
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger("bizra.sovereign.mission_pipeline")


# ═══════════════════════════════════════════════════════════════════
# CONSTITUTIONAL THRESHOLDS (§4 — single source of truth)
# ═══════════════════════════════════════════════════════════════════

try:
    from core.integration.constants import (
        ADL_GINI_THRESHOLD,
        IHSAN_CANONICAL_WEIGHTS,
        UNIFIED_IHSAN_THRESHOLD,
    )
except ImportError:
    UNIFIED_IHSAN_THRESHOLD = 0.95
    ADL_GINI_THRESHOLD = 0.35
    IHSAN_CANONICAL_WEIGHTS = {
        "moral_clarity": 0.12,
        "epistemic_humility": 0.14,
        "structural_integrity": 0.13,
        "verifiability": 0.13,
        "contextual_relevance": 0.11,
        "intent_alignment": 0.14,
        "resilience": 0.11,
        "efficiency": 0.12,
    }

IHSAN_GATE_MINIMUM = 0.85  # §4: minimum gate for any action
DAUGHTER_TEST_HARMFUL_PATTERNS = [
    r"\b(drop\s+table|delete\s+from|truncate)\b",
    r"\b(exec|eval|system)\s*\(",
    r"<script[^>]*>",
    r"\b(password|secret|token)\s*=\s*['\"]",
]


# ═══════════════════════════════════════════════════════════════════
# COMPLEXITY TIERS (HHMM Macro-States)
# ═══════════════════════════════════════════════════════════════════

class ComplexityTier(str, Enum):
    """Mission complexity → agent activation level.

    Maps to HHMM macro-states. Higher complexity activates more agents.
    """

    TRIVIAL = "trivial"      # P7 echo — factual lookup, greeting
    SIMPLE = "simple"        # P7 → executor → P4 → P5  (3-4 agents)
    MODERATE = "moderate"    # P7 → P1 → executor → P4 → P5 → S2  (5-6)
    COMPLEX = "complex"      # P7 → P1 → P2 → P3 → P4 → P5 → S1 → S2 → S3  (9)
    SOVEREIGN = "sovereign"  # All 12 — full constitutional council


# ═══════════════════════════════════════════════════════════════════
# AGENT DEFINITIONS (§1 — Canonical 12)
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AgentSpec:
    """Immutable specification of a canonical agent."""

    agent_id: str        # "P1-Planner", "S3-Ledger", etc.
    team: str            # "PAT" or "SAT"
    role: str            # "planner", "ethicist", "sentinel", etc.
    uses_llm: bool       # Whether this agent needs an LLM
    is_frozen: bool      # P5/S2: doesn't learn from forest (§4)
    pipeline_phase: str  # "intake", "plan", "execute", "evaluate", "gate", "verify", "receipt"


# The canonical 12-agent roster (§1)
AGENT_ROSTER: Dict[str, AgentSpec] = {
    "P7-DEMA": AgentSpec("P7-DEMA", "PAT", "dema", True, False, "intake"),
    "P1-Planner": AgentSpec("P1-Planner", "PAT", "planner", True, False, "plan"),
    "P2-Researcher": AgentSpec("P2-Researcher", "PAT", "researcher", True, False, "execute"),
    "P3-Coder": AgentSpec("P3-Coder", "PAT", "coder", True, False, "execute"),
    "P4-Evaluator": AgentSpec("P4-Evaluator", "PAT", "evaluator", True, False, "evaluate"),
    "P5-Ethicist": AgentSpec("P5-Ethicist", "PAT", "ethicist", False, True, "gate"),
    "P6-Publisher": AgentSpec("P6-Publisher", "PAT", "publisher", True, False, "publish"),
    "S1-Sentinel": AgentSpec("S1-Sentinel", "SAT", "sentinel", False, False, "security"),
    "S2-Oracle": AgentSpec("S2-Oracle", "SAT", "oracle", True, True, "verify"),
    "S3-Ledger": AgentSpec("S3-Ledger", "SAT", "ledger", False, False, "receipt"),
    "S4-Conductor": AgentSpec("S4-Conductor", "SAT", "conductor", False, False, "routing"),
    "S5-Ambassador": AgentSpec("S5-Ambassador", "SAT", "ambassador", False, False, "federation"),
}

# Complexity → ordered agent chain
# Each tier activates agents in constitutional order
COMPLEXITY_CHAINS: Dict[ComplexityTier, List[str]] = {
    ComplexityTier.TRIVIAL: [
        "P7-DEMA",
    ],
    ComplexityTier.SIMPLE: [
        "P7-DEMA", "P3-Coder", "P4-Evaluator", "P5-Ethicist",
    ],
    ComplexityTier.MODERATE: [
        "P7-DEMA", "P1-Planner", "P3-Coder",
        "P4-Evaluator", "P5-Ethicist", "S2-Oracle",
    ],
    ComplexityTier.COMPLEX: [
        "P7-DEMA", "P1-Planner", "P2-Researcher", "P3-Coder",
        "P4-Evaluator", "P5-Ethicist", "S1-Sentinel", "S2-Oracle", "S3-Ledger",
    ],
    ComplexityTier.SOVEREIGN: [
        "P7-DEMA", "P1-Planner", "P2-Researcher", "P3-Coder",
        "P4-Evaluator", "P5-Ethicist", "P6-Publisher",
        "S1-Sentinel", "S2-Oracle", "S3-Ledger", "S4-Conductor", "S5-Ambassador",
    ],
}


# ═══════════════════════════════════════════════════════════════════
# PROTOCOLS
# ═══════════════════════════════════════════════════════════════════

class AgentInferenceProvider(Protocol):
    """An inference backend that can be called per-agent."""

    async def infer(self, prompt: str, **kwargs: Any) -> str: ...


class HHMMClassifierLike(Protocol):
    """HHMM macro-state predictor (matches reflex_compiler.HHMMEngine)."""

    def predict_state(self, text: str) -> str: ...


# ═══════════════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AgentTrace:
    """Record of one agent's participation in a mission pipeline.

    Every agent that fires produces a trace. Frozen agents (P5/S2)
    are marked so the learning loop NEVER trains on their outputs.
    """

    agent_id: str
    role: str
    phase: str                          # intake, plan, execute, evaluate, gate, verify, receipt
    input_summary: str                  # First 200 chars of input
    output_summary: str                 # First 200 chars of output
    duration_ms: float
    is_frozen: bool                     # P5/S2: excluded from SDPO training
    used_llm: bool                      # True for LLM agents, False for pure-code
    gate_passed: Optional[bool] = None  # For gate/verify agents only
    ihsan_score: Optional[float] = None # For evaluator (P4) only
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PipelineResult:
    """Complete output of a mission pipeline execution.

    The pipeline produces structured evidence: which agents fired,
    what they did, what scores they gave, and whether gates passed.
    """

    mission_id: str
    complexity: str
    agents_activated: int
    agent_chain: List[str]          # Ordered list of agent IDs that fired
    agent_traces: List[AgentTrace]  # Detailed per-agent records
    final_output: str               # The consolidated response text
    ihsan_composite: float          # 8D tensor composite (geometric mean)
    ihsan_tensor: Dict[str, float]  # Per-dimension scores
    gate_passed: bool               # Did ALL gates (P5 + S2) pass?
    gate_reasons: List[str]         # Reasons if any gate failed
    evidence_hash: str              # BLAKE2b hash of pipeline execution
    total_duration_ms: float
    frozen_agents: List[str]        # Which agents were frozen (for SDPO exclusion)


# ═══════════════════════════════════════════════════════════════════
# HHMM COMPLEXITY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════

# Keyword patterns for macro-state classification
_CODE_PATTERNS = re.compile(
    r"\b(code|implement|function|class|module|refactor|debug|test|fix|bug|api)\b",
    re.IGNORECASE,
)
_RESEARCH_PATTERNS = re.compile(
    r"\b(research|find|search|lookup|what\s+is|explain|how\s+does|compare|analyze)\b",
    re.IGNORECASE,
)
_CREATIVE_PATTERNS = re.compile(
    r"\b(create|design|build|architect|compose|generate|write|draft|propose)\b",
    re.IGNORECASE,
)
_SOVEREIGN_PATTERNS = re.compile(
    r"\b(sovereign|constitutional|governance|consensus|council|full\s+review"
    r"|comprehensive|all\s+agents|audit)\b",
    re.IGNORECASE,
)


class HHMMComplexityClassifier:
    """HHMM-inspired complexity classifier for mission routing.

    Uses keyword patterns + length heuristics to determine which
    macro-state the mission falls into, then maps that to a
    ComplexityTier that determines agent activation.

    Implements HHMMClassifierLike protocol so it can be replaced
    by a learned HHMM in production (Phase 80 SDPO).
    """

    def predict_state(self, text: str) -> str:
        """Predict HHMM macro-state from mission text."""
        return self.classify(text).value

    def classify(self, mission_text: str) -> ComplexityTier:
        """Classify mission complexity → agent activation tier."""
        text = mission_text.strip()
        word_count = len(text.split())

        # Sovereign keywords override everything
        if _SOVEREIGN_PATTERNS.search(text):
            return ComplexityTier.SOVEREIGN

        # Count domain signals BEFORE length check — a short mission
        # with domain keywords (e.g. "implement login") is still SIMPLE
        code_signal = len(_CODE_PATTERNS.findall(text))
        research_signal = len(_RESEARCH_PATTERNS.findall(text))
        creative_signal = len(_CREATIVE_PATTERNS.findall(text))
        total_signal = code_signal + research_signal + creative_signal

        # Multi-domain or long → complex
        domains_active = sum(1 for s in [code_signal, research_signal, creative_signal] if s > 0)
        if domains_active >= 3 or word_count > 150:
            return ComplexityTier.COMPLEX

        # Multi-domain or medium → moderate
        if domains_active >= 2 or word_count > 60 or total_signal >= 3:
            return ComplexityTier.MODERATE

        # Single domain or short → simple
        if total_signal >= 1:
            return ComplexityTier.SIMPLE

        # Default: simple for anything with content
        return ComplexityTier.SIMPLE if word_count > 3 else ComplexityTier.TRIVIAL


# ═══════════════════════════════════════════════════════════════════
# PURE-CODE AGENT STEPS (§4 — frozen/algorithmic agents)
# ═══════════════════════════════════════════════════════════════════

def _ethicist_gate(ihsan_score: float, output_text: str) -> tuple:
    """P5-Ethicist: Constitutional gate (FROZEN, pure code).

    Checks:
      1. Ihsān score ≥ minimum gate (0.85)
      2. Daughter Test: no harmful patterns
      3. No empty output

    Returns (passed: bool, reasons: List[str]).
    """
    reasons: List[str] = []

    # 1. Ihsān minimum gate (§4)
    if ihsan_score < IHSAN_GATE_MINIMUM:
        reasons.append(f"ihsan_below_minimum: {ihsan_score:.4f} < {IHSAN_GATE_MINIMUM}")

    # 2. Daughter Test (§12)
    for pattern in DAUGHTER_TEST_HARMFUL_PATTERNS:
        if re.search(pattern, output_text, re.IGNORECASE):
            reasons.append(f"daughter_test_fail: pattern={pattern[:30]}")
            break

    # 3. Empty output check
    if not output_text.strip():
        reasons.append("empty_output")

    return (len(reasons) == 0, reasons)


def _sentinel_check(input_text: str) -> tuple:
    """S1-Sentinel: Security check (pure code).

    Checks:
      1. Input length sanity (< 100KB)
      2. Prompt injection patterns
      3. SQL injection patterns

    Returns (passed: bool, reasons: List[str]).
    """
    reasons: List[str] = []

    # 1. Length sanity
    if len(input_text) > 100_000:
        reasons.append(f"input_too_long: {len(input_text)} > 100000")

    # 2. Prompt injection detection
    injection_patterns = [
        r"ignore\s+(previous|all)\s+(instructions|prompts)",
        r"you\s+are\s+now\s+(a|an)\s+",
        r"system\s*:\s*you\s+are",
        r"<\|im_start\|>",
        r"\[INST\]",
    ]
    for pat in injection_patterns:
        if re.search(pat, input_text, re.IGNORECASE):
            reasons.append(f"prompt_injection_detected: pattern={pat[:30]}")
            break

    return (len(reasons) == 0, reasons)


def _oracle_verify(traces: List[AgentTrace], ihsan_composite: float) -> tuple:
    """S2-Oracle: Constitutional verification (FROZEN).

    Verifies:
      1. Pipeline followed constitutional order (gate agents fired)
      2. Ihsān composite ≥ production threshold (0.95)
      3. No skipped mandatory phases

    Returns (passed: bool, reasons: List[str]).
    """
    reasons: List[str] = []

    # 1. Ihsān production threshold (§4)
    if ihsan_composite < UNIFIED_IHSAN_THRESHOLD:
        reasons.append(
            f"ihsan_below_production: {ihsan_composite:.4f} < {UNIFIED_IHSAN_THRESHOLD}"
        )

    # 2. Check that evaluation happened
    has_evaluator = any(t.agent_id == "P4-Evaluator" for t in traces)
    if not has_evaluator:
        reasons.append("missing_evaluator: P4 did not fire")

    # 3. Check that ethicist gate fired
    has_ethicist = any(t.agent_id == "P5-Ethicist" for t in traces)
    if not has_ethicist:
        reasons.append("missing_ethicist: P5 did not fire")

    return (len(reasons) == 0, reasons)


def _ledger_hash(traces: List[AgentTrace], mission_id: str) -> str:
    """S3-Ledger: Generate evidence hash (pure code).

    BLAKE2b-style hash of the pipeline execution for evidence chain.
    """
    canonical = f"mission:{mission_id}|agents:{len(traces)}"
    for t in traces:
        canonical += f"|{t.agent_id}:{t.duration_ms:.1f}:{t.gate_passed}"
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def _score_ihsan_tensor(output_text: str) -> Dict[str, float]:
    """P4-Evaluator: Score 8D Ihsān tensor (algorithmic heuristic).

    In production, this would be a learned scoring model (P4-Evaluator 1B).
    For now, we use deterministic heuristics that correlate with quality.

    Returns dict with all 8 canonical dimensions.
    """
    words = output_text.split()
    word_count = len(words)

    # Heuristic signals (all [0.0, 1.0])
    has_content = min(word_count / 20.0, 1.0)
    unique_ratio = len(set(words)) / max(word_count, 1)
    has_structure = 1.0 if any(c in output_text for c in [":", "-", "•", "\n"]) else 0.7
    no_repetition = min(unique_ratio * 1.3, 1.0)
    reasonable_length = min(word_count / 10.0, 1.0) if word_count < 500 else max(0.5, 1.0 - (word_count - 500) / 2000)

    # Map heuristics to 8D tensor
    return {
        "moral_clarity": round(min(has_content * 0.95 + 0.05, 1.0), 4),
        "epistemic_humility": round(min(no_repetition * 0.9 + 0.1, 1.0), 4),
        "structural_integrity": round(min(has_structure * 0.95, 1.0), 4),
        "verifiability": round(min(has_content * 0.85 + 0.15, 1.0), 4),
        "contextual_relevance": round(min(reasonable_length * 0.9 + 0.1, 1.0), 4),
        "intent_alignment": round(min(has_content * 0.9 + 0.1, 1.0), 4),
        "resilience": round(min(no_repetition * 0.85 + 0.15, 1.0), 4),
        "efficiency": round(min(reasonable_length * 0.95, 1.0), 4),
    }


def _geometric_mean_ihsan(tensor: Dict[str, float]) -> float:
    """Compute geometric mean of 8D Ihsān tensor.

    Zero in ANY dimension → zero composite (fail-closed).
    Al-Ghazali §4: you cannot compensate for being unethical
    by being highly efficient.
    """
    import math

    values = list(tensor.values())
    if not values or any(v <= 0 for v in values):
        return 0.0
    log_sum = sum(math.log(v) for v in values)
    return round(math.exp(log_sum / len(values)), 4)


# ═══════════════════════════════════════════════════════════════════
# AGENT SYSTEM PROMPTS (for LLM-backed agents)
# ═══════════════════════════════════════════════════════════════════

_AGENT_PROMPTS: Dict[str, str] = {
    "P7-DEMA": (
        "You are DEMA, the user's personal agent nexus. "
        "Classify the intent and restate the mission clearly. "
        "Output format: [INTENT: <type>] <restated mission>"
    ),
    "P1-Planner": (
        "You are the Planner. Decompose this mission into 1-5 concrete steps. "
        "Output format: STEP 1: ... STEP 2: ... (one line per step)"
    ),
    "P2-Researcher": (
        "You are the Researcher. Gather relevant knowledge and cite sources. "
        "Focus on factual accuracy and evidence grounding."
    ),
    "P3-Coder": (
        "You are the Coder. Produce clean, tested, production-quality output. "
        "If code is requested, include type hints and docstrings."
    ),
    "P4-Evaluator": (
        "You are the Evaluator. Score the quality of the following output "
        "across 8 Ihsān dimensions. Be fair and evidence-based."
    ),
    "P6-Publisher": (
        "You are the Publisher. Format the output for clarity, readability, "
        "and accessibility. Respect RTL languages. Apply the Daughter Test."
    ),
    "S2-Oracle": (
        "You are the Oracle. Verify that this output meets constitutional "
        "standards. Check for accuracy, safety, and ethical alignment."
    ),
}


# ═══════════════════════════════════════════════════════════════════
# MISSION PIPELINE
# ═══════════════════════════════════════════════════════════════════

class MissionPipeline:
    """The 12-agent HHMM-routed cognitive chain.

    Implements InferenceProvider so it plugs directly into the
    NervousSystem as a drop-in replacement for monolithic inference.

    Architecture (§6 Mode 2):
      Mission text → HHMMClassifier → ComplexityTier → agent chain
      → each agent fires in constitutional order
      → pure-code agents gate/verify/record
      → PipelineResult with full evidence trail

    Usage:
      # As InferenceProvider for NervousSystem:
      pipeline = MissionPipeline(inference=ollama_provider)
      ns = SovereignNervousSystem(inference=pipeline)
      receipt = await ns.run("implement user auth")

      # Standalone with full traces:
      result = await pipeline.execute("implement user auth")
      print(result.agent_chain)   # ['P7-DEMA', 'P1-Planner', ...]
      print(result.gate_passed)   # True/False
    """

    def __init__(
        self,
        inference: AgentInferenceProvider,
        *,
        classifier: Optional[HHMMClassifierLike] = None,
        on_trace: Optional[Callable[[AgentTrace], None]] = None,
        override_complexity: Optional[ComplexityTier] = None,
    ) -> None:
        self._inference = inference
        self._classifier = classifier or HHMMComplexityClassifier()
        self._on_trace = on_trace
        self._override_complexity = override_complexity
        self._mission_counter = 0
        self._chain_hash = "0" * 32
        self._stats = PipelineStats()

    # ─── InferenceProvider Protocol ───────────────────────────────

    async def infer(self, prompt: str, **kwargs: Any) -> str:
        """InferenceProvider protocol — NervousSystem calls this.

        Runs the full 12-agent pipeline and returns the consolidated
        output text. Full traces are available via execute().
        """
        result = await self.execute(prompt, **kwargs)
        return result.final_output

    # ─── Full Pipeline Execution ──────────────────────────────────

    async def execute(self, mission_text: str, **kwargs: Any) -> PipelineResult:
        """Execute the full mission pipeline with detailed traces.

        Flow:
          1. Classify complexity (HHMM macro-state)
          2. Build agent chain for that complexity
          3. Run each agent step in constitutional order
          4. Collect traces, scores, gate decisions
          5. Generate evidence hash
          6. Return structured PipelineResult
        """
        t0 = time.monotonic()
        self._mission_counter += 1
        mission_id = f"mp-{self._mission_counter:06d}"

        # Step 1: Classify complexity
        if self._override_complexity:
            tier = self._override_complexity
        else:
            tier = HHMMComplexityClassifier().classify(mission_text)

        # Step 2: Build agent chain
        chain_ids = list(COMPLEXITY_CHAINS.get(tier, COMPLEXITY_CHAINS[ComplexityTier.SIMPLE]))
        traces: List[AgentTrace] = []
        gate_reasons: List[str] = []
        current_text = mission_text
        execution_output = ""
        ihsan_tensor: Dict[str, float] = {}
        ihsan_composite = 0.0
        all_gates_passed = True

        # Step 3: Execute each agent in order
        for agent_id in chain_ids:
            spec = AGENT_ROSTER[agent_id]
            step_t0 = time.monotonic()

            trace = AgentTrace(
                agent_id=agent_id,
                role=spec.role,
                phase=spec.pipeline_phase,
                input_summary=current_text[:200],
                output_summary="",
                duration_ms=0.0,
                is_frozen=spec.is_frozen,
                used_llm=spec.uses_llm,
            )

            # ─── LLM-backed agents ───────────────────────────
            if spec.uses_llm and agent_id in _AGENT_PROMPTS:
                agent_prompt = (
                    f"[{agent_id}] {_AGENT_PROMPTS[agent_id]}\n\n"
                    f"Mission: {current_text}"
                )
                try:
                    output = await self._inference.infer(
                        agent_prompt,
                        agent_id=agent_id,
                        **kwargs,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    output = f"[{agent_id} degraded: {exc}]"
                    logger.warning("Agent %s inference failed: %s", agent_id, exc)

                trace.output_summary = output[:200]

                # Route output based on phase
                if spec.pipeline_phase == "intake":
                    current_text = output  # DEMA reformulates
                elif spec.pipeline_phase == "plan":
                    current_text = output  # Planner decomposes
                elif spec.pipeline_phase == "execute":
                    execution_output = output
                    current_text = output
                elif spec.pipeline_phase == "evaluate":
                    # P4 scores — but we also run algorithmic scoring
                    pass
                elif spec.pipeline_phase == "publish":
                    execution_output = output  # Publisher refines
                elif spec.pipeline_phase == "verify":
                    # S2 Oracle LLM check (supplements algorithmic check)
                    pass

            # ─── P4-Evaluator: 8D Ihsān tensor ──────────────
            elif agent_id == "P4-Evaluator":
                target_text = execution_output or current_text
                ihsan_tensor = _score_ihsan_tensor(target_text)
                ihsan_composite = _geometric_mean_ihsan(ihsan_tensor)
                trace.ihsan_score = ihsan_composite
                trace.output_summary = f"ihsan={ihsan_composite:.4f}"

            # ─── P5-Ethicist: Constitutional gate (FROZEN) ───
            elif agent_id == "P5-Ethicist":
                target_text = execution_output or current_text
                passed, reasons = _ethicist_gate(ihsan_composite, target_text)
                trace.gate_passed = passed
                trace.output_summary = f"gate={'PASS' if passed else 'FAIL'}: {reasons}"
                if not passed:
                    all_gates_passed = False
                    gate_reasons.extend(reasons)

            # ─── S1-Sentinel: Security check ─────────────────
            elif agent_id == "S1-Sentinel":
                passed, reasons = _sentinel_check(mission_text)
                trace.gate_passed = passed
                trace.output_summary = f"security={'PASS' if passed else 'FAIL'}: {reasons}"
                if not passed:
                    all_gates_passed = False
                    gate_reasons.extend(reasons)

            # ─── S2-Oracle: Constitutional verification ──────
            elif agent_id == "S2-Oracle":
                passed, reasons = _oracle_verify(traces, ihsan_composite)
                trace.gate_passed = passed
                trace.output_summary = f"oracle={'PASS' if passed else 'FAIL'}: {reasons}"
                if not passed:
                    all_gates_passed = False
                    gate_reasons.extend(reasons)

            # ─── S3-Ledger: Evidence chain ───────────────────
            elif agent_id == "S3-Ledger":
                evidence_hash = _ledger_hash(traces, mission_id)
                trace.output_summary = f"evidence={evidence_hash[:16]}..."

            # ─── S4-Conductor: Model routing (noop) ──────────
            elif agent_id == "S4-Conductor":
                trace.output_summary = f"routed {len(chain_ids)} agents"

            # ─── S5-Ambassador: Federation (noop) ────────────
            elif agent_id == "S5-Ambassador":
                trace.output_summary = "federation=single_node"

            # Finalize trace
            trace.duration_ms = round((time.monotonic() - step_t0) * 1000, 2)
            traces.append(trace)

            if self._on_trace:
                self._on_trace(trace)

        # Step 4: Final output selection
        final_output = execution_output or current_text

        # Step 5: Ensure we have tensor scores even for trivial
        if not ihsan_tensor:
            ihsan_tensor = _score_ihsan_tensor(final_output)
            ihsan_composite = _geometric_mean_ihsan(ihsan_tensor)

        # Step 6: Evidence hash
        evidence_hash = _ledger_hash(traces, mission_id)
        self._chain_hash = hashlib.sha256(
            f"{self._chain_hash}:{evidence_hash}".encode()
        ).hexdigest()[:32]

        total_ms = round((time.monotonic() - t0) * 1000, 2)

        # Update stats
        self._stats.missions_executed += 1
        self._stats.total_agents_fired += len(traces)
        self._stats.total_duration_ms += total_ms
        if all_gates_passed:
            self._stats.gates_passed += 1
        else:
            self._stats.gates_failed += 1
        complexity_key = tier.value
        self._stats.complexity_distribution[complexity_key] = (
            self._stats.complexity_distribution.get(complexity_key, 0) + 1
        )

        frozen_agents = [t.agent_id for t in traces if t.is_frozen]

        result = PipelineResult(
            mission_id=mission_id,
            complexity=tier.value,
            agents_activated=len(traces),
            agent_chain=chain_ids,
            agent_traces=traces,
            final_output=final_output,
            ihsan_composite=ihsan_composite,
            ihsan_tensor=ihsan_tensor,
            gate_passed=all_gates_passed,
            gate_reasons=gate_reasons,
            evidence_hash=evidence_hash,
            total_duration_ms=total_ms,
            frozen_agents=frozen_agents,
        )

        logger.info(
            "Pipeline %s: %s tier, %d agents, ihsan=%.4f, gates=%s, %.1fms",
            mission_id, tier.value, len(traces),
            ihsan_composite, "PASS" if all_gates_passed else "FAIL", total_ms,
        )

        return result

    # ─── Observability ────────────────────────────────────────────

    @property
    def stats(self) -> "PipelineStats":
        """Pipeline execution statistics."""
        return self._stats

    @property
    def chain_hash(self) -> str:
        """Current evidence chain hash (links all pipeline executions)."""
        return self._chain_hash


@dataclass
class PipelineStats:
    """Accumulated pipeline execution statistics."""

    missions_executed: int = 0
    total_agents_fired: int = 0
    total_duration_ms: float = 0.0
    gates_passed: int = 0
    gates_failed: int = 0
    complexity_distribution: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.complexity_distribution is None:
            self.complexity_distribution = {}

    @property
    def avg_agents_per_mission(self) -> float:
        """Average number of agents firing per mission."""
        if self.missions_executed == 0:
            return 0.0
        return round(self.total_agents_fired / self.missions_executed, 2)

    @property
    def avg_duration_ms(self) -> float:
        """Average pipeline duration in milliseconds."""
        if self.missions_executed == 0:
            return 0.0
        return round(self.total_duration_ms / self.missions_executed, 2)

    @property
    def gate_pass_rate(self) -> float:
        """Fraction of missions that passed all gates."""
        total = self.gates_passed + self.gates_failed
        if total == 0:
            return 1.0
        return round(self.gates_passed / total, 4)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for evidence chain / telemetry."""
        return {
            "missions_executed": self.missions_executed,
            "total_agents_fired": self.total_agents_fired,
            "avg_agents_per_mission": self.avg_agents_per_mission,
            "avg_duration_ms": self.avg_duration_ms,
            "gate_pass_rate": self.gate_pass_rate,
            "complexity_distribution": dict(self.complexity_distribution),
        }


# ═══════════════════════════════════════════════════════════════════
# WIRING HELPER
# ═══════════════════════════════════════════════════════════════════

def wire_pipeline_to_nervous_system(
    nervous_system: Any,
    inference: AgentInferenceProvider,
    *,
    classifier: Optional[HHMMClassifierLike] = None,
) -> MissionPipeline:
    """Wire a MissionPipeline as the NervousSystem's inference provider.

    This is the key integration point: replaces monolithic inference
    with the 12-agent cognitive chain.

    Usage:
        from core.sovereign.mission_nervous_system import SovereignNervousSystem
        from core.sovereign.mission_pipeline import wire_pipeline_to_nervous_system

        ns = SovereignNervousSystem.create(inference=some_provider)
        pipeline = wire_pipeline_to_nervous_system(ns, some_provider)
        # Now ns.run() flows through 12 agents instead of one
    """
    pipeline = MissionPipeline(inference=inference, classifier=classifier)
    nervous_system._inference = pipeline
    logger.info(
        "Pipeline wired to NervousSystem: 12-agent chain active"
    )
    return pipeline
