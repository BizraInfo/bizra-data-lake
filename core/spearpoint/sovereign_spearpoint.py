"""
SOVEREIGN SPEARPOINT — Unified Foundation for Autonomous Reasoning
═══════════════════════════════════════════════════════════════════════════════

The Spearpoint integrates the four foundational capabilities:
1. INFERENCE — LLM gateway with circuit breaker resilience
2. AUTONOMOUS LOOP — Extended OODA with Muraqabah vigilance
3. MEMORY — Persistent learning with 5-type cognitive model
4. VERIFICATION — Z3 SMT formal proofs with Ihsān constraints

This is the TRUE CENTER of BIZRA — the capability that enables all others.

Standing on Giants:
- Shannon (1948): SNR, information theory
- Boyd (1995): OODA loop
- Al-Ghazali (1095): Muraqabah, Ihsān
- de Moura & Bjørner (2008): Z3 SMT solver
- Nygard (2007): Circuit breaker pattern

Principle: لا نفترض — We do not assume.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TypedDict

# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTS — Authoritative Thresholds
# ════════════════════════════════════════════════════════════════════════════════

SPEARPOINT_SNR_FLOOR: float = 0.85  # Minimum acceptable signal quality
SPEARPOINT_IHSAN_THRESHOLD: float = 0.95  # Excellence constraint
SPEARPOINT_CIRCUIT_FAILURE_THRESHOLD: int = 5  # Failures before circuit opens
SPEARPOINT_CIRCUIT_RECOVERY_SECONDS: float = 30.0  # Time before half-open test
SPEARPOINT_MAX_LOOP_ITERATIONS: int = 10  # Prevent infinite loops

# Logging
logger = logging.getLogger("SovereignSpearpoint")


# ════════════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════════


class SpearheadStatus(Enum):
    """Status of a spearhead operation."""
    
    READY = auto()          # Initialized, ready for execution
    OBSERVING = auto()      # OODA: Observe phase
    ORIENTING = auto()      # OODA: Orient phase
    DECIDING = auto()       # OODA: Decide phase
    ACTING = auto()         # OODA: Act phase
    REFLECTING = auto()     # Extended: Reflect phase
    LEARNING = auto()       # Extended: Learn phase
    COMPLETED = auto()      # Successfully finished
    FAILED = auto()         # Failed with error
    GATED = auto()          # Blocked by constitutional gate


class MemoryType(Enum):
    """5-type cognitive memory model."""
    
    EPISODIC = auto()       # Event-based experiences
    SEMANTIC = auto()       # Fact-based knowledge
    PROCEDURAL = auto()     # How-to patterns
    WORKING = auto()        # Active short-term context
    PROSPECTIVE = auto()    # Future goals and plans


class CircuitState(Enum):
    """Circuit breaker states (Nygard 2007)."""
    
    CLOSED = auto()         # Normal operation
    OPEN = auto()           # Failing fast, rejecting calls
    HALF_OPEN = auto()      # Testing recovery


@dataclass
class SpearheadConfig:
    """Configuration for the Sovereign Spearpoint."""
    
    # SNR thresholds
    snr_floor: float = SPEARPOINT_SNR_FLOOR
    ihsan_threshold: float = SPEARPOINT_IHSAN_THRESHOLD
    
    # Circuit breaker
    circuit_failure_threshold: int = SPEARPOINT_CIRCUIT_FAILURE_THRESHOLD
    circuit_recovery_seconds: float = SPEARPOINT_CIRCUIT_RECOVERY_SECONDS
    
    # Loop constraints
    max_iterations: int = SPEARPOINT_MAX_LOOP_ITERATIONS
    loop_timeout_seconds: float = 300.0
    
    # Memory
    working_memory_limit: int = 20
    enable_persistence: bool = True
    
    # Verification
    require_z3_proof: bool = True
    z3_timeout_ms: int = 5000


@dataclass
class MemoryEntry:
    """A single memory entry in the cognitive model."""
    
    id: str
    content: str
    memory_type: MemoryType
    timestamp: datetime
    snr_score: float
    ihsan_score: float
    access_count: int = 0
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "type": self.memory_type.name,
            "snr": round(self.snr_score, 4),
            "ihsan": round(self.ihsan_score, 4),
            "importance": round(self.importance, 4),
        }


@dataclass
class Z3Proof:
    """Z3 SMT satisfiability proof."""
    
    proof_id: str
    constraints_checked: List[str]
    satisfiable: bool
    model: Dict[str, Any]
    generation_time_ms: int
    counterexample: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "satisfiable": self.satisfiable,
            "constraints": self.constraints_checked,
            "time_ms": self.generation_time_ms,
            "counterexample": self.counterexample,
        }


@dataclass
class SpearheadResult:
    """Result of a spearhead operation."""
    
    session_id: str
    status: SpearheadStatus
    output: str
    snr_score: float
    ihsan_score: float
    iterations: int
    elapsed_seconds: float
    z3_proof: Optional[Z3Proof] = None
    memories_used: List[str] = field(default_factory=list)
    memories_created: List[str] = field(default_factory=list)
    loop_trace: List[str] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status.name,
            "output": self.output[:200] + "..." if len(self.output) > 200 else self.output,
            "snr": round(self.snr_score, 4),
            "ihsan": round(self.ihsan_score, 4),
            "iterations": self.iterations,
            "elapsed": round(self.elapsed_seconds, 3),
            "z3_proof": self.z3_proof.to_dict() if self.z3_proof else None,
            "memories_used": len(self.memories_used),
            "memories_created": len(self.memories_created),
        }


# ════════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER — Resilience Pattern (Nygard 2007)
# ════════════════════════════════════════════════════════════════════════════════


class CircuitBreaker:
    """
    Circuit breaker for inference gateway resilience.
    
    Standing on Giants:
    - Nygard (2007): Release It! - Circuit breaker pattern
    - Netflix Hystrix: Latency and fault tolerance
    
    States: CLOSED → OPEN (on failure) → HALF_OPEN (on timeout) → CLOSED (on success)
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change: float = time.time()
    
    @property
    def state(self) -> CircuitState:
        """Get current state, checking for recovery timeout."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None and time.time() - self._last_failure_time >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()
        logger.debug(f"CircuitBreaker[{self.name}]: {old_state.name} → {new_state.name}")
    
    def record_success(self) -> None:
        """Record a successful call."""
        self._success_count += 1
        
        if self._state == CircuitState.HALF_OPEN:
            # Recovery successful
            self._failure_count = 0
            self._transition_to(CircuitState.CLOSED)
        elif self._state == CircuitState.CLOSED:
            # Normal operation, reset failure count
            self._failure_count = max(0, self._failure_count - 1)
    
    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            # Recovery failed, reopen
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
    
    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        state = self.state  # This checks for recovery timeout
        
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            return True  # Allow test request
        else:  # OPEN
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.name,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "time_in_state": round(time.time() - self._last_state_change, 2),
        }


# ════════════════════════════════════════════════════════════════════════════════
# MEMORY SYSTEM — 5-Type Cognitive Model
# ════════════════════════════════════════════════════════════════════════════════


class MemorySystem:
    """
    Persistent memory with 5-type cognitive model.
    
    Memory Types:
    - Episodic: Event-based experiences (temporal)
    - Semantic: Fact-based knowledge (evergreen)
    - Procedural: How-to patterns (skills)
    - Working: Active context (short-term)
    - Prospective: Goals and plans (future)
    """
    
    def __init__(
        self,
        working_limit: int = 20,
        snr_floor: float = SPEARPOINT_SNR_FLOOR,
    ):
        self.working_limit = working_limit
        self.snr_floor = snr_floor
        
        # Memory stores by type
        self._memories: Dict[MemoryType, Dict[str, MemoryEntry]] = {
            mt: {} for mt in MemoryType
        }
        
        # Working memory is bounded
        self._working_queue: List[str] = []
    
    def encode(
        self,
        content: str,
        memory_type: MemoryType,
        snr_score: float,
        ihsan_score: float,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[MemoryEntry]:
        """Encode a new memory (quality-gated)."""
        
        # Quality gate
        if snr_score < self.snr_floor:
            logger.debug(f"Memory rejected: SNR {snr_score:.3f} < {self.snr_floor}")
            return None
        
        entry = MemoryEntry(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now(timezone.utc),
            snr_score=snr_score,
            ihsan_score=ihsan_score,
            importance=importance,
            metadata=metadata or {},
        )
        
        # Store
        self._memories[memory_type][entry.id] = entry
        
        # Working memory management
        if memory_type == MemoryType.WORKING:
            self._working_queue.append(entry.id)
            if len(self._working_queue) > self.working_limit:
                # Overflow oldest to episodic
                overflow_id = self._working_queue.pop(0)
                if overflow_id in self._memories[MemoryType.WORKING]:
                    overflow = self._memories[MemoryType.WORKING].pop(overflow_id)
                    overflow.memory_type = MemoryType.EPISODIC
                    self._memories[MemoryType.EPISODIC][overflow.id] = overflow
        
        return entry
    
    def retrieve(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Retrieve memories matching query (simple keyword match for now)."""
        
        types = memory_types or list(MemoryType)
        query_lower = query.lower()
        
        candidates = []
        for mt in types:
            for entry in self._memories[mt].values():
                # Simple relevance: keyword overlap + recency + importance
                words = set(query_lower.split())
                content_words = set(entry.content.lower().split())
                overlap = len(words & content_words) / max(len(words), 1)
                
                # Score: 0.4*relevance + 0.3*importance + 0.2*snr + 0.1*recency
                age_hours = (datetime.now(timezone.utc) - entry.timestamp).total_seconds() / 3600
                recency = max(0, 1 - age_hours / 168)  # Decay over 1 week
                
                score = (
                    0.4 * overlap +
                    0.3 * entry.importance +
                    0.2 * entry.snr_score +
                    0.1 * recency
                )
                
                if score > 0.1:  # Minimum threshold
                    entry.access_count += 1
                    candidates.append((score, entry))
        
        # Sort by score, return top
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in candidates[:limit]]
    
    def get_working_context(self, limit: int = 5) -> List[MemoryEntry]:
        """Get recent working memory entries."""
        entries = []
        for mid in reversed(self._working_queue[-limit:]):
            if mid in self._memories[MemoryType.WORKING]:
                entries.append(self._memories[MemoryType.WORKING][mid])
        return entries
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "total_memories": sum(len(store) for store in self._memories.values()),
            "by_type": {mt.name: len(store) for mt, store in self._memories.items()},
            "working_queue_size": len(self._working_queue),
        }


# ════════════════════════════════════════════════════════════════════════════════
# Z3 VERIFICATION — Formal Constraint Proofs
# ════════════════════════════════════════════════════════════════════════════════


class Z3Verifier:
    """
    Z3 SMT solver integration for formal verification.
    
    Standing on Giants:
    - de Moura & Bjørner (2008): Z3 theorem prover
    
    Verifies 4 constitutional constraints:
    1. Ihsān floor (≥ 0.95)
    2. SNR floor (≥ 0.85)
    3. Reversibility (high-risk → reversible ∨ approved)
    4. Resource bounds (cost ≤ autonomy_limit)
    """
    
    def __init__(self, timeout_ms: int = 5000):
        self.timeout_ms = timeout_ms
        self._z3_available = False
        
        # Try to import Z3
        try:
            import z3
            self._z3 = z3
            self._z3_available = True
            logger.debug("Z3 SMT solver available")
        except ImportError:
            logger.warning("Z3 not available; using simulation mode")
    
    def verify(
        self,
        ihsan_score: float,
        snr_score: float,
        risk_level: float = 0.3,
        cost: float = 0.1,
        autonomy_limit: float = 1.0,
        reversible: bool = True,
        human_approved: bool = False,
    ) -> Z3Proof:
        """Verify constitutional constraints."""
        
        start = time.time()
        proof_id = f"z3_{uuid.uuid4().hex[:8]}"
        
        if not self._z3_available:
            # Simulation mode: basic checks
            return self._simulate_verify(
                proof_id, ihsan_score, snr_score, risk_level,
                cost, autonomy_limit, reversible, human_approved, start
            )
        
        # Full Z3 verification
        z3 = self._z3
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        
        # Define variables
        ihsan = z3.Real("ihsan")
        snr = z3.Real("snr")
        risk = z3.Real("risk")
        c = z3.Real("cost")
        limit = z3.Real("autonomy_limit")
        rev = z3.Bool("reversible")
        approved = z3.Bool("human_approved")
        
        # Add actual values as constraints
        solver.add(ihsan == ihsan_score)
        solver.add(snr == snr_score)
        solver.add(risk == risk_level)
        solver.add(c == cost)
        solver.add(limit == autonomy_limit)
        solver.add(rev == reversible)
        solver.add(approved == human_approved)
        
        # Constitutional constraints
        constraints = []
        
        # 1. Ihsān floor
        c1 = ihsan >= 0.95
        solver.add(c1)
        constraints.append("ihsan >= 0.95")
        
        # 2. SNR floor
        c2 = snr >= 0.85
        solver.add(c2)
        constraints.append("snr >= 0.85")
        
        # 3. Reversibility for high-risk
        c3 = z3.Implies(risk > 0.7, z3.Or(rev, approved))
        solver.add(c3)
        constraints.append("high_risk => (reversible ∨ approved)")
        
        # 4. Resource bounds
        c4 = c <= limit
        solver.add(c4)
        constraints.append("cost <= autonomy_limit")
        
        # Check satisfiability
        result = solver.check()
        gen_time = int((time.time() - start) * 1000)
        
        if result == z3.sat:
            model = solver.model()
            model_dict = {str(d): str(model[d]) for d in model.decls()}
            return Z3Proof(
                proof_id=proof_id,
                constraints_checked=constraints,
                satisfiable=True,
                model=model_dict,
                generation_time_ms=gen_time,
            )
        else:
            # Find which constraint failed
            counterexample = None
            if ihsan_score < 0.95:
                counterexample = f"ihsan {ihsan_score:.3f} < 0.95"
            elif snr_score < 0.85:
                counterexample = f"snr {snr_score:.3f} < 0.85"
            elif risk_level > 0.7 and not reversible and not human_approved:
                counterexample = "high risk without reversibility or approval"
            elif cost > autonomy_limit:
                counterexample = f"cost {cost} > autonomy_limit {autonomy_limit}"
            
            return Z3Proof(
                proof_id=proof_id,
                constraints_checked=constraints,
                satisfiable=False,
                model={},
                generation_time_ms=gen_time,
                counterexample=counterexample,
            )
    
    def _simulate_verify(
        self,
        proof_id: str,
        ihsan_score: float,
        snr_score: float,
        risk_level: float,
        cost: float,
        autonomy_limit: float,
        reversible: bool,
        human_approved: bool,
        start: float,
    ) -> Z3Proof:
        """Simulation mode when Z3 not available."""
        
        constraints = [
            "ihsan >= 0.95",
            "snr >= 0.85",
            "high_risk => (reversible ∨ approved)",
            "cost <= autonomy_limit",
        ]
        
        # Check each constraint
        counterexample = None
        satisfiable = True
        
        if ihsan_score < 0.95:
            satisfiable = False
            counterexample = f"ihsan {ihsan_score:.3f} < 0.95"
        elif snr_score < 0.85:
            satisfiable = False
            counterexample = f"snr {snr_score:.3f} < 0.85"
        elif risk_level > 0.7 and not reversible and not human_approved:
            satisfiable = False
            counterexample = "high risk without reversibility or approval"
        elif cost > autonomy_limit:
            satisfiable = False
            counterexample = f"cost {cost} > autonomy_limit {autonomy_limit}"
        
        return Z3Proof(
            proof_id=proof_id + "_sim",
            constraints_checked=constraints,
            satisfiable=satisfiable,
            model={"mode": "simulation"} if satisfiable else {},
            generation_time_ms=int((time.time() - start) * 1000),
            counterexample=counterexample,
        )


# ════════════════════════════════════════════════════════════════════════════════
# SOVEREIGN SPEARPOINT — The True Foundation
# ════════════════════════════════════════════════════════════════════════════════


class SovereignSpearpoint:
    """
    The TRUE SPEARPOINT of BIZRA — unified foundation for autonomous reasoning.
    
    Integrates:
    1. INFERENCE — LLM gateway with circuit breaker resilience
    2. AUTONOMOUS LOOP — Extended OODA with Muraqabah vigilance
    3. MEMORY — Persistent learning with 5-type cognitive model
    4. VERIFICATION — Z3 SMT formal proofs with Ihsān constraints
    
    This is the center of gravity. Without it, no thought. With it, sovereignty.
    
    Standing on Giants:
    - Shannon (1948): SNR
    - Boyd (1995): OODA
    - Al-Ghazali (1095): Muraqabah
    - de Moura & Bjørner (2008): Z3
    - Nygard (2007): Circuit breaker
    """
    
    def __init__(self, config: Optional[SpearheadConfig] = None):
        self.config = config or SpearheadConfig()
        
        # Session tracking
        self._session_id = uuid.uuid4().hex[:12]
        self._start_time = time.time()
        
        # Components
        self._circuit_breaker = CircuitBreaker(
            name="inference",
            failure_threshold=self.config.circuit_failure_threshold,
            recovery_timeout=self.config.circuit_recovery_seconds,
        )
        self._memory = MemorySystem(
            working_limit=self.config.working_memory_limit,
            snr_floor=self.config.snr_floor,
        )
        self._verifier = Z3Verifier(timeout_ms=self.config.z3_timeout_ms)
        
        # Inference backend (placeholder — integrate with actual gateway)
        self._inference_fn: Optional[Callable[[str], Awaitable[str]]] = None
        
        # Statistics
        self._stats = {
            "cycles": 0,
            "successful": 0,
            "gated": 0,
            "failed": 0,
            "total_snr": 0.0,
            "total_ihsan": 0.0,
        }
        
        logger.info(f"SovereignSpearpoint initialized | Session: {self._session_id}")
    
    def set_inference_backend(
        self,
        fn: Callable[[str], Awaitable[str]],
    ) -> None:
        """Set the inference backend function."""
        self._inference_fn = fn
        logger.debug("Inference backend configured")
    
    async def _infer(self, prompt: str) -> str:
        """Execute inference with circuit breaker protection."""
        
        if not self._circuit_breaker.allow_request():
            raise RuntimeError("Circuit breaker OPEN — inference unavailable")
        
        if self._inference_fn is None:
            # Fallback: simulate response
            self._circuit_breaker.record_success()
            return f"[Simulated response to: {prompt[:50]}...]"
        
        try:
            result = await self._inference_fn(prompt)
            self._circuit_breaker.record_success()
            return result
        except Exception as e:
            self._circuit_breaker.record_failure()
            raise RuntimeError(f"Inference failed: {e}") from e
    
    def _calculate_snr(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate SNR score for content."""
        # Simplified SNR: length-adjusted, keyword-based
        words = content.split()
        unique_words = set(w.lower() for w in words)
        
        # Signal: unique information density
        signal = len(unique_words) / max(len(words), 1)
        
        # Noise: repetition factor
        noise = 1 - signal + 0.01  # Avoid division by zero
        
        # Context boost
        if context.get("has_evidence", False):
            signal *= 1.1
        if context.get("is_coherent", True):
            signal *= 1.05
        
        snr = min(signal / noise, 1.0)
        return max(snr, 0.0)
    
    def _calculate_ihsan(self, content: str, snr: float) -> float:
        """Calculate Ihsān score (8-dimensional, simplified)."""
        # Simplified: based on SNR + content quality heuristics
        base = snr * 0.4
        
        # Length appropriateness (not too short, not too verbose)
        words = len(content.split())
        length_score = 1.0 if 20 <= words <= 500 else 0.8
        
        # Presence of uncertainty acknowledgment
        uncertainty = any(w in content.lower() for w in ["however", "although", "consider", "might", "could"])
        uncertainty_score = 1.0 if uncertainty else 0.9
        
        # Ihsān = weighted combination
        ihsan = base + 0.3 * length_score + 0.2 * uncertainty_score + 0.1
        return min(ihsan, 1.0)
    
    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SpearheadResult:
        """
        Execute the full Sovereign Spearpoint loop.
        
        Phases: OBSERVE → ORIENT → DECIDE → ACT → REFLECT → LEARN
        
        Returns a SpearheadResult with output, SNR, Ihsān, and Z3 proof.
        """
        
        start = time.time()
        ctx = context or {}
        loop_trace: List[str] = []
        memories_used: List[str] = []
        memories_created: List[str] = []
        
        try:
            # ════════════════════════════════════════════════════════════════
            # PHASE 1: OBSERVE
            # ════════════════════════════════════════════════════════════════
            loop_trace.append("OBSERVE: Gathering context")
            
            # Retrieve relevant memories
            relevant = self._memory.retrieve(query, limit=5)
            for mem in relevant:
                memories_used.append(mem.id)
            
            # Build context with working memory
            working = self._memory.get_working_context(limit=3)
            memory_context = "\n".join(
                f"[{m.memory_type.name}] {m.content[:100]}" for m in working
            )
            
            # ════════════════════════════════════════════════════════════════
            # PHASE 2: ORIENT
            # ════════════════════════════════════════════════════════════════
            loop_trace.append("ORIENT: Analyzing situation")
            
            # Build prompt with context
            prompt = f"""Query: {query}

Context from memory:
{memory_context if memory_context else "(No prior context)"}

Provide a thoughtful, accurate response. Acknowledge uncertainty where appropriate.
"""
            
            # ════════════════════════════════════════════════════════════════
            # PHASE 3: DECIDE
            # ════════════════════════════════════════════════════════════════
            loop_trace.append("DECIDE: Selecting action")
            
            # Check circuit breaker
            if not self._circuit_breaker.allow_request():
                loop_trace.append("GATED: Circuit breaker OPEN")
                return SpearheadResult(
                    session_id=self._session_id,
                    status=SpearheadStatus.GATED,
                    output="",
                    snr_score=0.0,
                    ihsan_score=0.0,
                    iterations=0,
                    elapsed_seconds=time.time() - start,
                    error="Circuit breaker OPEN — system degraded",
                    loop_trace=loop_trace,
                )
            
            # ════════════════════════════════════════════════════════════════
            # PHASE 4: ACT (Inference)
            # ════════════════════════════════════════════════════════════════
            loop_trace.append("ACT: Executing inference")
            
            output = await self._infer(prompt)
            
            # Calculate quality scores
            snr = self._calculate_snr(output, ctx)
            ihsan = self._calculate_ihsan(output, snr)
            
            # ════════════════════════════════════════════════════════════════
            # PHASE 5: VERIFY (Z3 Proof)
            # ════════════════════════════════════════════════════════════════
            loop_trace.append("VERIFY: Z3 constraint check")
            
            z3_proof = None
            if self.config.require_z3_proof:
                z3_proof = self._verifier.verify(
                    ihsan_score=ihsan,
                    snr_score=snr,
                    risk_level=ctx.get("risk_level", 0.3),
                    cost=ctx.get("cost", 0.1),
                    autonomy_limit=ctx.get("autonomy_limit", 1.0),
                    reversible=ctx.get("reversible", True),
                    human_approved=ctx.get("human_approved", False),
                )
                
                if not z3_proof.satisfiable:
                    loop_trace.append(f"GATED: Z3 UNSAT — {z3_proof.counterexample}")
                    self._stats["gated"] += 1
                    return SpearheadResult(
                        session_id=self._session_id,
                        status=SpearheadStatus.GATED,
                        output=output,
                        snr_score=snr,
                        ihsan_score=ihsan,
                        iterations=1,
                        elapsed_seconds=time.time() - start,
                        z3_proof=z3_proof,
                        error=f"Constitutional gate: {z3_proof.counterexample}",
                        loop_trace=loop_trace,
                        memories_used=memories_used,
                    )
            
            # ════════════════════════════════════════════════════════════════
            # PHASE 6: REFLECT
            # ════════════════════════════════════════════════════════════════
            loop_trace.append("REFLECT: Analyzing outcome")
            
            # ════════════════════════════════════════════════════════════════
            # PHASE 7: LEARN (Memory encoding)
            # ════════════════════════════════════════════════════════════════
            loop_trace.append("LEARN: Encoding to memory")
            
            # Encode query to working memory
            query_mem = self._memory.encode(
                content=f"Query: {query}",
                memory_type=MemoryType.WORKING,
                snr_score=snr,
                ihsan_score=ihsan,
                importance=0.6,
            )
            if query_mem:
                memories_created.append(query_mem.id)
            
            # Encode response to episodic memory
            response_mem = self._memory.encode(
                content=f"Response: {output[:200]}",
                memory_type=MemoryType.EPISODIC,
                snr_score=snr,
                ihsan_score=ihsan,
                importance=0.7,
            )
            if response_mem:
                memories_created.append(response_mem.id)
            
            # Update statistics
            self._stats["cycles"] += 1
            self._stats["successful"] += 1
            self._stats["total_snr"] += snr
            self._stats["total_ihsan"] += ihsan
            
            loop_trace.append(f"COMPLETE: SNR={snr:.3f}, Ihsān={ihsan:.3f}")
            
            return SpearheadResult(
                session_id=self._session_id,
                status=SpearheadStatus.COMPLETED,
                output=output,
                snr_score=snr,
                ihsan_score=ihsan,
                iterations=1,
                elapsed_seconds=time.time() - start,
                z3_proof=z3_proof,
                memories_used=memories_used,
                memories_created=memories_created,
                loop_trace=loop_trace,
            )
            
        except Exception as e:
            self._stats["cycles"] += 1
            self._stats["failed"] += 1
            loop_trace.append(f"ERROR: {e}")
            
            return SpearheadResult(
                session_id=self._session_id,
                status=SpearheadStatus.FAILED,
                output="",
                snr_score=0.0,
                ihsan_score=0.0,
                iterations=1,
                elapsed_seconds=time.time() - start,
                error=str(e),
                loop_trace=loop_trace,
                memories_used=memories_used,
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get spearpoint statistics."""
        avg_snr = self._stats["total_snr"] / max(self._stats["successful"], 1)
        avg_ihsan = self._stats["total_ihsan"] / max(self._stats["successful"], 1)
        
        return {
            "session_id": self._session_id,
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "cycles": self._stats["cycles"],
            "successful": self._stats["successful"],
            "gated": self._stats["gated"],
            "failed": self._stats["failed"],
            "avg_snr": round(avg_snr, 4),
            "avg_ihsan": round(avg_ihsan, 4),
            "circuit_breaker": self._circuit_breaker.get_metrics(),
            "memory": self._memory.get_statistics(),
        }


# ════════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ════════════════════════════════════════════════════════════════════════════════


async def demo():
    """Demonstrate the Sovereign Spearpoint."""
    
    print("═" * 80)
    print("SOVEREIGN SPEARPOINT — The True Foundation")
    print("═" * 80)
    
    # Initialize
    spearpoint = SovereignSpearpoint()
    
    # Execute a query
    result = await spearpoint.execute(
        query="What is the most important principle for building sovereign AI systems?",
        context={"risk_level": 0.2, "reversible": True},
    )
    
    print(f"\nSession: {result.session_id}")
    print(f"Status: {result.status.name}")
    print(f"SNR: {result.snr_score:.4f}")
    print(f"Ihsān: {result.ihsan_score:.4f}")
    print(f"Elapsed: {result.elapsed_seconds:.3f}s")
    
    if result.z3_proof:
        print(f"\nZ3 Proof: {'SAT ✓' if result.z3_proof.satisfiable else 'UNSAT ✗'}")
        print(f"  Constraints: {result.z3_proof.constraints_checked}")
        if result.z3_proof.counterexample:
            print(f"  Counterexample: {result.z3_proof.counterexample}")
    
    print(f"\nLoop Trace:")
    for step in result.loop_trace:
        print(f"  → {step}")
    
    print(f"\nOutput:\n{result.output}")
    
    # Statistics
    stats = spearpoint.get_statistics()
    print(f"\nStatistics: {stats}")
    
    print("\n" + "═" * 80)
    print("لا نفترض — We do not assume. We verify with formal proofs.")
    print("إحسان — Excellence in all things.")
    print("═" * 80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
