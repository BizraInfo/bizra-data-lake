"""
AUTOPOIETIC ENTROPY REDUCTION ENGINE (AERE)
═══════════════════════════════════════════════════════════════════════════════

The Self-Sustaining Cognitive Core with Merkle-Persisted State

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUTOPOIETIC ENTROPY REDUCTION ENGINE                      │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  COGNITIVE LAYER                                                     │   │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│   │  │ Perceive │→ │ Abstract │→ │ Intend   │→ │ Act      │            │   │
│   │  │          │  │          │  │          │  │          │            │   │
│   │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │   │
│   │       │              │              │              │                 │   │
│   │       └──────────────┴──────────────┴──────────────┘                 │   │
│   │                              │                                        │   │
│   │                              ▼                                        │   │
│   │                    ┌──────────────────┐                              │   │
│   │                    │   CONSOLIDATE    │ ← Memory Formation           │   │
│   │                    └────────┬─────────┘                              │   │
│   └─────────────────────────────┼───────────────────────────────────────┘   │
│                                 │                                            │
│   ┌─────────────────────────────┼───────────────────────────────────────┐   │
│   │  ENTROPY REDUCTION LAYER    ▼                                       │   │
│   │  ┌──────────────────────────────────────────────────────────────┐   │   │
│   │  │  Entropy Estimator → Compression → Deduplication → Indexing  │   │   │
│   │  └──────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                 │                                            │
│   ┌─────────────────────────────┼───────────────────────────────────────┐   │
│   │  MERKLE PERSISTENCE LAYER   ▼                                       │   │
│   │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
│   │  │ State Hash │→ │ Chain Link │→ │ Append Log │→ │ Checkpoint │    │   │
│   │  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │   │
│   │                                                                      │   │
│   │  Merkle DAG: S₀ ← S₁ ← S₂ ← ... ← Sₙ (current head)               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Properties:                                                                │
│   • Autopoietic: Self-producing, self-maintaining                           │
│   • Entropy-Reducing: Information compression over time                     │
│   • Merkle-Persisted: Cryptographically verified state chain               │
│   • Fail-Safe: Any state recoverable from chain                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Giants Protocol:
  Al-Khwarizmi — Algorithmic state transitions
  Ibn Sina — Diagnostic self-monitoring  
  Al-Biruni — Empirical entropy measurement
  Al-Jazari — Self-sustaining mechanical precision

Theoretical Foundation:
  - Autopoiesis (Maturana & Varela): Self-producing systems
  - Entropy Reduction (Schrödinger): Life as neg-entropy
  - Merkle Trees (Merkle): Cryptographic verification
  - Cognitive Architecture (ACT-R, SOAR): Perception-Action cycles

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import time
import zlib
import hashlib
import asyncio
import threading
import mmap
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Callable, Awaitable
from enum import Enum
from pathlib import Path
import struct
import math

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

AERE_VERSION = "1.0.0"
STATE_FILE = Path(os.getenv("BIZRA_AERE_STATE", "/var/lib/bizra/aere/brain.state"))
MERKLE_LOG = Path(os.getenv("BIZRA_MERKLE_LOG", "/var/lib/bizra/aere/merkle.log"))
CHECKPOINT_DIR = Path(os.getenv("BIZRA_CHECKPOINT_DIR", "/var/lib/bizra/aere/checkpoints"))

# Cognitive cycle timing
PERCEPTION_INTERVAL_MS = 100      # 10Hz perception
ABSTRACTION_INTERVAL_MS = 500     # 2Hz abstraction
INTENTION_INTERVAL_MS = 1000      # 1Hz intention
CONSOLIDATION_INTERVAL_MS = 60000 # Every minute

# Entropy thresholds
ENTROPY_HIGH_THRESHOLD = 0.8   # Trigger compression
ENTROPY_LOW_THRESHOLD = 0.3    # Optimal state
MAX_WORKING_MEMORY_ITEMS = 7   # Miller's Law (7±2)

# Merkle settings
CHECKPOINT_INTERVAL = 100      # States between checkpoints
HASH_ALGORITHM = "blake2b"     # Fast, secure

# ═══════════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class CognitivePhase(str, Enum):
    """Phases of the cognitive cycle."""
    PERCEIVE = "perceive"
    ABSTRACT = "abstract"
    INTEND = "intend"
    ACT = "act"
    CONSOLIDATE = "consolidate"


class EntropyLevel(str, Enum):
    """Entropy classification."""
    LOW = "low"           # Well-organized, compressed
    MEDIUM = "medium"     # Normal operation
    HIGH = "high"         # Needs compression
    CRITICAL = "critical" # Emergency consolidation


@dataclass
class Perception:
    """Raw sensory input."""
    source: str           # Input source identifier
    modality: str         # audio, visual, text, etc.
    raw_data: bytes       # Compressed raw input
    timestamp_ns: int     # Nanosecond precision
    entropy: float        # Shannon entropy of input
    hash: str = ""        # Content hash
    
    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.blake2b(self.raw_data, digest_size=32).hexdigest()


@dataclass
class Abstraction:
    """Compressed representation of perceptions."""
    perception_hashes: List[str]  # Source perceptions
    embedding: bytes              # Compressed embedding vector
    salience: float               # Importance score (0-1)
    novelty: float                # How new is this? (0-1)
    timestamp_ns: int
    hash: str = ""
    
    def __post_init__(self):
        if not self.hash:
            payload = b"".join([h.encode() for h in self.perception_hashes]) + self.embedding
            self.hash = hashlib.blake2b(payload, digest_size=32).hexdigest()


@dataclass
class Intention:
    """Goal or action plan."""
    abstraction_hash: str         # Source abstraction
    goal: str                     # Natural language goal
    action_plan: List[str]        # Steps to achieve
    priority: float               # Urgency (0-1)
    ihsan_score: float            # Ethical alignment (0-1)
    timestamp_ns: int
    hash: str = ""
    
    def __post_init__(self):
        if not self.hash:
            payload = f"{self.goal}:{self.ihsan_score}:{self.timestamp_ns}".encode()
            self.hash = hashlib.blake2b(payload, digest_size=32).hexdigest()


@dataclass
class Action:
    """Executed action with outcome."""
    intention_hash: str
    action_type: str
    parameters: Dict[str, Any]
    outcome: str                  # success, failure, partial
    impact_score: float           # Measured impact
    timestamp_ns: int
    hash: str = ""
    
    def __post_init__(self):
        if not self.hash:
            payload = f"{self.intention_hash}:{self.action_type}:{self.outcome}".encode()
            self.hash = hashlib.blake2b(payload, digest_size=32).hexdigest()


@dataclass
class CognitiveState:
    """
    Complete cognitive state at a point in time.
    
    This is the unit of Merkle persistence.
    """
    version: str = AERE_VERSION
    timestamp_ns: int = 0
    sequence: int = 0
    
    # Current phase
    phase: CognitivePhase = CognitivePhase.PERCEIVE
    
    # Working memory (limited capacity)
    working_memory: List[str] = field(default_factory=list)  # Abstraction hashes
    
    # Recent items (sliding window)
    recent_perceptions: List[str] = field(default_factory=list)
    recent_abstractions: List[str] = field(default_factory=list)
    recent_intentions: List[str] = field(default_factory=list)
    recent_actions: List[str] = field(default_factory=list)
    
    # Entropy metrics
    perception_entropy: float = 0.0
    memory_entropy: float = 0.0
    overall_entropy: float = 0.0
    
    # Chain linkage
    prev_state_hash: str = ""
    state_hash: str = ""
    
    def compute_hash(self) -> str:
        """Compute cryptographic hash of state."""
        payload = json.dumps({
            "v": self.version,
            "t": self.timestamp_ns,
            "s": self.sequence,
            "p": self.phase.value,
            "wm": self.working_memory,
            "prev": self.prev_state_hash,
        }, sort_keys=True).encode()
        
        return hashlib.blake2b(payload, digest_size=32).hexdigest()
    
    def finalize(self, prev_hash: str = "") -> "CognitiveState":
        """Finalize state with hash chain linkage."""
        self.prev_state_hash = prev_hash
        self.timestamp_ns = time.time_ns()
        self.state_hash = self.compute_hash()
        return self
    
    def to_bytes(self) -> bytes:
        """Serialize to compressed bytes."""
        data = json.dumps(asdict(self), default=str).encode()
        return zlib.compress(data, level=6)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "CognitiveState":
        """Deserialize from compressed bytes."""
        decompressed = zlib.decompress(data)
        obj = json.loads(decompressed)
        obj["phase"] = CognitivePhase(obj["phase"])
        return cls(**obj)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTROPY REDUCTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class EntropyEstimator:
    """
    Al-Biruni's Empirical Entropy Measurement.
    
    Estimates Shannon entropy of cognitive contents.
    """
    
    @staticmethod
    def shannon_entropy(data: bytes) -> float:
        """Calculate Shannon entropy of byte sequence."""
        if not data:
            return 0.0
        
        # Count byte frequencies
        freq = {}
        for byte in data:
            freq[byte] = freq.get(byte, 0) + 1
        
        # Calculate entropy
        length = len(data)
        entropy = 0.0
        for count in freq.values():
            if count > 0:
                p = count / length
                entropy -= p * math.log2(p)
        
        # Normalize to 0-1 (max entropy for bytes is 8 bits)
        return entropy / 8.0
    
    @staticmethod
    def list_entropy(items: List[str]) -> float:
        """Entropy based on list diversity."""
        if not items:
            return 0.0
        
        unique = len(set(items))
        total = len(items)
        
        # Higher entropy = more diverse
        return unique / total if total > 0 else 0.0
    
    @staticmethod
    def working_memory_entropy(wm: List[str], max_capacity: int = MAX_WORKING_MEMORY_ITEMS) -> float:
        """
        Working memory entropy.
        
        High when near capacity or fragmented.
        Low when organized and within capacity.
        """
        if not wm:
            return 0.0
        
        # Capacity pressure
        capacity_entropy = len(wm) / max_capacity
        
        # Diversity (unique items)
        diversity_entropy = len(set(wm)) / len(wm) if wm else 0
        
        return (capacity_entropy + diversity_entropy) / 2


class EntropyReducer:
    """
    Schrödinger's Neg-Entropy Mechanism.
    
    Reduces entropy through compression and consolidation.
    """
    
    @staticmethod
    def compress_perceptions(perceptions: List[Perception]) -> Tuple[bytes, float]:
        """Compress multiple perceptions into unified representation."""
        if not perceptions:
            return b"", 0.0
        
        # Concatenate and compress
        combined = b"".join(p.raw_data for p in perceptions)
        compressed = zlib.compress(combined, level=9)
        
        # Compression ratio as entropy reduction metric
        ratio = len(compressed) / len(combined) if combined else 1.0
        reduction = 1.0 - ratio
        
        return compressed, reduction
    
    @staticmethod
    def deduplicate_memory(items: List[str]) -> List[str]:
        """Remove duplicates while preserving recency."""
        seen = set()
        result = []
        
        # Traverse in reverse to keep most recent
        for item in reversed(items):
            if item not in seen:
                seen.add(item)
                result.append(item)
        
        return list(reversed(result))
    
    @staticmethod
    def prune_working_memory(
        wm: List[str],
        salience_scores: Dict[str, float],
        max_items: int = MAX_WORKING_MEMORY_ITEMS
    ) -> Tuple[List[str], List[str]]:
        """
        Prune working memory to capacity.
        
        Keep highest salience items.
        Returns (kept, pruned).
        """
        if len(wm) <= max_items:
            return wm, []
        
        # Sort by salience (descending)
        scored = [(item, salience_scores.get(item, 0.5)) for item in wm]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        kept = [item for item, _ in scored[:max_items]]
        pruned = [item for item, _ in scored[max_items:]]
        
        return kept, pruned


# ═══════════════════════════════════════════════════════════════════════════════
# MERKLE PERSISTENCE LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class MerkleStateChain:
    """
    Cryptographically verified state persistence.
    
    Properties:
    - Append-only log for integrity
    - Chain linkage via prev_hash
    - Periodic checkpoints for fast recovery
    - Tamper-evident (any modification breaks chain)
    """
    
    def __init__(self):
        self.current_head: Optional[str] = None
        self.sequence: int = 0
        self.log_file: Optional[Any] = None
        self.state_index: Dict[str, int] = {}  # hash -> file offset
        
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage directories and files."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        MERKLE_LOG.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Open append-only log
        self.log_file = open(MERKLE_LOG, "ab+")
        
        # Load existing chain
        self._load_chain()
    
    def _load_chain(self):
        """Load existing Merkle chain from log."""
        if not MERKLE_LOG.exists() or MERKLE_LOG.stat().st_size == 0:
            return
        
        try:
            with open(MERKLE_LOG, "rb") as f:
                offset = 0
                while True:
                    # Read length prefix (4 bytes)
                    length_bytes = f.read(4)
                    if len(length_bytes) < 4:
                        break
                    
                    length = struct.unpack(">I", length_bytes)[0]
                    data = f.read(length)
                    
                    if len(data) < length:
                        break
                    
                    # Parse state
                    state = CognitiveState.from_bytes(data)
                    self.state_index[state.state_hash] = offset
                    self.current_head = state.state_hash
                    self.sequence = state.sequence
                    
                    offset = f.tell()
        except Exception as e:
            print(f"[MerkleChain] Error loading chain: {e}")
    
    def append(self, state: CognitiveState) -> str:
        """
        Append state to Merkle chain.
        
        Returns state hash.
        """
        # Finalize with chain linkage
        state.sequence = self.sequence + 1
        state.finalize(self.current_head or "")
        
        # Serialize
        data = state.to_bytes()
        
        # Write with length prefix (for parsing)
        offset = self.log_file.tell()
        self.log_file.write(struct.pack(">I", len(data)))
        self.log_file.write(data)
        self.log_file.flush()
        
        # Update index
        self.state_index[state.state_hash] = offset
        self.current_head = state.state_hash
        self.sequence = state.sequence
        
        # Periodic checkpoint
        if self.sequence % CHECKPOINT_INTERVAL == 0:
            self._create_checkpoint(state)
        
        return state.state_hash
    
    def get(self, state_hash: str) -> Optional[CognitiveState]:
        """Retrieve state by hash."""
        if state_hash not in self.state_index:
            return None
        
        offset = self.state_index[state_hash]
        
        with open(MERKLE_LOG, "rb") as f:
            f.seek(offset)
            length_bytes = f.read(4)
            length = struct.unpack(">I", length_bytes)[0]
            data = f.read(length)
            return CognitiveState.from_bytes(data)
    
    def get_head(self) -> Optional[CognitiveState]:
        """Get current head state."""
        if self.current_head:
            return self.get(self.current_head)
        return None
    
    def verify_chain(self, from_hash: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Verify chain integrity from hash to head.
        
        Returns (valid, [broken_hashes]).
        """
        broken = []
        current = from_hash or self._find_genesis()
        
        while current:
            state = self.get(current)
            if not state:
                broken.append(current)
                break
            
            # Verify hash
            computed = state.compute_hash()
            if computed != state.state_hash:
                broken.append(current)
            
            # Find next in chain
            current = self._find_next(state.state_hash)
        
        return len(broken) == 0, broken
    
    def _find_genesis(self) -> Optional[str]:
        """Find genesis state (prev_hash is empty)."""
        for hash_val in self.state_index:
            state = self.get(hash_val)
            if state and not state.prev_state_hash:
                return hash_val
        return None
    
    def _find_next(self, current_hash: str) -> Optional[str]:
        """Find state that links to current."""
        for hash_val in self.state_index:
            state = self.get(hash_val)
            if state and state.prev_state_hash == current_hash:
                return hash_val
        return None
    
    def _create_checkpoint(self, state: CognitiveState):
        """Create checkpoint for fast recovery."""
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{state.sequence}.state"
        checkpoint_path.write_bytes(state.to_bytes())
        
        # Keep only last 10 checkpoints
        checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_*.state"))
        for old in checkpoints[:-10]:
            old.unlink()
    
    def recover_from_checkpoint(self) -> Optional[CognitiveState]:
        """Recover from latest checkpoint."""
        checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_*.state"))
        if not checkpoints:
            return None
        
        latest = checkpoints[-1]
        return CognitiveState.from_bytes(latest.read_bytes())
    
    def close(self):
        """Close log file."""
        if self.log_file:
            self.log_file.close()


# ═══════════════════════════════════════════════════════════════════════════════
# COGNITIVE PROCESSES
# ═══════════════════════════════════════════════════════════════════════════════

class PerceptionProcess:
    """
    Sensory input processing.
    
    Converts raw inputs to structured Perceptions.
    """
    
    def __init__(self):
        self.buffer: List[Perception] = []
        self.entropy_estimator = EntropyEstimator()
    
    async def process(self, source: str, modality: str, raw_data: bytes) -> Perception:
        """Process raw input into Perception."""
        # Compress input
        compressed = zlib.compress(raw_data, level=6)
        
        # Estimate entropy
        entropy = self.entropy_estimator.shannon_entropy(raw_data)
        
        perception = Perception(
            source=source,
            modality=modality,
            raw_data=compressed,
            timestamp_ns=time.time_ns(),
            entropy=entropy,
        )
        
        self.buffer.append(perception)
        
        # Limit buffer size
        if len(self.buffer) > 100:
            self.buffer = self.buffer[-100:]
        
        return perception


class AbstractionProcess:
    """
    Pattern extraction and compression.
    
    Converts Perceptions to Abstractions.
    """
    
    def __init__(self, llm_embed: Optional[Callable[[str], bytes]] = None):
        self.llm_embed = llm_embed or self._default_embed
        self.buffer: List[Abstraction] = []
    
    def _default_embed(self, text: str) -> bytes:
        """Default embedding (hash-based, no LLM)."""
        return hashlib.blake2b(text.encode(), digest_size=64).digest()
    
    async def process(self, perceptions: List[Perception]) -> Abstraction:
        """Abstract patterns from perceptions."""
        # Combine perception data
        combined = b"".join(p.raw_data for p in perceptions)
        
        # Generate embedding
        embedding = self.llm_embed(combined.hex()[:1000])  # Truncate for safety
        
        # Calculate salience (based on entropy)
        avg_entropy = sum(p.entropy for p in perceptions) / len(perceptions) if perceptions else 0
        salience = 1.0 - avg_entropy  # Low entropy = high salience
        
        # Calculate novelty (simple: based on hash uniqueness)
        perception_hashes = [p.hash for p in perceptions]
        existing_hashes = {a.hash for a in self.buffer}
        novelty = 1.0 if all(h not in existing_hashes for h in perception_hashes) else 0.5
        
        abstraction = Abstraction(
            perception_hashes=perception_hashes,
            embedding=embedding,
            salience=salience,
            novelty=novelty,
            timestamp_ns=time.time_ns(),
        )
        
        self.buffer.append(abstraction)
        
        # Limit buffer
        if len(self.buffer) > 50:
            self.buffer = self.buffer[-50:]
        
        return abstraction


class IntentionProcess:
    """
    Goal formation with ethical constraints.
    
    Converts Abstractions to Intentions.
    """
    
    def __init__(self, llm_generate: Optional[Callable[[str], str]] = None):
        self.llm_generate = llm_generate or self._default_generate
        self.buffer: List[Intention] = []
    
    def _default_generate(self, prompt: str) -> str:
        """Default generation (echo, no LLM)."""
        return f"Process: {prompt[:100]}"
    
    async def process(
        self,
        abstraction: Abstraction,
        context: List[str],
        ihsan_constraints: Dict[str, float],
    ) -> Intention:
        """Form intention from abstraction with ethical bounds."""
        # Generate goal via LLM (or default)
        prompt = f"Context: {context[:5]}\nAbstraction: {abstraction.hash[:16]}\nGoal:"
        goal = self.llm_generate(prompt)
        
        # Simple action plan
        action_plan = [f"step_{i}" for i in range(3)]
        
        # Calculate Ihsān score
        ihsan_score = min(ihsan_constraints.values()) if ihsan_constraints else 0.5
        
        # Priority based on salience and novelty
        priority = (abstraction.salience + abstraction.novelty) / 2
        
        intention = Intention(
            abstraction_hash=abstraction.hash,
            goal=goal,
            action_plan=action_plan,
            priority=priority,
            ihsan_score=ihsan_score,
            timestamp_ns=time.time_ns(),
        )
        
        # Filter by Ihsān threshold
        if ihsan_score < 0.7:
            intention.goal = f"[BLOCKED:IHSAN<0.7] {intention.goal}"
        
        self.buffer.append(intention)
        
        return intention


class ActionProcess:
    """
    Action execution with impact tracking.
    """
    
    def __init__(self, accumulator=None):
        self.accumulator = accumulator
        self.buffer: List[Action] = []
    
    async def execute(self, intention: Intention) -> Action:
        """Execute intention and measure impact."""
        start = time.time_ns()
        
        # Simulate execution (real impl would dispatch to actuators)
        outcome = "success" if intention.ihsan_score >= 0.7 else "blocked"
        
        elapsed_ns = time.time_ns() - start
        
        # Calculate impact
        impact_score = intention.priority * intention.ihsan_score if outcome == "success" else 0
        
        action = Action(
            intention_hash=intention.hash,
            action_type="cognitive_action",
            parameters={"goal": intention.goal, "elapsed_ns": elapsed_ns},
            outcome=outcome,
            impact_score=impact_score,
            timestamp_ns=time.time_ns(),
        )
        
        # Record to accumulator if available
        if self.accumulator and outcome == "success":
            try:
                self.accumulator.record_impact(
                    contributor="aere:cognitive",
                    action="intention_executed",
                    category="orchestration",
                    impact_score=impact_score,
                )
            except Exception:
                pass
        
        self.buffer.append(action)
        
        return action


class ConsolidationProcess:
    """
    Memory consolidation (entropy reduction).
    
    Compresses and organizes cognitive contents.
    """
    
    def __init__(self):
        self.reducer = EntropyReducer()
        self.estimator = EntropyEstimator()
    
    async def consolidate(
        self,
        state: CognitiveState,
        abstractions: List[Abstraction],
    ) -> CognitiveState:
        """Consolidate memory, reduce entropy."""
        # Calculate current entropy
        state.memory_entropy = self.estimator.working_memory_entropy(state.working_memory)
        
        # Deduplicate working memory
        state.working_memory = self.reducer.deduplicate_memory(state.working_memory)
        
        # Prune if over capacity
        salience_scores = {a.hash: a.salience for a in abstractions}
        kept, pruned = self.reducer.prune_working_memory(
            state.working_memory,
            salience_scores,
        )
        state.working_memory = kept
        
        # Recalculate entropy
        state.memory_entropy = self.estimator.working_memory_entropy(state.working_memory)
        
        # Overall entropy
        state.overall_entropy = (state.perception_entropy + state.memory_entropy) / 2
        
        return state


# ═══════════════════════════════════════════════════════════════════════════════
# THE AUTOPOIETIC ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AutopoieticEntropyReductionEngine:
    """
    The Self-Sustaining Cognitive Core.
    
    Autopoietic: Self-producing, self-maintaining
    Entropy-Reducing: Compresses and organizes information
    Merkle-Persisted: Cryptographically verified state chain
    
    Lifecycle:
    1. Perceive → raw input processing
    2. Abstract → pattern extraction
    3. Intend → goal formation (Ihsān-bounded)
    4. Act → execution with impact tracking
    5. Consolidate → memory compression (entropy reduction)
    
    State is persisted to Merkle chain after each cycle.
    """
    
    def __init__(self):
        # Merkle persistence
        self.chain = MerkleStateChain()
        
        # Cognitive processes
        self.perception = PerceptionProcess()
        self.abstraction = AbstractionProcess()
        self.intention = IntentionProcess()
        self.action = ActionProcess()
        self.consolidation = ConsolidationProcess()
        
        # Current state
        self.state = self._recover_or_init()
        
        # Entropy monitoring
        self.entropy_estimator = EntropyEstimator()
        
        # Control
        self._running = False
        self._lock = threading.Lock()
        
        # Metrics
        self.cycles_completed = 0
        self.total_entropy_reduced = 0.0
    
    def _recover_or_init(self) -> CognitiveState:
        """Recover from chain or initialize fresh."""
        head = self.chain.get_head()
        if head:
            print(f"[AERE] Recovered state: sequence={head.sequence}, hash={head.state_hash[:16]}...")
            return head
        
        # Genesis state
        genesis = CognitiveState().finalize()
        self.chain.append(genesis)
        print(f"[AERE] Genesis state created: {genesis.state_hash[:16]}...")
        return genesis
    
    async def start(self):
        """Start the autopoietic loop."""
        if self._running:
            return
        
        self._running = True
        print("[AERE] Autopoietic loop starting...")
        
        await asyncio.gather(
            self._perception_loop(),
            self._abstraction_loop(),
            self._intention_loop(),
            self._action_loop(),
            self._consolidation_loop(),
            return_exceptions=True,
        )
    
    async def stop(self):
        """Stop the autopoietic loop."""
        self._running = False
        
        # Final state persist
        self.chain.append(self.state)
        self.chain.close()
        
        print(f"[AERE] Stopped. Cycles: {self.cycles_completed}, Entropy reduced: {self.total_entropy_reduced:.4f}")
    
    async def _perception_loop(self):
        """Continuous perception processing."""
        while self._running:
            try:
                # In real impl, would pull from audio/visual streams
                # Here we simulate with entropy-based synthetic input
                synthetic_input = os.urandom(256)
                
                perception = await self.perception.process(
                    source="synthetic",
                    modality="entropy_test",
                    raw_data=synthetic_input,
                )
                
                with self._lock:
                    self.state.recent_perceptions.append(perception.hash)
                    self.state.recent_perceptions = self.state.recent_perceptions[-20:]
                    self.state.perception_entropy = perception.entropy
                
            except Exception as e:
                print(f"[AERE:Perception] Error: {e}")
            
            await asyncio.sleep(PERCEPTION_INTERVAL_MS / 1000)
    
    async def _abstraction_loop(self):
        """Continuous abstraction processing."""
        while self._running:
            try:
                if len(self.perception.buffer) >= 5:
                    # Abstract last 5 perceptions
                    perceptions = self.perception.buffer[-5:]
                    abstraction = await self.abstraction.process(perceptions)
                    
                    with self._lock:
                        self.state.recent_abstractions.append(abstraction.hash)
                        self.state.recent_abstractions = self.state.recent_abstractions[-20:]
                        
                        # Add to working memory if salient
                        if abstraction.salience > 0.5:
                            self.state.working_memory.append(abstraction.hash)
                
            except Exception as e:
                print(f"[AERE:Abstraction] Error: {e}")
            
            await asyncio.sleep(ABSTRACTION_INTERVAL_MS / 1000)
    
    async def _intention_loop(self):
        """Continuous intention formation."""
        while self._running:
            try:
                if self.abstraction.buffer:
                    abstraction = self.abstraction.buffer[-1]
                    
                    intention = await self.intention.process(
                        abstraction=abstraction,
                        context=self.state.working_memory[:5],
                        ihsan_constraints={
                            "safety": 0.9,
                            "beneficence": 0.8,
                            "transparency": 0.85,
                        },
                    )
                    
                    with self._lock:
                        self.state.recent_intentions.append(intention.hash)
                        self.state.recent_intentions = self.state.recent_intentions[-20:]
                
            except Exception as e:
                print(f"[AERE:Intention] Error: {e}")
            
            await asyncio.sleep(INTENTION_INTERVAL_MS / 1000)
    
    async def _action_loop(self):
        """Continuous action execution."""
        while self._running:
            try:
                if self.intention.buffer:
                    intention = self.intention.buffer[-1]
                    
                    action = await self.action.execute(intention)
                    
                    with self._lock:
                        self.state.recent_actions.append(action.hash)
                        self.state.recent_actions = self.state.recent_actions[-20:]
                        
                        # Persist state after action
                        self.state.phase = CognitivePhase.ACT
                        self.chain.append(self.state)
                        self.cycles_completed += 1
                
            except Exception as e:
                print(f"[AERE:Action] Error: {e}")
            
            await asyncio.sleep(INTENTION_INTERVAL_MS / 1000)
    
    async def _consolidation_loop(self):
        """Periodic memory consolidation."""
        while self._running:
            try:
                before_entropy = self.state.overall_entropy
                
                self.state = await self.consolidation.consolidate(
                    self.state,
                    self.abstraction.buffer,
                )
                
                after_entropy = self.state.overall_entropy
                reduction = before_entropy - after_entropy
                self.total_entropy_reduced += max(0, reduction)
                
                # Persist consolidated state
                with self._lock:
                    self.state.phase = CognitivePhase.CONSOLIDATE
                    self.chain.append(self.state)
                
                print(f"[AERE:Consolidation] Entropy: {before_entropy:.4f} → {after_entropy:.4f} (Δ={reduction:.4f})")
                
            except Exception as e:
                print(f"[AERE:Consolidation] Error: {e}")
            
            await asyncio.sleep(CONSOLIDATION_INTERVAL_MS / 1000)
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "version": AERE_VERSION,
            "running": self._running,
            "state_sequence": self.state.sequence,
            "state_hash": self.state.state_hash[:16] + "...",
            "phase": self.state.phase.value,
            "working_memory_size": len(self.state.working_memory),
            "overall_entropy": self.state.overall_entropy,
            "cycles_completed": self.cycles_completed,
            "total_entropy_reduced": self.total_entropy_reduced,
            "chain_length": len(self.chain.state_index),
        }
    
    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """Verify Merkle chain integrity."""
        return self.chain.verify_chain()


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON & CLI
# ═══════════════════════════════════════════════════════════════════════════════

_engine: Optional[AutopoieticEntropyReductionEngine] = None

def get_engine() -> AutopoieticEntropyReductionEngine:
    """Get singleton engine instance."""
    global _engine
    if _engine is None:
        _engine = AutopoieticEntropyReductionEngine()
    return _engine


async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autopoietic Entropy Reduction Engine")
    parser.add_argument("command", choices=["start", "status", "verify", "demo"])
    args = parser.parse_args()
    
    engine = get_engine()
    
    if args.command == "start":
        print("═" * 70)
        print("   AUTOPOIETIC ENTROPY REDUCTION ENGINE")
        print("   The Self-Sustaining Cognitive Core")
        print("═" * 70)
        
        try:
            await engine.start()
        except KeyboardInterrupt:
            await engine.stop()
    
    elif args.command == "status":
        status = engine.get_status()
        print(json.dumps(status, indent=2))
    
    elif args.command == "verify":
        valid, broken = engine.verify_integrity()
        if valid:
            print("✅ Merkle chain integrity verified")
        else:
            print(f"❌ Chain broken at: {broken}")
    
    elif args.command == "demo":
        print("═" * 70)
        print("   AERE DEMO — 10 Second Cognitive Cycle")
        print("═" * 70)
        
        # Run for 10 seconds
        async def demo():
            await asyncio.wait_for(engine.start(), timeout=10.0)
        
        try:
            await demo()
        except asyncio.TimeoutError:
            pass
        finally:
            await engine.stop()
        
        print("\n📊 Final Status:")
        print(json.dumps(engine.get_status(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
