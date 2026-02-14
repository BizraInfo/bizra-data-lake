# BIZRA Integration Contracts v2.3.0

## Purpose

This document defines the exact API contracts between BIZRA components. All implementations MUST conform to these contracts to ensure interoperability.

---

## 1. NTU Contract

### 1.1 Core Types

```python
# Python Type Definitions
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from collections import deque

@dataclass
class NTUState:
    """Immutable state vector in [0,1]^3 + iteration counter."""
    belief: float      # Certainty score [0.0, 1.0]
    entropy: float     # Uncertainty measure [0.0, 1.0]
    potential: float   # Predictive capacity [0.0, 1.0]
    iteration: int     # Observation counter

@dataclass
class NTUConfig:
    """Configuration with auto-normalized weights."""
    window_size: int = 5
    alpha: float = 0.4        # Temporal consistency weight
    beta: float = 0.35        # Neural prior weight
    gamma: float = 0.25       # Belief persistence weight
    ihsan_threshold: float = 0.95
    epsilon: float = 0.01     # Convergence criterion

@dataclass
class Observation:
    """Single observation with optional metadata."""
    value: float              # [0.0, 1.0]
    timestamp: float          # Unix timestamp
    metadata: Optional[Dict[str, Any]] = None
```

### 1.2 Interface Contract

```python
class INTUProtocol:
    """NTU interface contract. All implementations MUST satisfy this."""

    def observe(self, value: float, metadata: Optional[Dict] = None) -> NTUState:
        """
        Process a single observation.

        Args:
            value: Observation value in [0.0, 1.0]. Values outside range are clamped.
            metadata: Optional metadata dict.

        Returns:
            Updated NTUState.

        Invariants:
            - POST: 0.0 <= state.belief <= 1.0
            - POST: 0.0 <= state.entropy <= 1.0
            - POST: 0.0 <= state.potential <= 1.0
            - POST: state.iteration == old_state.iteration + 1
        """
        ...

    @property
    def state(self) -> NTUState:
        """Current state (read-only)."""
        ...

    @property
    def memory(self) -> deque:
        """Observation window (read-only)."""
        ...

    def detect_pattern(self, threshold: Optional[float] = None) -> Tuple[bool, float]:
        """
        Check if current belief exceeds threshold.

        Args:
            threshold: Override config.ihsan_threshold if provided.

        Returns:
            (detected, confidence) where:
            - detected: True if belief >= threshold
            - confidence: Current belief value
        """
        ...

    def reset(self) -> None:
        """Reset to initial state."""
        ...
```

### 1.3 Functional API Contract

```python
def minimal_ntu_detect(
    observations: List[float],
    threshold: float = 0.95,
    window_size: int = 5
) -> Tuple[bool, float]:
    """
    Stateless pattern detection.

    Args:
        observations: List of values in [0.0, 1.0].
        threshold: Detection threshold (default: 0.95).
        window_size: Sliding window size (default: 5).

    Returns:
        (detected, final_belief)

    Complexity:
        Time: O(n) where n = len(observations)
        Space: O(window_size)
    """
    ...
```

---

## 2. FATE Gate Contract

### 2.1 Core Types

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any

class FATEDimension(Enum):
    """Constitutional AI dimensions."""
    FIDELITY = "fidelity"         # No secrets in input
    ACCOUNTABILITY = "accountability"  # All ops logged
    TRANSPARENCY = "transparency"     # No obfuscation
    ETHICS = "ethics"               # Block dangerous patterns

class GateAction(Enum):
    """Gate decision actions."""
    ALLOW = "allow"
    BLOCK = "block"

@dataclass
class GateResult:
    """Result of FATE validation."""
    passed: bool
    action: GateAction
    score: float                  # Aggregate score [0.0, 1.0]
    dimension_scores: Dict[FATEDimension, float]
    reason: str
    tool_name: str
    timestamp: float
```

### 2.2 Interface Contract

```python
class IFATEGateProtocol:
    """FATE Gate interface contract."""

    async def validate(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """
        Validate tool invocation against Constitutional AI principles.

        Args:
            tool_name: Name of tool being invoked (e.g., "Bash", "Write").
            tool_input: Tool input parameters.
            context: Optional execution context.

        Returns:
            GateResult with decision and reasoning.

        Thresholds:
            - Bash: 0.98
            - Write: 0.96
            - Edit: 0.96
            - WebFetch: 0.95
            - default: 0.95
        """
        ...

    def get_threshold(self, tool_name: str) -> float:
        """Get threshold for specific tool."""
        ...
```

### 2.3 Decorator Contract

```python
def fate_guarded(
    threshold: float = 0.95,
    dimensions: Optional[List[FATEDimension]] = None
):
    """
    Decorator for FATE-guarded functions.

    Args:
        threshold: Minimum FATE score required.
        dimensions: Specific dimensions to check (default: all).

    Raises:
        FATEGateError: If validation fails.

    Usage:
        @fate_guarded(threshold=0.95)
        async def risky_operation():
            ...
    """
    ...
```

---

## 3. Session DAG Contract

### 3.1 Core Types

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

class SessionState(Enum):
    """Session state machine states."""
    INIT = "init"
    ACTIVE = "active"
    COMPUTING = "computing"
    VALIDATED = "validated"
    COMMITTED = "committed"

@dataclass
class MerkleNode:
    """Node in the Merkle DAG."""
    state: SessionState
    timestamp: float
    parent_hash: Optional[str]
    payload_hash: str
    node_hash: str  # blake3(state || timestamp || parent_hash || payload_hash)

@dataclass
class SessionProof:
    """Merkle proof of session lineage."""
    root_hash: str
    path: List[MerkleNode]
    leaf_hash: str
```

### 3.2 Interface Contract

```python
class ISessionStateMachineProtocol:
    """Session state machine interface contract."""

    def begin(self) -> MerkleNode:
        """
        Transition: INIT -> ACTIVE

        Returns:
            New MerkleNode with state=ACTIVE.

        Raises:
            InvalidTransitionError: If current state is not INIT.
        """
        ...

    def start_compute(self, payload: Dict[str, Any]) -> MerkleNode:
        """
        Transition: ACTIVE -> COMPUTING

        Args:
            payload: Inference request payload.

        Returns:
            New MerkleNode with state=COMPUTING.
        """
        ...

    def validate(self, result: Dict[str, Any]) -> MerkleNode:
        """
        Transition: COMPUTING -> VALIDATED

        Args:
            result: Inference result to validate.

        Returns:
            New MerkleNode with state=VALIDATED.
        """
        ...

    def commit(self) -> MerkleNode:
        """
        Transition: VALIDATED -> COMMITTED

        Returns:
            Final MerkleNode (immutable).
        """
        ...

    def branch(self, name: str) -> 'ISessionStateMachineProtocol':
        """
        Create a branch from current state.

        Args:
            name: Branch identifier.

        Returns:
            New session machine starting from current node.
        """
        ...

    def get_proof(self) -> SessionProof:
        """Get Merkle proof of current state lineage."""
        ...
```

---

## 4. Cognitive Budget Contract

### 4.1 Core Types

```python
from enum import Enum
from dataclasses import dataclass

class BudgetTier(Enum):
    """Inference model tiers."""
    NANO = "nano"      # 0.5B-1.5B
    MICRO = "micro"    # 1.5B-3B
    MESO = "meso"      # 7B-13B
    MACRO = "macro"    # 13B-34B
    MEGA = "mega"      # 70B+

class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = 0.2
    MODERATE = 0.5
    COMPLEX = 0.7
    EXPERT = 0.9

@dataclass
class BudgetAllocation:
    """Budget allocation result."""
    tier: BudgetTier
    tokens: int
    timeout_seconds: float
    model_candidates: List[str]
```

### 4.2 Interface Contract

```python
class ICognitiveBudgetProtocol:
    """Cognitive budget allocator interface contract."""

    def allocate(
        self,
        task_complexity: TaskComplexity,
        domain: str = "general",
        latency_requirement: Optional[float] = None
    ) -> BudgetAllocation:
        """
        Allocate cognitive budget for a task.

        Args:
            task_complexity: Estimated task complexity.
            domain: Task domain (general, code, math, science, reasoning).
            latency_requirement: Max acceptable latency in seconds.

        Returns:
            BudgetAllocation with tier, tokens, timeout, and model candidates.

        Signature (7-3-6-9):
            - 70% of traffic -> NANO/MICRO
            - 30% allocation -> MESO
            - 6% allocation -> MACRO
            - 9% allocation -> MEGA
        """
        ...

    def track_usage(
        self,
        tier: BudgetTier,
        tokens_used: int,
        latency: float
    ) -> None:
        """Track budget consumption."""
        ...

    def get_utilization(self) -> Dict[BudgetTier, float]:
        """Get current utilization per tier."""
        ...
```

---

## 5. Compute Market Contract

### 5.1 Core Types

```python
from dataclasses import dataclass
from enum import Enum

class LicenseType(Enum):
    """License types in compute market."""
    INFERENCE = "inference"
    PATTERN = "pattern"
    FEDERATION = "federation"

@dataclass
class InferenceLicense:
    """Harberger-taxed inference license."""
    license_id: str
    node_id: str
    license_type: LicenseType
    self_assessed_value: float  # In compute credits
    tax_rate: float             # Annual rate (default: 0.07)
    issued_at: float            # Unix timestamp
    expires_at: float           # Unix timestamp

@dataclass
class MarketMetrics:
    """Compute market health metrics."""
    total_licenses: int
    active_licenses: int
    gini_coefficient: float     # Must be <= 0.35
    total_value_locked: float
```

### 5.2 Interface Contract

```python
class IComputeMarketProtocol:
    """Compute market interface contract."""

    def acquire_license(
        self,
        node_id: str,
        license_type: LicenseType,
        self_assessed_value: float
    ) -> InferenceLicense:
        """
        Acquire a new license via Harberger mechanism.

        Args:
            node_id: Requesting node ID.
            license_type: Type of license.
            self_assessed_value: Self-assessed value (determines tax).

        Returns:
            Newly issued license.

        Raises:
            GiniViolationError: If acquisition would push Gini > 0.35.
        """
        ...

    def force_sale(
        self,
        license_id: str,
        buyer_node_id: str
    ) -> InferenceLicense:
        """
        Force sale at self-assessed value.

        Args:
            license_id: License to purchase.
            buyer_node_id: Buyer's node ID.

        Returns:
            Transferred license with new owner.
        """
        ...

    def check_license(
        self,
        node_id: str,
        license_type: LicenseType
    ) -> Tuple[bool, Optional[InferenceLicense]]:
        """Check if node has valid license."""
        ...

    def collect_taxes(self) -> float:
        """Collect due taxes from all licenses."""
        ...

    def get_metrics(self) -> MarketMetrics:
        """Get current market metrics."""
        ...
```

---

## 6. PCI Protocol Contract

### 6.1 Core Types

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class AgentType(Enum):
    """Agent types in PCI protocol."""
    USER = "user"
    MODEL = "model"
    SOVEREIGN = "sovereign"

class RejectCode(Enum):
    """PCI rejection codes."""
    R001_TTL_EXPIRED = "R001"
    R002_INVALID_SIGNATURE = "R002"
    R003_IHSAN_BELOW_THRESHOLD = "R003"
    R004_SNR_BELOW_THRESHOLD = "R004"
    R005_MALFORMED_ENVELOPE = "R005"

@dataclass
class PCIEnvelope:
    """Proof-Carrying Inference envelope."""
    version: str = "1.0"
    agent_type: AgentType
    node_id: str
    timestamp: str              # ISO-8601
    ttl_seconds: int
    ihsan_score: float
    snr_score: float
    payload: Dict[str, Any]
    signature: str              # Base64 Ed25519 signature
    public_key: str             # Base64 Ed25519 public key

@dataclass
class VerificationResult:
    """Result of PCI verification."""
    valid: bool
    reject_code: Optional[RejectCode]
    reason: str
```

### 6.2 Interface Contract

```python
class IPCIProtocol:
    """PCI protocol interface contract."""

    def create_envelope(
        self,
        payload: Dict[str, Any],
        agent_type: AgentType,
        ihsan_score: float,
        snr_score: float,
        ttl_seconds: int = 3600
    ) -> PCIEnvelope:
        """
        Create and sign a PCI envelope.

        Args:
            payload: Message payload.
            agent_type: Type of agent creating envelope.
            ihsan_score: Ihsan quality score [0.0, 1.0].
            snr_score: Signal-to-noise ratio [0.0, 1.0].
            ttl_seconds: Time-to-live in seconds.

        Returns:
            Signed PCIEnvelope.

        Signature:
            sign(DOMAIN_PREFIX || canonical_json(payload))
        """
        ...

    def verify_envelope(
        self,
        envelope: PCIEnvelope
    ) -> VerificationResult:
        """
        Verify a PCI envelope.

        Verification chain:
            1. Check TTL not expired
            2. Verify Ed25519 signature
            3. Check ihsan_score >= 0.95
            4. Check snr_score >= 0.85

        Returns:
            VerificationResult with validity and rejection reason.
        """
        ...
```

---

## 7. Federation Contract

### 7.1 Core Types

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class NodeState(Enum):
    """Federation node states."""
    JOINING = "joining"
    ACTIVE = "active"
    SUSPECT = "suspect"
    DEAD = "dead"

class MessageType(Enum):
    """Gossip message types."""
    PING = "ping"
    PONG = "pong"
    SYNC = "sync"
    PATTERN = "pattern"

@dataclass
class NodeInfo:
    """Federation node information."""
    node_id: str
    address: str
    state: NodeState
    belief: float              # NTU belief
    patterns_count: int
    last_seen: float           # Unix timestamp

@dataclass
class GossipMessage:
    """Gossip protocol message."""
    message_type: MessageType
    sender: NodeInfo
    payload: Dict[str, Any]
    signature: str
```

### 7.2 Interface Contract

```python
class IFederationNodeProtocol:
    """Federation node interface contract."""

    async def start(self) -> None:
        """Start federation node (gossip + consensus)."""
        ...

    async def stop(self) -> None:
        """Gracefully stop federation node."""
        ...

    async def join(self, bootstrap_peers: List[str]) -> None:
        """
        Join federation via bootstrap peers.

        Args:
            bootstrap_peers: List of peer addresses (host:port).
        """
        ...

    async def broadcast_pattern(
        self,
        pattern_id: str,
        pattern_data: Dict[str, Any],
        snr_score: float
    ) -> None:
        """
        Broadcast elevated pattern to federation.

        Args:
            pattern_id: Unique pattern identifier.
            pattern_data: Pattern payload.
            snr_score: Pattern SNR score (must be >= 0.95 for elevation).

        Raises:
            SNRBelowThresholdError: If snr_score < 0.95.
        """
        ...

    def get_peers(self) -> List[NodeInfo]:
        """Get current peer list."""
        ...

    def get_state(self) -> NodeState:
        """Get current node state."""
        ...
```

### 7.3 Consensus Contract

```python
class IConsensusProtocol:
    """BFT consensus interface contract."""

    async def propose(
        self,
        proposal_id: str,
        proposal_data: Dict[str, Any]
    ) -> bool:
        """
        Propose a value for consensus.

        Args:
            proposal_id: Unique proposal identifier.
            proposal_data: Proposal payload.

        Returns:
            True if proposal was accepted (2f+1 votes).

        Timeout: 30 seconds
        """
        ...

    async def vote(
        self,
        proposal_id: str,
        accept: bool,
        reason: Optional[str] = None
    ) -> None:
        """
        Vote on a proposal.

        Args:
            proposal_id: Proposal to vote on.
            accept: True to accept, False to reject.
            reason: Optional rejection reason.
        """
        ...
```

---

## 8. Inference Gateway Contract

### 8.1 Core Types

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

class ComputeTier(Enum):
    """Inference compute tiers."""
    EDGE = "edge"      # Always-on, low power
    LOCAL = "local"    # On-demand, high power
    POOL = "pool"      # Federated compute

@dataclass
class InferenceRequest:
    """Inference request."""
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    tier_hint: Optional[ComputeTier] = None

@dataclass
class InferenceResult:
    """Inference result with PCI envelope."""
    text: str
    tokens_used: int
    latency_ms: float
    model_id: str
    tier: ComputeTier
    envelope: PCIEnvelope
```

### 8.2 Interface Contract

```python
class IInferenceGatewayProtocol:
    """Inference gateway interface contract."""

    async def infer(
        self,
        request: InferenceRequest
    ) -> InferenceResult:
        """
        Execute inference with automatic model selection.

        Args:
            request: Inference request.

        Returns:
            InferenceResult with PCI envelope.

        Backends (priority order):
            1. LM Studio (192.168.56.1:1234)
            2. Ollama (localhost:11434)
        """
        ...

    async def batch_infer(
        self,
        requests: List[InferenceRequest]
    ) -> List[InferenceResult]:
        """
        Execute batch inference.

        Args:
            requests: List of inference requests.

        Returns:
            List of results (same order as requests).
        """
        ...

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        ...

    def get_backend_status(self) -> Dict[str, bool]:
        """Get backend availability status."""
        ...
```

---

## 9. A2A Protocol Contract

### 9.1 Core Types

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class Capability(Enum):
    """Agent capabilities."""
    READ = "read"
    WRITE = "write"
    CODE = "code"
    ANALYZE = "analyze"
    RESEARCH = "research"

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentCard:
    """Agent identity and capabilities."""
    agent_id: str
    name: str
    capabilities: List[Capability]
    ihsan_score: float
    public_key: str

@dataclass
class TaskCard:
    """Task specification."""
    task_id: str
    title: str
    description: str
    required_capabilities: List[Capability]
    requester: AgentCard
    assignee: Optional[AgentCard]
    status: TaskStatus
    result: Optional[Dict[str, Any]]
```

### 9.2 Interface Contract

```python
class IA2AEngineProtocol:
    """A2A engine interface contract."""

    def register_agent(self, card: AgentCard) -> None:
        """Register an agent with the A2A network."""
        ...

    def submit_task(self, task: TaskCard) -> str:
        """
        Submit a task for execution.

        Args:
            task: Task specification.

        Returns:
            Task ID for tracking.
        """
        ...

    async def route_task(self, task_id: str) -> AgentCard:
        """
        Route task to capable agent.

        Args:
            task_id: Task to route.

        Returns:
            Selected agent card.

        Selection criteria:
            1. Has required capabilities
            2. Highest ihsan_score
            3. Lowest current load
        """
        ...

    async def get_task_result(
        self,
        task_id: str,
        timeout: float = 60.0
    ) -> TaskCard:
        """
        Wait for task completion and return result.

        Args:
            task_id: Task to wait for.
            timeout: Max wait time in seconds.

        Returns:
            Completed TaskCard with result.

        Raises:
            TimeoutError: If task not completed within timeout.
        """
        ...
```

---

## 10. Cross-Contract Constants

### 10.1 Global Thresholds

```python
# Quality thresholds
IHSAN_THRESHOLD: float = 0.95
SNR_THRESHOLD: float = 0.85
GINI_CONSTRAINT: float = 0.40  # Must match core/integration/constants.py

# NTU parameters
NTU_WINDOW_SIZE: int = 5
NTU_ALPHA: float = 0.4
NTU_BETA: float = 0.35
NTU_GAMMA: float = 0.25

# Federation parameters
GOSSIP_FANOUT: int = 3
GOSSIP_INTERVAL_SEC: float = 5.0
BFT_QUORUM: str = "2f+1"
CLOCK_SKEW_SEC: float = 120.0  # 2-minute tolerance window

# PCI parameters
DOMAIN_PREFIX: bytes = b"bizra-pci-v1:"
MAX_TTL_SECONDS: int = 3600
```

### 10.2 Error Codes

```python
class BIZRAErrorCode(Enum):
    """Standardized error codes across all contracts."""
    # NTU errors (1xx)
    E101_INVALID_OBSERVATION = 101
    E102_NAN_VALUE = 102
    E103_INF_VALUE = 103

    # FATE errors (2xx)
    E201_GATE_BLOCKED = 201
    E202_FIDELITY_VIOLATION = 202
    E203_ETHICS_VIOLATION = 203

    # Session errors (3xx)
    E301_INVALID_TRANSITION = 301
    E302_MERKLE_VERIFICATION_FAILED = 302

    # Budget errors (4xx)
    E401_BUDGET_EXHAUSTED = 401
    E402_TIER_UNAVAILABLE = 402

    # Market errors (5xx)
    E501_GINI_VIOLATION = 501
    E502_LICENSE_EXPIRED = 502
    E503_INSUFFICIENT_FUNDS = 503

    # PCI errors (6xx)
    E601_TTL_EXPIRED = 601
    E602_INVALID_SIGNATURE = 602
    E603_IHSAN_BELOW_THRESHOLD = 603
    E604_SNR_BELOW_THRESHOLD = 604

    # Federation errors (7xx)
    E701_CONSENSUS_TIMEOUT = 701
    E702_PEER_UNREACHABLE = 702
    E703_PATTERN_REJECTED = 703

    # Inference errors (8xx)
    E801_BACKEND_UNAVAILABLE = 801
    E802_MODEL_NOT_FOUND = 802
    E803_TOKEN_LIMIT_EXCEEDED = 803

    # A2A errors (9xx)
    E901_NO_CAPABLE_AGENT = 901
    E902_TASK_TIMEOUT = 902
```

---

## 11. Contract Versioning

```yaml
contract_version: "2.3.0"
compatible_versions:
  - "2.3.x"
  - "2.2.x"
deprecated_versions:
  - "2.1.x"
  - "2.0.x"
  - "1.x.x"

breaking_changes:
  "2.3.0":
    - "Added NTU contract"
    - "Added Session DAG contract"
    - "Added Cognitive Budget contract"
    - "Added Compute Market contract"
  "2.2.0":
    - "Added PCI envelope versioning"
    - "Changed FATE threshold for Bash to 0.98"
  "2.1.0":
    - "Initial contract specification"
```

---

**END OF CONTRACTS**

*All implementations MUST pass module-level tests in `tests/core/` corresponding to each contract domain.*
