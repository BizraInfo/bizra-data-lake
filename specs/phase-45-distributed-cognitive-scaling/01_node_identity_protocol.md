# Phase 45.1 — Node Identity Protocol

> **Version:** 0.1.0 | **Status:** Specification + Pseudocode
> **Standing on Giants:** Diffie-Hellman (1976) · Nakamoto (2008) · Zooko (triangle, 2001) · BIZRA Genesis (Phase 25)

## 1.1 Purpose

Define the minimum viable identity for a BIZRA node — the protocol that
turns "a person with a computer" into "a sovereign cognitive node in a
distributed intelligence system."

Human = Node. Every node must be:
- Uniquely identified (cryptographic keypair)
- Self-describing (compute + cognitive profile)
- Reputation-carrying (earned, not assigned)
- Privacy-preserving (nothing leaked by default)

## 1.2 Existing Infrastructure

```
REUSE:
  core/genesis/types.py       -- GenesisState, NodeDesignation
  core/genesis/hardware.py    -- hardware detection
  core/pci/crypto.py          -- Ed25519 signing
  core/pci/types.py           -- PCIEnvelope
  core/federation/gossip.py   -- NodeInfo, NodeState
```

## 1.3 Identity Structure — Pseudocode

```
MODULE: core.node_identity.identity

IMPORT: Ed25519 from core.pci.crypto
IMPORT: blake3_digest from core.proof_engine.canonical
IMPORT: UNIFIED_IHSAN_THRESHOLD from core.integration.constants

CLASS NodeIdentity:
  """
  Immutable cryptographic identity of a BIZRA node.
  Generated once at genesis. Public key is the root of trust.
  """

  FIELDS:
    public_key: bytes[32]         -- Ed25519 public key
    node_id: str                  -- BLAKE3(public_key).hex()[:16]
    genesis_timestamp: datetime   -- UTC, when this node was born
    human_attestation: bool       -- true = human steward exists
    designation: NodeDesignation  -- from core.genesis.types

  CONSTRUCTOR(private_key_seed: bytes | None):
    IF private_key_seed IS None:
      keypair = Ed25519.generate()
    ELSE:
      keypair = Ed25519.from_seed(private_key_seed)

    self.public_key = keypair.public_key
    self.node_id = blake3_digest(self.public_key).hex()[:16]
    self.genesis_timestamp = utc_now()
    self.human_attestation = true
    self.designation = NodeDesignation.SOVEREIGN

    -- Private key stored ONLY in encrypted local vault
    -- NEVER serialized, transmitted, or logged
    self._signer = keypair  -- kept in memory only

  METHOD sign(data: bytes) -> bytes:
    """Sign data with node's private key."""
    RETURN self._signer.sign(data)

  METHOD verify(data: bytes, signature: bytes) -> bool:
    """Verify signature against this node's public key."""
    RETURN Ed25519.verify(self.public_key, data, signature)

  METHOD to_public_card() -> NodePublicCard:
    """Export the shareable public identity (no secrets)."""
    RETURN NodePublicCard(
      node_id = self.node_id,
      public_key = self.public_key,
      genesis_timestamp = self.genesis_timestamp,
      human_attestation = self.human_attestation,
      designation = self.designation,
    )
```

## 1.4 Compute Profile — Pseudocode

```
MODULE: core.node_identity.compute_profile

IMPORT: hardware detection from core.genesis.hardware

CLASS ComputeProfile:
  """
  What this node can contribute to the mesh.
  Auto-detected at boot, updated periodically.
  """

  FIELDS:
    cpu_cores: int
    cpu_model: str
    gpu_vram_mb: int          -- 0 if no GPU
    gpu_model: str            -- "none" if no GPU
    ram_total_mb: int
    storage_available_gb: float
    bandwidth_mbps: float     -- measured via speed test
    availability: AvailabilitySchedule

  CLASSMETHOD detect() -> ComputeProfile:
    """Auto-detect hardware capabilities."""
    hw = core.genesis.hardware.detect()

    RETURN ComputeProfile(
      cpu_cores = hw.cpu_count,
      cpu_model = hw.cpu_model,
      gpu_vram_mb = hw.gpu_vram_mb OR 0,
      gpu_model = hw.gpu_model OR "none",
      ram_total_mb = hw.ram_mb,
      storage_available_gb = hw.free_disk_gb,
      bandwidth_mbps = 0.0,  -- measured lazily on first federation join
      availability = AvailabilitySchedule.always_on(),
    )

  METHOD compute_power_score() -> float:
    """
    Normalized compute contribution score [0.0, 1.0].

    Weighted: GPU > CPU > RAM > Storage > Bandwidth
    This score determines task assignment weight.
    """
    gpu_score = min(self.gpu_vram_mb / 24_000, 1.0) * 0.40
    cpu_score = min(self.cpu_cores / 32, 1.0) * 0.25
    ram_score = min(self.ram_total_mb / 131_072, 1.0) * 0.15
    storage_score = min(self.storage_available_gb / 500, 1.0) * 0.10
    bw_score = min(self.bandwidth_mbps / 1000, 1.0) * 0.10
    RETURN gpu_score + cpu_score + ram_score + storage_score + bw_score

CLASS AvailabilitySchedule:
  """When this node is available for mesh tasks."""

  FIELDS:
    mode: "always_on" | "scheduled" | "on_demand"
    hours_per_day: float  -- declared availability
    timezone: str
    active_windows: list[TimeWindow]  -- only if mode == "scheduled"

  CLASSMETHOD always_on() -> AvailabilitySchedule:
    RETURN AvailabilitySchedule(mode="always_on", hours_per_day=24.0, ...)

  METHOD is_available_now() -> bool:
    IF self.mode == "always_on": RETURN true
    IF self.mode == "on_demand": RETURN false
    RETURN current_time_in(self.timezone) FALLS_WITHIN self.active_windows
```

## 1.5 Cognitive Profile — Pseudocode

```
MODULE: core.node_identity.cognitive_profile

CLASS CognitiveProfile:
  """
  What kind of thinking this node specializes in.
  Combination of local LLM capability + human expertise.
  """

  FIELDS:
    local_llm_id: str              -- e.g., "deepseek-r1:8b"
    local_llm_context_window: int  -- e.g., 32768
    embedding_model_id: str        -- e.g., "nomic-embed-text"
    knowledge_domains: list[str]   -- self-declared + measured
    language_codes: list[str]      -- e.g., ["en", "ar", "es"]
    reasoning_style: str           -- "analytical" | "creative" | "balanced"
    specialization: NodeRole       -- assigned or self-declared

  METHOD cognitive_compatibility(other: CognitiveProfile) -> float:
    """
    How complementary are two nodes' cognitive profiles?
    High diversity = high complementarity (better for distributed reasoning).
    Low diversity = good for redundant validation.
    """
    domain_overlap = jaccard(self.knowledge_domains, other.knowledge_domains)
    -- Complementarity is INVERSE of overlap for reasoning tasks
    -- Same overlap for validation tasks
    RETURN 1.0 - domain_overlap  -- for task assignment diversity

ENUM NodeRole:
  ARCHITECT    = "architect"     -- system design, orchestration
  ANALYST      = "analyst"       -- code analysis, data processing
  PHILOSOPHER  = "philosopher"   -- ethics, long-term reasoning
  AUDITOR      = "auditor"       -- security, validation, testing
  GENERALIST   = "generalist"    -- no specialization (default)
  RESEARCHER   = "researcher"    -- deep investigation, knowledge synthesis
  BUILDER      = "builder"       -- code generation, implementation
```

## 1.6 Reputation Score — Pseudocode

```
MODULE: core.node_identity.reputation

IMPORT: ADL_GINI_THRESHOLD from core.integration.constants

CLASS ReputationScore:
  """
  Earned through verified impact, decays with inactivity.

  Standing on Giants:
    PageRank (1998) — reputation through link structure
    EigenTrust (2003) — distributed reputation in P2P
    Bitcoin proof-of-work (2008) — sybil resistance through cost
  """

  FIELDS:
    score: float              -- [0.0, 1.0]
    impact_history: list[ImpactRecord]
    last_active: datetime
    decay_rate: float         -- 0.01 per day of inactivity
    total_tasks_completed: int
    total_tasks_validated: int
    total_compute_contributed_hours: float

  CONSTRUCTOR():
    self.score = 0.10  -- new nodes start low but not zero
    self.impact_history = []
    self.last_active = utc_now()
    self.decay_rate = 0.01

  METHOD update_from_impact(record: ImpactRecord):
    """Increase reputation based on verified positive impact."""
    -- Only PCI-receipted impacts count
    IF NOT record.has_valid_receipt():
      RETURN  -- ZANN: no unverified claims

    -- SNR-weighted: higher quality work = more reputation gain
    gain = record.snr_score * record.impact_magnitude * 0.05
    self.score = min(1.0, self.score + gain)
    self.last_active = utc_now()
    self.impact_history.append(record)

  METHOD apply_decay():
    """Daily reputation decay for inactive nodes."""
    days_inactive = (utc_now() - self.last_active).days
    IF days_inactive > 0:
      decay = self.decay_rate * days_inactive
      self.score = max(0.0, self.score - decay)

  METHOD governance_weight() -> float:
    """
    Weight in collective decisions (Shura).
    Reputation-proportional but Gini-capped.
    """
    -- Raw weight is reputation score
    -- But ADL Gini gate ensures no node dominates governance
    RETURN self.score  -- actual Gini enforcement at voting level

DATACLASS ImpactRecord:
  task_id: str
  node_id: str
  snr_score: float           -- quality of output
  impact_magnitude: float    -- scale of contribution (0.0 - 1.0)
  receipt_digest: str        -- PCI receipt hash (proof it happened)
  timestamp: datetime
  verified_by: list[str]     -- node_ids that validated this
```

## 1.7 Node Public Card — Wire Format

```
MODULE: core.node_identity.public_card

DATACLASS NodePublicCard:
  """
  The shareable identity card — what other nodes see.
  Never contains private keys or raw personal data.
  """

  node_id: str                    -- BLAKE3(pubkey)[:16]
  public_key: bytes               -- Ed25519, 32 bytes
  genesis_timestamp: datetime
  designation: NodeDesignation
  compute_score: float            -- normalized [0, 1]
  cognitive_domains: list[str]    -- expertise areas
  languages: list[str]
  role: NodeRole
  reputation: float               -- current score
  availability_mode: str          -- "always_on" | "scheduled" | "on_demand"
  federation_endpoints: list[str] -- host:port for P2P connection
  bloom_filter: bytes             -- Phase 44 Bloom filter of known content
  signature: bytes                -- self-signed: sign(canonical(card_fields))

  METHOD verify_self_signature() -> bool:
    """Verify the card was signed by the claimed public key."""
    card_data = canonical_bytes(all_fields_except_signature)
    RETURN Ed25519.verify(self.public_key, card_data, self.signature)

  METHOD to_gossip_message() -> GossipMessage:
    """Convert to federation gossip format for SWIM propagation."""
    RETURN GossipMessage(
      type = MessageType.NODE_CARD,
      sender = self.node_id,
      payload = self.to_bytes(),
      ttl = 5,  -- max gossip hops
    )
```

## 1.8 TDD Anchors

```
TEST_SUITE: tests/core/node_identity/

  test_identity_generation:
    - generate() produces valid Ed25519 keypair
    - node_id is BLAKE3 of public key, 16 hex chars
    - two generates produce different identities
    - sign/verify roundtrip succeeds
    - verify with wrong key fails

  test_compute_profile:
    - detect() returns non-zero values on real hardware
    - compute_power_score() in [0.0, 1.0]
    - GPU-heavy node scores higher than CPU-only
    - score components sum correctly

  test_cognitive_profile:
    - cognitive_compatibility() of identical profiles = 0.0
    - cognitive_compatibility() of disjoint profiles = 1.0
    - domains are case-insensitive compared

  test_reputation:
    - new node starts at 0.10
    - valid impact increases score
    - unverified impact is rejected (ZANN)
    - decay reduces score over time
    - score clamped to [0.0, 1.0]
    - decay stops at 0.0

  test_public_card:
    - self-signature verifies
    - tampered card fails verification
    - card contains no private key bytes
    - to_gossip_message() produces valid GossipMessage
```
