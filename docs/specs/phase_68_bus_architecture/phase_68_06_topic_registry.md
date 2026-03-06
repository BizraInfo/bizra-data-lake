# Phase 68.06 — Topic Registry (Event Envelope + Routing)

## Context

The Rust event bus has 12 hardcoded topics. The Python EventBus supports
arbitrary topics via wildcard subscriptions. This spec defines a canonical
topic registry, event envelope schema, and routing rules that both runtimes
share.

---

## 1. Requirements

### FR-1: Canonical Topic Namespace
All topics follow a hierarchical dot-separated scheme.
Registration validates against the canonical set.

### FR-2: Typed Event Envelope
Every event carries: id, timestamp, type, actor, correlation_id,
prev_hash, payload, signature. Deterministic serialization for hashing.

### FR-3: Tiered Activation
Not all topics are active at all times. Topics activate based on
node degradation level and federation state.

### FR-4: Cross-Runtime Sync
Python and Rust topic constants must stay in sync. The registry
exports a JSON schema that both runtimes validate against.

---

## 2. Event Envelope

```python
@dataclass(frozen=True)
class BusEvent:
    """Canonical event envelope for both Event Bus and Event Log."""
    event_id: str               # blake3(canonical_content) hex
    timestamp: int              # unix ms (monotonic within node)
    event_type: str             # topic from registry
    actor: bytes                # ed25519 public key
    correlation_id: str         # mission/loop/action linkage
    prev_hash: bytes            # blake3 of previous event (chain)
    payload: dict               # event-specific data
    signature: bytes            # ed25519(actor, canonical_content)

    def canonical_bytes(self) -> bytes:
        """Deterministic byte representation for hashing/signing."""
        content = json.dumps({
            "ts": self.timestamp,
            "type": self.event_type,
            "actor": self.actor.hex(),
            "correlation_id": self.correlation_id,
            "prev": self.prev_hash.hex(),
            "payload": self.payload,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return content

    def verify(self, public_key: bytes) -> bool:
        """Verify Ed25519 signature."""
        return ed25519_verify(public_key, self.canonical_bytes(), self.signature)
```

---

## 3. Topic Registry

```python
# core/bus/topics.py

class TopicTier(Enum):
    CONSTITUTIONAL = 0      # Always active
    COGNITIVE = 1           # Active at degradation >= 2
    LIFECYCLE = 2           # Always active
    ECONOMIC = 3            # Active during ticker
    FEDERATION = 4          # Active when peers > 0
    POLICY = 5              # Always active
    MISSION = 6             # Active during orchestration
    OMEGA = 7               # Active during Omega loops

# Canonical topic definitions
TOPIC_REGISTRY: dict[str, TopicDef] = {
    # Tier 0: Constitutional
    "action.intent":            TopicDef(tier=CONSTITUTIONAL, schema="action_intent_v1"),
    "action.receipt":           TopicDef(tier=CONSTITUTIONAL, schema="action_receipt_v1"),
    "action.receipt.failed":    TopicDef(tier=CONSTITUTIONAL, schema="action_receipt_v1"),
    "action.cancelled":         TopicDef(tier=CONSTITUTIONAL, schema="action_cancelled_v1"),
    "ihsan.breach":             TopicDef(tier=CONSTITUTIONAL, schema="ihsan_breach_v1",
                                         min_priority=Priority.EMERGENCY),
    "poi.credit":               TopicDef(tier=CONSTITUTIONAL, schema="poi_credit_v1"),

    # Tier 1: Cognitive
    "memory.promoted":          TopicDef(tier=COGNITIVE, schema="memory_event_v1"),
    "memory.retrieved":         TopicDef(tier=COGNITIVE, schema="memory_event_v1"),
    "reflex.compiled":          TopicDef(tier=COGNITIVE, schema="reflex_event_v1"),
    "reflex.cache_hit":         TopicDef(tier=COGNITIVE, schema="reflex_event_v1"),
    "reflex.pruned":            TopicDef(tier=COGNITIVE, schema="reflex_event_v1"),

    # Tier 2: Lifecycle
    "node.lifecycle.boot":      TopicDef(tier=LIFECYCLE, schema="lifecycle_v1"),
    "node.lifecycle.shutdown":  TopicDef(tier=LIFECYCLE, schema="lifecycle_v1"),
    "node.lifecycle.upgrade":   TopicDef(tier=LIFECYCLE, schema="lifecycle_v1"),
    "session.end":              TopicDef(tier=LIFECYCLE, schema="session_v1"),
    "system.lifecycle":         TopicDef(tier=LIFECYCLE, schema="lifecycle_v1"),

    # Tier 3: Economic
    "economy.seed_minted":      TopicDef(tier=ECONOMIC, schema="economy_v1"),
    "economy.bloom_accrued":    TopicDef(tier=ECONOMIC, schema="economy_v1"),
    "economy.zakat":            TopicDef(tier=ECONOMIC, schema="economy_v1"),
    "economy.demurrage":        TopicDef(tier=ECONOMIC, schema="economy_v1"),
    "economy.asabiyyah":        TopicDef(tier=ECONOMIC, schema="economy_v1"),

    # Tier 4: Federation
    "federation.peer_seen":     TopicDef(tier=FEDERATION, schema="federation_v1"),
    "federation.attestation.sent":     TopicDef(tier=FEDERATION, schema="attestation_v1"),
    "federation.attestation.received": TopicDef(tier=FEDERATION, schema="attestation_v1"),
    "federation.attestation.reciprocal": TopicDef(tier=FEDERATION, schema="attestation_v1"),
    "federation.diffusion":     TopicDef(tier=FEDERATION, schema="diffusion_v1"),

    # Tier 5: Policy
    "policy.fate.vetoed":       TopicDef(tier=POLICY, schema="policy_v1"),
    "policy.telescript.denied": TopicDef(tier=POLICY, schema="policy_v1"),
    "policy.invariant.violation": TopicDef(tier=POLICY, schema="invariant_v1",
                                           min_priority=Priority.CRITICAL),

    # Tier 6: Mission
    "mission.created":          TopicDef(tier=MISSION, schema="mission_v1"),
    "mission.planned":          TopicDef(tier=MISSION, schema="mission_v1"),
    "mission.executed":         TopicDef(tier=MISSION, schema="mission_v1"),
    "mission.verified":         TopicDef(tier=MISSION, schema="mission_v1"),
    "mission.failed":           TopicDef(tier=MISSION, schema="mission_v1"),

    # Tier 7: Omega
    "omega.started":            TopicDef(tier=OMEGA, schema="omega_v1"),
    "omega.iteration":          TopicDef(tier=OMEGA, schema="omega_v1"),
    "omega.proved":             TopicDef(tier=OMEGA, schema="omega_v1"),
    "omega.cancelled":          TopicDef(tier=OMEGA, schema="omega_v1"),
    "omega.paused":             TopicDef(tier=OMEGA, schema="omega_v1"),
    "omega.completed":          TopicDef(tier=OMEGA, schema="omega_v1"),
}
```

---

## 4. Topic Validation

```
CLASS TopicRegistry:
    INIT():
        self._topics = TOPIC_REGISTRY
        self._active_tiers: set[TopicTier] = {CONSTITUTIONAL, LIFECYCLE, POLICY}

    DEF activate_tier(tier: TopicTier):
        self._active_tiers.add(tier)

    DEF deactivate_tier(tier: TopicTier):
        IF tier IN (CONSTITUTIONAL, LIFECYCLE, POLICY):
            RAISE CannotDeactivateConstitutional(tier)
        self._active_tiers.discard(tier)

    DEF validate(topic: str) -> bool:
        """Check if topic is known and its tier is active."""
        defn = self._topics.get(topic)
        IF defn IS None:
            # Allow wildcard parent topics (e.g., "economy.*")
            RETURN any(t.startswith(topic.rstrip("*")) for t in self._topics)
        RETURN defn.tier IN self._active_tiers

    DEF get_min_priority(topic: str) -> Priority:
        """Return minimum priority for a topic (e.g., ihsan.breach = EMERGENCY)."""
        defn = self._topics.get(topic)
        RETURN defn.min_priority IF defn ELSE Priority.NORMAL

    DEF export_json() -> str:
        """Export registry as JSON for Rust cross-validation."""
        RETURN json.dumps({
            topic: {"tier": defn.tier.value, "schema": defn.schema}
            FOR topic, defn IN self._topics.items()
        }, sort_keys=True, indent=2)
```

---

## 5. Cross-Runtime Sync

The topic registry exports a `topics.json` file that the Rust
`bizra-hooks` crate validates against at build time.

```
core/bus/topics.py              # Python SSoT
  |
  v (export_json())
core/bus/topics.json            # Generated artifact
  |
  v (build.rs include!)
bizra-hooks/src/subscribers.rs  # Rust validates topic strings
```

CI gate: `python -c "from core.bus.topics import TopicRegistry; r = TopicRegistry(); print(r.export_json())" | diff - core/bus/topics.json`

If they drift, CI fails.

---

## 6. Degradation-Level Topic Activation

```
Level 0 (Python-only):
  Active: Constitutional + Lifecycle + Policy + Mission

Level 1 (+ AHK):
  Active: + Economic

Level 2 (+ LLM):
  Active: + Cognitive + Omega

Level 3 (+ LivingMemory):
  Active: + Cognitive (memory.promoted, memory.retrieved)

Level 4 (+ Rust EventBus):
  Active: ALL tiers including Federation
```

---

## 7. TDD Anchors (10 tests)

```python
class TestTopicValidation:
    def test_known_topic_validates()
    def test_unknown_topic_rejected()
    def test_wildcard_parent_validates()
    def test_constitutional_always_active()
    def test_federation_inactive_by_default()

class TestTierActivation:
    def test_activate_economic_tier()
    def test_cannot_deactivate_constitutional()

class TestEventEnvelope:
    def test_canonical_bytes_deterministic()
    def test_signature_verification()
    def test_chain_linkage()

class TestCrossRuntime:
    def test_export_json_matches_rust_topics()
```

---

## 8. Non-Goals

- **No topic creation at runtime.** All topics are defined at build time.
  Dynamic topics are a federation protocol concern for later phases.
- **No schema enforcement on payload.** Payload is `dict`. Schema names
  in TopicDef are for documentation, not runtime validation (yet).
- **No event bus replacement.** This registry wraps the existing EventBus
  with validation, not replaces it.
