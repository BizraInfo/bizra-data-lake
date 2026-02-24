# Phase 45.2 — Node-to-Node Communication Protocol

> **Version:** 0.1.0 | **Status:** Specification + Pseudocode
> **Standing on Giants:** Lamport (logical clocks, 1978) · SWIM (gossip, 2002) · Noise Protocol (Perrin, 2018) · BIZRA Federation (Phase 7)

## 2.1 Purpose

Define how sovereign nodes discover each other, establish secure channels,
exchange knowledge, delegate tasks, and maintain privacy boundaries.

The key tension: **maximum collaboration with minimum information exposure.**

## 2.2 Existing Infrastructure

```
REUSE:
  core/federation/gossip.py          -- SWIM-style discovery, NodeInfo, NodeState
  core/federation/secure_transport.py -- DTLS/Noise encrypted channels
  core/federation/consensus.py       -- PBFT consensus engine
  core/federation/protocol.py        -- FederatedPattern, wire formats
  core/hashtable/bloom_filter.py     -- Phase 44, set reconciliation
  core/hashtable/merkle_tree.py      -- Phase 44, data integrity
```

## 2.3 Discovery Protocol — Pseudocode

```
MODULE: core.mesh.discovery

IMPORT: GossipEngine from core.federation.gossip
IMPORT: BloomFilter from core.hashtable.bloom_filter
IMPORT: NodePublicCard from core.node_identity.public_card

CLASS MeshDiscovery:
  """
  SWIM-based node discovery with capability-aware routing.

  Nodes announce themselves via gossip, sharing their public card
  and a Bloom filter of their knowledge domains. Other nodes use
  this for intelligent task routing.
  """

  FIELDS:
    gossip: GossipEngine
    known_nodes: dict[str, NodePublicCard]  -- node_id -> card
    capability_index: dict[str, set[str]]   -- domain -> {node_ids}

  CONSTRUCTOR(self_identity: NodeIdentity, bind_address: str):
    self.gossip = GossipEngine(
      node_id = self_identity.node_id,
      bind_addr = bind_address,
    )
    self.known_nodes = {}
    self.capability_index = {}

    -- Register handler for incoming node cards
    self.gossip.on_message(MessageType.NODE_CARD, self._handle_node_card)

  METHOD announce() -> None:
    """Broadcast our public card to the mesh."""
    card = self_identity.to_public_card()
    self.gossip.broadcast(card.to_gossip_message())

  METHOD _handle_node_card(msg: GossipMessage) -> None:
    """Process incoming node announcement."""
    card = NodePublicCard.from_bytes(msg.payload)

    -- Verify self-signature (reject tampered cards)
    IF NOT card.verify_self_signature():
      LOG.warning(f"Rejected card from {msg.sender}: invalid signature")
      RETURN

    -- Store / update
    self.known_nodes[card.node_id] = card

    -- Update capability index
    FOR domain IN card.cognitive_domains:
      IF domain NOT IN self.capability_index:
        self.capability_index[domain] = set()
      self.capability_index[domain].add(card.node_id)

  METHOD find_nodes_for_task(task: TaskDescriptor) -> list[NodePublicCard]:
    """
    Find best nodes for a given task based on:
    1. Domain match (cognitive profile)
    2. Compute fit (enough resources)
    3. Availability (online now)
    4. Reputation (quality history)
    """
    candidates = []

    FOR node_id, card IN self.known_nodes.items():
      -- Skip offline nodes
      IF NOT self.gossip.is_alive(node_id):
        CONTINUE

      -- Domain match score
      domain_score = domain_match(task.required_domains, card.cognitive_domains)

      -- Compute fit
      IF task.min_compute_score > card.compute_score:
        CONTINUE

      -- Reputation threshold
      IF card.reputation < task.min_reputation:
        CONTINUE

      candidates.append((card, domain_score))

    -- Sort by domain match * reputation (best first)
    candidates.sort(key=LAMBDA (card, ds): ds * card.reputation, reverse=true)
    RETURN [card for card, _ in candidates[:task.max_nodes]]
```

## 2.4 Shared/Private Boundaries — Pseudocode

```
MODULE: core.mesh.privacy_boundary

ENUM ShareLevel:
  """What granularity of data a node consents to share."""
  NOTHING     = 0   -- fully isolated (default for new nodes)
  PROOFS_ONLY = 1   -- share PCI receipts and Merkle proofs
  SUMMARIES   = 2   -- share compressed summaries of knowledge
  EMBEDDINGS  = 3   -- share vector embeddings (not raw text)
  TASK_RESULTS = 4  -- share outputs of assigned tasks
  FULL_SYNC   = 5   -- share everything (opted-in, not default)

CLASS PrivacyBoundary:
  """
  Per-node consent configuration for data sharing.
  Human sets this. System enforces it cryptographically.
  """

  FIELDS:
    share_level: ShareLevel            -- overall level
    per_domain_overrides: dict[str, ShareLevel]  -- fine-grained
    blocked_nodes: set[str]            -- node_ids I refuse to share with
    trusted_nodes: set[str]            -- node_ids with elevated access

  CONSTRUCTOR():
    -- PRIVACY BY DEFAULT: nothing shared until human opts in
    self.share_level = ShareLevel.NOTHING
    self.per_domain_overrides = {}
    self.blocked_nodes = set()
    self.trusted_nodes = set()

  METHOD can_share(data_type: str, recipient_node_id: str) -> bool:
    """Check if sharing this data to this node is permitted."""
    IF recipient_node_id IN self.blocked_nodes:
      RETURN false

    required_level = DATA_TYPE_TO_LEVEL[data_type]
    effective_level = self.share_level

    -- Trusted nodes get one level higher access
    IF recipient_node_id IN self.trusted_nodes:
      effective_level = min(ShareLevel.FULL_SYNC, effective_level + 1)

    RETURN effective_level >= required_level

  CONST DATA_TYPE_TO_LEVEL = {
    "receipt":     ShareLevel.PROOFS_ONLY,
    "merkle_proof": ShareLevel.PROOFS_ONLY,
    "summary":     ShareLevel.SUMMARIES,
    "embedding":   ShareLevel.EMBEDDINGS,
    "task_result": ShareLevel.TASK_RESULTS,
    "raw_memory":  ShareLevel.FULL_SYNC,
  }
```

## 2.5 Secure Messaging — Pseudocode

```
MODULE: core.mesh.messaging

IMPORT: SecureTransportManager from core.federation.secure_transport
IMPORT: canonical_bytes, blake3_digest from core.proof_engine.canonical
IMPORT: MerkleTree from core.hashtable.merkle_tree

ENUM MessageType:
  TASK_REQUEST      = "task_request"       -- ask node to do work
  TASK_RESPONSE     = "task_response"      -- return completed work
  KNOWLEDGE_SYNC    = "knowledge_sync"     -- share knowledge fragment
  BLOOM_EXCHANGE    = "bloom_exchange"     -- exchange Bloom filters
  CONSENSUS_VOTE    = "consensus_vote"     -- vote on proposal
  HEARTBEAT         = "heartbeat"          -- alive signal
  REPUTATION_UPDATE = "reputation_update"  -- broadcast impact proof

DATACLASS MeshMessage:
  """
  Signed, encrypted message between two nodes.
  """
  msg_type: MessageType
  sender_id: str
  recipient_id: str
  payload: bytes              -- encrypted content
  nonce: bytes                -- replay protection
  timestamp: float            -- UTC unix timestamp
  signature: bytes            -- sender signs (type + recipient + payload_hash + nonce + ts)
  receipt_chain: str          -- Merkle root of message audit trail

  METHOD verify_integrity(sender_pubkey: bytes) -> bool:
    """Verify signature and freshness."""
    -- Check timestamp within skew tolerance
    IF abs(time.time() - self.timestamp) > UNIFIED_CLOCK_SKEW_SECONDS:
      RETURN false

    -- Verify signature
    signed_data = canonical_bytes({
      "type": self.msg_type,
      "recipient": self.recipient_id,
      "payload_hash": blake3_digest(self.payload).hex(),
      "nonce": self.nonce.hex(),
      "timestamp": self.timestamp,
    })
    RETURN Ed25519.verify(sender_pubkey, signed_data, self.signature)

CLASS SecureChannel:
  """
  Bidirectional encrypted channel between two nodes.
  Built on core.federation.secure_transport (Noise Protocol).
  """

  FIELDS:
    transport: SecureTransportManager
    local_identity: NodeIdentity
    remote_card: NodePublicCard
    message_log: MerkleTree        -- audit trail of all messages
    privacy: PrivacyBoundary

  METHOD send(msg_type: MessageType, payload: dict) -> MeshMessage:
    """Encrypt, sign, and send a message."""
    -- Check privacy boundary
    data_type = MSG_TYPE_TO_DATA_TYPE[msg_type]
    IF NOT self.privacy.can_share(data_type, self.remote_card.node_id):
      RAISE PrivacyViolationError(f"Sharing {data_type} not permitted")

    -- Serialize and encrypt
    raw = canonical_bytes(payload)
    encrypted = self.transport.encrypt(raw)

    -- Build message
    msg = MeshMessage(
      msg_type = msg_type,
      sender_id = self.local_identity.node_id,
      recipient_id = self.remote_card.node_id,
      payload = encrypted,
      nonce = generate_nonce(),
      timestamp = time.time(),
      signature = None,  -- filled below
      receipt_chain = self.message_log.root_hex,
    )

    -- Sign
    msg.signature = self.local_identity.sign(msg.signed_data())

    -- Audit trail
    self.message_log.append(blake3_digest(msg.to_bytes()))

    -- Transmit
    self.transport.send(msg.to_bytes())
    RETURN msg

  METHOD receive() -> tuple[MessageType, dict]:
    """Receive, verify, decrypt a message."""
    raw = self.transport.receive()
    msg = MeshMessage.from_bytes(raw)

    -- Verify integrity
    IF NOT msg.verify_integrity(self.remote_card.public_key):
      RAISE IntegrityError("Message signature or freshness check failed")

    -- Decrypt
    plaintext = self.transport.decrypt(msg.payload)
    payload = json.loads(plaintext)

    -- Audit trail
    self.message_log.append(blake3_digest(raw))

    RETURN (msg.msg_type, payload)
```

## 2.6 Knowledge Sync Protocol — Pseudocode

```
MODULE: core.mesh.knowledge_sync

IMPORT: BloomFilter from core.hashtable.bloom_filter
IMPORT: MerkleTree, MerkleProof from core.hashtable.merkle_tree

CLASS KnowledgeSync:
  """
  Efficient knowledge reconciliation between nodes.

  Step 1: Exchange Bloom filters (O(1) — "what do you know?")
  Step 2: Identify gaps via set difference
  Step 3: Exchange only missing knowledge (bandwidth-efficient)
  Step 4: Verify integrity via Merkle proofs

  Standing on Giants: Minsky & Trachtenberg (set reconciliation, 2002)
  """

  FIELDS:
    local_bloom: BloomFilter       -- my knowledge fingerprint
    local_merkle: MerkleTree       -- integrity tree of my knowledge
    knowledge_store: KnowledgeStore

  METHOD sync_with(channel: SecureChannel) -> SyncResult:
    """Full sync protocol with a remote node."""

    -- Phase 1: Exchange Bloom filters
    channel.send(MessageType.BLOOM_EXCHANGE, {
      "bloom": self.local_bloom.to_bytes().hex(),
      "merkle_root": self.local_merkle.root_hex,
      "item_count": self.local_bloom.estimated_count(),
    })

    _, remote_data = channel.receive()
    remote_bloom = BloomFilter.from_bytes(bytes.fromhex(remote_data["bloom"]))

    -- Phase 2: Find what remote has that we don't
    missing_locally = []
    FOR item_id IN self.knowledge_store.all_ids():
      item_bytes = item_id.encode()
      IF item_bytes NOT IN remote_bloom:
        -- Remote probably doesn't have this (may be false negative: impossible.
        -- May be false positive: they DO have it. That's fine — we skip.)
        PASS

    -- Phase 3: Request items remote has that we lack
    -- (We send our Bloom; remote checks which of THEIR items
    --  are NOT in OUR Bloom, then sends those)
    channel.send(MessageType.KNOWLEDGE_SYNC, {
      "request": "diff",
      "our_bloom": self.local_bloom.to_bytes().hex(),
    })

    _, diff_response = channel.receive()
    new_items = diff_response.get("items", [])

    -- Phase 4: Verify each item with Merkle proof
    verified_count = 0
    FOR item IN new_items:
      proof = MerkleProof.from_dict(item["proof"])
      IF proof.verify() AND proof.root == remote_data["merkle_root"]:
        self.knowledge_store.insert(item["data"])
        self.local_bloom.add(item["id"].encode())
        self.local_merkle.append(item["data_bytes"])
        verified_count += 1
      ELSE:
        LOG.warning(f"Rejected item {item['id']}: proof verification failed")

    RETURN SyncResult(
      items_received = verified_count,
      items_rejected = len(new_items) - verified_count,
      new_merkle_root = self.local_merkle.root_hex,
    )
```

## 2.7 Task Delegation Protocol — Pseudocode

```
MODULE: core.mesh.task_delegation

DATACLASS TaskDescriptor:
  """What work needs to be done and what's needed to do it."""
  task_id: str                    -- unique, BLAKE3-based
  task_type: str                  -- "reasoning" | "computation" | "validation"
  description: str                -- natural language task description
  required_domains: list[str]     -- expertise needed
  min_compute_score: float        -- minimum compute capability
  min_reputation: float           -- minimum reputation to accept
  max_nodes: int                  -- how many nodes to assign
  timeout_seconds: int
  reward_seed: float              -- SEED tokens offered
  privacy_level: ShareLevel       -- what data the task involves

DATACLASS TaskAssignment:
  task: TaskDescriptor
  assigned_to: str                -- node_id
  assigned_at: datetime
  deadline: datetime
  status: "pending" | "accepted" | "in_progress" | "completed" | "failed"

DATACLASS TaskResult:
  task_id: str
  node_id: str
  result: dict                    -- the actual output
  snr_score: float                -- self-assessed quality
  compute_time_seconds: float
  receipt_digest: str             -- PCI receipt proving execution
  merkle_proof: MerkleProof       -- proof result is in node's ledger

CLASS TaskDelegator:
  """
  Assigns tasks to capable nodes and aggregates results.
  """

  METHOD delegate(task: TaskDescriptor, mesh: MeshDiscovery) -> list[TaskAssignment]:
    """Find suitable nodes and send task requests."""
    candidates = mesh.find_nodes_for_task(task)

    IF len(candidates) == 0:
      RAISE NoCapableNodesError(f"No nodes match task requirements")

    assignments = []
    FOR card IN candidates:
      channel = open_channel(card)
      channel.send(MessageType.TASK_REQUEST, task.to_dict())

      _, response = channel.receive()
      IF response["accepted"]:
        assignments.append(TaskAssignment(
          task = task,
          assigned_to = card.node_id,
          assigned_at = utc_now(),
          deadline = utc_now() + timedelta(seconds=task.timeout_seconds),
          status = "accepted",
        ))

    RETURN assignments

  METHOD collect_results(assignments: list[TaskAssignment]) -> AggregatedResult:
    """
    Collect and merge results from multiple nodes.

    Standing on Giants:
      Condorcet (jury theorem, 1785) — majority of independent judges
      converge on truth faster than individuals.
    """
    results = []
    FOR assignment IN assignments:
      IF assignment.status == "completed":
        result = TaskResult.from_dict(assignment.result_data)

        -- Verify PCI receipt (no unverified claims)
        IF NOT verify_receipt(result.receipt_digest):
          LOG.warning(f"Rejected result from {assignment.assigned_to}: invalid receipt")
          CONTINUE

        results.append(result)

    -- Aggregate: weight by SNR score * reputation
    IF len(results) == 0:
      RETURN AggregatedResult(status="no_valid_results")

    IF len(results) == 1:
      RETURN AggregatedResult(status="single", result=results[0])

    -- Multi-node: weighted consensus
    RETURN weighted_consensus(results)
```

## 2.8 TDD Anchors

```
TEST_SUITE: tests/core/mesh/

  test_discovery:
    - announce() broadcasts valid gossip message
    - node card with invalid signature is rejected
    - capability_index correctly populated
    - find_nodes_for_task() filters by domain, compute, reputation
    - offline nodes excluded from results

  test_privacy_boundary:
    - default share level is NOTHING
    - blocked node always rejected
    - trusted node gets one level upgrade
    - per-domain overrides work correctly

  test_messaging:
    - send/receive roundtrip succeeds
    - tampered message fails verify_integrity()
    - expired message (clock skew) rejected
    - message logged in Merkle audit trail

  test_knowledge_sync:
    - Bloom exchange identifies missing items
    - Merkle proof verification accepts valid items
    - tampered items rejected
    - sync is idempotent (re-sync same data = no change)

  test_task_delegation:
    - delegate() finds capable nodes
    - no capable nodes raises error
    - results with invalid receipts rejected
    - weighted consensus produces reasonable output
```
