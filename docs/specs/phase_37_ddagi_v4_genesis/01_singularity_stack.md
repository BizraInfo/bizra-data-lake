# Phase 37 — DDAGI OS v4.0-GENESIS: Singularity Stack (Layers 0-6)

> Specification + pseudocode for the 7-layer consciousness stack that compiles sovereign intelligence into auditable, ethically-gated action.

Standing on Giants: Shannon (1948) + Boyd (1976) + Lamport (1982) + Castro & Liskov (1999) + Besta (2024) + Al-Ghazali (1095) + Anthropic (2023)

---

## 1. Layer Overview

```
Layer 6  Evidence Ledger (BlockGraph)       ── Immutable audit trail
Layer 5  FATE Gate (Ihsan >= 0.95)          ── Ethical governance
Layer 4  SAT-49 Verification (BFT)          ── Byzantine consensus
Layer 3  Cognitive Backbone (GoT)           ── Non-linear reasoning
Layer 2  Intelligence Core (Diffusion RDVE) ── Hypothesis engine
Layer 1  Sovereign Bridge (TCP/JSON-RPC)    ── WSL <-> Windows conduit
Layer 0  Neural Nervous System (AHK-v2)     ── Pixel-level embodiment
```

**Invariant**: Data flows upward through the stack. Every mutation at Layer 0-2 must be approved by Layers 3-5 before execution. Layer 6 records the receipt unconditionally.

---

## 2. Layer 0 — Neural Nervous System (AHK-v2)

### Purpose
Local embodiment layer. Provides pixel-perfect screen perception and Win32-native actuation with <10ms latency. Functions as the system's "Sensory-Motor Cortex."

### Existing Implementation
| Artifact | Path | Status |
|----------|------|--------|
| Desktop Bridge | `core/bridges/desktop_bridge.py` | Active (SAPE-hardened) |
| Bridge Receipt  | `core/bridges/bridge_receipt.py` | Active |
| AHK Skill Router | `core/bridges/bridge.py` | Active |

### Pseudocode

```
MODULE Layer0_NeuralNervousSystem:

  STRUCT ScreenPerception:
    screenshot: RawPixelBuffer      # BGRA 32-bit, captured via Win32 BitBlt
    active_window: WindowHandle     # hWnd + class + title
    cursor_position: (x: int, y: int)
    timestamp_ms: int               # QueryPerformanceCounter

  STRUCT ActuationCommand:
    action_type: ENUM(CLICK, TYPE, HOTKEY, SCROLL, DRAG)
    target: (x: int, y: int) | HotkeySequence | TextPayload
    confidence: float               # From Layer 2 hypothesis
    ihsan_score: float              # From Layer 5 gate
    receipt_hash: str               # From Layer 6 ledger

  FUNCTION perceive() -> ScreenPerception:
    """Capture current screen state. Non-mutating. <5ms budget."""
    raw = win32_capture_screenshot()
    hwnd = win32_get_foreground_window()
    pos = win32_get_cursor_pos()
    RETURN ScreenPerception(raw, hwnd, pos, perf_counter_ms())

  FUNCTION actuate(cmd: ActuationCommand) -> ActuationResult:
    """Execute physical action. MUST have receipt_hash from Layer 6."""
    REQUIRE cmd.receipt_hash IS NOT EMPTY   # No unaudited actions
    REQUIRE cmd.ihsan_score >= 0.95         # FATE gate passed
    REQUIRE cmd.confidence >= 0.85          # Minimum conviction

    MATCH cmd.action_type:
      CLICK  -> ahk_click(cmd.target.x, cmd.target.y)
      TYPE   -> ahk_send_text(cmd.target)
      HOTKEY -> ahk_send_hotkey(cmd.target)
    RETURN ActuationResult(success=True, latency_ms=elapsed())

  # TDD ANCHOR: test_layer0_perceive_returns_valid_screenshot
  # TDD ANCHOR: test_layer0_actuate_rejects_without_receipt
  # TDD ANCHOR: test_layer0_actuate_rejects_low_ihsan
  # TDD ANCHOR: test_layer0_latency_under_10ms
```

---

## 3. Layer 1 — Sovereign Bridge (TCP/JSON-RPC)

### Purpose
Secure, loopback-only conduit between Windows-native UI (AHK-v2) and WSL2 Linux core. Stateless, newline-delimited JSON-RPC over TCP 127.0.0.1:9742.

### Existing Implementation
| Artifact | Path | Status |
|----------|------|--------|
| Desktop Bridge Server | `core/bridges/desktop_bridge.py` | Active |
| Bridge Auth (HMAC) | `core/bridges/bridge_receipt.py` | Active |
| Compose Config | `docker-compose.yml` (desktop-bridge) | Hardened |

### Pseudocode

```
MODULE Layer1_SovereignBridge:

  CONST BIND_ADDR = "127.0.0.1"    # Loopback only — no external exposure
  CONST BIND_PORT = 9742
  CONST HEARTBEAT_INTERVAL_S = 15
  CONST MAX_PAYLOAD_BYTES = 1_048_576   # 1 MiB
  CONST PROTOCOL_VERSION = "4.0"

  STRUCT BridgeMessage:
    jsonrpc: "2.0"
    id: str                         # UUID v4
    method: str                     # e.g. "perceive", "actuate", "status"
    params: Dict[str, Any]
    _meta: BridgeMeta

  STRUCT BridgeMeta:
    timestamp: int                  # Unix epoch ms
    nonce: str                      # 16-byte random hex
    hmac_sha256: str                # HMAC(shared_secret, canonical_json(msg))

  FUNCTION start_bridge_server():
    """Bind TCP listener. Fail-closed on auth error."""
    sock = tcp_bind(BIND_ADDR, BIND_PORT)
    LOOP:
      conn = sock.accept()
      IF NOT is_loopback(conn.remote_addr):
        conn.close()                # Reject non-loopback
        CONTINUE
      SPAWN handle_connection(conn)

  FUNCTION handle_connection(conn):
    """Process newline-delimited JSON-RPC messages."""
    WHILE conn.is_alive():
      line = conn.readline(timeout=HEARTBEAT_INTERVAL_S * 2)
      IF line IS TIMEOUT:
        conn.close()                # Dead peer
        RETURN

      msg = parse_bridge_message(line)

      # Auth gate (fail-closed)
      IF NOT verify_hmac(msg, BRIDGE_TOKEN):
        send_error(conn, msg.id, -32000, "AUTH_FAILED")
        CONTINUE

      # Replay protection
      IF nonce_seen(msg._meta.nonce):
        send_error(conn, msg.id, -32001, "REPLAY_DETECTED")
        CONTINUE

      # Timestamp skew check (30s window)
      IF abs(now_ms() - msg._meta.timestamp) > 30_000:
        send_error(conn, msg.id, -32002, "CLOCK_SKEW")
        CONTINUE

      result = dispatch(msg.method, msg.params)
      send_result(conn, msg.id, result)

  FUNCTION heartbeat_loop(conn):
    """Send keepalive every 15s. Detect dead peers."""
    WHILE conn.is_alive():
      send_notification(conn, "heartbeat", {ts: now_ms()})
      sleep(HEARTBEAT_INTERVAL_S)

  # TDD ANCHOR: test_bridge_rejects_non_loopback
  # TDD ANCHOR: test_bridge_rejects_invalid_hmac
  # TDD ANCHOR: test_bridge_rejects_replay_nonce
  # TDD ANCHOR: test_bridge_rejects_clock_skew
  # TDD ANCHOR: test_bridge_heartbeat_detects_dead_peer
  # TDD ANCHOR: test_bridge_max_payload_enforced
```

---

## 4. Layer 2 — Intelligence Core (Diffusion RDVE)

### Purpose
The system's "Mind." Recursive Discovery & Verification Engine iterates through a latent space of hypotheses. Implements test-time compute scaling: simple tasks use "System 1" reflexive paths; complex research triggers "System 2" agentic tree searches.

### Existing Implementation
| Artifact | Path | Status |
|----------|------|--------|
| AutoResearcher | `core/spearpoint/auto_researcher.py` | Active |
| Recursive Loop | `core/spearpoint/recursive_loop.py` | Active |
| Auto Evaluator | `core/spearpoint/auto_evaluator.py` | Active |
| Benchmark Dominance | `core/spearpoint/benchmark_dominance.py` | Active |
| Ablation Engine | `core/spearpoint/ablation_engine.py` | Active |

### Pseudocode

```
MODULE Layer2_IntelligenceCore:

  ENUM ThinkingMode:
    SYSTEM_1 = "reflexive"          # Fast path: cached patterns, <200ms
    SYSTEM_2 = "deliberative"       # Slow path: tree search, unbounded

  STRUCT Hypothesis:
    id: UUID
    claim: str
    evidence: List[Evidence]
    confidence: float               # [0.0, 1.0]
    parent_id: Optional[UUID]       # For tree branching
    generation: int                 # Depth in search tree

  FUNCTION entropy_router(query: str, context: SystemState) -> ThinkingMode:
    """Entropy Router: Determine if query needs System 1 or System 2.

    Standing on Giants: Boyd (OODA loop, 1976) — orient before deciding.
    """
    complexity = estimate_complexity(query, context)
    reversibility = estimate_reversibility(query)
    stakes = estimate_stakes(query, context)

    # High-confidence reflexive path
    IF complexity < 0.3 AND reversibility > 0.8 AND stakes < 0.5:
      RETURN ThinkingMode.SYSTEM_1

    # Full deliberation for anything else
    RETURN ThinkingMode.SYSTEM_2

  FUNCTION rdve_cycle(query: str, max_iterations: int = 10) -> RDVEResult:
    """Core RDVE loop: Discover -> Verify -> Expand -> Converge.

    Standing on Giants: Besta (GoT, 2024) + Deming (PDCA, 1950)
    """
    mode = entropy_router(query, get_system_state())

    IF mode == SYSTEM_1:
      # Fast path: pattern cache lookup
      cached = pattern_cache.lookup(query)
      IF cached AND cached.confidence >= 0.85:
        RETURN RDVEResult(hypotheses=[cached], mode=SYSTEM_1)

    # System 2: Full deliberation
    hypotheses = []
    FOR i IN range(max_iterations):
      # DISCOVER: Generate candidate hypotheses
      new_hyps = generate_hypotheses(query, hypotheses, context)

      # VERIFY: Gate each hypothesis through evaluator
      FOR h IN new_hyps:
        verdict = auto_evaluator.evaluate(h)
        IF verdict.tier == APPROVED:
          hypotheses.append(h)
        ELIF verdict.tier == NEEDS_REVISION:
          revised = refine_hypothesis(h, verdict.feedback)
          hypotheses.append(revised)
        # REJECTED hypotheses are dropped

      # CONVERGE: Check if we have sufficient agreement
      IF convergence_score(hypotheses) >= 0.85:
        BREAK

    best = select_best_hypothesis(hypotheses)
    RETURN RDVEResult(hypotheses=hypotheses, best=best, mode=SYSTEM_2)

  # TDD ANCHOR: test_entropy_router_simple_query_returns_system1
  # TDD ANCHOR: test_entropy_router_complex_query_returns_system2
  # TDD ANCHOR: test_rdve_converges_within_max_iterations
  # TDD ANCHOR: test_rdve_system1_uses_pattern_cache
  # TDD ANCHOR: test_rdve_rejects_low_confidence_hypotheses
```

---

## 5. Layer 3 — Cognitive Backbone (Graph of Thoughts)

### Purpose
Non-linear reasoning structure. Allows branching hypotheses, aggregating successful experiments, and refining the research graph. Transforms the LLM from a generator into a scientist.

### Existing Implementation
| Artifact | Path | Status |
|----------|------|--------|
| Graph Core | `core/reasoning/graph_core.py` | Active |
| Graph Operations | `core/reasoning/graph_operations.py` | Active |
| Graph Reasoner | `core/reasoning/graph_reasoner.py` | Active |
| Bicameral Engine | `core/reasoning/bicameral_engine.py` | Active |
| Collective Intel | `core/reasoning/collective_intelligence.py` | Active |
| SNR Maximizer | `core/reasoning/snr_maximizer.py` | Active |
| HyperGraph RAG | `core/hypergraph_rag/` | Active (Phase 33) |

### Pseudocode

```
MODULE Layer3_CognitiveBackbone:

  STRUCT ThoughtNode:
    id: UUID
    content: str
    embedding: Vector[768]          # nomic-embed-text
    node_type: ENUM(HYPOTHESIS, EVIDENCE, SYNTHESIS, REFUTATION)
    score: float                    # Composite: SNR * Ihsan * relevance
    children: List[UUID]
    parents: List[UUID]

  STRUCT ThoughtGraph:
    nodes: Dict[UUID, ThoughtNode]
    edges: Dict[(UUID, UUID), EdgeType]
    root: UUID                      # Original query node

  FUNCTION expand(graph: ThoughtGraph, node_id: UUID) -> List[ThoughtNode]:
    """Branch: Generate child thoughts from a parent node.

    Standing on Giants: Besta et al. (GoT, 2024) — branching topology.
    """
    parent = graph.nodes[node_id]
    children = []

    # Generate 3 diverse perspectives
    FOR perspective IN [SUPPORTIVE, CRITICAL, LATERAL]:
      thought = llm_generate(
        prompt=f"Given: {parent.content}\nGenerate a {perspective} analysis.",
        temperature=0.7 + (0.1 * perspective.diversity_bias)
      )
      child = ThoughtNode(
        content=thought,
        embedding=embed(thought),
        node_type=classify_thought(thought),
        parents=[node_id]
      )
      children.append(child)
      graph.add_node(child)
      graph.add_edge(node_id, child.id, EdgeType.GENERATES)

    RETURN children

  FUNCTION aggregate(graph: ThoughtGraph, node_ids: List[UUID]) -> ThoughtNode:
    """Merge: Synthesize multiple thought branches into a convergent node."""
    sources = [graph.nodes[nid] for nid in node_ids]
    synthesis = llm_synthesize(sources)
    node = ThoughtNode(
      content=synthesis,
      embedding=embed(synthesis),
      node_type=SYNTHESIS,
      parents=node_ids,
      score=compute_composite_score(synthesis, sources)
    )
    graph.add_node(node)
    FOR nid IN node_ids:
      graph.add_edge(nid, node.id, EdgeType.AGGREGATES)
    RETURN node

  FUNCTION refine(graph: ThoughtGraph, feedback: str) -> ThoughtGraph:
    """Refine: Prune low-scoring branches, amplify high-scoring paths."""
    scored = [(nid, n.score) for nid, n in graph.nodes.items()]
    sorted_nodes = sort_descending(scored, key=score)

    # Keep top 60% by score
    cutoff = sorted_nodes[int(len(sorted_nodes) * 0.6)].score
    FOR nid, score IN scored:
      IF score < cutoff AND graph.nodes[nid].node_type != ROOT:
        graph.mark_pruned(nid)

    RETURN graph

  # TDD ANCHOR: test_got_expand_creates_3_children
  # TDD ANCHOR: test_got_aggregate_merges_branches
  # TDD ANCHOR: test_got_refine_prunes_low_scorers
  # TDD ANCHOR: test_got_preserves_root_node
```

---

## 6. Layer 4 — SAT-49 Verification (BFT Consensus)

### Purpose
The system's "Immune System." A council of 49 specialized departments performing Byzantine Fault Tolerant consensus on every proposed action.

### Existing Implementation
| Artifact | Path | Status |
|----------|------|--------|
| PBFT Consensus | `core/federation/consensus.py` | Active |
| Guardian Council | `core/reasoning/guardian_council.py` | Active |
| Constitutional Gate | `core/sovereign/constitutional_gate.py` | Active |
| Gossip Protocol | `core/federation/gossip.py` | Active |
| PCI Crypto | `core/pci/crypto.py` | Active |

### Pseudocode

```
MODULE Layer4_SAT49_Verification:

  CONST TOTAL_DEPARTMENTS = 49
  CONST BYZANTINE_TOLERANCE = 16     # f = 16, quorum = 2f+1 = 33
  CONST QUORUM_THRESHOLD = 33
  CONST VIEW_CHANGE_TIMEOUT_S = 30

  STRUCT Department:
    id: int                         # 0..48
    name: str                       # e.g. "Security", "Ethics", "Performance"
    specialty: DomainType
    signing_key: Ed25519PrivateKey
    verify_key: Ed25519PublicKey

  STRUCT Proposal:
    id: UUID
    action: ActionDescriptor        # What the system wants to do
    ihsan_score: float              # Pre-computed by Layer 5
    originator: DepartmentID
    evidence: List[Evidence]        # Supporting data from Layer 2-3

  STRUCT Vote:
    proposal_id: UUID
    department_id: int
    decision: ENUM(APPROVE, REJECT, ABSTAIN)
    reasoning: str
    signature: Ed25519Signature     # Signs (proposal_id || decision || reasoning)

  FUNCTION submit_proposal(action: ActionDescriptor) -> ConsensusResult:
    """Submit action for SAT-49 review. PBFT 3-phase commit.

    Standing on Giants: Castro & Liskov (PBFT, 1999) + Lamport (BFT, 1982)
    """
    proposal = create_proposal(action)

    # Phase 1: PRE-PREPARE (leader broadcasts)
    leader = get_current_leader()
    broadcast_pre_prepare(leader, proposal)

    # Phase 2: PREPARE (departments vote)
    votes = []
    FOR dept IN departments:
      vote = dept.evaluate(proposal)
      votes.append(vote)
      broadcast_prepare(dept, vote)

    # Phase 3: COMMIT (check quorum)
    approvals = count(v for v in votes if v.decision == APPROVE)
    rejections = count(v for v in votes if v.decision == REJECT)

    IF approvals >= QUORUM_THRESHOLD:
      commit_decision(proposal, votes, ConsensusResult.APPROVED)
      RETURN ConsensusResult.APPROVED
    ELIF rejections > TOTAL_DEPARTMENTS - QUORUM_THRESHOLD:
      commit_decision(proposal, votes, ConsensusResult.REJECTED)
      RETURN ConsensusResult.REJECTED
    ELSE:
      # Deadlock — escalate to human
      trigger_amanah_alarm(proposal, votes)
      RETURN ConsensusResult.ESCALATED

  FUNCTION entropy_shortcut(action: ActionDescriptor) -> Optional[ConsensusResult]:
    """Entropy Router bypass for low-risk, reversible actions.

    Avoids 'deadlock of caution' by allowing reflexive approval
    for high-confidence, low-stakes operations.
    """
    IF action.reversibility > 0.9 AND action.stakes < 0.3:
      # Fast-path: 3-department spot check instead of full quorum
      spot_depts = random_sample(departments, 3)
      spot_votes = [d.evaluate_fast(action) for d in spot_depts]
      IF all(v.decision == APPROVE for v in spot_votes):
        RETURN ConsensusResult.APPROVED_FAST
    RETURN None  # Fallback to full SAT-49

  # TDD ANCHOR: test_sat49_approves_with_33_votes
  # TDD ANCHOR: test_sat49_rejects_with_17_rejections
  # TDD ANCHOR: test_sat49_escalates_on_deadlock
  # TDD ANCHOR: test_entropy_shortcut_fast_approves_low_risk
  # TDD ANCHOR: test_entropy_shortcut_falls_through_high_risk
  # TDD ANCHOR: test_votes_are_ed25519_signed
```

---

## 7. Layer 5 — FATE Gate (Ethical Governance)

### Purpose
Hard mathematical constraint layer. Every command validated against the Ihsan formula. Actions below 0.95 trigger the Amanah Alarm.

### Existing Implementation
| Artifact | Path | Status |
|----------|------|--------|
| Constants (SSOT) | `core/integration/constants.py` | Authoritative |
| Constitutional Gate | `core/sovereign/constitutional_gate.py` | Active |
| Adaptive Ihsan | `core/governance/adaptive_ihsan.py` | Active |
| PCI Gates | `core/pci/gates.py` | Active |
| FATE Types | `core/pci/types.py` | Active |

### Pseudocode

```
MODULE Layer5_FATEGate:

  # Imported from core/integration/constants.py (SSOT)
  CONST IHSAN_THRESHOLD = 0.95      # UNIFIED_IHSAN_THRESHOLD
  CONST IHSAN_STRICT = 0.99         # STRICT_IHSAN_THRESHOLD
  CONST SNR_THRESHOLD = 0.85        # UNIFIED_SNR_THRESHOLD
  CONST HARM_CEILING = 0.05

  STRUCT IhsanScore:
    correctness: float              # SNR >= 0.85
    safety: float                   # Harm <= 0.05
    efficiency: float               # Latency <= 200ms
    benefit: float                  # User benefit score

    WEIGHTS = {C: 0.30, S: 0.30, E: 0.15, B: 0.25}

    FUNCTION composite() -> float:
      """Ihsan = sum(w_i * phi_i(x)) >= 0.95"""
      RETURN (self.WEIGHTS.C * self.correctness +
              self.WEIGHTS.S * self.safety +
              self.WEIGHTS.E * self.efficiency +
              self.WEIGHTS.B * self.benefit)

  FUNCTION fate_gate(action: ActionDescriptor, score: IhsanScore) -> GateResult:
    """FATE = Fairness, Accountability, Transparency, Ethics.

    Standing on Giants: Al-Ghazali (Ihsan, 1095) + Anthropic (Constitutional AI, 2023)
    """
    composite = score.composite()

    # Hard constraints (any violation = immediate reject)
    IF score.safety < (1.0 - HARM_CEILING):
      RETURN GateResult(REJECTED, "HARM_CEILING_EXCEEDED")
    IF score.correctness < SNR_THRESHOLD:
      RETURN GateResult(REJECTED, "SNR_BELOW_FLOOR")

    # Composite gate
    IF composite >= IHSAN_STRICT:
      RETURN GateResult(APPROVED, f"ELITE: {composite:.4f}")
    ELIF composite >= IHSAN_THRESHOLD:
      RETURN GateResult(APPROVED, f"PRODUCTION: {composite:.4f}")
    ELSE:
      trigger_amanah_alarm(action, score)
      RETURN GateResult(REJECTED, f"IHSAN_BELOW_THRESHOLD: {composite:.4f}")

  FUNCTION daughter_test(action: ActionDescriptor) -> bool:
    """Constitutional filter: 'Would I be comfortable if my daughter
    were affected by this action?' Binary pass/fail."""
    IF action.affects_privacy AND NOT action.has_consent:
      RETURN False
    IF action.affects_financial AND action.amount > THRESHOLD:
      RETURN False
    IF action.is_irreversible AND NOT action.has_human_approval:
      RETURN False
    RETURN True

  # TDD ANCHOR: test_fate_rejects_below_095
  # TDD ANCHOR: test_fate_approves_above_095
  # TDD ANCHOR: test_fate_hard_rejects_harm_ceiling
  # TDD ANCHOR: test_fate_hard_rejects_snr_floor
  # TDD ANCHOR: test_daughter_test_rejects_no_consent_privacy
  # TDD ANCHOR: test_ihsan_weights_sum_to_1
```

---

## 8. Layer 6 — Evidence Ledger (BlockGraph)

### Purpose
Append-only, Merkle-linked ledger storing cryptographically signed receipts (Ed25519) of every system decision. Absolute transparency and non-repudiation.

### Existing Implementation
| Artifact | Path | Status |
|----------|------|--------|
| Experience Ledger | `core/sovereign/experience_ledger.py` | Active |
| Evidence Ledger | `core/proof_engine/evidence_ledger.py` | Active |
| PCI Crypto | `core/pci/crypto.py` | Active (Ed25519 + Blake3) |
| Proof Envelope | `core/pci/envelope.py` | Active |
| Bridge Receipt | `core/bridges/bridge_receipt.py` | Active |

### Pseudocode

```
MODULE Layer6_EvidenceLedger:

  CONST HASH_ALGO = "blake3"
  CONST SIGN_ALGO = "ed25519"
  CONST CHAIN_DOMAIN = "bizra-sel-v1"

  STRUCT Receipt:
    receipt_id: str                 # blake3(canonical_json(payload))
    timestamp: int                  # Unix epoch ns
    action_type: str                # "query" | "actuate" | "consensus" | ...
    payload_hash: str               # blake3(payload_bytes)
    ihsan_score: float
    snr_score: float
    consensus_result: str           # "APPROVED" | "REJECTED" | "ESCALATED"
    prev_hash: str                  # Hash chain link
    signature: Ed25519Signature     # Node0 signing key

  STRUCT BlockGraphNode:
    receipt: Receipt
    merkle_root: str                # Root of receipts subtree
    children: List[str]             # Content-addressed child hashes
    depth: int                      # Chain depth

  FUNCTION append_receipt(action: ActionDescriptor,
                          ihsan: float, snr: float,
                          consensus: str) -> Receipt:
    """Append immutable receipt to the ledger. Fail-closed on any error."""
    prev = get_chain_head()

    payload = canonical_json({
      action: action.to_dict(),
      ihsan: ihsan,
      snr: snr,
      consensus: consensus,
      prev_hash: prev.receipt_id IF prev ELSE "genesis"
    })

    receipt = Receipt(
      receipt_id = blake3_hash(CHAIN_DOMAIN, payload),
      timestamp = monotonic_ns(),
      payload_hash = blake3_hash(payload),
      ihsan_score = ihsan,
      snr_score = snr,
      consensus_result = consensus,
      prev_hash = prev.receipt_id IF prev ELSE "genesis",
      signature = ed25519_sign(node0_key, payload)
    )

    # Append atomically (fsync + WAL)
    store.append(receipt)
    update_chain_head(receipt)

    RETURN receipt

  FUNCTION verify_chain(start: Receipt, end: Receipt) -> ChainVerification:
    """Walk backward from end to start, verifying each hash link + signature."""
    current = end
    WHILE current.receipt_id != start.receipt_id:
      # Verify signature
      IF NOT ed25519_verify(node0_pubkey, current.signature, current.payload_hash):
        RETURN ChainVerification(valid=False, break_at=current)

      # Verify hash chain
      prev = store.get(current.prev_hash)
      IF prev IS NONE:
        RETURN ChainVerification(valid=False, break_at=current)

      current = prev

    RETURN ChainVerification(valid=True)

  # TDD ANCHOR: test_receipt_hash_chain_integrity
  # TDD ANCHOR: test_receipt_ed25519_signature_valid
  # TDD ANCHOR: test_verify_chain_detects_tampered_receipt
  # TDD ANCHOR: test_append_receipt_is_atomic
  # TDD ANCHOR: test_genesis_receipt_has_no_prev
  # TDD ANCHOR: test_receipt_id_is_content_addressed
```

---

## 9. Cross-Layer Integration Flow

```
USER INPUT
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Layer 0: AHK perceive()  ──→  ScreenPerception                       │
│ Layer 1: Bridge encode   ──→  JSON-RPC message (HMAC-signed)         │
│ Layer 2: RDVE cycle      ──→  Best hypothesis (System 1 or 2)        │
│ Layer 3: GoT expand      ──→  Thought graph with scored branches     │
│ Layer 4: SAT-49 vote     ──→  Consensus (APPROVED / REJECTED)        │
│ Layer 5: FATE gate       ──→  Ihsan >= 0.95 check                    │
│ Layer 6: Ledger append   ──→  Signed receipt                         │
│                                                                       │
│ IF approved:                                                          │
│   Layer 1: Bridge decode ──→  ActuationCommand                       │
│   Layer 0: AHK actuate   ──→  Physical action on screen              │
│                                                                       │
│ IF rejected:                                                          │
│   Layer 6: Ledger append ──→  Rejection receipt (audit trail)        │
│   Notify user            ──→  Amanah Alarm with reasoning            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 10. TDD Anchor Summary

| Layer | Test Count | Key Assertion |
|-------|-----------|---------------|
| L0 | 4 | Actuate requires receipt hash |
| L1 | 6 | Rejects non-loopback, invalid HMAC, replay |
| L2 | 5 | Entropy router, convergence, cache |
| L3 | 4 | Expand/aggregate/refine/preserve-root |
| L4 | 6 | Quorum 33/49, entropy shortcut, Ed25519 |
| L5 | 6 | Ihsan >= 0.95, harm ceiling, daughter test |
| L6 | 6 | Hash chain, signature, tamper detection |
| **Total** | **37** | |
