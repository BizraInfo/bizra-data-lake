# Phase 07 — Federation Network: Transport, Discovery, Reflex Diffusion

> Source: Atlas v5.0 — Diagrams D19 (Federation Transport), D29 (Reflex Diffusion Network)
> Status: SPECIFICATION SEALED | SNR: 0.95

---

## 1. Functional Requirements

### FR-070: Node Discovery

Multi-layer discovery ensures every sovereign node can find peers regardless of
network topology. Local-first, progressive expansion to global scope.

**Layer 1 -- mDNS (Local).** Zero-config UDP multicast to `224.0.0.251:5353`.
Announces `_bizra._tcp.local` with NodeID, DID, port, protocol version. Refresh:
`GOSSIP_INTERVAL_MS` (1000ms). For home/office clusters on shared broadcast domain.

**Layer 2 -- Kademlia DHT (Global).** XOR-distance routing, 160-bit IDs from
`BLAKE3(Ed25519_pubkey)`. K-buckets (k=20). Lookup: `O(log N)`. Bootstraps from
seed nodes. Refresh: 3600s per bucket.

**Layer 3 -- Bootstrap Nodes.** Hardcoded seed list (min 3, geo-distributed). No
governance privilege. Config: `discovery_timeout_secs` (30), `max_peers` (100),
`retry_interval_secs` (5). DNS-SD fallback via `enable_dns_sd` flag.

**Layer 4 -- NAT Traversal.** STUN (RFC 5389) for external address. TURN (RFC
5766) relay if symmetric NAT. ICE candidates: host > server-reflexive > relay.
STUN timeout: 5000ms. TURN TTL: 600s. TCP fallback for firewalled environments.

### FR-071: Trust Handshake

Mutual identity verification. No anonymous participation. Noise_XX protocol.

**4-Phase Protocol:**
1. **DID Exchange.** Initiator sends `did:bizra:<base58(pubkey)>` + version.
   Responder checks format and CRL (revocation list on BlockGraph).
2. **Nonce Challenge.** 32-byte nonce + timestamp. Window: `DEFAULT_CHALLENGE_WINDOW_SECS`
   (30s). Domain: `ATTESTATION_DOMAIN` (`bizra-attestation-v1:`).
3. **Sign + Verify.** Initiator signs `domain || nonce || responder_id || timestamp`
   with Ed25519. Nonce stored in replay cache (`MAX_NONCE_CACHE`: 10,000).
4. **DTLS Session.** Noise_XX_25519_ChaChaPoly_BLAKE2b. PFS via ephemeral X25519.
   Lifetime: `SESSION_TIMEOUT_SECONDS` (3600s). Rekey: `SESSION_REKEY_INTERVAL` (300s).

Post-handshake: peer must demonstrate `IHSAN_CONFORMANCE_JOIN` (0.95) within
`TAKAFUL_BOOTSTRAP_OBSERVATION_MINUTES` (10 min) or graceful disconnect.

### FR-072: Communication Channels

Four channel types, all over authenticated DTLS sessions:

| Channel   | Pattern       | Use Case                                       |
|-----------|--------------|------------------------------------------------|
| Gossip    | Fanout (k=3) | Membership state, heartbeats, health            |
| Direct    | Point-to-point| Encrypted N2N data transfer, pattern queries   |
| Broadcast | Flood        | Governance proposals, emergency announcements   |
| Priority  | Sorted queue | Consensus votes, FATE proofs (latency-critical) |

**Gossip.** SWIM-based. Heartbeat: 1000ms. States: `ALIVE -> SUSPECT` (5s silence)
`-> DEAD` (15s). Incarnation numbers resolve conflicts. Messages: PING, PING_ACK,
PING_REQ, ANNOUNCE, LEAVE, PATTERN_SHARE, PROPOSE, VOTE, COMMIT.

**Priority Queue.** priority 0 = consensus/FATE, 64 = patterns, 128 = heartbeats.
Depth: 1024/peer. Overflow: drop lowest priority.

**Rate Limiting.** 10/peer/sec. Max age: 300s. Max future: 30s. Violations
escalate suspicion.

### FR-073: Pattern Sharing Protocol

Success patterns flow from local discovery to network adoption via PCI Envelopes.

1. **Detection.** SAPE engine: `ELEVATION_THRESHOLD` (3) reps, SNR >= 0.85,
   delta >= `MIN_SNR_DELTA_FOR_ELEVATION` (0.10).
2. **Abstraction.** Strip PII. Only `ABSTRACT_OK`/`SHAREABLE` propagate.
   Extract `(intent_class, context_hash) -> (action_template, outcome)`.
3. **Signing.** PCI Envelope + Ed25519. Domain: `DOMAIN_POI_ATTESTATION`.
   Minimum Ihsan: `UNIFIED_IHSAN_THRESHOLD` (0.95).
4. **Transmission.** Gossip PATTERN_SHARE. TTL: 168h. Cache: 1000 max.
5. **Validation.** Receiver verifies signature, PoI proof, SNR >= 0.85,
   Ihsan >= 0.95. PBFT consensus (2f+1).
6. **Integration.** Adopted into knowledge graph. `LOCAL -> PROPOSED -> VALIDATED`.

### FR-074: Federated Learning

Local model improvement shared without raw data exposure.

1. **Local Gradient.** TTRL gradient from receipts. SSO constrains norm. Min 10 receipts.
2. **Secure Aggregation.** Additive homomorphic masking: share `g_i + r_i`, masks cancel.
3. **Global Update.** Aggregated gradient applied. SNR must exceed 0.85 or discard.
4. **Distribution.** BLAKE3-hashed checkpoint via gossip. Rollback if loss increases >10%.

Privacy: raw data never leaves node. Differential privacy: epsilon=1.0/round.

### FR-075: Network Consensus

PoI-weighted BFT consensus finalizes blocks on the BlockGraph.

1. **Propose.** PoI block referencing `active_tips()`. Rate: 100/hour.
2. **Validate.** `3f+1` committee. Weight: `impact_score * sqrt(stake)`.
3. **BFT Vote.** PBFT: PRE_PREPARE -> PREPARE -> COMMIT -> COMMITTED.
   Quorum: `2f+1` via `sat_frontier_quorum()`. Timeout: 5000ms.
4. **Finalize.** Append to DAG. Merkle root of validator sigs. Post-finalization
   Gini check: reject if > `ADL_GINI_THRESHOLD` (0.35).

### FR-076: Reflex Diffusion

The network's collective System-1 cache. A verified pattern from Node A becomes
a reflex on Node B, reducing LLM calls network-wide.

**9-Stage Pipeline:**

1. **S2 Discovery.** Node A solves novel task via diffusion cognition (FR-021).
   Aha moment: SNR >= `GOT_CONVERGENCE_SNR` (0.90), sigma < 0.20.
2. **UIA Verification.** 8-dim Ihsan tensor (`IHSAN_CANONICAL_WEIGHTS`).
   Composite >= `UNIFIED_IHSAN_THRESHOLD` (0.95). PoI receipt to evidence ledger.
3. **RLVR Reward.** Mean atomic reward >= 0.85 (FR-022). TTRL gradient queued.
4. **Capsule Compilation.** G.R.A.S.P. (FR-023) extracts canonical pattern:
   ```
   ReflexCapsule { skill_hash, trigger_hash, action_template, ihsan_score,
     snr_composite, poi_block_cid, author_did, signature, privacy_class, ttl_hours }
   ```
5. **Broadcast.** Gossip PATTERN_SHARE with proofs. Priority 64. TTL 7 days.
6. **Random Peer Replay.** k=3 SAT validators replay in Pillar-3 sandbox
   (SNR floor 0.70). Verify output SNR >= 0.85 and no FATE violations.
7. **Consensus.** PBFT 2f+1 among replay validators. `PROPOSED -> VALIDATED`.
8. **Receiving Node Adoption.** Verify: Ed25519 sig, PoI chain to genesis,
   Ihsan >= `REFLEX_PRECIPITATION_IHSAN` (0.90). TeleScript sandbox trial.
9. **S1 Integration.** Insert as `ReflexMode::Shadow`. After
   `REFLEX_PRECIPITATION_HITS` (3) hits with Ihsan >= 0.90, precipitate to
   `Active`. Collective S1 hit-rate rises. GDP scales `Theta(N / log N)`.

**Residual Monitor.** 10-execution window. Evict if success < 90%. Invalidate if
Ihsan drift > 0.05 or age > 30 days.

---

## 2. Edge Cases

**EC-070: Sybil Attack.** Fake identities for consensus weight. Mitigation:
Genesis ceremony (256+ bit entropy, Ed25519, Shamir 3-of-5), Axiom 1.6 eliminates
7/8 attack classes, attestation challenges (30s window), `sqrt(stake)` dampening,
rate limit 100/hour/node. 3+ poisoned patterns trigger `NodeStatus::Expelled`.

**EC-071: Poisoned Pattern.** Harmful actions adopted by peers. Mitigation:
(1) FATE gate on every derived action. (2) Random peer replay in Pillar-3 sandbox.
(3) Residual monitor evicts at success < 90%. Author reputation reduced on rejection.

**EC-072: NAT Traversal Failure.** STUN/TURN both fail. Node enters "island mode"
(local-only, no federation). Retry with exponential backoff (5s to 300s max).
Status: `NodeState::SUSPECT` until connectivity restored.

**EC-073: Network Partition.** Disconnected subgraphs run independent consensus.
On reconnection: merge via BLAKE3 hash comparison, PoI-weight conflict resolution,
Gini check on merged state. Minority partition nodes undergo 10-min re-conformance.

**EC-074: Stale Capsule.** Deprecated APIs or outdated context. TTL: 168h.
Staleness: reject if age > 30 days. Residual monitor detects drift (Ihsan delta
> 0.05). Newer capsule with same trigger_hash supersedes after validation.

---

## 3. Pseudocode

### 3.1 discover_peers()

```
FUNCTION discover_peers(config: BootstrapConfig, identity: NodeIdentity) -> BootstrapResult:
    discovered = []

    # Layer 1: mDNS
    IF config.enable_mdns:
        FOR peer IN mdns_query("_bizra._tcp.local", timeout_ms=2000):
            IF peer.protocol_version == PROTOCOL_VERSION AND peer.node_id != identity.node_id:
                discovered.append(PeerInfo(peer.node_id, peer.address, peer.did, source="mdns"))

    # Layer 2: Seed nodes + peer exchange
    FOR seed_addr IN config.seed_nodes:
        TRY:
            resp = udp_send_recv(seed_addr, JoinRequest(identity), timeout_ms=config.discovery_timeout_secs * 1000)
            IF verify_ed25519(resp.signature, resp.public_key, resp.payload_bytes()):
                discovered.append(PeerInfo(resp.node_id, seed_addr, resp.did, source="seed"))
                FOR ep IN udp_send_recv(seed_addr, PeerExchangeRequest(max=config.max_peers)).peers:
                    IF ep.node_id NOT IN [p.node_id FOR p IN discovered]: discovered.append(ep)
        EXCEPT TimeoutError: CONTINUE

    # Layer 3: Kademlia DHT expansion
    IF len(discovered) > 0:
        dht = KademliaDHT(identity.node_id_160bit, k_bucket_size=20)
        FOR peer IN discovered: dht.add_contact(peer)
        FOR dp IN dht.iterative_find_node(identity.node_id_160bit, alpha=3):
            IF dp.node_id NOT IN [p.node_id FOR p IN discovered]: discovered.append(dp)

    # Layer 4: NAT traversal for unreachable peers
    FOR peer IN discovered WHERE NOT peer.directly_reachable:
        ext = stun_binding_request(STUN_SERVER, timeout_ms=5000)
        IF ext IS NOT None: peer.relay = None
        ELSE:
            alloc = turn_allocate(TURN_SERVER, ttl=600)
            IF alloc IS NOT None: peer.relay = alloc.relay_address
            ELSE: peer.reachable = False

    RETURN BootstrapResult(local_addr=config.bind_addr, discovered_peers=discovered,
        connected_count=len([p FOR p IN discovered IF p.reachable]),
        failed_seeds=[s FOR s IN config.seed_nodes IF s NOT IN [p.address FOR p IN discovered]])
```

### 3.2 trust_handshake(peer)

```
FUNCTION trust_handshake(peer: PeerInfo, identity: NodeIdentity, crl: RevocationList) -> SessionResult:
    IF NOT peer.did.startswith("did:bizra:"): RETURN SessionResult(REJECTED, "invalid_did_scheme")
    IF crl.is_revoked(peer.did): RETURN SessionResult(REJECTED, "did_revoked")
    IF NOT verify_did_pubkey_binding(peer.did, peer.public_key):
        RETURN SessionResult(REJECTED, "did_pubkey_mismatch")

    nonce = random_bytes(32); timestamp = now_unix_seconds()
    send(peer.address, ChallengeMessage(Challenge(nonce, timestamp, peer.node_id), identity.did))

    response = recv(peer.address, timeout_ms=DTLS_HANDSHAKE_TIMEOUT_MS)
    IF response IS None: RETURN SessionResult(REJECTED, "handshake_timeout")
    IF abs(now_unix_seconds() - response.timestamp) > DEFAULT_CHALLENGE_WINDOW_SECS:
        RETURN SessionResult(REJECTED, "challenge_expired")

    payload = ATTESTATION_DOMAIN + nonce + peer.node_id.bytes() + to_bytes(response.timestamp)
    IF NOT verify_ed25519(response.signature, peer.public_key, payload):
        RETURN SessionResult(REJECTED, "invalid_challenge_signature")
    IF nonce_cache.contains(nonce): RETURN SessionResult(REJECTED, "nonce_replayed")
    nonce_cache.insert(nonce, timestamp)

    transport = NoiseTransport(NOISE_PROTOCOL_NAME, identity.keypair, peer.public_key)
    session = transport.handshake()
    session.expires_at = now_unix_seconds() + SESSION_TIMEOUT_SECONDS
    RETURN SessionResult(ACCEPTED, session=session, peer_did=peer.did)
```

### 3.3 share_pattern(pattern)

```
FUNCTION share_pattern(pattern: ElevatedPattern, identity: NodeIdentity, federation: FederationNode) -> ShareResult:
    IF pattern.privacy_class == "LOCAL_ONLY": RETURN ShareResult(BLOCKED, "privacy_local_only")
    IF pattern.snr < UNIFIED_SNR_THRESHOLD: RETURN ShareResult(REJECTED, "snr_below_floor")
    IF pattern.ihsan < UNIFIED_IHSAN_THRESHOLD: RETURN ShareResult(REJECTED, "ihsan_below_threshold")
    IF pattern.repetitions < ELEVATION_THRESHOLD: RETURN ShareResult(REJECTED, "insufficient_reps")
    IF federation.gossip.sent_count(window=3600) >= ACTION_BUS_MAX_PER_HOUR:
        RETURN ShareResult(REJECTED, "rate_limit_exceeded")

    abstract = CanonicalPattern(pattern.intent_class, BLAKE3(pattern.context),
                                pattern.action_template, pattern.outcome_summary)
    envelope = PCIEnvelope(payload=serialize(abstract), domain=DOMAIN_POI_ATTESTATION,
        author_did=identity.did, ihsan_score=pattern.ihsan, snr_composite=pattern.snr)
    envelope.signature = identity.keypair.sign(envelope.canonical_bytes())

    message = GossipMessage(type=MessageType.PATTERN_SHARE, payload=serialize(envelope),
                            priority=64, ttl_hours=PATTERN_TTL_HOURS)
    peer_count = federation.gossip.broadcast(message)
    pattern.status = PatternStatus.PROPOSED
    RETURN ShareResult(SHARED, peer_count=peer_count, pattern_id=abstract.skill_hash)
```

### 3.4 diffuse_reflex(capsule)

```
FUNCTION diffuse_reflex(capsule: ReflexCapsule, local_node: FederationNode) -> DiffuseResult:
    author_pubkey = resolve_did_to_pubkey(capsule.author_did, local_node.blockgraph)
    IF author_pubkey IS None: RETURN DiffuseResult(REJECTED, "author_did_unresolvable")
    IF NOT verify_ed25519(capsule.signature, author_pubkey, capsule.canonical_bytes()):
        RETURN DiffuseResult(REJECTED, "invalid_capsule_signature")
    IF NOT verify_poi_chain(capsule.poi_block_cid, local_node.blockgraph.genesis_hash):
        RETURN DiffuseResult(REJECTED, "broken_poi_chain")

    IF capsule.ihsan_score < REFLEX_PRECIPITATION_IHSAN: RETURN DiffuseResult(REJECTED, "ihsan_low")
    IF capsule.snr_composite < UNIFIED_SNR_THRESHOLD: RETURN DiffuseResult(REJECTED, "snr_low")
    age_hours = (now_utc() - capsule.created_at).total_hours()
    IF age_hours > capsule.ttl_hours: RETURN DiffuseResult(REJECTED, "expired")
    IF capsule.privacy_class NOT IN ("ABSTRACT_OK", "SHAREABLE"):
        RETURN DiffuseResult(REJECTED, "privacy_forbids_adoption")

    sandbox = TeleScriptSandbox(pillar=3, snr_floor=PILLAR_3_SANDBOX_SNR_FLOOR)
    trial = sandbox.execute(capsule.action_template, synthetic_input(capsule.trigger_hash))
    IF trial.fate_violations > 0: RETURN DiffuseResult(REJECTED, "sandbox_fate_violation")
    IF trial.snr < UNIFIED_SNR_THRESHOLD: RETURN DiffuseResult(REJECTED, "sandbox_snr_low")
    IF NOT trial.outcome_matches(capsule.expected_outcome_class):
        RETURN DiffuseResult(REJECTED, "sandbox_outcome_mismatch")

    local_node.reflex_cache.insert(ReflexRule(
        trigger_hash=TriggerHash(capsule.trigger_hash), policy_hash=current_policy_hash(),
        action=capsule.action_template, mode=ReflexMode.Shadow,
        ihsan_score=capsule.ihsan_score, source_did=capsule.author_did, adopted_at=now_ms()))
    RETURN DiffuseResult(ADOPTED, mode=Shadow, skill_hash=capsule.skill_hash)
```

### 3.5 federated_aggregate(local_gradients)

```
FUNCTION federated_aggregate(local_gradients: [MaskedGradient], coordinator: FederationNode) -> AggregateResult:
    IF len(local_gradients) < 3: RETURN AggregateResult(REJECTED, "insufficient_participants")

    verified = [g FOR g IN local_gradients
                IF verify_ed25519(g.signature, g.author_pubkey, g.payload_bytes()) AND g.receipt_count >= 10]
    IF len(verified) < 3: RETURN AggregateResult(REJECTED, "insufficient_valid_gradients")

    # Secure aggregation: masks cancel -> SUM(g_i)
    aggregated = SUM(g.masked_values FOR g IN verified) - SUM(g.mask_share FOR g IN verified)
    mean_gradient = aggregated / len(verified)

    gradient_snr = compute_gradient_snr(verified, mean_gradient)
    IF gradient_snr < UNIFIED_SNR_THRESHOLD: RETURN AggregateResult(REJECTED, "gradient_snr_low")

    # SSO spectral norm clamp
    s_norm = compute_spectral_norm(mean_gradient)
    IF s_norm > SSO_SPECTRAL_BOUND: mean_gradient *= (SSO_SPECTRAL_BOUND / s_norm)

    checkpoint = ModelCheckpoint(gradient=mean_gradient, hash=BLAKE3(serialize(mean_gradient)),
        participant_count=len(verified), gradient_snr=gradient_snr)
    checkpoint.signature = coordinator.identity.keypair.sign(checkpoint.canonical_bytes())
    coordinator.gossip.broadcast(GossipMessage(type=MessageType.COMMIT,
        payload=serialize(checkpoint), priority=32))
    RETURN AggregateResult(SUCCESS, checkpoint_hash=checkpoint.hash, participants=len(verified))
```

---

## 4. TDD Anchors

```
TEST discover_peers_finds_mdns_local:
    config = BootstrapConfig(enable_mdns=True, seed_nodes=[], max_peers=10)
    mock_mdns([PeerInfo("node-A", "192.168.1.10:7654"), PeerInfo("node-B", "192.168.1.11:7654")])
    result = discover_peers(config, test_identity)
    ASSERT result.connected_count >= 2
    ASSERT ALL(p.source == "mdns" FOR p IN result.discovered_peers)

TEST discover_peers_falls_back_to_island_mode:
    config = BootstrapConfig(enable_mdns=False, seed_nodes=["unreachable:7654"])
    result = discover_peers(config, test_identity)
    ASSERT result.connected_count == 0 AND len(result.failed_seeds) == 1

TEST trust_handshake_rejects_revoked_did:
    crl = MockCRL(revoked=["did:bizra:EXPIRED_NODE"])
    result = trust_handshake(make_peer(did="did:bizra:EXPIRED_NODE"), test_identity, crl)
    ASSERT result.status == REJECTED AND "did_revoked" IN result.reason

TEST trust_handshake_rejects_expired_challenge:
    mock_response(timestamp=now_unix_seconds() - 60)  # > 30s window
    result = trust_handshake(make_peer(valid_sig=True), test_identity, empty_crl)
    ASSERT result.status == REJECTED AND "challenge_expired" IN result.reason

TEST share_pattern_blocks_local_only_privacy:
    pattern = make_pattern(privacy_class="LOCAL_ONLY", snr=0.92, ihsan=0.97)
    result = share_pattern(pattern, test_identity, federation)
    ASSERT result.status == BLOCKED AND "privacy_local_only" IN result.reason

TEST share_pattern_rejects_below_ihsan:
    pattern = make_pattern(privacy_class="SHAREABLE", snr=0.92, ihsan=0.89)
    result = share_pattern(pattern, test_identity, federation)
    ASSERT result.status == REJECTED AND "ihsan" IN result.reason

TEST diffuse_reflex_adopts_valid_capsule:
    capsule = make_capsule(ihsan=0.96, snr=0.91, valid_sig=True, valid_poi=True)
    mock_sandbox(fate_violations=0, snr=0.88, outcome_matches=True)
    result = diffuse_reflex(capsule, local_node)
    ASSERT result.status == ADOPTED AND result.mode == Shadow

TEST federated_aggregate_rejects_low_snr_gradient:
    grads = [make_gradient(valid=True, receipts=20) FOR _ IN range(5)]
    mock_gradient_snr(0.70)  # < UNIFIED_SNR_THRESHOLD (0.85)
    result = federated_aggregate(grads, coordinator)
    ASSERT result.status == REJECTED AND "gradient_snr" IN result.reason
```

---

## 5. Cross-References

### Python Modules
- `core/federation/gossip.py` -- `GossipEngine`, `NodeInfo`, `NodeState`, `MessageType`. SWIM with `MAX_FANOUT` (3), `GOSSIP_INTERVAL_MS` (1000).
- `core/federation/consensus.py` -- `ConsensusEngine`, `ConsensusPhase`, `Proposal`, `Vote`. PBFT with view-change.
- `core/federation/propagation.py` -- `PropagationEngine`, `PatternStore`, `ElevatedPattern`, `PatternStatus`. `ELEVATION_THRESHOLD` (3), `PATTERN_TTL_HOURS` (168).
- `core/federation/secure_transport.py` -- `NoiseTransport`, `DTLSTransport`, `SecureSession`, `ReplayWindow`. Noise_XX_25519_ChaChaPoly_BLAKE2b.
- `core/federation/node.py` -- `FederationNode`, `SyncFederationNode`. Composes gossip + propagation + consensus.
- `core/federation/interaction_boundary.py` -- `AttackClass` (8 classes). Axiom 1.6: eliminates 7/8 attack classes.
- `core/federation/pool_consensus.py` -- Amended Theorem 2.4: pool-mediated BFT.
- `core/federation/protocol.py` -- `FederationProtocol`, `FederatedPattern`, `PatternImpact`.
- `core/integration/constants.py` -- All thresholds: `UNIFIED_IHSAN_THRESHOLD` (0.95), `UNIFIED_SNR_THRESHOLD` (0.85), `IHSAN_CONFORMANCE_JOIN` (0.95), `REFLEX_*`, `GOT_CONVERGENCE_SNR` (0.90), `ACTION_BUS_MAX_PER_HOUR` (100), `ADL_GINI_THRESHOLD` (0.35), `PRIVACY_CLASSES`, `sat_frontier_quorum()`.
- `core/pci/envelope.py` -- `PCIEnvelope`. `core/pci/gates.py` -- `PCIGateKeeper`, 7-gate chain.

### Rust Crates
- `bizra-omega/bizra-federation/` -- `GossipProtocol`, `ConsensusEngine`, `FederationNode`, `Bootstrapper` (lib.rs). `Member`, `NodeState`, `GossipMessage`, `SignedGossipMessage` (gossip.rs). `BootstrapConfig`, `PeerInfo` (bootstrap.rs). `Challenge`, `Attestor`, `ATTESTATION_DOMAIN` (attestation.rs).
- `bizra-omega/bizra-agent/` -- `ReflexCache`, `ReflexMode`, `ReflexRule` (reflex_cache.rs). S2-to-S1 compiler (reflex_compiler.rs).
- `bizra-omega/bizra-ttrl/` -- `TtrlEngine` (GRPO), `SSO` (spectral norm), `EngramCache`, `MetabolicLedger`.
- `bizra-omega/bizra-hooks/` -- Event bus (8 shards, FNV-1a).
- `bizra-omega/bizra-core/` -- `IHSAN_THRESHOLD` (0.95), `SNR_THRESHOLD` (0.85), `NodeId`.

### Atlas v5 Phases
- Phase 00 -- FR-001: Federation layer; FR-002: L0 = libp2p + Noise + mDNS + DHT + NAT; FR-003: steps 10-12
- Phase 01 -- FR-010: Genesis Ed25519 + DID; FR-012: identity lifecycle, Shamir recovery, CRL
- Phase 02 -- FR-021: Diffusion Cognition Aha moments; FR-023: G.R.A.S.P.; FR-025: federated gossip + Takaful
- Phase 05 -- FR-050: BlockGraph; FR-051: PoI BFT 3f+1; FR-055: GDP = Theta(N / log N)
- Phase 06 -- FR-062: FATE Gate; FR-064: CROWN H0/H1/H2; FR-065: Governance pipeline

### Standing on Giants
- Das et al. (2002): SWIM -- scalable failure detection
- Maymounkov & Mazieres (2002): Kademlia -- XOR-distance DHT
- Perrin (2018): Noise Framework -- mutual auth with forward secrecy
- Castro & Liskov (1999): PBFT -- Byzantine fault tolerance
- Lamport (1982): Byzantine Generals -- foundational consensus
- Needham & Schroeder (1978): Nonce-based authentication
- Bonawitz et al. (2017): Secure Aggregation -- privacy-preserving FL
- McMahan et al. (2017): Federated Learning -- distributed training
- Ibn Khaldun (1377): Asabiyyah -- solidarity through shared impact
- Al-Ghazali (1095): Ihsan -- excellence as the floor
- Shannon (1948): SNR -- universal quality metric
- Metcalfe (1980): Network Effect -- value proportional to n-squared
