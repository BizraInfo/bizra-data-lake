# 08 — Federation & A2A

> Federation = Node-to-URP, not Node-to-Node.
> Gossip propagates receipts and attestations through the URP mesh.
> A2A = Agent-to-Agent protocol for cross-node coordination.
> Federation amplifies already-live organisms. It does not create liveness.

## Pseudocode: Federation Topology

```
STRUCT FederationMesh:
    nodes:          HashMap<NodeId, NodeConnection>
    gossip_log:     Vec<GossipMessage>
    epoch:          u64                    // current federation epoch
    partition_map:  PartitionMap           // shard assignments

STRUCT NodeConnection:
    node_id:        NodeId
    endpoint:       Endpoint               // how to reach this node
    last_seen:      u64                    // last heartbeat timestamp
    sat_agents:     [SatAgentRef; 5]       // references to this node's SAT agents
    reputation:     f64                    // computed from receipt history
    is_alive:       bool                   // heartbeat-based liveness

INVARIANT federation_topology:
    // Federation is hub-and-spoke: Node → URP → Node.
    // Nodes do NOT communicate directly with each other.
    // All cross-node interaction goes through the URP.

    FOR node IN federation.nodes:
        ASSERT node.connects_to == URP
        ASSERT node.connects_to != OTHER_NODE  // no direct node-to-node
```

## Pseudocode: Node-to-URP Connection

```
FUNCTION connect_to_urp(
    node: &SovereignNode,
    urp_endpoint: Endpoint,
) -> ConnectionResult:
    // A sovereign node connects to the URP.
    // Connection is optional — node is alive without it.

    // Step 1: Prove identity
    challenge = request_challenge(urp_endpoint)
    response = sign_challenge(challenge, node.identity)

    IF NOT urp_verify_identity(response):
        RETURN ConnectionResult::Rejected("Identity verification failed")

    // Step 2: Register SAT agents (if first connection)
    IF NOT urp.has_node(node.node_id):
        urp_register_node(urp, node.node_id, node.sat_agents, node.substrate)

    // Step 3: Sync receipt chain
    local_head = node.local_ledger.head_hash()
    urp_head = urp.receipt_store.head_for(node.node_id)

    IF local_head != urp_head:
        // Reconcile: push local receipts that URP hasn't seen
        missing = node.local_ledger.since(urp_head)
        FOR receipt IN missing:
            urp_submit_for_validation(urp, receipt)

    // Step 4: Start heartbeat
    start_heartbeat(node.node_id, urp_endpoint, interval=30_seconds)

    RETURN ConnectionResult::Connected(node.node_id)

FUNCTION disconnect_from_urp(node: &SovereignNode):
    // Node can disconnect at any time.
    // It remains alive and functional locally.
    // Pending URP operations are queued for reconnection.

    stop_heartbeat(node.node_id)
    // Node continues operating in OFFLINE mode
    // Local missions still run, local receipts still chain
```

## Pseudocode: Gossip Protocol

```
STRUCT GossipMessage:
    origin:     NodeId
    payload:    GossipPayload
    hop_count:  u8               // max 5 hops
    signature:  Ed25519Signature
    timestamp:  u64

ENUM GossipPayload:
    ReceiptAnnouncement(ReceiptHash)      // "I have a new attested receipt"
    ReputationUpdate(NodeId, f64)          // "Node X reputation changed"
    CapabilityAdvertisement(Capability)    // "I offer this capability"
    FederationPolicy(PolicyUpdate)         // system-wide policy change

FUNCTION gossip_propagate(
    mesh: &mut FederationMesh,
    message: GossipMessage,
) -> PropagationResult:
    // Gossip propagates through the URP mesh.
    // SAT-5 Ambassador agents handle gossip on each node.

    // Verify message integrity
    IF NOT verify_ed25519(message.signature, message.origin):
        RETURN PropagationResult::Rejected("Invalid signature")

    // Check hop count (prevent infinite propagation)
    IF message.hop_count >= MAX_GOSSIP_HOPS:
        RETURN PropagationResult::Expired("Max hops reached")

    // Dedup — have we seen this message before?
    msg_hash = BLAKE3(serialize(message))
    IF mesh.gossip_log.contains(msg_hash):
        RETURN PropagationResult::Duplicate

    // Record and forward
    mesh.gossip_log.push(message)

    // Select peers to forward to (fanout)
    peers = select_gossip_peers(mesh, message.origin, fanout=3)
    forwarded = GossipMessage {
        ..message,
        hop_count: message.hop_count + 1,
    }

    FOR peer IN peers:
        send_to_peer(peer, forwarded)

    RETURN PropagationResult::Propagated(peers.len())

FUNCTION select_gossip_peers(
    mesh: &FederationMesh,
    exclude: NodeId,
    fanout: usize,
) -> Vec<NodeId>:
    // Select random alive peers, excluding the sender.
    // Bias toward high-reputation nodes for reliability.

    candidates = mesh.nodes.values()
        .filter(|n| n.node_id != exclude AND n.is_alive)
        .sorted_by(|a, b| b.reputation.partial_cmp(&a.reputation))

    // Take top-N with some randomization to prevent cliques
    RETURN candidates.take(fanout * 2).random_sample(fanout)
```

## Pseudocode: A2A (Agent-to-Agent) Protocol

```
STRUCT A2AMessage:
    from_agent:     AgentId          // e.g., NodeA::P2_Oracle
    to_agent:       AgentId          // e.g., NodeB::P2_Oracle
    payload:        A2APayload
    proof:          ProofTrace       // proof that this message is legitimate
    signature:      Ed25519Signature

ENUM A2APayload:
    KnowledgeQuery(String)              // "Do you know about X?"
    KnowledgeResponse(Vec<Evidence>)    // "Here's what I know about X"
    CapabilityRequest(Capability)       // "I need compute/storage/inference"
    CapabilityOffer(Capability, f64)    // "I can provide X for Y SEED"
    CollaborationProposal(Mission)      // "Let's work on this together"

FUNCTION a2a_send(
    from: &PatAgent,
    to_node: NodeId,
    to_agent_role: AgentRole,
    payload: A2APayload,
) -> A2AResult:
    // A2A messages go through the URP, not directly between nodes.
    // The message must be proof-carrying.

    // Step 1: Build proof that this agent is authorized
    proof = build_agent_proof(from)

    // Step 2: Wrap as A2A message
    message = A2AMessage {
        from_agent: from.id,
        to_agent:   AgentId::new(to_node, to_agent_role),
        payload:    payload,
        proof:      proof,
        signature:  sign(from.node_key, serialize(payload)),
    }

    // Step 3: Submit to URP for routing
    // URP validates proof, then routes to destination node's agent
    RETURN urp_route_a2a(message)

FUNCTION urp_route_a2a(message: A2AMessage) -> A2AResult:
    // URP validates and routes A2A messages.
    // This ensures all cross-node agent communication is:
    //   1. Authenticated (Ed25519)
    //   2. Proof-carrying (has ProofTrace)
    //   3. Policy-compliant (SAT-5 checks)

    // Validate sender
    IF NOT verify_agent_proof(message.proof):
        RETURN A2AResult::Rejected("Invalid agent proof")

    // SAT-5 policy check (S5 Ambassador)
    policy = S5_Ambassador.check_a2a_policy(message)
    IF NOT policy.ok:
        RETURN A2AResult::Rejected("A2A policy violation")

    // Route to destination
    dest_node = lookup_node(message.to_agent.node_id)
    IF dest_node IS None OR NOT dest_node.is_alive:
        RETURN A2AResult::Unreachable

    deliver_to_agent(dest_node, message.to_agent.role, message)
    RETURN A2AResult::Delivered
```

## Pseudocode: Offline Reconciliation

```
FUNCTION reconcile_after_offline(
    node: &SovereignNode,
    urp: &mut UniversalResourcePool,
) -> ReconciliationResult:
    // When a node reconnects after being offline,
    // it reconciles its local receipt chain with the URP.

    // Step 1: Find divergence point
    local_chain = node.local_ledger.all_receipts()
    urp_chain = urp.receipt_store.receipts_for(node.node_id)

    divergence = find_divergence(local_chain, urp_chain)

    // Step 2: Submit unattested local receipts for validation
    unattested = local_chain.since(divergence)
    results = []

    FOR receipt IN unattested:
        // Each receipt goes through full FATE → SAT pipeline
        request = wrap_as_proof_request(receipt, node.identity, receipt.chain_link)
        fate_result = fate_admit(request)

        IF fate_result IS Admit:
            sat_result = sat_validate(request)
            IF sat_result IS Attest:
                urp_settle_receipt(urp, request, sat_result)
                results.push(ReconcileStatus::Settled(receipt.hash))
            ELSE:
                results.push(ReconcileStatus::SatRejected(receipt.hash))
        ELSE:
            results.push(ReconcileStatus::FateRejected(receipt.hash))

    RETURN ReconciliationResult {
        total:    unattested.len(),
        settled:  results.filter(|r| r IS Settled).len(),
        rejected: results.filter(|r| r IS Rejected).len(),
    }
```

## Pseudocode: Scaling Model

```
FUNCTION compute_federation_capacity(urp: &UniversalResourcePool) -> FederationStats:
    // N nodes → 5N SAT agents → proportional validation capacity
    // More nodes = more validators = more trustworthy

    n_nodes = urp.sat_registry.len()
    n_sat = n_nodes * 5
    total_compute = urp.resource_manifest.total_compute()
    total_storage = urp.resource_manifest.total_storage()

    RETURN FederationStats {
        node_count:     n_nodes,
        sat_count:      n_sat,
        total_compute:  total_compute,
        total_storage:  total_storage,
        avg_reputation: urp.compute_avg_reputation(),
        gini:           compute_gini(urp.seed_ledger),
    }

INVARIANT federation_amplifies_not_creates:
    // Federation does NOT make dead nodes alive.
    // Federation amplifies nodes that are ALREADY alive.
    // A node with no identity, no genesis seal, or no PAT cannot join.

    FOR node_id IN urp.sat_registry.keys():
        node = lookup_node(node_id)
        ASSERT node.is_alive()               // was alive before federation
        ASSERT node.identity IS valid         // has identity
        ASSERT node.genesis_seal IS valid     // has constitutional root
        ASSERT node.pat_agents.len() == 7    // has full PAT council
```

## TDD Anchors

```
TEST node_connects_to_urp:
    node = make_sovereign_node()
    result = connect_to_urp(node, urp_endpoint)
    ASSERT result IS Connected
    ASSERT urp.has_node(node.node_id)

TEST node_disconnects_gracefully:
    node = make_connected_node(urp)
    disconnect_from_urp(node)
    ASSERT node.is_alive()  // still alive locally

TEST gossip_propagates_to_peers:
    mesh = make_mesh_with_nodes(5)
    message = make_gossip_message(origin=node_a)
    result = gossip_propagate(mesh, message)
    ASSERT result IS Propagated

TEST gossip_rejects_invalid_signature:
    mesh = make_mesh_with_nodes(5)
    message = make_gossip_message(signature=FORGED)
    result = gossip_propagate(mesh, message)
    ASSERT result IS Rejected

TEST gossip_expires_at_max_hops:
    mesh = make_mesh_with_nodes(5)
    message = make_gossip_message(hop_count=MAX_GOSSIP_HOPS)
    result = gossip_propagate(mesh, message)
    ASSERT result IS Expired

TEST gossip_deduplicates:
    mesh = make_mesh_with_nodes(5)
    message = make_gossip_message()
    gossip_propagate(mesh, message)
    result = gossip_propagate(mesh, message)  // same message again
    ASSERT result IS Duplicate

TEST a2a_routes_through_urp:
    node_a = make_connected_node(urp)
    node_b = make_connected_node(urp)
    result = a2a_send(
        node_a.P2_Oracle,
        node_b.node_id,
        P2_Oracle,
        A2APayload::KnowledgeQuery("test"),
    )
    ASSERT result IS Delivered

TEST a2a_rejects_invalid_proof:
    message = make_a2a_message(proof=INVALID)
    result = urp_route_a2a(message)
    ASSERT result IS Rejected

TEST offline_reconciliation_settles_valid:
    node = make_sovereign_node()
    // Node works offline, creating 5 local receipts
    FOR i IN 1..=5:
        node.execute_local_mission("mission_" + i)
    ASSERT node.local_ledger.len() == 5
    // Reconnect and reconcile
    result = reconcile_after_offline(node, urp)
    ASSERT result.total == 5
    ASSERT result.settled == 5

TEST scaling_is_linear:
    urp = make_urp()
    FOR i IN 1..=50:
        node = make_sovereign_node(seed=i)
        connect_to_urp(node, urp)
    stats = compute_federation_capacity(urp)
    ASSERT stats.node_count == 50
    ASSERT stats.sat_count == 250   // 50 × 5

TEST federation_requires_alive_nodes:
    dead_node = SovereignNode { identity: None }
    result = connect_to_urp(dead_node, urp)
    ASSERT result IS Rejected

TEST no_direct_node_to_node:
    node_a = make_connected_node(urp)
    node_b = make_connected_node(urp)
    // All communication goes through URP
    ASSERT node_a.direct_connections.is_empty()
    ASSERT node_b.direct_connections.is_empty()
```
