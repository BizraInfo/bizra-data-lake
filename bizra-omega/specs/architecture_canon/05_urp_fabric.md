# 05 — URP Fabric

> URP = Universal Resource Pool. The shared constitutional substrate.
> SAT-5 validators live here. Proofs settle here. Resources route here.
> URP serves the world, not any individual user.

## Pseudocode: URP Structure

```
STRUCT UniversalResourcePool:
    genesis_seal:       GenesisSeal           // root of trust (NODE0 minted)
    sat_registry:       HashMap<NodeId, [SatAgent; 5]>  // all registered SAT agents
    receipt_store:      ReceiptStore          // verified, attested receipts
    resource_manifest:  GlobalResourceManifest // contributed compute/storage
    federation_mesh:    FederationMesh        // node-to-URP connections
    seed_ledger:        SEEDLedger            // economic settlement layer
    marketplace:        Marketplace           // capability exchange
    constitution:       ConstitutionalParams  // compiled-in, immutable at runtime

FUNCTION urp_boot(genesis: GenesisSeal) -> UniversalResourcePool:
    // URP is created exactly once by NODE0 at genesis.
    // After genesis, URP accepts registrations but cannot be re-created.

    ASSERT genesis.is_valid()
    ASSERT genesis.minter == NODE0_IDENTITY

    urp = UniversalResourcePool {
        genesis_seal:      genesis,
        sat_registry:      HashMap::new(),
        receipt_store:     ReceiptStore::new(genesis.hash),
        resource_manifest: GlobalResourceManifest::empty(),
        federation_mesh:   FederationMesh::new(),
        seed_ledger:       SEEDLedger::new(genesis.hash),
        marketplace:       Marketplace::new(),
        constitution:      ConstitutionalParams::default(),  // compiled-in
    }

    RETURN urp
```

## Pseudocode: Node Registration

```
FUNCTION urp_register_node(
    urp: &mut UniversalResourcePool,
    node_id: NodeId,
    sat_agents: [SatAgent; 5],
    contributed_resources: ResourceContribution,
) -> RegistrationResult:
    // A new node joins the URP by registering its 5 SAT agents
    // and declaring its resource contribution.

    // Verify node identity
    IF NOT verify_node_identity(node_id):
        RETURN RegistrationResult::Reject("Invalid node identity")

    // Verify SAT agents are system-owned (not user-directed)
    FOR agent IN sat_agents:
        IF agent.owner != URP_SYSTEM_KEY:
            RETURN RegistrationResult::Reject("SAT agent not system-owned")
        IF agent.can_be_directed_by_user:
            RETURN RegistrationResult::Reject("SAT agent must not be user-directed")

    // Prevent duplicate registration
    IF urp.sat_registry.contains(node_id):
        RETURN RegistrationResult::Reject("Node already registered")

    // Register SAT agents into the pool
    urp.sat_registry.insert(node_id, sat_agents)

    // Add contributed resources
    urp.resource_manifest.add(node_id, contributed_resources)

    // Initialize SEED account for the node
    urp.seed_ledger.create_account(node_id, initial_balance=0.0)

    RETURN RegistrationResult::Accepted(node_id)
```

## Pseudocode: Receipt Settlement

```
FUNCTION urp_settle_receipt(
    urp: &mut UniversalResourcePool,
    request: ProofCarryingRequest,
    sat_verdict: SatVerdict,
) -> SettlementResult:
    // Only SAT-attested requests reach settlement.
    // Settlement = receipt stored + SEED minted + effects propagated.

    MATCH sat_verdict:
        SatVerdict::Reject(reason):
            RETURN SettlementResult::Rejected(reason)

        SatVerdict::Defer(reason):
            urp.queue_for_later(request)
            RETURN SettlementResult::Deferred(reason)

        SatVerdict::Attest(attested_request):
            // Step 1: Store the receipt
            receipt = Receipt {
                origin:        attested_request.origin_node,
                receipt_hash:  attested_request.receipt_hash,
                chain_link:    attested_request.chain_link,
                timestamp:     now(),
                sat_attestation: compute_attestation(attested_request),
            }
            urp.receipt_store.append(receipt)

            // Step 2: SEED settlement
            reward = compute_seed_reward(attested_request)
            zakat = reward * ZAKAT_RATE                    // 2.5%
            net_reward = reward - zakat

            urp.seed_ledger.credit(attested_request.origin_node, net_reward)
            urp.seed_ledger.credit(ZAKAT_POOL, zakat)

            // Step 3: Gini check (post-settlement)
            post_gini = urp.seed_ledger.compute_gini()
            IF post_gini > ADL_GINI_THRESHOLD:
                // Settlement still happens, but flag for redistribution
                urp.seed_ledger.flag_redistribution(attested_request.origin_node)

            // Step 4: Propagate to federation
            urp.federation_mesh.broadcast_receipt(receipt)

            RETURN SettlementResult::Settled(receipt, net_reward)

ENUM SettlementResult:
    Settled(Receipt, f64)     // receipt stored, SEED credited
    Rejected(String)          // SAT rejected
    Deferred(String)          // queued for later
```

## Pseudocode: Resource Routing

```
FUNCTION urp_route_resource(
    urp: &UniversalResourcePool,
    request: ResourceRequest,
) -> RouteResult:
    // S4 Conductor routes resource requests to available capacity.
    // Routing is fair — no node gets preferential access.

    // Find nodes with available capacity
    candidates = urp.resource_manifest.find_available(request.resource_type)

    IF candidates.is_empty():
        RETURN RouteResult::NoCapacity

    // Score candidates by proximity, capacity, and fairness
    scored = []
    FOR node IN candidates:
        score = ResourceScore {
            capacity:  node.available_capacity(request.resource_type),
            latency:   estimate_latency(request.origin, node.id),
            fairness:  urp.seed_ledger.usage_fairness(node.id),
        }
        scored.push((node, score))

    // Select best candidate (fairness-weighted)
    best = scored.sort_by(|a, b| b.score.weighted() - a.score.weighted()).first()

    RETURN RouteResult::Routed(best.node.id, best.score)

ENUM RouteResult:
    Routed(NodeId, ResourceScore)
    NoCapacity
```

## Pseudocode: Marketplace

```
FUNCTION marketplace_list_capability(
    urp: &mut UniversalResourcePool,
    node_id: NodeId,
    capability: Capability,
    price_seed: f64,
) -> ListingResult:
    // Nodes can offer capabilities (compute, storage, inference)
    // priced in SEED. Marketplace enforces fairness.

    // Verify the node actually has this capability
    IF NOT urp.resource_manifest.node_has(node_id, capability):
        RETURN ListingResult::Reject("Capability not verified")

    // Harberger tax: declared value determines tax rate
    tax_rate = HARBERGER_TAX_RATE   // 5% annual
    listing = MarketplaceListing {
        provider:   node_id,
        capability: capability,
        price_seed: price_seed,
        tax_rate:   tax_rate,
        listed_at:  now(),
    }

    urp.marketplace.add(listing)
    RETURN ListingResult::Listed(listing)

FUNCTION marketplace_consume(
    urp: &mut UniversalResourcePool,
    consumer: NodeId,
    listing_id: ListingId,
) -> ConsumeResult:
    listing = urp.marketplace.get(listing_id)
    IF listing IS None:
        RETURN ConsumeResult::NotFound

    // Check consumer balance
    balance = urp.seed_ledger.balance(consumer)
    IF balance < listing.price_seed:
        RETURN ConsumeResult::InsufficientFunds

    // Transfer SEED
    urp.seed_ledger.transfer(consumer, listing.provider, listing.price_seed)

    // Route the capability
    route = urp_route_resource(urp, ResourceRequest::from(listing))

    RETURN ConsumeResult::Consumed(route)
```

## Scaling Model

```
INVARIANT urp_scaling:
    // N nodes → 5N SAT agents in URP
    // Each node contributes SAT validators + resources
    // URP grows stronger with each node, but each node is independent

    total_sat_agents = urp.sat_registry.values().flatten().count()
    ASSERT total_sat_agents == urp.sat_registry.len() * 5

    // URP does NOT create liveness — nodes are alive alone
    // URP amplifies already-live organisms
    FOR node_id IN urp.sat_registry.keys():
        node = lookup_node(node_id)
        ASSERT node.is_alive()  // node was alive BEFORE joining URP
```

## TDD Anchors

```
TEST urp_boots_from_genesis:
    genesis = GenesisSeal::compute(ConstitutionalParams::default(), 1000)
    urp = urp_boot(genesis)
    ASSERT urp.genesis_seal == genesis
    ASSERT urp.sat_registry.is_empty()
    ASSERT urp.receipt_store.len() == 0

TEST urp_registers_node:
    urp = make_urp()
    node = make_sovereign_node()
    result = urp_register_node(urp, node.id, node.sat_agents, node.resources)
    ASSERT result IS Accepted
    ASSERT urp.sat_registry.len() == 1
    ASSERT urp.sat_registry[node.id].len() == 5

TEST urp_rejects_duplicate_registration:
    urp = make_urp()
    node = make_sovereign_node()
    urp_register_node(urp, node.id, node.sat_agents, node.resources)
    result = urp_register_node(urp, node.id, node.sat_agents, node.resources)
    ASSERT result IS Reject

TEST urp_settles_attested_receipt:
    urp = make_urp_with_node()
    request = make_valid_proof_request()
    verdict = SatVerdict::Attest(request)
    result = urp_settle_receipt(urp, request, verdict)
    ASSERT result IS Settled
    ASSERT urp.receipt_store.len() == 1

TEST urp_credits_seed_minus_zakat:
    urp = make_urp_with_node()
    request = make_valid_proof_request()
    verdict = SatVerdict::Attest(request)
    urp_settle_receipt(urp, request, verdict)
    balance = urp.seed_ledger.balance(request.origin_node)
    ASSERT balance == 1.0 - (1.0 * ZAKAT_RATE)   // 0.975

TEST urp_routes_to_available_node:
    urp = make_urp_with_nodes(3)
    request = ResourceRequest::compute(cores=4)
    result = urp_route_resource(urp, request)
    ASSERT result IS Routed

TEST urp_rejects_user_directed_sat:
    urp = make_urp()
    bad_agent = SatAgent { owner: USER_KEY, can_be_directed_by_user: true }
    result = urp_register_node(urp, node_id, [bad_agent; 5], resources)
    ASSERT result IS Reject

TEST scaling_invariant_holds:
    urp = make_urp()
    FOR i IN 1..=100:
        node = make_sovereign_node(seed=i)
        urp_register_node(urp, node.id, node.sat_agents, node.resources)
    ASSERT urp.sat_registry.len() == 100
    total_agents = urp.sat_registry.values().flatten().count()
    ASSERT total_agents == 500   // 100 nodes × 5 SAT each
```
