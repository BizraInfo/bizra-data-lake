# 01 — Node Sovereignty

> One user = one sovereign node.
> Local hardware, local data, local keys, local memory, local models.

## Pseudocode: Node Lifecycle

```
STRUCT SovereignNode:
    identity:       Ed25519KeyPair        // minted at genesis
    node_id:        BLAKE3(public_key)     // deterministic from key
    pat_agents:     [Agent; 7]             // user-owned council
    sat_agents:     [Agent; 5]             // system-owned validators (registered to URP)
    local_ledger:   ReceiptChain           // JSONL on disk, BLAKE3 chained
    local_memory:   HHMMMemory             // fast/slow/glacial layers
    local_models:   Vec<ModelRuntime>      // Ollama, LM Studio, GGUF
    substrate:      ResourceManifest       // CPU, RAM, GPU, disk
    genesis_seal:   GenesisSeal            // constitutional root of trust
    config:         ConstitutionalParams   // compiled-in, not user-editable

FUNCTION boot(node: SovereignNode):
    // Phase 1: Substrate discovery
    node.substrate = ResourceManifest::discover()

    // Phase 2: Identity — load or generate
    IF node.identity NOT exists on disk:
        node.identity = Ed25519KeyPair::generate()
        persist(node.identity)
    ELSE:
        node.identity = load_from_disk()

    // Phase 3: Genesis seal — deterministic root
    node.genesis_seal = GenesisSeal::compute(
        ConstitutionalParams::default(),
        current_timestamp_ms()
    )

    // Phase 4: Agent minting — 12 agents total
    node.pat_agents = mint_pat_7(node.identity)
    node.sat_agents = mint_sat_5(node.identity)

    // Phase 5: Memory restoration
    node.local_memory = HHMMMemory::restore_from_disk()

    // Phase 6: Model discovery
    node.local_models = discover_local_models(node.substrate)

    // Phase 7: Receipt chain — load existing
    node.local_ledger = ReceiptChain::load_or_create()

    RETURN node  // Node is now ALIVE, independent of network
```

## Offline-First Invariant

```
FUNCTION is_alive(node: SovereignNode) -> bool:
    // A node is alive if these LOCAL conditions hold.
    // Network connectivity is NOT required.
    RETURN node.identity IS valid
       AND node.genesis_seal IS computable
       AND node.pat_agents.len() == 7
       AND node.substrate.ram_gb >= 8.0
       // Models are optional — node degrades but lives
```

## Degradation Model

```
ENUM NodeState:
    SOVEREIGN   // all checks pass, models available, receipts valid
    DEGRADED    // some checks fail (e.g. no models, chain broken)
    MINIMAL     // identity + constitution only, no inference capability
    OFFLINE     // alive but URP unreachable (still fully functional locally)

FUNCTION compute_state(node: SovereignNode) -> NodeState:
    trust = evaluate_trust_surface(node)
    models = node.local_models.len()

    IF trust.all_pass AND models > 0:
        RETURN SOVEREIGN
    ELIF trust.all_pass:
        RETURN DEGRADED   // constitutional but no inference
    ELIF node.identity IS valid:
        RETURN MINIMAL
    ELSE:
        PANIC("Node cannot exist without identity")
```

## TDD Anchors

```
TEST node_boots_without_network:
    node = SovereignNode::boot(offline=true)
    ASSERT node.is_alive() == true
    ASSERT node.pat_agents.len() == 7
    ASSERT node.sat_agents.len() == 5

TEST node_identity_is_deterministic:
    key = Ed25519KeyPair::from_seed(FIXED_SEED)
    node_id_1 = BLAKE3(key.public)
    node_id_2 = BLAKE3(key.public)
    ASSERT node_id_1 == node_id_2

TEST genesis_seal_is_replayable:
    params = ConstitutionalParams::default()
    seal_1 = GenesisSeal::compute(params, timestamp=1000)
    seal_2 = GenesisSeal::compute(params, timestamp=1000)
    ASSERT seal_1.hash == seal_2.hash

TEST node_degrades_gracefully_without_models:
    node = SovereignNode::boot(models=[])
    ASSERT node.compute_state() == NodeState::DEGRADED
    ASSERT node.is_alive() == true  // still alive

TEST receipt_chain_survives_restart:
    node = SovereignNode::boot()
    receipt = node.execute_mission("test")
    node.shutdown()
    node2 = SovereignNode::boot()  // same disk
    ASSERT node2.local_ledger.len() == node.local_ledger.len()
```
