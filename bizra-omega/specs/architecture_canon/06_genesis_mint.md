# 06 — Genesis Mint

> Genesis = the one-time bootstrap that creates the constitutional root of trust.
> NODE0 mints the URP. Every subsequent node mints its own identity and 12 agents.
> Genesis is deterministic: same params + same timestamp = same seal.

## Pseudocode: Genesis Seal

```
STRUCT GenesisSeal:
    constitution_hash:  BLAKE3Hash        // hash of ConstitutionalParams
    timestamp_ms:       u64               // genesis moment
    seal_hash:          BLAKE3Hash        // BLAKE3(constitution_hash || timestamp)
    minter:             NodeId            // who computed this seal

FUNCTION GenesisSeal::compute(
    params: ConstitutionalParams,
    timestamp: u64,
) -> GenesisSeal:
    // Deterministic: same inputs always produce the same seal.
    // This is the root of the entire trust chain.

    constitution_hash = BLAKE3(serialize(params))
    seal_hash = BLAKE3(constitution_hash || timestamp.to_le_bytes())

    RETURN GenesisSeal {
        constitution_hash: constitution_hash,
        timestamp_ms:      timestamp,
        seal_hash:         seal_hash,
        minter:            NODE0_IDENTITY,   // only NODE0 can genesis
    }

FUNCTION GenesisSeal::verify(seal: GenesisSeal) -> bool:
    // Anyone can verify by recomputing
    expected = BLAKE3(seal.constitution_hash || seal.timestamp_ms.to_le_bytes())
    RETURN expected == seal.seal_hash
```

## Pseudocode: Constitutional Parameters

```
STRUCT ConstitutionalParams:
    // These are compiled-in. Not configurable at runtime.
    // Changing these requires a new genesis.

    ihsan_threshold:     f64    // 0.95 — excellence floor
    snr_threshold:       f64    // 0.85 — signal-to-noise minimum
    adl_gini_threshold:  f64    // 0.35 — fairness ceiling
    zakat_rate:          f64    // 0.025 — 2.5% redistribution
    harberger_tax_rate:  f64    // 0.05 — 5% annual on declared value
    pat_agent_count:     u8     // 7 — user-owned council
    sat_agent_count:     u8     // 5 — system-owned validators
    gate_order:          [Gate; 3]  // [Ihsan, Adl, Guardian] — fixed

FUNCTION ConstitutionalParams::default() -> ConstitutionalParams:
    RETURN ConstitutionalParams {
        ihsan_threshold:    0.95,
        snr_threshold:      0.85,
        adl_gini_threshold: 0.35,
        zakat_rate:         0.025,
        harberger_tax_rate: 0.05,
        pat_agent_count:    7,
        sat_agent_count:    5,
        gate_order:         [Gate::Ihsan, Gate::Adl, Gate::Guardian],
    }

INVARIANT constitutional_immutability:
    // ConstitutionalParams are set at genesis and NEVER change at runtime.
    // No user, no agent, no URP operation can modify these values.
    // They are the law. Everything else is derived.
    ASSERT params == ConstitutionalParams::default()  // always
```

## Pseudocode: NODE0 Genesis (One-Time)

```
FUNCTION genesis_node0() -> (SovereignNode, UniversalResourcePool):
    // This function runs EXACTLY ONCE in the history of the system.
    // NODE0 = the first node. It creates itself AND the URP.

    // Step 1: Mint NODE0 identity
    node0_key = Ed25519KeyPair::generate()
    node0_id = BLAKE3(node0_key.public)

    // Step 2: Compute genesis seal
    params = ConstitutionalParams::default()
    genesis = GenesisSeal::compute(params, now())
    genesis.minter = node0_id

    // Step 3: Mint NODE0's 12 agents
    pat_agents = mint_pat_7(node0_key)
    sat_agents = mint_sat_5(node0_key)

    // Step 4: Create the URP
    urp = urp_boot(genesis)

    // Step 5: Register NODE0's SAT agents into URP
    urp_register_node(urp, node0_id, sat_agents, node0_resources())

    // Step 6: Mint genesis block receipt
    genesis_receipt = CanonicalReceipt {
        receipt_hash:  BLAKE3("GENESIS:" || genesis.seal_hash),
        origin_node:   node0_id,
        timestamp:     now(),
        chain_link:    None,  // first receipt — no predecessor
        payload:       GenesisPayload {
            agents_minted: 12,
            urp_created:   true,
            seal:          genesis,
        },
    }
    genesis_receipt.sign(node0_key)
    urp.receipt_store.append(genesis_receipt)

    // Step 7: Build the sovereign node
    node0 = SovereignNode {
        identity:      node0_key,
        node_id:       node0_id,
        pat_agents:    pat_agents,
        sat_agents:    sat_agents,
        local_ledger:  ReceiptChain::from(genesis_receipt),
        local_memory:  HHMMMemory::new(),
        local_models:  discover_local_models(),
        substrate:     ResourceManifest::discover(),
        genesis_seal:  genesis,
        config:        params,
    }

    RETURN (node0, urp)
```

## Pseudocode: Subsequent Node Minting

```
FUNCTION mint_new_node(urp: &mut UniversalResourcePool) -> SovereignNode:
    // Every human after NODE0 goes through this process.
    // The node is alive BEFORE it registers with URP.

    // Step 1: Mint identity (local, no network needed)
    key = Ed25519KeyPair::generate()
    node_id = BLAKE3(key.public)

    // Step 2: Compute genesis seal (same params, different timestamp)
    params = ConstitutionalParams::default()
    genesis = GenesisSeal::compute(params, now())

    // Step 3: Mint 12 agents
    pat_agents = mint_pat_7(key)
    sat_agents = mint_sat_5(key)

    // Step 4: Build sovereign node (ALIVE at this point, no URP needed)
    node = SovereignNode {
        identity:      key,
        node_id:       node_id,
        pat_agents:    pat_agents,
        sat_agents:    sat_agents,
        local_ledger:  ReceiptChain::new(),
        local_memory:  HHMMMemory::new(),
        local_models:  discover_local_models(),
        substrate:     ResourceManifest::discover(),
        genesis_seal:  genesis,
        config:        params,
    }

    ASSERT node.is_alive()  // alive WITHOUT URP

    // Step 5: Register with URP (optional — amplifies but not required)
    IF urp IS reachable:
        urp_register_node(urp, node_id, sat_agents, node.substrate)

    RETURN node
```

## Pseudocode: Agent Minting

```
FUNCTION mint_pat_7(identity: Ed25519KeyPair) -> [PatAgent; 7]:
    // 7 PAT agents: user-owned, local-only, serve the human.
    RETURN [
        PatAgent::new(P1_Atlas,   identity, owner=identity.public, scope=LOCAL),
        PatAgent::new(P2_Oracle,  identity, owner=identity.public, scope=LOCAL),
        PatAgent::new(P3_Forge,   identity, owner=identity.public, scope=LOCAL),
        PatAgent::new(P4_Judge,   identity, owner=identity.public, scope=LOCAL),
        PatAgent::new(P5_Crown,   identity, owner=identity.public, scope=LOCAL),
        PatAgent::new(P6_Herald,  identity, owner=identity.public, scope=LOCAL),
        PatAgent::new(P7_Nexus,   identity, owner=identity.public, scope=LOCAL),
    ]

FUNCTION mint_sat_5(identity: Ed25519KeyPair) -> [SatAgent; 5]:
    // 5 SAT agents: system-owned, URP-registered, serve the constitution.
    // Note: minted FROM the user's key but OWNED BY the system.
    RETURN [
        SatAgent::new(S1_Sentinel,   identity, owner=URP_SYSTEM_KEY, scope=SYSTEM),
        SatAgent::new(S2_Oracle,     identity, owner=URP_SYSTEM_KEY, scope=SYSTEM),
        SatAgent::new(S3_Ledger,     identity, owner=URP_SYSTEM_KEY, scope=SYSTEM),
        SatAgent::new(S4_Conductor,  identity, owner=URP_SYSTEM_KEY, scope=SYSTEM),
        SatAgent::new(S5_Ambassador, identity, owner=URP_SYSTEM_KEY, scope=SYSTEM),
    ]

INVARIANT agent_count:
    // Every node always has exactly 12 agents: 7 PAT + 5 SAT.
    // No more, no less. No runtime creation or destruction.
    FOR node IN all_nodes:
        ASSERT node.pat_agents.len() == 7
        ASSERT node.sat_agents.len() == 5
```

## TDD Anchors

```
TEST genesis_seal_is_deterministic:
    params = ConstitutionalParams::default()
    seal_1 = GenesisSeal::compute(params, timestamp=1000)
    seal_2 = GenesisSeal::compute(params, timestamp=1000)
    ASSERT seal_1.seal_hash == seal_2.seal_hash

TEST genesis_seal_differs_with_timestamp:
    params = ConstitutionalParams::default()
    seal_1 = GenesisSeal::compute(params, timestamp=1000)
    seal_2 = GenesisSeal::compute(params, timestamp=2000)
    ASSERT seal_1.seal_hash != seal_2.seal_hash

TEST genesis_seal_is_verifiable:
    seal = GenesisSeal::compute(ConstitutionalParams::default(), 1000)
    ASSERT GenesisSeal::verify(seal) == true

TEST node0_creates_urp:
    (node0, urp) = genesis_node0()
    ASSERT node0.is_alive()
    ASSERT urp.genesis_seal.minter == node0.node_id
    ASSERT urp.receipt_store.len() == 1  // genesis receipt

TEST node0_mints_12_agents:
    (node0, _) = genesis_node0()
    ASSERT node0.pat_agents.len() == 7
    ASSERT node0.sat_agents.len() == 5

TEST subsequent_node_alive_before_urp:
    node = mint_new_node(urp=UNREACHABLE)
    ASSERT node.is_alive()
    ASSERT node.pat_agents.len() == 7
    ASSERT node.sat_agents.len() == 5

TEST pat_agents_are_user_owned:
    key = Ed25519KeyPair::generate()
    pat = mint_pat_7(key)
    FOR agent IN pat:
        ASSERT agent.owner == key.public
        ASSERT agent.scope == LOCAL

TEST sat_agents_are_system_owned:
    key = Ed25519KeyPair::generate()
    sat = mint_sat_5(key)
    FOR agent IN sat:
        ASSERT agent.owner == URP_SYSTEM_KEY
        ASSERT agent.scope == SYSTEM

TEST constitutional_params_are_immutable:
    params = ConstitutionalParams::default()
    ASSERT params.ihsan_threshold == 0.95
    ASSERT params.adl_gini_threshold == 0.35
    ASSERT params.zakat_rate == 0.025
    ASSERT params.pat_agent_count == 7
    ASSERT params.sat_agent_count == 5
    ASSERT params.gate_order == [Ihsan, Adl, Guardian]
```
