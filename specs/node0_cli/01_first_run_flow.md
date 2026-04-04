# Phase 1: First-Run Flow
## init → genesis → agents → first mission (< 15 min)
### References: 00_cli_master_spec.md §7

---

## 1. Design Principle

Onboarding is part of the product. The journey is:
**Genesis → Teach → Assembly → Mission**, not "install, configure, good luck."

The user should feel they are **minting a sovereign identity**, not setting up a tool.

## 2. Pseudocode: `bizra init`

```
COMMAND bizra_init():
    print_banner("Scanning your sovereign environment...")

    # Phase A: Hardware discovery (uses bizra-node substrate)
    hardware = discover_substrate()
    /*
        discover_substrate():
            cpu = detect_cpu()      # cores, model, arch
            ram = detect_ram()      # total, available
            gpu = detect_gpu()      # model, vram, cuda version
            disk = detect_disk()    # total, free, type
            RETURN SubstrateProfile(cpu, ram, gpu, disk)
    */
    print_substrate_card(hardware)

    # Phase B: LLM backend discovery
    backends = discover_backends()
    /*
        discover_backends():
            results = []
            TRY: probe_ollama("localhost:11434")     → results.append(OllamaBackend)
            TRY: probe_lmstudio(wsl_gateway:1234)    → results.append(LMStudioBackend)
            TRY: probe_gguf(scan_local_models())      → results.append(GGUFBackend)
            RETURN results  # empty = warn, non-empty = ready
    */
    print_backend_status(backends)

    # Phase C: Data root discovery
    data_roots = discover_data_roots()
    /*
        discover_data_roots():
            candidates = [
                env("BIZRA_DATA_LAKE_ROOT"),
                "~/bizra-data",
                "/mnt/c/BIZRA-DATA-LAKE",
                cwd(),
            ]
            FOR path IN candidates:
                IF has_gold_corpus(path):    # 04_GOLD/ exists
                    RETURN DataRoot(path, corpus_size, vector_count)
            RETURN None  # first-time user, will create
    */
    print_data_root_status(data_roots)

    # Phase D: Trust prerequisites
    trust = check_trust_prerequisites()
    /*
        check_trust_prerequisites():
            has_evidence = exists(EVIDENCE_JSONL)
            has_ledger = exists(TOKEN_LEDGER_JSONL)
            has_memory = exists(MEMORY_DB)
            RETURN TrustPrereqs(has_evidence, has_ledger, has_memory)
    */
    print_trust_status(trust)

    # Phase E: Summary + next step
    IF hardware.ready AND len(backends) > 0:
        print_success("Environment ready. Run: bizra genesis")
        save_init_state(hardware, backends, data_roots, trust)
    ELSE:
        print_issues(hardware.issues + backend_issues)
        print_hint("Fix issues above, then re-run: bizra init")

    RETURN InitResult(hardware, backends, data_roots, trust)
```

### Example output:

```
  Scanning your sovereign environment...

  ┌─ Hardware ─────────────────────────────────┐
  │ CPU:   Intel i9-14900HX (32 cores)         │
  │ RAM:   128 GB (89 GB available)            │
  │ GPU:   NVIDIA RTX 4090 (16 GB VRAM)       │
  │ Disk:  1 TB SSD (689 GB free)             │
  └────────────────────────────────────────────┘

  ┌─ Inference Backends ───────────────────────┐
  │ ● Ollama     localhost:11434  (7 models)   │
  │ ● LM Studio  172.22.48.1:1234 (18 models) │
  │ ○ GGUF       No local models found         │
  └────────────────────────────────────────────┘

  ┌─ Knowledge Base ───────────────────────────┐
  │ ● FAISS index: 102,714 vectors             │
  │ ● Corpus: 102,715 chunks across 4 parquets │
  │ ● Data root: /mnt/c/BIZRA-DATA-LAKE        │
  └────────────────────────────────────────────┘

  ┌─ Trust State ──────────────────────────────┐
  │ ● Evidence chain: 34 entries               │
  │ ● Token ledger: 108 transactions           │
  │ ● Living memory: 30 entries                │
  └────────────────────────────────────────────┘

  ✓ Environment ready. Run: bizra genesis
```

## 3. Pseudocode: `bizra genesis`

```
COMMAND bizra_genesis():
    init_state = load_init_state()
    IF init_state IS NONE:
        print_error("Run 'bizra init' first")
        RETURN

    print_banner("Minting your sovereign node...")

    # Phase A: Node identity
    node_key = ed25519_generate_keypair()
    node_id = blake3(node_key.public_key)[:16]
    genesis_seal = GenesisSeal(
        node_id=node_id,
        operator="operator-" + blake3(hostname())[:8],
        timestamp=utc_now(),
        public_key=node_key.public_key,
    )
    print_step("Node identity minted: node0-{node_id}")

    # Phase B: Agent parliament instantiation
    pat_agents = []
    FOR role IN PAT_7_ROLES:
        agent = mint_agent(role, type="PAT", signing_key=ed25519_generate())
        pat_agents.append(agent)
        print_step(f"  ♟ PAT: {role.name} ({role.giant})")

    sat_agents = []
    FOR role IN SAT_5_ROLES:
        agent = mint_agent(role, type="SAT", signing_key=ed25519_generate())
        sat_agents.append(agent)
        print_step(f"  🛡 SAT: {role.name} ({role.giant})")

    # Phase C: Wallet root
    wallet = WalletRoot(
        seed_balance=0.0,
        impt_balance=0.0,
        zakat_total=0.0,
    )
    print_step("Wallet initialized (SEED + IMPT + zakat)")

    # Phase D: First manifest (genesis block)
    manifest = ManifestArtifact(
        version="1.0.0",
        genesis_seal=genesis_seal,
        agents=pat_agents + sat_agents,
        wallet=wallet,
        evidence_seq=0,
    )
    receipt = sign_manifest(manifest, node_key)
    save_genesis(manifest, receipt)
    print_step(f"Genesis seal: {receipt.hash[:16]}...")

    # Phase E: Print genesis card
    print_genesis_card(genesis_seal, pat_agents, sat_agents, receipt)
    print_success("Your node is alive. Run: bizra agents")
```

### Example output:

```
  Minting your sovereign node...

  ✓ Node identity minted: node0-ce5af35c
  ✓ PAT: Strategist (Sun Tzu)
  ✓ PAT: Researcher (Shannon)
  ✓ PAT: Developer (Knuth)
  ✓ PAT: Analyst (Tukey)
  ✓ PAT: Reviewer (Fagan)
  ✓ PAT: Executor (Deming)
  ✓ PAT: Guardian (Al-Ghazali)
  ✓ SAT: FairVote (Rawls)
  ✓ SAT: HarmFilter (Hippocrates)
  ✓ SAT: Constitutional (Montesquieu)
  ✓ SAT: SecurityGate (Diffie)
  ✓ SAT: QualityAudit (Deming)
  ✓ Wallet initialized (SEED + IMPT + zakat)
  ✓ Genesis seal: a7f68f1f74f2c089...

  ╔═══════════════════════════════════════════╗
  ║  GENESIS COMPLETE                          ║
  ║  Node:    node0-ce5af35c                   ║
  ║  Agents:  7 PAT + 5 SAT = 12 active       ║
  ║  Wallet:  0.00 SEED / 0.00 IMPT           ║
  ║  Trust:   Genesis seal minted              ║
  ║                                             ║
  ║  Your sovereign node is alive.             ║
  ║  Run: bizra agents                         ║
  ╚═══════════════════════════════════════════╝
```

## 4. Pseudocode: `bizra agents`

```
COMMAND bizra_agents():
    state = load_node_state()

    # PAT-7: User's team
    print_section("PAT-7 — Your Personal Council")
    FOR agent IN state.pat_agents:
        status = agent.health_check()
        print_agent_row(
            icon=agent.icon,
            name=agent.name,
            role=agent.role_description,
            giant=agent.standing_giant,
            status=status,  # ● ACTIVE | ○ IDLE | ✗ ERROR
            last_receipt=agent.last_receipt_hash[:8] OR "—",
        )

    # SAT-5: System immune system
    print_section("SAT-5 — Constitutional Validators")
    FOR agent IN state.sat_agents:
        status = agent.health_check()
        print_agent_row(
            icon=agent.icon,
            name=agent.name,
            gate=agent.gate_description,
            status=status,
        )

    # Summary
    print_summary(
        total=12,
        active=count(a FOR a IN all_agents IF a.status == ACTIVE),
        last_mission=state.last_mission_id OR "none yet",
    )
    print_hint("Run: bizra mission \"your objective here\"")
```

## 5. First Mission (the killer moment)

```
COMMAND bizra_mission(objective: str):
    # Detailed in 02_mission_command.md
    # Summary: governed loop → proof card → memory persist

    # THE KILLER MOMENT:
    # "I gave BIZRA one real mission, it executed it locally,
    #  showed me which agents worked on it, proved what it did,
    #  remembered the result, and made the next run faster."
```

## 6. TDD Anchors

```rust
// bizra-cli/tests/test_first_run.rs

#[test]
fn test_init_discovers_hardware() {
    let result = bizra_init();
    assert!(result.hardware.cpu_cores > 0);
    assert!(result.hardware.ram_total_gb > 0);
}

#[test]
fn test_init_discovers_at_least_one_backend() {
    let result = bizra_init();
    assert!(!result.backends.is_empty(), "No LLM backend found");
}

#[test]
fn test_genesis_mints_12_agents() {
    let result = bizra_genesis();
    assert_eq!(result.pat_agents.len(), 7);
    assert_eq!(result.sat_agents.len(), 5);
}

#[test]
fn test_genesis_creates_ed25519_keys() {
    let result = bizra_genesis();
    assert!(result.genesis_seal.public_key.len() == 32);
    for agent in &result.all_agents {
        assert!(agent.signing_key.is_some());
    }
}

#[test]
fn test_genesis_emits_receipt() {
    let result = bizra_genesis();
    assert!(!result.genesis_receipt.hash.is_empty());
}

#[test]
fn test_first_run_under_15_minutes() {
    let start = Instant::now();
    bizra_init();
    bizra_genesis();
    bizra_agents();
    // Mission not included — user interaction required
    assert!(start.elapsed() < Duration::from_secs(120));
    // init + genesis + agents should complete in < 2 min
}
```

```python
# tests/integration/test_first_run.py

def test_init_finds_faiss_index():
    """Init discovers existing FAISS corpus."""
    result = run_bizra("init")
    assert "102,714 vectors" in result.stdout or "vectors" in result.stdout

def test_genesis_creates_state_dir():
    """Genesis creates ~/.bizra/node-1/ state directory."""
    run_bizra("genesis")
    assert Path("~/.bizra/node-1").expanduser().exists()
```

## 7. Validation Gate

```
[ ] bizra init completes in < 30s
[ ] bizra init discovers hardware, backends, corpus
[ ] bizra genesis mints 12 agents with Ed25519 keys
[ ] bizra genesis emits genesis seal receipt
[ ] bizra agents shows 7 PAT + 5 SAT with status
[ ] First mission completes and shows proof card
[ ] Total first-run time < 15 minutes
```

---

*The first 8 minutes are the product. Make them unforgettable.*
