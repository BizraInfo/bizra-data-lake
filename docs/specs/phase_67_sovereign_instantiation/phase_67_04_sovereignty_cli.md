# Phase 67.04 — Sovereignty CLI
# ══════════════════════════════

## Standing on Giants
- Thompson & Ritchie (1973): Unix CLI philosophy
- Nakamoto (2008): `bitcoin-cli` as reference for sovereign tooling
- Bernstein (2011): Ed25519 for identity
- Al-Ghazali (1095): Intent precedes all action

## Purpose

Four commands that transform a human from subject to sovereign:

```
bizra init    — Generate keypair, verify Declaration, become a node
bizra work    — Submit verified work, earn SEED
bizra attest  — Vouch for a neighbor's work, build Asabiyyah
bizra status  — View sovereign state: balance, governance, trajectory
```

## Source

Ω⁶ Synthesis CLI spec + existing `core/genesis/cli.py`

## Target

Extend `core/genesis/cli.py` and `core/sovereign/` to support all 4 commands.
The existing genesis CLI already handles `identity-genesis`, `hardware-scan`,
`pat-7`, `sat-5`. This spec adds the 4 sovereign commands as a separate
subcommand group.

```
core/sovereign/cli.py            # New: 4-command sovereign CLI
core/sovereign/sovereign_ops.py  # New: operation implementations
```

## Pseudocode

### Command: `bizra init`

```
COMMAND init
    ARGS:
        --name <str>       # Optional display name
        --declaration <path>  # Custom Declaration path (default: 00_CONSTITUTION/)

    FLOW:
        # Step 1: Verify Declaration
        declaration_text = load_declaration(args.declaration)
        IF NOT verify_declaration_hash(declaration_text):
            PRINT "ERROR: Declaration hash mismatch. Cannot establish sovereignty."
            EXIT 1
        PRINT "Declaration verified (Hash: 859649ea...)"

        # Step 2: Generate Ed25519 Keypair
        private_key, public_key = ed25519_generate()
        node_id = public_key

        # Step 3: Create genesis event
        genesis_event = create_genesis_event(declaration_text)

        # Step 4: Initialize WalletState
        wallet = WalletState(
            node_id=node_id,
            seed_balance=0,
            bloom_balance=0,
            created_at=now(),
            last_active=now()
        )

        # Step 5: Compute initial equity factor
        equity = ghazali_equity_factor(wallet, mean_balance=fp(100))  # Network mean

        # Step 6: Save to sovereign_state/
        save_keypair(private_key, "sovereign_state/identity.json")
        save_wallet(wallet, "sovereign_state/wallet.json")
        save_event_log([genesis_event], "sovereign_state/event_log.jsonl")

        # Step 7: Display
        PRINT "Node established."
        PRINT f"  ID:      {node_id.hex()[:16]}..."
        PRINT f"  Balance: 0 SEED"
        PRINT f"  Equity:  {fp_float(equity):.2f}x (newcomer advantage)"
        PRINT f"  Signed Declaration: {genesis_event.hash.hex()[:16]}..."
```

### Command: `bizra work`

```
COMMAND work
    ARGS:
        <description: str>  # Natural language description of work
        --type <str>        # "contribution" | "research" | "review" | "governance"
        --impact <float>    # Self-assessed impact (0.0 - 1.0)

    FLOW:
        # Step 1: Load identity and wallet
        identity = load_identity("sovereign_state/identity.json")
        wallet = load_wallet("sovereign_state/wallet.json")
        event_log = load_event_log("sovereign_state/event_log.jsonl")

        # Step 2: Construct ActionReceipt
        receipt = ActionReceipt(
            receipt_id=blake3(description.encode()),
            actor_id=identity.public_key,
            action_type=args.type OR "contribution",
            timestamp=now(),
            intent_score=fp(0.95),       # Self-reported, will be attested
            efficiency_score=fp(0.90),   # Default, refined by system
            impact_score=fp(args.impact OR 0.85),
            reproducibility_score=fp(0.90),
            oracle_signature=ed25519_sign(identity.private_key, receipt_id),
            metadata_hash=blake3(json.dumps({"desc": description}).encode()),
            co_actors=[]
        )

        # Step 3: Al-Ghazali Intent Gate
        IF NOT intent_gate(receipt):
            PRINT "REJECTED: Intent score below 0.90 floor."
            EXIT 1

        # Step 4: Ihsan Scoring
        passed, ihsan = full_ihsan_check(receipt)
        IF NOT passed:
            PRINT f"REJECTED: Ihsan {fp_float(ihsan):.4f} below 0.95 floor."
            EXIT 1

        # Step 5: Progressive Minting
        network_gini = estimate_network_gini()  # From local state or federation
        mean_balance = estimate_mean_balance()
        minted = progressive_mint(receipt, ihsan, wallet, network_gini, mean_balance)

        # Step 6: Apply Zakat at mint
        zakat = fp_mul(minted, ZAKAT_RATE)
        net_minted = fp_sub(minted, zakat)

        # Step 7: Update wallet
        wallet.seed_balance = fp_add(wallet.seed_balance, net_minted)
        wallet.bloom_balance = accrue_bloom(wallet, ihsan)
        wallet.total_actions += 1
        wallet.last_active = now()
        wallet.ihsan_history.append(ihsan)

        # Step 8: Append event
        append_event(event_log, "work", identity.public_key, {
            "receipt_hash": receipt.receipt_id.hex(),
            "ihsan": fp_float(ihsan),
            "minted": fp_float(net_minted),
            "zakat": fp_float(zakat),
            "description": description[:200]
        })

        # Step 9: Save and display
        save_wallet(wallet)
        save_event_log(event_log)

        PRINT f"Ihsan: {fp_float(ihsan):.4f} | Minted: {fp_float(net_minted):.4f} SEED"
        PRINT f"Zakat: {fp_float(zakat):.4f} SEED (purified)"
        PRINT f"Balance: {fp_float(wallet.seed_balance):.4f} SEED"
        PRINT f"Receipt: {receipt.receipt_id.hex()[:16]}..."
```

### Command: `bizra attest`

```
COMMAND attest
    ARGS:
        <node_id: str>     # Hex public key of node to attest
        <receipt_id: str>  # Hex receipt hash to attest

    FLOW:
        # Step 1: Load identity
        identity = load_identity("sovereign_state/identity.json")
        wallet = load_wallet("sovereign_state/wallet.json")

        # Step 2: Verify receipt exists (local or network lookup)
        receipt = lookup_receipt(args.receipt_id)
        IF receipt IS None:
            PRINT "ERROR: Receipt not found."
            EXIT 1

        # Step 3: Create attestation
        attestation = attest(wallet, target_wallet, receipt)

        # Step 4: Update Asabiyyah
        wallet.attestations_given.add(bytes.fromhex(args.node_id))
        wallet.cooperative_actions += 1

        old_asabiyyah = asabiyyah_score(wallet, network_size())

        # Step 5: Append event
        append_event(event_log, "attestation", identity.public_key, {
            "attestee": args.node_id,
            "receipt": args.receipt_id,
            "signature": attestation.signature.hex()
        })

        new_asabiyyah = asabiyyah_score(wallet, network_size())

        # Step 6: Display
        PRINT f"Attestation signed."
        PRINT f"Asabiyyah: {fp_float(old_asabiyyah):.4f} → {fp_float(new_asabiyyah):.4f}"
```

### Command: `bizra status`

```
COMMAND status
    ARGS:
        --verbose          # Show full event history
        --json             # Machine-readable output

    FLOW:
        identity = load_identity("sovereign_state/identity.json")
        wallet = load_wallet("sovereign_state/wallet.json")
        event_log = load_event_log("sovereign_state/event_log.jsonl")

        # Compute derived metrics
        trust = trust_score(wallet)
        asabiyyah = asabiyyah_score(wallet, network_size())
        backing = backing_ratio(total_seed(), total_verified_work())

        # Trajectory calculation (18-month path to median)
        IF wallet.total_actions > 0:
            avg_mint_per_action = fp_div(wallet.seed_balance, fp(wallet.total_actions))
            actions_to_median = fp_div(
                fp_sub(estimate_median_balance(), wallet.seed_balance),
                avg_mint_per_action
            ) IF avg_mint_per_action > 0 ELSE fp(0)
            months_to_median = fp_float(fp_div(actions_to_median, fp(30)))
        ELSE:
            months_to_median = 18.0  # Default projection

        IF args.json:
            PRINT json.dumps({...})
        ELSE:
            PRINT "╔═══════════════════════════════════════╗"
            PRINT "║      SOVEREIGN STATUS                 ║"
            PRINT "╠═══════════════════════════════════════╣"
            PRINT f"║  Node:       {wallet.node_id.hex()[:16]}...  ║"
            PRINT f"║  Balance:    {fp_float(wallet.seed_balance):>10.4f} SEED  ║"
            PRINT f"║  Governance: {fp_float(wallet.bloom_balance):>10.4f} BLOOM ║"
            PRINT f"║  Trust:      {fp_float(trust):>10.4f}       ║"
            PRINT f"║  Asabiyyah:  {fp_float(asabiyyah):>10.4f}       ║"
            PRINT f"║  Actions:    {wallet.total_actions:>10d}       ║"
            PRINT f"║  Trajectory: {months_to_median:>7.1f} months to median ║"
            PRINT "╚═══════════════════════════════════════╝"

            IF args.verbose:
                PRINT "\nEvent History:"
                FOR event IN event_log[-10:]:
                    PRINT f"  [{event.event_type}] {event.data}"
```

## Integration with Existing CLI

The existing `core/genesis/cli.py` has `build_genesis_parser()` that adds
a `genesis` subcommand. The sovereign CLI adds 4 new subcommands:

```python
# In core/sovereign/cli.py — extends the main CLI parser

def build_sovereign_parser(subparsers):
    """Add sovereign subcommands: init, work, attest, status."""

    init_parser = subparsers.add_parser("init", help="Establish sovereignty")
    init_parser.add_argument("--name", type=str)
    init_parser.set_defaults(func=cmd_init)

    work_parser = subparsers.add_parser("work", help="Submit verified work")
    work_parser.add_argument("description", type=str)
    work_parser.add_argument("--type", choices=["contribution", "research", "review"])
    work_parser.add_argument("--impact", type=float, default=0.85)
    work_parser.set_defaults(func=cmd_work)

    attest_parser = subparsers.add_parser("attest", help="Attest neighbor's work")
    attest_parser.add_argument("node_id", type=str)
    attest_parser.add_argument("receipt_id", type=str)
    attest_parser.set_defaults(func=cmd_attest)

    status_parser = subparsers.add_parser("status", help="View sovereign state")
    status_parser.add_argument("--verbose", action="store_true")
    status_parser.add_argument("--json", action="store_true")
    status_parser.set_defaults(func=cmd_status)
```

## TDD Anchors

```python
# tests/constitutional/test_sovereignty_cli.py

def test_init_creates_identity(tmp_path):
    """bizra init creates keypair and genesis event."""
    result = cmd_init(name="TestNode", state_dir=tmp_path)
    assert (tmp_path / "identity.json").exists()
    assert (tmp_path / "wallet.json").exists()
    assert (tmp_path / "event_log.jsonl").exists()

    wallet = json.loads((tmp_path / "wallet.json").read_text())
    assert wallet["seed_balance"] == 0
    assert wallet["bloom_balance"] == 0

def test_init_verifies_declaration(tmp_path):
    """bizra init fails if Declaration is tampered."""
    # Write tampered Declaration
    (tmp_path / "DECLARATION.md").write_text("TAMPERED")
    with pytest.raises(ConstitutionalViolation):
        cmd_init(state_dir=tmp_path, declaration=tmp_path / "DECLARATION.md")

def test_work_mints_seed(tmp_path):
    """bizra work mints SEED for quality work."""
    cmd_init(state_dir=tmp_path)
    result = cmd_work("Test contribution", state_dir=tmp_path)
    wallet = load_wallet(tmp_path / "wallet.json")
    assert wallet.seed_balance > 0
    assert wallet.total_actions == 1

def test_work_rejects_low_intent(tmp_path):
    """bizra work rejects actions below intent floor."""
    cmd_init(state_dir=tmp_path)
    # Force low intent score
    result = cmd_work("Test", state_dir=tmp_path, _override_intent=0.50)
    wallet = load_wallet(tmp_path / "wallet.json")
    assert wallet.seed_balance == 0  # Nothing minted

def test_attest_builds_asabiyyah(tmp_path):
    """bizra attest increases social cohesion score."""
    # Two nodes
    cmd_init(state_dir=tmp_path / "node1")
    cmd_init(state_dir=tmp_path / "node2")

    # Node1 does work
    cmd_work("Contribution", state_dir=tmp_path / "node1")
    receipt = get_latest_receipt(tmp_path / "node1")

    # Node2 attests Node1's work
    node1_id = load_identity(tmp_path / "node1").public_key.hex()
    cmd_attest(node1_id, receipt.hex(), state_dir=tmp_path / "node2")

    wallet2 = load_wallet(tmp_path / "node2" / "wallet.json")
    assert len(wallet2.attestations_given) == 1

def test_status_shows_trajectory(tmp_path):
    """bizra status computes months-to-median trajectory."""
    cmd_init(state_dir=tmp_path)
    for _ in range(5):
        cmd_work("Daily contribution", state_dir=tmp_path)

    result = cmd_status(state_dir=tmp_path, json_output=True)
    assert "months_to_median" in result
    assert result["months_to_median"] > 0
```
