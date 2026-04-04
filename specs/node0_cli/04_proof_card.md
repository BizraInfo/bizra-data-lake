# Phase 4: Proof Card + Trust Surface
## Receipt display, verification, replay
### References: 00_cli_master_spec.md §6

---

## 1. Design Principle

**Trust by verification, not assertion.** Every serious action shows:
gate status, permission envelope, receipt id, evidence status, replay path.

The proof card is the CLI translation of the Trust Panel differentiator.

## 2. Proof Card Format

```
╔══════════════════════════════════════════════════════════════╗
║  MISSION COMPLETE                                            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Objective: What is the Ihsan principle and how does         ║
║             BIZRA enforce it constitutionally?               ║
║                                                              ║
║  ┌─ Verdict ──────────────────────────────────────────────┐  ║
║  │  ✓ PERMIT                                              │  ║
║  │  SNR:   0.8712  ████████████████░░░░  (≥ 0.85 ✓)      │  ║
║  │  Ihsan: 95.20%  ████████████████████░  (≥ 95% ✓)      │  ║
║  └────────────────────────────────────────────────────────┘  ║
║                                                              ║
║  ┌─ Agents ───────────────────────────────────────────────┐  ║
║  │  ♟ Strategist   147 tok  ✓                             │  ║
║  │  🔍 Researcher   283 tok  ✓                             │  ║
║  │  ⚙ Developer    201 tok  ✓                             │  ║
║  │  📊 Analyst      165 tok  ✓                             │  ║
║  └────────────────────────────────────────────────────────┘  ║
║                                                              ║
║  ┌─ Evidence ─────────────────────────────────────────────┐  ║
║  │  Receipt:  fa5f4e98f5b6cdaa7cf5a26fa769b9ca            │  ║
║  │  Chain:    seq=35, prev=16f6e994a410...                │  ║
║  │  Status:   INTACT (35 entries, 0 gaps)                 │  ║
║  └────────────────────────────────────────────────────────┘  ║
║                                                              ║
║  ┌─ Economy ──────────────────────────────────────────────┐  ║
║  │  SEED:  +0.12 minted, -0.003 zakat                    │  ║
║  │  IMPT:  +30.28 impact tokens                           │  ║
║  │  Total: 1,124,695 SEED | 4,208.3 IMPT                 │  ║
║  └────────────────────────────────────────────────────────┘  ║
║                                                              ║
║  ┌─ Memory ───────────────────────────────────────────────┐  ║
║  │  Persisted: 31 entries (+1)                            │  ║
║  │  Reflex:    Promoted to warm path ✓                    │  ║
║  └────────────────────────────────────────────────────────┘  ║
║                                                              ║
║  Replay: bizra receipt replay fa5f4e98                        ║
║  Verify: bizra trust verify fa5f4e98                         ║
║  Export: bizra receipt export fa5f4e98 --json                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

## 3. Pseudocode: Proof Card Renderer

```
FUNCTION render_proof_card(
    envelope: MissionEnvelope,
    synthesis: SynthesisResult,
    verdict: GateVerdict,
    receipt: CanonicalReceipt,
    token_summary: TokenSummary,
    memory_delta: MemoryDelta,
) -> String:

    card = BoxRenderer(title="MISSION COMPLETE", width=62)

    # Objective (wrapped)
    card.section("Objective", wrap_text(envelope.objective, 55))

    # Verdict section
    verdict_section = card.section("Verdict")
    verdict_badge = MATCH verdict.decision:
        "PERMIT"      → "✓ PERMIT" (green)
        "REVIEW"      → "⚠ REVIEW" (yellow)
        "QUARANTINED" → "✗ QUARANTINED" (red)

    verdict_section.add(verdict_badge)
    verdict_section.add(gauge("SNR",   verdict.snr_score,   0.85))
    verdict_section.add(gauge("Ihsan", verdict.ihsan_score,  0.95))

    # Agents section
    agents_section = card.section("Agents")
    FOR result IN synthesis.agent_results:
        agents_section.add(
            f"{result.icon} {result.name:<12} {result.tokens:>3} tok  "
            + ("✓" IF result.success ELSE "✗")
        )

    # Evidence section
    evidence_section = card.section("Evidence")
    evidence_section.add(f"Receipt:  {receipt.receipt_id}")
    evidence_section.add(f"Chain:    seq={receipt.sequence}, prev={receipt.prev_hash[:16]}...")
    chain_status = "INTACT" IF verify_chain() ELSE "BROKEN"
    evidence_section.add(f"Status:   {chain_status} ({receipt.sequence} entries, 0 gaps)")

    # Economy section
    econ_section = card.section("Economy")
    econ_section.add(f"SEED:  +{token_summary.seed_minted:.2f} minted, -{token_summary.zakat:.3f} zakat")
    econ_section.add(f"IMPT:  +{token_summary.impt_minted:.2f} impact tokens")
    econ_section.add(f"Total: {wallet.seed:,.0f} SEED | {wallet.impt:,.1f} IMPT")

    # Memory section
    mem_section = card.section("Memory")
    mem_section.add(f"Persisted: {memory_delta.total} entries (+{memory_delta.added})")
    IF memory_delta.reflex_promoted:
        mem_section.add("Reflex:    Promoted to warm path ✓")

    # Action hints
    card.footer([
        f"Replay: bizra receipt replay {receipt.receipt_id[:8]}",
        f"Verify: bizra trust verify {receipt.receipt_id[:8]}",
        f"Export: bizra receipt export {receipt.receipt_id[:8]} --json",
    ])

    RETURN card.render()
```

## 4. Pseudocode: `bizra trust` Command

```
COMMAND bizra_trust(subcommand: str = "status"):
    MATCH subcommand:
        "status" | "" → trust_status()
        "chain"       → trust_chain()
        "verify" id   → trust_verify(id)

FUNCTION trust_status():
    chain = load_evidence_chain()
    valid, issues = chain.verify()
    wallet = load_wallet()
    parity = check_cross_lang_parity()

    print_box("Constitutional Trust Status"):
        row("Evidence Chain",  f"{len(chain)} entries", badge(valid))
        row("Cross-Lang Parity", f"{parity.pass_count}/6", badge(parity.all_pass))
        row("SEED Balance",    f"{wallet.seed:,.0f}")
        row("Zakat Total",     f"{wallet.zakat:,.3f}")
        row("Last Receipt",   chain.latest.receipt_id[:16])
        row("Last Verdict",    chain.latest.decision)

        IF issues:
            section("Issues")
            FOR issue IN issues:
                row("⚠", issue)

FUNCTION trust_chain(tail: int = 10):
    chain = load_evidence_chain()
    entries = chain.entries[-tail:]

    print_box(f"Evidence Chain (last {tail})")
    FOR entry IN entries:
        row(
            seq=entry.sequence,
            hash=entry.entry_hash[:12],
            prev=entry.prev_hash[:12],
            decision=badge(entry.receipt.decision),
            time=relative_time(entry.timestamp),
        )
        # Visual chain link
        IF entry != entries[-1]:
            print("  │")

FUNCTION trust_verify(receipt_id: str):
    chain = load_evidence_chain()
    entry = chain.find(receipt_id)

    IF entry IS NONE:
        print_error(f"Receipt {receipt_id} not found")
        RETURN

    # Verify hash integrity
    recomputed = blake3(serialize(entry.receipt) + entry.prev_hash)
    hash_valid = recomputed == entry.entry_hash

    # Verify chain continuity
    IF entry.sequence > 1:
        prev_entry = chain.entries[entry.sequence - 2]
        chain_valid = entry.prev_hash == prev_entry.entry_hash
    ELSE:
        chain_valid = True  # Genesis entry

    print_box(f"Verification: {receipt_id}")
        row("Hash Integrity", badge(hash_valid))
        row("Chain Link",     badge(chain_valid))
        row("Decision",       entry.receipt.decision)
        row("SNR Score",      f"{entry.receipt.snr_score:.4f}")
        row("Ihsan Score",    f"{entry.receipt.ihsan_score:.2%}")
        row("Timestamp",      entry.timestamp)
        row("Replay",         f"bizra receipt replay {receipt_id[:8]}")
```

## 5. Pseudocode: `bizra receipt` Command

```
COMMAND bizra_receipt(subcommand: str = "list"):
    MATCH subcommand:
        "list"          → receipt_list()
        "show" id       → receipt_show(id)
        "replay" id     → receipt_replay(id)
        "export" id fmt → receipt_export(id, fmt)

FUNCTION receipt_list(limit: int = 10):
    chain = load_evidence_chain()
    FOR entry IN chain.entries[-limit:]:
        compact_row(
            entry.receipt.receipt_id[:12],
            badge(entry.receipt.decision),
            f"SNR={entry.receipt.snr_score:.2f}",
            relative_time(entry.timestamp),
        )

FUNCTION receipt_show(receipt_id: str):
    entry = find_receipt(receipt_id)
    # Full receipt details in structured format
    print_yaml(entry.receipt)

FUNCTION receipt_replay(receipt_id: str):
    entry = find_receipt(receipt_id)
    manifest = load_manifest(entry.receipt.mission_id)

    IF manifest IS NONE:
        print_error("No manifest found — replay requires proof bundle")
        RETURN

    print_info(f"Replaying mission: {manifest.mission_id}")
    print_command(manifest.replay.command)
    confirmed = prompt_user("Execute replay? [y/N]")
    IF confirmed == "y":
        exec_shell(manifest.replay.command)

FUNCTION receipt_export(receipt_id: str, format: str = "json"):
    entry = find_receipt(receipt_id)
    MATCH format:
        "json" → print(json_serialize(entry))
        "yaml" → print(yaml_serialize(entry))
        "md"   → print(markdown_render(entry))
```

## 6. TDD Anchors

```rust
#[test]
fn test_proof_card_shows_all_sections() {
    let card = render_proof_card(test_envelope(), test_verdict(), test_receipt());
    assert!(card.contains("MISSION COMPLETE"));
    assert!(card.contains("Verdict"));
    assert!(card.contains("Agents"));
    assert!(card.contains("Evidence"));
    assert!(card.contains("Economy"));
    assert!(card.contains("Memory"));
    assert!(card.contains("Replay"));
}

#[test]
fn test_proof_card_permit_green() {
    let verdict = GateVerdict::permit(0.87, 0.95);
    let card = render_proof_card(test_envelope(), verdict, test_receipt());
    assert!(card.contains("PERMIT"));
    assert!(card.contains("✓"));
}

#[test]
fn test_proof_card_review_yellow() {
    let verdict = GateVerdict::review(0.61, 0.61);
    let card = render_proof_card(test_envelope(), verdict, test_receipt());
    assert!(card.contains("REVIEW"));
    assert!(card.contains("⚠"));
}

#[test]
fn test_trust_verify_detects_tampered_hash() {
    let mut chain = test_chain();
    chain.entries[5].entry_hash = "deadbeef".repeat(8);
    let result = trust_verify_chain(&chain);
    assert!(!result.valid);
}

#[test]
fn test_receipt_replay_command_correct() {
    let manifest = test_manifest();
    let cmd = manifest.replay.command;
    assert!(cmd.contains("python scripts/node0_activate.py mission"));
}
```

## 7. Validation Gate

```
[ ] Proof card renders with all 6 sections
[ ] PERMIT/REVIEW/QUARANTINED shown with correct badge color
[ ] SNR/Ihsan gauges show fill + threshold marker
[ ] Agent contribution shows tokens + success/fail
[ ] Receipt hash is from kernel (not computed in CLI)
[ ] bizra trust shows chain integrity
[ ] bizra trust verify validates hash + chain link
[ ] bizra receipt replay executes correct command
[ ] bizra receipt export produces valid JSON
```

---

*The receipt is the product. The proof card is how the user sees it.*
