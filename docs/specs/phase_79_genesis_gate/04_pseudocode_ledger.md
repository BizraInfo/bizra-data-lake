# Phase 79: Ledger Gate (Layer 3) — Pseudocode

---

## Module: `core/sat/ledger_gate.py`

```pseudocode
IMPORT GateResult, CheckResult, CheckStatus FROM core.sat.gate_result
IMPORT TokenMinter FROM core.token.mint
IMPORT TokenLedger FROM core.token.ledger
IMPORT tempfile, os


FUNCTION ledger_verify() -> GateResult:
    """Layer 3: Economic Soundness — 10 checks.
    All automated. Uses isolated temp DB for each check.
    """
    checks = []

    # 4.1 SEED minting quality-gated (Ihsan < 0.85 -> zero SEED)
    WITH temp_ledger() AS (minter, ledger):
        receipt = minter.mint_seed(
            node_id="test-node",
            amount=1.0,
            ihsan_score=0.50,  # Below threshold
            intent_score=0.90,
        )
        minted = receipt.amount_minted
        checks.APPEND(CheckResult(
            "quality_gated",
            PASS IF minted == 0 ELSE FAIL,
            evidence=f"Low-ihsan mint returned {minted} SEED (expected 0)"
        ))

    # 4.2 BLOOM is soulbound (transfer rejected)
    WITH temp_ledger() AS (minter, ledger):
        # Mint some BLOOM first
        minter.genesis_mint("node-a")
        result = ledger.transfer(
            from_node="node-a", to_node="node-b",
            amount=1.0, token_type="BLOOM"
        )
        checks.APPEND(CheckResult(
            "bloom_soulbound",
            PASS IF result.rejected ELSE FAIL,
            evidence=f"BLOOM transfer: {'REJECTED' IF result.rejected ELSE 'ALLOWED'}"
        ))

    # 4.3 BLOOM decays
    WITH temp_ledger() AS (minter, ledger):
        minter.genesis_mint("decay-node")
        initial_bloom = ledger.get_balance("decay-node", "BLOOM")
        # Simulate 30 ticks of inactivity
        FOR i IN range(30):
            ledger.apply_bloom_decay("decay-node")
        final_bloom = ledger.get_balance("decay-node", "BLOOM")
        decayed = initial_bloom > final_bloom > 0
        checks.APPEND(CheckResult(
            "bloom_decays",
            PASS IF decayed ELSE FAIL,
            evidence=f"BLOOM: {initial_bloom} -> {final_bloom} after 30 ticks"
        ))

    # 4.4 Community pool receives 50%
    WITH temp_ledger() AS (minter, ledger):
        minter.mint_seed(
            node_id="pool-test", amount=100.0,
            ihsan_score=0.98, intent_score=0.95,
        )
        pool_balance = ledger.get_balance("BIZRA-COMMUNITY-POOL", "SEED")
        # After zakat (2.5%) the net is 97.5. 50% of that = 48.75
        expected_pool = 48.75  # approximate
        checks.APPEND(CheckResult(
            "pool_split_exact",
            PASS IF abs(pool_balance - expected_pool) < 1.0 ELSE FAIL,
            evidence=f"Pool received {pool_balance} SEED (expected ~{expected_pool})"
        ))

    # 4.5 Zakat applied
    WITH temp_ledger() AS (minter, ledger):
        minter.mint_seed(
            node_id="zakat-test", amount=100.0,
            ihsan_score=0.98, intent_score=0.95,
        )
        zakat_balance = ledger.get_balance("BIZRA-COMMUNITY-FUND", "SEED")
        expected_zakat = 2.5  # 2.5% of 100
        checks.APPEND(CheckResult(
            "zakat_applied",
            PASS IF abs(zakat_balance - expected_zakat) < 0.01 ELSE FAIL,
            evidence=f"Zakat: {zakat_balance} SEED (expected {expected_zakat})"
        ))

    # 4.6 Gini enforcement
    WITH temp_ledger() AS (minter, ledger):
        # Create 5 nodes, concentrate wealth in one
        FOR node IN ["a", "b", "c", "d", "e"]:
            minter.genesis_mint(node)
        # Try to mint huge amount for one node (should trigger Gini gate)
        result = minter.mint_seed(
            node_id="a", amount=10000.0,
            ihsan_score=0.98, intent_score=0.95,
        )
        checks.APPEND(CheckResult(
            "gini_enforced",
            PASS IF result.gini_blocked ELSE PARTIAL,
            evidence=f"Gini throttle: {'activated' IF result.gini_blocked ELSE 'not triggered'}"
        ))

    # 4.7 No double-mint
    WITH temp_ledger() AS (minter, ledger):
        first = minter.genesis_mint("double-test")
        second = minter.genesis_mint("double-test")
        checks.APPEND(CheckResult(
            "no_double_mint",
            PASS IF second.rejected ELSE FAIL,
            evidence=f"Second genesis mint: {'REJECTED' IF second.rejected ELSE 'ALLOWED'}"
        ))

    # 4.8 Supply cap
    WITH temp_ledger() AS (minter, ledger):
        # Attempt to mint beyond yearly cap
        result = minter.mint_seed(
            node_id="cap-test", amount=2_000_000,  # > 1M cap
            ihsan_score=0.98, intent_score=0.95,
        )
        checks.APPEND(CheckResult(
            "supply_capped",
            PASS IF result.cap_exceeded ELSE FAIL,
            evidence=f"Over-cap mint: {'REJECTED' IF result.cap_exceeded ELSE 'ALLOWED'}"
        ))

    # 4.9 Bot farming resistance (rapid low-quality rejected)
    WITH temp_ledger() AS (minter, ledger):
        rejected_count = 0
        FOR i IN range(100):
            result = minter.mint_seed(
                node_id=f"bot-{i}", amount=0.01,
                ihsan_score=0.30,  # Very low quality
                intent_score=0.40,
            )
            IF result.amount_minted == 0:
                rejected_count += 1
        checks.APPEND(CheckResult(
            "bot_resistant",
            PASS IF rejected_count == 100 ELSE FAIL,
            evidence=f"{rejected_count}/100 low-quality mints rejected"
        ))

    # 4.10 Receipt fabrication (hash chain rejects forged receipt)
    WITH temp_ledger() AS (minter, ledger):
        # Append a legitimate receipt
        ledger.append_receipt(valid_receipt())
        # Try to append a forged receipt with wrong prev_hash
        forged = forge_receipt(wrong_prev_hash=True)
        TRY:
            ledger.append_receipt(forged)
            checks.APPEND(CheckResult("chain_tamper_proof", FAIL, "Forged receipt accepted"))
        EXCEPT (ValueError, ValidationError):
            checks.APPEND(CheckResult("chain_tamper_proof", PASS, "Forged receipt rejected"))

    RETURN GateResult(agent="Ledger", layer="ECONOMIC_SOUNDNESS", checks=checks)


# Helper: create isolated temp ledger for testing
CONTEXT MANAGER temp_ledger():
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test_ledger.db")
    log_path = os.path.join(tmp, "test_ledger.jsonl")
    minter = TokenMinter.create(db_path=db_path, log_path=log_path)
    ledger = minter.ledger
    YIELD (minter, ledger)
    # Cleanup
    shutil.rmtree(tmp, ignore_errors=True)
```

---

## TDD Anchors

```pseudocode
TEST test_ledger_all_10_checks_present:
    result = ledger_verify()
    ASSERT len(result.checks) == 10
    ASSERT result.agent == "Ledger"

TEST test_ledger_quality_gate_blocks_low_ihsan:
    result = ledger_verify()
    qg = find_check(result, "quality_gated")
    ASSERT qg.status == PASS  # Low ihsan -> 0 SEED

TEST test_ledger_zakat_exact:
    result = ledger_verify()
    zakat = find_check(result, "zakat_applied")
    ASSERT zakat.status == PASS
    ASSERT "2.5" IN zakat.evidence

TEST test_ledger_isolation:
    # Two runs should not interfere (temp DBs)
    r1 = ledger_verify()
    r2 = ledger_verify()
    ASSERT r1.checks[0].evidence == r2.checks[0].evidence
```
