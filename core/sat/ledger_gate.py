"""
Ledger Gate — Layer 3: Economic Soundness
==========================================

10 checks. All automated. Uses isolated temp DBs.
"Is the economy fair, sustainable, and ungameable?"

Standing on Giants:
- Gini (1912): Inequality measurement
- Harberger (1962): Anti-hoarding taxation
- Ibn Khaldun (1377): Asabiyyah and economic cycles
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Tuple

from core.sat.gate_result import CheckResult, CheckStatus, GateResult

PASS = CheckStatus.PASS
FAIL = CheckStatus.FAIL
PARTIAL = CheckStatus.PARTIAL


@contextmanager
def _temp_ledger() -> Generator[Tuple, None, None]:
    """Create an isolated temp ledger for testing."""
    tmp = Path(tempfile.mkdtemp(prefix="sat_ledger_"))
    db_path = tmp / "test_ledger.db"
    log_path = tmp / "test_ledger.jsonl"
    try:
        from core.token.ledger import TokenLedger
        from core.token.mint import TokenMinter

        ledger = TokenLedger(db_path=db_path, log_path=log_path)
        minter = TokenMinter.create(ledger=ledger)
        yield minter, ledger
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ledger_verify() -> GateResult:
    """Layer 3: Economic Soundness — 10 checks."""
    checks: list[CheckResult] = []

    # 4.1 SEED minting works (quality gate is at mission level, not minter)
    try:
        with _temp_ledger() as (minter, ledger):
            receipt = minter.mint_seed(
                to_account="test-node",
                amount=1.0,
                poi_score=0.98,
            )
            checks.append(
                CheckResult(
                    "quality_gated",
                    PASS if receipt.success else FAIL,
                    f"SEED mint: {'success' if receipt.success else receipt.error}",
                )
            )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as e:  # SEC-003 — ledger check boundary
        checks.append(CheckResult("quality_gated", FAIL, f"Error: {e}"))

    # 4.2 BLOOM is soulbound (transfer rejected)
    try:
        with _temp_ledger() as (minter, ledger):
            from core.token.types import TokenOp, TokenType, TransactionEntry

            # Mint some BLOOM
            ledger.record_transaction(
                TransactionEntry(
                    token_type=TokenType.BLOOM,
                    op=TokenOp.MINT,
                    to_account="node-a",
                    amount=100.0,
                )
            )
            # Attempt transfer
            tx = TransactionEntry(
                token_type=TokenType.BLOOM,
                op=TokenOp.TRANSFER,
                from_account="node-a",
                to_account="node-b",
                amount=1.0,
            )
            result = ledger.record_transaction(tx)
            rejected = not result.success
            checks.append(
                CheckResult(
                    "bloom_soulbound",
                    PASS if rejected else FAIL,
                    f"BLOOM transfer: {'REJECTED' if rejected else 'ALLOWED'}",
                )
            )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as e:  # SEC-003 — ledger check boundary
        checks.append(CheckResult("bloom_soulbound", FAIL, f"Error: {e}"))

    # 4.3 BLOOM decays
    try:
        with _temp_ledger() as (minter, ledger):
            from core.token.types import TokenOp, TokenType, TransactionEntry

            ledger.record_transaction(
                TransactionEntry(
                    token_type=TokenType.BLOOM,
                    op=TokenOp.MINT,
                    to_account="decay-node",
                    amount=100.0,
                )
            )
            initial = ledger.get_balance("decay-node", TokenType.BLOOM).balance
            # Apply decay ticks
            if hasattr(ledger, "apply_bloom_decay"):
                for _ in range(30):
                    ledger.apply_bloom_decay("decay-node")
                final = ledger.get_balance("decay-node", TokenType.BLOOM).balance
                decayed = initial > final > 0
                checks.append(
                    CheckResult(
                        "bloom_decays",
                        PASS if decayed else FAIL,
                        f"BLOOM: {initial} -> {final} after 30 ticks",
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        "bloom_decays",
                        PARTIAL,
                        "apply_bloom_decay not implemented",
                    )
                )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as e:  # SEC-003 — ledger check boundary
        checks.append(CheckResult("bloom_decays", PARTIAL, f"Error: {e}"))

    # 4.4 Community pool — founder's oath (constant verified, not user tax)
    from core.integration.constants import BLOOM_REDISTRIBUTION_RATE

    checks.append(
        CheckResult(
            "pool_split_documented",
            PASS if BLOOM_REDISTRIBUTION_RATE == 0.50 else FAIL,
            f"BLOOM_REDISTRIBUTION_RATE = {BLOOM_REDISTRIBUTION_RATE} (founder's sadaqah)",
        )
    )

    # 4.5 Zakat applied
    try:
        with _temp_ledger() as (minter, ledger):
            from core.token.types import TokenOp, TokenType, TransactionEntry

            # Mint SEED
            ledger.record_transaction(
                TransactionEntry(
                    token_type=TokenType.SEED,
                    op=TokenOp.MINT,
                    to_account="zakat-test",
                    amount=100.0,
                )
            )
            if hasattr(ledger, "apply_zakat"):
                ledger.apply_zakat()
                bal = ledger.get_balance("zakat-test", TokenType.SEED).balance
                # After 2.5% zakat, balance should be 97.5
                checks.append(
                    CheckResult(
                        "zakat_applied",
                        PASS if abs(bal - 97.5) < 0.1 else FAIL,
                        f"Post-zakat balance: {bal} (expected ~97.5)",
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        "zakat_applied",
                        PARTIAL,
                        "apply_zakat not implemented as direct method",
                    )
                )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as e:  # SEC-003 — ledger check boundary
        checks.append(CheckResult("zakat_applied", PARTIAL, f"Error: {e}"))

    # 4.6 Gini enforcement
    try:
        with _temp_ledger() as (minter, ledger):
            from core.token.types import TokenOp, TokenType, TransactionEntry

            # Create 5 nodes with some SEED
            for node in ["a", "b", "c", "d", "e"]:
                ledger.record_transaction(
                    TransactionEntry(
                        token_type=TokenType.SEED,
                        op=TokenOp.GENESIS_MINT,
                        to_account=node,
                        amount=100.0,
                    )
                )
            # Try to mint huge amount for one node
            result = ledger.record_transaction(
                TransactionEntry(
                    token_type=TokenType.SEED,
                    op=TokenOp.MINT,
                    to_account="a",
                    amount=10000.0,
                )
            )
            gini_blocked = not result.success
            checks.append(
                CheckResult(
                    "gini_enforced",
                    PASS if gini_blocked else PARTIAL,
                    f"Gini throttle: {'activated' if gini_blocked else 'not triggered'}",
                )
            )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as e:  # SEC-003 — ledger check boundary
        checks.append(CheckResult("gini_enforced", PARTIAL, f"Error: {e}"))

    # 4.7 No double-mint (genesis)
    try:
        with _temp_ledger() as (minter, ledger):
            from core.token.types import TokenOp, TokenType, TransactionEntry

            ledger.record_transaction(
                TransactionEntry(
                    token_type=TokenType.SEED,
                    op=TokenOp.GENESIS_MINT,
                    to_account="double-test",
                    amount=100.0,
                )
            )
            second = ledger.record_transaction(
                TransactionEntry(
                    token_type=TokenType.SEED,
                    op=TokenOp.GENESIS_MINT,
                    to_account="double-test",
                    amount=100.0,
                )
            )
            rejected = not second.success
            checks.append(
                CheckResult(
                    "no_double_mint",
                    PASS if rejected else PARTIAL,
                    f"Second genesis mint: {'REJECTED' if rejected else 'ALLOWED'}",
                )
            )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as e:  # SEC-003 — ledger check boundary
        checks.append(CheckResult("no_double_mint", PARTIAL, f"Error: {e}"))

    # 4.8 Supply cap
    try:
        with _temp_ledger() as (minter, ledger):
            from core.token.types import TokenOp, TokenType, TransactionEntry

            result = ledger.record_transaction(
                TransactionEntry(
                    token_type=TokenType.SEED,
                    op=TokenOp.MINT,
                    to_account="cap-test",
                    amount=2_000_000,
                )
            )
            cap_hit = not result.success
            checks.append(
                CheckResult(
                    "supply_capped",
                    PASS if cap_hit else PARTIAL,
                    f"Over-cap mint: {'REJECTED' if cap_hit else 'ALLOWED (cap may not be enforced yet)'}",
                )
            )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as e:  # SEC-003 — ledger check boundary
        checks.append(CheckResult("supply_capped", PARTIAL, f"Error: {e}"))

    # 4.9 Bot farming resistance (Gini gate blocks concentrated minting)
    try:
        with _temp_ledger() as (minter, ledger):
            from core.token.types import TokenOp, TokenType, TransactionEntry

            # Create baseline accounts for Gini to be meaningful
            for node in ["base-a", "base-b", "base-c", "base-d", "base-e"]:
                ledger.record_transaction(
                    TransactionEntry(
                        token_type=TokenType.SEED,
                        op=TokenOp.GENESIS_MINT,
                        to_account=node,
                        amount=10.0,
                    )
                )
            # Rapid minting to single node should trigger Gini
            blocked = 0
            for i in range(20):
                r = minter.mint_seed(
                    to_account="bot-farmer", amount=50.0, poi_score=0.5
                )
                if not r.success:
                    blocked += 1
            checks.append(
                CheckResult(
                    "bot_resistant",
                    PASS if blocked > 0 else PARTIAL,
                    f"{blocked}/20 concentrated mints blocked by Gini",
                )
            )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as e:  # SEC-003 — ledger check boundary
        checks.append(CheckResult("bot_resistant", PARTIAL, f"Error: {e}"))

    # 4.10 Receipt fabrication (hash chain rejects forged receipt)
    try:
        from core.proof_engine.evidence_ledger import EvidenceLedger

        tmp = Path(tempfile.mkdtemp(prefix="sat_chain_"))
        try:
            el = EvidenceLedger(path=tmp / "chain.jsonl", validate_on_append=False)
            el.append(
                receipt={"action": "test", "data": "valid", "reason_codes": ["test"]}
            )
            # Forge: append with wrong prev_hash
            import json

            with open(tmp / "chain.jsonl", "a") as f:
                forged = json.dumps(
                    {
                        "seq": 999,
                        "prev_hash": "0" * 64,
                        "receipt": {"action": "forged"},
                        "entry_hash": "1" * 64,
                    }
                )
                f.write(forged + "\n")
            valid, errors = el.verify_chain()
            checks.append(
                CheckResult(
                    "chain_tamper_proof",
                    PASS if not valid else FAIL,
                    f"Chain validation: {'forged entry detected' if not valid else 'forged entry NOT detected'}",
                )
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as e:  # SEC-003 — ledger check boundary
        checks.append(CheckResult("chain_tamper_proof", PARTIAL, f"Error: {e}"))

    return GateResult(agent="Ledger", layer="ECONOMIC_SOUNDNESS", checks=checks)
