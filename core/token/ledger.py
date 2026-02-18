"""
Token Ledger — Hash-Chained, Append-Only Transaction Log + Balance State
=========================================================================

The ledger that makes SEED and BLOOM tokens real.

Architecture:
    Dual storage:
    1. SQLite database — balances + queryable transaction history
    2. JSONL append log — immutable hash-chained transaction trail

    Every transaction is hash-chained to its predecessor (Merkle 1979).
    Every transaction includes a nonce for replay protection (Nakamoto 2008).
    The JSONL log is the source of truth; SQLite is a materialized view.

Standing on Giants:
- Nakamoto (2008): Hash-chained transaction ledger
- Lamport (1978): Logical clocks, monotonic sequence numbers
- Merkle (1979): Hash chains for tamper detection
- Szabo (1997): Smart contracts as automated enforcement
- Shannon (1948): SNR for quality gating
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.integration.constants import (
    ADL_GINI_MIN_ACCOUNTS,
    ADL_GINI_THRESHOLD,
    ADL_HARBERGER_TAX_RATE,
)
from core.sovereign.adl_invariant import UBC_POOL_ID, calculate_gini
from core.token.types import (
    TokenBalance,
    TokenOp,
    TokenReceipt,
    TokenType,
    TransactionEntry,
)

logger = logging.getLogger(__name__)

# Sentinel hash for the first transaction in the ledger
GENESIS_TX_HASH = "0" * 64

# System pool accounts — excluded from Gini calculation because they are
# communal redistribution pools, not individual wealth holdings.
# Gini measures inequality among INDIVIDUAL nodes, not system accounts.
SYSTEM_POOL_IDS: frozenset[str] = frozenset({
    "__UBC_POOL__",              # Universal Basic Compute pool
    "BIZRA-COMMUNITY-FUND",     # Computational zakat recipient
    "SYSTEM-TREASURY",          # System treasury
})

# Default paths — resolved relative to project root (no hardcoded absolutes)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = _PROJECT_ROOT / ".swarm" / "memory.db"
DEFAULT_LOG_PATH = _PROJECT_ROOT / "04_GOLD" / "token_ledger.jsonl"


class TokenLedger:
    """
    BIZRA Token Ledger — The source of truth for all token balances.

    Thread-safe. Hash-chained. Append-only transaction log.

    Usage:
        ledger = TokenLedger()
        balance = ledger.get_balance("node-0", TokenType.SEED)
        ledger.close()
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        log_path: Optional[Path] = None,
    ):
        self._db_path = db_path or DEFAULT_DB_PATH
        self._log_path = log_path or DEFAULT_LOG_PATH
        self._lock = threading.Lock()
        self._sequence = 0
        self._last_hash = GENESIS_TX_HASH

        # Ensure directories exist
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema and resume state
        self._ensure_schema()
        self._resume_chain_state()

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.cursor()

            # Token balances table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_balances (
                    account_id TEXT NOT NULL,
                    token_type TEXT NOT NULL,
                    balance REAL NOT NULL DEFAULT 0.0,
                    staked REAL NOT NULL DEFAULT 0.0,
                    last_updated TEXT NOT NULL,
                    PRIMARY KEY (account_id, token_type)
                )
            """)

            # Transaction log table (queryable mirror of JSONL)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_transactions (
                    tx_id TEXT PRIMARY KEY,
                    sequence INTEGER NOT NULL UNIQUE,
                    op TEXT NOT NULL,
                    token_type TEXT NOT NULL,
                    from_account TEXT NOT NULL DEFAULT '',
                    to_account TEXT NOT NULL DEFAULT '',
                    amount REAL NOT NULL,
                    memo TEXT DEFAULT '',
                    epoch_id TEXT DEFAULT '',
                    poi_score REAL DEFAULT 0.0,
                    prev_hash TEXT NOT NULL,
                    tx_hash TEXT NOT NULL,
                    signature TEXT DEFAULT '',
                    signer_pubkey TEXT DEFAULT '',
                    nonce TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)

            # Yearly supply tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_supply (
                    year INTEGER NOT NULL,
                    token_type TEXT NOT NULL,
                    total_minted REAL NOT NULL DEFAULT 0.0,
                    total_burned REAL NOT NULL DEFAULT 0.0,
                    PRIMARY KEY (year, token_type)
                )
            """)

            conn.commit()

    def _resume_chain_state(self) -> None:
        """Resume sequence and chain state from existing log."""
        if self._log_path.exists() and self._log_path.stat().st_size > 0:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self._sequence = data["seq"]
                    self._last_hash = data["tx_hash"]
            logger.info(
                "Token ledger resumed: sequence=%d, last_hash=%s...",
                self._sequence,
                self._last_hash[:16],
            )

    @property
    def sequence(self) -> int:
        """Current sequence number."""
        return self._sequence

    @property
    def last_hash(self) -> str:
        """Hash of the most recent transaction."""
        return self._last_hash

    # =========================================================================
    # BALANCE QUERIES
    # =========================================================================

    def get_balance(self, account_id: str, token_type: TokenType) -> TokenBalance:
        """Get current balance for an account + token type."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT balance, staked, last_updated
                FROM token_balances
                WHERE account_id = ? AND token_type = ?
                """,
                (account_id, token_type.value),
            )
            row = cursor.fetchone()

            if row:
                return TokenBalance(
                    account_id=account_id,
                    token_type=token_type,
                    balance=row[0],
                    staked=row[1],
                    last_updated=row[2],
                )
            return TokenBalance(
                account_id=account_id,
                token_type=token_type,
                balance=0.0,
                staked=0.0,
            )

    def get_all_balances(self, account_id: str) -> dict[TokenType, TokenBalance]:
        """Get all token balances for an account."""
        result: dict[TokenType, TokenBalance] = {}
        for tt in TokenType:
            bal = self.get_balance(account_id, tt)
            if bal.balance > 0 or bal.staked > 0:
                result[tt] = bal
        return result

    def list_accounts(self) -> list[str]:
        """Get all distinct account IDs with non-zero balances."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT account_id FROM token_balances WHERE balance > 0 OR staked > 0"
            )
            return [row[0] for row in cursor.fetchall()]

    def get_total_supply(self, token_type: TokenType) -> float:
        """Get total circulating supply of a token type."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COALESCE(SUM(balance), 0) FROM token_balances WHERE token_type = ?",
                (token_type.value,),
            )
            row = cursor.fetchone()
            return row[0] if row else 0.0

    def get_yearly_minted(
        self, token_type: TokenType, year: Optional[int] = None
    ) -> float:
        """Get total minted for a token type in a given year."""
        if year is None:
            year = datetime.now(timezone.utc).year
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT total_minted FROM token_supply WHERE year = ? AND token_type = ?",
                (year, token_type.value),
            )
            row = cursor.fetchone()
            return row[0] if row else 0.0

    # =========================================================================
    # TRANSACTION RECORDING
    # =========================================================================

    def record_transaction(self, tx: TransactionEntry) -> TokenReceipt:
        """
        Record a transaction in the ledger.

        This is the core operation. It:
        1. Validates the transaction
        2. Updates balances in SQLite
        3. Appends hash-chained entry to JSONL log
        4. Updates yearly supply tracking
        5. Returns a signed receipt

        Thread-safe via lock.
        """
        with self._lock:
            return self._record_locked(tx)

    def _record_locked(self, tx: TransactionEntry) -> TokenReceipt:
        """Record transaction while holding the lock."""
        # Assign sequence and chain link
        self._sequence += 1
        tx.sequence = self._sequence
        tx.prev_hash = self._last_hash

        # Compute hash
        tx.tx_hash = tx.compute_hash()

        # Validate
        error = self._validate_transaction(tx)
        if error:
            self._sequence -= 1  # Rollback sequence
            return TokenReceipt(success=False, error=error)

        # Apply to SQLite
        try:
            balance_after = self._apply_to_db(tx)
        except Exception as e:
            self._sequence -= 1
            logger.error("Failed to apply transaction %s: %s", tx.tx_id, e)
            return TokenReceipt(success=False, error=str(e))

        # Append to JSONL log (the immutable record)
        self._append_to_log(tx)

        # Update chain state
        self._last_hash = tx.tx_hash

        logger.info(
            "TX #%d %s %s %.2f %s -> %s (hash=%s...)",
            tx.sequence,
            tx.op.value,
            tx.token_type.value,
            tx.amount,
            tx.from_account or "MINT",
            tx.to_account or "BURN",
            tx.tx_hash[:16],
        )

        return TokenReceipt(
            success=True,
            tx_entry=tx,
            balance_after=balance_after,
            receipt_hash=tx.tx_hash,
        )

    def _validate_transaction(self, tx: TransactionEntry) -> Optional[str]:
        """Validate a transaction before recording."""
        if tx.amount <= 0:
            return f"Amount must be positive, got {tx.amount}"

        if tx.op == TokenOp.TRANSFER:
            if not tx.from_account:
                return "Transfer requires from_account"
            if not tx.to_account:
                return "Transfer requires to_account"
            if tx.from_account == tx.to_account:
                return "Cannot transfer to self"
            # Check sufficient balance
            bal = self.get_balance(tx.from_account, tx.token_type)
            if bal.available < tx.amount:
                return (
                    f"Insufficient balance: {bal.available:.4f} < {tx.amount:.4f} "
                    f"(account={tx.from_account}, token={tx.token_type.value})"
                )

        if tx.op == TokenOp.BURN:
            if not tx.from_account:
                return "Burn requires from_account"
            bal = self.get_balance(tx.from_account, tx.token_type)
            if bal.available < tx.amount:
                return f"Insufficient balance for burn: {bal.available:.4f} < {tx.amount:.4f}"

        if tx.op in (TokenOp.MINT, TokenOp.GENESIS_MINT):
            if not tx.to_account:
                return "Mint requires to_account"

        if tx.op == TokenOp.STAKE:
            if not tx.from_account:
                return "Stake requires from_account"
            bal = self.get_balance(tx.from_account, tx.token_type)
            if bal.available < tx.amount:
                return (
                    f"Insufficient available balance for staking: {bal.available:.4f}"
                )

        if tx.op == TokenOp.UNSTAKE:
            if not tx.from_account:
                return "Unstake requires from_account"
            bal = self.get_balance(tx.from_account, tx.token_type)
            if bal.staked < tx.amount:
                return f"Insufficient staked balance: {bal.staked:.4f}"

        # IMPT is non-transferable
        if tx.token_type == TokenType.IMPT and tx.op == TokenOp.TRANSFER:
            return "IMPT tokens are non-transferable (soulbound)"

        # ADL Gini gate — reject transactions that push inequality beyond threshold
        # Only applies to SEED (economic token). GENESIS_MINT and ZAKAT are exempt.
        if (
            tx.token_type == TokenType.SEED
            and tx.op in (TokenOp.TRANSFER, TokenOp.MINT)
        ):
            gini_error = self._check_gini_impact(tx)
            if gini_error:
                return gini_error

        return None

    def _check_gini_impact(self, tx: TransactionEntry) -> Optional[str]:
        """Check if a transaction would push the Gini coefficient above the ADL threshold.

        Simulates the post-transaction balance distribution and rejects if
        the resulting Gini coefficient exceeds ADL_GINI_THRESHOLD (0.35).

        System pool accounts (UBC pool, community fund, treasury) are excluded
        from the Gini calculation because they are communal redistribution pools,
        not individual wealth holdings.

        Standing on Giants - Gini (1912) + Rawls (1971):
        Justice is a hard gate, not a soft metric.
        """
        # Transfers TO system pools are always allowed (redistributive)
        target = tx.to_account if tx.op in (TokenOp.MINT, TokenOp.TRANSFER) else ""
        if target in SYSTEM_POOL_IDS:
            return None

        # Get current SEED balances for all accounts
        holdings_raw = self._get_seed_holdings()
        holdings = dict(holdings_raw)  # Copy for simulation

        # Simulate the transaction's effect on balances
        if tx.op == TokenOp.TRANSFER:
            holdings[tx.from_account] = holdings.get(tx.from_account, 0.0) - tx.amount
            holdings[tx.to_account] = holdings.get(tx.to_account, 0.0) + tx.amount
        elif tx.op == TokenOp.MINT:
            holdings[tx.to_account] = holdings.get(tx.to_account, 0.0) + tx.amount

        # Exclude system pools and zero/negative balances from Gini calculation
        projected = {
            k: v for k, v in holdings.items()
            if v > 0 and k not in SYSTEM_POOL_IDS
        }

        # Gini coefficient is meaningless with few data points.
        # During genesis bootstrap the system must distribute to initial
        # participants before equality enforcement can apply.
        if len(projected) < ADL_GINI_MIN_ACCOUNTS:
            return None

        # Compute pre-transaction Gini for comparison
        pre_holdings = {
            k: v for k, v in holdings_raw.items()
            if v > 0 and k not in SYSTEM_POOL_IDS
        }
        pre_gini = calculate_gini(pre_holdings) if len(pre_holdings) >= 2 else 0.0

        post_gini = calculate_gini(projected)

        # Allow transfers that REDUCE Gini (directionally improving justice)
        # even if the absolute level is still above threshold
        if post_gini <= pre_gini:
            return None

        if post_gini > ADL_GINI_THRESHOLD:
            return (
                f"ADL Gini gate: transaction would push Gini to {post_gini:.4f} "
                f"(threshold={ADL_GINI_THRESHOLD}). "
                f"Plutocratic concentration rejected."
            )

        return None

    def _get_seed_holdings(self) -> dict[str, float]:
        """Query all non-zero SEED balances from SQLite."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT account_id, balance FROM token_balances "
                "WHERE token_type = ? AND balance > 0",
                (TokenType.SEED.value,),
            )
            return {row[0]: row[1] for row in cursor.fetchall()}

    # =========================================================================
    # HARBERGER TAX — Continuous redistribution toward equality
    # =========================================================================

    def apply_harberger_tax(
        self,
        tax_rate: Optional[float] = None,
        epoch_id: str = "",
    ) -> dict:
        """Apply Harberger tax to all SEED holders, flowing proceeds to UBC pool.

        The Harberger mechanism ensures that resources held but not productively
        used are gradually redistributed to the Universal Basic Compute pool,
        which distributes equally to all active nodes.

        Standing on Giants - Harberger (1962):
        Self-assessed value with continuous taxation prevents hoarding.

        Args:
            tax_rate: Override rate (default: ADL_HARBERGER_TAX_RATE from constants.py)
            epoch_id: Epoch identifier for the tax sweep

        Returns:
            Summary dict with total_taxed, accounts_affected, ubc_pool_credit
        """
        rate = tax_rate if tax_rate is not None else ADL_HARBERGER_TAX_RATE
        holdings = self._get_seed_holdings()

        # Exclude UBC pool itself from taxation
        holdings.pop(UBC_POOL_ID, None)

        total_taxed = 0.0
        accounts_affected = 0

        for account_id, balance in holdings.items():
            tax_amount = balance * rate
            if tax_amount <= 0:
                continue

            # Create a TRANSFER from holder to UBC pool
            tx = TransactionEntry(
                op=TokenOp.TRANSFER,
                token_type=TokenType.SEED,
                from_account=account_id,
                to_account=UBC_POOL_ID,
                amount=tax_amount,
                memo=f"harberger_tax_epoch_{epoch_id}" if epoch_id else "harberger_tax",
                epoch_id=epoch_id,
            )

            # Record without Gini check (tax is redistributive by nature)
            # We bypass _validate_transaction's Gini gate by recording directly
            receipt = self._record_tax_transfer(tx)
            if receipt.success:
                total_taxed += tax_amount
                accounts_affected += 1
            else:
                logger.warning(
                    "Harberger tax failed for %s: %s", account_id, receipt.error
                )

        logger.info(
            "Harberger tax sweep: %.2f SEED collected from %d accounts → %s",
            total_taxed,
            accounts_affected,
            UBC_POOL_ID,
        )

        return {
            "total_taxed": total_taxed,
            "accounts_affected": accounts_affected,
            "ubc_pool_credit": total_taxed,
            "tax_rate": rate,
            "epoch_id": epoch_id,
        }

    def _record_tax_transfer(self, tx: TransactionEntry) -> TokenReceipt:
        """Record a Harberger tax transfer, bypassing the Gini gate.

        Tax transfers are inherently redistributive (reducing Gini), so
        they must not be blocked by the Gini gate that prevents concentration.
        All other validations (balance sufficiency, etc.) still apply.
        """
        with self._lock:
            # Assign sequence and chain link
            self._sequence += 1
            tx.sequence = self._sequence
            tx.prev_hash = self._last_hash
            tx.tx_hash = tx.compute_hash()

            # Validate balance only (skip Gini gate)
            if tx.amount <= 0:
                self._sequence -= 1
                return TokenReceipt(success=False, error="Amount must be positive")
            bal = self.get_balance(tx.from_account, tx.token_type)
            if bal.available < tx.amount:
                self._sequence -= 1
                return TokenReceipt(
                    success=False,
                    error=f"Insufficient balance for tax: {bal.available:.4f}",
                )

            try:
                balance_after = self._apply_to_db(tx)
            except Exception as e:
                self._sequence -= 1
                return TokenReceipt(success=False, error=str(e))

            self._append_to_log(tx)
            self._last_hash = tx.tx_hash

            return TokenReceipt(
                success=True,
                tx_entry=tx,
                balance_after=balance_after,
                receipt_hash=tx.tx_hash,
            )

    def _apply_to_db(self, tx: TransactionEntry) -> float:
        """Apply transaction to SQLite balances. Returns new balance of primary account."""
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        year = datetime.now(timezone.utc).year
        tt = tx.token_type.value

        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.cursor()

            if tx.op in (TokenOp.MINT, TokenOp.GENESIS_MINT, TokenOp.ZAKAT):
                self._credit(cursor, tx.to_account, tt, tx.amount, now)
                self._track_minted(cursor, year, tt, tx.amount)

            elif tx.op == TokenOp.TRANSFER:
                self._debit(cursor, tx.from_account, tt, tx.amount, now)
                self._credit(cursor, tx.to_account, tt, tx.amount, now)

            elif tx.op == TokenOp.BURN:
                self._debit(cursor, tx.from_account, tt, tx.amount, now)
                self._track_burned(cursor, year, tt, tx.amount)

            elif tx.op == TokenOp.STAKE:
                self._adjust_stake(cursor, tx.from_account, tt, tx.amount, now)

            elif tx.op == TokenOp.UNSTAKE:
                self._adjust_stake(cursor, tx.from_account, tt, -tx.amount, now)

            # Record transaction in SQL table
            cursor.execute(
                """
                INSERT INTO token_transactions
                (tx_id, sequence, op, token_type, from_account, to_account,
                 amount, memo, epoch_id, poi_score, prev_hash, tx_hash,
                 signature, signer_pubkey, nonce, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tx.tx_id,
                    tx.sequence,
                    tx.op.value,
                    tt,
                    tx.from_account,
                    tx.to_account,
                    tx.amount,
                    tx.memo,
                    tx.epoch_id,
                    tx.poi_score,
                    tx.prev_hash,
                    tx.tx_hash,
                    tx.signature,
                    tx.signer_pubkey,
                    tx.nonce,
                    tx.timestamp,
                ),
            )
            conn.commit()

            # Return balance of the primary account
            primary = (
                tx.to_account
                if tx.op in (TokenOp.MINT, TokenOp.GENESIS_MINT, TokenOp.ZAKAT)
                else tx.from_account
            )
            if primary:
                cursor.execute(
                    "SELECT balance FROM token_balances WHERE account_id = ? AND token_type = ?",
                    (primary, tt),
                )
                row = cursor.fetchone()
                return row[0] if row else 0.0
            return 0.0

    @staticmethod
    def _credit(
        cursor: sqlite3.Cursor, account: str, tt: str, amount: float, now: str
    ) -> None:
        """Credit tokens to an account (upsert)."""
        cursor.execute(
            """INSERT INTO token_balances (account_id, token_type, balance, staked, last_updated)
            VALUES (?, ?, ?, 0.0, ?)
            ON CONFLICT(account_id, token_type)
            DO UPDATE SET balance = balance + ?, last_updated = ?""",
            (account, tt, amount, now, amount, now),
        )

    @staticmethod
    def _debit(
        cursor: sqlite3.Cursor, account: str, tt: str, amount: float, now: str
    ) -> None:
        """Debit tokens from an account."""
        cursor.execute(
            """UPDATE token_balances SET balance = balance - ?, last_updated = ?
            WHERE account_id = ? AND token_type = ?""",
            (amount, now, account, tt),
        )

    @staticmethod
    def _adjust_stake(
        cursor: sqlite3.Cursor, account: str, tt: str, delta: float, now: str
    ) -> None:
        """Adjust staked amount (positive = stake, negative = unstake)."""
        cursor.execute(
            """UPDATE token_balances SET staked = staked + ?, last_updated = ?
            WHERE account_id = ? AND token_type = ?""",
            (delta, now, account, tt),
        )

    @staticmethod
    def _track_minted(
        cursor: sqlite3.Cursor, year: int, tt: str, amount: float
    ) -> None:
        """Track yearly minted supply."""
        cursor.execute(
            """INSERT INTO token_supply (year, token_type, total_minted, total_burned)
            VALUES (?, ?, ?, 0.0)
            ON CONFLICT(year, token_type) DO UPDATE SET total_minted = total_minted + ?""",
            (year, tt, amount, amount),
        )

    @staticmethod
    def _track_burned(
        cursor: sqlite3.Cursor, year: int, tt: str, amount: float
    ) -> None:
        """Track yearly burned supply."""
        cursor.execute(
            """INSERT INTO token_supply (year, token_type, total_minted, total_burned)
            VALUES (?, ?, 0.0, ?)
            ON CONFLICT(year, token_type) DO UPDATE SET total_burned = total_burned + ?""",
            (year, tt, amount, amount),
        )

    def _append_to_log(self, tx: TransactionEntry) -> None:
        """Append transaction to JSONL log (immutable record)."""
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(tx.to_jsonl() + "\n")

    # =========================================================================
    # CHAIN VERIFICATION
    # =========================================================================

    def verify_chain(self) -> tuple[bool, int, Optional[str]]:
        """
        Verify the hash chain integrity of the JSONL log.

        Returns: (is_valid, entries_checked, error_message)
        """
        if not self._log_path.exists():
            return True, 0, None

        prev_hash = GENESIS_TX_HASH
        count = 0

        with open(self._log_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                entry = TransactionEntry.from_dict(data)

                # Verify chain link
                if entry.prev_hash != prev_hash:
                    return (
                        False,
                        count,
                        f"Chain break at seq {entry.sequence} (line {line_num}): "
                        f"prev_hash {entry.prev_hash[:16]}... != expected {prev_hash[:16]}...",
                    )

                # Verify hash
                computed = entry.compute_hash()
                if computed != entry.tx_hash:
                    return (
                        False,
                        count,
                        f"Hash mismatch at seq {entry.sequence}: "
                        f"computed {computed[:16]}... != stored {entry.tx_hash[:16]}...",
                    )

                prev_hash = entry.tx_hash
                count += 1

        return True, count, None

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_transaction_history(
        self,
        account_id: Optional[str] = None,
        token_type: Optional[TokenType] = None,
        limit: int = 100,
    ) -> list[TransactionEntry]:
        """Get transaction history with optional filters."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            conditions = []
            params: list = []

            if account_id:
                conditions.append("(from_account = ? OR to_account = ?)")
                params.extend([account_id, account_id])
            if token_type:
                conditions.append("token_type = ?")
                params.append(token_type.value)

            where = "WHERE " + " AND ".join(conditions) if conditions else ""

            cursor.execute(
                f"""
                SELECT tx_id, sequence, op, token_type, from_account, to_account,
                       amount, memo, epoch_id, poi_score, prev_hash, tx_hash,
                       signature, signer_pubkey, nonce, timestamp
                FROM token_transactions
                {where}
                ORDER BY sequence DESC
                LIMIT ?
                """,
                params + [limit],
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    TransactionEntry(
                        tx_id=row["tx_id"],
                        sequence=row["sequence"],
                        op=TokenOp(row["op"]),
                        token_type=TokenType(row["token_type"]),
                        from_account=row["from_account"],
                        to_account=row["to_account"],
                        amount=row["amount"],
                        memo=row["memo"],
                        epoch_id=row["epoch_id"],
                        poi_score=row["poi_score"],
                        prev_hash=row["prev_hash"],
                        tx_hash=row["tx_hash"],
                        signature=row["signature"],
                        signer_pubkey=row["signer_pubkey"],
                        nonce=row["nonce"],
                        timestamp=row["timestamp"],
                    )
                )
            return results

    def close(self) -> None:
        """Explicit cleanup (connections are per-operation, but good practice)."""
        logger.info(
            "Token ledger closed: %d transactions, last_hash=%s...",
            self._sequence,
            self._last_hash[:16],
        )


__all__ = [
    "TokenLedger",
    "GENESIS_TX_HASH",
    "SYSTEM_POOL_IDS",
    "DEFAULT_DB_PATH",
    "DEFAULT_LOG_PATH",
]
