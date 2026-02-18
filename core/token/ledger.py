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

        return None

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
    "DEFAULT_DB_PATH",
    "DEFAULT_LOG_PATH",
]
