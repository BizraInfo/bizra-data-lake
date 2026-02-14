"""
Token Minter — SEED/BLOOM Minting Engine
==========================================

The engine that converts Proof of Impact into actual tokens.

Architecture:
    PoI Engine → TokenDistribution → TokenMinter → TokenLedger
                                         ↓
                                    Zakat (2.5%)
                                         ↓
                                  Community Fund

Every mint is:
- Signed with Ed25519 (signer must hold minter authority)
- Hash-chained to previous transaction
- Gated by yearly supply cap (1M SEED/year)
- Subject to 2.5% computational zakat

Standing on Giants:
- Nakamoto (2008): Genesis block and coin issuance
- Shannon (1948): SNR for quality gating
- Al-Ghazali (1058-1111): Zakat as distributive justice
- Gini (1912): Inequality measurement and rebalancing
- Lamport (1978): Signed messages and ordering
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.pci.crypto import (
    generate_keypair,
    sign_message,
    verify_signature,
)
from core.token.ledger import TokenLedger
from core.token.types import (
    FOUNDER_GENESIS_ALLOCATION,
    GENESIS_EPOCH_ID,
    SEED_SUPPLY_CAP_PER_YEAR,
    SYSTEM_TREASURY_ALLOCATION,
    ZAKAT_RATE,
    TokenOp,
    TokenReceipt,
    TokenType,
    TransactionEntry,
)

logger = logging.getLogger(__name__)

# Community fund account for zakat
COMMUNITY_FUND_ACCOUNT = "BIZRA-COMMUNITY-FUND"

# System treasury account
SYSTEM_TREASURY_ACCOUNT = "SYSTEM-TREASURY"

# Genesis Node0 account
GENESIS_NODE0_ACCOUNT = "BIZRA-00000000"


class TokenMinter:
    """
    BIZRA Token Minter — Converts Proof of Impact into tokens.

    The minter holds signing authority and can:
    1. Mint SEED tokens based on PoI scores
    2. Mint BLOOM tokens from SEED staking rewards
    3. Execute the Genesis Mint (one-time founder allocation)
    4. Transfer tokens between accounts
    5. Burn tokens

    All operations are signed and recorded in the token ledger.

    Usage:
        minter = TokenMinter.create()
        receipt = minter.mint_seed("node-42", 100.0, epoch_id="epoch-1", poi_score=0.87)
        genesis_receipts = minter.genesis_mint()
    """

    def __init__(
        self,
        private_key_hex: str,
        public_key_hex: str,
        ledger: TokenLedger,
    ):
        self._private_key = private_key_hex
        self._public_key = public_key_hex
        self._ledger = ledger
        self._genesis_minted = False

    @classmethod
    def create(
        cls,
        ledger: Optional[TokenLedger] = None,
        db_path: Optional[Path] = None,
        log_path: Optional[Path] = None,
    ) -> "TokenMinter":
        """Create a new minter with fresh keypair and ledger."""
        private_key, public_key = generate_keypair()
        if ledger is None:
            ledger = TokenLedger(db_path=db_path, log_path=log_path)
        return cls(
            private_key_hex=private_key,
            public_key_hex=public_key,
            ledger=ledger,
        )

    @property
    def public_key(self) -> str:
        """Minter's public key (safe to expose)."""
        return self._public_key

    @property
    def ledger(self) -> TokenLedger:
        """Access the underlying ledger."""
        return self._ledger

    # =========================================================================
    # CORE MINTING
    # =========================================================================

    def mint_seed(
        self,
        to_account: str,
        amount: float,
        *,
        epoch_id: str = "",
        poi_score: float = 0.0,
        memo: str = "",
    ) -> TokenReceipt:
        """
        Mint SEED tokens to an account.

        Enforces yearly supply cap (1M SEED/year).
        Applies 2.5% computational zakat to community fund.

        Args:
            to_account: Recipient account ID
            amount: Amount of SEED to mint
            epoch_id: PoI epoch that justified this mint
            poi_score: PoI score of the recipient
            memo: Human-readable memo

        Returns:
            TokenReceipt with transaction details
        """
        # Check yearly supply cap
        year = datetime.now(timezone.utc).year
        already_minted = self._ledger.get_yearly_minted(TokenType.SEED, year)
        if already_minted + amount > SEED_SUPPLY_CAP_PER_YEAR:
            remaining = SEED_SUPPLY_CAP_PER_YEAR - already_minted
            return TokenReceipt(
                success=False,
                error=(
                    f"Yearly supply cap exceeded: {already_minted:.2f} already minted, "
                    f"cap is {SEED_SUPPLY_CAP_PER_YEAR}, remaining: {remaining:.2f}"
                ),
            )

        # Compute zakat amount
        zakat_amount = amount * ZAKAT_RATE
        net_amount = amount - zakat_amount

        # Mint net amount to recipient
        tx = self._build_transaction(
            op=TokenOp.MINT,
            token_type=TokenType.SEED,
            to_account=to_account,
            amount=net_amount,
            epoch_id=epoch_id,
            poi_score=poi_score,
            memo=memo or f"SEED mint from epoch {epoch_id}",
        )
        receipt = self._record_and_sign(tx)

        if not receipt.success:
            return receipt

        # Mint zakat to community fund
        if zakat_amount > 0:
            zakat_tx = self._build_transaction(
                op=TokenOp.ZAKAT,
                token_type=TokenType.SEED,
                to_account=COMMUNITY_FUND_ACCOUNT,
                amount=zakat_amount,
                epoch_id=epoch_id,
                memo=f"Computational zakat (2.5%) from mint to {to_account}",
            )
            zakat_receipt = self._record_and_sign(zakat_tx)
            if not zakat_receipt.success:
                logger.warning("Zakat transaction failed: %s", zakat_receipt.error)

        return receipt

    def mint_bloom(
        self,
        to_account: str,
        amount: float,
        *,
        staking_epoch: str = "",
        memo: str = "",
    ) -> TokenReceipt:
        """
        Mint BLOOM governance tokens.

        BLOOM is earned through SEED staking, not direct PoI.

        Args:
            to_account: Recipient account ID
            amount: Amount of BLOOM to mint
            staking_epoch: Staking period reference
            memo: Human-readable memo
        """
        tx = self._build_transaction(
            op=TokenOp.MINT,
            token_type=TokenType.BLOOM,
            to_account=to_account,
            amount=amount,
            epoch_id=staking_epoch,
            memo=memo or f"BLOOM mint from staking epoch {staking_epoch}",
        )
        return self._record_and_sign(tx)

    def mint_impt(
        self,
        to_account: str,
        amount: float,
        *,
        epoch_id: str = "",
        poi_score: float = 0.0,
        memo: str = "",
    ) -> TokenReceipt:
        """
        Mint IMPT reputation tokens (non-transferable).

        IMPT is the soulbound reputation score that compounds reward multipliers.
        """
        tx = self._build_transaction(
            op=TokenOp.MINT,
            token_type=TokenType.IMPT,
            to_account=to_account,
            amount=amount,
            epoch_id=epoch_id,
            poi_score=poi_score,
            memo=memo or f"IMPT reputation from epoch {epoch_id}",
        )
        return self._record_and_sign(tx)

    # =========================================================================
    # GENESIS MINT
    # =========================================================================

    def genesis_mint(self) -> list[TokenReceipt]:
        """
        Execute the Genesis Mint — one-time founder allocation.

        This is the Nakamoto moment: the first tokens ever created
        in the BIZRA network, allocated to Node0 (MoMo) for 3 years
        of foundational architecture work.

        Allocations:
            - Node0 (MoMo): 100,000 SEED (founder allocation)
            - System Treasury: 50,000 SEED (operational reserves)
            - Community Fund: 3,750 SEED (zakat on above)
            - Node0 IMPT: 1,000 (founder reputation)

        Returns:
            list of TokenReceipts for each allocation
        """
        if self._genesis_minted:
            return [TokenReceipt(success=False, error="Genesis mint already executed")]

        # Check if genesis was already done (resume safety)
        existing = self._ledger.get_transaction_history(
            account_id=GENESIS_NODE0_ACCOUNT,
            token_type=TokenType.SEED,
            limit=1,
        )
        if existing and any(tx.op == TokenOp.GENESIS_MINT for tx in existing):
            self._genesis_minted = True
            return [
                TokenReceipt(
                    success=False, error="Genesis mint already recorded in ledger"
                )
            ]

        receipts: list[TokenReceipt] = []

        # 1. Founder SEED allocation — Node0 (MoMo)
        founder_tx = self._build_transaction(
            op=TokenOp.GENESIS_MINT,
            token_type=TokenType.SEED,
            to_account=GENESIS_NODE0_ACCOUNT,
            amount=FOUNDER_GENESIS_ALLOCATION,
            epoch_id=GENESIS_EPOCH_ID,
            poi_score=1.0,  # Maximum PoI — the founder
            memo=(
                "Genesis Mint: Node0 founder allocation. "
                "3 years of foundational architecture (2023-2026). "
                "بذرة — The first seed."
            ),
        )
        receipts.append(self._record_and_sign(founder_tx))

        # 2. System Treasury allocation
        treasury_tx = self._build_transaction(
            op=TokenOp.GENESIS_MINT,
            token_type=TokenType.SEED,
            to_account=SYSTEM_TREASURY_ACCOUNT,
            amount=SYSTEM_TREASURY_ALLOCATION,
            epoch_id=GENESIS_EPOCH_ID,
            memo="Genesis Mint: System treasury operational reserves",
        )
        receipts.append(self._record_and_sign(treasury_tx))

        # 3. Computational Zakat on genesis allocations
        total_genesis = FOUNDER_GENESIS_ALLOCATION + SYSTEM_TREASURY_ALLOCATION
        zakat_amount = total_genesis * ZAKAT_RATE
        zakat_tx = self._build_transaction(
            op=TokenOp.ZAKAT,
            token_type=TokenType.SEED,
            to_account=COMMUNITY_FUND_ACCOUNT,
            amount=zakat_amount,
            epoch_id=GENESIS_EPOCH_ID,
            memo=f"Computational zakat (2.5%) on genesis allocation of {total_genesis} SEED",
        )
        receipts.append(self._record_and_sign(zakat_tx))

        # 4. Founder IMPT (reputation) allocation
        impt_tx = self._build_transaction(
            op=TokenOp.GENESIS_MINT,
            token_type=TokenType.IMPT,
            to_account=GENESIS_NODE0_ACCOUNT,
            amount=1000.0,
            epoch_id=GENESIS_EPOCH_ID,
            poi_score=1.0,
            memo="Genesis Mint: Node0 founder reputation score — 3 years of impact",
        )
        receipts.append(self._record_and_sign(impt_tx))

        self._genesis_minted = True

        # Log the genesis event
        succeeded = sum(1 for r in receipts if r.success)
        logger.info(
            "GENESIS MINT COMPLETE: %d/%d transactions succeeded. "
            "Node0=%s SEED, Treasury=%s SEED, Zakat=%s SEED, IMPT=%s",
            succeeded,
            len(receipts),
            FOUNDER_GENESIS_ALLOCATION,
            SYSTEM_TREASURY_ALLOCATION,
            zakat_amount,
            1000.0,
        )

        return receipts

    # =========================================================================
    # TRANSFERS AND BURNS
    # =========================================================================

    def transfer(
        self,
        from_account: str,
        to_account: str,
        token_type: TokenType,
        amount: float,
        *,
        memo: str = "",
    ) -> TokenReceipt:
        """Transfer tokens between accounts."""
        tx = self._build_transaction(
            op=TokenOp.TRANSFER,
            token_type=token_type,
            from_account=from_account,
            to_account=to_account,
            amount=amount,
            memo=memo or f"Transfer {amount} {token_type.value}",
        )
        return self._record_and_sign(tx)

    def burn(
        self,
        from_account: str,
        token_type: TokenType,
        amount: float,
        *,
        memo: str = "",
    ) -> TokenReceipt:
        """Burn tokens (remove from circulation)."""
        tx = self._build_transaction(
            op=TokenOp.BURN,
            token_type=token_type,
            from_account=from_account,
            amount=amount,
            memo=memo or f"Burn {amount} {token_type.value}",
        )
        return self._record_and_sign(tx)

    def stake(
        self,
        account_id: str,
        token_type: TokenType,
        amount: float,
        *,
        memo: str = "",
    ) -> TokenReceipt:
        """Stake tokens (lock for governance/rewards)."""
        tx = self._build_transaction(
            op=TokenOp.STAKE,
            token_type=token_type,
            from_account=account_id,
            amount=amount,
            memo=memo or f"Stake {amount} {token_type.value}",
        )
        return self._record_and_sign(tx)

    def unstake(
        self,
        account_id: str,
        token_type: TokenType,
        amount: float,
        *,
        memo: str = "",
    ) -> TokenReceipt:
        """Unstake tokens (release from lock)."""
        tx = self._build_transaction(
            op=TokenOp.UNSTAKE,
            token_type=token_type,
            from_account=account_id,
            amount=amount,
            memo=memo or f"Unstake {amount} {token_type.value}",
        )
        return self._record_and_sign(tx)

    # =========================================================================
    # POI-DRIVEN DISTRIBUTION
    # =========================================================================

    def distribute_from_poi(
        self,
        distributions: dict[str, float],
        epoch_id: str,
        epoch_reward: float,
        poi_scores: Optional[dict[str, float]] = None,
    ) -> list[TokenReceipt]:
        """
        Distribute SEED tokens based on PoI engine output.

        This is the bridge from compute_token_distribution() to actual minting.

        Args:
            distributions: Mapping of account_id -> token amount
            epoch_id: PoI epoch identifier
            epoch_reward: Total epoch reward
            poi_scores: Optional mapping of account_id -> poi_score

        Returns:
            list of TokenReceipts for each distribution
        """
        poi_scores = poi_scores or {}
        receipts: list[TokenReceipt] = []

        for account_id, amount in sorted(distributions.items()):
            if amount <= 0:
                continue
            receipt = self.mint_seed(
                to_account=account_id,
                amount=amount,
                epoch_id=epoch_id,
                poi_score=poi_scores.get(account_id, 0.0),
                memo=f"PoI distribution: epoch={epoch_id}, reward={epoch_reward}",
            )
            receipts.append(receipt)

        succeeded = sum(1 for r in receipts if r.success)
        logger.info(
            "PoI distribution for epoch %s: %d/%d mints succeeded, "
            "total_reward=%s SEED",
            epoch_id,
            succeeded,
            len(receipts),
            epoch_reward,
        )

        return receipts

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _build_transaction(
        self,
        op: TokenOp,
        token_type: TokenType,
        amount: float,
        to_account: str = "",
        from_account: str = "",
        epoch_id: str = "",
        poi_score: float = 0.0,
        memo: str = "",
    ) -> TransactionEntry:
        """Build and sign a transaction."""
        tx = TransactionEntry(
            op=op,
            token_type=token_type,
            from_account=from_account,
            to_account=to_account,
            amount=amount,
            memo=memo,
            epoch_id=epoch_id,
            poi_score=poi_score,
            signer_pubkey=self._public_key,
        )

        # CRITICAL-5 FIX: Do NOT sign here. The hash will change when
        # ledger assigns sequence/prev_hash. Signing must happen AFTER
        # ledger finalizes the hash — see _record_and_sign().
        # Standing on Giants: Merkle (1979) — hash chain integrity requires
        # signing the FINAL hash, not a preliminary one.
        tx.tx_hash = tx.compute_hash()  # Preliminary, will be overwritten by ledger

        return tx

    def _record_and_sign(self, tx: TransactionEntry) -> "TokenReceipt":
        """Record transaction in ledger, then sign the FINAL hash.

        CRITICAL-5 FIX: Previously, _build_transaction signed before the
        ledger assigned sequence/prev_hash. The ledger then recomputed
        tx_hash (different!), making the signature cover the wrong hash.

        Fix: sign AFTER ledger finalizes sequence + prev_hash + tx_hash.
        """
        receipt = self._ledger.record_transaction(tx)
        if receipt.success and self._private_key:
            # NOW tx.tx_hash is final (includes sequence + prev_hash)
            tx.signature = sign_message(tx.tx_hash, self._private_key)
        return receipt

    def verify_transaction(self, tx: TransactionEntry) -> bool:
        """Verify a transaction's signature."""
        computed_hash = tx.compute_hash()
        if computed_hash != tx.tx_hash:
            return False
        return verify_signature(tx.tx_hash, tx.signature, tx.signer_pubkey)

    # =========================================================================
    # STATUS
    # =========================================================================

    def status(self) -> dict[str, Any]:
        """Get minter status summary."""
        seed_supply = self._ledger.get_total_supply(TokenType.SEED)
        bloom_supply = self._ledger.get_total_supply(TokenType.BLOOM)
        year = datetime.now(timezone.utc).year
        yearly = self._ledger.get_yearly_minted(TokenType.SEED, year)

        return {
            "minter_pubkey": self._public_key[:16] + "...",
            "genesis_minted": self._genesis_minted,
            "ledger_sequence": self._ledger.sequence,
            "total_supply": {
                "SEED": seed_supply,
                "BLOOM": bloom_supply,
            },
            "yearly_minted_seed": yearly,
            "yearly_cap_seed": SEED_SUPPLY_CAP_PER_YEAR,
            "yearly_remaining_seed": SEED_SUPPLY_CAP_PER_YEAR - yearly,
        }


__all__ = [
    "TokenMinter",
    "COMMUNITY_FUND_ACCOUNT",
    "SYSTEM_TREASURY_ACCOUNT",
    "GENESIS_NODE0_ACCOUNT",
]
