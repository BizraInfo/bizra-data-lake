"""
PoI → Token Bridge — Connects Proof of Impact to Token Minting
================================================================

This is the bridge that makes ongoing SEED distribution automatic:

    PoI Engine (compute_token_distribution)
           ↓
    poi_bridge.distribute_epoch()
           ↓
    TokenMinter.distribute_from_poi()
           ↓
    TokenLedger (hash-chained, signed)

Standing on Giants:
- Nakamoto (2008): Block rewards proportional to contribution
- Al-Ghazali (1058-1111): Zakat as distributive justice
- Gini (1912): Inequality measurement
- Shannon (1948): SNR for quality gating
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from core.proof_engine.poi_engine import (
    AuditTrail,
    compute_token_distribution,
)
from core.token.mint import TokenMinter
from core.token.types import TokenReceipt

logger = logging.getLogger(__name__)


class PoITokenBridge:
    """
    Bridge between PoI engine output and token minting.

    Handles:
    1. Taking PoI AuditTrail → computing token distribution
    2. Feeding distribution to TokenMinter
    3. Also minting IMPT (reputation) for each contributor
    4. Returning receipts for audit trail

    Usage:
        bridge = PoITokenBridge.create()
        receipts = bridge.distribute_epoch(audit_trail, epoch_reward=10000)
    """

    def __init__(self, minter: TokenMinter):
        self._minter = minter

    @classmethod
    def create(
        cls,
        db_path: Optional[Path] = None,
        log_path: Optional[Path] = None,
    ) -> "PoITokenBridge":
        """Create bridge with new minter."""
        minter = TokenMinter.create(db_path=db_path, log_path=log_path)
        return cls(minter)

    @property
    def minter(self) -> TokenMinter:
        """The underlying token minter instance."""
        return self._minter

    def distribute_epoch(
        self,
        audit: AuditTrail,
        epoch_reward: float,
        *,
        scaling_factor: float = 1.0,
        mint_impt: bool = True,
        impt_multiplier: float = 100.0,
    ) -> dict[str, Any]:
        """
        Execute token distribution for a completed PoI epoch.

        Args:
            audit: PoI AuditTrail from poi_engine.run_epoch()
            epoch_reward: Total SEED reward for this epoch
            scaling_factor: Distribution scaling factor
            mint_impt: Also mint IMPT reputation tokens
            impt_multiplier: IMPT = poi_score * multiplier

        Returns:
            dict with distribution summary and receipts
        """
        # 1. Compute distribution from PoI scores
        distribution = compute_token_distribution(audit, epoch_reward, scaling_factor)

        # 2. Extract PoI scores for each contributor
        poi_scores = {p.contributor_id: p.poi_score for p in audit.poi_scores}

        # 3. Mint SEED tokens via minter
        seed_receipts = self._minter.distribute_from_poi(
            distributions=distribution.distributions,
            epoch_id=distribution.epoch_id,
            epoch_reward=epoch_reward,
            poi_scores=poi_scores,
        )

        # 4. Optionally mint IMPT (reputation)
        impt_receipts: list[TokenReceipt] = []
        if mint_impt:
            for cid, score in sorted(poi_scores.items()):
                if score > 0:
                    impt_amount = score * impt_multiplier
                    receipt = self._minter.mint_impt(
                        to_account=cid,
                        amount=impt_amount,
                        epoch_id=distribution.epoch_id,
                        poi_score=score,
                        memo=f"IMPT reputation: PoI={score:.4f}, epoch={distribution.epoch_id}",
                    )
                    impt_receipts.append(receipt)

        # 5. Summary
        seed_ok = sum(1 for r in seed_receipts if r.success)
        impt_ok = sum(1 for r in impt_receipts if r.success)

        summary = {
            "epoch_id": distribution.epoch_id,
            "epoch_reward": epoch_reward,
            "total_minted_seed": distribution.total_minted,
            "gini_coefficient": distribution.gini_coefficient,
            "seed_distributions": len(seed_receipts),
            "seed_succeeded": seed_ok,
            "impt_distributions": len(impt_receipts),
            "impt_succeeded": impt_ok,
            "contributors": len(distribution.distributions),
        }

        logger.info(
            "Epoch %s distributed: %d SEED (%d/%d ok), %d IMPT (%d/%d ok), Gini=%.4f",
            distribution.epoch_id,
            distribution.total_minted,
            seed_ok,
            len(seed_receipts),
            sum(r.tx_entry.amount for r in impt_receipts if r.success and r.tx_entry),
            impt_ok,
            len(impt_receipts),
            distribution.gini_coefficient,
        )

        return {
            "summary": summary,
            "seed_receipts": seed_receipts,
            "impt_receipts": impt_receipts,
            "distribution": distribution.to_dict(),
        }

    def status(self) -> dict[str, Any]:
        """Bridge status (delegates to minter)."""
        return self._minter.status()


__all__ = ["PoITokenBridge"]
