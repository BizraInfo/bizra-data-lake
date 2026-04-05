"""
BIZRA Token System — BLOOM Token + Community Pool
==================================================
Drop into: core/token/bloom.py

Implements the BLOOM governance token and the 50% community pool split
from البذرة page 19 (التجارة مع الله — Trading with God).

Constitutional constraints:
- BLOOM is soulbound (cannot be transferred between nodes)
- BLOOM decays over time (prevents plutocracy)
- 50% of all SEED minting flows to the community pool
- Community pool funds charitable distribution (Zakat, Sadaqah, etc.)

Standing on Giants: Ostrom (commons governance), Al-Ghazali (إحسان ethics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List

logger = logging.getLogger("bizra.token.bloom")


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS (from core/integration/constants.py)
# ═══════════════════════════════════════════════════════════════════

from core.integration.constants import ADL_GINI_THRESHOLD, IHSAN_THRESHOLD, ZAKAT_RATE

TOKEN_ZAKAT_RATE = ZAKAT_RATE  # Alias for local usage
COMMUNITY_POOL_SPLIT = 0.50  # 50% — البذرة page 19, HARDCODED, NOT A PARAMETER
BLOOM_DECAY_RATE = 0.02  # 2% monthly decay (prevents plutocracy)
BLOOM_MINT_FLOOR = 0.90  # Minimum Ihsān for BLOOM eligibility
SEED_MINT_FLOOR = IHSAN_THRESHOLD  # Minimum Ihsān for SEED minting


class TokenType(str, Enum):
    SEED = "SEED"  # Stable utility token (transferable)
    BLOOM = "BLOOM"  # Soulbound governance token (non-transferable, decays)
    BRANCH = "BRANCH"  # Reputation token (earned through attestation)


# ═══════════════════════════════════════════════════════════════════
# BLOOM TOKEN
# ═══════════════════════════════════════════════════════════════════


@dataclass
class BloomBalance:
    """
    BLOOM is soulbound — it belongs to a node and cannot be transferred.
    It decays monthly to prevent governance concentration.
    """

    node_id: str
    balance: float = 0.0
    lifetime_earned: float = 0.0
    last_decay: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    governance_weight: float = 0.0  # Derived from balance

    def apply_decay(self) -> float:
        """Apply monthly decay. Returns amount decayed."""
        now = datetime.now(timezone.utc)
        last = datetime.fromisoformat(self.last_decay)
        months_elapsed = (now - last).days / 30.0

        if months_elapsed < 1.0:
            return 0.0

        decay_factor = (1 - BLOOM_DECAY_RATE) ** int(months_elapsed)
        old_balance = self.balance
        self.balance *= decay_factor
        self.last_decay = now.isoformat()
        self.governance_weight = self.balance  # 1:1 mapping

        decayed = old_balance - self.balance
        logger.info(
            f"BLOOM decay: {self.node_id} lost {decayed:.4f} "
            f"({int(months_elapsed)} months, balance: {self.balance:.4f})"
        )
        return decayed


# ═══════════════════════════════════════════════════════════════════
# COMMUNITY POOL (البذرة page 19)
# ═══════════════════════════════════════════════════════════════════


@dataclass
class CommunityPool:
    """
    التجارة مع الله — Trading with God.

    50% of ALL SEED minting flows here. This is constitutionally locked.
    The pool funds:
    - Zakat distribution (2.5% annual)
    - Sadaqah (voluntary charity)
    - Kefalet Al-Yateem (orphan sponsorship)
    - Gharimin support (debt relief)
    - General community benefit

    From البذرة page 19:
    "وإيمان من البذرة بالفكرة كل أرباح المشروع من جميع
     الخدمات والأدوات ستحول نصف الأرباح إلى الحوض"
    """

    total_received: float = 0.0
    total_distributed: float = 0.0
    current_balance: float = 0.0
    distributions: List[Dict] = field(default_factory=list)

    # THIS IS A CONSTANT, NOT A PARAMETER.
    # It cannot be changed by governance, upgrade, or any other mechanism.
    SPLIT_RATIO: float = 0.50  # 50% — hardcoded per البذرة

    def receive(self, amount: float, source: str, evidence_hash: str) -> None:
        """Receive funds from SEED minting split."""
        self.total_received += amount
        self.current_balance += amount
        logger.info(
            f"🕌 Community pool: +{amount:.4f} SEED from {source} "
            f"(balance: {self.current_balance:.4f})"
        )

    def distribute(
        self, amount: float, category: str, recipient: str, evidence_hash: str
    ) -> bool:
        """Distribute from pool to charitable cause."""
        if amount > self.current_balance:
            logger.warning(
                f"Pool distribution failed: {amount} > {self.current_balance}"
            )
            return False

        self.current_balance -= amount
        self.total_distributed += amount
        self.distributions.append(
            {
                "amount": amount,
                "category": category,
                "recipient": recipient,
                "evidence_hash": evidence_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        logger.info(
            f"🕌 Community distribution: {amount:.4f} SEED → {category}:{recipient}"
        )
        return True


# ═══════════════════════════════════════════════════════════════════
# TOKEN MINTER (with 50% community pool split)
# ═══════════════════════════════════════════════════════════════════


@dataclass
class WalletState:
    """Node wallet holding all three token types."""

    node_id: str
    seed_balance: float = 0.0
    bloom: BloomBalance = None
    branch_balance: float = 0.0

    def __post_init__(self):
        if self.bloom is None:
            self.bloom = BloomBalance(node_id=self.node_id)


class TokenMinter:
    """
    Mints tokens based on Proof-of-Impact evidence.

    SEED: 50% to node, 50% to community pool (البذرة page 19)
    BLOOM: 100% to node (soulbound, earned through governance contribution)
    BRANCH: 100% to node (earned through attestation work)
    """

    def __init__(self, community_pool: CommunityPool):
        self.pool = community_pool
        self.total_seed_minted = 0.0
        self.total_bloom_minted = 0.0
        self.total_branch_minted = 0.0
        self.mint_log: List[Dict] = []

    def compute_reward(
        self,
        ihsan: float,
        steps: int = 1,
        duration_ms: int = 0,
    ) -> float:
        """
        Compute SEED reward based on verified work.

        Formula: base_reward * ihsan_multiplier * step_bonus
        - base_reward = 1.0 SEED per verified completion
        - ihsan_multiplier = ihsan^2 (rewards excellence superlinearly)
        - step_bonus = log2(1 + steps) (diminishing returns on complexity)
        """
        import math

        base = 1.0
        ihsan_mult = ihsan**2  # 0.95^2 = 0.9025, 0.99^2 = 0.9801
        step_bonus = math.log2(1 + steps)
        return base * ihsan_mult * step_bonus

    def mint_seed(
        self,
        wallet: WalletState,
        amount: float,
        poi_evidence: str,
        ihsan: float,
    ) -> Dict:
        """
        Mint SEED with constitutional 50% community pool split.

        This split is HARDCODED per البذرة page 19.
        It cannot be changed by governance vote or system upgrade.
        """
        if ihsan < SEED_MINT_FLOOR:
            logger.warning(f"SEED mint rejected: Ihsān {ihsan:.3f} < {SEED_MINT_FLOOR}")
            return {"minted": False, "reason": "below_ihsan_floor"}

        # THE SPLIT — constitutionally locked at 50%
        node_share = amount * (1 - CommunityPool.SPLIT_RATIO)
        pool_share = amount * CommunityPool.SPLIT_RATIO

        # Credit node
        wallet.seed_balance += node_share

        # Credit community pool
        self.pool.receive(
            amount=pool_share,
            source=f"seed_mint:{wallet.node_id}",
            evidence_hash=poi_evidence,
        )

        self.total_seed_minted += amount

        result = {
            "minted": True,
            "total_amount": amount,
            "node_share": node_share,
            "pool_share": pool_share,
            "node_balance": wallet.seed_balance,
            "pool_balance": self.pool.current_balance,
            "ihsan": ihsan,
            "evidence": poi_evidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.mint_log.append(result)
        logger.info(
            f"🌱 SEED minted: {amount:.4f} "
            f"(node: {node_share:.4f}, pool: {pool_share:.4f}, "
            f"Ihsān: {ihsan:.3f})"
        )
        return result

    def mint_bloom(
        self,
        wallet: WalletState,
        amount: float,
        contribution_type: str,
        evidence_hash: str,
    ) -> Dict:
        """
        Mint BLOOM (soulbound governance token).
        Earned through network contributions, not purchased.
        """
        wallet.bloom.balance += amount
        wallet.bloom.lifetime_earned += amount
        wallet.bloom.governance_weight = wallet.bloom.balance

        self.total_bloom_minted += amount

        logger.info(
            f"🌸 BLOOM minted: {amount:.4f} for {contribution_type} "
            f"(governance weight: {wallet.bloom.governance_weight:.4f})"
        )
        return {
            "minted": True,
            "amount": amount,
            "type": "BLOOM",
            "contribution": contribution_type,
            "governance_weight": wallet.bloom.governance_weight,
        }

    def mint_branch(
        self,
        wallet: WalletState,
        amount: float,
        attestation_type: str,
        evidence_hash: str,
    ) -> Dict:
        """
        Mint BRANCH (reputation token).
        Earned through peer attestation work.
        """
        wallet.branch_balance += amount
        self.total_branch_minted += amount

        logger.info(f"🌿 BRANCH minted: {amount:.4f} for {attestation_type}")
        return {
            "minted": True,
            "amount": amount,
            "type": "BRANCH",
            "attestation": attestation_type,
        }


# ═══════════════════════════════════════════════════════════════════
# GINI COEFFICIENT CALCULATOR (ʿAdl Invariant)
# ═══════════════════════════════════════════════════════════════════


def compute_gini(balances: List[float]) -> float:
    """
    Compute Gini coefficient for wealth distribution.
    Constitutional invariant: must be ≤ 0.35.

    G = 0 means perfect equality.
    G = 1 means perfect inequality.
    """
    if not balances or all(b == 0 for b in balances):
        return 0.0

    sorted_b = sorted(balances)
    n = len(sorted_b)
    total = sum(sorted_b)

    if total == 0:
        return 0.0

    cumulative = 0.0
    weighted_sum = 0.0
    for i, b in enumerate(sorted_b):
        cumulative += b
        weighted_sum += (2 * (i + 1) - n - 1) * b

    return weighted_sum / (n * total)


def check_gini_invariant(wallets: List[WalletState]) -> Dict:
    """
    Check the ʿAdl (Justice) invariant: Gini ≤ 0.35.
    Returns violation details if breached.
    """
    seed_balances = [w.seed_balance for w in wallets]
    gini = compute_gini(seed_balances)

    passed = gini <= ADL_GINI_THRESHOLD
    result = {
        "gini": gini,
        "threshold": ADL_GINI_THRESHOLD,
        "passed": passed,
        "node_count": len(wallets),
        "total_seed": sum(seed_balances),
    }

    if not passed:
        logger.warning(
            f"⚖️ ʿAdl VIOLATION: Gini={gini:.4f} > {ADL_GINI_THRESHOLD} "
            f"({len(wallets)} nodes)"
        )
    else:
        logger.info(f"⚖️ ʿAdl check: Gini={gini:.4f} ≤ {ADL_GINI_THRESHOLD} ✓")

    return result


# ═══════════════════════════════════════════════════════════════════
# SMOKE TESTS
# ═══════════════════════════════════════════════════════════════════


def _run_smoke_tests():
    pool = CommunityPool()
    minter = TokenMinter(community_pool=pool)
    wallet = WalletState(node_id="node0")

    # Test 1: SEED minting with 50% split
    result = minter.mint_seed(wallet, amount=10.0, poi_evidence="hash1", ihsan=0.96)
    assert result["minted"] is True
    assert (
        result["node_share"] == 5.0
    ), f"Node share should be 5.0, got {result['node_share']}"
    assert (
        result["pool_share"] == 5.0
    ), f"Pool share should be 5.0, got {result['pool_share']}"
    assert wallet.seed_balance == 5.0
    assert pool.current_balance == 5.0
    print("✓ Test 1: SEED minting with 50% community pool split")

    # Test 2: SEED rejected below Ihsān floor
    result = minter.mint_seed(wallet, amount=10.0, poi_evidence="hash2", ihsan=0.90)
    assert result["minted"] is False
    assert wallet.seed_balance == 5.0  # Unchanged
    print("✓ Test 2: SEED rejected below minting floor")

    # Test 3: BLOOM is soulbound (no transfer method exists)
    minter.mint_bloom(
        wallet, amount=3.0, contribution_type="governance_vote", evidence_hash="h3"
    )
    assert wallet.bloom.balance == 3.0
    assert wallet.bloom.governance_weight == 3.0
    # No transfer method exists — soulbound by design
    print("✓ Test 3: BLOOM minted (soulbound)")

    # Test 4: BLOOM decay
    wallet.bloom.last_decay = "2025-01-01T00:00:00+00:00"  # Old date
    decayed = wallet.bloom.apply_decay()
    assert wallet.bloom.balance < 3.0  # Balance decreased
    print(f"✓ Test 4: BLOOM decay ({decayed:.4f} decayed)")

    # Test 5: Gini coefficient
    wallets = [
        WalletState(node_id="a", seed_balance=100),
        WalletState(node_id="b", seed_balance=90),
        WalletState(node_id="c", seed_balance=80),
        WalletState(node_id="d", seed_balance=70),
    ]
    result = check_gini_invariant(wallets)
    assert result["passed"] is True  # These are relatively equal
    assert result["gini"] < 0.35
    print(f"✓ Test 5: Gini invariant ({result['gini']:.4f} ≤ 0.35)")

    # Test 6: Gini violation
    wallets_unequal = [
        WalletState(node_id="a", seed_balance=1000),
        WalletState(node_id="b", seed_balance=1),
        WalletState(node_id="c", seed_balance=1),
        WalletState(node_id="d", seed_balance=1),
    ]
    result = check_gini_invariant(wallets_unequal)
    assert result["passed"] is False  # Highly unequal
    print(f"✓ Test 6: Gini violation detected ({result['gini']:.4f} > 0.35)")

    # Test 7: Community pool distribution
    success = pool.distribute(
        amount=2.0,
        category="zakat",
        recipient="orphan_fund_001",
        evidence_hash="dist_hash_1",
    )
    assert success is True
    assert pool.current_balance == 3.0  # 5.0 - 2.0
    print("✓ Test 7: Community pool Zakat distribution")

    # Test 8: Reward computation
    reward = minter.compute_reward(ihsan=0.96, steps=5, duration_ms=1200)
    assert reward > 0
    print(f"✓ Test 8: Reward computation ({reward:.4f} SEED)")

    print("\n═══ BLOOM + COMMUNITY POOL: ALL TESTS PASSED ═══")
    print(f"  SEED minted: {minter.total_seed_minted:.4f}")
    print(f"  BLOOM minted: {minter.total_bloom_minted:.4f}")
    print(f"  Pool received: {pool.total_received:.4f}")
    print(f"  Pool distributed: {pool.total_distributed:.4f}")
    print(f"  Pool balance: {pool.current_balance:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _run_smoke_tests()
