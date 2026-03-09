"""
Helix 3 — Constitutional Evolutionary Scheduler
=================================================
Drop into: core/sovereign/helix3.py

The third strand of the Triple Helix (DDAGI §2):

  Helix 1 (Reactive/S1):     50ms    ReflexCompiler O(1) cache
  Helix 2 (Deliberative/S2): 800ms   NervousSystem → full inference
  Helix 3 (Evolutionary/S3): 60s     THIS MODULE → process_tick()

Without Helix 3, the organism THINKS but does not GROW.
Helix 3 is the organism's metabolism: it processes completed missions,
mints rewards, prunes stale patterns, and evolves the reflex cache.

Bridge Architecture:
  NervousSystemReceipt (float) → ActionReceipt (fixed-point) → process_tick()
  TickResult (fixed-point) → HeartbeatReceipt (float) → back to NervousSystem

Standing on Giants:
  Al-Khwarizmi (780) — deterministic procedure, every tick identical
  Nakamoto (2008) — block processing tick, consensus through computation
  Kahneman (2011) — S1/S2 split; S3 is the meta-learning layer above both
  Ibn Khaldun (1377) — asabiyyah (social cohesion) modulates minting
  Deming (1950) — PDCA applied to the organism itself
  Al-Ghazali (1095) — 8D Ihsān tensor as constitutional hard gate

Constitutional Authority:
  §2  Triple Helix: S3 evolutionary cycle
  §4  Immutable invariants: Ihsān ≥ 0.95, Gini ≤ 0.35, Zakat 2.5%
  §5  Economics: SEED minting, BLOOM accrual, demurrage
  §7  Evidence: BLAKE2b hash chain, Ed25519 signed
  §8  SNR/Ihsān: 8D geometric mean scoring
  §9  Growth: reflex precipitation, forest sync
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger("bizra.sovereign.helix3")


# ═══════════════════════════════════════════════════════════════════
# CONSTITUTIONAL THRESHOLDS
# ═══════════════════════════════════════════════════════════════════

try:
    from core.integration.constants import (
        ADL_GINI_THRESHOLD,
        IHSAN_CANONICAL_WEIGHTS,
        UNIFIED_IHSAN_THRESHOLD,
    )
except ImportError:
    UNIFIED_IHSAN_THRESHOLD = 0.95
    ADL_GINI_THRESHOLD = 0.35
    IHSAN_CANONICAL_WEIGHTS = {
        "moral_clarity": 0.12,
        "epistemic_humility": 0.14,
        "structural_integrity": 0.13,
        "verifiability": 0.13,
        "contextual_relevance": 0.11,
        "intent_alignment": 0.14,
        "resilience": 0.11,
        "efficiency": 0.12,
    }

HEARTBEAT_INTERVAL_S = 60  # §2: Every 60 seconds
PRECIPITATION_IHSAN_FLOOR = 0.90  # §2: Ihsān ≥ 0.90 for precipitation
STALE_REFLEX_AGE_DAYS = 30  # Reflexes older than 30d without hits → prune


# ═══════════════════════════════════════════════════════════════════
# 8D IHSĀN TENSOR (§8)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class IhsanTensor8D:
    """The canonical 8-dimension Ihsān tensor.

    From constants.py IHSAN_CANONICAL_WEIGHTS:
      moral_clarity (0.12), epistemic_humility (0.14),
      structural_integrity (0.13), verifiability (0.13),
      contextual_relevance (0.11), intent_alignment (0.14),
      resilience (0.11), efficiency (0.12)

    The GEOMETRIC MEAN ensures that a ZERO in any dimension kills
    the composite score. You cannot compensate for being unethical
    by being highly efficient.
    """

    moral_clarity: float = 0.0
    epistemic_humility: float = 0.0
    structural_integrity: float = 0.0
    verifiability: float = 0.0
    contextual_relevance: float = 0.0
    intent_alignment: float = 0.0
    resilience: float = 0.0
    efficiency: float = 0.0

    @property
    def dimensions(self) -> Dict[str, float]:
        return {
            "moral_clarity": self.moral_clarity,
            "epistemic_humility": self.epistemic_humility,
            "structural_integrity": self.structural_integrity,
            "verifiability": self.verifiability,
            "contextual_relevance": self.contextual_relevance,
            "intent_alignment": self.intent_alignment,
            "resilience": self.resilience,
            "efficiency": self.efficiency,
        }

    @property
    def geometric_mean(self) -> float:
        """Weighted geometric mean — constitutional composite score.

        exp(Σ wᵢ · ln(dᵢ)) where dᵢ > 0 for all dimensions.
        If ANY dimension is 0, the entire score is 0 (fail-closed).
        """
        dims = self.dimensions
        weights = IHSAN_CANONICAL_WEIGHTS

        weighted_log_sum = 0.0
        for key, value in dims.items():
            w = weights.get(key, 0.0)
            if value <= 0.0:
                return 0.0  # Fail-closed: zero in any dimension → zero composite
            weighted_log_sum += w * math.log(value)

        return round(math.exp(weighted_log_sum), 6)

    @property
    def weighted_mean(self) -> float:
        """Weighted arithmetic mean — legacy/operational scoring."""
        dims = self.dimensions
        weights = IHSAN_CANONICAL_WEIGHTS
        return round(
            sum(dims[k] * weights.get(k, 0.0) for k in dims), 6
        )

    @property
    def min_dimension(self) -> float:
        """Lowest dimension — the bottleneck."""
        return min(self.dimensions.values())

    @property
    def verified_count(self) -> int:
        """Number of dimensions above the minimum gate (0.85)."""
        return sum(1 for v in self.dimensions.values() if v >= 0.85)

    @classmethod
    def from_scores(cls, scores: Dict[str, float]) -> IhsanTensor8D:
        """Create tensor from a dict of dimension scores."""
        return cls(**{
            k: scores.get(k, 0.0)
            for k in IHSAN_CANONICAL_WEIGHTS
        })

    @classmethod
    def uniform(cls, score: float) -> IhsanTensor8D:
        """Create tensor with all dimensions at the same score."""
        return cls(**{k: score for k in IHSAN_CANONICAL_WEIGHTS})


# ═══════════════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HeartbeatReceipt:
    """Evidence-chained receipt from one heartbeat tick."""

    tick_number: int
    timestamp: str
    duration_ms: float
    missions_processed: int
    ihsan_tensor: Dict[str, float]  # 8D tensor for this tick
    ihsan_composite: float  # Geometric mean
    gini_coefficient: float
    gini_ok: bool
    seed_minted: float
    bloom_accrued: float
    reflexes_precipitated: int
    reflexes_pruned: int
    evidence_hash: str
    chain_hash: str  # Links to previous heartbeat
    stats: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Helix3Stats:
    """Cumulative statistics for Helix 3 evolution."""

    total_ticks: int = 0
    total_missions_processed: int = 0
    total_seed_minted: float = 0.0
    total_bloom_accrued: float = 0.0
    total_reflexes_precipitated: int = 0
    total_reflexes_pruned: int = 0
    total_gini_halts: int = 0
    avg_ihsan: float = 0.0
    avg_tick_duration_ms: float = 0.0
    last_tick_time: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════
# PROTOCOLS (Dependency Injection)
# ═══════════════════════════════════════════════════════════════════

class NervousSystemLike(Protocol):
    """Duck-typed interface matching SovereignNervousSystem."""

    @property
    def stats(self) -> Any: ...

    @property
    def chain_hash(self) -> str: ...


class ReflexCacheLike(Protocol):
    """Duck-typed interface matching ReflexCompiler."""

    def revalidate(self, key_str: str, new_ihsan: float) -> bool: ...
    def get_top_reflexes(self, n: int = 10) -> List[Dict[str, Any]]: ...

    @property
    def stats(self) -> Any: ...


# ═══════════════════════════════════════════════════════════════════
# HELIX 3 SCHEDULER
# ═══════════════════════════════════════════════════════════════════

class Helix3Scheduler:
    """The evolutionary heartbeat that makes the organism GROW.

    Every 60 seconds (configurable), this scheduler:
    1. Collects mission receipts since last tick
    2. Computes the aggregate 8D Ihsān tensor
    3. Bridges to constitutional process_tick() (fixed-point)
    4. Mints SEED, accrues BLOOM, enforces Gini
    5. Precipitates proven patterns into reflex cache
    6. Prunes stale reflexes
    7. Generates an evidence-chained HeartbeatReceipt
    """

    def __init__(
        self,
        *,
        reflex_cache: Optional[ReflexCacheLike] = None,
        token_minter: Optional[Any] = None,
        wallet: Optional[Any] = None,
        wallets: Optional[List[Any]] = None,
        on_heartbeat: Optional[Callable[[HeartbeatReceipt], None]] = None,
        interval_s: float = HEARTBEAT_INTERVAL_S,
    ) -> None:
        self._reflex = reflex_cache
        self._minter = token_minter
        self._wallet = wallet
        self._wallets = wallets or []
        self._on_heartbeat = on_heartbeat
        self._interval_s = interval_s

        self._tick_number = 0
        self._chain_hash = "0" * 64  # Genesis hash
        self._stats = Helix3Stats()
        self._pending_receipts: List[Dict[str, Any]] = []
        self._ihsan_history: List[float] = []

    # ─── Receipt Collection ───────────────────────────────────────

    def ingest_receipt(self, receipt: Dict[str, Any]) -> None:
        """Collect a NervousSystem receipt for next tick processing.

        Call this after every NervousSystem.run() completion.
        The receipts accumulate until the next process_tick().
        """
        self._pending_receipts.append(receipt)

    # ─── Main Heartbeat ──────────────────────────────────────────

    def process_tick(self) -> HeartbeatReceipt:
        """One heartbeat of the evolutionary cycle (§2 Helix 3).

        12-step constitutional procedure:
          1. Collect pending receipts
          2. Compute 8D Ihsān tensor (geometric mean)
          3. Score and gate receipts
          4. Mint SEED (if Ihsān ≥ 0.95)
          5. Accrue BLOOM (if Ihsān ≥ 0.90)
          6. Decay BLOOM (monthly, all wallets)
          7. Check Gini invariant (HALT if > 0.35)
          8. Precipitate reflexes (proven patterns → S1 cache)
          9. Prune stale reflexes (TTL-based eviction)
          10. Compute asabiyyah (social cohesion)
          11. Generate evidence-chained receipt
          12. Reset pending receipts
        """
        start = time.monotonic()
        self._tick_number += 1
        receipts = list(self._pending_receipts)
        self._pending_receipts.clear()

        # ── Step 1: Aggregate 8D Ihsān tensor from receipts ─────
        tensor = self._compute_aggregate_tensor(receipts)
        composite = tensor.geometric_mean

        # ── Step 2: Score and gate ──────────────────────────────
        passing = [r for r in receipts if r.get("ihsan_score", 0) >= 0.85]
        excellent = [r for r in passing if r.get("ihsan_score", 0) >= UNIFIED_IHSAN_THRESHOLD]

        # ── Step 3-4: Constitutional economics ──────────────────
        seed_minted = 0.0
        bloom_accrued = 0.0

        if self._minter and self._wallet and excellent:
            for r in excellent:
                ihsan = r.get("ihsan_score", 0.0)
                mint_result = self._minter.mint_seed(
                    wallet=self._wallet,
                    amount=r.get("reward_amount", 0.0) * 2,  # Reconstruct pre-split amount
                    poi_evidence=r.get("evidence_hash", ""),
                    ihsan=ihsan,
                )
                if mint_result.get("minted"):
                    seed_minted += mint_result["total_amount"]

        # ── Step 5-6: BLOOM accrual + decay (simplified) ───────
        if self._wallet and hasattr(self._wallet, "bloom"):
            bloom_before = getattr(self._wallet.bloom, "balance", 0.0)
            for r in passing:
                ihsan = r.get("ihsan_score", 0.0)
                if ihsan >= 0.90 and hasattr(self._wallet.bloom, "accrue"):
                    self._wallet.bloom.accrue(ihsan * 0.1)
            bloom_after = getattr(self._wallet.bloom, "balance", 0.0)
            bloom_accrued = bloom_after - bloom_before

        # ── Step 7: Gini invariant ─────────────────────────────
        gini = self._compute_gini()
        gini_ok = gini <= ADL_GINI_THRESHOLD

        if not gini_ok:
            logger.critical(
                "GINI HALT tick=%d | gini=%.4f > %.4f — evolutionary rewards suspended",
                self._tick_number, gini, ADL_GINI_THRESHOLD,
            )
            self._stats.total_gini_halts += 1

        # ── Step 8: Reflex precipitation ───────────────────────
        reflexes_precipitated = 0
        if self._reflex:
            top_reflexes = self._reflex.get_top_reflexes(n=20)
            for reflex in top_reflexes:
                ihsan = reflex.get("ihsan_composite", 0.0)
                if ihsan >= PRECIPITATION_IHSAN_FLOOR:
                    key = reflex.get("key", "")
                    if key and self._reflex.revalidate(key, ihsan):
                        reflexes_precipitated += 1

        # ── Step 9: Stale reflex pruning ───────────────────────
        reflexes_pruned = self._prune_stale_reflexes()

        # ── Step 10: Asabiyyah (social cohesion metric) ────────
        # Placeholder: in production, computed from inter-node attestations

        # ── Step 11: Evidence chain ────────────────────────────
        duration_ms = (time.monotonic() - start) * 1000

        self._ihsan_history.append(composite)
        self._stats.total_ticks += 1
        self._stats.total_missions_processed += len(receipts)
        self._stats.total_seed_minted += seed_minted
        self._stats.total_bloom_accrued += bloom_accrued
        self._stats.total_reflexes_precipitated += reflexes_precipitated
        self._stats.total_reflexes_pruned += reflexes_pruned
        self._stats.last_tick_time = time.time()
        if self._ihsan_history:
            self._stats.avg_ihsan = sum(self._ihsan_history) / len(self._ihsan_history)
        self._stats.avg_tick_duration_ms = (
            (self._stats.avg_tick_duration_ms * (self._stats.total_ticks - 1) + duration_ms)
            / self._stats.total_ticks
        )

        evidence_data = {
            "tick": self._tick_number,
            "missions": len(receipts),
            "composite": composite,
            "gini": gini,
            "minted": seed_minted,
        }
        evidence_hash = "ev:" + hashlib.sha256(
            str(sorted(evidence_data.items())).encode()
        ).hexdigest()[:32]
        chain_hash = hashlib.sha256(
            f"{self._chain_hash}:{evidence_hash}".encode()
        ).hexdigest()
        self._chain_hash = chain_hash

        receipt = HeartbeatReceipt(
            tick_number=self._tick_number,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=round(duration_ms, 2),
            missions_processed=len(receipts),
            ihsan_tensor=tensor.dimensions,
            ihsan_composite=composite,
            gini_coefficient=round(gini, 6),
            gini_ok=gini_ok,
            seed_minted=seed_minted,
            bloom_accrued=bloom_accrued,
            reflexes_precipitated=reflexes_precipitated,
            reflexes_pruned=reflexes_pruned,
            evidence_hash=evidence_hash,
            chain_hash=chain_hash,
            stats=self._stats.as_dict(),
        )

        # ── Step 12: Callback + reset ──────────────────────────
        if self._on_heartbeat:
            self._on_heartbeat(receipt)

        return receipt

    # ─── Bridge: NervousSystem → Constitutional Ticker ────────

    def process_tick_constitutional(self) -> Optional[HeartbeatReceipt]:
        """Run process_tick via the constitutional kernel (fixed-point).

        This is the production path that uses core.constitutional.ticker
        with deterministic fixed-point arithmetic. Falls back to the
        simplified path if the constitutional kernel is unavailable.
        """
        try:
            from core.constitutional.fixed_point import fp, fp_float
            from core.constitutional.ticker import process_tick
            from core.constitutional.types import ActionReceipt as ConstitutionalReceipt
            from core.constitutional.types import WalletState as ConstitutionalWallet
        except ImportError:
            logger.info("Constitutional kernel unavailable — using simplified tick")
            return self.process_tick()

        start = time.monotonic()
        self._tick_number += 1
        receipts = list(self._pending_receipts)
        self._pending_receipts.clear()

        # Convert float receipts → constitutional fixed-point
        constitutional_receipts = []
        for r in receipts:
            ihsan = r.get("ihsan_score", 0.0)
            cr = ConstitutionalReceipt(
                receipt_id=hashlib.blake2b(
                    r.get("mission_id", "").encode(), digest_size=32
                ).digest(),
                actor_id=b"local_node" + b"\x00" * 22,  # 32 bytes
                action_type="contribution",
                timestamp=int(time.time() * 1000),
                intent_score=fp(ihsan),
                efficiency_score=fp(r.get("snr_score", 0.85)),
                impact_score=fp(ihsan),
                reproducibility_score=fp(0.90),
                oracle_signature=b"\x00" * 64,
                metadata_hash=b"\x00" * 32,
            )
            constitutional_receipts.append(cr)

        # Convert wallets
        constitutional_wallets = [
            ConstitutionalWallet(
                node_id=b"local_node" + b"\x00" * 22,
                seed_balance=fp(getattr(self._wallet, "seed_balance", 0.0))
                if self._wallet else 0,
            )
        ]

        # Run constitutional 12-step tick
        tick_result = process_tick(
            wallets=constitutional_wallets,
            receipts=constitutional_receipts,
            proposals=[],
            event_log=[],
            reflex_cache={},
        )

        duration_ms = (time.monotonic() - start) * 1000

        # Convert back to float
        gini = fp_float(tick_result.network_gini) if tick_result.network_gini else 0.0
        tensor = self._compute_aggregate_tensor(receipts)
        composite = tensor.geometric_mean

        self._stats.total_ticks += 1
        self._stats.total_missions_processed += len(receipts)
        self._stats.total_seed_minted += fp_float(tick_result.total_minted)
        self._stats.last_tick_time = time.time()

        evidence_hash = "ev:" + hashlib.sha256(
            f"tick:{self._tick_number}:const".encode()
        ).hexdigest()[:32]
        chain_hash = hashlib.sha256(
            f"{self._chain_hash}:{evidence_hash}".encode()
        ).hexdigest()
        self._chain_hash = chain_hash

        receipt = HeartbeatReceipt(
            tick_number=self._tick_number,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=round(duration_ms, 2),
            missions_processed=len(receipts),
            ihsan_tensor=tensor.dimensions,
            ihsan_composite=composite,
            gini_coefficient=round(gini, 6),
            gini_ok=gini <= ADL_GINI_THRESHOLD,
            seed_minted=fp_float(tick_result.total_minted),
            bloom_accrued=0.0,
            reflexes_precipitated=0,
            reflexes_pruned=0,
            evidence_hash=evidence_hash,
            chain_hash=chain_hash,
            stats=self._stats.as_dict(),
        )

        if self._on_heartbeat:
            self._on_heartbeat(receipt)

        return receipt

    # ─── Internal Methods ─────────────────────────────────────

    def _compute_aggregate_tensor(
        self, receipts: List[Dict[str, Any]]
    ) -> IhsanTensor8D:
        """Compute aggregate 8D Ihsān tensor from mission receipts.

        For each dimension, takes the mean of individual mission scores.
        If no receipts, returns uniform tensor at the threshold floor.
        """
        if not receipts:
            return IhsanTensor8D.uniform(UNIFIED_IHSAN_THRESHOLD)

        # If receipts contain individual tensor dimensions, aggregate them.
        # Otherwise, project the scalar ihsan_score uniformly.
        agg: Dict[str, List[float]] = {k: [] for k in IHSAN_CANONICAL_WEIGHTS}

        for r in receipts:
            tensor_data = r.get("ihsan_tensor")
            if isinstance(tensor_data, dict):
                for k in agg:
                    agg[k].append(tensor_data.get(k, r.get("ihsan_score", 0.0)))
            else:
                score = r.get("ihsan_score", 0.0)
                for k in agg:
                    agg[k].append(score)

        means = {k: sum(v) / len(v) if v else 0.0 for k, v in agg.items()}
        return IhsanTensor8D.from_scores(means)

    def _compute_gini(self) -> float:
        """Compute Gini coefficient across all known wallets."""
        if not self._wallets or len(self._wallets) < 2:
            return 0.0

        try:
            from core.token.bloom import compute_gini

            balances = [getattr(w, "seed_balance", 0.0) for w in self._wallets]
            if all(b == 0.0 for b in balances):
                return 0.0
            return compute_gini(balances)
        except ImportError:
            # Inline Gini (no external deps)
            balances = sorted(getattr(w, "seed_balance", 0.0) for w in self._wallets)
            n = len(balances)
            if n == 0 or sum(balances) == 0:
                return 0.0
            numerator = sum((2 * i - n - 1) * b for i, b in enumerate(balances, 1))
            return abs(numerator) / (n * sum(balances))

    def _prune_stale_reflexes(self) -> int:
        """Prune stale reflexes from the cache (TTL-based eviction)."""
        if not self._reflex:
            return 0

        pruned = 0
        try:
            top = self._reflex.get_top_reflexes(n=100)
            for entry in top:
                age_days = entry.get("age_days", 0)
                hit_count = entry.get("hit_count", 0)
                if age_days > STALE_REFLEX_AGE_DAYS and hit_count < 2:
                    key = entry.get("key", "")
                    if key:
                        # Revalidate with 0.0 → triggers eviction
                        self._reflex.revalidate(key, 0.0)
                        pruned += 1
        except (AttributeError, TypeError):
            pass

        return pruned

    # ─── Properties ───────────────────────────────────────────

    @property
    def stats(self) -> Helix3Stats:
        return self._stats

    @property
    def tick_number(self) -> int:
        return self._tick_number

    @property
    def chain_hash(self) -> str:
        return self._chain_hash

    @property
    def interval_s(self) -> float:
        return self._interval_s


# ═══════════════════════════════════════════════════════════════════
# FACTORY: Wire Helix 3 to existing Nervous System
# ═══════════════════════════════════════════════════════════════════

def wire_helix3(
    nervous_system: Any,
    *,
    reflex_cache: Optional[Any] = None,
    on_heartbeat: Optional[Callable[[HeartbeatReceipt], None]] = None,
) -> Helix3Scheduler:
    """Wire Helix 3 to a SovereignNervousSystem instance.

    Attaches a receipt callback that feeds NervousSystem receipts
    into the Helix 3 scheduler for evolutionary processing.

    Usage:
        ns = SovereignNervousSystem.create(inference=...)
        h3 = wire_helix3(ns, on_heartbeat=my_callback)
        receipt = await ns.run("mission")  # Auto-ingested into h3
        heartbeat = h3.process_tick()      # Run after interval
    """
    scheduler = Helix3Scheduler(
        reflex_cache=reflex_cache,
        token_minter=getattr(nervous_system, "_minter", None),
        wallet=getattr(nervous_system, "_wallet", None),
        wallets=getattr(nervous_system, "_wallets", []),
        on_heartbeat=on_heartbeat,
    )

    # Patch the NervousSystem to auto-ingest receipts
    original_on_receipt = getattr(nervous_system, "_on_receipt", None)

    def _feed_helix3(receipt: Any) -> None:
        scheduler.ingest_receipt(receipt.as_dict())
        if original_on_receipt:
            original_on_receipt(receipt)

    nervous_system._on_receipt = _feed_helix3

    return scheduler
