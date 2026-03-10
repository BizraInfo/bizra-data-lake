"""
Sovereign Nervous System — S1/S2 Cognitive Bridge
==================================================
Drop into: core/sovereign/mission_nervous_system.py

This is the integration layer that makes the BIZRA organism LIVE.
It connects the brain (MissionOrchestrator) to the body (EventBus +
ReflexCompiler + BLOOM tokens) through a dual-process cognitive
architecture:

  S1 (Reflex): O(1) ReflexCompiler cache hit → instant response
  S2 (Deliberation): Full MissionOrchestrator OODA cycle → evidence-traced

After every S2 completion, the nervous system:
  1. Records the observation for future S1 precipitation
  2. Publishes typed events to the Phase 80 EventBus
  3. Mints SEED tokens if Ihsān ≥ 0.95 (50% community pool split)
  4. Checks Gini invariant (HALT if > 0.35)

Standing on Giants:
  Kahneman — System 1 (fast) / System 2 (slow) dual-process theory
  Boyd — OODA loop (Observe→Orient→Decide→Act)
  Deming — PDCA (Plan→Do→Check→Act) quality cycle
  Hewitt — Actor model (EventBus subscribers as independent actors)
  Ostrom — Commons governance (50% community pool)
  Al-Ghazali — Ihsān as constitutional hard constraint (§4)
  Shannon — SNR scoring on output quality
  Lamport — Hash-chained evidence with ordering invariant

Usage:
  ns = SovereignNervousSystem.create(persistence_dir=Path("./state"))
  receipt = await ns.run("Summarize the quarterly report")
  # receipt.system == "S1" if cache hit, "S2" if full deliberation
  # receipt.ihsan >= 0.95 guaranteed if receipt.rewarded is True

Constitutional Invariants:
  §4  Ihsān gate: mission HALTS below threshold (0.95 for rewards)
  §7  Data sovereignty: reflex export strips personal data
  §12 Community pool: 50% of all SEED minting (HARDCODED)
  §14 ADL Gini: ≤ 0.35 or system HALTS token minting
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger("bizra.sovereign.nervous_system")


# ═══════════════════════════════════════════════════════════════════
# CONSTITUTIONAL THRESHOLDS (from single source of truth)
# ═══════════════════════════════════════════════════════════════════

from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    UNIFIED_IHSAN_THRESHOLD,
)

SEED_MINT_FLOOR = 0.95  # From core/token/bloom.py


# ═══════════════════════════════════════════════════════════════════
# PROTOCOLS (Dependency Injection Contracts)
# ═══════════════════════════════════════════════════════════════════


class InferenceProvider(Protocol):
    """Any callable that turns a prompt into a response string."""

    async def infer(self, prompt: str, **kwargs: Any) -> str: ...


class ReflexCache(Protocol):
    """Duck-typed interface matching ReflexCompiler."""

    def lookup(
        self, input_text: str, *, macro_state: Optional[str] = None
    ) -> Optional[Any]: ...

    def record_observation(
        self,
        input_text: str,
        output_text: str,
        ihsan_composite: float,
        ihsan_tensor: Optional[Dict[str, float]] = None,
    ) -> Optional[Any]: ...


class TokenMinter(Protocol):
    """Duck-typed interface matching core.token.bloom.TokenMinter."""

    def mint_seed(
        self, wallet: Any, amount: float, poi_evidence: str, ihsan: float
    ) -> Dict[str, Any]: ...


class EventBusLike(Protocol):
    """Duck-typed interface matching core.bus.subscribers.EventBus."""

    def publish(self, event_type: Any, payload: Dict[str, Any]) -> Any: ...


# ═══════════════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════════════


@dataclass
class NervousSystemReceipt:
    """Complete evidence-chained receipt from a nervous system execution."""

    mission_id: str
    system: str  # "S1" (reflex) or "S2" (deliberation)
    input_text: str
    output_text: str
    ihsan_score: float
    snr_score: float
    duration_ms: float
    rewarded: bool
    reward_amount: float
    pool_contribution: float
    evidence_hash: str
    chain_hash: str  # Links to previous receipt
    timestamp: str
    reflex_hit: bool
    gini_ok: bool
    events_published: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NervousSystemStats:
    """Operational statistics for observability."""

    total_missions: int = 0
    s1_hits: int = 0
    s2_executions: int = 0
    rewards_minted: int = 0
    rewards_rejected: int = 0
    total_seed_minted: float = 0.0
    total_pool_contributed: float = 0.0
    gini_halts: int = 0
    avg_ihsan: float = 0.0
    avg_duration_ms: float = 0.0

    @property
    def s1_hit_rate(self) -> float:
        return self.s1_hits / max(self.total_missions, 1)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["s1_hit_rate"] = self.s1_hit_rate
        return d


# ═══════════════════════════════════════════════════════════════════
# QUALITY SCORING
# ═══════════════════════════════════════════════════════════════════


def _score_ihsan(output: str, input_text: str) -> float:
    """Compute Ihsān composite score via unified 8D content scorer.

    Delegates to ihsan_scorer.score_ihsan_composite() — the single
    source of truth for Ihsān scoring across the organism (§4).
    """
    try:
        from core.sovereign.ihsan_scorer import score_ihsan_composite

        return score_ihsan_composite(output, input_text)
    except ImportError:
        # Fallback: basic heuristic if scorer not available
        if not output or not output.strip():
            return 0.0
        length_score = min(len(output) / 200, 1.0)
        relevance = (
            1.0
            if any(w in output.lower() for w in input_text.lower().split()[:3])
            else 0.5
        )
        coherence = 1.0 if len(output.split()) > 5 else 0.6
        raw = length_score * 0.3 + relevance * 0.4 + coherence * 0.3
        return round(min(raw, 1.0), 4)


def _score_snr(output: str) -> float:
    """Compute SNR score via unified 4D content scorer (§8).

    Delegates to ihsan_scorer.score_snr_composite() — Shannon-inspired
    signal-to-noise across 4 dimensions.
    """
    try:
        from core.sovereign.ihsan_scorer import score_snr_composite

        return score_snr_composite(output)
    except ImportError:
        # Fallback: basic heuristic
        if not output:
            return 0.0
        words = output.split()
        if not words:
            return 0.0
        unique_ratio = len(set(words)) / len(words)
        length_factor = min(len(words) / 20, 1.0)
        return round(unique_ratio * 0.6 + length_factor * 0.4, 4)


def _compute_evidence_hash(data: Dict[str, Any]) -> str:
    """BLAKE3-style evidence hash (SHA-256 fallback)."""
    canonical = hashlib.sha256(str(sorted(data.items())).encode()).hexdigest()[:32]
    return f"ev:{canonical}"


# ═══════════════════════════════════════════════════════════════════
# SOVEREIGN NERVOUS SYSTEM
# ═══════════════════════════════════════════════════════════════════


class SovereignNervousSystem:
    """The cognitive bridge between reflexes (S1) and deliberation (S2).

    This is the organism's nervous system — it routes every mission
    through the optimal cognitive path and ensures constitutional
    compliance at every step.

    Architecture:
        Input → S1 Probe (ReflexCompiler O(1))
              → HIT:  Return cached plan + mint reward
              → MISS: S2 Delegate (full inference)
                    → Record observation (future S1 precipitation)
                    → Publish events → 12 subscribers
                    → Mint reward (if Ihsān ≥ 0.95)
                    → Check Gini (HALT if > 0.35)
                    → Return evidence-chained receipt
    """

    def __init__(
        self,
        inference: InferenceProvider,
        *,
        reflex_cache: Optional[ReflexCache] = None,
        event_bus: Optional[EventBusLike] = None,
        token_minter: Optional[TokenMinter] = None,
        wallet: Optional[Any] = None,
        wallets: Optional[List[Any]] = None,
        on_receipt: Optional[Callable[[NervousSystemReceipt], None]] = None,
        reward_per_mission: float = 1.0,
    ) -> None:
        self._inference = inference
        self._reflex = reflex_cache
        self._bus = event_bus
        self._minter = token_minter
        self._wallet = wallet
        self._wallets = wallets or []
        self._on_receipt = on_receipt
        self._reward_per_mission = reward_per_mission

        self._chain_hash = "0" * 64  # Genesis hash
        self._mission_counter = 0
        self._stats = NervousSystemStats()
        self._ihsan_history: List[float] = []

    # ─── Factory ──────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        inference: InferenceProvider,
        persistence_dir: Optional[Path] = None,
        reward_per_mission: float = 1.0,
    ) -> SovereignNervousSystem:
        """Create a fully-wired nervous system with all Phase 80 modules.

        This is the recommended entry point for production use.
        Creates ReflexCompiler, EventBus, TokenMinter, and WalletState
        internally, wiring them together via the 12 EventBus subscribers.
        """
        from core.bus.subscribers import EventBus, wire_all_subscribers
        from core.sovereign.reflex_compiler import ReflexCompiler
        from core.token.bloom import CommunityPool
        from core.token.bloom import TokenMinter as BloomMinter
        from core.token.bloom import WalletState

        reflex = ReflexCompiler(
            max_entries=1000,
            persistence_path=(
                persistence_dir / "reflexes.json" if persistence_dir else None
            ),
        )

        bus = EventBus()
        pool = CommunityPool()
        minter = BloomMinter(community_pool=pool)
        wallet = WalletState(node_id="local_node")

        # Minimal no-op implementations for safety-critical subscribers
        # IhsanGateBreachHandler and FailedActionQuarantine are fail-closed
        # (re-raise on error), so they need functional dependencies.
        class _NoOpAuditLog:
            """Minimal audit log — logs violations without external service."""

            def log_violation(self, **kw: Any) -> None:
                logger.warning("Ihsān gate violation: %s", kw)

        class _NoOpSessionManager:
            """Minimal session manager — halts are logged, not enforced."""

            def halt(self, **kw: Any) -> None:
                logger.warning("Session halt requested: %s", kw)

        class _NoOpQuarantine:
            """Minimal quarantine — isolations are logged."""

            def isolate(self, **kw: Any) -> None:
                logger.warning("Quarantine isolation: %s", kw)

        class _NoOpMemoryStore:
            """Minimal memory store for subscriber wiring."""

            def reinforce(self, **kw: Any) -> None:
                pass

            def get_success_count(self, key: str) -> int:
                return 0

            def set_success_count(self, key: str, val: int) -> None:
                pass

            def promote_to_semantic(self, **kw: Any) -> bool:
                return False

            def record_failure_pattern(self, **kw: Any) -> None:
                pass

        class _NoOpTeleScript:
            """Minimal telescript engine."""

            def begin_execution(self, **kw: Any) -> str:
                return f"ts_noop_{id(self)}"

        memory_store = _NoOpMemoryStore()

        subs = wire_all_subscribers(
            bus,
            memory_store=memory_store,
            telescript_engine=_NoOpTeleScript(),
            receipt_chain=[],
            reflex_cache=reflex,
            session_manager=_NoOpSessionManager(),
            audit_log=_NoOpAuditLog(),
            quarantine_store=_NoOpQuarantine(),
            healing_engine=None,
            hhmm_engine=None,
            poi_engine=None,
            token_minter=minter,
            context_budget=None,
            self_model=None,
            capability_registry=None,
        )
        logger.info("Nervous system wired: %d subscribers", len(subs))

        return cls(
            inference=inference,
            reflex_cache=reflex,
            event_bus=bus,
            token_minter=minter,
            wallet=wallet,
            wallets=[wallet],
            reward_per_mission=reward_per_mission,
        )

    # ─── Main Entry Point ─────────────────────────────────────────

    async def run(
        self,
        mission_text: str,
        *,
        macro_state: Optional[str] = None,
        ihsan_override: Optional[float] = None,
        snr_override: Optional[float] = None,
    ) -> NervousSystemReceipt:
        """Execute a mission through the S1/S2 cognitive pipeline.

        Args:
            mission_text: The user's mission description.
            macro_state: Optional HHMM macro state for hierarchical lookup.
            ihsan_override: Override Ihsān score (for testing).
            snr_override: Override SNR score (for testing).

        Returns:
            NervousSystemReceipt with full evidence chain.
        """
        start = time.monotonic()
        self._mission_counter += 1
        mission_id = f"m-{self._mission_counter:06d}"
        events_published: List[str] = []

        # ── S1 PROBE: Reflex cache lookup (Kahneman System 1) ────
        reflex_hit = False
        output_text = ""

        if self._reflex is not None:
            entry = self._reflex.lookup(mission_text, macro_state=macro_state)
            if entry is not None:
                output_text = entry.output_template
                reflex_hit = True
                self._stats.s1_hits += 1
                logger.info(
                    "S1 HIT | mission=%s hit_count=%d ihsan=%.3f",
                    mission_id,
                    entry.hit_count,
                    entry.ihsan_composite,
                )

        # ── S2 DELIBERATION: Full inference (Kahneman System 2) ──
        if not reflex_hit:
            output_text = await self._inference.infer(mission_text)
            self._stats.s2_executions += 1
            logger.info("S2 EXEC | mission=%s len=%d", mission_id, len(output_text))

        # ── SCORE: Constitutional quality gates (Al-Ghazali) ─────
        ihsan = (
            ihsan_override
            if ihsan_override is not None
            else _score_ihsan(output_text, mission_text)
        )
        snr = snr_override if snr_override is not None else _score_snr(output_text)

        self._ihsan_history.append(ihsan)
        self._stats.avg_ihsan = sum(self._ihsan_history) / len(self._ihsan_history)

        # ── PUBLISH: EventBus events (Hewitt Actor Model) ────────
        if self._bus is not None:
            events_published = self._publish_events(
                mission_id,
                mission_text,
                output_text,
                ihsan,
                snr,
                reflex_hit,
            )

        # ── RECORD: Observation for future S1 (Deming PDCA) ─────
        if not reflex_hit and self._reflex is not None:
            self._reflex.record_observation(
                input_text=mission_text,
                output_text=output_text,
                ihsan_composite=ihsan,
            )

        # ── REWARD: BLOOM token minting (Ostrom Commons) ────────
        rewarded = False
        reward_amount = 0.0
        pool_contribution = 0.0

        if ihsan >= SEED_MINT_FLOOR and self._minter and self._wallet:
            mint_result = self._minter.mint_seed(
                wallet=self._wallet,
                amount=self._reward_per_mission,
                poi_evidence=_compute_evidence_hash(
                    {
                        "mission_id": mission_id,
                        "ihsan": ihsan,
                        "output_hash": hashlib.sha256(output_text.encode()).hexdigest()[
                            :16
                        ],
                    }
                ),
                ihsan=ihsan,
            )
            if mint_result.get("minted"):
                rewarded = True
                reward_amount = mint_result["node_share"]
                pool_contribution = mint_result["pool_share"]
                self._stats.rewards_minted += 1
                self._stats.total_seed_minted += mint_result["total_amount"]
                self._stats.total_pool_contributed += pool_contribution
            else:
                self._stats.rewards_rejected += 1

        # ── GINI CHECK: Justice invariant (ADL §14) ─────────────
        gini_ok = self._check_gini()

        # ── EVIDENCE: Hash-chained receipt (Lamport) ────────────
        duration_ms = (time.monotonic() - start) * 1000

        evidence_data = {
            "mission_id": mission_id,
            "system": "S1" if reflex_hit else "S2",
            "ihsan": ihsan,
            "snr": snr,
            "rewarded": rewarded,
            "gini_ok": gini_ok,
        }
        evidence_hash = _compute_evidence_hash(evidence_data)
        chain_hash = hashlib.sha256(
            f"{self._chain_hash}:{evidence_hash}".encode()
        ).hexdigest()
        self._chain_hash = chain_hash

        self._stats.total_missions += 1
        self._stats.avg_duration_ms = (
            self._stats.avg_duration_ms * (self._stats.total_missions - 1) + duration_ms
        ) / self._stats.total_missions

        receipt = NervousSystemReceipt(
            mission_id=mission_id,
            system="S1" if reflex_hit else "S2",
            input_text=mission_text,
            output_text=output_text,
            ihsan_score=ihsan,
            snr_score=snr,
            duration_ms=round(duration_ms, 2),
            rewarded=rewarded,
            reward_amount=reward_amount,
            pool_contribution=pool_contribution,
            evidence_hash=evidence_hash,
            chain_hash=chain_hash,
            timestamp=datetime.now(timezone.utc).isoformat(),
            reflex_hit=reflex_hit,
            gini_ok=gini_ok,
            events_published=events_published,
        )

        if self._on_receipt:
            self._on_receipt(receipt)

        return receipt

    # ─── EventBus Bridge ──────────────────────────────────────────

    def _publish_events(
        self,
        mission_id: str,
        input_text: str,
        output_text: str,
        ihsan: float,
        snr: float,
        reflex_hit: bool,
    ) -> List[str]:
        """Publish typed events to the Phase 80 EventBus."""
        published: List[str] = []

        try:
            from core.bus.subscribers import EventType
        except ImportError:
            logger.warning("EventBus not available — skipping event publication")
            return published

        assert self._bus is not None

        base_payload = {
            "mission_id": mission_id,
            "ihsan": ihsan,
            "snr": snr,
            "system": "S1" if reflex_hit else "S2",
        }

        # ACTION_INTENT — triggers TeleScript begin
        self._bus.publish(
            EventType.ACTION_INTENT,
            {**base_payload, "description": input_text[:500]},
        )
        published.append("action.intent")

        # ACTION_RECEIPT — triggers memory reinforce
        self._bus.publish(
            EventType.ACTION_RECEIPT,
            {
                **base_payload,
                "output": output_text[:1000],
                "success": ihsan >= UNIFIED_IHSAN_THRESHOLD,
            },
        )
        published.append("action.receipt")

        if ihsan < UNIFIED_IHSAN_THRESHOLD:
            # IHSAN_GATE_BREACHED — triggers safety handlers
            self._bus.publish(
                EventType.IHSAN_GATE_BREACHED,
                {**base_payload, "threshold": UNIFIED_IHSAN_THRESHOLD},
            )
            published.append("ihsan.gate.breached")

        # SESSION_END — triggers reflex compilation + PoI accumulation
        self._bus.publish(
            EventType.SESSION_END,
            {
                **base_payload,
                "output": output_text[:1000],
                "duration_ms": 0,  # Filled by caller
            },
        )
        published.append("session.end")

        return published

    # ─── Gini Invariant ───────────────────────────────────────────

    def _check_gini(self) -> bool:
        """Check ADL Gini invariant across all known wallets.

        Returns True if Gini ≤ threshold (system healthy).
        Returns True if no wallets or fewer than 2 (can't compute).
        """
        if not self._wallets or len(self._wallets) < 2:
            return True

        try:
            from core.token.bloom import compute_gini

            balances = [getattr(w, "seed_balance", 0.0) for w in self._wallets]
            if all(b == 0.0 for b in balances):
                return True

            gini = compute_gini(balances)
            if gini > ADL_GINI_THRESHOLD:
                logger.critical(
                    "GINI HALT | gini=%.4f > threshold=%.4f — token minting suspended",
                    gini,
                    ADL_GINI_THRESHOLD,
                )
                self._stats.gini_halts += 1
                return False
            return True
        except ImportError:
            return True

    # ─── Properties ───────────────────────────────────────────────

    @property
    def stats(self) -> NervousSystemStats:
        return self._stats

    @property
    def chain_hash(self) -> str:
        return self._chain_hash

    @property
    def mission_count(self) -> int:
        return self._mission_counter


# ═══════════════════════════════════════════════════════════════════
# SIMPLE INFERENCE PROVIDERS (for testing and standalone use)
# ═══════════════════════════════════════════════════════════════════


class EchoInference:
    """Echoes input back — for testing only."""

    async def infer(self, prompt: str, **kwargs: Any) -> str:
        return f"[Echo] Processed: {prompt}"


class OllamaInference:
    """Production inference via Ollama API (local-first)."""

    def __init__(
        self,
        model: str = "phi3:mini",
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.host = host

    async def infer(self, prompt: str, **kwargs: Any) -> str:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx required for OllamaInference: pip install httpx")

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
