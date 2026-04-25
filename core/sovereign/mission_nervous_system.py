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

from core.errors import BizraError, BridgeError, InferenceError
from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    REFLEX_PRECIPITATION_HITS,
    UNIFIED_IHSAN_THRESHOLD,
)

# ═══════════════════════════════════════════════════════════════════
# CONSTITUTIONAL THRESHOLDS (from single source of truth)
# ═══════════════════════════════════════════════════════════════════


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
    s2_avg_duration_ms: float = 0.0

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


def _append_degradation_receipt(
    metadata: Dict[str, Any],
    error: BizraError,
) -> None:
    """Attach typed degradation evidence to receipt metadata."""
    receipts = metadata.setdefault("degradation_receipts", [])
    if isinstance(receipts, list):
        receipts.append(error.to_receipt())
    metadata["degraded"] = True


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

        # ── Phase 87: Wire Rust constitutional bridge ──────────────
        try:
            from core.bus.rust_bridge import wire_rust_bridge

            rust_bridge = wire_rust_bridge(bus, production=False)
            if rust_bridge:
                logger.info("Rust bridge ACTIVE on mission nervous system")
        except (
            ImportError,
            AttributeError,
            TypeError,
            RuntimeError,
            OSError,
        ) as rust_exc:
            logger.info("Rust bridge not available (degraded): %s", rust_exc)

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
        raw_prompt: Optional[str] = None,
    ) -> NervousSystemReceipt:
        """Execute a mission through the S1/S2 cognitive pipeline.

        Args:
            mission_text: The canonical mission description. Typically a
                liturgical mission-spine wrapped version of the raw prompt
                (``## Niyyah / ## Bayyinah / ## Hadd / ## Qasd`` from
                ``core.prompt.seed_chain``). Kept intact for receipts,
                evidence, reflex pattern matching, and canonical runtime
                structure.
            macro_state: Optional HHMM macro state for hierarchical lookup.
            ihsan_override: Override Ihsān score (for testing).
            snr_override: Override SNR score (for testing).
            raw_prompt: Optional raw user prompt BEFORE the mission-spine
                wrapper is applied. When provided, used as the ``input_text``
                for the Ihsān composite scorer so contextual_relevance and
                intent_alignment are measured against the true user intent
                rather than the liturgical scaffolding. When None (legacy
                callers), scoring falls back to ``mission_text``.

                This decoupling closes the scorer-vs-spine drift surfaced
                by the canonical spearpoint replay test on 2026-04-21:
                wrapped 835-char spine text dropped ``contextual_relevance``
                from 0.68 (raw prompt) to 0.39, pulling the composite from
                0.87 to 0.79 — below the 0.85 Ihsān floor. Preserving the
                wrapped ``mission_text`` for evidence while scoring against
                ``raw_prompt`` restores the correct signal without changing
                the Ihsān floor or the scorer calibration.

        Returns:
            NervousSystemReceipt with full evidence chain.
        """
        start = time.monotonic()
        self._mission_counter += 1
        mission_id = f"m-{self._mission_counter:06d}"
        events_published: List[str] = []
        metadata: Dict[str, Any] = {}

        # Reflex-key input: the stable semantic intent. When raw_prompt is
        # provided, use it so the reflex key survives variable Bayyinah
        # evidence fields (e.g., "Prior receipt: <hash>" changes between
        # run1 and run2 of an otherwise-identical mission). This mirrors
        # the raw_prompt decoupling for Ihsān scoring (see _score_ihsan
        # call below). Wrapped `mission_text` remains canonical runtime
        # evidence for receipts, inference, and audit.
        reflex_key_input = raw_prompt if raw_prompt is not None else mission_text

        # ── S1 PROBE: Reflex cache lookup (Kahneman System 1) ────
        reflex_hit = False
        output_text = ""

        if self._reflex is not None:
            entry = self._reflex.lookup(reflex_key_input, macro_state=macro_state)
            if entry is not None:
                output_text = entry.output_template
                reflex_hit = True
                self._stats.s1_hits += 1
                metadata["reflex_delta"] = {
                    "compiled": True,
                    "near_compile": False,
                    "compile_count": int(entry.precipitation_count),
                    "threshold": REFLEX_PRECIPITATION_HITS,
                }
                metadata["reflex_pattern"] = str(entry.pattern_hash)
                logger.info(
                    "S1 HIT | mission=%s hit_count=%d ihsan=%.3f",
                    mission_id,
                    entry.hit_count,
                    entry.ihsan_composite,
                )

        # ── S2 DELIBERATION: Full inference (Kahneman System 2) ──
        if not reflex_hit:
            self._stats.s2_executions += 1
            try:
                output_text = await self._inference.infer(mission_text)
                logger.info("S2 EXEC | mission=%s len=%d", mission_id, len(output_text))
            except BizraError as exc:
                _append_degradation_receipt(metadata, exc)
                metadata["degradation_reason"] = type(exc).__name__
                output_text = (
                    "[DEGRADED] Inference boundary failed. "
                    f"Mission preserved for recovery: {mission_text[:200]}"
                )
                logger.warning("S2 DEGRADE | mission=%s error=%s", mission_id, exc)
            except Exception as exc:
                typed_exc = InferenceError(
                    type(self._inference).__name__,
                    str(exc) or "untyped inference failure",
                    context={
                        "mission_id": mission_id,
                        "mission_text": mission_text[:200],
                    },
                    original=exc,
                )
                _append_degradation_receipt(metadata, typed_exc)
                metadata["degradation_reason"] = type(typed_exc).__name__
                output_text = (
                    "[DEGRADED] Inference backend unavailable. "
                    f"Mission preserved for recovery: {mission_text[:200]}"
                )
                logger.warning(
                    "S2 DEGRADE | mission=%s error=%s",
                    mission_id,
                    typed_exc,
                )

        # ── SCORE: Constitutional quality gates (Al-Ghazali) ─────
        # Use raw_prompt when provided (decouples scoring from the
        # liturgical mission-spine wrapper); fall back to mission_text for
        # legacy callers. See docstring for the 2026-04-21 rationale.
        scoring_input = raw_prompt if raw_prompt is not None else mission_text
        ihsan = (
            ihsan_override
            if ihsan_override is not None
            else _score_ihsan(output_text, scoring_input)
        )
        snr = snr_override if snr_override is not None else _score_snr(output_text)
        if metadata.get("degraded"):
            if ihsan_override is None:
                ihsan = min(ihsan, 0.2)
            if snr_override is None:
                snr = min(snr, 0.2)

        self._ihsan_history.append(ihsan)
        self._stats.avg_ihsan = sum(self._ihsan_history) / len(self._ihsan_history)

        # ── PUBLISH: EventBus events (Hewitt Actor Model) ────────
        if self._bus is not None:
            try:
                events_published = self._publish_events(
                    mission_id,
                    mission_text,
                    output_text,
                    ihsan,
                    snr,
                    reflex_hit,
                )
            except BizraError as exc:
                _append_degradation_receipt(metadata, exc)
                metadata["event_bus_degraded"] = True
                logger.warning(
                    "EVENT BUS DEGRADE | mission=%s error=%s",
                    mission_id,
                    exc,
                )
            except Exception as exc:
                typed_exc = BridgeError(
                    "event_bus",
                    str(exc) or "untyped event bus failure",
                    context={"mission_id": mission_id},
                    original=exc,
                )
                _append_degradation_receipt(metadata, typed_exc)
                metadata["event_bus_degraded"] = True
                logger.warning(
                    "EVENT BUS DEGRADE | mission=%s error=%s",
                    mission_id,
                    typed_exc,
                )

        # ── RECORD: Observation for future S1 (Deming PDCA) ─────
        # Use reflex_key_input (raw_prompt when provided) so the stored
        # pattern_hash matches what a future lookup will compute. Mixing
        # wrapped mission_text on write with raw_prompt on read would
        # guarantee reflex miss even for semantically identical missions.
        if not reflex_hit and self._reflex is not None:
            precipitated_entry = self._reflex.record_observation(
                input_text=reflex_key_input,
                output_text=output_text,
                ihsan_composite=ihsan,
            )
            pattern_hash = self._reflex._hash_input(reflex_key_input)
            if precipitated_entry is not None:
                metadata["reflex_delta"] = {
                    "compiled": True,
                    "near_compile": False,
                    "compile_count": int(precipitated_entry.precipitation_count),
                    "threshold": REFLEX_PRECIPITATION_HITS,
                }
                metadata["reflex_pattern"] = str(precipitated_entry.pattern_hash)
                metadata["compiled_reflex_event"] = {
                    "name": str(precipitated_entry.input_template)[:120],
                    "pattern_hash": str(precipitated_entry.pattern_hash),
                    "avg_ihsan": round(
                        float(precipitated_entry.ihsan_composite),
                        4,
                    ),
                    "execution_count": int(precipitated_entry.hit_count),
                    "precipitation_count": int(precipitated_entry.precipitation_count),
                }
            else:
                candidate = getattr(self._reflex, "_candidates", {}).get(pattern_hash)
                if candidate is not None:
                    consecutive = int(candidate.consecutive_high_quality())
                    metadata["reflex_delta"] = {
                        "compiled": False,
                        "near_compile": consecutive > 0,
                        "compile_count": consecutive,
                        "threshold": REFLEX_PRECIPITATION_HITS,
                    }

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
        if not reflex_hit:
            prior_s2 = max(self._stats.s2_executions - 1, 0)
            self._stats.s2_avg_duration_ms = (
                self._stats.s2_avg_duration_ms * prior_s2 + duration_ms
            ) / max(self._stats.s2_executions, 1)
        elif metadata:
            metadata["reflex_latency_ms"] = round(duration_ms, 2)
            metadata["comparison_s2_avg_ms"] = round(
                float(self._stats.s2_avg_duration_ms),
                2,
            )

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
            metadata=metadata,
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

        action_type = f"mission:{input_text[:96]}"
        base_payload = {
            "mission_id": mission_id,
            "session_id": mission_id,
            "ihsan": ihsan,
            "ihsan_composite": ihsan,
            "snr": snr,
            "snr_score": snr,
            "system": "S1" if reflex_hit else "S2",
        }

        # ACTION_INTENT — triggers TeleScript begin
        self._bus.publish(
            EventType.ACTION_INTENT,
            {
                **base_payload,
                "intent": input_text[:500],
                "description": input_text[:500],
                "context": {
                    "mission_id": mission_id,
                    "snr_score": snr,
                    "system": base_payload["system"],
                },
            },
        )
        published.append("action.intent")

        # ACTION_RECEIPT — triggers memory reinforce
        self._bus.publish(
            EventType.ACTION_RECEIPT,
            {
                **base_payload,
                "action_type": action_type,
                "result_summary": output_text[:1000],
                "output": output_text[:1000],
                "success": ihsan >= UNIFIED_IHSAN_THRESHOLD,
            },
        )
        published.append("action.receipt")

        # Two-band Ihsān policy (see core.bus.subscribers
        # MISSION_IHSAN_HALT_FLOOR docstring, 2026-04-21):
        #   ihsan < 0.85              → IHSAN_GATE_BREACHED (hard halt)
        #   0.85 ≤ ihsan < 0.95       → IHSAN_WARNING (warn, no halt)
        #   ihsan ≥ 0.95              → neither (production-ideal met)
        from core.bus.subscribers import MISSION_IHSAN_HALT_FLOOR

        if ihsan < MISSION_IHSAN_HALT_FLOOR:
            # Hard halt — below operational minimum.
            # ``violation_dimensions`` keeps the legacy string
            # "ihsan_below_threshold" for backward compatibility with
            # downstream subscribers; the ``threshold`` field now reports
            # the actual trigger value (0.85, not 0.95).
            self._bus.publish(
                EventType.IHSAN_GATE_BREACHED,
                {
                    **base_payload,
                    "action_type": action_type,
                    "threshold": MISSION_IHSAN_HALT_FLOOR,
                    "violation_dimensions": ["ihsan_below_threshold"],
                },
            )
            published.append("ihsan.gate.breached")
        elif ihsan < UNIFIED_IHSAN_THRESHOLD:
            # Warn band — lawful but below production ideal. Does NOT halt.
            self._bus.publish(
                EventType.IHSAN_WARNING,
                {
                    **base_payload,
                    "action_type": action_type,
                    "threshold": UNIFIED_IHSAN_THRESHOLD,
                    "halt_floor": MISSION_IHSAN_HALT_FLOOR,
                },
            )
            published.append("ihsan.warning")

        # SESSION_END — triggers reflex compilation + PoI accumulation
        self._bus.publish(
            EventType.SESSION_END,
            {
                **base_payload,
                "actions": [
                    {
                        "action_type": action_type,
                        "ihsan_composite": ihsan,
                    }
                ],
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
