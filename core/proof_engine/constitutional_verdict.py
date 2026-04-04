"""
BIZRA Constitutional Computability Doctrine v1.0 — Implementation

Authority: البذرة → Enforceable Spine v1.1 → Doctrine v1.0 → this code
Ratified: 1 April 2026
CLAIM_MUST_BIND: Every type, function, and contract in this module traces
to the four axioms of the Constitutional Computability Doctrine.

Axiom 1: Kernel law must be decidable.
Axiom 2: Not all ethics are kernel law.
Axiom 3: Therefore the constitution is stratified.
Axiom 4: Claims of completeness are scoped.
"""

from __future__ import annotations

import enum
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime, timezone

try:
    import blake3

    def blake3_hash(data: bytes) -> str:
        return blake3.blake3(data).hexdigest()

except ImportError:

    def blake3_hash(data: bytes) -> str:
        return hashlib.sha256(b"BLAKE3_FALLBACK:" + data).hexdigest()


# ═══════════════════════════════════════════════════════════════════
# LAYER 0: FROZEN ANCHORS — Non-negotiable, pre-constitutional
# ═══════════════════════════════════════════════════════════════════

ZANN_ZERO: bool = True  # No unverified claims propagated
RIBA_ZERO: bool = True  # No extractive economic patterns
GINI_CEILING: float = 0.35
IHSAN_FLOOR: float = 0.95


# ═══════════════════════════════════════════════════════════════════
# THE VERDICT ENUM — Replaces binary ALLOW/DENY
# ═══════════════════════════════════════════════════════════════════


class ConstitutionalVerdict(enum.Enum):
    """
    Four-state verdict system. The kernel is physics (PERMIT/REJECT).
    The higher constitution is jurisprudence (REVIEW/SCORE_ONLY).

    Axiom 3: The constitution is stratified. This enum IS that stratification
    made executable. Without REVIEW, the system will either lie or deadlock.
    """

    PERMIT = "PERMIT"  # Layer 1 or 2: constitutionally admissible
    REJECT = "REJECT"  # Layer 1 or 2: violates hard law or timed out on high-risk
    REVIEW = "REVIEW"  # Layer 2: budget exceeded on low-risk action
    SCORE_ONLY = "SCORE_ONLY"  # Layer 3: advisory assessment, no blocking authority

    def is_blocking(self) -> bool:
        """Only PERMIT and REJECT are terminal blocking verdicts."""
        return self in (ConstitutionalVerdict.PERMIT, ConstitutionalVerdict.REJECT)

    def is_admissible(self) -> bool:
        """Only PERMIT allows execution to proceed."""
        return self == ConstitutionalVerdict.PERMIT

    def to_canonical_bytes(self) -> bytes:
        """Deterministic serialization for BLAKE3 hashing. Sorted JSON, no whitespace."""
        return json.dumps(
            {"verdict": self.value}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


# ═══════════════════════════════════════════════════════════════════
# CONSTITUTIONAL LAYER CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════


class ConstitutionalLayer(enum.Enum):
    """Which layer of the constitution governs this action type."""

    HARD_LAW = 1  # Decidable, blocking, replayable — kernel physics
    REVIEW_LAW = 2  # Bounded, timeout-aware, fail-closed for high-risk
    JUDICIARY = 3  # Advisory/deliberative, evidence-producing, not hot-path gate


class ActionRiskClass(enum.Enum):
    """Risk classification determines timeout fallback behavior."""

    LOW = "LOW"  # Timeout → REVIEW (escalate for deliberation)
    HIGH = "HIGH"  # Timeout → REJECT (fail-closed, sovereignty preserved)


# ═══════════════════════════════════════════════════════════════════
# KERNEL ACTION GRAMMAR — The finite set the microkernel may decide
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ActionType:
    """An action in the constitutional grammar with its layer classification."""

    name: str
    layer: ConstitutionalLayer
    risk_class: ActionRiskClass = ActionRiskClass.HIGH  # default: fail-closed
    description: str = ""


class KernelActionGrammar:
    """
    Axiom 1: Kernel law must be decidable.
    This class defines the FINITE set of actions the kernel may decide directly.
    Actions not in this grammar are escalated to Layer 2 or Layer 3.

    Axiom 4: Claims of completeness are scoped — only to this grammar.
    """

    # ── Layer 1: Hard Law (decidable, blocking) ──
    SIGNATURE_VALIDITY = ActionType(
        "signature_validity",
        ConstitutionalLayer.HARD_LAW,
        description="Ed25519 signature check",
    )
    SCHEMA_VALIDITY = ActionType(
        "schema_validity",
        ConstitutionalLayer.HARD_LAW,
        description="Input schema validation at ingress",
    )
    RECEIPT_TRANSITION = ActionType(
        "receipt_transition",
        ConstitutionalLayer.HARD_LAW,
        description="ReceiptStateMachine state change legality",
    )
    RIBA_ZERO_VETO = ActionType(
        "riba_zero_veto",
        ConstitutionalLayer.HARD_LAW,
        description="Non-bypassable RIBA_ZERO check",
    )
    ZANN_ZERO_VETO = ActionType(
        "zann_zero_veto",
        ConstitutionalLayer.HARD_LAW,
        description="Non-bypassable ZANN_ZERO check",
    )
    IHSAN_SCALAR_THRESHOLD = ActionType(
        "ihsan_scalar_threshold",
        ConstitutionalLayer.HARD_LAW,
        description="Pre-computed scalar comparison against floor",
    )
    RESOURCE_CAP = ActionType(
        "resource_cap",
        ConstitutionalLayer.HARD_LAW,
        description="Resource/risk cap enforcement",
    )
    LIVENESS_CONTRACT = ActionType(
        "liveness_contract",
        ConstitutionalLayer.HARD_LAW,
        description="Request → response within bounded time",
    )

    # ── Layer 2: Bounded Review (budgeted, timeout-aware) ──
    PROJECTED_IHSAN = ActionType(
        "projected_ihsan",
        ConstitutionalLayer.REVIEW_LAW,
        ActionRiskClass.LOW,
        "Budgeted Ihsān projection",
    )
    CACHED_ADL_DELTA = ActionType(
        "cached_adl_delta",
        ConstitutionalLayer.REVIEW_LAW,
        ActionRiskClass.LOW,
        "Bounded Adl delta from cached aggregates",
    )
    SMT_CONSTRAINT = ActionType(
        "smt_constraint",
        ConstitutionalLayer.REVIEW_LAW,
        ActionRiskClass.HIGH,
        "Z3/FATE solving with timeout",
    )
    PROVENANCE_TRAVERSAL = ActionType(
        "provenance_traversal",
        ConstitutionalLayer.REVIEW_LAW,
        ActionRiskClass.LOW,
        "Depth-limited provenance walk",
    )

    # ── Layer 3: Judiciary / Advisory (evidence-producing) ──
    FULL_8D_IHSAN = ActionType(
        "full_8d_ihsan",
        ConstitutionalLayer.JUDICIARY,
        description="Full contextual 8D Ihsān scoring",
    )
    GRAPH_ADL = ActionType(
        "graph_adl",
        ConstitutionalLayer.JUDICIARY,
        description="Whole-graph distributional justice",
    )
    DEEP_AMANAH = ActionType(
        "deep_amanah",
        ConstitutionalLayer.JUDICIARY,
        description="Long-horizon trust stewardship",
    )
    GUARDIAN_COUNCIL = ActionType(
        "guardian_council",
        ConstitutionalLayer.JUDICIARY,
        description="Deliberative constitutional interpretation",
    )
    COUNTERFACTUAL = ActionType(
        "counterfactual",
        ConstitutionalLayer.JUDICIARY,
        description="Simulation and scenario analysis",
    )

    _ALL_ACTIONS: dict[str, ActionType] = {}

    @classmethod
    def _init_registry(cls):
        if not cls._ALL_ACTIONS:
            for attr_name in dir(cls):
                val = getattr(cls, attr_name)
                if isinstance(val, ActionType):
                    cls._ALL_ACTIONS[val.name] = val

    @classmethod
    def classify(cls, action_name: str) -> ActionType:
        """Classify an action by name. Unknown actions default to Layer 3 (advisory)."""
        cls._init_registry()
        if action_name in cls._ALL_ACTIONS:
            return cls._ALL_ACTIONS[action_name]
        # Unknown action → Layer 3 (fail-safe: advisory only, no blocking authority)
        return ActionType(
            action_name,
            ConstitutionalLayer.JUDICIARY,
            description=f"Unknown action '{action_name}' — classified as advisory",
        )

    @classmethod
    def layer_1_actions(cls) -> list[ActionType]:
        cls._init_registry()
        return [
            a
            for a in cls._ALL_ACTIONS.values()
            if a.layer == ConstitutionalLayer.HARD_LAW
        ]

    @classmethod
    def layer_2_actions(cls) -> list[ActionType]:
        cls._init_registry()
        return [
            a
            for a in cls._ALL_ACTIONS.values()
            if a.layer == ConstitutionalLayer.REVIEW_LAW
        ]

    @classmethod
    def layer_3_actions(cls) -> list[ActionType]:
        cls._init_registry()
        return [
            a
            for a in cls._ALL_ACTIONS.values()
            if a.layer == ConstitutionalLayer.JUDICIARY
        ]


# ═══════════════════════════════════════════════════════════════════
# BUDGETED REVIEW CONTRACT
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ReviewBudget:
    """
    Axiom 2: Not all ethics are kernel law. Some checks need bounded time.
    This contract defines the budget for Layer 2 evaluations.
    """

    max_time_ms: float = 100.0  # Maximum wall-clock time in milliseconds
    max_provenance_depth: int = 10  # Maximum chain traversal depth
    approximation_mode: str = "cached"  # "cached" | "sketch" | "delta"
    fallback_by_risk: dict[ActionRiskClass, ConstitutionalVerdict] = field(
        default_factory=lambda: {
            ActionRiskClass.LOW: ConstitutionalVerdict.REVIEW,  # escalate
            ActionRiskClass.HIGH: ConstitutionalVerdict.REJECT,  # fail-closed
        }
    )


@dataclass
class VerdictReceipt:
    """Every verdict is receipted. This is the proof artifact."""

    action_name: str
    layer: int
    verdict: str
    risk_class: str
    elapsed_ms: float
    budget_exceeded: bool
    timestamp: str
    hash: str = ""

    def compute_hash(self, prev_hash: str = "") -> str:
        """BLAKE3 hash with chain linking."""
        canonical = json.dumps(
            {
                "action": self.action_name,
                "layer": self.layer,
                "verdict": self.verdict,
                "risk": self.risk_class,
                "elapsed_ms": round(self.elapsed_ms, 3),
                "budget_exceeded": self.budget_exceeded,
                "timestamp": self.timestamp,
                "prev_hash": prev_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.hash = blake3_hash(canonical)
        return self.hash


class ConstitutionalGate:
    """
    The unified constitutional gate implementing all three layers.

    Layer 1: Direct evaluation, always terminates, blocking.
    Layer 2: Budgeted evaluation with timeout, fail-closed for high-risk.
    Layer 3: Advisory scoring, never blocks.
    """

    def __init__(self, budget: Optional[ReviewBudget] = None):
        self.budget = budget or ReviewBudget()
        self._receipt_chain: list[VerdictReceipt] = []
        self._prev_hash: str = "0" * 64  # genesis hash

        # Layer 1 evaluators: action_name → callable returning bool (True=permit)
        self._layer1_evaluators: dict[str, Callable[..., bool]] = {
            "signature_validity": lambda ctx: ctx.get("signature_valid", False),
            "schema_validity": lambda ctx: ctx.get("schema_valid", False),
            "receipt_transition": lambda ctx: ctx.get("transition_legal", False),
            "riba_zero_veto": lambda ctx: not ctx.get("contains_riba", False),
            "zann_zero_veto": lambda ctx: not ctx.get(
                "contains_unverified_claim", False
            ),
            "ihsan_scalar_threshold": lambda ctx: ctx.get("ihsan_score", 0.0)
            >= IHSAN_FLOOR,
            "resource_cap": lambda ctx: ctx.get("within_resource_cap", True),
            "liveness_contract": lambda ctx: ctx.get("responds_within_budget", True),
        }

        # Layer 2 evaluators: action_name → callable returning Optional[bool]
        # None means "budget exceeded, use fallback"
        self._layer2_evaluators: dict[str, Callable[..., Optional[bool]]] = {
            "projected_ihsan": self._evaluate_projected_ihsan,
            "cached_adl_delta": self._evaluate_cached_adl,
            "smt_constraint": self._evaluate_smt,
            "provenance_traversal": self._evaluate_provenance,
        }

    def _evaluate_projected_ihsan(self, ctx: dict) -> Optional[bool]:
        """Simulate budgeted Ihsān projection."""
        projected = ctx.get("projected_ihsan", None)
        if projected is None:
            return None  # budget exceeded
        return projected >= IHSAN_FLOOR

    def _evaluate_cached_adl(self, ctx: dict) -> Optional[bool]:
        """Simulate cached Adl delta check."""
        gini_delta = ctx.get("gini_delta", None)
        if gini_delta is None:
            return None
        current_gini = ctx.get("current_gini", 0.0)
        return (current_gini + gini_delta) <= GINI_CEILING

    def _evaluate_smt(self, ctx: dict) -> Optional[bool]:
        """Simulate SMT constraint solving with timeout."""
        smt_result = ctx.get("smt_result", None)
        return smt_result  # None = timeout, True/False = result

    def _evaluate_provenance(self, ctx: dict) -> Optional[bool]:
        """Simulate depth-limited provenance traversal."""
        depth = ctx.get("provenance_depth", 0)
        if depth > self.budget.max_provenance_depth:
            return None  # exceeded depth budget
        return ctx.get("provenance_valid", False)

    def evaluate(
        self, action_name: str, context: dict
    ) -> tuple[ConstitutionalVerdict, VerdictReceipt]:
        """
        Evaluate a constitutional action and return a verdict with receipt.

        This is the main entry point. It routes to the correct layer,
        applies budgets and fallbacks, and emits a chained receipt.
        """
        action_type = KernelActionGrammar.classify(action_name)
        start_time = time.monotonic()

        if action_type.layer == ConstitutionalLayer.HARD_LAW:
            verdict = self._evaluate_layer1(action_name, context)
            budget_exceeded = False

        elif action_type.layer == ConstitutionalLayer.REVIEW_LAW:
            verdict, budget_exceeded = self._evaluate_layer2(
                action_name, action_type, context, start_time
            )

        else:  # JUDICIARY
            verdict = ConstitutionalVerdict.SCORE_ONLY
            budget_exceeded = False

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        receipt = VerdictReceipt(
            action_name=action_name,
            layer=action_type.layer.value,
            verdict=verdict.value,
            risk_class=action_type.risk_class.value,
            elapsed_ms=elapsed_ms,
            budget_exceeded=budget_exceeded,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        receipt.compute_hash(self._prev_hash)
        self._prev_hash = receipt.hash
        self._receipt_chain.append(receipt)

        return verdict, receipt

    def _evaluate_layer1(
        self, action_name: str, context: dict
    ) -> ConstitutionalVerdict:
        """Layer 1: Hard law. Always terminates. PERMIT or REJECT."""
        evaluator = self._layer1_evaluators.get(action_name)
        if evaluator is None:
            return ConstitutionalVerdict.REJECT  # fail-closed: unknown L1 action
        return (
            ConstitutionalVerdict.PERMIT
            if evaluator(context)
            else ConstitutionalVerdict.REJECT
        )

    def _evaluate_layer2(
        self,
        action_name: str,
        action_type: ActionType,
        context: dict,
        start_time: float,
    ) -> tuple[ConstitutionalVerdict, bool]:
        """Layer 2: Bounded review. Timeout-aware. Fallback by risk class."""
        evaluator = self._layer2_evaluators.get(action_name)
        if evaluator is None:
            # Unknown L2 evaluator → fallback by risk
            return self.budget.fallback_by_risk[action_type.risk_class], True

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        if elapsed_ms > self.budget.max_time_ms:
            return self.budget.fallback_by_risk[action_type.risk_class], True

        result = evaluator(context)

        if result is None:
            # Evaluator returned None = budget exceeded
            return self.budget.fallback_by_risk[action_type.risk_class], True

        verdict = (
            ConstitutionalVerdict.PERMIT if result else ConstitutionalVerdict.REJECT
        )
        return verdict, False

    @property
    def receipt_chain(self) -> list[VerdictReceipt]:
        return list(self._receipt_chain)

    def verify_chain_integrity(self) -> bool:
        """Verify the entire receipt chain has valid hash linking."""
        prev = "0" * 64
        for receipt in self._receipt_chain:
            canonical = json.dumps(
                {
                    "action": receipt.action_name,
                    "layer": receipt.layer,
                    "verdict": receipt.verdict,
                    "risk": receipt.risk_class,
                    "elapsed_ms": round(receipt.elapsed_ms, 3),
                    "budget_exceeded": receipt.budget_exceeded,
                    "timestamp": receipt.timestamp,
                    "prev_hash": prev,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            expected = blake3_hash(canonical)
            if receipt.hash != expected:
                return False
            prev = receipt.hash
        return True
