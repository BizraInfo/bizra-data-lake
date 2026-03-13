"""
Diffusion Reasoning Amplifier -- HMM-Informed Query Biasing for T2 Deliberation.

Bridges T1 cognitive micro-state forecasts into T2 reasoning controls:
- complexity hint for routing
- GoT branch/depth budgets
- SNR target tier hints

Fail-closed by design: if prediction confidence is below policy floor,
the original query is preserved and no amplification is applied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final, Optional

from core.integration.constants import (
    GOT_MAX_DEPTH,
    GOT_MAX_HYPOTHESES,
    SNR_THRESHOLD_T0_ELITE,
    SNR_THRESHOLD_T1_HIGH,
    UNIFIED_SNR_THRESHOLD,
)
from core.prediction import HMMState, PredictionResult
from core.prediction.hierarchical_hmm import HierarchicalPredictionResult, StrategicGoal

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-zA-Z0-9_]+")

_SYMBOL_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "search": ("search", "find", "query", "lookup", "where"),
    "edit": ("edit", "change", "modify", "update", "fix", "refactor"),
    "navigate": ("open", "go", "navigate", "show", "view"),
    "organize": ("organize", "sort", "move", "rename", "clean"),
    "review": ("review", "check", "verify", "audit", "inspect"),
    "compile": ("build", "compile", "run", "execute"),
    "test": ("test", "pytest", "assert", "validate"),
    "chat": ("chat", "ask", "tell", "explain", "summarize"),
    "deploy": ("deploy", "push", "release", "publish"),
    "file_open": ("read", "load", "import", "fetch"),
    "file_save": ("save", "write", "export", "commit"),
    "idle": ("idle", "wait", "pause"),
}


@dataclass(frozen=True)
class DiffusionAmplifierConfig:
    """Policy envelope for HMM-informed diffusion amplification."""

    min_prediction_confidence: float = UNIFIED_SNR_THRESHOLD
    max_hypotheses: int = GOT_MAX_HYPOTHESES
    max_depth: int = GOT_MAX_DEPTH
    max_complexity_hint: float = 0.35


@dataclass(frozen=True)
class AmplifiedReasoningContext:
    """Serializable output for routing, GoT, and SNR layers."""

    activated: bool
    observation_symbol: str
    predicted_state: str
    confidence: float
    complexity_hint: float
    got_hypotheses: int
    got_depth: int
    snr_target: float
    hhmm_layer: int
    strategic_goal: str
    focus: str

    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "activated": self.activated,
            "observation_symbol": self.observation_symbol,
            "predicted_state": self.predicted_state,
            "confidence": self.confidence,
            "complexity_hint": self.complexity_hint,
            "got_hypotheses": self.got_hypotheses,
            "got_depth": self.got_depth,
            "snr_target": self.snr_target,
            "hhmm_layer": self.hhmm_layer,
            "focus": self.focus,
            "reasons": list(self.reasons),
        }


_STATE_PROFILE: Final[dict[str, dict[str, Any]]] = {
    HMMState.IDLE.value: {"focus": "stabilize", "hhmm_layer": 0, "branch_bias": 0},
    HMMState.EXPLORING.value: {"focus": "diverge", "hhmm_layer": 1, "branch_bias": 1},
    HMMState.ORGANIZING.value: {
        "focus": "structure",
        "hhmm_layer": 2,
        "branch_bias": 0,
    },
    HMMState.CREATING.value: {"focus": "synthesize", "hhmm_layer": 2, "branch_bias": 1},
    HMMState.ANALYZING.value: {"focus": "verify", "hhmm_layer": 3, "branch_bias": 1},
    HMMState.COMMUNICATING.value: {
        "focus": "explain",
        "hhmm_layer": 1,
        "branch_bias": 0,
    },
}

_STRATEGIC_BIAS: Final[dict[StrategicGoal, dict[str, float]]] = {
    StrategicGoal.DEBUGGING: {"depth_mult": 1.5, "hyp_mult": 1.2},
    StrategicGoal.DEVELOPING: {"depth_mult": 1.2, "hyp_mult": 1.0},
    StrategicGoal.REFACTORING: {"depth_mult": 2.0, "hyp_mult": 0.8},
    StrategicGoal.SYNCING: {"depth_mult": 1.8, "hyp_mult": 1.5},
    StrategicGoal.IDLE: {"depth_mult": 1.0, "hyp_mult": 1.0},
}


class DiffusionReasoningAmplifier:
    """Translate T1 prediction signals into deterministic T2 control hints."""

    def __init__(self, config: Optional[DiffusionAmplifierConfig] = None) -> None:
        self.config = config or DiffusionAmplifierConfig()

    @staticmethod
    def query_to_observation_symbol(query: str) -> str:
        """Map free text to the closest HMM observation symbol."""
        words = set(_WORD_RE.findall(query.lower()))
        best_symbol = "idle"
        best_overlap = 0
        for symbol, keywords in _SYMBOL_KEYWORDS.items():
            overlap = len(words.intersection(keywords))
            if overlap > best_overlap:
                best_overlap = overlap
                best_symbol = symbol
        return best_symbol

    def amplify(
        self,
        query: str,
        prediction: Optional[PredictionResult | HierarchicalPredictionResult],
        observation_symbol: str = "",
    ) -> AmplifiedReasoningContext:
        """Compute fail-closed amplification context from an HMM prediction."""
        symbol = observation_symbol or self.query_to_observation_symbol(query)
        if prediction is None:
            return AmplifiedReasoningContext(
                activated=False,
                observation_symbol=symbol,
                predicted_state="unknown",
                confidence=0.0,
                complexity_hint=0.0,
                got_hypotheses=1,
                got_depth=1,
                snr_target=UNIFIED_SNR_THRESHOLD,
                hhmm_layer=0,
                strategic_goal="unknown",
                focus="baseline",
                reasons=("no_prediction",),
            )

        # Handle both flat and hierarchical predictions
        if isinstance(prediction, HierarchicalPredictionResult):
            state = prediction.tactical_state.value
            confidence = prediction.tactical_confidence
            strategic_goal = prediction.strategic_goal
        else:
            state = prediction.predicted_next_state.value
            confidence = float(prediction.prediction_confidence)
            strategic_goal = StrategicGoal.IDLE

        confidence = max(0.0, min(1.0, confidence))

        profile = _STATE_PROFILE.get(
            state, {"focus": "baseline", "hhmm_layer": 0, "branch_bias": 0}
        )
        activated = confidence >= self.config.min_prediction_confidence
        if not activated:
            return AmplifiedReasoningContext(
                activated=False,
                observation_symbol=symbol,
                predicted_state=state,
                confidence=confidence,
                complexity_hint=0.0,
                got_hypotheses=1,
                got_depth=1,
                snr_target=UNIFIED_SNR_THRESHOLD,
                hhmm_layer=int(profile["hhmm_layer"]),
                strategic_goal=strategic_goal.name,
                focus=str(profile["focus"]),
                reasons=("confidence_below_threshold",),
            )

        # 4. Multi-Layer Modulation
        strat_bias = _STRATEGIC_BIAS.get(
            strategic_goal, {"depth_mult": 1.0, "hyp_mult": 1.0}
        )

        complexity_hint = min(
            self.config.max_complexity_hint,
            confidence
            * self.config.max_complexity_hint
            * strat_bias["depth_mult"]
            / 2.0,
        )

        got_hypotheses = max(
            1,
            min(
                self.config.max_hypotheses,
                int(
                    round(
                        1
                        + (confidence * (self.config.max_hypotheses - 1))
                        + int(profile["branch_bias"])
                    )
                ),
            ),
        )
        got_depth = max(
            1,
            min(
                self.config.max_depth,
                int(
                    round(
                        (1 + confidence * (self.config.max_depth - 1))
                        * strat_bias["depth_mult"]
                    )
                ),
            ),
        )

        if confidence >= 0.98:
            snr_target = SNR_THRESHOLD_T0_ELITE
        elif confidence >= 0.90:
            snr_target = SNR_THRESHOLD_T1_HIGH
        else:
            snr_target = UNIFIED_SNR_THRESHOLD

        return AmplifiedReasoningContext(
            activated=True,
            observation_symbol=symbol,
            predicted_state=state,
            confidence=confidence,
            complexity_hint=complexity_hint,
            got_hypotheses=got_hypotheses,
            got_depth=got_depth,
            snr_target=snr_target,
            hhmm_layer=int(profile["hhmm_layer"]),
            strategic_goal=strategic_goal.name,
            focus=str(profile["focus"]),
            reasons=("hmm_diffusion_amplified",),
        )

    @staticmethod
    def augment_query(query: str, ctx: AmplifiedReasoningContext) -> str:
        """Inject structured context when amplification is active."""
        if not ctx.activated:
            return query
        prefix = (
            "[DIFFUSION_CONTEXT]\n"
            f"predicted_state={ctx.predicted_state}\n"
            f"strategic_goal={ctx.strategic_goal}\n"
            f"confidence={ctx.confidence:.3f}\n"
            f"focus={ctx.focus}\n"
            f"got_hypotheses={ctx.got_hypotheses}\n"
            f"got_depth={ctx.got_depth}\n"
            f"snr_target={ctx.snr_target:.2f}\n"
            "[/DIFFUSION_CONTEXT]\n"
        )

        return f"{prefix}{query}"

    @staticmethod
    def context_for_router(ctx: AmplifiedReasoningContext) -> dict[str, Any]:
        """Router-facing context payload derived from amplification state."""
        return {
            "complexity_hint": ctx.complexity_hint if ctx.activated else 0.0,
            "got_hypotheses": ctx.got_hypotheses,
            "got_depth": ctx.got_depth,
            "snr_target": ctx.snr_target,
            "hhmm_layer": ctx.hhmm_layer,
            "diffusion_focus": ctx.focus,
            "strategic_goal": ctx.strategic_goal,
            "diffusion_active": ctx.activated,
        }
