"""Equalizer Agent: cognitive-debt homeostasis control loop.

This module is additive and does not alter existing runtime wiring.
It provides mode detection + command generation around ihsan deficit,
backlog pressure, and human presence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OperationalMode(str, Enum):
    ACCUMULATION = "accumulation"
    SATURATION = "saturation"
    FLOW = "flow"
    RECOVERY = "recovery"
    STEADY = "steady"


class HaltReason(str, Enum):
    IHSAN_CRITICAL = "ihsan_critical"


class EqualizerCommandKind(str, Enum):
    ESCALATE = "escalate"
    ACCELERATE = "accelerate"
    HALT = "halt"
    RESUME = "resume"


@dataclass(frozen=True)
class EqualizerState:
    layer: int
    deficit: int
    backlog: int
    presence: int


@dataclass(frozen=True)
class PatternCandidate:
    hash32: bytes
    snr: float
    ihsan: float
    downstream_blocked: int


@dataclass(frozen=True)
class EqualizerCommand:
    kind: EqualizerCommandKind
    reason: str
    pattern_hash: bytes = b""
    batch_scale: int = 1


@dataclass
class EqualizerAgent:
    ihsan_target: float = 0.95
    accumulation_deficit: int = 13  # ~0.05 * 255
    saturation_deficit: int = 26  # ~0.10 * 255
    flow_deficit: int = 10
    flow_backlog: int = 8
    presence_threshold: int = 128
    history_limit: int = 64

    history: list[EqualizerState] = field(default_factory=list)
    pending_patterns: list[PatternCandidate] = field(default_factory=list)

    def observe(self, *, layer: int, ihsan_score: float, backlog: int, presence: int) -> EqualizerState:
        deficit = max(0.0, self.ihsan_target - ihsan_score)
        deficit_u8 = min(255, int(round(deficit * 255)))
        state = EqualizerState(
            layer=max(0, min(255, int(layer))),
            deficit=deficit_u8,
            backlog=max(0, min(65535, int(backlog))),
            presence=max(0, min(255, int(presence))),
        )
        self.history.append(state)
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit :]
        return state

    def detect_mode(self) -> OperationalMode:
        if not self.history:
            return OperationalMode.STEADY

        cur = self.history[-1]

        if cur.deficit >= self.saturation_deficit and cur.presence == 0:
            return OperationalMode.SATURATION

        if cur.deficit <= self.flow_deficit and cur.backlog <= self.flow_backlog:
            return OperationalMode.FLOW

        if len(self.history) >= 2:
            prev = self.history[-2]
            if cur.deficit < prev.deficit and cur.backlog <= prev.backlog:
                return OperationalMode.RECOVERY
            if cur.deficit > prev.deficit or cur.backlog > prev.backlog:
                return OperationalMode.ACCUMULATION

        if cur.deficit >= self.accumulation_deficit:
            return OperationalMode.ACCUMULATION

        return OperationalMode.STEADY

    def next_command(self) -> Optional[EqualizerCommand]:
        mode = self.detect_mode()
        cur = self.history[-1] if self.history else None

        if mode is OperationalMode.SATURATION:
            return EqualizerCommand(
                kind=EqualizerCommandKind.HALT,
                reason=HaltReason.IHSAN_CRITICAL.value,
            )

        if mode is OperationalMode.ACCUMULATION and cur is not None:
            if cur.presence > self.presence_threshold:
                return EqualizerCommand(
                    kind=EqualizerCommandKind.ESCALATE,
                    reason="human_present",
                    pattern_hash=self.select_critical_pattern(),
                )
            return EqualizerCommand(
                kind=EqualizerCommandKind.ACCELERATE,
                reason="human_absent",
                batch_scale=2,
            )

        if mode is OperationalMode.RECOVERY:
            return EqualizerCommand(kind=EqualizerCommandKind.RESUME, reason="recovery_detected")

        return None

    def select_critical_pattern(self) -> bytes:
        candidates = [p for p in self.pending_patterns if p.snr < 0.90 and p.ihsan < self.ihsan_target]
        if not candidates:
            return b""
        best = max(candidates, key=lambda p: p.downstream_blocked)
        return best.hash32
