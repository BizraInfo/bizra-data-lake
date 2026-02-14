"""
BIZRA BICAMERAL REASONING ENGINE v3.1-OMEGA
Standing on Giants: DeepSeek R1 (2025) + Karpathy (2024) + Jaynes (1976)

Bicameral architecture for robust inference:
- Right Hemisphere (R1/Local): Generative, creative, pattern-matching
- Left Hemisphere (Claude/API): Analytical, linguistic, verification
- Consensus: Fast generate-verify loops beat slow monolithic inference

Sovereignty: "Two minds are stronger than one. Verify, then trust."
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

logger = logging.getLogger(__name__)

DEFAULT_CONSENSUS_THRESHOLD = UNIFIED_IHSAN_THRESHOLD
DEFAULT_NUM_CANDIDATES = 3


class LocalInferenceProtocol(Protocol):
    """Protocol for local inference endpoint (R1-style generation)."""

    async def generate(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> str: ...


class AnalyticalClientProtocol(Protocol):
    """Protocol for analytical client (Claude-style verification)."""

    async def analyze(
        self, content: str, criteria: dict[str, Any]
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ReasoningCandidate:
    """A candidate solution from the generative hemisphere."""

    candidate_id: str
    content: str
    source: str  # "r1" or "claude"
    confidence: float
    reasoning_trace: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "reasoning_trace": self.reasoning_trace,
        }


@dataclass(frozen=True)
class VerificationResult:
    """Verification result from the analytical hemisphere."""

    candidate_id: str
    verified: bool
    critique: str
    adjusted_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "verified": self.verified,
            "critique": self.critique,
            "adjusted_confidence": self.adjusted_confidence,
        }


@dataclass
class BicameralResult:
    """Final result from bicameral reasoning process."""

    final_answer: str
    consensus_score: float
    candidates_generated: int
    candidates_verified: int
    reasoning_path: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "consensus_score": self.consensus_score,
            "candidates_generated": self.candidates_generated,
            "candidates_verified": self.candidates_verified,
            "reasoning_path": self.reasoning_path,
        }


class BicameralReasoningEngine:
    """
    Bicameral Reasoning Engine: Two hemispheres cooperating for robust inference.
    Standing on Giants: DeepSeek R1 (2025) + Karpathy (2024) + Jaynes (1976)
    """

    def __init__(
        self,
        local_endpoint: Optional[LocalInferenceProtocol] = None,
        api_client: Optional[AnalyticalClientProtocol] = None,
        consensus_threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
    ) -> None:
        self.local_endpoint = local_endpoint
        self.api_client = api_client
        self.consensus_threshold = consensus_threshold
        self._metrics = {
            "generations": 0,
            "verifications": 0,
            "consensus_hits": 0,
            "fallbacks": 0,
        }
        logger.info(
            "BicameralEngine init | local=%s api=%s",
            local_endpoint is not None,
            api_client is not None,
        )

    @property
    def is_bicameral(self) -> bool:
        return self.local_endpoint is not None and self.api_client is not None

    @property
    def metrics(self) -> dict[str, Any]:
        return {**self._metrics, "is_bicameral": self.is_bicameral}

    async def generate_candidates(
        self, problem: str, num_candidates: int = DEFAULT_NUM_CANDIDATES
    ) -> list[ReasoningCandidate]:
        """Generate multiple solution candidates using local model (R1-style)."""
        candidates: list[ReasoningCandidate] = []
        cot_prompt = (
            f"<|reasoning|>\nProblem: {problem}\nThink step by step.\n<|answer|>\n"
        )

        if self.local_endpoint:
            tasks = [self._gen_single(cot_prompt, i) for i in range(num_candidates)]
            for result in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(result, ReasoningCandidate):
                    candidates.append(result)
            self._metrics["generations"] += len(candidates)
        elif self.api_client:  # Fallback to API
            self._metrics["fallbacks"] += 1
            try:
                api_result = await self.api_client.analyze(problem, {"mode": "solve"})
                candidates.append(
                    ReasoningCandidate(
                        str(uuid.uuid4())[:8],
                        api_result.get("answer", str(api_result)),
                        "claude",
                        api_result.get("confidence", 0.7),
                        "[FALLBACK]",
                    )
                )
                self._metrics["generations"] += 1
            except Exception as e:
                logger.error("Fallback generation failed: %s", e)

        logger.info("Generated %d/%d candidates", len(candidates), num_candidates)
        return candidates

    async def verify_candidate(
        self, candidate: ReasoningCandidate, criteria: dict[str, Any]
    ) -> VerificationResult:
        """Verify a candidate using analytical model (Claude-style)."""
        if not self.api_client:
            return VerificationResult(
                candidate.candidate_id, True, "[FALLBACK]", candidate.confidence * 0.7
            )

        try:
            result = await asyncio.wait_for(
                self.api_client.analyze(candidate.content, criteria), timeout=30.0
            )
            self._metrics["verifications"] += 1
            adj = max(
                0.0,
                min(1.0, candidate.confidence + result.get("confidence_delta", 0.0)),
            )
            return VerificationResult(
                candidate.candidate_id,
                result.get("passes", False),
                result.get("critique", ""),
                adj,
            )
        except Exception as e:
            logger.error("Verification error: %s", e)
            return VerificationResult(
                candidate.candidate_id,
                False,
                f"[ERROR] {e}",
                candidate.confidence * 0.5,
            )

    async def reason(self, problem: str, context: dict[str, Any]) -> BicameralResult:
        """Full bicameral reasoning: generate, verify, select."""
        start = time.monotonic()
        path: list[dict[str, Any]] = []

        # Phase 1: Generate (Right Hemisphere)
        candidates = await self.generate_candidates(
            problem, context.get("num_candidates", DEFAULT_NUM_CANDIDATES)
        )
        path.append(
            {
                "phase": "generation",
                "ts": datetime.now(timezone.utc).isoformat(),
                "candidates": [c.to_dict() for c in candidates],
            }
        )

        if not candidates:
            return BicameralResult("[NO CANDIDATES]", 0.0, 0, 0, path)

        # Phase 2: Verify (Left Hemisphere)
        criteria = context.get("criteria", {"correctness": True})
        verifications = await asyncio.gather(
            *[self.verify_candidate(c, criteria) for c in candidates]
        )
        vmap = {v.candidate_id: v for v in verifications}
        path.append(
            {
                "phase": "verification",
                "ts": datetime.now(timezone.utc).isoformat(),
                "results": [v.to_dict() for v in verifications],
            }
        )

        # Phase 3: Select best
        verified = [
            (c, vmap[c.candidate_id])
            for c in candidates
            if vmap[c.candidate_id].verified
        ]
        if verified:
            best_c, best_v = self._select_best(verified)
            score = best_v.adjusted_confidence
            if score >= self.consensus_threshold:
                self._metrics["consensus_hits"] += 1
        else:
            best_c = max(candidates, key=lambda c: c.confidence)
            score = best_c.confidence * 0.5

        elapsed = (time.monotonic() - start) * 1000
        path.append(
            {
                "phase": "selection",
                "ts": datetime.now(timezone.utc).isoformat(),
                "selected": best_c.candidate_id,
                "score": score,
                "elapsed_ms": elapsed,
            }
        )
        logger.info(
            "Bicameral complete | gen=%d ver=%d score=%.2f",
            len(candidates),
            len(verified),
            score,
        )
        return BicameralResult(
            best_c.content, score, len(candidates), len(verified), path
        )

    def _select_best(
        self, verified: list[tuple[ReasoningCandidate, VerificationResult]]
    ) -> tuple[ReasoningCandidate, VerificationResult]:
        """Select best candidate by adjusted confidence."""
        return max(
            verified, key=lambda cv: (cv[1].adjusted_confidence, cv[0].confidence)
        )

    async def _gen_single(self, prompt: str, idx: int) -> ReasoningCandidate:
        """Generate single candidate with varying temperature."""
        if not self.local_endpoint:
            raise RuntimeError("Local inference endpoint not configured")
        temp = min(0.7 + idx * 0.1, 1.2)
        try:
            resp = await asyncio.wait_for(
                self.local_endpoint.generate(prompt, 2048, temp), timeout=60.0
            )
            content, trace = (resp.split("<|answer|>", 1) + [""])[:2]
            trace = trace.replace("<|reasoning|>", "").strip()
            conf = (
                0.6
                + (0.1 if len(trace.split()) > 50 else 0)
                + (0.1 if len(trace.split()) > 100 else 0)
            )
            return ReasoningCandidate(
                str(uuid.uuid4())[:8], content.strip(), "r1", min(conf, 1.0), trace
            )
        except Exception as e:
            raise RuntimeError(f"Gen {idx} failed: {e}")
