"""
SAT Validator — Governance-lane constitutional verdict on PAT output.

Promoted from MVDA v0.3 into core proof engine.
First LLM-powered SAT gate in the BIZRA runtime.

Evaluates PAT output against:
  - Ihsān threshold (excellence ≥ 0.95)
  - CLAIM_MUST_BIND (evidence presence)
  - Daughter Test (output dignity)

Returns strict schema verdicts: PASS | BLOCKED_BY_IHSAN | BLOCKED_BY_EVIDENCE | DEGRADED

Standing on Giants:
- Al-Ghazali (1095): Self-knowledge and moral excellence
- Shannon (1948): Signal quality measurement
- BIZRA Constitutional Computability Doctrine v1.0
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

# Configurable via environment — no hardcoded model names
SAT_MODEL = os.getenv("BIZRA_SAT_MODEL", "gemma4:26b-bizra-16k")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
EVIDENCE_MIN_COUNT = int(os.getenv("BIZRA_EVIDENCE_MIN_COUNT", "1"))

# Allowed verdicts — strict enum, no freeform
VALID_VERDICTS = frozenset(
    {"PASS", "BLOCKED_BY_IHSAN", "BLOCKED_BY_EVIDENCE", "DEGRADED"}
)

# Constitutional governance prompt — the core of the SAT boundary
_SAT_SYSTEM_PROMPT = (
    "You are a BIZRA SAT Validator — a constitutional governance agent. "
    "You evaluate whether a PAT agent's output meets BIZRA's constitutional standards.\n\n"
    "Evaluate the answer against these criteria:\n"
    "1. IHSAN (excellence): Is the answer high-quality, precise, and honest? Score 0.0-1.0.\n"
    "2. CLAIM_MUST_BIND: Does every claim have evidence backing it?\n"
    "3. DAUGHTER_TEST: Is this output appropriate and dignified?\n\n"
    "You MUST respond with ONLY a JSON object, no other text:\n"
    '{"verdict": "PASS|BLOCKED_BY_IHSAN|BLOCKED_BY_EVIDENCE|DEGRADED", '
    '"reason": "short explanation", '
    '"ihsan_score": 0.0-1.0, '
    '"evidence_sufficient": true|false}'
)


class PatOutput(Protocol):
    """Protocol for PAT output that SAT can evaluate."""

    @property
    def answer(self) -> str: ...

    @property
    def evidence_refs(self) -> List[str]: ...

    @property
    def confidence(self) -> str: ...


@dataclass
class SatVerdict:
    """Structured verdict from SAT governance evaluation."""

    verdict: str = "DEGRADED"
    reason: str = ""
    ihsan_score: float = 0.0
    evidence_sufficient: bool = False
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "ihsan_score": self.ihsan_score,
            "evidence_sufficient": self.evidence_sufficient,
            "model": self.model,
        }


@dataclass
class SimplePatOutput:
    """Minimal PAT output for testing and standalone use."""

    answer: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    confidence: str = "none"


def _extract_json_from_llm_response(raw: str) -> dict:
    """Extract JSON from LLM response that may contain markdown or thinking tokens."""
    if not raw:
        return {}

    # Try markdown code blocks first
    if "```" in raw:
        parts = raw.split("```")
        for part in parts[1:]:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    # Try to find JSON object in raw text
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

    # Last resort: parse entire string
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _call_sat_model(pat_output: PatOutput) -> dict:
    """Call governance-lane model to evaluate PAT output."""
    user_prompt = (
        f"PAT ANSWER:\n{pat_output.answer}\n\n"
        f"EVIDENCE REFS: {json.dumps(pat_output.evidence_refs)}\n"
        f"CONFIDENCE: {pat_output.confidence}\n"
        f"EVIDENCE COUNT: {len(pat_output.evidence_refs)}\n\n"
        "Evaluate now. Return JSON only."
    )

    payload = json.dumps(
        {
            "model": SAT_MODEL,
            "messages": [
                {"role": "system", "content": _SAT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1024},
        }
    ).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
            msg = data.get("message", {})
            # gemma4 thinking model: check both content and thinking fields
            raw = msg.get("content", "")
            if not raw:
                raw = msg.get("thinking", "")
            return _extract_json_from_llm_response(raw)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}


def validate(pat_output: PatOutput) -> SatVerdict:
    """Run SAT governance validation on PAT output.

    This is the constitutional boundary — every PAT output must pass through here.

    Args:
        pat_output: Any object implementing PatOutput protocol (answer, evidence_refs, confidence).

    Returns:
        SatVerdict with strict schema verdict.
    """
    # Pre-checks (code-level gates, no LLM needed)
    if not pat_output.answer or pat_output.answer.startswith("ERROR:"):
        return SatVerdict(
            verdict="DEGRADED",
            reason="PAT produced no answer or errored",
            ihsan_score=0.0,
            evidence_sufficient=False,
            model="code-gate",
        )

    if len(pat_output.evidence_refs) < EVIDENCE_MIN_COUNT:
        return SatVerdict(
            verdict="BLOCKED_BY_EVIDENCE",
            reason=f"CLAIM_MUST_BIND: {len(pat_output.evidence_refs)} evidence refs < minimum {EVIDENCE_MIN_COUNT}",
            ihsan_score=0.0,
            evidence_sufficient=False,
            model="code-gate",
        )

    # LLM-powered constitutional evaluation
    llm_result = _call_sat_model(pat_output)

    if not llm_result:
        return SatVerdict(
            verdict="DEGRADED",
            reason="SAT model unreachable or returned invalid JSON",
            ihsan_score=0.0,
            evidence_sufficient=False,
            model=SAT_MODEL,
        )

    verdict = llm_result.get("verdict", "DEGRADED")
    if verdict not in VALID_VERDICTS:
        verdict = "DEGRADED"

    ihsan_score = float(llm_result.get("ihsan_score", 0.0))
    evidence_ok = bool(llm_result.get("evidence_sufficient", False))

    # Enforce Ihsān threshold even if LLM says PASS
    if verdict == "PASS" and ihsan_score < UNIFIED_IHSAN_THRESHOLD:
        verdict = "BLOCKED_BY_IHSAN"

    reason = llm_result.get("reason", "no reason provided")

    return SatVerdict(
        verdict=verdict,
        reason=reason,
        ihsan_score=ihsan_score,
        evidence_sufficient=evidence_ok,
        model=SAT_MODEL,
    )
