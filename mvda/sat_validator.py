"""SAT Validator — governance-lane constitutional verdict on PAT output."""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from mvda.config import (
    CLAIM_MUST_BIND,
    IHSAN_THRESHOLD,
    EVIDENCE_MIN_COUNT,
    OLLAMA_URL,
    SAT_MODEL,
)
from mvda.pat_researcher import PatResult

# Allowed verdicts — strict enum, no freeform
VALID_VERDICTS = {"PASS", "BLOCKED_BY_IHSAN", "BLOCKED_BY_EVIDENCE", "DEGRADED"}


@dataclass
class SatVerdict:
    verdict: str = "DEGRADED"
    reason: str = ""
    ihsan_score: float = 0.0
    evidence_sufficient: bool = False
    model: str = ""


def _call_sat_model(pat_result: PatResult) -> dict:
    """Call governance-lane model to evaluate PAT output."""
    system_prompt = (
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

    user_prompt = (
        f"PAT ANSWER:\n{pat_result.answer}\n\n"
        f"EVIDENCE REFS: {json.dumps(pat_result.evidence_refs)}\n"
        f"CONFIDENCE: {pat_result.confidence}\n"
        f"EVIDENCE COUNT: {len(pat_result.evidence_refs)}\n\n"
        "Evaluate now. Return JSON only."
    )

    payload = json.dumps({
        "model": SAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }).encode()

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
            # Extract JSON from response (model may wrap in markdown or thinking)
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
            return json.loads(raw)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, IndexError):
        return {}


def run_sat_validator(pat_result: PatResult) -> SatVerdict:
    """Run SAT governance validation on PAT output."""
    # Pre-checks (code-level, no LLM needed)
    if not pat_result.answer or pat_result.answer.startswith("ERROR:"):
        return SatVerdict(
            verdict="DEGRADED",
            reason="PAT produced no answer or errored",
            ihsan_score=0.0,
            evidence_sufficient=False,
            model="code-gate",
        )

    if CLAIM_MUST_BIND and len(pat_result.evidence_refs) < EVIDENCE_MIN_COUNT:
        return SatVerdict(
            verdict="BLOCKED_BY_EVIDENCE",
            reason=f"CLAIM_MUST_BIND: {len(pat_result.evidence_refs)} evidence refs < minimum {EVIDENCE_MIN_COUNT}",
            ihsan_score=0.0,
            evidence_sufficient=False,
            model="code-gate",
        )

    # LLM-powered constitutional evaluation
    llm_result = _call_sat_model(pat_result)

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
    if verdict == "PASS" and ihsan_score < IHSAN_THRESHOLD:
        verdict = "BLOCKED_BY_IHSAN"

    reason = llm_result.get("reason", "no reason provided")

    return SatVerdict(
        verdict=verdict,
        reason=reason,
        ihsan_score=ihsan_score,
        evidence_sufficient=evidence_ok,
        model=SAT_MODEL,
    )
