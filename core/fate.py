from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("fate.gate")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MAX_CORRECTION_RETRIES = int(os.getenv("BIZRA_MAX_RETRIES", "2"))
EVIDENCE_PATH = Path(os.getenv("BIZRA_FATE_EVIDENCE", "docs/evidence/fate"))


CANONICAL_DIMENSIONS = [
    "correctness",
    "safety",
    "user_benefit",
    "efficiency",
    "auditability",
    "anti_centralization",
    "robustness",
    "adl_fairness",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_key(raw: str) -> str:
    return raw.strip().lower().replace("-", "_").replace(" ", "_")


def _find_constitution_path() -> Optional[Path]:
    configured = os.getenv("BIZRA_IHSAN_CONSTITUTION")
    if configured:
        p = Path(configured).expanduser()
        return p.resolve() if p.exists() else None

    start = Path(__file__).resolve()
    for parent in [start.parent, *start.parents]:
        cand = parent / "constitution" / "ihsan_v1.yaml"
        if cand.exists():
            return cand.resolve()
    return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class IhsanPolicy:
    loaded: bool
    constitution_id: str
    constitution_version: int
    constitution_sha256: Optional[str]
    constitution_path: Optional[str]
    weights: Dict[str, float]
    default_env: str
    thresholds_by_env: Dict[str, float]
    env_aliases: Dict[str, str]

    def normalize_env(self, env: str) -> str:
        key = _normalize_key(env)
        return self.env_aliases.get(key, key)

    def threshold_for_env(self, env: str) -> float:
        normalized = self.normalize_env(env)
        return float(self.thresholds_by_env.get(normalized, self.thresholds_by_env.get(self.default_env, 0.95)))


def load_ihsan_policy() -> IhsanPolicy:
    path = _find_constitution_path()
    if path is None:
        return IhsanPolicy(
            loaded=False,
            constitution_id="ihsan_v1",
            constitution_version=1,
            constitution_sha256=None,
            constitution_path=None,
            weights={},
            default_env="development",
            thresholds_by_env={"development": 0.80, "ci": 0.90, "production": 0.95},
            env_aliases={"dev": "development", "prod": "production"},
        )

    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"ihsan constitution must be a mapping: {path}")

    cid = str(data.get("id") or "ihsan_v1")
    version = int(data.get("version") or 1)
    dims = data.get("dimensions") or {}
    if not isinstance(dims, dict):
        raise ValueError(f"ihsan constitution dimensions must be a mapping: {path}")

    weights: Dict[str, float] = {}
    for dim in CANONICAL_DIMENSIONS:
        entry = dims.get(dim) or {}
        if not isinstance(entry, dict) or "weight" not in entry:
            raise ValueError(f"missing weight for dimension '{dim}' in {path}")
        weights[dim] = float(entry["weight"])

    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"ihsan weights must sum to 1.0 in {path}")

    policy = data.get("threshold_policy") or {}
    if not isinstance(policy, dict):
        policy = {}
    default_env = _normalize_key(str(policy.get("default_env") or "development"))

    thresholds_by_env_raw = policy.get("thresholds_by_env") or {}
    thresholds_by_env: Dict[str, float] = {}
    if isinstance(thresholds_by_env_raw, dict):
        for k, v in thresholds_by_env_raw.items():
            if not isinstance(k, str):
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            thresholds_by_env[_normalize_key(k)] = fv

    normalization = policy.get("normalization") or {}
    if not isinstance(normalization, dict):
        normalization = {}
    env_aliases_raw = normalization.get("env_aliases") or {}
    env_aliases: Dict[str, str] = {}
    if isinstance(env_aliases_raw, dict):
        for k, v in env_aliases_raw.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            env_aliases[_normalize_key(k)] = _normalize_key(v)

    return IhsanPolicy(
        loaded=True,
        constitution_id=cid,
        constitution_version=version,
        constitution_sha256=_sha256_file(path),
        constitution_path=str(path),
        weights=weights,
        default_env=default_env,
        thresholds_by_env=thresholds_by_env,
        env_aliases=env_aliases,
    )


class IhsanVector(BaseModel):
    correctness: float = Field(1.0, ge=0.0, le=1.0)
    safety: float = Field(1.0, ge=0.0, le=1.0)
    user_benefit: float = Field(1.0, ge=0.0, le=1.0)
    efficiency: float = Field(1.0, ge=0.0, le=1.0)
    auditability: float = Field(1.0, ge=0.0, le=1.0)
    anti_centralization: float = Field(1.0, ge=0.0, le=1.0)
    robustness: float = Field(1.0, ge=0.0, le=1.0)
    adl_fairness: float = Field(1.0, ge=0.0, le=1.0)

    def composite_score(self, weights: Dict[str, float]) -> float:
        data = self.model_dump()
        return float(sum(float(weights.get(k, 0.0)) * float(data.get(k, 0.0)) for k in CANONICAL_DIMENSIONS))


class FateSeal(BaseModel):
    id: str
    timestamp: str
    validator: str
    env: str
    threshold: float
    vector: IhsanVector
    composite_score: float
    verdict: Literal["APPROVED", "REJECTED"]
    reason: str
    policy: Dict[str, Optional[str]]
    intent_sha256: str
    context_sha256: str


class FateEngine:
    def __init__(self, *, strict_mode: bool = True, validator: str = "Node0_Sovereign_Kernel"):
        self.strict_mode = strict_mode
        self.validator = validator
        self.policy = load_ihsan_policy()

    def _env(self) -> str:
        return (os.getenv("BIZRA_ENV") or os.getenv("ENV") or self.policy.default_env) or self.policy.default_env

    def audit_request(self, *, intent: str, context: str = "") -> FateSeal:
        env = self._env()
        normalized_env = self.policy.normalize_env(env)

        intent = intent or ""
        context = context or ""
        intent_hash = sha256_text(intent)
        context_hash = sha256_text(context)

        if self.strict_mode and not self.policy.loaded:
            return self._seal(
                env=normalized_env,
                threshold=1.0,
                vector=IhsanVector.model_validate({k: 0.0 for k in CANONICAL_DIMENSIONS}),
                composite=0.0,
                verdict="REJECTED",
                reason="ihsan_policy_unavailable",
                intent_sha256=intent_hash,
                context_sha256=context_hash,
            )

        weights = self.policy.weights
        threshold = self.policy.threshold_for_env(normalized_env)

        vector, reason = self._score_intent(intent=intent, context=context)
        composite = vector.composite_score(weights)
        verdict: Literal["APPROVED", "REJECTED"] = "APPROVED" if composite >= threshold else "REJECTED"
        return self._seal(
            env=normalized_env,
            threshold=threshold,
            vector=vector,
            composite=composite,
            verdict=verdict,
            reason=reason,
            intent_sha256=intent_hash,
            context_sha256=context_hash,
        )

    def _seal(
        self,
        *,
        env: str,
        threshold: float,
        vector: IhsanVector,
        composite: float,
        verdict: Literal["APPROVED", "REJECTED"],
        reason: str,
        intent_sha256: str,
        context_sha256: str,
    ) -> FateSeal:
        ts = utc_now_iso()
        payload = {
            "env": env,
            "threshold": threshold,
            "vector": vector.model_dump(),
            "composite": composite,
            "verdict": verdict,
            "reason": reason,
            "intent_sha256": intent_sha256,
            "context_sha256": context_sha256,
            "timestamp": ts,
        }
        seal_id = sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))

        return FateSeal(
            id=seal_id,
            timestamp=ts,
            validator=self.validator,
            env=env,
            threshold=threshold,
            vector=vector,
            composite_score=composite,
            verdict=verdict,
            reason=reason,
            policy={
                "constitution_id": self.policy.constitution_id,
                "constitution_version": str(self.policy.constitution_version),
                "constitution_sha256": self.policy.constitution_sha256,
                "constitution_path": self.policy.constitution_path,
            },
            intent_sha256=intent_sha256,
            context_sha256=context_sha256,
        )

    def _score_intent(self, *, intent: str, context: str) -> Tuple[IhsanVector, str]:
        intent_l = intent.lower()
        ctx_l = context.lower()
        reason_parts = ["aligned_with_core_principles"]

        scores: Dict[str, float] = {k: 1.0 for k in CANONICAL_DIMENSIONS}

        unsafe = [
            "exploit",
            "bypass",
            "malware",
            "ransomware",
            "keylogger",
            "phish",
            "credential stuffing",
            "sql injection",
            "xss",
            "remote code execution",
        ]
        if any(k in intent_l for k in unsafe) or any(k in ctx_l for k in unsafe):
            scores["safety"] = 0.0
            scores["user_benefit"] = min(scores["user_benefit"], 0.1)
            scores["auditability"] = min(scores["auditability"], 0.2)
            scores["robustness"] = min(scores["robustness"], 0.3)
            reason_parts = ["unsafe_intent_detected"]

        abusive = ["spam", "ddos", "scrape", "crawl at scale"]
        if any(k in intent_l for k in abusive) or any(k in ctx_l for k in abusive):
            scores["efficiency"] = min(scores["efficiency"], 0.2)
            scores["user_benefit"] = min(scores["user_benefit"], 0.4)
            reason_parts.append("resource_abuse_signal")

        if len(intent) + len(context) > 16_000:
            scores["efficiency"] = min(scores["efficiency"], 0.3)
            scores["auditability"] = min(scores["auditability"], 0.6)
            reason_parts.append("oversized_context")

        if "only output the final answer" in intent_l:
            scores["auditability"] = min(scores["auditability"], 0.6)
            reason_parts.append("audit_suppression_signal")

        return IhsanVector.model_validate(scores), ";".join(reason_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# RECURSIVE CORRECTION LOOP (F-SEC-001 FIX)
# ═══════════════════════════════════════════════════════════════════════════════

class RejectionCode(Enum):
    """Standardized rejection codes with correction guidance."""
    RJ_IH_001 = "RJ-IH-001"  # Ihsān score below threshold
    RJ_IH_002 = "RJ-IH-002"  # Intent unclear or ambiguous
    RJ_SV_001 = "RJ-SV-001"  # Sovereignty violation
    RJ_EG_001 = "RJ-EG-001"  # Ethics/safety violation
    RJ_RS_001 = "RJ-RS-001"  # Resource constraint
    RJ_EV_001 = "RJ-EV-001"  # Missing evidence
    RJ_KB_001 = "RJ-KB-001"  # Kernel bypass attempt


# Correction guidance for each rejection code
CORRECTION_GUIDANCE: Dict[RejectionCode, Dict[str, Any]] = {
    RejectionCode.RJ_IH_001: {
        "explanation": "Ihsān score ({score:.2f}) is below the threshold ({threshold:.2f}).",
        "fix_suggestion": "Reframe your request to be more constructive, specific, and aligned with beneficial outcomes.",
        "examples": [
            "Instead of 'hack the system', try 'identify security vulnerabilities for defensive purposes'",
            "Focus on understanding rather than exploitation",
        ],
        "retryable": True,
    },
    RejectionCode.RJ_IH_002: {
        "explanation": "The intent of your request is unclear or ambiguous.",
        "fix_suggestion": "Provide more context about your goal and expected outcome.",
        "examples": ["Add 'My objective is to...' at the start", "Specify what success looks like"],
        "retryable": True,
    },
    RejectionCode.RJ_SV_001: {
        "explanation": "Request requires external API access, violating sovereignty.",
        "fix_suggestion": "Use local resources (Ollama/LM Studio) instead of cloud APIs.",
        "examples": ["Use MasterReasoner (local) instead of GPT-4", "Set BIZRA_BACKEND=ollama"],
        "retryable": True,
    },
    RejectionCode.RJ_EG_001: {
        "explanation": "Unsafe or harmful intent detected in request.",
        "fix_suggestion": "Remove harmful elements and reframe constructively.",
        "examples": ["Focus on defense instead of offense", "Consider impact on all stakeholders"],
        "retryable": True,
    },
    RejectionCode.RJ_RS_001: {
        "explanation": "Requested resources exceed available capacity.",
        "fix_suggestion": "Reduce resource requirements or wait for current tasks to complete.",
        "examples": ["Use a smaller model", "Queue for later execution"],
        "retryable": True,
    },
    RejectionCode.RJ_EV_001: {
        "explanation": "High-stakes action requires cryptographic evidence attestation.",
        "fix_suggestion": "Provide a valid genesis receipt or evidence hash.",
        "examples": ["Include 'evidence_hash' in payload", "Generate via genesis_receipt.py"],
        "retryable": True,
    },
    RejectionCode.RJ_KB_001: {
        "explanation": "Detected attempt to bypass kernel security gates.",
        "fix_suggestion": "All actions must flow through the Kernel API.",
        "examples": ["Use /v1/fate/evaluate for all actions"],
        "retryable": False,
    },
}


@dataclass
class CorrectionFeedback:
    """Structured feedback for rejected requests."""
    code: RejectionCode
    explanation: str
    fix_suggestion: str
    examples: List[str]
    retryable: bool
    retry_count: int
    max_retries: int
    composite_score: float
    required_threshold: float
    request_hash: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "explanation": self.explanation,
            "fix_suggestion": self.fix_suggestion,
            "examples": self.examples,
            "retryable": self.retryable,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "composite_score": self.composite_score,
            "required_threshold": self.required_threshold,
            "request_hash": self.request_hash,
            "timestamp": self.timestamp,
        }


class FateEngineWithCorrection(FateEngine):
    """
    Extended FATE Engine with Recursive Correction Loop.
    
    Doesn't just reject - provides structured guidance for alignment.
    Supports bounded retry (max 2 attempts by default).
    """
    
    def __init__(self, *, strict_mode: bool = True, validator: str = "Node0_Sovereign_Kernel"):
        super().__init__(strict_mode=strict_mode, validator=validator)
        self._retry_tracker: Dict[str, int] = {}
        self._cleanup_threshold = 1000
        
        # Create evidence directory
        EVIDENCE_PATH.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.total_evaluations = 0
        self.approvals = 0
        self.rejections = 0
        self.corrections_accepted = 0
    
    def _hash_request(self, intent: str, context: str) -> str:
        """Generate deterministic hash of request."""
        payload = json.dumps({"intent": intent, "context": context}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
    
    def _determine_rejection_code(self, seal: FateSeal, intent: str) -> RejectionCode:
        """Determine the most specific rejection code."""
        reason = seal.reason.lower()
        intent_l = intent.lower()
        
        if "unsafe" in reason or "safety" in reason:
            return RejectionCode.RJ_EG_001
        
        if "oversized" in reason or "resource" in reason:
            return RejectionCode.RJ_RS_001
        
        if any(ext in intent_l for ext in ["openai", "anthropic", "gpt-4", "claude"]):
            return RejectionCode.RJ_SV_001
        
        if seal.composite_score < 0.5:
            return RejectionCode.RJ_IH_002
        
        return RejectionCode.RJ_IH_001
    
    def _build_feedback(
        self,
        code: RejectionCode,
        seal: FateSeal,
        request_hash: str,
        retry_count: int
    ) -> CorrectionFeedback:
        """Build structured correction feedback."""
        guidance = CORRECTION_GUIDANCE.get(code, CORRECTION_GUIDANCE[RejectionCode.RJ_IH_001])
        
        explanation = guidance["explanation"]
        if "{score" in explanation:
            explanation = explanation.format(
                score=seal.composite_score,
                threshold=seal.threshold
            )
        
        return CorrectionFeedback(
            code=code,
            explanation=explanation,
            fix_suggestion=guidance["fix_suggestion"],
            examples=guidance["examples"],
            retryable=guidance["retryable"] and retry_count < MAX_CORRECTION_RETRIES,
            retry_count=retry_count,
            max_retries=MAX_CORRECTION_RETRIES,
            composite_score=seal.composite_score,
            required_threshold=seal.threshold,
            request_hash=request_hash,
        )
    
    def audit_request_with_feedback(
        self,
        *,
        intent: str,
        context: str = ""
    ) -> Tuple[FateSeal, Optional[CorrectionFeedback]]:
        """
        Audit request and provide correction feedback if rejected.
        
        Returns:
            (FateSeal, CorrectionFeedback or None)
        """
        self.total_evaluations += 1
        request_hash = self._hash_request(intent, context)
        retry_count = self._retry_tracker.get(request_hash, 0)
        
        # Get base seal
        seal = self.audit_request(intent=intent, context=context)
        
        feedback = None
        if seal.verdict == "REJECTED":
            self.rejections += 1
            
            code = self._determine_rejection_code(seal, intent)
            feedback = self._build_feedback(code, seal, request_hash, retry_count)
            
            # Track retry
            self._retry_tracker[request_hash] = retry_count + 1
            
            logger.warning(
                f"FATE REJECT: {code.value} | score={seal.composite_score:.3f} | "
                f"retry={retry_count}/{MAX_CORRECTION_RETRIES} | hash={request_hash}"
            )
            
            # Record evidence
            self._record_rejection(seal, feedback)
        else:
            self.approvals += 1
            
            # Check if this was a successful correction
            if request_hash in self._retry_tracker:
                self.corrections_accepted += 1
                del self._retry_tracker[request_hash]
            
            logger.info(
                f"FATE APPROVE: score={seal.composite_score:.3f} | "
                f"env={seal.env} | hash={request_hash}"
            )
        
        # Periodic cleanup
        if len(self._retry_tracker) > self._cleanup_threshold:
            self._retry_tracker.clear()
        
        return seal, feedback
    
    def submit_correction(
        self,
        original_hash: str,
        corrected_intent: str,
        context: str = ""
    ) -> Tuple[FateSeal, Optional[CorrectionFeedback]]:
        """
        Submit a corrected request (via /v1/sape/feedback endpoint).
        
        Args:
            original_hash: Hash of the original rejected request
            corrected_intent: The corrected intent
            context: Updated context
            
        Returns:
            (FateSeal, CorrectionFeedback or None)
        """
        retry_count = self._retry_tracker.get(original_hash, 0)
        
        if retry_count >= MAX_CORRECTION_RETRIES:
            # Create rejection seal for exhausted retries
            seal = FateSeal(
                id="exhausted",
                timestamp=utc_now_iso(),
                validator=self.validator,
                env=self._env(),
                threshold=1.0,
                vector=IhsanVector.model_validate({k: 0.0 for k in CANONICAL_DIMENSIONS}),
                composite_score=0.0,
                verdict="REJECTED",
                reason="max_retries_exhausted",
                policy={
                    "constitution_id": self.policy.constitution_id,
                    "constitution_version": str(self.policy.constitution_version),
                    "constitution_sha256": self.policy.constitution_sha256,
                    "constitution_path": self.policy.constitution_path,
                },
                intent_sha256=sha256_text(corrected_intent),
                context_sha256=sha256_text(context),
            )
            
            feedback = CorrectionFeedback(
                code=RejectionCode.RJ_IH_001,
                explanation=f"Maximum correction retries ({MAX_CORRECTION_RETRIES}) exhausted.",
                fix_suggestion="Please reformulate your request from scratch.",
                examples=[],
                retryable=False,
                retry_count=retry_count,
                max_retries=MAX_CORRECTION_RETRIES,
                composite_score=0.0,
                required_threshold=self.policy.threshold_for_env(self._env()),
                request_hash=original_hash,
            )
            
            return seal, feedback
        
        # Evaluate corrected request
        return self.audit_request_with_feedback(intent=corrected_intent, context=context)
    
    def _record_rejection(self, seal: FateSeal, feedback: CorrectionFeedback) -> None:
        """Record rejection to evidence log."""
        log_file = EVIDENCE_PATH / "rejections.jsonl"
        
        record = {
            "timestamp": seal.timestamp,
            "seal_id": seal.id,
            "verdict": seal.verdict,
            "composite_score": seal.composite_score,
            "threshold": seal.threshold,
            "rejection_code": feedback.code.value,
            "retryable": feedback.retryable,
            "retry_count": feedback.retry_count,
            "request_hash": feedback.request_hash,
        }
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.warning(f"Failed to record rejection: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_evaluations": self.total_evaluations,
            "approvals": self.approvals,
            "rejections": self.rejections,
            "corrections_accepted": self.corrections_accepted,
            "approval_rate": self.approvals / max(1, self.total_evaluations),
            "correction_success_rate": (
                self.corrections_accepted / max(1, self.rejections)
                if self.rejections > 0 else 0
            ),
            "max_retries": MAX_CORRECTION_RETRIES,
            "pending_corrections": len(self._retry_tracker),
        }


# Global instance for the enhanced engine
_fate_engine_with_correction: Optional[FateEngineWithCorrection] = None


def get_fate_engine() -> FateEngineWithCorrection:
    """Get or create the global FATE engine with correction support."""
    global _fate_engine_with_correction
    if _fate_engine_with_correction is None:
        _fate_engine_with_correction = FateEngineWithCorrection()
    return _fate_engine_with_correction
