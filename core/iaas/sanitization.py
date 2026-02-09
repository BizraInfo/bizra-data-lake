"""
Sanitization Engine — Privacy, Ethics, and Risk Removal

Standing on Giants:
- PII Detection (Microsoft Presidio patterns)
- Toxicity Detection (Perspective API heuristics)
- Ethics Filtering (Constitutional AI principles)
- DDAGI Constitution Article 5: Data Sovereignty

"Sanitization ensures that the data is free from privacy risks,
 toxic content, and ethical violations before training."
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of Personally Identifiable Information."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    API_KEY = "api_key"  # nosec B105 — PII category enum, not an actual key
    PASSWORD = "password"  # nosec B105 — PII category enum, not an actual password


class ToxicityType(Enum):
    """Categories of toxic content."""

    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    DANGEROUS = "dangerous"
    PROFANITY = "profanity"


@dataclass
class PIIMatch:
    """A detected PII instance."""

    pii_type: PIIType
    start: int
    end: int
    text: str
    confidence: float
    replacement: str = ""


@dataclass
class SanitizationResult:
    """Result of sanitization operation."""

    original_count: int
    sanitized_count: int
    removed_indices: List[int]
    pii_detections: Dict[int, List[PIIMatch]]
    toxicity_scores: Dict[int, float]
    ethics_violations: Dict[int, List[str]]
    method: str

    @property
    def pii_density(self) -> float:
        """PII detections per sample."""
        total_pii = sum(len(matches) for matches in self.pii_detections.values())
        return total_pii / max(self.original_count, 1)

    @property
    def mean_toxicity(self) -> float:
        """Average toxicity score."""
        if not self.toxicity_scores:
            return 0.0
        return sum(self.toxicity_scores.values()) / len(self.toxicity_scores)


class PIIAnonymizer:
    """
    PII Detection and Anonymization Engine.

    Implements pattern-based detection inspired by Microsoft Presidio,
    with deterministic anonymization for reproducibility.

    Anonymization strategies:
    - MASK: Replace with asterisks (e.g., ****@****.com)
    - HASH: Replace with deterministic hash prefix
    - REDACT: Replace with [PII_TYPE] token
    - SYNTHETIC: Replace with synthetic equivalent
    """

    # Regex patterns for PII detection
    PATTERNS = {
        PIIType.EMAIL: re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ),
        PIIType.PHONE: re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
        ),
        PIIType.SSN: re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
        PIIType.CREDIT_CARD: re.compile(
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"
        ),
        PIIType.IP_ADDRESS: re.compile(
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
        ),
        PIIType.API_KEY: re.compile(
            r"\b(?:sk-|pk-|api[_-]?key[=:\s]+)[A-Za-z0-9_-]{20,}\b", re.IGNORECASE
        ),
        PIIType.PASSWORD: re.compile(
            r"(?:password|passwd|pwd)[=:\s]+[^\s]{6,}", re.IGNORECASE
        ),
    }

    def __init__(
        self,
        anonymization_strategy: str = "redact",
        hash_salt: str = "bizra_pii_salt_v1",
        detect_names: bool = False,  # Requires NER model
    ):
        self.strategy = anonymization_strategy
        self.hash_salt = hash_salt
        self.detect_names = detect_names

    def _hash_pii(self, text: str) -> str:
        """Generate deterministic hash for PII."""
        salted = f"{self.hash_salt}:{text}"
        return hashlib.sha256(salted.encode()).hexdigest()[:12]

    def _anonymize(self, match: PIIMatch) -> str:
        """Generate replacement for PII match."""
        if self.strategy == "mask":
            # Preserve structure with asterisks
            if match.pii_type == PIIType.EMAIL:
                return "****@****.***"
            elif match.pii_type == PIIType.PHONE:
                return "***-***-****"
            elif match.pii_type == PIIType.SSN:
                return "***-**-****"
            elif match.pii_type == PIIType.CREDIT_CARD:
                return "****-****-****-****"
            else:
                return "*" * len(match.text)

        elif self.strategy == "hash":
            # Deterministic hash prefix
            hash_prefix = self._hash_pii(match.text)
            return f"[{match.pii_type.value.upper()}_{hash_prefix}]"

        elif self.strategy == "redact":
            # Simple token replacement
            return f"[{match.pii_type.value.upper()}]"

        elif self.strategy == "synthetic":
            # Synthetic equivalents
            synthetics = {
                PIIType.EMAIL: "user@example.com",
                PIIType.PHONE: "555-000-0000",
                PIIType.SSN: "000-00-0000",
                PIIType.CREDIT_CARD: "4111-1111-1111-1111",
                PIIType.IP_ADDRESS: "192.0.2.1",
                PIIType.API_KEY: "sk-synthetic-key-placeholder",
                PIIType.PASSWORD: "[REDACTED_PASSWORD]",
            }
            return synthetics.get(match.pii_type, f"[{match.pii_type.value.upper()}]")

        return f"[{match.pii_type.value.upper()}]"

    def detect_pii(self, text: str) -> List[PIIMatch]:
        """Detect all PII in text."""
        matches = []

        for pii_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                pii_match = PIIMatch(
                    pii_type=pii_type,
                    start=match.start(),
                    end=match.end(),
                    text=match.group(),
                    confidence=0.95,  # Pattern-based detection has high confidence
                )
                pii_match.replacement = self._anonymize(pii_match)
                matches.append(pii_match)

        # Sort by position (descending) for safe replacement
        matches.sort(key=lambda m: m.start, reverse=True)
        return matches

    def anonymize_text(self, text: str) -> Tuple[str, List[PIIMatch]]:
        """Detect and anonymize PII in text."""
        matches = self.detect_pii(text)

        anonymized = text
        for match in matches:
            anonymized = (
                anonymized[: match.start] + match.replacement + anonymized[match.end :]
            )

        return anonymized, matches

    def process_batch(
        self,
        texts: List[str],
        remove_if_pii: bool = False,
    ) -> SanitizationResult:
        """Process batch of texts for PII."""
        n = len(texts)
        pii_detections = {}
        removed_indices = []

        for i, text in enumerate(texts):
            _, matches = self.anonymize_text(text)
            if matches:
                pii_detections[i] = matches
                if remove_if_pii:
                    removed_indices.append(i)

        logger.info(
            f"PII scan: {n} texts, {len(pii_detections)} with PII, {len(removed_indices)} removed"
        )

        return SanitizationResult(
            original_count=n,
            sanitized_count=n - len(removed_indices),
            removed_indices=removed_indices,
            pii_detections=pii_detections,
            toxicity_scores={},
            ethics_violations={},
            method="pii_anonymization",
        )


class ToxicityDetector:
    """
    Toxicity Detection Engine.

    Implements heuristic-based detection inspired by Perspective API,
    using keyword matching and pattern analysis.

    For production use, integrate with:
    - Perspective API
    - OpenAI Moderation API
    - Local toxicity classifier (e.g., Detoxify)
    """

    # Toxicity keyword patterns (simplified, extend for production)
    TOXICITY_PATTERNS = {
        ToxicityType.PROFANITY: [
            r"\b(?:fuck|shit|damn|ass|bitch|bastard|crap)\b",
        ],
        ToxicityType.HATE_SPEECH: [
            r"\b(?:hate|kill|die|death to)\s+(?:all|every)?\s*(?:\w+s)\b",
        ],
        ToxicityType.VIOLENCE: [
            r"\b(?:kill|murder|attack|destroy|bomb|shoot|stab)\s+(?:you|them|him|her|people)\b",
        ],
        ToxicityType.SELF_HARM: [
            r"\b(?:suicide|self[- ]?harm|cut myself|end my life)\b",
        ],
    }

    def __init__(
        self,
        max_toxicity_threshold: float = 0.5,
        toxicity_fn: Optional[Any] = None,  # External classifier
    ):
        self.max_toxicity = max_toxicity_threshold
        self.toxicity_fn = toxicity_fn

        # Compile patterns
        self.compiled_patterns = {}
        for tox_type, patterns in self.TOXICITY_PATTERNS.items():
            self.compiled_patterns[tox_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def _heuristic_toxicity(self, text: str) -> Tuple[float, List[ToxicityType]]:
        """Estimate toxicity using heuristics."""
        detected_types = []
        max_score = 0.0

        text_lower = text.lower()

        for tox_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    detected_types.append(tox_type)
                    # Weight by severity
                    weights = {
                        ToxicityType.PROFANITY: 0.3,
                        ToxicityType.HATE_SPEECH: 0.9,
                        ToxicityType.VIOLENCE: 0.8,
                        ToxicityType.SELF_HARM: 0.7,
                        ToxicityType.HARASSMENT: 0.6,
                        ToxicityType.SEXUAL: 0.5,
                        ToxicityType.DANGEROUS: 0.7,
                    }
                    max_score = max(max_score, weights.get(tox_type, 0.5))
                    break  # One match per type is enough

        return max_score, detected_types

    def detect_toxicity(self, text: str) -> Tuple[float, List[ToxicityType]]:
        """Detect toxicity in text."""
        if self.toxicity_fn:
            # Use external classifier if available
            try:
                score = self.toxicity_fn(text)
                return score, []
            except Exception:
                pass

        return self._heuristic_toxicity(text)

    def process_batch(
        self,
        texts: List[str],
    ) -> SanitizationResult:
        """Process batch for toxicity detection."""
        n = len(texts)
        toxicity_scores = {}
        removed_indices = []

        for i, text in enumerate(texts):
            score, _ = self.detect_toxicity(text)
            toxicity_scores[i] = score
            if score > self.max_toxicity:
                removed_indices.append(i)

        logger.info(
            f"Toxicity scan: {n} texts, {len(removed_indices)} toxic "
            f"(threshold: {self.max_toxicity})"
        )

        return SanitizationResult(
            original_count=n,
            sanitized_count=n - len(removed_indices),
            removed_indices=removed_indices,
            pii_detections={},
            toxicity_scores=toxicity_scores,
            ethics_violations={},
            method="toxicity_detection",
        )


class EthicsFilter:
    """
    Ethics Compliance Filter.

    Implements Constitutional AI principles for data sanitization:
    - Harm avoidance
    - Truthfulness
    - Fairness
    - Privacy respect
    - Autonomy preservation

    Per DDAGI Constitution Article 5:
    "Data sovereignty means individual control over personal data,
     with explicit consent for any processing beyond direct service."
    """

    # Ethics violation patterns
    ETHICS_PATTERNS = {
        "manipulation": [
            r"\b(?:trick|deceive|manipulate|exploit|scam|fraud)\b.*\b(?:people|users|victims|customers)\b",
        ],
        "harm_instruction": [
            r"\b(?:how to|instructions? for|guide to)\b.*\b(?:hack|steal|hurt|harm|kill|attack)\b",
        ],
        "discrimination": [
            r"\b(?:only|just|never)\s+(?:for|allow|hire|accept)\s+(?:\w+)\s+(?:people|men|women|race|religion)\b",
        ],
        "illegal_activity": [
            r"\b(?:how to|buy|sell|make)\b.*\b(?:drugs|weapons|explosives|poison)\b",
        ],
        "consent_violation": [
            r"\b(?:without|bypass|ignore)\s+(?:consent|permission|approval)\b",
        ],
    }

    def __init__(
        self,
        strict_mode: bool = True,
        custom_rules: Optional[Dict[str, List[str]]] = None,
    ):
        self.strict_mode = strict_mode
        self.rules = {**self.ETHICS_PATTERNS}
        if custom_rules:
            self.rules.update(custom_rules)

        # Compile patterns
        self.compiled_rules = {}
        for category, patterns in self.rules.items():
            self.compiled_rules[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def check_ethics(self, text: str) -> List[str]:
        """Check text for ethics violations."""
        violations = []

        for category, patterns in self.compiled_rules.items():
            for pattern in patterns:
                if pattern.search(text):
                    violations.append(category)
                    break  # One violation per category

        return violations

    def process_batch(
        self,
        texts: List[str],
    ) -> SanitizationResult:
        """Process batch for ethics compliance."""
        n = len(texts)
        ethics_violations = {}
        removed_indices = []

        for i, text in enumerate(texts):
            violations = self.check_ethics(text)
            if violations:
                ethics_violations[i] = violations
                if self.strict_mode:
                    removed_indices.append(i)

        logger.info(
            f"Ethics scan: {n} texts, {len(ethics_violations)} with violations, "
            f"{len(removed_indices)} removed"
        )

        return SanitizationResult(
            original_count=n,
            sanitized_count=n - len(removed_indices),
            removed_indices=removed_indices,
            pii_detections={},
            toxicity_scores={},
            ethics_violations=ethics_violations,
            method="ethics_filter",
        )


class SanitizationEngine:
    """
    Unified Sanitization Pipeline.

    Combines:
    1. PII Detection & Anonymization
    2. Toxicity Detection
    3. Ethics Compliance

    Produces sanitization scores for IaaS quality assessment.
    """

    def __init__(
        self,
        enable_pii: bool = True,
        enable_toxicity: bool = True,
        enable_ethics: bool = True,
        pii_strategy: str = "redact",
        max_toxicity: float = 0.5,
        ethics_strict: bool = True,
    ):
        self.enable_pii = enable_pii
        self.enable_toxicity = enable_toxicity
        self.enable_ethics = enable_ethics

        self.pii_anonymizer = (
            PIIAnonymizer(anonymization_strategy=pii_strategy) if enable_pii else None
        )

        self.toxicity_detector = (
            ToxicityDetector(max_toxicity_threshold=max_toxicity)
            if enable_toxicity
            else None
        )

        self.ethics_filter = (
            EthicsFilter(strict_mode=ethics_strict) if enable_ethics else None
        )

    def sanitize(
        self,
        texts: List[str],
        anonymize_pii: bool = True,
        remove_toxic: bool = True,
        remove_unethical: bool = True,
    ) -> Tuple[List[str], List[int], Dict[str, SanitizationResult]]:
        """
        Run full sanitization pipeline.

        Returns:
            - Sanitized texts (with PII anonymized)
            - Indices to keep
            - Results from each sanitizer
        """
        n = len(texts)
        keep_set = set(range(n))
        sanitized_texts = list(texts)  # Copy for modification
        results = {}

        # PII Detection & Anonymization
        if self.enable_pii and self.pii_anonymizer:
            pii_result = self.pii_anonymizer.process_batch(texts, remove_if_pii=False)
            results["pii"] = pii_result

            if anonymize_pii:
                for idx, matches in pii_result.pii_detections.items():
                    if idx in keep_set:
                        sanitized_texts[idx], _ = self.pii_anonymizer.anonymize_text(
                            texts[idx]
                        )

        # Toxicity Detection
        if self.enable_toxicity and self.toxicity_detector:
            toxicity_result = self.toxicity_detector.process_batch(texts)
            results["toxicity"] = toxicity_result

            if remove_toxic:
                keep_set -= set(toxicity_result.removed_indices)

        # Ethics Compliance
        if self.enable_ethics and self.ethics_filter:
            ethics_result = self.ethics_filter.process_batch(texts)
            results["ethics"] = ethics_result

            if remove_unethical:
                keep_set -= set(ethics_result.removed_indices)

        # Compute aggregate metrics
        pii_density = results.get(
            "pii", SanitizationResult(n, n, [], {}, {}, {}, "")
        ).pii_density
        mean_toxicity = results.get(
            "toxicity", SanitizationResult(n, n, [], {}, {}, {}, "")
        ).mean_toxicity
        ethics_violations_count = len(
            results.get(
                "ethics", SanitizationResult(n, n, [], {}, {}, {}, "")
            ).ethics_violations
        )

        logger.info(
            f"Sanitization complete: {n} -> {len(keep_set)} "
            f"(PII density: {pii_density:.4f}, toxicity: {mean_toxicity:.3f}, "
            f"ethics violations: {ethics_violations_count})"
        )

        return sanitized_texts, sorted(keep_set), results

    def compute_sanitization_score(
        self,
        texts: List[str],
    ) -> Tuple[float, float, float, bool]:
        """
        Compute sanitization metrics for IaaS scoring.

        Returns:
            - pii_density
            - toxicity_score
            - bias_score (placeholder)
            - ethics_compliance
        """
        _, _, results = self.sanitize(
            texts,
            anonymize_pii=False,
            remove_toxic=False,
            remove_unethical=False,
        )

        pii_density = results.get(
            "pii", SanitizationResult(len(texts), len(texts), [], {}, {}, {}, "")
        ).pii_density
        toxicity_score = results.get(
            "toxicity", SanitizationResult(len(texts), len(texts), [], {}, {}, {}, "")
        ).mean_toxicity
        ethics_violations = len(
            results.get(
                "ethics", SanitizationResult(len(texts), len(texts), [], {}, {}, {}, "")
            ).ethics_violations
        )

        ethics_compliance = ethics_violations == 0
        bias_score = 0.0  # Placeholder for bias detection

        return pii_density, toxicity_score, bias_score, ethics_compliance
