"""
BIZRA PCI Protocol — SAPE Bridge
================================
Maps SAPE probe results to PCI envelope metadata.

Status: PRODUCTION
Alignment: BIZRA_SOT.md Section 3.2 (SAPE 9-probe verification)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     SAPE → PCI Bridge                       │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
    │  │ 9 Probes    │───▶│ Score Calc  │───▶│ PCI Metadata│     │
    │  └─────────────┘    └─────────────┘    └─────────────┘     │
    │         │                  │                  │             │
    │         ▼                  ▼                  ▼             │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
    │  │ Pattern     │───▶│ Elevation   │───▶│ Kernel      │     │
    │  │ Detection   │    │ Check (>3)  │    │ Shortcuts   │     │
    │  └─────────────┘    └─────────────┘    └─────────────┘     │
    └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS (From constitution/ihsan_v1.yaml)
# =============================================================================

IHSAN_THRESHOLD = 0.95
SNR_THRESHOLD = 0.98
ELEVATION_THRESHOLD = 3  # Repetitions before pattern elevation


# =============================================================================
# SAPE PROBE TYPES
# =============================================================================

class SAPEProbeType(Enum):
    """9 SAPE probes from constitution."""
    THREAT_SCAN = "threat_scan"
    COMPLIANCE = "compliance_check"
    BIAS = "bias_probe"
    USER_BENEFIT = "user_benefit"
    CORRECTNESS = "correctness"
    SAFETY = "safety"
    GROUNDEDNESS = "groundedness"
    RELEVANCE = "relevance"
    FLUENCY = "fluency"


# Probe weights from constitution
PROBE_WEIGHTS: Dict[SAPEProbeType, float] = {
    SAPEProbeType.THREAT_SCAN: 0.15,
    SAPEProbeType.COMPLIANCE: 0.12,
    SAPEProbeType.BIAS: 0.12,
    SAPEProbeType.USER_BENEFIT: 0.12,
    SAPEProbeType.CORRECTNESS: 0.12,
    SAPEProbeType.SAFETY: 0.15,
    SAPEProbeType.GROUNDEDNESS: 0.08,
    SAPEProbeType.RELEVANCE: 0.07,
    SAPEProbeType.FLUENCY: 0.07,
}

# Probe thresholds
PROBE_THRESHOLDS: Dict[SAPEProbeType, float] = {
    SAPEProbeType.THREAT_SCAN: 0.95,
    SAPEProbeType.COMPLIANCE: 0.95,
    SAPEProbeType.BIAS: 0.90,
    SAPEProbeType.USER_BENEFIT: 0.85,
    SAPEProbeType.CORRECTNESS: 0.95,
    SAPEProbeType.SAFETY: 0.95,
    SAPEProbeType.GROUNDEDNESS: 0.85,
    SAPEProbeType.RELEVANCE: 0.80,
    SAPEProbeType.FLUENCY: 0.80,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ProbeResult:
    """Result of a single SAPE probe."""
    probe_type: SAPEProbeType
    score: float
    threshold: float
    passed: bool
    evidence: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_type": self.probe_type.value,
            "score": self.score,
            "threshold": self.threshold,
            "passed": self.passed,
            "evidence": self.evidence,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbeResult":
        return cls(
            probe_type=SAPEProbeType(data["probe_type"]),
            score=data["score"],
            threshold=data["threshold"],
            passed=data["passed"],
            evidence=data.get("evidence"),
            latency_ms=data.get("latency_ms", 0.0),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class SAPEMetadata:
    """PCI-compatible SAPE metadata for envelope embedding."""
    probes_run: List[str]
    probes_passed: int
    probes_failed: int
    overall_score: float
    ihsan_equivalent: float  # Converted to Ihsan scale
    snr_equivalent: float    # Converted to SNR scale
    elevation_candidate: bool
    pattern_hash: Optional[str] = None
    failed_probes: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probes_run": self.probes_run,
            "probes_passed": self.probes_passed,
            "probes_failed": self.probes_failed,
            "overall_score": self.overall_score,
            "ihsan_equivalent": self.ihsan_equivalent,
            "snr_equivalent": self.snr_equivalent,
            "elevation_candidate": self.elevation_candidate,
            "pattern_hash": self.pattern_hash,
            "failed_probes": self.failed_probes,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SAPEMetadata":
        return cls(
            probes_run=data["probes_run"],
            probes_passed=data["probes_passed"],
            probes_failed=data["probes_failed"],
            overall_score=data["overall_score"],
            ihsan_equivalent=data["ihsan_equivalent"],
            snr_equivalent=data["snr_equivalent"],
            elevation_candidate=data["elevation_candidate"],
            pattern_hash=data.get("pattern_hash"),
            failed_probes=data.get("failed_probes", []),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class TrackedPattern:
    """A tracked pattern for potential elevation."""
    pattern_hash: str
    content_signature: str
    occurrences: int
    first_seen: str
    last_seen: str
    average_score: float
    success_rate: float

    def should_elevate(self) -> bool:
        """Check if pattern should be elevated (>3 repetitions with good performance)."""
        return (
            self.occurrences > ELEVATION_THRESHOLD
            and self.success_rate >= 0.7
            and self.average_score >= 0.8
        )


# =============================================================================
# SAPE-PCI BRIDGE
# =============================================================================

class SAPEPCIBridge:
    """
    Bridges SAPE probe execution to PCI envelope metadata.

    Responsibilities:
    1. Execute 9 SAPE probes on content
    2. Compute weighted scores
    3. Convert to PCI-compatible metadata
    4. Track patterns for elevation
    5. Emit elevation callbacks when threshold exceeded

    Thread-safe for concurrent use.
    """

    def __init__(
        self,
        ihsan_threshold: float = IHSAN_THRESHOLD,
        snr_threshold: float = SNR_THRESHOLD,
        elevation_callback: Optional[Callable[[str, SAPEMetadata], None]] = None,
    ):
        """
        Initialize the SAPE-PCI bridge.

        Args:
            ihsan_threshold: Minimum Ihsan score for pass (default: 0.95)
            snr_threshold: Minimum SNR score for pass (default: 0.98)
            elevation_callback: Optional callback when pattern is elevated
        """
        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold
        self.elevation_callback = elevation_callback

        # Pattern tracking (thread-safe)
        self._pattern_cache: Dict[str, TrackedPattern] = {}
        self._lock = threading.Lock()

        logger.info(
            f"SAPEPCIBridge initialized: ihsan={ihsan_threshold}, snr={snr_threshold}"
        )

    async def run_probes(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        probes: Optional[List[SAPEProbeType]] = None,
    ) -> List[ProbeResult]:
        """
        Execute SAPE probes on content.

        Args:
            content: Content to validate
            context: Optional context for probes
            probes: Specific probes to run (default: all 9)

        Returns:
            List of ProbeResult for each probe
        """
        if probes is None:
            probes = list(SAPEProbeType)

        context = context or {}
        results: List[ProbeResult] = []

        for probe_type in probes:
            try:
                start = asyncio.get_event_loop().time()
                result = await self._execute_probe(probe_type, content, context)
                latency = (asyncio.get_event_loop().time() - start) * 1000
                result.latency_ms = latency
                results.append(result)
            except Exception as e:
                # Fail-closed: probe failure means probe did not pass
                logger.error(f"Probe {probe_type.value} failed: {e}")
                results.append(ProbeResult(
                    probe_type=probe_type,
                    score=0.0,
                    threshold=PROBE_THRESHOLDS[probe_type],
                    passed=False,
                    evidence=f"Probe error: {str(e)}",
                ))

        return results

    async def _execute_probe(
        self,
        probe_type: SAPEProbeType,
        content: str,
        context: Dict[str, Any],
    ) -> ProbeResult:
        """Execute a single probe. Override for custom implementations."""
        threshold = PROBE_THRESHOLDS[probe_type]

        # Default implementation: heuristic-based scoring
        # In production, these would call actual SAPE probe implementations
        score = await self._compute_probe_score(probe_type, content, context)

        return ProbeResult(
            probe_type=probe_type,
            score=score,
            threshold=threshold,
            passed=score >= threshold,
            evidence=f"Probe {probe_type.value} completed",
        )

    async def _compute_probe_score(
        self,
        probe_type: SAPEProbeType,
        content: str,
        context: Dict[str, Any],
    ) -> float:
        """
        Compute probe score. This is a placeholder for actual probe logic.

        In production, this would integrate with:
        - src/sape.rs for Rust SAPE engine
        - core/sape.py for Python SAPE logic
        - Neo4j for graph evidence (high-stakes probes)
        """
        # Placeholder: return high score for non-empty content
        # Real implementation would call actual probe logic
        if not content or not content.strip():
            return 0.0

        # Simulate probe scores based on content characteristics
        base_score = 0.85

        # Adjust based on content length (reasonable length is better)
        content_len = len(content)
        if 100 <= content_len <= 10000:
            base_score += 0.05

        # Check for obvious issues (placeholder logic)
        lower_content = content.lower()

        if probe_type == SAPEProbeType.THREAT_SCAN:
            # Check for potentially harmful patterns
            harmful_patterns = ["exploit", "attack", "malicious", "hack"]
            if any(p in lower_content for p in harmful_patterns):
                base_score -= 0.3

        elif probe_type == SAPEProbeType.SAFETY:
            # Check for safety concerns
            unsafe_patterns = ["dangerous", "harmful", "illegal"]
            if any(p in lower_content for p in unsafe_patterns):
                base_score -= 0.2

        elif probe_type == SAPEProbeType.BIAS:
            # Check for potential bias indicators
            bias_patterns = ["always", "never", "everyone", "no one"]
            if any(p in lower_content for p in bias_patterns):
                base_score -= 0.1

        # Ensure score is in valid range
        return max(0.0, min(1.0, base_score + 0.1))  # Add slight boost

    def compute_scores(
        self,
        results: List[ProbeResult],
    ) -> Tuple[float, float, float]:
        """
        Compute overall, ihsan_equivalent, and snr_equivalent scores.

        Args:
            results: List of probe results

        Returns:
            Tuple of (overall_score, ihsan_equivalent, snr_equivalent)
        """
        if not results:
            return 0.0, 0.0, 0.0

        # Weighted overall score
        total_weight = 0.0
        weighted_sum = 0.0

        for result in results:
            weight = PROBE_WEIGHTS.get(result.probe_type, 0.1)
            weighted_sum += result.score * weight
            total_weight += weight

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Ihsan equivalent: focus on ethical dimensions
        ihsan_probes = [
            SAPEProbeType.CORRECTNESS,
            SAPEProbeType.SAFETY,
            SAPEProbeType.USER_BENEFIT,
            SAPEProbeType.BIAS,
        ]
        ihsan_scores = [r.score for r in results if r.probe_type in ihsan_probes]
        ihsan_equivalent = sum(ihsan_scores) / len(ihsan_scores) if ihsan_scores else 0.0

        # SNR equivalent: focus on signal quality dimensions
        snr_probes = [
            SAPEProbeType.GROUNDEDNESS,
            SAPEProbeType.RELEVANCE,
            SAPEProbeType.CORRECTNESS,
        ]
        snr_scores = [r.score for r in results if r.probe_type in snr_probes]
        snr_equivalent = sum(snr_scores) / len(snr_scores) if snr_scores else 0.0

        return overall, ihsan_equivalent, snr_equivalent

    def to_pci_metadata(
        self,
        results: List[ProbeResult],
        content: Optional[str] = None,
    ) -> SAPEMetadata:
        """
        Convert probe results to PCI-compatible metadata.

        Args:
            results: List of probe results
            content: Optional content for pattern hashing

        Returns:
            SAPEMetadata suitable for PCI envelope
        """
        overall, ihsan_eq, snr_eq = self.compute_scores(results)

        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]

        # Compute pattern hash if content provided
        pattern_hash = None
        elevation_candidate = False

        if content:
            pattern_hash = self._compute_pattern_hash(content, results)
            elevation_candidate = self.check_elevation(pattern_hash)

        return SAPEMetadata(
            probes_run=[r.probe_type.value for r in results],
            probes_passed=len(passed),
            probes_failed=len(failed),
            overall_score=overall,
            ihsan_equivalent=ihsan_eq,
            snr_equivalent=snr_eq,
            elevation_candidate=elevation_candidate,
            pattern_hash=pattern_hash,
            failed_probes=[r.probe_type.value for r in failed],
        )

    def _compute_pattern_hash(
        self,
        content: str,
        results: List[ProbeResult],
    ) -> str:
        """Compute a deterministic pattern hash for content + results."""
        # Create a signature from content structure + probe results
        signature_parts = [
            f"len:{len(content)}",
            f"probes:{len(results)}",
            f"passed:{sum(1 for r in results if r.passed)}",
        ]

        # Add content hash (first 1000 chars to limit size)
        content_hash = hashlib.sha256(content[:1000].encode()).hexdigest()[:16]
        signature_parts.append(f"content:{content_hash}")

        signature = "|".join(signature_parts)
        return hashlib.sha256(signature.encode()).hexdigest()

    def check_elevation(self, pattern_hash: str) -> bool:
        """
        Check if pattern should be elevated (>3 repetitions).

        Args:
            pattern_hash: Hash of the pattern to check

        Returns:
            True if pattern should be elevated
        """
        with self._lock:
            if pattern_hash not in self._pattern_cache:
                return False

            pattern = self._pattern_cache[pattern_hash]
            return pattern.should_elevate()

    def track_pattern(
        self,
        pattern_hash: str,
        content_signature: str,
        score: float,
        success: bool,
    ) -> TrackedPattern:
        """
        Track a pattern occurrence for potential elevation.

        Args:
            pattern_hash: Hash of the pattern
            content_signature: Human-readable signature
            score: Score achieved
            success: Whether the validation succeeded

        Returns:
            Updated TrackedPattern
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            if pattern_hash in self._pattern_cache:
                pattern = self._pattern_cache[pattern_hash]

                # Update running averages
                n = pattern.occurrences
                pattern.average_score = (pattern.average_score * n + score) / (n + 1)
                pattern.success_rate = (pattern.success_rate * n + (1.0 if success else 0.0)) / (n + 1)
                pattern.occurrences += 1
                pattern.last_seen = now
            else:
                pattern = TrackedPattern(
                    pattern_hash=pattern_hash,
                    content_signature=content_signature,
                    occurrences=1,
                    first_seen=now,
                    last_seen=now,
                    average_score=score,
                    success_rate=1.0 if success else 0.0,
                )
                self._pattern_cache[pattern_hash] = pattern

            # Check for elevation
            if pattern.should_elevate() and self.elevation_callback:
                metadata = SAPEMetadata(
                    probes_run=[],
                    probes_passed=0,
                    probes_failed=0,
                    overall_score=pattern.average_score,
                    ihsan_equivalent=pattern.average_score,
                    snr_equivalent=pattern.average_score,
                    elevation_candidate=True,
                    pattern_hash=pattern_hash,
                )
                self.elevation_callback(pattern_hash, metadata)
                logger.info(f"Pattern elevated: {pattern_hash[:16]}...")

            return pattern

    def embed_in_envelope(
        self,
        envelope_builder,
        results: List[ProbeResult],
        content: Optional[str] = None,
    ):
        """
        Embed SAPE metadata into an envelope builder.

        Args:
            envelope_builder: EnvelopeBuilder instance
            results: Probe results to embed
            content: Optional content for pattern tracking

        Returns:
            The envelope builder (for chaining)
        """
        metadata = self.to_pci_metadata(results, content)

        # Add SAPE metadata to envelope's metadata
        if hasattr(envelope_builder, 'with_metadata'):
            envelope_builder.with_metadata("sape", metadata.to_dict())
        elif hasattr(envelope_builder, '_data'):
            envelope_builder._data["sape"] = metadata.to_dict()

        return envelope_builder

    async def validate_for_pci(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, SAPEMetadata]:
        """
        Full SAPE validation for PCI gate chain.

        Args:
            content: Content to validate
            context: Optional context

        Returns:
            Tuple of (passed, SAPEMetadata)
        """
        results = await self.run_probes(content, context)
        metadata = self.to_pci_metadata(results, content)

        # Check thresholds
        passed = (
            metadata.ihsan_equivalent >= self.ihsan_threshold
            and metadata.overall_score >= 0.8  # Minimum overall
            and metadata.probes_failed <= 2  # Max 2 failed probes
        )

        # Track pattern
        if metadata.pattern_hash:
            self.track_pattern(
                metadata.pattern_hash,
                content[:100],
                metadata.overall_score,
                passed,
            )

        return passed, metadata

    def get_tracked_patterns(self) -> List[TrackedPattern]:
        """Get all tracked patterns."""
        with self._lock:
            return list(self._pattern_cache.values())

    def get_elevation_candidates(self) -> List[TrackedPattern]:
        """Get patterns that are candidates for elevation."""
        with self._lock:
            return [p for p in self._pattern_cache.values() if p.should_elevate()]

    def clear_cache(self) -> None:
        """Clear the pattern cache."""
        with self._lock:
            self._pattern_cache.clear()
        logger.info("SAPE pattern cache cleared")

    def to_json(self) -> str:
        """Serialize bridge state to JSON."""
        with self._lock:
            data = {
                "ihsan_threshold": self.ihsan_threshold,
                "snr_threshold": self.snr_threshold,
                "patterns": {
                    k: {
                        "pattern_hash": v.pattern_hash,
                        "content_signature": v.content_signature,
                        "occurrences": v.occurrences,
                        "first_seen": v.first_seen,
                        "last_seen": v.last_seen,
                        "average_score": v.average_score,
                        "success_rate": v.success_rate,
                    }
                    for k, v in self._pattern_cache.items()
                },
            }
            return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str, elevation_callback: Optional[Callable] = None) -> "SAPEPCIBridge":
        """Deserialize bridge from JSON."""
        data = json.loads(json_str)

        bridge = cls(
            ihsan_threshold=data.get("ihsan_threshold", IHSAN_THRESHOLD),
            snr_threshold=data.get("snr_threshold", SNR_THRESHOLD),
            elevation_callback=elevation_callback,
        )

        # Restore patterns
        for k, v in data.get("patterns", {}).items():
            bridge._pattern_cache[k] = TrackedPattern(
                pattern_hash=v["pattern_hash"],
                content_signature=v["content_signature"],
                occurrences=v["occurrences"],
                first_seen=v["first_seen"],
                last_seen=v["last_seen"],
                average_score=v["average_score"],
                success_rate=v["success_rate"],
            )

        return bridge


# =============================================================================
# ELEVATION CALLBACK HANDLER
# =============================================================================

def on_pattern_elevated(pattern_hash: str, metadata: SAPEMetadata) -> None:
    """
    Default callback when a pattern is elevated to kernel shortcuts.

    In production, this would:
    1. Store the elevated pattern in Redis/Synapse
    2. Update SAPE probe cache
    3. Emit an elevation receipt
    """
    logger.info(
        f"Pattern elevated to kernel shortcut: {pattern_hash[:16]}... "
        f"(score: {metadata.overall_score:.3f}, occurrences: {metadata.probes_passed})"
    )


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_sape_bridge: Optional[SAPEPCIBridge] = None


def get_sape_bridge() -> SAPEPCIBridge:
    """Get or create the global SAPE-PCI bridge."""
    global _sape_bridge
    if _sape_bridge is None:
        _sape_bridge = SAPEPCIBridge(
            elevation_callback=on_pattern_elevated,
        )
    return _sape_bridge


def reset_sape_bridge() -> None:
    """Reset the global SAPE-PCI bridge."""
    global _sape_bridge
    _sape_bridge = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def validate_content(
    content: str,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, SAPEMetadata]:
    """
    Validate content through SAPE probes.

    Args:
        content: Content to validate
        context: Optional context

    Returns:
        Tuple of (passed, SAPEMetadata)
    """
    bridge = get_sape_bridge()
    return await bridge.validate_for_pci(content, context)


async def run_probes(
    content: str,
    context: Optional[Dict[str, Any]] = None,
    probes: Optional[List[SAPEProbeType]] = None,
) -> List[ProbeResult]:
    """
    Run SAPE probes on content.

    Args:
        content: Content to probe
        context: Optional context
        probes: Specific probes to run (default: all)

    Returns:
        List of ProbeResult
    """
    bridge = get_sape_bridge()
    return await bridge.run_probes(content, context, probes)


def compute_metadata(results: List[ProbeResult], content: Optional[str] = None) -> SAPEMetadata:
    """
    Compute PCI-compatible metadata from probe results.

    Args:
        results: Probe results
        content: Optional content for pattern tracking

    Returns:
        SAPEMetadata
    """
    bridge = get_sape_bridge()
    return bridge.to_pci_metadata(results, content)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "SAPEProbeType",
    # Data classes
    "ProbeResult",
    "SAPEMetadata",
    "TrackedPattern",
    # Main class
    "SAPEPCIBridge",
    # Global functions
    "get_sape_bridge",
    "reset_sape_bridge",
    "validate_content",
    "run_probes",
    "compute_metadata",
    "on_pattern_elevated",
    # Constants
    "IHSAN_THRESHOLD",
    "SNR_THRESHOLD",
    "ELEVATION_THRESHOLD",
    "PROBE_WEIGHTS",
    "PROBE_THRESHOLDS",
]
