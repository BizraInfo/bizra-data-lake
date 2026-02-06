"""
Risk Management — Cascading Risk Mitigation with PMBOK Integration

Implements PMBOK Risk Management with Ihsān principles:
- Risk identification and assessment
- Cascading impact analysis
- Mitigation strategy selection
- Constitutional risk handling

Standing on Giants: PMBOK + FAIR + Constitutional AI
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
)

logger = logging.getLogger(__name__)


class RiskCategory(str, Enum):
    """Risk categories aligned with PMBOK."""

    TECHNICAL = "technical"
    SECURITY = "security"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    CONSTITUTIONAL = "constitutional"  # Ihsān violations


class RiskSeverity(str, Enum):
    """Risk severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    CONSTITUTIONAL = "constitutional"  # Never acceptable


class RiskStatus(str, Enum):
    """Risk status."""

    IDENTIFIED = "identified"
    ASSESSED = "assessed"
    MITIGATING = "mitigating"
    MITIGATED = "mitigated"
    ACCEPTED = "accepted"
    MATERIALIZED = "materialized"


class MitigationStrategy(str, Enum):
    """Risk mitigation strategies (PMBOK)."""

    AVOID = "avoid"  # Eliminate the threat
    TRANSFER = "transfer"  # Shift to third party
    MITIGATE = "mitigate"  # Reduce probability/impact
    ACCEPT = "accept"  # Acknowledge and monitor
    ESCALATE = "escalate"  # Push to higher authority


@dataclass
class Risk:
    """A risk entry."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    category: RiskCategory = RiskCategory.TECHNICAL
    severity: RiskSeverity = RiskSeverity.MEDIUM
    status: RiskStatus = RiskStatus.IDENTIFIED

    # Quantitative assessment
    probability: float = 0.5  # 0-1
    impact: float = 0.5  # 0-1

    # Cascading relationships
    triggers: Set[str] = field(default_factory=set)  # Risks that can trigger this
    cascades_to: Set[str] = field(default_factory=set)  # Risks this can trigger

    # Mitigation
    strategy: Optional[MitigationStrategy] = None
    mitigation_plan: str = ""
    owner: str = ""

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def risk_score(self) -> float:
        """Calculate risk score (probability × impact)."""
        return self.probability * self.impact

    @property
    def priority_score(self) -> float:
        """Calculate priority score with severity weighting."""
        severity_weights = {
            RiskSeverity.LOW: 1.0,
            RiskSeverity.MEDIUM: 2.0,
            RiskSeverity.HIGH: 3.0,
            RiskSeverity.CRITICAL: 4.0,
            RiskSeverity.CONSTITUTIONAL: 10.0,  # Always highest priority
        }
        return self.risk_score * severity_weights.get(self.severity, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "probability": self.probability,
            "impact": self.impact,
            "risk_score": self.risk_score,
            "priority_score": self.priority_score,
            "triggers": list(self.triggers),
            "cascades_to": list(self.cascades_to),
            "strategy": self.strategy.value if self.strategy else None,
            "mitigation_plan": self.mitigation_plan,
            "owner": self.owner,
        }


@dataclass
class CascadeAnalysis:
    """Result of cascading risk analysis."""

    source_risk: str
    affected_risks: List[str]
    total_impact: float
    cascade_depth: int
    critical_path: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_risk": self.source_risk,
            "affected_risks": self.affected_risks,
            "total_impact": self.total_impact,
            "cascade_depth": self.cascade_depth,
            "critical_path": self.critical_path,
        }


class RiskManager:
    """
    PMBOK-aligned Risk Manager with Constitutional Constraints.

    Implements:
    - Risk identification and assessment
    - Cascading impact analysis
    - Mitigation strategy selection
    - Ihsān-based risk prioritization
    """

    def __init__(
        self,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    ):
        self.ihsan_threshold = ihsan_threshold
        self._risks: Dict[str, Risk] = {}
        self._cascade_cache: Dict[str, CascadeAnalysis] = {}

        # Register standard BIZRA risks
        self._register_standard_risks()

    def _register_standard_risks(self) -> None:
        """Register standard BIZRA ecosystem risks."""
        standard_risks = [
            Risk(
                id="SEC-001",
                name="Cryptographic Key Compromise",
                description="Ed25519/Dilithium keys exposed or compromised",
                category=RiskCategory.SECURITY,
                severity=RiskSeverity.CONSTITUTIONAL,
                probability=0.1,
                impact=1.0,
                strategy=MitigationStrategy.AVOID,
                mitigation_plan="Hardware security modules, key rotation, zero-trust architecture",
            ),
            Risk(
                id="SEC-002",
                name="Ihsān Threshold Bypass",
                description="Constitutional constraint bypassed or weakened",
                category=RiskCategory.CONSTITUTIONAL,
                severity=RiskSeverity.CONSTITUTIONAL,
                probability=0.05,
                impact=1.0,
                strategy=MitigationStrategy.AVOID,
                mitigation_plan="Compile-time constants, immutable genes, formal verification",
            ),
            Risk(
                id="PERF-001",
                name="SNR Degradation",
                description="Signal-to-noise ratio falls below threshold",
                category=RiskCategory.PERFORMANCE,
                severity=RiskSeverity.HIGH,
                probability=0.2,
                impact=0.7,
                strategy=MitigationStrategy.MITIGATE,
                mitigation_plan="Continuous SNR monitoring, quality gates, data filtering",
            ),
            Risk(
                id="INT-001",
                name="Federation Consensus Failure",
                description="BFT consensus cannot reach agreement",
                category=RiskCategory.INTEGRATION,
                severity=RiskSeverity.HIGH,
                probability=0.15,
                impact=0.8,
                strategy=MitigationStrategy.MITIGATE,
                mitigation_plan="Fallback to local inference, timeout handling, leader election",
            ),
            Risk(
                id="OPS-001",
                name="Self-Healing Failure",
                description="Autopoietic loop fails to recover from error",
                category=RiskCategory.OPERATIONAL,
                severity=RiskSeverity.MEDIUM,
                probability=0.25,
                impact=0.5,
                strategy=MitigationStrategy.MITIGATE,
                mitigation_plan="Emergency diversity injection, guardian escalation, manual override",
            ),
            Risk(
                id="TECH-001",
                name="Memory Corruption",
                description="Living memory entries corrupted or inconsistent",
                category=RiskCategory.TECHNICAL,
                severity=RiskSeverity.MEDIUM,
                probability=0.2,
                impact=0.4,
                strategy=MitigationStrategy.MITIGATE,
                mitigation_plan="Memory healing, checksums, periodic consolidation",
            ),
        ]

        for risk in standard_risks:
            self._risks[risk.id] = risk

        # Define cascade relationships
        self._define_cascades()

    def _define_cascades(self) -> None:
        """Define risk cascade relationships."""
        cascades = {
            "SEC-001": [
                "SEC-002",
                "INT-001",
            ],  # Key compromise → Ihsān bypass, consensus failure
            "SEC-002": [
                "PERF-001",
                "OPS-001",
            ],  # Ihsān bypass → SNR degrade, self-healing fail
            "PERF-001": ["TECH-001"],  # SNR degrade → memory corruption
            "INT-001": ["OPS-001"],  # Consensus fail → self-healing fail
        }

        for source, targets in cascades.items():
            if source in self._risks:
                self._risks[source].cascades_to = set(targets)
                for target in targets:
                    if target in self._risks:
                        self._risks[target].triggers.add(source)

    def add_risk(self, risk: Risk) -> None:
        """Add a new risk."""
        self._risks[risk.id] = risk
        self._cascade_cache.clear()

    def update_risk(
        self,
        risk_id: str,
        **updates: Any,
    ) -> Optional[Risk]:
        """Update a risk."""
        if risk_id not in self._risks:
            return None

        risk = self._risks[risk_id]
        for key, value in updates.items():
            if hasattr(risk, key):
                setattr(risk, key, value)

        risk.updated_at = datetime.now(timezone.utc)
        self._cascade_cache.clear()

        return risk

    def get_risk(self, risk_id: str) -> Optional[Risk]:
        """Get a risk by ID."""
        return self._risks.get(risk_id)

    def analyze_cascade(
        self,
        source_risk_id: str,
        max_depth: int = 5,
    ) -> CascadeAnalysis:
        """
        Analyze cascading impact from a source risk.

        Uses BFS to find all affected risks and calculate total impact.
        """
        if source_risk_id in self._cascade_cache:
            return self._cascade_cache[source_risk_id]

        source = self._risks.get(source_risk_id)
        if not source:
            return CascadeAnalysis(
                source_risk=source_risk_id,
                affected_risks=[],
                total_impact=0.0,
                cascade_depth=0,
                critical_path=[],
            )

        # BFS to find all affected risks
        visited = set()
        affected = []
        queue = [(source_risk_id, 0, [source_risk_id])]  # (risk_id, depth, path)
        max_depth_reached = 0
        critical_path = []
        total_impact = source.impact

        while queue:
            current_id, depth, path = queue.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)
            current = self._risks.get(current_id)

            if current and current_id != source_risk_id:
                affected.append(current_id)
                # Cascading impact is dampened by probability
                cascade_impact = current.impact * current.probability
                total_impact += cascade_impact

                if depth > max_depth_reached:
                    max_depth_reached = depth
                    critical_path = path

            if current:
                for next_id in current.cascades_to:
                    if next_id not in visited:
                        queue.append((next_id, depth + 1, path + [next_id]))

        result = CascadeAnalysis(
            source_risk=source_risk_id,
            affected_risks=affected,
            total_impact=min(total_impact, 1.0),  # Cap at 1.0
            cascade_depth=max_depth_reached,
            critical_path=critical_path,
        )

        self._cascade_cache[source_risk_id] = result
        return result

    def get_prioritized_risks(self) -> List[Risk]:
        """Get risks ordered by priority score."""
        risks = list(self._risks.values())
        risks.sort(key=lambda r: r.priority_score, reverse=True)
        return risks

    def get_constitutional_risks(self) -> List[Risk]:
        """Get all constitutional-level risks."""
        return [
            r
            for r in self._risks.values()
            if r.severity == RiskSeverity.CONSTITUTIONAL
            or r.category == RiskCategory.CONSTITUTIONAL
        ]

    def recommend_mitigation(
        self,
        risk_id: str,
    ) -> MitigationStrategy:
        """Recommend a mitigation strategy based on risk characteristics."""
        risk = self._risks.get(risk_id)
        if not risk:
            return MitigationStrategy.ACCEPT

        # Constitutional risks must be avoided
        if risk.severity == RiskSeverity.CONSTITUTIONAL:
            return MitigationStrategy.AVOID

        # High impact risks should be mitigated or avoided
        if risk.impact >= 0.8:
            if risk.probability >= 0.5:
                return MitigationStrategy.AVOID
            else:
                return MitigationStrategy.MITIGATE

        # Medium risks can be mitigated
        if risk.risk_score >= 0.3:
            return MitigationStrategy.MITIGATE

        # Low risks can be accepted with monitoring
        return MitigationStrategy.ACCEPT

    def get_risk_matrix(self) -> Dict[str, Any]:
        """Generate risk matrix for visualization."""
        matrix = {
            "high_high": [],
            "high_low": [],
            "low_high": [],
            "low_low": [],
            "constitutional": [],
        }

        for risk in self._risks.values():
            if risk.severity == RiskSeverity.CONSTITUTIONAL:
                matrix["constitutional"].append(risk.id)
            elif risk.probability >= 0.5 and risk.impact >= 0.5:
                matrix["high_high"].append(risk.id)
            elif risk.probability >= 0.5 and risk.impact < 0.5:
                matrix["high_low"].append(risk.id)
            elif risk.probability < 0.5 and risk.impact >= 0.5:
                matrix["low_high"].append(risk.id)
            else:
                matrix["low_low"].append(risk.id)

        return matrix

    def get_summary(self) -> Dict[str, Any]:
        """Get risk management summary."""
        risks = list(self._risks.values())

        by_category = {}
        by_severity = {}
        by_status = {}

        for risk in risks:
            cat = risk.category.value
            sev = risk.severity.value
            stat = risk.status.value

            by_category[cat] = by_category.get(cat, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1
            by_status[stat] = by_status.get(stat, 0) + 1

        total_risk_score = sum(r.risk_score for r in risks)
        avg_risk_score = total_risk_score / max(len(risks), 1)

        return {
            "total_risks": len(risks),
            "by_category": by_category,
            "by_severity": by_severity,
            "by_status": by_status,
            "average_risk_score": avg_risk_score,
            "total_risk_exposure": total_risk_score,
            "constitutional_risks": len(self.get_constitutional_risks()),
            "top_priority": [r.id for r in self.get_prioritized_risks()[:5]],
        }
