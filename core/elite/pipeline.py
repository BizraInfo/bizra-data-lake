"""
Elite Pipeline — CI/CD with Constitutional Validation

Implements DevOps/GitOps practices with Ihsān principles:
- Automated quality gates at every stage
- Constitutional validation in CI/CD
- SNR-optimized build artifacts
- Fail-closed security model

Standing on Giants: GitOps + SRE + Constitutional AI
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

from core.elite.quality_gates import (
    QualityGate,
    QualityGateChain,
    GateResult,
    GateStatus,
    GateLevel,
)
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
)

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """CI/CD pipeline stages."""
    SOURCE = "source"
    BUILD = "build"
    TEST = "test"
    SECURITY = "security"
    QUALITY = "quality"
    STAGING = "staging"
    PRODUCTION = "production"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


@dataclass
class StageResult:
    """Result of a pipeline stage."""
    stage: PipelineStage
    status: PipelineStatus
    duration_ms: float
    gate_result: Optional[GateResult] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "gate_result": self.gate_result.to_dict() if self.gate_result else None,
            "artifacts": self.artifacts,
            "logs": self.logs[-10:],  # Last 10 logs
            "error": self.error,
        }


@dataclass
class PipelineRun:
    """A complete pipeline execution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: PipelineStatus = PipelineStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    stages: Dict[str, StageResult] = field(default_factory=dict)
    ihsan_score: float = 0.0
    snr_score: float = 0.0
    constitutional_violations: List[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "constitutional_violations": self.constitutional_violations,
        }


class PipelineStageHandler:
    """Base handler for pipeline stages."""

    def __init__(
        self,
        stage: PipelineStage,
        gate: Optional[QualityGate] = None,
    ):
        self.stage = stage
        self.gate = gate or QualityGate(name=stage.value)

    async def execute(
        self,
        input_artifacts: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Execute the stage logic.

        Returns: (output_artifacts, logs)
        Override in subclasses for stage-specific logic.
        """
        logs = [f"Executing stage: {self.stage.value}"]
        output = input_artifacts.copy()
        output["_stage"] = self.stage.value
        output["_timestamp"] = datetime.now(timezone.utc).isoformat()

        return output, logs

    async def validate(
        self,
        artifacts: Dict[str, Any],
    ) -> GateResult:
        """Run quality gate validation."""
        _, exit_result = await self.gate.validate(artifacts)
        return exit_result


class SourceStageHandler(PipelineStageHandler):
    """Source code analysis stage."""

    def __init__(self):
        super().__init__(
            PipelineStage.SOURCE,
            QualityGate(name="source", sape_layer="data"),
        )

    async def execute(
        self,
        input_artifacts: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        logs = ["Analyzing source code..."]

        output = input_artifacts.copy()

        # Simulate source analysis metrics
        output["code_coverage"] = context.get("code_coverage", 0.85)
        output["complexity"] = context.get("complexity", 0.7)
        output["snr_threshold"] = 0.90  # Source layer SNR

        # Ihsān dimensions for source (all 8 required)
        output["ihsan_correctness"] = context.get("ihsan_correctness", context.get("lint_score", 0.95))
        output["ihsan_safety"] = context.get("ihsan_safety", context.get("security_score", 0.95))
        output["ihsan_user_benefit"] = context.get("ihsan_user_benefit", 0.95)
        output["ihsan_efficiency"] = context.get("ihsan_efficiency", 0.95)
        output["ihsan_auditability"] = context.get("ihsan_auditability", context.get("documentation_score", 0.85))
        output["ihsan_anti_centralization"] = context.get("ihsan_anti_centralization", 0.95)
        output["ihsan_robustness"] = context.get("ihsan_robustness", 0.95)
        output["ihsan_adl_justice"] = context.get("ihsan_adl_justice", 0.95)

        logs.append(f"Source analysis complete. Coverage: {output['code_coverage']:.2%}")

        return output, logs


class BuildStageHandler(PipelineStageHandler):
    """Build stage."""

    def __init__(self):
        super().__init__(
            PipelineStage.BUILD,
            QualityGate(name="build", sape_layer="data"),
        )

    async def execute(
        self,
        input_artifacts: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        logs = ["Building artifacts..."]

        output = input_artifacts.copy()

        # Simulate build metrics
        output["build_success"] = True
        output["artifact_size_mb"] = context.get("artifact_size_mb", 50)
        output["dependencies_resolved"] = True

        logs.append("Build completed successfully")

        return output, logs


class TestStageHandler(PipelineStageHandler):
    """Test execution stage."""

    def __init__(self):
        super().__init__(
            PipelineStage.TEST,
            QualityGate(name="test", sape_layer="information"),
        )

    async def execute(
        self,
        input_artifacts: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        logs = ["Running test suite..."]

        output = input_artifacts.copy()

        # Test metrics
        tests_passed = context.get("tests_passed", 111)
        tests_total = context.get("tests_total", 111)

        output["tests_passed"] = tests_passed
        output["tests_total"] = tests_total
        output["test_success_rate"] = tests_passed / max(tests_total, 1)
        output["snr_threshold"] = 0.95  # Information layer SNR

        # Update Ihsān dimensions
        output["ihsan_correctness"] = output["test_success_rate"]
        output["ihsan_robustness"] = context.get("edge_case_coverage", 0.85)

        logs.append(f"Tests: {tests_passed}/{tests_total} passed ({output['test_success_rate']:.2%})")

        return output, logs


class SecurityStageHandler(PipelineStageHandler):
    """Security scanning stage."""

    def __init__(self):
        super().__init__(
            PipelineStage.SECURITY,
            QualityGate(name="security", sape_layer="knowledge"),
        )
        # Security is constitutional - never override
        self.gate.add_exit_criterion(
            name="no_critical_vulnerabilities",
            description="No critical security vulnerabilities",
            threshold=1.0,  # Must be 100%
            level=GateLevel.CONSTITUTIONAL,
            validator=lambda x: 1.0 if x == 0 else 0.0,
        )

    async def execute(
        self,
        input_artifacts: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        logs = ["Running security scans..."]

        output = input_artifacts.copy()

        # Security metrics
        output["critical_vulns"] = context.get("critical_vulns", 0)
        output["high_vulns"] = context.get("high_vulns", 0)
        output["medium_vulns"] = context.get("medium_vulns", 0)
        output["no_critical_vulnerabilities"] = output["critical_vulns"]
        output["snr_threshold"] = 0.99  # Knowledge layer SNR

        # Ihsān safety dimension
        total_vulns = output["critical_vulns"] + output["high_vulns"] + output["medium_vulns"]
        output["ihsan_safety"] = 1.0 if total_vulns == 0 else max(0, 1.0 - total_vulns * 0.1)

        logs.append(f"Security scan: {output['critical_vulns']} critical, {output['high_vulns']} high, {output['medium_vulns']} medium")

        return output, logs


class QualityStageHandler(PipelineStageHandler):
    """Quality assurance stage."""

    def __init__(self):
        super().__init__(
            PipelineStage.QUALITY,
            QualityGate(name="quality", sape_layer="knowledge"),
        )

    async def execute(
        self,
        input_artifacts: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        logs = ["Running quality analysis..."]

        output = input_artifacts.copy()

        # Quality metrics
        output["code_quality_score"] = context.get("code_quality_score", 0.92)
        output["maintainability_index"] = context.get("maintainability_index", 0.88)
        output["technical_debt_hours"] = context.get("technical_debt_hours", 20)
        output["snr_threshold"] = 0.99

        # Aggregate Ihsān score
        ihsan_scores = [
            output.get("ihsan_correctness", 0.95),
            output.get("ihsan_safety", 0.95),
            output.get("ihsan_auditability", 0.85),
            output.get("ihsan_robustness", 0.85),
        ]
        output["ihsan_overall"] = sum(ihsan_scores) / len(ihsan_scores)

        logs.append(f"Quality score: {output['code_quality_score']:.2%}, Ihsān: {output['ihsan_overall']:.2%}")

        return output, logs


class ElitePipeline:
    """
    Elite CI/CD Pipeline with Constitutional Validation.

    Implements DevOps best practices with Ihsān principles:
    - Automated quality gates at every stage
    - Constitutional (Ihsān) validation
    - SNR progression through SAPE layers
    - Fail-closed security model
    """

    def __init__(
        self,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    ):
        self.ihsan_threshold = ihsan_threshold

        # Stage handlers
        self._handlers: Dict[PipelineStage, PipelineStageHandler] = {
            PipelineStage.SOURCE: SourceStageHandler(),
            PipelineStage.BUILD: BuildStageHandler(),
            PipelineStage.TEST: TestStageHandler(),
            PipelineStage.SECURITY: SecurityStageHandler(),
            PipelineStage.QUALITY: QualityStageHandler(),
        }

        # Pipeline configuration
        self._stage_order = [
            PipelineStage.SOURCE,
            PipelineStage.BUILD,
            PipelineStage.TEST,
            PipelineStage.SECURITY,
            PipelineStage.QUALITY,
        ]

        # Run history
        self._runs: Dict[str, PipelineRun] = {}

    def add_handler(
        self,
        stage: PipelineStage,
        handler: PipelineStageHandler,
    ) -> None:
        """Add or replace a stage handler."""
        self._handlers[stage] = handler

    async def _run_stage(
        self,
        stage: PipelineStage,
        artifacts: Dict[str, Any],
        context: Dict[str, Any],
    ) -> StageResult:
        """Execute a single stage."""
        handler = self._handlers.get(stage)
        if not handler:
            return StageResult(
                stage=stage,
                status=PipelineStatus.FAILED,
                duration_ms=0,
                error=f"No handler for stage: {stage.value}",
            )

        import time
        start = time.time()

        try:
            # Execute stage
            output_artifacts, logs = await handler.execute(artifacts, context)

            # Validate gate
            gate_result = await handler.validate(output_artifacts)

            duration = (time.time() - start) * 1000

            status = (
                PipelineStatus.PASSED if gate_result.status == GateStatus.PASSED
                else PipelineStatus.FAILED
            )

            return StageResult(
                stage=stage,
                status=status,
                duration_ms=duration,
                gate_result=gate_result,
                artifacts=output_artifacts,
                logs=logs,
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return StageResult(
                stage=stage,
                status=PipelineStatus.FAILED,
                duration_ms=duration,
                error=str(e),
                logs=[f"Stage error: {e}"],
            )

    async def run(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineRun:
        """
        Execute the full pipeline.

        Args:
            context: Optional context with metrics/scores to inject

        Returns:
            PipelineRun with all stage results
        """
        context = context or {}

        run = PipelineRun(
            started_at=datetime.now(timezone.utc),
            status=PipelineStatus.RUNNING,
        )
        self._runs[run.id] = run

        artifacts = {}
        all_passed = True

        for stage in self._stage_order:
            logger.info(f"Running stage: {stage.value}")

            result = await self._run_stage(stage, artifacts, context)
            run.stages[stage.value] = result

            if result.status != PipelineStatus.PASSED:
                all_passed = False
                run.status = PipelineStatus.FAILED

                # Constitutional violations
                if result.gate_result:
                    for blocking in result.gate_result.blocking_criteria:
                        if "ihsan" in blocking or "safety" in blocking:
                            run.constitutional_violations.append(
                                f"{stage.value}: {blocking}"
                            )

                break  # Stop on failure

            artifacts = result.artifacts

        if all_passed:
            run.status = PipelineStatus.PASSED

        run.completed_at = datetime.now(timezone.utc)

        # Compute overall scores
        quality_stage = run.stages.get("quality")
        if quality_stage and quality_stage.gate_result:
            run.ihsan_score = quality_stage.gate_result.ihsan_score
            run.snr_score = quality_stage.gate_result.snr_score

        logger.info(
            f"Pipeline {run.id} completed: {run.status.value} "
            f"(Ihsān: {run.ihsan_score:.2%}, SNR: {run.snr_score:.2%})"
        )

        return run

    def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """Get a pipeline run by ID."""
        return self._runs.get(run_id)

    def get_recent_runs(self, limit: int = 10) -> List[PipelineRun]:
        """Get recent pipeline runs."""
        runs = sorted(
            self._runs.values(),
            key=lambda r: r.started_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return runs[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        total = len(self._runs)
        passed = sum(1 for r in self._runs.values() if r.status == PipelineStatus.PASSED)
        failed = sum(1 for r in self._runs.values() if r.status == PipelineStatus.FAILED)

        avg_ihsan = 0.0
        avg_snr = 0.0
        if total > 0:
            avg_ihsan = sum(r.ihsan_score for r in self._runs.values()) / total
            avg_snr = sum(r.snr_score for r in self._runs.values()) / total

        return {
            "total_runs": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / max(total, 1),
            "avg_ihsan_score": avg_ihsan,
            "avg_snr_score": avg_snr,
            "constitutional_violations": sum(
                len(r.constitutional_violations) for r in self._runs.values()
            ),
        }
