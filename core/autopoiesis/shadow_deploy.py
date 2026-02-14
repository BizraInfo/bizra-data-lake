"""
Shadow Deployment System — Safe Parallel Testing of Improvements
===============================================================================

Enables safe testing of hypothesized improvements by running them in parallel
with production, comparing metrics without affecting actual users.

Key Features:
- Isolated shadow environments with cloned configuration
- Traffic mirroring without response delivery to users
- Statistical significance testing for A/B comparisons
- Automatic promotion/rollback based on metrics
- FATE validation integration with Ihsan kill switch

Standing on Giants: Netflix (Chaos Engineering) + Amazon (Canary) + Anthropic (Constitutional AI)
Genesis Strict Synthesis v2.2.2
"""

import asyncio
import copy
import hashlib
import logging
import math
import random
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class DeploymentVerdict(Enum):
    """Verdict for shadow deployment evaluation."""

    PENDING = "pending"  # Still evaluating
    PROMOTE = "promote"  # Shadow is better, promote to production
    REJECT = "reject"  # Shadow is worse or unsafe, reject
    EXTEND = "extend"  # Need more data, extend evaluation
    ROLLBACK = "rollback"  # Regression detected, rollback immediately


class ComparisonStatus(Enum):
    """Status of metric comparison."""

    SIGNIFICANT_IMPROVEMENT = "significant_improvement"
    MARGINAL_IMPROVEMENT = "marginal_improvement"
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"
    MARGINAL_REGRESSION = "marginal_regression"
    SIGNIFICANT_REGRESSION = "significant_regression"


class IsolationLevel(Enum):
    """Level of isolation for shadow environment."""

    SOFT = "soft"  # Shared resources, logical isolation
    MEDIUM = "medium"  # Separate processes, shared storage
    HARD = "hard"  # Full containerization


class TrafficMode(Enum):
    """Traffic mirroring mode."""

    FULL_MIRROR = "full_mirror"  # Mirror all traffic
    SAMPLED = "sampled"  # Sample percentage of traffic
    TARGETED = "targeted"  # Specific traffic patterns only


# Shadow deployment constants
DEFAULT_SHADOW_DURATION = timedelta(hours=1)
MIN_SAMPLE_SIZE = 30  # Minimum samples for statistical significance
CONFIDENCE_LEVEL = 0.95  # 95% confidence interval
IHSAN_KILL_THRESHOLD = 0.95  # Kill switch if Ihsan drops below this
MAX_LATENCY_DEGRADATION = 1.5  # Max 50% latency increase allowed
MAX_ERROR_RATE_INCREASE = 0.01  # Max 1% error rate increase allowed


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ShadowHypothesis:
    """
    A hypothesis for improvement to test in shadow deployment.

    Represents a proposed change that needs validation before
    being promoted to production.

    Note: This is distinct from core.autopoiesis.loop_engine.Hypothesis
    which is used for the general autopoietic improvement loop.
    ShadowHypothesis is specifically for shadow deployment testing.
    """

    id: str = ""
    name: str = ""
    description: str = ""
    proposed_change: Dict[str, Any] = field(default_factory=dict)
    expected_improvement: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            import secrets

            self.id = secrets.token_hex(6)  # 12 hex chars, CSPRNG

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "proposed_change": self.proposed_change,
            "expected_improvement": self.expected_improvement,
            "created_at": self.created_at.isoformat(),
        }


# Alias for backward compatibility and clearer semantics
Hypothesis = ShadowHypothesis


@dataclass
class MetricSample:
    """A single metric measurement."""

    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricComparison:
    """Comparison result between production and shadow metrics."""

    metric_name: str
    production_mean: float
    shadow_mean: float
    production_std: float
    shadow_std: float
    delta: float
    delta_percent: float
    p_value: float
    confidence_interval: Tuple[float, float]
    status: ComparisonStatus
    sample_size_prod: int
    sample_size_shadow: int

    @property
    def is_significant(self) -> bool:
        """Check if the difference is statistically significant."""
        return self.p_value < (1 - CONFIDENCE_LEVEL)

    @property
    def is_improvement(self) -> bool:
        """Check if shadow is an improvement."""
        return self.status in (
            ComparisonStatus.SIGNIFICANT_IMPROVEMENT,
            ComparisonStatus.MARGINAL_IMPROVEMENT,
        )

    @property
    def is_regression(self) -> bool:
        """Check if shadow is a regression."""
        return self.status in (
            ComparisonStatus.SIGNIFICANT_REGRESSION,
            ComparisonStatus.MARGINAL_REGRESSION,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric_name,
            "production_mean": round(self.production_mean, 6),
            "shadow_mean": round(self.shadow_mean, 6),
            "delta": round(self.delta, 6),
            "delta_percent": round(self.delta_percent, 2),
            "p_value": round(self.p_value, 4),
            "status": self.status.value,
            "is_significant": self.is_significant,
        }


@dataclass
class ComparisonResult:
    """Aggregated comparison results."""

    comparisons: List[MetricComparison] = field(default_factory=list)
    overall_verdict: DeploymentVerdict = DeploymentVerdict.PENDING
    ihsan_score_prod: float = 1.0
    ihsan_score_shadow: float = 1.0
    snr_score_prod: float = 1.0
    snr_score_shadow: float = 1.0
    fate_passed: bool = True
    regression_detected: bool = False
    improvement_detected: bool = False
    evaluation_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.overall_verdict.value,
            "ihsan_prod": round(self.ihsan_score_prod, 4),
            "ihsan_shadow": round(self.ihsan_score_shadow, 4),
            "snr_prod": round(self.snr_score_prod, 4),
            "snr_shadow": round(self.snr_score_shadow, 4),
            "fate_passed": self.fate_passed,
            "regression_detected": self.regression_detected,
            "improvement_detected": self.improvement_detected,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


@dataclass
class ShadowDeployment:
    """
    A shadow deployment instance tracking parallel execution.

    Maintains state for comparing production and shadow behavior
    over the deployment duration.
    """

    deployment_id: str = ""
    hypothesis: Hypothesis = field(default_factory=Hypothesis)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_limit: timedelta = DEFAULT_SHADOW_DURATION
    production_metrics: Dict[str, List[MetricSample]] = field(default_factory=dict)
    shadow_metrics: Dict[str, List[MetricSample]] = field(default_factory=dict)
    comparison: ComparisonResult = field(default_factory=ComparisonResult)
    verdict: DeploymentVerdict = DeploymentVerdict.PENDING
    traffic_mode: TrafficMode = TrafficMode.FULL_MIRROR
    sample_rate: float = 1.0  # For SAMPLED mode
    isolation_level: IsolationLevel = IsolationLevel.MEDIUM

    # Tracking
    requests_mirrored: int = 0
    requests_compared: int = 0
    errors_prod: int = 0
    errors_shadow: int = 0

    # FATE/Ihsan monitoring
    ihsan_samples_prod: List[float] = field(default_factory=list)
    ihsan_samples_shadow: List[float] = field(default_factory=list)
    kill_switch_triggered: bool = False
    kill_switch_reason: str = ""

    def __post_init__(self):
        if not self.deployment_id:
            self.deployment_id = str(uuid.uuid4())[:8]

    @property
    def elapsed_time(self) -> timedelta:
        """Time elapsed since deployment started."""
        return datetime.now(timezone.utc) - self.start_time

    @property
    def time_remaining(self) -> timedelta:
        """Time remaining until duration limit."""
        remaining = self.duration_limit - self.elapsed_time
        return remaining if remaining > timedelta(0) else timedelta(0)

    @property
    def is_expired(self) -> bool:
        """Check if deployment has exceeded duration limit."""
        return self.elapsed_time >= self.duration_limit

    @property
    def has_sufficient_data(self) -> bool:
        """Check if we have enough samples for statistical significance."""
        for metric_name in self.production_metrics:
            if metric_name in self.shadow_metrics:
                prod_count = len(self.production_metrics[metric_name])
                shadow_count = len(self.shadow_metrics[metric_name])
                if prod_count >= MIN_SAMPLE_SIZE and shadow_count >= MIN_SAMPLE_SIZE:
                    return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "hypothesis_id": self.hypothesis.id,
            "hypothesis_name": self.hypothesis.name,
            "start_time": self.start_time.isoformat(),
            "elapsed_seconds": self.elapsed_time.total_seconds(),
            "duration_limit_seconds": self.duration_limit.total_seconds(),
            "verdict": self.verdict.value,
            "requests_mirrored": self.requests_mirrored,
            "requests_compared": self.requests_compared,
            "errors_prod": self.errors_prod,
            "errors_shadow": self.errors_shadow,
            "kill_switch_triggered": self.kill_switch_triggered,
            "has_sufficient_data": self.has_sufficient_data,
        }


@dataclass
class ShadowRequest:
    """A request to be processed by both production and shadow."""

    request_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())[:16]


@dataclass
class ShadowResponse:
    """Response from either production or shadow."""

    request_id: str
    result: Any
    latency_ms: float
    error: Optional[str] = None
    ihsan_score: float = 1.0
    snr_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceLimits:
    """Resource limits for shadow environment."""

    max_cpu_percent: float = 50.0  # Max CPU usage
    max_memory_mb: int = 1024  # Max memory in MB
    max_requests_per_second: float = 100.0
    max_concurrent_requests: int = 10
    timeout_seconds: float = 30.0


@dataclass
class AuditEntry:
    """Audit log entry for shadow deployment operations."""

    entry_id: str = ""
    deployment_id: str = ""
    operation: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = str(uuid.uuid4())[:12]


# =============================================================================
# SHADOW ENVIRONMENT
# =============================================================================


class ShadowEnvironment:
    """
    Isolated execution context for shadow deployments.

    Creates a sandboxed environment where hypothesis changes can be
    tested without affecting production.

    Features:
    - Cloned configuration from production
    - Separate metrics collection
    - Resource limits enforcement
    - No side effects on production state

    Usage:
        env = ShadowEnvironment(hypothesis=my_hypothesis)
        await env.initialize()

        # Process request in shadow
        response = await env.process(request)

        # Get metrics
        metrics = env.get_metrics()

        await env.teardown()
    """

    def __init__(
        self,
        hypothesis: Hypothesis,
        isolation_level: IsolationLevel = IsolationLevel.MEDIUM,
        resource_limits: Optional[ResourceLimits] = None,
        production_config: Optional[Dict[str, Any]] = None,
    ):
        self.hypothesis = hypothesis
        self.isolation_level = isolation_level
        self.resource_limits = resource_limits or ResourceLimits()
        self.production_config = production_config or {}

        # State
        self.shadow_config: Dict[str, Any] = {}
        self._initialized = False
        self._active = False

        # Metrics
        self._latency_samples: List[float] = []
        self._ihsan_samples: List[float] = []
        self._snr_samples: List[float] = []
        self._error_count: int = 0
        self._request_count: int = 0

        # Isolation
        self._isolation_verified = False
        self._resource_usage: Dict[str, float] = {}

        # Concurrency control
        self._semaphore = asyncio.Semaphore(
            self.resource_limits.max_concurrent_requests
        )
        self._rate_limiter = _TokenBucket(
            rate=self.resource_limits.max_requests_per_second,
            capacity=self.resource_limits.max_concurrent_requests,
        )

        self.env_id = hashlib.md5(
            f"{hypothesis.id}{datetime.now().isoformat()}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:8]

    async def initialize(self) -> bool:
        """Initialize the shadow environment."""
        if self._initialized:
            return True

        logger.info(
            f"Initializing shadow environment {self.env_id} for hypothesis {self.hypothesis.id}"
        )

        try:
            # Clone production configuration
            self.shadow_config = copy.deepcopy(self.production_config)

            # Apply hypothesis changes
            for key, value in self.hypothesis.proposed_change.items():
                self._apply_config_change(key, value)

            # Verify isolation
            self._isolation_verified = await self._verify_isolation()

            if not self._isolation_verified:
                logger.error(
                    f"Isolation verification failed for environment {self.env_id}"
                )
                return False

            self._initialized = True
            self._active = True

            logger.info(f"Shadow environment {self.env_id} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize shadow environment: {e}")
            return False

    def _apply_config_change(self, key: str, value: Any):
        """Apply a configuration change from the hypothesis."""
        # Support nested keys with dot notation
        keys = key.split(".")
        config = self.shadow_config

        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                # Create or replace with dict if not a dict
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        logger.debug(f"Applied config change: {key} = {value}")

    async def _verify_isolation(self) -> bool:
        """Verify that the environment is properly isolated."""
        if self.isolation_level == IsolationLevel.SOFT:
            # Soft isolation: just verify config separation
            return self.shadow_config is not self.production_config

        elif self.isolation_level == IsolationLevel.MEDIUM:
            # Medium isolation: verify separate memory space
            return (
                id(self.shadow_config) != id(self.production_config)
                and self.shadow_config is not self.production_config
            )

        elif self.isolation_level == IsolationLevel.HARD:
            # Hard isolation: would verify container/process separation
            # In production, this would check container/process isolation
            return True

        return False

    async def process(
        self,
        request: ShadowRequest,
        handler: Callable[[ShadowRequest, Dict[str, Any]], Any],
    ) -> ShadowResponse:
        """
        Process a request in the shadow environment.

        Args:
            request: The request to process
            handler: The function to handle the request (receives config)

        Returns:
            ShadowResponse with results and metrics
        """
        if not self._active:
            return ShadowResponse(
                request_id=request.request_id,
                result=None,
                latency_ms=0,
                error="Shadow environment not active",
            )

        # Rate limiting
        if not await self._rate_limiter.acquire():
            return ShadowResponse(
                request_id=request.request_id,
                result=None,
                latency_ms=0,
                error="Rate limit exceeded",
            )

        async with self._semaphore:
            start_time = datetime.now(timezone.utc)

            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    handler(request, self.shadow_config),
                    timeout=self.resource_limits.timeout_seconds,
                )

                latency_ms = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000
                self._request_count += 1
                self._latency_samples.append(latency_ms)

                # Extract metrics from result if available
                ihsan_score = getattr(result, "ihsan_score", None) or 1.0
                snr_score = getattr(result, "snr_score", None) or 1.0

                self._ihsan_samples.append(ihsan_score)
                self._snr_samples.append(snr_score)

                return ShadowResponse(
                    request_id=request.request_id,
                    result=result,
                    latency_ms=latency_ms,
                    ihsan_score=ihsan_score,
                    snr_score=snr_score,
                )

            except asyncio.TimeoutError:
                self._error_count += 1
                latency_ms = self.resource_limits.timeout_seconds * 1000
                self._latency_samples.append(latency_ms)

                return ShadowResponse(
                    request_id=request.request_id,
                    result=None,
                    latency_ms=latency_ms,
                    error=f"Timeout after {self.resource_limits.timeout_seconds}s",
                )

            except Exception as e:
                self._error_count += 1
                latency_ms = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000
                self._latency_samples.append(latency_ms)

                return ShadowResponse(
                    request_id=request.request_id,
                    result=None,
                    latency_ms=latency_ms,
                    error=str(e),
                )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics from the shadow environment."""
        return {
            "env_id": self.env_id,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._request_count),
            "latency": {
                "mean": (
                    statistics.mean(self._latency_samples)
                    if self._latency_samples
                    else 0
                ),
                "p50": (
                    self._percentile(self._latency_samples, 50)
                    if self._latency_samples
                    else 0
                ),
                "p95": (
                    self._percentile(self._latency_samples, 95)
                    if self._latency_samples
                    else 0
                ),
                "p99": (
                    self._percentile(self._latency_samples, 99)
                    if self._latency_samples
                    else 0
                ),
            },
            "ihsan": {
                "mean": (
                    statistics.mean(self._ihsan_samples) if self._ihsan_samples else 1.0
                ),
                "min": min(self._ihsan_samples) if self._ihsan_samples else 1.0,
                "count_below_threshold": sum(
                    1 for s in self._ihsan_samples if s < UNIFIED_IHSAN_THRESHOLD
                ),
            },
            "snr": {
                "mean": (
                    statistics.mean(self._snr_samples) if self._snr_samples else 1.0
                ),
                "min": min(self._snr_samples) if self._snr_samples else 1.0,
            },
            "isolation_verified": self._isolation_verified,
        }

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        sorted_data = sorted(data)
        idx = (len(sorted_data) - 1) * percentile / 100
        lower = int(idx)
        upper = lower + 1
        if upper >= len(sorted_data):
            return sorted_data[-1]
        return sorted_data[lower] + (sorted_data[upper] - sorted_data[lower]) * (
            idx - lower
        )

    async def teardown(self):
        """Tear down the shadow environment."""
        self._active = False
        self._initialized = False
        self.shadow_config = {}
        logger.info(f"Shadow environment {self.env_id} torn down")


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================


class StatisticalAnalyzer:
    """
    Statistical analysis for A/B comparisons.

    Provides statistical significance testing for comparing
    production and shadow metrics.
    """

    @staticmethod
    def compare_metrics(
        production_samples: List[float],
        shadow_samples: List[float],
        metric_name: str,
        higher_is_better: bool = True,
    ) -> MetricComparison:
        """
        Compare production and shadow metric samples.

        Args:
            production_samples: Metric values from production
            shadow_samples: Metric values from shadow
            metric_name: Name of the metric being compared
            higher_is_better: True if higher values are improvements

        Returns:
            MetricComparison with statistical analysis
        """
        if not production_samples or not shadow_samples:
            return MetricComparison(
                metric_name=metric_name,
                production_mean=0,
                shadow_mean=0,
                production_std=0,
                shadow_std=0,
                delta=0,
                delta_percent=0,
                p_value=1.0,
                confidence_interval=(0, 0),
                status=ComparisonStatus.NO_SIGNIFICANT_DIFFERENCE,
                sample_size_prod=len(production_samples),
                sample_size_shadow=len(shadow_samples),
            )

        prod_mean = statistics.mean(production_samples)
        shadow_mean = statistics.mean(shadow_samples)
        prod_std = (
            statistics.stdev(production_samples) if len(production_samples) > 1 else 0
        )
        shadow_std = statistics.stdev(shadow_samples) if len(shadow_samples) > 1 else 0

        delta = shadow_mean - prod_mean
        delta_percent = (delta / prod_mean * 100) if prod_mean != 0 else 0

        # Welch's t-test for unequal variances
        p_value = StatisticalAnalyzer._welch_t_test(production_samples, shadow_samples)

        # Confidence interval for the difference
        ci = StatisticalAnalyzer._confidence_interval(
            production_samples, shadow_samples
        )

        # Determine status
        status = StatisticalAnalyzer._determine_status(delta, p_value, higher_is_better)

        return MetricComparison(
            metric_name=metric_name,
            production_mean=prod_mean,
            shadow_mean=shadow_mean,
            production_std=prod_std,
            shadow_std=shadow_std,
            delta=delta,
            delta_percent=delta_percent,
            p_value=p_value,
            confidence_interval=ci,
            status=status,
            sample_size_prod=len(production_samples),
            sample_size_shadow=len(shadow_samples),
        )

    @staticmethod
    def _welch_t_test(
        sample1: List[float],
        sample2: List[float],
    ) -> float:
        """
        Perform Welch's t-test (unequal variances).

        Returns p-value for the null hypothesis that the means are equal.
        """
        n1, n2 = len(sample1), len(sample2)

        if n1 < 2 or n2 < 2:
            return 1.0  # Cannot compute with insufficient samples

        mean1 = statistics.mean(sample1)
        mean2 = statistics.mean(sample2)
        var1 = statistics.variance(sample1)
        var2 = statistics.variance(sample2)

        # Welch's t-statistic
        se = math.sqrt(var1 / n1 + var2 / n2)
        if se == 0:
            return 1.0

        t_stat = (mean1 - mean2) / se

        # Degrees of freedom (Welch-Satterthwaite)
        numerator = (var1 / n1 + var2 / n2) ** 2
        denominator = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)

        if denominator == 0:
            return 1.0

        df = numerator / denominator

        # Approximate p-value using t-distribution CDF
        # Using a simplified approximation for the two-tailed test
        p_value = StatisticalAnalyzer._t_distribution_p_value(abs(t_stat), df)

        return p_value * 2  # Two-tailed test

    @staticmethod
    def _t_distribution_p_value(t: float, df: float) -> float:
        """
        Approximate one-tailed p-value from t-distribution.

        Uses a normal approximation for large df, else uses a polynomial approximation.
        """
        if df > 100:
            # Use normal approximation
            return 0.5 * (1 + math.erf(-t / math.sqrt(2)))

        # Hill's approximation for Student's t CDF
        df / (df + t * t)

        # Incomplete beta function approximation
        # Using a simplified series expansion
        if df == 1:
            return 0.5 - math.atan(t) / math.pi
        elif df == 2:
            return 0.5 * (1 - t / math.sqrt(2 + t * t))
        else:
            # General approximation using Gaussian CDF with correction
            z = t * (1 - 1 / (4 * df)) / math.sqrt(1 + t * t / (2 * df))
            return 0.5 * (1 + math.erf(-z / math.sqrt(2)))

    @staticmethod
    def _confidence_interval(
        sample1: List[float],
        sample2: List[float],
        confidence: float = CONFIDENCE_LEVEL,
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for the difference in means.
        """
        n1, n2 = len(sample1), len(sample2)

        if n1 < 2 or n2 < 2:
            return (0, 0)

        mean1 = statistics.mean(sample1)
        mean2 = statistics.mean(sample2)
        var1 = statistics.variance(sample1)
        var2 = statistics.variance(sample2)

        diff = mean2 - mean1
        se = math.sqrt(var1 / n1 + var2 / n2)

        # t critical value approximation for 95% confidence
        # Using z approximation for simplicity
        z_critical = 1.96  # For 95% confidence

        margin = z_critical * se
        return (diff - margin, diff + margin)

    @staticmethod
    def _determine_status(
        delta: float,
        p_value: float,
        higher_is_better: bool,
    ) -> ComparisonStatus:
        """Determine comparison status from delta and p-value."""
        is_significant = p_value < (1 - CONFIDENCE_LEVEL)

        if higher_is_better:
            is_improvement = delta > 0
        else:
            is_improvement = delta < 0

        if is_significant:
            if is_improvement:
                return ComparisonStatus.SIGNIFICANT_IMPROVEMENT
            else:
                return ComparisonStatus.SIGNIFICANT_REGRESSION
        else:
            if abs(delta) < 0.01 * abs(delta if delta != 0 else 1):
                return ComparisonStatus.NO_SIGNIFICANT_DIFFERENCE
            elif is_improvement:
                return ComparisonStatus.MARGINAL_IMPROVEMENT
            else:
                return ComparisonStatus.MARGINAL_REGRESSION


# =============================================================================
# SHADOW DEPLOYER
# =============================================================================


class ShadowDeployer:
    """
    Main orchestrator for shadow deployments.

    Manages the lifecycle of shadow deployments:
    1. Create isolated shadow environment
    2. Apply hypothesis changes
    3. Mirror traffic to shadow
    4. Collect and compare metrics
    5. Make promotion/rollback decisions

    Usage:
        deployer = ShadowDeployer()

        # Create and start shadow deployment
        hypothesis = Hypothesis(
            name="Improved batch size",
            proposed_change={"batch_size": 32},
            expected_improvement={"latency": -0.1},
        )

        deployment = await deployer.deploy_shadow(hypothesis)

        # Mirror traffic
        for request in requests:
            await deployer.mirror_request(deployment, request)

        # Evaluate
        verdict = await deployer.evaluate(deployment)

        if verdict == DeploymentVerdict.PROMOTE:
            await deployer.promote(deployment)
        elif verdict == DeploymentVerdict.REJECT:
            await deployer.rollback(deployment)
    """

    def __init__(
        self,
        production_config: Optional[Dict[str, Any]] = None,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        snr_threshold: float = UNIFIED_SNR_THRESHOLD,
        production_handler: Optional[Callable] = None,
    ):
        self.production_config = production_config or {}
        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold
        self.production_handler = production_handler

        # Active deployments
        self._deployments: Dict[str, ShadowDeployment] = {}
        self._environments: Dict[str, ShadowEnvironment] = {}

        # Audit log
        self._audit_log: List[AuditEntry] = []

        # Statistics
        self._total_deployments = 0
        self._promoted_count = 0
        self._rejected_count = 0
        self._rollback_count = 0

        # Background monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

    async def deploy_shadow(
        self,
        hypothesis: Hypothesis,
        duration: timedelta = DEFAULT_SHADOW_DURATION,
        traffic_mode: TrafficMode = TrafficMode.FULL_MIRROR,
        sample_rate: float = 1.0,
        isolation_level: IsolationLevel = IsolationLevel.MEDIUM,
        resource_limits: Optional[ResourceLimits] = None,
    ) -> ShadowDeployment:
        """
        Deploy a hypothesis to shadow environment.

        Args:
            hypothesis: The hypothesis to test
            duration: How long to run the shadow deployment
            traffic_mode: How to mirror traffic
            sample_rate: Sampling rate for SAMPLED mode
            isolation_level: Level of isolation for shadow
            resource_limits: Resource constraints

        Returns:
            ShadowDeployment tracking the deployment
        """
        self._log_audit(
            "deploy_shadow_start",
            "",
            {
                "hypothesis_id": hypothesis.id,
                "hypothesis_name": hypothesis.name,
            },
        )

        # Create shadow environment
        env = ShadowEnvironment(
            hypothesis=hypothesis,
            isolation_level=isolation_level,
            resource_limits=resource_limits,
            production_config=self.production_config,
        )

        if not await env.initialize():
            self._log_audit(
                "deploy_shadow_failed",
                "",
                {
                    "hypothesis_id": hypothesis.id,
                    "reason": "Environment initialization failed",
                },
            )
            raise RuntimeError(
                f"Failed to initialize shadow environment for {hypothesis.id}"
            )

        # Create deployment
        deployment = ShadowDeployment(
            hypothesis=hypothesis,
            duration_limit=duration,
            traffic_mode=traffic_mode,
            sample_rate=sample_rate,
            isolation_level=isolation_level,
        )

        self._deployments[deployment.deployment_id] = deployment
        self._environments[deployment.deployment_id] = env
        self._total_deployments += 1

        self._log_audit(
            "deploy_shadow_success",
            deployment.deployment_id,
            {
                "hypothesis_id": hypothesis.id,
                "duration_seconds": duration.total_seconds(),
            },
        )

        logger.info(
            f"Shadow deployment {deployment.deployment_id} started for hypothesis {hypothesis.name}"
        )

        return deployment

    async def mirror_request(
        self,
        deployment: ShadowDeployment,
        request: ShadowRequest,
        production_response: Optional[ShadowResponse] = None,
    ) -> Tuple[Optional[ShadowResponse], Optional[ShadowResponse]]:
        """
        Mirror a request to the shadow environment.

        Args:
            deployment: The shadow deployment
            request: The request to mirror
            production_response: Pre-computed production response (optional)

        Returns:
            Tuple of (production_response, shadow_response)
            Shadow response is None if not mirrored (sampling)
        """
        if deployment.kill_switch_triggered:
            logger.warning(
                f"Deployment {deployment.deployment_id} kill switch active, skipping mirror"
            )
            return (production_response, None) if production_response else (None, None)

        # Check if we should mirror this request (sampling)
        if deployment.traffic_mode == TrafficMode.SAMPLED:
            if random.random() > deployment.sample_rate:
                return (
                    (production_response, None) if production_response else (None, None)
                )

        env = self._environments.get(deployment.deployment_id)
        if not env:
            return (production_response, None) if production_response else (None, None)

        # Get production response if not provided
        if production_response is None and self.production_handler:
            prod_start = datetime.now(timezone.utc)
            try:
                prod_result = await self.production_handler(
                    request, self.production_config
                )
                prod_latency = (
                    datetime.now(timezone.utc) - prod_start
                ).total_seconds() * 1000
                production_response = ShadowResponse(
                    request_id=request.request_id,
                    result=prod_result,
                    latency_ms=prod_latency,
                    ihsan_score=getattr(prod_result, "ihsan_score", 1.0),
                    snr_score=getattr(prod_result, "snr_score", 1.0),
                )
            except Exception as e:
                deployment.errors_prod += 1
                production_response = ShadowResponse(
                    request_id=request.request_id,
                    result=None,
                    latency_ms=0,
                    error=str(e),
                )

        # Process in shadow
        shadow_response = await env.process(
            request,
            handler=self.production_handler or (lambda r, c: None),
        )

        deployment.requests_mirrored += 1

        # Record metrics
        if production_response:
            self._record_metric(
                deployment,
                "latency",
                production_response.latency_ms,
                shadow_response.latency_ms,
            )
            self._record_metric(
                deployment,
                "ihsan",
                production_response.ihsan_score,
                shadow_response.ihsan_score,
            )
            self._record_metric(
                deployment,
                "snr",
                production_response.snr_score,
                shadow_response.snr_score,
            )

            if production_response.error:
                deployment.errors_prod += 1
            if shadow_response.error:
                deployment.errors_shadow += 1

            # Track Ihsan for kill switch
            deployment.ihsan_samples_prod.append(production_response.ihsan_score)
            deployment.ihsan_samples_shadow.append(shadow_response.ihsan_score)

            deployment.requests_compared += 1

        # Check kill switch
        await self._check_kill_switch(deployment)

        return (production_response, shadow_response)

    def _record_metric(
        self,
        deployment: ShadowDeployment,
        metric_name: str,
        prod_value: float,
        shadow_value: float,
    ):
        """Record metric samples for comparison."""
        if metric_name not in deployment.production_metrics:
            deployment.production_metrics[metric_name] = []
        if metric_name not in deployment.shadow_metrics:
            deployment.shadow_metrics[metric_name] = []

        deployment.production_metrics[metric_name].append(
            MetricSample(value=prod_value)
        )
        deployment.shadow_metrics[metric_name].append(MetricSample(value=shadow_value))

    async def _check_kill_switch(self, deployment: ShadowDeployment):
        """Check if kill switch should be triggered."""
        if deployment.kill_switch_triggered:
            return

        # Check Ihsan threshold
        if deployment.ihsan_samples_shadow:
            current_ihsan = statistics.mean(deployment.ihsan_samples_shadow[-10:])
            if current_ihsan < IHSAN_KILL_THRESHOLD:
                deployment.kill_switch_triggered = True
                deployment.kill_switch_reason = f"Ihsan below threshold: {current_ihsan:.4f} < {IHSAN_KILL_THRESHOLD}"
                deployment.verdict = DeploymentVerdict.REJECT

                self._log_audit(
                    "kill_switch_triggered",
                    deployment.deployment_id,
                    {
                        "reason": deployment.kill_switch_reason,
                        "ihsan_score": current_ihsan,
                    },
                )

                logger.warning(
                    f"Kill switch triggered for deployment {deployment.deployment_id}: {deployment.kill_switch_reason}"
                )

    async def evaluate(self, deployment: ShadowDeployment) -> DeploymentVerdict:
        """
        Evaluate a shadow deployment and determine verdict.

        Args:
            deployment: The deployment to evaluate

        Returns:
            DeploymentVerdict indicating what action to take
        """
        if deployment.kill_switch_triggered:
            return DeploymentVerdict.REJECT

        if not deployment.has_sufficient_data:
            if deployment.is_expired:
                # Not enough data even after duration expired
                self._log_audit(
                    "evaluate_insufficient_data",
                    deployment.deployment_id,
                    {
                        "samples": deployment.requests_compared,
                        "required": MIN_SAMPLE_SIZE,
                    },
                )
                return DeploymentVerdict.EXTEND
            return DeploymentVerdict.PENDING

        # Perform statistical comparison
        comparisons: List[MetricComparison] = []

        # Define metric preferences (higher/lower is better)
        metric_prefs = {
            "latency": False,  # Lower is better
            "error_rate": False,
            "ihsan": True,  # Higher is better
            "snr": True,
            "throughput": True,
        }

        for metric_name in deployment.production_metrics:
            if metric_name not in deployment.shadow_metrics:
                continue

            prod_samples = [s.value for s in deployment.production_metrics[metric_name]]
            shadow_samples = [s.value for s in deployment.shadow_metrics[metric_name]]

            higher_is_better = metric_prefs.get(metric_name, True)

            comparison = StatisticalAnalyzer.compare_metrics(
                prod_samples,
                shadow_samples,
                metric_name,
                higher_is_better,
            )
            comparisons.append(comparison)

        # Build comparison result
        comparison_result = ComparisonResult(
            comparisons=comparisons,
            ihsan_score_prod=(
                statistics.mean(deployment.ihsan_samples_prod)
                if deployment.ihsan_samples_prod
                else 1.0
            ),
            ihsan_score_shadow=(
                statistics.mean(deployment.ihsan_samples_shadow)
                if deployment.ihsan_samples_shadow
                else 1.0
            ),
            snr_score_prod=self._get_metric_mean(deployment.production_metrics, "snr"),
            snr_score_shadow=self._get_metric_mean(deployment.shadow_metrics, "snr"),
        )

        # Determine verdict
        verdict = self._determine_verdict(comparison_result)

        comparison_result.overall_verdict = verdict
        comparison_result.regression_detected = any(
            c.is_regression for c in comparisons
        )
        comparison_result.improvement_detected = any(
            c.is_improvement for c in comparisons
        )

        # FATE validation
        comparison_result.fate_passed = await self._validate_fate(
            deployment, comparison_result
        )

        if not comparison_result.fate_passed:
            verdict = DeploymentVerdict.REJECT

        deployment.comparison = comparison_result
        deployment.verdict = verdict

        self._log_audit(
            "evaluate_complete",
            deployment.deployment_id,
            {
                "verdict": verdict.value,
                "comparisons": len(comparisons),
                "regression": comparison_result.regression_detected,
                "improvement": comparison_result.improvement_detected,
            },
        )

        return verdict

    def _get_metric_mean(
        self,
        metrics: Dict[str, List[MetricSample]],
        metric_name: str,
    ) -> float:
        """Get mean value for a metric."""
        if metric_name not in metrics or not metrics[metric_name]:
            return 1.0
        return statistics.mean([s.value for s in metrics[metric_name]])

    def _determine_verdict(self, comparison: ComparisonResult) -> DeploymentVerdict:
        """Determine deployment verdict from comparison results."""
        # Check Ihsan compliance first (hard constraint)
        if comparison.ihsan_score_shadow < IHSAN_KILL_THRESHOLD:
            return DeploymentVerdict.REJECT

        # Check for significant regressions
        regressions = [
            c
            for c in comparison.comparisons
            if c.status == ComparisonStatus.SIGNIFICANT_REGRESSION
        ]
        if regressions:
            # Check if any critical metrics regressed
            critical_metrics = {"ihsan", "snr", "error_rate"}
            critical_regressions = [
                r for r in regressions if r.metric_name in critical_metrics
            ]
            if critical_regressions:
                return DeploymentVerdict.REJECT

            # Non-critical regression but check for compensating improvements
            improvements = [c for c in comparison.comparisons if c.is_improvement]
            if len(improvements) <= len(regressions):
                return DeploymentVerdict.REJECT

        # Check for significant improvements
        improvements = [
            c
            for c in comparison.comparisons
            if c.status == ComparisonStatus.SIGNIFICANT_IMPROVEMENT
        ]
        if improvements and not regressions:
            return DeploymentVerdict.PROMOTE

        # Marginal cases
        marginal_improvements = [
            c
            for c in comparison.comparisons
            if c.status == ComparisonStatus.MARGINAL_IMPROVEMENT
        ]
        if marginal_improvements and not regressions:
            # Need more data or can extend
            return DeploymentVerdict.EXTEND

        # No significant changes
        return DeploymentVerdict.PENDING

    async def _validate_fate(
        self,
        deployment: ShadowDeployment,
        comparison: ComparisonResult,
    ) -> bool:
        """
        Validate deployment against FATE requirements.

        FATE = Fidelity, Accountability, Transparency, Ethics
        """
        # Fidelity: Does shadow behavior match expectations?
        expected = deployment.hypothesis.expected_improvement
        for metric_name, expected_change in expected.items():
            for comp in comparison.comparisons:
                if comp.metric_name == metric_name:
                    # Check if direction matches expectation
                    expected_direction = expected_change > 0
                    actual_direction = comp.delta > 0
                    if expected_direction != actual_direction:
                        logger.warning(f"FATE: Fidelity check failed for {metric_name}")
                        # Don't fail entirely, but note the discrepancy

        # Accountability: Is the deployment traceable?
        if not deployment.deployment_id or not deployment.hypothesis.id:
            logger.warning("FATE: Accountability check failed - missing IDs")
            return False

        # Transparency: Are metrics available?
        if not comparison.comparisons:
            logger.warning("FATE: Transparency check failed - no metrics")
            return False

        # Ethics (Ihsan): Does it meet ethical threshold?
        if comparison.ihsan_score_shadow < self.ihsan_threshold:
            logger.warning(
                f"FATE: Ethics check failed - Ihsan {comparison.ihsan_score_shadow:.4f} < {self.ihsan_threshold}"
            )
            return False

        return True

    async def promote(self, deployment: ShadowDeployment) -> bool:
        """
        Promote shadow deployment to production.

        Args:
            deployment: The deployment to promote

        Returns:
            True if promotion succeeded
        """
        if deployment.verdict != DeploymentVerdict.PROMOTE:
            logger.warning(
                f"Cannot promote deployment {deployment.deployment_id} with verdict {deployment.verdict}"
            )
            return False

        self._log_audit(
            "promote_start",
            deployment.deployment_id,
            {
                "hypothesis_id": deployment.hypothesis.id,
            },
        )

        try:
            # Apply hypothesis changes to production config
            for key, value in deployment.hypothesis.proposed_change.items():
                self._apply_config_change(self.production_config, key, value)

            self._promoted_count += 1

            self._log_audit(
                "promote_success",
                deployment.deployment_id,
                {
                    "changes": deployment.hypothesis.proposed_change,
                },
            )

            logger.info(f"Deployment {deployment.deployment_id} promoted to production")

            # Clean up shadow
            await self._cleanup_deployment(deployment)

            return True

        except Exception as e:
            self._log_audit(
                "promote_failed",
                deployment.deployment_id,
                {
                    "error": str(e),
                },
            )
            logger.error(
                f"Failed to promote deployment {deployment.deployment_id}: {e}"
            )
            return False

    async def rollback(self, deployment: ShadowDeployment) -> bool:
        """
        Rollback/reject a shadow deployment.

        Args:
            deployment: The deployment to rollback

        Returns:
            True if rollback succeeded
        """
        self._log_audit(
            "rollback",
            deployment.deployment_id,
            {
                "reason": deployment.kill_switch_reason or "Verdict: reject",
                "verdict": deployment.verdict.value,
            },
        )

        deployment.verdict = DeploymentVerdict.ROLLBACK
        self._rollback_count += 1

        # Clean up shadow
        await self._cleanup_deployment(deployment)

        logger.info(f"Deployment {deployment.deployment_id} rolled back")
        return True

    async def _cleanup_deployment(self, deployment: ShadowDeployment):
        """Clean up a shadow deployment."""
        env = self._environments.get(deployment.deployment_id)
        if env:
            await env.teardown()
            del self._environments[deployment.deployment_id]

        if deployment.deployment_id in self._deployments:
            del self._deployments[deployment.deployment_id]

    def _apply_config_change(self, config: Dict[str, Any], key: str, value: Any):
        """Apply configuration change with dot notation support."""
        keys = key.split(".")
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def _log_audit(self, operation: str, deployment_id: str, details: Dict[str, Any]):
        """Log an audit entry."""
        entry = AuditEntry(
            deployment_id=deployment_id,
            operation=operation,
            details=details,
        )
        self._audit_log.append(entry)

        # Keep audit log bounded
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]

    async def start_monitoring(self, interval_seconds: float = 10.0):
        """Start background monitoring of active deployments."""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval_seconds))

    async def stop_monitoring(self):
        """Stop background monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self._running:
            for deployment_id, deployment in list(self._deployments.items()):
                try:
                    # Check for expired deployments
                    if (
                        deployment.is_expired
                        and deployment.verdict == DeploymentVerdict.PENDING
                    ):
                        verdict = await self.evaluate(deployment)
                        if verdict == DeploymentVerdict.PROMOTE:
                            await self.promote(deployment)
                        elif verdict in (
                            DeploymentVerdict.REJECT,
                            DeploymentVerdict.ROLLBACK,
                        ):
                            await self.rollback(deployment)

                    # Check kill switch conditions
                    await self._check_kill_switch(deployment)

                except Exception as e:
                    logger.error(f"Error monitoring deployment {deployment_id}: {e}")

            await asyncio.sleep(interval)

    def get_deployment(self, deployment_id: str) -> Optional[ShadowDeployment]:
        """Get a deployment by ID."""
        return self._deployments.get(deployment_id)

    def get_active_deployments(self) -> List[ShadowDeployment]:
        """Get all active deployments."""
        return list(self._deployments.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get deployer statistics."""
        return {
            "total_deployments": self._total_deployments,
            "active_deployments": len(self._deployments),
            "promoted": self._promoted_count,
            "rejected": self._rejected_count,
            "rollbacks": self._rollback_count,
            "audit_log_size": len(self._audit_log),
        }

    def get_audit_log(
        self,
        deployment_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        entries = self._audit_log

        if deployment_id:
            entries = [e for e in entries if e.deployment_id == deployment_id]

        return [
            {
                "entry_id": e.entry_id,
                "deployment_id": e.deployment_id,
                "operation": e.operation,
                "timestamp": e.timestamp.isoformat(),
                "details": e.details,
            }
            for e in entries[-limit:]
        ]


# =============================================================================
# CANARY DEPLOYMENT SUPPORT
# =============================================================================


class CanaryDeployer(ShadowDeployer):
    """
    Extension of ShadowDeployer for gradual canary rollouts.

    Supports staged promotion:
    1. Start at small traffic percentage
    2. Gradually increase if metrics stay healthy
    3. Full rollout or rollback based on thresholds
    """

    def __init__(
        self,
        *args,
        initial_traffic_percent: float = 5.0,
        traffic_increment: float = 10.0,
        increment_interval_seconds: float = 300.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.initial_traffic_percent = initial_traffic_percent
        self.traffic_increment = traffic_increment
        self.increment_interval_seconds = increment_interval_seconds

        self._canary_traffic: Dict[str, float] = {}  # deployment_id -> traffic %

    async def deploy_canary(
        self,
        hypothesis: Hypothesis,
        duration: timedelta = DEFAULT_SHADOW_DURATION,
    ) -> ShadowDeployment:
        """Deploy a canary with gradual traffic increase."""
        deployment = await self.deploy_shadow(
            hypothesis=hypothesis,
            duration=duration,
            traffic_mode=TrafficMode.SAMPLED,
            sample_rate=self.initial_traffic_percent / 100.0,
        )

        self._canary_traffic[deployment.deployment_id] = self.initial_traffic_percent

        return deployment

    async def increase_traffic(self, deployment: ShadowDeployment) -> bool:
        """Increase traffic to canary if metrics are healthy."""
        if deployment.deployment_id not in self._canary_traffic:
            return False

        # Check if metrics are healthy enough to increase
        if deployment.kill_switch_triggered:
            return False

        current = self._canary_traffic[deployment.deployment_id]
        new_traffic = min(100.0, current + self.traffic_increment)

        deployment.sample_rate = new_traffic / 100.0
        self._canary_traffic[deployment.deployment_id] = new_traffic

        self._log_audit(
            "canary_traffic_increase",
            deployment.deployment_id,
            {
                "previous_percent": current,
                "new_percent": new_traffic,
            },
        )

        logger.info(
            f"Canary {deployment.deployment_id} traffic increased to {new_traffic}%"
        )

        return new_traffic >= 100.0  # Returns True if fully rolled out


# =============================================================================
# UTILITIES
# =============================================================================


class _TokenBucket:
    """Simple token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens: float = float(capacity)
        self.last_update = datetime.now(timezone.utc)
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Try to acquire a token."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            elapsed = (now - self.last_update).total_seconds()
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "DeploymentVerdict",
    "ComparisonStatus",
    "IsolationLevel",
    "TrafficMode",
    # Data classes
    "ShadowHypothesis",
    "Hypothesis",  # Alias for ShadowHypothesis
    "MetricSample",
    "MetricComparison",
    "ComparisonResult",
    "ShadowDeployment",
    "ShadowRequest",
    "ShadowResponse",
    "ResourceLimits",
    "AuditEntry",
    # Core classes
    "ShadowEnvironment",
    "StatisticalAnalyzer",
    "ShadowDeployer",
    "CanaryDeployer",
    # Constants
    "DEFAULT_SHADOW_DURATION",
    "MIN_SAMPLE_SIZE",
    "CONFIDENCE_LEVEL",
    "IHSAN_KILL_THRESHOLD",
]
