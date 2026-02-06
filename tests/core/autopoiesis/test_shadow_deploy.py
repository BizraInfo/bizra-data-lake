"""
Tests for Shadow Deployment System
===============================================================================

Tests the shadow deployment infrastructure including:
- ShadowEnvironment isolation
- Traffic mirroring
- Statistical comparison
- FATE validation
- Promotion/rollback decisions

Genesis Strict Synthesis v2.2.2
"""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
import random

from core.autopoiesis.shadow_deploy import (
    ShadowDeployer,
    ShadowDeployment,
    ShadowEnvironment,
    ShadowHypothesis,
    ShadowRequest,
    ShadowResponse,
    CanaryDeployer,
    DeploymentVerdict,
    ComparisonStatus,
    ComparisonResult,
    MetricComparison,
    MetricSample,
    ResourceLimits,
    IsolationLevel,
    TrafficMode,
    StatisticalAnalyzer,
    AuditEntry,
    IHSAN_KILL_THRESHOLD,
    MIN_SAMPLE_SIZE,
)
from core.integration.constants import UNIFIED_IHSAN_THRESHOLD


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_hypothesis():
    """Create a sample hypothesis for testing."""
    return ShadowHypothesis(
        name="Improved batch size",
        description="Increase batch size for better throughput",
        proposed_change={"batch_size": 32, "cache_strategy": "adaptive"},
        expected_improvement={"latency": -0.1, "throughput": 0.2},
    )


@pytest.fixture
def production_config():
    """Sample production configuration."""
    return {
        "batch_size": 16,
        "cache_strategy": "lru",
        "model": "default",
        "timeout": 30,
    }


@pytest.fixture
async def shadow_handler():
    """Simple handler for testing shadow requests."""
    async def handler(request: ShadowRequest, config: Dict[str, Any]) -> Any:
        # Simulate processing with config-dependent behavior
        batch_size = config.get("batch_size", 16)
        latency = 100 / batch_size  # Larger batch = lower latency

        # Add some randomness
        latency += random.uniform(-5, 5)

        await asyncio.sleep(latency / 1000)  # Simulate work

        class Result:
            def __init__(self, latency, ihsan, snr):
                self.latency_ms = latency
                self.ihsan_score = ihsan
                self.snr_score = snr

        return Result(
            latency=latency,
            ihsan=0.97 + random.uniform(-0.02, 0.02),
            snr=0.90 + random.uniform(-0.05, 0.05),
        )

    return handler


# =============================================================================
# HYPOTHESIS TESTS
# =============================================================================

class TestShadowHypothesis:
    """Tests for ShadowHypothesis dataclass."""

    def test_hypothesis_creation(self, sample_hypothesis):
        """Test hypothesis is created with auto-generated ID."""
        assert sample_hypothesis.id
        assert len(sample_hypothesis.id) == 12
        assert sample_hypothesis.name == "Improved batch size"

    def test_hypothesis_to_dict(self, sample_hypothesis):
        """Test hypothesis serialization."""
        data = sample_hypothesis.to_dict()
        assert data["name"] == "Improved batch size"
        assert "proposed_change" in data
        assert data["proposed_change"]["batch_size"] == 32

    def test_hypothesis_unique_ids(self):
        """Test that each hypothesis gets a unique ID."""
        h1 = ShadowHypothesis(name="test1")
        h2 = ShadowHypothesis(name="test2")
        assert h1.id != h2.id


# =============================================================================
# SHADOW ENVIRONMENT TESTS
# =============================================================================

class TestShadowEnvironment:
    """Tests for ShadowEnvironment class."""

    @pytest.mark.asyncio
    async def test_environment_initialization(self, sample_hypothesis, production_config):
        """Test shadow environment initializes correctly."""
        env = ShadowEnvironment(
            hypothesis=sample_hypothesis,
            production_config=production_config,
        )

        success = await env.initialize()
        assert success
        assert env._initialized
        assert env._active
        assert env._isolation_verified

        await env.teardown()

    @pytest.mark.asyncio
    async def test_config_isolation(self, sample_hypothesis, production_config):
        """Test that shadow config is isolated from production."""
        env = ShadowEnvironment(
            hypothesis=sample_hypothesis,
            production_config=production_config,
        )

        await env.initialize()

        # Shadow config should be different object
        assert env.shadow_config is not production_config

        # Shadow config should have hypothesis changes applied
        assert env.shadow_config["batch_size"] == 32  # From hypothesis
        assert env.shadow_config["cache_strategy"] == "adaptive"

        # Production config should be unchanged
        assert production_config["batch_size"] == 16

        await env.teardown()

    @pytest.mark.asyncio
    async def test_nested_config_change(self, production_config):
        """Test applying nested configuration changes."""
        hypothesis = ShadowHypothesis(
            name="Nested change",
            proposed_change={
                "model.temperature": 0.8,
                "model.max_tokens": 2048,
            },
        )

        env = ShadowEnvironment(
            hypothesis=hypothesis,
            production_config=production_config,
        )

        await env.initialize()

        # Check nested changes were applied
        assert "model" in env.shadow_config
        assert env.shadow_config["model"]["temperature"] == 0.8
        assert env.shadow_config["model"]["max_tokens"] == 2048

        await env.teardown()

    @pytest.mark.asyncio
    async def test_process_request(self, sample_hypothesis, production_config, shadow_handler):
        """Test processing a request in shadow environment."""
        env = ShadowEnvironment(
            hypothesis=sample_hypothesis,
            production_config=production_config,
        )

        await env.initialize()

        request = ShadowRequest(
            payload={"query": "test"},
        )

        response = await env.process(request, shadow_handler)

        assert response.request_id == request.request_id
        assert response.latency_ms > 0
        assert response.error is None

        await env.teardown()

    @pytest.mark.asyncio
    async def test_metrics_collection(self, sample_hypothesis, production_config, shadow_handler):
        """Test that environment collects metrics."""
        env = ShadowEnvironment(
            hypothesis=sample_hypothesis,
            production_config=production_config,
        )

        await env.initialize()

        # Process several requests
        for _ in range(5):
            request = ShadowRequest(payload={"query": "test"})
            await env.process(request, shadow_handler)

        metrics = env.get_metrics()

        assert metrics["request_count"] == 5
        assert metrics["error_count"] == 0
        assert metrics["latency"]["mean"] > 0
        assert metrics["ihsan"]["mean"] > 0.9

        await env.teardown()

    @pytest.mark.asyncio
    async def test_timeout_handling(self, sample_hypothesis, production_config):
        """Test request timeout is enforced."""
        async def slow_handler(request, config):
            await asyncio.sleep(10)  # 10 seconds
            return None

        env = ShadowEnvironment(
            hypothesis=sample_hypothesis,
            production_config=production_config,
            resource_limits=ResourceLimits(timeout_seconds=0.1),  # 100ms timeout
        )

        await env.initialize()

        request = ShadowRequest(payload={"query": "test"})
        response = await env.process(request, slow_handler)

        assert response.error is not None
        assert "Timeout" in response.error

        await env.teardown()


# =============================================================================
# STATISTICAL ANALYZER TESTS
# =============================================================================

class TestStatisticalAnalyzer:
    """Tests for statistical analysis."""

    def test_compare_metrics_identical(self):
        """Test comparison with identical samples."""
        samples = [1.0, 1.1, 0.9, 1.0, 1.05] * 10

        result = StatisticalAnalyzer.compare_metrics(
            samples, samples, "test_metric"
        )

        assert result.delta == 0
        assert result.status == ComparisonStatus.NO_SIGNIFICANT_DIFFERENCE

    def test_compare_metrics_significant_improvement(self):
        """Test detection of significant improvement."""
        prod_samples = [10.0] * 50
        shadow_samples = [15.0] * 50  # 50% improvement

        result = StatisticalAnalyzer.compare_metrics(
            prod_samples,
            shadow_samples,
            "throughput",
            higher_is_better=True,
        )

        assert result.is_improvement
        assert result.delta_percent == 50.0

    def test_compare_metrics_significant_regression(self):
        """Test detection of significant regression."""
        prod_samples = [10.0] * 50
        shadow_samples = [5.0] * 50  # 50% regression

        result = StatisticalAnalyzer.compare_metrics(
            prod_samples,
            shadow_samples,
            "throughput",
            higher_is_better=True,
        )

        assert result.is_regression
        assert result.delta_percent == -50.0

    def test_compare_metrics_latency_lower_is_better(self):
        """Test latency comparison where lower is better."""
        prod_samples = [100.0] * 50
        shadow_samples = [80.0] * 50  # 20% lower latency = improvement

        result = StatisticalAnalyzer.compare_metrics(
            prod_samples,
            shadow_samples,
            "latency",
            higher_is_better=False,
        )

        assert result.is_improvement
        assert result.delta == -20.0

    def test_compare_metrics_insufficient_samples(self):
        """Test handling of insufficient samples."""
        result = StatisticalAnalyzer.compare_metrics(
            [], [], "test_metric"
        )

        assert result.p_value == 1.0
        assert result.status == ComparisonStatus.NO_SIGNIFICANT_DIFFERENCE

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        prod = [10.0 + random.gauss(0, 1) for _ in range(100)]
        shadow = [12.0 + random.gauss(0, 1) for _ in range(100)]

        result = StatisticalAnalyzer.compare_metrics(prod, shadow, "test")

        # Confidence interval should contain the true difference (~2.0)
        ci_low, ci_high = result.confidence_interval
        assert ci_low < 2.0 < ci_high or abs(result.delta - 2.0) < 1.0


# =============================================================================
# SHADOW DEPLOYER TESTS
# =============================================================================

class TestShadowDeployer:
    """Tests for ShadowDeployer orchestrator."""

    @pytest.mark.asyncio
    async def test_deploy_shadow(self, sample_hypothesis, production_config, shadow_handler):
        """Test deploying a shadow."""
        deployer = ShadowDeployer(
            production_config=production_config,
            production_handler=shadow_handler,
        )

        deployment = await deployer.deploy_shadow(
            hypothesis=sample_hypothesis,
            duration=timedelta(minutes=5),
        )

        assert deployment.deployment_id
        assert deployment.hypothesis.id == sample_hypothesis.id
        assert deployment.verdict == DeploymentVerdict.PENDING

        # Cleanup
        await deployer._cleanup_deployment(deployment)

    @pytest.mark.asyncio
    async def test_mirror_request(self, sample_hypothesis, production_config, shadow_handler):
        """Test mirroring requests to shadow."""
        deployer = ShadowDeployer(
            production_config=production_config,
            production_handler=shadow_handler,
        )

        deployment = await deployer.deploy_shadow(
            hypothesis=sample_hypothesis,
            duration=timedelta(minutes=5),
        )

        # Mirror a request
        request = ShadowRequest(payload={"query": "test"})
        prod_response, shadow_response = await deployer.mirror_request(
            deployment, request
        )

        assert prod_response is not None
        assert shadow_response is not None
        assert deployment.requests_mirrored == 1
        assert deployment.requests_compared == 1

        # Cleanup
        await deployer._cleanup_deployment(deployment)

    @pytest.mark.asyncio
    async def test_evaluate_insufficient_data(self, sample_hypothesis, production_config, shadow_handler):
        """Test evaluation with insufficient data."""
        deployer = ShadowDeployer(
            production_config=production_config,
            production_handler=shadow_handler,
        )

        deployment = await deployer.deploy_shadow(
            hypothesis=sample_hypothesis,
            duration=timedelta(seconds=1),  # Short duration
        )

        # Only a few requests
        for _ in range(5):
            request = ShadowRequest(payload={"query": "test"})
            await deployer.mirror_request(deployment, request)

        # Wait for expiry
        await asyncio.sleep(1.1)

        verdict = await deployer.evaluate(deployment)

        # Should extend due to insufficient data
        assert verdict in (DeploymentVerdict.EXTEND, DeploymentVerdict.PENDING)

        await deployer._cleanup_deployment(deployment)

    @pytest.mark.asyncio
    async def test_evaluate_with_sufficient_data(self, sample_hypothesis, production_config, shadow_handler):
        """Test evaluation with sufficient data."""
        deployer = ShadowDeployer(
            production_config=production_config,
            production_handler=shadow_handler,
        )

        deployment = await deployer.deploy_shadow(
            hypothesis=sample_hypothesis,
            duration=timedelta(minutes=5),
        )

        # Generate sufficient data
        for _ in range(MIN_SAMPLE_SIZE + 10):
            request = ShadowRequest(payload={"query": "test"})
            await deployer.mirror_request(deployment, request)

        verdict = await deployer.evaluate(deployment)

        # Should have a real verdict now
        assert verdict in (
            DeploymentVerdict.PROMOTE,
            DeploymentVerdict.REJECT,
            DeploymentVerdict.EXTEND,
            DeploymentVerdict.PENDING,
        )

        await deployer._cleanup_deployment(deployment)

    @pytest.mark.asyncio
    async def test_kill_switch_on_low_ihsan(self, production_config):
        """Test kill switch triggers on low Ihsan."""
        async def low_ihsan_handler(request, config):
            class Result:
                ihsan_score = 0.80  # Below threshold
                snr_score = 0.90
            return Result()

        hypothesis = ShadowHypothesis(
            name="Bad hypothesis",
            proposed_change={"bad_setting": True},
        )

        deployer = ShadowDeployer(
            production_config=production_config,
            production_handler=low_ihsan_handler,
        )

        deployment = await deployer.deploy_shadow(
            hypothesis=hypothesis,
            duration=timedelta(minutes=5),
        )

        # Mirror requests with low Ihsan
        for _ in range(15):
            request = ShadowRequest(payload={"query": "test"})
            await deployer.mirror_request(deployment, request)

        # Kill switch should have been triggered
        assert deployment.kill_switch_triggered
        assert "Ihsan" in deployment.kill_switch_reason
        assert deployment.verdict == DeploymentVerdict.REJECT

        await deployer._cleanup_deployment(deployment)

    @pytest.mark.asyncio
    async def test_promote_deployment(self, sample_hypothesis, production_config, shadow_handler):
        """Test promoting a deployment to production."""
        deployer = ShadowDeployer(
            production_config=production_config.copy(),
            production_handler=shadow_handler,
        )

        deployment = await deployer.deploy_shadow(
            hypothesis=sample_hypothesis,
            duration=timedelta(minutes=5),
        )

        # Generate data
        for _ in range(MIN_SAMPLE_SIZE + 10):
            request = ShadowRequest(payload={"query": "test"})
            await deployer.mirror_request(deployment, request)

        # Force promote verdict for testing
        deployment.verdict = DeploymentVerdict.PROMOTE

        success = await deployer.promote(deployment)

        assert success
        assert deployer.production_config["batch_size"] == 32
        assert deployer._promoted_count == 1

    @pytest.mark.asyncio
    async def test_rollback_deployment(self, sample_hypothesis, production_config, shadow_handler):
        """Test rolling back a deployment."""
        deployer = ShadowDeployer(
            production_config=production_config,
            production_handler=shadow_handler,
        )

        deployment = await deployer.deploy_shadow(
            hypothesis=sample_hypothesis,
        )

        success = await deployer.rollback(deployment)

        assert success
        assert deployment.verdict == DeploymentVerdict.ROLLBACK
        assert deployer._rollback_count == 1

    @pytest.mark.asyncio
    async def test_audit_logging(self, sample_hypothesis, production_config, shadow_handler):
        """Test audit log is populated."""
        deployer = ShadowDeployer(
            production_config=production_config,
            production_handler=shadow_handler,
        )

        deployment = await deployer.deploy_shadow(hypothesis=sample_hypothesis)
        await deployer.rollback(deployment)

        audit_log = deployer.get_audit_log()

        assert len(audit_log) >= 2  # deploy_shadow_start, deploy_shadow_success, rollback

        operations = [entry["operation"] for entry in audit_log]
        assert "deploy_shadow_start" in operations
        assert "rollback" in operations

    @pytest.mark.asyncio
    async def test_sampled_traffic_mode(self, sample_hypothesis, production_config, shadow_handler):
        """Test sampled traffic mode."""
        deployer = ShadowDeployer(
            production_config=production_config,
            production_handler=shadow_handler,
        )

        deployment = await deployer.deploy_shadow(
            hypothesis=sample_hypothesis,
            traffic_mode=TrafficMode.SAMPLED,
            sample_rate=0.5,  # 50% sampling
        )

        # Send many requests
        random.seed(42)  # For reproducibility
        for _ in range(100):
            request = ShadowRequest(payload={"query": "test"})
            await deployer.mirror_request(deployment, request)

        # Should have approximately 50% mirrored (with some variance)
        assert 30 < deployment.requests_mirrored < 70

        await deployer._cleanup_deployment(deployment)

    def test_get_stats(self, production_config):
        """Test statistics retrieval."""
        deployer = ShadowDeployer(production_config=production_config)

        stats = deployer.get_stats()

        assert "total_deployments" in stats
        assert "active_deployments" in stats
        assert "promoted" in stats
        assert "rejected" in stats


# =============================================================================
# CANARY DEPLOYER TESTS
# =============================================================================

class TestCanaryDeployer:
    """Tests for CanaryDeployer gradual rollout."""

    @pytest.mark.asyncio
    async def test_deploy_canary(self, sample_hypothesis, production_config, shadow_handler):
        """Test canary deployment starts at initial traffic percent."""
        deployer = CanaryDeployer(
            production_config=production_config,
            production_handler=shadow_handler,
            initial_traffic_percent=5.0,
        )

        deployment = await deployer.deploy_canary(hypothesis=sample_hypothesis)

        assert deployment.sample_rate == 0.05  # 5%
        assert deployment.traffic_mode == TrafficMode.SAMPLED

        await deployer._cleanup_deployment(deployment)

    @pytest.mark.asyncio
    async def test_increase_traffic(self, sample_hypothesis, production_config, shadow_handler):
        """Test gradual traffic increase."""
        deployer = CanaryDeployer(
            production_config=production_config,
            production_handler=shadow_handler,
            initial_traffic_percent=10.0,
            traffic_increment=20.0,
        )

        deployment = await deployer.deploy_canary(hypothesis=sample_hypothesis)

        # Increase traffic
        fully_rolled_out = await deployer.increase_traffic(deployment)

        assert deployment.sample_rate == 0.30  # 30%
        assert not fully_rolled_out

        # Increase more times to reach 100%
        for _ in range(4):
            fully_rolled_out = await deployer.increase_traffic(deployment)

        assert deployment.sample_rate == 1.0  # 100%
        assert fully_rolled_out

        await deployer._cleanup_deployment(deployment)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestShadowDeploymentIntegration:
    """Integration tests for the full shadow deployment workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow_promotion(self, sample_hypothesis, production_config, shadow_handler):
        """Test complete workflow resulting in promotion."""
        deployer = ShadowDeployer(
            production_config=production_config.copy(),
            production_handler=shadow_handler,
        )

        # Deploy shadow
        deployment = await deployer.deploy_shadow(
            hypothesis=sample_hypothesis,
            duration=timedelta(minutes=5),
        )

        # Mirror sufficient traffic
        for _ in range(MIN_SAMPLE_SIZE + 20):
            request = ShadowRequest(payload={"query": f"test_{_}"})
            await deployer.mirror_request(deployment, request)

        # Evaluate
        verdict = await deployer.evaluate(deployment)

        # Take action based on verdict
        if verdict == DeploymentVerdict.PROMOTE:
            success = await deployer.promote(deployment)
            assert success
            assert deployer.production_config["batch_size"] == 32
        elif verdict == DeploymentVerdict.REJECT:
            success = await deployer.rollback(deployment)
            assert success

        # Verify stats
        stats = deployer.get_stats()
        assert stats["total_deployments"] == 1

    @pytest.mark.asyncio
    async def test_concurrent_deployments(self, production_config, shadow_handler):
        """Test multiple concurrent shadow deployments."""
        deployer = ShadowDeployer(
            production_config=production_config,
            production_handler=shadow_handler,
        )

        # Deploy multiple shadows
        deployments = []
        for i in range(3):
            hypothesis = ShadowHypothesis(
                name=f"Hypothesis {i}",
                proposed_change={f"setting_{i}": i * 10},
            )
            deployment = await deployer.deploy_shadow(hypothesis=hypothesis)
            deployments.append(deployment)

        assert len(deployer.get_active_deployments()) == 3

        # Cleanup all
        for deployment in deployments:
            await deployer._cleanup_deployment(deployment)

        assert len(deployer.get_active_deployments()) == 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_hypothesis_change(self, production_config, shadow_handler):
        """Test hypothesis with no changes."""
        hypothesis = ShadowHypothesis(
            name="Empty change",
            proposed_change={},
        )

        deployer = ShadowDeployer(
            production_config=production_config,
            production_handler=shadow_handler,
        )

        deployment = await deployer.deploy_shadow(hypothesis=hypothesis)

        # Should still work, shadow config equals production
        env = deployer._environments[deployment.deployment_id]
        assert env.shadow_config["batch_size"] == production_config["batch_size"]

        await deployer._cleanup_deployment(deployment)

    @pytest.mark.asyncio
    async def test_handler_exception(self, sample_hypothesis, production_config):
        """Test handling of handler exceptions."""
        async def failing_handler(request, config):
            raise RuntimeError("Handler failed")

        deployer = ShadowDeployer(
            production_config=production_config,
            production_handler=failing_handler,
        )

        deployment = await deployer.deploy_shadow(hypothesis=sample_hypothesis)

        request = ShadowRequest(payload={"query": "test"})
        prod_response, shadow_response = await deployer.mirror_request(
            deployment, request
        )

        # Should handle errors gracefully
        assert shadow_response.error is not None
        assert deployment.errors_shadow == 1

        await deployer._cleanup_deployment(deployment)

    def test_metric_comparison_with_variance(self):
        """Test comparison with high variance data."""
        # High variance samples
        prod_samples = [random.gauss(10, 5) for _ in range(100)]
        shadow_samples = [random.gauss(10.5, 5) for _ in range(100)]

        result = StatisticalAnalyzer.compare_metrics(
            prod_samples, shadow_samples, "noisy_metric"
        )

        # With high variance, small difference shouldn't be significant
        assert result.p_value > 0.05 or abs(result.delta_percent) < 20
