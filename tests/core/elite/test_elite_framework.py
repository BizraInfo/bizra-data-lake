"""
Tests for Elite Framework — PMBOK + DevOps + Ihsān Integration

Tests:
- Quality gates (SNR/Ihsān validation)
- CI/CD pipeline with constitutional enforcement
- Risk management with cascading analysis
- SAPE optimizer with Graph-of-Thoughts
"""

import pytest
import asyncio
from typing import Dict, Any

from core.elite import (
    ELITE_VERSION,
    PMBOK_KNOWLEDGE_AREAS,
    IHSAN_DIMENSIONS,
    SAPE_LAYERS,
    SNR_TARGETS,
)
from core.elite.quality_gates import (
    QualityGate,
    QualityGateChain,
    GateCriterion,
    GateResult,
    GateStatus,
    GateLevel,
)
from core.elite.pipeline import (
    ElitePipeline,
    PipelineStage,
    PipelineStatus,
    PipelineRun,
    StageResult,
    SourceStageHandler,
    BuildStageHandler,
    TestStageHandler,
    SecurityStageHandler,
    QualityStageHandler,
)
from core.elite.risk import (
    RiskManager,
    Risk,
    RiskCategory,
    RiskSeverity,
    RiskStatus,
    MitigationStrategy,
    CascadeAnalysis,
)
from core.elite.sape import (
    SAPEOptimizer,
    GraphOfThoughts,
    ThoughtNode,
    ThoughtNodeType,
    SAPELayer,
    SAPEResult,
)


# =============================================================================
# MODULE CONSTANTS TESTS
# =============================================================================


class TestEliteConstants:
    """Tests for elite module constants."""

    def test_elite_version_format(self):
        """Version follows semantic versioning."""
        parts = ELITE_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_pmbok_knowledge_areas_complete(self):
        """All 10 PMBOK knowledge areas mapped."""
        expected_areas = [
            "integration", "scope", "schedule", "cost", "quality",
            "resource", "communications", "risk", "procurement", "stakeholder"
        ]
        for area in expected_areas:
            assert area in PMBOK_KNOWLEDGE_AREAS

    def test_ihsan_dimensions_sum_to_one(self):
        """Ihsān dimension weights sum to 1.0."""
        total = sum(IHSAN_DIMENSIONS.values())
        assert abs(total - 1.0) < 0.01  # Allow small float error

    def test_ihsan_correctness_safety_highest(self):
        """Correctness and safety have highest weights."""
        assert IHSAN_DIMENSIONS["correctness"] >= 0.20
        assert IHSAN_DIMENSIONS["safety"] >= 0.20

    def test_sape_layers_order(self):
        """SAPE layers in correct progression order."""
        assert SAPE_LAYERS == ["data", "information", "knowledge", "wisdom"]

    def test_snr_targets_increasing(self):
        """SNR targets increase through layers."""
        assert SNR_TARGETS["data"] < SNR_TARGETS["information"]
        assert SNR_TARGETS["information"] < SNR_TARGETS["knowledge"]
        assert SNR_TARGETS["knowledge"] < SNR_TARGETS["wisdom"]

    def test_snr_wisdom_near_perfect(self):
        """Wisdom layer requires near-perfect SNR."""
        assert SNR_TARGETS["wisdom"] >= 0.999


# =============================================================================
# QUALITY GATE TESTS
# =============================================================================


class TestGateCriterion:
    """Tests for individual gate criteria."""

    def test_criterion_creation(self):
        """Criterion can be created with required fields."""
        criterion = GateCriterion(
            name="test_criterion",
            description="Test description",
            threshold=0.9,
        )
        assert criterion.name == "test_criterion"
        assert criterion.threshold == 0.9
        assert criterion.level == GateLevel.STANDARD

    def test_criterion_validate_passing(self):
        """Value above threshold passes."""
        criterion = GateCriterion(
            name="test",
            description="Test",
            threshold=0.8,
        )
        passed, score = criterion.validate(0.9)
        assert passed is True
        assert score == 0.9

    def test_criterion_validate_failing(self):
        """Value below threshold fails."""
        criterion = GateCriterion(
            name="test",
            description="Test",
            threshold=0.8,
        )
        passed, score = criterion.validate(0.7)
        assert passed is False
        assert score == 0.7

    def test_criterion_custom_validator(self):
        """Custom validator function works."""
        criterion = GateCriterion(
            name="test",
            description="Test",
            threshold=0.5,
            validator=lambda x: x / 100,  # Convert percentage
        )
        passed, score = criterion.validate(80)
        assert passed is True
        assert score == 0.8


class TestQualityGate:
    """Tests for quality gates."""

    def test_gate_creation(self):
        """Gate can be created with defaults."""
        gate = QualityGate(name="test_gate")
        assert gate.name == "test_gate"
        assert gate.sape_layer == "data"

    def test_gate_snr_threshold_by_layer(self):
        """SNR threshold matches SAPE layer."""
        gate = QualityGate(name="knowledge_gate", sape_layer="knowledge")
        assert gate.snr_threshold == 0.99

    @pytest.mark.asyncio
    async def test_gate_entry_validation(self):
        """Entry validation works."""
        gate = QualityGate(name="test_gate")
        result = await gate.validate_entry({"_default": 0.9})
        assert isinstance(result, GateResult)
        assert result.gate_name == "test_gate_entry"

    @pytest.mark.asyncio
    async def test_gate_exit_validation(self):
        """Exit validation with Ihsān checking."""
        gate = QualityGate(name="test_gate")

        # Provide high scores for all Ihsān dimensions
        artifact = {
            "ihsan_correctness": 0.95,
            "ihsan_safety": 0.95,
            "ihsan_user_benefit": 0.95,
            "ihsan_efficiency": 0.95,
            "ihsan_auditability": 0.95,
            "ihsan_anti_centralization": 0.95,
            "ihsan_robustness": 0.95,
            "ihsan_adl_justice": 0.95,
            "snr_threshold": 0.95,
        }

        result = await gate.validate_exit(artifact)
        assert result.ihsan_score >= 0.9

    @pytest.mark.asyncio
    async def test_gate_full_validation(self):
        """Full validation (entry + exit) works."""
        gate = QualityGate(name="test_gate")
        entry_result, exit_result = await gate.validate({"_default": 0.95})
        assert isinstance(entry_result, GateResult)
        assert isinstance(exit_result, GateResult)

    def test_add_custom_criterion(self):
        """Custom criteria can be added."""
        gate = QualityGate(name="test_gate")
        gate.add_exit_criterion(
            name="custom_check",
            description="Custom validation",
            threshold=0.9,
            level=GateLevel.MANDATORY,
        )
        assert len(gate._exit_criteria) > 0


class TestQualityGateChain:
    """Tests for quality gate chains."""

    def test_chain_creation(self):
        """Chain creates gates for all SAPE layers."""
        chain = QualityGateChain()
        for layer in SAPE_LAYERS:
            assert layer in chain._gates

    def test_chain_get_gate(self):
        """Can retrieve gate by layer."""
        chain = QualityGateChain()
        gate = chain.get_gate("knowledge")
        assert gate.sape_layer == "knowledge"

    @pytest.mark.asyncio
    async def test_chain_progression(self):
        """Chain validates progression through layers."""
        chain = QualityGateChain()

        artifacts = {
            "data": {
                "snr_threshold": 0.95,
                "ihsan_correctness": 0.95,
                "ihsan_safety": 0.95,
            }
        }

        results = await chain.validate_progression(artifacts)
        assert "data" in results

    def test_chain_summary(self):
        """Chain summary includes all config."""
        chain = QualityGateChain()
        summary = chain.get_summary()
        assert "layers" in summary
        assert "snr_targets" in summary
        assert "ihsan_threshold" in summary


# =============================================================================
# PIPELINE TESTS
# =============================================================================


class TestPipelineStages:
    """Tests for pipeline stage handlers."""

    @pytest.mark.asyncio
    async def test_source_stage(self):
        """Source stage executes correctly."""
        handler = SourceStageHandler()
        artifacts, logs = await handler.execute({}, {"code_coverage": 0.9})
        assert "code_coverage" in artifacts
        assert artifacts["code_coverage"] == 0.9
        assert len(logs) > 0

    @pytest.mark.asyncio
    async def test_build_stage(self):
        """Build stage executes correctly."""
        handler = BuildStageHandler()
        artifacts, logs = await handler.execute({}, {})
        assert artifacts["build_success"] is True
        assert artifacts["dependencies_resolved"] is True

    @pytest.mark.asyncio
    async def test_test_stage(self):
        """Test stage calculates success rate."""
        handler = TestStageHandler()
        context = {"tests_passed": 100, "tests_total": 100}
        artifacts, logs = await handler.execute({}, context)
        assert artifacts["test_success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_security_stage_no_vulns(self):
        """Security stage passes with no vulnerabilities."""
        handler = SecurityStageHandler()
        artifacts, logs = await handler.execute({}, {"critical_vulns": 0})
        assert artifacts["ihsan_safety"] == 1.0

    @pytest.mark.asyncio
    async def test_security_stage_with_vulns(self):
        """Security stage reduces safety score with vulns."""
        handler = SecurityStageHandler()
        artifacts, logs = await handler.execute({}, {"critical_vulns": 2})
        assert artifacts["ihsan_safety"] < 1.0

    @pytest.mark.asyncio
    async def test_quality_stage_aggregates(self):
        """Quality stage aggregates Ihsān scores."""
        handler = QualityStageHandler()
        input_artifacts = {
            "ihsan_correctness": 0.95,
            "ihsan_safety": 0.95,
        }
        artifacts, logs = await handler.execute(input_artifacts, {})
        assert "ihsan_overall" in artifacts


class TestElitePipeline:
    """Tests for elite CI/CD pipeline."""

    def test_pipeline_creation(self):
        """Pipeline can be created with defaults."""
        pipeline = ElitePipeline()
        assert len(pipeline._handlers) > 0
        assert len(pipeline._stage_order) == 5

    @pytest.mark.asyncio
    async def test_pipeline_full_run_success(self):
        """Full pipeline run with passing context."""
        pipeline = ElitePipeline()

        # Must include ALL Ihsān dimensions for gates to pass
        context = {
            "code_coverage": 0.95,
            "tests_passed": 100,
            "tests_total": 100,
            "critical_vulns": 0,
            "high_vulns": 0,
            "medium_vulns": 0,
            "code_quality_score": 0.95,
            # Ihsān dimensions
            "ihsan_correctness": 0.96,
            "ihsan_safety": 0.96,
            "ihsan_user_benefit": 0.96,
            "ihsan_efficiency": 0.96,
            "ihsan_auditability": 0.96,
            "ihsan_anti_centralization": 0.96,
            "ihsan_robustness": 0.96,
            "ihsan_adl_justice": 0.96,
            # SNR thresholds
            "snr_threshold": 0.95,
            "lint_score": 0.95,
            "security_score": 0.95,
            "documentation_score": 0.95,
            "edge_case_coverage": 0.95,
            "maintainability_index": 0.95,
        }

        run = await pipeline.run(context)
        assert run.status == PipelineStatus.PASSED
        assert len(run.stages) == 5

    @pytest.mark.asyncio
    async def test_pipeline_fails_on_security(self):
        """Pipeline fails when security stage fails."""
        pipeline = ElitePipeline()

        # Provide all Ihsān dimensions so source/build/test pass,
        # but security fails due to critical vulnerabilities
        context = {
            "code_coverage": 0.95,
            "tests_passed": 100,
            "tests_total": 100,
            "critical_vulns": 5,  # Critical vulns = failure in security
            "high_vulns": 0,
            "medium_vulns": 0,
            "code_quality_score": 0.95,
            # Ihsān dimensions
            "ihsan_correctness": 0.96,
            "ihsan_safety": 0.96,
            "ihsan_user_benefit": 0.96,
            "ihsan_efficiency": 0.96,
            "ihsan_auditability": 0.96,
            "ihsan_anti_centralization": 0.96,
            "ihsan_robustness": 0.96,
            "ihsan_adl_justice": 0.96,
            # SNR thresholds
            "snr_threshold": 0.95,
            "lint_score": 0.95,
            "security_score": 0.95,
            "documentation_score": 0.95,
            "edge_case_coverage": 0.95,
            "maintainability_index": 0.95,
        }

        run = await pipeline.run(context)
        assert run.status == PipelineStatus.FAILED
        # Should stop at security stage
        assert "security" in run.stages

    def test_pipeline_run_history(self):
        """Pipeline maintains run history."""
        pipeline = ElitePipeline()
        # Check initially empty
        assert len(pipeline._runs) == 0

    def test_pipeline_stats(self):
        """Pipeline provides statistics."""
        pipeline = ElitePipeline()
        stats = pipeline.get_stats()
        assert "total_runs" in stats
        assert "success_rate" in stats
        assert "avg_ihsan_score" in stats


# =============================================================================
# RISK MANAGEMENT TESTS
# =============================================================================


class TestRisk:
    """Tests for risk entries."""

    def test_risk_creation(self):
        """Risk can be created with defaults."""
        risk = Risk(name="Test Risk", description="A test")
        assert risk.name == "Test Risk"
        assert risk.probability == 0.5
        assert risk.impact == 0.5

    def test_risk_score_calculation(self):
        """Risk score = probability × impact."""
        risk = Risk(probability=0.4, impact=0.5)
        assert risk.risk_score == 0.2

    def test_risk_priority_score(self):
        """Priority score includes severity weighting."""
        risk = Risk(
            probability=0.5,
            impact=0.5,
            severity=RiskSeverity.HIGH,
        )
        # HIGH weight = 3.0, so priority = 0.25 * 3.0 = 0.75
        assert risk.priority_score == 0.75

    def test_constitutional_priority_highest(self):
        """Constitutional risks have highest priority."""
        risk = Risk(
            probability=0.1,
            impact=0.1,
            severity=RiskSeverity.CONSTITUTIONAL,
        )
        # Even with low probability/impact, constitutional = weight 10
        assert risk.priority_score >= 0.1


class TestRiskManager:
    """Tests for risk manager."""

    def test_manager_creation(self):
        """Manager creates with standard risks."""
        manager = RiskManager()
        assert len(manager._risks) >= 6  # 6 standard risks

    def test_standard_risks_registered(self):
        """Standard BIZRA risks are pre-registered."""
        manager = RiskManager()
        assert "SEC-001" in manager._risks
        assert "SEC-002" in manager._risks
        assert "PERF-001" in manager._risks

    def test_cascade_relationships_defined(self):
        """Cascade relationships are set up."""
        manager = RiskManager()
        sec001 = manager.get_risk("SEC-001")
        assert len(sec001.cascades_to) > 0

    def test_add_custom_risk(self):
        """Custom risks can be added."""
        manager = RiskManager()
        custom = Risk(
            id="CUSTOM-001",
            name="Custom Risk",
            description="Test custom risk",
        )
        manager.add_risk(custom)
        assert "CUSTOM-001" in manager._risks

    def test_cascade_analysis(self):
        """Cascade analysis finds affected risks."""
        manager = RiskManager()
        analysis = manager.analyze_cascade("SEC-001")
        assert isinstance(analysis, CascadeAnalysis)
        assert len(analysis.affected_risks) > 0

    def test_prioritized_risks(self):
        """Risks can be retrieved by priority."""
        manager = RiskManager()
        prioritized = manager.get_prioritized_risks()
        assert len(prioritized) > 0
        # First should have highest priority
        assert prioritized[0].priority_score >= prioritized[-1].priority_score

    def test_constitutional_risks(self):
        """Can filter constitutional risks."""
        manager = RiskManager()
        constitutional = manager.get_constitutional_risks()
        assert len(constitutional) >= 2  # SEC-001, SEC-002

    def test_recommend_mitigation_constitutional(self):
        """Constitutional risks recommend AVOID."""
        manager = RiskManager()
        strategy = manager.recommend_mitigation("SEC-002")
        assert strategy == MitigationStrategy.AVOID

    def test_risk_matrix(self):
        """Risk matrix categorizes properly."""
        manager = RiskManager()
        matrix = manager.get_risk_matrix()
        assert "constitutional" in matrix
        assert len(matrix["constitutional"]) >= 2

    def test_risk_summary(self):
        """Summary provides useful stats."""
        manager = RiskManager()
        summary = manager.get_summary()
        assert "total_risks" in summary
        assert "by_category" in summary
        assert "constitutional_risks" in summary


# =============================================================================
# SAPE OPTIMIZER TESTS
# =============================================================================


class TestGraphOfThoughts:
    """Tests for Graph-of-Thoughts structure."""

    def test_graph_creation(self):
        """Graph can be created."""
        graph = GraphOfThoughts()
        assert len(graph._nodes) == 0
        assert graph._backtrack_count == 0

    def test_add_node(self):
        """Nodes can be added to graph."""
        graph = GraphOfThoughts()
        node = graph.add_node(
            content="Initial observation",
            node_type=ThoughtNodeType.OBSERVATION,
            layer=SAPELayer.DATA,
        )
        assert node.id in graph._nodes
        assert node.id in graph._root_ids

    def test_add_child_node(self):
        """Child nodes link to parents."""
        graph = GraphOfThoughts()
        parent = graph.add_node(
            content="Parent thought",
            node_type=ThoughtNodeType.OBSERVATION,
            layer=SAPELayer.DATA,
        )
        child = graph.add_node(
            content="Child thought",
            node_type=ThoughtNodeType.ANALYSIS,
            layer=SAPELayer.INFORMATION,
            parent_ids={parent.id},
        )
        assert child.id not in graph._root_ids
        assert child.id in parent.children

    def test_backtrack_creates_node(self):
        """Backtracking creates a backtrack node."""
        graph = GraphOfThoughts()
        node = graph.add_node(
            content="Low quality thought",
            node_type=ThoughtNodeType.HYPOTHESIS,
            layer=SAPELayer.DATA,
            snr_score=0.5,
        )
        backtrack = graph.backtrack(node.id, "SNR too low")
        assert backtrack is not None
        assert backtrack.node_type == ThoughtNodeType.BACKTRACK
        assert graph._backtrack_count == 1

    def test_synthesize_nodes(self):
        """Multiple nodes can be synthesized."""
        graph = GraphOfThoughts()

        nodes = []
        for i in range(3):
            node = graph.add_node(
                content=f"Observation {i}",
                node_type=ThoughtNodeType.OBSERVATION,
                layer=SAPELayer.DATA,
                snr_score=0.9,
            )
            nodes.append(node)

        synthesis = graph.synthesize(
            {n.id for n in nodes},
            "Synthesized understanding",
            SAPELayer.INFORMATION,
        )

        assert synthesis is not None
        assert synthesis.node_type == ThoughtNodeType.SYNTHESIS
        assert synthesis.layer == SAPELayer.INFORMATION

    def test_get_best_path(self):
        """Best path can be found through graph."""
        graph = GraphOfThoughts()

        root = graph.add_node(
            content="Root",
            node_type=ThoughtNodeType.OBSERVATION,
            layer=SAPELayer.DATA,
            snr_score=0.9,
            confidence=0.9,
        )
        child = graph.add_node(
            content="Child",
            node_type=ThoughtNodeType.CONCLUSION,
            layer=SAPELayer.KNOWLEDGE,
            parent_ids={root.id},
            snr_score=0.95,
            confidence=0.95,
        )

        best_path = graph.get_best_path()
        assert len(best_path) == 2
        assert best_path[-1].id == child.id

    def test_get_layer_nodes(self):
        """Can filter nodes by layer."""
        graph = GraphOfThoughts()

        graph.add_node("Data 1", ThoughtNodeType.OBSERVATION, SAPELayer.DATA)
        graph.add_node("Data 2", ThoughtNodeType.OBSERVATION, SAPELayer.DATA)
        graph.add_node("Info 1", ThoughtNodeType.ANALYSIS, SAPELayer.INFORMATION)

        data_nodes = graph.get_layer_nodes(SAPELayer.DATA)
        assert len(data_nodes) == 2

    def test_graph_stats(self):
        """Graph provides useful statistics."""
        graph = GraphOfThoughts()
        graph.add_node("Test", ThoughtNodeType.OBSERVATION, SAPELayer.DATA)

        stats = graph.get_stats()
        assert stats["total_nodes"] == 1
        assert "by_layer" in stats
        assert "avg_snr" in stats


class TestSAPEOptimizer:
    """Tests for SAPE optimizer."""

    def test_optimizer_creation(self):
        """Optimizer can be created."""
        optimizer = SAPEOptimizer()
        assert optimizer.ihsan_threshold >= 0.9

    def test_compute_snr_empty(self):
        """Empty text returns 0 SNR."""
        optimizer = SAPEOptimizer()
        snr = optimizer._compute_snr("")
        assert snr == 0.0

    def test_compute_snr_quality_text(self):
        """Quality text gets reasonable SNR."""
        optimizer = SAPEOptimizer()
        text = (
            "The BIZRA ecosystem implements constitutional AI principles "
            "with formal verification. Signal-to-noise optimization ensures "
            "high-quality outputs. Each layer adds abstraction value."
        )
        snr = optimizer._compute_snr(text)
        assert 0.3 <= snr <= 1.0

    def test_compute_ihsan_quality_text(self):
        """Quality text gets reasonable Ihsān score."""
        optimizer = SAPEOptimizer()
        text = (
            "This is a well-structured text that demonstrates clarity and purpose. "
            "It provides balanced information with proper context and reasoning."
        )
        ihsan = optimizer._compute_ihsan(text)
        assert 0.3 <= ihsan <= 1.0

    @pytest.mark.asyncio
    async def test_optimize_basic(self):
        """Basic optimization runs through layers."""
        optimizer = SAPEOptimizer()

        content = (
            "The BIZRA system uses constitutional AI for ethical constraints. "
            "SNR optimization ensures high-quality signal processing. "
            "Graph-of-Thoughts enables non-linear reasoning paths."
        )

        result = await optimizer.optimize(content)

        assert isinstance(result, SAPEResult)
        assert len(result.layers_traversed) > 0
        assert result.thought_graph_size > 0

    @pytest.mark.asyncio
    async def test_optimize_to_knowledge_layer(self):
        """Can optimize to specific layer."""
        optimizer = SAPEOptimizer()

        content = "Data input for knowledge processing with quality validation."

        result = await optimizer.optimize(content, target_layer=SAPELayer.KNOWLEDGE)

        assert SAPELayer.KNOWLEDGE in result.layers_traversed

    @pytest.mark.asyncio
    async def test_analyze_patterns(self):
        """Pattern analysis extracts insights."""
        optimizer = SAPEOptimizer()

        content = (
            "Complex content requiring multi-step reasoning and analysis. "
            "The patterns here suggest underlying structure that needs extraction. "
            "Quality signals emerge through progressive refinement."
        )

        patterns = await optimizer.analyze_unconventional_patterns(content)
        assert isinstance(patterns, list)

    def test_layer_info(self):
        """Layer info provides configuration."""
        optimizer = SAPEOptimizer()
        info = optimizer.get_layer_info()

        assert "layers" in info
        assert "snr_targets" in info
        assert "progression" in info


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestEliteIntegration:
    """Integration tests for elite framework."""

    @pytest.mark.asyncio
    async def test_pipeline_with_quality_gates(self):
        """Pipeline uses quality gates properly."""
        pipeline = ElitePipeline()

        context = {
            "code_coverage": 0.95,
            "tests_passed": 100,
            "tests_total": 100,
            "critical_vulns": 0,
            "code_quality_score": 0.95,
        }

        run = await pipeline.run(context)

        # Check gate results exist
        for stage_name, stage_result in run.stages.items():
            assert stage_result.gate_result is not None

    @pytest.mark.asyncio
    async def test_sape_with_got_reasoning(self):
        """SAPE uses Graph-of-Thoughts for reasoning."""
        optimizer = SAPEOptimizer()

        content = (
            "This content needs multi-hop reasoning to understand. "
            "First, we observe the data. Then, we analyze patterns. "
            "Finally, we synthesize knowledge from observations."
        )

        result = await optimizer.optimize(content)

        # Should use GoT
        assert result.thought_graph_size > 0
        assert "data" in result.snr_progression

    def test_risk_cascade_to_pipeline(self):
        """Risk cascades can affect pipeline decisions."""
        manager = RiskManager()

        # Analyze cascade from key compromise
        cascade = manager.analyze_cascade("SEC-001")

        # Should affect multiple risks
        assert cascade.total_impact > 0.5
        assert len(cascade.affected_risks) >= 2

    @pytest.mark.asyncio
    async def test_full_elite_workflow(self):
        """Full workflow: SAPE → Pipeline → Risk Check."""
        # 1. Optimize with SAPE
        optimizer = SAPEOptimizer()
        sape_result = await optimizer.optimize(
            "Code submission for review with constitutional validation."
        )

        # 2. Run through pipeline
        pipeline = ElitePipeline()
        context = {
            "code_coverage": 0.9,
            "tests_passed": 95,
            "tests_total": 100,
            "critical_vulns": 0,
            "snr_score": sape_result.snr_progression.get("data", 0.85),
        }
        run = await pipeline.run(context)

        # 3. Check risks
        manager = RiskManager()
        if run.status == PipelineStatus.FAILED:
            # Get top risks
            top_risks = manager.get_prioritized_risks()[:3]
            assert len(top_risks) > 0
        else:
            summary = manager.get_summary()
            assert summary["total_risks"] > 0


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestSerialization:
    """Tests for data serialization."""

    def test_thought_node_to_dict(self):
        """ThoughtNode serializes correctly."""
        node = ThoughtNode(
            content="Test content",
            node_type=ThoughtNodeType.OBSERVATION,
            layer=SAPELayer.DATA,
            snr_score=0.9,
        )
        data = node.to_dict()
        assert data["node_type"] == "observation"
        assert data["layer"] == "data"
        assert data["snr_score"] == 0.9

    def test_sape_result_to_dict(self):
        """SAPEResult serializes correctly."""
        result = SAPEResult(
            input_content="Input",
            output_content="Output",
            layers_traversed=[SAPELayer.DATA, SAPELayer.INFORMATION],
            snr_progression={"data": 0.9, "information": 0.95},
            ihsan_score=0.92,
            thought_graph_size=5,
            backtrack_count=1,
            duration_ms=100.0,
        )
        data = result.to_dict()
        assert data["layers_traversed"] == ["data", "information"]
        assert data["thought_graph_size"] == 5

    def test_gate_result_to_dict(self):
        """GateResult serializes correctly."""
        result = GateResult(
            gate_name="test_gate",
            status=GateStatus.PASSED,
            overall_score=0.9,
            ihsan_score=0.95,
            snr_score=0.92,
            criteria_results={"test": (True, 0.9)},
        )
        data = result.to_dict()
        assert data["status"] == "passed"
        assert "criteria_results" in data

    def test_risk_to_dict(self):
        """Risk serializes correctly."""
        risk = Risk(
            id="TEST-001",
            name="Test Risk",
            description="A test risk",
            category=RiskCategory.TECHNICAL,
            severity=RiskSeverity.MEDIUM,
        )
        data = risk.to_dict()
        assert data["id"] == "TEST-001"
        assert data["category"] == "technical"

    def test_pipeline_run_to_dict(self):
        """PipelineRun serializes correctly."""
        run = PipelineRun(
            status=PipelineStatus.PASSED,
            ihsan_score=0.95,
            snr_score=0.92,
        )
        data = run.to_dict()
        assert data["status"] == "passed"
        assert data["ihsan_score"] == 0.95
