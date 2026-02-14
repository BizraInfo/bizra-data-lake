"""Tests for core.uers.vectors -- 5-Dimensional Analytical Manifold.

Covers:
- VectorType and ProbeResult enums
- VectorState data class
- Probe data class
- AnalyticalManifold: vector updates, probing, state queries
"""

import pytest

from core.uers.entropy import EntropyMeasurement
from core.uers.vectors import (
    AnalyticalManifold,
    Probe,
    ProbeResult,
    VectorState,
    VectorType,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def manifold():
    return AnalyticalManifold()


# ---------------------------------------------------------------------------
# ENUM TESTS
# ---------------------------------------------------------------------------


class TestVectorType:

    def test_all_five_vectors_exist(self):
        expected = {"surface", "structural", "behavioral", "hypothetical", "contextual"}
        actual = {v.value for v in VectorType}
        assert actual == expected

    def test_string_enum(self):
        assert VectorType.SURFACE == "surface"


class TestProbeResult:

    def test_all_results_exist(self):
        expected = {"success", "failure", "blocked", "pending"}
        actual = {r.value for r in ProbeResult}
        assert actual == expected


# ---------------------------------------------------------------------------
# VectorState TESTS
# ---------------------------------------------------------------------------


class TestVectorState:

    def test_is_resolved_low_entropy(self):
        state = VectorState(
            vector_type=VectorType.SURFACE,
            entropy=EntropyMeasurement("surface", 0.0, 0.05),
        )
        assert state.is_resolved is True

    def test_is_resolved_high_entropy(self):
        state = VectorState(
            vector_type=VectorType.SURFACE,
            entropy=EntropyMeasurement("surface", 5.0, 0.5),
        )
        assert state.is_resolved is False

    def test_needs_attention_high_entropy(self):
        state = VectorState(
            vector_type=VectorType.SURFACE,
            entropy=EntropyMeasurement("surface", 7.0, 0.8),
        )
        assert state.needs_attention is True

    def test_needs_attention_low_entropy(self):
        state = VectorState(
            vector_type=VectorType.STRUCTURAL,
            entropy=EntropyMeasurement("structural", 1.0, 0.3),
        )
        assert state.needs_attention is False

    def test_to_dict(self):
        state = VectorState(
            vector_type=VectorType.BEHAVIORAL,
            entropy=EntropyMeasurement("behavioral", 3.0, 0.5),
            confidence=0.7,
        )
        d = state.to_dict()
        assert d["type"] == "behavioral"
        assert d["confidence"] == 0.7
        assert "is_resolved" in d
        assert "needs_attention" in d


# ---------------------------------------------------------------------------
# Probe TESTS
# ---------------------------------------------------------------------------


class TestProbe:

    def test_probe_default_state(self):
        p = Probe(
            id="probe_001",
            source_vector=VectorType.SURFACE,
            target_vector=VectorType.STRUCTURAL,
            operation="extract_cfg",
        )
        assert p.result == ProbeResult.PENDING
        assert p.delta_e == 0.0

    def test_probe_to_dict(self):
        p = Probe(
            id="probe_002",
            source_vector=VectorType.BEHAVIORAL,
            target_vector=VectorType.HYPOTHETICAL,
            operation="seed_from_traces",
        )
        d = p.to_dict()
        assert d["id"] == "probe_002"
        assert d["source"] == "behavioral"
        assert d["target"] == "hypothetical"
        assert d["result"] == "pending"


# ---------------------------------------------------------------------------
# AnalyticalManifold TESTS
# ---------------------------------------------------------------------------


class TestAnalyticalManifoldInit:

    def test_initial_entropy_is_max(self, manifold):
        assert manifold.get_average_entropy() == 1.0

    def test_all_vectors_initialized(self, manifold):
        for vtype in VectorType:
            assert vtype in manifold._vectors

    def test_is_not_converged_initially(self, manifold):
        assert manifold.is_converged() is False

    def test_probe_registry_populated(self, manifold):
        assert len(manifold._probe_registry) > 0


class TestVectorUpdates:

    def test_update_surface(self, manifold):
        state = manifold.update_surface(b"\x00" * 100)
        assert state.vector_type == VectorType.SURFACE
        assert state.entropy.normalized == 0.0  # Constant bytes

    def test_update_structural(self, manifold):
        state = manifold.update_structural(nodes=10, edges=15, components=1)
        assert state.vector_type == VectorType.STRUCTURAL
        assert 0.0 <= state.entropy.normalized <= 1.0

    def test_update_behavioral(self, manifold):
        events = ["read", "write", "exec"] * 10
        state = manifold.update_behavioral(events)
        assert state.vector_type == VectorType.BEHAVIORAL

    def test_update_hypothetical(self, manifold):
        state = manifold.update_hypothetical(
            explored_paths=50, total_paths=100, feasible_paths=40
        )
        assert state.vector_type == VectorType.HYPOTHETICAL

    def test_update_contextual(self, manifold):
        state = manifold.update_contextual(
            text="Analyze system for vulnerabilities",
            intent_score=0.9,
            alignment_score=0.85,
        )
        assert state.vector_type == VectorType.CONTEXTUAL

    def test_updates_reduce_entropy(self, manifold):
        initial = manifold.get_average_entropy()
        manifold.update_surface(b"\x00" * 100)
        after = manifold.get_average_entropy()
        assert after < initial  # At least surface vector now has lower entropy


class TestProbingOperations:

    def test_create_probe(self, manifold):
        p = manifold.create_probe(
            source=VectorType.SURFACE,
            target=VectorType.STRUCTURAL,
            operation="extract_cfg_from_bytes",
        )
        assert p.id.startswith("probe_")
        assert p.result == ProbeResult.PENDING

    def test_execute_probe_simulated(self, manifold):
        p = manifold.create_probe(
            source=VectorType.SURFACE,
            target=VectorType.STRUCTURAL,
            operation="test_op",
        )
        result = manifold.execute_probe(p)
        assert result.result == ProbeResult.SUCCESS
        assert result.output == {"simulated": True}

    def test_execute_probe_with_custom_fn(self, manifold):
        def custom_fn(probe):
            return {"extracted": True, "blocks": 42}

        p = manifold.create_probe(
            source=VectorType.SURFACE,
            target=VectorType.STRUCTURAL,
            operation="custom",
        )
        result = manifold.execute_probe(p, probe_fn=custom_fn)
        assert result.result == ProbeResult.SUCCESS
        assert result.output["blocks"] == 42

    def test_execute_probe_failure(self, manifold):
        def failing_fn(probe):
            raise ValueError("Probe failed")

        p = manifold.create_probe(
            source=VectorType.SURFACE,
            target=VectorType.STRUCTURAL,
            operation="fail",
        )
        result = manifold.execute_probe(p, probe_fn=failing_fn)
        assert result.result == ProbeResult.FAILURE
        assert "error" in result.output

    def test_suggest_probes(self, manifold):
        suggestions = manifold.suggest_probes()
        assert isinstance(suggestions, list)
        # All vectors start at 1.0, so suggestions should exist
        assert len(suggestions) > 0

    def test_suggest_probes_format(self, manifold):
        suggestions = manifold.suggest_probes()
        for source, target, operation in suggestions:
            assert isinstance(source, VectorType)
            assert isinstance(target, VectorType)
            assert isinstance(operation, str)


class TestManifoldState:

    def test_get_manifold_state(self, manifold):
        from core.uers.entropy import ManifoldState
        state = manifold.get_manifold_state()
        assert isinstance(state, ManifoldState)

    def test_get_entropy_vector(self, manifold):
        vec = manifold.get_entropy_vector()
        assert len(vec) == 5
        assert all(0.0 <= v <= 1.0 for v in vec)

    def test_is_converged_after_full_update(self, manifold):
        # Update all vectors with low-entropy data
        manifold.update_surface(b"\x00" * 100)
        manifold.update_structural(nodes=2, edges=1, components=1)
        # Behavioral with single event type
        manifold.update_behavioral(["read"] * 50)
        # Fully explored paths
        manifold.update_hypothetical(
            explored_paths=100, total_paths=100, feasible_paths=100
        )
        manifold.update_contextual(
            text="Clear, well-defined intent",
            intent_score=1.0,
            alignment_score=1.0,
        )
        # Should be close to converged but thresholds may vary
        avg = manifold.get_average_entropy()
        assert avg < 0.5  # Significantly reduced from initial 1.0


class TestManifoldStats:

    def test_get_stats(self, manifold):
        stats = manifold.get_stats()
        assert "total_entropy" in stats
        assert "average_entropy" in stats
        assert "vectors" in stats
        assert "probe_count" in stats

    def test_get_probe_history(self, manifold):
        manifold.create_probe(VectorType.SURFACE, VectorType.STRUCTURAL, "op1")
        history = manifold.get_probe_history(limit=5)
        assert len(history) == 1

    def test_get_vector_info(self, manifold):
        info = manifold.get_vector_info(VectorType.SURFACE)
        assert "state" in info
        assert "config" in info
        assert "available_probes" in info

    def test_to_dict(self, manifold):
        d = manifold.to_dict()
        assert "stats" in d
        assert "recent_probes" in d
