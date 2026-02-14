"""Tests for core.uers.entropy -- Multi-Dimensional Entropy Measurement.

Covers:
- EntropyMeasurement data class properties
- ManifoldState computed properties
- EntropyCalculator: shannon, structural, trace, path, contextual entropy
- Delta-E calculation and convergence tracking
"""

import math

import pytest

from core.uers.entropy import (
    EntropyCalculator,
    EntropyMeasurement,
    ManifoldState,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def calculator():
    return EntropyCalculator()


@pytest.fixture
def low_entropy_measurement():
    return EntropyMeasurement(vector="surface", value=1.0, normalized=0.1)


@pytest.fixture
def high_entropy_measurement():
    return EntropyMeasurement(vector="surface", value=7.0, normalized=0.9)


@pytest.fixture
def sample_manifold():
    return ManifoldState(
        surface=EntropyMeasurement("surface", 4.0, 0.5),
        structural=EntropyMeasurement("structural", 3.0, 0.4),
        behavioral=EntropyMeasurement("behavioral", 2.0, 0.3),
        hypothetical=EntropyMeasurement("hypothetical", 1.0, 0.2),
        contextual=EntropyMeasurement("contextual", 0.5, 0.1),
    )


# ---------------------------------------------------------------------------
# EntropyMeasurement TESTS
# ---------------------------------------------------------------------------


class TestEntropyMeasurement:

    def test_instantiation(self):
        m = EntropyMeasurement(vector="surface", value=5.0, normalized=0.625)
        assert m.vector == "surface"
        assert m.value == 5.0
        assert m.normalized == 0.625

    def test_is_high_entropy_true(self, high_entropy_measurement):
        assert high_entropy_measurement.is_high_entropy is True

    def test_is_high_entropy_false(self, low_entropy_measurement):
        assert low_entropy_measurement.is_high_entropy is False

    def test_is_low_entropy_true(self, low_entropy_measurement):
        assert low_entropy_measurement.is_low_entropy is True

    def test_is_low_entropy_false(self, high_entropy_measurement):
        assert high_entropy_measurement.is_low_entropy is False

    @pytest.mark.parametrize("normalized,expected_high,expected_low", [
        (0.0, False, True),
        (0.24, False, True),
        (0.25, False, False),
        (0.75, False, False),
        (0.76, True, False),
        (1.0, True, False),
    ])
    def test_entropy_boundaries(self, normalized, expected_high, expected_low):
        m = EntropyMeasurement("test", 0.0, normalized)
        assert m.is_high_entropy == expected_high
        assert m.is_low_entropy == expected_low

    def test_to_dict(self):
        m = EntropyMeasurement("surface", 3.0, 0.375)
        d = m.to_dict()
        assert d["vector"] == "surface"
        assert d["value"] == 3.0
        assert d["normalized"] == 0.375
        assert "timestamp" in d
        assert isinstance(d["is_high"], bool)
        assert isinstance(d["is_low"], bool)


# ---------------------------------------------------------------------------
# ManifoldState TESTS
# ---------------------------------------------------------------------------


class TestManifoldState:

    def test_total_entropy(self, sample_manifold):
        expected = 0.5 + 0.4 + 0.3 + 0.2 + 0.1
        assert abs(sample_manifold.total_entropy - expected) < 1e-9

    def test_average_entropy(self, sample_manifold):
        expected = (0.5 + 0.4 + 0.3 + 0.2 + 0.1) / 5.0
        assert abs(sample_manifold.average_entropy - expected) < 1e-9

    def test_entropy_vector(self, sample_manifold):
        vec = sample_manifold.entropy_vector
        assert len(vec) == 5
        assert vec == [0.5, 0.4, 0.3, 0.2, 0.1]

    def test_to_dict_has_all_vectors(self, sample_manifold):
        d = sample_manifold.to_dict()
        for key in ("surface", "structural", "behavioral", "hypothetical", "contextual"):
            assert key in d
        assert "total_entropy" in d
        assert "average_entropy" in d

    def test_zero_entropy_manifold(self):
        ms = ManifoldState(
            surface=EntropyMeasurement("surface", 0.0, 0.0),
            structural=EntropyMeasurement("structural", 0.0, 0.0),
            behavioral=EntropyMeasurement("behavioral", 0.0, 0.0),
            hypothetical=EntropyMeasurement("hypothetical", 0.0, 0.0),
            contextual=EntropyMeasurement("contextual", 0.0, 0.0),
        )
        assert ms.total_entropy == 0.0
        assert ms.average_entropy == 0.0


# ---------------------------------------------------------------------------
# EntropyCalculator TESTS
# ---------------------------------------------------------------------------


class TestShannonEntropy:

    def test_empty_data(self, calculator):
        m = calculator.shannon_entropy(b"")
        assert m.value == 0.0
        assert m.normalized == 0.0

    def test_uniform_data(self, calculator):
        m = calculator.shannon_entropy(bytes(range(256)) * 100)
        assert m.value > 7.9  # Near maximum for uniform distribution
        assert m.normalized > 0.99

    def test_constant_data(self, calculator):
        m = calculator.shannon_entropy(b"\x00" * 1000)
        assert m.value == 0.0
        assert m.normalized == 0.0

    def test_two_symbol_data(self, calculator):
        m = calculator.shannon_entropy(b"\x00\x01" * 500)
        assert abs(m.value - 1.0) < 0.01  # Should be ~1.0 bits

    def test_metadata_fields(self, calculator):
        data = b"Hello, world!"
        m = calculator.shannon_entropy(data)
        assert "byte_count" in m.metadata
        assert "unique_bytes" in m.metadata
        assert "is_encrypted" in m.metadata
        assert m.metadata["byte_count"] == len(data)

    def test_normalized_range(self, calculator):
        m = calculator.shannon_entropy(b"test data for entropy check")
        assert 0.0 <= m.normalized <= 1.0


class TestTextEntropy:

    def test_empty_text(self, calculator):
        m = calculator.text_entropy("")
        assert m.value == 0.0

    def test_single_char(self, calculator):
        m = calculator.text_entropy("aaaa")
        assert m.value == 0.0

    def test_high_diversity_text(self, calculator):
        text = "The quick brown fox jumps over the lazy dog"
        m = calculator.text_entropy(text)
        assert m.normalized > 0.5

    def test_normalized_capped_at_one(self, calculator):
        text = "".join(chr(i) for i in range(32, 127))
        m = calculator.text_entropy(text)
        assert m.normalized <= 1.0


class TestStructuralEntropy:

    def test_single_node(self, calculator):
        m = calculator.structural_entropy(nodes=1, edges=0)
        assert m.value == 0.0
        assert m.normalized == 0.0

    def test_simple_graph(self, calculator):
        m = calculator.structural_entropy(nodes=10, edges=15, components=1)
        assert 0.0 <= m.normalized <= 1.0
        assert "edge_density" in m.metadata

    def test_fragmented_graph(self, calculator):
        m_connected = calculator.structural_entropy(nodes=10, edges=15, components=1)
        m_fragmented = calculator.structural_entropy(nodes=10, edges=15, components=5)
        # More fragments should increase entropy
        assert m_fragmented.normalized > m_connected.normalized

    def test_cfg_entropy_single_block(self, calculator):
        m = calculator.cfg_entropy(basic_blocks=1, branch_edges=0, call_edges=0)
        assert m.normalized == 0.0

    def test_cfg_entropy_complex(self, calculator):
        m = calculator.cfg_entropy(basic_blocks=50, branch_edges=30, call_edges=20, loop_count=5)
        assert 0.0 < m.normalized <= 1.0
        assert "cyclomatic_complexity" in m.metadata


class TestTraceEntropy:

    def test_empty_events(self, calculator):
        m = calculator.trace_entropy([])
        assert m.value == 0.0

    def test_single_event_type(self, calculator):
        m = calculator.trace_entropy(["read"] * 100)
        assert m.value == 0.0

    def test_diverse_events(self, calculator):
        events = ["read", "write", "exec", "alloc", "free"] * 20
        m = calculator.trace_entropy(events)
        assert m.normalized > 0.8

    def test_api_sequence_entropy_empty(self, calculator):
        m = calculator.api_sequence_entropy([])
        assert m.value == 0.0

    def test_api_sequence_entropy_single_call(self, calculator):
        m = calculator.api_sequence_entropy(["open"])
        assert m.value == 0.0  # Falls back to trace_entropy with single element

    def test_api_sequence_entropy_varied(self, calculator):
        calls = ["open", "read", "write", "close", "open", "read", "seek", "close"]
        m = calculator.api_sequence_entropy(calls)
        assert m.normalized > 0.0


class TestPathEntropy:

    def test_zero_total_paths(self, calculator):
        m = calculator.path_entropy(explored_paths=0, total_paths=0, feasible_paths=0)
        assert m.normalized == 0.0

    def test_all_explored(self, calculator):
        m = calculator.path_entropy(explored_paths=100, total_paths=100, feasible_paths=80)
        assert m.normalized < 0.1  # Should be very low

    def test_none_explored(self, calculator):
        m = calculator.path_entropy(explored_paths=0, total_paths=100, feasible_paths=0)
        assert m.normalized > 0.5  # High remaining entropy

    def test_constraint_entropy_underconstrained(self, calculator):
        m = calculator.constraint_entropy(constraint_count=2, variable_count=10)
        assert m.normalized > 0.5  # Under-constrained -> high entropy

    def test_constraint_entropy_overconstrained(self, calculator):
        m = calculator.constraint_entropy(constraint_count=20, variable_count=10, satisfiable=True)
        assert m.normalized < 0.5

    def test_constraint_entropy_zero_variables(self, calculator):
        m = calculator.constraint_entropy(constraint_count=5, variable_count=0)
        assert m.normalized == 0.0


class TestContextualEntropy:

    def test_empty_text(self, calculator):
        m = calculator.contextual_entropy("   ")
        assert m.normalized == 1.0  # Maximum uncertainty

    def test_high_intent_alignment(self, calculator):
        m = calculator.contextual_entropy(
            "Analyze the system architecture for scalability concerns.",
            intent_score=0.95,
            alignment_score=0.95,
        )
        assert m.normalized < 0.3

    def test_low_intent_alignment(self, calculator):
        m = calculator.contextual_entropy(
            "Random stuff here.",
            intent_score=0.1,
            alignment_score=0.1,
        )
        assert m.normalized > 0.5

    def test_snr_to_entropy_high_snr(self, calculator):
        m = calculator.snr_to_entropy(0.95)
        assert abs(m.normalized - 0.05) < 1e-9

    def test_snr_to_entropy_low_snr(self, calculator):
        m = calculator.snr_to_entropy(0.1)
        assert abs(m.normalized - 0.9) < 1e-9


class TestManifoldOperations:

    def test_measure_manifold_defaults(self, calculator):
        state = calculator.measure_manifold()
        assert isinstance(state, ManifoldState)
        # All default to 0.5
        assert state.average_entropy == 0.5

    def test_measure_manifold_with_surface(self, calculator):
        state = calculator.measure_manifold(surface_data=b"\x00" * 100)
        assert state.surface.normalized == 0.0  # Constant data

    def test_calculate_delta_e(self, calculator):
        before = ManifoldState(
            surface=EntropyMeasurement("surface", 0, 0.8),
            structural=EntropyMeasurement("structural", 0, 0.8),
            behavioral=EntropyMeasurement("behavioral", 0, 0.8),
            hypothetical=EntropyMeasurement("hypothetical", 0, 0.8),
            contextual=EntropyMeasurement("contextual", 0, 0.8),
        )
        after = ManifoldState(
            surface=EntropyMeasurement("surface", 0, 0.3),
            structural=EntropyMeasurement("structural", 0, 0.3),
            behavioral=EntropyMeasurement("behavioral", 0, 0.3),
            hypothetical=EntropyMeasurement("hypothetical", 0, 0.3),
            contextual=EntropyMeasurement("contextual", 0, 0.3),
        )
        delta = calculator.calculate_delta_e(before, after)
        assert delta > 0  # Entropy reduced = positive delta

    def test_convergence_progress_insufficient_data(self, calculator):
        result = calculator.get_convergence_progress()
        assert result["progress"] == 0.0
        assert result["converging"] is False

    def test_measurement_history(self, calculator):
        calculator.measure_manifold()
        calculator.measure_manifold()
        history = calculator.get_measurement_history(limit=5)
        assert len(history) == 2
