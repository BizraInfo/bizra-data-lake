"""TDD anchors for Node State 7-Tuple (Amended Definition 1.1).

9 tests covering all three layers:
  Ontological: identity immutability
  Physical: body updates on model load, can_execute_locally delegation
  Cognitive: 5-tuple extraction, Ihsan EMA, cache growth, maturity monotonicity
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.sovereign.node_state import NodeState


# -- Mock objects using structural typing (no core.identity import) ----------


@dataclass
class MockIdentity:
    """Structural stand-in for IdentityGenesis (Definition 1.6)."""

    node_id: str = "node-0001"
    public_key: str = "pk_mock_ed25519"
    sovereignty_class: int = 0


@dataclass
class MockBody:
    """Structural stand-in for NodeBody (Definition 1.7).

    Must satisfy the duck-typing contract:
      .can_execute_local(mission) -> bool
      .loaded_models -> set
      .cpu_cores, .gpu_vram_mb, etc.
    """

    cpu_cores: int = 8
    gpu_vram_mb: int = 16000
    gpu_compute_cap: float = 8.9
    ram_bytes: int = 64 * 1024**3
    disk_bytes: int = 1024 * 1024**3
    loaded_models: set = field(default_factory=set)
    knowledge_graph_size: int = 0

    def can_execute_local(self, mission: dict) -> bool:
        """Check if this body can handle the mission locally."""
        required_models = mission.get("required_models", set())
        if required_models and not required_models.issubset(self.loaded_models):
            return False
        min_vram = mission.get("min_gpu_vram_mb", 0)
        if min_vram > self.gpu_vram_mb:
            return False
        return True


def _make_default_state(**overrides) -> NodeState:
    """Factory for test NodeState with sensible defaults."""
    defaults = {
        "identity": MockIdentity(),
        "body": MockBody(),
        "hhmm_state": "idle",
        "reflex_cache_size": 0,
        "ihsan_score": 0.0,
        "maturity_level": 0,
        "stress_tensor": 0.0,
    }
    defaults.update(overrides)
    return NodeState(**defaults)


# -- TDD Anchor 1: 7-component structure ------------------------------------


class TestNodeStateStructure:
    """Verify the 7-tuple structure and layer separation."""

    def test_node_state_has_7_components(self) -> None:
        """State is a 7-tuple: identity, body, H, C, I, L, Sigma."""
        state = _make_default_state()
        assert hasattr(state, "identity")
        assert hasattr(state, "body")
        assert hasattr(state, "hhmm_state")
        assert hasattr(state, "reflex_cache_size")
        assert hasattr(state, "ihsan_score")
        assert hasattr(state, "maturity_level")
        assert hasattr(state, "stress_tensor")

    def test_cognitive_state_is_original_5tuple(self) -> None:
        """cognitive_state() returns (H, C, I, L, Sigma)."""
        state = _make_default_state(
            hhmm_state="browsing",
            reflex_cache_size=42,
            ihsan_score=0.87,
            maturity_level=1,
            stress_tensor=2.5,
        )
        cog = state.cognitive_state()
        assert len(cog) == 5
        assert cog == ("browsing", 42, 0.87, 1, 2.5)


# -- TDD Anchor 2: Ontological layer immutability ---------------------------


class TestOntologicalLayer:
    """Identity is fixed across transitions (Definition 1.6)."""

    def test_identity_unchanged_after_transition(self) -> None:
        """Ontological layer is fixed (except wallet)."""
        state = _make_default_state()
        original_identity = state.identity
        new_state = state.transition(observation="email_received")
        assert new_state.identity is original_identity


# -- TDD Anchor 3: Physical layer -------------------------------------------


class TestPhysicalLayer:
    """Body updates only on model/KG changes (Definition 1.7)."""

    def test_body_updates_on_model_load(self) -> None:
        """Body changes when mission loads a new model."""
        state = _make_default_state()
        result = {"loaded_models": ["llama3"], "ihsan_score": 0.95}
        new_state = state.transition(
            observation="model_loaded", mission_result=result
        )
        assert "llama3" in new_state.body.loaded_models

    def test_body_unchanged_without_model_load(self) -> None:
        """Body does not change when no models are loaded."""
        state = _make_default_state()
        result = {"ihsan_score": 0.95}
        new_state = state.transition(observation="task_done", mission_result=result)
        assert new_state.body.loaded_models == state.body.loaded_models


# -- TDD Anchor 4: Routing delegation ---------------------------------------


class TestRoutingDecision:
    """can_execute_locally delegates to body.can_execute_local()."""

    def test_can_execute_locally_delegates_to_body(self) -> None:
        """Route decision uses body.can_execute_local()."""
        state = _make_default_state()
        state.body.loaded_models = {"phi3:mini"}
        state.body.gpu_vram_mb = 24000
        assert state.can_execute_locally(
            {"required_models": {"phi3:mini"}, "min_gpu_vram_mb": 8000}
        )

    def test_cannot_execute_locally_without_model(self) -> None:
        """Node without required model delegates to Pool."""
        state = _make_default_state()
        state.body.loaded_models = set()
        assert not state.can_execute_locally(
            {"required_models": {"llama3.1-70b"}}
        )


# -- TDD Anchor 5: Cognitive layer ------------------------------------------


class TestCognitiveLayer:
    """Ihsan EMA, cache growth, and maturity monotonicity."""

    def test_ihsan_ema_smoothing(self) -> None:
        """Ihsan updates via exponential moving average (alpha=0.1)."""
        state = _make_default_state(ihsan_score=0.90)
        new_state = state.transition(
            observation="mission_complete",
            mission_result={"ihsan_score": 1.0},
        )
        # EMA: 0.1 * 1.0 + 0.9 * 0.90 = 0.91
        assert abs(new_state.ihsan_score - 0.91) < 1e-6

    def test_reflex_cache_grows_on_high_ihsan(self) -> None:
        """Cache grows only when mission Ihsan >= 0.90."""
        state = _make_default_state(reflex_cache_size=5)

        good = state.transition("obs", {"ihsan_score": 0.95})
        assert good.reflex_cache_size == 6

        bad = state.transition("obs", {"ihsan_score": 0.50})
        assert bad.reflex_cache_size == 5  # no growth

    def test_maturity_is_monotonic(self) -> None:
        """Maturity level never decreases."""
        state = _make_default_state(
            maturity_level=2,
            reflex_cache_size=50,
            ihsan_score=0.80,
        )
        # Even with low cache/Ihsan, maturity cannot drop below 2
        new_state = state.transition(
            observation="test",
            mission_result={"ihsan_score": 0.5},
        )
        assert new_state.maturity_level >= 2
