"""Node State 7-Tuple -- Amended Definition 1.1.

Expands the cognitive 5-tuple (H, C, I, L, Sigma) to a full 7-tuple
by adding ontological identity (epsilon_i) and physical body (B_i(t)).

The HHMM transition function is amended:
    P(s_t | o_{1:t}, theta) -> P(s_t | o_{1:t}, B_i(t), theta)

Body conditioning makes classification and routing ONE decision:
    IF CanExecuteLocal(mission, B_i(t)): execute locally (SOVEREIGN tier)
    ELSE: submit PoolRequest for ComputeAllocation

Standing on Giants: Fine (HHMM, 1998) | Shannon (state entropy, 1948) | Besta (GoT state, 2024)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class NodeState:
    """The complete 7-tuple node state N_i(t) = (epsilon_i, B_i(t), H, C, I, L, Sigma).

    Three layers:
      Ontological: identity (fixed after genesis, except wallet balances)
      Physical:    body (dynamic hardware/model state)
      Cognitive:   hhmm_state, reflex_cache_size, ihsan_score, maturity_level, stress_tensor

    Identity and body use structural typing -- any object with the expected
    attributes works. This avoids circular imports with core.identity.
    """

    # -- Layer 0: Ontological (epsilon_i) --------------------------------
    identity: Any  # IdentityGenesis -- Definition 1.6 (structural typing)

    # -- Layer 0.5: Physical (B_i(t)) -----------------------------------
    body: Any  # NodeBody -- Definition 1.7 (structural typing)

    # -- Layer 1-3: Cognitive (H, C, I, L, Sigma) -----------------------
    hhmm_state: str = "idle"
    reflex_cache_size: int = 0
    ihsan_score: float = 0.0
    maturity_level: int = 0
    stress_tensor: float = 0.0

    def cognitive_state(self) -> tuple[str, int, float, int, float]:
        """Extract the original 5-tuple cognitive sub-state (H, C, I, L, Sigma).

        Provides backward compatibility with the pre-amendment Definition 1.1.
        """
        return (
            self.hhmm_state,
            self.reflex_cache_size,
            self.ihsan_score,
            self.maturity_level,
            self.stress_tensor,
        )

    def can_execute_locally(self, mission: dict[str, Any]) -> bool:
        """Route decision: execute locally or delegate to Pool.

        This is the HHMM body-conditioning function:
            P(s_t | o_{1:t}, B_i(t), theta)

        The HHMM classifies the mission AND determines routing
        in a single decision. Delegates to body.can_execute_local().
        """
        return self.body.can_execute_local(mission)

    def transition(
        self,
        observation: Any,
        mission_result: Optional[dict[str, Any]] = None,
    ) -> NodeState:
        """Apply state transition after an observation/mission.

        Returns a NEW NodeState (immutable transition pattern).
        Identity is always preserved. Body updates only on model changes.
        Maturity is monotonically non-decreasing.
        """
        new_cache = self._update_cache(mission_result)
        new_ihsan = self._update_ihsan(mission_result)
        new_maturity = self._compute_maturity(new_cache, new_ihsan)

        return NodeState(
            identity=self.identity,
            body=self._update_body(mission_result),
            hhmm_state=self._hhmm_transition(observation),
            reflex_cache_size=new_cache,
            ihsan_score=new_ihsan,
            maturity_level=new_maturity,
            stress_tensor=self._update_stress(observation),
        )

    # -- Private transition helpers --------------------------------------

    def _update_body(self, result: Optional[dict[str, Any]]) -> Any:
        """Body state changes only when models/KG change.

        Uses structural typing: body must have loaded_models, cpu_cores,
        gpu_vram_mb, gpu_compute_cap, ram_bytes, disk_bytes, knowledge_graph_size.
        """
        if result is None:
            return self.body

        new_models = result.get("loaded_models")
        if not new_models:
            return self.body

        # Create updated body with new models merged in.
        # Uses duck typing -- body must support attribute access.
        updated = _BodySnapshot(
            cpu_cores=self.body.cpu_cores,
            gpu_vram_mb=self.body.gpu_vram_mb,
            gpu_compute_cap=getattr(self.body, "gpu_compute_cap", 0.0),
            ram_bytes=getattr(self.body, "ram_bytes", 0),
            disk_bytes=getattr(self.body, "disk_bytes", 0),
            loaded_models=self.body.loaded_models | set(new_models),
            knowledge_graph_size=getattr(self.body, "knowledge_graph_size", 0),
        )
        # Preserve can_execute_local if the original body has custom logic
        if hasattr(self.body, "can_execute_local"):
            updated.can_execute_local = self.body.can_execute_local  # type: ignore[attr-defined]
        return updated

    def _hhmm_transition(self, observation: Any) -> str:
        """HHMM macro-state transition.

        Placeholder -- real implementation lives in core.sovereign.hhmm.
        Returns current state unchanged (identity transition).
        """
        return self.hhmm_state

    def _update_cache(self, result: Optional[dict[str, Any]]) -> int:
        """Reflex cache: grows on successful high-Ihsan missions.

        Cache precipitates a new reflex only when the mission's Ihsan >= 0.90.
        """
        if result and result.get("ihsan_score", 0) >= 0.90:
            return self.reflex_cache_size + 1
        return self.reflex_cache_size

    def _update_ihsan(self, result: Optional[dict[str, Any]]) -> float:
        """Ihsan: exponential moving average of mission scores.

        EMA formula: I_{t+1} = alpha * new_score + (1 - alpha) * I_t
        Alpha = 0.1 provides smooth convergence.
        """
        if result is None:
            return self.ihsan_score
        alpha = 0.1
        new_score = result.get("ihsan_score", self.ihsan_score)
        return alpha * new_score + (1 - alpha) * self.ihsan_score

    def _compute_maturity(self, new_cache: int, new_ihsan: float) -> int:
        """Maturity: monotonically non-decreasing, based on cache and Ihsan.

        Tiers:
          0 = SEED   (default)
          1 = SPROUT  (cache >= 10)
          2 = TREE    (cache >= 100 AND ihsan >= 0.90)
          3 = FOREST  (cache >= 1000 AND ihsan >= 0.95)
        """
        candidate = 0
        if new_cache >= 1000 and new_ihsan >= 0.95:
            candidate = 3  # FOREST
        elif new_cache >= 100 and new_ihsan >= 0.90:
            candidate = 2  # TREE
        elif new_cache >= 10:
            candidate = 1  # SPROUT

        # Monotonic: never decrease
        return max(self.maturity_level, candidate)

    def _update_stress(self, observation: Any) -> float:
        """Stress tensor update placeholder.

        Real implementation in Step 6 (core/sovereign/stress_tensor.py).
        Returns current stress unchanged.
        """
        return self.stress_tensor


@dataclass
class _BodySnapshot:
    """Internal snapshot of body state for immutable transitions.

    Not part of the public API -- used only when _update_body creates
    a new body with updated loaded_models.
    """

    cpu_cores: int = 0
    gpu_vram_mb: int = 0
    gpu_compute_cap: float = 0.0
    ram_bytes: int = 0
    disk_bytes: int = 0
    loaded_models: set = None  # type: ignore[assignment]
    knowledge_graph_size: int = 0

    def __post_init__(self) -> None:
        if self.loaded_models is None:
            self.loaded_models = set()

    def can_execute_local(self, mission: dict[str, Any]) -> bool:
        """Default body capability check.

        Verifies required models are loaded and GPU VRAM is sufficient.
        """
        required_models = mission.get("required_models", set())
        if required_models and not required_models.issubset(self.loaded_models):
            return False
        min_vram = mission.get("min_gpu_vram_mb", 0)
        if min_vram > self.gpu_vram_mb:
            return False
        return True
