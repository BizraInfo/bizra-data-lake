# Step 5: Node State 7-Tuple Amendment

## Standing on Giants: Fine (HHMM, 1998) | Shannon (state entropy, 1948) | Besta (GoT state, 2024)

**Date:** 2026-03-03
**Ω⁷ Gem:** Ω⁷-4 (Node state missing identity and body)
**Intent:** Expand Definition 1.1 from 5-tuple to 7-tuple, resolve HHMM routing question

---

## Problem Statement

Definition 1.1 defines node state as (H, C, I, L, Σ) — HHMM state, reflex cache,
Ihsan tensor, maturity level, stress tensor. This captures COGNITIVE state but
misses ONTOLOGICAL state (who am I?) and PHYSICAL state (what can I do?).

The Identity Genesis (Definition 1.6) and Node Body (Definition 1.7) add two
components. The HHMM transition function P(s_t | o_{1:t}, θ) should condition
on Body state because hardware determines which inference paths are possible.

**The question this resolves:** When the HHMM classifies a mission as SOVEREIGN
(requiring full 7-agent pipeline with LLM synthesis), should it check whether
the local node can handle it, or delegate to the Pool?

**Answer:** The HHMM conditions on B_i(t). Classification and routing are ONE
decision, not two.

---

## Mathematical Formalization

### Amended Definition 1.1 (Node State 7-Tuple)

```
Original:
  N_i(t) = (H_t, C_t, I_t, L_t, Σ_t)

Amended:
  N_i(t) = (ε_i, B_i(t), H_t, C_t, I_t, L_t, Σ_t)

Where:
  ε_i     — Identity state (Definition 1.6)     [STATIC, ontological]
  B_i(t)  — Body state (Definition 1.7)         [DYNAMIC, physical]
  H_t     — HHMM hidden state                   [DYNAMIC, cognitive]
  C_t     — Reflex cache state                   [DYNAMIC, cognitive]
  I_t     — Ihsan tensor ∈ [0,1]^d              [DYNAMIC, cognitive]
  L_t     — Maturity level ∈ {SEED..FOREST}     [MONOTONIC]
  Σ_t     — Stress tensor ∈ ℝ⁺                  [DYNAMIC, cognitive]

Three layers:
  Ontological: ε_i           — fixed after genesis (except wallet balances)
  Physical:    B_i(t)        — changes as resources are loaded/unloaded
  Cognitive:   (H,C,I,L,Σ)  — the original 5-tuple, now a sub-state

The HHMM transition function is amended:

  P(s_t | o_{1:t}, θ) → P(s_t | o_{1:t}, B_i(t), θ)

  Body conditioning determines routing:
    IF CanExecuteLocal(mission, B_i(t)):
      Execute locally (SOVEREIGN tier)
    ELSE:
      Submit PoolRequest for ComputeAllocation

  This makes classification and routing one decision:
    The HHMM's macro-state output includes both
    WHAT to do and WHERE to do it.
```

### State Transition Rules

```
Ontological layer (ε_i):
  ε_i is fixed after genesis EXCEPT:
    - Wallet balances W_i change on SEED/BLOOM transfers
    - Sovereignty class S_i can only increase (P4 from Def 1.6)

Physical layer (B_i(t)):
  B_i(t) → B_i(t+1) when:
    - A model is loaded/unloaded (Models_i changes)
    - Knowledge graph is updated (KG_i changes)
    - Disk space changes (Disk_i changes)
    - CPU/GPU/RAM are static (hardware doesn't change)

Cognitive layer (H, C, I, L, Σ):
  H_t → H_{t+1}: HHMM transition on new observation
  C_t → C_{t+1}: Reflex precipitation or eviction
  I_t → I_{t+1}: Ihsan score update after mission
  L_t → L_{t+1}: Maturity level promotion (monotonic)
  Σ_t → Σ_{t+1}: Stress tensor update (see Step 6)
```

---

## Pseudocode

### core/sovereign/node_state.py

```pseudocode
"""Node State 7-Tuple — Amended Definition 1.1.

Standing on Giants: Fine (HHMM) | Shannon (state)
"""

FROM __future__ IMPORT annotations
FROM dataclasses IMPORT dataclass, field
FROM typing IMPORT Optional, Any


@dataclass
CLASS NodeState:
    """The complete 7-tuple node state.

    Three layers:
      Ontological: identity (fixed after genesis)
      Physical:    body (dynamic hardware/model state)
      Cognitive:   hhmm_state, reflex_cache, ihsan, maturity, stress
    """

    # ── Layer 0: Ontological ──────────────────────────────────────
    identity: "IdentityGenesis"           # ε_i — Definition 1.6

    # ── Layer 0.5: Physical ───────────────────────────────────────
    body: "NodeBody"                      # B_i(t) — Definition 1.7

    # ── Layer 1-3: Cognitive ──────────────────────────────────────
    hhmm_state: str = "idle"              # H_t — macro-state label
    reflex_cache_size: int = 0            # C_t — number of cached reflexes
    ihsan_score: float = 0.0              # I_t — current Ihsan tensor scalar
    maturity_level: int = 0               # L_t — sovereignty tier (0-3)
    stress_tensor: float = 0.0            # Σ_t — epistemic tension

    FUNCTION cognitive_state(self) -> tuple:
        """Extract the original 5-tuple cognitive sub-state."""
        RETURN (
            self.hhmm_state,
            self.reflex_cache_size,
            self.ihsan_score,
            self.maturity_level,
            self.stress_tensor,
        )

    FUNCTION can_execute_locally(self, mission: dict) -> bool:
        """Route decision: execute locally or delegate to Pool.

        This is the HHMM body-conditioning function:
        P(s_t | o_{1:t}, B_i(t), θ)

        The HHMM classifies the mission AND determines routing
        in a single decision.
        """
        RETURN self.body.can_execute_local(mission)

    FUNCTION transition(
        self,
        observation: Any,
        mission_result: Optional[dict] = None,
    ) -> "NodeState":
        """Apply state transition after an observation/mission.

        Returns a NEW NodeState (immutable transition).
        """
        new_state = NodeState(
            identity=self.identity,         # ontological: unchanged
            body=self._update_body(mission_result),
            hhmm_state=self._hhmm_transition(observation),
            reflex_cache_size=self._update_cache(mission_result),
            ihsan_score=self._update_ihsan(mission_result),
            maturity_level=self._update_maturity(),
            stress_tensor=self._update_stress(observation),
        )
        RETURN new_state

    FUNCTION _update_body(self, result: Optional[dict]) -> "NodeBody":
        """Body state changes only when models/KG change."""
        IF result IS None:
            RETURN self.body
        # If mission loaded a new model, update body
        new_models = result.get("loaded_models")
        IF new_models:
            RETURN NodeBody(
                cpu_cores=self.body.cpu_cores,
                gpu_vram_mb=self.body.gpu_vram_mb,
                gpu_compute_cap=self.body.gpu_compute_cap,
                ram_bytes=self.body.ram_bytes,
                disk_bytes=self.body.disk_bytes,
                loaded_models=self.body.loaded_models | set(new_models),
                knowledge_graph_size=self.body.knowledge_graph_size,
            )
        RETURN self.body

    FUNCTION _hhmm_transition(self, observation: Any) -> str:
        """HHMM macro-state transition.
        In production: calls the actual HHMM classifier.
        """
        # Placeholder — real implementation in core.sovereign.hhmm
        RETURN self.hhmm_state

    FUNCTION _update_cache(self, result: Optional[dict]) -> int:
        """Reflex cache: grows on successful high-Ihsan missions."""
        IF result AND result.get("ihsan_score", 0) >= 0.90:
            RETURN self.reflex_cache_size + 1
        RETURN self.reflex_cache_size

    FUNCTION _update_ihsan(self, result: Optional[dict]) -> float:
        """Ihsan: exponential moving average of mission scores."""
        IF result IS None:
            RETURN self.ihsan_score
        alpha = 0.1  # smoothing factor
        new_score = result.get("ihsan_score", self.ihsan_score)
        RETURN alpha * new_score + (1 - alpha) * self.ihsan_score

    FUNCTION _update_maturity(self) -> int:
        """Maturity: monotonic, based on reflex cache and Ihsan."""
        # Simplified: advance tier when cache hits thresholds
        IF self.reflex_cache_size >= 1000 AND self.ihsan_score >= 0.95:
            RETURN max(self.maturity_level, 3)  # FOREST
        ELIF self.reflex_cache_size >= 100 AND self.ihsan_score >= 0.90:
            RETURN max(self.maturity_level, 2)  # TREE
        ELIF self.reflex_cache_size >= 10:
            RETURN max(self.maturity_level, 1)  # SPROUT
        RETURN self.maturity_level  # monotonic: never decrease

    FUNCTION _update_stress(self, observation: Any) -> float:
        """Stress: increases with unresolved tasks, decreases with resolution."""
        # Placeholder — real implementation in Step 6 (stress_action_bus)
        RETURN self.stress_tensor
```

---

## TDD Anchors

```pseudocode
# tests/core/sovereign/test_node_state_7tuple.py

TEST node_state_has_7_components:
    """State is a 7-tuple: identity, body, H, C, I, L, Σ."""
    state = _make_default_state()
    ASSERT hasattr(state, "identity")
    ASSERT hasattr(state, "body")
    ASSERT hasattr(state, "hhmm_state")
    ASSERT hasattr(state, "reflex_cache_size")
    ASSERT hasattr(state, "ihsan_score")
    ASSERT hasattr(state, "maturity_level")
    ASSERT hasattr(state, "stress_tensor")

TEST cognitive_state_is_original_5tuple:
    """cognitive_state() returns (H, C, I, L, Σ)."""
    state = _make_default_state()
    cog = state.cognitive_state()
    ASSERT len(cog) == 5

TEST identity_unchanged_after_transition:
    """Ontological layer is fixed (except wallet)."""
    state = _make_default_state()
    new_state = state.transition(observation="email_received")
    ASSERT new_state.identity == state.identity

TEST body_updates_on_model_load:
    """Body changes when mission loads a new model."""
    state = _make_default_state()
    result = {"loaded_models": ["llama3"], "ihsan_score": 0.95}
    new_state = state.transition(observation="model_loaded", mission_result=result)
    ASSERT "llama3" IN new_state.body.loaded_models

TEST maturity_is_monotonic:
    """Maturity level never decreases."""
    state = _make_default_state()
    state.maturity_level = 2  # TREE
    # Even with low Ihsan, maturity cannot decrease
    new_state = state.transition(
        observation="test",
        mission_result={"ihsan_score": 0.5}
    )
    ASSERT new_state.maturity_level >= 2

TEST can_execute_locally_delegates_to_body:
    """Route decision uses body.can_execute_local()."""
    state = _make_default_state()
    state.body.loaded_models = {"phi3:mini"}
    state.body.gpu_vram_mb = 24000
    ASSERT state.can_execute_locally({
        "required_models": {"phi3:mini"},
        "min_gpu_vram_mb": 8000,
    })

TEST cannot_execute_locally_without_model:
    """Node without required model delegates to Pool."""
    state = _make_default_state()
    state.body.loaded_models = set()
    ASSERT NOT state.can_execute_locally({
        "required_models": {"llama3.1-70b"},
    })

TEST ihsan_ema_smoothing:
    """Ihsan updates via exponential moving average."""
    state = _make_default_state()
    state.ihsan_score = 0.90
    new_state = state.transition(
        observation="mission_complete",
        mission_result={"ihsan_score": 1.0},
    )
    # EMA: 0.1 * 1.0 + 0.9 * 0.90 = 0.91
    ASSERT abs(new_state.ihsan_score - 0.91) < 1e-6

TEST reflex_cache_grows_on_high_ihsan:
    """Cache grows only when mission Ihsan >= 0.90."""
    state = _make_default_state()
    state.reflex_cache_size = 5
    good = state.transition("obs", {"ihsan_score": 0.95})
    ASSERT good.reflex_cache_size == 6
    bad = state.transition("obs", {"ihsan_score": 0.50})
    ASSERT bad.reflex_cache_size == 5  # no growth
```

---

## Acceptance Criteria

1. `NodeState` is a 7-component dataclass with clear layer separation
2. `cognitive_state()` returns the original 5-tuple for backward compatibility
3. Identity is immutable across transitions
4. Body updates only on model/KG changes
5. Maturity is monotonically non-decreasing
6. `can_execute_locally()` delegates to body capability predicate
7. All 9 TDD anchors GREEN
8. Full test suite GREEN
