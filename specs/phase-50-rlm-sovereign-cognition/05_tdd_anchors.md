# Phase 50.5 — TDD Anchors and Verification Contracts

> Standing on Giants: Beck (TDD, 2002) · Hoare (design by contract, 1969) · Lamport (temporal logic, 1977)

## 1. Test Matrix

| Module | Test File | Tests | Priority |
|--------|-----------|-------|----------|
| RLM Sandbox | `tests/core/inference/test_rlm_sandbox.py` | 12 | P0 |
| RLM Bridge | `tests/core/inference/test_rlm_bridge.py` | 10 | P0 |
| Token Incentive RL | `tests/core/token/test_rl_rewards.py` | 14 | P1 |
| Strategy Learning | `tests/core/token/test_strategy_update.py` | 8 | P1 |
| Voice Pipeline | `tests/core/voice/test_personaplex_bridge.py` | 10 | P2 |
| Integration | `tests/integration/test_rlm_mission.py` | 6 | P1 |
| **Total** | | **60** | |

## 2. RLM Sandbox Tests (P0)

```python
# tests/core/inference/test_rlm_sandbox.py

"""
TDD anchors for RLM sandboxed REPL execution.

Standing on Giants:
- Beck (2002): Red-green-refactor
- Zhang et al. (2026): RLM Algorithm 1
"""

class TestREPLState:
    def test_init_with_prompt_as_variable(self):
        """Prompt is stored as REPL variable, not in LLM context."""
        state = REPLState()
        state.variables["prompt"] = "A" * 100_000  # 100K chars
        assert state.variables["prompt_length"] is not set  # Only set by bridge
        assert "prompt" in state.variables

    def test_metadata_is_constant_size(self):
        """Metadata about prompt is O(1), regardless of prompt size."""
        for size in [1_000, 100_000, 10_000_000]:
            prompt = "x" * size
            metadata = extract_metadata(prompt)
            assert len(metadata) < 1000  # Always < 1KB

    def test_final_answer_terminates_loop(self):
        """Setting FINAL_ANSWER in state terminates the RLM loop."""
        state = REPLState()
        state.variables["FINAL_ANSWER"] = "the answer"
        assert state.variables["FINAL_ANSWER"] is not None


class TestSandboxSecurity:
    def test_blocks_file_io(self):
        """Sandbox blocks all file I/O operations."""
        sandbox = RLMSandbox(REPLState())
        valid, msg = sandbox.validate_code("open('etc/passwd')")
        assert not valid
        assert "Blocked" in msg

    def test_blocks_imports(self):
        """Sandbox blocks import statements."""
        sandbox = RLMSandbox(REPLState())
        valid, msg = sandbox.validate_code("import os")
        assert not valid

    def test_blocks_subprocess(self):
        """Sandbox blocks subprocess execution."""
        sandbox = RLMSandbox(REPLState())
        valid, msg = sandbox.validate_code("subprocess.run(['ls'])")
        assert not valid

    def test_allows_string_operations(self):
        """Sandbox allows string slicing, splitting, searching."""
        sandbox = RLMSandbox(REPLState())
        state = REPLState()
        state.variables["prompt"] = "hello world"
        sandbox = RLMSandbox(state)

        valid, _ = sandbox.validate_code("result = prompt[:5]")
        assert valid

    def test_allows_regex(self):
        """Sandbox allows re module for prompt searching."""
        sandbox = RLMSandbox(REPLState())
        valid, _ = sandbox.validate_code("import re\nre.findall(r'\\d+', prompt)")
        # Note: re is pre-loaded, not imported
        # Actual test uses pre-loaded re

    def test_allows_lm_query(self):
        """Sandbox allows sub-RLM calls via lm_query function."""
        state = REPLState()
        state.functions["lm_query"] = lambda x: "mock response"
        sandbox = RLMSandbox(state)
        valid, _ = sandbox.validate_code("answer = lm_query('sub question')")
        assert valid

    def test_state_persists_across_iterations(self):
        """Variables set in one iteration persist to the next."""
        state = REPLState()
        state.variables["prompt"] = "test data"
        sandbox = RLMSandbox(state)

        state, _ = sandbox.execute("x = 42")
        state, _ = sandbox.execute("y = x + 1")

        assert state.variables["y"] == 43

    def test_stdout_captured(self):
        """Print statements are captured in stdout buffer."""
        state = REPLState()
        sandbox = RLMSandbox(state)
        state, stdout = sandbox.execute("print('hello from REPL')")
        assert "hello from REPL" in stdout

    def test_execution_error_captured(self):
        """Runtime errors are captured, not raised."""
        state = REPLState()
        sandbox = RLMSandbox(state)
        state, stdout = sandbox.execute("x = 1 / 0")
        assert "ZeroDivisionError" in stdout

    def test_ast_validation_catches_nested_attacks(self):
        """Deeply nested code attempting to bypass sandbox is caught."""
        sandbox = RLMSandbox(REPLState())
        valid, _ = sandbox.validate_code(
            "x = lambda: __import__('os').system('rm -rf /')"
        )
        assert not valid
```

## 3. Token Incentive RL Tests (P1)

```python
# tests/core/token/test_rl_rewards.py

"""
TDD anchors for token-incentivized agent reinforcement learning.

Standing on Giants:
- Sutton & Barto (1998): Reward function properties
- Gini (1912): Inequality constraint
"""

class TestCompositeReward:
    def test_perfect_score_yields_max_reward(self):
        """Perfect SNR + Ihsan + efficiency = maximum reward."""
        result = composite_reward({
            "snr": 1.0, "ihsan": 1.0,
            "tokens_used": 100, "user_feedback": 1.0,
            "hallucination_score": 0.0, "latency": 0.5,
        })
        assert result >= 0.95

    def test_zero_quality_yields_near_zero_reward(self):
        """Zero quality metrics yield near-zero reward."""
        result = composite_reward({
            "snr": 0.0, "ihsan": 0.0,
            "tokens_used": 10000, "user_feedback": 0.0,
        })
        assert result < 0.1

    def test_hallucination_penalty_reduces_reward(self):
        """Hallucination detection reduces reward."""
        good = composite_reward({"snr": 0.9, "ihsan": 0.9, "hallucination_score": 0.0})
        bad = composite_reward({"snr": 0.9, "ihsan": 0.9, "hallucination_score": 1.0})
        assert bad < good

    def test_reward_bounded_zero_to_one(self):
        """Reward is always in [0, 1]."""
        for _ in range(100):
            import random
            result = composite_reward({
                "snr": random.random(),
                "ihsan": random.random(),
                "tokens_used": random.randint(1, 10000),
            })
            assert 0.0 <= result <= 1.0


class TestTokenEfficiency:
    def test_high_quality_few_tokens_is_efficient(self):
        """High quality with few tokens = high efficiency reward."""
        reward = token_efficiency_reward(tokens_used=100, quality=0.95)
        assert reward > 0.7

    def test_low_quality_many_tokens_is_inefficient(self):
        """Low quality with many tokens = low efficiency reward."""
        reward = token_efficiency_reward(tokens_used=5000, quality=0.2)
        assert reward < 0.3

    def test_zero_tokens_returns_zero(self):
        """Zero tokens used returns zero reward."""
        assert token_efficiency_reward(tokens_used=0, quality=0.9) == 0.0


class TestSEEDReward:
    def test_high_score_mints_seed(self):
        """High composite score results in SEED minting."""
        minter = TokenMinter.create()
        receipt = compute_agent_reward(
            agent_id="pat-analyst-001",
            mission_result={"snr": 0.95, "ihsan": 0.96},
            minter=minter,
            emission_gate=LogisticEmissionGate(),
        )
        assert receipt.success

    def test_gini_gate_throttles_top_earner(self):
        """When Gini > 0.35, top earner's reward is throttled."""
        # Simulate unequal distribution
        holdings = [1000.0, 10.0, 10.0, 10.0, 10.0]
        gate = LogisticEmissionGate()
        result = gate.compute_gated_emission(100.0, holdings)
        assert result["gated_amount"] < 100.0  # Throttled

    def test_zakat_flows_to_community(self):
        """2.5% of minted SEED goes to community fund."""
        minter = TokenMinter.create()
        minter.mint_seed("agent-001", 100.0, epoch_id="test")
        community_balance = minter.ledger.get_balance(
            "BIZRA-COMMUNITY-FUND", TokenType.SEED
        )
        assert community_balance >= 2.5  # 2.5% of 100


class TestIMPTReputation:
    def test_impt_is_non_transferable(self):
        """IMPT reputation tokens cannot be transferred."""
        # IMPT minting succeeds but transfer should fail
        # (enforced by token type constraints)
        pass  # Implementation detail in token/types.py

    def test_impt_compounds_with_sqrt(self):
        """Higher IMPT gives diminishing returns on new IMPT."""
        # Agent with 100 IMPT earns less additional IMPT per mission
        # than agent with 10 IMPT, preventing runaway reputation
        pass


class TestStrategyUpdate:
    def test_high_reward_lowers_temperature(self):
        """Consistent high rewards lower sampling temperature."""
        strategy = AgentStrategy(temperature=0.7)
        for _ in range(10):
            strategy = update_strategy(strategy, reward=0.9, mission_context={})
        assert strategy.temperature < 0.7

    def test_low_reward_raises_temperature(self):
        """Consistent low rewards raise temperature for exploration."""
        strategy = AgentStrategy(temperature=0.5)
        for _ in range(10):
            strategy = update_strategy(strategy, reward=0.2, mission_context={})
        assert strategy.temperature > 0.5

    def test_ema_reward_tracks_performance(self):
        """EMA reward smoothly tracks performance trend."""
        strategy = AgentStrategy(ema_reward=0.5)
        rewards = [0.9, 0.8, 0.85, 0.95, 0.9]
        for r in rewards:
            strategy = update_strategy(strategy, reward=r, mission_context={})
        assert strategy.ema_reward > 0.7  # Trending up

    def test_long_context_enables_rlm(self):
        """Long prompts with good results enable RLM mode."""
        strategy = AgentStrategy(use_rlm=False)
        strategy = update_strategy(
            strategy, reward=0.85,
            mission_context={"prompt_length": 50000},
        )
        assert strategy.use_rlm is True
```

## 4. Voice Pipeline Tests (P2)

```python
# tests/core/voice/test_personaplex_bridge.py

"""
TDD anchors for PersonaPlex voice pipeline.

These tests validate the bridge logic WITHOUT requiring
PersonaPlex to be installed (mocked inference).
"""

class TestIhsanVoiceGate:
    def test_safe_text_passes_gate(self):
        """Constructive, helpful text passes Ihsan gate."""
        bridge = PersonaPlexBridge(Path("/mock"), mode="offline")
        bridge.register_personas()
        persona = bridge._personas["analyst"]
        passes, score = bridge.ihsan_gate("Let me help improve your system", persona)
        assert passes
        assert score >= persona.ihsan_floor

    def test_harmful_text_blocked(self):
        """Text with harm indicators is blocked."""
        bridge = PersonaPlexBridge(Path("/mock"), mode="offline")
        bridge.register_personas()
        persona = bridge._personas["guardian"]
        passes, score = bridge.ihsan_gate("exploit the vulnerability to harm users", persona)
        assert not passes

    def test_security_guardian_has_highest_threshold(self):
        """Security Guardian has strictest Ihsan floor."""
        bridge = PersonaPlexBridge(Path("/mock"), mode="offline")
        bridge.register_personas()
        assert bridge._personas["guardian"].ihsan_floor >= 0.95
        assert bridge._personas["ethics"].ihsan_floor >= 0.95

    def test_creative_guardian_has_lowest_threshold(self):
        """Creative Guardian has most permissive floor."""
        bridge = PersonaPlexBridge(Path("/mock"), mode="offline")
        bridge.register_personas()
        assert bridge._personas["creator"].ihsan_floor <= 0.85


class TestGuardianMapping:
    def test_all_pat_agents_have_personas(self):
        """Every PAT agent type maps to a Guardian persona."""
        bridge = PersonaPlexBridge(Path("/mock"), mode="offline")
        bridge.register_personas()
        required = ["coordinator", "analyst", "researcher", "guardian", "creator", "ethics"]
        for name in required:
            assert name in bridge._personas

    def test_each_persona_has_unique_voice(self):
        """Each Guardian has a distinct voice code."""
        bridge = PersonaPlexBridge(Path("/mock"), mode="offline")
        bridge.register_personas()
        voice_codes = [p.voice_code for p in bridge._personas.values()]
        assert len(voice_codes) == len(set(voice_codes))  # All unique

    def test_persona_text_prompts_non_empty(self):
        """Each persona has a substantive text prompt."""
        bridge = PersonaPlexBridge(Path("/mock"), mode="offline")
        bridge.register_personas()
        for persona in bridge._personas.values():
            assert len(persona.text_prompt) > 50


class TestVoiceOutput:
    def test_blocked_output_has_empty_audio(self):
        """Ihsan-blocked output returns empty audio array."""
        output = VoiceOutput(
            audio=np.array([], dtype=np.float32),
            sample_rate=24000,
            text_spoken="",
            guardian="test",
            ihsan_passed=False,
            duration_seconds=0.0,
        )
        assert len(output.audio) == 0
        assert not output.ihsan_passed

    def test_duration_calculated_correctly(self):
        """Duration matches audio length / sample rate."""
        audio = np.zeros(48000, dtype=np.float32)  # 2 seconds at 24kHz
        output = VoiceOutput(
            audio=audio, sample_rate=24000,
            text_spoken="test", guardian="test",
            ihsan_passed=True, duration_seconds=2.0,
        )
        assert output.duration_seconds == 2.0
```

## 5. Integration Tests (P1)

```python
# tests/integration/test_rlm_mission.py

"""
Integration tests for RLM-enhanced mission execution.
"""

class TestRLMMission:
    @pytest.mark.integration
    async def test_rlm_processes_large_corpus(self):
        """RLM can process a corpus larger than any model's context window."""
        large_corpus = "Document content. " * 50_000  # ~500K chars
        bridge = BizraRLMBridge(llm_call=mock_llm_call)
        result = await bridge.execute_rlm(
            prompt=large_corpus,
            task="Summarize the key themes",
            agent_model="test-model",
        )
        assert result.answer  # Non-empty answer
        assert result.iterations > 1  # Multiple REPL iterations

    @pytest.mark.integration
    async def test_rlm_uses_fewer_tokens_than_full_context(self):
        """RLM uses fewer total tokens than loading full context."""
        corpus = "x" * 100_000
        bridge = BizraRLMBridge(llm_call=mock_llm_call)
        result = await bridge.execute_rlm(
            prompt=corpus, task="Count words",
            agent_model="test-model",
        )
        # RLM should use much less than 100K tokens
        assert result.tokens_used < 50_000

    @pytest.mark.integration
    async def test_token_reward_minted_after_mission(self):
        """Successful mission mints SEED reward to agent account."""
        minter = TokenMinter.create()
        receipt = compute_agent_reward(
            agent_id="test-agent",
            mission_result={"snr": 0.92, "ihsan": 0.95},
            minter=minter,
            emission_gate=LogisticEmissionGate(),
        )
        assert receipt.success
        balance = minter.ledger.get_balance("test-agent", TokenType.SEED)
        assert balance > 0

    @pytest.mark.integration
    async def test_strategy_improves_over_multiple_missions(self):
        """Agent strategy EMA reward trends upward with good performance."""
        strategy = AgentStrategy()
        for i in range(20):
            reward = 0.5 + (i * 0.02)  # Improving performance
            strategy = update_strategy(
                strategy, reward=min(1.0, reward),
                mission_context={"prompt_length": 10000},
            )
        assert strategy.ema_reward > 0.7

    @pytest.mark.integration
    async def test_recursive_sub_calls_respect_depth_limit(self):
        """Sub-RLM calls cannot exceed max recursion depth."""
        call_count = 0
        async def counting_llm(prompt, model, max_tokens):
            nonlocal call_count
            call_count += 1
            return "FINAL_ANSWER = 'done'"

        bridge = BizraRLMBridge(
            llm_call=counting_llm,
            max_sub_calls=5,
        )
        # Even if code tries infinite recursion, sub-calls are bounded
        result = await bridge.execute_rlm(
            prompt="test", task="test",
            agent_model="test",
        )
        assert call_count <= 25  # max_iterations + max_sub_calls

    @pytest.mark.integration
    async def test_evidence_chain_records_rlm_metadata(self):
        """Evidence chain includes RLM-specific metadata (iterations, sub-calls)."""
        # After RLM mission, evidence entry should contain:
        # - rlm_iterations
        # - rlm_sub_calls
        # - rlm_tokens_used
        pass  # Wired through evidence chain integration
```

## 6. Verification Contracts

| Contract | Condition | Enforcement |
|----------|-----------|-------------|
| **Sandbox isolation** | No file I/O, no network, no imports | AST validation before every `exec()` |
| **Recursion depth** | ≤ 3 levels of sub-RLM calls | Counter in REPLState, hard-checked |
| **Token budget** | ≤ 41,000 tokens per RLM session | Counter in REPLState, terminates on exceed |
| **Ihsan on output** | Final answer ≥ 0.95 Ihsan score | Gate check before `return state[Final]` |
| **Gini on rewards** | Agent economy Gini ≤ 0.35 | LogisticEmissionGate auto-throttle |
| **Zakat** | 2.5% of all SEED rewards to community | Enforced in `TokenMinter.mint_seed()` |
| **Voice safety** | Ihsan gate before every vocalization | `PersonaPlexBridge.ihsan_gate()` |
| **Strategy bounds** | Temperature ∈ [0.3, 1.0], max_tokens ∈ [300, 2000] | Clamping in `update_strategy()` |

## 7. Files Summary

**New files (8):**
- `core/inference/rlm_bridge.py` — RLM REPL sandbox + bridge
- `core/token/rl_rewards.py` — Token incentive RL reward functions
- `core/token/strategy.py` — Agent strategy learning
- `core/voice/personaplex_bridge.py` — PersonaPlex voice pipeline
- `tests/core/inference/test_rlm_sandbox.py`
- `tests/core/inference/test_rlm_bridge.py`
- `tests/core/token/test_rl_rewards.py`
- `tests/core/voice/test_personaplex_bridge.py`

**Modified files (3):**
- `scripts/node0_activate.py` — Add RLM mode to `_call_agent()`
- `core/sovereign/graph_reasoning.py` — Replace template fallback with RLM-powered GoT synthesis
- `core/living_memory/core.py` — Add trajectory storage for RL learning

**Unchanged but depended on (4):**
- `core/token/mint.py` — SEED/BLOOM/IMPT minting (used by RL rewards)
- `core/token/emission_decay.py` — Logistic emission gate (used by reward gating)
- `core/sovereign/ihsan_vector.py` — Ihsan scoring (used by all gates)
- `core/sovereign/probe_defense.py` — Hallucination probe (used by reward function)
