# Phase 50.3 — Token-Incentivized Agent Reinforcement Learning

> Standing on Giants: Sutton & Barto (RL theory, 1998) · Schulman (PPO, 2017) · Nakamoto (token incentives, 2008) · DeepSeek-AI (RL for reasoning, 2025) · Gini (inequality constraint, 1912) · Al-Ghazali (Ihsan as reward signal, 1095)

## 1. Concept

BIZRA's existing token economy (SEED/BLOOM/IMPT) already implements a **Proof of Impact** reward system. The missing piece is closing the loop: using token rewards as **reinforcement signals** that improve PAT agent behavior over time.

The paper "Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (DeepSeek-AI, arXiv:2501.12948, referenced in the RLM paper) demonstrates that RL can dramatically improve LLM reasoning. BIZRA can apply this principle using its own token economy as the reward function.

### The Feedback Loop

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Mission → PAT Agent → Response → Quality Score     │
│                                          │          │
│                                    ┌─────┴─────┐    │
│                                    │ Ihsan+SNR  │    │
│                                    │  Scoring   │    │
│                                    └─────┬─────┘    │
│                                          │          │
│                            ┌─────────────┴───┐      │
│                            │  Token Reward    │      │
│                            │  SEED = f(score) │      │
│                            └─────────┬───────┘      │
│                                      │              │
│                            ┌─────────┴───────┐      │
│                            │  Strategy Update │      │
│                            │  (RL Policy)     │      │
│                            └─────────┬───────┘      │
│                                      │              │
│  Next Mission ← Updated Strategy ←───┘              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 2. Reward Function Design

### Composite Reward

The reward for agent `a` on mission `m` is:

```
R(a, m) = w_snr × SNR(response)
        + w_ihsan × Ihsan(response)
        + w_token × TokenEfficiency(tokens_used)
        + w_user × UserFeedback(response)
        - penalty_hallucination × HallucinationProbe(response)
        - penalty_latency × max(0, latency - target_latency)
```

### Weight Vector

| Component | Weight | Source | Justification |
|-----------|--------|--------|---------------|
| SNR score | 0.30 | `core/integration/constants.py` | Signal quality is primary objective |
| Ihsan score | 0.25 | `core/sovereign/ihsan_vector.py` | Constitutional constraint |
| Token efficiency | 0.15 | `core/token/mint.py` | Economic sustainability |
| User feedback | 0.20 | Interactive denoiser | User satisfaction is ground truth |
| Hallucination penalty | -0.05 | `core/sovereign/probe_defense.py` | Safety constraint |
| Latency penalty | -0.05 | Wall-clock time | Responsiveness |

### Token Efficiency Sub-Reward

Encourages agents to solve tasks with fewer tokens (like RLM's cost-efficiency):

```python
def token_efficiency_reward(tokens_used: int, quality: float) -> float:
    """
    Reward quality per token. Higher quality with fewer tokens = more reward.

    Standing on Giants: Shannon (1948) — bits per symbol efficiency.
    """
    if tokens_used <= 0:
        return 0.0

    # Quality per 1000 tokens — normalized
    efficiency = (quality * 1000) / tokens_used

    # Logistic scaling to [0, 1]
    return 1.0 / (1.0 + math.exp(-5 * (efficiency - 0.5)))
```

## 3. Token-to-RL Reward Bridge

### SEED as Reward Signal

Each PAT agent has an account in the token ledger. After each mission:

```python
def compute_agent_reward(
    agent_id: str,
    mission_result: dict,
    minter: TokenMinter,
    emission_gate: LogisticEmissionGate,
) -> TokenReceipt:
    """
    Compute and mint SEED reward for agent performance.

    The reward amount is gated by:
    1. Composite quality score (R function above)
    2. Logistic emission gate (Gini-based throttling)
    3. Yearly supply cap (1M SEED/year)
    """
    # Compute composite reward
    reward_score = composite_reward(mission_result)

    # Base reward amount (scaled by score)
    base_amount = reward_score * 10.0  # 10 SEED for perfect score

    # Apply emission gate (Gini-based throttling)
    current_holdings = get_all_agent_balances()
    gate_result = emission_gate.compute_gated_emission(
        requested_amount=base_amount,
        current_holdings=current_holdings,
    )

    gated_amount = gate_result["gated_amount"]

    # Mint if above minimum threshold
    if gated_amount >= 0.01:
        return minter.mint_seed(
            to_account=agent_id,
            amount=gated_amount,
            epoch_id=f"mission-{mission_result['mission_id']}",
            poi_score=reward_score,
            memo=f"RL reward: SNR={mission_result.get('snr', 0):.3f}, "
                 f"Ihsan={mission_result.get('ihsan', 0):.3f}",
        )

    return TokenReceipt(success=True, error="Below minimum threshold")
```

### IMPT as Reputation Signal

IMPT (soulbound reputation) compounds over time and affects future mission assignment:

```python
def update_agent_reputation(agent_id: str, reward_score: float, minter: TokenMinter):
    """
    Update agent's IMPT reputation based on cumulative performance.

    IMPT is non-transferable — it represents the agent's track record.
    Higher IMPT → assigned to harder missions → higher potential reward.
    """
    # IMPT earned = reward_score * reputation_multiplier
    current_impt = minter.ledger.get_balance(agent_id, TokenType.IMPT)

    # Diminishing returns: sqrt scaling prevents runaway reputation
    multiplier = 1.0 + math.sqrt(current_impt / 100.0)
    impt_earned = reward_score * 0.1 * multiplier

    if impt_earned > 0:
        minter.mint_impt(
            to_account=agent_id,
            amount=impt_earned,
            poi_score=reward_score,
            memo=f"Reputation update from mission performance",
        )
```

## 4. Strategy Learning (Policy Update)

### Approach: Contextual Bandit with Memory

Full RL (PPO/DPO) requires gradient access to the LLM — not possible with API-based models. Instead, we use a **contextual bandit** approach where the "policy" is a set of **strategy parameters** stored in Living Memory:

```python
@dataclass
class AgentStrategy:
    """
    Learnable strategy parameters for a PAT agent.

    These parameters are stored in PROCEDURAL memory and
    updated after each mission based on token rewards.
    """
    # Prompt engineering parameters
    system_prompt_template: str          # Which template to use
    temperature: float = 0.7            # Sampling temperature
    max_tokens: int = 600               # Token budget

    # RLM parameters (when in RLM mode)
    use_rlm: bool = False               # Whether to use RLM
    rlm_max_iterations: int = 10        # REPL loop budget
    rlm_sub_call_budget: int = 20       # Sub-call budget
    preferred_probe_strategy: str = ""  # e.g., "regex_first", "chunk_split"

    # Task routing parameters
    confidence_threshold: float = 0.8   # When to request help from other agents
    delegation_preference: float = 0.5  # Prefer delegation vs. solo work

    # Performance history (exponential moving average)
    ema_reward: float = 0.5             # EMA of recent rewards
    ema_alpha: float = 0.1              # EMA decay rate
    total_missions: int = 0
    total_reward: float = 0.0


def update_strategy(
    strategy: AgentStrategy,
    reward: float,
    mission_context: dict,
) -> AgentStrategy:
    """
    Update agent strategy based on mission reward.

    Uses exponential moving average to smooth reward signal
    and epsilon-greedy exploration to try new strategies.

    Standing on Giants:
    - Sutton & Barto (1998): EMA reward tracking
    - Auer et al. (2002): UCB exploration
    """
    # Update EMA reward
    strategy.ema_reward = (
        strategy.ema_alpha * reward
        + (1 - strategy.ema_alpha) * strategy.ema_reward
    )
    strategy.total_missions += 1
    strategy.total_reward += reward

    # Adaptive temperature: lower temp when strategy is working well
    if strategy.ema_reward > 0.8:
        strategy.temperature = max(0.3, strategy.temperature - 0.05)
    elif strategy.ema_reward < 0.4:
        strategy.temperature = min(1.0, strategy.temperature + 0.05)

    # RLM activation: enable if tasks are getting more complex
    if mission_context.get("prompt_length", 0) > 16000:
        if reward > 0.7:
            strategy.use_rlm = True  # Good results on long context — keep RLM
        elif strategy.use_rlm and reward < 0.4:
            # RLM not helping — try single-shot next time
            strategy.rlm_max_iterations = max(5, strategy.rlm_max_iterations - 2)

    # Token budget adaptation
    tokens_used = mission_context.get("tokens_used", 0)
    if reward > 0.8 and tokens_used < strategy.max_tokens * 0.5:
        # High reward with few tokens — can reduce budget
        strategy.max_tokens = max(300, strategy.max_tokens - 50)
    elif reward < 0.4 and tokens_used >= strategy.max_tokens * 0.9:
        # Low reward at budget cap — need more tokens
        strategy.max_tokens = min(2000, strategy.max_tokens + 100)

    return strategy
```

## 5. Gini Constraint on Agent Economy

The ADL Gini invariant (G ≤ 0.35) applies to agent token holdings, preventing any single agent from accumulating disproportionate rewards:

```python
def enforce_agent_gini(minter: TokenMinter, agent_ids: list[str]):
    """
    Enforce Gini invariant across agent token holdings.

    If Gini exceeds 0.35, the LogisticEmissionGate automatically
    throttles rewards to top-earning agents while maintaining
    rewards for lower-performing ones.

    Standing on Giants: Gini (1912), Piketty (2013)
    """
    holdings = [
        minter.ledger.get_balance(aid, TokenType.SEED)
        for aid in agent_ids
    ]

    gini = _calculate_gini(holdings)

    if gini > 0.35:
        logger.warning(
            "Agent economy Gini %.3f exceeds threshold 0.35. "
            "Emission gate will throttle top earners.",
            gini,
        )
        # The LogisticEmissionGate handles this automatically
        # through compute_gated_emission()
```

## 6. Memory Persistence of Learned Strategies

Agent strategies are stored in Living Memory PROCEDURAL layer:

```python
async def persist_strategy(memory: LivingMemoryCore, agent_id: str, strategy: AgentStrategy):
    """Store strategy in PROCEDURAL memory for cross-session persistence."""
    await memory.encode(
        content=strategy.to_dict(),
        memory_type=MemoryType.PROCEDURAL,
        tags=[f"agent:{agent_id}", "strategy", "rl"],
        metadata={
            "ema_reward": strategy.ema_reward,
            "total_missions": strategy.total_missions,
        },
    )

async def load_strategy(memory: LivingMemoryCore, agent_id: str) -> AgentStrategy:
    """Load strategy from PROCEDURAL memory."""
    results = await memory.retrieve(
        query=f"agent:{agent_id} strategy rl",
        memory_type=MemoryType.PROCEDURAL,
        limit=1,
    )
    if results:
        return AgentStrategy.from_dict(results[0].content)
    return AgentStrategy()  # Default strategy for new agents
```

## 7. Training Signal from RLM Trajectories

The RLM paper's Observation 6 shows that fine-tuning on just 1,000 trajectories improves performance by 28.3%. BIZRA can collect trajectories from RLM sessions and use them for future strategy improvement:

```python
@dataclass
class RLMTrajectory:
    """
    A recorded RLM execution trajectory for future learning.

    Trajectories that achieved high rewards become training data
    for strategy optimization.
    """
    task: str
    prompt_length: int
    iterations: list[dict]  # [{code, stdout, metadata}, ...]
    final_answer: str
    reward: float
    tokens_used: int

    @property
    def is_high_quality(self) -> bool:
        """Trajectory worth learning from?"""
        return self.reward >= 0.8 and not self.partial


def collect_trajectory(rlm_result: RLMResult, reward: float) -> RLMTrajectory:
    """Package an RLM session into a trajectory for future learning."""
    return RLMTrajectory(
        task=rlm_result.trace[0] if rlm_result.trace else "",
        prompt_length=0,  # Filled by caller
        iterations=[
            {"step": i, "content": t[:2000]}
            for i, t in enumerate(rlm_result.trace[1:])
        ],
        final_answer=rlm_result.answer[:5000],
        reward=reward,
        tokens_used=rlm_result.tokens_used,
    )
```

## 8. Economic Equilibrium

The token incentive RL system reaches equilibrium when:

1. **Agent quality improves** → Higher rewards → Strategy reinforcement
2. **Gini gate activates** → Top agents throttled → Bottom agents catch up
3. **Emission decay** → As total supply grows, per-mission rewards shrink → Quality must improve to earn same SEED
4. **Zakat redistribution** → 2.5% of all rewards flow to community fund → Prevents pure accumulation

This creates a **self-regulating** agent economy where quality improves monotonically but wealth concentration is bounded by constitutional constraints.
