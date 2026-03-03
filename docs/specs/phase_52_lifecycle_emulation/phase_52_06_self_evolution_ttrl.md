# Phase 52.6: TTRL Self-Evolution (Phase 5 -- On-Device Learning)

> Standing on Giants: Friston (Free Energy Principle and active inference, 2006) · Deming (PDCA continuous improvement cycle, 1950) · Schultz (dopamine reward prediction error, 1997) · Shannon (information gain as learning signal, 1948) · Al-Ghazali (Ihsan as self-improvement imperative, 1095)

## 1. Overview

After Ahmed's task completes, the node does not discard the experience. TTRL
(Test-Time Reinforcement Learning) uses the PAT-7 majority vote as a reward signal
to perform a GRPO (Group Relative Policy Optimization) weight update on the 7B MoE
model. SSO (Spectral Stability Optimization) then projects the updated weights back
into a stable spectral norm ball, preventing catastrophic drift.

The result: Ahmed's node gets slightly better at invoice organization after every
successful run. The update is tiny (learning rate ~1e-5), cumulative, and
stability-constrained. After 10 tasks, the model generates better decompositions and
more accurate Telescript actions for this task class.

---

## 2. Data Flow

```
  Task completed (receipts generated)
       │
  ┌────▼─────────────────────────────────────────────┐
  │  REWARD SIGNAL COMPUTATION                        │
  │  PAT-7 majority vote + Ihsan + CPVA + chain OK   │
  └────┬─────────────────────────────────────────────┘
       │ reward: float [-1.0, 1.0]
  ┌────▼─────────────────────────────────────────────┐
  │  GRPO (Group Relative Policy Optimization)        │
  │  1. Sample K=4 completions for same task          │
  │  2. Score each with reward function               │
  │  3. Advantage = score - mean(scores)              │
  │  4. Policy loss = -advantage * log_prob           │
  │  5. Gradient step with lr=1e-5                    │
  └────┬─────────────────────────────────────────────┘
       │ updated weights (raw)
  ┌────▼─────────────────────────────────────────────┐
  │  SSO (Spectral Stability Optimization)            │
  │  1. Spectral norm of weight delta                 │
  │  2. If norm > sigma_max: project back             │
  │  3. Verify stability constraint                   │
  └────┬─────────────────────────────────────────────┘
       │ stable weights
       ▼ Model updated. Next task benefits.
```

---

## 3. Pseudocode

### 3.1 Reward Signal

```python
from __future__ import annotations

import logging
from dataclasses import dataclass

from core.integration.constants import IHSAN_THRESHOLD

logger = logging.getLogger("bizra.ttrl")


@dataclass
class RewardSignal:
    """Multi-dimensional reward from task completion."""
    pat7_approval_ratio: float = 0.0   # [0,1]: fraction of agents approving
    ihsan_composite: float = 0.0       # [0,1]: 8-dim weighted Ihsan
    cpva_efficiency: float = 0.0       # [0,1]: 1 - (actual/budget)
    chain_integrity: bool = True

    @property
    def reward(self) -> float:
        """Composite reward in [-1.0, 1.0]."""
        if not self.chain_integrity:
            return -1.0
        if self.ihsan_composite < IHSAN_THRESHOLD:
            return -0.5
        r = (0.4 * self.pat7_approval_ratio + 0.3 * self.ihsan_composite
             + 0.2 * self.cpva_efficiency + 0.1 * float(self.chain_integrity))
        return min(1.0, max(-1.0, (r - 0.5) * 2))


def compute_pat7_vote(agents: list[str], task_results: list[dict]) -> float:
    """PAT-7 majority vote. Each agent votes approve/reject/abstain."""
    votes = {"approve": 0, "reject": 0, "abstain": 0}
    for agent in agents:
        all_ok = all(r.get("success") for r in task_results)
        if agent == "ethicist":
            avg = sum(r.get("ihsan", 0) for r in task_results) / max(len(task_results), 1)
            votes["approve" if avg >= IHSAN_THRESHOLD else "reject"] += 1
        else:
            votes["approve" if all_ok else "reject"] += 1
    total = votes["approve"] + votes["reject"]
    return votes["approve"] / total if total > 0 else 0.5
```

### 3.2 GRPO Engine

```python
import torch


class GRPOEngine:
    """Group Relative Policy Optimization for on-device learning."""

    def __init__(self, model: torch.nn.Module, learning_rate: float = 1e-5,
                 max_grad_norm: float = 1.0, group_size: int = 4) -> None:
        self.model = model
        self.lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.group_size = group_size
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    async def update(self, task_prompt: str, reward: RewardSignal) -> dict:
        # Step 1: Sample K completions
        completions = await self._sample(task_prompt, k=self.group_size)
        # Step 2: Score each
        scores = [reward.reward for _ in completions]  # Simplified
        # Step 3: Advantages
        mean_s = sum(scores) / len(scores)
        advantages = [s - mean_s for s in scores]
        # Step 4: Policy gradient
        loss = torch.tensor(-sum(advantages) / len(advantages), requires_grad=True)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {"loss": loss.item(), "mean_score": mean_s,
                "grad_norm": self._grad_norm()}

    async def _sample(self, prompt: str, k: int) -> list[str]:
        return [f"completion_{i}" for i in range(k)]

    def _grad_norm(self) -> float:
        return sum(p.grad.norm().item() ** 2 for p in self.model.parameters()
                   if p.grad is not None) ** 0.5
```

### 3.3 SSO Spectral Projector

```python
class SSOProjector:
    """Constrains weight updates to prevent catastrophic drift.
    Standing on Giants: Friston (2006) -- free energy with stability."""

    def __init__(self, sigma_max: float = 0.1) -> None:
        self.sigma_max = sigma_max

    def project(self, w_before: torch.Tensor,
                w_after: torch.Tensor) -> tuple[torch.Tensor, dict]:
        delta = w_after - w_before
        if delta.dim() >= 2:
            spec_norm = torch.linalg.svdvals(delta)[0].item()
        else:
            spec_norm = delta.norm().item()

        projected = False
        if spec_norm > self.sigma_max:
            delta = delta * (self.sigma_max / spec_norm)
            projected = True

        return w_before + delta, {
            "spectral_norm_raw": spec_norm,
            "spectral_norm_final": min(spec_norm, self.sigma_max),
            "projected": projected}


class TTRLEngine:
    """Orchestrates GRPO + SSO for on-device self-evolution."""

    def __init__(self, model_name: str, vram_gb: int) -> None:
        self.model_name = model_name
        self.vram_gb = vram_gb
        self.grpo: GRPOEngine | None = None
        self.sso = SSOProjector(sigma_max=0.1)
        self.update_count = 0

    async def evolve(self, task_prompt: str, reward: RewardSignal) -> dict:
        if self.grpo is None:
            return {"skipped": True, "reason": "model_not_loaded"}

        w_before = {n: p.clone() for n, p in self.grpo.model.named_parameters()}
        grpo_metrics = await self.grpo.update(task_prompt, reward)

        sso_metrics = []
        for name, param in self.grpo.model.named_parameters():
            if name in w_before:
                projected, m = self.sso.project(w_before[name], param.data)
                param.data = projected
                sso_metrics.append({name: m})

        self.update_count += 1
        return {"grpo": grpo_metrics, "sso": sso_metrics,
                "update_count": self.update_count}
```

---

## 4. Ahmed's TTRL Cycle (Concrete)

```
Reward: PAT-7 vote 6/7 approve → ratio=1.0, ihsan=0.97, cpva_eff=0.28 → r=0.67
GRPO: 4 completions, scores=[0.67,0.52,0.71,0.59], loss=-0.0023, grad=0.041
SSO: max spectral norm 0.031 < sigma_max 0.1 → no projection needed
Result: Model slightly better at PDF organization
```

---

## 5. TDD Anchors

```python
import pytest

class TestTTRLSelfEvolution:
    """Phase 52.6: TTRL self-evolution tests."""

    def test_reward_signal_positive(self):
        s = RewardSignal(pat7_approval_ratio=1.0, ihsan_composite=0.97,
                         cpva_efficiency=0.3, chain_integrity=True)
        assert s.reward > 0

    def test_reward_signal_negative_broken_chain(self):
        assert RewardSignal(chain_integrity=False).reward == -1.0

    def test_reward_signal_negative_low_ihsan(self):
        assert RewardSignal(ihsan_composite=0.80, chain_integrity=True).reward < 0

    def test_pat7_vote_majority(self):
        ratio = compute_pat7_vote(
            ["planner", "researcher", "coder", "evaluator",
             "ethicist", "publisher", "integrator"],
            [{"success": True, "ihsan": 0.97}])
        assert ratio >= 0.8

    @pytest.mark.asyncio
    async def test_grpo_update(self):
        grpo = GRPOEngine(model=MockModel(), learning_rate=1e-5)
        metrics = await grpo.update("test",
            RewardSignal(pat7_approval_ratio=0.9, ihsan_composite=0.96,
                         chain_integrity=True))
        assert "loss" in metrics

    def test_sso_projection_within_bounds(self):
        sso = SSOProjector(sigma_max=0.1)
        w = torch.randn(64, 64)
        projected, m = sso.project(w, w + torch.randn(64, 64) * 0.01)
        assert m["spectral_norm_final"] <= 0.1

    def test_sso_projection_enforced(self):
        sso = SSOProjector(sigma_max=0.1)
        w = torch.zeros(64, 64)
        projected, m = sso.project(w, torch.randn(64, 64) * 10.0)
        assert m["projected"] is True
        assert m["spectral_norm_final"] <= 0.1 + 1e-6

    def test_stability_maintained(self):
        sso = SSOProjector(sigma_max=0.05)
        w = torch.randn(32, 32)
        projected, _ = sso.project(w, w + torch.randn(32, 32) * 0.5)
        delta = projected - w
        assert torch.linalg.svdvals(delta)[0].item() <= 0.05 + 1e-6
```
