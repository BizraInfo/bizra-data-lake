# Architecture Playbook (Ablation-Driven Upgrades)

Use this reference during the Architect stage.

## Principle
Apply architecture changes only when ablation evidence shows measurable gain.

## Priority Upgrade Paths
1. Federated routing first:
- Route simple tasks to smaller models.
- Route high-complexity tasks to stronger solvers.
- Keep verifier path independent.

2. MoE pattern:
- Split by task family (reasoning, coding, retrieval-heavy).
- Preserve deterministic routing keys in benchmarks.

3. Sequential attention:
- Use one-pass feature selection for long-context or memory-heavy tasks.
- Favor deterministic scoring when leaderboard reproducibility is required.

4. Memory integration:
- Preserve long-horizon context with explicit memory merge policy.
- Add forgetting protection for recurring benchmark loops.

## Inference Optimization
- Quantization (4-bit or 8-bit) when cost or throughput gate is the bottleneck.
- Speculative decoding when latency gate is the bottleneck.

## Selection Rules
- If efficacy bottleneck: improve planner/solver path.
- If assurance bottleneck: strengthen verifier and safety gates.
- If reliability bottleneck: reduce stochasticity and improve routing determinism.
- If cost or latency bottleneck: quantize, cache, and route more aggressively.

## Do Not Do
- Do not ship architecture changes without ablation effect size.
- Do not trade reliability for isolated accuracy gains.
