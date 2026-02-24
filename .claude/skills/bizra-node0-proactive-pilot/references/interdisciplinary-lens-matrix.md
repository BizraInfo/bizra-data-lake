# Interdisciplinary Lens Matrix

> Standing on Giants: Shannon (signal fidelity across domains) · Boyd (multi-perspective orientation) · Al-Ghazali (ethical evaluation as first-class lens)

Evaluate every non-trivial operational decision through all 7 lenses before convergence. Any single lens can **veto** a recommendation if its veto condition is triggered.

## 7-Lens Framework

### 1. Systems

**Domain**: Runtime topology, dependency resolution, state machine correctness.

| Aspect | Detail |
| --- | --- |
| Key Questions | Is the runtime topology healthy? Are all dependencies resolved? Is the state machine in a valid transition? |
| Required Evidence | `python scripts/node0_activate.py status` output, PID state from `sovereign_state/proactive.pid`, process tree |
| Veto Condition | Circular dependency detected; state machine in undefined transition; orphaned child processes |
| Repo Anchor | `core/sovereign/runtime_core.py`, `scripts/node0_activate.py` |

### 2. Reliability

**Domain**: Durability, restart survivability, checkpoint integrity.

| Aspect | Detail |
| --- | --- |
| Key Questions | Can the node survive restart? Is state durable? Are checkpoints current (within 300s cycle)? |
| Required Evidence | Checkpoint files in `sovereign_state/checkpoints/`, PID lifecycle logs in `logs/proactive/startup.log` |
| Veto Condition | No checkpoint written in > 300s; PID file present but process dead (stale PID); log directory missing |
| Repo Anchor | `config/proactive_config.yaml` (checkpoint interval), `scripts/stop_proactive.sh` (shutdown lifecycle) |

### 3. Security

**Domain**: Token scope, network trust, secret hygiene.

| Aspect | Detail |
| --- | --- |
| Key Questions | Are tokens properly scoped? Is the LM Studio network path trusted? Are secrets excluded from logs? |
| Required Evidence | `LM_API_TOKEN` / `LM_STUDIO_API_KEY` state, LM Studio endpoint TLS status, log content audit |
| Veto Condition | Token value leaked to stdout/logs; untrusted network path to LM Studio; secrets committed to repo |
| Repo Anchor | `scripts/set_lm_studio_key.sh`, `SECURITY.md`, `deploy/node0/secrets.env.template` |

### 4. Economics

**Domain**: Token budget, cost per mission, resource utilization.

| Aspect | Detail |
| --- | --- |
| Key Questions | Is token budget within limits? Cost per mission vs. baseline? GPU/memory utilization acceptable? |
| Required Evidence | Mission token usage from `node0_activate.py mission` output, system resource metrics |
| Veto Condition | Token burn > 2× established baseline; GPU OOM risk; runaway inference loop detected |
| Repo Anchor | `core/integration/constants.py` (ADL thresholds), `NODE0_IDENTITY.yaml` (hardware specs) |

### 5. Ethics (Ihsān)

**Domain**: Constitutional compliance, ethical threshold adherence, Daughter Test.

| Aspect | Detail |
| --- | --- |
| Key Questions | Does the action meet Ihsān threshold (≥ 0.95 production)? Does it pass the Daughter Test? Is ADL Gini ≤ 0.40? |
| Required Evidence | Ihsān score from mission output, `ConstitutionalGate` result (APPROVED/NEEDS_REVIEW/REJECTED) |
| Veto Condition | Ihsān < 0.95 for production ops; Daughter Test fails; ADL Gini exceeds 0.40; constitutional REJECTED |
| Repo Anchor | `core/integration/constants.py` (Ihsān thresholds), `core/governance/constitutional_gate.py`, `docs/PROACTIVE_SOVEREIGN_ENTITY.md` § Constitutional Filters |

### 6. Operations

**Domain**: Run/stop lifecycle, log capture, operational hygiene.

| Aspect | Detail |
| --- | --- |
| Key Questions | Is the run/stop lifecycle clean? Are logs being captured? Is the PID managed correctly? |
| Required Evidence | Log artifacts in `logs/proactive/`, PID file state, `startup.log` timestamps, graceful shutdown evidence |
| Veto Condition | Stale PID with no running process; missing log directory; start without prior clean stop; unlogged force-kill |
| Repo Anchor | `scripts/start_proactive.sh`, `scripts/stop_proactive.sh`, `sovereign_state/proactive.pid` |

### 7. Product Impact

**Domain**: User intent alignment, mission output quality, end-value delivery.

| Aspect | Detail |
| --- | --- |
| Key Questions | Does the mission output serve user intent? Is the output actionable and specific? Does it advance the stated goal? |
| Required Evidence | Mission result vs. task description comparison, PAT agent outputs, SNR score of response |
| Veto Condition | Output diverges from task intent; response is generic/templated without specific evidence; SNR < 0.85 |
| Repo Anchor | `scripts/node0_activate.py` (PAT agent dispatch), `core/integration/constants.py` (SNR floor) |

## Application Rules

1. **All 7 lenses are mandatory** for mutating operation recommendations. For non-mutating diagnostics, lenses 1 (Systems), 2 (Reliability), and 6 (Operations) are the minimum required set.
2. **Any single veto blocks** the recommendation. Document the vetoing lens and condition clearly.
3. **Lens evaluation order** is fixed (1→7) to ensure deterministic reasoning traces.
4. When a lens cannot be evaluated due to missing evidence, it produces a **DEFER** result — which blocks mutating ops but allows diagnostics.
5. For the Response Contract v2, summarize lens results as: `[lens]: PASS | VETO(reason) | DEFER(missing evidence)`.
