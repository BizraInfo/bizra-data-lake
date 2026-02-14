# Phase 05: Integration Wiring & Skill Registration

## Status: SPEC
## Depends On: Phases 01-04
## Produces: Unified gateway wiring, skill registration, CLI entry point

---

## 1. Context

The True Spearpoint v9 introduces several new modules that must be wired
into the existing BIZRA infrastructure:

- **Gateway unification**: SovereignSpearpoint._inference_fn vs DominanceLoop._gateway
  need a single injection point
- **Skill registration**: TrueSpearpointLoop needs a Desktop Bridge skill handler
  (like RDVE's `RDVESkillHandler`)
- **Metrics flow**: MetricsProvider → AutoEvaluator → RecursiveLoop → v9 Loop
  feedback cycle must be explicit
- **CLI entry point**: `python -m core.spearpoint.true_spearpoint --run`

---

## 2. Gateway Injection Protocol

### Problem
Two inference injection points exist:
1. `SovereignSpearpoint.set_inference_backend(fn)` — callback function
2. `DominanceLoop.__init__(inference_gateway=gateway)` — gateway object

### Solution
Create a thin adapter that presents both interfaces from a single source.

### Pseudocode

```
MODULE core.spearpoint.gateway_adapter

CLASS GatewayAdapter:
    """
    Unifies InferenceGateway (object) with inference_fn (callback) interfaces.
    Single source of truth for inference in Spearpoint subsystem.
    """

    INIT(gateway: Any = None):
        self._gateway = gateway

    @classmethod
    from_default() -> "GatewayAdapter":
        """Load default gateway from core.inference.gateway."""
        TRY:
            FROM core.inference.gateway IMPORT get_inference_gateway
            gw = get_inference_gateway()
            RETURN GatewayAdapter(gateway=gw)
        EXCEPT ImportError:
            RETURN GatewayAdapter(gateway=None)

    @property
    available -> bool:
        RETURN self._gateway IS NOT None

    ASYNC infer(prompt: str, **kwargs) -> dict:
        """Unified inference call. Returns dict with content, model, tokens."""
        IF self._gateway IS None:
            RETURN {
                "content": f"[simulation] Response to: {prompt[:80]}",
                "model": "simulation",
                "tokens_generated": 0,
                "latency_ms": 0.0,
            }

        result = AWAIT self._gateway.infer(prompt, **kwargs)
        RETURN {
            "content": result.content,
            "model": result.model,
            "tokens_generated": result.tokens_generated,
            "latency_ms": getattr(result, 'latency_ms', 0.0),
        }

    as_callback() -> callable:
        """Return a callback function for SovereignSpearpoint.set_inference_backend."""
        ASYNC DEF _callback(prompt, **kwargs):
            result = AWAIT self.infer(prompt, **kwargs)
            RETURN result["content"]
        RETURN _callback

    as_gateway() -> Any:
        """Return the raw gateway object for DominanceLoop.__init__."""
        RETURN self._gateway
```

### Wiring Pattern

```
# At startup / skill registration:
adapter = GatewayAdapter.from_default()

# For SovereignSpearpoint:
spearpoint = SovereignSpearpoint()
spearpoint.set_inference_backend(adapter.as_callback())

# For DominanceLoop:
loop = DominanceLoop(inference_gateway=adapter.as_gateway())

# For TrueSpearpointLoop:
v9 = TrueSpearpointLoop(config=config, gateway=adapter.as_gateway())
```

---

## 3. Metrics Flow — End-to-End

### Data Flow Diagram

```
TrueSpearpointLoop._run_iteration()
    │
    ├── _phase_evaluate()
    │       │
    │       ├── DominanceLoop._phase_evaluate()
    │       │       └── CLEARFramework.evaluate() → EvaluationContext
    │       │
    │       ├── HALOrchestrator.evaluate_consistency()
    │       │       └── CLEARFramework.evaluate() × k runs
    │       │
    │       └── MetricsProvider.record_cycle_metrics()
    │               ├── Updates running averages
    │               └── Feeds back to AutoEvaluator on next cycle
    │
    ├── _phase_ablate()
    │       ├── AblationEngine (existing)
    │       └── MIRASMemory.retrieve() for past ablation context
    │
    ├── _phase_architect()
    │       ├── ZScorer.score() for routing decisions
    │       └── MoERouter for tier recommendations
    │
    ├── _phase_submit()
    │       └── CampaignOrchestrator.execute()
    │               └── IntegrityValidator → LeaderboardManager
    │
    └── _phase_analyze()
            ├── Compute SNR (weighted: 70% CLEAR + 30% Pass@k)
            ├── Update Pareto frontier
            └── MIRASMemory.store() + .store_episodic()
```

### MetricsProvider Integration

```
# Inside TrueSpearpointLoop._phase_evaluate():

snapshot = self._metrics.current_snapshot()

# Feed real metrics to AutoEvaluator (via AutoResearcher)
IF self._researcher:
    self._researcher.research_with_pattern(
        pattern_id=selected_pattern,
        metrics=snapshot.to_clear_metrics(),
        ihsan_components=snapshot.to_ihsan_components(),
    )

# Record this cycle's results back
self._metrics.record_cycle_metrics(
    approved=(consistency.pass_at_1 > 0.5),
    rejected=(consistency.pass_at_1 <= 0.5),
    clear_score=clear_result.compute_overall_score(),
    ihsan_score=ihsan_result,
)
```

---

## 4. Skill Handler — Desktop Bridge Registration

### Pseudocode

```
MODULE core.spearpoint.spearpoint_skill

IMPORT TrueSpearpointLoop, LoopConfig FROM core.spearpoint.true_spearpoint_loop
IMPORT Battlefield FROM core.benchmark.battlefield_registry
IMPORT UNIFIED_IHSAN_THRESHOLD FROM core.integration.constants

CLASS SpearpointSkillHandler:
    """
    Desktop Bridge adapter for True Spearpoint v9.
    Same pattern as RDVESkillHandler and SmartFileHandler.

    Invocation:
        {"jsonrpc": "2.0", "method": "invoke_skill", "params": {
            "skill": "true_spearpoint",
            "inputs": {"operation": "run", "max_iterations": 5}
        }, "id": 1}
    """

    SKILL_NAME = "true_spearpoint"
    AGENT_NAME = "spearpoint-engine"
    DESCRIPTION = (
        "True Spearpoint v9 — Recursive Benchmark Dominance Loop "
        "with CLEAR evaluation, Pareto optimization, and multi-battlefield campaigns"
    )
    TAGS = ["spearpoint", "benchmark", "dominance", "clear", "pareto", "cowork"]
    VERSION = "9.0.0"

    INIT():
        self._loop: TrueSpearpointLoop = None  # Lazy-loaded
        self._invocation_count = 0

    register(router: Any):
        """Register on SkillRouter (same pattern as RDVE/SmartFiles)."""
        FROM core.skills.registry IMPORT (
            RegisteredSkill, SkillContext, SkillManifest, SkillStatus,
        )

        manifest = SkillManifest(
            name=self.SKILL_NAME,
            description=self.DESCRIPTION,
            version=self.VERSION,
            author="BIZRA Node0",
            context=SkillContext.INLINE,
            agent=self.AGENT_NAME,
            tags=self.TAGS,
            required_inputs=["operation"],
            optional_inputs=[
                "max_iterations", "target_snr", "budget_usd",
                "battlefields", "enable_campaign",
            ],
            outputs=["report"],
            ihsan_floor=UNIFIED_IHSAN_THRESHOLD,
        )

        skill = RegisteredSkill(
            manifest=manifest,
            path="core/spearpoint/spearpoint_skill.py",
            status=SkillStatus.AVAILABLE,
        )

        router.registry._skills[self.SKILL_NAME] = skill

        FOR tag IN self.TAGS:
            tag_list = router.registry._by_tag.setdefault(tag, [])
            IF self.SKILL_NAME NOT IN tag_list:
                tag_list.append(self.SKILL_NAME)

        agent_list = router.registry._by_agent.setdefault(self.AGENT_NAME, [])
        IF self.SKILL_NAME NOT IN agent_list:
            agent_list.append(self.SKILL_NAME)

        router.register_handler(self.AGENT_NAME, self._handle)

    ASYNC _handle(skill, inputs: dict, context: dict = None) -> dict:
        """Dispatch to operation handlers."""
        operation = inputs.get("operation", "")
        self._invocation_count += 1

        dispatch = {
            "run": self._op_run,
            "status": self._op_status,
            "pareto": self._op_pareto,
            "memory": self._op_memory,
        }

        handler = dispatch.get(operation)
        IF handler IS None:
            RETURN {
                "error": f"Unknown operation: '{operation}'",
                "available_operations": list(dispatch.keys()),
            }

        RETURN AWAIT handler(inputs)

    ASYNC _op_run(inputs: dict) -> dict:
        """Run the dominance loop."""
        config = LoopConfig(
            max_iterations=int(inputs.get("max_iterations", 5)),
            target_snr=float(inputs.get("target_snr", 0.99)),
            budget_usd=float(inputs.get("budget_usd", 500.0)),
            enable_campaign=bool(inputs.get("enable_campaign", False)),
            battlefields=[
                Battlefield(bf) FOR bf IN inputs.get("battlefields", [])
            ],
        )

        self._loop = TrueSpearpointLoop(config=config)
        report = AWAIT self._loop.run()

        RETURN {
            "operation": "run",
            "iterations": report.iterations_completed,
            "final_snr": report.final_snr,
            "convergence_reason": report.convergence_reason,
            "battlefields_won": [bf.value FOR bf IN report.battlefields_won],
            "total_cost_usd": report.total_cost_usd,
        }

    ASYNC _op_status(inputs: dict) -> dict:
        """Return current loop status."""
        IF self._loop IS None:
            RETURN {"status": "idle", "invocation_count": self._invocation_count}

        RETURN {
            "status": "running" IF self._loop._iteration > 0 ELSE "ready",
            "iteration": self._loop._iteration,
            "best_snr": self._loop._best_snr,
            "spent_usd": self._loop._spent_usd,
            "memory": self._loop._memory.get_stats(),
        }

    ASYNC _op_pareto(inputs: dict) -> dict:
        """Return current Pareto frontier."""
        IF self._loop IS None OR NOT self._loop._pareto_history:
            RETURN {"frontier": [], "count": 0}

        frontier = self._loop._pareto_history[-1]
        RETURN {
            "frontier": [
                {"agent_id": p.agent_id, "efficacy": p.efficacy,
                 "cost": p.cost, "reliability": p.reliability}
                FOR p IN frontier
            ],
            "count": len(frontier),
        }

    ASYNC _op_memory(inputs: dict) -> dict:
        """Query the loop's MIRAS memory."""
        IF self._loop IS None:
            RETURN {"error": "No loop running", "entries": []}

        query = inputs.get("query", "recent results")
        k = int(inputs.get("k", 10))
        result = self._loop._memory.retrieve(query, k=k)

        RETURN {
            "entries": [
                {"content": e.content[:200], "tier": e.tier}
                FOR e IN result.entries
            ],
            "total": result.total_retrieved,
            "sources": result.sources,
        }


# Module-level convenience
def register_spearpoint_skill(router: Any) -> SpearpointSkillHandler:
    handler = SpearpointSkillHandler()
    handler.register(router)
    RETURN handler
```

---

## 5. Desktop Bridge Registration

```
# In core/bridges/desktop_bridge.py :: _get_skill_router()

# Auto-register True Spearpoint skill (best-effort)
TRY:
    FROM core.spearpoint.spearpoint_skill IMPORT register_spearpoint_skill
    register_spearpoint_skill(self._skill_router)
    logger.info("True Spearpoint skill auto-registered on bridge SkillRouter")
EXCEPT Exception AS exc:
    logger.debug(f"True Spearpoint registration skipped: {exc}")
```

---

## 6. CLI Entry Point

### Pseudocode

```
MODULE core.spearpoint.__main__

IMPORT TrueSpearpointLoop, LoopConfig FROM core.spearpoint.true_spearpoint_loop
IMPORT GatewayAdapter FROM core.spearpoint.gateway_adapter

ASYNC DEF main():
    IMPORT argparse

    parser = argparse.ArgumentParser(description="True Spearpoint v9")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--target-snr", type=float, default=0.99)
    parser.add_argument("--budget", type=float, default=500.0)
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--battlefields", nargs="*", default=[])
    args = parser.parse_args()

    config = LoopConfig(
        max_iterations=args.max_iterations,
        target_snr=args.target_snr,
        budget_usd=args.budget,
        enable_campaign=args.campaign,
    )

    adapter = GatewayAdapter.from_default()
    loop = TrueSpearpointLoop(config=config, gateway=adapter.as_gateway())

    PRINT(f"True Spearpoint v9 — Target SNR: {config.target_snr}")
    PRINT(f"Max iterations: {config.max_iterations}, Budget: ${config.budget_usd}")

    report = AWAIT loop.run()

    PRINT(f"\nCompleted in {report.iterations_completed} iterations")
    PRINT(f"Final SNR: {report.final_snr:.4f}")
    PRINT(f"Convergence: {report.convergence_reason}")
    PRINT(f"Cost: ${report.total_cost_usd:.2f}")
    IF report.battlefields_won:
        PRINT(f"Battlefields won: {[bf.value FOR bf IN report.battlefields_won]}")

IF __name__ == "__main__":
    IMPORT asyncio
    asyncio.run(main())
```

---

## 7. TDD Anchors

```
TEST test_gateway_adapter_simulation_mode:
    adapter = GatewayAdapter(gateway=None)
    ASSERT adapter.available == False
    result = AWAIT adapter.infer("test prompt")
    ASSERT "simulation" IN result["content"]
    ASSERT result["model"] == "simulation"

TEST test_gateway_adapter_callback:
    adapter = GatewayAdapter(gateway=None)
    callback = adapter.as_callback()
    result = AWAIT callback("test")
    ASSERT isinstance(result, str)

TEST test_skill_registration:
    FROM core.skills.registry IMPORT SkillRegistry
    FROM core.skills.mcp_bridge IMPORT MCPBridge
    FROM core.skills.router IMPORT SkillRouter

    registry = SkillRegistry(skills_dir="/nonexistent")
    router = SkillRouter(registry=registry, mcp_bridge=MCPBridge())

    handler = SpearpointSkillHandler()
    handler.register(router)

    skill = router.registry.get("true_spearpoint")
    ASSERT skill IS NOT None
    ASSERT skill.manifest.agent == "spearpoint-engine"
    ASSERT "spearpoint" IN skill.manifest.tags
    ASSERT "spearpoint-engine" IN router._handlers

TEST test_skill_unknown_operation:
    handler = SpearpointSkillHandler()
    result = AWAIT handler._handle(None, {"operation": "invalid"})
    ASSERT "error" IN result
    ASSERT "available_operations" IN result

TEST test_skill_status_when_idle:
    handler = SpearpointSkillHandler()
    result = AWAIT handler._handle(None, {"operation": "status"})
    ASSERT result["status"] == "idle"
```

---

## 8. File Deliverables

| File | Lines | Purpose |
|------|-------|---------|
| `core/spearpoint/gateway_adapter.py` | ~60 | Unified gateway interface |
| `core/spearpoint/spearpoint_skill.py` | ~180 | Skill handler for bridge |
| `tests/core/spearpoint/test_gateway_adapter.py` | ~50 | Adapter tests |
| `tests/core/spearpoint/test_spearpoint_skill.py` | ~80 | Skill handler tests |

---

## 9. Implementation Order

```
1. gateway_adapter.py          (no deps beyond existing gateway)
2. battlefield_registry.py     (Phase 03 — standalone data)
3. z_scorer.py                 (Phase 02 — extends ComplexityClassifier)
4. miras_memory.py             (Phase 02 — stdlib only)
5. zero_failure_trainer.py     (Phase 02 — standalone)
6. hal_orchestrator.py         (Phase 01 — composes CLEARFramework)
7. steering_adapter.py         (Phase 01 — standalone)
8. abc_enhanced.py             (Phase 01 — extends AgenticBenchmarkChecklist)
9. integrity_validator.py      (Phase 03 — extends AntiGamingValidator)
10. campaign_orchestrator.py   (Phase 03 — composes leaderboard + integrity)
11. true_spearpoint_loop.py    (Phase 04 — composes ALL of above)
12. spearpoint_skill.py        (Phase 05 — skill handler)
13. Update desktop_bridge.py   (Phase 05 — registration)
14. Tests for all modules
```
