"""
Agent Executor — PEK Proposal → PAT Agent Execution Bridge
============================================================
Bridges the OpportunityPipeline's execution stage to activated
PAT agents via InferenceGateway.

This is the second half of the True Spearpoint: AgentActivator creates
the agents; AgentExecutor dispatches work to them.

The executor plugs into OpportunityPipeline.set_execution_callback()
to handle the EXECUTION stage of every approved opportunity.

Architecture:
    PEK → OpportunityPipeline → _stage_execution → AgentExecutor → ActiveAgent → InferenceGateway

Standing on Giants:
- Boyd (1976): OODA ACT phase — the executor IS the actuator
- Deming (1950): Plan-Do-Check-Act — executor handles "Do"
- Shannon (1948): SNR filtering on agent output
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger("sovereign.agent_executor")


@dataclass
class ExecutionResult:
    """Result of executing a task through a PAT agent."""

    success: bool = False
    agent_id: str = ""
    agent_role: str = ""
    content: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "content": self.content,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class AgentExecutor:
    """
    Dispatches PEK-approved opportunities to activated PAT agents.

    The executor:
    1. Receives PipelineOpportunity from OpportunityPipeline's execution stage
    2. Selects appropriate activated agents via AgentActivator
    3. Dispatches work through InferenceGateway
    4. Collects and synthesizes multi-agent responses
    5. Returns structured execution results

    Plugs into: OpportunityPipeline.set_execution_callback(executor.execute)

    Usage:
        executor = AgentExecutor(activator=activator, gateway=gateway)
        pipeline.set_execution_callback(executor.execute)
        # Now PEK proposals flow: PEK → Pipeline → Executor → PAT → Gateway → LLM
    """

    def __init__(
        self,
        activator: Optional[object] = None,
        gateway: Optional[object] = None,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        snr_threshold: float = UNIFIED_SNR_THRESHOLD,
        max_agents_per_task: int = 3,
        agent_timeout: float = 120.0,
    ) -> None:
        self._activator = activator
        self._gateway = gateway
        self._ihsan_threshold = ihsan_threshold
        self._snr_threshold = snr_threshold
        self._max_agents_per_task = max_agents_per_task
        self._agent_timeout = agent_timeout

        # Metrics
        self._executions: int = 0
        self._successes: int = 0
        self._failures: int = 0
        self._total_tokens: int = 0

    def set_activator(self, activator: object) -> None:
        """Inject or update the AgentActivator reference."""
        self._activator = activator

    def set_gateway(self, gateway: object) -> None:
        """Inject or update the InferenceGateway reference."""
        self._gateway = gateway

    async def execute(self, opportunity: object) -> Dict[str, Any]:
        """
        Execute an opportunity using activated PAT agents.

        This method is designed to be passed to
        OpportunityPipeline.set_execution_callback().

        Args:
            opportunity: A PipelineOpportunity with description and context.

        Returns:
            Dict with 'success', 'content', 'agents_used', 'tokens_used'.
        """
        self._executions += 1
        start = time.perf_counter()

        description = getattr(opportunity, "description", "")
        domain = getattr(opportunity, "domain", "unknown")
        opp_id = getattr(opportunity, "id", "unknown")
        context = getattr(opportunity, "context", {})

        logger.info("Executing opportunity %s [%s]: %.80s...", opp_id, domain, description)

        # Select agents
        agents = self._select_agents(description)
        if not agents:
            self._failures += 1
            logger.warning("No agents available for opportunity %s", opp_id)
            return {
                "success": False,
                "error": "No activated agents available",
                "opportunity_id": opp_id,
            }

        # Dispatch to each selected agent
        results: List[ExecutionResult] = []
        for agent in agents[: self._max_agents_per_task]:
            result = await self._dispatch_to_agent(agent, description, context)
            results.append(result)

        # Synthesize results
        synthesis = self._synthesize_results(results)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if synthesis["success"]:
            self._successes += 1
        else:
            self._failures += 1

        synthesis["opportunity_id"] = opp_id
        synthesis["domain"] = domain
        synthesis["elapsed_ms"] = elapsed_ms
        synthesis["agents_used"] = [r.agent_id for r in results]

        logger.info(
            "Opportunity %s executed: success=%s agents=%d tokens=%d %.0fms",
            opp_id,
            synthesis["success"],
            len(results),
            synthesis.get("tokens_used", 0),
            elapsed_ms,
        )
        return synthesis

    def _select_agents(self, description: str) -> list:
        """Select agents from the activator for the given task."""
        if self._activator is None:
            return []

        select_fn = getattr(self._activator, "select_agents_for_task", None)
        if select_fn is None:
            return []

        try:
            return select_fn(description)
        except Exception as e:
            logger.warning("Agent selection failed: %s", e)
            return []

    async def _dispatch_to_agent(
        self,
        agent: object,
        description: str,
        context: Dict[str, Any],
    ) -> ExecutionResult:
        """Dispatch a task to a single activated agent via InferenceGateway."""
        from .agent_activator import ActivationStatus

        agent_id = getattr(agent, "agent_id", "unknown")
        role = getattr(agent, "role", "unknown")
        system_prompt = getattr(agent, "system_prompt", "")

        result = ExecutionResult(agent_id=agent_id, agent_role=role)
        start = time.perf_counter()

        # Mark agent as busy
        if hasattr(agent, "status"):
            agent.status = ActivationStatus.BUSY

        try:
            content = await self._call_gateway(system_prompt, description, context)
            elapsed_ms = (time.perf_counter() - start) * 1000

            result.success = True
            result.content = content
            result.latency_ms = elapsed_ms

            # Update agent metrics
            if hasattr(agent, "tasks_completed"):
                agent.tasks_completed += 1
            if hasattr(agent, "last_active"):
                agent.last_active = time.time()

        except asyncio.TimeoutError:
            result.error = f"Agent {agent_id} timed out after {self._agent_timeout}s"
            result.latency_ms = (time.perf_counter() - start) * 1000
            if hasattr(agent, "tasks_failed"):
                agent.tasks_failed += 1
            logger.warning(result.error)

        except Exception as e:
            result.error = str(e)
            result.latency_ms = (time.perf_counter() - start) * 1000
            if hasattr(agent, "tasks_failed"):
                agent.tasks_failed += 1
            logger.warning("Agent %s execution error: %s", agent_id, e)

        finally:
            # Restore agent to ready state
            if hasattr(agent, "status"):
                agent.status = ActivationStatus.READY

        return result

    async def _call_gateway(
        self,
        system_prompt: str,
        user_message: str,
        context: Dict[str, Any],
    ) -> str:
        """Call InferenceGateway with the agent's system prompt and task."""
        if self._gateway is None:
            return f"[No gateway configured] Task: {user_message[:200]}"

        # Build messages for the gateway
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Try the gateway's infer method
        infer_fn = getattr(self._gateway, "infer", None)
        if infer_fn is None:
            return f"[Gateway has no infer method] Task: {user_message[:200]}"

        try:
            result = await asyncio.wait_for(
                infer_fn(
                    messages=messages,
                    max_tokens=800,
                ),
                timeout=self._agent_timeout,
            )

            # Extract content from various response formats
            if hasattr(result, "content"):
                content = result.content
            elif isinstance(result, dict):
                content = result.get("content", result.get("text", str(result)))
            else:
                content = str(result)

            # Track tokens
            tokens = 0
            if hasattr(result, "usage"):
                usage = result.usage
                tokens = getattr(usage, "total_tokens", 0)
            elif isinstance(result, dict) and "usage" in result:
                tokens = result["usage"].get("total_tokens", 0)

            self._total_tokens += tokens
            return content

        except asyncio.TimeoutError:
            raise
        except Exception as e:
            raise RuntimeError(f"Gateway call failed: {e}") from e

    def _synthesize_results(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """Synthesize multi-agent results into a single response."""
        if not results:
            return {"success": False, "content": "", "error": "No results"}

        successful = [r for r in results if r.success]
        if not successful:
            errors = "; ".join(r.error for r in results if r.error)
            return {"success": False, "content": "", "error": errors}

        # Combine outputs from all successful agents
        parts = []
        total_tokens = 0
        for r in successful:
            role_label = r.agent_role.upper()
            parts.append(f"[{role_label}] {r.content}")
            total_tokens += r.tokens_used

        combined = "\n\n".join(parts)
        self._total_tokens += total_tokens

        return {
            "success": True,
            "content": combined,
            "tokens_used": total_tokens,
            "agent_count": len(successful),
            "failed_count": len(results) - len(successful),
        }

    def metrics(self) -> Dict[str, Any]:
        """Execution metrics summary."""
        return {
            "executions": self._executions,
            "successes": self._successes,
            "failures": self._failures,
            "total_tokens": self._total_tokens,
            "success_rate": (
                self._successes / self._executions
                if self._executions > 0
                else 0.0
            ),
        }
