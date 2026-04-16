"""
PAT Runtime Daemon — Persistent Agent Loop
============================================
Instantiates PAT-7 agents from GenesisState, activates them,
and runs a mission-processing loop. Each action emits a receipt.

Standing on Giants: Actor Model (Hewitt) + Event Loop (Libevent)
Constitutional Constraint: Ihsan >= 0.95
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MissionRequest:
    """A mission for a PAT agent to process."""

    mission_id: str
    content: str
    requester_id: str = ""
    target_role: Optional[str] = None
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class MissionResult:
    """Result of a PAT agent processing a mission."""

    mission_id: str
    agent_id: str
    agent_role: str
    success: bool
    response: str = ""
    receipt_hash: str = ""
    elapsed_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "success": self.success,
            "response": self.response,
            "receipt_hash": self.receipt_hash,
            "elapsed_ms": self.elapsed_ms,
            "timestamp": self.timestamp,
        }


# Type for the query function the runtime uses
QueryFn = Callable[[str, Optional[Dict[str, Any]]], Coroutine[Any, Any, str]]


class PATRuntime:
    """
    Persistent runtime for PAT-7 agents.

    Lifecycle:
        1. Load agents from GenesisState (or mint fresh)
        2. Activate all agents (DORMANT → ACTIVE)
        3. Run async mission loop
        4. Each mission is routed to the best agent by role
        5. Each result emits a BLAKE3-chained receipt

    Usage:
        runtime = PATRuntime(agents=genesis.pat_team, query_fn=sovereign.query)
        await runtime.start()
        result = await runtime.submit_mission(MissionRequest(...))
        await runtime.stop()
    """

    def __init__(
        self,
        agents: Optional[List[Any]] = None,
        query_fn: Optional[QueryFn] = None,
        receipt_dir: Optional[Path] = None,
        fate_boundary: Optional[Any] = None,
    ):
        self._agents: List[Any] = agents or []
        self._agent_map: Dict[str, Any] = {}  # agent_id → agent
        self._role_map: Dict[str, List[Any]] = {}  # role → [agents]
        self._query_fn = query_fn
        self._receipt_dir = receipt_dir
        self._fate_boundary = fate_boundary

        self._mission_queue: asyncio.Queue[MissionRequest] = asyncio.Queue()
        self._running = False
        self._loop_task: Optional[asyncio.Task[Any]] = None
        self._results: Dict[str, MissionResult] = {}
        self._receipt_count = 0
        self._prev_hash = "0" * 64

        # Metrics
        self._missions_processed = 0
        self._missions_failed = 0
        self._total_elapsed_ms = 0.0

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    @property
    def active_count(self) -> int:
        return sum(
            1
            for a in self._agents
            if hasattr(a, "status") and str(a.status).endswith("ACTIVE")
        )

    @property
    def is_running(self) -> bool:
        return self._running

    def _index_agents(self) -> None:
        """Build lookup maps for agents."""
        self._agent_map.clear()
        self._role_map.clear()
        for agent in self._agents:
            aid = getattr(agent, "agent_id", None) or str(id(agent))
            self._agent_map[aid] = agent
            role = getattr(agent, "agent_type", None)
            if role is not None:
                role_key = role.value if hasattr(role, "value") else str(role)
                self._role_map.setdefault(role_key, []).append(agent)

    def _activate_all(self) -> int:
        """Activate all dormant agents, return count activated."""
        activated = 0
        for agent in self._agents:
            if hasattr(agent, "activate") and hasattr(agent, "status"):
                status_str = str(agent.status)
                if "ACTIVE" not in status_str:
                    if agent.activate():
                        activated += 1
        return activated

    def _select_agent(self, request: MissionRequest) -> Optional[Any]:
        """Select the best agent for a mission."""
        if request.target_role:
            # Direct role targeting
            candidates = self._role_map.get(request.target_role, [])
            if candidates:
                # Pick least busy
                active = [
                    a
                    for a in candidates
                    if hasattr(a, "status") and "ACTIVE" in str(a.status)
                ]
                return active[0] if active else candidates[0]

        # Keyword-based routing (use select_pat_agent if available)
        try:
            from core.sovereign.user_context import select_pat_agent

            role = select_pat_agent(request.content, self._agents)
            if role:
                candidates = self._role_map.get(role, [])
                if candidates:
                    return candidates[0]
        except ImportError:
            pass

        # Fallback: round-robin across active agents
        active = [
            a
            for a in self._agents
            if hasattr(a, "status") and "ACTIVE" in str(a.status)
        ]
        if active:
            idx = self._missions_processed % len(active)
            return active[idx]

        return self._agents[0] if self._agents else None

    def _emit_receipt(self, result: MissionResult) -> str:
        """Emit a BLAKE3-chained receipt for a mission result."""
        try:
            import blake3

            content = json.dumps(result.to_dict(), sort_keys=True).encode()
            receipt_hash = blake3.blake3(self._prev_hash.encode() + content).hexdigest()
            self._prev_hash = receipt_hash
            self._receipt_count += 1

            if self._receipt_dir:
                self._receipt_dir.mkdir(parents=True, exist_ok=True)
                receipt_path = (
                    self._receipt_dir / f"pat_mission_{result.mission_id}.json"
                )
                receipt_doc = {
                    "event": "pat_mission_complete",
                    "receipt_hash": receipt_hash,
                    "prev_hash": self._prev_hash,
                    **result.to_dict(),
                }
                receipt_path.write_text(json.dumps(receipt_doc, indent=2))

            return receipt_hash
        except ImportError:
            import hashlib

            content = json.dumps(result.to_dict(), sort_keys=True).encode()
            receipt_hash = hashlib.sha256(
                self._prev_hash.encode() + content
            ).hexdigest()
            self._prev_hash = receipt_hash
            return receipt_hash

    async def _process_mission(self, request: MissionRequest) -> MissionResult:
        """Process a single mission through the agent pipeline."""
        t0 = time.monotonic()
        agent = self._select_agent(request)

        if agent is None:
            return MissionResult(
                mission_id=request.mission_id,
                agent_id="none",
                agent_role="none",
                success=False,
                response="No agent available",
            )

        agent_id = getattr(agent, "agent_id", "unknown")
        agent_role = getattr(agent, "agent_type", "unknown")
        if hasattr(agent_role, "value"):
            agent_role = agent_role.value

        try:
            # FATE boundary check (if wired)
            if self._fate_boundary and hasattr(self._fate_boundary, "check_crossing"):
                fate_ok = await self._fate_boundary.check_crossing(
                    agent_id=agent_id,
                    content=request.content,
                    direction="pat_to_urp",
                )
                if not fate_ok:
                    return MissionResult(
                        mission_id=request.mission_id,
                        agent_id=agent_id,
                        agent_role=str(agent_role),
                        success=False,
                        response="FATE gate blocked this crossing",
                        elapsed_ms=(time.monotonic() - t0) * 1000,
                    )

            # Execute through query function
            response = ""
            if self._query_fn:
                context = {
                    "agent_id": agent_id,
                    "agent_role": str(agent_role),
                    "mission_id": request.mission_id,
                    "priority": request.priority,
                }
                response = await self._query_fn(request.content, context)

            # Record task on agent
            if hasattr(agent, "record_task_completion"):
                agent.record_task_completion(success=True)

            elapsed = (time.monotonic() - t0) * 1000
            result = MissionResult(
                mission_id=request.mission_id,
                agent_id=agent_id,
                agent_role=str(agent_role),
                success=True,
                response=response,
                elapsed_ms=elapsed,
            )

            # Emit receipt
            result.receipt_hash = self._emit_receipt(result)
            return result

        except (asyncio.CancelledError, RuntimeError, OSError, ValueError) as e:
            elapsed = (time.monotonic() - t0) * 1000
            if hasattr(agent, "record_task_completion"):
                agent.record_task_completion(success=False)
            return MissionResult(
                mission_id=request.mission_id,
                agent_id=agent_id,
                agent_role=str(agent_role),
                success=False,
                response=f"Error: {e}",
                elapsed_ms=elapsed,
            )

    async def _mission_loop(self) -> None:
        """Main mission processing loop."""
        logger.info(f"PAT runtime loop started ({self.active_count} active agents)")
        while self._running:
            try:
                request = await asyncio.wait_for(self._mission_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            result = await self._process_mission(request)
            self._results[result.mission_id] = result
            self._missions_processed += 1
            if not result.success:
                self._missions_failed += 1
            self._total_elapsed_ms += result.elapsed_ms
            logger.debug(
                f"Mission {result.mission_id} → {result.agent_role}: "
                f"{'OK' if result.success else 'FAIL'} ({result.elapsed_ms:.0f}ms)"
            )

    async def start(self) -> None:
        """Start the PAT runtime."""
        if self._running:
            return

        self._index_agents()
        activated = self._activate_all()
        logger.info(
            f"PAT Runtime: {self.agent_count} agents loaded, "
            f"{activated} activated, {self.active_count} active"
        )

        self._running = True
        self._loop_task = asyncio.create_task(
            self._mission_loop(), name="pat_runtime_loop"
        )

    async def stop(self) -> None:
        """Stop the PAT runtime."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info(
            f"PAT Runtime stopped: {self._missions_processed} missions, "
            f"{self._missions_failed} failed"
        )

    async def submit_mission(self, request: MissionRequest) -> MissionResult:
        """Submit a mission and wait for result."""
        await self._mission_queue.put(request)
        # Wait for result (with timeout)
        for _ in range(300):  # 30s max
            await asyncio.sleep(0.1)
            if request.mission_id in self._results:
                return self._results[request.mission_id]
        return MissionResult(
            mission_id=request.mission_id,
            agent_id="timeout",
            agent_role="none",
            success=False,
            response="Mission timed out",
        )

    def submit_fire_and_forget(self, request: MissionRequest) -> None:
        """Submit a mission without waiting for result."""
        self._mission_queue.put_nowait(request)

    def get_status(self) -> Dict[str, Any]:
        """Get runtime status."""
        return {
            "running": self._running,
            "agents_total": self.agent_count,
            "agents_active": self.active_count,
            "missions_processed": self._missions_processed,
            "missions_failed": self._missions_failed,
            "missions_queued": self._mission_queue.qsize(),
            "avg_latency_ms": (
                self._total_elapsed_ms / max(self._missions_processed, 1)
            ),
            "receipt_count": self._receipt_count,
        }
