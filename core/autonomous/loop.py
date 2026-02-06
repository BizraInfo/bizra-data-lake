"""
Sovereign Loop — Autonomous Reasoning Cycle

The closed-loop autonomous reasoning system:
OBSERVE → ORIENT → REASON → SYNTHESIZE → ACT → REFLECT

Standing on Giants:
- Boyd (OODA Loop)
- Shannon (Information Theory)
- Besta (Graph-of-Thoughts)
- Anthropic (Constitutional AI)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import time

from core.autonomous import SNR_THRESHOLDS, CONSTITUTIONAL_CONSTRAINTS
from core.autonomous.nodes import (
    ReasoningGraph,
    ReasoningNode,
    ReasoningPath,
    NodeType,
    NodeState,
)
from core.autonomous.giants import GiantsProtocol, Giant, ProvenanceRecord

logger = logging.getLogger(__name__)


class LoopPhase(str, Enum):
    """Phases of the Sovereign Loop."""
    OBSERVE = "observe"       # Input processing
    ORIENT = "orient"         # Context establishment
    REASON = "reason"         # Core inference
    SYNTHESIZE = "synthesize" # Integration
    ACT = "act"              # Output generation
    REFLECT = "reflect"       # Meta-cognition


class LoopState(str, Enum):
    """State of the loop execution."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PhaseResult:
    """Result of a single phase execution."""
    phase: LoopPhase
    success: bool
    content: str
    snr_score: float
    ihsan_score: float
    duration_ms: float
    node_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "success": self.success,
            "content_preview": self.content[:100] if self.content else "",
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
            "duration_ms": self.duration_ms,
            "node_id": self.node_id,
        }


@dataclass
class LoopExecution:
    """Complete execution of a sovereign loop."""
    id: str
    input_content: str
    output_content: str
    phases: List[PhaseResult]
    best_path: Optional[ReasoningPath] = None
    provenance: Optional[ProvenanceRecord] = None
    total_duration_ms: float = 0.0
    loop_count: int = 0
    backtrack_count: int = 0
    state: LoopState = LoopState.COMPLETED

    @property
    def final_snr(self) -> float:
        if not self.phases:
            return 0.0
        return self.phases[-1].snr_score

    @property
    def final_ihsan(self) -> float:
        if not self.phases:
            return 0.0
        return self.phases[-1].ihsan_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "input_preview": self.input_content[:100],
            "output_preview": self.output_content[:200],
            "phases": [p.to_dict() for p in self.phases],
            "best_path": self.best_path.to_dict() if self.best_path else None,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "total_duration_ms": self.total_duration_ms,
            "loop_count": self.loop_count,
            "backtrack_count": self.backtrack_count,
            "final_snr": self.final_snr,
            "final_ihsan": self.final_ihsan,
            "state": self.state.value,
        }


class SovereignLoop:
    """
    The Sovereign Autonomous Reasoning Loop.

    Implements a closed-loop reasoning cycle with:
    - Phase-based execution (OODA-inspired)
    - Graph-of-Thoughts reasoning
    - SNR optimization at every step
    - Constitutional validation
    - Automatic backtracking on quality degradation
    - Meta-cognitive reflection
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        max_loops: int = 3,
        max_backtrack: int = 5,
    ):
        self.llm_fn = llm_fn
        self.max_loops = max_loops
        self.max_backtrack = max_backtrack

        # Components
        self._graph: Optional[ReasoningGraph] = None
        self._giants = GiantsProtocol()

        # State
        self._current_phase = LoopPhase.OBSERVE
        self._loop_count = 0
        self._backtrack_count = 0
        self._executions: List[LoopExecution] = []

        # Thresholds
        self._snr_thresholds = SNR_THRESHOLDS
        self._constitutional = CONSTITUTIONAL_CONSTRAINTS

    # =========================================================================
    # PHASE IMPLEMENTATIONS
    # =========================================================================

    async def _observe(self, content: str, context: Dict[str, Any]) -> PhaseResult:
        """
        OBSERVE Phase: Process raw input.

        Shannon: Maximize information extraction
        Goal: Transform input into high-SNR observations
        """
        start = time.time()

        # Invoke Shannon SNR technique
        snr, _ = self._giants.invoke(Giant.SHANNON, "snr", content)

        # Create observation node
        node = self._graph.add_node(
            content=content,
            node_type=NodeType.OBSERVATION,
            technique="snr",
            giant="shannon",
            metadata={"phase": "observe", "context_keys": list(context.keys())},
        )

        duration = (time.time() - start) * 1000

        return PhaseResult(
            phase=LoopPhase.OBSERVE,
            success=snr >= self._snr_thresholds["observation"],
            content=content,
            snr_score=node.snr_score,
            ihsan_score=node.ihsan_score,
            duration_ms=duration,
            node_id=node.id,
        )

    async def _orient(
        self,
        observation: PhaseResult,
        context: Dict[str, Any],
    ) -> PhaseResult:
        """
        ORIENT Phase: Establish context.

        Vaswani: Apply attention to weight context
        Goal: Situate observation in relevant context
        """
        start = time.time()

        # Gather context elements
        context_items = list(context.keys())

        # Apply attention weighting
        if context_items:
            weighted, _ = self._giants.invoke(
                Giant.VASWANI,
                "attention",
                observation.content,
                context_items,
                list(context.values()),
            )
            top_context = weighted[:3] if weighted else []
        else:
            top_context = []

        # Create orientation content
        orientation_content = f"Context: {observation.content[:200]}"
        if top_context:
            context_summary = ", ".join(str(c[0])[:50] for c in top_context)
            orientation_content += f". Relevant: {context_summary}"

        # Create orientation node
        node = self._graph.add_node(
            content=orientation_content,
            node_type=NodeType.ORIENTATION,
            parent_ids={observation.node_id} if observation.node_id else None,
            technique="attention",
            giant="vaswani",
            metadata={"phase": "orient", "top_context": [str(c[0])[:50] for c in top_context]},
        )

        duration = (time.time() - start) * 1000

        return PhaseResult(
            phase=LoopPhase.ORIENT,
            success=node.snr_score >= self._snr_thresholds["orientation"],
            content=orientation_content,
            snr_score=node.snr_score,
            ihsan_score=node.ihsan_score,
            duration_ms=duration,
            node_id=node.id,
        )

    async def _reason(
        self,
        orientation: PhaseResult,
        context: Dict[str, Any],
    ) -> PhaseResult:
        """
        REASON Phase: Core inference.

        Besta: Graph-of-Thoughts reasoning
        Goal: Generate high-quality inferences
        """
        start = time.time()

        # Generate reasoning content
        if self.llm_fn:
            prompt = f"""Given this context:
{orientation.content}

Provide a clear, structured analysis with key insights.
Focus on the most relevant patterns and implications."""

            reasoning_content = self.llm_fn(prompt)
        else:
            # Heuristic reasoning without LLM
            reasoning_content = f"Analysis of: {orientation.content[:100]}. Key patterns identified through structured examination."

        # Create reasoning node
        node = self._graph.add_node(
            content=reasoning_content,
            node_type=NodeType.ANALYSIS,
            parent_ids={orientation.node_id} if orientation.node_id else None,
            technique="got",
            giant="besta",
            metadata={"phase": "reason"},
        )

        # Check if backtracking needed
        should_bt, reason = self._graph.should_backtrack(node.id)
        if should_bt and self._backtrack_count < self.max_backtrack:
            logger.info(f"Backtracking: {reason}")
            backtrack_node = self._graph.backtrack(node.id, reason)
            self._backtrack_count += 1

            if backtrack_node:
                node = backtrack_node

        duration = (time.time() - start) * 1000

        return PhaseResult(
            phase=LoopPhase.REASON,
            success=node.snr_score >= self._snr_thresholds["reasoning"],
            content=reasoning_content,
            snr_score=node.snr_score,
            ihsan_score=node.ihsan_score,
            duration_ms=duration,
            node_id=node.id,
            metadata={"backtracked": should_bt},
        )

    async def _synthesize(
        self,
        phases: List[PhaseResult],
        context: Dict[str, Any],
    ) -> PhaseResult:
        """
        SYNTHESIZE Phase: Integration.

        Besta: Multi-path synthesis
        Goal: Integrate all reasoning paths
        """
        start = time.time()

        # Gather all node IDs from phases
        node_ids = {p.node_id for p in phases if p.node_id}

        # Generate synthesis content
        if self.llm_fn:
            phase_summaries = "\n".join([
                f"- {p.phase.value}: {p.content[:100]}..."
                for p in phases if p.content
            ])

            prompt = f"""Synthesize these insights into a coherent understanding:

{phase_summaries}

Provide an integrated conclusion that captures the key insights."""

            synthesis_content = self.llm_fn(prompt)
        else:
            synthesis_content = f"Synthesis of {len(phases)} reasoning phases. Integrated conclusion based on observation, orientation, and analysis."

        # Create synthesis node
        synthesis_node = self._graph.synthesize(
            node_ids=node_ids,
            synthesis_content=synthesis_content,
            target_type=NodeType.SYNTHESIS,
        )

        if not synthesis_node:
            synthesis_node = self._graph.add_node(
                content=synthesis_content,
                node_type=NodeType.SYNTHESIS,
                parent_ids=node_ids,
                technique="synthesis",
                giant="besta",
            )

        duration = (time.time() - start) * 1000

        return PhaseResult(
            phase=LoopPhase.SYNTHESIZE,
            success=synthesis_node.snr_score >= self._snr_thresholds["synthesis"],
            content=synthesis_content,
            snr_score=synthesis_node.snr_score,
            ihsan_score=synthesis_node.ihsan_score,
            duration_ms=duration,
            node_id=synthesis_node.id,
        )

    async def _act(
        self,
        synthesis: PhaseResult,
        context: Dict[str, Any],
    ) -> PhaseResult:
        """
        ACT Phase: Output generation.

        Anthropic: Constitutional validation
        Goal: Generate constitutionally-valid output
        """
        start = time.time()

        # Generate action content
        if self.llm_fn:
            prompt = f"""Based on this synthesis:
{synthesis.content}

Provide a clear, actionable response that:
1. Addresses the original query
2. Is helpful and accurate
3. Maintains high quality standards"""

            action_content = self.llm_fn(prompt)
        else:
            action_content = synthesis.content

        # Constitutional validation
        validation, _ = self._giants.invoke(
            Giant.ANTHROPIC,
            "constitutional",
            action_content,
        )

        # Ihsān scoring
        ihsan, _ = self._giants.invoke(
            Giant.ANTHROPIC,
            "ihsan",
            action_content,
        )

        # Create action node
        node = self._graph.add_node(
            content=action_content,
            node_type=NodeType.CONCLUSION,
            parent_ids={synthesis.node_id} if synthesis.node_id else None,
            technique="constitutional",
            giant="anthropic",
            metadata={
                "phase": "act",
                "validation": validation,
            },
        )

        # Override scores with constitutional results
        node.ihsan_score = ihsan
        node.snr_score = max(node.snr_score, ihsan)  # Ihsān implies high SNR

        duration = (time.time() - start) * 1000

        success = (
            validation.get("passed", False) and
            node.ihsan_score >= self._constitutional["ihsan_threshold"]
        )

        return PhaseResult(
            phase=LoopPhase.ACT,
            success=success,
            content=action_content,
            snr_score=node.snr_score,
            ihsan_score=node.ihsan_score,
            duration_ms=duration,
            node_id=node.id,
            metadata={"validation": validation},
        )

    async def _reflect(
        self,
        phases: List[PhaseResult],
        context: Dict[str, Any],
    ) -> PhaseResult:
        """
        REFLECT Phase: Meta-cognition.

        Goal: Assess loop quality and determine if re-iteration needed
        """
        start = time.time()

        # Calculate overall quality
        avg_snr = sum(p.snr_score for p in phases) / len(phases)
        avg_ihsan = sum(p.ihsan_score for p in phases) / len(phases)

        # Reflection content
        reflection_content = f"""Meta-cognitive assessment:
- Phases completed: {len(phases)}
- Average SNR: {avg_snr:.3f}
- Average Ihsān: {avg_ihsan:.3f}
- Backtrack count: {self._backtrack_count}
- Quality threshold: {self._snr_thresholds['reflection']:.3f}"""

        # Create reflection node
        node = self._graph.add_node(
            content=reflection_content,
            node_type=NodeType.META,
            parent_ids={phases[-1].node_id} if phases and phases[-1].node_id else None,
            technique="reflection",
            giant="anthropic",
            metadata={
                "phase": "reflect",
                "avg_snr": avg_snr,
                "avg_ihsan": avg_ihsan,
            },
        )

        duration = (time.time() - start) * 1000

        # Determine if loop should continue
        should_continue = (
            avg_snr < self._snr_thresholds["reflection"] and
            self._loop_count < self.max_loops
        )

        return PhaseResult(
            phase=LoopPhase.REFLECT,
            success=not should_continue,  # Success means we're done
            content=reflection_content,
            snr_score=node.snr_score,
            ihsan_score=node.ihsan_score,
            duration_ms=duration,
            node_id=node.id,
            metadata={"should_continue": should_continue},
        )

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    async def execute(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> LoopExecution:
        """
        Execute the sovereign loop.

        Returns a complete execution record with provenance.
        """
        import uuid as uuid_module
        execution_id = f"exec_{uuid_module.uuid4().hex[:8]}"
        context = context or {}
        start_time = time.time()

        # Initialize fresh graph
        self._graph = ReasoningGraph()
        self._loop_count = 0
        self._backtrack_count = 0

        all_phases: List[PhaseResult] = []
        output_content = ""

        try:
            while self._loop_count < self.max_loops:
                self._loop_count += 1
                logger.info(f"Sovereign Loop iteration {self._loop_count}")

                # Execute phases
                observe_result = await self._observe(content, context)
                all_phases.append(observe_result)

                orient_result = await self._orient(observe_result, context)
                all_phases.append(orient_result)

                reason_result = await self._reason(orient_result, context)
                all_phases.append(reason_result)

                loop_phases = [observe_result, orient_result, reason_result]

                synthesize_result = await self._synthesize(loop_phases, context)
                all_phases.append(synthesize_result)

                act_result = await self._act(synthesize_result, context)
                all_phases.append(act_result)
                output_content = act_result.content

                reflect_result = await self._reflect(all_phases, context)
                all_phases.append(reflect_result)

                # Check if we should continue
                if not reflect_result.metadata.get("should_continue", False):
                    break

                # Update content for next iteration
                content = f"Refining: {act_result.content}"

            # Get best path
            best_path = self._graph.find_best_path()

            # Create provenance
            provenance = self._giants.create_provenance(
                output=output_content,
                snr_score=all_phases[-2].snr_score if len(all_phases) >= 2 else 0.0,
                ihsan_score=all_phases[-2].ihsan_score if len(all_phases) >= 2 else 0.0,
                reasoning_path=[p.node_id for p in all_phases if p.node_id],
            )

            total_duration = (time.time() - start_time) * 1000

            execution = LoopExecution(
                id=execution_id,
                input_content=content,
                output_content=output_content,
                phases=all_phases,
                best_path=best_path,
                provenance=provenance,
                total_duration_ms=total_duration,
                loop_count=self._loop_count,
                backtrack_count=self._backtrack_count,
                state=LoopState.COMPLETED,
            )

        except Exception as e:
            logger.error(f"Sovereign loop failed: {e}")
            execution = LoopExecution(
                id=execution_id,
                input_content=content,
                output_content=f"Error: {str(e)}",
                phases=all_phases,
                total_duration_ms=(time.time() - start_time) * 1000,
                loop_count=self._loop_count,
                backtrack_count=self._backtrack_count,
                state=LoopState.FAILED,
            )

        self._executions.append(execution)
        return execution

    def get_graph(self) -> Optional[ReasoningGraph]:
        """Get the current reasoning graph."""
        return self._graph

    def get_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent executions."""
        return [e.to_dict() for e in self._executions[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        """Get loop statistics."""
        if not self._executions:
            return {"total_executions": 0}

        return {
            "total_executions": len(self._executions),
            "avg_loop_count": sum(e.loop_count for e in self._executions) / len(self._executions),
            "avg_backtrack_count": sum(e.backtrack_count for e in self._executions) / len(self._executions),
            "avg_snr": sum(e.final_snr for e in self._executions) / len(self._executions),
            "avg_ihsan": sum(e.final_ihsan for e in self._executions) / len(self._executions),
            "success_rate": sum(1 for e in self._executions if e.state == LoopState.COMPLETED) / len(self._executions),
        }
