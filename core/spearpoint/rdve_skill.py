"""
RDVE Skill — Recursive Discovery & Verification Engine (Bridge Adapter)
========================================================================

Registers the Spearpoint orchestration layer as an invocable skill
accessible via the Desktop Bridge's ``invoke_skill`` JSON-RPC method.

Exposed operations:
  research_pattern  — Pattern-aware research using 15 Sci-Reasoning patterns
  reproduce         — Verify/reproduce a claim via AutoEvaluator
  improve           — Generate improvement hypotheses via AutoResearcher
  statistics        — Return orchestrator + bridge statistics

When invoked via the Desktop Bridge, operations flow through the FATE gate,
Rust gate chain, and PCI receipt pipeline. Direct instantiation bypasses
the bridge security surface — callers must validate Ihsan scores themselves.

Standing on Giants:
- Boyd (OODA): Each operation is one OODA cycle
- Li et al. (2025): Sci-Reasoning 15 thinking patterns
- Nygard (2007): Circuit breaker in RecursiveLoop
- Shannon (1948): SNR quality gate on every output

Created: 2026-02-13 | BIZRA RDVE Skill v1.0
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

from .config import SpearpointConfig
from .orchestrator import SpearpointOrchestrator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RDVE Skill Handler
# ---------------------------------------------------------------------------


class RDVESkillHandler:
    """
    Bridge adapter that wraps SpearpointOrchestrator as an invocable skill.

    Registration:
        handler = RDVESkillHandler()
        handler.register(router)   # Registers skill + handler on SkillRouter

    Invocation (via bridge):
        {"jsonrpc": "2.0", "method": "invoke_skill", "params": {
            "skill": "rdve_research",
            "inputs": {
                "operation": "research_pattern",
                "pattern_id": "P01",
                "claim_context": "optimize transformer attention"
            }
        }, "id": 1}
    """

    SKILL_NAME = "rdve_research"
    AGENT_NAME = "rdve-researcher"
    DESCRIPTION = (
        "Recursive Discovery & Verification Engine — "
        "pattern-aware research using 15 Sci-Reasoning thinking patterns"
    )
    TAGS = ["research", "spearpoint", "rdve", "sci-reasoning", "got"]
    VERSION = "1.0.0"

    def __init__(
        self,
        config: Optional[SpearpointConfig] = None,
        orchestrator: Optional[SpearpointOrchestrator] = None,
    ):
        self._config = config or SpearpointConfig()
        self._orchestrator = orchestrator or SpearpointOrchestrator(config=self._config)
        self._invocation_count = 0
        self._created_at = datetime.now(timezone.utc).isoformat()

    @property
    def orchestrator(self) -> SpearpointOrchestrator:
        """Access the underlying orchestrator."""
        return self._orchestrator

    def register(self, router: Any) -> None:
        """Register this skill on a SkillRouter.

        Creates a RegisteredSkill in the registry and wires the async handler.
        """
        from core.skills.registry import (
            RegisteredSkill,
            SkillContext,
            SkillManifest,
            SkillStatus,
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
                "pattern_id",
                "claim",
                "claim_context",
                "top_k",
                "observation",
                "mission_id",
            ],
            outputs=["mission_result"],
            ihsan_floor=UNIFIED_IHSAN_THRESHOLD,
        )

        skill = RegisteredSkill(
            manifest=manifest,
            path="core/spearpoint/rdve_skill.py",
            status=SkillStatus.AVAILABLE,
        )

        # Register in registry
        router.registry._skills[self.SKILL_NAME] = skill

        # Index by tag
        for tag in self.TAGS:
            tag_list = router.registry._by_tag.setdefault(tag, [])
            if self.SKILL_NAME not in tag_list:
                tag_list.append(self.SKILL_NAME)

        # Index by agent
        agent_list = router.registry._by_agent.setdefault(self.AGENT_NAME, [])
        if self.SKILL_NAME not in agent_list:
            agent_list.append(self.SKILL_NAME)

        # Register handler
        router.register_handler(self.AGENT_NAME, self._handle)

        logger.info(f"RDVE skill '{self.SKILL_NAME}' registered on SkillRouter")

    async def _handle(
        self,
        skill: Any,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async handler invoked by SkillRouter.invoke().

        Dispatches to the appropriate orchestrator method based on
        ``inputs["operation"]``.
        """
        operation = inputs.get("operation", "")
        self._invocation_count += 1

        dispatch = {
            "research_pattern": self._op_research_pattern,
            "reproduce": self._op_reproduce,
            "improve": self._op_improve,
            "statistics": self._op_statistics,
        }

        handler = dispatch.get(operation)
        if handler is None:
            return {
                "error": f"Unknown RDVE operation: '{operation}'",
                "available_operations": list(dispatch.keys()),
            }

        return await handler(inputs, context)

    # -- Operation handlers ---------------------------------------------------

    async def _op_research_pattern(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Pattern-aware research using Sci-Reasoning thinking patterns.

        Required inputs:
            pattern_id: str — e.g. "P01" through "P15"
        Optional:
            claim_context: str — additional research context
            top_k: int — max hypotheses to evaluate (default 3)
            mission_id: str — provenance ID
        """
        pattern_id = inputs.get("pattern_id", "")
        if not pattern_id:
            return {"error": "Missing required input: pattern_id"}

        # M-11: Whitelist pattern IDs to P01-P15
        import re

        if not re.fullmatch(r"P(?:0[1-9]|1[0-5])", pattern_id):
            return {
                "error": f"Invalid pattern_id: must be P01-P15, got '{pattern_id[:10]}'"
            }

        claim_context = inputs.get("claim_context", "")
        top_k = max(1, min(int(inputs.get("top_k", 3)), 50))
        mission_id = inputs.get("mission_id", f"rdve_{uuid.uuid4().hex[:12]}")

        start = time.perf_counter()

        result = self._orchestrator.research_pattern(
            pattern_id=pattern_id,
            claim_context=claim_context,
            top_k=top_k,
            mission_id=mission_id,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "operation": "research_pattern",
            "pattern_id": pattern_id,
            "mission": result.to_dict(),
            "elapsed_ms": round(elapsed_ms, 2),
        }

    async def _op_reproduce(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify/reproduce a claim via AutoEvaluator.

        Required inputs:
            claim: str — the claim to verify
        Optional:
            proposed_change: str
            prompt: str
            response: str
            metrics: dict
            mission_id: str
        """
        claim = inputs.get("claim", "")
        if not claim:
            return {"error": "Missing required input: claim"}

        mission_id = inputs.get("mission_id", f"rdve_{uuid.uuid4().hex[:12]}")

        start = time.perf_counter()

        result = self._orchestrator.reproduce(
            claim=claim,
            proposed_change=inputs.get("proposed_change", ""),
            prompt=inputs.get("prompt", ""),
            response=inputs.get("response", ""),
            metrics=inputs.get("metrics"),
            mission_id=mission_id,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "operation": "reproduce",
            "claim": claim[:100],
            "mission": result.to_dict(),
            "elapsed_ms": round(elapsed_ms, 2),
        }

    async def _op_improve(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate improvement hypotheses via AutoResearcher.

        Optional inputs:
            observation: dict — system state snapshot
            top_k: int — max hypotheses (default 3)
            mission_id: str
        """
        observation = inputs.get("observation")
        top_k = max(1, min(int(inputs.get("top_k", 3)), 50))
        mission_id = inputs.get("mission_id", f"rdve_{uuid.uuid4().hex[:12]}")

        start = time.perf_counter()

        result = self._orchestrator.improve(
            observation=observation,
            top_k=top_k,
            mission_id=mission_id,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "operation": "improve",
            "mission": result.to_dict(),
            "elapsed_ms": round(elapsed_ms, 2),
        }

    async def _op_statistics(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return orchestrator + RDVE skill statistics."""
        return {
            "operation": "statistics",
            "rdve": {
                "version": self.VERSION,
                "invocation_count": self._invocation_count,
                "created_at": self._created_at,
                "skill_name": self.SKILL_NAME,
            },
            "orchestrator": self._orchestrator.get_statistics(),
            "mission_history": self._orchestrator.get_mission_history(limit=5),
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_handler: Optional[RDVESkillHandler] = None


def get_rdve_handler(
    config: Optional[SpearpointConfig] = None,
) -> RDVESkillHandler:
    """Get the singleton RDVE skill handler."""
    global _default_handler
    if _default_handler is None:
        _default_handler = RDVESkillHandler(config=config)
    return _default_handler


def register_rdve_skill(router: Any) -> RDVESkillHandler:
    """Register the RDVE skill on a SkillRouter. Returns the handler."""
    handler = get_rdve_handler()
    handler.register(router)
    return handler


__all__ = [
    "RDVESkillHandler",
    "get_rdve_handler",
    "register_rdve_skill",
]
