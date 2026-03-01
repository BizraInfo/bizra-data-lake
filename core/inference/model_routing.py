"""Model routing configuration — shared by core/ and scripts/ layers.

Extracted from scripts/node0_activate.py to eliminate the layering
violation where core/inference/auto_model_router.py imported from the
scripts layer.

Standing on Giants: Shannon (capacity planning) . Boyd (OODA pre-staging)
"""

from __future__ import annotations

from typing import Any, Dict

# ── PAT agent purpose → fleet role mapping ────────────────────────────

PURPOSE_TO_ROLE: Dict[str, str] = {
    "reasoning": "reasoner",
    "reasoning_large": "reasoner_large",
    "thinking": "thinker",
    "general": "general",
    "creative": "creative",
    "agentic": "planner",
    "nano": "nano",
    "vision": "vision",
    "voice": "voice",
}

# ── Default model routing: role → LM Studio model identifier ─────────

DEFAULT_MODEL_ROUTING: Dict[str, str] = {
    "planner": "agentflow-planner-7b-i1",
    "reasoner": "qwen/qwen3-4b-thinking-2507",  # 4B fits RTX 4090 alongside other models
    "reasoner_large": "deepseek/deepseek-r1-0528-qwen3-8b",  # Promoted from default reasoner
    "thinker": "qwen/qwen3-4b-thinking-2507",
    "general": "liquid/lfm2.5-1.2b",
    "creative": "chuanli11_-_llama-3.2-3b-instruct-uncensored",
    "nano": "qwen2.5-0.5b-instruct",
    "vision": "qwen/qwen3-vl-4b",
    "vision_large": "qwen/qwen3-vl-8b",
    "voice": "deephat-v1-7b",
    "embedding": "text-embedding-nomic-embed-text-v1.5",
}

# ── PAT agent definitions (model_purpose field only) ──────────────────
# Full PAT_AGENTS dict lives in node0_activate.py; this subset is enough
# for model resolution without pulling in the entire scripts layer.

_PAT_AGENT_PURPOSE: Dict[str, str] = {
    "strategist": "thinking",
    "researcher": "reasoning",
    "analyst": "thinking",
    "creator": "creative",
    "executor": "agentic",
    "guardian": "reasoning_large",
    "coordinator": "reasoning",
}


def resolve_model_for_agent(agent_id: str, config: Dict[str, Any]) -> str:
    """Resolve the correct LM Studio model identifier for a PAT agent.

    ``config`` may contain a ``model_routing`` key that overrides the
    defaults.  Falls back gracefully to the ``reasoner`` role if the
    agent or purpose is unknown.
    """
    routing = config.get("model_routing", DEFAULT_MODEL_ROUTING)
    purpose = _PAT_AGENT_PURPOSE.get(agent_id, "reasoning")
    role = PURPOSE_TO_ROLE.get(purpose, "reasoner")
    return routing.get(
        role, routing.get("reasoner", "deepseek/deepseek-r1-0528-qwen3-8b")
    )


__all__ = [
    "DEFAULT_MODEL_ROUTING",
    "PURPOSE_TO_ROLE",
    "resolve_model_for_agent",
]
