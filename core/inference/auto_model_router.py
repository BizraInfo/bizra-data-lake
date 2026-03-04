"""AutoModelRouter: pre-loads optimal models into VRAM before inference.

Bridges the mission pipeline (node0_activate.py) with LM Studio's model
management API so that cold-start latency is absorbed BEFORE agent calls
begin.  Optionally consumes EqualizerAgent commands to swap models under
load (ESCALATE / HALT / RESUME).

Standing on Giants: Shannon (capacity planning) . Boyd (OODA pre-staging)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("AutoModelRouter")

# ── Escalation map: role -> (current_model, larger_variant) ──────────
_ESCALATION_MAP: Dict[str, tuple[str, str]] = {
    "reasoner": (
        "deepseek/deepseek-r1-0528-qwen3-8b",
        "mistralai/ministral-3-14b-reasoning",
    ),
    "thinker": (
        "qwen/qwen3-4b-thinking-2507",
        "deepseek/deepseek-r1-0528-qwen3-8b",
    ),
    "general": (
        "liquid/lfm2.5-1.2b",
        "qwen2.5-0.5b-instruct",
    ),
    "creative": (
        "chuanli11_-_llama-3.2-3b-instruct-uncensored",
        "deepseek/deepseek-r1-0528-qwen3-8b",
    ),
    "planner": (
        "agentflow-planner-7b-i1",
        "mistralai/ministral-3-14b-reasoning",
    ),
}


class AutoModelRouter:
    """Pre-loads optimal models into VRAM before inference starts.

    Usage::

        router = AutoModelRouter(base_url, token, equalizer=EqualizerAgent())
        fleet = await router.preload_mission_fleet(agent_ids, config)
        # ... run agent calls ...
        action = await router.check_equalizer(ihsan, backlog, presence)
    """

    def __init__(
        self,
        base_url: str,
        token: str = "",
        *,
        equalizer: Any = None,
        load_timeout: float = 180.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._equalizer = equalizer
        self._load_timeout = load_timeout

        self._loaded_models: set[str] = set()
        self._unloaded_by_halt: set[str] = set()
        self._load_lock = asyncio.Lock()

    # ── Core public API ──────────────────────────────────────────────

    async def ensure_model_loaded(self, model_id: str) -> bool:
        """Pre-load a single model into VRAM.

        Returns True if the model is ready (already loaded or successfully
        loaded), False on failure (graceful degradation).
        """
        if model_id in self._loaded_models:
            return True

        async with self._load_lock:
            # Double-check after acquiring lock
            if model_id in self._loaded_models:
                return True
            return await self._load_model(model_id)

    async def preload_mission_fleet(
        self,
        agent_ids: list[str],
        config: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Resolve all agent->model mappings, deduplicate, and pre-load.

        Returns ``{model_id: loaded_ok}`` status dict.
        """
        from core.inference.model_routing import (
            resolve_model_for_agent as _resolve_model_for_agent,
        )

        # Resolve unique models needed
        model_for_agent: Dict[str, str] = {}
        for aid in agent_ids:
            model_for_agent[aid] = _resolve_model_for_agent(aid, config)

        unique_models = set(model_for_agent.values())

        # Refresh what's already in VRAM
        try:
            self._loaded_models = await self._get_loaded_models()
        except Exception:
            pass  # proceed with stale cache

        status: Dict[str, bool] = {}
        for model_id in unique_models:
            ok = await self.ensure_model_loaded(model_id)
            status[model_id] = ok

        return status

    async def check_equalizer(
        self,
        ihsan_score: float,
        backlog: int,
        presence: int,
    ) -> Optional[str]:
        """Feed state to EqualizerAgent, consume its command, and act.

        Returns a human-readable action string, or None if no action taken.
        """
        if self._equalizer is None:
            return None

        try:
            self._equalizer.observe(
                layer=0,
                ihsan_score=ihsan_score,
                backlog=backlog,
                presence=presence,
            )
            cmd = self._equalizer.next_command()
        except Exception as exc:
            logger.debug("Equalizer observation failed: %s", exc)
            return None

        if cmd is None:
            return None

        from core.sovereign.equalizer_agent import EqualizerCommandKind

        kind = cmd.kind

        if kind == EqualizerCommandKind.ESCALATE:
            return await self._handle_escalate()

        if kind == EqualizerCommandKind.HALT:
            return await self._handle_halt()

        if kind == EqualizerCommandKind.RESUME:
            return await self._handle_resume()

        # ACCELERATE — no model change needed
        return None

    # ── Internal helpers ─────────────────────────────────────────────

    async def _get_loaded_models(self) -> set[str]:
        """GET /api/v1/models — return set of currently loaded model IDs."""
        headers = self._auth_headers()
        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            resp = await client.get(f"{self._base_url}/api/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                # Native API: {"models": [...]} with "key" and "loaded_instances"
                models = data.get("models", data.get("data", []))
                return {
                    m.get("key", m.get("id", ""))
                    for m in models
                    if m.get("loaded_instances") or m.get("loaded")
                }
            # Fall back to /v1/models (OpenAI-compat — no loaded field)
            resp = await client.get(f"{self._base_url}/v1/models")
            if resp.status_code == 200:
                # OpenAI-compat has no "loaded" field — assume all available
                # models COULD be loaded; caller must verify via inference
                return {
                    m["id"]
                    for m in resp.json().get("data", [])
                }
        return set()

    async def _load_model(
        self,
        model_id: str,
        context_length: Optional[int] = None,
    ) -> bool:
        """POST /api/v1/models/load with 1 retry."""
        payload: Dict[str, Any] = {"model": model_id}
        if context_length is not None:
            payload["context_length"] = context_length

        headers = self._auth_headers()

        for attempt in range(2):
            try:
                async with httpx.AsyncClient(
                    headers=headers,
                    timeout=self._load_timeout,
                ) as client:
                    resp = await client.post(
                        f"{self._base_url}/api/v1/models/load",
                        json=payload,
                    )
                    if resp.status_code == 200:
                        self._loaded_models.add(model_id)
                        self._unloaded_by_halt.discard(model_id)
                        logger.info("Model loaded: %s", model_id)
                        return True
                    logger.warning(
                        "Model load failed (attempt %d): %s -> HTTP %d",
                        attempt + 1,
                        model_id,
                        resp.status_code,
                    )
            except Exception as exc:
                logger.warning(
                    "Model load error (attempt %d): %s -> %s",
                    attempt + 1,
                    model_id,
                    exc,
                )

            if attempt == 0:
                await asyncio.sleep(10)

        return False

    async def _unload_model(self, model_id: str) -> bool:
        """POST /api/v1/models/unload."""
        headers = self._auth_headers()
        try:
            async with httpx.AsyncClient(
                headers=headers,
                timeout=30.0,
            ) as client:
                resp = await client.post(
                    f"{self._base_url}/api/v1/models/unload",
                    json={"model": model_id},
                )
                if resp.status_code == 200:
                    self._loaded_models.discard(model_id)
                    logger.info("Model unloaded: %s", model_id)
                    return True
        except Exception as exc:
            logger.warning("Model unload error: %s -> %s", model_id, exc)
        return False

    # ── Equalizer command handlers ───────────────────────────────────

    async def _handle_escalate(self) -> str:
        """ESCALATE: load a larger model variant for active roles."""
        loaded_any = False
        for _role, (current, larger) in _ESCALATION_MAP.items():
            if current in self._loaded_models and larger not in self._loaded_models:
                ok = await self.ensure_model_loaded(larger)
                if ok:
                    loaded_any = True
                    logger.info("Escalated: %s -> %s", current, larger)
                break  # one escalation per cycle

        if loaded_any:
            return "ESCALATE: loaded larger variant"
        return "ESCALATE: no upgrade available"

    async def _handle_halt(self) -> str:
        """HALT: unload non-critical models to free VRAM."""
        # Keep only the first loaded model (assumed to be the active one)
        keep = next(iter(self._loaded_models), None)
        to_unload = [m for m in list(self._loaded_models) if m != keep]

        for model_id in to_unload:
            ok = await self._unload_model(model_id)
            if ok:
                self._unloaded_by_halt.add(model_id)

        return f"HALT: unloaded {len(to_unload)} non-critical models"

    async def _handle_resume(self) -> str:
        """RESUME: reload previously unloaded models."""
        reloaded = 0
        for model_id in list(self._unloaded_by_halt):
            ok = await self.ensure_model_loaded(model_id)
            if ok:
                reloaded += 1
                self._unloaded_by_halt.discard(model_id)

        return f"RESUME: reloaded {reloaded} models"

    def _auth_headers(self) -> Dict[str, str]:
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}
