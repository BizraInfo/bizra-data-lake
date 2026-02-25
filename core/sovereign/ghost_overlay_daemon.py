"""
Ghost Overlay Daemon -- Bridge between Node0 Proactive Loop and Ghost Overlay UI.

Subscribes to the sovereign event bus for HHMM predictions and Muraqabah
opportunities, feeds them through the PredictionDebouncer and ConstitutionalGate,
and emits overlay events to connected Ghost Overlay clients via the WS bridge.

Architecture:
  EventBus("proactive.prediction") -> GhostOverlayDaemon -> ghost_ws.emit_overlay_event()
  EventBus("muraqabah.opportunity") -> GhostOverlayDaemon -> ghost_ws.emit_overlay_event()

Standing on Giants:
- Boyd (OODA): prediction -> overlay is the Orient->Decide UI loop
- Norman (invisible design): overlay only appears when signal exceeds threshold
- Shannon (SNR): predictions gated by UNIFIED_SNR_THRESHOLD
- Al-Ghazali (Ihsan): every suggestion passes ConstitutionalGate before dispatch

Created: 2026-02-25 | BIZRA Ghost Overlay Daemon v0.1
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from core.bridges.ghost_ws import (
    MAX_SUGGESTIONS,
    OverlayEvent,
    OverlaySuggestion,
    PredictionDebouncer,
    emit_overlay_event,
)
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.sovereign.event_bus import Event, EventPriority, get_event_bus

logger = logging.getLogger("bizra.ghost_overlay_daemon")

# Event topics
TOPIC_PREDICTION = "proactive.prediction"
TOPIC_OPPORTUNITY = "muraqabah.opportunity"
TOPIC_OVERLAY_GESTURE = "ghost.gesture"
TOPIC_OVERLAY_DISPATCH = "ghost.action_dispatched"


class GhostOverlayDaemon:
    """Connects the Node0 proactive loop to the Ghost Overlay UI.

    Lifecycle:
      1. start() -- subscribes to event bus topics
      2. Event bus fires predictions/opportunities
      3. Daemon debounces, gates, and builds overlay events
      4. WS bridge pushes to connected Ghost Overlay clients
      5. stop() -- unsubscribes and cleans up
    """

    def __init__(
        self,
        constitutional_gate: Optional[Any] = None,
        debounce_ms: int = 500,
    ) -> None:
        self._event_bus = get_event_bus()
        self._debouncer = PredictionDebouncer(debounce_ms=debounce_ms)
        self._debouncer.set_callback(self._on_debounced_prediction)
        self._constitutional_gate = constitutional_gate
        self._overlay_visible = False
        self._running = False
        self._events_received = 0
        self._events_emitted = 0

    @property
    def overlay_visible(self) -> bool:
        return self._overlay_visible

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "overlay_visible": self._overlay_visible,
            "events_received": self._events_received,
            "events_emitted": self._events_emitted,
        }

    async def start(self) -> None:
        """Subscribe to event bus and begin processing predictions."""
        self._event_bus.subscribe(TOPIC_PREDICTION, self._on_prediction_event)
        self._event_bus.subscribe(TOPIC_OPPORTUNITY, self._on_opportunity_event)
        self._event_bus.subscribe(TOPIC_OVERLAY_GESTURE, self._on_gesture_event)
        self._running = True
        logger.info(
            "Ghost Overlay Daemon started -- subscribed to %s, %s, %s",
            TOPIC_PREDICTION,
            TOPIC_OPPORTUNITY,
            TOPIC_OVERLAY_GESTURE,
        )

    async def stop(self) -> None:
        """Unsubscribe and clean up."""
        self._event_bus.unsubscribe(TOPIC_PREDICTION, self._on_prediction_event)
        self._event_bus.unsubscribe(TOPIC_OPPORTUNITY, self._on_opportunity_event)
        self._event_bus.unsubscribe(TOPIC_OVERLAY_GESTURE, self._on_gesture_event)
        self._running = False
        logger.info("Ghost Overlay Daemon stopped")

    # -----------------------------------------------------------------
    # Event Handlers
    # -----------------------------------------------------------------

    async def _on_prediction_event(self, event: Event) -> None:
        """Handle an HHMM prediction from the proactive loop."""
        self._events_received += 1
        prediction = event.payload
        await self._debouncer.on_prediction(prediction)

    async def _on_opportunity_event(self, event: Event) -> None:
        """Handle a Muraqabah opportunity detection."""
        self._events_received += 1
        opportunity = event.payload

        confidence = opportunity.get("confidence", 0.0)
        if confidence < UNIFIED_SNR_THRESHOLD:
            return

        # Convert opportunity to prediction format for unified processing
        prediction = {
            "intent": opportunity.get("type", "opportunity"),
            "confidence": confidence,
            "context": opportunity.get("context", {}),
            "source": "muraqabah",
        }
        await self._debouncer.on_prediction(prediction)

    async def _on_gesture_event(self, event: Event) -> None:
        """Handle a sovereign gesture from the overlay UI."""
        gesture = event.payload.get("gesture")
        if gesture == "dismiss":
            self._overlay_visible = False
            await emit_overlay_event(OverlayEvent(type="dismiss_overlay"))
        elif gesture == "solidify":
            action_id = event.payload.get("action_id")
            if action_id:
                await self._dispatch_action(action_id, event.payload)

    # -----------------------------------------------------------------
    # Core Logic
    # -----------------------------------------------------------------

    async def _on_debounced_prediction(self, prediction: Dict[str, Any]) -> None:
        """Process a debounced prediction into overlay suggestions."""
        if self._overlay_visible:
            return  # Don't stack overlays

        suggestions = await self._build_suggestions(prediction)
        if not suggestions:
            return

        overlay_event = OverlayEvent(
            type="show_overlay",
            suggestions=[asdict(s) for s in suggestions],
            position=prediction.get("cursor_position"),
        )

        sent = await emit_overlay_event(overlay_event)
        if sent > 0:
            self._overlay_visible = True
            self._events_emitted += 1
            logger.info(
                "Ghost Overlay shown -- %d suggestions, %d clients",
                len(suggestions),
                sent,
            )

    async def _build_suggestions(
        self, prediction: Dict[str, Any]
    ) -> List[OverlaySuggestion]:
        """Build up to MAX_SUGGESTIONS from a prediction, each gated by Ihsan."""
        intent = prediction.get("intent", "unknown")
        confidence = prediction.get("confidence", 0.0)

        # Generate candidate suggestions from the prediction
        candidates = self._generate_candidates(intent, confidence, prediction)

        # Gate each candidate through ConstitutionalGate
        gated: List[OverlaySuggestion] = []
        for candidate in candidates[:MAX_SUGGESTIONS]:
            if self._constitutional_gate is not None:
                try:
                    result = await self._constitutional_gate.check(
                        candidate.ahk_action_id, candidate.intent_summary
                    )
                    candidate.ihsan_precheck = "pass" if result.approved else "blocked"
                    candidate.ihsan_score = getattr(result, "ihsan_score", 0.0)
                    if not result.approved:
                        candidate.block_reason = getattr(result, "reason", "Ihsan gate")
                except Exception as exc:
                    logger.warning("ConstitutionalGate error: %s", exc)
                    candidate.ihsan_precheck = "blocked"
                    candidate.block_reason = "Gate unavailable"
            else:
                # No gate configured -- mark as pending (dev mode)
                candidate.ihsan_precheck = "pending"
                candidate.ihsan_score = 0.0

            gated.append(candidate)

        return gated

    def _generate_candidates(
        self,
        intent: str,
        confidence: float,
        prediction: Dict[str, Any],
    ) -> List[OverlaySuggestion]:
        """Map an HHMM intent to candidate overlay suggestions."""
        suggestion = OverlaySuggestion(
            id=str(uuid.uuid4())[:8],
            action_label=self._intent_to_label(intent),
            intent_summary=f"HHMM: {intent} ({int(confidence * 100)}% confident)",
            hhmm_confidence=confidence,
            ihsan_precheck="pending",
            ihsan_score=0.0,
            ahk_action_id=f"act_{intent}_{uuid.uuid4().hex[:6]}",
        )
        return [suggestion]

    @staticmethod
    def _intent_to_label(intent: str) -> str:
        """Convert an HHMM intent to a human-readable action label."""
        labels = {
            "batch_rename": "Batch rename selected files",
            "merge_region": "Merge selected region",
            "auto_fill": "Auto-fill detected fields",
            "sort": "Sort selected data",
            "opportunity": "Act on detected opportunity",
        }
        return labels.get(intent, f"Execute: {intent.replace('_', ' ').title()}")

    async def _dispatch_action(self, action_id: str, context: Dict[str, Any]) -> None:
        """Dispatch a solidified action to the Action Bus."""
        # Emit dispatch event for the Action Bus to pick up
        await self._event_bus.emit(
            topic=TOPIC_OVERLAY_DISPATCH,
            payload={
                "action_id": action_id,
                "channel": "Ahk",
                "permit_scope": "ghost_overlay",
                "context": context,
            },
            priority=EventPriority.HIGH,
            source="ghost_overlay_daemon",
        )
        self._overlay_visible = False
        await emit_overlay_event(OverlayEvent(type="dismiss_overlay"))
        logger.info("Ghost Overlay: dispatched action %s", action_id)
