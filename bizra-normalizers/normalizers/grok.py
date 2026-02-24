"""Grok (xAI) export normalizer with deterministic role/content extraction."""

from __future__ import annotations

import re
from typing import Any

from schemas import FragmentHint, FragmentKind

from .base import (
    PlatformParser,
    canonical_role,
    collect_text,
    normalize_whitespace,
    parse_timestamp,
    stable_turn_id,
)


_GOAL_RE = re.compile(
    r"\b(goal|objective|plan|roadmap|next step|launch|ship|complete)\b",
    re.IGNORECASE,
)
_DOMAIN_RE = re.compile(
    r"\b(api|llm|model|prompt|agent|memory|compiler|protocol|security|architecture)\b",
    re.IGNORECASE,
)
_STYLE_RE = re.compile(
    r"(^|\s)/[A-Z0-9#]{1,6}\b|(^|\n)#{1,4}\s|\b(mode|protocol|structured)\b",
    re.IGNORECASE,
)
_EMOTION_RE = re.compile(
    r"\b(frustrated|excited|grateful|worried|urgent|joy|happy|fear)\b",
    re.IGNORECASE,
)
_RELATIONSHIP_RE = re.compile(
    r"\b(team|community|users|customers|investors|partners)\b",
    re.IGNORECASE,
)

_MAX_HINTS_PER_TURN = 6
_MIN_CONFIDENCE = 0.72


class GrokParser(PlatformParser):
    platform = "grok"

    def parse_payload(self, payload: Any, source_path: str = "") -> list:
        turns = []
        conversations = self._as_conversation_list(payload)

        for convo_index, conversation in enumerate(conversations):
            nested = conversation.get("conversation")
            convo = nested if isinstance(nested, dict) else conversation
            conversation_id = str(
                convo.get("conversation_id")
                or convo.get("id")
                or conversation.get("id")
                or self._conversation_id(convo, convo_index)
            )
            model = str(
                convo.get("model")
                or conversation.get("model")
                or convo.get("model_id")
                or ""
            )
            messages = self._extract_messages(convo)

            for index, message in enumerate(messages):
                role = canonical_role(
                    message.get("role")
                    or message.get("sender")
                    or (message.get("author") or {}).get("role")
                )
                if role == "unknown":
                    continue

                content = normalize_whitespace(
                    collect_text(
                        message.get("content")
                        or message.get("text")
                        or message.get("message")
                    )
                )
                if not content:
                    continue

                turn_id = str(
                    message.get("id")
                    or message.get("message_id")
                    or stable_turn_id(self.platform, conversation_id, index, content)
                )
                timestamp = parse_timestamp(
                    message.get("created_at")
                    or message.get("timestamp")
                    or message.get("created")
                    or convo.get("updated_at")
                    or convo.get("created_at")
                )

                turns.append(
                    self._build_turn(
                        provider=self.platform,
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        role=role,
                        content=content,
                        timestamp=timestamp,
                        model=model,
                        hints=self._hints_for_message(content, role),
                        metadata={"source_path": source_path},
                    )
                )

        return turns

    def _extract_messages(self, conversation: dict[str, Any]) -> list[dict[str, Any]]:
        messages = conversation.get("messages")
        if isinstance(messages, list):
            return [item for item in messages if isinstance(item, dict)]

        turns = conversation.get("turns")
        if isinstance(turns, list):
            return [item for item in turns if isinstance(item, dict)]

        if any(key in conversation for key in ("prompt", "response", "answer")):
            prompt = collect_text(conversation.get("prompt") or conversation.get("query"))
            response = collect_text(conversation.get("response") or conversation.get("answer"))
            out: list[dict[str, Any]] = []
            if prompt.strip():
                out.append(
                    {
                        "id": "prompt-0",
                        "role": "user",
                        "content": prompt,
                        "created_at": conversation.get("created_at"),
                    }
                )
            if response.strip():
                out.append(
                    {
                        "id": "response-0",
                        "role": "assistant",
                        "content": response,
                        "created_at": conversation.get("created_at"),
                    }
                )
            return out

        return []

    def _hints_for_message(self, content: str, role: str) -> list[FragmentHint]:
        if role not in {"user", "assistant"}:
            return []

        text = content.strip()
        if not text:
            return []

        hints: list[FragmentHint] = []
        dedupe: set[tuple[str, str]] = set()

        def push(kind: FragmentKind, signal: str, confidence: float, source: str) -> None:
            if confidence < _MIN_CONFIDENCE or len(hints) >= _MAX_HINTS_PER_TURN:
                return
            normalized = normalize_whitespace(signal).lower()
            key = (kind.value, normalized)
            if not normalized or key in dedupe:
                return
            dedupe.add(key)
            hints.append(
                FragmentHint(
                    kind=kind,
                    signal=normalized,
                    confidence=confidence,
                    source=source,
                )
            )

        if _GOAL_RE.search(text):
            push(FragmentKind.GOAL, "goal_or_plan_language", 0.92, "grok.goal")
        if _DOMAIN_RE.search(text):
            push(FragmentKind.DOMAIN, "software_engineering_domain", 0.88, "grok.domain")
        if _STYLE_RE.search(text):
            push(FragmentKind.STYLE, "protocol_or_structured_style", 0.86, "grok.style")
        if _RELATIONSHIP_RE.search(text):
            push(FragmentKind.RELATIONSHIP, "stakeholder_reference", 0.84, "grok.relationship")
        if _EMOTION_RE.search(text):
            push(FragmentKind.EMOTION, "emotion_signal", 0.82, "grok.emotion")

        if not hints and len(text) >= 180:
            push(FragmentKind.PATTERN, "long_form_exchange", 0.80, "grok.length")

        return hints

