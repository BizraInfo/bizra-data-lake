"""Claude export normalizer with deterministic hint extraction."""

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

_FACT_RE = re.compile(
    r"\b(i am|i'm|my name is|i live in|based in|founder|ceo|i work as)\b",
    re.IGNORECASE,
)
_GOAL_RE = re.compile(
    r"\b(goal|objective|plan|roadmap|i want to|we need to|next step|launch|ship|complete)\b",
    re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(
    r"\b(i prefer|i like|i love|i dislike|rather than|prefer)\b",
    re.IGNORECASE,
)
_PATTERN_RE = re.compile(
    r"\b(always|usually|often|every day|routine|workflow|habit)\b",
    re.IGNORECASE,
)
_EXPERTISE_RE = re.compile(
    r"\b(expert|specialize|experienced|architecture|distributed systems|consensus|rust)\b",
    re.IGNORECASE,
)
_RELATIONSHIP_RE = re.compile(
    r"\b(team|community|users|customers|investors|partners|co-builders)\b",
    re.IGNORECASE,
)
_TEMPORAL_RE = re.compile(
    r"\b(today|tomorrow|this week|next week|deadline|q[1-4]\b|by\s+\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
_EMOTION_RE = re.compile(
    r"\b(frustrated|excited|grateful|worried|urgent|joy|happy|fear)\b",
    re.IGNORECASE,
)
_STYLE_RE = re.compile(
    r"(^|\s)/[A-Z0-9#]{1,6}\b|(^|\n)#{1,4}\s|\b(mode|protocol|structured)\b",
    re.IGNORECASE,
)
_DOMAIN_RE = re.compile(
    r"\b(api|llm|rust|python|agent|memory|compiler|protocol|security|architecture)\b",
    re.IGNORECASE,
)

_MAX_HINTS_PER_TURN = 6
_MIN_CONFIDENCE = 0.72


class ClaudeParser(PlatformParser):
    platform = "claude"

    def parse_payload(self, payload: Any, source_path: str = "") -> list:
        turns = []
        conversations = self._as_conversation_list(payload)

        for convo_index, conversation in enumerate(conversations):
            conversation_id = str(
                conversation.get("uuid")
                or conversation.get("conversation_id")
                or self._conversation_id(conversation, convo_index)
            )
            model = str(
                conversation.get("model") or conversation.get("assistant") or ""
            )
            messages = self._extract_messages(conversation)

            for index, message in enumerate(messages):
                role = canonical_role(
                    message.get("sender")
                    or message.get("role")
                    or (message.get("author") or {}).get("role")
                )
                if role not in {"user", "assistant", "system", "tool"}:
                    continue

                content = normalize_whitespace(
                    collect_text(
                        message.get("text")
                        or message.get("content")
                        or message.get("message")
                    )
                )
                if not content:
                    continue

                timestamp = parse_timestamp(
                    message.get("created_at")
                    or message.get("timestamp")
                    or message.get("updated_at")
                    or conversation.get("created_at")
                    or conversation.get("updated_at")
                )
                turn_id = str(
                    message.get("uuid")
                    or message.get("id")
                    or message.get("message_id")
                    or stable_turn_id(self.platform, conversation_id, index, content)
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
        if isinstance(conversation.get("chat_messages"), list):
            return [
                item
                for item in conversation.get("chat_messages", [])
                if isinstance(item, dict)
            ]

        if isinstance(conversation.get("messages"), list):
            return [item for item in conversation["messages"] if isinstance(item, dict)]

        if isinstance(conversation.get("data"), dict) and isinstance(
            conversation["data"].get("messages"), list
        ):
            return [
                item
                for item in conversation["data"]["messages"]
                if isinstance(item, dict)
            ]

        return []

    def _hints_for_message(self, content: str, role: str) -> list[FragmentHint]:
        if role not in {"user", "assistant"}:
            return []

        text = content.strip()
        if not text:
            return []

        hints: list[FragmentHint] = []
        dedupe: set[tuple[str, str]] = set()

        def push(
            kind: FragmentKind, signal: str, confidence: float, source: str
        ) -> None:
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

        if _FACT_RE.search(text):
            push(FragmentKind.FACT, "self_identity_statement", 0.95, "claude.fact")
        if _GOAL_RE.search(text):
            push(FragmentKind.GOAL, "goal_or_plan_language", 0.95, "claude.goal")
        if _PREFERENCE_RE.search(text):
            push(
                FragmentKind.PREFERENCE,
                "preference_expression",
                0.93,
                "claude.preference",
            )
        if _PATTERN_RE.search(text):
            push(
                FragmentKind.PATTERN,
                "workflow_or_repetition_pattern",
                0.90,
                "claude.pattern",
            )
        if _EXPERTISE_RE.search(text):
            push(
                FragmentKind.EXPERTISE,
                "technical_expertise_signal",
                0.92,
                "claude.expertise",
            )
        if _RELATIONSHIP_RE.search(text):
            push(
                FragmentKind.RELATIONSHIP,
                "stakeholder_reference",
                0.86,
                "claude.relationship",
            )
        if _TEMPORAL_RE.search(text):
            push(
                FragmentKind.TEMPORAL,
                "time_bound_commitment",
                0.88,
                "claude.temporal",
            )
        if _EMOTION_RE.search(text):
            push(FragmentKind.EMOTION, "emotion_signal", 0.84, "claude.emotion")
        if _STYLE_RE.search(text):
            push(
                FragmentKind.STYLE,
                "protocol_or_structured_style",
                0.90,
                "claude.style",
            )
        if _DOMAIN_RE.search(text):
            push(
                FragmentKind.DOMAIN,
                "software_engineering_domain",
                0.88,
                "claude.domain",
            )

        if not hints and len(text) >= 180:
            push(
                FragmentKind.PATTERN,
                "long_form_exchange",
                0.82,
                "claude.length",
            )

        return hints
