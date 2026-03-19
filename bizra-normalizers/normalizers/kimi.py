"""Kimi/Moonshot export normalizer with long-context signal extraction."""

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

_TEMPORAL_RE = re.compile(
    r"\b(deadline|by\s+\d{4}-\d{2}-\d{2}|tomorrow|next\s+week|q[1-4]|milestone|before\s+\w+)\b",
    re.IGNORECASE,
)
_REFERENCE_RE = re.compile(
    r"\b(section\s+\d+|appendix\s+[a-z0-9]+|document\s+\w+|doc\s+\w+|file\s+\w+|as\s+discussed)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


class KimiParser(PlatformParser):
    platform = "kimi"

    def parse_payload(self, payload: Any, source_path: str = "") -> list:
        turns = []
        conversations = self._as_conversation_list(payload)

        for convo_index, conversation in enumerate(conversations):
            conversation_id = self._conversation_id(conversation, convo_index)
            model = str(conversation.get("model") or conversation.get("model_id") or "")
            messages = self._extract_messages(conversation)

            for index, message in enumerate(messages):
                role = canonical_role(message.get("role") or message.get("type"))
                content = normalize_whitespace(
                    collect_text(
                        message.get("content")
                        or message.get("text")
                        or message.get("value")
                    )
                )
                if not content:
                    continue

                hints = self._hints_for_message(content)
                turn_id = str(
                    message.get("id")
                    or message.get("segment_id")
                    or stable_turn_id(self.platform, conversation_id, index, content)
                )
                timestamp = parse_timestamp(
                    message.get("created_at")
                    or message.get("timestamp")
                    or conversation.get("created_at")
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
                        hints=hints,
                        metadata={
                            "source_path": source_path,
                            "content_length": len(content),
                        },
                    )
                )

        return turns

    def _extract_messages(self, conversation: dict[str, Any]) -> list[dict[str, Any]]:
        messages = conversation.get("messages")
        if isinstance(messages, list):
            return [item for item in messages if isinstance(item, dict)]

        segments = conversation.get("segments")
        if isinstance(segments, list):
            return [item for item in segments if isinstance(item, dict)]

        items = conversation.get("items")
        if isinstance(items, list):
            out: list[dict[str, Any]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                nested = item.get("messages")
                if isinstance(nested, list):
                    out.extend([row for row in nested if isinstance(row, dict)])
            return out

        data = conversation.get("data")
        if isinstance(data, dict) and isinstance(data.get("messages"), list):
            return [item for item in data["messages"] if isinstance(item, dict)]

        return []

    def _hints_for_message(self, content: str) -> list[FragmentHint]:
        hints: list[FragmentHint] = []

        if _TEMPORAL_RE.search(content):
            hints.append(
                FragmentHint(
                    kind=FragmentKind.TEMPORAL,
                    signal="time_bound_commitment",
                    confidence=0.86,
                    source="kimi.temporal",
                )
            )

        if _REFERENCE_RE.search(content) or _URL_RE.search(content):
            hints.append(
                FragmentHint(
                    kind=FragmentKind.RELATIONSHIP,
                    signal="cross_reference_dependency",
                    confidence=0.83,
                    source="kimi.cross_reference",
                )
            )

        return hints
