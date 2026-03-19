"""DeepSeek export normalizer with reasoning-trace signal extraction."""

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

_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


class DeepSeekParser(PlatformParser):
    platform = "deepseek"

    def parse_payload(self, payload: Any, source_path: str = "") -> list:
        turns = []
        conversations = self._as_conversation_list(payload)

        for convo_index, conversation in enumerate(conversations):
            conversation_id = self._conversation_id(conversation, convo_index)
            model = str(
                conversation.get("model")
                or conversation.get("model_slug")
                or conversation.get("provider_model")
                or ""
            )
            messages = self._extract_messages(conversation)

            for index, message in enumerate(messages):
                role = canonical_role(
                    message.get("role")
                    or message.get("sender")
                    or (message.get("author") or {}).get("role")
                )
                content = collect_text(
                    message.get("content")
                    or message.get("text")
                    or message.get("message")
                )
                if not content.strip():
                    continue

                hints, clean_content = self._reasoning_hints(content, message)
                turn_id = str(
                    message.get("id")
                    or message.get("message_id")
                    or stable_turn_id(
                        self.platform, conversation_id, index, clean_content
                    )
                )
                timestamp = parse_timestamp(
                    message.get("created_at")
                    or message.get("timestamp")
                    or message.get("inserted_at")
                    or message.get("create_time")
                    or conversation.get("created_at")
                )

                turns.append(
                    self._build_turn(
                        provider=self.platform,
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        role=role,
                        content=clean_content,
                        timestamp=timestamp,
                        model=model,
                        hints=hints,
                        metadata={"source_path": source_path},
                    )
                )

        return turns

    def _extract_messages(self, conversation: dict[str, Any]) -> list[dict[str, Any]]:
        messages = conversation.get("messages")
        if isinstance(messages, list):
            return [item for item in messages if isinstance(item, dict)]

        mapping = conversation.get("mapping")
        if isinstance(mapping, dict):
            out: list[dict[str, Any]] = []
            for node_id, node in mapping.items():
                if not isinstance(node, dict):
                    continue
                msg = node.get("message")
                if not isinstance(msg, dict):
                    continue

                author_role = canonical_role(
                    (msg.get("author") or {}).get("role")
                    if isinstance(msg.get("author"), dict)
                    else None
                )
                if author_role != "unknown":
                    out.append(
                        {
                            "id": msg.get("id") or node_id,
                            "role": author_role,
                            "content": msg.get("content"),
                            "created_at": msg.get("create_time")
                            or msg.get("inserted_at"),
                            "reasoning_content": msg.get("reasoning_content"),
                        }
                    )

                fragments = msg.get("fragments")
                if isinstance(fragments, list):
                    for frag_index, fragment in enumerate(fragments):
                        if not isinstance(fragment, dict):
                            continue
                        frag_type = str(fragment.get("type") or "").strip().upper()
                        role = {
                            "REQUEST": "user",
                            "RESPONSE": "assistant",
                            "THINK": "assistant",
                            "SEARCH": "tool",
                        }.get(frag_type, "unknown")
                        if role == "unknown":
                            continue
                        out.append(
                            {
                                "id": f"{msg.get('id') or node_id}:{frag_index}",
                                "role": role,
                                "content": fragment.get("content"),
                                "created_at": msg.get("inserted_at")
                                or msg.get("create_time"),
                            }
                        )
            return out

        return []

    def _reasoning_hints(
        self,
        content: str,
        message: dict[str, Any],
    ) -> tuple[list[FragmentHint], str]:
        hints: list[FragmentHint] = []

        reasoning_chunks = [
            normalize_whitespace(chunk) for chunk in _THINK_BLOCK_RE.findall(content)
        ]
        reasoning_field = collect_text(message.get("reasoning_content"))
        if reasoning_field.strip():
            reasoning_chunks.append(normalize_whitespace(reasoning_field))

        clean_content = normalize_whitespace(_THINK_BLOCK_RE.sub("", content))
        if not clean_content:
            clean_content = normalize_whitespace(content)

        for chunk in reasoning_chunks:
            if not chunk:
                continue
            hints.append(
                FragmentHint(
                    kind=FragmentKind.PATTERN,
                    signal=f"reasoning_trace:{chunk[:200]}",
                    confidence=0.95,
                    source="deepseek.reasoning_trace",
                )
            )
            hints.append(
                FragmentHint(
                    kind=FragmentKind.EXPERTISE,
                    signal="engages_with_complex_reasoning",
                    confidence=0.87,
                    source="deepseek.reasoning_trace",
                )
            )

        return hints, clean_content
