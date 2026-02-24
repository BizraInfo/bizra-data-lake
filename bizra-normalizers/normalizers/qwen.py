"""Qwen export normalizer with multilingual code-switch signal extraction."""

from __future__ import annotations

import re
from typing import Any

from schemas import FragmentHint, FragmentKind

from .base import (
    PlatformParser,
    canonical_role,
    collect_text,
    contains_cjk,
    contains_latin,
    normalize_whitespace,
    parse_timestamp,
    stable_turn_id,
)


_CODE_RE = re.compile(r"```|\b(def|class|import|function|try|except|return)\b")


class QwenParser(PlatformParser):
    platform = "qwen"

    def parse_payload(self, payload: Any, source_path: str = "") -> list:
        turns = []
        conversations = self._as_conversation_list(payload)

        for convo_index, conversation in enumerate(conversations):
            conversation_id = self._conversation_id(conversation, convo_index)
            model = str(conversation.get("model_id") or conversation.get("model") or "")
            messages = self._extract_messages(conversation)

            for index, message in enumerate(messages):
                role = canonical_role(message.get("role"))
                content = normalize_whitespace(collect_text(message.get("content") or message.get("text")))
                if not content:
                    continue

                hints = self._hints_for_message(content)
                turn_id = str(message.get("id") or message.get("message_id") or stable_turn_id(self.platform, conversation_id, index, content))
                timestamp = parse_timestamp(
                    message.get("created_at")
                    or message.get("timestamp")
                    or message.get("updated_at")
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
                        metadata={"source_path": source_path},
                    )
                )

        return turns

    def _extract_messages(self, conversation: dict[str, Any]) -> list[dict[str, Any]]:
        messages = conversation.get("messages")
        if isinstance(messages, list):
            return [item for item in messages if isinstance(item, dict)]

        history = conversation.get("history")
        if isinstance(history, list):
            out: list[dict[str, Any]] = []
            for idx, pair in enumerate(history):
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                user_text = collect_text(pair[0])
                bot_text = collect_text(pair[1])
                if user_text.strip():
                    out.append({"id": f"hist-{idx}-u", "role": "user", "content": user_text})
                if bot_text.strip():
                    out.append({"id": f"hist-{idx}-a", "role": "assistant", "content": bot_text})
            return out

        data = conversation.get("data")
        if isinstance(data, dict) and isinstance(data.get("messages"), list):
            return [item for item in data["messages"] if isinstance(item, dict)]

        return []

    def _hints_for_message(self, content: str) -> list[FragmentHint]:
        hints: list[FragmentHint] = []

        if contains_cjk(content) and contains_latin(content):
            hints.append(
                FragmentHint(
                    kind=FragmentKind.STYLE,
                    signal="multilingual_code_switching",
                    confidence=0.88,
                    source="qwen.multilingual",
                )
            )
            hints.append(
                FragmentHint(
                    kind=FragmentKind.DOMAIN,
                    signal="bilingual_technical_context",
                    confidence=0.84,
                    source="qwen.multilingual",
                )
            )

        if _CODE_RE.search(content):
            hints.append(
                FragmentHint(
                    kind=FragmentKind.DOMAIN,
                    signal="software_engineering",
                    confidence=0.84,
                    source="qwen.code_pattern",
                )
            )

        return hints
