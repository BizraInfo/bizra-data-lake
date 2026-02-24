"""Zhipu/GLM export normalizer with structured-output and tool-call signals."""

from __future__ import annotations

from typing import Any

from schemas import FragmentHint, FragmentKind

from .base import (
    PlatformParser,
    canonical_role,
    collect_text,
    maybe_parse_json_object,
    normalize_whitespace,
    parse_timestamp,
    stable_turn_id,
)


class ZhipuParser(PlatformParser):
    platform = "zhipu"

    def parse_payload(self, payload: Any, source_path: str = "") -> list:
        turns = []
        conversations = self._as_conversation_list(payload)

        for convo_index, conversation in enumerate(conversations):
            conversation_id = self._conversation_id(conversation, convo_index)
            model = str(conversation.get("model") or conversation.get("model_id") or "")
            messages = self._extract_messages(conversation)

            for index, message in enumerate(messages):
                role = canonical_role(message.get("role") or "assistant")
                content = normalize_whitespace(collect_text(message.get("content") or message.get("text") or message.get("message")))
                if not content:
                    continue

                hints = self._hints_for_message(content, message)
                turn_id = str(message.get("id") or message.get("message_id") or stable_turn_id(self.platform, conversation_id, index, content))
                timestamp = parse_timestamp(
                    message.get("created")
                    or message.get("created_at")
                    or message.get("timestamp")
                    or conversation.get("created")
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

        choices = conversation.get("choices")
        if isinstance(choices, list):
            out: list[dict[str, Any]] = []
            for idx, choice in enumerate(choices):
                if not isinstance(choice, dict):
                    continue
                msg = choice.get("message")
                if isinstance(msg, dict):
                    out.append(msg)
                else:
                    out.append(
                        {
                            "id": choice.get("index", idx),
                            "role": "assistant",
                            "content": choice.get("content"),
                        }
                    )
            return out

        if any(key in conversation for key in ("prompt", "query", "input")):
            prompt = collect_text(conversation.get("prompt") or conversation.get("query") or conversation.get("input"))
            response = collect_text(conversation.get("response") or conversation.get("output") or conversation.get("answer"))
            out = []
            if prompt.strip():
                out.append({"id": "prompt-0", "role": "user", "content": prompt})
            if response.strip():
                out.append({"id": "response-0", "role": "assistant", "content": response})
            return out

        data = conversation.get("data")
        if isinstance(data, dict):
            if isinstance(data.get("choices"), list):
                return self._extract_messages({"choices": data["choices"]})
            if isinstance(data.get("messages"), list):
                return [item for item in data["messages"] if isinstance(item, dict)]

        return []

    def _hints_for_message(self, content: str, message: dict[str, Any]) -> list[FragmentHint]:
        hints: list[FragmentHint] = []

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name") or call.get("function", {}).get("name") or "tool")
                hints.append(
                    FragmentHint(
                        kind=FragmentKind.GOAL,
                        signal=f"tool_call:{name}",
                        confidence=0.85,
                        source="zhipu.tool_call",
                        metadata={"arguments": call.get("arguments") or call.get("function", {}).get("arguments")},
                    )
                )

        citations = message.get("citations")
        if isinstance(citations, list) and citations:
            hints.append(
                FragmentHint(
                    kind=FragmentKind.FACT,
                    signal="citations_present",
                    confidence=0.80,
                    source="zhipu.citations",
                    metadata={"count": len(citations)},
                )
            )

        parsed = maybe_parse_json_object(content)
        if parsed is not None:
            hints.append(
                FragmentHint(
                    kind=FragmentKind.FACT,
                    signal="structured_output",
                    confidence=0.82,
                    source="zhipu.structured_output",
                    metadata={"keys": sorted(parsed.keys())},
                )
            )
            if any(key in parsed for key in ("plan", "steps", "goal", "objective")):
                hints.append(
                    FragmentHint(
                        kind=FragmentKind.GOAL,
                        signal="explicit_plan_structure",
                        confidence=0.85,
                        source="zhipu.structured_output",
                    )
                )

        lowered = content.lower()
        if any(keyword in lowered for keyword in ("plan", "next step", "objective", "roadmap")):
            hints.append(
                FragmentHint(
                    kind=FragmentKind.GOAL,
                    signal="goal_or_plan_language",
                    confidence=0.85,
                    source="zhipu.language",
                )
            )

        return hints
