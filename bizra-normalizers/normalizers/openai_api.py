"""OpenAI API conversation normalizer for request/response logs and JSONL traces."""

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
_PREFERENCE_RE = re.compile(r"\b(i prefer|i like|rather than|prefer)\b", re.IGNORECASE)
_FACT_RE = re.compile(
    r"\b(i am|i'm|my name is|i live in|based in|founder|ceo|i work as)\b",
    re.IGNORECASE,
)
_DOMAIN_RE = re.compile(
    r"\b(api|sdk|llm|python|rust|agent|memory|compiler|protocol|security|architecture)\b",
    re.IGNORECASE,
)
_TEMPORAL_RE = re.compile(
    r"\b(today|tomorrow|this week|next week|deadline|q[1-4]\b|by\s+\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)

_MAX_HINTS_PER_TURN = 6
_MIN_CONFIDENCE = 0.72


class OpenAIAPIParser(PlatformParser):
    platform = "openai_api"

    def parse_payload(self, payload: Any, source_path: str = "") -> list:
        turns = []
        conversations = self._as_conversation_list(payload)

        for convo_index, conversation in enumerate(conversations):
            conversation_id = str(
                conversation.get("conversation_id")
                or conversation.get("id")
                or conversation.get("request_id")
                or conversation.get("trace_id")
                or self._conversation_id(conversation, convo_index)
            )
            model = self._extract_model(conversation)
            messages = self._extract_messages(conversation)

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
                        or message.get("output_text")
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
                        hints=self._hints_for_message(content, role),
                        metadata={
                            "source_path": source_path,
                            "request_id": conversation.get("request_id"),
                        },
                    )
                )

        return turns

    def _extract_model(self, conversation: dict[str, Any]) -> str:
        request = conversation.get("request")
        response = conversation.get("response")
        if isinstance(conversation.get("model"), str):
            return str(conversation["model"])
        if isinstance(request, dict) and isinstance(request.get("model"), str):
            return str(request["model"])
        if isinstance(response, dict) and isinstance(response.get("model"), str):
            return str(response["model"])
        return str(conversation.get("model_id") or "")

    def _extract_messages(self, conversation: dict[str, Any]) -> list[dict[str, Any]]:
        messages = conversation.get("messages")
        if isinstance(messages, list):
            return [item for item in messages if isinstance(item, dict)]

        out: list[dict[str, Any]] = []

        request = conversation.get("request")
        if isinstance(request, dict):
            req_messages = request.get("messages")
            if isinstance(req_messages, list):
                for item in req_messages:
                    if isinstance(item, dict):
                        out.append(item)
            req_input = collect_text(request.get("input"))
            if req_input.strip():
                out.append(
                    {
                        "id": "request-input-0",
                        "role": "user",
                        "content": req_input,
                        "created_at": conversation.get("created_at"),
                    }
                )

        response = conversation.get("response")
        if isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list):
                for idx, choice in enumerate(choices):
                    if not isinstance(choice, dict):
                        continue
                    message = choice.get("message")
                    if isinstance(message, dict):
                        out.append(message)
                        continue
                    content = collect_text(choice.get("text") or choice.get("content"))
                    if content.strip():
                        out.append(
                            {
                                "id": f"response-choice-{idx}",
                                "role": "assistant",
                                "content": content,
                                "created_at": response.get("created"),
                            }
                        )

            response_output = response.get("output")
            if isinstance(response_output, list):
                for idx, item in enumerate(response_output):
                    if not isinstance(item, dict):
                        continue
                    for block in item.get("content") or []:
                        text = collect_text(block)
                        if text.strip():
                            out.append(
                                {
                                    "id": f"response-output-{idx}",
                                    "role": "assistant",
                                    "content": text,
                                    "created_at": response.get("created"),
                                }
                            )

            output_text = collect_text(response.get("output_text"))
            if output_text.strip():
                out.append(
                    {
                        "id": "response-output-text-0",
                        "role": "assistant",
                        "content": output_text,
                        "created_at": response.get("created"),
                    }
                )

        if not out:
            choices = conversation.get("choices")
            if isinstance(choices, list):
                for idx, choice in enumerate(choices):
                    if not isinstance(choice, dict):
                        continue
                    message = choice.get("message")
                    if isinstance(message, dict):
                        out.append(message)
                        continue
                    content = collect_text(choice.get("text") or choice.get("content"))
                    if content.strip():
                        out.append(
                            {
                                "id": f"choice-{idx}",
                                "role": "assistant",
                                "content": content,
                            }
                        )

        return out

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

        if _FACT_RE.search(text):
            push(FragmentKind.FACT, "self_identity_statement", 0.94, "openai_api.fact")
        if _GOAL_RE.search(text):
            push(FragmentKind.GOAL, "goal_or_plan_language", 0.94, "openai_api.goal")
        if _PREFERENCE_RE.search(text):
            push(
                FragmentKind.PREFERENCE,
                "preference_expression",
                0.90,
                "openai_api.preference",
            )
        if _TEMPORAL_RE.search(text):
            push(
                FragmentKind.TEMPORAL,
                "time_bound_commitment",
                0.86,
                "openai_api.temporal",
            )
        if _DOMAIN_RE.search(text):
            push(
                FragmentKind.DOMAIN,
                "software_engineering_domain",
                0.88,
                "openai_api.domain",
            )

        if not hints and len(text) >= 180:
            push(
                FragmentKind.PATTERN,
                "long_form_exchange",
                0.80,
                "openai_api.length",
            )

        return hints

