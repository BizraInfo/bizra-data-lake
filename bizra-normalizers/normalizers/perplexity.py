"""Perplexity export normalizer with research-centric hint extraction."""

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


_CITATION_RE = re.compile(r"\[[0-9]+\]|https?://\S+", re.IGNORECASE)
_RESEARCH_RE = re.compile(
    r"\b(source|citation|references?|study|paper|benchmark|evidence|report)\b",
    re.IGNORECASE,
)
_GOAL_RE = re.compile(
    r"\b(goal|objective|plan|roadmap|next step|launch|ship|complete)\b",
    re.IGNORECASE,
)
_DOMAIN_RE = re.compile(
    r"\b(api|llm|rust|python|agent|memory|compiler|protocol|security|architecture)\b",
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

_MAX_HINTS_PER_TURN = 6
_MIN_CONFIDENCE = 0.72


class PerplexityParser(PlatformParser):
    platform = "perplexity"

    def parse_payload(self, payload: Any, source_path: str = "") -> list:
        turns = []
        conversations = self._as_conversation_list(payload)

        for convo_index, conversation in enumerate(conversations):
            conversation_id = self._conversation_id(conversation, convo_index)
            model = str(conversation.get("model") or conversation.get("model_id") or "")
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
                        hints=self._hints_for_message(content, role, message),
                        metadata={"source_path": source_path},
                    )
                )

        return turns

    def _extract_messages(self, conversation: dict[str, Any]) -> list[dict[str, Any]]:
        messages = conversation.get("messages")
        if isinstance(messages, list):
            return [item for item in messages if isinstance(item, dict)]

        if any(key in conversation for key in ("query", "prompt", "input")):
            prompt = collect_text(
                conversation.get("query")
                or conversation.get("prompt")
                or conversation.get("input")
            )
            answer = collect_text(
                conversation.get("answer")
                or conversation.get("response")
                or conversation.get("output")
            )
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
            if answer.strip():
                out.append(
                    {
                        "id": "answer-0",
                        "role": "assistant",
                        "content": answer,
                        "created_at": conversation.get("created_at"),
                        "citations": conversation.get("citations"),
                    }
                )
            return out

        choices = conversation.get("choices")
        if isinstance(choices, list):
            out = []
            for idx, choice in enumerate(choices):
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict):
                    out.append(message)
                    continue
                out.append(
                    {
                        "id": f"choice-{idx}",
                        "role": "assistant",
                        "content": choice.get("content"),
                    }
                )
            return out

        data = conversation.get("data")
        if isinstance(data, dict):
            if isinstance(data.get("messages"), list):
                return [item for item in data["messages"] if isinstance(item, dict)]
            if isinstance(data.get("choices"), list):
                return self._extract_messages({"choices": data["choices"]})

        return []

    def _hints_for_message(
        self,
        content: str,
        role: str,
        message: dict[str, Any],
    ) -> list[FragmentHint]:
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

        citations = message.get("citations")
        if (isinstance(citations, list) and citations) or _CITATION_RE.search(text):
            push(
                FragmentKind.FACT,
                "citation_backed_claim",
                0.90,
                "perplexity.citation",
            )

        if _RESEARCH_RE.search(text):
            push(
                FragmentKind.EXPERTISE,
                "research_synthesis_language",
                0.86,
                "perplexity.research",
            )
        if _GOAL_RE.search(text):
            push(FragmentKind.GOAL, "goal_or_plan_language", 0.85, "perplexity.goal")
        if _DOMAIN_RE.search(text):
            push(
                FragmentKind.DOMAIN,
                "software_engineering_domain",
                0.83,
                "perplexity.domain",
            )
        if _TEMPORAL_RE.search(text):
            push(
                FragmentKind.TEMPORAL,
                "time_bound_commitment",
                0.82,
                "perplexity.temporal",
            )
        if _EMOTION_RE.search(text):
            push(FragmentKind.EMOTION, "emotion_signal", 0.80, "perplexity.emotion")

        if not hints and len(text) >= 180:
            push(
                FragmentKind.PATTERN,
                "long_form_exchange",
                0.80,
                "perplexity.length",
            )

        return hints
