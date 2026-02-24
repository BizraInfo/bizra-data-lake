"""Gemini export normalizer with deterministic hint extraction."""

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


_GOAL_RE = re.compile(
    r"\b(goal|objective|plan|roadmap|next step|launch|ship|complete)\b",
    re.IGNORECASE,
)
_FACT_RE = re.compile(
    r"\b(according to|evidence|source|citation|reported|study|paper)\b",
    re.IGNORECASE,
)
_EXPERTISE_RE = re.compile(
    r"\b(architecture|distributed systems|consensus|compiler|protocol|optimization)\b",
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
_CODE_RE = re.compile(r"```|\b(def|class|import|function|try|except|return)\b")
_DOMAIN_RE = re.compile(
    r"\b(api|llm|rust|python|agent|memory|compiler|protocol|security|architecture)\b",
    re.IGNORECASE,
)

_MAX_HINTS_PER_TURN = 6
_MIN_CONFIDENCE = 0.72


class GeminiParser(PlatformParser):
    platform = "gemini"

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
                        or message.get("parts")
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
                        hints=self._hints_for_message(content, role),
                        metadata={"source_path": source_path},
                    )
                )

        return turns

    def _extract_messages(self, conversation: dict[str, Any]) -> list[dict[str, Any]]:
        messages = conversation.get("messages")
        if isinstance(messages, list):
            return [item for item in messages if isinstance(item, dict)]

        contents = conversation.get("contents")
        if isinstance(contents, list):
            out: list[dict[str, Any]] = []
            for idx, item in enumerate(contents):
                if not isinstance(item, dict):
                    continue
                out.append(
                    {
                        "id": item.get("id") or f"contents-{idx}",
                        "role": "assistant"
                        if str(item.get("role") or "").strip().lower() == "model"
                        else item.get("role"),
                        "content": item.get("parts") or item.get("content"),
                        "timestamp": item.get("create_time"),
                    }
                )
            return out

        candidates = conversation.get("candidates")
        if isinstance(candidates, list):
            out = []
            for idx, item in enumerate(candidates):
                if not isinstance(item, dict):
                    continue
                content = item.get("content") if isinstance(item.get("content"), dict) else {}
                out.append(
                    {
                        "id": item.get("id") or f"candidate-{idx}",
                        "role": "assistant",
                        "content": content.get("parts") or content.get("content"),
                        "timestamp": item.get("create_time"),
                    }
                )
            return out

        if any(key in conversation for key in ("prompt", "query", "input")):
            prompt = collect_text(
                conversation.get("prompt")
                or conversation.get("query")
                or conversation.get("input")
            )
            response = collect_text(
                conversation.get("response")
                or conversation.get("output")
                or conversation.get("answer")
            )
            out: list[dict[str, Any]] = []
            if prompt.strip():
                out.append({"id": "prompt-0", "role": "user", "content": prompt})
            if response.strip():
                out.append({"id": "response-0", "role": "assistant", "content": response})
            return out

        data = conversation.get("data")
        if isinstance(data, dict):
            if isinstance(data.get("messages"), list):
                return [item for item in data["messages"] if isinstance(item, dict)]
            if isinstance(data.get("contents"), list):
                return self._extract_messages({"contents": data["contents"]})

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
            push(FragmentKind.GOAL, "goal_or_plan_language", 0.90, "gemini.goal")
        if _FACT_RE.search(text):
            push(FragmentKind.FACT, "evidence_or_citation_language", 0.88, "gemini.fact")
        if _EXPERTISE_RE.search(text):
            push(
                FragmentKind.EXPERTISE,
                "technical_expertise_signal",
                0.88,
                "gemini.expertise",
            )
        if _TEMPORAL_RE.search(text):
            push(FragmentKind.TEMPORAL, "time_bound_commitment", 0.85, "gemini.temporal")
        if _EMOTION_RE.search(text):
            push(FragmentKind.EMOTION, "emotion_signal", 0.80, "gemini.emotion")
        if contains_cjk(text) and contains_latin(text):
            push(
                FragmentKind.STYLE,
                "multilingual_code_switching",
                0.84,
                "gemini.multilingual",
            )
        if _CODE_RE.search(text) or _DOMAIN_RE.search(text):
            push(
                FragmentKind.DOMAIN,
                "software_engineering_domain",
                0.84,
                "gemini.domain",
            )
        if "i prefer" in text.lower() or "prefer " in text.lower():
            push(
                FragmentKind.PREFERENCE,
                "preference_expression",
                0.84,
                "gemini.preference",
            )

        if not hints and len(text) >= 180:
            push(FragmentKind.PATTERN, "long_form_exchange", 0.80, "gemini.length")

        return hints
