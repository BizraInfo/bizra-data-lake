"""Shared parser base and heuristics for platform normalizers."""

from __future__ import annotations

import abc
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from schemas import ConversationTurn, FragmentHint


def canonical_role(raw: Any) -> str:
    """Normalize provider-specific role values into canonical roles."""
    value = str(raw or "").strip().lower()
    if value in {"user", "human", "requester"}:
        return "user"
    if value in {"assistant", "ai", "model", "bot", "response"}:
        return "assistant"
    if value in {"system"}:
        return "system"
    if value in {"tool", "function", "plugin"}:
        return "tool"
    return "unknown"


def parse_timestamp(value: Any) -> int:
    """Parse mixed timestamp formats into UNIX seconds."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        out = int(value)
        # heuristic: millisecond epoch
        if out > 10_000_000_000:
            return out // 1000
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        if text.isdigit():
            out = int(text)
            if out > 10_000_000_000:
                return out // 1000
            return out
        try:
            parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return int(parsed.timestamp())
        except ValueError:
            return 0
    return 0


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def collect_text(content: Any) -> str:
    """Extract text recursively from common export value shapes."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            chunk = collect_text(item)
            if chunk:
                parts.append(chunk)
        return "\n".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content", "message", "value", "output"):
            if key in content:
                return collect_text(content.get(key))
        if "parts" in content:
            return collect_text(content.get("parts"))
    return ""


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def contains_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def stable_turn_id(provider: str, conversation_id: str, index: int, content: str) -> str:
    material = f"{provider}|{conversation_id}|{index}|{normalize_whitespace(content)}"
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]
    return f"{provider}-{digest}"


def apply_cross_platform_boost(
    confidence: float,
    supporting_platforms: Iterable[str],
    min_platforms: int = 3,
    multiplier: float = 1.5,
) -> float:
    """Apply stereoscopic confidence boost when a signal appears on 3+ platforms."""
    platforms = {p for p in supporting_platforms if p}
    if len(platforms) >= min_platforms:
        # Keep a stable decimal value for deterministic comparisons and JSON output.
        return round(min(1.0, confidence * multiplier), 6)
    return confidence


def maybe_parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return None
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


class PlatformParser(abc.ABC):
    """Interface for provider-specific conversation export parsers."""

    platform: str

    @abc.abstractmethod
    def parse_payload(self, payload: Any, source_path: str = "") -> list[ConversationTurn]:
        raise NotImplementedError

    def parse_file(self, path: str | Path) -> list[ConversationTurn]:
        p = Path(path)
        try:
            payload = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except (json.JSONDecodeError, OSError):
            return []
        return self.parse_payload(payload, source_path=str(p))

    @staticmethod
    def _as_conversation_list(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("conversations", "threads", "items"):
                maybe = payload.get(key)
                if isinstance(maybe, list):
                    return [item for item in maybe if isinstance(item, dict)]
            if isinstance(payload.get("data"), dict):
                nested = payload["data"].get("conversations")
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
            return [payload]
        return []

    @staticmethod
    def _conversation_id(conversation: dict[str, Any], fallback_index: int) -> str:
        return str(
            conversation.get("id")
            or conversation.get("uuid")
            or conversation.get("conversation_id")
            or conversation.get("thread_id")
            or f"conv-{fallback_index}"
        )

    @staticmethod
    def _build_turn(
        provider: str,
        conversation_id: str,
        turn_id: str,
        role: str,
        content: str,
        timestamp: int,
        model: str,
        hints: list[FragmentHint] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        return ConversationTurn(
            provider=provider,
            conversation_id=conversation_id,
            turn_id=turn_id,
            role=role,
            content=content,
            timestamp=timestamp,
            model=model,
            fragment_hints=hints or [],
            metadata=metadata or {},
        )


class GenericJsonlParser(PlatformParser):
    """Parser for generic JSONL with role + content fields.

    Handles any JSONL file where each line has at minimum:
    - A role field (user/assistant/system)
    - A content field (text)

    Supports configurable field names for maximum compatibility
    with local model exports (LM Studio, Ollama, LocalAI, etc.).
    """

    platform = "generic_jsonl"

    def __init__(
        self,
        role_field: str = "role",
        content_field: str = "content",
        timestamp_field: str = "timestamp",
        model_field: str = "model",
        conversation_id_field: str = "conversation_id",
    ) -> None:
        self._role_field = role_field
        self._content_field = content_field
        self._timestamp_field = timestamp_field
        self._model_field = model_field
        self._conversation_id_field = conversation_id_field

    def parse_payload(
        self, payload: Any, source_path: str = ""
    ) -> list[ConversationTurn]:
        conversations = self._as_conversation_list(payload)
        turns: list[ConversationTurn] = []

        for idx, item in enumerate(conversations):
            role_raw = item.get(self._role_field, "")
            content_raw = item.get(self._content_field, "")

            role = canonical_role(role_raw)
            content = collect_text(content_raw)
            if not content.strip():
                continue

            timestamp = parse_timestamp(item.get(self._timestamp_field))
            model = str(item.get(self._model_field, "local"))
            conv_id = str(
                item.get(
                    self._conversation_id_field,
                    f"session-{hash(source_path) % 10000}",
                )
            )
            turn_id = stable_turn_id(self.platform, conv_id, idx, content)

            turns.append(
                self._build_turn(
                    provider=self.platform,
                    conversation_id=conv_id,
                    turn_id=turn_id,
                    role=role,
                    content=content,
                    timestamp=timestamp,
                    model=model,
                )
            )

        return turns


class GenericOpenAIParser(PlatformParser):
    """Parser for OpenAI-compatible API format.

    Handles conversation logs from any OpenAI-compatible API server:
    LM Studio, Ollama, LocalAI, vLLM, text-generation-webui, etc.

    Expected format: list of objects with ``messages`` array containing
    ``{role, content}`` pairs, similar to the chat.completions API format.
    Also handles flat ``{role, content}`` objects without a wrapping
    ``messages`` array.
    """

    platform = "generic_openai"

    def parse_payload(
        self, payload: Any, source_path: str = ""
    ) -> list[ConversationTurn]:
        conversations = self._as_conversation_list(payload)
        turns: list[ConversationTurn] = []
        global_idx = 0

        for conv_idx, item in enumerate(conversations):
            messages = item.get("messages")
            if not isinstance(messages, list):
                # Single message format (flat role+content object).
                role_raw = item.get("role", "")
                content_raw = item.get("content", "")
                role = canonical_role(role_raw)
                content = collect_text(content_raw)
                if content.strip():
                    conv_id = str(
                        item.get(
                            "conversation_id",
                            item.get("id", f"conv-{conv_idx}"),
                        )
                    )
                    turn_id = stable_turn_id(
                        self.platform, conv_id, global_idx, content
                    )
                    timestamp = parse_timestamp(
                        item.get(
                            "timestamp",
                            item.get("created_at", item.get("created", 0)),
                        )
                    )
                    model = str(item.get("model", "local"))
                    turns.append(
                        self._build_turn(
                            provider=self.platform,
                            conversation_id=conv_id,
                            turn_id=turn_id,
                            role=role,
                            content=content,
                            timestamp=timestamp,
                            model=model,
                        )
                    )
                    global_idx += 1
                continue

            conv_id = str(
                item.get("id", item.get("conversation_id", f"conv-{conv_idx}"))
            )
            model = str(item.get("model", "local"))
            base_ts = parse_timestamp(
                item.get(
                    "timestamp",
                    item.get("created_at", item.get("created", 0)),
                )
            )

            for msg_idx, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    continue
                role = canonical_role(msg.get("role", ""))
                content = collect_text(msg.get("content", ""))
                if not content.strip():
                    continue

                turn_id = stable_turn_id(
                    self.platform, conv_id, global_idx, content
                )
                timestamp = parse_timestamp(msg.get("timestamp", 0)) or (
                    base_ts + msg_idx
                )

                turns.append(
                    self._build_turn(
                        provider=self.platform,
                        conversation_id=conv_id,
                        turn_id=turn_id,
                        role=role,
                        content=content,
                        timestamp=timestamp,
                        model=model,
                    )
                )
                global_idx += 1

        return turns


__all__ = [
    "PlatformParser",
    "GenericJsonlParser",
    "GenericOpenAIParser",
    "ConversationTurn",
    "FragmentHint",
    "canonical_role",
    "parse_timestamp",
    "normalize_whitespace",
    "collect_text",
    "contains_cjk",
    "contains_latin",
    "stable_turn_id",
    "apply_cross_platform_boost",
    "maybe_parse_json_object",
]
