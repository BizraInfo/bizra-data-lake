"""Platform-specific conversation parsers.

Each parser implements detect() + parse() for its platform format.
Parsers never panic; malformed records produce warnings and skip.

Ref: specs/user-zero-bootstrap/phase_01_multi_platform_ingestion.md S2-3

Standing on Giants: Shannon (channel coding) - Besta (graph-of-thought)
"""

from __future__ import annotations

import abc
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.genesis.ingestion.schema import ConversationTurn, Platform, Role

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hashing utility
# ---------------------------------------------------------------------------


def _blake3_conversation_id(platform: str, conv_id: str, index: int) -> str:
    """Domain-separated BLAKE3 hash for conversation turn ID.

    Uses hashlib.blake2b fallback when the blake3 package is not installed.
    """
    try:
        import blake3

        h = blake3.blake3(derive_key_context="genesis/conversation/v1")
        h.update(platform.encode())
        h.update(conv_id.encode())
        h.update(index.to_bytes(4, "little"))
        return h.hexdigest()[:32]
    except ImportError:
        import hashlib

        h = hashlib.blake2b(
            f"genesis/conversation/v1:{platform}:{conv_id}:{index}".encode(),
            digest_size=16,
        )
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class PlatformParser(abc.ABC):
    """Base class for platform-specific conversation parsers."""

    @abc.abstractmethod
    def platform_name(self) -> Platform: ...

    @abc.abstractmethod
    def detect(self, raw_bytes: bytes) -> bool:
        """Return True if the first 4096 bytes match this platform's format."""
        ...

    @abc.abstractmethod
    def parse(
        self, data: Any, source_path: Path | None = None
    ) -> list[ConversationTurn]:
        """Parse raw data into unified ConversationTurn records."""
        ...

    def _make_turn(
        self,
        conv_id: str,
        index: int,
        role: Role,
        content: str,
        model: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        """Helper to construct a ConversationTurn with a deterministic ID."""
        return ConversationTurn(
            id=_blake3_conversation_id(self.platform_name().value, conv_id, index),
            platform=self.platform_name(),
            conversation_id=conv_id,
            turn_index=index,
            role=role,
            content=content,
            model=model,
            timestamp=timestamp,
            metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# ChatGPT
# ---------------------------------------------------------------------------


class ChatGPTParser(PlatformParser):
    """ChatGPT conversation export parser (JSON with mapping tree).

    Ref: ingest_conversations.py:extract_turns() for the tree traversal pattern.
    """

    def platform_name(self) -> Platform:
        return Platform.CHATGPT

    def detect(self, raw_bytes: bytes) -> bool:
        try:
            text = raw_bytes.decode("utf-8", errors="ignore")
            return '"mapping"' in text and '"message"' in text and '"author"' in text
        except Exception:  # noqa: BLE001 — boundary boundary
            return False

    def parse(
        self, data: Any, source_path: Path | None = None
    ) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        if isinstance(data, list):
            for conv in data:
                turns.extend(self._parse_single(conv))
        elif isinstance(data, dict):
            turns.extend(self._parse_single(data))
        return turns

    def _parse_single(self, conv: dict[str, Any]) -> list[ConversationTurn]:
        mapping = conv.get("mapping", {})
        conv_id = conv.get("conversation_id", conv.get("id", "unknown"))
        model_slug = conv.get("default_model_slug")
        raw_turns: list[tuple[float, str, str, datetime | None, str | None]] = []

        for _node_id, node in mapping.items():
            msg = node.get("message")
            if msg is None:
                continue
            author = msg.get("author", {})
            role_str = author.get("role", "")
            if role_str not in ("user", "assistant", "system", "tool"):
                continue

            content_obj = msg.get("content", {})
            if not isinstance(content_obj, dict):
                continue

            parts = content_obj.get("parts", [])
            text_parts = [
                p for p in parts if isinstance(p, str) and len(p.strip()) >= 10
            ]
            if not text_parts:
                continue

            combined = "\n\n".join(text_parts)
            create_time = msg.get("create_time")
            ts: datetime | None = None
            if create_time is not None:
                try:
                    ts = datetime.fromtimestamp(float(create_time), tz=timezone.utc)
                except (ValueError, TypeError, OSError):
                    pass

            model = msg.get("metadata", {}).get("model_slug") or model_slug
            raw_turns.append((create_time or 0, role_str, combined, ts, model))

        raw_turns.sort(key=lambda t: t[0])
        result: list[ConversationTurn] = []
        for idx, (_, role_str, text, ts, model) in enumerate(raw_turns):
            try:
                role = Role(role_str)
            except ValueError:
                role = Role.ASSISTANT
            result.append(
                self._make_turn(conv_id, idx, role, text, model=model, timestamp=ts)
            )
        return result


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class ClaudeParser(PlatformParser):
    """Claude conversation export parser (JSON with chat_messages)."""

    def platform_name(self) -> Platform:
        return Platform.CLAUDE

    def detect(self, raw_bytes: bytes) -> bool:
        text = raw_bytes.decode("utf-8", errors="ignore")
        return '"chat_messages"' in text or ('"sender"' in text and '"text"' in text)

    def parse(
        self, data: Any, source_path: Path | None = None
    ) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        if isinstance(data, list):
            for conv in data:
                turns.extend(self._parse_single(conv))
        elif isinstance(data, dict):
            turns.extend(self._parse_single(data))
        return turns

    def _parse_single(self, conv: dict[str, Any]) -> list[ConversationTurn]:
        messages = conv.get("chat_messages", [])
        conv_id = conv.get("uuid", conv.get("id", "unknown"))
        result: list[ConversationTurn] = []
        for idx, msg in enumerate(messages):
            sender = msg.get("sender", "")
            role = Role.USER if sender == "human" else Role.ASSISTANT
            text = msg.get("text", "")
            if not text or len(text.strip()) < 10:
                continue
            ts: datetime | None = None
            if "created_at" in msg:
                try:
                    ts = datetime.fromisoformat(
                        msg["created_at"].replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    pass
            model = msg.get("model")
            result.append(
                self._make_turn(conv_id, idx, role, text, model=model, timestamp=ts)
            )
        return result


# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------


class DeepSeekParser(PlatformParser):
    """DeepSeek conversation export parser."""

    def platform_name(self) -> Platform:
        return Platform.DEEPSEEK

    def detect(self, raw_bytes: bytes) -> bool:
        text = raw_bytes.decode("utf-8", errors="ignore").lower()
        return (
            '"messages"' in text and "deepseek" in text
        ) or '"reasoning_content"' in text

    def parse(
        self, data: Any, source_path: Path | None = None
    ) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        if isinstance(data, list):
            for conv in data:
                turns.extend(self._parse_single(conv))
        elif isinstance(data, dict):
            turns.extend(self._parse_single(data))
        return turns

    def _parse_single(self, conv: dict[str, Any]) -> list[ConversationTurn]:
        messages = conv.get("messages", [])
        conv_id = conv.get("id", conv.get("conversation_id", "unknown"))
        result: list[ConversationTurn] = []
        for idx, msg in enumerate(messages):
            role_str = msg.get("role", "assistant")
            try:
                role = Role(role_str)
            except ValueError:
                role = Role.ASSISTANT
            content = msg.get("content", "")
            if not content or len(content.strip()) < 10:
                continue

            metadata: dict[str, Any] = {}
            reasoning = msg.get("reasoning_content")
            if reasoning:
                metadata["reasoning_content"] = reasoning

            ts: datetime | None = None
            raw_ts = msg.get("created_at")
            if raw_ts:
                try:
                    if isinstance(raw_ts, (int, float)):
                        ts = datetime.fromtimestamp(raw_ts, tz=timezone.utc)
                    else:
                        ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
                except (ValueError, TypeError, OSError):
                    pass

            model = msg.get("model") or conv.get("model")
            result.append(
                self._make_turn(
                    conv_id,
                    idx,
                    role,
                    content,
                    model=model,
                    timestamp=ts,
                    metadata=metadata,
                )
            )
        return result


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class GeminiParser(PlatformParser):
    """Google Gemini / Google Takeout conversation parser."""

    def platform_name(self) -> Platform:
        return Platform.GEMINI

    def detect(self, raw_bytes: bytes) -> bool:
        text = raw_bytes.decode("utf-8", errors="ignore")
        return '"chunks"' in text or ('"createTime"' in text and '"parts"' in text)

    def parse(
        self, data: Any, source_path: Path | None = None
    ) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        if isinstance(data, list):
            for conv in data:
                turns.extend(self._parse_single(conv))
        elif isinstance(data, dict):
            turns.extend(self._parse_single(data))
        return turns

    def _parse_single(self, conv: dict[str, Any]) -> list[ConversationTurn]:
        chunks = conv.get("chunks", [])
        conv_id = conv.get("id", conv.get("conversationId", "unknown"))
        result: list[ConversationTurn] = []
        turn_idx = 0
        for chunk in chunks:
            author_num = chunk.get("author")
            role = Role.USER if author_num == 0 else Role.ASSISTANT

            parts = chunk.get("parts", [])
            text_parts = [
                p.get("text", "")
                for p in parts
                if isinstance(p, dict) and p.get("text")
            ]
            combined = "\n".join(text_parts)
            if not combined or len(combined.strip()) < 10:
                continue

            ts: datetime | None = None
            raw_ts = chunk.get("createTime") or chunk.get("create_time")
            if raw_ts:
                try:
                    ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

            model = conv.get("model", "gemini-pro")
            result.append(
                self._make_turn(
                    conv_id, turn_idx, role, combined, model=model, timestamp=ts
                )
            )
            turn_idx += 1
        return result


# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------


class OpenAIAPIParser(PlatformParser):
    """OpenAI API log parser (JSONL format)."""

    def platform_name(self) -> Platform:
        return Platform.OPENAI_API

    def detect(self, raw_bytes: bytes) -> bool:
        text = raw_bytes.decode("utf-8", errors="ignore")
        text_lower = text.lower()
        # Reject if another platform's signature is present
        for sig in (
            "deepseek",
            "moonshot",
            "kimi",
            "qwen",
            "tongyi",
            "chatglm",
            "zhipu",
            "glm-",
            "grok",
            "xai",
            "perplexity",
            "chat_messages",
            "chunks",
        ):
            if sig in text_lower:
                return False
        first_line = text.split("\n")[0].strip()
        if not first_line:
            return False
        try:
            obj = json.loads(first_line)
            return "messages" in obj and isinstance(obj["messages"], list)
        except (json.JSONDecodeError, TypeError):
            return False

    def parse(
        self, data: Any, source_path: Path | None = None
    ) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        if isinstance(data, str):
            for line_num, line in enumerate(data.strip().split("\n")):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    turns.extend(
                        self._parse_request(obj, f"req_{line_num}", source_path)
                    )
                except json.JSONDecodeError:
                    log.warning("Skipping malformed JSONL line %d", line_num)
        elif isinstance(data, list):
            for i, obj in enumerate(data):
                turns.extend(self._parse_request(obj, f"req_{i}", source_path))
        return turns

    def _parse_request(
        self, obj: dict[str, Any], conv_id: str, source_path: Path | None
    ) -> list[ConversationTurn]:
        messages = obj.get("messages", [])
        model = obj.get("model")
        file_mtime: datetime | None = None
        if source_path and source_path.exists():
            file_mtime = datetime.fromtimestamp(
                source_path.stat().st_mtime, tz=timezone.utc
            )

        result: list[ConversationTurn] = []
        for idx, msg in enumerate(messages):
            role_str = msg.get("role", "user")
            try:
                role = Role(role_str)
            except ValueError:
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", str(p)) for p in content if isinstance(p, (str, dict))
                )
            if not content or len(str(content).strip()) < 10:
                continue
            result.append(
                self._make_turn(
                    conv_id,
                    idx,
                    role,
                    str(content),
                    model=model,
                    timestamp=file_mtime,
                )
            )
        return result


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


class PerplexityParser(PlatformParser):
    """Perplexity conversation parser (Markdown/YAML or JSON)."""

    def platform_name(self) -> Platform:
        return Platform.PERPLEXITY

    def detect(self, raw_bytes: bytes) -> bool:
        text = raw_bytes.decode("utf-8", errors="ignore")
        return (
            "## Query" in text and "## Answer" in text
        ) or "perplexity" in text.lower()

    def parse(
        self, data: Any, source_path: Path | None = None
    ) -> list[ConversationTurn]:
        if isinstance(data, str):
            return self._parse_markdown(data, source_path)
        elif isinstance(data, dict):
            return self._parse_json(data)
        elif isinstance(data, list):
            turns: list[ConversationTurn] = []
            for item in data:
                if isinstance(item, dict):
                    turns.extend(self._parse_json(item))
            return turns
        return []

    def _parse_markdown(
        self, text: str, source_path: Path | None
    ) -> list[ConversationTurn]:
        conv_id = source_path.stem if source_path else "perplexity_unknown"
        sections = re.split(r"(?m)^## (Query|Answer)\s*$", text)
        turns: list[ConversationTurn] = []
        idx = 0
        i = 1
        while i < len(sections) - 1:
            section_type = sections[i].strip()
            content = sections[i + 1].strip()
            # Strip citation markers [1], [2], etc.
            content_clean = re.sub(r"\[\d+\]", "", content)
            if len(content_clean) >= 10:
                role = Role.USER if section_type == "Query" else Role.ASSISTANT
                turns.append(
                    self._make_turn(
                        conv_id, idx, role, content_clean, model="perplexity"
                    )
                )
                idx += 1
            i += 2
        return turns

    def _parse_json(self, conv: dict[str, Any]) -> list[ConversationTurn]:
        messages = conv.get("messages", [])
        conv_id = conv.get("id", "unknown")
        result: list[ConversationTurn] = []
        for idx, msg in enumerate(messages):
            role_str = msg.get("role", "assistant")
            try:
                role = Role(role_str)
            except ValueError:
                role = Role.ASSISTANT
            content = msg.get("content", "")
            if not content or len(content.strip()) < 10:
                continue
            result.append(
                self._make_turn(conv_id, idx, role, content, model="perplexity")
            )
        return result


# ---------------------------------------------------------------------------
# Qwen
# ---------------------------------------------------------------------------


class QwenParser(PlatformParser):
    """Qwen/Tongyi conversation parser."""

    def platform_name(self) -> Platform:
        return Platform.QWEN

    def detect(self, raw_bytes: bytes) -> bool:
        text = raw_bytes.decode("utf-8", errors="ignore").lower()
        return "qwen" in text or '"model_id"' in text or "tongyi" in text

    def parse(
        self, data: Any, source_path: Path | None = None
    ) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        if isinstance(data, list):
            for conv in data:
                turns.extend(self._parse_single(conv))
        elif isinstance(data, dict):
            turns.extend(self._parse_single(data))
        return turns

    def _parse_single(self, conv: dict[str, Any]) -> list[ConversationTurn]:
        messages = conv.get("messages", [])
        conv_id = conv.get("id", "unknown")
        result: list[ConversationTurn] = []
        for idx, msg in enumerate(messages):
            role_str = msg.get("role", "assistant")
            try:
                role = Role(role_str)
            except ValueError:
                role = Role.ASSISTANT
            content = msg.get("content", "")
            if not content or len(content.strip()) < 10:
                continue

            ts: datetime | None = None
            raw_ts = msg.get("created_at") or msg.get("timestamp")
            if raw_ts:
                try:
                    if isinstance(raw_ts, int) and raw_ts > 1e12:
                        ts = datetime.fromtimestamp(raw_ts / 1000, tz=timezone.utc)
                    elif isinstance(raw_ts, (int, float)):
                        ts = datetime.fromtimestamp(raw_ts, tz=timezone.utc)
                except (ValueError, TypeError, OSError):
                    pass

            model = msg.get("model_id") or msg.get("model") or conv.get("model_id")
            result.append(
                self._make_turn(conv_id, idx, role, content, model=model, timestamp=ts)
            )
        return result


# ---------------------------------------------------------------------------
# Kimi
# ---------------------------------------------------------------------------


class KimiParser(PlatformParser):
    """Kimi (Moonshot) conversation parser."""

    def platform_name(self) -> Platform:
        return Platform.KIMI

    def detect(self, raw_bytes: bytes) -> bool:
        text = raw_bytes.decode("utf-8", errors="ignore").lower()
        return "moonshot" in text or "kimi" in text

    def parse(
        self, data: Any, source_path: Path | None = None
    ) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        if isinstance(data, list):
            for conv in data:
                turns.extend(self._parse_single(conv))
        elif isinstance(data, dict):
            turns.extend(self._parse_single(data))
        return turns

    def _parse_single(self, conv: dict[str, Any]) -> list[ConversationTurn]:
        messages = conv.get("messages", [])
        conv_id = conv.get("id", "unknown")
        result: list[ConversationTurn] = []
        for idx, msg in enumerate(messages):
            role_str = msg.get("role", "assistant")
            try:
                role = Role(role_str)
            except ValueError:
                role = Role.ASSISTANT
            content = msg.get("content", "")
            if len(content) > 8000:
                content = content[:8000]  # 128K context; chunk large content
            if not content or len(content.strip()) < 10:
                continue

            ts: datetime | None = None
            raw_ts = msg.get("created_at")
            if raw_ts:
                try:
                    ts = datetime.fromtimestamp(int(raw_ts), tz=timezone.utc)
                except (ValueError, TypeError, OSError):
                    pass

            model = msg.get("model") or conv.get("model")
            result.append(
                self._make_turn(conv_id, idx, role, content, model=model, timestamp=ts)
            )
        return result


# ---------------------------------------------------------------------------
# Zhipu
# ---------------------------------------------------------------------------


class ZhipuParser(PlatformParser):
    """Zhipu (ChatGLM) conversation parser."""

    def platform_name(self) -> Platform:
        return Platform.ZHIPU

    def detect(self, raw_bytes: bytes) -> bool:
        text = raw_bytes.decode("utf-8", errors="ignore").lower()
        return "chatglm" in text or "zhipu" in text or "glm-" in text

    def parse(
        self, data: Any, source_path: Path | None = None
    ) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        if isinstance(data, list):
            for conv in data:
                turns.extend(self._parse_single(conv))
        elif isinstance(data, dict):
            turns.extend(self._parse_single(data))
        return turns

    def _parse_single(self, conv: dict[str, Any]) -> list[ConversationTurn]:
        messages = conv.get("messages", [])
        conv_id = conv.get("id", "unknown")
        result: list[ConversationTurn] = []
        for idx, msg in enumerate(messages):
            role_str = msg.get("role", "assistant")
            try:
                role = Role(role_str)
            except ValueError:
                role = Role.ASSISTANT
            content = msg.get("content", "")
            if not content or len(content.strip()) < 10:
                continue

            metadata: dict[str, Any] = {}
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                metadata["tool_calls"] = tool_calls

            ts: datetime | None = None
            raw_ts = msg.get("created")
            if raw_ts:
                try:
                    ts = datetime.fromtimestamp(int(raw_ts), tz=timezone.utc)
                except (ValueError, TypeError, OSError):
                    pass

            model = msg.get("model") or conv.get("model")
            result.append(
                self._make_turn(
                    conv_id,
                    idx,
                    role,
                    content,
                    model=model,
                    timestamp=ts,
                    metadata=metadata,
                )
            )
        return result


# ---------------------------------------------------------------------------
# Grok
# ---------------------------------------------------------------------------


class GrokParser(PlatformParser):
    """Grok conversation parser (JSON or CSV)."""

    def platform_name(self) -> Platform:
        return Platform.GROK

    def detect(self, raw_bytes: bytes) -> bool:
        text = raw_bytes.decode("utf-8", errors="ignore").lower()
        return "grok" in text or ('"messages"' in text and "xai" in text)

    def parse(
        self, data: Any, source_path: Path | None = None
    ) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        if isinstance(data, list):
            for conv in data:
                turns.extend(self._parse_single(conv))
        elif isinstance(data, dict):
            turns.extend(self._parse_single(data))
        return turns

    def _parse_single(self, conv: dict[str, Any]) -> list[ConversationTurn]:
        messages = conv.get("messages", [])
        conv_id = conv.get("id", "unknown")
        result: list[ConversationTurn] = []
        for idx, msg in enumerate(messages):
            role_str = msg.get("role", "assistant")
            try:
                role = Role(role_str)
            except ValueError:
                role = Role.ASSISTANT
            content = msg.get("content", "")
            if not content or len(content.strip()) < 10:
                continue

            ts: datetime | None = None
            raw_ts = msg.get("timestamp")
            if raw_ts:
                try:
                    ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

            model = msg.get("model") or conv.get("model")
            result.append(
                self._make_turn(conv_id, idx, role, content, model=model, timestamp=ts)
            )
        return result


# ---------------------------------------------------------------------------
# Parser Registry
# ---------------------------------------------------------------------------

# Order matters: platform-specific parsers first, generic OpenAI API last.
# Parsers with distinctive signatures (mapping tree, chat_messages, chunks)
# come first, followed by those that match model-name substrings, with the
# generic messages-array parser as the fallback.
ALL_PARSERS: list[PlatformParser] = [
    ChatGPTParser(),  # "mapping" + "message" + "author"
    ClaudeParser(),  # "chat_messages" or "sender" + "text"
    GeminiParser(),  # "chunks" or "createTime" + "parts"
    PerplexityParser(),  # "## Query" + "## Answer" or "perplexity"
    DeepSeekParser(),  # "deepseek" or "reasoning_content"
    QwenParser(),  # "qwen" or "tongyi" or "model_id"
    KimiParser(),  # "moonshot" or "kimi"
    ZhipuParser(),  # "chatglm" or "zhipu" or "glm-"
    GrokParser(),  # "grok" or "xai"
    OpenAIAPIParser(),  # Generic: {"messages": [...]} (fallback)
]

PARSER_MAP: dict[Platform, PlatformParser] = {p.platform_name(): p for p in ALL_PARSERS}


def detect_platform(raw_bytes: bytes) -> Platform | None:
    """Auto-detect platform from raw file bytes (first 4096).

    Iterates through all registered parsers and returns the first match.
    Returns None if no parser claims the format.
    """
    sample = raw_bytes[:4096]
    for parser in ALL_PARSERS:
        try:
            if parser.detect(sample):
                return parser.platform_name()
        except Exception:  # noqa: BLE001 — boundary boundary
            continue
    return None
