"""Unified conversation schema -- platform-agnostic ConversationTurn.

All platforms normalize to this single shape.
Ref: specs/user-zero-bootstrap/phase_01_multi_platform_ingestion.md S1

Standing on Giants: Shannon (information entropy) - Lamport (logical clocks)
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Platform(str, enum.Enum):
    """Supported conversation platforms.

    Each value corresponds to a parser in parsers.py and maps 1:1
    to a platform-specific export format.
    """

    CHATGPT = "chatgpt"
    OPENAI_API = "openai_api"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    PERPLEXITY = "perplexity"
    QWEN = "qwen"
    KIMI = "kimi"
    ZHIPU = "zhipu"
    CLAUDE = "claude"
    GROK = "grok"


class Role(str, enum.Enum):
    """Conversation participant role."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ConversationTurn(BaseModel):
    """A single turn in a conversation, normalized across all platforms.

    Fields:
        id: Domain-separated BLAKE3 hash hex (32 chars).
        platform: Source platform enum value.
        conversation_id: Platform-native conversation identifier.
        turn_index: Zero-based position within the conversation.
        role: Speaker role (user, assistant, system, tool).
        content: The actual text content of the turn.
        model: LLM model identifier if known.
        timestamp: UTC timestamp of the turn if available.
        metadata: Platform-specific metadata (tool calls, reasoning, etc.).
        token_count: Estimated token count (enrichment phase).
        content_hash: BLAKE3 dedup hash of normalized content.
        language: ISO 639-1 language code (enrichment phase).
        language_conf: Language detection confidence [0, 1].
        topics: Extracted keyword topics (enrichment phase).
    """

    id: str
    platform: Platform
    conversation_id: str
    turn_index: int
    role: Role
    content: str
    model: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_count: int | None = None
    content_hash: str | None = None
    language: str | None = None
    language_conf: float | None = None
    topics: list[str] = Field(default_factory=list)
