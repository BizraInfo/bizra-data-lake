"""BIZRA platform normalizer registry and provider detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import GenericJsonlParser, GenericOpenAIParser, PlatformParser
from .chatgpt import ChatGPTParser
from .claude import ClaudeParser
from .deepseek import DeepSeekParser
from .gemini import GeminiParser
from .grok import GrokParser
from .kimi import KimiParser
from .openai_api import OpenAIAPIParser
from .perplexity import PerplexityParser
from .qwen import QwenParser
from .zhipu import ZhipuParser

# Conversation platforms: where users have identity-building dialogues.
LEGACY_CORE3 = {"chatgpt", "claude", "gemini"}
NEW_CORE4 = {"deepseek", "qwen", "kimi", "zhipu"}
CONVERSATION_PLATFORMS = LEGACY_CORE3 | NEW_CORE4

# Search aggregators: route queries to other models' APIs, primarily search engines.
# Data is still collected (it's the user's data) but has different signal density
# for identity compilation vs genuine conversation platforms.
SEARCH_AGGREGATORS = {"perplexity"}

# Backward-compatible alias — includes search aggregators.
LEGACY_CORE4 = LEGACY_CORE3 | SEARCH_AGGREGATORS

# CORE8: all 8 supported platforms (conversation + search).
CORE8 = tuple(sorted(CONVERSATION_PLATFORMS | SEARCH_AGGREGATORS))
EXTENDED2 = {"openai_api", "grok"}
CORE10 = tuple(sorted(set(CORE8) | EXTENDED2))

# Providers with confirmed export/download capability as of 2025-12.
# Exportable-now means providers with practical first-party export path.
# Keep this derived from canonical sets to avoid drift.
EXPORTABLE_NOW = NEW_CORE4 | {"chatgpt", "claude"}
# Collection gaps are the remainder of CORE8 not currently exportable.
COLLECTION_GAP = set(CORE8) - set(EXPORTABLE_NOW)
# Conversation-only gap: identity-building platforms not yet exportable.
CONVERSATION_GAP = set(CONVERSATION_PLATFORMS) - set(EXPORTABLE_NOW)

_REGISTRY: dict[str, PlatformParser] = {
    "chatgpt": ChatGPTParser(),
    "openai_api": OpenAIAPIParser(),
    "claude": ClaudeParser(),
    "grok": GrokParser(),
    "gemini": GeminiParser(),
    "perplexity": PerplexityParser(),
    "deepseek": DeepSeekParser(),
    "qwen": QwenParser(),
    "kimi": KimiParser(),
    "zhipu": ZhipuParser(),
    # Generic parsers for local/custom model exports.
    "generic_jsonl": GenericJsonlParser(),
    "generic_openai": GenericOpenAIParser(),
}

# Track custom (user-registered) providers separately from built-in ones.
_CUSTOM_PROVIDERS: dict[str, bool] = {}


def register_provider(
    name: str,
    parser: PlatformParser,
    is_conversation_platform: bool = True,
) -> None:
    """Register a custom provider parser at runtime.

    Allows users with local models (LM Studio, Ollama, LocalAI, vLLM,
    text-generation-webui, etc.) to import their conversation history
    using custom or pre-built parsers.

    Args:
        name: Provider name (lowercase, alphanumeric + underscore only).
        parser: ``PlatformParser`` instance to handle this provider's format.
        is_conversation_platform: If ``True``, the provider is added to
            :data:`CONVERSATION_PLATFORMS` so it participates in
            identity-building analysis.

    Raises:
        ValueError: If *name* is empty or contains characters other than
            lowercase alphanumeric and underscore.
        TypeError: If *parser* is not a :class:`PlatformParser` instance.
    """
    if not name or not name.replace("_", "").isalnum():
        raise ValueError(
            f"Invalid provider name: {name!r} "
            "(use lowercase alphanumeric + underscore)"
        )
    if not isinstance(parser, PlatformParser):
        raise TypeError(
            f"parser must be a PlatformParser instance, got {type(parser).__name__}"
        )
    name = name.lower()
    _REGISTRY[name] = parser
    _CUSTOM_PROVIDERS[name] = is_conversation_platform
    if is_conversation_platform:
        CONVERSATION_PLATFORMS.add(name)


def unregister_provider(name: str) -> bool:
    """Remove a previously registered custom provider.

    Built-in providers cannot be removed. Returns ``True`` if the
    provider was found and removed, ``False`` otherwise.
    """
    if name not in _CUSTOM_PROVIDERS:
        return False
    _REGISTRY.pop(name, None)
    CONVERSATION_PLATFORMS.discard(name)
    del _CUSTOM_PROVIDERS[name]
    return True


def registered_providers() -> list[str]:
    """List all registered provider names (built-in + custom), sorted."""
    return sorted(_REGISTRY.keys())


def custom_providers() -> list[str]:
    """List only custom-registered provider names, sorted."""
    return sorted(_CUSTOM_PROVIDERS.keys())


def registry() -> dict[str, PlatformParser]:
    return dict(_REGISTRY)


def parser_for(provider: str) -> PlatformParser | None:
    return _REGISTRY.get(provider)


def _provider_from_signal(value: str) -> str | None:
    text = value.lower()
    if "grok" in text or "xai" in text:
        return "grok"
    if "openai api" in text or "responses api" in text or "chat.completions" in text:
        return "openai_api"
    if "deepseek" in text:
        return "deepseek"
    if "qwen" in text:
        return "qwen"
    if "kimi" in text or "moonshot" in text:
        return "kimi"
    if "zhipu" in text or "glm" in text or "chatglm" in text:
        return "zhipu"
    if "chatgpt" in text or "openai" in text or text.startswith("gpt-"):
        return "chatgpt"
    if "claude" in text or "anthropic" in text:
        return "claude"
    if "gemini" in text or "google" in text:
        return "gemini"
    if "perplexity" in text:
        return "perplexity"
    return None


def _sample_dicts(payload: Any, limit: int = 24) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = 0

    def _push(node: dict[str, Any]) -> None:
        nonlocal seen
        if seen >= limit:
            return
        out.append(node)
        seen += 1

    if isinstance(payload, dict):
        _push(payload)
        for key in ("data", "result"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                _push(nested)
            elif isinstance(nested, list):
                for item in nested:
                    if isinstance(item, dict):
                        _push(item)
        for key in ("conversations", "threads", "items"):
            nested = payload.get(key)
            if isinstance(nested, list):
                for item in nested:
                    if isinstance(item, dict):
                        _push(item)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                _push(item)
            if seen >= limit:
                break

    return out


def _detect_schema_provider(payload: Any) -> str | None:
    # Schema-first truth: strong structural signals outweigh filename hints.
    samples = _sample_dicts(payload)
    if not samples:
        return None

    # OpenAI API request/response logs signature.
    for item in samples:
        req = item.get("request")
        resp = item.get("response")
        if (
            isinstance(req, dict)
            and isinstance(resp, dict)
            and isinstance(req.get("messages"), list)
            and (
                isinstance(resp.get("choices"), list)
                or isinstance(resp.get("output"), list)
            )
        ):
            return "openai_api"
        if isinstance(item.get("messages"), list) and (
            item.get("usage") is not None or item.get("system_fingerprint") is not None
        ):
            model = str(item.get("model") or item.get("model_id") or "").lower()
            if model.startswith(("gpt-", "o1", "o3", "o4")):
                return "openai_api"

    # Grok (xAI) export signature.
    for item in samples:
        model = str(item.get("model") or item.get("model_id") or "").lower()
        source = str(item.get("source") or item.get("platform") or "").lower()
        if isinstance(item.get("messages"), list) and (
            "grok" in model or "xai" in source or "grok" in source
        ):
            return "grok"
        if isinstance(item.get("conversation"), dict):
            convo = item["conversation"]
            convo_model = str(convo.get("model") or "").lower()
            if isinstance(convo.get("messages"), list) and "grok" in convo_model:
                return "grok"

    # ChatGPT export signature.
    for item in samples:
        if isinstance(item.get("mapping"), dict) and item.get("conversation_id"):
            return "chatgpt"

    # Claude export signature.
    for item in samples:
        if isinstance(item.get("chat_messages"), list) and (
            item.get("uuid") is not None
            or item.get("account") is not None
            or item.get("summary") is not None
        ):
            return "claude"

    # DeepSeek export signature.
    for item in samples:
        if isinstance(item.get("mapping"), dict) and (
            item.get("inserted_at") is not None or item.get("updated_at") is not None
        ):
            return "deepseek"

    # Qwen export signature.
    for item in samples:
        model = str(item.get("model") or item.get("model_id") or "").lower()
        if isinstance(item.get("history"), list):
            return "qwen"
        if isinstance(item.get("messages"), list) and "qwen" in model:
            return "qwen"

    # Kimi export signature.
    for item in samples:
        model = str(item.get("model") or item.get("model_id") or "").lower()
        if isinstance(item.get("segments"), list):
            return "kimi"
        if isinstance(item.get("messages"), list) and (
            "kimi" in model or "moonshot" in model
        ):
            return "kimi"

    # Zhipu export signature.
    for item in samples:
        model = str(item.get("model") or item.get("model_id") or "").lower()
        if isinstance(item.get("choices"), list) and (
            item.get("task_id") is not None or "glm" in model or "chatglm" in model
        ):
            return "zhipu"
        if isinstance(item.get("messages"), list) and (
            "glm" in model or "chatglm" in model or "zhipu" in model
        ):
            return "zhipu"

    # Gemini export signature.
    for item in samples:
        model = str(item.get("model") or item.get("model_id") or "").lower()
        if isinstance(item.get("contents"), list):
            return "gemini"
        if isinstance(item.get("candidates"), list) and (
            "gemini" in model or "google" in model
        ):
            return "gemini"

    # Perplexity export signature.
    for item in samples:
        model = str(item.get("model") or item.get("model_id") or "").lower()
        source = str(item.get("source") or item.get("platform") or "").lower()
        if "perplexity" in model or "perplexity" in source:
            return "perplexity"
        if (
            isinstance(item.get("messages"), list)
            and item.get("query") is not None
            and item.get("answer") is not None
        ):
            return "perplexity"

    return None


def _walk_signals(payload: Any) -> list[str]:
    keys = {
        "provider",
        "platform",
        "source",
        "model",
        "model_id",
        "model_slug",
        "default_model_slug",
        "assistant",
    }
    out: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in keys and isinstance(value, str):
                    out.append(value)
                if isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    _walk(item)

    _walk(payload)
    return out


def detect_provider(payload: Any, source_path: str = "") -> str:
    # 1) Schema-first truth.
    schema_provider = _detect_schema_provider(payload)
    if schema_provider:
        return schema_provider

    # 2) Payload metadata signals.
    votes: dict[str, int] = {}
    for signal in _walk_signals(payload):
        provider = _provider_from_signal(signal)
        if provider:
            votes[provider] = votes.get(provider, 0) + 1
    if votes:
        return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    # 3) Filename/path fallback (weakest signal).
    lower_path = source_path.lower()
    for provider, keywords in (
        ("deepseek", ("deepseek",)),
        ("qwen", ("qwen",)),
        ("kimi", ("kimi", "moonshot")),
        ("zhipu", ("zhipu", "glm")),
        ("chatgpt", ("chatgpt", "openai")),
        ("claude", ("claude",)),
        ("gemini", ("gemini", "google")),
        ("perplexity", ("perplexity",)),
    ):
        if any(token in lower_path for token in keywords):
            return provider

    return "unknown"


def parse_payload(payload: Any, source_path: str = "") -> list:
    provider = detect_provider(payload, source_path=source_path)
    parser = parser_for(provider)
    if parser is None:
        return []
    return parser.parse_payload(payload, source_path=source_path)


def parse_file_with_receipt(path: str | Path) -> tuple[list, dict[str, Any]]:
    """Parse one export file and return turns plus a machine-readable receipt."""
    p = Path(path)
    receipt: dict[str, Any] = {
        "path": str(p),
        "provider": "unknown",
        "status": "failed",
        "turn_count": 0,
        "reason_code": None,
        "reason_detail": None,
        "invalid_jsonl_lines": 0,
        "valid_jsonl_records": 0,
    }
    try:
        import json

        raw = p.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        receipt["reason_code"] = "FILE_READ_ERROR"
        receipt["reason_detail"] = str(exc)
        return [], receipt

    if p.suffix.lower() == ".jsonl":
        payload: Any = []
        invalid_jsonl_lines = 0
        for line in raw.splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                invalid_jsonl_lines += 1
                continue
            if isinstance(item, dict):
                payload.append(item)
        receipt["invalid_jsonl_lines"] = invalid_jsonl_lines
        receipt["valid_jsonl_records"] = len(payload)
        if not payload:
            receipt["provider"] = detect_provider([], source_path=str(p))
            receipt["reason_code"] = (
                "JSONL_NO_VALID_OBJECTS" if invalid_jsonl_lines > 0 else "JSONL_EMPTY"
            )
            if invalid_jsonl_lines > 0:
                receipt["reason_detail"] = f"invalid_jsonl_lines={invalid_jsonl_lines}"
            return [], receipt
    else:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            receipt["provider"] = detect_provider({}, source_path=str(p))
            receipt["reason_code"] = "JSON_DECODE_ERROR"
            receipt["reason_detail"] = str(exc)
            return [], receipt

    provider = detect_provider(payload, source_path=str(p))
    receipt["provider"] = provider

    parser = parser_for(provider)
    if parser is None:
        receipt["reason_code"] = "UNSUPPORTED_PROVIDER"
        return [], receipt

    try:
        turns = parser.parse_payload(payload, source_path=str(p))
    except Exception as exc:  # pragma: no cover - defensive parser boundary
        receipt["reason_code"] = "PARSER_EXCEPTION"
        receipt["reason_detail"] = f"{type(exc).__name__}: {exc}"
        return [], receipt

    receipt["turn_count"] = len(turns)
    if not turns:
        receipt["reason_code"] = "NO_TURNS_EXTRACTED"
        return [], receipt

    receipt["status"] = "parsed"
    return turns, receipt


def parse_file(path: str | Path) -> list:
    turns, _receipt = parse_file_with_receipt(path)
    return turns


__all__ = [
    # Set constants
    "CORE8",
    "CORE10",
    "LEGACY_CORE3",
    "LEGACY_CORE4",
    "NEW_CORE4",
    "CONVERSATION_PLATFORMS",
    "SEARCH_AGGREGATORS",
    "EXTENDED2",
    "EXPORTABLE_NOW",
    "COLLECTION_GAP",
    "CONVERSATION_GAP",
    # Built-in parsers
    "ChatGPTParser",
    "OpenAIAPIParser",
    "ClaudeParser",
    "GrokParser",
    "GeminiParser",
    "PerplexityParser",
    "DeepSeekParser",
    "QwenParser",
    "KimiParser",
    "ZhipuParser",
    # Generic parsers for local/custom models
    "GenericJsonlParser",
    "GenericOpenAIParser",
    # Registry functions
    "registry",
    "parser_for",
    "detect_provider",
    "_detect_schema_provider",
    "parse_payload",
    "parse_file_with_receipt",
    "parse_file",
    # Custom provider registration
    "register_provider",
    "unregister_provider",
    "registered_providers",
    "custom_providers",
]
