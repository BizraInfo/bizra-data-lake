#!/usr/bin/env python3
"""Provider normalizers for Core 8 corpus ingestion.

Canonical output fields (CorpusRecord v1):
- provider
- account_scope
- conversation_id
- message_id
- role
- timestamp
- content_hash
- source_path
- import_run_id
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

CORE8 = {
    "chatgpt_openai",
    "claude",
    "gemini_google",
    "deepseek",
    "qwen",
    "kimi",
    "perplexity",
    "zhipu",
}


@dataclass(frozen=True)
class CorpusRecord:
    provider: str
    account_scope: str
    conversation_id: str
    message_id: str
    role: str
    timestamp: int
    content_hash: str
    source_path: str
    import_run_id: str


def _hash_text(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _role(raw: str | None) -> str:
    if not raw:
        return "unknown"
    v = raw.strip().lower()
    if v in {"user", "human"}:
        return "user"
    if v in {"assistant", "ai", "model"}:
        return "assistant"
    if v in {"system"}:
        return "system"
    if v in {"tool", "function"}:
        return "tool"
    return "unknown"


def _parse_ts(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return 0
        if s.isdigit():
            return int(s)
        try:
            # Handle ISO-like timestamps.
            parsed = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            return int(parsed.timestamp())
        except ValueError:
            return 0
    return 0


def _collect_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            part = _collect_text(item)
            if part:
                parts.append(part)
        return "\n".join(parts)
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        if "parts" in content:
            return _collect_text(content["parts"])
        if "content" in content:
            return _collect_text(content["content"])
        if "fragments" in content:
            return _collect_text(content["fragments"])
        if "message" in content:
            return _collect_text(content["message"])
    return ""


def detect_provider(path: Path) -> str:
    return detect_provider_with_payload(path=path, payload=None)


def _provider_from_signal(value: str) -> str | None:
    v = value.strip().lower()
    if not v:
        return None

    if "deepseek" in v:
        return "deepseek"
    if "qwen" in v:
        return "qwen"
    if "kimi" in v or "moonshot" in v:
        return "kimi"
    if "zhipu" in v or "chatglm" in v or re.search(r"\bglm[-_a-z0-9]*\b", v):
        return "zhipu"
    if "gemini" in v or "google" in v:
        return "gemini_google"
    if "claude" in v or "anthropic" in v:
        return "claude"
    if "perplexity" in v:
        return "perplexity"
    if "chatgpt" in v or "openai" in v or v.startswith("gpt-") or v in {"o3", "o3-pro", "o4-mini", "o4-mini-high"}:
        return "chatgpt_openai"
    return None


def _iter_structured_provider_signals(payload: Any) -> Iterable[str]:
    signal_keys = {
        "provider",
        "model",
        "model_slug",
        "default_model_slug",
        "source",
        "platform",
        "app",
        "assistant",
    }

    def walk(x: Any) -> Iterable[str]:
        if isinstance(x, dict):
            for k, v in x.items():
                if k in signal_keys and isinstance(v, str) and v.strip():
                    yield v
                if isinstance(v, (dict, list)):
                    yield from walk(v)
        elif isinstance(x, list):
            for item in x:
                if isinstance(item, (dict, list)):
                    yield from walk(item)

    return walk(payload)


def _infer_provider_from_payload(payload: Any) -> str | None:
    votes: dict[str, int] = {}
    for signal in _iter_structured_provider_signals(payload):
        inferred = _provider_from_signal(signal)
        if inferred is None:
            continue
        votes[inferred] = votes.get(inferred, 0) + 1
    if not votes:
        return None
    # Deterministic tie-breaker: highest votes, then lexical provider name.
    return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def detect_provider_with_payload(path: Path, payload: Any | None) -> str:
    lpath = str(path).lower()

    # Specific providers MUST be checked before chatgpt_openai because
    # chatgpt_openai uses the broad "00_intake" keyword that would match
    # any file under 00_INTAKE/, including qwen-export/, kimi-export/, etc.
    keyword_map_ordered = [
        ("deepseek", ["deepseek"]),
        ("qwen", ["qwen"]),
        ("kimi", ["kimi", "moonshot"]),
        ("zhipu", ["zhipu", "glm"]),
        ("claude", ["claude"]),
        ("gemini_google", ["gemini", "google"]),
        ("perplexity", ["perplexity"]),
        ("chatgpt_openai", ["chatgpt", "openai", "00_intake", "conversations-"]),
    ]
    for provider, keys in keyword_map_ordered:
        if any(k in lpath for k in keys):
            return provider

    if payload is not None:
        inferred = _infer_provider_from_payload(payload)
        if inferred is not None:
            return inferred

    return "generic"


def _extract_mapping_messages(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node_id, node in mapping.items():
        msg = (node or {}).get("message") or {}
        if not isinstance(msg, dict):
            continue
        author = ((msg.get("author") or {}).get("role") or "") if isinstance(msg.get("author"), dict) else ""
        author_role = _role(author)
        if author_role != "unknown":
            content = _collect_text(msg.get("content"))
            if content.strip():
                rows.append(
                    {
                        "message_id": str(msg.get("id") or node_id),
                        "role": author_role,
                        "text": content,
                        "timestamp": _parse_ts(msg.get("create_time") or msg.get("inserted_at")),
                    }
                )
            continue

        # DeepSeek-style exports encode interaction turns as fragments with typed roles.
        fragments = msg.get("fragments")
        if isinstance(fragments, list):
            fragment_ts = _parse_ts(msg.get("inserted_at") or msg.get("create_time") or msg.get("created_at"))
            for idx, fragment in enumerate(fragments):
                if not isinstance(fragment, dict):
                    continue
                ftype = str(fragment.get("type") or "").strip().upper()
                role = {
                    "REQUEST": "user",
                    "RESPONSE": "assistant",
                    "THINK": "assistant",
                    "SEARCH": "tool",
                }.get(ftype, "unknown")
                if role == "unknown":
                    continue
                text = _collect_text(fragment.get("content"))
                if not text.strip():
                    continue
                rows.append(
                    {
                        "message_id": f"{msg.get('id') or node_id}:{idx}",
                        "role": role,
                        "text": text,
                        "timestamp": fragment_ts,
                    }
                )
    rows.sort(key=lambda x: (x["timestamp"], x["message_id"]))
    return rows


def _extract_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        sender = msg.get("sender") or msg.get("role") or ((msg.get("author") or {}).get("role") if isinstance(msg.get("author"), dict) else "")
        role = _role(str(sender) if sender is not None else None)
        if role == "unknown":
            continue
        content = _collect_text(msg.get("text") or msg.get("content") or msg.get("message"))
        if not content.strip():
            continue
        rows.append(
            {
                "message_id": str(msg.get("uuid") or msg.get("id") or msg.get("message_id") or idx),
                "role": role,
                "text": content,
                "timestamp": _parse_ts(msg.get("created_at") or msg.get("inserted_at") or msg.get("timestamp")),
            }
        )
    rows.sort(key=lambda x: (x["timestamp"], x["message_id"]))
    return rows


def _extract_history_pairs(history: list[Any]) -> list[dict[str, Any]]:
    """Extract messages from Qwen-style history pairs: [[user, bot], ...]."""
    rows: list[dict[str, Any]] = []
    for idx, pair in enumerate(history):
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        user_text = _collect_text(pair[0])
        bot_text = _collect_text(pair[1])
        if user_text.strip():
            rows.append({"message_id": f"hist-{idx}-user", "role": "user", "text": user_text, "timestamp": 0})
        if bot_text.strip():
            rows.append({"message_id": f"hist-{idx}-assistant", "role": "assistant", "text": bot_text, "timestamp": 0})
    return rows


def _extract_segments(segments: list[Any]) -> list[dict[str, Any]]:
    """Extract messages from Kimi/Moonshot-style segments array."""
    rows: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        role = _role(str(seg.get("role") or seg.get("type") or ""))
        if role == "unknown":
            continue
        content = _collect_text(seg.get("content") or seg.get("text") or seg.get("value"))
        if not content.strip():
            continue
        rows.append(
            {
                "message_id": str(seg.get("id") or seg.get("segment_id") or idx),
                "role": role,
                "text": content,
                "timestamp": _parse_ts(seg.get("created_at") or seg.get("timestamp")),
            }
        )
    rows.sort(key=lambda x: (x["timestamp"], x["message_id"]))
    return rows


def _extract_choices(choices: list[Any]) -> list[dict[str, Any]]:
    """Extract messages from Zhipu/GLM-style OpenAI-compatible choices array."""
    rows: list[dict[str, Any]] = []
    for idx, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message") or choice
        if not isinstance(msg, dict):
            continue
        role = _role(str(msg.get("role") or "assistant"))
        content = _collect_text(msg.get("content"))
        if not content.strip():
            continue
        rows.append(
            {
                "message_id": str(msg.get("id") or choice.get("index") or idx),
                "role": role,
                "text": content,
                "timestamp": 0,
            }
        )
    return rows


def _unwrap_data_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Unwrap common {data: {...}} or {result: {...}} envelopes used by Chinese AI platforms."""
    inner = payload.get("data") or payload.get("result")
    if isinstance(inner, dict):
        return inner
    return payload


def _extract_prompt_response(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract from prompt+response/output pair format."""
    prompt = _collect_text(payload.get("prompt") or payload.get("query") or payload.get("input"))
    response = _collect_text(payload.get("response") or payload.get("output") or payload.get("answer"))
    rows: list[dict[str, Any]] = []
    if prompt.strip():
        rows.append({"message_id": "prompt-0", "role": "user", "text": prompt, "timestamp": _parse_ts(payload.get("created_at") or payload.get("timestamp"))})
    if response.strip():
        rows.append({"message_id": "response-0", "role": "assistant", "text": response, "timestamp": _parse_ts(payload.get("created_at") or payload.get("timestamp"))})
    return rows


def _convo_id_from_dict(d: dict[str, Any]) -> str:
    return str(
        d.get("id") or d.get("uuid") or d.get("conversation_id")
        or d.get("task_id") or d.get("request_id") or d.get("kimiplus_id")
        or d.get("invocation_id") or d.get("title") or d.get("name") or "single"
    )


def _acct_from_dict(d: dict[str, Any]) -> str:
    return str(
        ((d.get("account") or {}).get("uuid") if isinstance(d.get("account"), dict) else None)
        or d.get("owner_id") or d.get("user_id") or "default"
    )


def _try_extract_messages(d: dict[str, Any]) -> list[dict[str, Any]]:
    """Try all known message extraction strategies on a dict."""
    # Strategy 1: ChatGPT/DeepSeek mapping
    if isinstance(d.get("mapping"), dict):
        return _extract_mapping_messages(d["mapping"])
    # Strategy 2: Claude chat_messages
    if isinstance(d.get("chat_messages"), list):
        return _extract_chat_messages(d["chat_messages"])
    # Strategy 3: Generic messages array
    if isinstance(d.get("messages"), list):
        return _extract_chat_messages(d["messages"])
    # Strategy 4: Kimi/Moonshot segments
    if isinstance(d.get("segments"), list):
        return _extract_segments(d["segments"])
    # Strategy 5: Qwen history pairs [[user, bot], ...]
    if isinstance(d.get("history"), list):
        return _extract_history_pairs(d["history"])
    # Strategy 6: Zhipu/GLM choices[].message
    if isinstance(d.get("choices"), list):
        return _extract_choices(d["choices"])
    # Strategy 7: items sub-array (Kimi exports)
    if isinstance(d.get("items"), list):
        msgs: list[dict[str, Any]] = []
        for item in d["items"]:
            if isinstance(item, dict):
                msgs.extend(_try_extract_messages(item))
        return msgs
    # Strategy 8: prompt/response pair
    prompt = d.get("prompt") or d.get("query") or d.get("input")
    response = d.get("response") or d.get("output") or d.get("answer")
    if prompt is not None or response is not None:
        return _extract_prompt_response(d)
    return []


def _conversation_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        # Unwrap {data: {...}} / {result: {...}} envelopes common in Chinese AI exports.
        inner = _unwrap_data_envelope(payload)
        # If unwrapping found a list (e.g. {data: [conversations...]}), recurse.
        unwrapped_list = inner if inner is not payload else None
        data_list = payload.get("data") or payload.get("result")
        if isinstance(data_list, list):
            return _conversation_rows(data_list)
        # Use inner (unwrapped) dict for extraction.
        target = inner if inner is not payload else payload
        msgs = _try_extract_messages(target)
        if msgs:
            return [
                {
                    "conversation_id": _convo_id_from_dict(payload),
                    "account_scope": _acct_from_dict(payload),
                    "messages": msgs,
                }
            ]
        # If unwrapped dict had no messages but differs from payload, try payload directly.
        if target is not payload:
            msgs = _try_extract_messages(payload)
            if msgs:
                return [
                    {
                        "conversation_id": _convo_id_from_dict(payload),
                        "account_scope": _acct_from_dict(payload),
                        "messages": msgs,
                    }
                ]
        return []

    if isinstance(payload, list):
        out: list[dict[str, Any]] = []
        for i, convo in enumerate(payload):
            if not isinstance(convo, dict):
                continue
            convo_id = _convo_id_from_dict(convo)
            if convo_id == "single":
                convo_id = str(i)
            acct = _acct_from_dict(convo)

            msgs = _try_extract_messages(convo)
            # Try unwrapping data envelope for list items too.
            if not msgs:
                inner = _unwrap_data_envelope(convo)
                if inner is not convo:
                    msgs = _try_extract_messages(inner)

            if msgs:
                out.append(
                    {
                        "conversation_id": convo_id,
                        "account_scope": acct,
                        "messages": msgs,
                    }
                )
        return out

    return []


def iter_records_from_file(path: Path, run_id: str) -> Iterable[CorpusRecord]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return []

    provider = detect_provider_with_payload(path=path, payload=payload)
    conversations = _conversation_rows(payload)

    rows: list[CorpusRecord] = []
    for convo in conversations:
        conversation_id = convo["conversation_id"]
        account_scope = convo["account_scope"]
        for msg in convo["messages"]:
            text = msg.get("text", "")
            if not text.strip():
                continue
            rows.append(
                CorpusRecord(
                    provider=provider,
                    account_scope=account_scope,
                    conversation_id=conversation_id,
                    message_id=str(msg.get("message_id") or ""),
                    role=_role(msg.get("role")),
                    timestamp=int(msg.get("timestamp") or 0),
                    content_hash=_hash_text(text),
                    source_path=str(path),
                    import_run_id=run_id,
                )
            )

    return rows


def iter_candidate_json_files(roots: list[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if path in seen:
                continue
            seen.add(path)
            yield path
