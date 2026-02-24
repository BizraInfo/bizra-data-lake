from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    module_path = Path("scripts/corpus/provider_normalizers.py").resolve()
    spec = importlib.util.spec_from_file_location("provider_normalizers", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------


def test_detect_provider_from_payload_signal() -> None:
    m = _load_module()
    payload = {"provider": "zhipu", "messages": [{"role": "user", "content": "hello"}]}
    provider = m.detect_provider_with_payload(Path("tmp/random-export.json"), payload)
    assert provider == "zhipu"


def test_no_provider_inference_from_free_text_only() -> None:
    m = _load_module()
    payload = {"messages": [{"role": "user", "content": "compare qwen and kimi models"}]}
    provider = m.detect_provider_with_payload(Path("tmp/random-export.json"), payload)
    assert provider == "generic"


def test_detect_qwen_from_path() -> None:
    m = _load_module()
    assert m.detect_provider(Path("/data/qwen-export/conversations.json")) == "qwen"


def test_detect_kimi_from_path() -> None:
    m = _load_module()
    assert m.detect_provider(Path("/data/kimi-chats/export.json")) == "kimi"
    assert m.detect_provider(Path("/data/moonshot/export.json")) == "kimi"


def test_detect_zhipu_from_path() -> None:
    m = _load_module()
    assert m.detect_provider(Path("/data/zhipu/export.json")) == "zhipu"
    assert m.detect_provider(Path("/data/glm-export/chat.json")) == "zhipu"


def test_detect_qwen_from_model_signal() -> None:
    m = _load_module()
    payload = {"model": "qwen-max", "messages": [{"role": "user", "content": "hi"}]}
    assert m.detect_provider_with_payload(Path("tmp/export.json"), payload) == "qwen"


def test_detect_kimi_from_model_signal() -> None:
    m = _load_module()
    payload = {"model": "moonshot-v1-8k", "messages": [{"role": "user", "content": "hi"}]}
    assert m.detect_provider_with_payload(Path("tmp/export.json"), payload) == "kimi"


def test_detect_zhipu_from_model_signal() -> None:
    m = _load_module()
    payload = {"model": "glm-4", "messages": [{"role": "user", "content": "hi"}]}
    assert m.detect_provider_with_payload(Path("tmp/export.json"), payload) == "zhipu"


def test_detect_all_core8_from_signals() -> None:
    """Every Core-8 provider is detectable from model name signals."""
    m = _load_module()
    signal_map = {
        "chatgpt_openai": "gpt-4",
        "claude": "claude-3-opus",
        "gemini_google": "gemini-pro",
        "deepseek": "deepseek-chat",
        "qwen": "qwen-turbo",
        "kimi": "moonshot-v1-32k",
        "perplexity": "perplexity",
        "zhipu": "chatglm_turbo",
    }
    for expected_provider, model_name in signal_map.items():
        payload = {"model": model_name, "messages": []}
        detected = m.detect_provider_with_payload(Path("tmp/export.json"), payload)
        assert detected == expected_provider, f"Expected {expected_provider} for model={model_name}, got {detected}"


# ---------------------------------------------------------------------------
# DeepSeek parsing (existing, expanded)
# ---------------------------------------------------------------------------


def test_deepseek_fragment_mapping_is_parsed(tmp_path: Path) -> None:
    m = _load_module()
    fixture = [
        {
            "id": "conv-1",
            "mapping": {
                "node-1": {
                    "id": "node-1",
                    "message": {
                        "id": "msg-1",
                        "model": "deepseek-chat",
                        "inserted_at": "2026-02-20T10:00:00+00:00",
                        "fragments": [
                            {"type": "REQUEST", "content": "Hello"},
                            {"type": "THINK", "content": "Reasoning"},
                            {"type": "RESPONSE", "content": "Hi there"},
                        ],
                    },
                }
            },
        }
    ]
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 3
    assert {r.provider for r in records} == {"deepseek"}
    assert [r.role for r in records] == ["user", "assistant", "assistant"]
    assert all(r.content_hash for r in records)


# ---------------------------------------------------------------------------
# Qwen parsing
# ---------------------------------------------------------------------------


def test_qwen_messages_format(tmp_path: Path) -> None:
    """Qwen standard export with messages array."""
    m = _load_module()
    fixture = [
        {
            "id": "qwen-conv-1",
            "title": "Math question",
            "model": "qwen-max",
            "messages": [
                {"role": "user", "content": "What is 2+2?", "created_at": "2026-01-15T08:00:00Z"},
                {"role": "assistant", "content": "2+2 equals 4.", "created_at": "2026-01-15T08:00:01Z"},
            ],
        }
    ]
    path = tmp_path / "qwen-export" / "conversations.json"
    path.parent.mkdir()
    path.write_text(json.dumps(fixture), encoding="utf-8")

    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 2
    assert {r.provider for r in records} == {"qwen"}
    assert records[0].role == "user"
    assert records[1].role == "assistant"
    assert records[0].conversation_id == "qwen-conv-1"


def test_qwen_history_pairs_format(tmp_path: Path) -> None:
    """Qwen history-style export with [[user, bot], ...] pairs."""
    m = _load_module()
    fixture = {
        "id": "qwen-hist-1",
        "model": "qwen-plus",
        "history": [
            ["Explain recursion", "Recursion is when a function calls itself."],
            ["Give an example", "def factorial(n): return 1 if n<=1 else n*factorial(n-1)"],
        ],
    }
    path = tmp_path / "qwen-export" / "history.json"
    path.parent.mkdir()
    path.write_text(json.dumps(fixture), encoding="utf-8")

    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 4
    assert {r.provider for r in records} == {"qwen"}
    roles = [r.role for r in records]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_qwen_data_envelope(tmp_path: Path) -> None:
    """Qwen export wrapped in {data: {messages: [...]}}."""
    m = _load_module()
    fixture = {
        "id": "qwen-wrapped-1",
        "model": "qwen-turbo",
        "data": {
            "messages": [
                {"role": "user", "content": "Hello Qwen"},
                {"role": "assistant", "content": "Hello! How can I help?"},
            ]
        },
    }
    path = tmp_path / "qwen-export" / "wrapped.json"
    path.parent.mkdir()
    path.write_text(json.dumps(fixture), encoding="utf-8")

    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 2
    assert {r.provider for r in records} == {"qwen"}


# ---------------------------------------------------------------------------
# Kimi / Moonshot parsing
# ---------------------------------------------------------------------------


def test_kimi_messages_format(tmp_path: Path) -> None:
    """Kimi standard export with messages array."""
    m = _load_module()
    fixture = [
        {
            "id": "kimi-conv-1",
            "title": "Code review",
            "model": "moonshot-v1-8k",
            "messages": [
                {"role": "user", "content": "Review this code", "created_at": 1708000000},
                {"role": "assistant", "content": "The code looks clean. Two suggestions...", "created_at": 1708000001},
            ],
        }
    ]
    path = tmp_path / "kimi-export" / "conversations.json"
    path.parent.mkdir()
    path.write_text(json.dumps(fixture), encoding="utf-8")

    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 2
    assert {r.provider for r in records} == {"kimi"}
    assert records[0].role == "user"
    assert records[1].role == "assistant"


def test_kimi_segments_format(tmp_path: Path) -> None:
    """Kimi export with segments array instead of messages."""
    m = _load_module()
    fixture = {
        "id": "kimi-seg-1",
        "kimiplus_id": "kp-abc123",
        "model": "moonshot-v1-32k",
        "segments": [
            {"id": "seg-1", "role": "user", "content": "Summarize this paper", "created_at": "2026-01-10T10:00:00Z"},
            {"id": "seg-2", "role": "assistant", "content": "The paper presents a novel approach to...", "created_at": "2026-01-10T10:00:02Z"},
            {"id": "seg-3", "role": "user", "content": "What about the methodology?", "created_at": "2026-01-10T10:01:00Z"},
            {"id": "seg-4", "role": "assistant", "content": "The methodology section describes...", "created_at": "2026-01-10T10:01:05Z"},
        ],
    }
    path = tmp_path / "kimi-export" / "segments.json"
    path.parent.mkdir()
    path.write_text(json.dumps(fixture), encoding="utf-8")

    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 4
    assert {r.provider for r in records} == {"kimi"}
    assert [r.role for r in records] == ["user", "assistant", "user", "assistant"]
    assert records[0].conversation_id == "kimi-seg-1"


def test_kimi_items_wrapper(tmp_path: Path) -> None:
    """Kimi export with items sub-array wrapping conversations."""
    m = _load_module()
    fixture = {
        "id": "kimi-batch-1",
        "model": "moonshot-v1-8k",
        "items": [
            {
                "messages": [
                    {"role": "user", "content": "What is RLHF?"},
                    {"role": "assistant", "content": "RLHF stands for Reinforcement Learning from Human Feedback."},
                ]
            }
        ],
    }
    path = tmp_path / "kimi-export" / "items.json"
    path.parent.mkdir()
    path.write_text(json.dumps(fixture), encoding="utf-8")

    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 2
    assert {r.provider for r in records} == {"kimi"}


# ---------------------------------------------------------------------------
# Zhipu / ChatGLM parsing
# ---------------------------------------------------------------------------


def test_zhipu_messages_format(tmp_path: Path) -> None:
    """Zhipu standard export with messages array."""
    m = _load_module()
    fixture = [
        {
            "id": "zhipu-conv-1",
            "model": "glm-4",
            "messages": [
                {"role": "user", "content": "Translate to English: AI is amazing"},
                {"role": "assistant", "content": "AI is amazing"},
            ],
        }
    ]
    path = tmp_path / "zhipu-export" / "conversations.json"
    path.parent.mkdir()
    path.write_text(json.dumps(fixture), encoding="utf-8")

    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 2
    assert {r.provider for r in records} == {"zhipu"}
    assert records[0].role == "user"
    assert records[1].role == "assistant"


def test_zhipu_choices_format(tmp_path: Path) -> None:
    """Zhipu API-style export with choices[].message format."""
    m = _load_module()
    fixture = {
        "task_id": "zhipu-task-1",
        "request_id": "req-abc",
        "model": "glm-4",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "The answer is 42."}},
        ],
    }
    path = tmp_path / "zhipu-export" / "api_response.json"
    path.parent.mkdir()
    path.write_text(json.dumps(fixture), encoding="utf-8")

    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 1
    assert records[0].provider == "zhipu"
    assert records[0].role == "assistant"
    assert records[0].conversation_id == "zhipu-task-1"


def test_zhipu_prompt_response_format(tmp_path: Path) -> None:
    """Zhipu export with prompt/response pair format."""
    m = _load_module()
    fixture = {
        "invocation_id": "zhipu-inv-1",
        "model": "chatglm_turbo",
        "prompt": "Explain quantum computing",
        "response": "Quantum computing uses quantum-mechanical phenomena such as superposition and entanglement.",
        "created_at": "2026-01-20T14:30:00Z",
    }
    path = tmp_path / "zhipu-export" / "prompt_response.json"
    path.parent.mkdir()
    path.write_text(json.dumps(fixture), encoding="utf-8")

    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 2
    assert {r.provider for r in records} == {"zhipu"}
    assert records[0].role == "user"
    assert records[1].role == "assistant"
    assert records[0].conversation_id == "zhipu-inv-1"


def test_zhipu_data_choices_envelope(tmp_path: Path) -> None:
    """Zhipu export with {data: {choices: [...]}} envelope."""
    m = _load_module()
    fixture = {
        "task_id": "zhipu-task-2",
        "model": "glm-3-turbo",
        "data": {
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "Here is the analysis."}},
            ]
        },
    }
    path = tmp_path / "zhipu-export" / "wrapped.json"
    path.parent.mkdir()
    path.write_text(json.dumps(fixture), encoding="utf-8")

    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 1
    assert records[0].provider == "zhipu"


# ---------------------------------------------------------------------------
# Cross-platform and edge cases
# ---------------------------------------------------------------------------


def test_chatgpt_standard_export(tmp_path: Path) -> None:
    """Standard ChatGPT export with mapping structure."""
    m = _load_module()
    fixture = [
        {
            "id": "chatgpt-conv-1",
            "title": "Hello World",
            "mapping": {
                "n1": {
                    "id": "n1",
                    "message": {
                        "id": "msg-1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hello, ChatGPT!"]},
                        "create_time": 1700000000,
                    },
                },
                "n2": {
                    "id": "n2",
                    "message": {
                        "id": "msg-2",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Hello! How can I help you today?"]},
                        "create_time": 1700000001,
                    },
                },
            },
        }
    ]
    path = tmp_path / "chatgpt-conversations.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 2
    assert {r.provider for r in records} == {"chatgpt_openai"}
    assert records[0].role == "user"
    assert records[1].role == "assistant"


def test_claude_chat_messages_export(tmp_path: Path) -> None:
    """Claude export with chat_messages array."""
    m = _load_module()
    fixture = [
        {
            "uuid": "claude-conv-1",
            "name": "Code help",
            "chat_messages": [
                {"uuid": "cm-1", "sender": "human", "text": "Help me write a function", "created_at": "2026-01-05T09:00:00Z"},
                {"uuid": "cm-2", "sender": "assistant", "text": "Here is a function that...", "created_at": "2026-01-05T09:00:03Z"},
            ],
        }
    ]
    path = tmp_path / "claude-chats" / "export.json"
    path.parent.mkdir()
    path.write_text(json.dumps(fixture), encoding="utf-8")

    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 2
    assert {r.provider for r in records} == {"claude"}
    assert records[0].role == "user"
    assert records[1].role == "assistant"


def test_perplexity_messages_export(tmp_path: Path) -> None:
    """Perplexity export with messages array."""
    m = _load_module()
    fixture = [
        {
            "id": "pplx-conv-1",
            "model": "perplexity",
            "messages": [
                {"role": "user", "content": "What is the latest news on AI?"},
                {"role": "assistant", "content": "Here are the latest developments..."},
            ],
        }
    ]
    path = tmp_path / "perplexity-export" / "threads.json"
    path.parent.mkdir()
    path.write_text(json.dumps(fixture), encoding="utf-8")

    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 2
    assert {r.provider for r in records} == {"perplexity"}


def test_gemini_messages_export(tmp_path: Path) -> None:
    """Gemini export with messages array."""
    m = _load_module()
    fixture = [
        {
            "id": "gemini-conv-1",
            "model": "gemini-pro",
            "messages": [
                {"role": "user", "content": "Explain transformers"},
                {"role": "model", "content": "Transformers are a type of neural network architecture..."},
            ],
        }
    ]
    path = tmp_path / "gemini-export" / "chats.json"
    path.parent.mkdir()
    path.write_text(json.dumps(fixture), encoding="utf-8")

    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 2
    assert {r.provider for r in records} == {"gemini_google"}


def test_empty_conversation_produces_no_records(tmp_path: Path) -> None:
    """Empty messages array produces zero records."""
    m = _load_module()
    fixture = [{"id": "empty-1", "messages": []}]
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 0


def test_malformed_json_produces_no_records(tmp_path: Path) -> None:
    """Malformed JSON returns empty list."""
    m = _load_module()
    path = tmp_path / "bad.json"
    path.write_text("not json at all", encoding="utf-8")
    records = list(m.iter_records_from_file(path, run_id="test-run"))
    assert len(records) == 0


def test_content_hash_deterministic(tmp_path: Path) -> None:
    """Same content in different conversations produces same content_hash."""
    m = _load_module()
    fixture1 = [{"id": "c1", "messages": [{"role": "user", "content": "Hello world"}]}]
    fixture2 = [{"id": "c2", "messages": [{"role": "user", "content": "Hello world"}]}]
    p1 = tmp_path / "f1.json"
    p2 = tmp_path / "f2.json"
    p1.write_text(json.dumps(fixture1), encoding="utf-8")
    p2.write_text(json.dumps(fixture2), encoding="utf-8")

    r1 = list(m.iter_records_from_file(p1, run_id="r1"))
    r2 = list(m.iter_records_from_file(p2, run_id="r2"))
    assert r1[0].content_hash == r2[0].content_hash


def test_content_hash_whitespace_normalization(tmp_path: Path) -> None:
    """Content hashing normalizes whitespace."""
    m = _load_module()
    fixture1 = [{"id": "c1", "messages": [{"role": "user", "content": "Hello   world"}]}]
    fixture2 = [{"id": "c2", "messages": [{"role": "user", "content": "Hello world"}]}]
    p1 = tmp_path / "f1.json"
    p2 = tmp_path / "f2.json"
    p1.write_text(json.dumps(fixture1), encoding="utf-8")
    p2.write_text(json.dumps(fixture2), encoding="utf-8")

    r1 = list(m.iter_records_from_file(p1, run_id="r1"))
    r2 = list(m.iter_records_from_file(p2, run_id="r2"))
    assert r1[0].content_hash == r2[0].content_hash


def test_role_normalization() -> None:
    """Various role strings normalize to canonical values."""
    m = _load_module()
    assert m._role("user") == "user"
    assert m._role("human") == "user"
    assert m._role("Human") == "user"
    assert m._role("assistant") == "assistant"
    assert m._role("ai") == "assistant"
    assert m._role("model") == "assistant"
    assert m._role("AI") == "assistant"
    assert m._role("system") == "system"
    assert m._role("tool") == "tool"
    assert m._role("function") == "tool"
    assert m._role(None) == "unknown"
    assert m._role("") == "unknown"


def test_timestamp_parsing() -> None:
    """Various timestamp formats are parsed correctly."""
    m = _load_module()
    assert m._parse_ts(1700000000) == 1700000000
    assert m._parse_ts("1700000000") == 1700000000
    assert m._parse_ts(1700000000.5) == 1700000000
    assert m._parse_ts(None) == 0
    assert m._parse_ts("") == 0
    ts = m._parse_ts("2026-01-15T08:00:00Z")
    assert ts > 0
    ts2 = m._parse_ts("2026-01-15T08:00:00+00:00")
    assert ts == ts2


def test_core8_set_complete() -> None:
    """CORE8 constant contains exactly 8 providers (conversation + search)."""
    m = _load_module()
    assert len(m.CORE8) == 8
    expected = {"chatgpt_openai", "claude", "gemini_google", "deepseek", "qwen", "kimi", "perplexity", "zhipu"}
    assert m.CORE8 == expected


def test_conversation_platforms_excludes_search_aggregators() -> None:
    """CONVERSATION_PLATFORMS contains 7 identity-building conversation platforms."""
    m = _load_module()
    assert len(m.CONVERSATION_PLATFORMS) == 7
    assert "perplexity" not in m.CONVERSATION_PLATFORMS
    assert m.SEARCH_AGGREGATORS == {"perplexity"}
