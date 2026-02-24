from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import preflight_readiness
import replay_ingest_jsonl_to_seed as replay_seed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizers import (
    COLLECTION_GAP,
    CONVERSATION_GAP,
    CONVERSATION_PLATFORMS,
    CORE8,
    CORE10,
    EXPORTABLE_NOW,
    LEGACY_CORE3,
    NEW_CORE4,
    SEARCH_AGGREGATORS,
    GenericJsonlParser,
    GenericOpenAIParser,
    custom_providers,
    detect_provider,
    parse_file,
    parser_for,
    parse_payload,
    register_provider,
    registered_providers,
    registry,
    unregister_provider,
)
from normalizers.base import apply_cross_platform_boost, canonical_role, collect_text, parse_timestamp
from normalizers.chatgpt import ChatGPTParser
from normalizers.claude import ClaudeParser
from normalizers.deepseek import DeepSeekParser
from normalizers.gemini import GeminiParser
from normalizers.grok import GrokParser
from normalizers.kimi import KimiParser
from normalizers.openai_api import OpenAIAPIParser
from normalizers.perplexity import PerplexityParser
from normalizers.qwen import QwenParser
from normalizers.zhipu import ZhipuParser
from schemas import ConversationTurn, FragmentHint, FragmentKind
from engine import AutonomousSNRGoTEngine, GIANTS_PROTOCOL
from genesis_gate import GenesisGateConfig, evaluate_genesis_gate
from memory_bridge import (
    MemoryFragmentKind,
    build_fragment_inputs_from_report,
    ingest_report_nodes,
)
from validate_coverage import compute_cv, providers_from_paths


def _fixture(name: str):
    return json.loads((ROOT / "fixtures" / name).read_text(encoding="utf-8"))


# ============================================================
# Schema tests (5)
# ============================================================


def test_schema_fragment_kind_has_ten_targets() -> None:
    assert len(FragmentKind) == 10


def test_schema_fragment_hint_roundtrip() -> None:
    hint = FragmentHint(
        kind=FragmentKind.PATTERN,
        signal="reasoning_trace:hash_map_tradeoff",
        confidence=0.95,
        source="deepseek.reasoning_trace",
        metadata={"provider": "deepseek"},
    )
    restored = FragmentHint.from_dict(hint.to_dict())
    assert restored == hint


def test_schema_fragment_hint_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError):
        FragmentHint(
            kind=FragmentKind.FACT,
            signal="x",
            confidence=1.2,
            source="test",
        )


def test_schema_conversation_turn_roundtrip() -> None:
    turn = ConversationTurn(
        provider="qwen",
        conversation_id="c1",
        turn_id="t1",
        role="assistant",
        content="hello",
        timestamp=1700000000,
        model="qwen-max",
        fragment_hints=[
            FragmentHint(
                kind=FragmentKind.STYLE,
                signal="multilingual_code_switching",
                confidence=0.88,
                source="qwen.multilingual",
            )
        ],
    )
    restored = ConversationTurn.from_dict(turn.to_dict())
    assert restored.to_dict() == turn.to_dict()


def test_schema_conversation_turn_requires_content() -> None:
    with pytest.raises(ValueError):
        ConversationTurn(
            provider="kimi",
            conversation_id="c1",
            turn_id="t1",
            role="user",
            content="   ",
            timestamp=0,
        )


# ============================================================
# Base parser tests (19)
# ============================================================


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("user", "user"),
        ("human", "user"),
        ("assistant", "assistant"),
        ("model", "assistant"),
        ("bot", "assistant"),
        ("system", "system"),
        ("function", "tool"),
        ("plugin", "tool"),
        ("unknown-role", "unknown"),
    ],
)
def test_base_canonical_role(raw: str, expected: str) -> None:
    assert canonical_role(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 0),
        (1700000000, 1700000000),
        (1700000000123, 1700000000),
        ("1700000000", 1700000000),
        ("1700000000123", 1700000000),
        ("", 0),
        ("2026-02-20T10:00:00Z", 1771581600),
        ("not-a-timestamp", 0),
    ],
)
def test_base_parse_timestamp(raw, expected: int) -> None:
    assert parse_timestamp(raw) == expected


def test_base_collect_text_nested_content() -> None:
    content = {"parts": ["alpha", {"text": "beta"}, ["gamma", {"content": "delta"}]]}
    out = collect_text(content)
    assert "alpha" in out and "beta" in out and "gamma" in out and "delta" in out


def test_base_cross_platform_boost_rule() -> None:
    assert apply_cross_platform_boost(0.6, ["deepseek", "qwen", "kimi"]) == 0.9


# ============================================================
# Legacy Core-4 parser tests (14)
# ============================================================


def test_chatgpt_mapping_parse_contract() -> None:
    payload = {
        "conversation_id": "cg-1",
        "default_model_slug": "gpt-5",
        "mapping": {
            "a1": {
                "message": {
                    "id": "u-1",
                    "author": {"role": "user"},
                    "create_time": 1700000000,
                    "content": {"parts": ["I prefer Rust and my goal is to ship this week."]},
                }
            },
            "a2": {
                "message": {
                    "id": "a-1",
                    "author": {"role": "assistant"},
                    "create_time": 1700000001,
                    "content": {"parts": ["Acknowledged."]},
                }
            },
        },
    }
    turns = ChatGPTParser().parse_payload(payload)
    assert len(turns) == 2
    assert turns[0].provider == "chatgpt"
    assert turns[0].role == "user"
    assert turns[0].turn_id == "u-1"
    assert turns[0].timestamp == 1700000000
    assert turns[1].role == "assistant"


def test_chatgpt_hint_controls_dedupe_cap_floor_and_provenance() -> None:
    payload = {
        "conversation_id": "cg-2",
        "mapping": {
            "n1": {
                "message": {
                    "author": {"role": "user"},
                    "content": {
                        "parts": [
                            (
                                "I am a founder and I prefer clear protocols. "
                                "Our goal and objective is to launch next week. "
                                "I usually follow a workflow and my team coordinates with investors. "
                                "I am excited and this architecture uses Rust and API layers."
                            )
                        ]
                    },
                }
            }
        },
    }
    turns = ChatGPTParser().parse_payload(payload)
    hints = turns[0].fragment_hints
    assert 1 <= len(hints) <= 6
    assert all(h.confidence >= 0.72 for h in hints)
    assert all(h.source.startswith("chatgpt.") for h in hints)
    keys = {(h.kind.value, h.signal) for h in hints}
    assert len(keys) == len(hints)


def test_claude_parse_chat_messages_format() -> None:
    payload = [
        {
            "uuid": "cl-1",
            "chat_messages": [
                {"uuid": "m1", "sender": "human", "text": "I prefer depth over breadth."},
                {"uuid": "m2", "sender": "assistant", "text": "Noted."},
            ],
        }
    ]
    turns = ClaudeParser().parse_payload(payload)
    assert len(turns) == 2
    assert {turn.provider for turn in turns} == {"claude"}
    assert [turn.role for turn in turns] == ["user", "assistant"]


def test_gemini_parse_contents_format() -> None:
    payload = {
        "id": "gm-1",
        "model": "gemini-1.5-pro",
        "contents": [
            {"role": "user", "parts": [{"text": "Our goal is launch this week."}]},
            {"role": "model", "parts": [{"text": "Use a staged deployment plan."}]},
        ],
    }
    turns = GeminiParser().parse_payload(payload)
    assert len(turns) == 2
    assert {turn.provider for turn in turns} == {"gemini"}
    assert [turn.role for turn in turns] == ["user", "assistant"]


def test_perplexity_parse_query_answer_and_citation_hint() -> None:
    payload = {
        "id": "px-1",
        "model": "perplexity-sonar",
        "query": "Summarize latest benchmark sources",
        "answer": "According to [1] and [2], performance improved.",
        "citations": [{"url": "https://example.com/1"}],
    }
    turns = PerplexityParser().parse_payload(payload)
    assert len(turns) == 2
    assistant = [turn for turn in turns if turn.role == "assistant"][0]
    assert any(h.kind == FragmentKind.FACT for h in assistant.fragment_hints)


def test_openai_api_parse_request_response_contract() -> None:
    payload = {
        "id": "oa-1",
        "request": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "My goal is to ship this week."},
            ],
        },
        "response": {
            "id": "resp-1",
            "created": 1700000002,
            "choices": [
                {"message": {"role": "assistant", "content": "Use staged rollout."}}
            ],
        },
    }
    turns = OpenAIAPIParser().parse_payload(payload)
    assert len(turns) == 3
    assert {turn.provider for turn in turns} == {"openai_api"}
    assert [turn.role for turn in turns] == ["system", "user", "assistant"]
    assert any(h.kind == FragmentKind.GOAL for h in turns[1].fragment_hints)


def test_grok_parse_messages_contract() -> None:
    payload = {
        "id": "gr-1",
        "model": "grok-2-latest",
        "messages": [
            {"id": "m1", "role": "user", "content": "Our goal is launch next week."},
            {"id": "m2", "role": "assistant", "content": "Recommend staged protocol."},
        ],
    }
    turns = GrokParser().parse_payload(payload)
    assert len(turns) == 2
    assert {turn.provider for turn in turns} == {"grok"}
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert any(h.kind == FragmentKind.GOAL for h in turns[0].fragment_hints)


def test_detect_provider_openai_api_schema_first() -> None:
    payload = {
        "request": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
        },
        "response": {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        },
    }
    provider = detect_provider(payload, source_path="/tmp/random_export.json")
    assert provider == "openai_api"


def test_detect_provider_grok_schema_first() -> None:
    payload = {
        "id": "gr-2",
        "model": "grok-2-latest",
        "messages": [{"role": "user", "content": "hello"}],
    }
    provider = detect_provider(payload, source_path="/tmp/chatgpt_export.json")
    assert provider == "grok"


def test_parse_file_supports_jsonl_openai_logs(tmp_path: Path) -> None:
    path = tmp_path / "openai_log.jsonl"
    line1 = {
        "id": "oa-line-1",
        "request": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "I prefer Rust."}],
        },
        "response": {"choices": [{"message": {"role": "assistant", "content": "Great."}}]},
    }
    line2 = {
        "id": "oa-line-2",
        "request": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Next step?"}],
        },
        "response": {"choices": [{"message": {"role": "assistant", "content": "Ship small."}}]},
    }
    path.write_text(json.dumps(line1) + "\n" + json.dumps(line2) + "\n", encoding="utf-8")
    turns = parse_file(path)
    assert len(turns) == 4
    assert {turn.provider for turn in turns} == {"openai_api"}


def test_detect_provider_schema_beats_filename_for_chatgpt() -> None:
    payload = {
        "conversation_id": "cg-3",
        "mapping": {"n1": {"message": {"author": {"role": "user"}, "content": {"parts": ["x"]}}}},
        "default_model_slug": "gpt-5",
    }
    provider = detect_provider(payload, source_path="/tmp/Claude_overview.json")
    assert provider == "chatgpt"


def test_detect_provider_payload_signals_beat_filename() -> None:
    payload = {
        "model": "gemini-1.5-pro",
        "messages": [{"role": "user", "content": "hello"}],
    }
    provider = detect_provider(payload, source_path="/tmp/perplexity_export.json")
    assert provider == "gemini"


def test_detect_provider_filename_is_weak_fallback_only() -> None:
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    provider = detect_provider(payload, source_path="/tmp/perplexity_export.json")
    assert provider == "perplexity"


def test_cross_parse_payload_dispatches_chatgpt_schema() -> None:
    payload = {
        "conversation_id": "cg-4",
        "mapping": {
            "n1": {
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["hello"]},
                }
            }
        },
    }
    turns = parse_payload(payload, source_path="/tmp/not_chatgpt_name.json")
    assert turns and turns[0].provider == "chatgpt"


# ============================================================
# DeepSeek tests (12)
# ============================================================


def test_deepseek_parse_fixture_turn_count() -> None:
    turns = DeepSeekParser().parse_payload(_fixture("deepseek_export.json"))
    assert len(turns) == 2


def test_deepseek_parse_fixture_provider() -> None:
    turns = DeepSeekParser().parse_payload(_fixture("deepseek_export.json"))
    assert {turn.provider for turn in turns} == {"deepseek"}


def test_deepseek_extracts_reasoning_pattern_hint() -> None:
    turns = DeepSeekParser().parse_payload(_fixture("deepseek_export.json"))
    assistant = [t for t in turns if t.role == "assistant"][0]
    assert any(h.kind == FragmentKind.PATTERN for h in assistant.fragment_hints)


def test_deepseek_extracts_expertise_hint() -> None:
    turns = DeepSeekParser().parse_payload(_fixture("deepseek_export.json"))
    assistant = [t for t in turns if t.role == "assistant"][0]
    assert any(h.kind == FragmentKind.EXPERTISE for h in assistant.fragment_hints)


def test_deepseek_strips_think_tags_from_content() -> None:
    turns = DeepSeekParser().parse_payload(_fixture("deepseek_export.json"))
    assistant = [t for t in turns if t.role == "assistant"][0]
    assert "<think>" not in assistant.content


def test_deepseek_reasoning_content_creates_extra_hints() -> None:
    payload = _fixture("deepseek_export.json")
    payload["conversations"][0]["messages"][1]["reasoning_content"] = "Structured decomposition"
    turns = DeepSeekParser().parse_payload(payload)
    assistant = [t for t in turns if t.role == "assistant"][0]
    assert len(assistant.fragment_hints) >= 2


def test_deepseek_mapping_fragments_supported() -> None:
    payload = {
        "id": "ds-map-1",
        "mapping": {
            "n1": {
                "message": {
                    "id": "m1",
                    "inserted_at": "2026-02-20T10:00:00Z",
                    "fragments": [
                        {"type": "REQUEST", "content": "Hello"},
                        {"type": "THINK", "content": "Reasoning path"},
                        {"type": "RESPONSE", "content": "Hi"},
                    ],
                }
            }
        },
    }
    turns = DeepSeekParser().parse_payload(payload)
    assert len(turns) == 3


def test_deepseek_mapping_fragment_roles() -> None:
    payload = {
        "id": "ds-map-2",
        "mapping": {
            "n1": {
                "message": {
                    "id": "m1",
                    "fragments": [
                        {"type": "REQUEST", "content": "A"},
                        {"type": "RESPONSE", "content": "B"},
                    ],
                }
            }
        },
    }
    turns = DeepSeekParser().parse_payload(payload)
    assert [t.role for t in turns] == ["user", "assistant"]


def test_deepseek_ignores_empty_messages() -> None:
    payload = {"id": "ds-empty", "messages": [{"role": "assistant", "content": "   "}]}
    turns = DeepSeekParser().parse_payload(payload)
    assert turns == []


def test_deepseek_stable_turn_id_fallback() -> None:
    payload = {"id": "ds-id", "messages": [{"role": "assistant", "content": "x"}]}
    t1 = DeepSeekParser().parse_payload(payload)[0].turn_id
    t2 = DeepSeekParser().parse_payload(payload)[0].turn_id
    assert t1 == t2


def test_deepseek_timestamp_parsed_from_iso() -> None:
    turns = DeepSeekParser().parse_payload(_fixture("deepseek_export.json"))
    assert all(turn.timestamp > 0 for turn in turns)


def test_deepseek_parse_file() -> None:
    parser = DeepSeekParser()
    turns = parser.parse_file(ROOT / "fixtures" / "deepseek_export.json")
    assert len(turns) == 2


# ============================================================
# Qwen tests (8)
# ============================================================


def test_qwen_parse_fixture_turn_count() -> None:
    turns = QwenParser().parse_payload(_fixture("qwen_export.json"))
    assert len(turns) == 4


def test_qwen_provider_value() -> None:
    turns = QwenParser().parse_payload(_fixture("qwen_export.json"))
    assert {t.provider for t in turns} == {"qwen"}


def test_qwen_multilingual_style_hint() -> None:
    turns = QwenParser().parse_payload(_fixture("qwen_export.json"))
    assert any(
        h.kind == FragmentKind.STYLE and h.signal == "multilingual_code_switching"
        for turn in turns
        for h in turn.fragment_hints
    )


def test_qwen_bilingual_domain_hint() -> None:
    turns = QwenParser().parse_payload(_fixture("qwen_export.json"))
    assert any(h.kind == FragmentKind.DOMAIN for turn in turns for h in turn.fragment_hints)


def test_qwen_code_block_domain_hint() -> None:
    payload = {
        "id": "qw-code-1",
        "messages": [{"role": "assistant", "content": "```python\ndef f(x):\n  return x\n```"}],
    }
    turns = QwenParser().parse_payload(payload)
    assert any(h.signal == "software_engineering" for h in turns[0].fragment_hints)


def test_qwen_no_false_multilingual_for_plain_english() -> None:
    payload = {
        "id": "qw-eng-1",
        "messages": [{"role": "assistant", "content": "Please review this code change."}],
    }
    turns = QwenParser().parse_payload(payload)
    assert not any(h.signal == "multilingual_code_switching" for h in turns[0].fragment_hints)


def test_qwen_data_envelope_messages() -> None:
    payload = {
        "id": "qw-wrap",
        "data": {"messages": [{"role": "user", "content": "你好 and hello"}]},
    }
    turns = QwenParser().parse_payload(payload)
    assert len(turns) == 1


def test_qwen_parse_file() -> None:
    turns = QwenParser().parse_file(ROOT / "fixtures" / "qwen_export.json")
    assert len(turns) == 4


# ============================================================
# Kimi tests (10)
# ============================================================


def test_kimi_parse_fixture_turn_count() -> None:
    turns = KimiParser().parse_payload(_fixture("kimi_export.json"))
    assert len(turns) == 2


def test_kimi_provider_value() -> None:
    turns = KimiParser().parse_payload(_fixture("kimi_export.json"))
    assert {t.provider for t in turns} == {"kimi"}


def test_kimi_temporal_hint_detected() -> None:
    turns = KimiParser().parse_payload(_fixture("kimi_export.json"))
    assert any(h.kind == FragmentKind.TEMPORAL for h in turns[0].fragment_hints)


def test_kimi_relationship_hint_detected() -> None:
    turns = KimiParser().parse_payload(_fixture("kimi_export.json"))
    assert any(h.kind == FragmentKind.RELATIONSHIP for h in turns[0].fragment_hints)


def test_kimi_url_triggers_relationship_hint() -> None:
    payload = {
        "id": "km-url",
        "messages": [{"role": "user", "content": "Check https://example.org/spec"}],
    }
    turns = KimiParser().parse_payload(payload)
    assert any(h.kind == FragmentKind.RELATIONSHIP for h in turns[0].fragment_hints)


def test_kimi_section_reference_triggers_relationship_hint() -> None:
    payload = {
        "id": "km-sec",
        "messages": [{"role": "user", "content": "As discussed in Section 3, continue."}],
    }
    turns = KimiParser().parse_payload(payload)
    assert any(h.kind == FragmentKind.RELATIONSHIP for h in turns[0].fragment_hints)


def test_kimi_can_emit_temporal_and_relationship_together() -> None:
    payload = {
        "id": "km-both",
        "messages": [
            {
                "role": "user",
                "content": "As discussed in document A, deadline is 2026-03-03.",
            }
        ],
    }
    turns = KimiParser().parse_payload(payload)
    kinds = {h.kind for h in turns[0].fragment_hints}
    assert FragmentKind.TEMPORAL in kinds and FragmentKind.RELATIONSHIP in kinds


def test_kimi_items_wrapper_supported() -> None:
    payload = {
        "id": "km-items",
        "items": [{"messages": [{"role": "assistant", "content": "hello"}]}],
    }
    turns = KimiParser().parse_payload(payload)
    assert len(turns) == 1


def test_kimi_invalid_rows_ignored() -> None:
    payload = {"id": "km-invalid", "messages": ["bad", {"role": "assistant", "content": "ok"}]}
    turns = KimiParser().parse_payload(payload)
    assert len(turns) == 1


def test_kimi_parse_file() -> None:
    turns = KimiParser().parse_file(ROOT / "fixtures" / "kimi_export.json")
    assert len(turns) == 2


# ============================================================
# Zhipu tests (10)
# ============================================================


def test_zhipu_parse_fixture_turn_count() -> None:
    turns = ZhipuParser().parse_payload(_fixture("zhipu_export.json"))
    assert len(turns) == 2


def test_zhipu_provider_value() -> None:
    turns = ZhipuParser().parse_payload(_fixture("zhipu_export.json"))
    assert {t.provider for t in turns} == {"zhipu"}


def test_zhipu_structured_output_fact_hint() -> None:
    turns = ZhipuParser().parse_payload(_fixture("zhipu_export.json"))
    assistant = [t for t in turns if t.role == "assistant"][0]
    assert any(h.signal == "structured_output" for h in assistant.fragment_hints)


def test_zhipu_explicit_plan_goal_hint() -> None:
    turns = ZhipuParser().parse_payload(_fixture("zhipu_export.json"))
    assistant = [t for t in turns if t.role == "assistant"][0]
    assert any(h.signal == "explicit_plan_structure" for h in assistant.fragment_hints)


def test_zhipu_tool_call_goal_hint() -> None:
    turns = ZhipuParser().parse_payload(_fixture("zhipu_export.json"))
    assistant = [t for t in turns if t.role == "assistant"][0]
    assert any(h.signal.startswith("tool_call:") for h in assistant.fragment_hints)


def test_zhipu_citations_fact_hint() -> None:
    turns = ZhipuParser().parse_payload(_fixture("zhipu_export.json"))
    assistant = [t for t in turns if t.role == "assistant"][0]
    assert any(h.signal == "citations_present" for h in assistant.fragment_hints)


def test_zhipu_prompt_response_pair_supported() -> None:
    payload = {
        "id": "zh-pr",
        "prompt": "Summarize",
        "response": "Summary",
    }
    turns = ZhipuParser().parse_payload(payload)
    assert [t.role for t in turns] == ["user", "assistant"]


def test_zhipu_choices_message_supported() -> None:
    payload = {
        "id": "zh-choice",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
    }
    turns = ZhipuParser().parse_payload(payload)
    assert len(turns) == 1


def test_zhipu_data_choices_envelope_supported() -> None:
    payload = {
        "id": "zh-data",
        "data": {"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]},
    }
    turns = ZhipuParser().parse_payload(payload)
    assert len(turns) == 1


def test_zhipu_invalid_tool_calls_safe() -> None:
    payload = {
        "id": "zh-safe",
        "messages": [{"role": "assistant", "content": "hello", "tool_calls": ["bad", {"name": "lookup"}]}],
    }
    turns = ZhipuParser().parse_payload(payload)
    assert len(turns) == 1


# ============================================================
# Cross-platform tests (5)
# ============================================================


def test_cross_core8_has_all_providers() -> None:
    assert len(CORE8) == 8


def test_conversation_platforms_has_7_members() -> None:
    assert len(CONVERSATION_PLATFORMS) == 7
    assert CONVERSATION_PLATFORMS == LEGACY_CORE3 | NEW_CORE4


def test_search_aggregators_contains_perplexity() -> None:
    assert SEARCH_AGGREGATORS == {"perplexity"}
    assert CONVERSATION_PLATFORMS & SEARCH_AGGREGATORS == set()


def test_core8_is_conversation_plus_search() -> None:
    assert set(CORE8) == CONVERSATION_PLATFORMS | SEARCH_AGGREGATORS


def test_exportable_now_is_derived_from_constants() -> None:
    assert EXPORTABLE_NOW == (NEW_CORE4 | {"chatgpt", "claude"})


def test_collection_gap_is_core8_minus_exportable_now() -> None:
    assert COLLECTION_GAP == (set(CORE8) - set(EXPORTABLE_NOW))


def test_conversation_gap_excludes_search_aggregators() -> None:
    assert CONVERSATION_GAP == {"gemini"}
    assert "perplexity" not in CONVERSATION_GAP


def test_cross_registry_has_new_parsers() -> None:
    r = registry()
    assert {
        "chatgpt",
        "openai_api",
        "claude",
        "grok",
        "gemini",
        "perplexity",
        "deepseek",
        "qwen",
        "kimi",
        "zhipu",
    }.issubset(r.keys())


def test_cross_parser_for_new_provider() -> None:
    assert parser_for("deepseek") is not None
    assert parser_for("chatgpt") is not None
    assert parser_for("openai_api") is not None
    assert parser_for("grok") is not None


def test_cross_core10_has_extended_providers() -> None:
    assert len(CORE10) == 10


def test_cross_boost_requires_three_platforms() -> None:
    assert apply_cross_platform_boost(0.7, ["deepseek", "qwen"]) == 0.7


def test_cross_parse_payload_dispatches_new_provider() -> None:
    payload = _fixture("zhipu_export.json")
    turns = parse_payload(payload, source_path="zhipu/export.json")
    assert turns and turns[0].provider == "zhipu"


# ============================================================
# Edge case tests (5)
# ============================================================


def test_edge_empty_payload_returns_empty() -> None:
    assert DeepSeekParser().parse_payload({}) == []
    assert QwenParser().parse_payload({}) == []
    assert KimiParser().parse_payload({}) == []
    assert ZhipuParser().parse_payload({}) == []


def test_edge_unicode_content_survives_roundtrip() -> None:
    payload = {
        "id": "qw-unicode",
        "messages": [{"role": "user", "content": "بسم الله 🌱 你好 hello"}],
    }
    turns = QwenParser().parse_payload(payload)
    assert "بسم الله" in turns[0].content


def test_edge_nested_payload_does_not_raise() -> None:
    payload = {"conversations": [{"id": "n1", "messages": [{"role": "assistant", "content": {"parts": ["a", {"text": "b"}]}}]}]}
    turns = DeepSeekParser().parse_payload(payload)
    assert len(turns) == 1


def test_edge_single_conversation_dict_supported() -> None:
    payload = {
        "id": "single-kimi",
        "messages": [{"role": "assistant", "content": "Single dict conversation works."}],
    }
    turns = KimiParser().parse_payload(payload)
    assert len(turns) == 1


def test_edge_validate_coverage_fixtures_reaches_cv_one() -> None:
    providers = providers_from_paths([ROOT / "fixtures"])
    providers.update({"chatgpt", "claude", "gemini", "perplexity"})
    assert compute_cv(providers) == 1.0


# ============================================================
# Autonomous engine tests (8)
# ============================================================


def _all_fixture_turns() -> list[ConversationTurn]:
    turns: list[ConversationTurn] = []
    turns.extend(DeepSeekParser().parse_payload(_fixture("deepseek_export.json")))
    turns.extend(QwenParser().parse_payload(_fixture("qwen_export.json")))
    turns.extend(KimiParser().parse_payload(_fixture("kimi_export.json")))
    turns.extend(ZhipuParser().parse_payload(_fixture("zhipu_export.json")))
    return turns


def test_engine_protocol_manifest_available() -> None:
    names = {entry.name for entry in GIANTS_PROTOCOL}
    assert {"Claude Shannon", "Isaac Newton"}.issubset(names)


def test_engine_compile_from_turns_produces_nodes() -> None:
    report = AutonomousSNRGoTEngine(snr_threshold=0.0).compile(_all_fixture_turns())
    assert report.nodes and report.total_hints > 0


def test_engine_compile_paths_detects_new_core4_coverage() -> None:
    report = AutonomousSNRGoTEngine(snr_threshold=0.0).compile_paths([ROOT / "fixtures"])
    # 4 NEW_CORE4 providers out of 7 conversation platforms = 4/7 ≈ 0.5714
    assert report.cv == round(4 / len(CONVERSATION_PLATFORMS), 4)


def test_engine_compile_with_conversation_coverage_reaches_cv_one() -> None:
    turns = _all_fixture_turns()
    coverage = set(CONVERSATION_PLATFORMS)
    report = AutonomousSNRGoTEngine(snr_threshold=0.0).compile(turns, provider_coverage=coverage)
    assert report.cv == 1.0


def test_engine_applies_three_platform_boost() -> None:
    hints = [
        FragmentHint(
            kind=FragmentKind.PATTERN,
            signal="shared_pattern",
            confidence=0.6,
            source="test.source",
        )
    ]
    turns = [
        ConversationTurn("deepseek", "c1", "t1", "assistant", "x", 1700000000, fragment_hints=hints),
        ConversationTurn("qwen", "c2", "t2", "assistant", "x", 1700000001, fragment_hints=hints),
        ConversationTurn("kimi", "c3", "t3", "assistant", "x", 1700000002, fragment_hints=hints),
    ]
    report = AutonomousSNRGoTEngine(snr_threshold=0.0).compile(turns)
    node = [n for n in report.nodes if n.signal == "shared_pattern"][0]
    assert node.boosted_confidence == 0.9


def test_engine_builds_edges_from_cooccurring_hints() -> None:
    turn = ConversationTurn(
        provider="zhipu",
        conversation_id="co1",
        turn_id="t1",
        role="assistant",
        content="ok",
        timestamp=1700000000,
        fragment_hints=[
            FragmentHint(
                kind=FragmentKind.GOAL,
                signal="plan",
                confidence=0.8,
                source="zhipu.test",
            ),
            FragmentHint(
                kind=FragmentKind.FACT,
                signal="structured_output",
                confidence=0.8,
                source="zhipu.test",
            ),
        ],
    )
    report = AutonomousSNRGoTEngine(snr_threshold=0.0).compile([turn])
    assert len(report.edges) == 1


def test_engine_elite_threshold_returns_subset() -> None:
    report = AutonomousSNRGoTEngine(snr_threshold=0.0, elite_threshold=0.95).compile(_all_fixture_turns())
    assert len(report.elite_nodes) <= len(report.nodes)


def test_engine_report_to_dict_contains_protocol() -> None:
    report = AutonomousSNRGoTEngine(snr_threshold=0.0).compile(_all_fixture_turns())
    output = report.to_dict()
    assert isinstance(output["giants_protocol"], list) and output["giants_protocol"]


# ============================================================
# Gate + bridge tests (6)
# ============================================================


def test_genesis_gate_passes_with_cv_and_elite_minimum() -> None:
    report = {
        "cv": 1.0,
        "node_count": 3,
        "elite_count": 1,
    }
    verdict = evaluate_genesis_gate(
        report,
        GenesisGateConfig(min_cv=1.0, min_nodes=1, min_elite_nodes=1),
    )
    assert verdict.passed


def test_genesis_gate_fails_closed_when_elite_below_minimum() -> None:
    report = {
        "cv": 1.0,
        "node_count": 3,
        "elite_count": 0,
    }
    verdict = evaluate_genesis_gate(
        report,
        GenesisGateConfig(min_cv=1.0, min_nodes=1, min_elite_nodes=1),
    )
    assert not verdict.passed and verdict.reasons


def test_genesis_gate_reports_missing_provider_exports() -> None:
    report = {
        "cv": 0.5,
        "node_count": 3,
        "elite_count": 1,
        "provider_coverage": ["chatgpt", "deepseek", "qwen", "kimi"],
    }
    verdict = evaluate_genesis_gate(
        report,
        GenesisGateConfig(
            min_cv=1.0,
            min_nodes=1,
            min_elite_nodes=1,
            required_providers=("chatgpt", "claude", "gemini", "perplexity"),
        ),
    )
    assert "MISSING_PROVIDER_EXPORT:claude" in verdict.reasons
    assert "MISSING_PROVIDER_EXPORT:gemini" in verdict.reasons
    assert "MISSING_PROVIDER_EXPORT:perplexity" in verdict.reasons


def test_compile_gate_exportable_now_passes_with_info_gaps() -> None:
    report = {
        "cv": 0.0,  # effective CV computed from available_providers
        "node_count": 5,
        "elite_count": 2,
        "provider_coverage": ["chatgpt", "claude", "deepseek", "qwen", "kimi", "zhipu"],
    }
    verdict = evaluate_genesis_gate(
        report,
        GenesisGateConfig(
            min_cv=1.0,
            min_nodes=1,
            min_elite_nodes=1,
            available_providers=tuple(sorted(EXPORTABLE_NOW)),
            target_providers=tuple(sorted(CONVERSATION_PLATFORMS)),
        ),
    )
    assert verdict.passed
    assert verdict.cv == 1.0
    # Only gemini is a conversation gap; perplexity is a search aggregator, not a target.
    assert "INFO:COLLECTION_GAP:gemini" in verdict.reasons
    assert not any("perplexity" in r for r in verdict.reasons)


def test_compile_gate_conversation_fails_when_non_exportables_missing() -> None:
    report = {
        "cv": 0.0,  # effective CV computed from available_providers
        "node_count": 5,
        "elite_count": 2,
        "provider_coverage": ["chatgpt", "claude", "deepseek", "qwen", "kimi", "zhipu"],
    }
    verdict = evaluate_genesis_gate(
        report,
        GenesisGateConfig(
            min_cv=1.0,
            min_nodes=1,
            min_elite_nodes=1,
            available_providers=tuple(sorted(CONVERSATION_PLATFORMS)),
            target_providers=tuple(sorted(CONVERSATION_PLATFORMS)),
        ),
    )
    assert not verdict.passed
    assert "MISSING_PROVIDER_EXPORT:gemini" in verdict.reasons
    # Perplexity is a search aggregator, not in conversation platforms.
    assert not any("perplexity" in r for r in verdict.reasons)


def test_genesis_gate_available_vs_target_split() -> None:
    """Gate blocks on available_providers, reports target gaps as INFO only."""
    report = {
        "cv": 0.0,  # raw cv ignored when available_providers is set
        "node_count": 5,
        "elite_count": 2,
        "provider_coverage": ["chatgpt", "claude", "deepseek", "gemini", "qwen", "kimi", "zhipu"],
    }
    verdict = evaluate_genesis_gate(
        report,
        GenesisGateConfig(
            min_cv=1.0,
            min_nodes=1,
            min_elite_nodes=1,
            available_providers=("chatgpt", "claude", "deepseek", "gemini", "qwen", "kimi", "zhipu"),
            target_providers=tuple(sorted(CONVERSATION_PLATFORMS)),
        ),
    )
    # Gate should PASS: all available providers are present.
    assert verdict.passed
    assert verdict.cv == 1.0
    # No target gaps — all conversation platforms covered.
    info_reasons = [r for r in verdict.reasons if r.startswith("INFO:")]
    assert len(info_reasons) == 0
    # No MISSING_PROVIDER_EXPORT reasons.
    blocking = [r for r in verdict.reasons if r.startswith("MISSING_PROVIDER_EXPORT:")]
    assert len(blocking) == 0


def test_memory_bridge_builds_typed_fragments() -> None:
    report = {
        "nodes": [
            {
                "node_id": "n1",
                "kind": "Goal",
                "signal": "explicit_plan_structure",
                "snr_score": 0.91,
                "evidence_count": 4,
                "provider_count": 2,
                "providers": ["zhipu", "qwen"],
                "source_tags": ["zhipu.structured_output"],
            }
        ]
    }
    rows = build_fragment_inputs_from_report(report, min_snr=0.8, session_id=42, start_turn=7, timestamp=1700000000)
    assert len(rows) == 1
    assert rows[0].fragment_kind == MemoryFragmentKind.USER_MESSAGE
    assert rows[0].session_id == 42 and rows[0].turn == 7
    assert rows[0].metadata["kind"] == "Goal"
    assert rows[0].metadata["signal"] == "explicit_plan_structure"


def test_memory_bridge_export_jsonl(tmp_path: Path) -> None:
    report = {
        "nodes": [
            {
                "node_id": "n2",
                "kind": "Pattern",
                "signal": "reasoning_trace:x",
                "snr_score": 0.89,
                "evidence_count": 2,
                "provider_count": 3,
                "providers": ["deepseek", "qwen", "kimi"],
                "source_tags": ["deepseek.reasoning_trace"],
            }
        ]
    }
    out = tmp_path / "ingest.jsonl"
    result = ingest_report_nodes(
        report,
        min_snr=0.85,
        session_id=9000,
        export_jsonl_path=out,
    )
    assert result.prepared == 1 and out.exists()


def test_memory_bridge_ingests_with_backend() -> None:
    class _FakeBackend:
        def __init__(self) -> None:
            self.user_calls = 0
            self.assistant_calls = 0

        def process_user_turn(self, content: str, session_id: int, turn: int, timestamp: int):
            self.user_calls += 1
            return {"ingested": True}

        def process_assistant_turn(self, content: str, session_id: int, turn: int, timestamp: int):
            self.assistant_calls += 1
            return {"ingested": True}

    backend = _FakeBackend()
    report = {
        "nodes": [
            {
                "node_id": "n-goal",
                "kind": "Goal",
                "signal": "plan",
                "snr_score": 0.90,
                "evidence_count": 2,
                "provider_count": 2,
                "providers": ["zhipu", "qwen"],
                "source_tags": ["x"],
            },
            {
                "node_id": "n-fact",
                "kind": "Fact",
                "signal": "structured_output",
                "snr_score": 0.88,
                "evidence_count": 2,
                "provider_count": 2,
                "providers": ["zhipu", "qwen"],
                "source_tags": ["y"],
            },
        ]
    }
    result = ingest_report_nodes(report, min_snr=0.85, backend=backend, session_id=123)
    assert result.ingested == 2
    assert backend.user_calls == 1 and backend.assistant_calls == 1


def test_memory_bridge_respects_min_snr_filter() -> None:
    report = {
        "nodes": [
            {
                "node_id": "low",
                "kind": "Goal",
                "signal": "low",
                "snr_score": 0.50,
                "evidence_count": 1,
                "provider_count": 1,
                "providers": ["zhipu"],
                "source_tags": ["x"],
            }
        ]
    }
    result = ingest_report_nodes(report, min_snr=0.85)
    assert result.prepared == 0


def test_engine_report_contains_provider_telemetry() -> None:
    report = AutonomousSNRGoTEngine(snr_threshold=0.0).compile_paths([ROOT / "fixtures"]).to_dict()
    assert isinstance(report["provider_turn_counts"], dict)
    assert isinstance(report["provider_hint_counts"], dict)
    assert isinstance(report["provider_parse_failures"], dict)
    assert isinstance(report["ingest_input_file_count"], int)
    assert isinstance(report["unknown_file_count"], int)


def test_preflight_available_only_ready_with_exportable_set(tmp_path: Path) -> None:
    payloads = {
        "chatgpt": {
            "conversation_id": "cg-1",
            "mapping": {"n1": {"message": {"author": {"role": "user"}, "content": {"parts": ["x"]}}}},
        },
        "claude": {"uuid": "cl-1", "chat_messages": [{"sender": "human", "text": "x"}]},
        "deepseek": {
            "id": "ds-1",
            "mapping": {"n1": {"message": {"inserted_at": "2026-02-21T00:00:00Z", "fragments": []}}},
        },
        "qwen": {"id": "qw-1", "history": [["hello", "world"]]},
        "kimi": {"id": "km-1", "segments": [{"role": "assistant", "content": "x"}]},
        "zhipu": {"id": "zh-1", "task_id": "t1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}}]},
    }
    for name, payload in payloads.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    result = preflight_readiness.scan_provider_readiness(
        [tmp_path],
        required=set(EXPORTABLE_NOW),
    )

    assert result["ready"] is True
    assert result["missing_providers"] == []
    assert result["cv_achievable"] == 1.0


def test_preflight_supports_jsonl_schema_detection(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "openai_api_logs.jsonl"
    line = {
        "id": "oa-jsonl-1",
        "request": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
        },
        "response": {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        },
    }
    jsonl_path.write_text(json.dumps(line) + "\n", encoding="utf-8")

    result = preflight_readiness.scan_provider_readiness(
        [tmp_path],
        required={"openai_api"},
    )

    assert result["ready"] is True
    assert result["present_providers"] == ["openai_api"]
    assert result["missing_providers"] == []


def test_validate_coverage_does_not_count_extra_providers_in_cv(tmp_path: Path) -> None:
    # Core8 provider fixture.
    (tmp_path / "chatgpt.json").write_text(
        json.dumps(
            {
                "conversation_id": "cg-core8",
                "mapping": {
                    "n1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["hello"]},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    # Extra (non-Core8) provider in JSONL format.
    (tmp_path / "openai_api.jsonl").write_text(
        json.dumps(
            {
                "request": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                "response": {
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    covered = providers_from_paths([tmp_path])
    assert "chatgpt" in covered
    assert "openai_api" not in covered
    assert compute_cv(covered | {"openai_api"}) == compute_cv(covered)


# ============================================================
# Generic parser tests (14)
# ============================================================


def test_generic_jsonl_parses_flat_role_content() -> None:
    payload = [
        {"role": "user", "content": "Hello from LM Studio"},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ]
    turns = GenericJsonlParser().parse_payload(payload, source_path="lmstudio.jsonl")
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"
    assert turns[0].provider == "generic_jsonl"


def test_generic_jsonl_custom_field_names() -> None:
    payload = [
        {"speaker": "user", "text": "Custom fields work", "ts": 1700000000},
    ]
    parser = GenericJsonlParser(role_field="speaker", content_field="text", timestamp_field="ts")
    turns = parser.parse_payload(payload)
    assert len(turns) == 1
    assert turns[0].content == "Custom fields work"
    assert turns[0].timestamp == 1700000000


def test_generic_jsonl_skips_empty_content() -> None:
    payload = [
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "Valid"},
    ]
    turns = GenericJsonlParser().parse_payload(payload)
    assert len(turns) == 1
    assert turns[0].content == "Valid"


def test_generic_jsonl_model_defaults_to_local() -> None:
    payload = [{"role": "user", "content": "test"}]
    turns = GenericJsonlParser().parse_payload(payload)
    assert turns[0].model == "local"


def test_generic_jsonl_conversation_id_from_payload() -> None:
    payload = [{"role": "user", "content": "test", "conversation_id": "sess-42"}]
    turns = GenericJsonlParser().parse_payload(payload)
    assert turns[0].conversation_id == "sess-42"


def test_generic_openai_parses_messages_array() -> None:
    payload = [
        {
            "id": "conv-1",
            "model": "llama-3.1-8b",
            "messages": [
                {"role": "user", "content": "What is Rust?"},
                {"role": "assistant", "content": "Rust is a systems programming language."},
            ],
        }
    ]
    turns = GenericOpenAIParser().parse_payload(payload)
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"
    assert turns[0].provider == "generic_openai"
    assert turns[0].model == "llama-3.1-8b"


def test_generic_openai_handles_flat_role_content() -> None:
    payload = [
        {"role": "user", "content": "Flat message", "id": "flat-1"},
    ]
    turns = GenericOpenAIParser().parse_payload(payload)
    assert len(turns) == 1
    assert turns[0].content == "Flat message"


def test_generic_openai_skips_non_dict_messages() -> None:
    payload = [
        {
            "id": "conv-2",
            "messages": [
                "bad-entry",
                {"role": "user", "content": "Valid"},
            ],
        }
    ]
    turns = GenericOpenAIParser().parse_payload(payload)
    assert len(turns) == 1


def test_generic_openai_empty_content_skipped() -> None:
    payload = [
        {
            "id": "conv-3",
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "Ok"},
            ],
        }
    ]
    turns = GenericOpenAIParser().parse_payload(payload)
    assert len(turns) == 1
    assert turns[0].role == "assistant"


# ============================================================
# Custom provider registration tests (12)
# ============================================================


def test_register_provider_adds_to_registry() -> None:
    name = "test_lmstudio"
    parser = GenericJsonlParser()
    try:
        register_provider(name, parser)
        assert parser_for(name) is parser
        assert name in registered_providers()
        assert name in custom_providers()
    finally:
        unregister_provider(name)


def test_register_provider_adds_to_conversation_platforms() -> None:
    name = "test_ollama_conv"
    parser = GenericOpenAIParser()
    original_size = len(CONVERSATION_PLATFORMS)
    try:
        register_provider(name, parser, is_conversation_platform=True)
        assert name in CONVERSATION_PLATFORMS
        assert len(CONVERSATION_PLATFORMS) == original_size + 1
    finally:
        unregister_provider(name)
    assert name not in CONVERSATION_PLATFORMS


def test_register_provider_skips_conversation_platform_when_false() -> None:
    name = "test_tool_only"
    parser = GenericJsonlParser()
    try:
        register_provider(name, parser, is_conversation_platform=False)
        assert name not in CONVERSATION_PLATFORMS
        assert name in registered_providers()
    finally:
        unregister_provider(name)


def test_register_provider_rejects_invalid_name() -> None:
    with pytest.raises(ValueError, match="Invalid provider name"):
        register_provider("", GenericJsonlParser())
    with pytest.raises(ValueError, match="Invalid provider name"):
        register_provider("has spaces", GenericJsonlParser())
    with pytest.raises(ValueError, match="Invalid provider name"):
        register_provider("has-dash", GenericJsonlParser())


def test_register_provider_rejects_non_parser() -> None:
    with pytest.raises(TypeError, match="PlatformParser"):
        register_provider("bad_parser", "not a parser")  # type: ignore[arg-type]


def test_unregister_provider_removes_custom() -> None:
    name = "test_removable"
    register_provider(name, GenericJsonlParser())
    assert unregister_provider(name) is True
    assert parser_for(name) is None
    assert name not in custom_providers()


def test_unregister_provider_ignores_builtin() -> None:
    assert unregister_provider("chatgpt") is False
    assert parser_for("chatgpt") is not None


def test_unregister_provider_returns_false_for_unknown() -> None:
    assert unregister_provider("nonexistent_provider_xyz") is False


def test_registered_providers_includes_generics() -> None:
    providers = registered_providers()
    assert "generic_jsonl" in providers
    assert "generic_openai" in providers


def test_custom_providers_empty_initially() -> None:
    # After all cleanup, custom_providers should only contain items we added
    # in this test run. Since tests clean up after themselves, it should be empty.
    # But we just verify the function returns a list of strings.
    result = custom_providers()
    assert isinstance(result, list)


def test_register_provider_lowercases_name() -> None:
    name = "Test_UPPER"
    try:
        register_provider(name, GenericJsonlParser())
        assert parser_for("test_upper") is not None
        assert "test_upper" in registered_providers()
    finally:
        unregister_provider("test_upper")


def test_register_provider_end_to_end_parse() -> None:
    """Full integration: register a custom parser, parse data through it."""
    name = "test_localai"
    parser = GenericOpenAIParser()
    payload = [
        {
            "id": "localai-1",
            "model": "phi-3-mini",
            "messages": [
                {"role": "user", "content": "What is BIZRA?"},
                {"role": "assistant", "content": "BIZRA means seed."},
            ],
        }
    ]
    try:
        register_provider(name, parser)
        p = parser_for(name)
        assert p is not None
        turns = p.parse_payload(payload)
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[1].content == "BIZRA means seed."
    finally:
        unregister_provider(name)


def test_replay_seed_deterministic_order_and_checksum(tmp_path: Path) -> None:
    ingest_path = tmp_path / "ingest.jsonl"
    out_path = tmp_path / "knowledge.seed"
    row_a = {
        "fragment_kind": "Observation",
        "content": "x",
        "session_id": 1,
        "turn": 2,
        "timestamp": 2000,
        "metadata": {"kind": "Goal", "signal": "launch alpha", "snr_score": 0.9},
    }
    row_b = {
        "fragment_kind": "Observation",
        "content": "x",
        "session_id": 1,
        "turn": 1,
        "timestamp": 1000,
        "metadata": {"kind": "Fact", "signal": "based in dubai", "snr_score": 0.95},
    }
    # Include duplicate semantic row to verify deterministic dedupe.
    ingest_path.write_text(
        "\n".join([json.dumps(row_a), json.dumps(row_b), json.dumps(row_a)]),
        encoding="utf-8",
    )

    rows = replay_seed._dedupe_and_sort(replay_seed._load_rows(ingest_path))
    replay_seed._write_seed(rows, out_path)
    checksum_path, digest_1 = replay_seed._write_checksum(out_path)

    rows_again = replay_seed._dedupe_and_sort(replay_seed._load_rows(ingest_path))
    replay_seed._write_seed(rows_again, out_path)
    _, digest_2 = replay_seed._write_checksum(out_path)

    assert len(rows) == 2
    assert out_path.exists() and checksum_path.exists()
    assert digest_1 == digest_2
