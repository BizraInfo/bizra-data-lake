from types import SimpleNamespace

import pytest

from core.sovereign.data_import import (
    DataImporter,
    ImportScanLimits,
    ingest_chat_history,
)


class DummyMemory:
    def __init__(self) -> None:
        self.entries = []

    async def encode(self, *, content, memory_type, source, importance, **kwargs):
        self.entries.append(
            {
                "content": content,
                "memory_type": memory_type,
                "source": source,
                "importance": importance,
            }
        )
        return SimpleNamespace(id=str(len(self.entries)))


def _make_conversation(title: str, num_messages: int) -> dict:
    long_text = "x" * 120
    messages = []
    for i in range(num_messages):
        role = "human" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"{long_text}-{i}"})
    return {"title": title, "messages": messages}


@pytest.mark.asyncio
async def test_store_conversations_chunks_and_stats():
    memory = DummyMemory()
    importer = DataImporter(living_memory=memory, user_context=None)

    conversations = [
        _make_conversation("First", num_messages=7),
        _make_conversation("Second", num_messages=5),
    ]

    total_messages = sum(len(conv["messages"]) for conv in conversations)
    expected_chunks = sum((len(conv["messages"]) + 5) // 6 for conv in conversations)

    stored_chunks = await importer._store_conversations(
        conversations, source_prefix="test-src"
    )

    stats = importer.get_stats()

    assert stored_chunks == expected_chunks
    assert stats["conversations"] == len(conversations)
    assert stats["messages"] == total_messages
    assert stats["chunks_stored"] == stored_chunks
    assert stats["skipped"] == 0
    assert len(memory.entries) == stored_chunks

    for entry in memory.entries:
        assert entry["source"].startswith("test-src:")


@pytest.mark.asyncio
async def test_ingest_chat_history_respects_max_depth(tmp_path):
    memory = DummyMemory()
    deep_export = tmp_path / "nested" / "export"
    deep_export.mkdir(parents=True)
    (deep_export / "conversations.json").write_text("[]", encoding="utf-8")
    (deep_export / "memories.json").write_text("[]", encoding="utf-8")

    stats = await ingest_chat_history(
        tmp_path,
        memory,
        scan_limits=ImportScanLimits(max_depth=0),
    )

    assert stats["sources"] == {}
    assert stats["scan_audit"]["limit_hit"] is True
    assert any(
        skipped["reason"] == "max_depth" for skipped in stats["scan_audit"]["skipped"]
    )


@pytest.mark.asyncio
async def test_ingest_chat_history_respects_json_byte_budget(tmp_path):
    memory = DummyMemory()
    (tmp_path / "conversations.json").write_text(
        '[{"title":"Large","mapping":{}}]',
        encoding="utf-8",
    )
    (tmp_path / "memories.json").write_text("[]", encoding="utf-8")

    stats = await ingest_chat_history(
        tmp_path,
        memory,
        scan_limits=ImportScanLimits(max_total_json_bytes=8),
    )

    assert stats["sources"] == {}
    assert stats["scan_audit"]["limit_hit"] is True
    assert any(
        skipped["reason"] == "max_total_json_bytes"
        for skipped in stats["scan_audit"]["skipped"]
    )


@pytest.mark.asyncio
async def test_ingest_chat_history_counts_export_json_once(tmp_path):
    memory = DummyMemory()
    conversations = tmp_path / "conversations.json"
    memories = tmp_path / "memories.json"
    conversations.write_text("[]", encoding="utf-8")
    memories.write_text("[]", encoding="utf-8")
    exact_budget = conversations.stat().st_size + memories.stat().st_size

    stats = await ingest_chat_history(
        tmp_path,
        memory,
        scan_limits=ImportScanLimits(max_total_json_bytes=exact_budget),
    )

    assert stats["sources"] == {
        "chatgpt/memories.json": 0,
        "chatgpt/conversations.json": 0,
    }
    assert stats["scan_audit"]["total_json_bytes"] == exact_budget
    assert stats["scan_audit"]["limit_hit"] is False


@pytest.mark.asyncio
async def test_ingest_chat_history_rejects_named_json_symlinks(tmp_path):
    memory = DummyMemory()
    import_root = tmp_path / "import"
    import_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "conversations.json").write_text("[]", encoding="utf-8")
    (outside / "memories.json").write_text("[]", encoding="utf-8")
    (import_root / "conversations.json").symlink_to(outside / "conversations.json")
    (import_root / "memories.json").symlink_to(outside / "memories.json")

    stats = await ingest_chat_history(import_root, memory)

    assert stats["sources"] == {}
    assert stats["scan_audit"]["limit_hit"] is True
    assert any(
        skipped["reason"] == "symlink_file"
        for skipped in stats["scan_audit"]["skipped"]
    )


@pytest.mark.asyncio
async def test_ingest_chat_history_bounds_chatgpt_subdirectories(tmp_path):
    memory = DummyMemory()
    (tmp_path / "conversations.json").write_text("[]", encoding="utf-8")
    (tmp_path / "memories.json").write_text("[]", encoding="utf-8")
    child = tmp_path / "child"
    child.mkdir()
    (child / "conversation.json").write_text(
        '{"title":"Nested","mapping":{}}',
        encoding="utf-8",
    )

    stats = await ingest_chat_history(
        tmp_path,
        memory,
        scan_limits=ImportScanLimits(max_depth=0),
    )

    assert stats["sources"] == {
        "chatgpt/memories.json": 0,
        "chatgpt/conversations.json": 0,
    }
    assert any(
        skipped["reason"] == "max_depth" for skipped in stats["scan_audit"]["skipped"]
    )
