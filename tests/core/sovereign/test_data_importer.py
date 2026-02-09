import pytest
from types import SimpleNamespace

from core.sovereign.data_import import DataImporter


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
