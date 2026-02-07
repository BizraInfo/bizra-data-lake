"""
Data Import — Feed Your History to Your PAT
============================================
Imports external data (chat logs, documents, work artifacts) into the
Living Memory system so your PAT team can reason over your 3 years of work.

Supports:
- ChatGPT exports (tree-format mapping with parent/children)
- DeepSeek exports (same OpenAI mapping format)
- ChatGPT memories.json (AI's memory of the user)
- Plain text files (.txt, .md)
- Individual conversation JSONs

Each conversation is:
1. Parsed from tree/mapping to flat message list
2. Chunked into conversation segments (groups of turns)
3. Stored as EPISODIC memory in Living Memory
4. Key insights extracted as SEMANTIC memory

Standing on Giants: Tulving (memory types) + Shannon (information) + RAG
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sovereign.data_import")


# =============================================================================
# CONVERSATION PARSERS
# =============================================================================


def parse_chatgpt_mapping(data: dict) -> List[Dict[str, str]]:
    """
    Parse ChatGPT tree-format conversation into flat message list.

    ChatGPT exports conversations as a tree of nodes with parent/children
    relationships. We traverse from root to current_node to get the
    linear conversation thread.
    """
    mapping = data.get("mapping", {})
    if not mapping:
        return []

    # Find root node (no parent or parent not in mapping)
    root_id = None
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if parent is None or parent not in mapping:
            root_id = node_id
            break

    if not root_id:
        return []

    # Traverse from root following children (take first child at each level)
    messages = []
    current_id = root_id
    visited = set()

    while current_id and current_id not in visited:
        visited.add(current_id)
        node = mapping.get(current_id)
        if not node:
            break

        msg = node.get("message")
        if msg:
            author = msg.get("author", {}).get("role", "unknown")
            content = msg.get("content", {})
            parts = content.get("parts", [])
            # Extract text parts (skip image/file references)
            text_parts = [p for p in parts if isinstance(p, str) and p.strip()]
            text = "\n".join(text_parts)

            if text and len(text) > 5 and author in ("user", "assistant"):
                timestamp = msg.get("create_time")
                messages.append({
                    "role": "human" if author == "user" else "assistant",
                    "content": text,
                    "timestamp": timestamp,
                })

        # Follow first child
        children = node.get("children", [])
        current_id = children[0] if children else None

    return messages


def parse_chatgpt_bulk(data: list) -> List[Dict[str, Any]]:
    """
    Parse ChatGPT bulk export (conversations.json).

    Returns list of {title, created, messages: [{role, content}]}.
    """
    conversations = []
    for conv in data:
        title = conv.get("name") or conv.get("title") or "Untitled"
        created = conv.get("created_at") or conv.get("create_time")

        # Try chat_messages first (newer format)
        chat_msgs = conv.get("chat_messages", [])
        if chat_msgs:
            messages = []
            for m in chat_msgs:
                role = m.get("role", "unknown")
                content = m.get("content", "")
                if isinstance(content, dict):
                    parts = content.get("parts", [])
                    content = "\n".join(p for p in parts if isinstance(p, str))
                if content and len(content) > 5 and role in ("user", "assistant"):
                    messages.append({
                        "role": "human" if role == "user" else "assistant",
                        "content": content,
                    })
            if messages:
                conversations.append({
                    "title": title,
                    "created": created,
                    "messages": messages,
                })
            continue

        # Fall back to mapping format (tree structure)
        messages = parse_chatgpt_mapping(conv)
        if messages:
            conversations.append({
                "title": title,
                "created": created,
                "messages": messages,
            })

    return conversations


def chunk_conversation(
    title: str, messages: List[Dict[str, str]], chunk_size: int = 6
) -> List[str]:
    """
    Chunk a conversation into segments for memory storage.

    Groups messages into chunks of `chunk_size` turns, with title prefix.
    Each chunk is self-contained enough for RAG retrieval.
    """
    chunks = []
    for i in range(0, len(messages), chunk_size):
        segment = messages[i : i + chunk_size]
        lines = [f"[Conversation: {title}]"]
        for msg in segment:
            role = "Human" if msg["role"] == "human" else "AI"
            content = msg["content"]
            # Truncate very long messages to keep chunks manageable
            if len(content) > 2000:
                content = content[:2000] + "..."
            lines.append(f"{role}: {content}")
        chunks.append("\n".join(lines))
    return chunks


# =============================================================================
# DATA IMPORTER
# =============================================================================


class DataImporter:
    """Import external data into Living Memory."""

    def __init__(self, living_memory: Any, user_context: Any) -> None:
        self._memory = living_memory
        self._user_context = user_context
        self._stats = {
            "conversations": 0,
            "messages": 0,
            "chunks_stored": 0,
            "memories_imported": 0,
            "skipped": 0,
            "errors": 0,
        }

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    async def import_chatgpt_export(self, export_dir: Path) -> Dict[str, int]:
        """
        Import a full ChatGPT data export directory.

        Handles:
        - conversations.json (bulk conversations)
        - memories.json (ChatGPT's memory of the user)
        - Individual conversation JSONs in subdirectories
        """
        results = {}

        # 1. Import memories.json (highest value — AI's memory of the user)
        memories_file = export_dir / "memories.json"
        if memories_file.exists():
            count = await self._import_chatgpt_memories(memories_file)
            results["memories.json"] = count

        # 2. Import conversations.json (bulk export)
        convos_file = export_dir / "conversations.json"
        if convos_file.exists():
            count = await self._import_conversations_file(convos_file)
            results["conversations.json"] = count

        # 3. Import individual conversation JSONs from subdirectories
        for subdir in export_dir.iterdir():
            if subdir.is_dir():
                json_files = list(subdir.glob("*.json"))
                if json_files:
                    sub_total = 0
                    for jf in json_files:
                        try:
                            count = await self._import_single_conversation(jf)
                            sub_total += count
                        except Exception as e:
                            logger.warning(f"Skipping {jf.name}: {e}")
                            self._stats["errors"] += 1
                    if sub_total:
                        results[subdir.name] = sub_total

        return results

    async def import_deepseek_export(self, export_dir: Path) -> Dict[str, int]:
        """Import a DeepSeek data export directory."""
        results = {}

        convos_file = export_dir / "conversations.json"
        if convos_file.exists():
            count = await self._import_conversations_file(convos_file)
            results["conversations.json"] = count

        return results

    async def _import_chatgpt_memories(self, file_path: Path) -> int:
        """
        Import ChatGPT's memory of the user into profile + semantic memory.

        This is gold — it's what ChatGPT learned about the user over months/years.
        """
        from core.living_memory.core import MemoryType

        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            return 0

        imported = 0
        for memory_block in data:
            # conversations_memory: ChatGPT's full memory text
            conv_memory = memory_block.get("conversations_memory", "")
            if conv_memory:
                # Store as high-importance semantic memory
                # Split into sections by ** headers
                sections = conv_memory.split("\n\n")
                for section in sections:
                    section = section.strip()
                    if len(section) > 30:
                        entry = await self._memory.encode(
                            content=section,
                            memory_type=MemoryType.SEMANTIC,
                            source="chatgpt_memory",
                            importance=0.95,  # Highest — AI's distilled knowledge of user
                            emotional_weight=0.8,
                        )
                        if entry:
                            imported += 1

            # project_memories: ChatGPT project-specific memories
            proj_memories = memory_block.get("project_memories", [])
            if isinstance(proj_memories, list):
                for pm in proj_memories:
                    content = pm if isinstance(pm, str) else json.dumps(pm)
                    if len(content) > 30:
                        entry = await self._memory.encode(
                            content=content,
                            memory_type=MemoryType.SEMANTIC,
                            source="chatgpt_project_memory",
                            importance=0.9,
                        )
                        if entry:
                            imported += 1

        self._stats["memories_imported"] += imported
        logger.info(f"Imported {imported} memory entries from ChatGPT memories")
        return imported

    async def _import_conversations_file(self, file_path: Path) -> int:
        """Import a conversations.json file (ChatGPT or DeepSeek format)."""
        from core.living_memory.core import MemoryType

        logger.info(f"Loading conversations from {file_path.name}...")
        data = json.loads(file_path.read_text(encoding="utf-8"))

        if not isinstance(data, list):
            return 0

        conversations = parse_chatgpt_bulk(data)
        logger.info(f"Parsed {len(conversations)} conversations with content")

        total_chunks = 0
        for conv in conversations:
            title = conv["title"]
            messages = conv["messages"]
            self._stats["conversations"] += 1
            self._stats["messages"] += len(messages)

            # Chunk and store
            chunks = chunk_conversation(title, messages)
            for chunk in chunks:
                if len(chunk) > 50:
                    entry = await self._memory.encode(
                        content=chunk,
                        memory_type=MemoryType.EPISODIC,
                        source=f"chat:{title[:50]}",
                        importance=0.7,
                    )
                    if entry:
                        total_chunks += 1
                        self._stats["chunks_stored"] += 1
                    else:
                        self._stats["skipped"] += 1

        logger.info(f"Stored {total_chunks} chunks from {len(conversations)} conversations")
        return total_chunks

    async def _import_single_conversation(self, file_path: Path) -> int:
        """Import a single conversation JSON file (ChatGPT individual export)."""
        from core.living_memory.core import MemoryType

        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return 0

        title = data.get("title", file_path.stem.split("-", 4)[-1].replace("_", " "))
        messages = parse_chatgpt_mapping(data)

        if not messages:
            return 0

        self._stats["conversations"] += 1
        self._stats["messages"] += len(messages)

        chunks = chunk_conversation(title, messages)
        stored = 0
        for chunk in chunks:
            if len(chunk) > 50:
                entry = await self._memory.encode(
                    content=chunk,
                    memory_type=MemoryType.EPISODIC,
                    source=f"chat:{title[:50]}",
                    importance=0.7,
                )
                if entry:
                    stored += 1
                    self._stats["chunks_stored"] += 1

        return stored

    async def import_text_file(
        self, file_path: Path, memory_type: str = "semantic"
    ) -> int:
        """Import a plain text file as semantic memory."""
        from core.living_memory.core import MemoryType

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.strip().split("\n")

        # Chunk into paragraphs (blank line delimited)
        chunks = []
        current_chunk: List[str] = []
        for line in lines:
            if line.strip():
                current_chunk.append(line)
            elif current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        imported = 0
        mem_type = MemoryType(memory_type)
        for chunk in chunks:
            if len(chunk) > 20:
                entry = await self._memory.encode(
                    content=chunk,
                    memory_type=mem_type,
                    source=f"import:{file_path.name}",
                    importance=0.7,
                )
                if entry:
                    imported += 1

        logger.info(f"Imported {imported} chunks from {file_path}")
        return imported

    async def import_markdown_notes(self, file_path: Path) -> int:
        """Import Markdown notes as semantic memory."""
        from core.living_memory.core import MemoryType

        import re

        content = file_path.read_text(encoding="utf-8", errors="ignore")
        sections = re.split(r"\n##+ ", content)
        imported = 0

        for section in sections:
            section = section.strip()
            if len(section) > 50:
                entry = await self._memory.encode(
                    content=section,
                    memory_type=MemoryType.SEMANTIC,
                    source=f"notes:{file_path.name}",
                    importance=0.8,
                )
                if entry:
                    imported += 1

        logger.info(f"Imported {imported} sections from {file_path}")
        return imported


# =============================================================================
# BATCH INGESTION PIPELINE
# =============================================================================


async def ingest_chat_history(
    import_dir: Path,
    living_memory: Any,
    user_context: Any = None,
) -> Dict[str, Any]:
    """
    Batch ingest all chat history from an extracted export directory.

    Automatically detects ChatGPT vs DeepSeek format and processes accordingly.

    Args:
        import_dir: Directory containing extracted chat export
        living_memory: LivingMemoryCore instance
        user_context: Optional UserContextManager for profile enrichment

    Returns:
        Dict with ingestion statistics
    """
    importer = DataImporter(living_memory, user_context)
    all_results: Dict[str, int] = {}

    # Walk the directory tree looking for data sources
    for item in import_dir.rglob("*"):
        if not item.is_dir():
            continue

        # ChatGPT bulk export (has conversations.json + memories.json)
        if (item / "conversations.json").exists() and (item / "memories.json").exists():
            logger.info(f"Found ChatGPT export: {item.name}")
            results = await importer.import_chatgpt_export(item)
            all_results.update({f"chatgpt/{k}": v for k, v in results.items()})
            continue

        # DeepSeek export (has conversations.json + user.json)
        if (item / "conversations.json").exists() and (item / "user.json").exists():
            logger.info(f"Found DeepSeek export: {item.name}")
            results = await importer.import_deepseek_export(item)
            all_results.update({f"deepseek/{k}": v for k, v in results.items()})
            continue

        # Directory of individual conversation JSONs
        json_files = list(item.glob("*.json"))
        if len(json_files) > 5:
            logger.info(f"Found conversation directory: {item.name} ({len(json_files)} files)")
            for jf in json_files:
                try:
                    count = await importer._import_single_conversation(jf)
                    if count:
                        all_results[f"individual/{jf.stem[:40]}"] = count
                except Exception as e:
                    logger.warning(f"Skipping {jf.name}: {e}")

    stats = importer.get_stats()
    stats["sources"] = all_results
    return stats


async def run_import_wizard(runtime: Any) -> None:
    """Interactive wizard for importing data."""
    from core.living_memory.core import LivingMemoryCore

    print("\n" + "=" * 60)
    print("DATA IMPORT WIZARD")
    print("=" * 60)
    print("\nThis will import your chat history into Living Memory")
    print("so your PAT team can reason over your work history.\n")

    living_memory = getattr(runtime, "_living_memory", None)
    if not living_memory:
        living_memory = LivingMemoryCore(
            memory_dir=Path("sovereign_state/living_memory")
        )
        await living_memory.initialize()
        runtime._living_memory = living_memory

    user_context = getattr(runtime, "_user_context", None)

    print("What would you like to import?")
    print("  1. Chat history directory (ChatGPT/DeepSeek exports)")
    print("  2. Single text file (.txt, .md)")
    print("  3. Cancel")

    choice = input("\nChoice (1-3): ").strip()

    if choice == "1":
        path_str = input("Export directory path: ").strip()
        import_dir = Path(path_str)
        if not import_dir.is_dir():
            print(f"\n  Not a directory: {import_dir}")
            return

        print(f"\nIngesting from {import_dir}...")
        stats = await ingest_chat_history(import_dir, living_memory, user_context)

        print(f"\n  Conversations: {stats['conversations']}")
        print(f"  Messages: {stats['messages']}")
        print(f"  Chunks stored: {stats['chunks_stored']}")
        print(f"  Memories imported: {stats['memories_imported']}")
        print(f"  Skipped: {stats['skipped']}")
        if stats['errors']:
            print(f"  Errors: {stats['errors']}")

    elif choice == "2":
        path_str = input("File path: ").strip()
        path = Path(path_str)
        importer = DataImporter(living_memory, user_context)
        try:
            if path.suffix == ".md":
                count = await importer.import_markdown_notes(path)
            else:
                count = await importer.import_text_file(path)
            print(f"\n  Imported {count} items from {path.name}")
        except Exception as e:
            print(f"\n  Import failed: {e}")

    elif choice == "3":
        print("\nImport cancelled.")
        return

    if living_memory:
        await living_memory._save_memories()
        print("\n  Living Memory saved")

    print("\nYour PAT team can now reason over this data.")
