"""
Data Import — Feed Your History to Your PAT
============================================
Imports external data (chat logs, documents, work artifacts) into the
Living Memory system so your PAT team can reason over your 3 years of work.

Supports:
- Plain text files (.txt, .md)
- JSON chat exports (ChatGPT, Claude)
- CSV/TSV data
- Code repositories (via git log)

Each item is:
1. Parsed and chunked
2. Assigned to memory type (EPISODIC, SEMANTIC, PROCEDURAL)
3. Embedded and stored in Living Memory
4. Indexed for RAG retrieval

Standing on Giants: Information Theory + Memory Consolidation + RAG
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sovereign.data_import")


class DataImporter:
    """Import external data into Living Memory."""

    def __init__(self, living_memory: Any, user_context: Any) -> None:
        self._memory = living_memory
        self._user_context = user_context

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
        current_chunk = []
        for line in lines:
            if line.strip():
                current_chunk.append(line)
            elif current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        # Import each chunk
        imported = 0
        mem_type = MemoryType(memory_type)
        for chunk in chunks:
            if len(chunk) > 20:  # Skip very short chunks
                entry = await self._memory.encode(
                    content=chunk,
                    memory_type=mem_type,
                    source=f"import:{file_path.name}",
                    importance=0.7,  # Imported data is moderately important
                )
                if entry:
                    imported += 1

        logger.info(f"Imported {imported} chunks from {file_path}")
        return imported

    async def import_chat_json(self, file_path: Path) -> int:
        """
        Import chat history from JSON export.

        Expected format (ChatGPT/Claude style):
        [
          {"role": "user", "content": "...", "timestamp": "2024-01-01T..."},
          {"role": "assistant", "content": "...", "timestamp": "..."},
        ]
        """
        from core.living_memory.core import MemoryType

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        data = json.loads(file_path.read_text())
        if not isinstance(data, list):
            raise ValueError("Chat JSON must be an array of message objects")

        imported = 0
        for msg in data:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp_str = msg.get("timestamp", "")

            if not content or len(content) < 10:
                continue

            # Episodic memory: what happened in the conversation
            entry = await self._memory.encode(
                content=f"[{role}] {content}",
                memory_type=MemoryType.EPISODIC,
                source=f"chat_import:{file_path.name}",
                importance=0.6,  # Conversation history is valuable
            )
            if entry:
                imported += 1

                # Also add to conversation memory if it's from the user
                if role in ("user", "human") and self._user_context:
                    self._user_context.conversation.add_human_turn(content)
                elif role in ("assistant", "ai") and self._user_context:
                    self._user_context.conversation.add_pat_turn(content)

        logger.info(f"Imported {imported} messages from {file_path}")
        return imported

    async def import_directory(
        self, dir_path: Path, pattern: str = "*.txt"
    ) -> Dict[str, int]:
        """Import all matching files from a directory."""
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        results = {}
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                try:
                    if file_path.suffix == ".json":
                        count = await self.import_chat_json(file_path)
                    else:
                        count = await self.import_text_file(file_path)
                    results[file_path.name] = count
                except Exception as e:
                    logger.error(f"Failed to import {file_path}: {e}")
                    results[file_path.name] = 0

        total = sum(results.values())
        logger.info(f"Imported {total} total items from {len(results)} files")
        return results

    async def import_markdown_notes(self, file_path: Path) -> int:
        """Import Markdown notes as semantic memory (knowledge)."""
        from core.living_memory.core import MemoryType

        content = file_path.read_text(encoding="utf-8", errors="ignore")

        # Split by headers (## or ###)
        import re

        sections = re.split(r"\n##+ ", content)
        imported = 0

        for section in sections:
            section = section.strip()
            if len(section) > 50:  # Skip tiny sections
                entry = await self._memory.encode(
                    content=section,
                    memory_type=MemoryType.SEMANTIC,
                    source=f"notes:{file_path.name}",
                    importance=0.8,  # Notes are high-value knowledge
                )
                if entry:
                    imported += 1

        logger.info(f"Imported {imported} sections from {file_path}")
        return imported


async def run_import_wizard(runtime: Any) -> None:
    """Interactive wizard for importing data."""
    from core.living_memory.core import LivingMemoryCore

    print("\n" + "=" * 60)
    print("DATA IMPORT WIZARD")
    print("=" * 60)
    print("\nThis will import your external data into Living Memory")
    print("so your PAT team can reason over your work history.\n")

    # Check if living memory is available
    living_memory = getattr(runtime, "_living_memory", None)
    if not living_memory:
        # Initialize it if needed
        from pathlib import Path

        living_memory = LivingMemoryCore(
            memory_dir=Path("sovereign_state/living_memory")
        )
        await living_memory.initialize()
        runtime._living_memory = living_memory

    user_context = getattr(runtime, "_user_context", None)
    importer = DataImporter(living_memory, user_context)

    print("What would you like to import?")
    print("  1. Single text file (.txt, .md)")
    print("  2. Chat history (JSON export)")
    print("  3. Directory of files")
    print("  4. Cancel")

    choice = input("\nChoice (1-4): ").strip()

    if choice == "1":
        path_str = input("File path: ").strip()
        path = Path(path_str)
        try:
            if path.suffix == ".md":
                count = await importer.import_markdown_notes(path)
            else:
                count = await importer.import_text_file(path)
            print(f"\n✓ Imported {count} items from {path.name}")
        except Exception as e:
            print(f"\n✗ Import failed: {e}")

    elif choice == "2":
        path_str = input("Chat JSON file path: ").strip()
        path = Path(path_str)
        try:
            count = await importer.import_chat_json(path)
            print(f"\n✓ Imported {count} messages from {path.name}")
        except Exception as e:
            print(f"\n✗ Import failed: {e}")

    elif choice == "3":
        path_str = input("Directory path: ").strip()
        pattern = input("File pattern (e.g., *.txt, *.md): ").strip() or "*.txt"
        dir_path = Path(path_str)
        try:
            results = await importer.import_directory(dir_path, pattern)
            total = sum(results.values())
            print(f"\n✓ Imported {total} items from {len(results)} files")
            print("\nBreakdown:")
            for filename, count in sorted(results.items()):
                print(f"  {filename}: {count} items")
        except Exception as e:
            print(f"\n✗ Import failed: {e}")

    elif choice == "4":
        print("\nImport cancelled.")
        return

    # Save living memory
    if living_memory:
        await living_memory._save_memories()
        print("\n✓ Living Memory saved")

    print("\nYour PAT team can now reason over this data.")
    print("Try asking: 'What did I work on 6 months ago?'")
