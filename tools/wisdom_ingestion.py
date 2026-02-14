#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    WISDOM INGESTION ENGINE                                    ║
║                                                                              ║
║  The House of Wisdom's intake protocol.                                       ║
║                                                                              ║
║  Purpose: Transform external knowledge into indexed wisdom.                   ║
║           - Repos → Pattern extraction                                        ║
║           - Links → Content distillation                                      ║
║           - Ideas → Concept mapping                                           ║
║           - Files → Structured knowledge entries                              ║
║                                                                              ║
║  This is what the Data Lake DOES. Not chains. Not coordination.              ║
║  INGEST → PROCESS → INDEX → SERVE                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import hashlib
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PATHS = {
    "knowledge": Path(r"C:\BIZRA-DATA-LAKE\03_INDEXED\knowledge"),
    "graph": Path(r"C:\BIZRA-DATA-LAKE\03_INDEXED\graph"),
    "gold": Path(r"C:\BIZRA-DATA-LAKE\04_GOLD"),
    "intake": Path(r"C:\BIZRA-DATA-LAKE\00_INTAKE"),
}

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("WisdomIngestion")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KnowledgeEntry:
    """A unit of wisdom in the Data Lake."""
    id: str
    type: str  # organization, technology, pattern, insight, concept, reference
    name: str
    description: str
    source: str
    ingested_at: str
    tags: List[str]
    lesson_for_bizra: Optional[str] = None
    parent: Optional[str] = None
    relationships: Optional[List[Dict[str, str]]] = None
    
    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class WisdomExtraction:
    """Result of processing an external source."""
    source_type: str  # repo, link, idea, file
    source_url: str
    entries: List[KnowledgeEntry]
    patterns_found: List[str]
    lessons_extracted: List[str]
    processing_time: float


# ═══════════════════════════════════════════════════════════════════════════════
# WISDOM INGESTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class WisdomIngestionEngine:
    """
    The intake system for the House of Wisdom.
    
    Transforms raw external knowledge into structured, queryable wisdom.
    """
    
    def __init__(self):
        self.knowledge_store = PATHS["knowledge"]
        self.knowledge_store.mkdir(parents=True, exist_ok=True)
        self.entry_count = 0
        log.info("📚 Wisdom Ingestion Engine initialized")
    
    def generate_id(self, prefix: str, name: str) -> str:
        """Generate a unique ID for a knowledge entry."""
        slug = re.sub(r'[^a-z0-9]+', '_', name.lower())[:30]
        hash_suffix = hashlib.blake2b(f"{prefix}:{name}".encode(), digest_size=16).hexdigest()[:6]
        return f"{prefix}_{slug}_{hash_suffix}"
    
    def ingest_concept(
        self,
        name: str,
        description: str,
        source: str,
        entry_type: str = "concept",
        tags: List[str] = None,
        lesson: str = None,
        parent: str = None,
    ) -> KnowledgeEntry:
        """
        Ingest a single concept into the knowledge base.
        
        This is the atomic unit of wisdom ingestion.
        """
        entry = KnowledgeEntry(
            id=self.generate_id(entry_type[:3], name),
            type=entry_type,
            name=name,
            description=description,
            source=source,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            tags=tags or [],
            lesson_for_bizra=lesson,
            parent=parent,
        )
        self.entry_count += 1
        return entry
    
    def ingest_repo_analysis(
        self,
        repo_url: str,
        repo_name: str,
        analysis: Dict[str, Any]
    ) -> WisdomExtraction:
        """
        Ingest analysis from an external repository.
        
        Expected analysis format:
        {
            "technologies": [...],
            "patterns": [...],
            "architecture": {...},
            "lessons": [...],
        }
        """
        import time
        start = time.time()
        entries = []
        
        # Create parent entry for the repo
        repo_entry = self.ingest_concept(
            name=repo_name,
            description=f"External repository analyzed for patterns and wisdom.",
            source=repo_url,
            entry_type="repository",
            tags=["external", "reference", "analyzed"],
        )
        entries.append(repo_entry)
        
        # Extract technologies
        for tech in analysis.get("technologies", []):
            entry = self.ingest_concept(
                name=tech.get("name", "Unknown"),
                description=tech.get("description", ""),
                source=repo_url,
                entry_type="technology",
                tags=tech.get("tags", []),
                lesson=tech.get("lesson"),
                parent=repo_entry.id,
            )
            entries.append(entry)
        
        # Extract patterns
        for pattern in analysis.get("patterns", []):
            entry = self.ingest_concept(
                name=pattern.get("name", "Unknown Pattern"),
                description=pattern.get("description", ""),
                source=repo_url,
                entry_type="pattern",
                tags=["pattern"] + pattern.get("tags", []),
                lesson=pattern.get("lesson"),
                parent=repo_entry.id,
            )
            entries.append(entry)
        
        # Extract lessons
        lessons = analysis.get("lessons", [])
        for i, lesson in enumerate(lessons):
            entry = self.ingest_concept(
                name=f"Lesson {i+1} from {repo_name}",
                description=lesson,
                source=repo_url,
                entry_type="insight",
                tags=["lesson", "extracted"],
                parent=repo_entry.id,
            )
            entries.append(entry)
        
        elapsed = time.time() - start
        
        return WisdomExtraction(
            source_type="repo",
            source_url=repo_url,
            entries=entries,
            patterns_found=[p.get("name", "") for p in analysis.get("patterns", [])],
            lessons_extracted=lessons,
            processing_time=elapsed,
        )
    
    def ingest_link_content(
        self,
        url: str,
        title: str,
        content_summary: str,
        entities_found: List[Dict[str, str]] = None,
        lessons: List[str] = None,
    ) -> WisdomExtraction:
        """
        Ingest distilled content from a web link.
        
        The agent should have already fetched and summarized the content.
        This method structures it into queryable knowledge.
        """
        import time
        start = time.time()
        entries = []
        
        # Main link entry
        link_entry = self.ingest_concept(
            name=title,
            description=content_summary,
            source=url,
            entry_type="reference",
            tags=["web", "external", "link"],
        )
        entries.append(link_entry)
        
        # Extract entities
        for entity in (entities_found or []):
            entry = self.ingest_concept(
                name=entity.get("name", "Unknown"),
                description=entity.get("description", ""),
                source=url,
                entry_type=entity.get("type", "concept"),
                tags=entity.get("tags", []),
                lesson=entity.get("lesson"),
                parent=link_entry.id,
            )
            entries.append(entry)
        
        elapsed = time.time() - start
        
        return WisdomExtraction(
            source_type="link",
            source_url=url,
            entries=entries,
            patterns_found=[],
            lessons_extracted=lessons or [],
            processing_time=elapsed,
        )
    
    def ingest_idea(
        self,
        idea_title: str,
        idea_description: str,
        author: str = "MoMo",
        tags: List[str] = None,
        related_concepts: List[str] = None,
    ) -> KnowledgeEntry:
        """
        Ingest a raw idea from the BIZRA architect.
        
        Ideas are first-class citizens in the House of Wisdom.
        """
        entry = self.ingest_concept(
            name=idea_title,
            description=idea_description,
            source=f"author:{author}",
            entry_type="idea",
            tags=["idea", "original"] + (tags or []),
        )
        
        if related_concepts:
            entry.relationships = [
                {"type": "related_to", "target": c} for c in related_concepts
            ]
        
        return entry
    
    def save_extraction(self, extraction: WisdomExtraction, filename: str = None) -> Path:
        """
        Persist a wisdom extraction to the knowledge store.
        """
        if not filename:
            slug = re.sub(r'[^a-z0-9]+', '_', extraction.source_url.lower())[:40]
            filename = f"{slug}.jsonl"
        
        output_path = self.knowledge_store / filename
        
        with open(output_path, 'a', encoding='utf-8') as f:
            for entry in extraction.entries:
                f.write(entry.to_jsonl() + '\n')
        
        log.info(f"💾 Saved {len(extraction.entries)} entries to {filename}")
        return output_path
    
    def query_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Simple keyword search across knowledge entries.
        
        For production: integrate with vector_engine.py for semantic search.
        """
        results = []
        query_lower = query.lower()
        
        for jsonl_file in self.knowledge_store.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        # Simple relevance check
                        text = f"{entry.get('name', '')} {entry.get('description', '')}".lower()
                        if query_lower in text:
                            results.append(entry)
                            if len(results) >= limit:
                                return results
                    except json.JSONDecodeError:
                        continue
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        total_entries = 0
        files = list(self.knowledge_store.glob("*.jsonl"))
        
        for jsonl_file in files:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                total_entries += sum(1 for line in f if line.strip())
        
        return {
            "knowledge_files": len(files),
            "total_entries": total_entries,
            "knowledge_path": str(self.knowledge_store),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import sys
    
    engine = WisdomIngestionEngine()
    
    if len(sys.argv) < 2:
        print("""
╔════════════════════════════════════════════════════════════════╗
║          WISDOM INGESTION ENGINE — House of Wisdom             ║
╠════════════════════════════════════════════════════════════════╣
║  Usage:                                                         ║
║    python wisdom_ingestion.py stats     — Show KB statistics    ║
║    python wisdom_ingestion.py query X   — Search for X          ║
║    python wisdom_ingestion.py list      — List knowledge files  ║
╚════════════════════════════════════════════════════════════════╝
        """)
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == "stats":
        stats = engine.get_statistics()
        print("\n📊 KNOWLEDGE BASE STATISTICS")
        print("═" * 40)
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print("═" * 40)
        
    elif cmd == "query" and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        results = engine.query_knowledge(query)
        print(f"\n🔍 Results for '{query}':")
        print("═" * 40)
        for r in results:
            print(f"  [{r.get('type')}] {r.get('name')}")
            print(f"    {r.get('description', '')[:80]}...")
            print()
        if not results:
            print("  No matches found.")
        print("═" * 40)
        
    elif cmd == "list":
        print("\n📁 KNOWLEDGE FILES")
        print("═" * 40)
        for f in PATHS["knowledge"].glob("*.jsonl"):
            with open(f, 'r') as file:
                count = sum(1 for line in file if line.strip())
            print(f"  {f.name}: {count} entries")
        print("═" * 40)
        
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
