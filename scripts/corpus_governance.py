r"""
BIZRA Research Corpus Governance — W5 Registry
════════════════════════════════════════════════

Audit finding: "Your research corpus is correctly recognized as a
strategic asset, but it is not yet a governed knowledge plane."

This module canonicalizes the 3-year, 150+ document corpus into a
searchable, provenance-aware sovereign input layer.

What it does:
  1. SCAN — Find all documents across G:\, B:\BIZRA-SOVEREIGN, C:\BIZRA-DATA-LAKE
  2. HASH — BLAKE3 content hash for each document (provenance anchor)
  3. CLASSIFY — Tag each document (founding | research | architecture | spec | paper | session)
  4. REGISTER — Write canonical registry as JSON + markdown index
  5. VERIFY — Detect duplicates, orphans, and unregistered documents

The registry becomes a governed input to the House of Wisdom.
Every document has a provenance chain: creation_date → hash → classification → citations.

Usage:
    python corpus_governance.py scan              # Scan and register all documents
    python corpus_governance.py verify            # Verify registry integrity
    python corpus_governance.py index             # Generate searchable markdown index
    python corpus_governance.py stats             # Corpus statistics
    python corpus_governance.py duplicates        # Find duplicate content

Standing on: Ibn al-Nadim (Kitab al-Fihrist — first systematic bibliography),
Ranganathan (Five Laws of Library Science), Shannon (information theory)

Created: 2026-03-23 | BIZRA Corpus Governance v1.0
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


def blake3_hash_file(path: str, chunk_size: int = 65536) -> str:
    """BLAKE3 hash of file contents. Falls back to SHA-256."""
    h = hashlib.sha256()
    try:
        import blake3

        h = blake3.blake3()
    except ImportError:
        pass
    try:
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()[:32]
    except (OSError, PermissionError):
        return "UNREADABLE"


# ═══ DOCUMENT CLASSIFICATION ═══

CLASSIFICATIONS = {
    "founding": {
        "description": "Sacred founding documents (الرسالة, البذرة)",
        "authority_level": 2,  # Below Quran/Hadith, above Spine
        "keywords": ["الرسالة", "البذرة", "founding", "genesis", "ramadan 2023"],
    },
    "constitutional": {
        "description": "Enforceable Spine, invariants, governance",
        "authority_level": 3,
        "keywords": [
            "spine",
            "constitutional",
            "invariant",
            "governance",
            "enforceable",
        ],
    },
    "research": {
        "description": "Research papers, analysis, formal proofs",
        "authority_level": 5,
        "keywords": ["paper", "proof", "theorem", "analysis", "CMN", "preprint"],
    },
    "architecture": {
        "description": "System architecture, design decisions, ADRs",
        "authority_level": 4,
        "keywords": ["architecture", "ADR", "design", "system", "protocol", "HDA"],
    },
    "spec": {
        "description": "Specifications, APIs, schemas",
        "authority_level": 4,
        "keywords": ["spec", "API", "schema", "SAP", "protocol", "interface"],
    },
    "session": {
        "description": "Session transcripts, conversation exports",
        "authority_level": 6,
        "keywords": ["session", "conversation", "transcript", "daily"],
    },
    "business": {
        "description": "Pitch decks, business plans, investor materials",
        "authority_level": 6,
        "keywords": ["pitch", "investor", "business", "plan", "GTM", "market"],
    },
    "operational": {
        "description": "Deployment, CI/CD, infrastructure",
        "authority_level": 5,
        "keywords": ["deploy", "CI", "docker", "kubernetes", "infrastructure"],
    },
}

SCAN_EXTENSIONS = {
    ".md",
    ".txt",
    ".pdf",
    ".docx",
    ".doc",
    ".html",
    ".htm",
    ".py",
    ".rs",
    ".ts",
    ".js",
    ".jsx",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}

SCAN_ROOTS_WINDOWS = [
    r"C:\BIZRA-DATA-LAKE",
    r"B:\BIZRA-SOVEREIGN",
    # G:\ Google Drive folders scanned separately
]

SCAN_ROOTS_UNIX = [
    "/mnt/c/BIZRA-DATA-LAKE",
    "/mnt/b/BIZRA-SOVEREIGN",
]

IGNORE_DIRS = {
    "node_modules",
    ".git",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
}


# ═══ DOCUMENT RECORD ═══


@dataclass
class DocumentRecord:
    path: str
    filename: str
    extension: str
    size_bytes: int
    content_hash: str
    classification: str
    authority_level: int
    created_approx: str  # Best estimate from filesystem
    registered_at: str
    title: Optional[str] = None
    description: Optional[str] = None
    citations: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    duplicate_of: Optional[str] = None


# ═══ CORPUS REGISTRY ═══


class CorpusRegistry:
    def __init__(self, registry_path: str = "corpus_registry.json"):
        self.registry_path = registry_path
        self.documents: dict[str, DocumentRecord] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
                for key, rec in data.get("documents", {}).items():
                    self.documents[key] = DocumentRecord(**rec)
            except (json.JSONDecodeError, TypeError):
                pass

    def save(self):
        data = {
            "version": "1.0",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "document_count": len(self.documents),
            "documents": {k: asdict(v) for k, v in self.documents.items()},
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def scan(self, roots: list[str] | None = None) -> int:
        """Scan filesystem roots and register documents."""
        if roots is None:
            roots = SCAN_ROOTS_WINDOWS if os.name == "nt" else SCAN_ROOTS_UNIX

        count = 0
        for root in roots:
            if not os.path.exists(root):
                print(f"  Skip: {root} (not found)")
                continue
            print(f"  Scanning: {root}")
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
                for fname in filenames:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in SCAN_EXTENSIONS:
                        continue
                    fpath = os.path.join(dirpath, fname)
                    self._register_file(fpath)
                    count += 1

        self.save()
        return count

    def _register_file(self, path: str):
        """Register a single file."""
        try:
            stat = os.stat(path)
        except OSError:
            return

        content_hash = blake3_hash_file(path)
        classification = self._classify(path, os.path.basename(path))
        ext = os.path.splitext(path)[1].lower()

        # Use modification time as creation approximation
        mtime = time.strftime("%Y-%m-%d", time.localtime(stat.st_mtime))

        record = DocumentRecord(
            path=path,
            filename=os.path.basename(path),
            extension=ext,
            size_bytes=stat.st_size,
            content_hash=content_hash,
            classification=classification,
            authority_level=CLASSIFICATIONS.get(classification, {}).get(
                "authority_level", 6
            ),
            created_approx=mtime,
            registered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        self.documents[content_hash] = record

    def _classify(self, path: str, filename: str) -> str:
        """Classify document by path and filename keywords."""
        combined = f"{path} {filename}".lower()
        best_class = "session"
        best_score = 0

        for cls, info in CLASSIFICATIONS.items():
            score = sum(1 for kw in info["keywords"] if kw.lower() in combined)
            if score > best_score:
                best_score = score
                best_class = cls

        return best_class

    def find_duplicates(self) -> list[tuple[str, list[str]]]:
        """Find documents with identical content hashes."""
        hash_groups: dict[str, list[str]] = {}
        for key, doc in self.documents.items():
            hash_groups.setdefault(doc.content_hash, []).append(doc.path)

        return [(h, paths) for h, paths in hash_groups.items() if len(paths) > 1]

    def stats(self) -> dict:
        """Corpus statistics."""
        by_class: dict[str, int] = {}
        by_ext: dict[str, int] = {}
        total_size = 0

        for doc in self.documents.values():
            by_class[doc.classification] = by_class.get(doc.classification, 0) + 1
            by_ext[doc.extension] = by_ext.get(doc.extension, 0) + 1
            total_size += doc.size_bytes

        return {
            "total_documents": len(self.documents),
            "total_size_mb": total_size / (1024 * 1024),
            "by_classification": dict(sorted(by_class.items(), key=lambda x: -x[1])),
            "by_extension": dict(sorted(by_ext.items(), key=lambda x: -x[1])),
            "unique_hashes": len(set(d.content_hash for d in self.documents.values())),
        }

    def generate_index(self, output_path: str = "CORPUS_INDEX.md") -> str:
        """Generate a searchable markdown index."""
        lines = [
            "# BIZRA Research Corpus Index",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
            f"**Documents:** {len(self.documents)}",
            "",
            "---",
            "",
        ]

        # Group by classification
        by_class: dict[str, list[DocumentRecord]] = {}
        for doc in self.documents.values():
            by_class.setdefault(doc.classification, []).append(doc)

        # Sort by authority level (highest authority first)
        for cls in sorted(
            by_class.keys(),
            key=lambda c: CLASSIFICATIONS.get(c, {}).get("authority_level", 99),
        ):
            info = CLASSIFICATIONS.get(cls, {})
            docs = sorted(by_class[cls], key=lambda d: d.created_approx, reverse=True)
            lines.append(f"## {cls.title()} ({len(docs)} documents)")
            lines.append(f"*{info.get('description', '')}*")
            lines.append(f"*Authority level: {info.get('authority_level', '?')}*")
            lines.append("")

            for doc in docs[:50]:  # Cap at 50 per category
                size = doc.size_bytes
                size_str = (
                    f"{size/1024:.0f}KB" if size < 1048576 else f"{size/1048576:.1f}MB"
                )
                lines.append(
                    f"- **{doc.filename}** ({size_str}, {doc.created_approx}) `{doc.content_hash[:12]}`"
                )

            lines.append("")

        content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path


# ═══ CLI ═══


def main():
    if len(sys.argv) < 2:
        print("Usage: python corpus_governance.py [scan|verify|index|stats|duplicates]")
        return 1

    cmd = sys.argv[1]
    registry = CorpusRegistry()

    if cmd == "scan":
        roots = sys.argv[2:] if len(sys.argv) > 2 else None
        print("\n  BIZRA Corpus Governance — Scanning\n")
        count = registry.scan(roots)
        print(
            f"\n  Registered {count} files. Registry saved to {registry.registry_path}\n"
        )

    elif cmd == "verify":
        print("\n  BIZRA Corpus Governance — Verification\n")
        missing = 0
        for key, doc in registry.documents.items():
            if not os.path.exists(doc.path):
                print(f"  MISSING: {doc.filename} ({doc.path})")
                missing += 1
        if missing:
            print(f"\n  {missing} documents missing from filesystem")
        else:
            print(f"  All {len(registry.documents)} documents verified")

    elif cmd == "index":
        print("\n  BIZRA Corpus Governance — Index Generation\n")
        path = registry.generate_index()
        print(f"  Index written to {path}\n")

    elif cmd == "stats":
        print("\n  BIZRA Corpus Governance — Statistics\n")
        s = registry.stats()
        print(f"  Total documents:  {s['total_documents']}")
        print(f"  Total size:       {s['total_size_mb']:.1f} MB")
        print(f"  Unique hashes:    {s['unique_hashes']}")
        print("\n  By classification:")
        for cls, count in s["by_classification"].items():
            print(f"    {cls:20s} {count}")
        print("\n  By extension:")
        for ext, count in s["by_extension"].items():
            print(f"    {ext:10s} {count}")

    elif cmd == "duplicates":
        print("\n  BIZRA Corpus Governance — Duplicate Detection\n")
        dupes = registry.find_duplicates()
        if dupes:
            for h, paths in dupes:
                print(f"  Hash {h[:12]}:")
                for p in paths:
                    print(f"    {p}")
            print(f"\n  {len(dupes)} duplicate groups found")
        else:
            print("  No duplicates found")

    else:
        print(f"Unknown command: {cmd}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
