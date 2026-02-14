#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                      BIZRA UNIVERSAL INGESTION PIPELINE v1.0                                                ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  Purpose: Ingest ALL 1.1M+ BIZRA assets from Downloads, OneDrive, Google Drive                              ║
║  Scope: Images, HTML, PDF, MD, TXT, DOCX, Python, JSON, SVG, and more                                       ║
║  Author: BIZRA Genesis NODE0 | Date: 2026-01-28                                                             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import hashlib
import logging
import mimetypes
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s | INGEST | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("UniversalIngestion")

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_BASE = PROJECT_ROOT / "02_PROCESSED"
MANIFEST_PATH = PROJECT_ROOT / "04_GOLD" / "universal_ingestion_manifest.jsonl"
CHECKPOINT_PATH = PROJECT_ROOT / "04_GOLD" / "ingestion_checkpoint.json"

# Hashing configuration — SEC-001: BLAKE3 for Rust interop
HASH_ALGO = "blake3"
HASH_PREFIX_LEN = 16
LEGACY_PREFIX_LEN = 8

# Source locations
SOURCE_LOCATIONS = [
    Path(r"C:\Users\BIZRA-OS\Downloads"),
    Path(r"C:\Users\BIZRA-OS\OneDrive"),
    Path(r"H:\My Drive"),
]

# File type categories
FILE_CATEGORIES = {
    "text": [".md", ".txt", ".rst", ".log", ".csv"],
    "code": [".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs", ".rb", ".php", ".sh", ".bat", ".ps1", ".vue", ".svelte", ".d.ts", ".mts", ".cts"],
    "config": [".json", ".yaml", ".yml", ".toml", ".ini", ".env", ".xml", ".lock", ".config"],
    "document": [".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".odt"],
    "html": [".html", ".htm", ".mhtml", ".xhtml"],
    "image": [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".tiff"],
    "vector": [".svg", ".drawio", ".vsdx"],
    "notebook": [".ipynb"],
}

# Reverse lookup
EXT_TO_CATEGORY = {}
for cat, exts in FILE_CATEGORIES.items():
    for ext in exts:
        EXT_TO_CATEGORY[ext] = cat


@dataclass
class FileRecord:
    """Record of a processed file."""
    path: str
    name: str
    extension: str
    category: str
    size_bytes: int
    # Note: kept as "hash_md5" for backward compatibility with existing manifests.
    # New runs store a SHA-256 hex digest here.
    hash_md5: str
    processed_at: str
    output_path: Optional[str] = None
    content_length: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class IngestionStats:
    """Statistics for the ingestion run."""
    total_discovered: int = 0
    total_processed: int = 0
    total_skipped: int = 0
    total_errors: int = 0
    bytes_processed: int = 0
    by_category: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_extension: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class UniversalIngestionPipeline:
    """
    Universal pipeline to ingest all BIZRA assets.
    
    Stage 1: Text files (MD, TXT, RST, LOG)
    Stage 2: Code files (PY, JS, TS, etc.)
    Stage 3: Config files (JSON, YAML, TOML)
    Stage 4: HTML content
    Stage 5: Documents (PDF, DOCX) - requires special libs
    Stage 6: Images (with OCR) - requires special libs
    """
    
    def __init__(self, batch_size: int = 1000, max_workers: int = 8):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.stats = IngestionStats()
        # Store full hashes separately from legacy short prefixes for collision safety.
        self.processed_hashes_full: Set[str] = set()
        self.processed_hashes_prefix8: Set[str] = set()
        self.processed_hashes_prefix16: Set[str] = set()
        self.checkpoint: Dict = {}
        
        # Ensure ALL output directories exist
        OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
        for cat in FILE_CATEGORIES:
            (OUTPUT_BASE / cat).mkdir(exist_ok=True)
        
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing hashes from processed files for smart resume
        self._load_existing_hashes()
        self._load_checkpoint()
    
    def _load_existing_hashes(self):
        """Scan existing processed files to build hash set for smart resume."""
        log.info("Scanning existing processed files for resume capability...")
        hash_pattern = re.compile(r'_([a-f0-9]{8,64})\.md$')
        
        for cat_dir in OUTPUT_BASE.iterdir():
            if cat_dir.is_dir() and cat_dir.name in FILE_CATEGORIES:
                for file_path in cat_dir.glob("*.md"):
                    match = hash_pattern.search(file_path.name)
                    if match:
                        digest = match.group(1)
                        if len(digest) >= 64:
                            self.processed_hashes_full.add(digest)
                        elif len(digest) <= LEGACY_PREFIX_LEN:
                            self.processed_hashes_prefix8.add(digest)
                        else:
                            self.processed_hashes_prefix16.add(digest)
        
        existing = (
            len(self.processed_hashes_full)
            + len(self.processed_hashes_prefix8)
            + len(self.processed_hashes_prefix16)
        )
        if existing:
            log.info(f"Found {existing:,} already processed file hashes")
    
    def _load_checkpoint(self):
        """Load processing checkpoint for resume capability."""
        if CHECKPOINT_PATH.exists():
            try:
                with open(CHECKPOINT_PATH, 'r') as f:
                    self.checkpoint = json.load(f)
                full_hashes = self.checkpoint.get("processed_hashes_full")
                legacy_hashes = self.checkpoint.get("processed_hashes")
                prefix8 = self.checkpoint.get("processed_hashes_prefix8")
                prefix16 = self.checkpoint.get("processed_hashes_prefix16")

                if isinstance(full_hashes, list):
                    self.processed_hashes_full.update(full_hashes)

                if isinstance(prefix8, list):
                    self.processed_hashes_prefix8.update(prefix8)
                if isinstance(prefix16, list):
                    self.processed_hashes_prefix16.update(prefix16)

                if isinstance(legacy_hashes, list):
                    # Backward-compatible load (older checkpoints)
                    for digest in legacy_hashes:
                        if len(digest) >= 64:
                            self.processed_hashes_full.add(digest)
                        elif len(digest) <= LEGACY_PREFIX_LEN:
                            self.processed_hashes_prefix8.add(digest)
                        else:
                            self.processed_hashes_prefix16.add(digest)

                total = (
                    len(self.processed_hashes_full)
                    + len(self.processed_hashes_prefix8)
                    + len(self.processed_hashes_prefix16)
                )
                log.info(f"Loaded checkpoint: {total:,} files already processed")
            except Exception as e:
                log.warning(f"Could not load checkpoint: {e}")
    
    def _save_checkpoint(self):
        """Save processing checkpoint."""
        self.checkpoint["processed_hashes_full"] = list(self.processed_hashes_full)
        self.checkpoint["processed_hashes_prefix8"] = list(self.processed_hashes_prefix8)
        self.checkpoint["processed_hashes_prefix16"] = list(self.processed_hashes_prefix16)
        self.checkpoint["hash_algo"] = HASH_ALGO
        self.checkpoint["hash_prefix_len"] = HASH_PREFIX_LEN
        self.checkpoint["last_updated"] = datetime.now().isoformat()
        self.checkpoint["stats"] = {
            "total_processed": self.stats.total_processed,
            "total_errors": self.stats.total_errors,
            "bytes_processed": self.stats.bytes_processed,
        }
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate hash for deduplication."""
        # Use content-based hashing to prevent collisions and false skips.
        try:
            import blake3
            hasher = blake3.blake3()
        except ImportError:
            hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _is_already_processed(self, full_hash: str) -> bool:
        """Check hash against full and legacy prefix caches."""
        if full_hash in self.processed_hashes_full:
            return True
        prefix8 = full_hash[:LEGACY_PREFIX_LEN]
        if prefix8 in self.processed_hashes_prefix8:
            return True
        prefix16 = full_hash[:HASH_PREFIX_LEN]
        if prefix16 in self.processed_hashes_prefix16:
            return True
        return False
    
    def discover_files(self, categories: Optional[List[str]] = None) -> Generator[Path, None, None]:
        """Discover all files from source locations."""
        target_exts = set()
        if categories:
            for cat in categories:
                target_exts.update(FILE_CATEGORIES.get(cat, []))
        else:
            for exts in FILE_CATEGORIES.values():
                target_exts.update(exts)
        
        for source in SOURCE_LOCATIONS:
            if not source.exists():
                log.warning(f"Source not found: {source}")
                continue
            
            log.info(f"Scanning: {source}")
            try:
                for file_path in source.rglob("*"):
                    if file_path.is_file():
                        ext = file_path.suffix.lower()
                        if ext in target_exts:
                            self.stats.total_discovered += 1
                            yield file_path
            except PermissionError as e:
                log.warning(f"Permission denied: {e}")
            except Exception as e:
                log.error(f"Error scanning {source}: {e}")
    
    def process_text_file(self, file_path: Path) -> Tuple[str, int]:
        """Extract content from text-based files."""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
                return content, len(content)
            except Exception:
                continue
        
        raise ValueError(f"Could not decode file with any encoding")
    
    def process_code_file(self, file_path: Path) -> Tuple[str, int]:
        """Extract content from code files with metadata."""
        content, length = self.process_text_file(file_path)
        
        # Add metadata header
        ext = file_path.suffix.lower()
        lang_map = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.go': 'go',
            '.rs': 'rust', '.rb': 'ruby', '.php': 'php', '.sh': 'bash',
        }
        language = lang_map.get(ext, 'code')
        
        header = f"""---
source: {file_path}
type: code
language: {language}
size: {file_path.stat().st_size}
processed: {datetime.now().isoformat()}
---

"""
        full_content = header + content
        return full_content, len(full_content)
    
    def process_html_file(self, file_path: Path) -> Tuple[str, int]:
        """Extract text content from HTML files."""
        content, _ = self.process_text_file(file_path)
        
        # Simple HTML to text conversion
        # Remove script and style tags
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags but keep content
        content = re.sub(r'<[^>]+>', ' ', content)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        # Add metadata
        header = f"""---
source: {file_path}
type: html
size: {file_path.stat().st_size}
processed: {datetime.now().isoformat()}
---

"""
        full_content = header + content
        return full_content, len(full_content)
    
    def process_json_file(self, file_path: Path) -> Tuple[str, int]:
        """Process JSON files - extract structure and content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to readable format
            if isinstance(data, dict):
                # Try to extract meaningful content
                content_parts = []
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 10:
                        content_parts.append(f"{key}: {value[:1000]}")
                    elif isinstance(value, list) and len(value) > 0:
                        content_parts.append(f"{key}: [{len(value)} items]")
                content = "\n".join(content_parts) if content_parts else json.dumps(data, indent=2)[:5000]
            else:
                content = json.dumps(data, indent=2)[:5000]
            
            header = f"""---
source: {file_path}
type: json
size: {file_path.stat().st_size}
processed: {datetime.now().isoformat()}
---

"""
            full_content = header + content
            return full_content, len(full_content)
        except json.JSONDecodeError:
            # Fall back to text processing
            return self.process_text_file(file_path)
    
    def process_file(self, file_path: Path) -> Optional[FileRecord]:
        """Process a single file based on its type."""
        try:
            ext = file_path.suffix.lower()
            category = EXT_TO_CATEGORY.get(ext, "other")
            
            # Get file hash for deduplication
            file_hash = self.get_file_hash(file_path)
            
            # Check if already processed (using hash prefix stored in filename)
            if self._is_already_processed(file_hash):
                self.stats.total_skipped += 1
                return None
            
            # Process based on category
            if category in ["text", "code"]:
                if category == "code":
                    content, length = self.process_code_file(file_path)
                else:
                    content, length = self.process_text_file(file_path)
            elif category == "html":
                content, length = self.process_html_file(file_path)
            elif category == "config":
                if ext == ".json":
                    content, length = self.process_json_file(file_path)
                else:
                    content, length = self.process_text_file(file_path)
            elif category == "notebook":
                # Extract code and markdown from Jupyter notebooks
                content, length = self.process_notebook_file(file_path)
            else:
                # Skip binary files for now (images, PDFs need special processing)
                self.stats.total_skipped += 1
                return None
            
            # Generate output path
            safe_name = re.sub(r'[^\w\-_.]', '_', file_path.stem)[:100]
            output_name = f"{safe_name}_{file_hash[:HASH_PREFIX_LEN]}.md"
            output_path = OUTPUT_BASE / category / output_name
            
            # Write processed content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Create record
            record = FileRecord(
                path=str(file_path),
                name=file_path.name,
                extension=ext,
                category=category,
                size_bytes=file_path.stat().st_size,
                hash_md5=file_hash,
                processed_at=datetime.now().isoformat(),
                output_path=str(output_path),
                content_length=length,
            )
            
            # Update stats - track with hash prefix for resume capability
            self.processed_hashes_full.add(file_hash)
            self.stats.total_processed += 1
            self.stats.bytes_processed += file_path.stat().st_size
            self.stats.by_category[category] += 1
            self.stats.by_extension[ext] += 1
            
            return record
            
        except Exception as e:
            self.stats.total_errors += 1
            self.stats.errors_by_type[type(e).__name__] += 1
            return FileRecord(
                path=str(file_path),
                name=file_path.name,
                extension=file_path.suffix.lower(),
                category=EXT_TO_CATEGORY.get(file_path.suffix.lower(), "other"),
                size_bytes=0,
                hash_md5="",
                processed_at=datetime.now().isoformat(),
                error=str(e)[:200],
            )
    
    def process_notebook_file(self, file_path: Path) -> Tuple[str, int]:
        """Extract content from Jupyter notebooks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            
            cells = nb.get('cells', [])
            content_parts = []
            
            for i, cell in enumerate(cells):
                cell_type = cell.get('cell_type', 'unknown')
                source = ''.join(cell.get('source', []))
                
                if cell_type == 'markdown':
                    content_parts.append(f"### Cell {i+1} (Markdown)\n{source}\n")
                elif cell_type == 'code':
                    content_parts.append(f"### Cell {i+1} (Code)\n```python\n{source}\n```\n")
            
            header = f"""---
source: {file_path}
type: notebook
cells: {len(cells)}
processed: {datetime.now().isoformat()}
---

"""
            content = header + "\n".join(content_parts)
            return content, len(content)
        except Exception as e:
            raise ValueError(f"Could not parse notebook: {e}")
    
    def run_stage(self, categories: List[str], stage_name: str) -> int:
        """Run ingestion for specific file categories."""
        print(f"\n{'=' * 80}")
        print(f"  STAGE: {stage_name}")
        print(f"  Categories: {', '.join(categories)}")
        print(f"{'=' * 80}")

        processed = 0
        scanned = 0
        batch_records = []
        
        # Open manifest for appending
        with open(MANIFEST_PATH, 'a', encoding='utf-8') as manifest:
            try:
                for file_path in self.discover_files(categories):
                    try:
                        scanned += 1
                        record = self.process_file(file_path)
                        
                        if record and not record.error:
                            batch_records.append(record)
                            processed += 1
                        
                        # Write batch to manifest
                        if len(batch_records) >= 100:
                            for rec in batch_records:
                                manifest.write(json.dumps(rec.to_dict()) + "\n")
                            batch_records = []
                        
                        # Progress update
                        if scanned % 1000 == 0:
                            print(f"  Progress: {scanned:,} scanned | Processed: {processed:,}")
                            sys.stdout.flush()
                            self._save_checkpoint()
                    except Exception as e:
                        log.error(f"CRITICAL ERROR processing file {file_path}: {e}", exc_info=True)
                        continue
            except Exception as e:
                log.critical(f"FATAL ERROR in discovery loop: {e}", exc_info=True)
                with open("crash_report.log", "w") as f:
                    import traceback
                    traceback.print_exc(file=f)
            
            # Write remaining records
            for rec in batch_records:
                manifest.write(json.dumps(rec.to_dict()) + "\n")
        
        self._save_checkpoint()
        print(f"  ✅ Stage complete: {processed:,} files processed (scanned: {scanned:,})")
        return processed
    
    def run_full_pipeline(self):
        """Run all ingestion stages."""
        print("╔" + "═" * 78 + "╗")
        print("║" + "  BIZRA UNIVERSAL INGESTION PIPELINE".center(78) + "║")
        print("║" + f"  Processing 1.1M+ BIZRA Assets".center(78) + "║")
        print("╚" + "═" * 78 + "╝")
        print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Sources: {len(SOURCE_LOCATIONS)}")
        already = (
            len(self.processed_hashes_full)
            + len(self.processed_hashes_prefix8)
            + len(self.processed_hashes_prefix16)
        )
        print(f"Already processed: {already:,}")
        
        # Stage 1: Text documents (fastest)
        self.run_stage(["text"], "TEXT DOCUMENTS (MD, TXT, RST)")
        
        # Stage 2: HTML content
        self.run_stage(["html"], "HTML CONTENT")
        
        # Stage 3: Code files
        self.run_stage(["code"], "CODE FILES (PY, JS, TS, etc.)")
        
        # Stage 4: Config files
        self.run_stage(["config"], "CONFIG FILES (JSON, YAML, TOML)")
        
        # Stage 5: Notebooks
        self.run_stage(["notebook"], "JUPYTER NOTEBOOKS")
        
        # Final summary
        self._print_summary()
    
    def _print_summary(self):
        """Print final ingestion summary."""
        print("\n" + "═" * 80)
        print("  INGESTION COMPLETE")
        print("═" * 80)
        print(f"""
  📊 Summary:
     Total Discovered:  {self.stats.total_discovered:,}
     Total Processed:   {self.stats.total_processed:,}
     Total Skipped:     {self.stats.total_skipped:,}
     Total Errors:      {self.stats.total_errors:,}
     Bytes Processed:   {self.stats.bytes_processed / (1024**3):.2f} GB
  
  📁 By Category:""")
        for cat, count in sorted(self.stats.by_category.items(), key=lambda x: -x[1]):
            print(f"     {cat:15} {count:,}")
        
        print(f"\n  💾 Output: {OUTPUT_BASE}")
        print(f"  📋 Manifest: {MANIFEST_PATH}")
        print("═" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BIZRA Universal Ingestion Pipeline")
    parser.add_argument("--stage", choices=["text", "html", "code", "config", "notebook", "all"], 
                       default="all", help="Which stage to run")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    
    args = parser.parse_args()
    
    pipeline = UniversalIngestionPipeline(batch_size=args.batch_size)
    
    if args.stage == "all":
        pipeline.run_full_pipeline()
    else:
        pipeline.run_stage([args.stage], args.stage.upper())


if __name__ == "__main__":
    main()
