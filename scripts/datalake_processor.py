#!/usr/bin/env python3
"""
BIZRA Data Lake Pipeline — Stage 1→2 Processor
═══════════════════════════════════════════════

Walks 00_INTAKE slots, deduplicates by SHA-256,
sorts by type into 01_RAW, generates BLAKE3 manifests
into 02_PROCESSED.

Standing on Giants:
  Shannon (1948) — information entropy for dedup detection
  Knuth (1973) — hash table for O(1) duplicate lookup
  Salsa20/BLAKE3 (2020) — content-addressed storage

Constitutional Authority:
  ZANN_ZERO — every file gets a verified manifest
  IHSAN_FLOOR — only quality-checked content reaches GOLD

Usage:
    python datalake_processor.py [--intake-only SLOT] [--dry-run] [--batch N]

Example:
    python datalake_processor.py --intake-only gmail_main --batch 100
"""

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

SOVEREIGN_ROOT = Path(r"B:\BIZRA-SOVEREIGN\05_DATA_LAKE")
INTAKE_DIR = SOVEREIGN_ROOT / "00_INTAKE"
RAW_DIR = SOVEREIGN_ROOT / "01_RAW"
PROCESSED_DIR = SOVEREIGN_ROOT / "02_PROCESSED" / "manifests"
QUARANTINE_DIR = SOVEREIGN_ROOT / "99_QUARANTINE" / "duplicates"

# File type → RAW subdirectory mapping
TYPE_MAP = {
    # Documents
    ".pdf": "documents",
    ".docx": "documents",
    ".doc": "documents",
    ".txt": "documents",
    ".md": "documents",
    ".rtf": "documents",
    ".odt": "documents",
    ".tex": "documents",
    # Notebooks
    ".ipynb": "notebooks",
    # Conversations (AI chat exports)
    ".json": "data",  # default, reclassified if conversation-shaped
    # Media
    ".png": "media",
    ".jpg": "media",
    ".jpeg": "media",
    ".gif": "media",
    ".svg": "media",
    ".webp": "media",
    ".mp4": "media",
    ".mov": "media",
    ".avi": "media",
    ".mkv": "media",
    ".mp3": "media",
    ".wav": "media",
    # Code
    ".py": "code",
    ".rs": "code",
    ".ts": "code",
    ".js": "code",
    ".tsx": "code",
    ".jsx": "code",
    ".sh": "code",
    ".bat": "code",
    ".ps1": "code",
    ".toml": "code",
    ".yaml": "code",
    ".yml": "code",
    # Data
    ".csv": "data",
    ".tsv": "data",
    ".xml": "data",
    ".db": "data",
    ".sqlite": "data",
    # Presentations
    ".pptx": "presentations",
    ".ppt": "presentations",
    ".key": "presentations",
    # Archives (skip contents, log existence)
    ".zip": "data",
    ".tar": "data",
    ".gz": "data",
    ".7z": "data",
    ".rar": "data",
}

# Mindset era detection by source account
ERA_MAP = {
    "gmail_main": "genesis_searcher_2023",
    "gmail_2": "unknown",
    "gmail_3": "unknown",
    "outlook_1": "ideologist_2023",
    "outlook_2": "unknown",
    "local_node0": "builder_2024_2026",
}

# ═══════════════════════════════════════════════════════════════
# CORE: SHA-256 DEDUP ENGINE
# ═══════════════════════════════════════════════════════════════


def sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of file contents. O(n) in file size."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def blake3_content(filepath: Path) -> str:
    """BLAKE3 content hash. Falls back to SHA-256 if blake3 not installed."""
    try:
        import blake3

        h = blake3.blake3()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except ImportError:
        return "sha256:" + sha256_file(filepath)


def classify_file(filepath: Path) -> str:
    """Map file extension to RAW subdirectory."""
    ext = filepath.suffix.lower()
    return TYPE_MAP.get(ext, "data")


def extract_text_preview(filepath: Path, max_chars: int = 500) -> str:
    """Extract first N chars of text content. Safe for all file types."""
    ext = filepath.suffix.lower()
    if ext in (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".mp4",
        ".mov",
        ".avi",
        ".mp3",
        ".wav",
        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".rar",
        ".db",
        ".sqlite",
        ".pyd",
        ".exe",
    ):
        return f"[binary:{ext}]"
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except Exception:
        return "[unreadable]"


def generate_manifest(filepath: Path, source_account: str, content_hash: str) -> dict:
    """Generate JSON manifest for a single file. ZANN_ZERO compliant."""
    stat = filepath.stat()
    return {
        "id": content_hash,
        "source_account": source_account,
        "original_path": str(filepath),
        "original_filename": filepath.name,
        "created": datetime.fromtimestamp(stat.st_ctime, timezone.utc).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "file_type": classify_file(filepath),
        "extension": filepath.suffix.lower(),
        "size_bytes": stat.st_size,
        "extracted_text_preview": extract_text_preview(filepath),
        "mindset_era": ERA_MAP.get(source_account, "unknown"),
        "connections": [],
        "tags": [],
        "verified": False,
        "manifest_created": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════════
# PIPELINE: INTAKE → RAW → PROCESSED
# ═══════════════════════════════════════════════════════════════


class DataLakePipeline:
    """5-stage data lake processor with SHA-256 dedup and BLAKE3 manifests."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.seen_hashes: dict[str, str] = {}  # hash → first_seen_path
        self.stats = {
            "scanned": 0,
            "unique": 0,
            "duplicate": 0,
            "manifested": 0,
            "errors": 0,
            "skipped": 0,
        }
        self._load_existing_hashes()

    def _load_existing_hashes(self):
        """Load previously processed hashes to resume interrupted runs."""
        manifest_dir = PROCESSED_DIR
        if not manifest_dir.exists():
            return
        for mf in manifest_dir.glob("*.json"):
            try:
                with open(mf, "r") as f:
                    data = json.load(f)
                h = data.get("id", "")
                if h:
                    self.seen_hashes[h] = data.get("original_path", "")
            except Exception:
                pass
        if self.seen_hashes:
            print(f"  Resumed: {len(self.seen_hashes)} previously processed files")

    def process_intake_slot(self, slot_name: str, batch_limit: int = 0):
        """Process one intake slot end-to-end."""
        slot_path = INTAKE_DIR / slot_name
        if not slot_path.exists():
            print(f"  SKIP: {slot_name} (not found)")
            return

        print(f"\n{'='*60}")
        print(f"  Processing: {slot_name}")
        print(f"  Source: {slot_path}")
        print(f"  Era: {ERA_MAP.get(slot_name, 'unknown')}")
        print(f"{'='*60}")

        count = 0
        for root, dirs, files in os.walk(slot_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                if fname.startswith(".") or fname == "desktop.ini":
                    self.stats["skipped"] += 1
                    continue
                filepath = Path(root) / fname
                self._process_file(filepath, slot_name)
                count += 1
                if batch_limit and count >= batch_limit:
                    print(f"\n  Batch limit reached ({batch_limit})")
                    return

    def _process_file(self, filepath: Path, source_account: str):
        """Process a single file: hash → dedup → sort → manifest."""
        self.stats["scanned"] += 1
        try:
            file_hash = sha256_file(filepath)
        except (PermissionError, OSError) as e:
            print(f"  ERR: {filepath.name} — {e}")
            self.stats["errors"] += 1
            return

        # DEDUP CHECK (O(1) hash table lookup — Knuth)
        if file_hash in self.seen_hashes:
            first_seen = self.seen_hashes[file_hash]
            self.stats["duplicate"] += 1
            if not self.dry_run:
                log_path = QUARANTINE_DIR / "dedup_log.jsonl"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "duplicate": str(filepath),
                                "original": first_seen,
                                "hash": file_hash,
                                "time": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        + "\n"
                    )
            if self.stats["scanned"] % 50 == 0:
                print(f"  [{self.stats['scanned']}] DUP: {filepath.name}")
            return

        # NEW FILE — register hash
        self.seen_hashes[file_hash] = str(filepath)
        self.stats["unique"] += 1

        # SORT INTO RAW BY TYPE
        file_type = classify_file(filepath)
        raw_dest = RAW_DIR / file_type / filepath.name
        if not self.dry_run:
            raw_dest.parent.mkdir(parents=True, exist_ok=True)
            # Handle name collisions
            if raw_dest.exists():
                stem = filepath.stem
                suffix = filepath.suffix
                raw_dest = RAW_DIR / file_type / f"{stem}_{file_hash[:8]}{suffix}"
            shutil.copy2(filepath, raw_dest)

        # GENERATE MANIFEST (BLAKE3 content-addressed)
        content_hash = blake3_content(filepath)
        manifest = generate_manifest(filepath, source_account, content_hash)
        manifest["sha256"] = file_hash
        manifest["raw_path"] = str(raw_dest)

        if not self.dry_run:
            manifest_path = PROCESSED_DIR / f"{file_hash[:16]}.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        self.stats["manifested"] += 1

        if self.stats["scanned"] % 25 == 0:
            print(f"  [{self.stats['scanned']}] NEW: {filepath.name} -> {file_type}/")

    def run_all(self, intake_only: Optional[str] = None, batch_limit: int = 0):
        """Run pipeline across all intake slots (or one specific slot)."""
        start = time.time()
        print(f"BIZRA Data Lake Pipeline -- {datetime.now(timezone.utc).isoformat()}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")

        if intake_only:
            self.process_intake_slot(intake_only, batch_limit)
        else:
            for slot in sorted(INTAKE_DIR.iterdir()):
                if slot.is_dir():
                    self.process_intake_slot(slot.name, batch_limit)

        elapsed = time.time() - start
        self._print_report(elapsed)
        self._save_report(elapsed)

    def _print_report(self, elapsed: float):
        """Print pipeline summary."""
        s = self.stats
        print(f"\n{'='*60}")
        print("  PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"  Scanned:    {s['scanned']}")
        print(f"  Unique:     {s['unique']}")
        print(f"  Duplicate:  {s['duplicate']}")
        print(f"  Manifested: {s['manifested']}")
        print(f"  Errors:     {s['errors']}")
        print(f"  Skipped:    {s['skipped']}")
        print(f"  Duration:   {elapsed:.1f}s")
        if s["scanned"] > 0:
            dedup_rate = s["duplicate"] / s["scanned"] * 100
            print(f"  Dedup rate: {dedup_rate:.1f}%")

    def _save_report(self, elapsed: float):
        """Save pipeline run report as evidence."""
        report = {
            "pipeline": "BIZRA Data Lake Stage 1→2",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(elapsed, 1),
            "stats": self.stats,
            "dry_run": self.dry_run,
        }
        report_path = SOVEREIGN_ROOT / "pipeline_runs.jsonl"
        with open(report_path, "a") as f:
            f.write(json.dumps(report) + "\n")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="BIZRA Data Lake Pipeline — Stage 1→2 Processor"
    )
    parser.add_argument(
        "--intake-only", type=str, default=None, help="Process only this intake slot"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Scan and report without copying/writing"
    )
    parser.add_argument(
        "--batch", type=int, default=0, help="Limit files per slot (0=unlimited)"
    )
    args = parser.parse_args()

    pipeline = DataLakePipeline(dry_run=args.dry_run)
    pipeline.run_all(intake_only=args.intake_only, batch_limit=args.batch)


if __name__ == "__main__":
    main()
