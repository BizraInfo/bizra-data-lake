#!/usr/bin/env python3
"""
BIZRA Parallel Ingestion Engine v2.0
=====================================
High-performance parallel file processing using multiprocessing.
Designed for 500k+ file ingestion with SHA-256 hashing.
"""

import sys
import os
import time
import hashlib
import json
import re
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
from typing import Set, Dict, List, Optional, Tuple
import traceback

# Configuration
PROJECT_ROOT = Path(__file__).parent
OUTPUT_BASE = PROJECT_ROOT / "02_PROCESSED"
MANIFEST_PATH = PROJECT_ROOT / "04_GOLD" / "universal_ingestion_manifest.jsonl"
CHECKPOINT_PATH = PROJECT_ROOT / "04_GOLD" / "ingestion_checkpoint.json"
HASH_PREFIX_LEN = 16

# Source locations
SOURCE_LOCATIONS = [
    Path(r"C:\Users\BIZRA-OS\Downloads"),
    Path(r"C:\Users\BIZRA-OS\OneDrive"),
    Path(r"H:\My Drive"),
]

# Extended file categories
FILE_CATEGORIES = {
    "text": [".md", ".txt", ".rst", ".log", ".csv"],
    "code": [".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".java", ".cpp", ".c", 
             ".h", ".hpp", ".go", ".rs", ".rb", ".php", ".sh", ".bat", ".ps1", ".vue", 
             ".svelte", ".d.ts", ".mts", ".cts"],
    "config": [".json", ".yaml", ".yml", ".toml", ".ini", ".env", ".xml", ".lock", ".config"],
    "html": [".html", ".htm", ".mhtml", ".xhtml"],
}

# Build extension to category map
EXT_TO_CATEGORY = {}
for cat, exts in FILE_CATEGORIES.items():
    for ext in exts:
        EXT_TO_CATEGORY[ext] = cat


def get_file_hash(file_path: Path) -> str:
    """Generate SHA-256 hash for deduplication."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_checkpoint() -> Tuple[Set[str], Set[str], Set[str]]:
    """Load existing checkpoint data."""
    full_hashes = set()
    prefix8_hashes = set()
    prefix16_hashes = set()
    
    # Load from checkpoint file
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                cp = json.load(f)
            
            if "processed_hashes_full" in cp:
                full_hashes.update(cp["processed_hashes_full"])
            if "processed_hashes_prefix8" in cp:
                prefix8_hashes.update(cp["processed_hashes_prefix8"])
            if "processed_hashes_prefix16" in cp:
                prefix16_hashes.update(cp["processed_hashes_prefix16"])
            
            print(f"Loaded checkpoint: full={len(full_hashes)}, p8={len(prefix8_hashes)}, p16={len(prefix16_hashes)}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    # Scan existing processed files for hash extraction
    print("Scanning existing processed files...")
    for category_dir in OUTPUT_BASE.iterdir():
        if category_dir.is_dir():
            for processed_file in category_dir.glob("*.md"):
                name = processed_file.stem
                if "_" in name:
                    hash_part = name.split("_")[-1]
                    if len(hash_part) >= 16:
                        prefix16_hashes.add(hash_part[:16])
                    elif len(hash_part) >= 8:
                        prefix8_hashes.add(hash_part[:8])
    
    total = len(full_hashes) + len(prefix8_hashes) + len(prefix16_hashes)
    print(f"Total hashes loaded: {total:,}")
    
    return full_hashes, prefix8_hashes, prefix16_hashes


def save_checkpoint(full_hashes: Set[str], prefix8_hashes: Set[str], prefix16_hashes: Set[str], stats: Dict):
    """Save checkpoint to disk."""
    checkpoint = {
        "processed_hashes_full": list(full_hashes),
        "processed_hashes_prefix8": list(prefix8_hashes),
        "processed_hashes_prefix16": list(prefix16_hashes),
        "hash_algo": "sha256",
        "last_updated": datetime.now().isoformat(),
        "stats": stats
    }
    
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)


def discover_files(categories: List[str]) -> List[str]:
    """Discover all files from source locations."""
    target_exts = set()
    for cat in categories:
        target_exts.update(FILE_CATEGORIES.get(cat, []))
    
    files = []
    for source in SOURCE_LOCATIONS:
        if not source.exists():
            print(f"Warning: Source not found: {source}")
            continue
        
        print(f"Scanning: {source}")
        try:
            for file_path in source.rglob("*"):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in target_exts:
                        files.append(str(file_path))
                        if len(files) % 100000 == 0:
                            print(f"  Discovered {len(files):,} files...")
        except PermissionError as e:
            print(f"Permission denied: {e}")
        except Exception as e:
            print(f"Error scanning {source}: {e}")
    
    return files


def process_file_batch(file_paths: List[str], hash_sets: Tuple[Set, Set, Set]) -> List[Dict]:
    """Process a batch of files."""
    full_set, prefix8_set, prefix16_set = hash_sets
    results = []
    
    for file_path_str in file_paths:
        try:
            file_path = Path(file_path_str)
            if not file_path.exists():
                continue
                
            ext = file_path.suffix.lower()
            category = EXT_TO_CATEGORY.get(ext, "other")
            
            if category not in ["code", "text", "html", "config"]:
                continue
            
            # Hash file
            file_hash = get_file_hash(file_path)
            
            # Skip if already processed
            if (file_hash in full_set or 
                file_hash[:8] in prefix8_set or 
                file_hash[:16] in prefix16_set):
                results.append({"skipped": True})
                continue
            
            # Read content
            encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
            content = None
            for enc in encodings:
                try:
                    with open(file_path, "r", encoding=enc, errors="replace") as f:
                        content = f.read()
                    break
                except:
                    continue
            
            if not content:
                results.append({"error": "Could not decode"})
                continue
            
            # Generate output
            safe_name = re.sub(r'[^\w\-_.]', '_', file_path.stem)[:100]
            output_name = f"{safe_name}_{file_hash[:HASH_PREFIX_LEN]}.md"
            output_path = OUTPUT_BASE / category / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            header = f"---\nsource: {file_path}\ntype: {category}\nsize: {file_path.stat().st_size}\nprocessed: {datetime.now().isoformat()}\n---\n\n"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(header + content)
            
            results.append({
                "hash": file_hash,
                "size": file_path.stat().st_size,
                "category": category
            })
        except Exception as e:
            results.append({"error": str(e)})
    
    return results


def main():
    """Main ingestion entry point."""
    print("=" * 80)
    print("  BIZRA PARALLEL INGESTION ENGINE v2.0")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load existing hashes
    full_hashes, prefix8_hashes, prefix16_hashes = load_checkpoint()
    hash_sets = (full_hashes, prefix8_hashes, prefix16_hashes)
    
    # Discover files
    print()
    print("Discovering files...")
    files_to_process = discover_files(["code"])
    print(f"Total files to scan: {len(files_to_process):,}")
    
    # Shuffle files to prevent getting stuck on one bad directory
    print("Shuffling processing order...")
    random.shuffle(files_to_process)
    
    print()
    
    # Process sequentially (avoid multiprocessing pickling issues with sets)
    processed = 0
    skipped = 0
    errors = 0
    new_hashes = []
    bytes_processed = 0
    category_stats = defaultdict(int)
    
    start_time = time.time()
    batch_size = 500
    total_batches = (len(files_to_process) + batch_size - 1) // batch_size
    
    print(f"Processing {total_batches} batches of {batch_size} files...")
    print()
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(files_to_process))
        batch = files_to_process[batch_start:batch_end]
        
        results = process_file_batch(batch, hash_sets)
        
        for result in results:
            if result.get("skipped"):
                skipped += 1
            elif "error" in result:
                errors += 1
            elif "hash" in result:
                processed += 1
                new_hashes.append(result["hash"])
                # Add to hash set immediately to avoid reprocessing
                full_hashes.add(result["hash"])
                bytes_processed += result.get("size", 0)
                category_stats[result.get("category", "unknown")] += 1
        
        # Progress update
        total_scanned = batch_end
        elapsed = time.time() - start_time
        rate = total_scanned / elapsed if elapsed > 0 else 0
        
        if batch_idx % 10 == 0:
            print(f"Progress: {total_scanned:,}/{len(files_to_process):,} "
                  f"| New: {processed:,} | Skip: {skipped:,} | Err: {errors:,} "
                  f"| Rate: {rate:.0f}/s", flush=True)
        
        # Periodic checkpoint
        if batch_idx % 200 == 0 and batch_idx > 0:
            save_checkpoint(full_hashes, prefix8_hashes, prefix16_hashes, {
                "total_processed": processed,
                "total_skipped": skipped,
                "total_errors": errors,
                "bytes_processed": bytes_processed
            })
            print(f"  [Checkpoint saved]")
    
    # Final save
    save_checkpoint(full_hashes, prefix8_hashes, prefix16_hashes, {
        "total_processed": processed,
        "total_skipped": skipped,
        "total_errors": errors,
        "bytes_processed": bytes_processed
    })
    
    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 80)
    print("  INGESTION COMPLETE")
    print("=" * 80)
    print(f"""
  Duration:    {elapsed/60:.1f} minutes
  Total Files: {len(files_to_process):,}
  Processed:   {processed:,}
  Skipped:     {skipped:,}
  Errors:      {errors:,}
  Data:        {bytes_processed / (1024**3):.2f} GB

  By Category:""")
    for cat, count in sorted(category_stats.items(), key=lambda x: -x[1]):
        print(f"    {cat:15} {count:,}")
    
    print(f"""
  Output:      {OUTPUT_BASE}
  Checkpoint:  {CHECKPOINT_PATH}
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
