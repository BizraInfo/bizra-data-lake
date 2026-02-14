#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
    FILE TYPE INDEXER — Sort Everything by Type First
    
    "Before mining for gold, organize the mine."
    
    Scans directories and organizes files by type:
    - Images (jpg, png, gif, webp, svg, ico, bmp, tiff)
    - Videos (mp4, mkv, avi, mov, webm, flv)
    - Audio (mp3, wav, flac, ogg, m4a, aac)
    - Documents (pdf, doc, docx, txt, rtf, odt)
    - Code (py, js, ts, java, cpp, c, cs, go, rs, rb, php, html, css)
    - Data (json, csv, xml, yaml, yml, parquet, sqlite, db)
    - Archives (zip, rar, 7z, tar, gz, bz2)
    - Notebooks (ipynb)
    - Diagrams (drawio, vsdx, mermaid)
    - Models (onnx, pt, pth, h5, pkl, joblib, safetensors)
    - Configs (ini, cfg, conf, env, toml)
    
    Created: 2026-01-23 | BIZRA Data Lake
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
from datetime import datetime
import time

# ═══════════════════════════════════════════════════════════════════════════════
# FILE TYPE CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════

FILE_CATEGORIES = {
    "images": {
        "extensions": [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp", ".tiff", ".tif", ".heic", ".heif", ".raw", ".cr2", ".nef"],
        "icon": "🖼️",
        "priority": "high"
    },
    "videos": {
        "extensions": [".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg", ".3gp"],
        "icon": "🎬",
        "priority": "high"
    },
    "audio": {
        "extensions": [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus", ".aiff"],
        "icon": "🎵",
        "priority": "medium"
    },
    "documents": {
        "extensions": [".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt", ".xls", ".xlsx", ".ppt", ".pptx", ".md", ".epub", ".mobi"],
        "icon": "📄",
        "priority": "high"
    },
    "code": {
        "extensions": [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", ".cs", ".go", ".rs", ".rb", ".php", ".html", ".htm", ".css", ".scss", ".sass", ".vue", ".svelte", ".swift", ".kt", ".scala", ".r", ".m", ".sh", ".bash", ".ps1", ".bat", ".cmd"],
        "icon": "💻",
        "priority": "critical"
    },
    "data": {
        "extensions": [".json", ".csv", ".xml", ".yaml", ".yml", ".parquet", ".sqlite", ".db", ".sql", ".jsonl", ".ndjson", ".tsv", ".arrow", ".feather", ".hdf5", ".h5"],
        "icon": "📊",
        "priority": "critical"
    },
    "archives": {
        "extensions": [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".tgz", ".tar.gz", ".tar.bz2"],
        "icon": "📦",
        "priority": "high"
    },
    "notebooks": {
        "extensions": [".ipynb"],
        "icon": "📓",
        "priority": "critical"
    },
    "diagrams": {
        "extensions": [".drawio", ".vsdx", ".puml", ".mermaid", ".excalidraw", ".fig", ".sketch"],
        "icon": "📐",
        "priority": "high"
    },
    "models": {
        "extensions": [".onnx", ".pt", ".pth", ".h5", ".pkl", ".joblib", ".safetensors", ".ckpt", ".bin", ".model", ".weights"],
        "icon": "🧠",
        "priority": "critical"
    },
    "configs": {
        "extensions": [".ini", ".cfg", ".conf", ".env", ".toml", ".properties", ".config"],
        "icon": "⚙️",
        "priority": "medium"
    },
    "fonts": {
        "extensions": [".ttf", ".otf", ".woff", ".woff2", ".eot"],
        "icon": "🔤",
        "priority": "low"
    },
    "executables": {
        "extensions": [".exe", ".msi", ".dll", ".so", ".dylib", ".app"],
        "icon": "⚡",
        "priority": "low"
    },
    "logs": {
        "extensions": [".log", ".logs"],
        "icon": "📋",
        "priority": "low"
    }
}

# Build extension lookup
EXT_TO_CATEGORY = {}
for category, config in FILE_CATEGORIES.items():
    for ext in config["extensions"]:
        EXT_TO_CATEGORY[ext.lower()] = category


@dataclass
class FileInfo:
    """Information about a single file."""
    path: str
    name: str
    extension: str
    category: str
    size_bytes: int
    modified: str
    created: str
    is_hidden: bool = False
    content_hash: Optional[str] = None  # BLAKE2b hash when computed
    
    @property
    def size_human(self) -> str:
        """Human readable size."""
        size = self.size_bytes
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"


@dataclass
class CategoryStats:
    """Statistics for a file category."""
    category: str
    icon: str
    count: int = 0
    total_size: int = 0
    files: List[str] = field(default_factory=list)
    extensions: Dict[str, int] = field(default_factory=dict)
    
    @property
    def size_human(self) -> str:
        size = self.total_size
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"


class FileTypeIndexer:
    """
    Indexes files by type across directories.
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(r"C:\BIZRA-DATA-LAKE\03_INDEXED\file_index")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.files: List[FileInfo] = []
        self.categories: Dict[str, CategoryStats] = {}
        self.errors: List[str] = []
        
        # Initialize category stats
        for cat, config in FILE_CATEGORIES.items():
            self.categories[cat] = CategoryStats(
                category=cat,
                icon=config["icon"]
            )
        self.categories["other"] = CategoryStats(category="other", icon="❓")
    
    def scan_directory(self, path: Path, recursive: bool = True, 
                       skip_hidden: bool = False, max_files: int = None) -> int:
        """Scan a directory and index all files."""
        path = Path(path)
        if not path.exists():
            print(f"  ⚠️  Path does not exist: {path}")
            return 0
        
        print(f"  📂 Scanning: {path}")
        count = 0
        
        try:
            iterator = path.rglob("*") if recursive else path.glob("*")
            
            for item in iterator:
                if max_files and count >= max_files:
                    print(f"     ⚠️  Reached max files limit: {max_files}")
                    break
                
                if not item.is_file():
                    continue
                
                # Skip hidden files if requested
                if skip_hidden and any(part.startswith('.') for part in item.parts):
                    continue
                
                try:
                    file_info = self._index_file(item)
                    if file_info:
                        self.files.append(file_info)
                        self._update_stats(file_info)
                        count += 1
                        
                        if count % 1000 == 0:
                            print(f"     ... {count:,} files indexed")
                            
                except PermissionError:
                    self.errors.append(f"Permission denied: {item}")
                except Exception as e:
                    self.errors.append(f"Error indexing {item}: {e}")
        
        except PermissionError:
            print(f"  ⚠️  Permission denied for: {path}")
        except Exception as e:
            print(f"  ⚠️  Error scanning {path}: {e}")
        
        print(f"     ✓ Indexed {count:,} files from {path.name}")
        return count
    
    def _index_file(self, path: Path) -> Optional[FileInfo]:
        """Index a single file."""
        try:
            stat = path.stat()
            ext = path.suffix.lower()
            category = EXT_TO_CATEGORY.get(ext, "other")
            
            return FileInfo(
                path=str(path),
                name=path.name,
                extension=ext,
                category=category,
                size_bytes=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                created=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                is_hidden=path.name.startswith('.')
            )
        except Exception:
            return None
    
    def _update_stats(self, file_info: FileInfo):
        """Update category statistics."""
        cat = self.categories.get(file_info.category, self.categories["other"])
        cat.count += 1
        cat.total_size += file_info.size_bytes
        cat.extensions[file_info.extension] = cat.extensions.get(file_info.extension, 0) + 1
        
        # Store file paths (limit to avoid memory issues)
        if len(cat.files) < 10000:
            cat.files.append(file_info.path)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of indexed files."""
        total_size = sum(cat.total_size for cat in self.categories.values())
        
        summary = {
            "total_files": len(self.files),
            "total_size_bytes": total_size,
            "total_size_human": self._size_human(total_size),
            "categories": {},
            "errors": len(self.errors)
        }
        
        for cat_name, cat in sorted(self.categories.items(), key=lambda x: x[1].count, reverse=True):
            if cat.count > 0:
                summary["categories"][cat_name] = {
                    "icon": cat.icon,
                    "count": cat.count,
                    "size": cat.size_human,
                    "size_bytes": cat.total_size,
                    "top_extensions": dict(sorted(cat.extensions.items(), key=lambda x: x[1], reverse=True)[:10])
                }
        
        return summary
    
    def _size_human(self, size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        
        print("\n" + "═" * 70)
        print("  📊 FILE TYPE INDEX SUMMARY")
        print("═" * 70)
        print(f"  Total Files: {summary['total_files']:,}")
        print(f"  Total Size:  {summary['total_size_human']}")
        print("═" * 70)
        
        print("\n  BY CATEGORY:")
        print("  " + "-" * 60)
        
        for cat_name, data in summary["categories"].items():
            icon = data["icon"]
            count = data["count"]
            size = data["size"]
            bar = "█" * min(count // 100, 20)
            print(f"  {icon} {cat_name:15} {count:8,} files  {size:>12}  {bar}")
        
        print("\n  " + "-" * 60)
        
        # Show top extensions per critical category
        for cat_name in ["code", "data", "archives", "documents"]:
            if cat_name in summary["categories"]:
                exts = summary["categories"][cat_name].get("top_extensions", {})
                if exts:
                    ext_str = ", ".join([f"{e}({c})" for e, c in list(exts.items())[:5]])
                    print(f"     {cat_name}: {ext_str}")
        
        if summary["errors"] > 0:
            print(f"\n  ⚠️  {summary['errors']} errors during indexing")
    
    def save_index(self, name: str = "file_index"):
        """Save index to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_file = self.output_dir / f"{name}_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_summary(), f, indent=2, ensure_ascii=False)
        
        # Save detailed index by category
        for cat_name, cat in self.categories.items():
            if cat.count > 0:
                cat_file = self.output_dir / f"{name}_{cat_name}_{timestamp}.json"
                cat_data = {
                    "category": cat_name,
                    "icon": cat.icon,
                    "count": cat.count,
                    "size": cat.size_human,
                    "files": cat.files[:10000]  # Limit to prevent huge files
                }
                with open(cat_file, 'w', encoding='utf-8') as f:
                    json.dump(cat_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n  ✓ Index saved to: {self.output_dir}")
        return summary_file
    
    def find_golden_gems(self) -> List[Dict[str, Any]]:
        """Identify potentially valuable files."""
        gems = []
        
        # Large archives (might contain datasets)
        for f in self.files:
            if f.category == "archives" and f.size_bytes > 10 * 1024 * 1024:  # >10MB
                gems.append({
                    "type": "large_archive",
                    "path": f.path,
                    "size": f.size_human,
                    "reason": "Large archive - may contain datasets"
                })
        
        # Database files
        for f in self.files:
            if f.extension in [".sqlite", ".db", ".sql"]:
                gems.append({
                    "type": "database",
                    "path": f.path,
                    "size": f.size_human,
                    "reason": "Database file - structured data"
                })
        
        # Jupyter notebooks
        for f in self.files:
            if f.extension == ".ipynb":
                gems.append({
                    "type": "notebook",
                    "path": f.path,
                    "size": f.size_human,
                    "reason": "Jupyter notebook - code + analysis"
                })
        
        # Model files
        for f in self.files:
            if f.category == "models":
                gems.append({
                    "type": "ml_model",
                    "path": f.path,
                    "size": f.size_human,
                    "reason": "ML model file"
                })
        
        # Large data files
        for f in self.files:
            if f.category == "data" and f.size_bytes > 1 * 1024 * 1024:  # >1MB
                gems.append({
                    "type": "large_data",
                    "path": f.path,
                    "size": f.size_human,
                    "reason": f"Large {f.extension} data file"
                })
        
        return gems[:100]  # Limit output


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="File Type Indexer")
    parser.add_argument("paths", nargs="*", help="Paths to scan")
    parser.add_argument("--downloads", action="store_true", help="Scan Downloads folder")
    parser.add_argument("--all", action="store_true", help="Scan all common locations")
    parser.add_argument("--max-files", type=int, default=None, help="Max files to index")
    parser.add_argument("--gems", action="store_true", help="Show golden gems")
    
    args = parser.parse_args()
    
    indexer = FileTypeIndexer()
    
    print("═" * 70)
    print("  📂 FILE TYPE INDEXER — Sort Everything by Type")
    print("═" * 70)
    
    paths_to_scan = []
    
    if args.downloads:
        downloads = Path.home() / "Downloads"
        paths_to_scan.append(downloads)
        # Also check OneDrive Downloads
        onedrive_downloads = Path.home() / "OneDrive" / "Downloads"
        if onedrive_downloads.exists():
            paths_to_scan.append(onedrive_downloads)
        # Check the nested path from earlier
        nested_downloads = Path(r"C:\Users\BIZRA-OS\Downloads\OneDrive\Desktop\OneDrive\Downloads")
        if nested_downloads.exists():
            paths_to_scan.append(nested_downloads)
    
    if args.all:
        # Common locations
        paths_to_scan.extend([
            Path.home() / "Downloads",
            Path.home() / "Documents",
            Path.home() / "Desktop",
            Path(r"C:\BIZRA-DATA-LAKE"),
        ])
        # Check for additional drives
        for drive in ["D:", "E:", "F:"]:
            if Path(f"{drive}\\").exists():
                paths_to_scan.append(Path(f"{drive}\\"))
    
    if args.paths:
        paths_to_scan.extend([Path(p) for p in args.paths])
    
    if not paths_to_scan:
        # Default: scan Downloads
        paths_to_scan = [Path.home() / "Downloads"]
    
    # Scan all paths
    start = time.time()
    
    for path in paths_to_scan:
        if path.exists():
            indexer.scan_directory(path, max_files=args.max_files)
    
    elapsed = time.time() - start
    
    # Print summary
    indexer.print_summary()
    
    print(f"\n  ⏱️  Indexing completed in {elapsed:.1f}s")
    
    # Show golden gems
    if args.gems:
        gems = indexer.find_golden_gems()
        if gems:
            print("\n" + "═" * 70)
            print("  💎 GOLDEN GEMS FOUND")
            print("═" * 70)
            for gem in gems[:20]:
                print(f"  [{gem['type']:15}] {gem['size']:>10} | {gem['reason']}")
                print(f"                      {gem['path']}")
    
    # Save index
    indexer.save_index()
    
    return indexer


if __name__ == "__main__":
    main()
