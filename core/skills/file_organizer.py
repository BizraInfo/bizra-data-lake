"""
File Organizer — The skill that MOVES files.

Scans a directory, classifies files, creates folders, moves files.
Every operation produces a receipt-ready action log.

Standing on: Unix philosophy (do one thing well), Guardian veto (safe-by-default).
"""

import shutil
from pathlib import Path
from typing import Dict, List
from collections import Counter

# Classification rules (extension → category)
CATEGORIES = {
    # Documents
    "pdf": "Documents/PDFs",
    "doc": "Documents/Word",
    "docx": "Documents/Word",
    "txt": "Documents/Text",
    "md": "Documents/Markdown",
    "rtf": "Documents/Other",
    "odt": "Documents/Other",
    "csv": "Documents/Data",
    "xlsx": "Documents/Spreadsheets",
    "xls": "Documents/Spreadsheets",
    "pptx": "Documents/Presentations",
    "ppt": "Documents/Presentations",
    # Images
    "jpg": "Images/Photos",
    "jpeg": "Images/Photos",
    "png": "Images/Screenshots",
    "gif": "Images/GIFs",
    "svg": "Images/Vector",
    "webp": "Images/Web",
    "bmp": "Images/Other",
    "ico": "Images/Icons",
    # Code
    "py": "Code/Python",
    "rs": "Code/Rust",
    "js": "Code/JavaScript",
    "ts": "Code/TypeScript",
    "jsx": "Code/React",
    "tsx": "Code/React",
    "html": "Code/Web",
    "css": "Code/Web",
    "json": "Code/Config",
    "yaml": "Code/Config",
    "yml": "Code/Config",
    "toml": "Code/Config",
    "sh": "Code/Scripts",
    "bat": "Code/Scripts",
    "ps1": "Code/Scripts",
    # Media
    "mp4": "Media/Video",
    "mkv": "Media/Video",
    "avi": "Media/Video",
    "mov": "Media/Video",
    "mp3": "Media/Audio",
    "wav": "Media/Audio",
    "flac": "Media/Audio",
    # Archives
    "zip": "Archives",
    "tar": "Archives",
    "gz": "Archives",
    "7z": "Archives",
    "rar": "Archives",
    # Installers
    "exe": "Installers",
    "msi": "Installers",
    "deb": "Installers",
    "AppImage": "Installers",
}


def scan_directory(target: str) -> List[Dict]:
    """Scan directory and classify all files."""
    target_path = Path(target).expanduser().resolve()
    if not target_path.exists():
        return []

    files = []
    for f in target_path.iterdir():
        if f.is_file() and not f.name.startswith("."):
            ext = f.suffix.lstrip(".").lower()
            category = CATEGORIES.get(ext, "Other")
            files.append(
                {
                    "name": f.name,
                    "path": str(f),
                    "ext": ext,
                    "size": f.stat().st_size,
                    "category": category,
                    "modified": f.stat().st_mtime,
                }
            )

    return sorted(files, key=lambda x: x["category"])


def generate_plan(files: List[Dict], target: str) -> Dict:
    """Generate organization plan."""
    categories = Counter(f["category"] for f in files)
    return {
        "target": target,
        "total_files": len(files),
        "categories": dict(categories.most_common()),
        "folders_to_create": sorted(set(f["category"] for f in files)),
        "files": files,
    }


def execute_plan(plan: Dict, dry_run: bool = False) -> Dict:
    """Execute the organization plan. Move files into category folders."""
    target = Path(plan["target"]).expanduser().resolve()
    moved = 0
    errors = 0
    actions = []
    undo_log = []

    for folder in plan["folders_to_create"]:
        dest = target / folder
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)

    for f in plan["files"]:
        src = Path(f["path"])
        dest_dir = target / f["category"]
        dest = dest_dir / f["name"]

        # Handle name collision
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            counter = 1
            while dest.exists():
                dest = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        try:
            if not dry_run:
                shutil.move(str(src), str(dest))
            moved += 1
            actions.append({"action": "move", "from": str(src), "to": str(dest)})
            undo_log.append({"from": str(dest), "to": str(src)})
        except Exception as e:
            errors += 1
            actions.append({"action": "error", "file": str(src), "error": str(e)})

    return {
        "moved": moved,
        "errors": errors,
        "actions": actions,
        "undo_log": undo_log,
        "dry_run": dry_run,
    }


def format_plan(plan: Dict) -> str:
    """Format plan for user display."""
    lines = [f"Scanned {plan['total_files']} files in {plan['target']}"]
    lines.append("Plan:")
    for cat, count in sorted(plan["categories"].items(), key=lambda x: -x[1]):
        lines.append(f"  {cat}: {count} files")
    return "\n".join(lines)


def format_result(result: Dict) -> str:
    """Format execution result."""
    if result["dry_run"]:
        return (
            f"DRY RUN: Would move {result['moved']} files ({result['errors']} errors)"
        )
    return f"Moved {result['moved']} files ({result['errors']} errors)"


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "~/Downloads"
    files = scan_directory(target)
    plan = generate_plan(files, target)
    print(format_plan(plan))
    print(
        f"\nTotal: {plan['total_files']} files → {len(plan['folders_to_create'])} folders"
    )
