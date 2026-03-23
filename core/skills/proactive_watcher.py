"""
Proactive Filesystem Watcher — Detect changes, suggest actions.

Scans key directories for new/modified files since last check.
Returns actionable suggestions for the Ghost Panel.

Standing on: Boyd (OODA observe), OpenClaw (24/7 awareness).
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List


def _state_path() -> Path:
    d = Path.home() / ".bizra"
    d.mkdir(parents=True, exist_ok=True)
    return d / "watcher_state.json"


def _load_state() -> Dict:
    p = _state_path()
    if p.exists():
        return json.loads(p.read_text())
    return {"last_check": 0, "known_files": {}}


def _save_state(state: Dict):
    _state_path().write_text(json.dumps(state))


def scan_for_changes(directories: List[str] = None) -> List[Dict]:
    """
    Scan directories for new/modified files since last check.
    Returns list of change dicts.
    """
    if directories is None:
        directories = [
            str(Path.home() / "Downloads"),
            str(Path.home() / "Desktop"),
            str(Path.home() / "Documents"),
        ]

    state = _load_state()
    _last_check = state.get("last_check", 0)  # noqa: F841
    known = state.get("known_files", {})
    changes = []

    for dir_path in directories:
        d = Path(dir_path)
        if not d.exists():
            continue

        try:
            for f in d.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    key = str(f)
                    mtime = f.stat().st_mtime
                    size = f.stat().st_size

                    if key not in known:
                        changes.append(
                            {
                                "type": "new",
                                "path": key,
                                "name": f.name,
                                "dir": dir_path,
                                "size": size,
                                "ext": f.suffix.lstrip(".").lower(),
                            }
                        )
                    elif mtime > known.get(key, {}).get("mtime", 0):
                        changes.append(
                            {
                                "type": "modified",
                                "path": key,
                                "name": f.name,
                                "dir": dir_path,
                                "size": size,
                                "ext": f.suffix.lstrip(".").lower(),
                            }
                        )

                    known[key] = {"mtime": mtime, "size": size}
        except PermissionError:
            pass

    state["last_check"] = time.time()
    state["known_files"] = known
    _save_state(state)

    return changes


def format_suggestions(changes: List[Dict]) -> List[str]:
    """Convert changes into Ghost Panel suggestions."""
    suggestions = []

    new_files = [c for c in changes if c["type"] == "new"]
    if new_files:
        by_dir = {}
        for f in new_files:
            by_dir.setdefault(os.path.basename(f["dir"]), []).append(f)

        for dir_name, files in by_dir.items():
            if len(files) == 1:
                suggestions.append(
                    f"New file in {dir_name}: {files[0]['name']}. Classify it?"
                )
            else:
                suggestions.append(
                    f"{len(files)} new files in {dir_name}. Want me to organize?"
                )

    modified = [c for c in changes if c["type"] == "modified"]
    if modified:
        suggestions.append(f"{len(modified)} files modified since last check.")

    return suggestions
