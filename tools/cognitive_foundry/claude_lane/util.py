"""Shared utilities for the Claude Cognitive Archive Pilot.

Keep stdlib-only so the pilot runs in any Python 3.10+ environment without
extra dependencies.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
import zipfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

# ----------------------------------------------------------------------------
# Archive loading
# ----------------------------------------------------------------------------

EXPECTED_FILES = ("users.json", "projects.json", "memories.json", "conversations.json")


def load_archive(zip_path: Path) -> Dict[str, Any]:
    """Open a Claude export zip and return the parsed JSON for each expected file.

    Tries root-level names first, then walks the archive for first-match of each
    expected filename. Raises FileNotFoundError if any expected file is missing.
    """

    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a zip archive: {zip_path}")

    parsed: Dict[str, Any] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        for expected in EXPECTED_FILES:
            matches = [n for n in all_names if n.endswith(expected)]
            if not matches:
                raise FileNotFoundError(
                    f"Expected file '{expected}' not found in archive {zip_path}"
                )
            # Pick the shortest path (prefers root over nested).
            matches.sort(key=len)
            with zf.open(matches[0]) as f:
                try:
                    parsed[expected] = json.loads(f.read().decode("utf-8"))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"File '{matches[0]}' inside archive is not valid JSON: {e}"
                    ) from e
    return parsed


def archive_digest(zip_path: Path) -> str:
    """Return a stable SHA-256 hex digest of the archive bytes (short form)."""

    h = hashlib.sha256()
    with zip_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def make_run_id(zip_path: Path, when: Optional[datetime] = None) -> str:
    """Compose a deterministic-ish run id from archive digest + UTC date."""

    when = when or datetime.now(tz=timezone.utc)
    return f"{when.strftime('%Y%m%dT%H%M%SZ')}_{archive_digest(zip_path)}"


# ----------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_EDGE_RE = re.compile(r"^[^\w]+|[^\w]+$")


def normalize_text(text: str) -> str:
    """Normalize for hashing + clustering.

    Lowercase, NFKC, collapse whitespace, strip leading/trailing punctuation.
    NOT lossy for substring matching — we keep words intact.
    """

    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.lower()
    t = _WHITESPACE_RE.sub(" ", t).strip()
    t = _PUNCT_EDGE_RE.sub("", t)
    return t


def candidate_id(
    candidate_type: str,
    normalized_text_value: str,
    source_message_uuid: str,
) -> str:
    """Deterministic 16-hex-char candidate id."""

    key = f"{candidate_type}|{normalized_text_value}|{source_message_uuid or ''}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def cluster_id(candidate_type: str, normalized_entity_predicate: str) -> str:
    """Deterministic cluster id for grouping."""

    key = f"{candidate_type}::{normalized_entity_predicate}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Parsing Claude conversation turns
# ----------------------------------------------------------------------------


def iter_turns(conversation: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield normalized turn dicts from a Claude conversation object.

    Yields:
        {
            "conversation_uuid": str,
            "conversation_name": str,
            "message_uuid": str,
            "speaker": "human" | "assistant" | "unknown",
            "text": str,
            "created_at": str (ISO),
        }
    """

    convo_uuid = conversation.get("uuid", "")
    convo_name = conversation.get("name", "") or ""
    messages = conversation.get("chat_messages") or []
    for msg in messages:
        raw_sender = (msg.get("sender") or "").lower()
        if raw_sender in ("human", "user"):
            speaker = "human"
        elif raw_sender in ("assistant", "model", "claude"):
            speaker = "assistant"
        else:
            speaker = "unknown"
        # Claude exports can use either 'text' or 'content' (structured list).
        text = msg.get("text") or ""
        if not text and isinstance(msg.get("content"), list):
            parts: List[str] = []
            for c in msg["content"]:
                if isinstance(c, dict):
                    if c.get("type") == "text" and c.get("text"):
                        parts.append(str(c["text"]))
            text = "\n".join(parts)
        yield {
            "conversation_uuid": convo_uuid,
            "conversation_name": convo_name,
            "message_uuid": msg.get("uuid", ""),
            "speaker": speaker,
            "text": text or "",
            "created_at": msg.get("created_at", "") or "",
        }


# ----------------------------------------------------------------------------
# Deterministic CSV writing
# ----------------------------------------------------------------------------


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    columns: List[str],
    line_terminator: str = "\n",
    encoding: str = "utf-8",
) -> None:
    """Write a deterministic CSV.

    - columns must be exactly the expected schema; extra keys in rows are
      silently dropped, missing keys emit empty strings.
    - rows are NOT re-sorted here; caller decides sort order.
    - newline="" + explicit line_terminator makes output stable across OSs.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            lineterminator=line_terminator,
            extrasaction="ignore",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({c: _csv_value(r.get(c, "")) for c in columns})


def _csv_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple, set)):
        return "|".join(str(x) for x in v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if is_dataclass(v):
        return json.dumps(asdict(v), sort_keys=True, ensure_ascii=False)
    return str(v)


# ----------------------------------------------------------------------------
# Manifest + output paths
# ----------------------------------------------------------------------------


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON manifest with stable key ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def stage_dir(run_root: Path, stage_num: int, stage_name: str) -> Path:
    """Return the stable stage directory path."""

    return run_root / f"{stage_num:02d}_{stage_name}"


# ----------------------------------------------------------------------------
# Topic bucket matching
# ----------------------------------------------------------------------------


def match_topic_buckets(
    text: str, buckets: List[Any]
) -> List[str]:
    """Return the list of bucket names whose keywords appear in text (case-insensitive).

    Uses whole-word-ish containment; does NOT tokenize. Callers should pass a
    pre-lowercased text if they want case-insensitive match without per-keyword
    lowering. This function lowercases the input internally to be safe.
    """

    if not text:
        return []
    hay = text.lower()
    hits: List[str] = []
    for b in buckets:
        name = getattr(b, "name", None)
        kws = getattr(b, "keywords", None) or []
        if not name:
            continue
        for kw in kws:
            if kw.lower() in hay:
                hits.append(name)
                break
    return hits
