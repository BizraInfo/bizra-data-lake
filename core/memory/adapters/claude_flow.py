"""
Claude-Flow adapter for importing memory.db and JSON artifacts into AgentDB.

This bridge is read-only. It imports richer historical context from:
- .swarm/memory.db
- .claude-flow/memory/*.json

Malformed JSON artifacts are reported and skipped. They never crash the pipeline.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from core.proof_engine.canonical import hex_digest

from ..types import MemoryKind, MemoryRecord, RecordState

logger = logging.getLogger(__name__)

PARSER_VERSION = "1.0.0"
DEFAULT_DB_PATH = Path(".swarm") / "memory.db"
DEFAULT_ARTIFACT_DIR = Path(".claude-flow") / "memory"
_KNOWN_ARTIFACT_SOURCES = {
    "session-index.json": "claude_flow_session_index",
    "project-patterns.json": "claude_flow_project_patterns",
}


@dataclass
class ClaudeFlowImportIssue:
    source: str
    code: str
    message: str
    path: str | None = None

    def to_message(self) -> str:
        location = f" ({self.path})" if self.path else ""
        return f"{self.source}:{self.code}{location}: {self.message}"


@dataclass
class ClaudeFlowImportBatch:
    records: list[MemoryRecord]
    issues: list[ClaudeFlowImportIssue]


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        raw = float(value)
        if raw > 1_000_000_000_000:
            raw /= 1000.0
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    if isinstance(value, str) and value.strip():
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _normalize_kind(value: Any, default: MemoryKind = MemoryKind.SEMANTIC) -> MemoryKind:
    if isinstance(value, str):
        try:
            return MemoryKind(value.lower())
        except ValueError:
            return default
    return default


def _normalize_state(value: Any) -> RecordState:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "archived":
            return RecordState.ARCHIVED
        if lowered == "deleted":
            return RecordState.DELETED
    return RecordState.ACTIVE


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return list(dict.fromkeys(items))
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                return _normalize_tags(decoded)
        parts = [part.strip() for part in raw.replace(",", " ").split() if part.strip()]
        return list(dict.fromkeys(parts))
    return [str(value).strip()]


def _normalize_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        if isinstance(decoded, dict):
            return decoded
        return {"raw": decoded}
    return {}


def _stable_record_id(content: str, source: str, source_id: str | None) -> str:
    payload = content + source + (source_id or "")
    return hex_digest(payload.encode("utf-8"))[:16]


def _json_preview(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


class ClaudeFlowAdapter:
    """Read-only importer for Claude-flow memory surfaces."""

    def __init__(
        self,
        db_path: Path | None = None,
        artifact_dir: Path | None = None,
        strict_json: bool = False,
    ) -> None:
        self._db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self._artifact_dir = (
            Path(artifact_dir) if artifact_dir is not None else DEFAULT_ARTIFACT_DIR
        )
        self._strict_json = strict_json

    def export_db(self) -> ClaudeFlowImportBatch:
        records: list[MemoryRecord] = []
        issues: list[ClaudeFlowImportIssue] = []
        if not self._db_path.exists():
            return ClaudeFlowImportBatch(records=records, issues=issues)

        try:
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            issues.append(
                ClaudeFlowImportIssue(
                    source="claude_flow_db",
                    code="open_failed",
                    message=str(exc),
                    path=str(self._db_path),
                )
            )
            return ClaudeFlowImportBatch(records=records, issues=issues)

        try:
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "memory_entries" in tables:
                records.extend(self._export_memory_entries(conn))
            if "patterns" in tables:
                records.extend(self._export_patterns(conn))
        except sqlite3.Error as exc:
            issues.append(
                ClaudeFlowImportIssue(
                    source="claude_flow_db",
                    code="query_failed",
                    message=str(exc),
                    path=str(self._db_path),
                )
            )
        finally:
            conn.close()

        return ClaudeFlowImportBatch(records=records, issues=issues)

    def export_artifacts(self) -> ClaudeFlowImportBatch:
        records: list[MemoryRecord] = []
        issues: list[ClaudeFlowImportIssue] = []
        if not self._artifact_dir.exists():
            return ClaudeFlowImportBatch(records=records, issues=issues)

        for path in sorted(self._artifact_dir.glob("*.json")):
            source = _KNOWN_ARTIFACT_SOURCES.get(path.name, "claude_flow_artifact")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                issues.append(
                    ClaudeFlowImportIssue(
                        source=source,
                        code="invalid_json",
                        message=str(exc),
                        path=str(path),
                    )
                )
                if self._strict_json:
                    logger.warning("Claude-flow JSON rejected: %s", path)
                continue

            records.extend(self._records_from_artifact(source, path, payload))

        return ClaudeFlowImportBatch(records=records, issues=issues)

    def _export_memory_entries(self, conn: sqlite3.Connection) -> list[MemoryRecord]:
        rows = conn.execute(
            """
            SELECT id, key, namespace, type, content, tags, metadata, owner_id,
                   created_at, updated_at, last_accessed_at, access_count, status
            FROM memory_entries
            WHERE status != 'deleted'
            ORDER BY updated_at DESC
            """
        ).fetchall()
        records: list[MemoryRecord] = []
        for row in rows:
            record = self._memory_entry_to_record(dict(row))
            if record is not None:
                records.append(record)
        return records

    def _export_patterns(self, conn: sqlite3.Connection) -> list[MemoryRecord]:
        rows = conn.execute("SELECT * FROM patterns ORDER BY rowid DESC").fetchall()
        records: list[MemoryRecord] = []
        for row in rows:
            record = self._pattern_row_to_record(dict(row))
            if record is not None:
                records.append(record)
        return records

    def _memory_entry_to_record(self, row: dict[str, Any]) -> MemoryRecord | None:
        content = str(row.get("content") or row.get("key") or "").strip()
        if not content:
            return None
        source = "claude_flow_db"
        source_id = str(row.get("id") or row.get("key") or "")
        metadata = _normalize_metadata(row.get("metadata"))
        related_ids = metadata.get("related_ids") or metadata.get("relatedIds") or []
        if not isinstance(related_ids, list):
            related_ids = []
        created_at = _coerce_datetime(row.get("created_at"))
        updated_at = _coerce_datetime(row.get("updated_at"))
        last_accessed = _coerce_datetime(
            row.get("last_accessed_at") or row.get("updated_at") or row.get("created_at")
        )

        metadata.update(
            {
                "origin": "claude_flow",
                "source_path": str(self._db_path),
                "table_name": "memory_entries",
                "original_namespace": row.get("namespace"),
                "original_type": row.get("type"),
                "original_key": row.get("key"),
                "original_status": row.get("status"),
                "parser_version": PARSER_VERSION,
            }
        )

        tags = _normalize_tags(row.get("tags"))
        if row.get("namespace"):
            tags = list(dict.fromkeys(tags + [str(row.get("namespace")).strip()]))

        return MemoryRecord(
            id=_stable_record_id(content, source, source_id),
            content=content,
            kind=_normalize_kind(row.get("type")),
            state=_normalize_state(row.get("status")),
            importance=0.6,
            source=source,
            source_id=source_id,
            related_ids=[str(item) for item in related_ids if str(item).strip()],
            tags=tags,
            created_at=created_at,
            updated_at=updated_at,
            last_accessed=last_accessed,
            access_count=int(row.get("access_count") or 0),
            metadata=metadata,
        )

    def _pattern_row_to_record(self, row: dict[str, Any]) -> MemoryRecord | None:
        pattern_name = str(row.get("name") or row.get("pattern_name") or row.get("id") or "").strip()
        pattern_type = str(row.get("pattern_type") or row.get("type") or "pattern").strip()
        content_bits = [
            bit
            for bit in (
                pattern_name,
                str(row.get("description") or "").strip(),
                str(row.get("pattern") or "").strip(),
                str(row.get("summary") or "").strip(),
            )
            if bit
        ]
        content = "\n".join(content_bits) if content_bits else _json_preview(row)
        source = "claude_flow_patterns"
        source_id = str(row.get("id") or pattern_name or pattern_type)
        created_at = _coerce_datetime(row.get("created_at") or row.get("updated_at"))
        updated_at = _coerce_datetime(row.get("updated_at") or row.get("created_at"))

        return MemoryRecord(
            id=_stable_record_id(content, source, source_id),
            content=content,
            kind=MemoryKind.PROCEDURAL,
            state=RecordState.ACTIVE,
            importance=float(row.get("confidence") or row.get("score") or 0.7),
            source=source,
            source_id=source_id,
            tags=_normalize_tags(row.get("tags")) + [pattern_type, "pattern"],
            created_at=created_at,
            updated_at=updated_at,
            last_accessed=updated_at,
            metadata={
                "origin": "claude_flow",
                "source_path": str(self._db_path),
                "table_name": "patterns",
                "pattern_type": pattern_type,
                "parser_version": PARSER_VERSION,
                "original_row": row,
            },
        )

    def _records_from_artifact(
        self,
        source: str,
        path: Path,
        payload: Any,
    ) -> list[MemoryRecord]:
        if source == "claude_flow_session_index" and isinstance(payload, dict):
            return self._session_index_records(path, payload)
        if source == "claude_flow_project_patterns" and isinstance(payload, dict):
            return self._project_pattern_records(path, payload)
        return self._generic_artifact_records(source, path, payload)

    def _session_index_records(self, path: Path, payload: dict[str, Any]) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for session in payload.get("sessions", []):
            if not isinstance(session, dict):
                continue
            content = str(session.get("summary") or _json_preview(session)).strip()
            source_id = str(session.get("session_id") or session.get("timestamp") or len(records))
            timestamp = _coerce_datetime(session.get("timestamp") or payload.get("updated"))
            records.append(
                MemoryRecord(
                    id=_stable_record_id(content, "claude_flow_session_index", source_id),
                    content=content,
                    kind=MemoryKind.EPISODIC,
                    state=RecordState.ACTIVE,
                    importance=0.65,
                    source="claude_flow_session_index",
                    source_id=source_id,
                    tags=["session", "claude_flow"],
                    created_at=timestamp,
                    updated_at=timestamp,
                    last_accessed=timestamp,
                    metadata={
                        "origin": "claude_flow_artifact",
                        "source_path": str(path),
                        "artifact_type": "session-index",
                        "parser_version": PARSER_VERSION,
                        "session": session,
                    },
                )
            )
        return records

    def _project_pattern_records(self, path: Path, payload: dict[str, Any]) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        patterns = payload.get("patterns", {})
        if not isinstance(patterns, dict):
            return records
        for key, value in patterns.items():
            content = f"{key}\n{_json_preview(value)}"
            source_id = str(key)
            timestamp = datetime.now(timezone.utc)
            records.append(
                MemoryRecord(
                    id=_stable_record_id(content, "claude_flow_project_patterns", source_id),
                    content=content,
                    kind=MemoryKind.PROCEDURAL,
                    state=RecordState.ACTIVE,
                    importance=0.7,
                    source="claude_flow_project_patterns",
                    source_id=source_id,
                    tags=["pattern", "project"],
                    created_at=timestamp,
                    updated_at=timestamp,
                    last_accessed=timestamp,
                    metadata={
                        "origin": "claude_flow_artifact",
                        "source_path": str(path),
                        "artifact_type": "project-patterns",
                        "parser_version": PARSER_VERSION,
                    },
                )
            )
        return records

    def _generic_artifact_records(
        self,
        source: str,
        path: Path,
        payload: Any,
    ) -> list[MemoryRecord]:
        items: Iterable[tuple[str, Any]]
        if isinstance(payload, dict):
            items = payload.items()
        elif isinstance(payload, list):
            items = [(str(index), item) for index, item in enumerate(payload)]
        else:
            items = [(path.stem, payload)]

        timestamp = datetime.now(timezone.utc)
        records: list[MemoryRecord] = []
        for key, value in items:
            content = f"{key}\n{_json_preview(value)}"
            records.append(
                MemoryRecord(
                    id=_stable_record_id(content, source, str(key)),
                    content=content,
                    kind=MemoryKind.SEMANTIC,
                    state=RecordState.ACTIVE,
                    importance=0.55,
                    source=source,
                    source_id=str(key),
                    tags=["artifact", path.stem],
                    created_at=timestamp,
                    updated_at=timestamp,
                    last_accessed=timestamp,
                    metadata={
                        "origin": "claude_flow_artifact",
                        "source_path": str(path),
                        "parser_version": PARSER_VERSION,
                    },
                )
            )
        return records
