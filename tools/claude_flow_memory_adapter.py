#!/usr/bin/env python3
"""
Claude-Flow Memory Adapter

Reads the local .swarm/memory.db SQLite database and provides basic
stats, query, and export operations for BIZRA workflows.
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DB = PROJECT_ROOT / ".swarm" / "memory.db"
DEFAULT_EXPORT = PROJECT_ROOT / "04_GOLD" / "claude_flow_memory_export.jsonl"


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _list_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [row["name"] for row in rows]


def _table_counts(conn: sqlite3.Connection, tables: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for table in tables:
        try:
            row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()  # nosec B608 — table names from sqlite_master
            counts[table] = int(row["count"]) if row else 0
        except sqlite3.Error:
            counts[table] = -1
    return counts


def _build_where(
    namespace: Optional[str],
    entry_type: Optional[str],
    status: Optional[str],
    query: Optional[str],
) -> Tuple[str, List[Any]]:
    clauses: List[str] = []
    params: List[Any] = []

    if status:
        clauses.append("status = ?")
        params.append(status)
    if namespace:
        clauses.append("namespace = ?")
        params.append(namespace)
    if entry_type:
        clauses.append("type = ?")
        params.append(entry_type)
    if query:
        like = f"%{query}%"
        clauses.append("(content LIKE ? OR key LIKE ? OR tags LIKE ? OR metadata LIKE ?)")
        params.extend([like, like, like, like])

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def _ms_to_iso(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _format_times(row: Dict[str, Any], iso_times: bool) -> Dict[str, Any]:
    if not iso_times:
        return row
    for key in list(row.keys()):
        if key.endswith("_at"):
            row[key] = _ms_to_iso(row.get(key))
    return row


def _memory_entries_query(
    conn: sqlite3.Connection,
    query: Optional[str],
    namespace: Optional[str],
    entry_type: Optional[str],
    status: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    where, params = _build_where(namespace, entry_type, status, query)
    sql = (
        "SELECT id, key, namespace, type, content, tags, metadata, owner_id, "
        "created_at, updated_at, expires_at, last_accessed_at, access_count, status "
        "FROM memory_entries "
        f"{where} "  # nosec B608 — where clause uses ? params
        "ORDER BY updated_at DESC "
        "LIMIT ?"
    )
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def _memory_entries_iter(
    conn: sqlite3.Connection,
    namespace: Optional[str],
    entry_type: Optional[str],
    status: Optional[str],
) -> Iterable[Dict[str, Any]]:
    where, params = _build_where(namespace, entry_type, status, None)
    sql = (
        "SELECT id, key, namespace, type, content, tags, metadata, owner_id, "
        "created_at, updated_at, expires_at, last_accessed_at, access_count, status "
        "FROM memory_entries "
        f"{where} "  # nosec B608 — where clause uses ? params
        "ORDER BY updated_at DESC"
    )
    for row in conn.execute(sql, params):
        yield _row_to_dict(row)


def _to_sovereign_row(row: Dict[str, Any], snr_score: float, iso_times: bool) -> Dict[str, Any]:
    content = row.get("content") or ""
    modified_ms = row.get("updated_at") or row.get("created_at")
    modified = _ms_to_iso(modified_ms) if iso_times else modified_ms
    size_bytes = len(content.encode("utf-8")) if isinstance(content, str) else 0

    return {
        "id": row.get("id"),
        "path": f"claude-flow://memory_entries/{row.get('id')}",
        "name": row.get("key"),
        "kind": f"Memory/{row.get('type')}",
        "size_bytes": size_bytes,
        "modified": modified,
        "snr_score": snr_score,
        "domain_source": "claude-flow",
        "metadata": {
            "namespace": row.get("namespace"),
            "status": row.get("status"),
            "tags": row.get("tags"),
            "owner_id": row.get("owner_id"),
            "access_count": row.get("access_count"),
            "created_at": _ms_to_iso(row.get("created_at")) if iso_times else row.get("created_at"),
            "last_accessed_at": _ms_to_iso(row.get("last_accessed_at")) if iso_times else row.get("last_accessed_at"),
        },
    }


def cmd_stats(args: argparse.Namespace) -> int:
    conn = _connect(args.db)
    try:
        tables = _list_tables(conn)
        counts = _table_counts(conn, tables)

        data: Dict[str, Any] = {
            "db_path": str(args.db),
            "tables": counts,
        }

        if "metadata" in tables:
            meta_rows = conn.execute("SELECT key, value FROM metadata").fetchall()
            data["metadata"] = {r["key"]: r["value"] for r in meta_rows}

        if "memory_entries" in tables:
            for field, key in (("namespace", "by_namespace"), ("type", "by_type"), ("status", "by_status")):
                rows = conn.execute(
                    f"SELECT {field} AS label, COUNT(*) AS count FROM memory_entries GROUP BY {field} ORDER BY count DESC"  # nosec B608 — field is internal literal
                ).fetchall()
                data.setdefault("memory_entries", {})[key] = {r["label"]: int(r["count"]) for r in rows}

        if "patterns" in tables:
            rows = conn.execute(
                "SELECT pattern_type AS label, COUNT(*) AS count FROM patterns GROUP BY pattern_type ORDER BY count DESC"
            ).fetchall()
            data["patterns"] = {r["label"]: int(r["count"]) for r in rows}

        if args.json:
            print(json.dumps(data, indent=2))
            return 0

        print(f"DB: {data['db_path']}")
        if "metadata" in data:
            print("Metadata:")
            for k, v in data["metadata"].items():
                print(f"  {k}: {v}")
        print("Table counts:")
        for name, count in counts.items():
            print(f"  {name}: {count}")

        if "memory_entries" in data:
            print("Memory entries:")
            for key, mapping in data["memory_entries"].items():
                print(f"  {key}:")
                for label, count in mapping.items():
                    print(f"    {label}: {count}")

        if "patterns" in data:
            print("Patterns by type:")
            for label, count in data["patterns"].items():
                print(f"  {label}: {count}")

        return 0
    finally:
        conn.close()


def cmd_query(args: argparse.Namespace) -> int:
    conn = _connect(args.db)
    try:
        rows = _memory_entries_query(
            conn,
            query=args.text,
            namespace=args.namespace,
            entry_type=args.entry_type,
            status=args.status,
            limit=args.limit,
        )
        formatted = [_format_times(r, args.iso_times) for r in rows]

        if args.json or args.jsonl:
            if args.jsonl:
                for row in formatted:
                    print(json.dumps(row))
            else:
                print(json.dumps(formatted, indent=2))
            return 0

        for row in formatted:
            content = row.get("content") or ""
            preview = content.replace("\n", " ").strip()
            if len(preview) > 120:
                preview = preview[:117] + "..."
            print(f"[{row.get('namespace')}] {row.get('key')} ({row.get('type')}) -> {preview}")
        return 0
    finally:
        conn.close()


def cmd_export(args: argparse.Namespace) -> int:
    conn = _connect(args.db)
    try:
        out_path = Path(args.out or DEFAULT_EXPORT)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.format == "json":
            buffer: List[Dict[str, Any]] = []
            for row in _memory_entries_iter(conn, args.namespace, args.entry_type, args.status):
                if args.as_sovereign:
                    row = _to_sovereign_row(row, args.snr, args.iso_times)
                else:
                    row = _format_times(row, args.iso_times)
                buffer.append(row)

            out_path.write_text(json.dumps(buffer, indent=2))
            print(f"Exported {len(buffer)} records to {out_path}")
            return 0

        # jsonl
        count = 0
        with out_path.open("w", encoding="utf-8") as handle:
            for row in _memory_entries_iter(conn, args.namespace, args.entry_type, args.status):
                if args.as_sovereign:
                    row = _to_sovereign_row(row, args.snr, args.iso_times)
                else:
                    row = _format_times(row, args.iso_times)
                handle.write(json.dumps(row))
                handle.write("\n")
                count += 1

        print(f"Exported {count} records to {out_path}")
        return 0
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Claude-Flow memory adapter")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to memory.db")

    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("stats", help="Show table counts and metadata")
    stats.add_argument("--json", action="store_true", help="Emit JSON output")

    query = sub.add_parser("query", help="Query memory entries")
    query.add_argument("text", help="Search text")
    query.add_argument("--namespace", help="Filter by namespace")
    query.add_argument("--type", dest="entry_type", help="Filter by entry type")
    query.add_argument("--status", default="active", help="Filter by status")
    query.add_argument("--limit", type=int, default=20, help="Max results")
    query.add_argument("--json", action="store_true", help="Emit JSON output")
    query.add_argument("--jsonl", action="store_true", help="Emit JSONL output")
    query.add_argument("--iso-times", action="store_true", help="Convert timestamps to ISO-8601")

    export = sub.add_parser("export", help="Export memory entries")
    export.add_argument("--out", help="Output path")
    export.add_argument("--format", choices=["jsonl", "json"], default="jsonl")
    export.add_argument("--namespace", help="Filter by namespace")
    export.add_argument("--type", dest="entry_type", help="Filter by entry type")
    export.add_argument("--status", default="active", help="Filter by status")
    export.add_argument("--as-sovereign", action="store_true", help="Map entries to Sovereign catalog style")
    export.add_argument("--snr", type=float, default=0.0, help="SNR score for Sovereign mapping")
    export.add_argument("--iso-times", action="store_true", help="Convert timestamps to ISO-8601")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "stats":
            return cmd_stats(args)
        if args.command == "query":
            return cmd_query(args)
        if args.command == "export":
            return cmd_export(args)
        parser.print_help()
        return 1
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except sqlite3.Error as exc:
        print(f"SQLite error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
