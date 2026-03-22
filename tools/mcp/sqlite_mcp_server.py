#!/usr/bin/env python3
"""
Local SQLite MCP server for BIZRA workspaces.

This replaces the broken npm-based sqlite MCP dependency with a small,
read-only FastMCP server that works against a configured SQLite database.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

LOG = logging.getLogger("sqlite-mcp")
DEFAULT_MAX_ROWS = 200


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] sqlite-mcp | %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only SQLite MCP server")
    parser.add_argument("--db-path", required=True, help="Path to the SQLite database")
    parser.add_argument(
        "--server-name",
        default="sqlite-mcp",
        help="MCP server display name",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Maximum rows returned by query tools",
    )
    return parser.parse_args()


def _resolve_db_path(raw_path: str) -> Path:
    db_path = Path(raw_path).expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database does not exist: {db_path}")
    if not db_path.is_file():
        raise ValueError(f"SQLite database path is not a file: {db_path}")
    return db_path


def _read_only_uri(db_path: Path) -> str:
    return f"file:{db_path.as_posix()}?mode=ro"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(_read_only_uri(db_path), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _normalize_sql(sql: str) -> str:
    normalized = sql.strip()
    if not normalized:
        raise ValueError("SQL query must not be empty.")
    if normalized.endswith(";"):
        normalized = normalized[:-1].strip()
    if ";" in normalized:
        raise ValueError("Only a single SQL statement is allowed.")
    return normalized


def _ensure_read_only_sql(sql: str) -> str:
    normalized = _normalize_sql(sql)
    lowered = normalized.lower()
    allowed_prefixes = (
        "select ",
        "with ",
        "explain query plan ",
    )
    if lowered not in {
        "select",
        "with",
        "explain query plan",
    } and not lowered.startswith(allowed_prefixes):
        raise ValueError(
            "Only read-only SELECT/CTE/EXPLAIN QUERY PLAN statements are allowed."
        )
    return normalized


def _fetch_rows(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple[Any, ...] = (),
    *,
    max_rows: int,
) -> dict[str, Any]:
    cursor = conn.execute(sql, params)
    rows = cursor.fetchmany(max_rows + 1)
    columns = [item[0] for item in cursor.description or []]
    truncated = len(rows) > max_rows
    visible_rows = rows[:max_rows]
    return {
        "columns": columns,
        "rows": [dict(row) for row in visible_rows],
        "returned_rows": len(visible_rows),
        "truncated": truncated,
    }


def build_server(db_path: Path, server_name: str, max_rows: int) -> FastMCP:
    mcp = FastMCP(server_name)

    @mcp.tool(
        description="Return connection metadata for the configured SQLite database."
    )
    def database_info() -> dict[str, Any]:
        with _connect(db_path) as conn:
            version = conn.execute("select sqlite_version()").fetchone()[0]
            tables = conn.execute("""
                select count(*)
                from sqlite_master
                where type in ('table', 'view') and name not like 'sqlite_%'
                """).fetchone()[0]
        return {
            "server": server_name,
            "database_path": str(db_path),
            "sqlite_version": version,
            "table_like_objects": tables,
            "read_only": True,
            "max_rows": max_rows,
        }

    @mcp.tool(description="List non-system tables and views in the SQLite database.")
    def list_tables() -> dict[str, Any]:
        with _connect(db_path) as conn:
            data = _fetch_rows(
                conn,
                """
                select
                    name,
                    type,
                    sql
                from sqlite_master
                where type in ('table', 'view')
                  and name not like 'sqlite_%'
                order by type, name
                """,
                max_rows=max_rows,
            )
        data["database_path"] = str(db_path)
        return data

    @mcp.tool(description="Describe a table or view, including columns and indexes.")
    def describe_table(table_name: str) -> dict[str, Any]:
        if not table_name.strip():
            raise ValueError("table_name must not be empty.")

        with _connect(db_path) as conn:
            object_row = conn.execute(
                """
                select name, type, sql
                from sqlite_master
                where name = ?
                  and type in ('table', 'view')
                """,
                (table_name,),
            ).fetchone()

            if object_row is None:
                raise ValueError(f"Table or view not found: {table_name}")

            columns = [
                dict(row)
                for row in conn.execute(
                    "select * from pragma_table_info(?) order by cid",
                    (table_name,),
                ).fetchall()
            ]
            indexes = []
            for index_row in conn.execute(
                "select * from pragma_index_list(?) order by name",
                (table_name,),
            ).fetchall():
                index_name = index_row["name"]
                index_columns = [
                    dict(col_row)
                    for col_row in conn.execute(
                        "select * from pragma_index_info(?) order by seqno",
                        (index_name,),
                    ).fetchall()
                ]
                entry = dict(index_row)
                entry["columns"] = index_columns
                indexes.append(entry)

        return {
            "database_path": str(db_path),
            "object": dict(object_row),
            "columns": columns,
            "indexes": indexes,
        }

    @mcp.tool(
        description="Return sample rows from a table or view using a simple row limit."
    )
    def sample_table(table_name: str, limit: int = 20) -> dict[str, Any]:
        if not table_name.strip():
            raise ValueError("table_name must not be empty.")
        safe_limit = max(1, min(limit, max_rows))
        with _connect(db_path) as conn:
            exists = conn.execute(
                """
                select 1
                from sqlite_master
                where name = ?
                  and type in ('table', 'view')
                """,
                (table_name,),
            ).fetchone()
            if exists is None:
                raise ValueError(f"Table or view not found: {table_name}")

            quoted_name = '"' + table_name.replace('"', '""') + '"'
            data = _fetch_rows(
                conn,
                f"select * from {quoted_name} limit {safe_limit}",
                max_rows=safe_limit,
            )
        data.update(
            {
                "database_path": str(db_path),
                "table_name": table_name,
                "limit": safe_limit,
            }
        )
        return data

    @mcp.tool(
        description=(
            "Run a read-only SQL query against the configured SQLite database. "
            "Only SELECT, WITH, and EXPLAIN QUERY PLAN statements are allowed."
        )
    )
    def read_query(sql: str) -> dict[str, Any]:
        safe_sql = _ensure_read_only_sql(sql)
        with _connect(db_path) as conn:
            data = _fetch_rows(conn, safe_sql, max_rows=max_rows)
        data.update(
            {
                "database_path": str(db_path),
                "sql": safe_sql,
            }
        )
        return data

    return mcp


def main() -> int:
    _configure_logging()
    args = _parse_args()
    db_path = _resolve_db_path(args.db_path)
    LOG.info(
        "Starting SQLite MCP server '%s' for %s",
        args.server_name,
        db_path,
    )
    server = build_server(
        db_path=db_path,
        server_name=args.server_name,
        max_rows=max(1, args.max_rows),
    )
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
