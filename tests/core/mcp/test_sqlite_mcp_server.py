from pathlib import Path
import sqlite3

import pytest

from tools.mcp.sqlite_mcp_server import build_server, _ensure_read_only_sql


def test_read_only_sql_guard_allows_select_shapes() -> None:
    assert _ensure_read_only_sql("select 1") == "select 1"
    assert _ensure_read_only_sql("WITH x AS (SELECT 1) SELECT * FROM x;") == (
        "WITH x AS (SELECT 1) SELECT * FROM x"
    )
    assert _ensure_read_only_sql("EXPLAIN QUERY PLAN SELECT 1") == (
        "EXPLAIN QUERY PLAN SELECT 1"
    )


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "delete from items",
        "pragma journal_mode=WAL",
        "select 1; drop table users",
    ],
)
def test_read_only_sql_guard_blocks_mutating_or_multiple_statements(sql: str) -> None:
    with pytest.raises(ValueError):
        _ensure_read_only_sql(sql)


def test_database_tools_return_expected_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("create table users (id integer primary key, name text not null)")
        conn.execute("insert into users (name) values ('Amina'), ('Bilal')")
        conn.commit()
    finally:
        conn.close()

    mcp = build_server(db_path=db_path, server_name="test-sqlite", max_rows=10)
    tools = {tool.name: tool for tool in mcp._tool_manager.list_tools()}  # noqa: SLF001

    info = tools["database_info"].fn()
    assert info["read_only"] is True
    assert info["table_like_objects"] == 1

    tables = tools["list_tables"].fn()
    assert tables["rows"][0]["name"] == "users"

    schema = tools["describe_table"].fn("users")
    assert schema["object"]["name"] == "users"
    assert [col["name"] for col in schema["columns"]] == ["id", "name"]

    sample = tools["sample_table"].fn("users", limit=1)
    assert sample["returned_rows"] == 1
    assert sample["rows"][0]["name"] == "Amina"

    query = tools["read_query"].fn("select name from users order by id")
    assert [row["name"] for row in query["rows"]] == ["Amina", "Bilal"]
