"""Pattern Registry — execution flywheel kernel v0.1.

Loads patterns.yaml, validates entries, and exposes query helpers. Stdlib-only.

Includes a deliberately minimal YAML subset parser that handles exactly the
block-style grammar used by patterns.yaml:

  key: scalar
  key:
    - scalar
    - scalar
  key:
    - nested_key: scalar
      other_key: scalar

Values containing '#' or ':' must be quoted. This is NOT a general YAML parser.
"""

from __future__ import annotations

from pathlib import Path

from .schemas import Pattern


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _unquote(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v


def _strip_comment(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith("#"):
        return ""
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '"':
            j = line.find('"', i + 1)
            if j == -1:
                break
            i = j + 1
            continue
        if ch == "'":
            j = line.find("'", i + 1)
            if j == -1:
                break
            i = j + 1
            continue
        if ch == "#" and i > 0 and line[i - 1].isspace():
            return line[:i].rstrip()
        i += 1
    return line.rstrip()


def parse_minimal_yaml(text: str):
    raw = text.splitlines()
    lines: list[str] = []
    for ln in raw:
        s = _strip_comment(ln)
        if s.strip() == "":
            continue
        lines.append(s)
    if not lines:
        return {}
    value, _ = _parse_value(lines, 0, -1)
    return value if value is not None else {}


def _parse_value(lines: list[str], start: int, parent_indent: int):
    if start >= len(lines):
        return None, start
    first = lines[start]
    first_ind = _indent(first)
    if first_ind <= parent_indent:
        return None, start
    if first.lstrip().startswith("- "):
        return _parse_list(lines, start, first_ind)
    return _parse_mapping(lines, start, first_ind)


def _parse_mapping(lines: list[str], start: int, indent: int):
    result: dict = {}
    i = start
    while i < len(lines):
        ln = lines[i]
        ind = _indent(ln)
        if ind < indent:
            break
        if ind > indent:
            raise ValueError(f"Unexpected indent at line {i + 1}: {ln!r}")
        stripped = ln.strip()
        if stripped.startswith("- "):
            break
        if ":" not in stripped:
            raise ValueError(f"Expected 'key: value' at line {i + 1}: {ln!r}")
        key, _, rest = stripped.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest == "":
            i += 1
            child, i = _parse_value(lines, i, indent)
            result[key] = child
        else:
            result[key] = _unquote(rest)
            i += 1
    return result, i


def _parse_list(lines: list[str], start: int, indent: int):
    result: list = []
    i = start
    while i < len(lines):
        ln = lines[i]
        ind = _indent(ln)
        if ind < indent:
            break
        if ind > indent:
            raise ValueError(f"Unexpected indent at line {i + 1}: {ln!r}")
        stripped = ln.strip()
        if not stripped.startswith("- "):
            break
        content = stripped[2:].strip()
        if content and content[0] in ('"', "'"):
            result.append(_unquote(content))
            i += 1
            continue
        if content == "":
            i += 1
            child, i = _parse_value(lines, i, indent)
            result.append(child)
            continue
        if ":" in content:
            k, _, v = content.partition(":")
            v = v.strip()
            item: dict = {k.strip(): _unquote(v) if v else None}
            i += 1
            while i < len(lines):
                nxt = lines[i]
                nxt_ind = _indent(nxt)
                nxt_stripped = nxt.strip()
                if nxt_ind <= indent:
                    break
                if nxt_stripped.startswith("- "):
                    break
                if ":" not in nxt_stripped:
                    break
                nk, _, nv = nxt_stripped.partition(":")
                nv = nv.strip()
                if nv == "":
                    i += 1
                    child, i = _parse_value(lines, i, nxt_ind)
                    item[nk.strip()] = child
                else:
                    item[nk.strip()] = _unquote(nv)
                    i += 1
            result.append(item)
        else:
            result.append(_unquote(content))
            i += 1
    return result, i


def load_patterns(path: str | Path) -> list[Pattern]:
    text = Path(path).read_text(encoding="utf-8")
    parsed = parse_minimal_yaml(text)
    if not isinstance(parsed, dict):
        raise ValueError("Top-level patterns file must be a mapping")
    raw_patterns = parsed.get("patterns", []) or []
    if not isinstance(raw_patterns, list):
        raise ValueError("'patterns' must be a list")
    return [Pattern.from_dict(p) for p in raw_patterns if isinstance(p, dict)]


def list_patterns(patterns: list[Pattern]) -> list[str]:
    return [p.pattern_id for p in patterns]


def get_pattern(patterns: list[Pattern], pattern_id: str) -> Pattern | None:
    for p in patterns:
        if p.pattern_id == pattern_id:
            return p
    return None


def query_by_trigger(patterns: list[Pattern], keyword: str) -> list[Pattern]:
    kw = keyword.lower()
    return [p for p in patterns if any(t.keyword.lower() == kw for t in p.triggers)]


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Inspect the execution-flywheel pattern registry")
    parser.add_argument("--patterns", default=str(Path(__file__).parent / "patterns.yaml"))
    parser.add_argument("--query", help="Filter by trigger keyword")
    parser.add_argument("--id", help="Return a single pattern by id")
    args = parser.parse_args()
    patterns = load_patterns(args.patterns)
    if args.id:
        p = get_pattern(patterns, args.id)
        if p is None:
            raise SystemExit(f"pattern {args.id!r} not found")
        print(json.dumps(p.to_dict(), indent=2))
        return
    if args.query:
        patterns = query_by_trigger(patterns, args.query)
    print(json.dumps([p.to_dict() for p in patterns], indent=2))


if __name__ == "__main__":
    main()
