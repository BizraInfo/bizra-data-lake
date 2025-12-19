import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from zipfile import ZipFile


DEFAULT_MAX_TEXT_CHARS = 4000

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "as",
    "is",
    "are",
    "be",
    "this",
    "that",
    "it",
    "we",
    "you",
    "i",
    "our",
    "your",
    "from",
    "by",
    "at",
    "not",
    "if",
    "then",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "let",
    "just",
    "like",
    "also",
    "now",
    "here",
    "there",
    "get",
    "got",
    "set",
    "run",
    "running",
    "using",
    "use",
    "used",
    "into",
    "over",
    "under",
    "more",
    "most",
    "less",
    "very",
    "much",
    "yet",
    "current",
    "supported",
}


@dataclass(frozen=True)
class SourceFile:
    path: str
    size_bytes: int
    sha256: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def truncate_text(text: str, max_chars: int) -> Tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    return text[: max_chars - 12] + "\n...[TRUNCATED]", True


def detect_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_base(repo_root: Path) -> Path:
    indexed = os.environ.get("BIZRA_DATALAKE_INDEXED")
    if indexed:
        return Path(indexed) / "chat_history"

    data_lake = os.environ.get("BIZRA_DATA_LAKE_ROOT")
    if data_lake:
        return Path(data_lake) / "03_INDEXED" / "chat_history"

    if Path(r"C:\BIZRA-DATA-LAKE").exists():
        return Path(r"C:\BIZRA-DATA-LAKE") / "03_INDEXED" / "chat_history"

    return repo_root / "docs" / "evidence" / "chat_ingest"


def iter_zip_members(z: ZipFile) -> Iterator[str]:
    for info in z.infolist():
        if info.is_dir():
            continue
        yield info.filename


def read_zip_json(z: ZipFile, member: str) -> Optional[Any]:
    try:
        with z.open(member) as f:
            raw = f.read()
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None


def is_conversation_mapping_export(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    return "mapping" in obj and "title" in obj and isinstance(obj.get("mapping"), dict)


def extract_mapping_messages(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    mapping = obj.get("mapping") or {}
    items: List[Dict[str, Any]] = []
    for node in mapping.values():
        msg = node.get("message") if isinstance(node, dict) else None
        if not isinstance(msg, dict):
            continue
        author = msg.get("author") or {}
        role = safe_text(author.get("role")).strip() or "unknown"
        content = msg.get("content") or {}
        if not isinstance(content, dict):
            continue
        if content.get("content_type") != "text":
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        text = "\n".join(safe_text(p) for p in parts).strip()
        if not text:
            continue
        created = msg.get("create_time")
        items.append(
            {
                "message_id": safe_text(msg.get("id") or node.get("id")),
                "role": role,
                "created": created,
                "text": text,
            }
        )
    items.sort(key=lambda r: (r["created"] is None, r["created"] or 0))
    return items


def is_batch_conversations_export(obj: Any) -> bool:
    if not isinstance(obj, list):
        return False
    if not obj:
        return True
    head = obj[0]
    return isinstance(head, dict) and "uuid" in head and ("chat_messages" in head or "created_at" in head)


def iter_conversations_from_batch(obj: List[Dict[str, Any]]) -> Iterator[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    for c in obj:
        if not isinstance(c, dict):
            continue
        conv = {
            "conversation_id": safe_text(c.get("uuid")),
            "title": safe_text(c.get("name") or c.get("title")),
            "created_at": safe_text(c.get("created_at")),
            "updated_at": safe_text(c.get("updated_at")),
            "summary": safe_text(c.get("summary")),
        }
        msgs: List[Dict[str, Any]] = []
        chat_messages = c.get("chat_messages")
        if isinstance(chat_messages, list):
            for m in chat_messages:
                if not isinstance(m, dict):
                    continue
                role = safe_text(m.get("sender") or m.get("role") or m.get("author") or "unknown").strip()
                text = safe_text(m.get("text") or m.get("content") or "").strip()
                created_at = safe_text(m.get("created_at") or m.get("timestamp"))
                if not text:
                    continue
                msgs.append(
                    {
                        "message_id": safe_text(m.get("uuid") or m.get("id") or ""),
                        "role": role,
                        "created_at": created_at,
                        "text": text,
                    }
                )
        yield conv, msgs


PATH_START_RE = re.compile(r"[A-Za-z]:\\\\")
CMD_RE = re.compile(
    r"\b(docker|git|cargo|npm|ollama|powershell|cmd|robocopy|manage-bde|vssadmin)\b", re.I
)

KNOWN_PATH_EXTS = [
    ".exe",
    ".ps1",
    ".bat",
    ".cmd",
    ".py",
    ".js",
    ".ts",
    ".rs",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".log",
    ".zip",
    ".png",
    ".pdf",
]


def extract_signals(text: str) -> Dict[str, List[str]]:
    norm_paths: List[str] = []
    for m in PATH_START_RE.finditer(text):
        start = m.start()
        window = text[start : start + 300]

        end_limit = len(window)
        i = 0
        stop_chars = set(['\r', '\n', '"', "'", "`", "|"])

        while i < end_limit:
            ch = window[i]

            if ch in stop_chars:
                break

            # Stop on separators commonly used after paths in logs/commands.
            if ch in [';', ',', ')', ']', '}']:
                break

            # Stop on ":" if it's not the drive letter colon.
            if ch == ":" and i > 1:
                break

            if ch in [" ", "\t"]:
                # Keep spaces only when the next chunk looks like it continues a path (has a backslash)
                ahead = window[i + 1 :]
                next_stop = None
                for j, c2 in enumerate(ahead):
                    if c2 in stop_chars or c2 in [';', ',', ')', ']', '}', ':']:
                        next_stop = j
                        break
                segment = ahead[:next_stop] if next_stop is not None else ahead
                if "\\" not in segment:
                    break

            i += 1

        candidate = window[:i].strip().rstrip(").,;:`")
        if len(candidate) >= 4 and "\\" in candidate[3:]:
            # If this is an executable path, trim to .exe when present within the token.
            lower = candidate.lower()
            exe_pos = lower.find(".exe")
            if exe_pos != -1:
                candidate = candidate[: exe_pos + 4]
            norm_paths.append(candidate)

    paths = sorted(set(norm_paths))[:25]
    cmds = sorted(set(m.lower() for m in CMD_RE.findall(text)))[:25]
    return {"paths": paths, "commands": cmds}


def top_terms(text: str, k: int = 10) -> List[Tuple[str, int]]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\\-]{2,}", text.lower())
    freq: Dict[str, int] = {}
    for t in tokens:
        if t in STOPWORDS:
            continue
        if t.isdigit():
            continue
        freq[t] = freq.get(t, 0) + 1
    return sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_jsonl_gz(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: Sequence[str]) -> int:
    repo_root = detect_repo_root()

    p = argparse.ArgumentParser(description="Ingest chat export zips into a high-SNR indexed format.")
    p.add_argument("--input-dir", default=str(repo_root / "chat data sample"), help="Directory containing .zip exports")
    p.add_argument("--out-dir", default="", help="Output directory (defaults to Data Lake 03_INDEXED when available)")
    p.add_argument("--max-text-chars", type=int, default=DEFAULT_MAX_TEXT_CHARS, help="Max chars stored per message in index")
    p.add_argument("--store-full-text", action="store_true", help="Also store full messages to messages_full.jsonl.gz")
    args = p.parse_args(list(argv))

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        print(f"ERROR: input dir not found: {input_dir}", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (default_output_base(repo_root) / ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources: List[SourceFile] = []
    conversations_rows: List[Dict[str, Any]] = []
    messages_index_rows: List[Dict[str, Any]] = []
    messages_full_rows: List[Dict[str, Any]] = []
    conv_topics: Dict[str, List[Tuple[str, int]]] = {}
    conv_signals: Dict[str, Dict[str, List[str]]] = {}

    zip_paths = sorted(input_dir.glob("*.zip"))
    if not zip_paths:
        print(f"ERROR: no .zip files found in: {input_dir}", file=sys.stderr)
        return 2

    for zp in zip_paths:
        sources.append(SourceFile(path=str(zp), size_bytes=zp.stat().st_size, sha256=sha256_file(zp)))

        with ZipFile(zp) as z:
            members = list(iter_zip_members(z))

            # Batch format: conversations.json at root
            if "conversations.json" in members:
                convs = read_zip_json(z, "conversations.json")
                if isinstance(convs, list) and is_batch_conversations_export(convs):
                    for conv, msgs in iter_conversations_from_batch(convs):
                        conv_id = conv["conversation_id"] or f"{zp.name}::unknown"
                        conv["source_zip"] = zp.name
                        conv["source_member"] = "conversations.json"
                        conv["truth_label"] = "MEASURED"
                        conversations_rows.append(conv)

                        topic_text = (conv.get("title") or "") + "\n" + (conv.get("summary") or "")
                        for m in msgs[:10]:
                            topic_text += "\n" + safe_text(m.get("text"))
                        conv_topics[conv_id] = top_terms(topic_text, k=12)
                        conv_signals[conv_id] = extract_signals(topic_text)

                        for m in msgs:
                            text = safe_text(m.get("text"))
                            text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
                            truncated, was_truncated = truncate_text(text, args.max_text_chars)
                            row = {
                                "conversation_id": conv_id,
                                "message_id": m.get("message_id") or "",
                                "role": m.get("role") or "unknown",
                                "created_at": m.get("created_at") or "",
                                "text": truncated,
                                "text_len": len(text),
                                "text_sha256": text_hash,
                                "truncated": was_truncated,
                                "source_zip": zp.name,
                                "source_member": "conversations.json",
                                "truth_label": "MEASURED",
                            }
                            messages_index_rows.append(row)
                            if args.store_full_text:
                                messages_full_rows.append({**row, "text": text})

            # ChatGPT mapping export format: many per-conversation json files
            for member in members:
                if not member.lower().endswith(".json"):
                    continue
                if member.lower().endswith("conversations.json") or member.lower().endswith("projects.json"):
                    continue
                if member.lower().endswith("memories.json") or member.lower().endswith("users.json"):
                    continue

                obj = read_zip_json(z, member)
                if not is_conversation_mapping_export(obj):
                    continue

                conv_id = f"{zp.name}::{member}"
                title = safe_text(obj.get("title"))
                conv_row = {
                    "conversation_id": conv_id,
                    "title": title,
                    "created_at": safe_text(obj.get("create_time")),
                    "updated_at": safe_text(obj.get("update_time")),
                    "source_zip": zp.name,
                    "source_member": member,
                    "truth_label": "MEASURED",
                }
                conversations_rows.append(conv_row)

                msgs = extract_mapping_messages(obj)
                topic_text = title
                for m in msgs[:10]:
                    topic_text += "\n" + safe_text(m.get("text"))
                conv_topics[conv_id] = top_terms(topic_text, k=12)
                conv_signals[conv_id] = extract_signals(topic_text)

                for m in msgs:
                    text = safe_text(m.get("text"))
                    text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
                    truncated, was_truncated = truncate_text(text, args.max_text_chars)
                    row = {
                        "conversation_id": conv_id,
                        "message_id": m.get("message_id") or "",
                        "role": m.get("role") or "unknown",
                        "created_at": m.get("created") or "",
                        "text": truncated,
                        "text_len": len(text),
                        "text_sha256": text_hash,
                        "truncated": was_truncated,
                        "source_zip": zp.name,
                        "source_member": member,
                        "truth_label": "MEASURED",
                    }
                    messages_index_rows.append(row)
                    if args.store_full_text:
                        messages_full_rows.append({**row, "text": text})

    # Write outputs
    sources_dicts = [s.__dict__ for s in sources]
    write_json(out_dir / "sources.json", {"generated_at": utc_now_iso(), "sources": sources_dicts})

    conv_count = write_jsonl(out_dir / "conversations.jsonl", conversations_rows)
    msg_count = write_jsonl(out_dir / "messages_index.jsonl", messages_index_rows)
    full_msg_count = 0
    if args.store_full_text:
        full_msg_count = write_jsonl_gz(out_dir / "messages_full.jsonl.gz", messages_full_rows)

    topics_rows: List[Dict[str, Any]] = []
    for conv_id, terms in conv_topics.items():
        topics_rows.append(
            {
                "conversation_id": conv_id,
                "terms": [{"term": t, "count": c} for (t, c) in terms],
                "signals": conv_signals.get(conv_id, {}),
                "truth_label": "DERIVED",
            }
        )
    write_json(out_dir / "topics.json", {"generated_at": utc_now_iso(), "topics": topics_rows})

    # Minimal graph (conversations -> terms, conversations -> paths/commands)
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    node_ids: set = set()

    def add_node(node_id: str, kind: str, label: str) -> None:
        if node_id in node_ids:
            return
        node_ids.add(node_id)
        nodes.append({"id": node_id, "kind": kind, "label": label})

    for conv in conversations_rows:
        conv_id = conv.get("conversation_id") or ""
        add_node(conv_id, "conversation", safe_text(conv.get("title") or conv_id))

        for term in conv_topics.get(conv_id, []):
            term_id = f"term::{term[0]}"
            add_node(term_id, "term", term[0])
            edges.append({"from": conv_id, "to": term_id, "kind": "has_term", "weight": term[1]})

        sig = conv_signals.get(conv_id, {})
        for pth in sig.get("paths", [])[:10]:
            pid = f"path::{pth}"
            add_node(pid, "path", pth)
            edges.append({"from": conv_id, "to": pid, "kind": "mentions_path"})
        for cmd in sig.get("commands", [])[:10]:
            cid = f"cmd::{cmd}"
            add_node(cid, "command", cmd)
            edges.append({"from": conv_id, "to": cid, "kind": "mentions_command"})

    write_json(out_dir / "graph.json", {"generated_at": utc_now_iso(), "nodes": nodes, "edges": edges})

    summary = {
        "run_id": ts,
        "generated_at": utc_now_iso(),
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "zip_files": len(zip_paths),
        "conversations": conv_count,
        "messages_index": msg_count,
        "messages_full": full_msg_count,
        "truth_label": "MEASURED",
    }
    write_json(out_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
