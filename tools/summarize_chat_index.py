import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def iter_topic_rows(topics_obj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    topics = topics_obj.get("topics")
    if isinstance(topics, list):
        for row in topics:
            if isinstance(row, dict):
                yield row


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate a chat index (topics.json) into high-SNR summaries.")
    p.add_argument("--run-dir", required=True, help="Path to chat_history/<run_id> directory")
    p.add_argument("--top-n", type=int, default=100, help="How many top items to keep per aggregate")
    args = p.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    topics_path = run_dir / "topics.json"
    if not topics_path.exists():
        raise SystemExit(f"Missing: {topics_path}")

    topics_obj = read_json(topics_path)

    term_total = Counter()
    term_conv = Counter()
    path_total = Counter()
    path_conv = Counter()
    cmd_total = Counter()
    cmd_conv = Counter()

    conv_scores: List[Tuple[str, int]] = []

    for row in iter_topic_rows(topics_obj):
        conv_id = str(row.get("conversation_id") or "")
        terms = row.get("terms") or []
        signals = row.get("signals") or {}
        paths = signals.get("paths") or []
        cmds = signals.get("commands") or []

        conv_score = 0

        if isinstance(terms, list):
            for t in terms:
                if not isinstance(t, dict):
                    continue
                term = str(t.get("term") or "").strip()
                count = int(t.get("count") or 0)
                if not term:
                    continue
                term_total[term] += count
                term_conv[term] += 1
                conv_score += 1

        if isinstance(paths, list):
            for pth in paths:
                pth = str(pth).strip()
                if not pth:
                    continue
                path_total[pth] += 1
                path_conv[pth] += 1
                conv_score += 1

        if isinstance(cmds, list):
            for cmd in cmds:
                cmd = str(cmd).strip().lower()
                if not cmd:
                    continue
                cmd_total[cmd] += 1
                cmd_conv[cmd] += 1
                conv_score += 1

        conv_scores.append((conv_id, conv_score))

    conv_scores.sort(key=lambda kv: (-kv[1], kv[0]))

    def counter_to_rows(total: Counter, conv: Counter, top_n: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for key, total_count in total.most_common(top_n):
            rows.append(
                {
                    "key": key,
                    "total_count": int(total_count),
                    "conversation_count": int(conv.get(key, 0)),
                    "truth_label": "DERIVED",
                }
            )
        return rows

    out = {
        "generated_at": utc_now_iso(),
        "inputs": [{"path": str(topics_path), "sha256": sha256_file(topics_path)}],
        "truth_label": "DERIVED",
        "top_terms": counter_to_rows(term_total, term_conv, args.top_n),
        "top_paths": counter_to_rows(path_total, path_conv, args.top_n),
        "top_commands": counter_to_rows(cmd_total, cmd_conv, args.top_n),
        "high_signal_conversations": [
            {"conversation_id": cid, "score": score, "truth_label": "DERIVED"}
            for cid, score in conv_scores[: min(args.top_n, len(conv_scores))]
        ],
    }

    write_json(run_dir / "aggregate.json", out)
    print(str(run_dir / "aggregate.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

