import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zipfile import ZipFile


DEFAULT_INCLUDE_EXTS = {
    ".zip",
    ".dmp",
    ".mdmp",
    ".wer",
    ".log",
    ".txt",
    ".json",
    ".xml",
    ".evtx",
}

DEFAULT_EXTRACT_EXTS = {
    ".txt",
    ".log",
    ".wer",
    ".xml",
    ".json",
    ".md",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_label(root: Path) -> str:
    name = root.name.strip().lower()
    if not name:
        name = "root"
    return "".join(ch for ch in name if ch.isalnum() or ch in ("_", "-"))[:48] or "root"


def iter_files(root: Path) -> Iterable[Path]:
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            for child in cur.iterdir():
                if child.is_dir():
                    stack.append(child)
                elif child.is_file():
                    yield child
        except Exception:
            continue


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def sanitize_zip_member(name: str) -> Path:
    # Keep relative structure, but prevent path traversal.
    parts = []
    for p in name.replace("\\", "/").split("/"):
        p = p.strip()
        if not p or p in (".", ".."):
            continue
        parts.append(p)
    return Path(*parts)


def ingest_root(
    root: Path,
    out_dir: Path,
    include_exts: Sequence[str],
    copy_max_bytes: int,
    extract_max_bytes: int,
    extract_exts: Sequence[str],
) -> Dict[str, Any]:
    label = safe_label(root)
    files_dir = out_dir / "files" / label
    extracted_dir = out_dir / "extracted" / label

    file_rows: List[Dict[str, Any]] = []
    extracted_rows: List[Dict[str, Any]] = []

    for f in iter_files(root):
        ext = f.suffix.lower()
        if ext not in include_exts:
            continue
        try:
            st = f.stat()
            size = int(st.st_size)
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        except Exception:
            size = 0
            mtime = None

        rel = None
        try:
            rel = str(f.relative_to(root)).replace("/", "\\")
        except Exception:
            rel = f.name

        row: Dict[str, Any] = {
            "root": str(root),
            "relative_path": rel,
            "path": str(f),
            "size_bytes": size,
            "mtime_utc": mtime,
            "truth_label": "MEASURED",
            "copied": False,
            "copy_path": None,
            "sha256": None,
            "sha256_skipped_reason": None,
            "zip": None,
        }

        if size <= copy_max_bytes:
            dest = files_dir / rel
            copy_file(f, dest)
            row["copied"] = True
            row["copy_path"] = str(dest)
            row["sha256"] = sha256_file(dest)
        else:
            row["sha256_skipped_reason"] = f"size>{copy_max_bytes}"

        if ext == ".zip" and row.get("copied") and row.get("copy_path"):
            zpath = Path(row["copy_path"])
            try:
                zinfo_rows = []
                with ZipFile(zpath) as z:
                    for info in z.infolist():
                        zinfo_rows.append(
                            {
                                "name": info.filename,
                                "size_bytes": int(info.file_size),
                                "compressed_bytes": int(info.compress_size),
                            }
                        )

                        mext = Path(info.filename).suffix.lower()
                        if mext in extract_exts and info.file_size <= extract_max_bytes:
                            member_rel = sanitize_zip_member(info.filename)
                            if not member_rel:
                                continue
                            out_path = extracted_dir / zpath.stem / member_rel
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            with z.open(info) as src_f, out_path.open("wb") as dst_f:
                                shutil.copyfileobj(src_f, dst_f)
                            extracted_rows.append(
                                {
                                    "zip": str(zpath),
                                    "member": info.filename,
                                    "path": str(out_path),
                                    "size_bytes": int(info.file_size),
                                    "sha256": sha256_file(out_path),
                                    "truth_label": "MEASURED",
                                }
                            )

                row["zip"] = {"members": zinfo_rows, "truth_label": "MEASURED"}
            except Exception as e:
                row["zip"] = {"error": f"{e}", "truth_label": "MEASURED"}

        file_rows.append(row)

    return {
        "root": str(root),
        "label": label,
        "files": file_rows,
        "extracted": extracted_rows,
        "truth_label": "MEASURED",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Safely ingest Windows crash artifacts (index + copy small files).")
    ap.add_argument("--roots", nargs="*", default=[], help="Roots to scan (defaults from env or C:\\temp C:\\tmp)")
    ap.add_argument("--out-dir", default="", help="Output directory (defaults to Data Lake indexed/windows_crash/<run_id>)")
    ap.add_argument("--copy-max-mb", type=int, default=50, help="Copy files up to this size (MB)")
    ap.add_argument("--extract-max-mb", type=int, default=5, help="Extract text members up to this size (MB)")
    args = ap.parse_args()

    roots = [Path(r).expanduser() for r in args.roots if r]
    if not roots:
        env_roots = []
        for k in ("BIZRA_VAULT_CRASH_TEMP", "BIZRA_VAULT_CRASH_TMP"):
            v = os.environ.get(k)
            if v:
                env_roots.append(v)
        roots = [Path(r) for r in env_roots] if env_roots else [Path("C:\\temp"), Path("C:\\tmp")]

    roots = [r.resolve() for r in roots if r and r.exists()]
    if not roots:
        raise SystemExit("No valid roots found to ingest.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    out_dir: Path
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        indexed = os.environ.get("BIZRA_DATALAKE_INDEXED") or os.environ.get("BIZRA_DATA_LAKE_ROOT")
        if indexed:
            base = Path(indexed)
            if base.name.upper() == "BIZRA-DATA-LAKE":
                base = base / "03_INDEXED"
            out_dir = base / "windows_crash" / run_id
        else:
            out_dir = Path.cwd() / "windows_crash" / run_id

    out_dir.mkdir(parents=True, exist_ok=True)

    include_exts = list(DEFAULT_INCLUDE_EXTS)
    extract_exts = list(DEFAULT_EXTRACT_EXTS)
    copy_max_bytes = int(max(args.copy_max_mb, 1) * 1024 * 1024)
    extract_max_bytes = int(max(args.extract_max_mb, 1) * 1024 * 1024)

    roots_rows = []
    for r in roots:
        roots_rows.append(ingest_root(r, out_dir, include_exts, copy_max_bytes, extract_max_bytes, extract_exts))

    summary = {
        "run_type": "windows_crash_ingest",
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "truth_label": "MEASURED",
        "roots": roots_rows,
    }

    summary_path = out_dir / "summary.json"
    write_json(summary_path, summary)

    # Compact index (for LLM / quick inspection)
    index_rows = []
    extracted_count = 0
    for rr in roots_rows:
        for f in rr.get("files") or []:
            index_rows.append(
                {
                    "path": f.get("path"),
                    "size_bytes": f.get("size_bytes"),
                    "mtime_utc": f.get("mtime_utc"),
                    "sha256": f.get("sha256"),
                    "copy_path": f.get("copy_path"),
                    "truth_label": "MEASURED",
                }
            )
        extracted_count += len(rr.get("extracted") or [])

    index = {
        "generated_at": utc_now_iso(),
        "truth_label": "DERIVED",
        "summary": {
            "roots": len(roots_rows),
            "files_indexed": len(index_rows),
            "members_extracted": extracted_count,
        },
        "files": index_rows,
    }
    write_json(out_dir / "index.json", index)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "out_dir": str(out_dir),
                "summary_json": str(summary_path),
                "files_indexed": len(index_rows),
                "members_extracted": extracted_count,
                "truth_label": "MEASURED",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

