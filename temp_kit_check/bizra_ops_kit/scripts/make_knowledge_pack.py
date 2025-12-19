#!/usr/bin/env python3

import argparse, json, os, pathlib, collections, datetime, zipfile

def read_manifest(path: pathlib.Path):
    recs = []
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs

def summarize(recs):
    total_files = len(recs)
    total_bytes = sum(int(r.get("size_bytes", 0)) for r in recs)
    exts = collections.Counter(r.get("ext","") or "" for r in recs)
    top_exts = exts.most_common(30)

    # size buckets
    buckets = {
        "<1KB": 0, "1KB-1MB": 0, "1MB-100MB": 0, "100MB-1GB": 0, ">=1GB": 0
    }
    for r in recs:
        s = int(r.get("size_bytes", 0))
        if s < 1024: buckets["<1KB"] += 1
        elif s < 1024*1024: buckets["1KB-1MB"] += 1
        elif s < 100*1024*1024: buckets["1MB-100MB"] += 1
        elif s < 1024*1024*1024: buckets["100MB-1GB"] += 1
        else: buckets[">=1GB"] += 1

    # newest/oldest
    mtimes = []
    for r in recs:
        m = r.get("mtime_utc")
        if m:
            mtimes.append(m)
    oldest = min(mtimes) if mtimes else None
    newest = max(mtimes) if mtimes else None

    return {
        "generated_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total_files": total_files,
        "total_bytes": total_bytes,
        "top_extensions": top_exts,
        "size_buckets": buckets,
        "oldest_mtime_utc": oldest,
        "newest_mtime_utc": newest,
    }

def write_summary(out_dir: pathlib.Path, summary: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "catalog_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

def write_index_md(out_dir: pathlib.Path, summary: dict):
    md = []
    md.append("# Catalog Summary\n")
    md.append(f"- Generated (UTC): `{summary['generated_utc']}`")
    md.append(f"- Total files: **{summary['total_files']}**")
    md.append(f"- Total bytes: **{summary['total_bytes']}**")
    md.append("\n## Size buckets\n")
    for k,v in summary["size_buckets"].items():
        md.append(f"- {k}: {v}")
    md.append("\n## Top extensions\n")
    for ext, cnt in summary["top_extensions"]:
        label = ext if ext else "(no extension)"
        md.append(f"- {label}: {cnt}")
    (out_dir / "CATALOG_INDEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")

def maybe_zip(out_dir: pathlib.Path, zip_name: str):
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file() and p.name != zip_name:
                z.write(p, p.relative_to(out_dir))
    return zip_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root path you scanned (for provenance only).")
    ap.add_argument("--catalog", required=True, help="Folder containing manifest.jsonl.")
    ap.add_argument("--out", required=True, help="Output folder (default: same as catalog).")
    ap.add_argument("--zip", action="store_true", help="Create knowledge_pack.zip from outputs.")
    args = ap.parse_args()

    catalog = pathlib.Path(args.catalog)
    out_dir = pathlib.Path(args.out)
    manifest = catalog / "manifest.jsonl"

    recs = read_manifest(manifest)
    summary = summarize(recs)
    summary["scanned_root"] = args.root

    write_summary(out_dir, summary)
    write_index_md(out_dir, summary)

    if args.zip:
        zp = maybe_zip(out_dir, "knowledge_pack.zip")
        print(f"Created: {zp}")
    else:
        print(f"Wrote summary to: {out_dir}")

if __name__ == "__main__":
    main()
