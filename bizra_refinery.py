import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Set, Tuple


GENESIS_FILE_DEFAULT = Path("BIZRA_GENESIS_BLOCK_0.json")
DEFAULT_SCAN_ROOT = Path("bizra_data_vault") / "roots" / "sovereign_data"
MANIFEST_FILE_DEFAULT = Path("BIZRA_KNOWLEDGE_MANIFEST.json")
LEDGER_FILE_DEFAULT = Path("BIZRA_KNOWLEDGE_LEDGER.jsonl")


SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    "target",
    "dist",
    "build",
    ".next",
    ".turbo",
    ".cache",
}


IMPACT_MULTIPLIERS: Dict[str, float] = {
    # Source Code (High Value - The Logic)
    ".rs": 50.0,
    ".py": 40.0,
    ".js": 35.0,
    ".ts": 35.0,
    ".go": 40.0,
    # Structured Knowledge (The Wisdom)
    ".pdf": 25.0,
    ".md": 15.0,
    ".json": 10.0,
    ".xml": 10.0,
    # Raw Data (The Ore)
    ".csv": 5.0,
    ".txt": 2.0,
    ".html": 2.0,
}


def _configure_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_hex_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def iter_walk(root: Path) -> Iterator[Path]:
    root_str = str(root)
    for cur_root, dirs, files in os.walk(root_str, topdown=True):
        # Deterministic walk order for audit stability
        dirs[:] = sorted([d for d in dirs if d not in SKIP_DIR_NAMES], key=str.casefold)
        for name in sorted(files, key=str.casefold):
            if name.startswith(".") or name.startswith("__"):
                continue
            yield Path(cur_root) / name


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return os.path.relpath(str(path), str(root))
    except Exception:
        return str(path)


def hash_file_contents(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass
class RefineryConfig:
    scan_root: Path
    genesis_file: Path
    out_manifest: Path
    out_ledger: Path
    hash_mode: str  # "metadata" | "content"
    ledger_format: str  # "inline" | "jsonl"
    include_exts: Optional[Set[str]]
    exclude_exts: Set[str]
    force: bool
    print_impact_min: float
    progress_every_sec: float


class IhsanRefinery:
    def __init__(self, cfg: RefineryConfig):
        self.cfg = cfg
        self.total_value = 0.0
        self.file_count = 0
        self.error_count = 0
        self.total_bytes = 0
        self.by_ext: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"files": 0, "bytes": 0, "value": 0.0})

        self.genesis_hash = self.load_genesis_hash()
        self.ledger_chain = hashlib.sha256()
        if self.genesis_hash:
            self.ledger_chain.update(self.genesis_hash.encode("utf-8"))
            self.ledger_chain.update(b"\0")

    def load_genesis_hash(self) -> Optional[str]:
        try:
            with open(self.cfg.genesis_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            genesis_hash = data.get("genesis_hash")
            if not isinstance(genesis_hash, str) or not genesis_hash.strip():
                raise ValueError("genesis_hash missing or invalid")
            return genesis_hash.strip()
        except FileNotFoundError:
            print("⚠️ CRITICAL: Genesis Block not found. Cannot mint assets without a root.")
            print(f"   Missing: {self.cfg.genesis_file}")
            return None
        except Exception as e:
            print("⚠️ CRITICAL: Failed to load Genesis Block. Cannot mint assets without a root.")
            print(f"   Error: {e}")
            return None

    def calculate_impact(self, path: Path, size_mb: float) -> float:
        ext = path.suffix.lower()
        multiplier = IMPACT_MULTIPLIERS.get(ext, 1.0)
        return round(size_mb * multiplier, 4)

    def should_process(self, path: Path) -> bool:
        ext = path.suffix.lower()
        if self.cfg.include_exts is not None and ext not in self.cfg.include_exts:
            return False
        if ext in self.cfg.exclude_exts:
            return False
        return True

    def fingerprint(self, path: Path, st: os.stat_result) -> Tuple[str, str]:
        rel_path = safe_relpath(path, self.cfg.scan_root)
        if self.cfg.hash_mode == "content":
            return hash_file_contents(path), "content_sha256"

        # Fast audit hash: relative path + size (stable across copies where mtimes change)
        ident = f"{rel_path}\0{st.st_size}"
        return sha256_hex_bytes(ident.encode("utf-8", errors="surrogatepass")), "metadata_sha256"

    def _ensure_outputs(self) -> bool:
        if self.cfg.force:
            return True

        outputs = [self.cfg.out_manifest]
        if self.cfg.ledger_format == "jsonl":
            outputs.append(self.cfg.out_ledger)
        existing = [p for p in outputs if p.exists()]
        if not existing:
            return True

        print("⚠️  OUTPUT EXISTS. REFUSING TO OVERWRITE.")
        for p in existing:
            print(f"   Present: {p}")
        print("   Re-run with `--force` or choose different output paths.")
        return False

    def process_path(self, *, path: Path, scan_root: Path) -> Optional[Dict[str, Any]]:
        try:
            if not self.should_process(path):
                return None

            st = path.stat()
            size_bytes = int(st.st_size)
            size_mb = size_bytes / (1024 * 1024)
            impact = self.calculate_impact(path, size_mb)
            file_hash, hash_kind = self.fingerprint(path, st)

            rel_path = safe_relpath(path, scan_root)
            record = {
                "filename": path.name,
                "path": rel_path,
                "hash": file_hash,
                "hash_kind": hash_kind,
                "size_mb": round(size_mb, 4),
                "impact_value": impact,
                "type": "SOVEREIGN_ASSET",
            }

            self.ledger_chain.update(file_hash.encode("utf-8"))
            self.ledger_chain.update(b"\0")

            self.total_value += impact
            self.file_count += 1
            self.total_bytes += size_bytes

            ext = path.suffix.lower()
            ext_entry = self.by_ext[ext]
            ext_entry["files"] += 1
            ext_entry["bytes"] += size_bytes
            ext_entry["value"] += impact

            if impact >= self.cfg.print_impact_min:
                label = path.name[:40].ljust(40)
                print(f"💎 Refined: {label} | Impact: {impact:,.2f} BZR")

            return record
        except Exception as e:
            self.error_count += 1
            try:
                print(f"⚠️ Error processing {path.name}: {e}")
            except Exception:
                print("⚠️ Error processing a file (unprintable filename).")
            return None

    def _write_manifest_jsonl(self, duration_sec: float) -> int:
        manifest = {
            "genesis_link": self.genesis_hash,
            "timestamp": utc_now_iso(),
            "architect": "Node0",
            "scan_root": str(self.cfg.scan_root),
            "hash_mode": self.cfg.hash_mode,
            "ledger_format": "jsonl",
            "filters": {
                "include_extensions": sorted(self.cfg.include_exts) if self.cfg.include_exts is not None else None,
                "exclude_extensions": sorted(self.cfg.exclude_exts),
            },
            "total_files": self.file_count,
            "total_bytes": self.total_bytes,
            "total_knowledge_value": round(self.total_value, 4),
            "processing_time_sec": round(duration_sec, 2),
            "errors": self.error_count,
            "asset_ledger_file": str(self.cfg.out_ledger),
            "asset_ledger_format": "jsonl",
            "asset_ledger_chain_sha256": self.ledger_chain.hexdigest(),
            "by_extension": {
                ext: {
                    "files": v["files"],
                    "total_mb": round(v["bytes"] / (1024 * 1024), 4),
                    "total_value": round(float(v["value"]), 4),
                }
                for ext, v in sorted(self.by_ext.items(), key=lambda kv: kv[0].casefold())
            },
        }

        mode = "x" if not self.cfg.force else "w"
        try:
            with open(self.cfg.out_manifest, mode, encoding="utf-8", newline="\n") as f:
                json.dump(manifest, f, indent=4, ensure_ascii=False, sort_keys=True)
                f.write("\n")
        except Exception as e:
            print("⚠️  FAILED TO WRITE MANIFEST.")
            print(f"   Path: {self.cfg.out_manifest}")
            print(f"   Error: {e}")
            return 2

        print("----------------------------------------------------------------")
        print("✅ REFINERY COMPLETE.")
        print(f"📚 Total Artifacts Ingested: {self.file_count:,}")
        print(f"💰 Total Intrinsic Value:    {self.total_value:,.2f} BZR-G")
        print(f"💾 Manifest Saved:           {self.cfg.out_manifest}")
        print(f"🧾 Ledger Saved:             {self.cfg.out_ledger}")
        print(f"🔐 Ledger Chain:             {self.ledger_chain.hexdigest()}")
        print("----------------------------------------------------------------")
        return 0

    def _scan_to_jsonl(self, *, scan_root: Path) -> int:
        try:
            ledger_f = open(self.cfg.out_ledger, "x" if not self.cfg.force else "w", encoding="utf-8", newline="\n")
        except Exception as e:
            print("⚠️  FAILED TO OPEN LEDGER OUTPUT.")
            print(f"   Path: {self.cfg.out_ledger}")
            print(f"   Error: {e}")
            return 2

        start = time.monotonic()
        last_progress = start

        with ledger_f:
            for path in iter_walk(scan_root):
                # Avoid self-ingestion if scanning broader paths
                try:
                    if path.resolve() in {self.cfg.out_manifest.resolve(), self.cfg.out_ledger.resolve(), self.cfg.genesis_file.resolve()}:
                        continue
                except Exception:
                    pass

                record = self.process_path(path=path, scan_root=scan_root)
                if record is not None:
                    ledger_f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

                now = time.monotonic()
                if self.cfg.progress_every_sec > 0 and (now - last_progress) >= self.cfg.progress_every_sec:
                    elapsed = now - start
                    print(
                        f"… Progress: {self.file_count:,} files | {self.total_value:,.2f} BZR-G | "
                        f"{(self.total_bytes / (1024 * 1024)):,.1f} MB | {self.error_count} errors | {elapsed:,.1f}s"
                    )
                    last_progress = now

        duration = time.monotonic() - start
        return self._write_manifest_jsonl(duration)

    def _scan_to_inline_manifest(self, *, scan_root: Path) -> int:
        mode = "x" if not self.cfg.force else "w"
        tmp = self.cfg.out_manifest.with_suffix(self.cfg.out_manifest.suffix + ".tmp")

        try:
            f = open(tmp, mode, encoding="utf-8", newline="\n")
        except Exception as e:
            print("⚠️  FAILED TO OPEN MANIFEST OUTPUT.")
            print(f"   Path: {tmp}")
            print(f"   Error: {e}")
            return 2

        start = time.monotonic()
        last_progress = start

        with f:
            f.write("{\n")
            f.write(f'    "genesis_link": {json.dumps(self.genesis_hash, ensure_ascii=False)},\n')
            f.write(f'    "architect": {json.dumps("Node0")},\n')
            f.write(f'    "scan_root": {json.dumps(str(self.cfg.scan_root), ensure_ascii=False)},\n')
            f.write(f'    "hash_mode": {json.dumps(self.cfg.hash_mode)},\n')
            f.write(
                f'    "filters": {json.dumps({"include_extensions": sorted(self.cfg.include_exts) if self.cfg.include_exts is not None else None, "exclude_extensions": sorted(self.cfg.exclude_exts)}, ensure_ascii=False, sort_keys=True)},\n'
            )
            f.write('    "asset_ledger": [\n')

            first = True
            for path in iter_walk(scan_root):
                try:
                    if path.resolve() in {tmp.resolve(), self.cfg.out_manifest.resolve(), self.cfg.genesis_file.resolve()}:
                        continue
                except Exception:
                    pass

                record = self.process_path(path=path, scan_root=scan_root)
                if record is None:
                    continue

                if not first:
                    f.write(",\n")
                f.write("        " + json.dumps(record, ensure_ascii=False, sort_keys=True))
                first = False

                now = time.monotonic()
                if self.cfg.progress_every_sec > 0 and (now - last_progress) >= self.cfg.progress_every_sec:
                    elapsed = now - start
                    print(
                        f"… Progress: {self.file_count:,} files | {self.total_value:,.2f} BZR-G | "
                        f"{(self.total_bytes / (1024 * 1024)):,.1f} MB | {self.error_count} errors | {elapsed:,.1f}s"
                    )
                    last_progress = now

            f.write("\n    ],\n")

            duration = time.monotonic() - start
            f.write(f'    "timestamp": {json.dumps(utc_now_iso())},\n')
            f.write(f'    "total_files": {self.file_count},\n')
            f.write(f'    "total_bytes": {self.total_bytes},\n')
            f.write(f'    "total_knowledge_value": {round(self.total_value, 4)},\n')
            f.write(f'    "processing_time_sec": {round(duration, 2)},\n')
            f.write(f'    "errors": {self.error_count},\n')
            f.write(f'    "asset_ledger_chain_sha256": {json.dumps(self.ledger_chain.hexdigest())},\n')
            f.write('    "by_extension": ')
            f.write(
                json.dumps(
                    {
                        ext: {
                            "files": v["files"],
                            "total_mb": round(v["bytes"] / (1024 * 1024), 4),
                            "total_value": round(float(v["value"]), 4),
                        }
                        for ext, v in sorted(self.by_ext.items(), key=lambda kv: kv[0].casefold())
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            f.write("\n}\n")

        try:
            os.replace(str(tmp), str(self.cfg.out_manifest))
        except Exception as e:
            print("⚠️  FAILED TO FINALIZE MANIFEST (RENAME).")
            print(f"   From: {tmp}")
            print(f"   To:   {self.cfg.out_manifest}")
            print(f"   Error: {e}")
            return 2

        print("----------------------------------------------------------------")
        print("✅ REFINERY COMPLETE.")
        print(f"📚 Total Artifacts Ingested: {self.file_count:,}")
        print(f"💰 Total Intrinsic Value:    {self.total_value:,.2f} BZR-G")
        print(f"💾 Manifest Saved:           {self.cfg.out_manifest}")
        print(f"🔐 Ledger Chain:             {self.ledger_chain.hexdigest()}")
        print("----------------------------------------------------------------")
        return 0

    def scan(self) -> int:
        if not self.genesis_hash:
            return 2

        scan_root = self.cfg.scan_root
        if not scan_root.exists():
            print("⚠️  SCAN ROOT NOT FOUND.")
            print(f"   Missing: {scan_root}")
            return 2

        if not self._ensure_outputs():
            return 1

        print(f"🏭 REFINERY ACTIVATED. Linked to Genesis: {self.genesis_hash[:16]}...")
        print(f"📂 Scanning: {scan_root}")
        print("----------------------------------------------------------------")

        if self.cfg.ledger_format == "jsonl":
            return self._scan_to_jsonl(scan_root=scan_root)
        return self._scan_to_inline_manifest(scan_root=scan_root)


def parse_args(argv: Optional[Iterable[str]] = None) -> RefineryConfig:
    p = argparse.ArgumentParser(description="BIZRA Refinery: ingest sovereign data and compute Proof-of-Impact value.")
    p.add_argument(
        "--path",
        dest="scan_root",
        default=str(DEFAULT_SCAN_ROOT),
        help=f"Scan root (default: {DEFAULT_SCAN_ROOT})",
    )
    p.add_argument(
        "--genesis",
        dest="genesis_file",
        default=str(GENESIS_FILE_DEFAULT),
        help=f"Genesis block JSON (default: {GENESIS_FILE_DEFAULT})",
    )
    p.add_argument(
        "--out-manifest",
        dest="out_manifest",
        default=str(MANIFEST_FILE_DEFAULT),
        help=f"Manifest output path (default: {MANIFEST_FILE_DEFAULT})",
    )
    p.add_argument(
        "--out-ledger",
        dest="out_ledger",
        default=str(LEDGER_FILE_DEFAULT),
        help=f"Ledger output path (default: {LEDGER_FILE_DEFAULT})",
    )
    p.add_argument(
        "--hash-mode",
        choices=["metadata", "content"],
        default="metadata",
        help="Fingerprint mode: fast metadata hash or full content SHA-256",
    )
    p.add_argument(
        "--ledger-format",
        choices=["inline", "jsonl"],
        default="jsonl",
        help="Ledger output: separate JSONL ledger file (default) or inline `asset_ledger` in manifest.",
    )
    p.add_argument(
        "--extensions",
        default="",
        help="Comma-separated allowlist of file extensions (e.g. 'md,pdf,rs,py'); empty means all.",
    )
    p.add_argument(
        "--exclude-extensions",
        default="",
        help="Comma-separated denylist of file extensions (e.g. 'mp4,iso'); empty means none.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    p.add_argument(
        "--print-impact-min",
        type=float,
        default=1.0,
        help="Only print per-file lines at/above this impact value (default: 1.0).",
    )
    p.add_argument(
        "--progress-every-sec",
        type=float,
        default=0.0,
        help="Print periodic progress every N seconds (default: 0 = off).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    include_exts: Optional[Set[str]] = None
    if str(args.extensions).strip():
        include_exts = set()
        for raw in str(args.extensions).split(","):
            item = raw.strip().lower()
            if not item:
                continue
            if not item.startswith("."):
                item = f".{item}"
            include_exts.add(item)

    exclude_exts: Set[str] = set()
    if str(args.exclude_extensions).strip():
        for raw in str(args.exclude_extensions).split(","):
            item = raw.strip().lower()
            if not item:
                continue
            if not item.startswith("."):
                item = f".{item}"
            exclude_exts.add(item)

    return RefineryConfig(
        scan_root=Path(args.scan_root),
        genesis_file=Path(args.genesis_file),
        out_manifest=Path(args.out_manifest),
        out_ledger=Path(args.out_ledger),
        hash_mode=str(args.hash_mode),
        ledger_format=str(args.ledger_format),
        include_exts=include_exts,
        exclude_exts=exclude_exts,
        force=bool(args.force),
        print_impact_min=float(args.print_impact_min),
        progress_every_sec=float(args.progress_every_sec),
    )


def main() -> int:
    _configure_stdout_utf8()
    cfg = parse_args()
    refinery = IhsanRefinery(cfg)
    return refinery.scan()


if __name__ == "__main__":
    raise SystemExit(main())
