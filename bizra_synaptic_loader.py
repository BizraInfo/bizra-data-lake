import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


LEDGER_DEFAULT = Path("BIZRA_KNOWLEDGE_LEDGER.jsonl")
MANIFEST_DEFAULT = Path("BIZRA_KNOWLEDGE_MANIFEST.json")
TOKENOMICS_DEFAULT = Path("BIZRA_TOKENOMICS_GENESIS.yaml")
RECEIPTS_DEFAULT_DIR = Path("docs") / "evidence" / "receipts"


def _configure_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def env_first(*names: str) -> Optional[str]:
    for name in names:
        v = os.environ.get(name)
        if v and v.strip():
            return v.strip()
    return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def try_git_head(repo_root: Path) -> Optional[str]:
    try:
        p = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode != 0:
            return None
        head = (p.stdout or "").strip()
        return head or None
    except Exception:
        return None


def iter_ledger(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def compute_ledger_chain(genesis_hash: str, ledger_path: Path) -> Tuple[str, int]:
    h = hashlib.sha256()
    h.update(genesis_hash.encode("utf-8"))
    h.update(b"\0")
    count = 0
    for rec in iter_ledger(ledger_path):
        fh = rec.get("hash")
        if isinstance(fh, str) and fh:
            h.update(fh.encode("utf-8"))
            h.update(b"\0")
            count += 1
    return h.hexdigest(), count


def classify_asset(ext: str) -> str:
    e = ext.lower()
    if e in {".rs", ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".c", ".cpp", ".h", ".cs"}:
        return "code"
    if e in {".pdf", ".md", ".txt"}:
        return "docs"
    if e in {".json", ".csv", ".xml", ".sql"}:
        return "data"
    if e in {".yaml", ".yml", ".toml", ".ini", ".env"}:
        return "config"
    if e in {".ipynb"}:
        return "notebook"
    if e in {".ps1", ".bat", ".sh"}:
        return "script"
    if e in {".html", ".css", ".scss"}:
        return "web"
    return "other"


@dataclass
class LoaderConfig:
    ledger_path: Path
    manifest_path: Path
    tokenomics_path: Optional[Path]
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_db: Optional[str]
    batch_size: int
    verify_chain: bool
    create_constraints: bool
    dry_run: bool
    emit_receipt: bool
    receipt_dir: Path
    receipt_out: Optional[Path]
    receipt_include_paths: bool


class SynapticLoader:
    def __init__(self, cfg: LoaderConfig):
        self.cfg = cfg
        self.nodes_written = 0
        self.bad_lines = 0
        self.start_time = time.monotonic()

        self.manifest = self._load_manifest()
        self.genesis_hash = self._require_manifest_str("genesis_link")
        self.ledger_chain_expected = self._require_manifest_str("asset_ledger_chain_sha256")
        self.ledger_chain_computed: Optional[str] = None
        self.ledger_line_count: Optional[int] = None

        self.scan_root = self.manifest.get("scan_root")
        self.hash_mode = self.manifest.get("hash_mode")
        self.total_files_claimed = self.manifest.get("total_files")
        self.total_value_claimed = self.manifest.get("total_knowledge_value")

        self._validate_tokenomics_optional()

        self.driver = None
        if not self.cfg.dry_run:
            try:
                from neo4j import GraphDatabase  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise SystemExit(
                    f"neo4j driver not available: {exc}\nInstall with: pip install neo4j"
                ) from exc
            self.driver = GraphDatabase.driver(
                self.cfg.neo4j_uri, auth=(self.cfg.neo4j_user, self.cfg.neo4j_password)
            )

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.cfg.manifest_path.exists():
            raise SystemExit(f"manifest not found: {self.cfg.manifest_path}")
        with open(self.cfg.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise SystemExit("manifest must be a JSON object")
        return data

    def _require_manifest_str(self, key: str) -> str:
        v = self.manifest.get(key)
        if not isinstance(v, str) or not v.strip():
            raise SystemExit(f"manifest missing non-empty string: {key}")
        return v.strip()

    def _validate_tokenomics_optional(self) -> None:
        if not self.cfg.tokenomics_path:
            return
        if not self.cfg.tokenomics_path.exists():
            return
        try:
            import yaml  # type: ignore
        except Exception:
            print("NOTE: PyYAML not installed; skipping tokenomics cross-check.")
            return

        try:
            data = yaml.safe_load(self.cfg.tokenomics_path.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(data, dict) or not isinstance(data.get("genesis"), dict):
                return
            g = data["genesis"]
            token_hash = g.get("hash")
            if isinstance(token_hash, str) and token_hash.strip() and token_hash.strip() != self.ledger_chain_expected:
                print("WARN: Tokenomics genesis hash does not match manifest ledger chain hash.")
                print(f"   tokenomics: {token_hash.strip()}")
                print(f"   manifest:   {self.ledger_chain_expected}")

            token_total = g.get("total_artifacts")
            if isinstance(token_total, int) and isinstance(self.total_files_claimed, int) and token_total != self.total_files_claimed:
                print("WARN: Tokenomics total_artifacts does not match manifest total_files.")
                print(f"   tokenomics: {token_total}")
                print(f"   manifest:   {self.total_files_claimed}")

            valuation = g.get("valuation")
            if isinstance(valuation, dict) and isinstance(self.total_value_claimed, (int, float)):
                supply = valuation.get("bzr_growth_supply")
                if isinstance(supply, (int, float)):
                    if abs(float(supply) - float(self.total_value_claimed)) > 0.02:
                        print("WARN: Tokenomics bzr_growth_supply differs from manifest total_knowledge_value.")
                        print(f"   tokenomics: {float(supply):.4f}")
                        print(f"   manifest:   {float(self.total_value_claimed):.4f}")
        except Exception as e:
            print(f"NOTE: tokenomics cross-check skipped ({e})")

    def verify_ledger_chain(self) -> int:
        computed, count = compute_ledger_chain(self.genesis_hash, self.cfg.ledger_path)
        self.ledger_chain_computed = computed
        self.ledger_line_count = count
        if computed != self.ledger_chain_expected:
            print("ERROR: Ledger chain verification failed.")
            print(f"   expected: {self.ledger_chain_expected}")
            print(f"   computed: {computed}")
            return 2
        if isinstance(self.total_files_claimed, int) and count != self.total_files_claimed:
            print("WARN: Ledger line count differs from manifest total_files.")
            print(f"   manifest: {self.total_files_claimed}")
            print(f"   ledger:   {count}")
        print("OK: Ledger chain verified.")
        print(f"   chain: {computed}")
        print(f"   files: {count:,}")
        return 0

    def _session(self):
        assert self.driver is not None
        if self.cfg.neo4j_db:
            return self.driver.session(database=self.cfg.neo4j_db)
        return self.driver.session()

    def _create_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT artifact_hash IF NOT EXISTS FOR (a:Artifact) REQUIRE a.hash IS UNIQUE",
            "CREATE CONSTRAINT manifest_chain IF NOT EXISTS FOR (m:KnowledgeManifest) REQUIRE m.ledger_chain_sha256 IS UNIQUE",
            "CREATE CONSTRAINT genesis_hash IF NOT EXISTS FOR (g:GenesisBlock) REQUIRE g.hash IS UNIQUE",
            "CREATE CONSTRAINT file_ext IF NOT EXISTS FOR (e:FileExtension) REQUIRE e.ext IS UNIQUE",
            "CREATE CONSTRAINT asset_class IF NOT EXISTS FOR (c:AssetClass) REQUIRE c.name IS UNIQUE",
        ]
        with self._session() as s:
            for stmt in statements:
                try:
                    s.run(stmt)
                except Exception:
                    pass

    def _upsert_manifest_nodes(self) -> None:
        q = """
        MERGE (g:GenesisBlock {hash: $genesis_hash})
        MERGE (m:KnowledgeManifest {ledger_chain_sha256: $ledger_chain})
        ON CREATE SET
          m.created_at = datetime(),
          m.scan_root = $scan_root,
          m.hash_mode = $hash_mode,
          m.total_files = $total_files,
          m.total_value_bzr_g = $total_value,
          m.source_manifest = $manifest_path
        MERGE (m)-[:ANCHORED_TO]->(g)
        RETURN m.ledger_chain_sha256 AS id
        """
        params = {
            "genesis_hash": self.genesis_hash,
            "ledger_chain": self.ledger_chain_expected,
            "scan_root": str(self.scan_root) if isinstance(self.scan_root, str) else None,
            "hash_mode": str(self.hash_mode) if isinstance(self.hash_mode, str) else None,
            "total_files": int(self.total_files_claimed) if isinstance(self.total_files_claimed, int) else None,
            "total_value": float(self.total_value_claimed) if isinstance(self.total_value_claimed, (int, float)) else None,
            "manifest_path": str(self.cfg.manifest_path),
        }
        with self._session() as s:
            s.run(q, **params)

    def _commit_batch(self, batch: List[Dict[str, Any]]) -> None:
        q = """
        MATCH (m:KnowledgeManifest {ledger_chain_sha256: $ledger_chain})
        UNWIND $batch AS item

        MERGE (a:Artifact {hash: item.hash})
        ON CREATE SET
          a.filename = item.filename,
          a.path = item.path,
          a.size_mb = item.size_mb,
          a.impact_value = item.impact_value,
          a.hash_kind = item.hash_kind,
          a.extension = item.extension,
          a.asset_class = item.asset_class,
          a.ingested_at = datetime()

        MERGE (a)-[:IN_MANIFEST]->(m)

        MERGE (e:FileExtension {ext: item.extension})
        MERGE (a)-[:HAS_EXTENSION]->(e)

        MERGE (c:AssetClass {name: item.asset_class})
        MERGE (a)-[:CLASSIFIED_AS]->(c)
        """

        with self._session() as s:
            s.run(q, ledger_chain=self.ledger_chain_expected, batch=batch)

        self.nodes_written += len(batch)
        elapsed = time.monotonic() - self.start_time
        print(f"... {self.nodes_written:,} artifacts loaded ({elapsed:,.1f}s)")

    def _post_ingest_counts(self) -> Dict[str, Any]:
        q = """
        MATCH (m:KnowledgeManifest {ledger_chain_sha256: $ledger_chain})
        OPTIONAL MATCH (a:Artifact)-[:IN_MANIFEST]->(m)
        OPTIONAL MATCH (m)-[:ANCHORED_TO]->(g:GenesisBlock {hash: $genesis_hash})
        RETURN count(a) AS artifacts_in_manifest,
               count(g) AS anchored_genesis_nodes,
               count(m) AS manifest_nodes
        """
        with self._session() as s:
            row = s.run(q, ledger_chain=self.ledger_chain_expected, genesis_hash=self.genesis_hash).single()
        if not row:
            return {}
        return {
            "artifacts_in_manifest": int(row.get("artifacts_in_manifest", 0)),
            "manifest_nodes": int(row.get("manifest_nodes", 0)),
            "anchored_genesis_nodes": int(row.get("anchored_genesis_nodes", 0)),
        }

    def _write_receipt(self, *, status: str, rc: int, constraints_attempted: bool) -> Optional[Path]:
        if not self.cfg.emit_receipt:
            return None

        repo_root = Path(__file__).resolve().parent
        head = try_git_head(repo_root)

        ledger_exists = self.cfg.ledger_path.exists()
        manifest_sha = sha256_file(self.cfg.manifest_path) if self.cfg.manifest_path.exists() else None
        ledger_sha = sha256_file(self.cfg.ledger_path) if ledger_exists else None
        token_sha = (
            sha256_file(self.cfg.tokenomics_path)
            if self.cfg.tokenomics_path and self.cfg.tokenomics_path.exists()
            else None
        )

        scan_root_raw = self.scan_root if isinstance(self.scan_root, str) else None
        scan_root_sha = sha256_text(scan_root_raw) if scan_root_raw else None

        elapsed = time.monotonic() - self.start_time

        receipt: Dict[str, Any] = {
            "schema": "bizra_knowledge_graph_ingest_receipt_v1",
            "generated_at": utc_now_iso(),
            "truth_label": "MEASURED",
            "status": status,
            "return_code": int(rc),
            "tool": {
                "name": "bizra_synaptic_loader.py",
                "git_head": head,
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "inputs": {
                "manifest_file": self.cfg.manifest_path.name,
                "manifest_sha256": manifest_sha,
                "ledger_file": self.cfg.ledger_path.name,
                "ledger_sha256": ledger_sha,
                "tokenomics_file": self.cfg.tokenomics_path.name if self.cfg.tokenomics_path else None,
                "tokenomics_sha256": token_sha,
            },
            "anchors": {
                "genesis_hash": self.genesis_hash,
                "ledger_chain_expected_sha256": self.ledger_chain_expected,
                "ledger_chain_computed_sha256": self.ledger_chain_computed,
                "ledger_line_count": self.ledger_line_count,
                "total_files_claimed": self.total_files_claimed,
                "total_value_claimed": self.total_value_claimed,
                "hash_mode": self.hash_mode,
                "scan_root_sha256": scan_root_sha,
            },
            "neo4j": {
                "uri": self.cfg.neo4j_uri,
                "database": self.cfg.neo4j_db,
                "dry_run": bool(self.cfg.dry_run),
                "constraints_enabled": bool(self.cfg.create_constraints),
                "constraints_attempted": bool(constraints_attempted),
            },
            "results": {
                "nodes_written": int(self.nodes_written),
                "bad_lines_skipped": int(self.bad_lines),
                "duration_sec": round(elapsed, 3),
            },
        }

        if self.cfg.receipt_include_paths:
            receipt["inputs"]["manifest_path"] = str(self.cfg.manifest_path)
            receipt["inputs"]["ledger_path"] = str(self.cfg.ledger_path)
            receipt["inputs"]["tokenomics_path"] = str(self.cfg.tokenomics_path) if self.cfg.tokenomics_path else None
            receipt["anchors"]["scan_root"] = scan_root_raw

        if self.driver is not None:
            try:
                receipt["results"]["graph_counts"] = self._post_ingest_counts()
            except Exception as e:
                receipt["results"]["graph_counts_error"] = str(e)

        out_path: Path
        if self.cfg.receipt_out is not None:
            out_path = self.cfg.receipt_out
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            receipt_dir = self.cfg.receipt_dir / f"knowledge_graph_ingest_{utc_now_compact()}"
            receipt_dir.mkdir(parents=True, exist_ok=True)
            out_path = receipt_dir / "receipt.json"

        out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return out_path

    def load(self) -> int:
        status = "started"
        rc = 0
        constraints_attempted = False

        try:
            if not self.cfg.ledger_path.exists():
                print(f"ERROR: ledger not found: {self.cfg.ledger_path}")
                status = "ledger_missing"
                rc = 2
                return rc

            if self.cfg.verify_chain:
                rc = self.verify_ledger_chain()
                if rc != 0:
                    status = "chain_verification_failed"
                    return rc

            if self.cfg.dry_run:
                status = "dry_run_ok"
                rc = 0
                return rc

            if self.cfg.create_constraints:
                constraints_attempted = True
                self._create_constraints()

            self._upsert_manifest_nodes()

            batch: List[Dict[str, Any]] = []
            for rec in iter_ledger(self.cfg.ledger_path):
                try:
                    fh = rec.get("hash")
                    if not isinstance(fh, str) or not fh:
                        continue
                    filename = rec.get("filename") if isinstance(rec.get("filename"), str) else ""
                    rel_path = rec.get("path") if isinstance(rec.get("path"), str) else ""
                    size_mb = float(rec.get("size_mb")) if isinstance(rec.get("size_mb"), (int, float)) else 0.0
                    impact = float(rec.get("impact_value")) if isinstance(rec.get("impact_value"), (int, float)) else 0.0
                    hash_kind = rec.get("hash_kind") if isinstance(rec.get("hash_kind"), str) else "metadata_sha256"

                    ext = Path(filename).suffix.lower() if filename else Path(rel_path).suffix.lower()
                    asset_class = classify_asset(ext)

                    batch.append(
                        {
                            "hash": fh,
                            "filename": filename,
                            "path": rel_path,
                            "size_mb": size_mb,
                            "impact_value": impact,
                            "hash_kind": hash_kind,
                            "extension": ext or "",
                            "asset_class": asset_class,
                        }
                    )

                    if len(batch) >= self.cfg.batch_size:
                        self._commit_batch(batch)
                        batch = []
                except Exception:
                    self.bad_lines += 1

            if batch:
                self._commit_batch(batch)

            elapsed = time.monotonic() - self.start_time
            print("------------------------------------------------")
            print("OK: Brain activation complete.")
            print(f"Nodes created:     {self.nodes_written:,}")
            print(f"Bad lines skipped: {self.bad_lines}")
            print(f"Time elapsed:      {elapsed:,.2f}s")
            print("------------------------------------------------")
            status = "ingest_ok"
            rc = 0
            return rc
        except KeyboardInterrupt:
            status = "interrupted"
            rc = 130
            print("ERROR: interrupted")
            return rc
        except Exception as e:
            status = "error"
            rc = 1
            print(f"ERROR: ingestion failed: {e}")
            return rc
        finally:
            try:
                out = self._write_receipt(status=status, rc=rc, constraints_attempted=constraints_attempted)
                if out is not None:
                    print(f"Receipt: {out}")
            except Exception as e:
                print(f"NOTE: failed to write receipt: {e}")


def parse_args(argv: Optional[Iterable[str]] = None) -> LoaderConfig:
    p = argparse.ArgumentParser(description="Load BIZRA knowledge ledger into Neo4j (batch ingestion).")
    p.add_argument("--ledger", default=str(LEDGER_DEFAULT), help="Path to BIZRA_KNOWLEDGE_LEDGER.jsonl")
    p.add_argument("--manifest", default=str(MANIFEST_DEFAULT), help="Path to BIZRA_KNOWLEDGE_MANIFEST.json")
    p.add_argument(
        "--tokenomics",
        default=str(TOKENOMICS_DEFAULT),
        help="Optional tokenomics YAML to cross-check against manifest",
    )
    p.add_argument("--uri", default=env_first("NEO4J_URI", "BIZRA_NEO4J_URI") or "bolt://localhost:7687")
    p.add_argument("--user", default=env_first("NEO4J_USER", "BIZRA_NEO4J_USER") or "neo4j")
    p.add_argument(
        "--password",
        default=env_first("GRAPH_PASSWORD", "NEO4J_PASSWORD", "BIZRA_NEO4J_PASSWORD") or "",
        help="Neo4j password (or set GRAPH_PASSWORD / NEO4J_PASSWORD env var)",
    )
    p.add_argument("--database", default=env_first("NEO4J_DATABASE", "BIZRA_NEO4J_DATABASE") or "", help="Neo4j database")
    p.add_argument("--batch-size", type=int, default=1000, help="Artifacts per transaction commit")
    p.add_argument("--no-verify-chain", action="store_true", help="Skip ledger chain verification")
    p.add_argument("--no-constraints", action="store_true", help="Do not create uniqueness constraints")
    p.add_argument("--dry-run", action="store_true", help="Verify/read only; do not write to Neo4j")
    p.add_argument("--no-receipt", action="store_true", help="Do not write an evidence receipt")
    p.add_argument(
        "--receipt-dir",
        default=str(RECEIPTS_DEFAULT_DIR),
        help="Base directory for receipts (default: docs/evidence/receipts)",
    )
    p.add_argument("--receipt-out", default="", help="Write receipt to this exact path (overrides --receipt-dir)")
    p.add_argument(
        "--receipt-include-paths",
        action="store_true",
        help="Include raw file paths and scan_root in the receipt (privacy risk)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.password and not args.dry_run:
        raise SystemExit("Neo4j password is required (set GRAPH_PASSWORD / NEO4J_PASSWORD or pass --password).")

    tokenomics_path = Path(args.tokenomics) if str(args.tokenomics).strip() else None
    if tokenomics_path is not None and not tokenomics_path.exists():
        tokenomics_path = None

    neo4j_db = str(args.database).strip() or None

    receipt_out = Path(args.receipt_out).expanduser().resolve() if str(args.receipt_out).strip() else None
    receipt_dir = Path(args.receipt_dir).expanduser().resolve()

    return LoaderConfig(
        ledger_path=Path(args.ledger),
        manifest_path=Path(args.manifest),
        tokenomics_path=tokenomics_path,
        neo4j_uri=str(args.uri),
        neo4j_user=str(args.user),
        neo4j_password=str(args.password),
        neo4j_db=neo4j_db,
        batch_size=max(1, int(args.batch_size)),
        verify_chain=not bool(args.no_verify_chain),
        create_constraints=not bool(args.no_constraints),
        dry_run=bool(args.dry_run),
        emit_receipt=not bool(args.no_receipt),
        receipt_dir=receipt_dir,
        receipt_out=receipt_out,
        receipt_include_paths=bool(args.receipt_include_paths),
    )


def main() -> int:
    _configure_stdout_utf8()
    cfg = parse_args()
    loader = SynapticLoader(cfg)
    try:
        return loader.load()
    finally:
        loader.close()


if __name__ == "__main__":
    raise SystemExit(main())
