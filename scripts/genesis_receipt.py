from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_file_normalized(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return sha256_bytes(normalized.encode("utf-8"))


def _parse_env_file(env_path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for raw in read_text(env_path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        if not key:
            continue
        env[key] = v.strip()
    return env


def _get_env(env_file: Optional[Path]) -> Dict[str, str]:
    merged = dict(os.environ)
    if env_file and env_file.exists():
        merged.update(_parse_env_file(env_file))
    return merged


def _run_git_rev_parse(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.STDOUT)
    except Exception as e:
        raise RuntimeError(f"git_commit_sha_unavailable: {e}") from e
    sha = out.decode("utf-8", errors="replace").strip()
    if not sha or len(sha) != 40:
        raise RuntimeError(f"git_commit_sha_invalid: {sha!r}")
    return sha.lower()


def _git_dirty(repo_root: Path) -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root), stderr=subprocess.STDOUT)
    except Exception:
        return True
    return bool(out.strip())


def http_json(
    *,
    url: str,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 1.0,
) -> Tuple[int, Dict[str, Any]]:
    payload = None
    final_headers = {"Accept": "application/json"}
    if headers:
        final_headers.update(headers)
    if body is not None:
        payload = canonical_json_bytes(body)
        final_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url=url, method=method, data=payload, headers=final_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # nosec - local-only URLs
            raw = resp.read()
            if not raw:
                return resp.status, {}
            return resp.status, json.loads(raw.decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            data = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
        except Exception:
            data = {"error": raw.decode("utf-8", errors="replace")}
        return e.code, data


def _basic_auth_header(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


@dataclass(frozen=True)
class Neo4jCounts:
    node_count: int
    relationship_count: int
    response_sha256: str


def neo4j_counts_http(*, http_base: str, user: str, password: str, timeout_s: float = 1.0) -> Neo4jCounts:
    url = http_base.rstrip("/") + "/db/neo4j/tx/commit"
    payload: Dict[str, Any] = {
        "statements": [
            {"statement": "MATCH (n) RETURN count(n) as c"},
            {"statement": "MATCH ()-[r]->() RETURN count(r) as c"},
        ]
    }
    code, data = http_json(
        url=url,
        method="POST",
        body=payload,
        headers={"Authorization": _basic_auth_header(user, password)},
        timeout_s=timeout_s,
    )
    if code < 200 or code >= 300:
        raise RuntimeError(f"neo4j_http_error: {code} {data}")

    results = data.get("results")
    if not isinstance(results, list) or len(results) < 2:
        raise RuntimeError(f"neo4j_unexpected_response: {data}")

    def _extract_count(idx: int) -> int:
        rows = results[idx].get("data")
        if not (isinstance(rows, list) and rows and isinstance(rows[0], dict)):
            raise RuntimeError(f"neo4j_missing_rows: idx={idx} {data}")
        row = rows[0]
        vals = row.get("row")
        if not (isinstance(vals, list) and vals and isinstance(vals[0], (int, float))):
            raise RuntimeError(f"neo4j_missing_count: idx={idx} {data}")
        return int(vals[0])

    return Neo4jCounts(
        node_count=_extract_count(0),
        relationship_count=_extract_count(1),
        response_sha256=sha256_json(data),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a cryptographically replayable Genesis Receipt for Node0 ignition.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]), help="Repo root (default: auto).")
    parser.add_argument("--env-file", default=".env", help="Env file (default: .env in repo root).")
    parser.add_argument("--compose-file", default="docker-compose.yml", help="Compose file path (relative to repo root).")
    parser.add_argument("--model-family-manifest", default="", help="Override model-family manifest path (relative to repo root).")
    parser.add_argument("--healthz-url", default="http://127.0.0.1:8010/healthz", help="Kernel healthz URL.")
    parser.add_argument("--sape-url", default="http://127.0.0.1:8010/v1/sape/plan", help="Kernel SAPE plan URL.")
    parser.add_argument("--neo4j-http", default="http://127.0.0.1:7474", help="Neo4j HTTP base URL (host port).")
    parser.add_argument("--out", default="", help="Receipt output path (relative to repo root). When omitted, a timestamped path is used.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    env_path = (repo_root / args.env_file).resolve()
    out_rel = args.out.strip()
    if out_rel:
        out_path = (repo_root / out_rel).resolve()
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        out_path = (repo_root / f"docs/evidence/receipts/genesis_receipt_v1_{ts}.json").resolve()

    print(f"[genesis_receipt] repo_root: {repo_root}")
    print(f"[genesis_receipt] env_file:  {env_path} ({'exists' if env_path.exists() else 'missing'})")
    print(f"[genesis_receipt] out:       {out_path}")

    env = _get_env(env_path if env_path.exists() else None)

    compose_path = (repo_root / args.compose_file).resolve()
    if not compose_path.exists():
        raise RuntimeError(f"compose_file_missing: {compose_path}")
    compose_sha = sha256_file_normalized(compose_path)

    manifest_path = (repo_root / args.model_family_manifest).resolve() if args.model_family_manifest else None
    if not manifest_path:
        manifest_path = Path(env.get("BIZRA_MODEL_FAMILY_MANIFEST") or "").expanduser().resolve() if env.get("BIZRA_MODEL_FAMILY_MANIFEST") else (repo_root / "model-family-genesis-v1-SEALED.yaml").resolve()
    if not manifest_path.exists():
        raise RuntimeError(f"model_family_manifest_missing: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)

    git_sha = _run_git_rev_parse(repo_root)
    git_dirty = _git_dirty(repo_root)

    gates_path = (repo_root / "docs/evidence/gates/node0_gates_latest.json").resolve()
    if not gates_path.exists():
        raise RuntimeError(f"gates_snapshot_missing: {gates_path}")
    gates_data = json.loads(gates_path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(gates_data, dict):
        raise RuntimeError("gates_snapshot_invalid: expected JSON object")
    if not gates_data.get("passed"):
        raise RuntimeError("gates_snapshot_not_passed: refuse_to_mint")

    gates_sha = sha256_json(gates_data)

    def _env_float(key: str, default: float) -> float:
        raw = (env.get(key) or "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except Exception:
            return default

    healthz_cfg = {
        "budget_s": _env_float("BIZRA_HEALTHZ_BUDGET_S", 0.85),
        "redis_timeout_s": _env_float("BIZRA_HEALTHZ_REDIS_TIMEOUT_S", 0.35),
        "neo4j_timeout_s": _env_float("BIZRA_HEALTHZ_NEO4J_TIMEOUT_S", 0.55),
        "llm_timeout_s": _env_float("BIZRA_HEALTHZ_LLM_TIMEOUT_S", 0.25),
        "cache_ttl_s": _env_float("BIZRA_HEALTHZ_CACHE_TTL_S", 0.5),
    }

    token = (env.get("BIZRA_API_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("missing BIZRA_API_TOKEN (set it in .env or environment)")

    print("[genesis_receipt] checking /healthz (must be ok)...")
    code, health = http_json(url=args.healthz_url, timeout_s=1.0)
    if code != 200 or health.get("status") != "ok":
        raise RuntimeError(f"kernel_unhealthy: http={code} body.status={health.get('status')!r}")

    sape_request: Dict[str, Any] = {
        "domain": "Economic Sovereignty",
        "objective": "Validate Proof of Impact tokenomics model against inflation risks.",
        "stakes": "H",
        "constraints": "No web. Use graph evidence only. Fail-closed if evidence is missing.",
        "success_criteria": "Identify inflation risk vectors, propose mitigations, cite evidence artifacts.",
        "forbidden_moves": ["hallucination", "hidden assumptions", "skipped proof", "missing verification steps"],
        "lenses": ["Systems Architect", "Pragmatic Engineer", "Ethicist"],
        "rarity_path_moves": 5,
        "require_graph_evidence": True,
        "evidence_topics": ["PoI", "tokenomics", "inflation", "supply_cap"],
        "evidence_limit": 8,
        "extra_instructions": "",
    }
    sape_req_sha = sha256_json(sape_request)

    print("[genesis_receipt] executing first SAPE plan (/v1/sape/plan)...")
    t_sape = time.monotonic()
    sape_code, sape_resp = http_json(
        url=args.sape_url,
        method="POST",
        body=sape_request,
        headers={"X-BIZRA-TOKEN": token},
        timeout_s=10.0,
    )
    sape_latency_ms = round((time.monotonic() - t_sape) * 1000.0, 2)
    sape_resp_sha = sha256_json(sape_resp)

    neo4j_auth = (env.get("NEO4J_AUTH") or "").strip()
    neo_counts: Optional[Neo4jCounts] = None
    neo_error: Optional[str] = None
    if neo4j_auth and "/" in neo4j_auth:
        neo_user, neo_pass = neo4j_auth.split("/", 1)
        try:
            print("[genesis_receipt] querying Neo4j counts (HTTP)...")
            neo_counts = neo4j_counts_http(http_base=args.neo4j_http, user=neo_user, password=neo_pass, timeout_s=1.0)
        except Exception as e:
            neo_error = str(e)
    else:
        neo_error = "neo4j_auth_missing_or_invalid"

    receipt: Dict[str, Any] = {
        "schema": "bizra_genesis_receipt_v1",
        "version": 1,
        "truth_label": "MEASURED",
        "timestamp_utc": utc_now_iso(),
        "git_commit_sha": git_sha,
        "git_dirty": git_dirty,
        "artifacts": {
            "compose_file": str(compose_path.relative_to(repo_root)).replace("\\", "/"),
            "compose_sha256": compose_sha,
            "model_family_manifest": str(manifest_path.relative_to(repo_root)).replace("\\", "/"),
            "model_family_manifest_sha256": manifest_sha,
            "gates_file": str(gates_path.relative_to(repo_root)).replace("\\", "/"),
            "gates_sha256": gates_sha,
        },
        "healthz_config": healthz_cfg,
        "healthz_snapshot": {
            "url": args.healthz_url,
            "status_code": code,
            "body_sha256": sha256_json(health),
            "elapsed_ms": health.get("elapsed_ms"),
            "checks": {
                "synapse": (health.get("checks") or {}).get("synapse"),
                "wisdom": (health.get("checks") or {}).get("wisdom"),
                "ollama": (health.get("checks") or {}).get("ollama"),
                "lmstudio": (health.get("checks") or {}).get("lmstudio"),
            },
        },
        "first_sape_plan": {
            "url": args.sape_url,
            "request_sha256": sape_req_sha,
            "response_sha256": sape_resp_sha,
            "http_status": sape_code,
            "latency_ms": sape_latency_ms,
            "response_status": sape_resp.get("status"),
            "slot": sape_resp.get("slot"),
            "prompt_sha256": sape_resp.get("prompt_sha256"),
            "audit_id": sape_resp.get("audit_id"),
            "missing_artifacts": sape_resp.get("missing_artifacts"),
            "warnings": sape_resp.get("warnings"),
        },
        "neo4j_snapshot": {
            "http_url": args.neo4j_http,
            "node_count": neo_counts.node_count if neo_counts else None,
            "relationship_count": neo_counts.relationship_count if neo_counts else None,
            "query_response_sha256": neo_counts.response_sha256 if neo_counts else None,
            "error": neo_error,
        },
        "gates_snapshot": gates_data,
    }
    receipt["receipt_sha256"] = sha256_json(receipt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[genesis_receipt] wrote: {out_path}")
    print(f"[genesis_receipt] receipt_sha256: {receipt['receipt_sha256']}")
    print("[genesis_receipt] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
