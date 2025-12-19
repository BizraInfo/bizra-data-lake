from __future__ import annotations

import argparse
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import urllib.request
import urllib.error


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_env_file(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        if not key:
            continue
        env[key] = v
    return env


def merge_env(env_file: Path) -> Dict[str, str]:
    merged = dict(os.environ)
    merged.update(parse_env_file(env_file))
    return merged


def require_auth(raw: str | None) -> tuple[str, str]:
    if not raw:
        raise SystemExit("NEO4J_AUTH missing (expected user/password)")
    if "/" in raw:
        user, password = raw.split("/", 1)
    elif ":" in raw:
        user, password = raw.split(":", 1)
    else:
        raise SystemExit("NEO4J_AUTH invalid (expected user/password)")
    user = user.strip()
    password = password.strip()
    if not user or not password:
        raise SystemExit("NEO4J_AUTH invalid (empty user/password)")
    return user, password


def neo4j_request(http_url: str, user: str, password: str, statements: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = json.dumps({"statements": statements}).encode("utf-8")
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Basic {token}",
    }
    url = http_url.rstrip("/") + "/db/neo4j/tx/commit"
    req = urllib.request.Request(url=url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:  # nosec - local service call
            raw = resp.read()
    except urllib.error.HTTPError as e:  # pragma: no cover - best effort diagnostics
        data = e.read()
        raise SystemExit(f"neo4j_http_error: {e.code} {data.decode('utf-8', errors='replace')}") from e
    return json.loads(raw.decode("utf-8", errors="replace"))


def seed_artifacts(http_url: str, user: str, password: str) -> None:
    seeded_at = utc_iso()
    artifacts: List[Dict[str, Any]] = [
        {
            "hash": "seed-poi-inflation-whitepaper",
            "filename": "PoI_inflation_playbook.md",
            "path": "knowledge/tokenomics/poi_inflation_playbook.md",
            "impact_value": 5.0,
            "size_mb": 0.12,
            "asset_class": "docs",
            "extension": ".md",
            "hash_kind": "sha256",
            "topics": ["PoI", "inflation", "tokenomics"],
        },
        {
            "hash": "seed-tokenomics-guardrails",
            "filename": "tokenomics_supply_cap_guardrails.pdf",
            "path": "knowledge/tokenomics/supply_cap_guardrails.pdf",
            "impact_value": 4.0,
            "size_mb": 0.32,
            "asset_class": "docs",
            "extension": ".pdf",
            "hash_kind": "sha256",
            "topics": ["tokenomics", "supply_cap"],
        },
        {
            "hash": "seed-poi-stability-brief",
            "filename": "PoI_stability_brief.txt",
            "path": "knowledge/poi/poi_macro_stability_brief.txt",
            "impact_value": 3.0,
            "size_mb": 0.05,
            "asset_class": "docs",
            "extension": ".txt",
            "hash_kind": "sha256",
            "topics": ["PoI", "economic_sovereignty"],
        },
    ]

    statements: List[Dict[str, Any]] = [
        {
            "statement": (
                "UNWIND $rows AS row "
                "MERGE (a:Artifact {hash: row.hash}) "
                "SET a.filename = row.filename, "
                "    a.path = row.path, "
                "    a.impact_value = row.impact_value, "
                "    a.size_mb = row.size_mb, "
                "    a.asset_class = row.asset_class, "
                "    a.extension = row.extension, "
                "    a.hash_kind = row.hash_kind, "
                "    a.seeded_at_utc = row.seeded_at "
                "WITH a, row "
                "FOREACH (topic IN row.topics | MERGE (t:Topic {name: topic}) MERGE (a)-[:TAGGED_WITH]->(t))"
            ),
            "parameters": {"rows": [{**a, "seeded_at": seeded_at} for a in artifacts]},
        },
    ]

    response = neo4j_request(http_url=http_url, user=user, password=password, statements=statements)
    if response.get("errors"):
        raise SystemExit(f"neo4j_seed_errors: {response['errors']}")

    stats_stmt = {
        "statement": "MATCH (a:Artifact) RETURN count(a) AS artifacts",
        "parameters": {},
    }
    stats_resp = neo4j_request(http_url=http_url, user=user, password=password, statements=[stats_stmt])
    count = 0
    try:
        count = int(stats_resp["results"][0]["data"][0]["row"][0])
    except Exception:
        pass
    print(f"[seed_min_evidence] Seeded artifacts (total_artifacts={count})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed minimal deterministic evidence for SAPE Gate E2.")
    parser.add_argument("--env-file", default=".env", help="Path to .env file (default: .env in repo root).")
    parser.add_argument("--neo4j-http", default="http://127.0.0.1:7474", help="Neo4j HTTP endpoint (host-accessible).")
    args = parser.parse_args()

    env_file = Path(args.env_file).expanduser().resolve()
    merged_env = merge_env(env_file)

    auth_raw = merged_env.get("NEO4J_AUTH") or os.getenv("NEO4J_AUTH")
    user, password = require_auth(auth_raw)

    seed_artifacts(http_url=args.neo4j_http, user=user, password=password)
    print("[seed_min_evidence] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
