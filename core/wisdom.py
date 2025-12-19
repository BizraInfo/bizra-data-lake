from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase  # type: ignore


@dataclass(frozen=True)
class GraphConfig:
    uri: str
    user: str
    password: str
    database: Optional[str]


def _parse_neo4j_auth(raw: str) -> Optional[tuple[str, str]]:
    raw = (raw or "").strip()
    if not raw or raw.lower() == "none":
        return None
    if "/" in raw:
        user, password = raw.split("/", 1)
    elif ":" in raw:
        user, password = raw.split(":", 1)
    else:
        return None
    user = user.strip()
    password = password.strip()
    if not user or not password:
        return None
    return user, password


def _graph_config() -> GraphConfig:
    uri = os.getenv("WISDOM_URL", os.getenv("NEO4J_URI", os.getenv("BIZRA_NEO4J_URI", "bolt://localhost:7687"))).strip()

    auth = _parse_neo4j_auth(os.getenv("NEO4J_AUTH", ""))
    user = os.getenv("NEO4J_USER", os.getenv("BIZRA_NEO4J_USER", "neo4j")).strip()
    password = os.getenv("GRAPH_PASSWORD", os.getenv("NEO4J_PASSWORD", "")).strip()
    if auth:
        user, password = auth
    database = os.getenv("NEO4J_DATABASE", os.getenv("BIZRA_NEO4J_DATABASE", "")).strip() or None
    return GraphConfig(uri=uri, user=user, password=password, database=database)


class HouseOfWisdom:
    def __init__(self) -> None:
        self.cfg = _graph_config()
        self.driver = None
        self.last_error: Optional[str] = None
        self._connect()

    def _connect(self) -> None:
        self.cfg = _graph_config()
        if not self.cfg.password:
            self.driver = None
            self.last_error = "NEO4J_AUTH or GRAPH_PASSWORD/NEO4J_PASSWORD not set"
            return
        try:
            self.driver = GraphDatabase.driver(self.cfg.uri, auth=(self.cfg.user, self.cfg.password))
            with self._session() as s:
                s.run("RETURN 1").consume()
            self.last_error = None
        except Exception as e:
            self.driver = None
            self.last_error = str(e)

    def _session(self):
        assert self.driver is not None
        if self.cfg.database:
            return self.driver.session(database=self.cfg.database)
        return self.driver.session()

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
        self.driver = None

    def is_online(self) -> bool:
        if self.driver is None:
            return False
        try:
            with self._session() as s:
                s.run("RETURN 1").consume()
            return True
        except Exception as e:
            self.last_error = str(e)
            self.driver = None
            return False

    def ensure_online(self) -> bool:
        if self.is_online():
            return True
        self._connect()
        return self.is_online()

    def query_knowledge(self, *, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.ensure_online():
            raise RuntimeError(f"graph offline: {self.last_error}")

        topic = (topic or "").strip().lower()
        limit = max(1, min(int(limit), 100))

        cypher = """
        MATCH (a:Artifact)
        WHERE toLower(a.path) CONTAINS $topic OR toLower(a.filename) CONTAINS $topic
        RETURN
          a.hash AS hash,
          a.filename AS filename,
          a.path AS path,
          a.impact_value AS impact_value,
          a.size_mb AS size_mb,
          a.asset_class AS asset_class,
          a.extension AS extension,
          a.hash_kind AS hash_kind
        ORDER BY a.impact_value DESC
        LIMIT $limit
        """
        t0 = time.monotonic()
        with self._session() as session:
            result = session.run(cypher, topic=topic, limit=limit)
            rows = [record.data() for record in result]
        _ = time.monotonic() - t0
        return rows

    def get_stats(self) -> Dict[str, Any]:
        if not self.ensure_online():
            return {"status": "OFFLINE", "error": self.last_error}

        cypher = """
        OPTIONAL MATCH (m:KnowledgeManifest)-[:ANCHORED_TO]->(g:GenesisBlock)
        WITH m, g
        ORDER BY m.created_at DESC
        LIMIT 1
        RETURN
          m.ledger_chain_sha256 AS ledger_chain_sha256,
          m.total_files AS total_files,
          m.total_value_bzr_g AS total_value_bzr_g,
          g.hash AS genesis_hash
        """
        with self._session() as session:
            row = session.run(cypher).single()
        if row and row.get("ledger_chain_sha256"):
            return {
                "status": "ONLINE",
                "ledger_chain_sha256": row.get("ledger_chain_sha256"),
                "genesis_hash": row.get("genesis_hash"),
                "total_files": row.get("total_files"),
                "total_value_bzr_g": row.get("total_value_bzr_g"),
            }

        with self._session() as session:
            row2 = session.run("MATCH (a:Artifact) RETURN count(a) AS total_files, sum(a.impact_value) AS total_value_bzr_g").single()
        return {
            "status": "ONLINE",
            "total_files": row2.get("total_files") if row2 else None,
            "total_value_bzr_g": row2.get("total_value_bzr_g") if row2 else None,
        }
