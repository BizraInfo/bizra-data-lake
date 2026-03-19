#!/usr/bin/env python3
"""
Deterministic memory quality gate for CI/CD.

This script turns the unified memory blueprint into an executable gate:
- build a known-good Claude-flow fixture
- run full convergence into AgentDB
- seed benchmark vectors
- rebuild indexes
- measure hybrid search latency
- emit a JSON receipt and fail closed on threshold violations
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.memory import AgentDB, ConvergencePolicy, MemoryConfig, run_convergence
from core.memory.config import HNSWConfig
from core.memory.types import MemoryKind, MemoryRecord


@dataclass(frozen=True)
class GateThresholds:
    """Deterministic thresholds for the CI memory quality gate."""

    convergence_max_ms: float = 5000.0
    rebuild_max_ms: float = 1000.0
    search_p95_ms: float = 25.0
    search_p50_ms: float = 10.0
    min_imported_records: int = 5
    min_indexed_vectors: int = 8
    search_iterations: int = 25


def create_fixture_sources(root: Path) -> dict[str, Path]:
    """Create a small known-good Claude-flow fixture surface."""
    db_path = root / ".swarm" / "memory.db"
    artifact_dir = root / ".claude-flow" / "memory"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE memory_entries (
            id TEXT PRIMARY KEY,
            key TEXT,
            namespace TEXT,
            type TEXT,
            content TEXT,
            tags TEXT,
            metadata TEXT,
            owner_id TEXT,
            created_at INTEGER,
            updated_at INTEGER,
            last_accessed_at INTEGER,
            access_count INTEGER,
            status TEXT
        );
        CREATE TABLE patterns (
            id TEXT PRIMARY KEY,
            name TEXT,
            pattern_type TEXT,
            description TEXT,
            tags TEXT,
            created_at INTEGER,
            updated_at INTEGER,
            confidence REAL
        );
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            project_path TEXT
        );
        """)
    now_ms = 1_773_210_000_000
    conn.executemany(
        """
        INSERT INTO memory_entries (
            id, key, namespace, type, content, tags, metadata, owner_id,
            created_at, updated_at, last_accessed_at, access_count, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "mem-1",
                "alpha",
                "bizra",
                "semantic",
                "Alpha bridge memory for CI quality gating",
                json.dumps(["ops", "alpha"]),
                json.dumps({"related_ids": ["ctx-alpha"]}),
                "owner",
                now_ms,
                now_ms + 10,
                now_ms + 20,
                2,
                "active",
            ),
            (
                "mem-2",
                "beta",
                "bizra",
                "procedural",
                "Beta procedure for memory convergence",
                json.dumps(["ops", "beta"]),
                json.dumps({"related_ids": ["ctx-beta"]}),
                "owner",
                now_ms,
                now_ms + 10,
                now_ms + 20,
                1,
                "active",
            ),
        ],
    )
    conn.execute(
        """
        INSERT INTO patterns (
            id, name, pattern_type, description, tags, created_at, updated_at, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "pat-1",
            "Observe/Throttle",
            "governance",
            "Quarantine-not-evict pattern for CI smoke",
            json.dumps(["pattern", "governance"]),
            now_ms,
            now_ms + 10,
            0.91,
        ),
    )
    conn.execute(
        "INSERT INTO sessions (id, project_path) VALUES (?, ?)",
        ("session-1", str(root)),
    )
    conn.commit()
    conn.close()

    (artifact_dir / "session-index.json").write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "session_id": "fixture-session",
                        "timestamp": "2026-03-11T07:00:00Z",
                        "summary": "Fixture session summary for CI convergence",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "project-patterns.json").write_text(
        json.dumps(
            {
                "patterns": {
                    "memory_ci_gate": {
                        "owner": "platform",
                        "rule": "fail closed on stale memory quality",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    return {"db_path": db_path, "artifact_dir": artifact_dir}


def seed_benchmark_vectors(db: AgentDB, count: int = 8) -> int:
    """Seed deterministic vectors to exercise rebuild + hybrid search."""
    timestamp = datetime.now(timezone.utc)
    for index in range(count):
        vec = [0.0] * 8
        vec[index % 8] = 1.0
        db.store_record(
            MemoryRecord(
                id=f"bench-{index}",
                content=f"Benchmark alpha memory {index}",
                kind=MemoryKind.SEMANTIC,
                embedding=vec,
                importance=0.8,
                source="memory_quality_gate",
                source_id=str(index),
                tags=["benchmark", "alpha"],
                related_ids=["ctx-alpha"],
                created_at=timestamp,
                updated_at=timestamp,
                last_accessed=timestamp,
                metadata={"fixture": True, "slot": index},
            )
        )
    return count


def measure_search_latency(
    db: AgentDB,
    *,
    iterations: int,
    query: str = "Benchmark alpha memory",
    query_embedding: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Measure repeated search latency over a deterministic dataset."""
    query_embedding = query_embedding or [1.0] + [0.0] * 7
    samples_ms: list[float] = []
    result_count = 0

    for _ in range(iterations):
        started = perf_counter()
        results = db.search(
            query=query,
            query_embedding=query_embedding,
            top_k=5,
            min_score=0.0,
            tags=["benchmark"],
            context_ids=["ctx-alpha"],
        )
        samples_ms.append(round((perf_counter() - started) * 1000, 4))
        result_count = max(result_count, len(results))

    return {
        "iterations": iterations,
        "p50_ms": round(_percentile(samples_ms, 50), 4),
        "p95_ms": round(_percentile(samples_ms, 95), 4),
        "max_ms": round(max(samples_ms) if samples_ms else 0.0, 4),
        "result_count": result_count,
        "samples_ms": samples_ms,
    }


def run_memory_quality_gate(
    *,
    report_out: Path | None = None,
    thresholds: GateThresholds | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run the deterministic memory quality gate and return (exit_code, report)."""
    gate = thresholds or GateThresholds()

    with tempfile.TemporaryDirectory(prefix="bizra-memory-gate-") as tmp:
        root = Path(tmp)
        fixture = create_fixture_sources(root)
        config = MemoryConfig(
            data_dir=root / "agent_db",
            hnsw=HNSWConfig(dimensions=8, max_elements=256),
        )
        config.auto_embed = False

        started = perf_counter()
        convergence_exit_code, convergence = run_convergence(
            config=config,
            claude_flow_db=fixture["db_path"],
            claude_flow_dir=fixture["artifact_dir"],
            dry_run=False,
            policy=ConvergencePolicy(),
        )
        convergence_ms = round((perf_counter() - started) * 1000, 3)

        db = AgentDB(config)
        db.initialize()
        benchmark_seeded = seed_benchmark_vectors(db)
        rebuild = db.rebuild_indexes()
        search_latency = measure_search_latency(
            db,
            iterations=gate.search_iterations,
        )
        stats = db.stats()

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds": asdict(gate),
        "convergence": convergence,
        "convergence_exit_code": convergence_exit_code,
        "convergence_duration_ms": convergence_ms,
        "benchmark_seeded_records": benchmark_seeded,
        "rebuild": rebuild,
        "stats": stats,
        "search_latency": search_latency,
    }
    gate_result = _evaluate_quality_gate(report, gate)
    report["gate"] = gate_result

    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )

    return (0 if gate_result["passed"] else 1), report


def _evaluate_quality_gate(
    report: dict[str, Any],
    thresholds: GateThresholds,
) -> dict[str, Any]:
    reasons: list[str] = []

    if report["convergence_exit_code"] != 0:
        reasons.append("convergence_failed")
    if report["convergence_duration_ms"] > thresholds.convergence_max_ms:
        reasons.append(f"convergence_duration_ms={report['convergence_duration_ms']}")
    if (
        report["convergence"]["migration"]["total_imported"]
        < thresholds.min_imported_records
    ):
        reasons.append(
            f"imported_records={report['convergence']['migration']['total_imported']}"
        )
    if report["stats"]["index_health"]["status"] != "healthy":
        reasons.append(f"index_health={report['stats']['index_health']['status']}")
    if report["rebuild"]["duration_ms"] > thresholds.rebuild_max_ms:
        reasons.append(f"rebuild_duration_ms={report['rebuild']['duration_ms']}")
    if report["rebuild"]["indexed_vectors"] < thresholds.min_indexed_vectors:
        reasons.append(f"indexed_vectors={report['rebuild']['indexed_vectors']}")
    if report["search_latency"]["p50_ms"] > thresholds.search_p50_ms:
        reasons.append(f"search_p50_ms={report['search_latency']['p50_ms']}")
    if report["search_latency"]["p95_ms"] > thresholds.search_p95_ms:
        reasons.append(f"search_p95_ms={report['search_latency']['p95_ms']}")
    if report["search_latency"]["result_count"] == 0:
        reasons.append("search_results=0")

    return {"passed": not reasons, "reasons": reasons}


def _percentile(samples: Sequence[float], pct: int) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    rank = max(0, min(len(ordered) - 1, round((pct / 100) * (len(ordered) - 1))))
    return ordered[rank]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic memory convergence and performance quality gate",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("artifacts/memory/memory_quality_gate.json"),
        help="Write the gate receipt JSON to this path",
    )
    parser.add_argument(
        "--convergence-max-ms",
        type=float,
        default=GateThresholds.convergence_max_ms,
        help="Fail if end-to-end convergence exceeds this duration",
    )
    parser.add_argument(
        "--rebuild-max-ms",
        type=float,
        default=GateThresholds.rebuild_max_ms,
        help="Fail if index rebuild exceeds this duration",
    )
    parser.add_argument(
        "--search-p50-ms",
        type=float,
        default=GateThresholds.search_p50_ms,
        help="Fail if search p50 exceeds this duration",
    )
    parser.add_argument(
        "--search-p95-ms",
        type=float,
        default=GateThresholds.search_p95_ms,
        help="Fail if search p95 exceeds this duration",
    )
    parser.add_argument(
        "--min-imported-records",
        type=int,
        default=GateThresholds.min_imported_records,
        help="Fail if convergence imports fewer records than this floor",
    )
    parser.add_argument(
        "--min-indexed-vectors",
        type=int,
        default=GateThresholds.min_indexed_vectors,
        help="Fail if rebuild yields fewer indexed vectors than this floor",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=GateThresholds.search_iterations,
        help="Number of repeated searches used for latency measurement",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    thresholds = GateThresholds(
        convergence_max_ms=args.convergence_max_ms,
        rebuild_max_ms=args.rebuild_max_ms,
        search_p95_ms=args.search_p95_ms,
        search_p50_ms=args.search_p50_ms,
        min_imported_records=args.min_imported_records,
        min_indexed_vectors=args.min_indexed_vectors,
        search_iterations=args.search_iterations,
    )
    exit_code, report = run_memory_quality_gate(
        report_out=args.report_out,
        thresholds=thresholds,
    )
    print("Memory quality gate:")
    print(
        f"  convergence: exit={report['convergence_exit_code']} "
        f"duration_ms={report['convergence_duration_ms']} "
        f"imported={report['convergence']['migration']['total_imported']}"
    )
    print(
        f"  indexes: status={report['stats']['index_health']['status']} "
        f"vectors={report['rebuild']['indexed_vectors']} "
        f"rebuild_ms={report['rebuild']['duration_ms']}"
    )
    print(
        f"  search: p50_ms={report['search_latency']['p50_ms']} "
        f"p95_ms={report['search_latency']['p95_ms']} "
        f"results={report['search_latency']['result_count']}"
    )
    print(f"  gate: {'PASSED' if report['gate']['passed'] else 'BLOCKED'}")
    for reason in report["gate"]["reasons"]:
        print(f"    - {reason}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
