"""Typed bridge from StereoscopicReport nodes into bizra-memory ingestion."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class MemoryFragmentKind(str, Enum):
    """Rust bizra-memory fragment kinds (string form for bridge transport)."""

    USER_MESSAGE = "UserMessage"
    ASSISTANT_MESSAGE = "AssistantMessage"
    FILE_CONTENT = "FileContent"
    OBSERVATION = "Observation"
    SYSTEM_EVENT = "SystemEvent"
    EXTERNAL_DATA = "ExternalData"


@dataclass(frozen=True)
class MemoryFragmentInput:
    """Typed fragment payload for bizra-memory ingestion."""

    fragment_kind: MemoryFragmentKind
    content: str
    session_id: int
    turn: int
    timestamp: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_kind": self.fragment_kind.value,
            "content": self.content,
            "session_id": self.session_id,
            "turn": self.turn,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class BridgeIngestResult:
    """Outcome of adapting and ingesting stereoscopic nodes."""

    prepared: int
    ingested: int
    skipped: int
    backend_available: bool
    backend_mode: str
    exported_jsonl: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prepared": self.prepared,
            "ingested": self.ingested,
            "skipped": self.skipped,
            "backend_available": self.backend_available,
            "backend_mode": self.backend_mode,
            "exported_jsonl": self.exported_jsonl,
        }


class _BizraMemoryBackend(Protocol):
    def process_user_turn(
        self, content: str, session_id: int, turn: int, timestamp: int
    ) -> Any: ...

    def process_assistant_turn(
        self, content: str, session_id: int, turn: int, timestamp: int
    ) -> Any: ...


def _node_to_fragment_kind(node_kind: str) -> MemoryFragmentKind:
    norm = (node_kind or "").strip().lower()
    if norm in {"preference", "goal", "expertise"}:
        return MemoryFragmentKind.USER_MESSAGE
    if norm in {"style", "domain"}:
        return MemoryFragmentKind.ASSISTANT_MESSAGE
    if norm in {"fact", "pattern", "emotion", "relationship", "temporal"}:
        return MemoryFragmentKind.OBSERVATION
    return MemoryFragmentKind.OBSERVATION


def _node_to_content(node: dict[str, Any]) -> str:
    providers = ", ".join(node.get("providers") or [])
    source_tags = ", ".join(node.get("source_tags") or [])
    signal = str(node.get("signal") or "").strip()
    kind = str(node.get("kind") or "Unknown").strip()
    snr = float(node.get("snr_score", 0.0))
    evidence = int(node.get("evidence_count", 0))
    provider_count = int(node.get("provider_count", 0))
    return (
        f"[StereoscopicNode] kind={kind}; signal={signal}; snr={snr:.6f}; "
        f"evidence={evidence}; provider_count={provider_count}; "
        f"providers={providers}; sources={source_tags}"
    )


def build_fragment_inputs_from_report(
    report: dict[str, Any],
    *,
    min_snr: float = 0.0,
    session_id: int = 9000,
    start_turn: int = 1,
    timestamp: int | None = None,
) -> list[MemoryFragmentInput]:
    """Convert report nodes to typed fragment inputs."""
    ts = timestamp
    if ts is None:
        ts = int(dt.datetime.now(dt.timezone.utc).timestamp())

    rows = sorted(
        [n for n in (report.get("nodes") or []) if float(n.get("snr_score", 0.0)) >= min_snr],
        key=lambda n: (-float(n.get("snr_score", 0.0)), str(n.get("node_id") or "")),
    )

    out: list[MemoryFragmentInput] = []
    for idx, node in enumerate(rows):
        turn = start_turn + idx
        out.append(
            MemoryFragmentInput(
                fragment_kind=_node_to_fragment_kind(str(node.get("kind") or "")),
                content=_node_to_content(node),
                session_id=session_id,
                turn=turn,
                timestamp=ts,
                metadata={
                    "node_id": node.get("node_id"),
                    "kind": node.get("kind"),
                    "signal": node.get("signal"),
                    "snr_score": node.get("snr_score"),
                    "provider_count": node.get("provider_count"),
                    "providers": node.get("providers") or [],
                    "source_tags": node.get("source_tags") or [],
                },
            )
        )
    return out


def _load_default_backend() -> tuple[_BizraMemoryBackend | None, str]:
    try:
        from bizra_python import BizraMemory  # type: ignore

        return BizraMemory(), "bizra_python.BizraMemory"
    except Exception:
        return None, "unavailable"


def _ingest_fragment_with_backend(
    backend: _BizraMemoryBackend,
    fragment: MemoryFragmentInput,
) -> bool:
    # Current PyO3 binding exposes user/assistant entry points, so route typed
    # fragments through the closest semantic path until direct typed ingest is exposed.
    if fragment.fragment_kind in {
        MemoryFragmentKind.USER_MESSAGE,
        MemoryFragmentKind.FILE_CONTENT,
    }:
        result = backend.process_user_turn(
            fragment.content, fragment.session_id, fragment.turn, fragment.timestamp
        )
    else:
        result = backend.process_assistant_turn(
            fragment.content, fragment.session_id, fragment.turn, fragment.timestamp
        )

    if isinstance(result, dict):
        return bool(result.get("ingested", False))
    return True


def ingest_report_nodes(
    report: dict[str, Any],
    *,
    min_snr: float = 0.0,
    session_id: int = 9000,
    start_turn: int = 1,
    timestamp: int | None = None,
    backend: _BizraMemoryBackend | None = None,
    export_jsonl_path: str | Path | None = None,
) -> BridgeIngestResult:
    """Bridge stereoscopic nodes into bizra-memory ingest with typed payloads."""
    fragments = build_fragment_inputs_from_report(
        report,
        min_snr=min_snr,
        session_id=session_id,
        start_turn=start_turn,
        timestamp=timestamp,
    )

    loaded_backend = backend
    backend_mode = "explicit"
    if loaded_backend is None:
        loaded_backend, backend_mode = _load_default_backend()

    ingested = 0
    skipped = 0
    if loaded_backend is not None:
        for fragment in fragments:
            if _ingest_fragment_with_backend(loaded_backend, fragment):
                ingested += 1
            else:
                skipped += 1
    else:
        skipped = len(fragments)

    exported_jsonl = ""
    if export_jsonl_path:
        path = Path(export_jsonl_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for fragment in fragments:
                fh.write(json.dumps(fragment.to_dict(), ensure_ascii=False) + "\n")
        exported_jsonl = str(path)

    return BridgeIngestResult(
        prepared=len(fragments),
        ingested=ingested,
        skipped=skipped,
        backend_available=loaded_backend is not None,
        backend_mode=backend_mode,
        exported_jsonl=exported_jsonl,
    )


# ── AgentDB bridge ───────────────────────────────────────────────────────

# Mapping from StereoscopicReport node kinds to AgentDB MemoryKind values.
# Lazy-imported to avoid hard dependency on core.memory at module load time.
_NODE_KIND_TO_MEMORY_KIND: dict[str, str] = {
    "fact": "SEMANTIC",
    "preference": "SEMANTIC",
    "goal": "SEMANTIC",
    "expertise": "SEMANTIC",
    "pattern": "PROCEDURAL",
    "relationship": "EPISODIC",
    "emotion": "EPISODIC",
    "style": "SEMANTIC",
    "domain": "SEMANTIC",
    "temporal": "EPISODIC",
}


def ingest_report_to_agent_db(
    report: dict[str, Any],
    agent_db: Any,  # AgentDB instance
    *,
    min_snr: float = 0.0,
    source: str = "stereoscopic_compilation",
) -> dict[str, int]:
    """Bridge stereoscopic signal nodes into AgentDB unified memory.

    This closes the cold-start bootstrap loop: TEACH atoms -> self-compilation ->
    stereoscopic report -> AgentDB, making identity signals searchable via
    the unified /v1/memory/search endpoint.

    Parameters
    ----------
    report:
        StereoscopicReport dict with a ``nodes`` list.  Each node dict must
        contain at least ``kind``, ``signal``, and ``snr_score``.
    agent_db:
        An initialised ``AgentDB`` instance (``core.memory.agent_db.AgentDB``).
    min_snr:
        Drop signal nodes whose ``snr_score`` is below this threshold.
    source:
        Provenance tag written to every stored ``MemoryRecord``.

    Returns
    -------
    dict with ``stored``, ``skipped``, and ``errors`` counts.
    """
    from core.memory.types import MemoryKind

    nodes: list[dict[str, Any]] = report.get("nodes") or []

    stored = 0
    skipped = 0
    errors = 0

    for node in nodes:
        snr = float(node.get("snr_score", 0.0))

        # Gate: skip nodes below the SNR floor.
        if snr < min_snr:
            skipped += 1
            continue

        signal = str(node.get("signal") or "").strip()
        if not signal:
            skipped += 1
            continue

        node_kind = str(node.get("kind") or "").strip().lower()
        kind_name = _NODE_KIND_TO_MEMORY_KIND.get(node_kind, "SEMANTIC")
        memory_kind = MemoryKind(kind_name.lower())

        ihsan = min(snr, 1.0)

        try:
            agent_db.store(
                content=signal,
                kind=memory_kind,
                importance=snr,
                source=source,
                tags=[node_kind] if node_kind else [],
                metadata={
                    "ihsan_score": ihsan,
                    "node_kind": node_kind,
                    "confidence": float(node.get("confidence", 0.0)),
                    "providers": node.get("providers") or [],
                    "bridge": "stereoscopic_to_agentdb",
                },
            )
            stored += 1
        except Exception:
            errors += 1

    return {"stored": stored, "skipped": skipped, "errors": errors}
