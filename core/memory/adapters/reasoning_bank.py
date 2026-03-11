"""
ReasoningBank Adapter — Wraps hook-level ReasoningBank for AgentDB.

Converts ReasoningBank experiences and patterns (stored as JSONL/JSON in
.claude/state/reasoning_bank/) into MemoryRecords for semantic vector
search via AgentDB's HNSW index.

This enables:
  - Semantic retrieval of past experiences by context similarity (150x faster)
  - Cross-session pattern matching via vector embeddings
  - Unified search across all memory systems

Standing on Giants: ADR-006 (Unified Memory Service), Boyd (OODA, 1976)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.memory.types import MemoryKind, MemoryRecord, RecordState

logger = logging.getLogger(__name__)

# Default path for ReasoningBank state (hook system uses .claude/state/)
_DEFAULT_RB_DIR = Path(".claude") / "state" / "reasoning_bank"


class ReasoningBankAdapter:
    """Adapts ReasoningBank experiences/patterns to AgentDB MemoryRecords.

    Usage:
        adapter = ReasoningBankAdapter()
        records = adapter.export_all()
        for record in records:
            agent_db.store_record(record)

    Or via MigrationOrchestrator:
        orch.set_reasoning_bank(adapter)
    """

    def __init__(
        self,
        state_dir: Optional[Path] = None,
    ) -> None:
        self._state_dir = state_dir or _DEFAULT_RB_DIR

    @property
    def available(self) -> bool:
        """Check if ReasoningBank state exists."""
        return (self._state_dir / "experiences.jsonl").exists()

    def export_all(self) -> List[MemoryRecord]:
        """Export all experiences and patterns as MemoryRecords."""
        records: List[MemoryRecord] = []
        records.extend(self._export_experiences())
        records.extend(self._export_patterns())
        records.extend(self._export_strategies())
        logger.info(
            f"ReasoningBankAdapter: exported {len(records)} records "
            f"({self._state_dir})"
        )
        return records

    def export_experiences(self) -> List[MemoryRecord]:
        """Export only experiences (for incremental sync)."""
        return self._export_experiences()

    def export_patterns(self) -> List[MemoryRecord]:
        """Export only patterns."""
        return self._export_patterns()

    def export_strategies(self) -> List[MemoryRecord]:
        """Export only strategies."""
        return self._export_strategies()

    # ── Internal ─────────────────────────────────────────────────────────

    def _export_experiences(self) -> List[MemoryRecord]:
        """Convert experiences.jsonl to MemoryRecords."""
        records: List[MemoryRecord] = []
        exp_file = self._state_dir / "experiences.jsonl"
        if not exp_file.exists():
            return records

        for line_num, line in enumerate(exp_file.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                exp = json.loads(line)
                record = self._experience_to_record(exp)
                if record is not None:
                    records.append(record)
            except json.JSONDecodeError:
                logger.debug(f"Skipping malformed experience at line {line_num}")
        return records

    def _export_patterns(self) -> List[MemoryRecord]:
        """Convert patterns.json to MemoryRecords."""
        records: List[MemoryRecord] = []
        pat_file = self._state_dir / "patterns.json"
        if not pat_file.exists():
            return records

        try:
            patterns = json.loads(pat_file.read_text())
        except json.JSONDecodeError:
            logger.warning("patterns.json is corrupt — skipping")
            return records

        for pid, pattern in patterns.items():
            record = self._pattern_to_record(pid, pattern)
            if record is not None:
                records.append(record)
        return records

    def _export_strategies(self) -> List[MemoryRecord]:
        """Convert strategies.json to MemoryRecords."""
        records: List[MemoryRecord] = []
        strat_file = self._state_dir / "strategies.json"
        if not strat_file.exists():
            return records

        try:
            strategies = json.loads(strat_file.read_text())
        except json.JSONDecodeError:
            logger.warning("strategies.json is corrupt — skipping")
            return records

        for task_type, strategy in strategies.items():
            record = self._strategy_to_record(task_type, strategy)
            if record is not None:
                records.append(record)
        return records

    def _experience_to_record(self, exp: Dict[str, Any]) -> Optional[MemoryRecord]:
        """Convert a single experience dict to a MemoryRecord.

        Content is a natural-language summary suitable for embedding:
          "Task: edit | Approach: edit_py | Success: True | Score: 0.95"
        """
        try:
            task = exp.get("task", "unknown")
            approach = exp.get("approach", "default")
            outcome = exp.get("outcome", {})
            context = exp.get("context", {})
            ihsan = exp.get("ihsan_score", 0.5)
            ts = exp.get("timestamp", datetime.now(timezone.utc).isoformat())

            success = outcome.get("success", False)
            metrics = outcome.get("metrics", {})
            quality = metrics.get("quality_score", 0.5)

            # Build searchable content string
            content_parts = [
                f"Task: {task}",
                f"Approach: {approach}",
                f"Success: {success}",
                f"Quality: {quality:.2f}",
            ]
            for k, v in context.items():
                if k not in ("source",) and isinstance(v, (str, int, float, bool)):
                    content_parts.append(f"{k}: {v}")
            content = " | ".join(content_parts)

            return MemoryRecord(
                id=exp.get("id", f"rb_exp_{hash(content) & 0xFFFFFFFF:08x}"),
                content=content,
                kind=MemoryKind.EPISODIC,
                state=RecordState.ACTIVE,
                ihsan_score=ihsan,
                snr_score=quality,
                importance=ihsan * 0.7 + quality * 0.3,
                source="reasoning_bank",
                source_id=exp.get("id"),
                tags=["reasoning_bank", "experience", task, approach],
                created_at=datetime.fromisoformat(ts),
                updated_at=datetime.fromisoformat(ts),
                last_accessed=datetime.fromisoformat(ts),
                metadata={
                    "task": task,
                    "approach": approach,
                    "success": success,
                    "metrics": metrics,
                    "context": context,
                    "origin": "reasoning_bank_experience",
                },
            )
        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.warning(f"Failed to convert experience: {e}")
            return None

    def _pattern_to_record(
        self, pid: str, pattern: Dict[str, Any]
    ) -> Optional[MemoryRecord]:
        """Convert a learned pattern to a MemoryRecord."""
        try:
            triggers = pattern.get("triggers", [])
            actions = pattern.get("actions", [])
            confidence = pattern.get("confidence", 0.0)
            occurrences = pattern.get("occurrences", 0)

            content = (
                f"Pattern: {' + '.join(triggers)} → {' + '.join(actions)} | "
                f"Confidence: {confidence:.2f} | Occurrences: {occurrences}"
            )

            return MemoryRecord(
                id=f"rb_pat_{pid}",
                content=content,
                kind=MemoryKind.PROCEDURAL,
                state=RecordState.ACTIVE,
                ihsan_score=confidence,
                snr_score=confidence,
                importance=min(1.0, confidence * (1 + occurrences / 20)),
                source="reasoning_bank",
                source_id=pid,
                tags=["reasoning_bank", "pattern"] + triggers[:3] + actions[:3],
                metadata={
                    "triggers": triggers,
                    "actions": actions,
                    "confidence": confidence,
                    "occurrences": occurrences,
                    "origin": "reasoning_bank_pattern",
                },
            )
        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.warning(f"Failed to convert pattern {pid}: {e}")
            return None

    def _strategy_to_record(
        self, task_type: str, strategy: Dict[str, Any]
    ) -> Optional[MemoryRecord]:
        """Convert a distilled strategy to a MemoryRecord."""
        try:
            best_approach = strategy.get("best_approach", "default")
            score = strategy.get("score", 0.0)
            confidence = strategy.get("confidence", 0.0)
            success_rate = strategy.get("success_rate", 0.0)
            alternatives = strategy.get("alternatives", [])

            alt_names = [a.get("approach", "?") for a in alternatives[:3]]
            content = (
                f"Strategy for {task_type}: {best_approach} | "
                f"Score: {score:.2f} | Confidence: {confidence:.2f} | "
                f"Success rate: {success_rate:.2f}"
            )
            if alt_names:
                content += f" | Alternatives: {', '.join(alt_names)}"

            return MemoryRecord(
                id=f"rb_strat_{task_type}",
                content=content,
                kind=MemoryKind.PROCEDURAL,
                state=RecordState.ACTIVE,
                ihsan_score=score,
                snr_score=confidence,
                importance=score * confidence,
                source="reasoning_bank",
                source_id=task_type,
                tags=["reasoning_bank", "strategy", task_type, best_approach],
                metadata={
                    "task_type": task_type,
                    "best_approach": best_approach,
                    "score": score,
                    "confidence": confidence,
                    "success_rate": success_rate,
                    "alternatives": alternatives,
                    "origin": "reasoning_bank_strategy",
                },
            )
        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.warning(f"Failed to convert strategy {task_type}: {e}")
            return None
