"""
Pattern Memory Adapter — Optional bridge from Rust autopoiesis PatternMemory.

Requires the PyO3 bindings (`bizra_python`) to be built via maturin.
If not available, this module provides a no-op stub.

The Rust PatternMemory (bizra-autopoiesis crate) tracks learned patterns
with confidence scoring and temporal decay. This adapter exports them
as MemoryRecords with kind=PROCEDURAL and source="pattern_memory".
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from core.memory.types import MemoryKind, MemoryRecord, RecordState

logger = logging.getLogger(__name__)

# Try importing the PyO3 bindings
try:
    from bizra_python import PatternMemory as RustPatternMemory  # noqa: F401

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    logger.debug("Rust PatternMemory bindings not available — using stub")


class PatternMemoryAdapter:
    """Adapter for Rust autopoiesis PatternMemory.

    If Rust bindings are not available, all methods return empty results
    (graceful degradation).
    """

    def __init__(self, pattern_memory=None) -> None:
        self._pm = pattern_memory
        self._available = _HAS_RUST and pattern_memory is not None

    @property
    def available(self) -> bool:
        return self._available

    def export_all(self) -> List[MemoryRecord]:
        """Export all learned patterns as MemoryRecords."""
        if not self._available:
            return []

        records = []
        try:
            patterns = self._pm.list_patterns()  # type: ignore[union-attr]
            for pattern in patterns:
                record = self._pattern_to_record(pattern)
                if record is not None:
                    records.append(record)
            logger.info(f"PatternMemoryAdapter: exported {len(records)} patterns")
        except Exception as e:
            logger.warning(f"PatternMemory export failed: {e}")

        return records

    def _pattern_to_record(self, pattern) -> Optional[MemoryRecord]:
        """Convert a Rust Pattern to a MemoryRecord."""
        try:
            name = getattr(pattern, "name", "unknown_pattern")
            description = getattr(pattern, "description", "")
            confidence = getattr(pattern, "confidence", 0.5)
            pattern_type = getattr(pattern, "pattern_type", "learning")

            content = f"Pattern: {name}\n{description}"
            pattern_id = getattr(pattern, "id", name)

            return MemoryRecord(
                id=f"pat_{pattern_id[:12]}",
                content=content,
                kind=MemoryKind.PROCEDURAL,
                state=RecordState.ACTIVE,
                importance=confidence,
                source="pattern_memory",
                source_id=str(pattern_id),
                tags=["pattern", pattern_type],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
                metadata={
                    "confidence": confidence,
                    "pattern_type": pattern_type,
                    "origin": "pattern_memory",
                },
            )
        except Exception as e:
            logger.warning(f"Failed to convert pattern: {e}")
            return None
