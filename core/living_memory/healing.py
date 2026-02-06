"""
Memory Healer — Self-Repairing Knowledge System

Implements autonomous memory repair and optimization:
- Corruption detection and repair
- Consistency verification
- Quality improvement
- Knowledge deduplication

Standing on Giants: Autopoiesis (Self-Repair) + Shannon (Error Correction)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.living_memory.core import (
    LivingMemoryCore,
    MemoryState,
)

logger = logging.getLogger(__name__)


class CorruptionType(str):
    """Types of memory corruption."""

    ENCODING_ERROR = "encoding_error"
    EMBEDDING_MISMATCH = "embedding_mismatch"
    QUALITY_DEGRADATION = "quality_degradation"
    ORPHANED_REFERENCE = "orphaned_reference"
    INCONSISTENT_STATE = "inconsistent_state"
    DUPLICATE_CONTENT = "duplicate_content"


@dataclass
class CorruptionReport:
    """Report of detected corruption."""

    entry_id: str
    corruption_type: str
    severity: float  # 0-1
    description: str
    repairable: bool
    suggested_action: str


@dataclass
class RepairResult:
    """Result of a repair operation."""

    entry_id: str
    success: bool
    action_taken: str
    before_state: str
    after_state: str


@dataclass
class HealingStats:
    """Statistics from healing operations."""

    entries_scanned: int = 0
    corruptions_found: int = 0
    repairs_attempted: int = 0
    repairs_successful: int = 0
    entries_quarantined: int = 0
    entries_deleted: int = 0
    duplicates_merged: int = 0
    scan_duration_seconds: float = 0.0


class MemoryHealer:
    """
    Autonomous memory healing and optimization system.

    Continuously monitors memory health and performs:
    - Corruption detection
    - Automatic repair
    - Optimization passes
    - Garbage collection
    """

    def __init__(
        self,
        memory: LivingMemoryCore,
        llm_fn: Optional[Callable[[str], str]] = None,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        snr_threshold: float = UNIFIED_SNR_THRESHOLD,
    ):
        self.memory = memory
        self.llm_fn = llm_fn
        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold

        # Healing state
        self._last_full_scan: Optional[datetime] = None
        self._corruption_history: List[CorruptionReport] = []
        self._quarantine: Set[str] = set()

    async def scan_for_corruption(
        self,
        entry_ids: Optional[List[str]] = None,
    ) -> List[CorruptionReport]:
        """
        Scan memories for corruption.

        Checks:
        - Encoding validity
        - Embedding consistency
        - Quality thresholds
        - Reference integrity
        - State consistency
        """
        reports = []
        datetime.now(timezone.utc)

        entries_to_scan = entry_ids or list(self.memory._memories.keys())

        for entry_id in entries_to_scan:
            entry = self.memory._memories.get(entry_id)
            if not entry:
                continue

            # Check encoding
            try:
                entry.content.encode("utf-8")
            except UnicodeError:
                reports.append(
                    CorruptionReport(
                        entry_id=entry_id,
                        corruption_type=CorruptionType.ENCODING_ERROR,
                        severity=0.9,
                        description="Content contains invalid encoding",
                        repairable=False,
                        suggested_action="delete",
                    )
                )
                continue

            # Check quality degradation
            if entry.state == MemoryState.ACTIVE:
                if entry.ihsan_score < self.ihsan_threshold * 0.7:
                    reports.append(
                        CorruptionReport(
                            entry_id=entry_id,
                            corruption_type=CorruptionType.QUALITY_DEGRADATION,
                            severity=0.6,
                            description=f"Ihsān score degraded to {entry.ihsan_score:.3f}",
                            repairable=True,
                            suggested_action="revalidate",
                        )
                    )

                if entry.snr_score < self.snr_threshold * 0.7:
                    reports.append(
                        CorruptionReport(
                            entry_id=entry_id,
                            corruption_type=CorruptionType.QUALITY_DEGRADATION,
                            severity=0.5,
                            description=f"SNR score degraded to {entry.snr_score:.3f}",
                            repairable=True,
                            suggested_action="revalidate",
                        )
                    )

            # Check orphaned references
            for related_id in entry.related_ids:
                if related_id not in self.memory._memories:
                    reports.append(
                        CorruptionReport(
                            entry_id=entry_id,
                            corruption_type=CorruptionType.ORPHANED_REFERENCE,
                            severity=0.3,
                            description=f"Reference to non-existent entry: {related_id[:8]}",
                            repairable=True,
                            suggested_action="remove_reference",
                        )
                    )

            # Check state consistency
            if (
                entry.state == MemoryState.DELETED
                and entry_id in self.memory._type_index[entry.memory_type]
            ):
                reports.append(
                    CorruptionReport(
                        entry_id=entry_id,
                        corruption_type=CorruptionType.INCONSISTENT_STATE,
                        severity=0.4,
                        description="Deleted entry still in type index",
                        repairable=True,
                        suggested_action="cleanup_index",
                    )
                )

            # Check embedding consistency
            if entry.embedding is not None:
                if entry_id not in self.memory._embedding_index:
                    reports.append(
                        CorruptionReport(
                            entry_id=entry_id,
                            corruption_type=CorruptionType.EMBEDDING_MISMATCH,
                            severity=0.5,
                            description="Entry has embedding but not in index",
                            repairable=True,
                            suggested_action="rebuild_embedding_index",
                        )
                    )

        self._corruption_history.extend(reports)
        self._last_full_scan = datetime.now(timezone.utc)

        return reports

    async def repair(
        self,
        report: CorruptionReport,
    ) -> RepairResult:
        """
        Attempt to repair a detected corruption.
        """
        entry = self.memory._memories.get(report.entry_id)
        if not entry:
            return RepairResult(
                entry_id=report.entry_id,
                success=False,
                action_taken="none",
                before_state="missing",
                after_state="missing",
            )

        before_state = entry.state.value

        if report.suggested_action == "delete":
            await self.memory.forget(report.entry_id, hard_delete=True)
            return RepairResult(
                entry_id=report.entry_id,
                success=True,
                action_taken="deleted",
                before_state=before_state,
                after_state="deleted",
            )

        elif report.suggested_action == "revalidate":
            # Recompute quality scores
            ihsan, snr = await self.memory._compute_quality(entry.content)
            entry.ihsan_score = ihsan
            entry.snr_score = snr

            if ihsan >= self.ihsan_threshold * 0.8:
                entry.state = MemoryState.ACTIVE
                return RepairResult(
                    entry_id=report.entry_id,
                    success=True,
                    action_taken="revalidated",
                    before_state=before_state,
                    after_state=entry.state.value,
                )
            else:
                entry.state = MemoryState.CORRUPTED
                self._quarantine.add(report.entry_id)
                return RepairResult(
                    entry_id=report.entry_id,
                    success=False,
                    action_taken="quarantined",
                    before_state=before_state,
                    after_state="quarantined",
                )

        elif report.suggested_action == "remove_reference":
            # Remove orphaned references
            entry.related_ids = {
                rid for rid in entry.related_ids if rid in self.memory._memories
            }
            return RepairResult(
                entry_id=report.entry_id,
                success=True,
                action_taken="removed_orphaned_references",
                before_state=before_state,
                after_state=entry.state.value,
            )

        elif report.suggested_action == "cleanup_index":
            # Fix index inconsistency
            if entry.state == MemoryState.DELETED:
                self.memory._type_index[entry.memory_type].discard(report.entry_id)
            return RepairResult(
                entry_id=report.entry_id,
                success=True,
                action_taken="cleaned_index",
                before_state=before_state,
                after_state=entry.state.value,
            )

        elif report.suggested_action == "rebuild_embedding_index":
            # Add missing embedding to index
            if entry.embedding is not None:
                self.memory._embedding_index[report.entry_id] = entry.embedding
            return RepairResult(
                entry_id=report.entry_id,
                success=True,
                action_taken="rebuilt_embedding_index",
                before_state=before_state,
                after_state=entry.state.value,
            )

        return RepairResult(
            entry_id=report.entry_id,
            success=False,
            action_taken="unknown",
            before_state=before_state,
            after_state=entry.state.value,
        )

    async def find_duplicates(
        self,
        similarity_threshold: float = 0.95,
    ) -> List[Tuple[str, str, float]]:
        """
        Find duplicate or near-duplicate memories.

        Returns list of (entry1_id, entry2_id, similarity) tuples.
        """
        duplicates = []

        # Get all entries with embeddings
        entries_with_embeddings = [
            (eid, entry)
            for eid, entry in self.memory._memories.items()
            if entry.embedding is not None and entry.state == MemoryState.ACTIVE
        ]

        # O(n^2) comparison - could be optimized with LSH
        for i, (id1, entry1) in enumerate(entries_with_embeddings):
            for j, (id2, entry2) in enumerate(entries_with_embeddings[i + 1 :], i + 1):
                # Compute cosine similarity
                sim = np.dot(entry1.embedding, entry2.embedding) / (
                    np.linalg.norm(entry1.embedding) * np.linalg.norm(entry2.embedding)
                    + 1e-10
                )
                if sim >= similarity_threshold:
                    duplicates.append((id1, id2, float(sim)))

        return duplicates

    async def merge_duplicates(
        self,
        keep_id: str,
        merge_id: str,
    ) -> bool:
        """
        Merge two duplicate entries, keeping the higher quality one.
        """
        keep = self.memory._memories.get(keep_id)
        merge = self.memory._memories.get(merge_id)

        if not keep or not merge:
            return False

        # Merge metadata
        keep.access_count += merge.access_count
        keep.related_ids.update(merge.related_ids)
        keep.related_ids.discard(merge_id)  # Don't relate to self

        # Keep better quality scores
        if merge.ihsan_score > keep.ihsan_score:
            keep.ihsan_score = merge.ihsan_score
        if merge.snr_score > keep.snr_score:
            keep.snr_score = merge.snr_score

        # Delete merged entry
        await self.memory.forget(merge_id, hard_delete=True)

        return True

    async def optimize(self) -> HealingStats:
        """
        Run full optimization pass.

        Combines:
        - Corruption scan and repair
        - Duplicate merging
        - Garbage collection
        """
        start_time = datetime.now(timezone.utc)
        stats = HealingStats()

        # 1. Scan for corruption
        reports = await self.scan_for_corruption()
        stats.entries_scanned = len(self.memory._memories)
        stats.corruptions_found = len(reports)

        # 2. Repair corruptions
        for report in reports:
            if report.repairable:
                result = await self.repair(report)
                stats.repairs_attempted += 1
                if result.success:
                    stats.repairs_successful += 1
                elif result.after_state == "quarantined":
                    stats.entries_quarantined += 1
            else:
                await self.memory.forget(report.entry_id, hard_delete=True)
                stats.entries_deleted += 1

        # 3. Find and merge duplicates
        duplicates = await self.find_duplicates()
        for id1, id2, sim in duplicates:
            entry1 = self.memory._memories.get(id1)
            entry2 = self.memory._memories.get(id2)

            if entry1 and entry2:
                # Keep the one with higher Ihsān score
                if entry1.ihsan_score >= entry2.ihsan_score:
                    await self.merge_duplicates(id1, id2)
                else:
                    await self.merge_duplicates(id2, id1)
                stats.duplicates_merged += 1

        # 4. Save after optimization
        await self.memory._save_memories()

        stats.scan_duration_seconds = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()

        logger.info(
            f"Healing complete: scanned={stats.entries_scanned}, "
            f"corruptions={stats.corruptions_found}, "
            f"repaired={stats.repairs_successful}, "
            f"duplicates_merged={stats.duplicates_merged}"
        )

        return stats

    def get_health_report(self) -> Dict[str, Any]:
        """Get overall memory health report."""
        memory_stats = self.memory.get_stats()

        health_score = 1.0

        # Penalize for corrupted entries
        if memory_stats.total_entries > 0:
            corruption_ratio = (
                memory_stats.corrupted_entries / memory_stats.total_entries
            )
            health_score -= corruption_ratio * 0.5

        # Penalize for low average quality
        if memory_stats.avg_ihsan < self.ihsan_threshold:
            health_score -= 0.2

        if memory_stats.avg_snr < self.snr_threshold:
            health_score -= 0.1

        # Penalize for many quarantined entries
        quarantine_ratio = len(self._quarantine) / max(memory_stats.total_entries, 1)
        health_score -= quarantine_ratio * 0.2

        return {
            "health_score": max(0, min(1, health_score)),
            "total_entries": memory_stats.total_entries,
            "active_entries": memory_stats.active_entries,
            "corrupted_entries": memory_stats.corrupted_entries,
            "quarantined_entries": len(self._quarantine),
            "avg_ihsan": memory_stats.avg_ihsan,
            "avg_snr": memory_stats.avg_snr,
            "last_full_scan": (
                self._last_full_scan.isoformat() if self._last_full_scan else None
            ),
            "recent_corruptions": len([r for r in self._corruption_history[-100:]]),
        }
