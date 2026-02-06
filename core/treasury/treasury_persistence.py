"""
Treasury Persistence — SQLite-based State Storage
=================================================
Handles persistence of treasury state and transition history.

Standing on Giants: Lamport (1982) - State machine replication
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .treasury_types import (
    DEFAULT_BURN_RATE,
    EthicsAssessment,
    TransitionEvent,
    TransitionTrigger,
    TreasuryMode,
    TreasuryState,
)

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path("/mnt/c/BIZRA-DATA-LAKE/.swarm/memory.db")


class TreasuryPersistence:
    """
    SQLite-based persistence for treasury state and transitions.

    Stores state in .swarm/memory.db for consistency with other BIZRA modules.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Treasury state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS treasury_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    mode TEXT NOT NULL,
                    reserves_days REAL NOT NULL,
                    ethical_score REAL NOT NULL,
                    last_transition TEXT NOT NULL,
                    transition_reason TEXT NOT NULL,
                    burn_rate_seed_per_day REAL DEFAULT 100.0,
                    total_reserves_seed REAL DEFAULT 0.0,
                    locked_treasury_seed REAL DEFAULT 0.0,
                    unlocked_treasury_seed REAL DEFAULT 0.0,
                    updated_at TEXT NOT NULL
                )
            """)

            # Transition history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS treasury_transitions (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    from_mode TEXT NOT NULL,
                    to_mode TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    ethical_score_at_transition REAL,
                    reserves_days_at_transition REAL,
                    reason TEXT,
                    metadata TEXT
                )
            """)

            # Ethics assessments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS treasury_ethics_assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    transparency_score REAL,
                    fairness_score REAL,
                    sustainability_score REAL,
                    compliance_score REAL,
                    ihsan_alignment REAL,
                    confidence REAL,
                    data_sources TEXT
                )
            """)

            conn.commit()

    def save_state(self, state: TreasuryState) -> None:
        """Save current treasury state."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO treasury_state
                (id, mode, reserves_days, ethical_score, last_transition,
                 transition_reason, burn_rate_seed_per_day, total_reserves_seed,
                 locked_treasury_seed, unlocked_treasury_seed, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    state.mode.value,
                    state.reserves_days,
                    state.ethical_score,
                    state.last_transition.isoformat(),
                    state.transition_reason,
                    state.burn_rate_seed_per_day,
                    state.total_reserves_seed,
                    state.locked_treasury_seed,
                    state.unlocked_treasury_seed,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def load_state(self) -> Optional[TreasuryState]:
        """Load current treasury state."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT mode, reserves_days, ethical_score, last_transition,
                       transition_reason, burn_rate_seed_per_day, total_reserves_seed,
                       locked_treasury_seed, unlocked_treasury_seed
                FROM treasury_state WHERE id = 1
            """)
            row = cursor.fetchone()

            if row:
                return TreasuryState(
                    mode=TreasuryMode(row[0]),
                    reserves_days=row[1],
                    ethical_score=row[2],
                    last_transition=datetime.fromisoformat(row[3]),
                    transition_reason=row[4],
                    burn_rate_seed_per_day=row[5] or DEFAULT_BURN_RATE,
                    total_reserves_seed=row[6] or 0.0,
                    locked_treasury_seed=row[7] or 0.0,
                    unlocked_treasury_seed=row[8] or 0.0,
                )
            return None

    def record_transition(self, event: TransitionEvent) -> None:
        """Record a transition event."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO treasury_transitions
                (event_id, timestamp, from_mode, to_mode, trigger,
                 ethical_score_at_transition, reserves_days_at_transition,
                 reason, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.from_mode.value,
                    event.to_mode.value,
                    event.trigger.value,
                    event.ethical_score_at_transition,
                    event.reserves_days_at_transition,
                    event.reason,
                    json.dumps(event.metadata),
                ),
            )
            conn.commit()

    def record_ethics_assessment(self, assessment: EthicsAssessment) -> None:
        """Record an ethics assessment."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO treasury_ethics_assessments
                (timestamp, overall_score, transparency_score, fairness_score,
                 sustainability_score, compliance_score, ihsan_alignment,
                 confidence, data_sources)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    assessment.assessment_timestamp.isoformat(),
                    assessment.overall_score,
                    assessment.transparency_score,
                    assessment.fairness_score,
                    assessment.sustainability_score,
                    assessment.compliance_score,
                    assessment.ihsan_alignment,
                    assessment.confidence,
                    json.dumps(assessment.data_sources),
                ),
            )
            conn.commit()

    def get_transition_history(self, limit: int = 100) -> List[TransitionEvent]:
        """Get recent transition history."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT event_id, timestamp, from_mode, to_mode, trigger,
                       ethical_score_at_transition, reserves_days_at_transition,
                       reason, metadata
                FROM treasury_transitions
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )

            events = []
            for row in cursor.fetchall():
                events.append(
                    TransitionEvent(
                        event_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        from_mode=TreasuryMode(row[2]),
                        to_mode=TreasuryMode(row[3]),
                        trigger=TransitionTrigger(row[4]),
                        ethical_score_at_transition=row[5] or 0.0,
                        reserves_days_at_transition=row[6] or 0.0,
                        reason=row[7] or "",
                        metadata=json.loads(row[8]) if row[8] else {},
                    )
                )
            return events


__all__ = [
    "TreasuryPersistence",
    "DEFAULT_DB_PATH",
]
