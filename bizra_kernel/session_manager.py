"""
Session Manager — Protocol-Hashed Session Tracking
===================================================
From the Blueprint:
  Every session_start logs protocol_version + SHA256(protocol_config)
  to the Layer 1 BlockGraph as a PoI transaction.

This creates cryptographically verifiable traceability across the APEX stack.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

from .ihsan_vector import IhsanVector


def canonical_bytes(obj) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


class SessionState(Enum):
    """Session lifecycle states."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"  # Escalated to FATE
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SessionEvent:
    """An event within a session (for audit trail)."""
    event_type: str
    timestamp: str
    data: Dict[str, Any]
    ihsan_score: Optional[float] = None


@dataclass
class Session:
    """
    A protocol-hashed session with full audit trail.
    
    Each session is a unit of work with:
    - Unique ID
    - Protocol hash (for version verification)
    - Agent assignments
    - Ihsān vector tracking
    - Event log for auditability
    """
    
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    protocol_version: str = "2.0.0"
    protocol_hash: str = ""
    state: SessionState = SessionState.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Agent context
    primary_agent: str = ""
    supporting_agents: List[str] = field(default_factory=list)
    
    # User context
    user_id: str = "anonymous"
    user_message: str = ""
    
    # Ihsān tracking
    ihsan_vectors: List[IhsanVector] = field(default_factory=list)
    
    # Event log
    events: List[SessionEvent] = field(default_factory=list)
    
    # Metrics
    total_tokens: int = 0
    useful_tokens: int = 0
    total_latency_ms: int = 0
    
    def __post_init__(self):
        if not self.protocol_hash:
            self.protocol_hash = self._compute_protocol_hash()
    
    def _compute_protocol_hash(self) -> str:
        """Compute SHA256 hash of protocol configuration."""
        config = {
            "version": self.protocol_version,
            "ihsan_threshold": 0.95,
            "sape_elevation_threshold": 3,
            "snr_target": 0.90,
            "fate_circuit_breaker_ms": 100,
        }
        canonical = canonical_bytes(config)
        return hashlib.sha256(canonical).hexdigest()

    def to_poi_receipt(self) -> str:
        # 1. Construct the Full Fact
        payload = {
            "type": "session_receipt", # Using generic type as event_type is per-event
            "timestamp_utc": self.created_at, # Use creation time or updated? User snippet implies per-event but this is Session method.
            # Adapting user snippet to Session context: "data_hash" isn't a Session field. 
            # User snippet: "type": self.event_type. 
            # But the method is on Session class. 
            # I will assume this generates a receipt for the Session itself.
            "session_id": self.session_id,
            "protocol_hash": self.protocol_hash,
            "protocol_version": self.protocol_version,
            # "ihsan_score": self.ihsan_vectors[-1].score if self.ihsan_vectors else 0.0, # Example adaptation
            # But strictly following the snippet architecture:
            "agent_id": self.primary_agent
        }

        # 2. Canonical Serialization (RFC 8785 style)
        # sort_keys=True ensures {a:1, b:2} == {b:2, a:1}
        # separators removes whitespace noise
        canonical_bytes = json.dumps(
            payload, 
            sort_keys=True, 
            separators=(',', ':')
        ).encode('utf-8')

        # 3. The Immutable Seal (Full SHA-256)
        # Truncation is forbidden.
        return hashlib.sha256(canonical_bytes).hexdigest()
    
    def start(self, agent: str, message: str) -> "Session":
        """Start the session."""
        self.state = SessionState.ACTIVE
        self.primary_agent = agent
        self.user_message = message
        self.updated_at = datetime.utcnow().isoformat()
        self.log_event("session_start", {
            "agent": agent,
            "message_length": len(message),
            "protocol_hash": self.protocol_hash,
        })
        return self
    
    def pause_for_fate(self, reason: str) -> "Session":
        """Pause session and escalate to FATE protocol."""
        self.state = SessionState.PAUSED
        self.updated_at = datetime.utcnow().isoformat()
        self.log_event("fate_escalation", {
            "reason": reason,
            "current_ihsan": self.current_ihsan_score,
        })
        return self
    
    def complete(self, response: str, ihsan_vec: IhsanVector) -> "Session":
        """Complete the session successfully."""
        self.state = SessionState.COMPLETED
        self.updated_at = datetime.utcnow().isoformat()
        self.ihsan_vectors.append(ihsan_vec)
        self.log_event("session_complete", {
            "response_length": len(response),
            "ihsan_score": ihsan_vec.composite_score,
            "passes_threshold": ihsan_vec.passes_threshold,
        })
        return self
    
    def fail(self, error: str) -> "Session":
        """Mark session as failed."""
        self.state = SessionState.FAILED
        self.updated_at = datetime.utcnow().isoformat()
        self.log_event("session_failed", {"error": error})
        return self
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event to the session audit trail."""
        self.events.append(SessionEvent(
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            data=data,
            ihsan_score=self.current_ihsan_score,
        ))
    
    @property
    def current_ihsan_score(self) -> float:
        """Get the most recent Ihsān score."""
        if not self.ihsan_vectors:
            return 1.0  # Perfect until proven otherwise
        return self.ihsan_vectors[-1].composite_score
    
    @property
    def snr_score(self) -> float:
        """Calculate Signal-to-Noise Ratio for this session."""
        if self.total_tokens == 0:
            return 0.0
        return self.useful_tokens / self.total_tokens
    
    def to_dict(self) -> dict:
        """Serialize session to dictionary."""
        return {
            "session_id": self.session_id,
            "protocol_version": self.protocol_version,
            "protocol_hash": self.protocol_hash,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "primary_agent": self.primary_agent,
            "supporting_agents": self.supporting_agents,
            "user_id": self.user_id,
            "ihsan_score": self.current_ihsan_score,
            "snr_score": self.snr_score,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "event_count": len(self.events),
        }
    
    def to_poi_receipt(self) -> dict:
        """Generate PoI receipt for Layer 1 BlockGraph."""
        return {
            "receipt_type": "session_completion",
            "session_id": self.session_id,
            "protocol_hash": self.protocol_hash,
            "ihsan_score": self.current_ihsan_score,
            "snr_score": self.snr_score,
            "agent": self.primary_agent,
            "timestamp": self.updated_at,
            "events_hash": hashlib.sha256(
                json.dumps([e.event_type for e in self.events]).encode()
            ).hexdigest(),
        }


class SessionManager:
    """
    Manages sessions with protocol hashing and persistence.
    
    Acts as the Session OS layer described in the Blueprint.
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.sessions: Dict[str, Session] = {}
        self.storage_dir = storage_dir or Path("./sessions")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.protocol_version = "2.0.0"
    
    def create_session(
        self,
        agent: str,
        message: str,
        user_id: str = "anonymous",
    ) -> Session:
        """Create and start a new session."""
        session = Session(
            protocol_version=self.protocol_version,
            user_id=user_id,
        )
        session.start(agent, message)
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID."""
        return self.sessions.get(session_id)
    
    def complete_session(
        self,
        session_id: str,
        response: str,
        ihsan_vec: IhsanVector,
    ) -> Optional[Session]:
        """Complete a session with response and Ihsān vector."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        # Check if Ihsān threshold is met
        if not ihsan_vec.passes_threshold:
            session.pause_for_fate(
                f"Ihsān score {ihsan_vec.composite_score:.3f} < threshold 0.95"
            )
            return session
        
        session.complete(response, ihsan_vec)
        self._persist_session(session)
        return session
    
    def _persist_session(self, session: Session) -> None:
        """Persist session to storage for audit trail."""
        filepath = self.storage_dir / f"{session.session_id}.json"
        with open(filepath, "w") as f:
            json.dump(session.to_dict(), f, indent=2)
    
    def get_statistics(self) -> dict:
        """Get aggregate statistics across all sessions."""
        completed = [s for s in self.sessions.values() if s.state == SessionState.COMPLETED]
        failed = [s for s in self.sessions.values() if s.state == SessionState.FAILED]
        paused = [s for s in self.sessions.values() if s.state == SessionState.PAUSED]
        
        avg_ihsan = (
            sum(s.current_ihsan_score for s in completed) / len(completed)
            if completed else 0.0
        )
        avg_snr = (
            sum(s.snr_score for s in completed) / len(completed)
            if completed else 0.0
        )
        
        return {
            "total_sessions": len(self.sessions),
            "completed": len(completed),
            "failed": len(failed),
            "paused_for_fate": len(paused),
            "avg_ihsan_score": avg_ihsan,
            "avg_snr_score": avg_snr,
            "protocol_version": self.protocol_version,
        }
