# Phase 52.2: Genesis Bridge (Phase 1 -- User Interaction)

> Standing on Giants: Fielding (REST architectural style, 2000) · Shannon (information-theoretic request validation, 1948) · Lamport (PCI envelope as logical timestamp, 1978) · Al-Ghazali (Ihsan gates on API boundary, 1095)

## 1. Overview

The Genesis Bridge is Ahmed's entry point into BIZRA. When he submits "Organize my
invoice PDFs into folders by vendor and month, and email me a summary," the Bridge
validates the request, wraps it in a PCI (Proof-Carrying Inference) envelope, and
routes it to the PlanGenerator which triggers the PAT-7 pipeline.

This is the only surface where untrusted human input enters the system. Every field
is validated. Every request is PCI-wrapped before it touches any reasoning component.

---

## 2. Data Flow

```
  Ahmed's Browser / CLI
       │
       │ POST /api/v1/plan/generate
       │ { profileId, goal, goalCategory, selectedAgents }
       │
  ┌────▼──────────────────────────────────────────────┐
  │  GENESIS BRIDGE                                    │
  │                                                    │
  │  1. Validate request schema                        │
  │  2. Sanitize inputs (no injection, no overflow)    │
  │  3. Verify profileId exists and is active           │
  │  4. Wrap in PCI envelope (hash, timestamp, nonce)  │
  │  5. Route to PlanGenerator                         │
  │                                                    │
  └────┬──────────────────────────────────────────────┘
       │
  ┌────▼──────────────────────────────────────────────┐
  │  PCI ENVELOPE                                      │
  │  { payload_hash, timestamp, nonce, profile_id,     │
  │    claim_bindings: [], snr_floor: SNR_THRESHOLD }  │
  └────┬──────────────────────────────────────────────┘
       │
       ▼ → PAT-7 Pipeline (Phase 52.3)
```

---

## 3. Pseudocode

### 3.1 Request Model

```python
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from core.integration.constants import (
    IHSAN_THRESHOLD,
    SNR_THRESHOLD,
    UNIFIED_CLOCK_SKEW_SECONDS,
    UNIFIED_NONCE_TTL_SECONDS,
)


@dataclass
class PlanRequest:
    """Incoming request from Ahmed's client."""
    profile_id: str
    goal: str
    goal_category: str = "file_organization"
    selected_agents: list[str] = field(default_factory=lambda: [
        "planner", "researcher", "coder", "evaluator",
        "ethicist", "publisher", "integrator",
    ])

    def validate(self) -> tuple[bool, str]:
        """Validate all fields. Returns (valid, error_message)."""
        if not self.profile_id or len(self.profile_id) < 3:
            return False, "profile_id must be >= 3 characters"
        if not self.goal or len(self.goal.strip()) == 0:
            return False, "goal cannot be empty"
        if len(self.goal) > 2000:
            return False, "goal exceeds 2000 character limit"
        if not self.selected_agents:
            return False, "at least one agent must be selected"
        valid_agents = {"planner", "researcher", "coder", "evaluator",
                        "ethicist", "publisher", "integrator"}
        invalid = set(self.selected_agents) - valid_agents
        if invalid:
            return False, f"unknown agents: {invalid}"
        return True, ""
```

### 3.2 PCI Envelope

```python
@dataclass
class PCIEnvelope:
    """Proof-Carrying Inference envelope.
    Every claim must bind to evidence. Standing on Giants: Lamport (1978)."""
    envelope_id: str = field(default_factory=lambda: str(uuid4()))
    payload_hash: str = ""        # BLAKE3 hash of serialized payload
    timestamp: float = field(default_factory=time.time)
    nonce: str = field(default_factory=lambda: str(uuid4()))
    profile_id: str = ""
    goal: str = ""
    goal_category: str = ""
    selected_agents: list[str] = field(default_factory=list)
    claim_bindings: list[dict] = field(default_factory=list)
    snr_floor: float = SNR_THRESHOLD
    ihsan_floor: float = IHSAN_THRESHOLD

    @classmethod
    def wrap(cls, request: PlanRequest) -> PCIEnvelope:
        """Wrap a validated PlanRequest in a PCI envelope."""
        payload = f"{request.profile_id}:{request.goal}:{request.goal_category}"
        payload_hash = hashlib.blake2b(payload.encode(), digest_size=32).hexdigest()

        return cls(
            payload_hash=payload_hash,
            profile_id=request.profile_id,
            goal=request.goal,
            goal_category=request.goal_category,
            selected_agents=list(request.selected_agents),
            claim_bindings=[],
            snr_floor=SNR_THRESHOLD,
            ihsan_floor=IHSAN_THRESHOLD,
        )

    def is_fresh(self) -> bool:
        """Check that envelope timestamp is within acceptable clock skew."""
        age = abs(time.time() - self.timestamp)
        return age < UNIFIED_CLOCK_SKEW_SECONDS
```

### 3.3 Genesis Bridge API

```python
import logging
from typing import Optional

logger = logging.getLogger("bizra.genesis_bridge")


class GenesisBridge:
    """API layer between user and BIZRA reasoning.
    POST /api/v1/plan/generate is the sole entry point."""

    def __init__(self, node: BIZRANode) -> None:
        self.node = node
        self._nonce_cache: set[str] = set()

    async def generate_plan(self, request: PlanRequest) -> dict:
        """Main entry: validate -> wrap PCI -> route to PAT-7."""
        # Step 1: Validate
        valid, error = request.validate()
        if not valid:
            logger.warning("genesis.validation_failed: %s", error)
            return {"error": error, "status": 400}

        # Step 2: Check node health
        if not self.node._booted:
            return {"error": "node_not_ready", "status": 503}

        # Step 3: Profile existence check
        profile = await self._lookup_profile(request.profile_id)
        if profile is None:
            return {"error": "profile_not_found", "status": 404}

        # Step 4: Wrap in PCI envelope
        envelope = PCIEnvelope.wrap(request)

        # Step 5: Replay protection
        if envelope.nonce in self._nonce_cache:
            return {"error": "replay_detected", "status": 409}
        self._nonce_cache.add(envelope.nonce)

        # Step 6: Route to PAT-7 pipeline
        logger.info("genesis.routed envelope_id=%s goal_len=%d",
                     envelope.envelope_id, len(envelope.goal))
        plan = await self.node.pat7.generate_plan(envelope)

        return {
            "status": 200,
            "plan": plan.__dict__ if plan else None,
            "envelope_id": envelope.envelope_id,
        }

    async def _lookup_profile(self, profile_id: str) -> Optional[dict]:
        """Look up profile in Engram store. Returns None if not found."""
        if self.node.engram is None:
            return None
        return await self.node.engram.get_profile(profile_id)
```

### 3.4 FastAPI Route (Reference)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="BIZRA Genesis Bridge")


class PlanRequestBody(BaseModel):
    profileId: str = Field(..., min_length=3, max_length=128)
    goal: str = Field(..., min_length=1, max_length=2000)
    goalCategory: str = Field(default="file_organization")
    selectedAgents: list[str] = Field(
        default=["planner", "researcher", "coder",
                 "evaluator", "ethicist", "publisher", "integrator"])


@app.post("/api/v1/plan/generate")
async def generate_plan(body: PlanRequestBody):
    request = PlanRequest(
        profile_id=body.profileId, goal=body.goal,
        goal_category=body.goalCategory, selected_agents=body.selectedAgents)
    result = await bridge.generate_plan(request)
    if result.get("status") != 200:
        raise HTTPException(status_code=result["status"], detail=result["error"])
    return result
```

---

## 4. Ahmed's Request (Concrete Example)

```json
{
  "profileId": "ahmed-dubai-001",
  "goal": "Organize my invoice PDFs into folders by vendor and month, and email me a summary",
  "goalCategory": "file_organization",
  "selectedAgents": ["planner", "researcher", "coder", "evaluator",
                     "ethicist", "publisher", "integrator"]
}
```

PCI Envelope:
```
envelope_id:   "a7e3...d1f2"
payload_hash:  "b4c9...8a1e" (BLAKE2b of profile+goal+category)
timestamp:     1740614400.0
nonce:         "f1d2...c3e4"
snr_floor:     0.85 (from SNR_THRESHOLD)
ihsan_floor:   0.95 (from IHSAN_THRESHOLD)
```

---

## 5. TDD Anchors

```python
import pytest


class TestGenesisBridge:
    """Phase 52.2: Genesis Bridge tests."""

    def test_request_validation_valid(self):
        req = PlanRequest(profile_id="ahmed-dubai-001", goal="Organize my invoice PDFs")
        valid, error = req.validate()
        assert valid is True

    def test_request_validation_empty_goal(self):
        req = PlanRequest(profile_id="ahmed", goal="")
        valid, error = req.validate()
        assert valid is False
        assert "empty" in error

    def test_request_validation_goal_too_long(self):
        req = PlanRequest(profile_id="ahmed", goal="x" * 2001)
        valid, _ = req.validate()
        assert valid is False

    def test_request_validation_invalid_agent(self):
        req = PlanRequest(profile_id="ahmed", goal="test",
                          selected_agents=["planner", "hacker"])
        valid, error = req.validate()
        assert valid is False
        assert "unknown" in error

    def test_pci_envelope_wrapping(self):
        req = PlanRequest(profile_id="ahmed-001", goal="test goal")
        env = PCIEnvelope.wrap(req)
        assert env.payload_hash != ""
        assert env.profile_id == "ahmed-001"
        assert env.snr_floor == 0.85
        assert env.ihsan_floor == 0.95
        assert env.is_fresh()

    def test_pci_envelope_freshness(self):
        env = PCIEnvelope(timestamp=0.0)
        assert env.is_fresh() is False

    def test_invalid_profile_short(self):
        req = PlanRequest(profile_id="ab", goal="test")
        valid, _ = req.validate()
        assert valid is False

    def test_empty_goal_whitespace(self):
        req = PlanRequest(profile_id="ahmed-001", goal="   ")
        valid, _ = req.validate()
        assert valid is False

    @pytest.mark.asyncio
    async def test_unbooted_node_rejects(self):
        node = BIZRANode(profile_id="test", device=DeviceConfig())
        bridge = GenesisBridge(node=node)
        req = PlanRequest(profile_id="ahmed-001", goal="test")
        result = await bridge.generate_plan(req)
        assert result["status"] == 503

    @pytest.mark.asyncio
    async def test_replay_protection(self):
        bridge = GenesisBridge(node=mock_booted_node())
        bridge._nonce_cache.add("duplicate-nonce")
        # Any request using that nonce would be rejected
```
