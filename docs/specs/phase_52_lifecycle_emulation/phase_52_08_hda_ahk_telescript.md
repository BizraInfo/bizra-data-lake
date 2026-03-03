# Phase 52.8: HDA + AHK + Telescript Integration (Phase 7)

> Standing on Giants: General Magic (Telescript permits, places, mobile agents, 1994) · Fitts (target acquisition law, 1954) · Norman (affordance and feedback, 1988) · Boyd (OODA perception-action loop, 1976) · Shannon (entropy as action selection, 1948) · Al-Ghazali (permit as ethical gate, 1095)

## 1. Overview

The HDA (Human Desktop Automation) layer is the "body" of Ahmed's node. While PAT-7
is the "brain" (reasoning, planning, evaluating), HDA is where digital thought becomes
physical action: moving files, creating folders, composing emails, reading PDFs.

This specification details the brain/body split, AHK bridge protocol, 8 HDA verbs,
permit system, Ghost Overlay proactive suggestions, and perception-action verification.

Cross-reference: [Phase 48](../phase_48_ahk_hda_desktop_automation.md) for backend.

---

## 2. Data Flow

```
  ┌──────────────────────────────────────────────┐
  │  BRAIN (PAT-7 + TTRL)                        │
  │  PlanResponse with TelescriptActions          │
  └──────────────────┬───────────────────────────┘
                     │ Telescript envelope
  ┌──────────────────▼───────────────────────────┐
  │  SPINAL CORD (ActionExecutor)                 │
  │  1. Validate permit (capability + budget)     │
  │  2. FATE gate (Ihsan + SNR + ADL)             │
  │  3. Envelope → AHK bridge command             │
  └──────────────────┬───────────────────────────┘
                     │ JSON-RPC over TCP:9742
  ┌──────────────────▼───────────────────────────┐
  │  BODY (AHK Bridge, 1018 LOC)                  │
  │  8 HDA Verbs:                                 │
  │  open_app | switch_win | type_text | click    │
  │  screenshot | clipboard | file_open | browser │
  └──────────────────┬───────────────────────────┘
                     │ result + post_hash
  ┌──────────────────▼───────────────────────────┐
  │  SENSES (Receipt Pipeline)                    │
  │  pre_hash vs post_hash → verification         │
  │  receipt → chain → Engram reinforcement       │
  └──────────────────────────────────────────────┘
```

---

## 3. Pseudocode

### 3.1 HDA Verbs and Brain/Body Architecture

```python
from __future__ import annotations
import asyncio, json, logging
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4
from core.integration.constants import IHSAN_THRESHOLD, SNR_THRESHOLD, TIMESCALE_T1_CYCLE_MS

logger = logging.getLogger("bizra.hda")


@dataclass(frozen=True)
class HDAVerb:
    name: str
    description: str
    capabilities_required: list[str]
    is_mutating: bool
    requires_confirmation: bool = False


HDA_VERBS: dict[str, HDAVerb] = {
    "open_app": HDAVerb("open_app", "Launch application", ["app_launch"], True),
    "switch_window": HDAVerb("switch_window", "Focus window", ["window_manage"], False),
    "type_text": HDAVerb("type_text", "Keyboard input (Unicode)", ["keyboard"], True),
    "click_element": HDAVerb("click_element", "Mouse click", ["mouse"], True),
    "screenshot": HDAVerb("screenshot", "Capture screen + BLAKE3 hash", ["screen_read"], False),
    "read_clipboard": HDAVerb("read_clipboard", "Read/write clipboard", ["clipboard"], False),
    "file_open": HDAVerb("file_open", "File open/move/copy/delete", ["filesystem"], True),
    "browser_navigate": HDAVerb("browser_navigate", "Navigate URL/fill/submit",
                                 ["network", "keyboard"], True, requires_confirmation=True),
}

TELESCRIPT_TO_HDA: dict[str, str] = {
    "ocr": "file_open", "filesystem": "file_open", "email": "browser_navigate",
    "analysis": "type_text", "app_launch": "open_app", "window": "switch_window",
    "input": "type_text", "click": "click_element", "capture": "screenshot",
    "clipboard": "read_clipboard",
}
```

### 3.2 AHK Bridge IPC

```python
@dataclass
class AHKRequest:
    jsonrpc: str = "2.0"
    method: str = ""
    params: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))

    def to_json(self) -> bytes:
        return json.dumps({"jsonrpc": self.jsonrpc, "method": self.method,
                           "params": self.params, "id": self.id}).encode() + b"\n"

@dataclass
class AHKResponse:
    id: str = ""
    result: Optional[dict] = None
    error: Optional[dict] = None
    success: bool = False

    @classmethod
    def from_json(cls, data: bytes) -> AHKResponse:
        p = json.loads(data.decode())
        return cls(id=p.get("id", ""), result=p.get("result"), error=p.get("error"),
                   success=p.get("result", {}).get("success", False))


class AHKBridgeClient:
    """TCP client for AHK bridge (port 9742, JSON-RPC, newline-delimited)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9742) -> None:
        self.host = host
        self.port = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port), timeout=5.0)

    async def disconnect(self) -> None:
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()

    async def invoke(self, verb: str, params: dict) -> AHKResponse:
        if self._writer is None:
            await self.connect()
        req = AHKRequest(method=verb, params=params)
        self._writer.write(req.to_json())
        await self._writer.drain()
        line = await asyncio.wait_for(self._reader.readline(), timeout=30.0)
        return AHKResponse.from_json(line)

    async def screenshot_hash(self) -> str:
        resp = await self.invoke("screenshot", {"return_hash": True})
        return resp.result.get("hash", "") if resp.result else ""
```

### 3.3 Permit Enforcement

```python
import hashlib, hmac, time

@dataclass
class HDAPermit:
    """Telescript-inspired permit. General Magic (1994) + HMAC-SHA256."""
    permit_id: str
    profile_id: str
    capabilities: list[str]
    budget_usd: float
    budget_spent: float = 0.0
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    hmac_signature: str = ""

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.budget_usd - self.budget_spent)

    def has_capability(self, cap: str) -> bool: return cap in self.capabilities
    def has_budget(self, cost: float) -> bool: return self.budget_remaining >= cost
    def is_expired(self) -> bool: return time.time() > self.expires_at
    def spend(self, amount: float) -> bool:
        if amount > self.budget_remaining: return False
        self.budget_spent += amount
        return True


class PermitEnforcer:
    def __init__(self, signing_key: bytes) -> None:
        self._key = signing_key

    def check(self, permit: HDAPermit, verb: HDAVerb, cost: float) -> tuple[bool, str]:
        if permit.is_expired(): return False, "permit_expired"
        for cap in verb.capabilities_required:
            if not permit.has_capability(cap): return False, f"missing_capability:{cap}"
        if not permit.has_budget(cost): return False, "insufficient_budget"
        if not self._verify(permit): return False, "invalid_signature"
        return True, "permitted"

    def _verify(self, p: HDAPermit) -> bool:
        payload = f"{p.permit_id}:{','.join(sorted(p.capabilities))}:{p.budget_usd}"
        expected = hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(p.hmac_signature, expected)
```

### 3.4 Ghost Overlay

```python
@dataclass
class GhostSuggestion:
    suggestion_id: str
    title: str
    description: str
    confidence: float
    estimated_cpva: float
    telescript_preview: list[dict]
    source: str   # "hmm_prediction" | "reflex_match" | "pattern_detection"

class GhostOverlay:
    """Proactive suggestion engine. Phase 48: ghost_overlay.ahk (403 LOC)."""

    def __init__(self, engram: EngramStore, reflex_ledger: ReflexLedger):
        self.engram = engram
        self.reflex_ledger = reflex_ledger

    async def generate_suggestions(self, desktop_state: dict) -> list[GhostSuggestion]:
        suggestions = []
        # File pattern detection
        files = desktop_state.get("open_files", [])
        pdf_count = sum(1 for f in files if f.lower().endswith(".pdf"))
        if pdf_count >= 10:
            suggestions.append(GhostSuggestion(
                suggestion_id=str(uuid4()), title=f"Organize {pdf_count} PDFs?",
                description="I can sort these by vendor and date",
                confidence=0.75, estimated_cpva=0.07,
                telescript_preview=[{"action": "organize_pdfs"}],
                source="pattern_detection"))
        # Reflex match
        ctx = " ".join(desktop_state.get("recent_actions", []))
        if ctx:
            reflex = await self.reflex_ledger.match(ctx)
            if reflex:
                suggestions.append(GhostSuggestion(
                    suggestion_id=str(uuid4()),
                    title=f"Repeat: {reflex.pattern_name}?",
                    description=f"Done {reflex.success_count}x before",
                    confidence=0.9, estimated_cpva=reflex.cpva_range[2],
                    telescript_preview=reflex.telescript_template[:3],
                    source="reflex_match"))
        return suggestions
```

### 3.5 Perception-Action Loop

```python
class PerceptionActionLoop:
    """Pre/post screenshot hash verification. Boyd (OODA, 1976)."""

    def __init__(self, ahk: AHKBridgeClient):
        self.ahk = ahk

    async def execute_with_verification(self, verb: str, params: dict,
            expect_state_change: bool = True) -> tuple[AHKResponse, str, str]:
        pre_hash = await self.ahk.screenshot_hash()
        response = await self.ahk.invoke(verb, params)
        await asyncio.sleep(TIMESCALE_T1_CYCLE_MS / 1000.0)
        post_hash = await self.ahk.screenshot_hash()

        if expect_state_change and pre_hash == post_hash:
            logger.warning("perception.unchanged: verb=%s hash=%s",
                           verb, pre_hash[:16])
        return response, pre_hash, post_hash
```

---

## 4. TDD Anchors

```python
import pytest

class TestHDAAHKTelescript:
    """Phase 52.8: HDA + AHK + Telescript tests."""

    def test_ahk_connection(self):
        client = AHKBridgeClient()
        assert client.host == "127.0.0.1" and client.port == 9742

    def test_verb_dispatch(self):
        assert len(HDA_VERBS) == 8
        for name in ["open_app", "switch_window", "type_text", "click_element",
                      "screenshot", "read_clipboard", "file_open", "browser_navigate"]:
            assert name in HDA_VERBS

    def test_telescript_to_hda_mapping(self):
        assert TELESCRIPT_TO_HDA["filesystem"] == "file_open"
        assert TELESCRIPT_TO_HDA["email"] == "browser_navigate"

    def test_permit_enforcement_valid(self):
        enforcer = PermitEnforcer(signing_key=b"test_key")
        permit = make_signed_permit(enforcer._key, capabilities=["filesystem"], budget=1.0)
        ok, reason = enforcer.check(permit, HDA_VERBS["file_open"], 0.01)
        assert ok is True and reason == "permitted"

    def test_permit_enforcement_missing_cap(self):
        enforcer = PermitEnforcer(signing_key=b"test_key")
        permit = make_signed_permit(enforcer._key, capabilities=["keyboard"], budget=1.0)
        ok, reason = enforcer.check(permit, HDA_VERBS["file_open"], 0.01)
        assert ok is False and "missing" in reason

    def test_permit_enforcement_budget(self):
        enforcer = PermitEnforcer(signing_key=b"test_key")
        permit = make_signed_permit(enforcer._key, capabilities=["filesystem"], budget=0.001)
        ok, _ = enforcer.check(permit, HDA_VERBS["file_open"], 0.01)
        assert ok is False

    def test_permit_expired(self):
        enforcer = PermitEnforcer(signing_key=b"test_key")
        permit = make_signed_permit(enforcer._key, capabilities=["filesystem"], budget=1.0)
        permit.expires_at = 0.0
        ok, reason = enforcer.check(permit, HDA_VERBS["file_open"], 0.01)
        assert ok is False and reason == "permit_expired"

    @pytest.mark.asyncio
    async def test_ghost_suggestion_pdf(self):
        ghost = GhostOverlay(engram=mock_engram(), reflex_ledger=mock_ledger())
        suggestions = await ghost.generate_suggestions(
            {"open_files": [f"inv_{i}.pdf" for i in range(20)]})
        assert len(suggestions) >= 1

    @pytest.mark.asyncio
    async def test_perception_action_hash(self):
        ahk = MockAHKClient(hashes=["before", "after"], response={"success": True})
        loop = PerceptionActionLoop(ahk)
        resp, pre, post = await loop.execute_with_verification(
            "file_open", {"action": "move"}, True)
        assert pre == "before" and post == "after" and pre != post
```
