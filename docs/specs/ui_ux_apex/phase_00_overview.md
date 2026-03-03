# UI/UX APEX — Phase 00: Master Overview & Prototype Recommendation

**INVOCATION ID:** `ui_ux_apex_0xAF88`
**TIMESTAMP:** `2026-02-24T12:35:10Z`
**CPO DIRECTIVE:** Masterpiece UI/UX Framework — Sovereign Presence over Chat Bubble

> Standing on Giants: Norman (affordances, 1988) · Gibson (ecological perception, 1979) ·
> Csikszentmihalyi (flow theory, 1990) · Al-Ghazali (Ihsān ethics, 1095) ·
> Nakamoto (sovereign ownership, 2008) · Anthropic (constitutional AI, 2023)

---

## 1. Purpose

Transform BIZRA from a "tool the user opens" into an **extension of their digital nervous system**.
The six UI components below collectively produce the "Phantom Limb" effect:
when BIZRA is closed, every other OS feels dead, reactive, and extractive.

---

## 2. Six Component Overview

| # | Component | Core Emotion | Primary Infrastructure |
|---|-----------|-------------|----------------------|
| 1 | **Cognitive Helix** | Sovereignty / self-knowledge | `core/reasoning/graph_types.py`, WebGL |
| 2 | **Ghost Overlay** | Super-Perception | `bizra-action` AHK channel, HDA |
| 3 | **Iḥsān Gauge** | Virtue / guardian loyalty | `core/governance/constitutional_gate.py` |
| 4 | **MMORPG Progression** | Growth / mastery | `core/guild/`, `core/quest/` |
| 5 | **Sovereign Key** | Ownership / identity | `bizra-omega/fate-binding/` (Ed25519) |
| 6 | **Resonance Feed** | VIP status / market power | `core/a2a/`, HHMM predictor |

---

## 3. Prototype Recommendation: Ghost Overlay First

**Recommendation: Prototype Phase 02 (Ghost Overlay) before Phase 01 (Cognitive Helix).**

### Rationale

```
GHOST OVERLAY                         COGNITIVE HELIX
─────────────────────────────         ─────────────────────────────
Infrastructure: AHK channel EXISTS    Infrastructure: WebGL (new dep)
Data dependency: NONE (proactive)     Data dependency: ≥N GoT nodes
Impact surface: ALL apps user runs    Impact surface: BIZRA UI only
Phantom Limb onset: Day 1             Phantom Limb onset: Week 2+
Build risk: LOW (JSON-RPC stub ready) Build risk: MEDIUM (3D engine)
```

The AHK channel in `bizra-omega/bizra-action/src/channels/mod.rs` already defines the
`ChannelHandler` interface. The Ghost Overlay is its first full production consumer.

The Cognitive Helix requires a populated GoT graph to be *meaningful* — without memories
and reflexes, a helix is an empty wireframe. Ghost Overlay works from minute one.

**Sprint sequence:**
1. Ghost Overlay (Phase 02) — 2-week sprint → immediate Phantom Limb
2. Iḥsān Gauge (Phase 03) — 1-week sprint → ethical feedback loop
3. Cognitive Helix (Phase 01) — 3-week sprint → after graph is populated
4. MMORPG Progression (Phase 04) — 2-week sprint
5. Sovereign Key (Phase 05) — 1-week sprint (hardware dependency)
6. Resonance Feed (Phase 06) — 2-week sprint

---

## 4. Shared Design System Constraints

```
CONSTRAINT: No hard-coded secrets or config values in UI layer.
            All thresholds from core/integration/constants.py via API.

DESIGN TOKENS (sourced from BIZRA Brand Identity v2.0 Elite):
  Background:   #050B14  (Celestial Navy)
  Gold:         #C9A962  (Genesis Gold)
  Green:        #2eb86a  (Operational)
  Red:          #c93a4a  (Alert)
  Glass:        rgba(255,255,255,0.03) + backdrop-filter:blur(10px)
  Grid:         rgba(201,169,98,0.04) on 50px grid
  Font-Serif:   Playfair Display (headings, emotional weight)
  Font-UI:      Inter (data, controls)
  Font-Arabic:  Amiri (Iḥsān labels, Arabic tagline البذرة)
  Font-Mono:    JetBrains Mono (hashes, code, receipts)

ACCESSIBILITY:
  - All overlays must have ESC / Sovereign Gesture dismiss
  - WCAG 2.1 AA contrast on all text
  - Reduced-motion mode: disable animations, keep data

CONSTITUTIONAL:
  - No UI action executes without Ihsān ≥ 0.95 gate
  - All rendered data is read-only; mutations require explicit Sovereign Gesture
  - Ghost Overlay must not capture keystrokes from target application
```

---

## 5. Architecture Topology

```
┌─────────────────────────────────────────────────────────────┐
│  SOVEREIGN PRESENCE LAYER  (browser/Electron/AHK overlay)   │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Cognitive│  │  Ghost   │  │  Iḥsān  │  │ MMORPG   │  │
│  │  Helix   │  │ Overlay  │  │  Gauge   │  │  Tier    │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       └─────────────┴──────────────┴──────────────┘        │
│                          │                                  │
│              Sovereign Bridge (WebSocket / IPC)             │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  NODE0 KERNEL (Python)                                      │
│  core/reasoning/  core/governance/  core/a2a/               │
│  core/living_memory/  core/sovereign/                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  BIZRA ACTION BUS (Rust — bizra-omega/bizra-action)         │
│  AHK channel · LLM channel · Memory channel · Response ch.  │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Spec File Index

| File | Component | Lines |
|------|-----------|-------|
| `phase_00_overview.md` | Master brief (this file) | ~130 |
| `phase_01_cognitive_helix.md` | 3D GoT visualization | <500 |
| `phase_02_ghost_overlay.md` | AHK Ghost Layer + HDA | <500 |
| `phase_03_ihsan_gauge.md` | Ethical telemetry dashboard | <500 |
| `phase_04_mmorpg_progression.md` | Atlas Tier + Sovereign Quests | <500 |
| `phase_05_sovereign_key.md` | Ed25519 biometric key UX | <500 |
| `phase_06_resonance_feed.md` | A2A gift feed (Guild Hall) | <500 |

---

## 7. TDD Anchors (shared across all phases)

```python
# tests/ui_ux_apex/conftest.py

DESIGN_TOKENS = {
    "bg": "#050B14",
    "gold": "#C9A962",
    "ihsan_threshold": 0.95,   # from constants.py
    "snr_threshold": 0.85,     # from constants.py
}

def assert_no_hardcoded_secrets(component_source: str) -> None:
    """No API keys, tokens, or IP addresses in UI source."""
    import re
    forbidden = [r"[A-Za-z0-9+/]{32,}={0,2}", r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"]
    for pattern in forbidden:
        assert not re.search(pattern, component_source), \
            f"Hardcoded secret candidate found: {pattern}"

def assert_constitutional_gate_called(mock_gate, action_name: str) -> None:
    """Every UI-triggered mutation must pass through ConstitutionalGate."""
    calls = [c for c in mock_gate.call_args_list if action_name in str(c)]
    assert len(calls) >= 1, f"ConstitutionalGate not called for {action_name}"
```
