# Spec 01: FounderContext Builder

Standing on Giants:
- Friston (2010): Active Inference — an agent's generative model of ITSELF
- Deming (1950): Process awareness — you must know what you are before you act
- Al-Ghazali (1095): Ma'rifa (self-knowledge) precedes Ihsan (excellence)

## Problem

PAT agents have no structured awareness of the founder, the assets, the mission,
or the current goals. The user_profile.json and node0_baseline.json exist in
sovereign_state/ but are never loaded into agent prompts.

## Solution

Create a `FounderContext` class that:
1. Loads user_profile.json and node0_baseline.json at startup
2. Builds tiered context strings (full / standard / minimal)
3. Caches the result (identity doesn't change mid-session)
4. Provides a clean API for the PAT prompt builder

## Location

`core/sovereign/founder_context.py` — new file (~120 lines)

## Pseudocode

```python
"""
FounderContext — Structured identity awareness for PAT agents.

Standing on Giants:
- Friston (2010): Self-model in active inference
- Deming (1950): Process self-awareness
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FounderIdentity:
    """Immutable snapshot of the founder's identity and assets."""

    name: str                        # "Mohammed Beshr"
    node_name: str                   # "MoMo (محمد)"
    node_id: str                     # "node0_ce5af35c848ce889"
    bio: str                         # One-line bio
    mission: str                     # Core mission statement
    expertise: list[str]             # Domain expertise list
    values: list[str]               # Core values
    active_focus: str                # Current priority
    goals_short: list[str]          # Weekly goals
    pain_points: list[str]          # Known friction areas
    assets: dict[str, Any]          # Node0 asset inventory


class FounderContext:
    """
    Loads founder identity from sovereign_state/ and builds
    tiered context strings for PAT agent prompts.

    Tiers:
    - FULL:     ~180 tokens — identity + assets + goals + values + pain points
    - STANDARD: ~100 tokens — identity + assets + goals
    - MINIMAL:  ~40 tokens  — identity + mission one-liner

    Usage:
        ctx = FounderContext(sovereign_state_dir)
        prompt_fragment = ctx.build("standard")
    """

    def __init__(self, sovereign_state_dir: str | Path):
        self._dir = Path(sovereign_state_dir)
        self._identity: Optional[FounderIdentity] = None
        self._cache: dict[str, str] = {}  # tier -> formatted string
        self._load()

    def _load(self):
        """Load user_profile.json + node0_baseline.json into FounderIdentity."""
        profile_path = self._dir / "user_profile.json"
        baseline_path = self._dir / "node0_baseline.json"

        profile = {}
        baseline = {}

        if profile_path.exists():
            try:
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"FounderContext: failed to load profile: {e}")

        if baseline_path.exists():
            try:
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"FounderContext: failed to load baseline: {e}")

        if not profile and not baseline:
            logger.warning("FounderContext: no identity files found — PAT agents run anonymous")
            return

        self._identity = FounderIdentity(
            name=profile.get("name", "Unknown"),
            node_name=profile.get("node_name", "Node0"),
            node_id=profile.get("node_id", baseline.get("node_id", "")),
            bio=profile.get("bio", ""),
            mission=profile.get("mission", ""),
            expertise=profile.get("expertise", []),
            values=profile.get("values", []),
            active_focus=profile.get("active_focus", ""),
            goals_short=profile.get("goals_short", baseline.get("weekly_goals", [])),
            pain_points=baseline.get("pain_points", profile.get("pain_points", [])),
            assets=baseline.get("node0_assets", {}),
        )

    @property
    def loaded(self) -> bool:
        return self._identity is not None

    @property
    def identity(self) -> Optional[FounderIdentity]:
        return self._identity

    def build(self, tier: str = "standard") -> str:
        """
        Build a context string at the given tier.

        Returns empty string if no identity loaded (graceful degradation).
        """
        if not self._identity:
            return ""

        if tier in self._cache:
            return self._cache[tier]

        result = self._format(tier)
        self._cache[tier] = result
        return result

    def _format(self, tier: str) -> str:
        """Format identity into a prompt-ready string."""
        i = self._identity

        if tier == "minimal":
            # ~40 tokens
            return (
                f"You serve {i.node_name} ({i.name}), founder of BIZRA. "
                f"{i.mission}"
            )

        if tier == "standard":
            # ~100 tokens
            assets = i.assets
            goals = "; ".join(i.goals_short[:3]) if i.goals_short else "Not set"
            return (
                f"You serve {i.node_name} ({i.name}), founder of BIZRA — Node0.\n"
                f"Mission: {i.mission}\n"
                f"Focus: {i.active_focus}\n"
                f"Goals: {goals}\n"
                f"Assets: {assets.get('data_volume_tb', '?')}TB data, "
                f"{assets.get('research_papers', '?')} papers, "
                f"{assets.get('github_repos', '?')} repos, "
                f"{assets.get('ai_conversations', '?')} AI conversations, "
                f"{assets.get('rd_hours', '?')} R&D hours"
            )

        # tier == "full" — ~180 tokens
        assets = i.assets
        goals = "\n".join(f"  - {g}" for g in i.goals_short) if i.goals_short else "  Not set"
        pains = "\n".join(f"  - {p}" for p in i.pain_points[:3]) if i.pain_points else "  None"
        expertise = ", ".join(i.expertise[:5]) if i.expertise else "General"

        return (
            f"You serve {i.node_name} ({i.name}), founder of BIZRA — Node0.\n"
            f"Mission: {i.mission}\n"
            f"Bio: {i.bio}\n"
            f"Expertise: {expertise}\n"
            f"Current focus: {i.active_focus}\n"
            f"Weekly goals:\n{goals}\n"
            f"Pain points:\n{pains}\n"
            f"Node0 assets: {assets.get('data_volume_tb', '?')}TB data, "
            f"{assets.get('research_papers', '?')} papers, "
            f"{assets.get('github_repos', '?')} repos, "
            f"{assets.get('ai_conversations', '?')} AI conversations, "
            f"{assets.get('rd_hours', '?')} R&D hours, "
            f"{assets.get('faiss_vectors_indexed', '?')} indexed vectors"
        )
```

## Test Anchors

```python
# tests/core/sovereign/test_founder_context.py

class TestFounderContextLoad:
    def test_loads_from_real_sovereign_state(self):
        """FounderContext loads from the actual sovereign_state/ directory."""
        ctx = FounderContext(PROJECT_ROOT / "sovereign_state")
        assert ctx.loaded
        assert ctx.identity.name == "Mohammed Beshr"
        assert ctx.identity.node_id == "node0_ce5af35c848ce889"

    def test_graceful_when_no_files(self, tmp_path):
        """Missing files produce empty context, not crash."""
        ctx = FounderContext(tmp_path)
        assert not ctx.loaded
        assert ctx.build("full") == ""

    def test_assets_from_baseline(self):
        """Asset inventory comes from node0_baseline.json."""
        ctx = FounderContext(PROJECT_ROOT / "sovereign_state")
        assert ctx.identity.assets.get("rd_hours") == 15000
        assert ctx.identity.assets.get("data_volume_tb") == 1.3


class TestFounderContextTiers:
    def test_minimal_under_50_tokens(self):
        """Minimal tier stays under 50 tokens."""
        ctx = FounderContext(PROJECT_ROOT / "sovereign_state")
        text = ctx.build("minimal")
        # Rough token estimate: words * 1.3
        assert len(text.split()) < 50

    def test_standard_under_120_tokens(self):
        """Standard tier stays under 120 tokens."""
        ctx = FounderContext(PROJECT_ROOT / "sovereign_state")
        text = ctx.build("standard")
        assert len(text.split()) < 120

    def test_full_under_200_tokens(self):
        """Full tier stays under 200 tokens."""
        ctx = FounderContext(PROJECT_ROOT / "sovereign_state")
        text = ctx.build("full")
        assert len(text.split()) < 200

    def test_caches_result(self):
        """Same tier returns cached string."""
        ctx = FounderContext(PROJECT_ROOT / "sovereign_state")
        a = ctx.build("standard")
        b = ctx.build("standard")
        assert a is b  # Same object (cached)

    def test_minimal_contains_name_and_mission(self):
        """Minimal tier includes name and mission."""
        ctx = FounderContext(PROJECT_ROOT / "sovereign_state")
        text = ctx.build("minimal")
        assert "MoMo" in text
        assert "BIZRA" in text

    def test_standard_contains_goals_and_assets(self):
        """Standard tier includes goals and asset summary."""
        ctx = FounderContext(PROJECT_ROOT / "sovereign_state")
        text = ctx.build("standard")
        assert "Goals:" in text
        assert "Assets:" in text
        assert "1.3TB" in text or "1.3" in text

    def test_full_contains_pain_points(self):
        """Full tier includes pain points."""
        ctx = FounderContext(PROJECT_ROOT / "sovereign_state")
        text = ctx.build("full")
        assert "Pain points" in text
```

## File Budget

- `core/sovereign/founder_context.py`: ~120 lines
- `tests/core/sovereign/test_founder_context.py`: ~80 lines
