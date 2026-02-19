# Spec 04: Genesis Covenant Loader

Standing on Giants:
- Al-Ghazali (1095): The covenant precedes the code
- Lamport (1978): The genesis block defines all subsequent blocks
- Friston (2010): Priors constrain inference — the covenant is the prior

## Problem

The Guardian agent enforces Ihsan and constitutional constraints, but it has
no access to the actual covenant — the Three Invariants (RIBA_ZERO,
ZANN_ZERO, IHSAN_FLOOR) and the origin story from البذرة.

The covenant themes are already extracted in `00_GENESIS/LINEAGE_START.md`
and the constitutional constants are in `core/integration/constants.py`.
We don't need to re-parse the PDFs — we need to deliver the essence to the
Guardian's system prompt.

## Solution

A `GenesisCovenant` class that:
1. Loads LINEAGE_START.md for the origin narrative
2. Loads constitutional constants for the Three Invariants
3. Builds a compact covenant preamble (~80 tokens)
4. Injected specifically into the Guardian agent's system prompt

## Location

Added to `core/sovereign/founder_context.py` (extends the module, ~60 lines)

## Pseudocode

```python
# Added to core/sovereign/founder_context.py

@dataclass(frozen=True)
class GenesisCovenant:
    """
    The immutable covenant from Ramadan 2023.

    Three Invariants:
    - RIBA_ZERO: No exploitation. No interest. No harm.
    - ZANN_ZERO: No unverified claims. Every assertion has evidence.
    - IHSAN_FLOOR: Excellence is the minimum. score >= threshold.

    Origin: Two files written in Ramadan 2023 (رمضان ١٤٤٤):
    - البذرة (The Seed): Formal rule-set grounded in Quran and Sunnah
    - الرسالة (The Message): Personal covenant of surrender and purpose
    """

    riba_zero: str = "No exploitation. No interest. No harm."
    zann_zero: str = "No unverified claims. Every assertion has evidence."
    ihsan_floor: float = 0.95  # From UNIFIED_IHSAN_THRESHOLD
    origin_summary: str = ""   # Extracted from LINEAGE_START.md

    def to_prompt(self) -> str:
        """Build a compact covenant preamble for the Guardian agent."""
        return (
            "--- Genesis Covenant (Ramadan 2023) ---\n"
            "You enforce three immutable invariants:\n"
            f"  RIBA_ZERO: {self.riba_zero}\n"
            f"  ZANN_ZERO: {self.zann_zero}\n"
            f"  IHSAN_FLOOR: Excellence >= {self.ihsan_floor}\n"
            "Origin: Two files written in solitude — البذرة (The Seed) and الرسالة (The Message).\n"
            f"{self.origin_summary}\n"
            "If any gate fails, reject. No exceptions."
        )


def load_genesis_covenant(project_root: str | Path) -> GenesisCovenant:
    """
    Load the genesis covenant from 00_GENESIS/ and constants.

    Graceful: if files missing, returns covenant with defaults from constants.py.
    """
    from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

    root = Path(project_root)
    lineage_path = root / "00_GENESIS" / "LINEAGE_START.md"

    origin_summary = ""
    if lineage_path.exists():
        try:
            text = lineage_path.read_text(encoding="utf-8")
            # Extract the Lineage Signal section (the concise summary)
            for section in text.split("## "):
                if section.startswith("3) Lineage Signal"):
                    # Get just the summary paragraph
                    lines = section.strip().split("\n")
                    origin_summary = " ".join(
                        line.strip() for line in lines[1:]
                        if line.strip() and not line.startswith("---")
                    )
                    break
        except Exception:
            pass

    return GenesisCovenant(
        ihsan_floor=UNIFIED_IHSAN_THRESHOLD,
        origin_summary=origin_summary or (
            "Spiritual and ethical genesis: a covenantal origin, "
            "a moral operating system, placing Ihsan and discipline at the center."
        ),
    )
```

## Integration into Guardian Prompt

```python
# In scripts/node0_activate.py — only for the Guardian agent:

if agent_id == "guardian":
    covenant = load_genesis_covenant(PROJECT_ROOT)
    covenant_text = covenant.to_prompt()
    system_prompt = (
        f"You are the PAT Guardian. Your role is {agent['role']}.\n"
        f"Standing on Giants: {agent['giants']}.\n\n"
        f"{covenant_text}\n\n"
        f"--- Founder Context ---\n"
        f"{founder_preamble}\n"
        f"--- End Context ---\n\n"
        f"Evaluate this mission against the Three Invariants. "
        f"Flag any violation. Be the conscience of Node0."
    )
```

## Token Budget

Covenant preamble: ~80 tokens
Only applied to Guardian agent: 1 agent * 80 tokens = 80 tokens additional
Total with founder context (full): 80 + 180 = 260 tokens for Guardian
Still well under any model's context budget.

## Test Anchors

```python
class TestGenesisCovenant:
    def test_loads_from_real_genesis(self):
        """Covenant loads from actual 00_GENESIS/ directory."""
        covenant = load_genesis_covenant(PROJECT_ROOT)
        assert covenant.ihsan_floor == 0.95
        assert len(covenant.origin_summary) > 0

    def test_covenant_prompt_contains_three_invariants(self):
        """Covenant prompt mentions all three invariants."""
        covenant = load_genesis_covenant(PROJECT_ROOT)
        text = covenant.to_prompt()
        assert "RIBA_ZERO" in text
        assert "ZANN_ZERO" in text
        assert "IHSAN_FLOOR" in text

    def test_covenant_prompt_mentions_origin(self):
        """Covenant prompt references البذرة and الرسالة."""
        covenant = load_genesis_covenant(PROJECT_ROOT)
        text = covenant.to_prompt()
        assert "البذرة" in text or "The Seed" in text
        assert "الرسالة" in text or "The Message" in text

    def test_graceful_without_lineage_file(self, tmp_path):
        """Missing LINEAGE_START.md produces default origin summary."""
        covenant = load_genesis_covenant(tmp_path)
        assert covenant.ihsan_floor == 0.95
        assert "Ihsan" in covenant.origin_summary

    def test_covenant_prompt_under_100_tokens(self):
        """Covenant prompt is compact enough for system prompt injection."""
        covenant = load_genesis_covenant(PROJECT_ROOT)
        text = covenant.to_prompt()
        assert len(text.split()) < 100
```

## Why This Matters

Without the covenant, the Guardian is a generic "ethical oversight" agent.
With the covenant, the Guardian becomes the conscience of Node0 — aware of
the Three Invariants, the origin story, and the founder's mission. When it
evaluates a mission output, it checks not just "is this high quality?" but
"does this honor the seed that was planted in Ramadan 2023?"

This is the difference between a compliance checker and a guardian of identity.
