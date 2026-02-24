#!/usr/bin/env python3
"""
NTU Monitor Hook for Claude Code
=================================
Tracks temporal patterns in tool usage for anomaly detection.

Uses the NTU (NeuroTemporal Unit) for O(n log n) pattern detection.

Usage in .claude/settings.json:
{
  "hooks": {
    "PostToolUse": [{
      "matcher": { "tool_name": ".*" },
      "hooks": ["python .claude/hooks/ntu_monitor.py"]
    }]
  }
}
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from collections import deque
import math

# NTU Parameters (from core/ntu/ntu.py)
WINDOW_SIZE = 64
ALPHA = 0.4  # Belief weight
BETA = 0.35  # Entropy weight
GAMMA = 0.25  # Potential weight

# Observation file
OBSERVATIONS_FILE = Path(".claude/logs/ntu_observations.jsonl")


class SimpleNTU:
    """Simplified NTU for hook context (no numpy dependency)."""

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window = deque(maxlen=window_size)
        self.belief = 0.5
        self.entropy = 1.0
        self.potential = 0.5

    def observe(self, value: float) -> tuple:
        """Process observation and update state."""
        # Clamp value to [0, 1]
        value = max(0.0, min(1.0, value))
        self.window.append(value)

        if len(self.window) < 2:
            return self.belief, False

        # Compute components
        mean = sum(self.window) / len(self.window)

        # Entropy approximation
        if 0 < mean < 1:
            self.entropy = -mean * math.log2(mean + 1e-10) - (1 - mean) * math.log2(1 - mean + 1e-10)
        else:
            self.entropy = 0.0

        # Temporal consistency (autocorrelation proxy)
        diffs = [abs(self.window[i] - self.window[i - 1]) for i in range(1, len(self.window))]
        consistency = 1.0 - (sum(diffs) / len(diffs)) if diffs else 0.5

        # Update belief with conjugate prior style
        prior = self.belief
        likelihood = value
        self.belief = ALPHA * prior + (1 - ALPHA) * likelihood

        # Update potential
        self.potential = GAMMA * consistency + (1 - GAMMA) * self.potential

        # Anomaly detection: sudden drop in belief with high entropy
        anomaly = self.belief < 0.3 and self.entropy > 0.8

        return self.belief, anomaly


def compute_tool_score(tool_name: str, tool_result: dict) -> float:
    """Convert tool result to NTU observation value."""
    # Base score
    score = 0.5

    # Success increases score
    if tool_result.get("is_error", False):
        score -= 0.3
    else:
        score += 0.3

    # Tool-specific adjustments
    if tool_name == "Bash":
        # Check exit code
        output = tool_result.get("output", "")
        if "error" in output.lower() or "failed" in output.lower():
            score -= 0.2
    elif tool_name in ("Read", "Glob", "Grep"):
        # Search tools: finding results is good
        output = tool_result.get("output", "")
        if output and len(output) > 10:
            score += 0.1
    elif tool_name in ("Write", "Edit"):
        # Modification tools: completion is good
        score += 0.1

    return max(0.0, min(1.0, score))


def load_or_create_ntu() -> SimpleNTU:
    """Load NTU state from file or create new."""
    state_file = Path(".claude/logs/ntu_state.json")

    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
                ntu = SimpleNTU()
                ntu.belief = state.get("belief", 0.5)
                ntu.entropy = state.get("entropy", 1.0)
                ntu.potential = state.get("potential", 0.5)
                for v in state.get("window", []):
                    ntu.window.append(v)
                return ntu
        except Exception:
            pass

    return SimpleNTU()


def save_ntu_state(ntu: SimpleNTU):
    """Save NTU state to file."""
    state_file = Path(".claude/logs/ntu_state.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "belief": ntu.belief,
        "entropy": ntu.entropy,
        "potential": ntu.potential,
        "window": list(ntu.window),
        "updated": datetime.utcnow().isoformat() + "Z",
    }

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def main():
    """Main hook entry point."""
    # Read hook input from stdin
    input_data = json.load(sys.stdin)

    tool_name = input_data.get("tool_name", "unknown")
    tool_result = input_data.get("tool_result", {})

    # Compute observation value
    score = compute_tool_score(tool_name, tool_result)

    # Load NTU and process observation
    ntu = load_or_create_ntu()
    belief, anomaly = ntu.observe(score)

    # Save state
    save_ntu_state(ntu)

    # Log observation
    OBSERVATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    observation = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tool": tool_name,
        "score": score,
        "belief": belief,
        "entropy": ntu.entropy,
        "anomaly": anomaly,
    }

    with open(OBSERVATIONS_FILE, "a") as f:
        f.write(json.dumps(observation) + "\n")

    # If anomaly detected, output warning (PostToolUse can't block, but can notify)
    if anomaly:
        warning = {
            "type": "warning",
            "message": f"NTU anomaly detected: belief={belief:.3f}, entropy={ntu.entropy:.3f}",
        }
        print(json.dumps(warning))


if __name__ == "__main__":
    main()
