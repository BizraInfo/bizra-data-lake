#!/usr/bin/env python3
"""
Session Memory Hook for Claude Code
====================================
Persists session context to .claude-flow/memory for cross-session continuity.

Events:
- Stop: Save session summary
- Notification (subagent complete): Merge agent learnings

Usage in .claude/settings.json:
{
  "hooks": {
    "Stop": [{
      "matcher": {},
      "hooks": ["python .claude/hooks/session_memory.py stop"]
    }],
    "Notification": [{
      "matcher": { "type": "agent_complete" },
      "hooks": ["python .claude/hooks/session_memory.py notification"]
    }]
  }
}
"""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path(".claude-flow/memory")


def compute_session_hash(content: str) -> str:
    """Compute short hash for deduplication."""
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def save_session_summary(input_data: dict):
    """Save session summary on Stop event."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    # Extract session info
    session_id = input_data.get("session_id", "unknown")
    stop_reason = input_data.get("stop_reason", "user_request")

    # Create session summary
    summary = {
        "updated": datetime.utcnow().isoformat() + "Z",
        "category": "session",
        "session_id": session_id,
        "stop_reason": stop_reason,
        "cwd": os.getcwd(),
    }

    # Read conversation stats if available
    stats_file = Path(".claude/session_stats.json")
    if stats_file.exists():
        try:
            with open(stats_file) as f:
                stats = json.load(f)
                summary["stats"] = stats
        except Exception:
            pass

    # Save to memory
    filename = f"session-{session_id[:8]}.json"
    with open(MEMORY_DIR / filename, "w") as f:
        json.dump(summary, f, indent=2)

    # Update session index
    update_session_index(session_id, summary)


def update_session_index(session_id: str, summary: dict):
    """Update the session index with new entry."""
    index_file = MEMORY_DIR / "session-index.json"

    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
    else:
        index = {
            "updated": datetime.utcnow().isoformat() + "Z",
            "category": "index",
            "name": "Session History Index",
            "sessions": [],
        }

    # Add session entry
    entry = {
        "session_id": session_id,
        "timestamp": summary["updated"],
        "stop_reason": summary.get("stop_reason"),
    }

    # Keep last 50 sessions
    index["sessions"] = [entry] + index["sessions"][:49]
    index["updated"] = datetime.utcnow().isoformat() + "Z"

    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)


def merge_agent_learnings(input_data: dict):
    """Merge learnings from completed subagent."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    agent_id = input_data.get("agent_id", "unknown")
    result = input_data.get("result", "")

    # Extract key insights (simple heuristic)
    insights = []
    for line in result.split("\n"):
        line = line.strip()
        if any(marker in line.lower() for marker in ["key:", "insight:", "learned:", "important:"]):
            insights.append(line)

    if not insights:
        return

    # Save agent learnings
    learnings_file = MEMORY_DIR / "agent-learnings.jsonl"
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "agent_id": agent_id,
        "insights": insights,
    }

    with open(learnings_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    """Main hook entry point."""
    if len(sys.argv) < 2:
        print("Usage: session_memory.py <event_type>", file=sys.stderr)
        sys.exit(1)

    event_type = sys.argv[1]

    # Read hook input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        input_data = {}

    if event_type == "stop":
        save_session_summary(input_data)
    elif event_type == "notification":
        merge_agent_learnings(input_data)
    else:
        print(f"Unknown event type: {event_type}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
