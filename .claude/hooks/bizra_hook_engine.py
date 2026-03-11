#!/usr/bin/env python3
"""
BIZRA Hook Engine — Enterprise-Grade Claude Code Hooks
=======================================================
Unified hook system that maximizes BIZRA ecosystem performance.

Architecture:
┌────────────────────────────────────────────────────────────────────────────┐
│                           BIZRA HOOK ENGINE                                │
├────────────────────────────────────────────────────────────────────────────┤
│  SessionStart  │  PreToolUse  │  PostToolUse  │  Stop  │  SubagentStop    │
├────────────────┴──────────────┴───────────────┴────────┴──────────────────┤
│                                                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ FATE Gate   │  │ SNR Filter  │  │ NTU Monitor │  │ Self-Healer │       │
│  │ (Ethics)    │  │ (Quality)   │  │ (Temporal)  │  │ (Resilience)│       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
│                                                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ Memory Sync │  │ Audit Log   │  │ Performance │  │ Auto-Suggest│       │
│  │ (Persist)   │  │ (Tamper-Ev) │  │ (Metrics)   │  │ (Learning)  │       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

Standing on Giants: Shannon • Lamport • Anthropic • Bizra Ihsan Principles

Usage:
  python3 bizra_hook_engine.py <event> [args]

Events:
  session_start     - Initialize session context
  pre_tool          - Validate before tool execution
  post_tool         - Monitor after tool execution
  post_tool_failure - Handle and learn from failures
  stop              - Session wrap-up and persistence
  subagent_stop     - Subagent completion handling
  notification      - Notification processing
  pre_compact       - Pre-compaction memory save
"""

import json
import sys
import os
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from collections import deque
import math
import re

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — Aligned with core/integration/constants.py
# ═══════════════════════════════════════════════════════════════════════════════

# Ihsan thresholds
IHSAN_THRESHOLD = 0.95
IHSAN_THRESHOLD_STRICT = 0.99
IHSAN_THRESHOLD_CI = 0.90

# SNR thresholds
SNR_THRESHOLD_MINIMUM = 0.85
SNR_THRESHOLD_T1_HIGH = 0.95
SNR_THRESHOLD_T0_ELITE = 0.98

# FATE dimensions
FATE_DIMENSIONS = ["fidelity", "accountability", "transparency", "ethics"]

# NTU parameters
NTU_WINDOW_SIZE = 64
NTU_ALPHA = 0.4   # Belief weight
NTU_BETA = 0.35   # Entropy weight
NTU_GAMMA = 0.25  # Potential weight

# Tool risk levels - thresholds for validation
HIGH_RISK_TOOLS = {
    "Bash": 0.85,
    "Write": 0.80,
    "Edit": 0.80,
    "WebFetch": 0.75,
    "WebSearch": 0.70,
    "Task": 0.70,
}

# Blocked command patterns (Ethics gate) - ONLY block truly dangerous operations
BLOCKED_COMMAND_PATTERNS = [
    r"rm\s+-rf\s+/\s*$",           # rm -rf / (root only)
    r"rm\s+-rf\s+/\*",             # rm -rf /*
    r":\(\)\{\s*:\|:\&\s*\};:",    # Fork bomb
    r"dd\s+if=/dev/zero\s+of=/dev/sd", # Disk wipe
    r"mkfs\.\w+\s+/dev/sd",        # Format disk
    r">\s*/dev/sda",               # Overwrite disk
    r"chmod\s+-R\s+777\s+/\s*$",   # chmod 777 root
]

# Paths
PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", "/mnt/c/BIZRA-DATA-LAKE"))
HOOKS_LOG_DIR = PROJECT_DIR / ".claude" / "logs"
MEMORY_DIR = PROJECT_DIR / ".claude-flow" / "memory"
STATE_DIR = PROJECT_DIR / ".claude" / "state"


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def now_utc() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs():
    """Ensure all required directories exist."""
    for d in [HOOKS_LOG_DIR, MEMORY_DIR, STATE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def compute_hash(content: str, length: int = 12) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()[:length]


def log_entry(filename: str, entry: dict):
    """Append entry to JSONL log file."""
    ensure_dirs()
    log_file = HOOKS_LOG_DIR / filename
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_state(name: str) -> dict:
    """Load state from file."""
    ensure_dirs()
    state_file = STATE_DIR / f"{name}.json"
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except Exception:
            pass
    return {}


def save_state(name: str, state: dict):
    """Save state to file."""
    ensure_dirs()
    state_file = STATE_DIR / f"{name}.json"
    state["_updated"] = now_utc()
    state_file.write_text(json.dumps(state, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# FATE GATE — Constitutional AI Validation (Improved)
# ═══════════════════════════════════════════════════════════════════════════════

class FATEGate:
    """
    FATE (Fidelity, Accountability, Transparency, Ethics) validation gate.

    Improved version that only blocks truly dangerous operations.
    """

    def __init__(self):
        self.blocked_patterns = [re.compile(p, re.IGNORECASE) for p in BLOCKED_COMMAND_PATTERNS]

    def compute_score(self, tool_name: str, tool_input: dict) -> dict:
        """Compute FATE dimensions for a tool invocation."""
        scores = {dim: 1.0 for dim in FATE_DIMENSIONS}

        # Fidelity: Only penalize if ACTUAL hardcoded credentials detected
        # (not variable names or patterns in code)
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            # Only check for actual credential assignment in commands
            if re.search(r'(export\s+)?(API_KEY|TOKEN|PASS)\s*=\s*["\'][^"\']{10,}["\']', command):
                scores["fidelity"] *= 0.7

        # Accountability: Check for description
        if tool_name == "Bash" and not tool_input.get("description"):
            scores["accountability"] *= 0.95

        # Transparency: Check for obfuscated commands in Bash only
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            # Piped curl/wget to shell is suspicious
            if re.search(r"(curl|wget)\s+[^\|]+\|\s*(ba)?sh", command, re.IGNORECASE):
                scores["transparency"] *= 0.6

        # Ethics: Block dangerous command patterns
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            for pattern in self.blocked_patterns:
                if pattern.search(command):
                    scores["ethics"] = 0.0
                    break

        # Compute composite (geometric mean)
        composite = 1.0
        for score in scores.values():
            composite *= score
        composite = composite ** (1.0 / len(scores))

        threshold = HIGH_RISK_TOOLS.get(tool_name, 0.70)

        return {
            "dimensions": scores,
            "composite": composite,
            "threshold": threshold,
            "passed": composite >= threshold,
        }

    def validate(self, tool_name: str, tool_input: dict) -> tuple:
        """Validate a tool invocation against FATE gate."""
        result = self.compute_score(tool_name, tool_input)

        if result["passed"]:
            return True, "", result

        # Build reason string
        failing_dims = []
        for dim, score in result["dimensions"].items():
            if score < 0.9:
                failing_dims.append(f"{dim}={score:.2f}")

        reason = f"FATE Gate: composite={result['composite']:.3f} < threshold={result['threshold']}"
        if failing_dims:
            reason += f" ({', '.join(failing_dims)})"

        return False, reason, result


# ═══════════════════════════════════════════════════════════════════════════════
# NTU MONITOR — Temporal Pattern Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class NTUMonitor:
    """NeuroTemporal Unit for O(n log n) pattern detection."""

    def __init__(self, window_size: int = NTU_WINDOW_SIZE):
        self.window = deque(maxlen=window_size)
        self.belief = 0.5
        self.entropy = 1.0
        self.potential = 0.5
        self._load_state()

    def _load_state(self):
        """Load persisted state."""
        state = load_state("ntu_state")
        if state:
            self.belief = state.get("belief", 0.5)
            self.entropy = state.get("entropy", 1.0)
            self.potential = state.get("potential", 0.5)
            for v in state.get("window", []):
                self.window.append(v)

    def _save_state(self):
        """Persist current state."""
        save_state("ntu_state", {
            "belief": self.belief,
            "entropy": self.entropy,
            "potential": self.potential,
            "window": list(self.window),
        })

    def observe(self, value: float) -> tuple:
        """Process observation and detect anomalies."""
        value = max(0.0, min(1.0, value))
        self.window.append(value)

        if len(self.window) < 2:
            return self.belief, False

        mean = sum(self.window) / len(self.window)

        if 0 < mean < 1:
            self.entropy = -mean * math.log2(mean + 1e-10) - (1 - mean) * math.log2(1 - mean + 1e-10)
        else:
            self.entropy = 0.0

        diffs = [abs(self.window[i] - self.window[i - 1]) for i in range(1, len(self.window))]
        consistency = 1.0 - (sum(diffs) / len(diffs)) if diffs else 0.5

        self.belief = NTU_ALPHA * self.belief + (1 - NTU_ALPHA) * value
        self.potential = NTU_GAMMA * consistency + (1 - NTU_GAMMA) * self.potential

        anomaly = self.belief < 0.3 and self.entropy > 0.8

        self._save_state()
        return self.belief, anomaly

    def compute_tool_score(self, tool_name: str, tool_result: dict) -> float:
        """Convert tool result to observation value."""
        score = 0.5

        if tool_result.get("is_error", False):
            score -= 0.3
        else:
            score += 0.2

        if tool_name == "Bash":
            output = str(tool_result.get("output", "")).lower()
            if "error" in output or "failed" in output:
                score -= 0.2
            if "success" in output or "completed" in output:
                score += 0.1
        elif tool_name in ("Write", "Edit"):
            if tool_result.get("success", False):
                score += 0.15

        return max(0.0, min(1.0, score))


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-HEALER — Automatic Error Recovery
# ═══════════════════════════════════════════════════════════════════════════════

class SelfHealer:
    """Self-healing system that learns from errors."""

    KNOWN_PATTERNS = {
        "import_error": {
            "pattern": r"ImportError.*cannot import name '(\w+)'",
            "fix": "Check module exports in __init__.py",
        },
        "module_not_found": {
            "pattern": r"ModuleNotFoundError: No module named '(\S+)'",
            "fix": "pip install {0}",
        },
        "file_not_found": {
            "pattern": r"FileNotFoundError:.*'([^']+)'",
            "fix": "Verify path exists: {0}",
        },
        "permission_denied": {
            "pattern": r"PermissionError:.*'([^']+)'",
            "fix": "Check file permissions: {0}",
        },
        "syntax_error": {
            "pattern": r"SyntaxError: (.+)",
            "fix": "Fix syntax: {0}",
        },
    }

    def __init__(self):
        self.error_history = []

    def analyze_error(self, error_output: str) -> Optional[dict]:
        """Analyze error and suggest fix."""
        for error_type, info in self.KNOWN_PATTERNS.items():
            match = re.search(info["pattern"], error_output, re.IGNORECASE)
            if match:
                groups = match.groups()
                fix = info["fix"].format(*groups) if groups else info["fix"]
                return {
                    "error_type": error_type,
                    "suggested_fix": fix,
                }
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY SYNC — Cross-Session Persistence
# ═══════════════════════════════════════════════════════════════════════════════

class MemorySync:
    """Persistent memory synchronization across sessions."""

    def __init__(self):
        ensure_dirs()

    def save_session_context(self, session_id: str, context: dict):
        """Save session context for persistence."""
        context_file = MEMORY_DIR / f"session-{session_id[:8]}.json"
        context["_session_id"] = session_id
        context["_saved_at"] = now_utc()
        context_file.write_text(json.dumps(context, indent=2))
        self._update_index(session_id, context)

    def load_session_context(self, session_id: str) -> Optional[dict]:
        """Load previous session context."""
        context_file = MEMORY_DIR / f"session-{session_id[:8]}.json"
        if context_file.exists():
            return json.loads(context_file.read_text())
        return None

    def _update_index(self, session_id: str, context: dict):
        """Update session index."""
        index_file = MEMORY_DIR / "session-index.json"
        if index_file.exists():
            index = json.loads(index_file.read_text())
        else:
            index = {"sessions": [], "updated": now_utc()}

        entry = {
            "session_id": session_id,
            "timestamp": now_utc(),
            "summary": context.get("summary", ""),
        }

        index["sessions"] = [entry] + [s for s in index["sessions"] if s["session_id"] != session_id][:49]
        index["updated"] = now_utc()
        index_file.write_text(json.dumps(index, indent=2))

    def get_recent_sessions(self, limit: int = 10) -> list:
        """Get recent session summaries."""
        index_file = MEMORY_DIR / "session-index.json"
        if index_file.exists():
            index = json.loads(index_file.read_text())
            return index.get("sessions", [])[:limit]
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT LOGGER — Tamper-Evident Logging
# ═══════════════════════════════════════════════════════════════════════════════

class AuditLogger:
    """Tamper-evident audit logging with hash chains."""

    def __init__(self):
        ensure_dirs()
        self.log_file = HOOKS_LOG_DIR / "audit.jsonl"
        self.last_hash = self._get_last_hash()

    def _get_last_hash(self) -> str:
        """Get hash of last log entry."""
        if not self.log_file.exists():
            return "genesis"
        try:
            with open(self.log_file, "rb") as f:
                f.seek(-2, 2)
                while f.read(1) != b'\n':
                    f.seek(-2, 1)
                last_line = f.readline().decode()
                entry = json.loads(last_line)
                return entry.get("_hash", "genesis")
        except Exception:
            return "genesis"

    def log(self, event_type: str, data: dict, decision: str = "allow"):
        """Log event with hash chain."""
        entry = {
            "timestamp": now_utc(),
            "event_type": event_type,
            "decision": decision,
            "data": data,
            "_prev_hash": self.last_hash,
        }
        entry_str = json.dumps(entry, sort_keys=True)
        entry["_hash"] = compute_hash(entry_str)
        self.last_hash = entry["_hash"]

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class PerformanceTracker:
    """Performance metrics collection."""

    def __init__(self):
        self.metrics = load_state("performance_metrics")
        if not self.metrics:
            self.metrics = {
                "tool_timings": {},
                "error_counts": {},
                "success_rate": 1.0,
                "total_operations": 0,
            }

    def record_operation(self, tool_name: str, duration_ms: float, success: bool):
        """Record tool operation metrics."""
        self.metrics["total_operations"] += 1

        if tool_name not in self.metrics["tool_timings"]:
            self.metrics["tool_timings"][tool_name] = {"count": 0, "total_ms": 0}

        timing = self.metrics["tool_timings"][tool_name]
        timing["count"] += 1
        timing["total_ms"] += duration_ms

        if not success:
            self.metrics["error_counts"][tool_name] = self.metrics["error_counts"].get(tool_name, 0) + 1

        alpha = 0.1
        self.metrics["success_rate"] = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.metrics["success_rate"]

        save_state("performance_metrics", self.metrics)

    def get_summary(self) -> dict:
        """Get performance summary."""
        return {
            "total_operations": self.metrics["total_operations"],
            "success_rate": self.metrics["success_rate"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN HOOK ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class BizraHookEngine:
    """Main hook engine orchestrating all components."""

    def __init__(self):
        self.fate_gate = FATEGate()
        self.ntu_monitor = NTUMonitor()
        self.self_healer = SelfHealer()
        self.memory_sync = MemorySync()
        self.audit_logger = AuditLogger()
        self.perf_tracker = PerformanceTracker()
        # ReasoningBank: adaptive learning from hook outcomes
        try:
            from reasoning_bank import ReasoningBank
            self.reasoning_bank = ReasoningBank()
        except ImportError:
            self.reasoning_bank = None

    def handle_session_start(self, input_data: dict) -> dict:
        """Handle SessionStart event."""
        session_id = input_data.get("session_id", "unknown")
        source = input_data.get("source", "startup")

        context = ""
        if source == "resume":
            prev_context = self.memory_sync.load_session_context(session_id)
            if prev_context:
                context = f"Resumed session. Previous: {prev_context.get('summary', 'N/A')}"

        self.audit_logger.log("session_start", {"session_id": session_id, "source": source})

        output = {}
        if context:
            output["hookSpecificOutput"] = {
                "hookEventName": "SessionStart",
                "additionalContext": context,
            }
        return output

    def handle_pre_tool_use(self, input_data: dict) -> dict:
        """Handle PreToolUse event with FATE gating + ReasoningBank advisory."""
        tool_name = input_data.get("tool_name", "unknown")
        tool_input = input_data.get("tool_input", {})

        passed, reason, fate_result = self.fate_gate.validate(tool_name, tool_input)

        self.audit_logger.log("pre_tool_use", {
            "tool_name": tool_name,
            "fate_score": fate_result["composite"],
        }, decision="allow" if passed else "deny")

        if not passed:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }

        # Closed-loop advisory: query ReasoningBank for strategy before execution
        # Boyd OODA: Orient phase — match current context against learned patterns
        advice_context = ""
        if self.reasoning_bank:
            try:
                ctx = self._extract_tool_context(tool_name, tool_input)
                strategy = self.reasoning_bank.recommend_strategy(
                    tool_name.lower(), ctx
                )
                if strategy.get("confidence", 0) >= 0.70:
                    advice_context = (
                        f"ReasoningBank: {strategy['best_approach']} "
                        f"(score={strategy.get('score', 0):.2f}, "
                        f"conf={strategy['confidence']:.2f})"
                    )
                patterns = self.reasoning_bank.match_patterns(
                    {"tool": tool_name, **ctx}
                )
                if patterns and patterns[0].get("confidence", 0) >= 0.70:
                    p = patterns[0]
                    advice_context += (
                        f" | Pattern: {','.join(p.get('actions', [])[:2])}"
                    )
            except Exception as e:
                log_entry("hook_errors.jsonl", {
                    "timestamp": now_utc(),
                    "event": "pre_tool_advisory",
                    "error": str(e),
                })

        if advice_context:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": advice_context,
                }
            }
        return {}

    @staticmethod
    def _extract_tool_context(tool_name: str, tool_input: dict) -> dict:
        """Extract meaningful context from tool input for pattern matching."""
        ctx: dict = {"tool": tool_name}
        if tool_name in ("Edit", "Write", "MultiEdit"):
            path = tool_input.get("file_path", tool_input.get("path", ""))
            if path:
                ext = path.rsplit(".", 1)[-1] if "." in path else "unknown"
                ctx["file_ext"] = ext
                ctx["language"] = {
                    "py": "python", "rs": "rust", "ts": "typescript",
                    "tsx": "typescript", "js": "javascript", "json": "json",
                    "toml": "toml", "yml": "yaml", "yaml": "yaml",
                    "md": "markdown", "sh": "shell",
                }.get(ext, ext)
        elif tool_name == "Bash":
            cmd = tool_input.get("command", "")
            if cmd:
                first_word = cmd.strip().split()[0] if cmd.strip() else ""
                ctx["command_type"] = first_word
        elif tool_name == "Task":
            ctx["subagent"] = True
        return ctx

    def handle_post_tool_use(self, input_data: dict) -> dict:
        """Handle PostToolUse event with NTU monitoring + ReasoningBank learning."""
        tool_name = input_data.get("tool_name", "unknown")
        tool_input = input_data.get("tool_input", {})
        tool_result = input_data.get("tool_response", {})

        score = self.ntu_monitor.compute_tool_score(tool_name, tool_result)
        belief, anomaly = self.ntu_monitor.observe(score)

        self.perf_tracker.record_operation(tool_name, 0, True)

        log_entry("ntu_observations.jsonl", {
            "timestamp": now_utc(),
            "tool": tool_name,
            "score": score,
            "belief": belief,
            "anomaly": anomaly,
        })

        output = {}
        if anomaly:
            output["hookSpecificOutput"] = {
                "hookEventName": "PostToolUse",
                "additionalContext": f"NTU anomaly: belief={belief:.3f}",
            }

        # ReasoningBank: record experience with rich context (closed-loop learning)
        # Deming PDCA: Check phase — record actual outcome for strategy evaluation
        if self.reasoning_bank:
            try:
                ctx = self._extract_tool_context(tool_name, tool_input)
                ctx["anomaly"] = anomaly
                approach = self._infer_approach(tool_name, tool_input)
                self.reasoning_bank.record_experience(
                    task=tool_name.lower(),
                    approach=approach,
                    outcome={"success": True, "metrics": {"quality_score": score}},
                    context=ctx,
                )
            except Exception as e:
                log_entry("hook_errors.jsonl", {
                    "timestamp": now_utc(),
                    "event": "post_tool_record",
                    "error": str(e),
                })

        return output

    @staticmethod
    def _infer_approach(tool_name: str, tool_input: dict) -> str:
        """Infer the approach/strategy from tool input for richer learning."""
        if tool_name in ("Edit", "Write", "MultiEdit"):
            path = tool_input.get("file_path", tool_input.get("path", ""))
            ext = path.rsplit(".", 1)[-1] if "." in path else "unknown"
            return f"edit_{ext}"
        elif tool_name == "Bash":
            cmd = tool_input.get("command", "")
            first = cmd.strip().split()[0] if cmd.strip() else "unknown"
            return f"bash_{first}"
        elif tool_name == "Task":
            return "subagent_dispatch"
        elif tool_name in ("Read", "Glob", "Grep"):
            return f"search_{tool_name.lower()}"
        return "default"

    def handle_post_tool_failure(self, input_data: dict) -> dict:
        """Handle PostToolUseFailure event."""
        tool_name = input_data.get("tool_name", "unknown")
        error = input_data.get("error", "")

        analysis = self.self_healer.analyze_error(error)
        self.perf_tracker.record_operation(tool_name, 0, False)

        self.audit_logger.log("tool_failure", {
            "tool_name": tool_name,
            "error_type": analysis["error_type"] if analysis else "unknown",
        }, decision="logged")

        output = {}
        if analysis:
            output["hookSpecificOutput"] = {
                "hookEventName": "PostToolUseFailure",
                "additionalContext": f"Suggested fix: {analysis['suggested_fix']}",
            }

        # ReasoningBank: record failure with rich context for pattern learning
        if self.reasoning_bank:
            try:
                ctx = self._extract_tool_context(tool_name, tool_input if 'tool_input' in dir() else {})
                approach = self._infer_approach(tool_name, input_data.get("tool_input", {}))
                self.reasoning_bank.record_experience(
                    task=tool_name.lower(),
                    approach=approach,
                    outcome={
                        "success": False,
                        "metrics": {"error_count": 1},
                    },
                    context={
                        **ctx,
                        "error_type": analysis["error_type"] if analysis else "unknown",
                    },
                )
            except Exception as e:
                log_entry("hook_errors.jsonl", {
                    "timestamp": now_utc(),
                    "event": "post_tool_failure_record",
                    "error": str(e),
                })

        return output

    def handle_stop(self, input_data: dict) -> dict:
        """Handle Stop event."""
        session_id = input_data.get("session_id", "unknown")
        stop_hook_active = input_data.get("stop_hook_active", False)

        if stop_hook_active:
            return {}

        perf_summary = self.perf_tracker.get_summary()
        self.memory_sync.save_session_context(session_id, {
            "summary": f"Ops: {perf_summary['total_operations']}, Success: {perf_summary['success_rate']:.1%}",
        })

        self.audit_logger.log("session_stop", {
            "session_id": session_id,
            "operations": perf_summary["total_operations"],
        })
        return {}

    def handle_subagent_stop(self, input_data: dict) -> dict:
        """Handle SubagentStop event."""
        self.audit_logger.log("subagent_stop", {
            "agent_id": input_data.get("agent_id", "unknown"),
            "agent_type": input_data.get("agent_type", "unknown"),
        })
        return {}

    def handle_notification(self, input_data: dict) -> dict:
        """Handle Notification event."""
        log_entry("notifications.jsonl", {
            "timestamp": now_utc(),
            "type": input_data.get("notification_type", "unknown"),
            "message": input_data.get("message", "")[:200],
        })
        return {}

    def handle_pre_compact(self, input_data: dict) -> dict:
        """Handle PreCompact event."""
        session_id = input_data.get("session_id", "unknown")
        self.memory_sync.save_session_context(session_id, {
            "summary": "Pre-compaction snapshot",
            "trigger": input_data.get("trigger", "auto"),
        })
        return {}

    def handle_user_prompt_submit(self, input_data: dict) -> dict:
        """Handle UserPromptSubmit event - validate/enhance prompts."""
        prompt = input_data.get("prompt", "")

        # Log prompt submission
        self.audit_logger.log("user_prompt", {
            "prompt_length": len(prompt),
            "prompt_preview": prompt[:100] if prompt else "",
        })

        # Add context based on prompt analysis
        context = None

        # Detect if prompt mentions specific BIZRA components
        bizra_keywords = ["sovereign", "federation", "ihsan", "snr", "fate", "omega"]
        for keyword in bizra_keywords:
            if keyword.lower() in prompt.lower():
                context = f"BIZRA context: {keyword} module referenced"
                break

        output = {}
        if context:
            output["hookSpecificOutput"] = {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": context,
            }
        return output

    def handle_permission_request(self, input_data: dict) -> dict:
        """Handle PermissionRequest event - auto-approve safe operations."""
        tool_name = input_data.get("tool_name", "unknown")
        tool_input = input_data.get("tool_input", {})

        self.audit_logger.log("permission_request", {
            "tool_name": tool_name,
        })

        # Auto-approve safe read-only operations
        safe_tools = ["Read", "Glob", "Grep", "LS"]
        if tool_name in safe_tools:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PermissionRequest",
                    "decision": {
                        "behavior": "allow",
                    }
                }
            }

        # For other tools, let user decide (no output = ask user)
        return {}

    def handle_subagent_start(self, input_data: dict) -> dict:
        """Handle SubagentStart event - inject context into subagents."""
        agent_id = input_data.get("agent_id", "unknown")
        agent_type = input_data.get("agent_type", "unknown")

        self.audit_logger.log("subagent_start", {
            "agent_id": agent_id,
            "agent_type": agent_type,
        })

        # Inject BIZRA context for subagents
        context = f"BIZRA subagent ({agent_type}): Follow Ihsan threshold >= 0.95"

        return {
            "hookSpecificOutput": {
                "hookEventName": "SubagentStart",
                "additionalContext": context,
            }
        }

    def handle_session_end(self, input_data: dict) -> dict:
        """Handle SessionEnd event - cleanup and final persistence."""
        session_id = input_data.get("session_id", "unknown")
        reason = input_data.get("reason", "other")

        # Final state save
        perf_summary = self.perf_tracker.get_summary()
        self.memory_sync.save_session_context(session_id, {
            "summary": f"Session ended ({reason}). Ops: {perf_summary['total_operations']}",
            "end_reason": reason,
            "performance": perf_summary,
        })

        self.audit_logger.log("session_end", {
            "session_id": session_id,
            "reason": reason,
            "operations": perf_summary["total_operations"],
        })

        # ReasoningBank: consolidate patterns at session end
        # Deming PDCA: Act phase — update strategies from accumulated experience
        if self.reasoning_bank:
            try:
                self.reasoning_bank.consolidate()
            except Exception as e:
                log_entry("hook_errors.jsonl", {
                    "timestamp": now_utc(),
                    "event": "session_end_consolidate",
                    "error": str(e),
                })

        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point for hook invocation."""
    if len(sys.argv) < 2:
        print("Usage: bizra_hook_engine.py <event>", file=sys.stderr)
        sys.exit(1)

    event = sys.argv[1]

    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        input_data = {}

    engine = BizraHookEngine()

    handlers = {
        # Session lifecycle (3 events)
        "session_start": engine.handle_session_start,
        "session_end": engine.handle_session_end,
        "pre_compact": engine.handle_pre_compact,
        # User interaction (2 events)
        "user_prompt_submit": engine.handle_user_prompt_submit,
        "permission_request": engine.handle_permission_request,
        # Tool lifecycle (3 events)
        "pre_tool": engine.handle_pre_tool_use,
        "post_tool": engine.handle_post_tool_use,
        "post_tool_failure": engine.handle_post_tool_failure,
        # Agent lifecycle (2 events)
        "subagent_start": engine.handle_subagent_start,
        "subagent_stop": engine.handle_subagent_stop,
        # System events (2 events)
        "stop": engine.handle_stop,
        "notification": engine.handle_notification,
    }

    handler = handlers.get(event)
    if not handler:
        print(f"Unknown event: {event}", file=sys.stderr)
        sys.exit(1)

    try:
        result = handler(input_data)
        if result:
            print(json.dumps(result))
    except Exception as e:
        log_entry("hook_errors.jsonl", {
            "timestamp": now_utc(),
            "event": event,
            "error": str(e),
        })
        sys.exit(0)


if __name__ == "__main__":
    main()
