#!/usr/bin/env python3
"""
ReasoningBank — Adaptive Learning for BIZRA Hook System
========================================================
Implements the 4-step learning loop: RETRIEVE → JUDGE → DISTILL → CONSOLIDATE

Standing on Giants: Kahneman (dual-process, 2011) · Boyd (OODA, 1976) · Deming (PDCA, 1950)

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                         REASONING BANK                              │
├──────────────┬──────────────┬──────────────┬───────────────────────┤
│   RETRIEVE   │    JUDGE     │   DISTILL    │     CONSOLIDATE       │
│  Past exps   │  Score new   │  Extract     │  Deduplicate,         │
│  by context  │  outcomes    │  strategies  │  prune, merge         │
├──────────────┴──────────────┴──────────────┴───────────────────────┤
│                    experiences.jsonl (append-only)                   │
│                    strategies.json   (optimized)                     │
│                    patterns.json     (learned)                       │
└─────────────────────────────────────────────────────────────────────┘

Usage:
    from reasoning_bank import ReasoningBank
    rb = ReasoningBank()
    rb.record_experience(task="edit", approach="pre-edit-routing", outcome={"success": True}, context={...})
    strategy = rb.recommend_strategy("edit", {"language": "python", "complexity": "high"})
"""

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS — Aligned with core/integration/constants.py thresholds
# ═══════════════════════════════════════════════════════════════════════

IHSAN_THRESHOLD = 0.95
LEARNING_CONFIDENCE_FLOOR = 0.70  # Only learn from outcomes above this
PATTERN_MIN_OCCURRENCES = 3       # Minimum repeats to form a pattern
STRATEGY_DECAY_RATE = 0.02        # 2%/month aligned with BLOOM decay
MAX_EXPERIENCES = 10000           # Prune beyond this
MAX_STRATEGIES = 200              # Cap strategy count
MAX_PATTERNS = 500                # Cap pattern count


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(content: str, length: int = 12) -> str:
    return hashlib.blake2b(content.encode(), digest_size=length).hexdigest()


class ReasoningBank:
    """Adaptive learning system for hook-level pattern recognition and strategy optimization."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.environ.get(
                "CLAUDE_PROJECT_DIR",
                str(Path(__file__).resolve().parent.parent.parent)
            )
        self.state_dir = Path(base_dir) / ".claude" / "state" / "reasoning_bank"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.experiences_file = self.state_dir / "experiences.jsonl"
        self.strategies_file = self.state_dir / "strategies.json"
        self.patterns_file = self.state_dir / "patterns.json"
        self.metrics_file = self.state_dir / "metrics.json"

        self._strategies = self._load_json(self.strategies_file, {})
        self._patterns = self._load_json(self.patterns_file, {})
        self._metrics = self._load_json(self.metrics_file, {
            "total_experiences": 0,
            "patterns_learned": 0,
            "strategy_success_rate": 0.0,
            "recommendations_made": 0,
            "last_consolidation": None,
        })

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: RETRIEVE — Find relevant past experiences
    # ═══════════════════════════════════════════════════════════════════

    def retrieve(self, task_type: str, context: Optional[Dict] = None,
                 limit: int = 20) -> List[Dict]:
        """Retrieve past experiences matching task type and context."""
        matches = []
        if not self.experiences_file.exists():
            return matches

        with open(self.experiences_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    exp = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if exp.get("task") != task_type:
                    continue

                score = self._context_similarity(exp.get("context", {}), context or {})
                exp["_match_score"] = score
                matches.append(exp)

        matches.sort(key=lambda x: x["_match_score"], reverse=True)
        return matches[:limit]

    def _context_similarity(self, ctx_a: Dict, ctx_b: Dict) -> float:
        """Simple key-overlap similarity (0.0-1.0)."""
        if not ctx_a or not ctx_b:
            return 0.0
        keys_a = set(ctx_a.keys())
        keys_b = set(ctx_b.keys())
        shared = keys_a & keys_b
        if not shared:
            return 0.0
        match_count = sum(1 for k in shared if ctx_a[k] == ctx_b[k])
        return match_count / max(len(keys_a | keys_b), 1)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: JUDGE — Score and evaluate outcomes
    # ═══════════════════════════════════════════════════════════════════

    def judge(self, outcome: Dict) -> float:
        """Score an outcome on the Ihsan scale (0.0-1.0)."""
        success = 1.0 if outcome.get("success", False) else 0.0
        metrics = outcome.get("metrics", {})

        components = [success]
        if "duration_ms" in metrics:
            # Faster is better: score inversely proportional to time
            duration = max(metrics["duration_ms"], 1)
            time_score = min(1.0, 5000.0 / duration)  # 5s = 1.0
            components.append(time_score)
        if "error_count" in metrics:
            err_score = max(0.0, 1.0 - metrics["error_count"] * 0.2)
            components.append(err_score)
        if "quality_score" in metrics:
            components.append(min(1.0, max(0.0, metrics["quality_score"])))

        # Geometric mean (aligned with 8D Ihsan tensor approach)
        if not components:
            return 0.0
        product = 1.0
        for c in components:
            product *= max(c, 0.001)  # Avoid zero
        return product ** (1.0 / len(components))

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: DISTILL — Extract strategy from experiences
    # ═══════════════════════════════════════════════════════════════════

    def distill(self, task_type: str) -> Optional[Dict]:
        """Extract the best strategy for a task type from accumulated experiences."""
        experiences = self.retrieve(task_type, limit=100)
        if len(experiences) < PATTERN_MIN_OCCURRENCES:
            return None

        # Group by approach
        approach_stats: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0, "total_score": 0.0, "successes": 0, "failures": 0
        })

        for exp in experiences:
            approach = exp.get("approach", "unknown")
            score = self.judge(exp.get("outcome", {}))
            stats = approach_stats[approach]
            stats["count"] += 1
            stats["total_score"] += score
            if exp.get("outcome", {}).get("success"):
                stats["successes"] += 1
            else:
                stats["failures"] += 1

        # Rank approaches by average score
        ranked = []
        for approach, stats in approach_stats.items():
            avg_score = stats["total_score"] / max(stats["count"], 1)
            success_rate = stats["successes"] / max(stats["count"], 1)
            confidence = min(1.0, stats["count"] / 10.0)  # More data = more confident
            ranked.append({
                "approach": approach,
                "avg_score": round(avg_score, 4),
                "success_rate": round(success_rate, 4),
                "confidence": round(confidence, 4),
                "sample_size": stats["count"],
            })

        ranked.sort(key=lambda x: x["avg_score"] * x["confidence"], reverse=True)

        if not ranked:
            return None

        best = ranked[0]
        strategy = {
            "task_type": task_type,
            "best_approach": best["approach"],
            "score": best["avg_score"],
            "success_rate": best["success_rate"],
            "confidence": best["confidence"],
            "alternatives": ranked[1:5],
            "distilled_at": _now_iso(),
        }

        # Store strategy
        self._strategies[task_type] = strategy
        self._save_json(self.strategies_file, self._strategies)
        return strategy

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: CONSOLIDATE — Deduplicate, prune, merge patterns
    # ═══════════════════════════════════════════════════════════════════

    def consolidate(self) -> Dict:
        """Prune old/low-confidence data, deduplicate patterns, update metrics."""
        result = {"pruned_experiences": 0, "merged_patterns": 0, "strategies_updated": 0}

        # Prune experiences if over limit
        if self.experiences_file.exists():
            lines = self.experiences_file.read_text().strip().split("\n")
            if len(lines) > MAX_EXPERIENCES:
                # Keep most recent
                pruned = lines[-MAX_EXPERIENCES:]
                self.experiences_file.write_text("\n".join(pruned) + "\n")
                result["pruned_experiences"] = len(lines) - MAX_EXPERIENCES

        # Prune low-confidence patterns
        to_remove = []
        for pid, pattern in self._patterns.items():
            if pattern.get("confidence", 0) < LEARNING_CONFIDENCE_FLOOR:
                if pattern.get("occurrences", 0) < PATTERN_MIN_OCCURRENCES:
                    to_remove.append(pid)
        for pid in to_remove:
            del self._patterns[pid]
            result["merged_patterns"] += 1
        self._save_json(self.patterns_file, self._patterns)

        # Re-distill all known task types
        task_types = set(self._strategies.keys())
        for tt in task_types:
            strategy = self.distill(tt)
            if strategy:
                result["strategies_updated"] += 1

        # Update metrics
        exp_count = 0
        if self.experiences_file.exists():
            exp_count = sum(1 for _ in open(self.experiences_file))
        self._metrics["total_experiences"] = exp_count
        self._metrics["patterns_learned"] = len(self._patterns)
        self._metrics["last_consolidation"] = _now_iso()

        # Compute overall strategy success rate
        rates = [s.get("success_rate", 0) for s in self._strategies.values()]
        self._metrics["strategy_success_rate"] = round(
            sum(rates) / max(len(rates), 1), 4
        )
        self._save_json(self.metrics_file, self._metrics)

        return result

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    def record_experience(self, task: str, approach: str,
                          outcome: Dict, context: Optional[Dict] = None) -> Dict:
        """Record a task outcome for learning. Returns the scored experience."""
        score = self.judge(outcome)
        experience = {
            "id": _hash(f"{task}:{approach}:{time.time_ns()}", 8),
            "task": task,
            "approach": approach,
            "outcome": outcome,
            "context": context or {},
            "ihsan_score": round(score, 4),
            "timestamp": _now_iso(),
        }

        # Append to experiences log (append-only)
        with open(self.experiences_file, "a") as f:
            f.write(json.dumps(experience) + "\n")

        # Update pattern tracking
        self._update_patterns(task, approach, score, context or {})

        self._metrics["total_experiences"] = self._metrics.get("total_experiences", 0) + 1
        self._save_json(self.metrics_file, self._metrics)

        return experience

    def recommend_strategy(self, task_type: str,
                           context: Optional[Dict] = None) -> Dict:
        """Recommend the best strategy for a task type given context."""
        self._metrics["recommendations_made"] = self._metrics.get("recommendations_made", 0) + 1
        self._save_json(self.metrics_file, self._metrics)

        # Check cached strategy first
        if task_type in self._strategies:
            strategy = self._strategies[task_type]
            if strategy.get("confidence", 0) >= LEARNING_CONFIDENCE_FLOOR:
                return strategy

        # Try to distill from experiences
        distilled = self.distill(task_type)
        if distilled:
            return distilled

        # Fallback: retrieve similar and return best approach
        similar = self.retrieve(task_type, context, limit=10)
        if similar:
            best = max(similar, key=lambda x: x.get("ihsan_score", 0))
            return {
                "task_type": task_type,
                "best_approach": best.get("approach", "default"),
                "score": best.get("ihsan_score", 0),
                "confidence": 0.3,  # Low confidence — insufficient data
                "source": "single_experience_fallback",
            }

        return {
            "task_type": task_type,
            "best_approach": "default",
            "score": 0.0,
            "confidence": 0.0,
            "source": "no_data",
        }

    def learn_pattern(self, pattern_id: str, triggers: List[str],
                      actions: List[str], confidence: float,
                      context: Optional[Dict] = None) -> Dict:
        """Explicitly register a learned pattern."""
        pattern = {
            "id": pattern_id,
            "triggers": triggers,
            "actions": actions,
            "confidence": round(min(1.0, max(0.0, confidence)), 4),
            "occurrences": self._patterns.get(pattern_id, {}).get("occurrences", 0) + 1,
            "context": context or {},
            "learned_at": _now_iso(),
        }
        self._patterns[pattern_id] = pattern
        self._metrics["patterns_learned"] = len(self._patterns)
        self._save_json(self.patterns_file, self._patterns)
        self._save_json(self.metrics_file, self._metrics)
        return pattern

    def match_patterns(self, situation: Dict) -> List[Dict]:
        """Find patterns matching the current situation."""
        matches = []
        sit_triggers = set()
        for v in situation.values():
            if isinstance(v, str):
                sit_triggers.add(v.lower())
            elif isinstance(v, list):
                sit_triggers.update(str(x).lower() for x in v)

        for pid, pattern in self._patterns.items():
            pattern_triggers = set(t.lower() for t in pattern.get("triggers", []))
            overlap = sit_triggers & pattern_triggers
            if overlap:
                match_score = len(overlap) / max(len(pattern_triggers), 1)
                matches.append({
                    **pattern,
                    "_match_score": round(match_score, 4),
                })

        matches.sort(key=lambda x: x["_match_score"] * x.get("confidence", 0), reverse=True)
        return matches

    def get_metrics(self) -> Dict:
        """Return current learning metrics."""
        return {**self._metrics}

    def seed_from_performance_metrics(self, metrics_path: str) -> int:
        """Bootstrap from existing .claude/state/performance_metrics.json."""
        try:
            with open(metrics_path) as f:
                perf = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return 0

        count = 0
        tool_timings = perf.get("tool_timings", {})
        error_counts = perf.get("error_counts", {})

        for tool, timing in tool_timings.items():
            tool_count = timing.get("count", 0)
            errors = error_counts.get(tool, 0)
            successes = max(tool_count - errors, 0)
            success_rate = successes / max(tool_count, 1)

            self.record_experience(
                task=tool.lower().replace("mcp__filesystem__", "fs_"),
                approach="default",
                outcome={
                    "success": success_rate > 0.8,
                    "metrics": {
                        "quality_score": success_rate,
                        "error_count": errors,
                    },
                },
                context={
                    "tool": tool,
                    "total_invocations": tool_count,
                    "source": "performance_metrics_seed",
                },
            )
            count += 1

        return count

    # ═══════════════════════════════════════════════════════════════════
    # INTERNALS
    # ═══════════════════════════════════════════════════════════════════

    def _update_patterns(self, task: str, approach: str, score: float,
                         context: Dict) -> None:
        """Auto-detect patterns from repeated task+approach combos."""
        pid = _hash(f"{task}:{approach}", 8)
        existing = self._patterns.get(pid, {
            "id": pid,
            "triggers": [task],
            "actions": [approach],
            "confidence": 0.0,
            "occurrences": 0,
            "scores": [],
            "context": {},
        })

        existing["occurrences"] = existing.get("occurrences", 0) + 1
        scores = existing.get("scores", [])
        scores.append(score)
        if len(scores) > 50:
            scores = scores[-50:]
        existing["scores"] = scores

        # Update confidence based on consistency of scores
        if len(scores) >= PATTERN_MIN_OCCURRENCES:
            avg = sum(scores) / len(scores)
            variance = sum((s - avg) ** 2 for s in scores) / len(scores)
            std_dev = math.sqrt(variance) if variance > 0 else 0
            # High average + low variance = high confidence
            existing["confidence"] = round(min(1.0, avg * (1.0 - std_dev)), 4)
        existing["learned_at"] = _now_iso()

        self._patterns[pid] = existing
        self._save_json(self.patterns_file, self._patterns)

    def _load_json(self, path: Path, default: Any) -> Any:
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def _save_json(self, path: Path, data: Any) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# CLI interface — callable from hooks
# ═══════════════════════════════════════════════════════════════════════

def main():
    import sys

    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: reasoning_bank.py <command> [args]"}))
        sys.exit(1)

    cmd = sys.argv[1]
    rb = ReasoningBank()

    if cmd == "record":
        # reasoning_bank.py record <task> <approach> <success:0|1> [context_json]
        if len(sys.argv) < 5:
            print(json.dumps({"error": "Usage: record <task> <approach> <success:0|1>"}))
            sys.exit(1)
        task, approach, success_str = sys.argv[2], sys.argv[3], sys.argv[4]
        context = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
        exp = rb.record_experience(
            task=task,
            approach=approach,
            outcome={"success": success_str == "1"},
            context=context,
        )
        print(json.dumps(exp, indent=2))

    elif cmd == "recommend":
        # reasoning_bank.py recommend <task_type> [context_json]
        if len(sys.argv) < 3:
            print(json.dumps({"error": "Usage: recommend <task_type>"}))
            sys.exit(1)
        task_type = sys.argv[2]
        context = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        strategy = rb.recommend_strategy(task_type, context)
        print(json.dumps(strategy, indent=2))

    elif cmd == "consolidate":
        result = rb.consolidate()
        print(json.dumps(result, indent=2))

    elif cmd == "metrics":
        print(json.dumps(rb.get_metrics(), indent=2))

    elif cmd == "seed":
        metrics_path = sys.argv[2] if len(sys.argv) > 2 else str(
            Path(__file__).resolve().parent.parent / "state" / "performance_metrics.json"
        )
        count = rb.seed_from_performance_metrics(metrics_path)
        print(json.dumps({"seeded_experiences": count}))

    elif cmd == "advise":
        # reasoning_bank.py advise <tool_name> [context_json]
        # Returns strategy + patterns as compact JSON for PreToolUse additionalContext
        if len(sys.argv) < 3:
            print(json.dumps({"error": "Usage: advise <tool_name> [context_json]"}))
            sys.exit(1)
        tool_name = sys.argv[2]
        context = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        strategy = rb.recommend_strategy(tool_name.lower(), context)
        patterns = rb.match_patterns({"tool": tool_name, **context})
        # Compact advice format for hook injection
        advice_parts = []
        if strategy.get("confidence", 0) >= LEARNING_CONFIDENCE_FLOOR:
            advice_parts.append(
                f"Strategy({strategy['best_approach']}|"
                f"score={strategy.get('score', 0):.2f}|"
                f"conf={strategy['confidence']:.2f})"
            )
        if patterns:
            top = patterns[0]
            advice_parts.append(
                f"Pattern({top.get('id', '?')}|"
                f"actions={','.join(top.get('actions', [])[:2])}|"
                f"conf={top.get('confidence', 0):.2f})"
            )
        advice_text = " ".join(advice_parts) if advice_parts else ""
        print(json.dumps({
            "strategy": strategy,
            "patterns": patterns[:3],
            "advice": advice_text,
        }, indent=2))

    elif cmd == "match":
        # reasoning_bank.py match <situation_json>
        situation = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        matches = rb.match_patterns(situation)
        print(json.dumps(matches, indent=2))

    else:
        print(json.dumps({"error": f"Unknown command: {cmd}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
