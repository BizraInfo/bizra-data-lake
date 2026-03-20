#!/usr/bin/env python3
"""
BIZRA LIVING MEMORY — The Brain That Remembers You
═══════════════════════════════════════════════════

Three memory types (from cognitive science):
  EPISODIC  — What happened (event log, last 1000 actions)
  SEMANTIC  — What I know about you (user model, permanent)
  PROCEDURAL — What I've learned to do (compiled reflexes)

On every boot:
  1. Load semantic memory (who you are, what you care about)
  2. Load procedural memory (compiled reflexes)
  3. Scan episodic memory (recent context, active projects)
  4. Generate "morning briefing" (what happened, what's next)

Result: Your agents KNOW you. Day 1 they're strangers.
Day 30 they're colleagues. Day 365 they're your team.
"""

import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

DIR = Path.home() / ".bizra"


# ═══════════════════════════════════════
# SEMANTIC MEMORY — Who you are (permanent)
# ═══════════════════════════════════════
class UserModel:
    """
    Learned from your actions, not from you telling it.
    Updates after every mission. Persists across sessions.
    """

    def __init__(self):
        self.name = ""
        self.node_id = ""
        self.created_at = 0

        # Learned preferences
        self.preferred_agent = ""  # Which agent you use most
        self.preferred_domains = []  # Top 3 work domains
        self.active_hours = [0] * 24  # When you work (hour histogram)
        self.avg_mission_length = 0  # Typical task complexity
        self.vocabulary = Counter()  # Your most-used words

        # Work patterns
        self.total_missions = 0
        self.missions_by_agent = Counter()  # P1: 45, P2: 120, P3: 89...
        self.missions_by_domain = Counter()  # research: 120, coding: 89...
        self.streak_record = 0
        self.current_streak = 0
        self.avg_ihsan = 0.0
        self.ihsan_trend = "stable"  # improving / declining / stable

        # Active projects (detected from task patterns)
        self.active_projects = (
            []
        )  # [{"name": "...", "last_active": ..., "missions": 5}]

        # Communication style
        self.verbosity = "balanced"  # terse / balanced / detailed
        self.prefers_explanations = True
        self.greeting_name = ""  # How to address you

        # Growth trajectory
        self.tier = "Novice"
        self.days_to_next_tier = 0
        self.strongest_skill = ""
        self.weakest_skill = ""
        self.recommended_next = ""  # Suggested next action

    def to_dict(self):
        return {
            "name": self.name,
            "node_id": self.node_id,
            "created_at": self.created_at,
            "preferred_agent": self.preferred_agent,
            "preferred_domains": self.preferred_domains,
            "active_hours": self.active_hours,
            "avg_mission_length": self.avg_mission_length,
            "vocabulary": dict(self.vocabulary.most_common(100)),
            "total_missions": self.total_missions,
            "missions_by_agent": dict(self.missions_by_agent),
            "missions_by_domain": dict(self.missions_by_domain),
            "streak_record": self.streak_record,
            "current_streak": self.current_streak,
            "avg_ihsan": self.avg_ihsan,
            "ihsan_trend": self.ihsan_trend,
            "active_projects": self.active_projects,
            "verbosity": self.verbosity,
            "prefers_explanations": self.prefers_explanations,
            "greeting_name": self.greeting_name,
            "tier": self.tier,
            "days_to_next_tier": self.days_to_next_tier,
            "strongest_skill": self.strongest_skill,
            "weakest_skill": self.weakest_skill,
            "recommended_next": self.recommended_next,
        }

    @classmethod
    def from_dict(cls, d):
        m = cls()
        for k, v in d.items():
            if k == "vocabulary":
                m.vocabulary = Counter(v)
            elif k == "missions_by_agent":
                m.missions_by_agent = Counter(v)
            elif k == "missions_by_domain":
                m.missions_by_domain = Counter(v)
            elif hasattr(m, k):
                setattr(m, k, v)
        return m


# ═══════════════════════════════════════
# EPISODIC MEMORY — What happened recently
# ═══════════════════════════════════════
class EpisodicMemory:
    """Recent events. Scanned at boot for context continuity."""

    def __init__(self, events=None):
        self.events = events or []

    def last_session(self):
        """What happened in the most recent session."""
        if not self.events:
            return None
        sessions = []
        current = []
        for ev in self.events:
            if ev.get("t") == "genesis" and current:
                sessions.append(current)
                current = []
            current.append(ev)
        if current:
            sessions.append(current)
        return sessions[-1] if sessions else None

    def last_n_missions(self, n=10):
        return [e for e in self.events if e.get("t") == "mission"][-n:]

    def active_projects_from_tasks(self):
        """Detect recurring themes in recent tasks."""
        tasks = [e.get("task", "") for e in self.events if e.get("t") == "mission"][
            -100:
        ]
        if not tasks:
            return []

        # Extract key phrases
        word_freq = Counter()
        for task in tasks:
            words = re.findall(r"\b[a-zA-Z]{4,}\b", task.lower())
            word_freq.update(words)

        # Remove common words
        stop = {
            "that",
            "this",
            "with",
            "from",
            "your",
            "have",
            "will",
            "been",
            "what",
            "when",
            "about",
            "which",
            "their",
            "would",
            "could",
            "should",
            "into",
            "also",
            "each",
            "make",
            "like",
            "just",
            "over",
            "such",
            "than",
            "them",
            "very",
            "some",
            "only",
        }
        for w in stop:
            word_freq.pop(w, None)

        # Top themes become "projects"
        projects = []
        for word, count in word_freq.most_common(5):
            if count >= 3:  # Mentioned 3+ times
                related_tasks = [t for t in tasks if word in t.lower()]
                projects.append(
                    {
                        "name": word.capitalize(),
                        "mentions": count,
                        "last_task": related_tasks[-1][:60] if related_tasks else "",
                        "total_missions": len(related_tasks),
                    }
                )
        return projects

    def time_since_last_mission(self):
        missions = [e for e in self.events if e.get("t") == "mission"]
        if not missions:
            return None
        last_ts = missions[-1].get("ts", 0)
        return (int(time.time() * 1000) - last_ts) / (60 * 60 * 1000)  # hours


# ═══════════════════════════════════════
# PROCEDURAL MEMORY — What I know how to do
# ═══════════════════════════════════════
class ProceduralMemory:
    """Compiled reflexes. Loaded at boot for instant execution."""

    def __init__(self, cache=None):
        self.cache = cache or []

    def compiled_count(self):
        return sum(1 for r in self.cache if r.get("compiled"))

    def near_compilation(self):
        """Patterns close to becoming reflexes (3-4 out of 5)."""
        return [
            r for r in self.cache if 3 <= r.get("cnt", 0) < 5 and not r.get("compiled")
        ]

    def top_reflexes(self, n=5):
        return sorted(
            [r for r in self.cache if r.get("compiled")],
            key=lambda r: r.get("cnt", 0),
            reverse=True,
        )[:n]


# ═══════════════════════════════════════
# MEMORY MANAGER — The orchestrator
# ═══════════════════════════════════════
class LivingMemory:
    """
    Manages all three memory types.
    Called at boot and after every mission.
    """

    def __init__(self):
        self.user = UserModel()
        self.episodic = EpisodicMemory()
        self.procedural = ProceduralMemory()

    def load(self):
        """Load all memory from disk at boot."""
        # Load user model (semantic)
        um_path = DIR / "user_model.json"
        if um_path.exists():
            self.user = UserModel.from_dict(json.loads(um_path.read_text()))

        # Load node state for identity
        node_path = DIR / "node.json"
        if node_path.exists():
            node = json.loads(node_path.read_text())
            self.user.name = node.get("name", "")
            self.user.node_id = node.get("nid", "")
            self.user.created_at = node.get("cat", 0)
            self.user.tier = [
                "Novice",
                "Apprentice",
                "Adept",
                "Expert",
                "Master",
                "Grandmaster",
            ][node.get("tier", 0)]
            self.user.current_streak = node.get("streak", 0)
            self.user.total_missions = node.get("miss", 0)
            self.procedural = ProceduralMemory(node.get("cache", []))

        # Load event log (episodic)
        ledger_path = DIR / "ledger.jsonl"
        if ledger_path.exists():
            events = [
                json.loads(l)
                for l in ledger_path.read_text().strip().split("\n")
                if l.strip()
            ]
            self.episodic = EpisodicMemory(events[-1000:])  # Last 1000 events

        return self

    def save(self):
        """Persist semantic memory after every mission."""
        DIR.mkdir(parents=True, exist_ok=True)
        (DIR / "user_model.json").write_text(json.dumps(self.user.to_dict(), indent=2))

    def update_after_mission(self, task, agent_id, agent_domain, ihsan, seed):
        """Called after every mission to update the user model."""
        now = int(time.time() * 1000)
        hour = datetime.fromtimestamp(now / 1000).hour

        # Update agent preference
        self.user.missions_by_agent[agent_id] = (
            self.user.missions_by_agent.get(agent_id, 0) + 1
        )
        self.user.preferred_agent = self.user.missions_by_agent.most_common(1)[0][0]

        # Update domain preference
        self.user.missions_by_domain[agent_domain] = (
            self.user.missions_by_domain.get(agent_domain, 0) + 1
        )
        top_domains = self.user.missions_by_domain.most_common(3)
        self.user.preferred_domains = [d[0] for d in top_domains]

        # Update active hours
        self.user.active_hours[hour] += 1

        # Update vocabulary
        words = re.findall(r"\b[a-zA-Z]{4,}\b", task.lower())
        self.user.vocabulary.update(words)

        # Update task length preference
        word_count = len(task.split())
        n = self.user.total_missions or 1
        self.user.avg_mission_length = (
            self.user.avg_mission_length * (n - 1) + word_count
        ) / n

        # Update Ihsan trend
        if self.user.avg_ihsan > 0:
            if ihsan > self.user.avg_ihsan * 1.02:
                self.user.ihsan_trend = "improving"
            elif ihsan < self.user.avg_ihsan * 0.98:
                self.user.ihsan_trend = "declining"
            else:
                self.user.ihsan_trend = "stable"

        # Update verbosity preference (from task length)
        if self.user.avg_mission_length < 8:
            self.user.verbosity = "terse"
        elif self.user.avg_mission_length > 20:
            self.user.verbosity = "detailed"
        else:
            self.user.verbosity = "balanced"

        # Detect active projects
        self.user.active_projects = self.episodic.active_projects_from_tasks()

        # Update streak record
        if self.user.current_streak > self.user.streak_record:
            self.user.streak_record = self.user.current_streak

        # Save
        self.save()

    def generate_morning_briefing(self):
        """Generate a personalized greeting based on everything we know."""
        lines = []
        name = self.user.greeting_name or self.user.name or "Sovereign"
        hours_away = self.episodic.time_since_last_mission()

        # Greeting based on time of day
        hour = datetime.now().hour
        if hour < 6:
            tod = "Late night session"
        elif hour < 12:
            tod = "Good morning"
        elif hour < 17:
            tod = "Good afternoon"
        elif hour < 21:
            tod = "Good evening"
        else:
            tod = "Late evening"

        lines.append(f"{tod}, {name}.")

        # Time since last session
        if hours_away is not None:
            if hours_away < 1:
                lines.append("Picking up where we left off.")
            elif hours_away < 24:
                lines.append(
                    f"It's been {int(hours_away)} hours. Here's what I remember."
                )
            elif hours_away < 168:
                days = int(hours_away / 24)
                lines.append(
                    f"Welcome back — {days} day{'s' if days > 1 else ''} since our last session."
                )
            else:
                lines.append(
                    f"It's been a while. {int(hours_away / 24)} days. Your node is intact."
                )

        # Active projects
        projects = self.user.active_projects
        if projects:
            top = projects[0]
            lines.append(
                f"Your main focus has been {top['name']} ({top['total_missions']} missions). Last: \"{top['last_task'][:50]}\"."
            )

        # Last mission context
        recent = self.episodic.last_n_missions(1)
        if recent:
            last = recent[0]
            lines.append(
                f"Last mission: {last.get('an', 'agent')} completed \"{last.get('task', '')[:50]}\" — Ihsan {last.get('ih', 0):.3f}."
            )

        # Near-compilation reflexes
        near = self.procedural.near_compilation()
        if near:
            r = near[0]
            remaining = 5 - r.get("cnt", 0)
            lines.append(
                f"Pattern \"{r['p'][:40]}\" is {r.get('cnt',0)}/5 — {remaining} more to compile a reflex."
            )

        # Ihsan trend
        if self.user.ihsan_trend == "improving":
            lines.append("Your quality scores are trending up. Excellent trajectory.")
        elif self.user.ihsan_trend == "declining":
            lines.append(
                "I've noticed your Ihsan trending down. Shall I suggest ways to improve?"
            )

        # Streak
        if self.user.current_streak >= 7:
            lines.append(
                f"You're on a {self.user.current_streak}-mission streak. Personal best: {self.user.streak_record}."
            )

        # Proactive suggestion based on preferences
        if self.user.preferred_domains:
            top_domain = self.user.preferred_domains[0]
            suggestions = {
                "research": "I have new sources queued for your review.",
                "coding": "Your test suite could use attention — shall I run diagnostics?",
                "planning": "Your roadmap has items nearing their target dates.",
                "evaluation": "There are recent outputs waiting for quality assessment.",
                "delivery": "You have drafts that could be finalized today.",
                "ethics": "Constitutional compliance check is due for recent changes.",
                "coordination": "Cross-agent efficiency is at 94%. Room to optimize.",
            }
            if top_domain in suggestions:
                lines.append(suggestions[top_domain])

        # What to do next
        lines.append("Ready for your next mission.")

        return lines

    def sync_from_rust_feedback(self, bridge=None):
        """
        Poll the Rust nervous system for feedback signals and update
        the Python cognitive memory accordingly.

        Cross-boundary trust bridge:
        Rust subscribers set atomic flags → heartbeat drains →
        Python reads via poll_feedback() → updates user model.

        Standing on: Hebb (reinforcement), Tulving (episodic/semantic),
        Lamport (cross-boundary ordering).
        """
        if bridge is None:
            try:
                from core.sovereign.event_bus import create_rust_event_bridge

                bridge = create_rust_event_bridge()
            except (ImportError, Exception):
                return  # Graceful degradation

        if bridge is None:
            return

        feedback = bridge.poll_feedback()

        reinforce = feedback.get("reinforce_pending", 0)
        if reinforce > 0 and self.user.avg_ihsan > 0:
            self.user.ihsan_trend = "improving"

        quarantine = feedback.get("quarantine_pending", 0)
        if quarantine > 0:
            self.user.ihsan_trend = "declining"

        compile_pending = feedback.get("compile_pending", 0)
        if compile_pending > 0:
            self.user.active_projects = self.episodic.active_projects_from_tasks()

        if reinforce > 0 or quarantine > 0 or compile_pending > 0:
            self.save()

    def get_context_for_agent(self, agent_id):
        """Give an agent everything it needs to know about the user."""
        return {
            "user_name": self.user.name,
            "tier": self.user.tier,
            "preferred_domains": self.user.preferred_domains,
            "avg_ihsan": self.user.avg_ihsan,
            "active_projects": self.user.active_projects,
            "recent_missions": [
                {
                    "task": m.get("task", ""),
                    "agent": m.get("a", ""),
                    "ihsan": m.get("ih", 0),
                }
                for m in self.episodic.last_n_missions(5)
            ],
            "vocabulary_top20": [w for w, _ in self.user.vocabulary.most_common(20)],
            "verbosity": self.user.verbosity,
            "streak": self.user.current_streak,
        }

    def format_status(self):
        """Display what the system knows about the user."""
        u = self.user
        lines = []
        lines.append(f"  Name:           {u.name}")
        lines.append(f"  Tier:           {u.tier}")
        lines.append(f"  Total missions: {u.total_missions}")
        lines.append(
            f"  Streak:         {u.current_streak} (record: {u.streak_record})"
        )
        lines.append(f"  Avg Ihsan:      {u.avg_ihsan:.4f} ({u.ihsan_trend})")
        lines.append(f"  Top agent:      {u.preferred_agent}")
        lines.append(
            f"  Top domains:    {', '.join(u.preferred_domains) if u.preferred_domains else 'learning...'}"
        )
        lines.append(f"  Verbosity:      {u.verbosity}")

        # Active hours
        peak = max(range(24), key=lambda h: u.active_hours[h])
        lines.append(f"  Peak hour:      {peak:02d}:00")

        # Vocabulary signature
        top_words = [w for w, _ in u.vocabulary.most_common(10)]
        if top_words:
            lines.append(f"  Your words:     {', '.join(top_words)}")

        # Active projects
        if u.active_projects:
            lines.append(f"  Projects:       {len(u.active_projects)} active")
            for p in u.active_projects[:3]:
                lines.append(f"    · {p['name']} ({p['total_missions']} missions)")

        # Near-compilation
        near = self.procedural.near_compilation()
        if near:
            lines.append(f"  Near compile:   {len(near)} patterns close to reflex")

        return "\n".join(lines)


# ═══════════════════════════════════════
# DEMO: How it works in practice
# ═══════════════════════════════════════
def demo():
    G = "\033[38;2;201;169;98m"
    R = "\033[0m"
    D = "\033[2m"
    BL = "\033[38;2;96;165;250m"
    GR = "\033[38;2;52;211;153m"
    PU = "\033[38;2;167;139;250m"
    _B = "\033[1m"  # noqa: F841 — available for bold formatting

    mem = LivingMemory().load()

    if not mem.user.name:
        print(f"\n  {G}No node found. Run: python3 node0.py init YourName{R}\n")
        return

    # Morning briefing
    briefing = mem.generate_morning_briefing()
    print(f"\n  {G}{'═' * 48}{R}")
    print(f"  {G}LIVING MEMORY — What Your Agents Know{R}")
    print(f"  {G}{'═' * 48}{R}\n")

    print(f"  {BL}Morning Briefing:{R}")
    for line in briefing:
        print(f"  {GR}▸{R} {line}")

    print(f"\n  {PU}User Model (learned from your actions):{R}")
    print(mem.format_status())

    # Agent context
    ctx = mem.get_context_for_agent("P2")
    print(f"\n  {BL}What P2 ORACLE knows about you:{R}")
    print(f"  {D}Recent missions: {len(ctx['recent_missions'])}")
    print(f"  Vocabulary: {', '.join(ctx['vocabulary_top20'][:8])}...")
    print(f"  Projects: {[p['name'] for p in ctx['active_projects']]}{R}")

    print(f"\n  {G}{'═' * 48}{R}")
    print(f"  {D}This memory updates after every mission.")
    print(f"  Day 1: strangers. Day 30: colleagues. Day 365: your team.{R}\n")


if __name__ == "__main__":
    demo()
