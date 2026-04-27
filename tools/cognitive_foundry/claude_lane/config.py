"""Configuration for the Claude Cognitive Archive Pilot.

All thresholds, topic heuristics, and regex patterns live here so downstream
stages can reference a single source of truth. Override by editing this file
before running the pipeline, or by passing --config /path/to/override.py which
is import-exec'd in run_pipeline.py.

Rationale: stdlib-only, no YAML dependency.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class DistillationThresholds:
    """Length and pattern thresholds for Stage 2 extraction heuristics."""

    # Fact candidates — short declarative statements from the user.
    fact_min_chars: int = 20
    fact_max_chars: int = 300

    # Reasoning exemplars — long multi-step turns.
    exemplar_min_chars: int = 1500
    exemplar_requires_markers: bool = True  # require 'because'/'therefore'/numbered list

    # Decision candidates — stronger user-intent signals.
    decision_min_chars: int = 15
    decision_max_chars: int = 500


@dataclass(frozen=True)
class AdjudicationThresholds:
    """Clustering + hypothesis-flagging thresholds for Stage 3."""

    # Cluster by normalized-text hash (trivial) + optional keyword overlap.
    # Anything below this cosine-free Jaccard keyword-overlap is NOT grouped.
    keyword_jaccard_min: float = 0.6

    # Hypothesis flag: if a claim appears N or fewer times, mark as hypothesis.
    hypothesis_max_occurrences: int = 1

    # Obsolete flag: if a cluster has an older fact contradicted by a newer fact,
    # mark the older one obsolete.
    # Age delta in days required to prefer the newer one as canonical.
    obsolete_newer_preferred_days: int = 7


@dataclass(frozen=True)
class TopSignalConfig:
    """Which conversations count as 'top signal' in Stage 1."""

    # Rank by (turn count × user-message ratio). Top-K emitted.
    top_k: int = 50
    min_turns: int = 4


@dataclass(frozen=True)
class TopicBucket:
    """A single topic bucket with keyword triggers."""

    name: str
    keywords: List[str] = field(default_factory=list)


# Default topic buckets. Triggered by simple case-insensitive keyword match on
# conversation name + project name + message text (sampled). Counts are
# per-conversation, not per-message; a conversation belongs to each bucket it
# matches (buckets are not mutually exclusive).
DEFAULT_TOPIC_BUCKETS: List[TopicBucket] = [
    TopicBucket(
        name="bizra_architecture",
        keywords=[
            "bizra", "dema", "node0", "mission", "receipt",
            "pat-7", "sat-5", "ihsan", "fate gate", "snr",
            "pci", "hyperblocktree", "seed", "bloom",
        ],
    ),
    TopicBucket(
        name="validation_sprint",
        keywords=[
            "validation", "interview", "a3", "wedge", "signed record",
            "guardrail", "hypothesis", "founder prep", "cardinal",
        ],
    ),
    TopicBucket(
        name="strategy_audit",
        keywords=[
            "strategy", "market audit", "archetype", "mission inventory",
            "wedge decision", "competitor", "business model",
        ],
    ),
    TopicBucket(
        name="code_implementation",
        keywords=[
            "rust", "python", "implement", "refactor", "pr #",
            "commit", "branch", "pytest", "cargo", "lint", "ci gate",
        ],
    ),
    TopicBucket(
        name="personal_context",
        keywords=[
            "mumu", "family", "dubai", "travel", "health",
            "routine", "sleep", "workstation",
        ],
    ),
    TopicBucket(
        name="llm_tools_meta",
        keywords=[
            "claude code cli", "chatgpt", "openai", "anthropic",
            "ollama", "lm studio", "gemini", "context window",
        ],
    ),
    # Added 2026-04-24 after first real run revealed 575/806 uncategorized.
    TopicBucket(
        name="os_system_setup",
        keywords=[
            "linux", "ubuntu", "wsl", "partition", "raid", "boot ",
            "grub", "efi", "systemd", "kernel", "driver", "dual boot",
            "install", "msi titan", "nvme", "bios", "uefi",
            "fstab", "mount ", "udev", "dual-boot",
        ],
    ),
    TopicBucket(
        name="cli_workflow",
        keywords=[
            "command reference", "cli ", "shell", " bash", " zsh",
            "terminal", "shortcut", "keybind", "alias ", "tmux",
            "vim", "neovim", "emacs", "hotkey", "keyboard shortcut",
            "i3wm", "dotfiles", "command syntax",
        ],
    ),
    TopicBucket(
        name="research_knowledge",
        keywords=[
            "reverse engineer", "arxiv", "paper", "benchmark",
            "experiment", "literature", "citation", "persistence",
            "semantic", "synonym", "embedding", "ontology",
            "taxonomy", "reference",
        ],
    ),
    # A catch-all bucket is NOT added automatically; unmatched conversations
    # get counted in a synthetic 'uncategorized' bucket by the inventory stage.
]


# Regex patterns for Stage 2 heuristic extraction. Kept plain ASCII; multi-line
# not required for line-level detection. Each pattern is documented inline with
# what kind of candidate it produces.

FACT_PATTERNS: List[str] = [
    # Self-declaration
    r"\bI\s+(am|'m)\s+([a-zA-Z][^\.\?\!]{5,180})[\.!]",
    r"\bMy\s+(name|role|job|company|tool|workstation|workflow)\s+(is|are)\s+([^\.\?\!]{3,180})[\.!]",
    # Named-entity is-a  (tightened 2026-04-24: pronouns/articles/determiners excluded
    # via negative lookahead to stop pattern from treating "It"/"This"/"These" etc. as
    # entities.)
    r"\b(?!(?:It|This|That|These|Those|He|She|They|We|I|You|There|Here|My|Our|Your|Its|Their|A|An|The)\b)([A-Z][A-Za-z0-9_]+)\s+is\s+(a|an|the)\s+([^\.\?\!]{3,180})[\.!]",
    # Location / timezone anchors
    r"\bI\s+(live|work|am based)\s+in\s+([A-Z][a-zA-Z]+)[\.,!]",
]

DECISION_PATTERNS: List[str] = [
    # Explicit decision markers
    r"\bI\s+(decided|chose|will|am going to)\s+([^\.\?\!]{5,300})[\.!]",
    r"\b(Let'?s|We'?ll|We\s+will)\s+([^\.\?\!]{5,300})[\.!]",
    # Directive language
    r"\b(Go\s+with|Choose|Use|Pick|Adopt)\s+([^\.\?\!]{3,200})[\.!]",
    # Explicit plan
    r"\bThe\s+(plan|approach|decision)\s+is\s+([^\.\?\!]{3,300})[\.!]",
]

REASONING_MARKERS: List[str] = [
    "because", "therefore", "so that", "which means",
    "consequently", "as a result", "given that",
    "the reason is", "that implies", "it follows that",
]


@dataclass(frozen=True)
class Config:
    """Top-level configuration. Immutable dataclass for determinism."""

    distillation: DistillationThresholds = field(default_factory=DistillationThresholds)
    adjudication: AdjudicationThresholds = field(default_factory=AdjudicationThresholds)
    top_signal: TopSignalConfig = field(default_factory=TopSignalConfig)

    topic_buckets: List[TopicBucket] = field(default_factory=lambda: list(DEFAULT_TOPIC_BUCKETS))

    fact_patterns: List[str] = field(default_factory=lambda: list(FACT_PATTERNS))
    decision_patterns: List[str] = field(default_factory=lambda: list(DECISION_PATTERNS))
    reasoning_markers: List[str] = field(default_factory=lambda: list(REASONING_MARKERS))

    # Provenance discipline
    include_assistant_text_in_distillation: bool = False
    # False = only extract facts/decisions from user messages (more disciplined).
    # True = also scan assistant turns (useful for reasoning exemplars only).

    # Added 2026-04-24: the first real run produced 4651 reasoning exemplars
    # dominated by Claude's own long replies. When True, only human turns are
    # considered for exemplar extraction — the reviewer sees their own reasoning,
    # not Claude's.
    reasoning_exemplars_human_only: bool = True

    # Deterministic output
    csv_line_terminator: str = "\n"
    csv_encoding: str = "utf-8"


def default_config() -> Config:
    """Return the default Config. Callers can copy + mutate if needed."""
    return Config()
