"""
سلسلة البذرة — The Seed Chain v1.0

Constitutional prompt architecture for BIZRA's PAT-7 agent system.
Every prompt routed through DEMA follows six links, each mapped to
the governed execution pipeline.

Links:
    1. نِيَّة  (Niyyah)   — Intent, not role. Purpose over persona.
    2. بَيِّنَة (Bayyinah) — Evidence. Every fact tagged VERIFIED/PLANNED/DERIVED/UNKNOWN.
    3. حَدّ    (Hadd)     — Boundary. Constitutional negation.
    4. أَمَانَة (Amanah)   — Trust contract. Reasoning under oath.
    5. ثَمَرَة  (Thamara)  — Fruit. Output with evidence inheritance.
    6. إِيصَال (Iisal)    — Receipt. Verification loop -> feeds back to Niyyah.

The key innovation: Iisal feeds back into Niyyah (autopoietic closure).
Linear chains die at the output. The Seed Chain grows.

Created: 2026-03-22 | BIZRA-LAB
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bizra.seed_chain")


# ═══════════════════════════════════════════════════════════════
# Evidence Classification (CLAIM_MUST_BIND at prompt level)
# ═══════════════════════════════════════════════════════════════


class EvidenceTag(str, Enum):
    """Every fact in a Seed Chain must carry one of these tags."""

    VERIFIED = "VERIFIED"
    PLANNED = "PLANNED"
    DERIVED = "DERIVED"
    UNKNOWN = "UNKNOWN"


# ═══════════════════════════════════════════════════════════════
# Link 1: نِيَّة (Niyyah) — Intent
# ═══════════════════════════════════════════════════════════════


@dataclass
class Niyyah:
    """Intent declaration. Not a role — a purpose."""

    purpose: str
    requester: str = "user"
    target_agent: str = "P7_DEMA"
    urgency: str = "normal"
    context_keys: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Link 2: بَيِّنَة (Bayyinah) — Evidence
# ═══════════════════════════════════════════════════════════════


@dataclass
class BayyinahItem:
    """A single evidence-tagged fact."""

    claim: str
    tag: EvidenceTag
    source: str = ""
    confidence: float = 1.0


@dataclass
class Bayyinah:
    """Evidence bundle. No untagged claims allowed."""

    items: List[BayyinahItem] = field(default_factory=list)

    def add(
        self, claim: str, tag: EvidenceTag, source: str = "", confidence: float = 1.0
    ) -> None:
        if tag == EvidenceTag.UNKNOWN:
            confidence = min(confidence, 0.5)
        if tag == EvidenceTag.DERIVED:
            confidence = min(confidence, 0.9)
        self.items.append(
            BayyinahItem(
                claim=claim,
                tag=tag,
                source=source,
                confidence=confidence,
            )
        )

    @property
    def verified_count(self) -> int:
        return sum(1 for i in self.items if i.tag == EvidenceTag.VERIFIED)

    @property
    def unknown_count(self) -> int:
        return sum(1 for i in self.items if i.tag == EvidenceTag.UNKNOWN)


# ═══════════════════════════════════════════════════════════════
# Link 3: حَدّ (Hadd) — Boundary
# ═══════════════════════════════════════════════════════════════


@dataclass
class Hadd:
    """Constitutional negation — what the agent CANNOT do."""

    prohibitions: List[str] = field(default_factory=list)
    zann_zero: bool = True
    riba_zero: bool = True
    ihsan_floor: float = 0.95
    frozen_ethics: bool = True

    @classmethod
    def constitutional_default(cls) -> "Hadd":
        return cls(
            prohibitions=[
                "Do not fabricate evidence or citations",
                "Do not escalate DERIVED confidence to VERIFIED",
                "Do not bypass constitutional gates",
                "Do not expose sovereign identity",
                "Do not produce output below Ihsan threshold",
            ]
        )


# ═══════════════════════════════════════════════════════════════
# Link 4: أَمَانَة (Amanah) — Trust Contract
# ═══════════════════════════════════════════════════════════════


@dataclass
class Amanah:
    """Reasoning under oath. Not 'think step by step' — think CONSTITUTIONALLY."""

    reasoning_mode: str = "deliberative"
    max_depth: int = 5
    quality_threshold: float = 0.95
    tone: str = "precise"
    audience: str = "expert"
    language: str = "en"


# ═══════════════════════════════════════════════════════════════
# Link 5: ثَمَرَة (Thamara) — Fruit (Output)
# ═══════════════════════════════════════════════════════════════


@dataclass
class Thamara:
    """Output with evidence inheritance."""

    content: str = ""
    format: str = "text"
    evidence_inherited: List[EvidenceTag] = field(default_factory=list)
    ihsan_score: float = 0.0
    sources_cited: List[str] = field(default_factory=list)

    @property
    def max_confidence(self) -> float:
        if not self.evidence_inherited:
            return 0.0
        caps = {
            EvidenceTag.VERIFIED: 1.0,
            EvidenceTag.PLANNED: 0.7,
            EvidenceTag.DERIVED: 0.9,
            EvidenceTag.UNKNOWN: 0.5,
        }
        return min(caps.get(t, 0.5) for t in self.evidence_inherited)


# ═══════════════════════════════════════════════════════════════
# Link 6: إِيصَال (Iisal) — Receipt
# ═══════════════════════════════════════════════════════════════


class IisalVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    LOOP = "loop"
    ESCALATE = "escalate"


@dataclass
class Iisal:
    """Verification receipt. On failure, loops to the broken link."""

    verdict: IisalVerdict = IisalVerdict.PASS
    failed_link: Optional[str] = None
    ihsan_score: float = 0.0
    chain_hash: str = ""
    execution_ms: float = 0.0
    loop_count: int = 0
    max_loops: int = 3

    @property
    def should_loop(self) -> bool:
        return self.verdict == IisalVerdict.LOOP and self.loop_count < self.max_loops


# ═══════════════════════════════════════════════════════════════
# The Complete Seed Chain
# ═══════════════════════════════════════════════════════════════


@dataclass
class SeedChain:
    """
    سلسلة البذرة — The complete 6-link prompt chain.
    The autopoietic innovation: Iisal feeds back into Niyyah.
    """

    niyyah: Niyyah
    bayyinah: Bayyinah = field(default_factory=Bayyinah)
    hadd: Hadd = field(default_factory=Hadd.constitutional_default)
    amanah: Amanah = field(default_factory=Amanah)
    thamara: Thamara = field(default_factory=Thamara)
    iisal: Iisal = field(default_factory=Iisal)

    def to_prompt(self) -> str:
        """Render the chain as a structured prompt for any LLM."""
        sections = []
        sections.append(f"## Niyyah (Intent)\n{self.niyyah.purpose}")
        if self.niyyah.context_keys:
            sections.append(f"Context: {', '.join(self.niyyah.context_keys)}")
        if self.bayyinah.items:
            lines = []
            for item in self.bayyinah.items:
                lines.append(f"- [{item.tag.value}] {item.claim}")
            sections.append("## Bayyinah (Evidence)\n" + "\n".join(lines))
        constraints = []
        if self.hadd.zann_zero:
            constraints.append("ZANN_ZERO: No unverified claims")
        if self.hadd.riba_zero:
            constraints.append("RIBA_ZERO: No extractive economics")
        constraints.append(f"IHSAN_FLOOR: >= {self.hadd.ihsan_floor}")
        for p in self.hadd.prohibitions:
            constraints.append(f"PROHIBITED: {p}")
        sections.append(
            "## Hadd (Boundaries)\n" + "\n".join(f"- {c}" for c in constraints)
        )
        sections.append(
            f"## Amanah (Trust Contract)\n"
            f"Reasoning: {self.amanah.reasoning_mode} | "
            f"Depth: {self.amanah.max_depth} | "
            f"Quality: >= {self.amanah.quality_threshold}\n"
            f"Tone: {self.amanah.tone} | "
            f"Audience: {self.amanah.audience} | "
            f"Language: {self.amanah.language}"
        )
        sections.append(f"## Thamara (Expected Output)\nFormat: {self.thamara.format}")
        return "\n\n".join(sections)

    def compute_hash(self) -> str:
        """BLAKE3-style hash of the chain for receipt linking."""
        content = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()

    def validate(self) -> List[str]:
        """Check chain integrity before execution."""
        errors = []
        if not self.niyyah.purpose:
            errors.append("niyyah: empty purpose")
        if self.bayyinah.unknown_count > 0 and self.hadd.zann_zero:
            errors.append(
                f"bayyinah: {self.bayyinah.unknown_count} UNKNOWN claims "
                f"violate ZANN_ZERO"
            )
        if self.amanah.quality_threshold < self.hadd.ihsan_floor:
            errors.append(
                f"amanah: quality threshold {self.amanah.quality_threshold} "
                f"below hadd ihsan floor {self.hadd.ihsan_floor}"
            )
        return errors


# ═══════════════════════════════════════════════════════════════
# Factory Functions
# ═══════════════════════════════════════════════════════════════


def small_seed(purpose: str, *, agent: str = "P7_DEMA") -> SeedChain:
    """Two-link chain for simple tasks: just Niyyah + Thamara.
    Hadd inherited from constitutional defaults."""
    return SeedChain(
        niyyah=Niyyah(purpose=purpose, target_agent=agent),
        hadd=Hadd.constitutional_default(),
        amanah=Amanah(reasoning_mode="reflex", max_depth=1),
    )


def full_seed(
    purpose: str,
    evidence: Optional[List[Dict[str, Any]]] = None,
    *,
    agent: str = "P7_DEMA",
    audience: str = "expert",
    tone: str = "precise",
) -> SeedChain:
    """Full 6-link chain for governed agent execution."""
    chain = SeedChain(
        niyyah=Niyyah(purpose=purpose, target_agent=agent),
        hadd=Hadd.constitutional_default(),
        amanah=Amanah(tone=tone, audience=audience),
    )
    if evidence:
        for e in evidence:
            chain.bayyinah.add(
                claim=e.get("claim", ""),
                tag=EvidenceTag(e.get("tag", "UNKNOWN")),
                source=e.get("source", ""),
            )
    return chain
