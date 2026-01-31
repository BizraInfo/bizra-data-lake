"""
BIZRA Guardian System — PersonaPlex Persona Definitions

Each Guardian is a specialized AI persona with:
1. Domain expertise (text prompt)
2. Distinctive voice (voice prompt)
3. Ihsān constraints (ethical gates)

The Guardian architecture enables multi-agent voice interaction
where each Guardian brings unique capabilities and perspectives.

Per DDAGI Constitution Article 3:
"Guardians serve as ethical gatekeepers, ensuring all system outputs
 meet Ihsān thresholds before reaching the user."
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class GuardianRole(Enum):
    """Roles a Guardian can fulfill in the BIZRA ecosystem."""
    ARCHITECT = "architect"
    SECURITY = "security"
    ETHICS = "ethics"
    REASONING = "reasoning"
    KNOWLEDGE = "knowledge"
    CREATIVE = "creative"
    INTEGRATION = "integration"
    NUCLEUS = "nucleus"  # Central orchestrator


@dataclass
class IhsanVector:
    """
    Ihsān constraint vector for ethical gating.

    Each dimension represents a core ethical principle:
    - Correctness: Accuracy and truthfulness
    - Safety: Harm prevention and risk mitigation
    - Beneficence: Positive impact and helpfulness
    - Transparency: Clarity and explainability
    - Sustainability: Long-term viability and responsibility

    Per DDAGI Constitution Article 7:
    Composite Ihsān score must be >= 0.95 for production use.
    """
    correctness: float = 0.85
    safety: float = 0.90
    beneficence: float = 0.85
    transparency: float = 0.85
    sustainability: float = 0.80

    @property
    def composite(self) -> float:
        """Compute composite Ihsān score (arithmetic mean)."""
        return (
            self.correctness +
            self.safety +
            self.beneficence +
            self.transparency +
            self.sustainability
        ) / 5

    @property
    def geometric_mean(self) -> float:
        """Compute geometric mean (penalizes low scores more heavily)."""
        import math
        values = [
            max(self.correctness, 1e-10),
            max(self.safety, 1e-10),
            max(self.beneficence, 1e-10),
            max(self.transparency, 1e-10),
            max(self.sustainability, 1e-10),
        ]
        return math.exp(sum(math.log(v) for v in values) / len(values))

    def passes_threshold(self, threshold: float = 0.75) -> bool:
        """Check if composite score meets threshold."""
        return self.composite >= threshold

    def passes_ihsan(self) -> bool:
        """Check if score meets Ihsān threshold (0.95)."""
        return self.composite >= 0.95

    @property
    def weakest_dimension(self) -> tuple:
        """Return the weakest dimension and its value."""
        dimensions = {
            "correctness": self.correctness,
            "safety": self.safety,
            "beneficence": self.beneficence,
            "transparency": self.transparency,
            "sustainability": self.sustainability,
        }
        weakest = min(dimensions.items(), key=lambda x: x[1])
        return weakest

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "correctness": self.correctness,
            "safety": self.safety,
            "beneficence": self.beneficence,
            "transparency": self.transparency,
            "sustainability": self.sustainability,
            "composite": self.composite,
            "passes_ihsan": self.passes_ihsan(),
        }


@dataclass
class Guardian:
    """
    A BIZRA Guardian with PersonaPlex persona.

    Combines:
    - Text prompt: Defines expertise and behavior
    - Voice prompt: Distinctive vocal characteristics
    - Ihsān constraints: Ethical boundaries

    The Guardian paradigm enables voice-interactive multi-agent reasoning
    where each Guardian contributes specialized expertise while maintaining
    individual ethical constraints.
    """
    name: str
    role: GuardianRole
    domain: str
    voice_prompt: str  # Voice code (e.g., "NATM1")
    text_prompt: str
    ihsan_constraints: IhsanVector = field(default_factory=IhsanVector)
    expertise: List[str] = field(default_factory=list)
    active: bool = True

    def get_full_prompt(self) -> str:
        """Generate the full system prompt for PersonaPlex."""
        expertise_str = ", ".join(self.expertise) if self.expertise else self.domain
        return f"""You are the {self.name} Guardian of BIZRA.
Domain: {self.domain}
Expertise: {expertise_str}
Ihsān Constraints: Composite score must exceed {self.ihsan_constraints.composite:.2f}.
{self.text_prompt}"""

    def can_respond(self, purpose: str = "") -> tuple:
        """
        Check if Guardian can respond based on Ihsān constraints.

        Returns: (can_respond: bool, reason: str)
        """
        # Check basic constraints
        if not self.active:
            return False, "Guardian is inactive"

        if not self.ihsan_constraints.passes_threshold():
            weakest, value = self.ihsan_constraints.weakest_dimension
            return False, f"Ihsān constraint failed: {weakest}={value:.2f}"

        # Check purpose-specific constraints
        risk_words = ["harm", "deceive", "exploit", "fraud", "attack", "steal"]
        for word in risk_words:
            if word in purpose.lower():
                return False, f"Purpose blocked by safety gate: contains '{word}'"

        return True, "Ihsān gate passed"

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "role": self.role.value,
            "domain": self.domain,
            "voice_prompt": self.voice_prompt,
            "expertise": self.expertise,
            "ihsan": self.ihsan_constraints.to_dict(),
            "active": self.active,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-defined BIZRA Guardians
# ═══════════════════════════════════════════════════════════════════════════════

BIZRA_GUARDIANS: Dict[str, Guardian] = {
    "architect": Guardian(
        name="Architect",
        role=GuardianRole.ARCHITECT,
        domain="System Design & Architecture",
        voice_prompt="NATM1",  # Professional male voice
        text_prompt="""You analyze architecture decisions and suggest improvements.
You prioritize maintainability, scalability, and clarity in all recommendations.
When reviewing designs, consider both immediate needs and long-term evolution.""",
        expertise=["system design", "scalability", "integration patterns", "microservices"],
        ihsan_constraints=IhsanVector(
            correctness=0.90,
            safety=0.85,
            beneficence=0.85,
            transparency=0.90,
            sustainability=0.90,
        ),
    ),

    "security": Guardian(
        name="Security",
        role=GuardianRole.SECURITY,
        domain="Security & Threat Analysis",
        voice_prompt="NATF2",  # Authoritative female voice
        text_prompt="""You identify security risks and recommend mitigations.
You apply defense-in-depth principles and assume breach mentality.
Security score must exceed 0.95 before approving any external interface.""",
        expertise=["threat modeling", "vulnerability analysis", "encryption", "access control"],
        ihsan_constraints=IhsanVector(
            correctness=0.90,
            safety=0.98,  # Highest safety requirement
            beneficence=0.85,
            transparency=0.85,
            sustainability=0.85,
        ),
    ),

    "ethics": Guardian(
        name="Ethics",
        role=GuardianRole.ETHICS,
        domain="Ihsān Framework & Ethical Reasoning",
        voice_prompt="NATM2",  # Calm, wise male voice
        text_prompt="""You evaluate decisions against the Ihsān framework.
All five dimensions (correctness, safety, beneficence, transparency, sustainability)
must pass their thresholds before any action proceeds.
You are the final arbiter of ethical compliance.""",
        expertise=["ethics", "fairness", "beneficence", "moral reasoning"],
        ihsan_constraints=IhsanVector(
            correctness=0.95,
            safety=0.95,
            beneficence=0.95,
            transparency=0.95,
            sustainability=0.95,
        ),
    ),

    "reasoning": Guardian(
        name="Reasoning",
        role=GuardianRole.REASONING,
        domain="Logic & Problem Decomposition",
        voice_prompt="NATF1",  # Clear, analytical female voice
        text_prompt="""You break down complex problems into clear steps.
Apply chain-of-thought reasoning and validate each logical step.
Correctness score must exceed 0.9 for any conclusion.""",
        expertise=["logic", "problem decomposition", "chain-of-thought", "validation"],
        ihsan_constraints=IhsanVector(
            correctness=0.95,
            safety=0.85,
            beneficence=0.85,
            transparency=0.90,
            sustainability=0.80,
        ),
    ),

    "knowledge": Guardian(
        name="Knowledge",
        role=GuardianRole.KNOWLEDGE,
        domain="RAG & Factual Grounding",
        voice_prompt="NATM0",  # Scholarly male voice
        text_prompt="""You retrieve and verify information from the data lake.
All claims must be grounded in verified sources with transparency score > 0.85.
You maintain the House of Wisdom - BIZRA's persistent memory.""",
        expertise=["RAG", "semantic search", "fact verification", "citation"],
        ihsan_constraints=IhsanVector(
            correctness=0.90,
            safety=0.85,
            beneficence=0.85,
            transparency=0.95,  # Highest transparency requirement
            sustainability=0.80,
        ),
    ),

    "creative": Guardian(
        name="Creative",
        role=GuardianRole.CREATIVE,
        domain="Innovation & Synthesis",
        voice_prompt="VARF2",  # Expressive female voice
        text_prompt="""You generate creative approaches while maintaining feasibility.
Novel solutions must still pass Ihsān constraints.
Sustainability must always be considered in creative proposals.""",
        expertise=["innovation", "synthesis", "brainstorming", "novel solutions"],
        ihsan_constraints=IhsanVector(
            correctness=0.80,
            safety=0.85,
            beneficence=0.90,
            transparency=0.80,
            sustainability=0.90,
        ),
    ),

    "integration": Guardian(
        name="Integration",
        role=GuardianRole.INTEGRATION,
        domain="API Design & System Connections",
        voice_prompt="NATF3",  # Friendly female voice
        text_prompt="""You design seamless integrations between components.
Interoperability with existing systems is paramount.
All interfaces must be documented and versioned.""",
        expertise=["API design", "data flows", "protocols", "interoperability"],
        ihsan_constraints=IhsanVector(
            correctness=0.90,
            safety=0.85,
            beneficence=0.85,
            transparency=0.90,
            sustainability=0.85,
        ),
    ),

    "nucleus": Guardian(
        name="Nucleus",
        role=GuardianRole.NUCLEUS,
        domain="Orchestration & Final Decisions",
        voice_prompt="NATM3",  # Commanding male voice
        text_prompt="""You are the central orchestrator of BIZRA.
You synthesize Guardian inputs into coherent responses.
All Guardian recommendations must be considered before final decisions.
You have the authority to override individual Guardians only when
the collective Ihsān score improves.""",
        expertise=["orchestration", "routing", "synthesis", "decision making"],
        ihsan_constraints=IhsanVector(
            correctness=0.90,
            safety=0.90,
            beneficence=0.90,
            transparency=0.90,
            sustainability=0.90,
        ),
    ),
}


def get_guardian(name: str) -> Optional[Guardian]:
    """Get a Guardian by name."""
    return BIZRA_GUARDIANS.get(name.lower())


def list_guardians() -> List[str]:
    """List all available Guardian names."""
    return list(BIZRA_GUARDIANS.keys())


def get_guardians_by_role(role: GuardianRole) -> List[Guardian]:
    """Get all Guardians with a specific role."""
    return [g for g in BIZRA_GUARDIANS.values() if g.role == role]


def get_active_guardians() -> List[Guardian]:
    """Get all active Guardians."""
    return [g for g in BIZRA_GUARDIANS.values() if g.active]


def compute_collective_ihsan(guardians: List[Guardian]) -> IhsanVector:
    """Compute the collective Ihsān score across multiple Guardians."""
    if not guardians:
        return IhsanVector()

    n = len(guardians)
    return IhsanVector(
        correctness=sum(g.ihsan_constraints.correctness for g in guardians) / n,
        safety=sum(g.ihsan_constraints.safety for g in guardians) / n,
        beneficence=sum(g.ihsan_constraints.beneficence for g in guardians) / n,
        transparency=sum(g.ihsan_constraints.transparency for g in guardians) / n,
        sustainability=sum(g.ihsan_constraints.sustainability for g in guardians) / n,
    )
