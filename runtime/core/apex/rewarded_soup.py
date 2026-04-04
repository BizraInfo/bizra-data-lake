"""
BIZRA Apex - Rewarded Soups for Persona Interpolation
======================================================

Implements Rewarded Soups for BIZRA persona interpolation, allowing
weighted combination of multiple personas into a unified "soup" persona
with interpolated voice embeddings and weighted prompt composition.

Key Concepts:
    - PersonaSoup: Interpolated persona with combined voice embedding
    - Lambda weights: Must sum to 1.0, control persona contribution
    - Veto domains: Union of all constituent persona veto domains (never interpolated)
    - SNR contribution: Estimated signal-to-noise ratio based on persona alignment

The interpolation formula is:
    theta_final = sum(lambda_i * theta_i) for all personas

Where theta represents the voice embedding vector.

Reference:
    Rewarded Soups: Towards Pareto-Optimal Alignment by Interpolating Weights
    Fine-Tuned on Diverse Rewards (Rame et al., 2023)

Domain: bizra-pci-v1:apex:rewarded-soup
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# Optional numpy - pure Python fallback for environments without it
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

from core.personaplex.persona import (
    PersonaDefinition,
    VetoDomain,
    create_security_guardian,
    create_ethics_validator,
    create_master_reasoner,
    create_memory_architect,
    create_creative_synthesizer,
    create_compliance_guardian,
    create_safety_guardian,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS (imported from unified constants.py)
# =============================================================================

from core.constants import IHSAN_THRESHOLD

VOICE_EMBEDDING_DIM: int = 256
DEFAULT_SNR_BASE: float = IHSAN_THRESHOLD  # 0.95 - uses constitutional threshold
DOMAIN_PREFIX: str = "bizra-pci-v1:apex:rewarded-soup"
VERSION: str = "1.0.0"


# Soup preset names
class SoupPreset(str, Enum):
    """Standard soup presets for common use cases."""

    SECURITY_FOCUSED = "security_focused"
    CREATIVE_FOCUSED = "creative_focused"
    ANALYSIS_FOCUSED = "analysis_focused"
    BALANCED = "balanced"
    GUARDIAN_COUNCIL = "guardian_council"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class PersonaSoupComponent:
    """
    A component within a PersonaSoup blend.

    Tracks the persona and its lambda weight for interpolation,
    along with contribution metrics.

    Attributes:
        persona: The PersonaDefinition being blended
        lambda_weight: Weight for this persona (0.0-1.0)
        contribution_score: Computed contribution to soup quality
        active_domains: Domains this persona contributes expertise in
    """

    persona: PersonaDefinition
    lambda_weight: float
    contribution_score: float = 0.0
    active_domains: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate weight constraints."""
        if not 0.0 <= self.lambda_weight <= 1.0:
            raise ValueError(
                f"lambda_weight must be in [0.0, 1.0], got {self.lambda_weight}"
            )

        # Set active domains from persona
        if not self.active_domains:
            self.active_domains = list(self.persona.expertise_domains)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "persona_id": self.persona.persona_id,
            "lambda_weight": self.lambda_weight,
            "contribution_score": self.contribution_score,
            "active_domains": self.active_domains,
            "has_veto_power": self.persona.has_veto_power,
            "veto_domains": [v.name for v in self.persona.veto_domains],
        }


@dataclass
class PersonaSoup:
    """
    A blended persona created from interpolating multiple PersonaDefinitions.

    PersonaSoup implements the Rewarded Soups concept where multiple personas
    are combined via weighted interpolation of their voice embeddings. The
    resulting "soup" persona has:

    - Interpolated voice embedding (256-dim, L2 normalized)
    - Union of all constituent veto domains (veto is never diluted)
    - Weighted system prompt composition
    - Combined expertise domains
    - Estimated SNR contribution

    The interpolation formula:
        theta_final = sum(lambda_i * theta_i) for i in 1..n

    Where theta_i is the voice embedding and lambda_i is the weight.

    Attributes:
        soup_id: Unique identifier for this soup
        components: List of PersonaSoupComponent with weights
        voice_embedding: Interpolated 256-dim embedding (L2 normalized)
        system_prompt: Weighted composition of persona prompts
        veto_domains: Union of all component veto domains
        expertise_domains: Combined expertise from all components
        capabilities: Combined capabilities from all components
        snr_estimate: Estimated signal-to-noise ratio
        soup_hash: BLAKE3/SHA-256 hash for integrity verification
        created_at: ISO timestamp of soup creation
        version: Version of the soup schema
        metadata: Additional soup metadata

    Example:
        >>> security = create_security_guardian()
        >>> ethics = create_ethics_validator()
        >>> soup = interpolate_soup(
        ...     personas=[security, ethics],
        ...     lambdas=[0.6, 0.4]
        ... )
        >>> soup.soup_id
        'soup-security-guardian-0.60-ethics-validator-0.40'
        >>> len(soup.veto_domains)  # Union of SECURITY and ETHICS
        2
    """

    soup_id: str
    components: List[PersonaSoupComponent]
    voice_embedding: List[float]
    system_prompt: str
    veto_domains: Set[VetoDomain]
    expertise_domains: List[str]
    capabilities: List[str]
    snr_estimate: float
    soup_hash: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    version: str = VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate soup constraints and compute hash."""
        # Validate embedding dimension
        if len(self.voice_embedding) != VOICE_EMBEDDING_DIM:
            raise ValueError(
                f"voice_embedding must be {VOICE_EMBEDDING_DIM}-dim, "
                f"got {len(self.voice_embedding)}"
            )

        # Validate lambda weights sum to 1.0
        total_weight = sum(c.lambda_weight for c in self.components)
        if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
            raise ValueError(f"Lambda weights must sum to 1.0, got {total_weight}")

        # Compute hash if not provided
        if not self.soup_hash:
            self.soup_hash = self._compute_hash()

    @property
    def has_veto_power(self) -> bool:
        """Check if this soup has veto authority in any domain."""
        return len(self.veto_domains) > 0

    @property
    def dominant_persona(self) -> PersonaSoupComponent:
        """Get the component with highest lambda weight."""
        return max(self.components, key=lambda c: c.lambda_weight)

    @property
    def base_vote_weight(self) -> float:
        """Compute weighted average vote weight from components."""
        return sum(
            c.lambda_weight * c.persona.base_vote_weight for c in self.components
        )

    @property
    def name(self) -> str:
        """Generate a human-readable name for the soup."""
        if len(self.components) == 1:
            return self.components[0].persona.name

        # Use dominant persona + blend indicator
        dominant = self.dominant_persona
        return f"{dominant.persona.name} Blend"

    def _compute_hash(self) -> str:
        """
        Compute hash for integrity verification.

        Covers all immutable soup attributes for tamper detection.
        """
        hash_payload = {
            "soup_id": self.soup_id,
            "components": [
                {"persona_id": c.persona.persona_id, "lambda": c.lambda_weight}
                for c in self.components
            ],
            "voice_embedding": self.voice_embedding[
                :16
            ],  # First 16 dims for efficiency
            "veto_domains": sorted(v.name for v in self.veto_domains),
            "expertise_domains": sorted(self.expertise_domains),
            "version": self.version,
        }

        payload_bytes = json.dumps(
            hash_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

        return hashlib.sha256(payload_bytes).hexdigest()

    def verify_hash(self) -> bool:
        """Verify soup integrity by recomputing hash."""
        return self._compute_hash() == self.soup_hash

    def get_component_by_persona(
        self, persona_id: str
    ) -> Optional[PersonaSoupComponent]:
        """Get a component by persona ID."""
        for component in self.components:
            if component.persona.persona_id == persona_id:
                return component
        return None

    def compute_task_alignment(self, task_domains: List[str]) -> float:
        """
        Compute weighted task alignment across all components.

        Args:
            task_domains: List of domain identifiers for the task

        Returns:
            Weighted alignment score in [0.0, 1.0]
        """
        if not task_domains:
            return 0.0

        total_alignment = 0.0
        for component in self.components:
            persona_alignment = component.persona.compute_task_alignment(task_domains)
            total_alignment += component.lambda_weight * persona_alignment

        return total_alignment

    def to_dict(self) -> Dict[str, Any]:
        """Serialize soup to dictionary."""
        return {
            "soup_id": self.soup_id,
            "components": [c.to_dict() for c in self.components],
            "voice_embedding": self.voice_embedding,
            "system_prompt": self.system_prompt,
            "veto_domains": [v.name for v in self.veto_domains],
            "expertise_domains": self.expertise_domains,
            "capabilities": self.capabilities,
            "snr_estimate": self.snr_estimate,
            "soup_hash": self.soup_hash,
            "created_at": self.created_at,
            "version": self.version,
            "metadata": self.metadata,
            # Computed properties
            "has_veto_power": self.has_veto_power,
            "base_vote_weight": self.base_vote_weight,
            "name": self.name,
        }

    def to_persona_definition(self) -> PersonaDefinition:
        """
        Convert the soup to a PersonaDefinition for compatibility.

        This allows the soup to be used wherever a PersonaDefinition
        is expected (e.g., in consensus voting, routing).

        Returns:
            PersonaDefinition with interpolated attributes
        """
        return PersonaDefinition(
            persona_id=self.soup_id,
            text_prompt=self.system_prompt,
            expertise_domains=self.expertise_domains,
            capabilities=self.capabilities,
            veto_domains=self.veto_domains,
            base_vote_weight=self.base_vote_weight,
            voice_embedding=self.voice_embedding,
        )


@dataclass
class SNRContribution:
    """
    Signal-to-Noise Ratio contribution estimate for a persona soup.

    Estimates how the soup composition affects SNR based on:
    - Persona expertise alignment
    - Weight distribution entropy
    - Veto domain coverage

    Attributes:
        base_snr: Starting SNR estimate (typically 0.95)
        alignment_boost: Boost from expertise alignment
        entropy_penalty: Penalty from weight distribution entropy
        veto_coverage_boost: Boost from veto domain coverage
        final_snr: Computed final SNR estimate
        breakdown: Detailed breakdown of contributions
    """

    base_snr: float
    alignment_boost: float
    entropy_penalty: float
    veto_coverage_boost: float
    final_snr: float
    breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "base_snr": self.base_snr,
            "alignment_boost": self.alignment_boost,
            "entropy_penalty": self.entropy_penalty,
            "veto_coverage_boost": self.veto_coverage_boost,
            "final_snr": self.final_snr,
            "breakdown": self.breakdown,
        }


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


def l2_normalize(embedding: List[float]) -> List[float]:
    """
    L2 normalize an embedding vector.

    Args:
        embedding: Input embedding vector

    Returns:
        L2 normalized embedding (unit vector)
    """
    if HAS_NUMPY:
        np_emb = np.array(embedding)
        norm = np.linalg.norm(np_emb)
        if norm < 1e-10:
            return embedding
        return (np_emb / norm).tolist()
    else:
        # Pure Python fallback
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm < 1e-10:
            return embedding
        return [x / norm for x in embedding]


def interpolate_embeddings(
    embeddings: List[List[float]],
    lambdas: List[float],
) -> List[float]:
    """
    Interpolate multiple embeddings with lambda weights.

    Implements the core Rewarded Soups formula:
        theta_final = sum(lambda_i * theta_i)

    The result is L2 normalized to maintain unit vector properties.

    Args:
        embeddings: List of embedding vectors (each 256-dim)
        lambdas: List of lambda weights (must sum to 1.0)

    Returns:
        Interpolated and L2 normalized embedding

    Raises:
        ValueError: If lambdas don't sum to 1.0 or dimensions mismatch
    """
    if not embeddings:
        raise ValueError("At least one embedding required")

    if len(embeddings) != len(lambdas):
        raise ValueError(
            f"Embeddings and lambdas length mismatch: "
            f"{len(embeddings)} vs {len(lambdas)}"
        )

    # Validate lambda sum
    total = sum(lambdas)
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"Lambdas must sum to 1.0, got {total}")

    # Validate dimensions
    dim = len(embeddings[0])
    for i, emb in enumerate(embeddings):
        if len(emb) != dim:
            raise ValueError(f"Embedding {i} dimension mismatch: {len(emb)} vs {dim}")

    if HAS_NUMPY:
        # Numpy implementation
        np_embeddings = [np.array(e) for e in embeddings]
        np_lambdas = np.array(lambdas)

        interpolated = np.zeros(dim)
        for emb, lam in zip(np_embeddings, np_lambdas):
            interpolated += lam * emb

        # L2 normalize and return
        return l2_normalize(interpolated.tolist())
    else:
        # Pure Python fallback
        interpolated = [0.0] * dim
        for emb, lam in zip(embeddings, lambdas):
            for i in range(dim):
                interpolated[i] += lam * emb[i]

        # L2 normalize and return
        return l2_normalize(interpolated)


def compose_weighted_prompt(
    personas: List[PersonaDefinition],
    lambdas: List[float],
) -> str:
    """
    Compose a weighted system prompt from multiple personas.

    Creates a unified prompt that incorporates weighted contributions
    from each persona's text_prompt, with higher-weighted personas
    having more prominent representation.

    Args:
        personas: List of PersonaDefinitions
        lambdas: List of lambda weights

    Returns:
        Composed system prompt string
    """
    if not personas:
        return ""

    # Sort by weight descending
    weighted_personas = sorted(zip(personas, lambdas), key=lambda x: x[1], reverse=True)

    # Build header
    header = "You are a BIZRA Persona Soup - a blended persona combining:\n"
    for persona, weight in weighted_personas:
        header += f"  - {persona.name} ({weight:.0%} influence)\n"
    header += "\n"

    # Build core capabilities section
    capabilities = "Your blended capabilities:\n"
    for persona, weight in weighted_personas:
        if weight >= 0.1:  # Only include personas with >= 10% weight
            # Extract key capability from prompt
            prompt_lines = persona.text_prompt.strip().split("\n")
            core_line = prompt_lines[0] if prompt_lines else ""
            capabilities += f"  [{persona.name}] {core_line}\n"
    capabilities += "\n"

    # Build veto section for personas with veto power
    veto_section = ""
    veto_personas = [p for p, _ in weighted_personas if p.has_veto_power]
    if veto_personas:
        veto_section = "VETO AUTHORITY (inherited from component personas):\n"
        for persona in veto_personas:
            veto_domains = ", ".join(v.name for v in persona.veto_domains)
            veto_section += f"  - {persona.name}: {veto_domains}\n"
        veto_section += "\nVeto power is NEVER diluted by interpolation.\n\n"

    # Build behavioral guidance from dominant persona
    dominant_persona, dominant_weight = weighted_personas[0]
    guidance = (
        f"Primary behavioral model ({dominant_persona.name}, {dominant_weight:.0%}):\n"
    )
    guidance += dominant_persona.text_prompt + "\n\n"

    # Add secondary influences
    if len(weighted_personas) > 1:
        guidance += "Secondary influences (integrate as appropriate):\n"
        for persona, weight in weighted_personas[1:]:
            if weight >= 0.15:  # Only include meaningful contributions
                # Get first paragraph of persona prompt
                first_para = persona.text_prompt.split("\n\n")[0]
                guidance += f"[{persona.name}, {weight:.0%}]: {first_para}\n\n"

    return header + capabilities + veto_section + guidance


def compute_snr_contribution(
    personas: List[PersonaDefinition],
    lambdas: List[float],
    task_domains: Optional[List[str]] = None,
) -> SNRContribution:
    """
    Estimate SNR contribution based on persona weights and alignment.

    The SNR estimate considers:
    1. Base SNR (0.95 default, from Ihsan threshold)
    2. Alignment boost: Higher alignment to task domains improves SNR
    3. Entropy penalty: More dispersed weights slightly reduce SNR
    4. Veto coverage boost: More veto domain coverage improves SNR

    Args:
        personas: List of PersonaDefinitions
        lambdas: List of lambda weights
        task_domains: Optional task domains for alignment calculation

    Returns:
        SNRContribution with detailed breakdown
    """
    base_snr = DEFAULT_SNR_BASE

    # 1. Compute alignment boost
    alignment_boost = 0.0
    if task_domains:
        total_alignment = 0.0
        for persona, lam in zip(personas, lambdas):
            alignment = persona.compute_task_alignment(task_domains)
            total_alignment += lam * alignment
        # Scale alignment boost (max +0.02)
        alignment_boost = min(0.02, total_alignment * 0.025)

    # 2. Compute entropy penalty
    # Higher entropy (more uniform distribution) = slight penalty
    # because specialized personas are more precise
    nonzero_lambdas = [l for l in lambdas if l > 0]
    if HAS_NUMPY:
        lambdas_array = np.array(nonzero_lambdas)
        entropy = -np.sum(lambdas_array * np.log(lambdas_array + 1e-10))
        max_entropy = np.log(len(lambdas_array))
    else:
        # Pure Python fallback
        entropy = -sum(l * math.log(l + 1e-10) for l in nonzero_lambdas)
        max_entropy = math.log(len(nonzero_lambdas))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    entropy_penalty = normalized_entropy * 0.01  # Max -0.01

    # 3. Compute veto coverage boost
    all_veto_domains: Set[VetoDomain] = set()
    for persona in personas:
        all_veto_domains.update(persona.veto_domains)
    # Coverage of 4 domains = +0.015 boost
    veto_coverage_boost = (len(all_veto_domains) / 4) * 0.015

    # Compute final SNR
    final_snr = min(
        0.99,  # Cap at 0.99
        base_snr + alignment_boost - entropy_penalty + veto_coverage_boost,
    )

    breakdown = {
        "base": base_snr,
        "alignment": alignment_boost,
        "entropy": -entropy_penalty,
        "veto_coverage": veto_coverage_boost,
        "num_personas": len(personas),
        "num_veto_domains": len(all_veto_domains),
    }

    return SNRContribution(
        base_snr=base_snr,
        alignment_boost=alignment_boost,
        entropy_penalty=entropy_penalty,
        veto_coverage_boost=veto_coverage_boost,
        final_snr=final_snr,
        breakdown=breakdown,
    )


def interpolate_soup(
    personas: List[PersonaDefinition],
    lambdas: List[float],
    task_domains: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PersonaSoup:
    """
    Create a PersonaSoup by interpolating multiple personas.

    This is the main entry point for creating Rewarded Soups. It:
    1. Validates lambda weights sum to 1.0
    2. Interpolates voice embeddings using theta_final = sum(lambda_i * theta_i)
    3. L2 normalizes the resulting embedding
    4. Computes union of all veto domains (veto is never interpolated)
    5. Composes weighted system prompt
    6. Estimates SNR contribution

    Args:
        personas: List of PersonaDefinitions to blend
        lambdas: List of lambda weights (must sum to 1.0)
        task_domains: Optional task domains for SNR estimation
        metadata: Optional additional metadata

    Returns:
        PersonaSoup with interpolated attributes

    Raises:
        ValueError: If inputs are invalid (empty, length mismatch, bad weights)

    Example:
        >>> security = create_security_guardian()
        >>> ethics = create_ethics_validator()
        >>> soup = interpolate_soup(
        ...     personas=[security, ethics],
        ...     lambdas=[0.6, 0.4]
        ... )
        >>> print(f"Soup has {len(soup.veto_domains)} veto domains")
        Soup has 2 veto domains
    """
    # Validate inputs
    if not personas:
        raise ValueError("At least one persona required")

    if len(personas) != len(lambdas):
        raise ValueError(
            f"Personas and lambdas length mismatch: "
            f"{len(personas)} vs {len(lambdas)}"
        )

    # Validate lambda sum
    total = sum(lambdas)
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"Lambdas must sum to 1.0, got {total}")

    # Validate individual lambdas
    for i, lam in enumerate(lambdas):
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"Lambda {i} must be in [0.0, 1.0], got {lam}")

    logger.info(
        f"Creating persona soup from {len(personas)} personas: "
        f"{[p.persona_id for p in personas]}"
    )

    # Generate soup ID from components
    soup_id_parts = []
    for persona, lam in zip(personas, lambdas):
        soup_id_parts.append(f"{persona.persona_id}-{lam:.2f}")
    soup_id = "soup-" + "-".join(soup_id_parts)

    # Interpolate voice embeddings
    embeddings = [p.voice_embedding for p in personas]
    interpolated_embedding = interpolate_embeddings(embeddings, lambdas)

    # Union of veto domains (NEVER interpolated)
    all_veto_domains: Set[VetoDomain] = set()
    for persona in personas:
        all_veto_domains.update(persona.veto_domains)

    # Union of expertise domains (deduplicated)
    all_expertise: Set[str] = set()
    for persona in personas:
        all_expertise.update(persona.expertise_domains)

    # Union of capabilities (deduplicated)
    all_capabilities: Set[str] = set()
    for persona in personas:
        all_capabilities.update(persona.capabilities)

    # Compose weighted prompt
    system_prompt = compose_weighted_prompt(personas, lambdas)

    # Estimate SNR contribution
    snr_contribution = compute_snr_contribution(personas, lambdas, task_domains)

    # Create components
    components = []
    for persona, lam in zip(personas, lambdas):
        component = PersonaSoupComponent(
            persona=persona,
            lambda_weight=lam,
            contribution_score=lam * persona.base_vote_weight,
            active_domains=list(persona.expertise_domains),
        )
        components.append(component)

    # Create the soup
    soup = PersonaSoup(
        soup_id=soup_id,
        components=components,
        voice_embedding=interpolated_embedding,
        system_prompt=system_prompt,
        veto_domains=all_veto_domains,
        expertise_domains=sorted(all_expertise),
        capabilities=sorted(all_capabilities),
        snr_estimate=snr_contribution.final_snr,
        metadata={
            "snr_breakdown": snr_contribution.to_dict(),
            "task_domains": task_domains or [],
            "preset": None,
            **(metadata or {}),
        },
    )

    logger.info(
        f"Created persona soup '{soup_id}': "
        f"veto_domains={[v.name for v in all_veto_domains]}, "
        f"snr_estimate={snr_contribution.final_snr:.4f}"
    )

    return soup


# =============================================================================
# FACTORY FUNCTIONS - STANDARD SOUPS
# =============================================================================


def create_security_focused_soup() -> PersonaSoup:
    """
    Create a security-focused persona soup.

    Composition:
        - Security Guardian: 50%
        - Compliance Guardian: 25%
        - Safety Guardian: 15%
        - Ethics Validator: 10%

    Veto domains: SECURITY, COMPLIANCE, SAFETY, ETHICS (all four)

    Use case: Security audits, threat modeling, vulnerability assessment

    Returns:
        PersonaSoup optimized for security tasks
    """
    personas = [
        create_security_guardian(),
        create_compliance_guardian(),
        create_safety_guardian(),
        create_ethics_validator(),
    ]
    lambdas = [0.50, 0.25, 0.15, 0.10]

    soup = interpolate_soup(
        personas=personas,
        lambdas=lambdas,
        task_domains=["security", "threat-analysis", "compliance", "audit"],
        metadata={"preset": SoupPreset.SECURITY_FOCUSED.value},
    )

    return soup


def create_creative_focused_soup() -> PersonaSoup:
    """
    Create a creative-focused persona soup.

    Composition:
        - Creative Synthesizer: 50%
        - Master Reasoner: 30%
        - Memory Architect: 15%
        - Ethics Validator: 5%

    Veto domains: ETHICS (from Ethics Validator)

    Use case: Ideation, content generation, novel solution synthesis

    Returns:
        PersonaSoup optimized for creative tasks
    """
    personas = [
        create_creative_synthesizer(),
        create_master_reasoner(),
        create_memory_architect(),
        create_ethics_validator(),
    ]
    lambdas = [0.50, 0.30, 0.15, 0.05]

    soup = interpolate_soup(
        personas=personas,
        lambdas=lambdas,
        task_domains=["creativity", "synthesis", "ideation", "reasoning"],
        metadata={"preset": SoupPreset.CREATIVE_FOCUSED.value},
    )

    return soup


def create_analysis_focused_soup() -> PersonaSoup:
    """
    Create an analysis-focused persona soup.

    Composition:
        - Master Reasoner: 40%
        - Memory Architect: 30%
        - Security Guardian: 15%
        - Ethics Validator: 15%

    Veto domains: SECURITY, ETHICS

    Use case: Deep analysis, strategic planning, knowledge synthesis

    Returns:
        PersonaSoup optimized for analytical tasks
    """
    personas = [
        create_master_reasoner(),
        create_memory_architect(),
        create_security_guardian(),
        create_ethics_validator(),
    ]
    lambdas = [0.40, 0.30, 0.15, 0.15]

    soup = interpolate_soup(
        personas=personas,
        lambdas=lambdas,
        task_domains=["reasoning", "analysis", "planning", "synthesis"],
        metadata={"preset": SoupPreset.ANALYSIS_FOCUSED.value},
    )

    return soup


def create_balanced_soup() -> PersonaSoup:
    """
    Create a balanced persona soup with equal guardian representation.

    Composition:
        - Master Reasoner: 25%
        - Security Guardian: 18.75%
        - Ethics Validator: 18.75%
        - Compliance Guardian: 18.75%
        - Safety Guardian: 18.75%

    Veto domains: SECURITY, ETHICS, COMPLIANCE, SAFETY (all four)

    Use case: General-purpose tasks requiring balanced judgment

    Returns:
        PersonaSoup with balanced guardian coverage
    """
    personas = [
        create_master_reasoner(),
        create_security_guardian(),
        create_ethics_validator(),
        create_compliance_guardian(),
        create_safety_guardian(),
    ]
    lambdas = [0.25, 0.1875, 0.1875, 0.1875, 0.1875]

    soup = interpolate_soup(
        personas=personas,
        lambdas=lambdas,
        task_domains=["general", "reasoning", "security", "ethics"],
        metadata={"preset": SoupPreset.BALANCED.value},
    )

    return soup


def create_guardian_council_soup() -> PersonaSoup:
    """
    Create a guardian council soup with all veto personas.

    Composition:
        - Security Guardian: 25%
        - Ethics Validator: 25%
        - Compliance Guardian: 25%
        - Safety Guardian: 25%

    Veto domains: SECURITY, ETHICS, COMPLIANCE, SAFETY (all four)

    Use case: High-stakes decisions requiring maximum validation

    Returns:
        PersonaSoup with full guardian council coverage
    """
    personas = [
        create_security_guardian(),
        create_ethics_validator(),
        create_compliance_guardian(),
        create_safety_guardian(),
    ]
    lambdas = [0.25, 0.25, 0.25, 0.25]

    soup = interpolate_soup(
        personas=personas,
        lambdas=lambdas,
        task_domains=["security", "ethics", "compliance", "safety"],
        metadata={"preset": SoupPreset.GUARDIAN_COUNCIL.value},
    )

    return soup


def create_standard_soups() -> Dict[str, PersonaSoup]:
    """
    Create all standard persona soups.

    Factory function that generates the standard set of pre-configured
    persona soups for common use cases:

    - security_focused: Security audits, threat modeling
    - creative_focused: Ideation, content generation
    - analysis_focused: Deep analysis, strategic planning
    - balanced: General-purpose balanced judgment
    - guardian_council: High-stakes maximum validation

    Returns:
        Dictionary mapping preset names to PersonaSoup instances

    Example:
        >>> soups = create_standard_soups()
        >>> security_soup = soups["security_focused"]
        >>> print(len(security_soup.veto_domains))
        4
    """
    logger.info("Creating standard persona soups")

    soups = {
        SoupPreset.SECURITY_FOCUSED.value: create_security_focused_soup(),
        SoupPreset.CREATIVE_FOCUSED.value: create_creative_focused_soup(),
        SoupPreset.ANALYSIS_FOCUSED.value: create_analysis_focused_soup(),
        SoupPreset.BALANCED.value: create_balanced_soup(),
        SoupPreset.GUARDIAN_COUNCIL.value: create_guardian_council_soup(),
    }

    logger.info(f"Created {len(soups)} standard soups: {list(soups.keys())}")

    return soups


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_soup_for_task(
    task_domains: List[str],
    soups: Optional[Dict[str, PersonaSoup]] = None,
) -> Tuple[str, PersonaSoup]:
    """
    Select the best soup for a given task based on domain alignment.

    Args:
        task_domains: List of domain identifiers for the task
        soups: Optional pre-created soups dict (creates if not provided)

    Returns:
        Tuple of (soup_preset_name, PersonaSoup)
    """
    if soups is None:
        soups = create_standard_soups()

    best_preset = SoupPreset.BALANCED.value
    best_alignment = 0.0

    for preset_name, soup in soups.items():
        alignment = soup.compute_task_alignment(task_domains)
        if alignment > best_alignment:
            best_alignment = alignment
            best_preset = preset_name

    logger.debug(
        f"Selected soup '{best_preset}' for task domains {task_domains} "
        f"with alignment {best_alignment:.3f}"
    )

    return best_preset, soups[best_preset]


def validate_soup_integrity(soup: PersonaSoup) -> bool:
    """
    Validate the integrity of a PersonaSoup.

    Checks:
    1. Hash verification
    2. Lambda weights sum to 1.0
    3. Voice embedding dimension
    4. Veto domains are union of components

    Args:
        soup: PersonaSoup to validate

    Returns:
        True if soup passes all integrity checks
    """
    # Check hash
    if not soup.verify_hash():
        logger.error(f"Soup {soup.soup_id} failed hash verification")
        return False

    # Check lambda sum
    total_weight = sum(c.lambda_weight for c in soup.components)
    if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
        logger.error(
            f"Soup {soup.soup_id} lambda weights sum to {total_weight}, not 1.0"
        )
        return False

    # Check embedding dimension
    if len(soup.voice_embedding) != VOICE_EMBEDDING_DIM:
        logger.error(
            f"Soup {soup.soup_id} embedding dimension is {len(soup.voice_embedding)}, "
            f"expected {VOICE_EMBEDDING_DIM}"
        )
        return False

    # Check veto domains are union
    expected_veto: Set[VetoDomain] = set()
    for component in soup.components:
        expected_veto.update(component.persona.veto_domains)
    if soup.veto_domains != expected_veto:
        logger.error(
            f"Soup {soup.soup_id} veto domains mismatch: "
            f"expected {expected_veto}, got {soup.veto_domains}"
        )
        return False

    logger.debug(f"Soup {soup.soup_id} passed integrity validation")
    return True


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Constants
    "VOICE_EMBEDDING_DIM",
    "DEFAULT_SNR_BASE",
    "DOMAIN_PREFIX",
    "VERSION",
    "SoupPreset",
    # Data classes
    "PersonaSoupComponent",
    "PersonaSoup",
    "SNRContribution",
    # Core functions
    "l2_normalize",
    "interpolate_embeddings",
    "compose_weighted_prompt",
    "compute_snr_contribution",
    "interpolate_soup",
    # Factory functions
    "create_security_focused_soup",
    "create_creative_focused_soup",
    "create_analysis_focused_soup",
    "create_balanced_soup",
    "create_guardian_council_soup",
    "create_standard_soups",
    # Utility functions
    "get_soup_for_task",
    "validate_soup_integrity",
]
