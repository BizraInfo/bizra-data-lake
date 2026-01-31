"""
IaaS Framework Core — The Foundational Data Quality Standard

Standing on Giants:
- DATA4LLM IaaS Framework (Tsinghua University, 2024)
- DDAGI Constitution v1.1.0 (Ihsān Constraint)

"A good dataset is a purposefully balanced and rigorously sanitized
 collection of broad, diverse, and well-articulated data."
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import math
import logging

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """The four pillars of IaaS data quality."""
    INCLUSIVENESS = "inclusiveness"   # Domains, Languages, Modalities, Sources
    ABUNDANCE = "abundance"           # Scale with quality preservation
    ARTICULATION = "articulation"     # Well-formatted, instructive, step-by-step
    SANITIZATION = "sanitization"     # Privacy, ethics, risk removal


@dataclass
class IaaSConfig:
    """Configuration for IaaS quality enforcement."""

    # Ihsān thresholds (DDAGI Constitution Article 7)
    ihsan_minimum: float = 0.95
    ihsan_target: float = 0.99

    # Dimension weights (empirically derived from DATA4LLM)
    weights: Dict[QualityDimension, float] = field(default_factory=lambda: {
        QualityDimension.INCLUSIVENESS: 0.25,
        QualityDimension.ABUNDANCE: 0.20,
        QualityDimension.ARTICULATION: 0.30,
        QualityDimension.SANITIZATION: 0.25,
    })

    # Inclusiveness requirements
    min_domains: int = 3
    min_languages: int = 1
    min_modalities: int = 1

    # Abundance requirements
    min_chunks: int = 100
    max_redundancy: float = 0.15

    # Articulation requirements
    max_perplexity: float = 100.0
    min_instruction_clarity: float = 0.7

    # Sanitization requirements
    max_pii_density: float = 0.001
    max_toxicity_score: float = 0.1
    ethics_compliance: bool = True

    def validate(self) -> bool:
        """Validate configuration consistency."""
        total_weight = sum(self.weights.values())
        if not math.isclose(total_weight, 1.0, rel_tol=1e-5):
            logger.warning(f"IaaS weights sum to {total_weight}, normalizing...")
            for dim in self.weights:
                self.weights[dim] /= total_weight
        return True


@dataclass
class DimensionScore:
    """Score for a single IaaS dimension."""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    components: Dict[str, float] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0 and self.score >= 0.7


@dataclass
class IaaSScore:
    """
    Comprehensive IaaS quality score for a dataset or chunk collection.

    The IaaS score is computed as a weighted geometric mean of dimension scores,
    following the principle that ALL dimensions must be adequate (geometric mean
    penalizes low scores more heavily than arithmetic mean).
    """

    inclusiveness: DimensionScore
    abundance: DimensionScore
    articulation: DimensionScore
    sanitization: DimensionScore

    weights: Dict[QualityDimension, float] = field(default_factory=dict)

    @property
    def composite_score(self) -> float:
        """
        Compute weighted geometric mean of dimension scores.

        Formula: exp(Σ wᵢ × log(scoreᵢ))

        This ensures that a zero in any dimension produces a zero composite,
        and low scores are penalized more heavily than in arithmetic mean.
        """
        if not self.weights:
            self.weights = {
                QualityDimension.INCLUSIVENESS: 0.25,
                QualityDimension.ABUNDANCE: 0.20,
                QualityDimension.ARTICULATION: 0.30,
                QualityDimension.SANITIZATION: 0.25,
            }

        scores = {
            QualityDimension.INCLUSIVENESS: max(self.inclusiveness.score, 1e-10),
            QualityDimension.ABUNDANCE: max(self.abundance.score, 1e-10),
            QualityDimension.ARTICULATION: max(self.articulation.score, 1e-10),
            QualityDimension.SANITIZATION: max(self.sanitization.score, 1e-10),
        }

        weighted_log_sum = sum(
            self.weights[dim] * math.log(scores[dim])
            for dim in QualityDimension
        )

        return math.exp(weighted_log_sum)

    @property
    def ihsan_achieved(self) -> bool:
        """Check if Ihsān threshold (0.95) is achieved."""
        return self.composite_score >= 0.95

    @property
    def all_dimensions_passed(self) -> bool:
        """Check if all individual dimensions passed their thresholds."""
        return all([
            self.inclusiveness.passed,
            self.abundance.passed,
            self.articulation.passed,
            self.sanitization.passed,
        ])

    @property
    def violations(self) -> List[str]:
        """Aggregate all violations across dimensions."""
        all_violations = []
        all_violations.extend(self.inclusiveness.violations)
        all_violations.extend(self.abundance.violations)
        all_violations.extend(self.articulation.violations)
        all_violations.extend(self.sanitization.violations)
        return all_violations

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging and persistence."""
        return {
            "composite_score": self.composite_score,
            "ihsan_achieved": self.ihsan_achieved,
            "dimensions": {
                "inclusiveness": {
                    "score": self.inclusiveness.score,
                    "components": self.inclusiveness.components,
                    "violations": self.inclusiveness.violations,
                },
                "abundance": {
                    "score": self.abundance.score,
                    "components": self.abundance.components,
                    "violations": self.abundance.violations,
                },
                "articulation": {
                    "score": self.articulation.score,
                    "components": self.articulation.components,
                    "violations": self.articulation.violations,
                },
                "sanitization": {
                    "score": self.sanitization.score,
                    "components": self.sanitization.components,
                    "violations": self.sanitization.violations,
                },
            },
            "all_violations": self.violations,
        }


class IaaSFramework:
    """
    The IaaS Framework Engine — Core data quality enforcement.

    Integrates:
    - Deduplication (MinHash, SimHash, Semantic, SoftDeDup)
    - Filtering (Perplexity, IFD, Cluster Complexity)
    - Sanitization (PII, Ethics, Toxicity)
    - Mixing (Domain reweighting, DRO)

    Produces:
    - IaaSScore for any dataset or chunk collection
    - Actionable violations and remediation hints
    """

    def __init__(self, config: Optional[IaaSConfig] = None):
        self.config = config or IaaSConfig()
        self.config.validate()

        # Lazy-loaded components
        self._deduplication_engine = None
        self._quality_filter = None
        self._sanitization_engine = None
        self._domain_mixer = None

    def score_inclusiveness(
        self,
        domains: List[str],
        languages: List[str],
        modalities: List[str],
        sources: List[str],
    ) -> DimensionScore:
        """
        Score the Inclusiveness dimension.

        Measures: Domain coverage, language diversity, modality support, source variety.
        """
        violations = []
        components = {}

        # Domain coverage
        n_domains = len(set(domains))
        domain_score = min(n_domains / max(self.config.min_domains, 1), 1.0)
        components["domain_coverage"] = domain_score
        if n_domains < self.config.min_domains:
            violations.append(f"Insufficient domains: {n_domains} < {self.config.min_domains}")

        # Language diversity
        n_languages = len(set(languages))
        language_score = min(n_languages / max(self.config.min_languages, 1), 1.0)
        components["language_diversity"] = language_score
        if n_languages < self.config.min_languages:
            violations.append(f"Insufficient languages: {n_languages} < {self.config.min_languages}")

        # Modality support
        n_modalities = len(set(modalities))
        modality_score = min(n_modalities / max(self.config.min_modalities, 1), 1.0)
        components["modality_support"] = modality_score
        if n_modalities < self.config.min_modalities:
            violations.append(f"Insufficient modalities: {n_modalities} < {self.config.min_modalities}")

        # Source variety (bonus, not required)
        n_sources = len(set(sources))
        source_score = min(n_sources / 10, 1.0)  # Cap at 10 sources for max score
        components["source_variety"] = source_score

        # Composite inclusiveness score
        composite = (
            0.35 * domain_score +
            0.25 * language_score +
            0.25 * modality_score +
            0.15 * source_score
        )

        return DimensionScore(
            dimension=QualityDimension.INCLUSIVENESS,
            score=composite,
            components=components,
            violations=violations,
        )

    def score_abundance(
        self,
        total_chunks: int,
        unique_chunks: int,
        total_tokens: int,
    ) -> DimensionScore:
        """
        Score the Abundance dimension.

        Measures: Scale, uniqueness, redundancy rate.
        """
        violations = []
        components = {}

        # Scale score (logarithmic, rewards larger datasets)
        scale_score = min(math.log10(max(total_chunks, 1)) / 6, 1.0)  # Max at 1M chunks
        components["scale"] = scale_score
        if total_chunks < self.config.min_chunks:
            violations.append(f"Insufficient chunks: {total_chunks} < {self.config.min_chunks}")

        # Uniqueness score
        uniqueness = unique_chunks / max(total_chunks, 1)
        uniqueness_score = uniqueness
        components["uniqueness"] = uniqueness_score

        # Redundancy check
        redundancy = 1 - uniqueness
        if redundancy > self.config.max_redundancy:
            violations.append(f"High redundancy: {redundancy:.2%} > {self.config.max_redundancy:.2%}")
        components["redundancy_rate"] = redundancy

        # Token density (tokens per chunk)
        token_density = total_tokens / max(total_chunks, 1)
        density_score = min(token_density / 500, 1.0)  # Optimal ~500 tokens/chunk
        components["token_density"] = density_score

        # Composite abundance score
        composite = (
            0.30 * scale_score +
            0.40 * uniqueness_score +
            0.30 * density_score
        )

        return DimensionScore(
            dimension=QualityDimension.ABUNDANCE,
            score=composite,
            components=components,
            violations=violations,
        )

    def score_articulation(
        self,
        mean_perplexity: float,
        format_compliance_rate: float,
        instruction_clarity: float,
        reasoning_depth: float,
    ) -> DimensionScore:
        """
        Score the Articulation dimension.

        Measures: Perplexity (fluency), format compliance, instruction clarity, reasoning depth.
        """
        violations = []
        components = {}

        # Perplexity score (lower is better, inverse mapping)
        # PPL < 20 = excellent, PPL > 100 = poor
        perplexity_score = max(0, 1 - (mean_perplexity / self.config.max_perplexity))
        components["perplexity"] = perplexity_score
        if mean_perplexity > self.config.max_perplexity:
            violations.append(f"High perplexity: {mean_perplexity:.1f} > {self.config.max_perplexity}")

        # Format compliance
        components["format_compliance"] = format_compliance_rate
        if format_compliance_rate < 0.9:
            violations.append(f"Low format compliance: {format_compliance_rate:.2%}")

        # Instruction clarity
        components["instruction_clarity"] = instruction_clarity
        if instruction_clarity < self.config.min_instruction_clarity:
            violations.append(f"Low instruction clarity: {instruction_clarity:.2f} < {self.config.min_instruction_clarity}")

        # Reasoning depth (step-by-step)
        components["reasoning_depth"] = reasoning_depth

        # Composite articulation score
        composite = (
            0.30 * perplexity_score +
            0.25 * format_compliance_rate +
            0.25 * instruction_clarity +
            0.20 * reasoning_depth
        )

        return DimensionScore(
            dimension=QualityDimension.ARTICULATION,
            score=composite,
            components=components,
            violations=violations,
        )

    def score_sanitization(
        self,
        pii_density: float,
        toxicity_score: float,
        bias_score: float,
        ethics_compliance: bool,
    ) -> DimensionScore:
        """
        Score the Sanitization dimension.

        Measures: PII exposure, toxicity, bias, ethics compliance.
        """
        violations = []
        components = {}

        # PII density (lower is better)
        pii_score = max(0, 1 - (pii_density / self.config.max_pii_density))
        components["pii_protection"] = pii_score
        if pii_density > self.config.max_pii_density:
            violations.append(f"PII density exceeded: {pii_density:.4f} > {self.config.max_pii_density}")

        # Toxicity (lower is better)
        toxicity_clean = max(0, 1 - toxicity_score)
        components["toxicity_free"] = toxicity_clean
        if toxicity_score > self.config.max_toxicity_score:
            violations.append(f"Toxicity threshold exceeded: {toxicity_score:.2f} > {self.config.max_toxicity_score}")

        # Bias score (lower is better)
        bias_free = max(0, 1 - bias_score)
        components["bias_free"] = bias_free
        if bias_score > 0.2:
            violations.append(f"Bias detected: {bias_score:.2f}")

        # Ethics compliance (binary)
        ethics_score = 1.0 if ethics_compliance else 0.0
        components["ethics_compliance"] = ethics_score
        if not ethics_compliance and self.config.ethics_compliance:
            violations.append("Ethics compliance requirement not met")

        # Composite sanitization score
        composite = (
            0.30 * pii_score +
            0.30 * toxicity_clean +
            0.20 * bias_free +
            0.20 * ethics_score
        )

        return DimensionScore(
            dimension=QualityDimension.SANITIZATION,
            score=composite,
            components=components,
            violations=violations,
        )

    def compute_iaas_score(
        self,
        # Inclusiveness inputs
        domains: List[str],
        languages: List[str],
        modalities: List[str],
        sources: List[str],
        # Abundance inputs
        total_chunks: int,
        unique_chunks: int,
        total_tokens: int,
        # Articulation inputs
        mean_perplexity: float,
        format_compliance_rate: float,
        instruction_clarity: float,
        reasoning_depth: float,
        # Sanitization inputs
        pii_density: float,
        toxicity_score: float,
        bias_score: float,
        ethics_compliance: bool,
    ) -> IaaSScore:
        """
        Compute comprehensive IaaS score for a dataset.

        This is the main entry point for IaaS quality assessment.
        """
        inclusiveness = self.score_inclusiveness(domains, languages, modalities, sources)
        abundance = self.score_abundance(total_chunks, unique_chunks, total_tokens)
        articulation = self.score_articulation(
            mean_perplexity, format_compliance_rate, instruction_clarity, reasoning_depth
        )
        sanitization = self.score_sanitization(
            pii_density, toxicity_score, bias_score, ethics_compliance
        )

        return IaaSScore(
            inclusiveness=inclusiveness,
            abundance=abundance,
            articulation=articulation,
            sanitization=sanitization,
            weights=self.config.weights,
        )

    def validate_for_production(self, score: IaaSScore) -> bool:
        """
        Validate if a dataset meets production requirements.

        Per DDAGI Constitution Article 7:
        - Composite IaaS score must be >= 0.95 (Ihsān threshold)
        - All individual dimensions must pass their thresholds
        - No critical violations
        """
        if not score.ihsan_achieved:
            logger.warning(
                f"IaaS score {score.composite_score:.4f} below Ihsān threshold 0.95"
            )
            return False

        if not score.all_dimensions_passed:
            logger.warning(
                f"Not all IaaS dimensions passed: {score.violations}"
            )
            return False

        logger.info(f"IaaS validation PASSED: {score.composite_score:.4f} >= 0.95")
        return True
