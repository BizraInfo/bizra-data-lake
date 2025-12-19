"""
core/snr.py — Signal-to-Noise Ratio budget enforcement

SNR is governance, not style. This module enforces output contracts
to prevent verbosity inflation and ensure high-value responses.

Usage:
    from core.snr import SNRBudget, enforce_snr_budget
    
    budget = SNRBudget.from_context(stakes="H", task_type="query")
    result = enforce_snr_budget(output_text, budget)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class BudgetTier(str, Enum):
    """SNR budget tiers."""
    MINIMAL = "minimal"      # Ultra-concise: receipts, status, errors
    CONCISE = "concise"      # Default: answer + evidence refs, no elaboration
    STANDARD = "standard"    # Normal: answer + brief context
    EXPANDED = "expanded"    # Explicit request for depth
    UNLIMITED = "unlimited"  # No budget (testing only)


@dataclass
class SNRBudget:
    """
    SNR budget configuration.
    
    Attributes:
        tier: Budget tier
        max_tokens: Hard token limit for output
        max_words: Soft word limit (warning, not enforced)
        require_evidence: Require evidence_refs in output
        allow_speculation: Allow speculative content
        warn_ratio: SNR ratio below which to warn
    """
    tier: BudgetTier
    max_tokens: int
    max_words: int
    require_evidence: bool = True
    allow_speculation: bool = False
    warn_ratio: float = 0.5
    
    @classmethod
    def from_context(
        cls,
        stakes: str = "M",
        task_type: str = "general",
        explicit_tier: Optional[str] = None
    ) -> "SNRBudget":
        """
        Create budget from context.
        
        Args:
            stakes: H/M/L stakes level
            task_type: query/ingest/elevation/feedback
            explicit_tier: Override tier if user explicitly requests
        
        Returns:
            Configured SNRBudget
        """
        # Explicit override wins
        if explicit_tier:
            tier = BudgetTier(explicit_tier)
        # High stakes → more concise
        elif stakes == "H":
            tier = BudgetTier.MINIMAL
        # Query tasks → concise
        elif task_type in ("query", "gate"):
            tier = BudgetTier.CONCISE
        else:
            tier = BudgetTier.STANDARD
        
        return cls.for_tier(tier)
    
    @classmethod
    def for_tier(cls, tier: BudgetTier) -> "SNRBudget":
        """Get budget configuration for a tier."""
        configs = {
            BudgetTier.MINIMAL: cls(
                tier=tier,
                max_tokens=500,
                max_words=100,
                require_evidence=True,
                allow_speculation=False,
                warn_ratio=0.7
            ),
            BudgetTier.CONCISE: cls(
                tier=tier,
                max_tokens=1500,
                max_words=300,
                require_evidence=True,
                allow_speculation=False,
                warn_ratio=0.5
            ),
            BudgetTier.STANDARD: cls(
                tier=tier,
                max_tokens=4000,
                max_words=800,
                require_evidence=True,
                allow_speculation=True,
                warn_ratio=0.4
            ),
            BudgetTier.EXPANDED: cls(
                tier=tier,
                max_tokens=8000,
                max_words=2000,
                require_evidence=False,
                allow_speculation=True,
                warn_ratio=0.3
            ),
            BudgetTier.UNLIMITED: cls(
                tier=tier,
                max_tokens=100000,
                max_words=50000,
                require_evidence=False,
                allow_speculation=True,
                warn_ratio=0.0
            ),
        }
        return configs.get(tier, configs[BudgetTier.CONCISE])


@dataclass
class SNRResult:
    """Result of SNR enforcement check."""
    passed: bool
    tier: BudgetTier
    token_count: int
    word_count: int
    estimated_ratio: float
    warnings: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "tier": self.tier.value,
            "token_count": self.token_count,
            "word_count": self.word_count,
            "estimated_ratio": self.estimated_ratio,
            "warnings": self.warnings,
            "violations": self.violations
        }


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.
    
    Uses simple heuristic: ~4 chars per token for English.
    For accurate counts, use tiktoken or model-specific tokenizer.
    """
    return len(text) // 4


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def estimate_snr(
    output_text: str,
    input_text: str = "",
    evidence_count: int = 0
) -> float:
    """
    Estimate Signal-to-Noise Ratio.
    
    Higher is better:
    - 1.0 = pure signal (evidence-backed, minimal fluff)
    - 0.0 = pure noise (no evidence, all filler)
    
    Factors:
    - Evidence density: evidence_count / word_count
    - Conciseness: output/input ratio (lower is more concise)
    - Content markers: presence of speculation/filler phrases
    """
    output_words = count_words(output_text)
    if output_words == 0:
        return 1.0  # Empty output is technically infinite SNR
    
    # Evidence density (0-0.5 contribution)
    evidence_score = min(evidence_count / max(output_words / 50, 1), 1.0) * 0.5
    
    # Conciseness (0-0.3 contribution)
    input_words = count_words(input_text) or 1
    ratio = output_words / input_words
    # Ideal ratio is 1-3x input; penalize >5x
    if ratio <= 3:
        concise_score = 0.3
    elif ratio <= 5:
        concise_score = 0.2
    else:
        concise_score = max(0, 0.3 - (ratio - 5) * 0.05)
    
    # Noise markers (0-0.2 penalty)
    noise_phrases = [
        "in essence", "fundamentally", "it's worth noting",
        "as we can see", "let me explain", "to put it simply",
        "in other words", "that being said", "having said that",
        "needless to say", "it goes without saying",
        "at the end of the day", "when all is said and done"
    ]
    noise_count = sum(1 for phrase in noise_phrases if phrase.lower() in output_text.lower())
    noise_penalty = min(noise_count * 0.05, 0.2)
    
    # Speculation markers (0-0.1 penalty)
    speculation_phrases = [
        "might be", "could be", "possibly", "perhaps",
        "it seems", "appears to", "likely", "probably"
    ]
    spec_count = sum(1 for phrase in speculation_phrases if phrase.lower() in output_text.lower())
    spec_penalty = min(spec_count * 0.02, 0.1)
    
    # Base score
    base_score = 0.5  # Start at middle
    
    return max(0, min(1, base_score + evidence_score + concise_score - noise_penalty - spec_penalty))


def enforce_snr_budget(
    output_text: str,
    budget: SNRBudget,
    input_text: str = "",
    evidence_refs: Optional[List[Any]] = None
) -> SNRResult:
    """
    Enforce SNR budget on output.
    
    Args:
        output_text: Generated output to check
        budget: SNR budget to enforce
        input_text: Original input (for ratio calculation)
        evidence_refs: Evidence references in output
    
    Returns:
        SNRResult with pass/fail and diagnostics
    """
    evidence_refs = evidence_refs or []
    
    token_count = estimate_tokens(output_text)
    word_count = count_words(output_text)
    snr = estimate_snr(output_text, input_text, len(evidence_refs))
    
    warnings: List[str] = []
    violations: List[str] = []
    
    # Check token limit
    if token_count > budget.max_tokens:
        violations.append(f"Token limit exceeded: {token_count} > {budget.max_tokens}")
    elif token_count > budget.max_tokens * 0.9:
        warnings.append(f"Near token limit: {token_count}/{budget.max_tokens}")
    
    # Check word limit (soft)
    if word_count > budget.max_words:
        warnings.append(f"Word count high: {word_count} > {budget.max_words}")
    
    # Check evidence requirement
    if budget.require_evidence and len(evidence_refs) == 0:
        violations.append("No evidence references provided (required by budget)")
    
    # Check SNR ratio
    if snr < budget.warn_ratio:
        warnings.append(f"Low SNR: {snr:.2f} < {budget.warn_ratio}")
    
    passed = len(violations) == 0
    
    return SNRResult(
        passed=passed,
        tier=budget.tier,
        token_count=token_count,
        word_count=word_count,
        estimated_ratio=snr,
        warnings=warnings,
        violations=violations
    )


def get_snr_guidance(budget: SNRBudget) -> str:
    """
    Get SNR guidance text for prompts.
    
    Include this in system prompts to guide model output.
    """
    tier_guidance = {
        BudgetTier.MINIMAL: """
Output requirements (MINIMAL tier):
- Maximum 100 words
- Evidence-backed claims only
- No elaboration or context
- Format: direct answer + receipt reference
""",
        BudgetTier.CONCISE: """
Output requirements (CONCISE tier):
- Maximum 300 words
- Evidence-backed claims only
- Brief context if essential
- No speculation or qualifiers
- Format: answer + evidence refs + brief explanation
""",
        BudgetTier.STANDARD: """
Output requirements (STANDARD tier):
- Maximum 800 words
- Evidence-backed claims preferred
- Speculation marked explicitly
- Include relevant context
- Format: structured response with evidence
""",
        BudgetTier.EXPANDED: """
Output requirements (EXPANDED tier):
- Extended analysis allowed
- Include background and context
- Multiple perspectives welcome
- Mark speculation vs evidence
""",
        BudgetTier.UNLIMITED: """
Output requirements (UNLIMITED tier):
- No restrictions
- Full analysis permitted
""",
    }
    
    return tier_guidance.get(budget.tier, tier_guidance[BudgetTier.CONCISE])


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

def get_default_budget() -> SNRBudget:
    """Get default SNR budget from environment."""
    tier_str = os.environ.get("BIZRA_SNR_TIER", "concise")
    try:
        tier = BudgetTier(tier_str)
    except ValueError:
        tier = BudgetTier.CONCISE
    return SNRBudget.for_tier(tier)
