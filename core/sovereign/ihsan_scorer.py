"""
Unified 8D Ihsān Content Scorer (§4, §8)
==========================================

Single source of truth for quality scoring across the organism.
Replaces the duplicate heuristics in mission_nervous_system.py and
mission_pipeline.py with a unified, content-based scorer.

Each of the 8 canonical dimensions is scored by analyzing actual
content — structure, evidence markers, vocabulary, coherence —
not surface statistics like word count.

DDAGI Pilot v2.0 §4: Constitutional invariants (Ihsān ≥ 0.95 production)
DDAGI Pilot v2.0 §8: SNR/Ihsān dimensions and output gates

Standing on Giants:
  Al-Ghazali (intent gate, 1096) · Shannon (SNR, 1948)
  Kahneman (dual-process, 2011) · البذرة Rule 2 (Ihsān floor 0.85)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from core.integration.constants import IHSAN_CANONICAL_WEIGHTS

# ═══════════════════════════════════════════════════════════════════
# §8 SNR WEIGHTS (from DDAGI Pilot v2.0)
# ═══════════════════════════════════════════════════════════════════

SNR_WEIGHTS: Dict[str, float] = {
    "signal_density": 0.35,
    "evidence_grounding": 0.25,
    "contradiction_resolution": 0.20,
    "actionability": 0.20,
}

# ═══════════════════════════════════════════════════════════════════
# CONTENT ANALYSIS PRIMITIVES
# ═══════════════════════════════════════════════════════════════════

# Harmful content patterns (P5-Ethicist assist)
_HARM_PATTERNS = [
    re.compile(
        r"\b(kill|harm|destroy|attack|exploit)\s+(the\s+)?(user|person|people|system)",
        re.I,
    ),
    re.compile(r"\b(bypass|disable|ignore)\s+(security|safety|gate|auth)", re.I),
    re.compile(r"\b(steal|hack|crack|brute.?force)\b", re.I),
]

# Epistemic humility markers
_HUMBLE_MARKERS = re.compile(
    r"\b(may|might|could|possibly|likely|uncertain|approximate|"
    r"it.s possible|not certain|more research|caveat|limitation|"
    r"trade.?off|depends on|context.?dependent)\b",
    re.I,
)

# Overclaiming markers
_OVERCLAIM_MARKERS = re.compile(
    r"\b(always|never|definitely|guaranteed|impossible|"
    r"perfect|flawless|absolutely|undoubtedly|100%)\b",
    re.I,
)

# Evidence markers (verifiability)
_EVIDENCE_MARKERS = re.compile(
    r"(?:"
    r"\b[a-zA-Z_/][a-zA-Z0-9_./]+\.(py|rs|ts|tsx|js|yml|yaml|json|toml|md)\b"  # file paths
    r"|:?\d{1,5}(?:[-–]\d{1,5})?"  # line numbers
    r"|\bhttps?://\S+"  # URLs
    r"|\b(?:test|assert|expect|verify|pytest|cargo test)\b"  # test refs
    r"|\b(?:Ihsān|ihsan|SNR|snr|Gini|gini)\s*[≥><=]+\s*[\d.]+"  # threshold refs
    r"|\b(?:§\d+|Spine|DDAGI|constitutional)\b"  # spec refs
    r")",
    re.I,
)

# Structural markers
_STRUCTURE_MARKERS = re.compile(
    r"(?:"
    r"^#{1,6}\s"  # Markdown headers
    r"|^[-*•]\s"  # List items
    r"|^\d+\.\s"  # Numbered lists
    r"|```"  # Code blocks
    r"|\n\n"  # Paragraph breaks
    r"|:\s*$"  # Colon-terminated labels
    r")",
    re.MULTILINE,
)

# Actionable step markers
_ACTION_MARKERS = re.compile(
    r"(?:"
    r"^\s*[-*•]\s"  # List steps
    r"|^\s*\d+\.\s"  # Numbered steps
    r"|\b(?:run|execute|install|create|build|deploy|test|configure|add|remove|update)\b"
    r"|\b(?:should|must|need to|recommend|suggest)\b"
    r"|```(?:bash|python|sh|rust)"  # Code snippets
    r")",
    re.MULTILINE | re.I,
)

# Contradiction markers
_CONTRADICTION_MARKERS = re.compile(
    r"\b(?:however|but|although|conversely|on the other hand|"
    r"contradicts|inconsistent|conflict|despite|whereas|"
    r"nevertheless|in contrast)\b",
    re.I,
)

# Resolution markers (show contradictions were addressed)
_RESOLUTION_MARKERS = re.compile(
    r"\b(?:therefore|thus|consequently|as a result|"
    r"the solution|to resolve|balancing|reconcil|"
    r"trade.?off|given this|accordingly|in summary)\b",
    re.I,
)


@dataclass
class IhsanTensor:
    """8D Ihsān quality tensor (§4).

    Each dimension is scored [0.0, 1.0] based on content analysis.
    Composite is the weighted geometric mean — zero in ANY dimension
    means zero composite (Al-Ghazali fail-closed principle).
    """

    moral_clarity: float = 0.0
    epistemic_humility: float = 0.0
    structural_integrity: float = 0.0
    verifiability: float = 0.0
    contextual_relevance: float = 0.0
    intent_alignment: float = 0.0
    resilience: float = 0.0
    efficiency: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "moral_clarity": self.moral_clarity,
            "epistemic_humility": self.epistemic_humility,
            "structural_integrity": self.structural_integrity,
            "verifiability": self.verifiability,
            "contextual_relevance": self.contextual_relevance,
            "intent_alignment": self.intent_alignment,
            "resilience": self.resilience,
            "efficiency": self.efficiency,
        }

    @property
    def composite(self) -> float:
        """Weighted geometric mean (§4: zero in any → zero composite)."""
        return _weighted_geometric_mean(self.as_dict(), IHSAN_CANONICAL_WEIGHTS)


@dataclass
class SNRScore:
    """4D SNR quality score (§8).

    Shannon-inspired signal-to-noise measurement across four dimensions.
    """

    signal_density: float = 0.0
    evidence_grounding: float = 0.0
    contradiction_resolution: float = 0.0
    actionability: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "signal_density": self.signal_density,
            "evidence_grounding": self.evidence_grounding,
            "contradiction_resolution": self.contradiction_resolution,
            "actionability": self.actionability,
        }

    @property
    def composite(self) -> float:
        """Weighted linear combination per §8 weights."""
        d = self.as_dict()
        return round(sum(d[k] * SNR_WEIGHTS[k] for k in SNR_WEIGHTS), 4)


# ═══════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════


def _weighted_geometric_mean(
    scores: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """Weighted geometric mean — zero in any dimension → zero composite.

    Al-Ghazali §4: you cannot compensate for being unethical
    by being highly efficient.
    """
    log_sum = 0.0
    weight_sum = 0.0
    for dim, score in scores.items():
        w = weights.get(dim, 0.0)
        if w <= 0:
            continue
        if score <= 0.0:
            return 0.0  # Fail-closed
        log_sum += w * math.log(score)
        weight_sum += w

    if weight_sum <= 0:
        return 0.0
    return round(math.exp(log_sum / weight_sum), 4)


def _tokenize(text: str) -> List[str]:
    """Simple word tokenizer."""
    return text.split()


def _count_sentences(text: str) -> int:
    """Approximate sentence count."""
    return max(len(re.split(r"[.!?]+", text.strip())) - 1, 1)


def _ngrams(words: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generate n-grams from word list."""
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


# ═══════════════════════════════════════════════════════════════════
# 8D IHSĀN CONTENT SCORING (§4)
# ═══════════════════════════════════════════════════════════════════


def score_moral_clarity(output: str) -> float:
    """Dimension 1: وضوح أخلاقي — ethical transparency.

    Measures absence of harmful patterns and presence of clear,
    honest framing. Base = 1.0 minus harm penalties.
    """
    if not output.strip():
        return 0.0

    score = 1.0

    # Penalize harmful content patterns
    for pattern in _HARM_PATTERNS:
        matches = pattern.findall(output)
        score -= 0.3 * len(matches)

    # Reward transparent framing
    words = _tokenize(output)
    if len(words) < 3:
        return max(0.1, score)

    # Check for deception markers (false urgency, manipulation)
    deception = re.findall(
        r"\b(act now|limited time|don.t tell anyone|secret|"
        r"no one will know|trust me)\b",
        output,
        re.I,
    )
    score -= 0.15 * len(deception)

    return round(max(0.0, min(1.0, score)), 4)


def score_epistemic_humility(output: str) -> float:
    """Dimension 2: تواضع معرفي — knowing what you don't know.

    Rewards hedging/qualification and penalizes overclaiming.
    """
    if not output.strip():
        return 0.0

    words = _tokenize(output)
    word_count = len(words)
    if word_count < 3:
        return 0.3

    humble_count = len(_HUMBLE_MARKERS.findall(output))
    overclaim_count = len(_OVERCLAIM_MARKERS.findall(output))

    # Normalize by text length (per 100 words)
    humble_rate = min(humble_count / max(word_count / 100, 1), 1.0)
    overclaim_rate = min(overclaim_count / max(word_count / 100, 1), 1.0)

    # Balance: some humility is good, overclaiming is bad
    base = 0.6
    base += humble_rate * 0.3
    base -= overclaim_rate * 0.25

    # Very short outputs get a slight penalty (less room for nuance)
    if word_count < 15:
        base *= 0.85

    return round(max(0.0, min(1.0, base)), 4)


def score_structural_integrity(output: str) -> float:
    """Dimension 3: سلامة بنيوية — coherent architecture.

    Measures structural markers: headers, lists, code blocks,
    paragraph breaks, and logical flow.
    """
    if not output.strip():
        return 0.0

    markers = _STRUCTURE_MARKERS.findall(output)
    marker_count = len(markers)

    words = _tokenize(output)
    word_count = len(words)
    sentences = _count_sentences(output)

    # Structural density (markers per 50 words)
    struct_density = min(marker_count / max(word_count / 50, 1), 1.0)

    # Paragraph coherence (multiple sentences suggest structure)
    para_score = min(sentences / 3, 1.0)

    # Balanced sentence length (not all one-word or all 100-word sentences)
    avg_sentence_len = word_count / max(sentences, 1)
    length_balance = 1.0
    if avg_sentence_len < 3:
        length_balance = 0.5
    elif avg_sentence_len > 50:
        length_balance = 0.7

    base = struct_density * 0.4 + para_score * 0.35 + length_balance * 0.25

    # Minimum floor for any non-empty output
    return round(max(0.15, min(1.0, base)), 4)


def score_verifiability(output: str, input_text: str = "") -> float:
    """Dimension 4: قابلية التحقق — provable claims.

    Counts evidence markers: file paths, line numbers, URLs,
    test references, spec references, threshold citations.
    """
    if not output.strip():
        return 0.0

    evidence_hits = _EVIDENCE_MARKERS.findall(output)
    evidence_count = len(evidence_hits)

    sentences = _count_sentences(output)

    # Evidence density (per sentence)
    evidence_density = min(evidence_count / max(sentences, 1), 1.0)

    # Has code or structured data
    has_code = 1.0 if "```" in output or "    " in output else 0.0

    # Claims vs evidence ratio
    claim_markers = len(
        re.findall(r"\b(is|are|was|were|will|should|must)\b", output, re.I)
    )
    evidence_ratio = min(evidence_count / max(claim_markers / 3, 1), 1.0)

    base = evidence_density * 0.45 + evidence_ratio * 0.35 + has_code * 0.20

    return round(max(0.1, min(1.0, base)), 4)


def score_contextual_relevance(output: str, input_text: str) -> float:
    """Dimension 5: ملاءمة سياقية — right answer, right time.

    Measures how well the output addresses the input via
    token overlap and n-gram similarity.
    """
    if not output.strip():
        return 0.0
    if not input_text.strip():
        return 0.5  # No context to judge relevance — neutral, not penalizing

    input_words = set(w.lower() for w in _tokenize(input_text) if len(w) > 2)
    output_words = set(w.lower() for w in _tokenize(output) if len(w) > 2)

    if not input_words:
        return 0.5

    # Word overlap (Jaccard-like)
    overlap = input_words & output_words
    coverage = len(overlap) / max(len(input_words), 1)

    # Bigram overlap for phrase-level relevance
    input_bigrams = set(_ngrams([w.lower() for w in _tokenize(input_text)], 2))
    output_bigrams = set(_ngrams([w.lower() for w in _tokenize(output)], 2))
    bigram_overlap = len(input_bigrams & output_bigrams) / max(len(input_bigrams), 1)

    base = coverage * 0.6 + bigram_overlap * 0.2 + 0.2  # 0.2 baseline

    return round(max(0.1, min(1.0, base)), 4)


def score_intent_alignment(output: str, input_text: str) -> float:
    """Dimension 6: توافق النية — serves the user's true need.

    Checks if the output matches the input's question type
    (how/what/why/action request) and provides appropriate response.
    """
    if not output.strip():
        return 0.0
    if not input_text.strip():
        return 0.5  # No context to judge intent — neutral, not penalizing

    input_lower = input_text.lower()
    output_lower = output.lower()

    score = 0.5  # Baseline

    # Question type detection
    is_how = re.search(r"\bhow\b", input_lower) is not None
    is_what = re.search(r"\bwhat\b", input_lower) is not None
    is_why = re.search(r"\bwhy\b", input_lower) is not None
    is_action = (
        re.search(
            r"\b(implement|create|build|fix|add|remove|update|deploy|test|run)\b",
            input_lower,
        )
        is not None
    )

    # Check if response type matches
    has_steps = bool(_ACTION_MARKERS.findall(output))
    has_explanation = len(output_lower) > 100 and _count_sentences(output) > 2
    has_code = (
        "```" in output or re.search(r"def |class |fn |let |const ", output) is not None
    )

    if is_action and (has_steps or has_code):
        score += 0.3
    elif is_how and has_steps:
        score += 0.25
    elif is_what and has_explanation:
        score += 0.25
    elif is_why and has_explanation:
        score += 0.25
    elif has_explanation:
        score += 0.15

    # Reward direct address of input keywords
    input_keywords = set(w.lower() for w in _tokenize(input_text) if len(w) > 3)
    output_words = set(w.lower() for w in _tokenize(output))
    keyword_hit = len(input_keywords & output_words) / max(len(input_keywords), 1)
    score += keyword_hit * 0.2

    return round(max(0.1, min(1.0, score)), 4)


def score_resilience(output: str) -> float:
    """Dimension 7: مرونة — graceful under failure.

    Measures vocabulary diversity (not repetitive), error handling
    mentions, and edge case consideration.
    """
    if not output.strip():
        return 0.0

    words = _tokenize(output)
    word_count = len(words)
    if word_count < 3:
        return 0.2

    # Vocabulary diversity
    unique_ratio = len(set(w.lower() for w in words)) / word_count

    # Error handling awareness
    error_awareness = len(
        re.findall(
            r"\b(error|exception|fallback|degrad|recover|retry|timeout|"
            r"edge case|corner case|failure|handle|catch|safeguard)\b",
            output,
            re.I,
        )
    )
    error_score = min(error_awareness / 3, 1.0)

    # Not just a single approach — considers alternatives
    alternative_markers = len(
        re.findall(
            r"\b(alternatively|another approach|option|trade.?off|"
            r"if .+ fails|backup|plan B)\b",
            output,
            re.I,
        )
    )
    alternative_score = min(alternative_markers / 2, 1.0)

    base = unique_ratio * 0.5 + error_score * 0.3 + alternative_score * 0.2

    return round(max(0.1, min(1.0, base)), 4)


def score_efficiency(output: str) -> float:
    """Dimension 8: كفاءة — minimum waste, maximum signal.

    Measures signal density: content words vs filler, unique ideas
    vs repetition, conciseness relative to content.
    """
    if not output.strip():
        return 0.0

    words = _tokenize(output)
    word_count = len(words)
    if word_count < 3:
        return 0.3

    # Filler detection
    filler_words = {
        "very",
        "really",
        "just",
        "actually",
        "basically",
        "literally",
        "simply",
        "obviously",
        "clearly",
        "of",
        "the",
        "a",
        "an",
        "that",
        "this",
        "it",
        "is",
        "are",
        "was",
        "were",
        "be",
    }
    content_words = [w for w in words if w.lower() not in filler_words and len(w) > 2]
    content_ratio = len(content_words) / max(word_count, 1)

    # Repetition penalty (repeated phrases)
    bigrams = _ngrams([w.lower() for w in words], 2)
    unique_bigrams = set(bigrams)
    repetition_ratio = len(unique_bigrams) / max(len(bigrams), 1)

    # Length appropriateness (not absurdly long or trivially short)
    length_score = 1.0
    if word_count < 10:
        length_score = 0.6
    elif word_count > 1000:
        # Diminishing returns for very long outputs
        length_score = max(0.5, 1.0 - (word_count - 1000) / 5000)

    base = content_ratio * 0.35 + repetition_ratio * 0.35 + length_score * 0.30

    return round(max(0.1, min(1.0, base)), 4)


# ═══════════════════════════════════════════════════════════════════
# SNR CONTENT SCORING (§8)
# ═══════════════════════════════════════════════════════════════════


def score_signal_density(output: str) -> float:
    """SNR §8: actionable_insights / total_tokens (weight: 0.35)."""
    if not output.strip():
        return 0.0

    words = _tokenize(output)
    word_count = len(words)
    if word_count == 0:
        return 0.0

    # Actionable content: commands, decisions, recommendations
    actionable = len(
        re.findall(
            r"\b(should|must|recommend|implement|create|fix|add|run|"
            r"deploy|configure|verify|test|ensure|use)\b",
            output,
            re.I,
        )
    )

    return round(min(actionable / max(word_count / 10, 1), 1.0), 4)


def score_evidence_grounding(output: str) -> float:
    """SNR §8: sourced_claims / total_claims (weight: 0.25)."""
    if not output.strip():
        return 0.0

    evidence_count = len(_EVIDENCE_MARKERS.findall(output))
    sentences = _count_sentences(output)

    return round(min(evidence_count / max(sentences, 1), 1.0), 4)


def score_contradiction_resolution(output: str) -> float:
    """SNR §8: 1.0 - (unresolved / total_branches) (weight: 0.20)."""
    if not output.strip():
        return 0.0

    contradictions = len(_CONTRADICTION_MARKERS.findall(output))
    resolutions = len(_RESOLUTION_MARKERS.findall(output))

    if contradictions == 0:
        return 0.9  # No contradictions is good but not perfect

    resolution_rate = min(resolutions / max(contradictions, 1), 1.0)
    return round(0.5 + resolution_rate * 0.5, 4)


def score_actionability(output: str) -> float:
    """SNR §8: executable_steps / total_recommendations (weight: 0.20)."""
    if not output.strip():
        return 0.0

    action_items = len(_ACTION_MARKERS.findall(output))
    sentences = _count_sentences(output)

    return round(min(action_items / max(sentences, 1), 1.0), 4)


# ═══════════════════════════════════════════════════════════════════
# UNIFIED PUBLIC API
# ═══════════════════════════════════════════════════════════════════


def score_ihsan_8d(output: str, input_text: str = "") -> IhsanTensor:
    """Score output quality across all 8 canonical Ihsān dimensions.

    This is the single source of truth for Ihsān scoring in the organism.
    Used by P4-Evaluator (mission_pipeline) and NervousSystem (S2 path).

    Args:
        output: The text to evaluate (inference output).
        input_text: The original mission/query text (for relevance scoring).

    Returns:
        IhsanTensor with all 8 dimensions scored [0.0, 1.0].
    """
    return IhsanTensor(
        moral_clarity=score_moral_clarity(output),
        epistemic_humility=score_epistemic_humility(output),
        structural_integrity=score_structural_integrity(output),
        verifiability=score_verifiability(output, input_text),
        contextual_relevance=score_contextual_relevance(output, input_text),
        intent_alignment=score_intent_alignment(output, input_text),
        resilience=score_resilience(output),
        efficiency=score_efficiency(output),
    )


def score_ihsan_composite(output: str, input_text: str = "") -> float:
    """Convenience: compute Ihsān composite (geometric mean of 8D tensor).

    Returns [0.0, 1.0]. Zero in ANY dimension → zero composite.
    """
    return score_ihsan_8d(output, input_text).composite


def score_snr(output: str, input_text: str = "") -> SNRScore:
    """Score output quality across 4 SNR dimensions (§8).

    Args:
        output: The text to evaluate.
        input_text: The original query (unused for most SNR dims).

    Returns:
        SNRScore with 4 dimensions per §8 weights.
    """
    return SNRScore(
        signal_density=score_signal_density(output),
        evidence_grounding=score_evidence_grounding(output),
        contradiction_resolution=score_contradiction_resolution(output),
        actionability=score_actionability(output),
    )


def score_snr_composite(output: str, input_text: str = "") -> float:
    """Convenience: compute SNR composite (weighted linear combination)."""
    return score_snr(output, input_text).composite
