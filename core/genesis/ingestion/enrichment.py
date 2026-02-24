"""Metadata enrichment: token counts, language detection, topic extraction.

Enrichment is a non-destructive pass that fills in computed fields on
each ConversationTurn without altering the original content.

Ref: specs/user-zero-bootstrap/phase_01_multi_platform_ingestion.md S5

Standing on Giants: Shannon (information content) - Zipf (word frequency)
"""

from __future__ import annotations

import logging
import re

from core.genesis.ingestion.schema import ConversationTurn

log = logging.getLogger(__name__)


def estimate_tokens(content: str, model: str | None = None) -> int:
    """Estimate token count for content.

    Uses tiktoken for OpenAI-family models when available; falls back to a
    character-based heuristic (4 chars/token for Latin, 2 chars/token for CJK).
    """
    if model and any(model.startswith(p) for p in ("gpt", "o1", "o3")):
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(content))
        except Exception:
            pass

    if _contains_cjk(content):
        return max(1, len(content) // 2)
    return max(1, len(content) // 4)


def _contains_cjk(text: str) -> bool:
    """Check if text contains CJK characters (sample first 500 chars)."""
    for ch in text[:500]:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            return True
    return False


def detect_language(text: str) -> tuple[str, float]:
    """Detect language of text.

    Returns (ISO 639-1 code, confidence).
    Falls back to ("en", 0.5) when langdetect is not available.
    """
    try:
        from langdetect import detect_langs

        results = detect_langs(text[:1000])
        if results:
            return results[0].lang, results[0].prob
    except ImportError:
        pass
    except Exception:
        pass
    return "en", 0.5


# Common English stop words for topic extraction (4+ letters only).
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "this",
        "that",
        "with",
        "from",
        "have",
        "been",
        "will",
        "would",
        "could",
        "should",
        "about",
        "which",
        "their",
        "there",
        "were",
        "what",
        "when",
        "where",
        "then",
        "than",
        "them",
        "they",
        "your",
        "just",
        "like",
        "also",
        "some",
        "into",
        "more",
        "very",
        "each",
        "make",
        "made",
        "does",
        "here",
        "only",
        "well",
        "back",
        "much",
    }
)


def extract_topics(text: str, max_topics: int = 5) -> list[str]:
    """Extract keyword topics using simple frequency-based extraction.

    Filters stop words and returns the top ``max_topics`` most frequent
    words of length >= 4.
    """
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    filtered = [w for w in words if w not in _STOP_WORDS]

    freq: dict[str, int] = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:max_topics]]


def enrich(turns: list[ConversationTurn]) -> list[ConversationTurn]:
    """Enrich turns with token counts, language, and topics.

    Modifies turns in-place and returns the same list for chaining.
    """
    for turn in turns:
        turn.token_count = estimate_tokens(turn.content, turn.model)

        lang, conf = detect_language(turn.content)
        turn.language = lang
        turn.language_conf = conf

        turn.topics = extract_topics(turn.content)

    log.info("Enriched %d turns with tokens, language, topics", len(turns))
    return turns
