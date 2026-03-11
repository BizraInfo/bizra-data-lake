"""Cross-platform deduplication via content-hash.

When the same prompt or response appears across multiple platforms (e.g. the
user copied a prompt from ChatGPT into Claude), this module detects the
overlap via normalized BLAKE3 hashing and keeps only the earliest instance.

Ref: specs/user-zero-bootstrap/phase_01_multi_platform_ingestion.md S4

Standing on Giants: Shannon (information entropy) - Broder (min-hash)
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict

from core.genesis.ingestion.schema import ConversationTurn

log = logging.getLogger(__name__)

DEDUP_DOMAIN = "genesis/dedup/v1"


def _content_hash(content: str) -> str:
    """BLAKE3 content hash with whitespace normalization for cross-platform dedup.

    Uses hashlib.blake2b fallback when the blake3 package is not installed.
    """
    normalized = re.sub(r"\s+", " ", content.strip().lower())
    try:
        import blake3

        h = blake3.blake3(derive_key_context=DEDUP_DOMAIN)
        h.update(normalized.encode())
        return h.hexdigest()[:32]
    except ImportError:
        import hashlib

        h = hashlib.blake2b(
            f"{DEDUP_DOMAIN}:{normalized}".encode(),
            digest_size=16,
        )
        return h.hexdigest()


def deduplicate(turns: list[ConversationTurn]) -> list[ConversationTurn]:
    """Deduplicate conversation turns by normalized content hash.

    When duplicates span platforms, the earliest turn (by insertion order)
    is kept and duplicate platform info is recorded in the canonical turn's
    metadata under ``duplicate_platforms`` and ``duplicate_ids``.

    Returns:
        Sorted list of unique turns (by timestamp, nulls last).
    """
    seen: OrderedDict[str, ConversationTurn] = OrderedDict()
    dup_count = 0

    for turn in turns:
        ch = _content_hash(turn.content)
        turn_copy = turn.model_copy()
        turn_copy.content_hash = ch

        if ch not in seen:
            seen[ch] = turn_copy
        else:
            existing = seen[ch]
            dup_platforms: list[str] = existing.metadata.get("duplicate_platforms", [])
            dup_platforms.append(turn.platform.value)
            existing.metadata["duplicate_platforms"] = dup_platforms
            dup_ids: list[str] = existing.metadata.get("duplicate_ids", [])
            dup_ids.append(turn.id)
            existing.metadata["duplicate_ids"] = dup_ids
            dup_count += 1
            log.debug(
                "Duplicate: %s (%s) matches %s (%s)",
                turn.id[:8],
                turn.platform.value,
                existing.id[:8],
                existing.platform.value,
            )

    # Sort by timestamp; turns without timestamps sort to the end.
    def _sort_key(t: ConversationTurn) -> str:
        if t.timestamp is not None:
            return t.timestamp.isoformat()
        return "\xff"  # Sorts after all valid ISO timestamps

    result = sorted(seen.values(), key=_sort_key)
    log.info(
        "Dedup: %d turns -> %d unique (%d duplicates removed)",
        len(turns),
        len(result),
        dup_count,
    )
    return result
