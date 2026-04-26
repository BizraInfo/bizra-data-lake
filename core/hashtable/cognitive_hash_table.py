"""
Cognitive Hash Table — State-of-the-Art O(1) Context Retrieval.

Standing on Giants:
    Pagh & Rodler (2001) — Cuckoo Hashing
    Knuth (1973) — The Art of Computer Programming (Vol 3: Searching and Sorting)
    Bloom (1970) — Space/Time Trade-offs in Hash Coding
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger("bizra.hashtable.cognitive")


class CognitiveHashTable:
    """A high-performance O(1) hash table using Cuckoo Hashing.

    Cuckoo hashing uses two hash functions to ensure worst-case O(1) lookup
    and deletion. Insertion is amortized O(1).

    This replaces traditional O(log N) or high-load O(N) structures for
    storing symbolic-neural bridge mappings.
    """

    def __init__(self, initial_capacity: int = 1024, max_load_factor: float = 0.5):
        self._capacity = initial_capacity
        self._max_load_factor = max_load_factor
        self._size = 0

        # Two tables for cuckoo hashing
        self._table1: List[Optional[Tuple[str, Any]]] = [None] * self._capacity
        self._table2: List[Optional[Tuple[str, Any]]] = [None] * self._capacity

        self._max_displacements = 50  # Prevent infinite loops during re-insertion

    def _hash1(self, key: str) -> int:
        # MD5 here is a non-security hash used only to spread keys across
        # cuckoo-table slot indices. usedforsecurity=False satisfies bandit B324
        # without changing behavior (Python 3.9+ stdlib).
        return int(hashlib.md5(key.encode(), usedforsecurity=False).hexdigest(), 16) % self._capacity

    def _hash2(self, key: str) -> int:
        # SHA-1 here is a non-security hash used only to spread keys across
        # cuckoo-table slot indices. usedforsecurity=False satisfies bandit B324
        # without changing behavior (Python 3.9+ stdlib).
        return int(hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest(), 16) % self._capacity

    def put(self, key: str, value: Any) -> None:
        """Insert or update a key-value pair."""
        if self._size / (2 * self._capacity) > self._max_load_factor:
            self._rehash()

        if self.get(key) is not None:
            # Update existing
            self._update(key, value)
            return

        curr_key, curr_val = key, value
        for _ in range(self._max_displacements):
            # Try table 1
            idx1 = self._hash1(curr_key)
            entry1 = self._table1[idx1]
            if entry1 is None:
                self._table1[idx1] = (curr_key, curr_val)
                self._size += 1
                return

            # Evict from table 1
            prev_key, prev_val = entry1
            self._table1[idx1] = (curr_key, curr_val)

            # Move evicted item to table 2
            curr_key, curr_val = prev_key, prev_val
            idx2 = self._hash2(curr_key)
            entry2 = self._table2[idx2]
            if entry2 is None:
                self._table2[idx2] = (curr_key, curr_val)
                self._size += 1
                return

            # Evict from table 2
            prev_key, prev_val = entry2
            self._table2[idx2] = (curr_key, curr_val)
            curr_key, curr_val = prev_key, prev_val

        # If we reach here, we need to rehash
        self._rehash()
        self.put(curr_key, curr_val)

    def get(self, key: str) -> Optional[Any]:
        """O(1) lookup in two possible slots."""
        idx1 = self._hash1(key)
        entry1 = self._table1[idx1]
        if entry1 is not None and entry1[0] == key:
            return entry1[1]

        idx2 = self._hash2(key)
        entry2 = self._table2[idx2]
        if entry2 is not None and entry2[0] == key:
            return entry2[1]

        return None

    def _update(self, key: str, value: Any) -> None:
        idx1 = self._hash1(key)
        entry1 = self._table1[idx1]
        if entry1 is not None and entry1[0] == key:
            self._table1[idx1] = (key, value)
            return

        idx2 = self._hash2(key)
        entry2 = self._table2[idx2]
        if entry2 is not None and entry2[0] == key:
            self._table2[idx2] = (key, value)

    def _rehash(self) -> None:
        """Double size and re-insert all elements."""
        old_table1 = self._table1
        old_table2 = self._table2

        self._capacity *= 2
        self._size = 0
        self._table1 = [None] * self._capacity
        self._table2 = [None] * self._capacity

        for entry in old_table1:
            if entry is not None:
                self.put(entry[0], entry[1])
        for entry in old_table2:
            if entry is not None:
                self.put(entry[0], entry[1])

        logger.info(f"CognitiveHashTable rehashed to capacity {self._capacity}")

    def __len__(self) -> int:
        return self._size
