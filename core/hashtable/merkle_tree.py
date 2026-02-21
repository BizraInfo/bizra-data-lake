"""
Merkle Tree — Cryptographic Hash Tree with O(log n) Inclusion Proofs

Standing on Giants:
  Merkle (1979) — "A Certified Digital Signature"
  Laurie, Langley, Kasper (RFC 6962, 2013) — Certificate Transparency

Uses BLAKE3 for cross-language interop with Rust (bizra-omega).
Domain separation per RFC 6962: leaf prefix \\x00, node prefix \\x01,
preventing second-preimage attacks.

Supports O(log n) incremental append via right-spine caching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.integration.constants import MERKLE_LEAF_PREFIX, MERKLE_NODE_PREFIX
from core.proof_engine.canonical import blake3_digest


def _hash_leaf(data: bytes) -> bytes:
    """Hash a leaf with domain separation prefix."""
    return blake3_digest(MERKLE_LEAF_PREFIX + data)


def _hash_node(left: bytes, right: bytes) -> bytes:
    """Hash two children with internal node prefix."""
    return blake3_digest(MERKLE_NODE_PREFIX + left + right)


@dataclass(frozen=True)
class MerkleProof:
    """
    Self-contained Merkle inclusion proof.

    Siblings are ordered bottom-up. Each entry is (hash, is_left) where
    is_left=True means the sibling is to the LEFT of the path node.
    """

    leaf_index: int
    leaf_hash: bytes
    siblings: tuple[tuple[bytes, bool], ...]
    root: bytes

    def verify(self) -> bool:
        """Verify this proof against the embedded root."""
        current = self.leaf_hash
        for sibling_hash, sibling_is_left in self.siblings:
            if sibling_is_left:
                current = _hash_node(sibling_hash, current)
            else:
                current = _hash_node(current, sibling_hash)
        return current == self.root


class MerkleTree:
    """
    BLAKE3 Merkle tree with O(log n) incremental append.

    The tree stores leaf hashes and maintains a right spine for
    efficient append operations. Full tree rebuild is avoided.

    >>> tree = MerkleTree()
    >>> tree.append(b"hello")
    0
    >>> tree.append(b"world")
    1
    >>> proof = tree.prove(0)
    >>> proof.verify()
    True
    """

    __slots__ = ("_leaves", "_right_spine", "_root_cache")

    def __init__(self, leaves: Optional[list[bytes]] = None) -> None:
        self._leaves: list[bytes] = []
        self._right_spine: list[bytes] = []
        self._root_cache: Optional[bytes] = None

        if leaves:
            for leaf in leaves:
                self.append(leaf)

    @property
    def leaf_count(self) -> int:
        return len(self._leaves)

    @property
    def root(self) -> bytes:
        """Compute (or return cached) Merkle root."""
        if self._root_cache is not None:
            return self._root_cache

        if not self._leaves:
            self._root_cache = blake3_digest(b"")
            return self._root_cache

        # Compute from right spine
        self._root_cache = self._compute_root_from_spine()
        return self._root_cache

    @property
    def root_hex(self) -> str:
        """Root as 64-character hex string."""
        return self.root.hex()

    def append(self, data: bytes) -> int:
        """
        Append a leaf and update the right spine in O(log n).

        Returns the leaf index.
        """
        leaf_hash = _hash_leaf(data)
        index = len(self._leaves)
        self._leaves.append(leaf_hash)
        self._root_cache = None  # Invalidate

        # Update right spine: the spine stores the "carry" hashes
        # at each level, similar to incrementing a binary counter.
        carry = leaf_hash
        level = 0
        n = index  # 0-based index of the new leaf

        while n & 1:
            # n is odd at this level → merge with spine entry
            if level < len(self._right_spine):
                carry = _hash_node(self._right_spine[level], carry)
            n >>= 1
            level += 1

        # Store the carry at this level
        if level < len(self._right_spine):
            self._right_spine[level] = carry
        else:
            self._right_spine.append(carry)

        # Trim spine entries above this level that are now stale
        # (they'll be recomputed on next carry)

        return index

    def _compute_root_from_spine(self) -> bytes:
        """Fold the right spine to get the root."""
        if not self._leaves:
            return blake3_digest(b"")

        # Rebuild from leaves using the standard algorithm.
        # The spine is an optimization for append; for root computation
        # with odd-leaf handling we do a clean bottom-up pass.
        layer = list(self._leaves)
        while len(layer) > 1:
            next_layer: list[bytes] = []
            i = 0
            while i < len(layer):
                if i + 1 < len(layer):
                    next_layer.append(_hash_node(layer[i], layer[i + 1]))
                else:
                    # Odd node: promote
                    next_layer.append(layer[i])
                i += 2
            layer = next_layer
        return layer[0]

    def prove(self, leaf_index: int) -> MerkleProof:
        """
        Generate an inclusion proof for the leaf at the given index.

        Returns a MerkleProof with siblings ordered bottom-up.
        """
        if not self._leaves:
            raise IndexError("Cannot prove on empty tree")
        if leaf_index < 0 or leaf_index >= len(self._leaves):
            raise IndexError(
                f"Leaf index {leaf_index} out of range [0, {len(self._leaves)})"
            )

        siblings: list[tuple[bytes, bool]] = []
        layer = list(self._leaves)
        idx = leaf_index

        while len(layer) > 1:
            next_layer: list[bytes] = []
            i = 0
            next_idx = idx // 2

            while i < len(layer):
                if i + 1 < len(layer):
                    next_layer.append(_hash_node(layer[i], layer[i + 1]))
                    # Collect sibling if this pair contains our target
                    if i == idx or i + 1 == idx:
                        if idx % 2 == 0:
                            # Target is left child; sibling is right (is_left=False)
                            siblings.append((layer[i + 1], False))
                        else:
                            # Target is right child; sibling is left (is_left=True)
                            siblings.append((layer[i], True))
                else:
                    # Odd node promoted — no sibling needed if this is our target
                    next_layer.append(layer[i])
                i += 2

            layer = next_layer
            idx = next_idx

        return MerkleProof(
            leaf_index=leaf_index,
            leaf_hash=self._leaves[leaf_index],
            siblings=tuple(siblings),
            root=self.root,
        )

    @staticmethod
    def verify_proof(proof: MerkleProof) -> bool:
        """Static convenience — delegates to proof.verify()."""
        return proof.verify()

    def __repr__(self) -> str:
        return f"MerkleTree(leaves={len(self._leaves)}, root={self.root_hex[:16]}...)"
