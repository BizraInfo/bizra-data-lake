"""
Tests for MerkleTree — Phase 44

Standing on Giants: Merkle (1979), RFC 6962 (2013)
"""

import pytest

from core.hashtable.merkle_tree import MerkleProof, MerkleTree, _hash_leaf, _hash_node
from core.integration.constants import MERKLE_LEAF_PREFIX, MERKLE_NODE_PREFIX
from core.proof_engine.canonical import blake3_digest


class TestMerkleTreeConstruction:
    """Tree building and root computation."""

    def test_single_leaf_root(self):
        tree = MerkleTree([b"hello"])
        expected = blake3_digest(MERKLE_LEAF_PREFIX + b"hello")
        assert tree.root == expected

    def test_two_leaf_root(self):
        tree = MerkleTree([b"left", b"right"])
        h_left = blake3_digest(MERKLE_LEAF_PREFIX + b"left")
        h_right = blake3_digest(MERKLE_LEAF_PREFIX + b"right")
        expected = blake3_digest(MERKLE_NODE_PREFIX + h_left + h_right)
        assert tree.root == expected

    def test_three_leaf_root(self):
        """Odd leaf count — third leaf is promoted."""
        tree = MerkleTree([b"a", b"b", b"c"])
        ha = _hash_leaf(b"a")
        hb = _hash_leaf(b"b")
        hc = _hash_leaf(b"c")
        hab = _hash_node(ha, hb)
        root = _hash_node(hab, hc)
        assert tree.root == root

    def test_four_leaf_root(self):
        tree = MerkleTree([b"a", b"b", b"c", b"d"])
        ha, hb = _hash_leaf(b"a"), _hash_leaf(b"b")
        hc, hd = _hash_leaf(b"c"), _hash_leaf(b"d")
        hab = _hash_node(ha, hb)
        hcd = _hash_node(hc, hd)
        root = _hash_node(hab, hcd)
        assert tree.root == root

    def test_empty_tree_root(self):
        tree = MerkleTree()
        assert tree.root == blake3_digest(b"")

    def test_root_deterministic(self):
        """Same leaves → same root, always."""
        leaves = [f"leaf-{i}".encode() for i in range(10)]
        tree1 = MerkleTree(leaves)
        tree2 = MerkleTree(leaves)
        assert tree1.root == tree2.root

    def test_different_leaves_different_root(self):
        tree1 = MerkleTree([b"a", b"b"])
        tree2 = MerkleTree([b"c", b"d"])
        assert tree1.root != tree2.root


class TestMerkleTreeRootHex:
    """Root hex representation."""

    def test_root_hex_is_64_chars(self):
        tree = MerkleTree([b"test"])
        assert len(tree.root_hex) == 64

    def test_root_hex_matches_root(self):
        tree = MerkleTree([b"test"])
        assert tree.root_hex == tree.root.hex()

    def test_root_hex_is_valid_hex(self):
        tree = MerkleTree([b"test"])
        # Should not raise
        bytes.fromhex(tree.root_hex)


class TestMerkleTreeAppend:
    """Incremental append — Fix 3 from plan."""

    def test_append_returns_index(self):
        tree = MerkleTree()
        assert tree.append(b"first") == 0
        assert tree.append(b"second") == 1
        assert tree.append(b"third") == 2

    def test_append_matches_batch_construction(self):
        """Incremental append should give the same root as batch init."""
        leaves = [f"leaf-{i}".encode() for i in range(20)]
        batch_tree = MerkleTree(leaves)

        append_tree = MerkleTree()
        for leaf in leaves:
            append_tree.append(leaf)

        assert append_tree.root == batch_tree.root

    def test_append_single_leaf(self):
        tree = MerkleTree()
        tree.append(b"only")
        expected = blake3_digest(MERKLE_LEAF_PREFIX + b"only")
        assert tree.root == expected

    def test_append_updates_leaf_count(self):
        tree = MerkleTree()
        assert tree.leaf_count == 0
        tree.append(b"a")
        assert tree.leaf_count == 1
        tree.append(b"b")
        assert tree.leaf_count == 2

    def test_append_many_matches_batch(self):
        """Stress test: 100 leaves via append == batch."""
        leaves = [f"data-{i}".encode() for i in range(100)]
        batch = MerkleTree(leaves)
        incremental = MerkleTree()
        for leaf in leaves:
            incremental.append(leaf)
        assert incremental.root == batch.root


class TestMerkleProof:
    """Inclusion proof generation and verification."""

    def test_single_leaf_proof(self):
        tree = MerkleTree([b"only"])
        proof = tree.prove(0)
        assert proof.verify()
        assert proof.leaf_index == 0

    def test_two_leaf_proof_left(self):
        tree = MerkleTree([b"left", b"right"])
        proof = tree.prove(0)
        assert proof.verify()

    def test_two_leaf_proof_right(self):
        tree = MerkleTree([b"left", b"right"])
        proof = tree.prove(1)
        assert proof.verify()

    def test_four_leaf_all_proofs(self):
        tree = MerkleTree([b"a", b"b", b"c", b"d"])
        for i in range(4):
            proof = tree.prove(i)
            assert proof.verify(), f"Proof failed for leaf {i}"

    def test_odd_leaf_count_proofs(self):
        tree = MerkleTree([b"a", b"b", b"c"])
        for i in range(3):
            proof = tree.prove(i)
            assert proof.verify(), f"Proof failed for leaf {i}"

    def test_large_tree_proofs(self):
        leaves = [f"leaf-{i}".encode() for i in range(50)]
        tree = MerkleTree(leaves)
        # Check a sample of proofs
        for i in [0, 1, 24, 25, 49]:
            proof = tree.prove(i)
            assert proof.verify(), f"Proof failed for leaf {i}"

    def test_proof_root_matches_tree_root(self):
        tree = MerkleTree([b"a", b"b", b"c"])
        proof = tree.prove(1)
        assert proof.root == tree.root

    def test_static_verify_proof(self):
        tree = MerkleTree([b"a", b"b"])
        proof = tree.prove(0)
        assert MerkleTree.verify_proof(proof)


class TestMerkleProofTamperDetection:
    """Tampered proofs must fail verification."""

    def test_tampered_leaf_hash(self):
        tree = MerkleTree([b"a", b"b"])
        proof = tree.prove(0)
        tampered = MerkleProof(
            leaf_index=proof.leaf_index,
            leaf_hash=b"\xff" * 32,  # wrong hash
            siblings=proof.siblings,
            root=proof.root,
        )
        assert not tampered.verify()

    def test_tampered_sibling(self):
        tree = MerkleTree([b"a", b"b"])
        proof = tree.prove(0)
        tampered_siblings = ((b"\xff" * 32, False),)
        tampered = MerkleProof(
            leaf_index=proof.leaf_index,
            leaf_hash=proof.leaf_hash,
            siblings=tampered_siblings,
            root=proof.root,
        )
        assert not tampered.verify()

    def test_tampered_root(self):
        tree = MerkleTree([b"a", b"b"])
        proof = tree.prove(0)
        tampered = MerkleProof(
            leaf_index=proof.leaf_index,
            leaf_hash=proof.leaf_hash,
            siblings=proof.siblings,
            root=b"\x00" * 32,
        )
        assert not tampered.verify()


class TestMerkleDomainSeparation:
    """RFC 6962 domain separation prevents second-preimage attacks."""

    def test_leaf_and_node_produce_different_hashes(self):
        """A leaf hash of data X must differ from a node hash of data X."""
        data = b"test-data"
        leaf_h = blake3_digest(MERKLE_LEAF_PREFIX + data)
        node_h = blake3_digest(MERKLE_NODE_PREFIX + data)
        assert leaf_h != node_h

    def test_prefixes_are_different(self):
        assert MERKLE_LEAF_PREFIX != MERKLE_NODE_PREFIX
        assert MERKLE_LEAF_PREFIX == b"\x00"
        assert MERKLE_NODE_PREFIX == b"\x01"


class TestMerkleTreeEdgeCases:
    """Boundary conditions."""

    def test_prove_on_empty_tree_raises(self):
        tree = MerkleTree()
        with pytest.raises(IndexError, match="empty tree"):
            tree.prove(0)

    def test_prove_out_of_range_raises(self):
        tree = MerkleTree([b"only"])
        with pytest.raises(IndexError, match="out of range"):
            tree.prove(1)
        with pytest.raises(IndexError, match="out of range"):
            tree.prove(-1)

    def test_power_of_two_leaves(self):
        """Perfect binary tree (8 leaves)."""
        leaves = [f"leaf-{i}".encode() for i in range(8)]
        tree = MerkleTree(leaves)
        for i in range(8):
            proof = tree.prove(i)
            assert proof.verify()
            # log2(8) = 3 siblings expected
            assert len(proof.siblings) == 3

    def test_duplicate_leaves(self):
        tree = MerkleTree([b"same", b"same"])
        # Both leaves hash the same, but proofs should still work
        p0 = tree.prove(0)
        p1 = tree.prove(1)
        assert p0.verify()
        assert p1.verify()


class TestMerkleTreeRepr:

    def test_repr(self):
        tree = MerkleTree([b"a"])
        r = repr(tree)
        assert "MerkleTree" in r
        assert "leaves=1" in r
