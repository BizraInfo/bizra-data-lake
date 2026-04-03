// src/merkle.rs - Batch Merkle Tree for BIZRA Proof Chain
//
// Provides O(log n) verification for batch receipt anchoring.
// Uses domain separation to prevent second preimage attacks.
//
// # Integration Example
//
// ```rust,ignore
// use bizra::merkle::{MerkleTree, hash_leaf};
// use bizra::autopoietic::proof_chain::ProofChain;
//
// let mut chain = ProofChain::new();
// // ... append multiple generations ...
//
// // Batch anchor: collect receipt hashes from multiple generations
// let receipt_hashes: Vec<String> = (1..=100)
//     .filter_map(|gen| chain.get(gen))
//     .map(|node| node.evolution_proof.receipt_id.clone())
//     .collect();
//
// // Build Merkle tree
// let tree = MerkleTree::build(&receipt_hashes);
// let merkle_root = tree.root();
//
// // Generate proof for specific receipt
// let proof = tree.proof(42).unwrap();
//
// // Verify proof (can be done off-chain)
// assert!(MerkleTree::verify(&merkle_root, receipt_hashes[42].as_bytes(), &proof));
//
// // Anchor merkle_root to blockchain (single transaction for 100 receipts)
// // anchor_to_blockchain(merkle_root, generations: 1..=100);
// ```

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Side of sibling in Merkle path
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Left,
    Right,
}

/// Merkle inclusion proof
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerkleProof {
    /// Siblings along path from leaf to root
    pub siblings: Vec<([u8; 32], Side)>,
    /// Index of the leaf
    pub leaf_index: usize,
    /// Hash of the leaf
    pub leaf_hash: [u8; 32],
}

/// Merkle tree for batch verification
#[derive(Debug, Clone)]
pub struct MerkleTree {
    /// All node hashes (leaves + internal nodes)
    nodes: Vec<[u8; 32]>,
    /// Number of leaves
    leaf_count: usize,
}

impl MerkleTree {
    /// Build Merkle tree from leaf data
    pub fn build(items: &[impl AsRef<[u8]>]) -> Self {
        if items.is_empty() {
            return Self {
                nodes: vec![],
                leaf_count: 0,
            };
        }

        // Hash all leaves with domain separation
        let mut leaves: Vec<[u8; 32]> = items.iter().map(|item| hash_leaf(item.as_ref())).collect();

        let original_count = leaves.len();

        // For odd number of leaves, duplicate the last one
        if leaves.len() % 2 == 1 {
            leaves.push(*leaves.last().unwrap());
        }

        let mut nodes = leaves.clone();
        let mut current_level = leaves;

        // Build tree bottom-up
        while current_level.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in current_level.chunks(2) {
                let left = chunk[0];
                let right = chunk.get(1).copied().unwrap_or(left);
                let parent = hash_node(&left, &right);
                next_level.push(parent);
            }

            nodes.extend_from_slice(&next_level);
            current_level = next_level;

            // Duplicate last node if odd
            if current_level.len() > 1 && current_level.len() % 2 == 1 {
                current_level.push(*current_level.last().unwrap());
            }
        }

        Self {
            nodes,
            leaf_count: original_count,
        }
    }

    /// Get root hash
    pub fn root(&self) -> [u8; 32] {
        if self.nodes.is_empty() {
            return [0u8; 32];
        }
        *self.nodes.last().unwrap()
    }

    /// Get number of leaves
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// Generate inclusion proof for leaf at index
    pub fn proof(&self, index: usize) -> Option<MerkleProof> {
        if index >= self.leaf_count || self.nodes.is_empty() {
            return None;
        }

        let leaf_hash = self.nodes[index];
        let mut siblings = Vec::new();

        // Adjust for duplication if odd
        let padded_count = if self.leaf_count % 2 == 1 {
            self.leaf_count + 1
        } else {
            self.leaf_count
        };

        let mut current_index = index;
        let mut level_size = padded_count;
        let mut level_offset = 0;

        while level_size > 1 {
            let sibling_index = if current_index.is_multiple_of(2) {
                current_index + 1
            } else {
                current_index - 1
            };

            let sibling_hash = if sibling_index < level_size {
                self.nodes[level_offset + sibling_index]
            } else {
                // If sibling doesn't exist, use current node (duplication)
                self.nodes[level_offset + current_index]
            };

            let side = if current_index.is_multiple_of(2) {
                Side::Right
            } else {
                Side::Left
            };

            siblings.push((sibling_hash, side));

            // Move to next level
            level_offset += level_size;
            current_index /= 2;
            level_size = level_size.div_ceil(2);
        }

        Some(MerkleProof {
            siblings,
            leaf_index: index,
            leaf_hash,
        })
    }

    /// Verify inclusion proof
    pub fn verify(root: &[u8; 32], leaf_data: &[u8], proof: &MerkleProof) -> bool {
        // Hash the leaf data and verify it matches proof
        let computed_leaf = hash_leaf(leaf_data);
        if computed_leaf != proof.leaf_hash {
            return false;
        }

        // Walk up the tree
        let mut current_hash = proof.leaf_hash;

        for (sibling_hash, side) in &proof.siblings {
            current_hash = match side {
                Side::Left => hash_node(sibling_hash, &current_hash),
                Side::Right => hash_node(&current_hash, sibling_hash),
            };
        }

        &current_hash == root
    }

    /// Verify multiple proofs against same root
    pub fn batch_verify(root: &[u8; 32], items: &[(usize, &[u8])], proofs: &[MerkleProof]) -> bool {
        if items.len() != proofs.len() {
            return false;
        }

        for ((index, data), proof) in items.iter().zip(proofs.iter()) {
            if proof.leaf_index != *index {
                return false;
            }
            if !Self::verify(root, data, proof) {
                return false;
            }
        }

        true
    }
}

/// Hash a leaf with domain separation (0x00 prefix)
pub fn hash_leaf(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update([0x00]); // Leaf domain separator
    hasher.update(data);
    hasher.finalize().into()
}

/// Hash an internal node with domain separation (0x01 prefix)
pub fn hash_node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update([0x01]); // Node domain separator
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree() {
        let items: Vec<&[u8]> = vec![];
        let tree = MerkleTree::build(&items);
        assert_eq!(tree.leaf_count(), 0);
        assert_eq!(tree.root(), [0u8; 32]);
        assert!(tree.proof(0).is_none());
    }

    #[test]
    fn test_single_leaf() {
        let items = vec![b"single"];
        let tree = MerkleTree::build(&items);
        assert_eq!(tree.leaf_count(), 1);

        // Single leaf gets duplicated, so root = hash_node(leaf, leaf)
        let leaf_hash = hash_leaf(b"single");
        let expected_root = hash_node(&leaf_hash, &leaf_hash);
        assert_eq!(tree.root(), expected_root);

        let proof = tree.proof(0).unwrap();
        assert!(MerkleTree::verify(&tree.root(), b"single", &proof));
    }

    #[test]
    fn test_two_leaves() {
        let items = vec![b"leaf0", b"leaf1"];
        let tree = MerkleTree::build(&items);
        assert_eq!(tree.leaf_count(), 2);

        let leaf0 = hash_leaf(b"leaf0");
        let leaf1 = hash_leaf(b"leaf1");
        let expected_root = hash_node(&leaf0, &leaf1);
        assert_eq!(tree.root(), expected_root);

        // Verify both proofs
        let proof0 = tree.proof(0).unwrap();
        assert!(MerkleTree::verify(&tree.root(), b"leaf0", &proof0));

        let proof1 = tree.proof(1).unwrap();
        assert!(MerkleTree::verify(&tree.root(), b"leaf1", &proof1));
    }

    #[test]
    fn test_power_of_two_tree() {
        let items: Vec<String> = (0..8).map(|i| format!("item{}", i)).collect();
        let items_ref: Vec<&str> = items.iter().map(|s| s.as_str()).collect();

        let tree = MerkleTree::build(&items_ref);
        assert_eq!(tree.leaf_count(), 8);

        // Verify all proofs
        for i in 0..8 {
            let proof = tree.proof(i).unwrap();
            assert!(
                MerkleTree::verify(&tree.root(), items[i].as_bytes(), &proof),
                "Failed to verify item {}",
                i
            );
        }
    }

    #[test]
    fn test_non_power_of_two_tree() {
        let items: Vec<String> = (0..7).map(|i| format!("item{}", i)).collect();
        let items_ref: Vec<&str> = items.iter().map(|s| s.as_str()).collect();

        let tree = MerkleTree::build(&items_ref);
        assert_eq!(tree.leaf_count(), 7);

        // Verify all proofs
        for i in 0..7 {
            let proof = tree.proof(i).unwrap();
            assert!(
                MerkleTree::verify(&tree.root(), items[i].as_bytes(), &proof),
                "Failed to verify item {}",
                i
            );
        }
    }

    #[test]
    fn test_large_tree() {
        let items: Vec<String> = (0..1000).map(|i| format!("receipt_{}", i)).collect();
        let items_ref: Vec<&str> = items.iter().map(|s| s.as_str()).collect();

        let tree = MerkleTree::build(&items_ref);
        assert_eq!(tree.leaf_count(), 1000);

        // Verify random proofs
        for &i in &[0, 50, 123, 456, 789, 999] {
            let proof = tree.proof(i).unwrap();
            assert!(
                MerkleTree::verify(&tree.root(), items[i].as_bytes(), &proof),
                "Failed to verify item {}",
                i
            );
        }
    }

    #[test]
    fn test_tamper_detection() {
        let items = vec![b"leaf0", b"leaf1", b"leaf2", b"leaf3"];
        let tree = MerkleTree::build(&items);

        let proof = tree.proof(1).unwrap();

        // Correct data verifies
        assert!(MerkleTree::verify(&tree.root(), b"leaf1", &proof));

        // Tampered data fails
        assert!(!MerkleTree::verify(&tree.root(), b"tampered", &proof));
        assert!(!MerkleTree::verify(&tree.root(), b"leaf2", &proof));
    }

    #[test]
    fn test_wrong_root() {
        let items = vec![b"leaf0", b"leaf1", b"leaf2"];
        let tree = MerkleTree::build(&items);

        let proof = tree.proof(1).unwrap();

        // Wrong root fails
        let wrong_root = [0xff; 32];
        assert!(!MerkleTree::verify(&wrong_root, b"leaf1", &proof));
    }

    #[test]
    fn test_batch_verify() {
        let items: Vec<String> = (0..16).map(|i| format!("item{}", i)).collect();
        let items_ref: Vec<&str> = items.iter().map(|s| s.as_str()).collect();

        let tree = MerkleTree::build(&items_ref);
        let root = tree.root();

        // Generate multiple proofs
        let indices = vec![2, 5, 9, 14];
        let proofs: Vec<MerkleProof> = indices.iter().map(|&i| tree.proof(i).unwrap()).collect();

        let batch_items: Vec<(usize, &[u8])> = indices
            .iter()
            .map(|&i| (i, items[i].as_bytes()))
            .collect();

        // Batch verify should succeed
        assert!(MerkleTree::batch_verify(&root, &batch_items, &proofs));

        // Tamper one item - batch verify should fail
        let mut tampered_batch = batch_items.clone();
        tampered_batch[1].1 = b"tampered";
        assert!(!MerkleTree::batch_verify(&root, &tampered_batch, &proofs));
    }

    #[test]
    fn test_out_of_bounds_proof() {
        let items = vec![b"leaf0", b"leaf1", b"leaf2"];
        let tree = MerkleTree::build(&items);

        assert!(tree.proof(100).is_none());
    }

    #[test]
    fn test_determinism() {
        let items: Vec<String> = (0..100).map(|i| format!("data{}", i)).collect();
        let items_ref: Vec<&str> = items.iter().map(|s| s.as_str()).collect();

        let tree1 = MerkleTree::build(&items_ref);
        let tree2 = MerkleTree::build(&items_ref);

        assert_eq!(tree1.root(), tree2.root());
        assert_eq!(tree1.leaf_count(), tree2.leaf_count());

        // Same proofs
        for i in 0..10 {
            let proof1 = tree1.proof(i).unwrap();
            let proof2 = tree2.proof(i).unwrap();
            assert_eq!(proof1, proof2);
        }
    }

    #[test]
    fn test_domain_separation() {
        // Leaf and node hashes should be different even with same data
        let data = b"test_data";
        let leaf_hash = hash_leaf(data);

        let mut hasher = Sha256::new();
        hasher.update(data);
        let raw_hash: [u8; 32] = hasher.finalize().into();

        // Leaf hash should differ from raw hash
        assert_ne!(leaf_hash, raw_hash);

        // Node hash should differ from concatenating hashes
        let hash1 = [0u8; 32];
        let hash2 = [1u8; 32];
        let node_hash = hash_node(&hash1, &hash2);

        let mut hasher = Sha256::new();
        hasher.update(&hash1);
        hasher.update(&hash2);
        let concat_hash: [u8; 32] = hasher.finalize().into();

        assert_ne!(node_hash, concat_hash);
    }

    #[test]
    fn test_proof_structure() {
        let items = vec![b"a", b"b", b"c", b"d"];
        let tree = MerkleTree::build(&items);

        let proof = tree.proof(0).unwrap();
        assert_eq!(proof.leaf_index, 0);
        assert_eq!(proof.leaf_hash, hash_leaf(b"a"));

        // For 4 leaves, path length should be 2 (log2(4))
        assert_eq!(proof.siblings.len(), 2);

        // First sibling should be on the right (sibling of leaf 0 is leaf 1)
        assert_eq!(proof.siblings[0].1, Side::Right);
    }
}
