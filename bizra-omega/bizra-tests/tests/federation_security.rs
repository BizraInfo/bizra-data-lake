//! Federation Protocol Security Tests
//!
//! SECURITY: Tests for message signing, verification, and attack prevention
//!
//! Standing on Giants:
//! - Das et al. (2002): SWIM Protocol
//! - Lamport (1982): Byzantine Generals Problem

use bizra_core::{NodeId, NodeIdentity};
use bizra_federation::{
    consensus::{ConsensusEngine, ConsensusError, SignedVote, Vote},
    gossip::{FederationError, GossipMessage, GossipProtocol, SignedGossipMessage},
};
use chrono::{Duration as ChronoDuration, Utc};
use ed25519_dalek::{Signer, SigningKey};

// ============================================================================
// GOSSIP MESSAGE SECURITY
// ============================================================================

/// Test: Valid messages are accepted
#[tokio::test]
async fn test_valid_message_acceptance() {
    let local_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let local_id = "local_node".to_string();
    let peer_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let peer_id = "peer_node".to_string();

    let protocol = GossipProtocol::new(
        NodeId(local_id.clone()),
        "127.0.0.1:7946".parse().unwrap(),
        local_key,
    );

    // Register peer's public key
    protocol
        .register_peer_pubkey(NodeId(peer_id.clone()), peer_key.verifying_key().to_bytes())
        .await;

    // Create valid signed message from peer
    let msg = GossipMessage::Ping {
        from: NodeId(peer_id),
        incarnation: 1,
    };
    let signed = SignedGossipMessage::sign(msg, &peer_key);

    // Should be accepted
    let result = protocol.handle_signed_message(signed).await;
    assert!(result.is_ok(), "Valid message should be accepted");
}

/// Test: Messages with invalid signatures are rejected
#[tokio::test]
async fn test_invalid_signature_rejection() {
    let local_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let peer_key = SigningKey::generate(&mut rand::rngs::OsRng);

    let protocol = GossipProtocol::new(
        NodeId("local".to_string()),
        "127.0.0.1:7946".parse().unwrap(),
        local_key,
    );

    // Create message and sign it
    let msg = GossipMessage::Ping {
        from: NodeId("peer".to_string()),
        incarnation: 1,
    };
    let mut signed = SignedGossipMessage::sign(msg, &peer_key);

    // Corrupt the signature
    signed.signature[0] ^= 0xFF;
    signed.signature[32] ^= 0xFF;

    let result = protocol.handle_signed_message(signed).await;
    assert!(
        matches!(result, Err(FederationError::InvalidSignature)),
        "Corrupted signature should be rejected"
    );
}

/// Test: Messages from unknown senders are rejected
#[tokio::test]
async fn test_unknown_sender_rejection() {
    let local_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let unknown_key = SigningKey::generate(&mut rand::rngs::OsRng);

    let protocol = GossipProtocol::new(
        NodeId("local".to_string()),
        "127.0.0.1:7946".parse().unwrap(),
        local_key,
    );

    // Note: We do NOT register unknown_key's public key

    let msg = GossipMessage::Ping {
        from: NodeId("unknown".to_string()),
        incarnation: 1,
    };
    let signed = SignedGossipMessage::sign(msg, &unknown_key);

    let result = protocol.handle_signed_message(signed).await;
    assert!(
        matches!(result, Err(FederationError::UnknownSender)),
        "Unknown sender should be rejected"
    );
}

/// Test: Replay attack prevention - old messages are rejected
#[tokio::test]
async fn test_replay_attack_old_message() {
    let local_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let peer_key = SigningKey::generate(&mut rand::rngs::OsRng);

    let protocol = GossipProtocol::new(
        NodeId("local".to_string()),
        "127.0.0.1:7946".parse().unwrap(),
        local_key,
    );

    protocol
        .register_peer_pubkey(NodeId("peer".to_string()), peer_key.verifying_key().to_bytes())
        .await;

    // Create message with old timestamp (6 minutes ago, beyond 5-minute window)
    let old_timestamp = Utc::now() - ChronoDuration::seconds(360);
    let msg = GossipMessage::Ping {
        from: NodeId("peer".to_string()),
        incarnation: 1,
    };

    // Manually construct with old timestamp
    let msg_bytes = msg.to_bytes();
    let ts_bytes = old_timestamp.timestamp_millis().to_le_bytes();
    let payload = [msg_bytes, ts_bytes.to_vec()].concat();
    let signature = peer_key.sign(&payload);

    let old_signed = SignedGossipMessage {
        message: msg,
        signature: signature.to_bytes(),
        sender_pubkey: peer_key.verifying_key().to_bytes(),
        timestamp: old_timestamp,
    };

    let result = protocol.handle_signed_message(old_signed).await;
    assert!(
        matches!(result, Err(FederationError::MessageExpired)),
        "Old message should be rejected for replay protection"
    );
}

/// Test: Replay attack prevention - future messages are rejected
#[tokio::test]
async fn test_replay_attack_future_message() {
    let local_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let peer_key = SigningKey::generate(&mut rand::rngs::OsRng);

    let protocol = GossipProtocol::new(
        NodeId("local".to_string()),
        "127.0.0.1:7946".parse().unwrap(),
        local_key,
    );

    protocol
        .register_peer_pubkey(NodeId("peer".to_string()), peer_key.verifying_key().to_bytes())
        .await;

    // Create message from the future (10 minutes ahead)
    let future_timestamp = Utc::now() + ChronoDuration::seconds(600);
    let msg = GossipMessage::Ping {
        from: NodeId("peer".to_string()),
        incarnation: 1,
    };

    let msg_bytes = msg.to_bytes();
    let ts_bytes = future_timestamp.timestamp_millis().to_le_bytes();
    let payload = [msg_bytes, ts_bytes.to_vec()].concat();
    let signature = peer_key.sign(&payload);

    let future_signed = SignedGossipMessage {
        message: msg,
        signature: signature.to_bytes(),
        sender_pubkey: peer_key.verifying_key().to_bytes(),
        timestamp: future_timestamp,
    };

    let result = protocol.handle_signed_message(future_signed).await;
    assert!(
        matches!(result, Err(FederationError::MessageExpired)),
        "Future message should be rejected"
    );
}

/// Test: Node ID spoofing prevention
#[tokio::test]
async fn test_node_id_spoofing_prevention() {
    let local_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let attacker_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let victim_key = SigningKey::generate(&mut rand::rngs::OsRng);

    let protocol = GossipProtocol::new(
        NodeId("local".to_string()),
        "127.0.0.1:7946".parse().unwrap(),
        local_key,
    );

    // Register victim's public key
    protocol
        .register_peer_pubkey(NodeId("victim".to_string()), victim_key.verifying_key().to_bytes())
        .await;

    // Attacker creates message claiming to be victim
    let spoofed_msg = GossipMessage::Ping {
        from: NodeId("victim".to_string()), // Claims to be victim
        incarnation: 999,
    };

    // But signs with attacker's key
    let signed = SignedGossipMessage::sign(spoofed_msg, &attacker_key);

    // Should be rejected - pubkey doesn't match registered victim key
    let result = protocol.handle_signed_message(signed).await;
    assert!(
        matches!(result, Err(FederationError::UnknownSender)),
        "Spoofed message should be rejected"
    );
}

// ============================================================================
// MESSAGE SERIALIZATION SECURITY
// ============================================================================

/// Test: Truncated messages are rejected
#[test]
fn test_truncated_message_rejection() {
    let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let msg = GossipMessage::Ping {
        from: NodeId("test".to_string()),
        incarnation: 1,
    };
    let signed = SignedGossipMessage::sign(msg, &signing_key);
    let full_bytes = signed.to_bytes();

    // Test various truncation points
    let truncation_points = [0, 1, 64, 65, 96, 97, 104, 105];

    for &len in &truncation_points {
        if len < full_bytes.len() {
            let truncated = &full_bytes[..len];
            let result = SignedGossipMessage::from_bytes(truncated);
            assert!(
                result.is_err(),
                "Truncated message at {} bytes should fail parsing",
                len
            );
        }
    }
}

/// Test: Invalid version byte is rejected
#[test]
fn test_invalid_version_rejection() {
    let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let msg = GossipMessage::Ping {
        from: NodeId("test".to_string()),
        incarnation: 1,
    };
    let signed = SignedGossipMessage::sign(msg, &signing_key);
    let mut bytes = signed.to_bytes();

    // Change version byte (first byte)
    bytes[0] = 2; // Unsupported version

    let result = SignedGossipMessage::from_bytes(&bytes);
    assert!(
        matches!(result, Err(FederationError::UnsupportedVersion(2))),
        "Invalid version should be rejected"
    );
}

/// Test: Message round-trip serialization
#[test]
fn test_message_round_trip() {
    let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);

    let messages = vec![
        GossipMessage::Ping {
            from: NodeId("node_1".to_string()),
            incarnation: 42,
        },
        GossipMessage::Ack {
            from: NodeId("node_2".to_string()),
            incarnation: 123,
        },
        GossipMessage::Leave {
            node_id: NodeId("leaving_node".to_string()),
        },
    ];

    for msg in messages {
        let signed = SignedGossipMessage::sign(msg.clone(), &signing_key);
        let bytes = signed.to_bytes();
        let parsed = SignedGossipMessage::from_bytes(&bytes).unwrap();

        // Verify signature still valid after round-trip
        assert!(parsed.verify().is_ok());

        // Verify content preserved
        match (&msg, &parsed.message) {
            (
                GossipMessage::Ping {
                    from: f1,
                    incarnation: i1,
                },
                GossipMessage::Ping {
                    from: f2,
                    incarnation: i2,
                },
            ) => {
                assert_eq!(f1, f2);
                assert_eq!(i1, i2);
            }
            (
                GossipMessage::Ack {
                    from: f1,
                    incarnation: i1,
                },
                GossipMessage::Ack {
                    from: f2,
                    incarnation: i2,
                },
            ) => {
                assert_eq!(f1, f2);
                assert_eq!(i1, i2);
            }
            (GossipMessage::Leave { node_id: n1 }, GossipMessage::Leave { node_id: n2 }) => {
                assert_eq!(n1, n2);
            }
            _ => panic!("Message type mismatch"),
        }
    }
}

// ============================================================================
// CONSENSUS SECURITY
// ============================================================================

/// Test: Votes require valid signatures
#[test]
fn test_consensus_vote_signature_verification() {
    let identity = NodeIdentity::generate();
    let mut engine = ConsensusEngine::new(identity);
    engine.set_node_count(3);

    let proposal = engine
        .propose(serde_json::json!({"pattern": "test"}), 0.96)
        .unwrap();

    // Create legitimate vote
    let vote = engine.vote(&proposal.id, true, 0.96).unwrap();

    // Verify the signature is valid
    assert!(vote.verify().is_ok());
}

/// Test: Forged votes are rejected
#[test]
fn test_consensus_forged_vote_rejection() {
    let identity = NodeIdentity::generate();
    let identity_node_id = identity.node_id().clone(); // Save before move
    let forger = NodeIdentity::generate();

    let mut engine = ConsensusEngine::new(identity);
    engine.set_node_count(3);

    let proposal = engine
        .propose(serde_json::json!({"pattern": "test"}), 0.96)
        .unwrap();

    // Forger creates vote claiming to be identity
    let vote = Vote {
        proposal_id: proposal.id.clone(),
        voter_id: identity_node_id.clone(), // Claims to be original
        approve: true,
        ihsan_score: 0.96,
        timestamp: Utc::now(),
    };

    // But signs with forger's key
    let forged = SignedVote::new(vote, forger.signing_key());

    // Receive should reject - pubkey mismatch
    let result = engine.receive_vote(forged);
    assert!(
        matches!(result, Err(ConsensusError::PubkeyMismatch { .. })),
        "Forged vote should be rejected"
    );
}

/// Test: Votes from unknown voters are rejected
#[test]
fn test_consensus_unknown_voter_rejection() {
    let identity = NodeIdentity::generate();
    let unknown = NodeIdentity::generate();

    let mut engine = ConsensusEngine::new(identity);
    engine.set_node_count(3);

    let proposal = engine
        .propose(serde_json::json!({"pattern": "test"}), 0.96)
        .unwrap();

    // Unknown voter tries to vote
    let vote = Vote {
        proposal_id: proposal.id.clone(),
        voter_id: unknown.node_id().clone(),
        approve: true,
        ihsan_score: 0.96,
        timestamp: Utc::now(),
    };
    let signed = SignedVote::new(vote, unknown.signing_key());

    let result = engine.receive_vote(signed);
    assert!(
        matches!(result, Err(ConsensusError::UnknownVoter(_))),
        "Unknown voter should be rejected"
    );
}

/// Test: Low Ihsan votes are rejected
#[test]
fn test_consensus_low_ihsan_rejection() {
    let identity = NodeIdentity::generate();
    let voter = NodeIdentity::generate();

    let mut engine = ConsensusEngine::new(identity);
    engine.set_node_count(3);
    engine.register_peer(voter.node_id().clone(), voter.public_key_bytes());

    let proposal = engine
        .propose(serde_json::json!({"pattern": "test"}), 0.96)
        .unwrap();

    // Vote with low Ihsan score
    let vote = Vote {
        proposal_id: proposal.id.clone(),
        voter_id: voter.node_id().clone(),
        approve: true,
        ihsan_score: 0.80, // Below 0.95 threshold
        timestamp: Utc::now(),
    };
    let signed = SignedVote::new(vote, voter.signing_key());

    let result = engine.receive_vote(signed);
    assert!(
        matches!(result, Err(ConsensusError::IhsanThreshold(_))),
        "Low Ihsan vote should be rejected"
    );
}

/// Test: Proposals below Ihsan threshold are rejected
#[test]
fn test_consensus_low_ihsan_proposal_rejection() {
    let identity = NodeIdentity::generate();
    let mut engine = ConsensusEngine::new(identity);

    let result = engine.propose(serde_json::json!({"pattern": "test"}), 0.90);

    assert!(
        matches!(result, Err(ConsensusError::IhsanThreshold(_))),
        "Low Ihsan proposal should be rejected"
    );
}

/// Test: Quorum calculation for BFT
#[test]
fn test_consensus_bft_quorum() {
    // Generate all voters first
    let mut voters: Vec<NodeIdentity> = (0..5).map(|_| NodeIdentity::generate()).collect();

    // Use the first voter as the engine identity
    let identity = voters.remove(0);
    let _identity_node_id = identity.node_id().clone();
    let _identity_pubkey = identity.public_key_bytes();
    let identity_signing_key_bytes = identity.secret_bytes();

    let mut engine = ConsensusEngine::new(identity);

    // For n=5 nodes, f=1 Byzantine nodes, quorum = 2*1 + 1 = 3... but formula is (2n/3)+1
    // n=5: quorum = (2*5/3)+1 = 3+1 = 4
    engine.set_node_count(5);

    let proposal = engine
        .propose(serde_json::json!({"pattern": "test"}), 0.96)
        .unwrap();

    // Register remaining peers
    for voter in &voters {
        engine.register_peer(voter.node_id().clone(), voter.public_key_bytes());
    }

    // Restore the first identity for voting
    let first_voter = NodeIdentity::from_secret_bytes(&identity_signing_key_bytes);
    let mut all_voters = vec![first_voter];
    all_voters.extend(voters);

    // Add votes one by one until quorum
    for (i, voter) in all_voters.iter().take(4).enumerate() {
        let vote = Vote {
            proposal_id: proposal.id.clone(),
            voter_id: voter.node_id().clone(),
            approve: true,
            ihsan_score: 0.96,
            timestamp: Utc::now(),
        };
        let signed = SignedVote::new(vote, voter.signing_key());

        let result = engine.receive_vote(signed).unwrap();

        if i < 3 {
            // Not enough votes yet (need 4 for quorum with 5 nodes)
            assert!(!result, "Should not have consensus with {} votes", i + 1);
        } else {
            // Quorum reached at 4 votes
            assert!(result, "Should have consensus with {} votes", i + 1);
        }
    }
}

// ============================================================================
// FUZZ-LIKE EDGE CASES
// ============================================================================

/// Test: Random byte sequences don't panic
#[test]
fn test_random_bytes_no_panic() {
    // Various malformed inputs
    let test_cases: Vec<Vec<u8>> = vec![
        vec![],                            // Empty
        vec![0],                           // Single byte
        vec![1],                           // Version only
        vec![1; 64],                       // Partial signature
        vec![1; 97],                       // Partial pubkey
        vec![1; 105],                      // Partial timestamp
        vec![255; 200],                    // All 0xFF
        vec![0; 200],                      // All 0x00
        (0..=255).collect(),               // All byte values
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9],   // Short sequence
    ];

    for bytes in test_cases {
        // Should not panic, just return Err
        let _ = SignedGossipMessage::from_bytes(&bytes);
    }
}

/// Test: Very long node IDs are handled
#[test]
fn test_long_node_id_handling() {
    let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);

    // Create message with very long node_id
    let long_id = "x".repeat(10_000);
    let msg = GossipMessage::Ping {
        from: NodeId(long_id),
        incarnation: 1,
    };

    // Should handle without panic
    let signed = SignedGossipMessage::sign(msg, &signing_key);
    let bytes = signed.to_bytes();
    let parsed = SignedGossipMessage::from_bytes(&bytes).unwrap();

    assert!(parsed.verify().is_ok());
}

/// Test: Special characters in node IDs
#[test]
fn test_special_chars_node_id() {
    let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);

    let special_ids = vec![
        "node\0null",
        "node\ttab",
        "node\nnewline",
        "node\"quote\"",
        "node'apostrophe'",
        "node\\backslash",
        "node<script>alert('xss')</script>",
        "node\u{202E}rtl", // RTL override
    ];

    for id in special_ids {
        let msg = GossipMessage::Ping {
            from: NodeId(id.to_string()),
            incarnation: 1,
        };
        let signed = SignedGossipMessage::sign(msg, &signing_key);

        // Should handle round-trip
        let bytes = signed.to_bytes();
        let result = SignedGossipMessage::from_bytes(&bytes);
        assert!(result.is_ok(), "Failed to handle node_id: {:?}", id);
    }
}

// ============================================================================
// TIMING TESTS
// ============================================================================

/// Test: Message within acceptable time window
#[tokio::test]
async fn test_message_within_time_window() {
    let local_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let peer_key = SigningKey::generate(&mut rand::rngs::OsRng);

    let protocol = GossipProtocol::new(
        NodeId("local".to_string()),
        "127.0.0.1:7946".parse().unwrap(),
        local_key,
    );

    protocol
        .register_peer_pubkey(NodeId("peer".to_string()), peer_key.verifying_key().to_bytes())
        .await;

    // Message from 2 minutes ago (within 5-minute window)
    let recent_timestamp = Utc::now() - ChronoDuration::seconds(120);
    let msg = GossipMessage::Ping {
        from: NodeId("peer".to_string()),
        incarnation: 1,
    };

    let msg_bytes = msg.to_bytes();
    let ts_bytes = recent_timestamp.timestamp_millis().to_le_bytes();
    let payload = [msg_bytes, ts_bytes.to_vec()].concat();
    let signature = peer_key.sign(&payload);

    let recent_signed = SignedGossipMessage {
        message: msg,
        signature: signature.to_bytes(),
        sender_pubkey: peer_key.verifying_key().to_bytes(),
        timestamp: recent_timestamp,
    };

    let result = protocol.handle_signed_message(recent_signed).await;
    assert!(result.is_ok(), "Recent message should be accepted");
}
