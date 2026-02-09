//! Comprehensive tests for gossip protocol — SWIM, signed messages, wire format
//!
//! Phase 13: Test Sprint — Closing coverage gap on federation gossip layer

use bizra_core::NodeId;
use bizra_federation::gossip::*;
use ed25519_dalek::SigningKey;
use std::net::SocketAddr;

// ---------------------------------------------------------------------------
// Member
// ---------------------------------------------------------------------------

#[test]
fn member_new_defaults_to_alive() {
    let m = Member::new(NodeId("node_abc123".into()), "127.0.0.1:7946".parse().unwrap());
    assert_eq!(m.state, NodeState::Alive);
    assert_eq!(m.incarnation, 0);
    assert!(m.is_alive());
}

#[test]
fn member_is_alive_false_for_dead() {
    let mut m = Member::new(NodeId("node_dead".into()), "127.0.0.1:7946".parse().unwrap());
    m.state = NodeState::Dead;
    assert!(!m.is_alive());
}

#[test]
fn member_is_alive_false_for_suspect() {
    let mut m = Member::new(NodeId("node_sus".into()), "127.0.0.1:7946".parse().unwrap());
    m.state = NodeState::Suspect;
    assert!(!m.is_alive());
}

#[test]
fn member_is_alive_false_for_left() {
    let mut m = Member::new(NodeId("node_left".into()), "127.0.0.1:7946".parse().unwrap());
    m.state = NodeState::Left;
    assert!(!m.is_alive());
}

// ---------------------------------------------------------------------------
// GossipMessage — serialization round-trip
// ---------------------------------------------------------------------------

#[test]
fn gossip_message_ping_roundtrip() {
    let msg = GossipMessage::Ping {
        from: NodeId("node_1".into()),
        incarnation: 42,
    };
    let bytes = msg.to_bytes();
    let decoded = GossipMessage::from_bytes(&bytes).expect("decode");
    match decoded {
        GossipMessage::Ping { from, incarnation } => {
            assert_eq!(from.0, "node_1");
            assert_eq!(incarnation, 42);
        }
        _ => panic!("expected Ping"),
    }
}

#[test]
fn gossip_message_ack_roundtrip() {
    let msg = GossipMessage::Ack {
        from: NodeId("ack_node".into()),
        incarnation: 7,
    };
    let bytes = msg.to_bytes();
    let decoded = GossipMessage::from_bytes(&bytes).unwrap();
    match decoded {
        GossipMessage::Ack { from, incarnation } => {
            assert_eq!(from.0, "ack_node");
            assert_eq!(incarnation, 7);
        }
        _ => panic!("expected Ack"),
    }
}

#[test]
fn gossip_message_join_roundtrip() {
    let member = Member::new(NodeId("joiner".into()), "10.0.0.1:7946".parse().unwrap());
    let msg = GossipMessage::Join {
        member: member.clone(),
    };
    let bytes = msg.to_bytes();
    let decoded = GossipMessage::from_bytes(&bytes).unwrap();
    match decoded {
        GossipMessage::Join { member: m } => {
            assert_eq!(m.node_id.0, "joiner");
        }
        _ => panic!("expected Join"),
    }
}

#[test]
fn gossip_message_leave_roundtrip() {
    let msg = GossipMessage::Leave {
        node_id: NodeId("leaver".into()),
    };
    let bytes = msg.to_bytes();
    let decoded = GossipMessage::from_bytes(&bytes).unwrap();
    match decoded {
        GossipMessage::Leave { node_id } => assert_eq!(node_id.0, "leaver"),
        _ => panic!("expected Leave"),
    }
}

#[test]
fn gossip_message_update_roundtrip() {
    let mut member = Member::new(NodeId("updater".into()), "10.0.0.2:7946".parse().unwrap());
    member.incarnation = 5;
    let msg = GossipMessage::Update { member };
    let bytes = msg.to_bytes();
    let decoded = GossipMessage::from_bytes(&bytes).unwrap();
    match decoded {
        GossipMessage::Update { member: m } => {
            assert_eq!(m.node_id.0, "updater");
            assert_eq!(m.incarnation, 5);
        }
        _ => panic!("expected Update"),
    }
}

#[test]
fn gossip_message_from_bytes_rejects_bad_prefix() {
    let bad = b"wrong-prefix:{}";
    assert!(GossipMessage::from_bytes(bad).is_err());
}

#[test]
fn gossip_message_from_bytes_rejects_empty() {
    assert!(GossipMessage::from_bytes(&[]).is_err());
}

#[test]
fn gossip_message_sender_id_variants() {
    let ping = GossipMessage::Ping {
        from: NodeId("p".into()),
        incarnation: 0,
    };
    assert_eq!(ping.sender_id().0, "p");

    let ack = GossipMessage::Ack {
        from: NodeId("a".into()),
        incarnation: 0,
    };
    assert_eq!(ack.sender_id().0, "a");

    let join = GossipMessage::Join {
        member: Member::new(NodeId("j".into()), "127.0.0.1:1".parse().unwrap()),
    };
    assert_eq!(join.sender_id().0, "j");

    let leave = GossipMessage::Leave {
        node_id: NodeId("l".into()),
    };
    assert_eq!(leave.sender_id().0, "l");

    let update = GossipMessage::Update {
        member: Member::new(NodeId("u".into()), "127.0.0.1:2".parse().unwrap()),
    };
    assert_eq!(update.sender_id().0, "u");
}

// ---------------------------------------------------------------------------
// SignedGossipMessage — crypto
// ---------------------------------------------------------------------------

fn test_signing_key() -> SigningKey {
    SigningKey::generate(&mut rand::rngs::OsRng)
}

#[test]
fn signed_message_sign_and_verify() {
    let key = test_signing_key();
    let msg = GossipMessage::Ping {
        from: NodeId("signed_node".into()),
        incarnation: 1,
    };
    let signed = SignedGossipMessage::sign(msg, &key);
    assert!(signed.verify().is_ok());
}

#[test]
fn signed_message_rejects_tampered_signature() {
    let key = test_signing_key();
    let msg = GossipMessage::Ping {
        from: NodeId("tampered".into()),
        incarnation: 1,
    };
    let mut signed = SignedGossipMessage::sign(msg, &key);
    signed.signature[0] ^= 0xFF; // flip bits
    assert!(signed.verify().is_err());
}

#[test]
fn signed_message_rejects_wrong_pubkey() {
    let key = test_signing_key();
    let other_key = test_signing_key();
    let msg = GossipMessage::Ping {
        from: NodeId("wrong_key".into()),
        incarnation: 1,
    };
    let mut signed = SignedGossipMessage::sign(msg, &key);
    signed.sender_pubkey = other_key.verifying_key().to_bytes();
    assert!(signed.verify().is_err());
}

#[test]
fn signed_message_wire_format_roundtrip() {
    let key = test_signing_key();
    let msg = GossipMessage::Ack {
        from: NodeId("wire_test".into()),
        incarnation: 99,
    };
    let signed = SignedGossipMessage::sign(msg, &key);
    let wire = signed.to_bytes();
    let decoded = SignedGossipMessage::from_bytes(&wire).expect("decode wire");
    assert!(decoded.verify().is_ok());
    match &decoded.message {
        GossipMessage::Ack { from, incarnation } => {
            assert_eq!(from.0, "wire_test");
            assert_eq!(*incarnation, 99);
        }
        _ => panic!("expected Ack"),
    }
}

#[test]
fn signed_message_from_bytes_rejects_too_short() {
    assert!(SignedGossipMessage::from_bytes(&[0u8; 10]).is_err());
}

#[test]
fn signed_message_from_bytes_rejects_wrong_version() {
    let mut data = vec![0u8; 1 + 64 + 32 + 8 + 32]; // minimum size
    data[0] = 99; // bad version
    assert!(SignedGossipMessage::from_bytes(&data).is_err());
}

#[test]
fn signed_message_pubkey_hex_format() {
    let key = test_signing_key();
    let msg = GossipMessage::Ping {
        from: NodeId("hex_test".into()),
        incarnation: 0,
    };
    let signed = SignedGossipMessage::sign(msg, &key);
    let hex = signed.pubkey_hex();
    assert_eq!(hex.len(), 64); // 32 bytes → 64 hex chars
}

// ---------------------------------------------------------------------------
// GossipProtocol — state management
// ---------------------------------------------------------------------------

#[tokio::test]
async fn protocol_new_starts_with_self() {
    let id = NodeId("self_node".into());
    let addr: SocketAddr = "127.0.0.1:7946".parse().unwrap();
    let proto = GossipProtocol::new_with_generated_key(id.clone(), addr);
    assert_eq!(proto.member_count().await, 1);
    let alive = proto.alive_members().await;
    assert_eq!(alive.len(), 1);
    assert_eq!(alive[0].node_id.0, "self_node");
}

#[tokio::test]
async fn protocol_add_seed_increases_count() {
    let id = NodeId("host".into());
    let addr: SocketAddr = "127.0.0.1:7946".parse().unwrap();
    let proto = GossipProtocol::new_with_generated_key(id, addr);
    proto
        .add_seed(NodeId("seed1".into()), "10.0.0.1:7946".parse().unwrap())
        .await;
    assert_eq!(proto.member_count().await, 2);
}

#[tokio::test]
async fn protocol_sign_message_verifiable() {
    let id = NodeId("signer".into());
    let addr: SocketAddr = "127.0.0.1:7946".parse().unwrap();
    let proto = GossipProtocol::new_with_generated_key(id, addr);
    let msg = GossipMessage::Ping {
        from: NodeId("signer".into()),
        incarnation: 0,
    };
    let signed = proto.sign_message(msg);
    assert!(signed.verify().is_ok());
}

#[tokio::test]
async fn protocol_public_key_matches_signed_messages() {
    let id = NodeId("pk_test".into());
    let addr: SocketAddr = "127.0.0.1:7946".parse().unwrap();
    let proto = GossipProtocol::new_with_generated_key(id, addr);
    let pk = proto.public_key();
    let msg = GossipMessage::Ping {
        from: NodeId("pk_test".into()),
        incarnation: 0,
    };
    let signed = proto.sign_message(msg);
    assert_eq!(signed.sender_pubkey, pk);
}

#[tokio::test]
async fn protocol_handle_signed_ping_returns_ack() {
    let id_a = NodeId("node_a".into());
    let id_b = NodeId("node_b".into());
    let addr_a: SocketAddr = "127.0.0.1:7946".parse().unwrap();
    let addr_b: SocketAddr = "127.0.0.2:7946".parse().unwrap();

    let proto_a = GossipProtocol::new_with_generated_key(id_a.clone(), addr_a);
    let proto_b = GossipProtocol::new_with_generated_key(id_b.clone(), addr_b);

    // Register b's key in a
    proto_a
        .register_peer_pubkey(id_b.clone(), proto_b.public_key())
        .await;
    // Add b as member so incarnation update works
    proto_a.add_seed(id_b.clone(), addr_b).await;

    let ping = GossipMessage::Ping {
        from: id_b.clone(),
        incarnation: 1,
    };
    let signed_ping = proto_b.sign_message(ping);

    let response = proto_a.handle_signed_message(signed_ping).await;
    assert!(response.is_ok());
    let resp = response.unwrap();
    assert!(resp.is_some()); // Should return signed Ack
}

#[tokio::test]
async fn protocol_rejects_unknown_sender() {
    let id = NodeId("host".into());
    let addr: SocketAddr = "127.0.0.1:7946".parse().unwrap();
    let proto = GossipProtocol::new_with_generated_key(id, addr);

    // Create message from unknown node (not registered)
    let unknown_key = SigningKey::generate(&mut rand::rngs::OsRng);
    let msg = GossipMessage::Ping {
        from: NodeId("unknown".into()),
        incarnation: 0,
    };
    let signed = SignedGossipMessage::sign(msg, &unknown_key);

    let result = proto.handle_signed_message(signed).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn protocol_create_leave_message() {
    let id = NodeId("leaver".into());
    let addr: SocketAddr = "127.0.0.1:7946".parse().unwrap();
    let proto = GossipProtocol::new_with_generated_key(id, addr);
    let leave = proto.create_leave_message();
    match leave {
        GossipMessage::Leave { node_id } => assert_eq!(node_id.0, "leaver"),
        _ => panic!("expected Leave"),
    }
}

// ---------------------------------------------------------------------------
// FederationError display
// ---------------------------------------------------------------------------

#[test]
fn federation_error_display_messages() {
    let e1 = FederationError::InvalidPublicKey;
    assert!(e1.to_string().contains("public key"));

    let e2 = FederationError::InvalidSignature;
    assert!(e2.to_string().contains("signature"));

    let e3 = FederationError::InvalidMessageFormat;
    assert!(e3.to_string().contains("message format"));

    let e4 = FederationError::UnsupportedVersion(99);
    assert!(e4.to_string().contains("99"));

    let e5 = FederationError::InvalidTimestamp;
    assert!(e5.to_string().contains("timestamp"));

    let e6 = FederationError::MessageExpired;
    assert!(e6.to_string().contains("old"));

    let e7 = FederationError::UnknownSender;
    assert!(e7.to_string().contains("sender"));
}
