//! Comprehensive tests for FederationNode — lifecycle, gossip, consensus integration
//!
//! Phase 13: Test Sprint

use bizra_core::{Constitution, NodeId, NodeIdentity};
use bizra_federation::node::*;
use bizra_federation::{BootstrapConfig, PeerInfo};

fn make_node(name: &str) -> FederationNode {
    let identity = NodeIdentity::generate();
    let config = NodeConfig {
        name: name.into(),
        gossip_addr: "127.0.0.1:0".parse().unwrap(), // OS-assigned port
        seeds: Vec::new(),
        data_dir: "./test_data".into(),
    };
    FederationNode::new(config, identity, Constitution::default())
}

// ---------------------------------------------------------------------------
// NodeConfig
// ---------------------------------------------------------------------------

#[test]
fn node_config_default_values() {
    let config = NodeConfig::default();
    assert_eq!(config.name, "bizra-node");
    assert_eq!(
        config.gossip_addr.port(),
        bizra_federation::DEFAULT_GOSSIP_PORT
    );
    assert!(config.seeds.is_empty());
    assert_eq!(config.data_dir, "./data");
}

// ---------------------------------------------------------------------------
// FederationNode lifecycle
// ---------------------------------------------------------------------------

#[tokio::test]
async fn node_new_not_running() {
    let node = make_node("test-node");
    assert!(!node.is_running().await);
}

#[tokio::test]
async fn node_name_matches_config() {
    let node = make_node("my-node");
    assert_eq!(node.name(), "my-node");
}

#[tokio::test]
async fn node_id_is_deterministic_from_identity() {
    let identity = NodeIdentity::generate();
    let expected_id = identity.node_id().clone();
    let config = NodeConfig {
        name: "det-node".into(),
        gossip_addr: "127.0.0.1:0".parse().unwrap(),
        seeds: Vec::new(),
        data_dir: "./test".into(),
    };
    let node = FederationNode::new(config, identity, Constitution::default());
    assert_eq!(node.node_id(), &expected_id);
}

#[tokio::test]
async fn node_start_and_stop() {
    let node = make_node("lifecycle-node");
    assert!(!node.is_running().await);

    node.start().await.expect("start");
    assert!(node.is_running().await);

    node.stop().await.expect("stop");
    assert!(!node.is_running().await);
}

#[tokio::test]
async fn node_double_start_errors() {
    let node = make_node("double-start");
    node.start().await.unwrap();
    let result = node.start().await;
    assert!(result.is_err());
    match result.unwrap_err() {
        NodeError::AlreadyRunning => {}
        e => panic!("expected AlreadyRunning, got {:?}", e),
    }
    node.stop().await.unwrap(); // cleanup
}

#[tokio::test]
async fn node_double_stop_errors() {
    let node = make_node("double-stop");
    let result = node.stop().await;
    assert!(result.is_err());
    match result.unwrap_err() {
        NodeError::NotRunning => {}
        e => panic!("expected NotRunning, got {:?}", e),
    }
}

#[tokio::test]
async fn node_peer_count_starts_zero() {
    let node = make_node("peer-count");
    // peer_count = member_count - 1 (excludes self)
    assert_eq!(node.peer_count().await, 0);
}

#[tokio::test]
async fn node_propose_pattern_succeeds() {
    let node = make_node("proposer");
    let result = node
        .propose_pattern(serde_json::json!({"pattern": "test"}), 0.96)
        .await;
    assert!(result.is_ok());
    let pid = result.unwrap();
    assert!(pid.starts_with("prop_"));
}

#[tokio::test]
async fn node_propose_pattern_fails_below_ihsan() {
    let node = make_node("low-ihsan");
    let result = node
        .propose_pattern(serde_json::json!({"bad": true}), 0.3)
        .await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// NodeError display
// ---------------------------------------------------------------------------

#[test]
fn node_error_display() {
    assert!(NodeError::AlreadyRunning.to_string().contains("running"));
    assert!(NodeError::NotRunning.to_string().contains("running"));
}

// ---------------------------------------------------------------------------
// BootstrapConfig
// ---------------------------------------------------------------------------

#[test]
fn bootstrap_config_default() {
    let config = BootstrapConfig::default();
    assert!(config.seed_nodes.is_empty());
    assert_eq!(config.bind_addr, "0.0.0.0:7946");
    assert_eq!(config.discovery_timeout_secs, 30);
    assert_eq!(config.max_peers, 100);
    assert!(config.enable_mdns);
    assert!(!config.enable_dns_sd);
}

// ---------------------------------------------------------------------------
// PeerInfo
// ---------------------------------------------------------------------------

#[test]
fn peer_info_serialization_roundtrip() {
    let peer = PeerInfo {
        node_id: NodeId("peer_1".into()),
        address: "10.0.0.1:7946".parse().unwrap(),
        version: "0.1.0".into(),
        capabilities: vec!["inference".into(), "consensus".into()],
    };
    let json = serde_json::to_string(&peer).unwrap();
    let decoded: PeerInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.node_id.0, "peer_1");
    assert_eq!(decoded.capabilities.len(), 2);
}
