// src/sovereignty/interop.rs - Interoperability Sovereignty (Pillar 6: Exit + Federation)
//
// Principle: You can fork, migrate, and interconnect without vendor lock.
// Standard protocols at boundaries (A2A/MCP-style) with your own policy gate.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Protocol types for interoperability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteropProtocol {
    /// Pattern Federation Protocol (BIZRA-native)
    Pfp,
    /// Model Context Protocol (Anthropic)
    Mcp,
    /// Agent-to-Agent (Google)
    A2a,
    /// JSON-LD for data export
    JsonLd,
    /// Custom/proprietary (discouraged)
    Custom,
}

impl InteropProtocol {
    /// Is this a standard protocol?
    pub fn is_standard(&self) -> bool {
        !matches!(self, Self::Custom)
    }

    /// Get protocol specification URL
    pub fn spec_url(&self) -> Option<&'static str> {
        match self {
            Self::Pfp => Some("https://bizra.info/protocols/pfp/v1"),
            Self::Mcp => Some("https://modelcontextprotocol.io/specification"),
            Self::A2a => Some("https://google.github.io/a2a"),
            Self::JsonLd => Some("https://json-ld.org/spec/latest/json-ld/"),
            Self::Custom => None,
        }
    }
}

/// Boundary type (where interop happens)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoundaryType {
    /// Incoming from external system
    Inbound,
    /// Outgoing to external system
    Outbound,
    /// Federation peer-to-peer
    Federation,
    /// Data export
    Export,
    /// Data import
    Import,
}

/// Boundary policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryPolicy {
    /// Boundary type
    pub boundary_type: BoundaryType,
    /// Allowed protocols
    pub allowed_protocols: Vec<InteropProtocol>,
    /// Require authentication
    pub require_auth: bool,
    /// Require encryption (TLS)
    pub require_encryption: bool,
    /// Policy gate (PCI) enabled
    pub policy_gate_enabled: bool,
    /// Rate limit (requests per minute)
    pub rate_limit: Option<u32>,
    /// Audit logging
    pub audit_enabled: bool,
}

impl BoundaryPolicy {
    /// Create strict inbound policy
    pub fn strict_inbound() -> Self {
        Self {
            boundary_type: BoundaryType::Inbound,
            allowed_protocols: vec![InteropProtocol::Pfp, InteropProtocol::Mcp],
            require_auth: true,
            require_encryption: true,
            policy_gate_enabled: true,
            rate_limit: Some(100),
            audit_enabled: true,
        }
    }

    /// Create federation policy
    pub fn federation() -> Self {
        Self {
            boundary_type: BoundaryType::Federation,
            allowed_protocols: vec![InteropProtocol::Pfp],
            require_auth: true,
            require_encryption: true,
            policy_gate_enabled: true,
            rate_limit: Some(1000),
            audit_enabled: true,
        }
    }

    /// Create export policy
    pub fn export() -> Self {
        Self {
            boundary_type: BoundaryType::Export,
            allowed_protocols: vec![InteropProtocol::JsonLd],
            require_auth: true,
            require_encryption: true,
            policy_gate_enabled: true,
            rate_limit: Some(10),
            audit_enabled: true,
        }
    }

    /// Check if protocol is allowed
    pub fn is_protocol_allowed(&self, protocol: InteropProtocol) -> bool {
        self.allowed_protocols.contains(&protocol)
    }
}

/// Node migration package (for portability)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPackage {
    /// Package format version
    pub format_version: String,
    /// Source node ID
    pub source_node: String,
    /// Generated timestamp
    pub generated_at: chrono::DateTime<chrono::Utc>,
    /// Identity (encrypted)
    pub identity: MigrationIdentity,
    /// Configuration
    pub config: MigrationConfig,
    /// Patterns
    pub patterns: Vec<MigrationPattern>,
    /// Evidence (summary)
    pub evidence_summary: MigrationEvidence,
    /// Package signature
    pub signature: String,
}

/// Identity for migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationIdentity {
    /// Node public key
    pub public_key: String,
    /// Encrypted private key (passphrase-protected)
    pub encrypted_private_key: String,
    /// Agent public keys
    pub agent_keys: HashMap<String, String>,
    /// Federation memberships
    pub federation_memberships: Vec<String>,
}

/// Config for migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationConfig {
    /// Constitution version
    pub constitution_version: String,
    /// Model family version
    pub model_family_version: String,
    /// Custom settings
    pub settings: HashMap<String, String>,
}

/// Pattern for migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPattern {
    /// Pattern ID
    pub id: String,
    /// Pattern content hash
    pub content_hash: String,
    /// Origin node
    pub origin: String,
    /// Adoption timestamp
    pub adopted_at: chrono::DateTime<chrono::Utc>,
}

/// Evidence summary for migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationEvidence {
    /// Total receipts
    pub total_receipts: u64,
    /// Date range
    pub date_range: (String, String),
    /// Evidence archive hash (for verification)
    pub archive_hash: Option<String>,
}

/// Interoperability manager
pub struct InteropManager {
    /// Boundary policies
    policies: HashMap<BoundaryType, BoundaryPolicy>,
    /// Active federation peers
    federation_peers: Vec<String>,
    /// Pending migrations
    pending_migrations: Vec<MigrationPackage>,
}

impl InteropManager {
    /// Create with default policies
    pub fn new() -> Self {
        let mut policies = HashMap::new();
        policies.insert(BoundaryType::Inbound, BoundaryPolicy::strict_inbound());
        policies.insert(BoundaryType::Federation, BoundaryPolicy::federation());
        policies.insert(BoundaryType::Export, BoundaryPolicy::export());

        Self {
            policies,
            federation_peers: Vec::new(),
            pending_migrations: Vec::new(),
        }
    }

    /// Get policy for boundary
    pub fn policy(&self, boundary: BoundaryType) -> Option<&BoundaryPolicy> {
        self.policies.get(&boundary)
    }

    /// Check if request allowed at boundary
    pub fn is_allowed(
        &self,
        boundary: BoundaryType,
        protocol: InteropProtocol,
        is_authenticated: bool,
        is_encrypted: bool,
    ) -> bool {
        let Some(policy) = self.policies.get(&boundary) else {
            return false;
        };

        if !policy.is_protocol_allowed(protocol) {
            return false;
        }

        if policy.require_auth && !is_authenticated {
            return false;
        }

        if policy.require_encryption && !is_encrypted {
            return false;
        }

        true
    }

    /// Add federation peer
    pub fn add_peer(&mut self, peer_id: String) {
        if !self.federation_peers.contains(&peer_id) {
            self.federation_peers.push(peer_id);
        }
    }

    /// Get federation peers
    pub fn peers(&self) -> &[String] {
        &self.federation_peers
    }

    /// Check if protocol is standard (no vendor lock)
    pub fn is_vendor_neutral(&self, protocol: InteropProtocol) -> bool {
        protocol.is_standard()
    }
}

impl Default for InteropManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Export capabilities
pub trait Exportable {
    /// Export as JSON-LD
    fn to_json_ld(&self) -> serde_json::Value;

    /// Export format identifier
    fn export_format(&self) -> &'static str {
        "application/ld+json"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_is_standard() {
        assert!(InteropProtocol::Pfp.is_standard());
        assert!(InteropProtocol::Mcp.is_standard());
        assert!(InteropProtocol::A2a.is_standard());
        assert!(!InteropProtocol::Custom.is_standard());
    }

    #[test]
    fn test_strict_inbound_policy() {
        let policy = BoundaryPolicy::strict_inbound();

        assert!(policy.require_auth);
        assert!(policy.require_encryption);
        assert!(policy.policy_gate_enabled);
        assert!(policy.is_protocol_allowed(InteropProtocol::Pfp));
        assert!(!policy.is_protocol_allowed(InteropProtocol::Custom));
    }

    #[test]
    fn test_interop_manager() {
        let manager = InteropManager::new();

        // Authenticated, encrypted PFP should be allowed
        assert!(manager.is_allowed(BoundaryType::Inbound, InteropProtocol::Pfp, true, true,));

        // Unauthenticated should be denied
        assert!(!manager.is_allowed(BoundaryType::Inbound, InteropProtocol::Pfp, false, true,));

        // Custom protocol should be denied
        assert!(!manager.is_allowed(BoundaryType::Inbound, InteropProtocol::Custom, true, true,));
    }

    #[test]
    fn test_vendor_neutral() {
        let manager = InteropManager::new();

        assert!(manager.is_vendor_neutral(InteropProtocol::Pfp));
        assert!(manager.is_vendor_neutral(InteropProtocol::Mcp));
        assert!(!manager.is_vendor_neutral(InteropProtocol::Custom));
    }
}
