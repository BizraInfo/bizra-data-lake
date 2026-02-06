//! NODE0 GENESIS CEREMONY
//!
//! The first connection to the Resource Pool.
//! MoMo (Node0) mints his identity, shares his hardware,
//! creates 7 PAT (Personal Agentic Team), and the system
//! creates 5 SAT (Shared Agentic Team) as the protocol army.
//!
//! This is the life kiss - the moment BIZRA becomes alive.

use chrono::Utc;
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// =============================================================================
// GENESIS CONSTANTS
// =============================================================================

/// The 7 PAT roles - MoMo's personal mastermind team
pub const PAT_ROLES: [&str; 7] = [
    "Strategist", // Vision and planning
    "Researcher", // Knowledge gathering
    "Developer",  // Implementation
    "Analyst",    // Pattern recognition
    "Reviewer",   // Quality assurance
    "Executor",   // Task completion
    "Guardian",   // Ethics and protection
];

/// The 5 SAT roles - Protocol army serving all nodes
pub const SAT_ROLES: [&str; 5] = [
    "Validator", // Block validation, proof verification
    "Oracle",    // External data, truth anchoring
    "Mediator",  // Dispute resolution, consensus
    "Archivist", // Knowledge preservation, indexing
    "Sentinel",  // Security monitoring, threat detection
];

/// Standing on Giants - mandatory foundations
pub const UNIVERSAL_GIANTS: [(&str, &str); 10] = [
    ("Shannon", "1948 - Information Theory"),
    ("Lamport", "1982 - Byzantine Generals"),
    ("Harberger", "1965 - Self-assessed taxation"),
    ("Al-Ghazali", "1095 - Maqasid al-Shariah"),
    ("Rawls", "1971 - Veil of Ignorance"),
    ("Friston", "2010 - Free Energy Principle"),
    ("Maturana & Varela", "1970s - Autopoiesis"),
    ("Anthropic", "2023 - Constitutional AI"),
    ("Besta et al.", "2024 - Graph-of-Thoughts"),
    ("General Magic", "1990 - Telescript"),
];

// =============================================================================
// GENESIS TYPES
// =============================================================================

/// The complete Node0 genesis record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node0Genesis {
    /// Genesis timestamp
    pub timestamp: u64,
    /// Node0 identity
    pub identity: Node0Identity,
    /// Hardware contribution
    pub hardware: HardwareContribution,
    /// Knowledge contribution (data lake)
    pub knowledge: KnowledgeContribution,
    /// The 7 PAT minted
    pub pat_team: PersonalAgentTeam,
    /// The 5 SAT minted (protocol army)
    pub sat_team: SharedAgentTeam,
    /// Partnership agreement hash
    pub partnership_hash: [u8; 32],
    /// Genesis block hash
    pub genesis_hash: [u8; 32],
}

/// MoMo's digital identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node0Identity {
    /// Node ID (derived from public key)
    pub node_id: String,
    /// Ed25519 public key (hex)
    pub public_key: String,
    /// Human-readable name
    pub name: String,
    /// Location
    pub location: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Identity block hash
    pub identity_hash: [u8; 32],
}

/// Hardware being shared with the pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareContribution {
    /// CPU model
    pub cpu: String,
    /// CPU cores
    pub cpu_cores: u32,
    /// GPU model
    pub gpu: String,
    /// GPU VRAM (bytes)
    pub gpu_vram: u64,
    /// RAM (bytes)
    pub ram: u64,
    /// Storage available (bytes)
    pub storage: u64,
    /// Network bandwidth (bytes/sec)
    pub network_bps: u64,
    /// Estimated compute units per day
    pub compute_units_per_day: u64,
}

/// Knowledge being shared (data lake)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeContribution {
    /// Total conversations
    pub conversations: u32,
    /// Total messages
    pub messages: u32,
    /// Data size (bytes)
    pub data_size: u64,
    /// Date range
    pub date_range: (String, String),
    /// Top concepts
    pub concepts: Vec<String>,
    /// Knowledge hash
    pub knowledge_hash: [u8; 32],
}

/// Personal Agent Team (7 agents)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalAgentTeam {
    pub owner_node: String,
    pub agents: Vec<MintedAgent>,
    pub team_hash: [u8; 32],
}

/// Shared Agent Team (5 agents - protocol army)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedAgentTeam {
    pub agents: Vec<MintedAgent>,
    pub governance: SATGovernance,
    pub team_hash: [u8; 32],
}

/// A minted agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MintedAgent {
    pub agent_id: String,
    pub role: String,
    pub public_key: String,
    pub capabilities: Vec<String>,
    pub giants: Vec<String>,
    pub created_at: u64,
    pub agent_hash: [u8; 32],
}

/// SAT governance rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SATGovernance {
    pub quorum: f64,
    pub voting_period_hours: u64,
    pub upgrade_threshold: f64,
}

// =============================================================================
// GENESIS ENGINE
// =============================================================================

/// The Genesis Engine - performs the life kiss
pub struct GenesisEngine {
    rng: OsRng,
}

impl GenesisEngine {
    pub fn new() -> Self {
        GenesisEngine { rng: OsRng }
    }

    /// Execute the full Node0 genesis ceremony
    pub fn execute_genesis(
        &mut self,
        name: &str,
        location: &str,
        hardware: HardwareContribution,
        knowledge: KnowledgeContribution,
    ) -> Node0Genesis {
        let timestamp = Utc::now().timestamp_millis() as u64;

        println!("═══════════════════════════════════════════════════════════════════");
        println!("              BIZRA NODE0 GENESIS CEREMONY");
        println!("═══════════════════════════════════════════════════════════════════");
        println!();

        // Step 1: Mint Node0 Identity
        println!("▸ Step 1: Minting Node0 Identity...");
        let identity = self.mint_identity(name, location);
        println!(
            "  ✓ Identity: {} ({})",
            identity.name,
            &identity.node_id[..16]
        );
        println!("  ✓ Public Key: {}...", &identity.public_key[..32]);
        println!();

        // Step 2: Register Hardware Contribution
        println!("▸ Step 2: Registering Hardware Contribution...");
        println!("  CPU: {} ({} cores)", hardware.cpu, hardware.cpu_cores);
        println!(
            "  GPU: {} ({} GB VRAM)",
            hardware.gpu,
            hardware.gpu_vram / (1024 * 1024 * 1024)
        );
        println!("  RAM: {} GB", hardware.ram / (1024 * 1024 * 1024));
        println!("  Storage: {} GB", hardware.storage / (1024 * 1024 * 1024));
        println!(
            "  ✓ Estimated {} compute units/day",
            hardware.compute_units_per_day
        );
        println!();

        // Step 3: Register Knowledge Contribution
        println!("▸ Step 3: Registering Knowledge Contribution...");
        println!("  Conversations: {}", knowledge.conversations);
        println!("  Messages: {}", knowledge.messages);
        println!(
            "  Data Size: {} GB",
            knowledge.data_size / (1024 * 1024 * 1024)
        );
        println!(
            "  Date Range: {} to {}",
            knowledge.date_range.0, knowledge.date_range.1
        );
        println!(
            "  ✓ Knowledge hash: {}...",
            hex::encode(&knowledge.knowledge_hash[..8])
        );
        println!();

        // Step 4: Mint 7 PAT (Personal Agentic Team)
        println!("▸ Step 4: Minting Personal Agentic Team (7 PAT)...");
        let pat_team = self.mint_pat(&identity.node_id);
        for agent in &pat_team.agents {
            println!(
                "  ✓ {} - {} ({}...)",
                agent.role,
                agent.agent_id,
                &agent.public_key[..16]
            );
        }
        println!();

        // Step 5: Mint 5 SAT (Protocol Army)
        println!("▸ Step 5: Minting Shared Agentic Team (5 SAT - Protocol Army)...");
        let sat_team = self.mint_sat();
        for agent in &sat_team.agents {
            println!(
                "  ✓ {} - {} ({}...)",
                agent.role,
                agent.agent_id,
                &agent.public_key[..16]
            );
        }
        println!(
            "  Governance: Quorum {:.0}%, Voting {} hours",
            sat_team.governance.quorum * 100.0,
            sat_team.governance.voting_period_hours
        );
        println!();

        // Step 6: Create Partnership Agreement
        println!("▸ Step 6: Creating Partnership Agreement...");
        let partnership_hash = self.create_partnership_hash(&identity, &hardware, &knowledge);
        println!(
            "  ✓ Partnership hash: {}",
            hex::encode(&partnership_hash[..16])
        );
        println!("  Agreement: Hardware + Knowledge → Shared with Resource Pool");
        println!("  Terms: Musharakah (partnership), profit/loss shared proportionally");
        println!();

        // Step 7: Compute Genesis Hash
        println!("▸ Step 7: Computing Genesis Hash...");
        let genesis_hash = self.compute_genesis_hash(
            &identity,
            &hardware,
            &knowledge,
            &pat_team,
            &sat_team,
            &partnership_hash,
        );
        println!("  ✓ GENESIS HASH: {}", hex::encode(genesis_hash));
        println!();

        // Standing on Giants
        println!("═══════════════════════════════════════════════════════════════════");
        println!("                    STANDING ON GIANTS");
        println!("═══════════════════════════════════════════════════════════════════");
        for (giant, contribution) in UNIVERSAL_GIANTS.iter() {
            println!("  {} - {}", giant, contribution);
        }
        println!();

        println!("═══════════════════════════════════════════════════════════════════");
        println!("                    GENESIS COMPLETE");
        println!("═══════════════════════════════════════════════════════════════════");
        println!();
        println!("  Node0 is now LIVE.");
        println!("  The Resource Pool has received the life kiss.");
        println!("  7 PAT connected to you.");
        println!("  5 SAT serving the protocol.");
        println!();
        println!("  بذرة واحدة تصنع غابة");
        println!("  One seed makes a forest.");
        println!();

        Node0Genesis {
            timestamp,
            identity,
            hardware,
            knowledge,
            pat_team,
            sat_team,
            partnership_hash,
            genesis_hash,
        }
    }

    /// Mint Node0 identity
    fn mint_identity(&mut self, name: &str, location: &str) -> Node0Identity {
        let signing_key = SigningKey::generate(&mut self.rng);
        let verifying_key = signing_key.verifying_key();
        let public_key = hex::encode(verifying_key.as_bytes());
        let node_id = self.derive_node_id(&public_key);
        let created_at = Utc::now().timestamp_millis() as u64;

        let mut hasher = Sha256::new();
        hasher.update(node_id.as_bytes());
        hasher.update(public_key.as_bytes());
        hasher.update(name.as_bytes());
        hasher.update(location.as_bytes());
        let identity_hash: [u8; 32] = hasher.finalize().into();

        Node0Identity {
            node_id,
            public_key,
            name: name.to_string(),
            location: location.to_string(),
            created_at,
            identity_hash,
        }
    }

    /// Derive node ID from public key
    fn derive_node_id(&self, public_key: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"BIZRA_NODE_ID_V1:");
        hasher.update(public_key.as_bytes());
        let hash = hasher.finalize();
        format!("node0_{}", hex::encode(&hash[..8]))
    }

    /// Mint the 7 PAT agents
    fn mint_pat(&mut self, owner_node: &str) -> PersonalAgentTeam {
        let mut agents = Vec::with_capacity(7);

        let pat_capabilities = [
            vec!["plan", "analyze", "decide"],       // Strategist
            vec!["search", "synthesize", "cite"],    // Researcher
            vec!["code", "test", "deploy"],          // Developer
            vec!["pattern", "measure", "predict"],   // Analyst
            vec!["validate", "critique", "improve"], // Reviewer
            vec!["execute", "monitor", "report"],    // Executor
            vec!["protect", "audit", "enforce"],     // Guardian
        ];

        let pat_giants = [
            vec!["Sun Tzu", "Clausewitz", "Porter"],  // Strategist
            vec!["Shannon", "Besta", "Hinton"],       // Researcher
            vec!["Knuth", "Dijkstra", "Thompson"],    // Developer
            vec!["Tukey", "Fisher", "Bayes"],         // Analyst
            vec!["Hoare", "Dijkstra", "Meyer"],       // Reviewer
            vec!["Deming", "Taylor", "Ohno"],         // Executor
            vec!["Al-Ghazali", "Rawls", "Anthropic"], // Guardian
        ];

        for (i, role) in PAT_ROLES.iter().enumerate() {
            let signing_key = SigningKey::generate(&mut self.rng);
            let public_key = hex::encode(signing_key.verifying_key().as_bytes());
            let agent_id = format!("pat_{}_{}", role.to_lowercase(), &public_key[..8]);

            let mut hasher = Sha256::new();
            hasher.update(agent_id.as_bytes());
            hasher.update(public_key.as_bytes());
            hasher.update(role.as_bytes());
            let agent_hash: [u8; 32] = hasher.finalize().into();

            agents.push(MintedAgent {
                agent_id,
                role: role.to_string(),
                public_key,
                capabilities: pat_capabilities[i].iter().map(|s| s.to_string()).collect(),
                giants: pat_giants[i].iter().map(|s| s.to_string()).collect(),
                created_at: Utc::now().timestamp_millis() as u64,
                agent_hash,
            });
        }

        let mut hasher = Sha256::new();
        hasher.update(owner_node.as_bytes());
        for agent in &agents {
            hasher.update(agent.agent_hash);
        }
        let team_hash: [u8; 32] = hasher.finalize().into();

        PersonalAgentTeam {
            owner_node: owner_node.to_string(),
            agents,
            team_hash,
        }
    }

    /// Mint the 5 SAT agents (protocol army)
    fn mint_sat(&mut self) -> SharedAgentTeam {
        let mut agents = Vec::with_capacity(5);

        let sat_capabilities = [
            vec!["verify", "attest", "reject"],        // Validator
            vec!["fetch", "anchor", "timestamp"],      // Oracle
            vec!["arbitrate", "negotiate", "resolve"], // Mediator
            vec!["index", "preserve", "retrieve"],     // Archivist
            vec!["detect", "alert", "defend"],         // Sentinel
        ];

        let sat_giants = [
            vec!["Lamport", "Nakamoto", "Buterin"],    // Validator
            vec!["Shannon", "Wolfram", "Oracles"],     // Oracle
            vec!["Nash", "Schelling", "Ostrom"],       // Mediator
            vec!["Shannon", "Huffman", "Berners-Lee"], // Archivist
            vec!["Schneier", "Anderson", "Diffie"],    // Sentinel
        ];

        for (i, role) in SAT_ROLES.iter().enumerate() {
            let signing_key = SigningKey::generate(&mut self.rng);
            let public_key = hex::encode(signing_key.verifying_key().as_bytes());
            let agent_id = format!("sat_{}_{}", role.to_lowercase(), &public_key[..8]);

            let mut hasher = Sha256::new();
            hasher.update(agent_id.as_bytes());
            hasher.update(public_key.as_bytes());
            hasher.update(role.as_bytes());
            let agent_hash: [u8; 32] = hasher.finalize().into();

            agents.push(MintedAgent {
                agent_id,
                role: role.to_string(),
                public_key,
                capabilities: sat_capabilities[i].iter().map(|s| s.to_string()).collect(),
                giants: sat_giants[i].iter().map(|s| s.to_string()).collect(),
                created_at: Utc::now().timestamp_millis() as u64,
                agent_hash,
            });
        }

        let mut hasher = Sha256::new();
        hasher.update(b"BIZRA_SAT_GENESIS");
        for agent in &agents {
            hasher.update(agent.agent_hash);
        }
        let team_hash: [u8; 32] = hasher.finalize().into();

        SharedAgentTeam {
            agents,
            governance: SATGovernance {
                quorum: 0.67,            // 2/3 majority
                voting_period_hours: 72, // 3 days
                upgrade_threshold: 0.80, // 80% for upgrades
            },
            team_hash,
        }
    }

    /// Create partnership agreement hash
    fn create_partnership_hash(
        &self,
        identity: &Node0Identity,
        hardware: &HardwareContribution,
        knowledge: &KnowledgeContribution,
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"BIZRA_PARTNERSHIP_V1:");
        hasher.update(identity.identity_hash);
        hasher.update(hardware.compute_units_per_day.to_le_bytes());
        hasher.update(knowledge.knowledge_hash);
        hasher.update(b":MUSHARAKAH"); // Partnership model
        hasher.finalize().into()
    }

    /// Compute the genesis block hash
    fn compute_genesis_hash(
        &self,
        identity: &Node0Identity,
        hardware: &HardwareContribution,
        knowledge: &KnowledgeContribution,
        pat_team: &PersonalAgentTeam,
        sat_team: &SharedAgentTeam,
        partnership_hash: &[u8; 32],
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"BIZRA_GENESIS_BLOCK_V1:");
        hasher.update(identity.identity_hash);
        hasher.update(hardware.compute_units_per_day.to_le_bytes());
        hasher.update(knowledge.knowledge_hash);
        hasher.update(pat_team.team_hash);
        hasher.update(sat_team.team_hash);
        hasher.update(partnership_hash);
        hasher.update(b":NODE0");
        hasher.finalize().into()
    }
}

impl Default for GenesisEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hardware() -> HardwareContribution {
        HardwareContribution {
            cpu: "Intel i9-14900HX".to_string(),
            cpu_cores: 24,
            gpu: "NVIDIA RTX 4090".to_string(),
            gpu_vram: 16 * 1024 * 1024 * 1024,      // 16 GB
            ram: 128 * 1024 * 1024 * 1024,          // 128 GB
            storage: 4 * 1024 * 1024 * 1024 * 1024, // 4 TB
            network_bps: 1000 * 1024 * 1024,        // 1 Gbps
            compute_units_per_day: 100_000,
        }
    }

    fn test_knowledge() -> KnowledgeContribution {
        let mut hasher = Sha256::new();
        hasher.update(b"BIZRA_DATA_LAKE_2023_2025");
        KnowledgeContribution {
            conversations: 1241,
            messages: 24746,
            data_size: 5 * 1024 * 1024 * 1024, // 5.4 GB
            date_range: ("2023-07-22".to_string(), "2025-12-16".to_string()),
            concepts: vec![
                "BIZRA".to_string(),
                "Agents".to_string(),
                "Ihsān".to_string(),
                "Federation".to_string(),
            ],
            knowledge_hash: hasher.finalize().into(),
        }
    }

    #[test]
    fn test_identity_minting() {
        let mut engine = GenesisEngine::new();
        let identity = engine.mint_identity("MoMo", "Dubai, UAE");

        assert!(identity.node_id.starts_with("node0_"));
        assert_eq!(identity.public_key.len(), 64); // Ed25519 pubkey
        assert_eq!(identity.name, "MoMo");
        assert_eq!(identity.location, "Dubai, UAE");
    }

    #[test]
    fn test_pat_minting() {
        let mut engine = GenesisEngine::new();
        let pat = engine.mint_pat("node0_test");

        assert_eq!(pat.agents.len(), 7);
        assert_eq!(pat.owner_node, "node0_test");

        // Verify all roles present
        let roles: Vec<_> = pat.agents.iter().map(|a| a.role.as_str()).collect();
        assert!(roles.contains(&"Strategist"));
        assert!(roles.contains(&"Guardian"));
    }

    #[test]
    fn test_sat_minting() {
        let mut engine = GenesisEngine::new();
        let sat = engine.mint_sat();

        assert_eq!(sat.agents.len(), 5);
        assert_eq!(sat.governance.quorum, 0.67);

        // Verify all roles present
        let roles: Vec<_> = sat.agents.iter().map(|a| a.role.as_str()).collect();
        assert!(roles.contains(&"Validator"));
        assert!(roles.contains(&"Sentinel"));
    }

    #[test]
    fn test_full_genesis() {
        let mut engine = GenesisEngine::new();
        let genesis =
            engine.execute_genesis("MoMo", "Dubai, UAE", test_hardware(), test_knowledge());

        assert_eq!(genesis.pat_team.agents.len(), 7);
        assert_eq!(genesis.sat_team.agents.len(), 5);
        assert!(!genesis.genesis_hash.iter().all(|&b| b == 0));
    }
}
