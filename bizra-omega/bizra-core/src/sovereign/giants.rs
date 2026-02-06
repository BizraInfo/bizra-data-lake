//! Standing on the Shoulders of Giants — Intellectual Lineage Registry
//!
//! This module codifies the intellectual foundations of BIZRA,
//! honoring the giants whose work made this system possible.
//!
//! "If I have seen further, it is by standing on the shoulders of Giants."
//! — Isaac Newton, 1675

use std::collections::HashMap;

/// A giant in the intellectual lineage
#[derive(Clone, Debug)]
pub struct Giant {
    /// Name of the giant
    pub name: String,
    /// Domain of expertise
    pub domain: String,
    /// Key contributions
    pub contributions: Vec<Contribution>,
    /// Where their work is used in BIZRA
    pub bizra_usage: Vec<String>,
    /// Citation or reference
    pub citation: Option<String>,
}

impl Giant {
    /// Create a new giant entry
    pub fn new(name: &str, domain: &str) -> Self {
        Self {
            name: name.to_string(),
            domain: domain.to_string(),
            contributions: Vec::new(),
            bizra_usage: Vec::new(),
            citation: None,
        }
    }

    /// Add a contribution
    pub fn with_contribution(mut self, name: &str, description: &str) -> Self {
        self.contributions.push(Contribution {
            name: name.to_string(),
            description: description.to_string(),
        });
        self
    }

    /// Add BIZRA usage
    pub fn used_in(mut self, component: &str) -> Self {
        self.bizra_usage.push(component.to_string());
        self
    }

    /// Add citation
    pub fn with_citation(mut self, citation: &str) -> Self {
        self.citation = Some(citation.to_string());
        self
    }
}

/// A specific contribution from a giant
#[derive(Clone, Debug)]
pub struct Contribution {
    /// Name of the contribution
    pub name: String,
    /// Description
    pub description: String,
}

/// The Giants Registry — Intellectual lineage of BIZRA
pub struct GiantRegistry {
    giants: HashMap<String, Giant>,
}

impl GiantRegistry {
    /// Create the complete giants registry
    pub fn new() -> Self {
        let mut registry = Self {
            giants: HashMap::new(),
        };
        registry.populate();
        registry
    }

    /// Populate with all giants
    fn populate(&mut self) {
        // Claude Shannon — Information Theory
        self.add(
            Giant::new("Claude Shannon", "Information Theory")
                .with_contribution(
                    "Information Entropy",
                    "H(X) = -Σ p(x) log p(x) — measuring information content"
                )
                .with_contribution(
                    "Channel Capacity",
                    "Maximum rate of reliable communication"
                )
                .with_contribution(
                    "SNR Relationship",
                    "C = B log₂(1 + S/N) — capacity from signal-to-noise"
                )
                .used_in("SNR Engine")
                .used_in("Constitution thresholds")
                .with_citation("A Mathematical Theory of Communication, 1948")
        );

        // Leslie Lamport — Distributed Systems
        self.add(
            Giant::new("Leslie Lamport", "Distributed Systems")
                .with_contribution(
                    "Paxos Consensus",
                    "Fault-tolerant distributed agreement protocol"
                )
                .with_contribution(
                    "Byzantine Generals",
                    "Problem formulation for adversarial consensus"
                )
                .with_contribution(
                    "Logical Clocks",
                    "Happened-before relation for event ordering"
                )
                .used_in("BFT Consensus Engine")
                .used_in("Federation Protocol")
                .with_citation("Paxos Made Simple, 2001")
        );

        // Ashish Vaswani et al. — Transformers
        self.add(
            Giant::new("Ashish Vaswani", "Deep Learning")
                .with_contribution(
                    "Transformer Architecture",
                    "Self-attention mechanism for sequence modeling"
                )
                .with_contribution(
                    "Multi-Head Attention",
                    "Parallel attention for diverse representations"
                )
                .used_in("Inference Gateway")
                .used_in("Task Complexity Estimation")
                .with_citation("Attention Is All You Need, 2017")
        );

        // Maciej Besta et al. — Graph of Thoughts
        self.add(
            Giant::new("Maciej Besta", "AI Reasoning")
                .with_contribution(
                    "Graph-of-Thoughts",
                    "Non-linear reasoning with thought graphs"
                )
                .with_contribution(
                    "Thought Aggregation",
                    "Combining multiple reasoning paths"
                )
                .used_in("Sovereign Orchestrator")
                .used_in("Validation Reasoning")
                .with_citation("Graph of Thoughts: Solving Elaborate Problems, 2023")
        );

        // Linus Torvalds — Unix Philosophy
        self.add(
            Giant::new("Linus Torvalds", "Operating Systems")
                .with_contribution(
                    "Unix Philosophy",
                    "Do one thing well, compose with pipes"
                )
                .with_contribution(
                    "Git",
                    "Distributed version control with integrity"
                )
                .used_in("CLI Design")
                .used_in("Module Architecture")
                .with_citation("Just for Fun, 2001")
        );

        // Roy Fielding — REST
        self.add(
            Giant::new("Roy Fielding", "Web Architecture")
                .with_contribution(
                    "REST",
                    "Representational State Transfer architectural style"
                )
                .with_contribution(
                    "Uniform Interface",
                    "Resource-based, self-descriptive messages"
                )
                .used_in("API Gateway")
                .used_in("Federation Protocol")
                .with_citation("Architectural Styles and REST, 2000")
        );

        // Georgi Gerganov — llama.cpp
        self.add(
            Giant::new("Georgi Gerganov", "Efficient Inference")
                .with_contribution(
                    "llama.cpp",
                    "Efficient C++ implementation of LLM inference"
                )
                .with_contribution(
                    "GGML",
                    "Tensor library optimized for LLMs"
                )
                .with_contribution(
                    "Quantization",
                    "Memory-efficient model representation"
                )
                .used_in("LlamaCpp FFI Backend")
                .used_in("SIMD Optimizations")
                .with_citation("llama.cpp, 2023")
        );

        // Daniel Bernstein — Cryptography
        self.add(
            Giant::new("Daniel J. Bernstein", "Cryptography")
                .with_contribution(
                    "Curve25519",
                    "High-performance elliptic curve cryptography"
                )
                .with_contribution(
                    "Ed25519",
                    "Fast and secure digital signatures"
                )
                .with_contribution(
                    "ChaCha20",
                    "Stream cipher for authenticated encryption"
                )
                .used_in("Node Identity")
                .used_in("PCI Signatures")
                .with_citation("High-speed high-security signatures, 2012")
        );

        // Jack O'Connor — BLAKE3
        self.add(
            Giant::new("Jack O'Connor", "Cryptographic Hashing")
                .with_contribution(
                    "BLAKE3",
                    "Fast, parallel, secure cryptographic hash"
                )
                .with_contribution(
                    "Merkle Tree Mode",
                    "Unlimited parallelism for large data"
                )
                .used_in("Domain Separation")
                .used_in("Content Hashing")
                .with_citation("BLAKE3 Specification, 2020")
        );

        // Anthropic — Claude
        self.add(
            Giant::new("Anthropic", "AI Safety")
                .with_contribution(
                    "Constitutional AI",
                    "AI systems governed by principles"
                )
                .with_contribution(
                    "RLHF",
                    "Reinforcement learning from human feedback"
                )
                .with_contribution(
                    "Claude",
                    "Helpful, harmless, honest AI assistant"
                )
                .used_in("Constitution Framework")
                .used_in("Ihsān Constraints")
                .with_citation("Constitutional AI, 2022")
        );
    }

    /// Add a giant to the registry
    pub fn add(&mut self, giant: Giant) {
        self.giants.insert(giant.name.clone(), giant);
    }

    /// Get a giant by name
    pub fn get(&self, name: &str) -> Option<&Giant> {
        self.giants.get(name)
    }

    /// Get all giants
    pub fn all(&self) -> impl Iterator<Item = &Giant> {
        self.giants.values()
    }

    /// Get giants by domain
    pub fn by_domain(&self, domain: &str) -> Vec<&Giant> {
        self.giants.values()
            .filter(|g| g.domain == domain)
            .collect()
    }

    /// Get giants used in a component
    pub fn used_in(&self, component: &str) -> Vec<&Giant> {
        self.giants.values()
            .filter(|g| g.bizra_usage.iter().any(|u| u == component))
            .collect()
    }

    /// Format attribution string
    pub fn attribution(&self) -> String {
        let mut s = String::new();
        s.push_str("BIZRA Omega — Standing on the Shoulders of Giants\n\n");
        
        for giant in self.giants.values() {
            s.push_str(&format!("• {} ({})\n", giant.name, giant.domain));
            for contrib in &giant.contributions {
                s.push_str(&format!("  - {}: {}\n", contrib.name, contrib.description));
            }
            if let Some(ref citation) = giant.citation {
                s.push_str(&format!("  Reference: {}\n", citation));
            }
            s.push('\n');
        }
        
        s
    }

    /// Get count
    pub fn count(&self) -> usize {
        self.giants.len()
    }
}

impl Default for GiantRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = GiantRegistry::new();
        assert!(registry.count() >= 10, "Should have at least 10 giants");
    }

    #[test]
    fn test_get_giant() {
        let registry = GiantRegistry::new();
        let shannon = registry.get("Claude Shannon");
        assert!(shannon.is_some());
        assert_eq!(shannon.unwrap().domain, "Information Theory");
    }

    #[test]
    fn test_by_domain() {
        let registry = GiantRegistry::new();
        let crypto = registry.by_domain("Cryptography");
        assert!(!crypto.is_empty());
    }

    #[test]
    fn test_used_in() {
        let registry = GiantRegistry::new();
        let snr_giants = registry.used_in("SNR Engine");
        assert!(!snr_giants.is_empty());
    }

    #[test]
    fn test_attribution() {
        let registry = GiantRegistry::new();
        let attribution = registry.attribution();
        assert!(attribution.contains("Shannon"));
        assert!(attribution.contains("Lamport"));
    }
}
