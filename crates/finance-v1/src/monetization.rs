use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tier 1: Freemium Academic Access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcademicFreemiumModel {
    pub free_tier_limit_n: u32,
    pub features: Vec<String>,
    pub rate_limit: String,
    pub attribution_required: bool,
}

impl Default for AcademicFreemiumModel {
    fn default() -> Self {
        Self {
            free_tier_limit_n: 1000,
            features: vec![
                "constructibility_check".to_string(),
                "basic_tower".to_string(),
                "historical_notes".to_string(),
            ],
            rate_limit: "10 queries/hour".to_string(),
            attribution_required: true,
        }
    }
}

/// Tier 2: Professional Engineering API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineeringAPITier {
    pub name: String,
    pub price: String,
    pub quota: String,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineeringAPI {
    pub tiers: HashMap<String, EngineeringAPITier>,
}

impl Default for EngineeringAPI {
    fn default() -> Self {
        let mut tiers = HashMap::new();
        tiers.insert(
            "startup".to_string(),
            EngineeringAPITier {
                name: "Startup".to_string(),
                price: "$99/month".to_string(),
                quota: "1000 API calls".to_string(),
                features: vec![
                    "cad_export".to_string(),
                    "construction_steps".to_string(),
                    "error_bounds".to_string(),
                ],
            },
        );
        tiers.insert(
            "enterprise".to_string(),
            EngineeringAPITier {
                name: "Enterprise".to_string(),
                price: "$5000/month".to_string(),
                quota: "Unlimited calls".to_string(),
                features: vec![
                    "white_label_cad".to_string(),
                    "formal_verification_certificate".to_string(),
                    "blockchain_attestation".to_string(),
                    "priority_support".to_string(),
                    "custom_polygon_implementations".to_string(),
                ],
            },
        );
        Self { tiers }
    }
}

/// Tier 3: Mathematical Proof NFT Marketplace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofNFTMetadata {
    pub name: String,
    pub description: String,
    pub attributes: Vec<NFTAttribute>,
    pub proof_content: NFTProofContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NFTAttribute {
    pub trait_type: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NFTProofContent {
    pub lean_proof: String,
    pub quantum_circuit_qasm: String,
    pub cad_file_hash: String,
}

/// Tier 4: Research Grant Pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrantOpportunity {
    pub agency: String,
    pub amount: String,
    pub deliverables: Vec<String>,
    pub overhead_rate: String,
}

/// Tier 5: Institutional Licensing (The "Gauss Tier")
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussTierLicense {
    pub price: String,
    pub duration: String,
    pub includes: Vec<String>,
    pub equity_clause: String,
    pub exit_strategy: String,
}

impl Default for GaussTierLicense {
    fn default() -> Self {
        Self {
            price: "$1M/year + equity stake".to_string(),
            duration: "5 years exclusive".to_string(),
            includes: vec![
                "Full source code with modifications".to_string(),
                "Dedicated support team".to_string(),
                "Custom Fermat prime research".to_string(),
                "Private blockchain attestation".to_string(),
            ],
            equity_clause: "2% of licensee revenue".to_string(),
            exit_strategy: "Right to acquire system for $10M after 5 years".to_string(),
        }
    }
}

/// Tier 6: Ihsān Impact Fund
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IhsanImpactFund {
    pub source: String,
    pub governance: String,
    pub recipients: Vec<String>,
    pub transparency: String,
}

impl Default for IhsanImpactFund {
    fn default() -> Self {
        Self {
            source: "20% gross revenue from all tiers".to_string(),
            governance: "Token holders vote on grants".to_string(),
            recipients: vec![
                "Free tier expansion".to_string(),
                "Lean 4 workshops".to_string(),
                "Quantum computing access".to_string(),
            ],
            transparency: "All transactions on public blockchain".to_string(),
        }
    }
}

/// Financial Projection Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancialProjection {
    pub academic_tier: String,
    pub engineering_api: String,
    pub nft_marketplace: String,
    pub grants: String,
    pub total_revenue: String,
    pub margin_or_valuation: String,
}

pub fn get_roadmap_projections() -> (FinancialProjection, FinancialProjection) {
    (
        FinancialProjection {
            academic_tier: "$500K".to_string(),
            engineering_api: "$1.2M".to_string(),
            nft_marketplace: "$300K".to_string(),
            grants: "$500K".to_string(),
            total_revenue: "$2.5M".to_string(),
            margin_or_valuation: "60% margin".to_string(),
        },
        FinancialProjection {
            academic_tier: "$5M".to_string(),
            engineering_api: "$30M".to_string(),
            nft_marketplace: "$5M".to_string(),
            grants: "$2M".to_string(),
            total_revenue: "$52M".to_string(),
            margin_or_valuation: "$500M Enterprise Value".to_string(),
        },
    )
}
