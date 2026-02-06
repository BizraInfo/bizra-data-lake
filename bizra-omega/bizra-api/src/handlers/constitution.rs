//! Constitution Handlers — Governance rules and compliance

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::state::AppState;
use bizra_core::{IHSAN_THRESHOLD, SNR_THRESHOLD};

#[derive(Serialize)]
pub struct ConstitutionResponse {
    pub version: String,
    pub ihsan_threshold: f64,
    pub snr_threshold: f64,
    pub rules: Vec<RuleInfo>,
    pub penalties: Vec<PenaltyInfo>,
}

#[derive(Serialize)]
pub struct RuleInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub enforcement: String,
}

#[derive(Serialize)]
pub struct PenaltyInfo {
    pub violation: String,
    pub penalty: String,
    pub severity: String,
}

/// Get constitution
pub async fn get_constitution(State(state): State<Arc<AppState>>) -> Json<ConstitutionResponse> {
    let c = &state.constitution;

    Json(ConstitutionResponse {
        version: c.version.clone(),
        ihsan_threshold: IHSAN_THRESHOLD,
        snr_threshold: SNR_THRESHOLD,
        rules: vec![
            RuleInfo {
                id: "R001".into(),
                name: "Ihsan Constraint".into(),
                description: "All outputs must meet excellence threshold (>= 0.95)".into(),
                enforcement: "Hard".into(),
            },
            RuleInfo {
                id: "R002".into(),
                name: "SNR Minimum".into(),
                description: "Signal-to-noise ratio must exceed threshold (>= 0.85)".into(),
                enforcement: "Hard".into(),
            },
            RuleInfo {
                id: "R003".into(),
                name: "Cryptographic Integrity".into(),
                description: "All messages must be signed with Ed25519".into(),
                enforcement: "Hard".into(),
            },
            RuleInfo {
                id: "R004".into(),
                name: "Provenance Chain".into(),
                description: "Full lineage must be traceable for all derived content".into(),
                enforcement: "Soft".into(),
            },
            RuleInfo {
                id: "R005".into(),
                name: "Byzantine Tolerance".into(),
                description: "Consensus requires 2f+1 votes to tolerate f malicious nodes".into(),
                enforcement: "Hard".into(),
            },
        ],
        penalties: vec![
            PenaltyInfo {
                violation: "Below Ihsan threshold".into(),
                penalty: "Message rejected, not propagated".into(),
                severity: "High".into(),
            },
            PenaltyInfo {
                violation: "Invalid signature".into(),
                penalty: "Sender marked suspicious, rate limited".into(),
                severity: "Critical".into(),
            },
            PenaltyInfo {
                violation: "Repeated low-quality outputs".into(),
                penalty: "Node reputation decreased".into(),
                severity: "Medium".into(),
            },
        ],
    })
}

#[derive(Deserialize)]
pub struct CheckComplianceRequest {
    pub ihsan_score: Option<f64>,
    pub snr_score: Option<f64>,
    pub has_signature: bool,
    pub has_provenance: bool,
}

#[derive(Serialize)]
pub struct ComplianceResult {
    pub compliant: bool,
    pub checks: Vec<ComplianceCheck>,
}

#[derive(Serialize)]
pub struct ComplianceCheck {
    pub rule: String,
    pub passed: bool,
    pub value: Option<String>,
    pub threshold: Option<String>,
}

/// Check compliance against constitution
pub async fn check_compliance(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CheckComplianceRequest>,
) -> Json<ComplianceResult> {
    let mut checks = Vec::new();
    let mut all_passed = true;

    // Check Ihsan
    if let Some(score) = req.ihsan_score {
        let passed = state.constitution.check_ihsan(score);
        if !passed {
            all_passed = false;
        }
        checks.push(ComplianceCheck {
            rule: "Ihsan Constraint".into(),
            passed,
            value: Some(format!("{:.4}", score)),
            threshold: Some(format!("{:.4}", IHSAN_THRESHOLD)),
        });
    }

    // Check SNR
    if let Some(score) = req.snr_score {
        let passed = state.constitution.check_snr(score);
        if !passed {
            all_passed = false;
        }
        checks.push(ComplianceCheck {
            rule: "SNR Minimum".into(),
            passed,
            value: Some(format!("{:.4}", score)),
            threshold: Some(format!("{:.4}", SNR_THRESHOLD)),
        });
    }

    // Check signature
    if !req.has_signature {
        all_passed = false;
    }
    checks.push(ComplianceCheck {
        rule: "Cryptographic Integrity".into(),
        passed: req.has_signature,
        value: Some(req.has_signature.to_string()),
        threshold: Some("true".into()),
    });

    // Check provenance (soft enforcement)
    checks.push(ComplianceCheck {
        rule: "Provenance Chain".into(),
        passed: req.has_provenance,
        value: Some(req.has_provenance.to_string()),
        threshold: Some("recommended".into()),
    });

    Json(ComplianceResult {
        compliant: all_passed,
        checks,
    })
}
