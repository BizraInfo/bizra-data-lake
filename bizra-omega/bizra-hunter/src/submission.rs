//! Submission + Bonding primitives

use crate::pipeline::VulnType;

#[derive(Debug, Clone)]
pub struct BondedSubmission {
    pub contract_addr: [u8; 20],
    pub vuln_type: VulnType,
    pub bond_cents: u64,
    pub poc: String,
}

#[derive(Debug, Clone)]
pub struct SubmissionResult {
    pub accepted: bool,
    pub reason: String,
}

impl BondedSubmission {
    pub fn validate(&self) -> SubmissionResult {
        if self.bond_cents == 0 {
            return SubmissionResult { accepted: false, reason: "Bond required".to_string() };
        }
        if self.poc.trim().is_empty() {
            return SubmissionResult { accepted: false, reason: "PoC missing".to_string() };
        }
        SubmissionResult { accepted: true, reason: "Accepted".to_string() }
    }
}
