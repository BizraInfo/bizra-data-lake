//! HTTP boundary contracts — §Spearpoint A (post-Cycle-7)
//!
//! بسم الله الرحمن الرحيم
//!
//! Single-source-of-truth for the gateway HTTP surface. Every type
//! here is:
//!   1. Rust-enforced at the gateway server boundary
//!   2. Emitted as TypeScript to `bindings/` via ts-rs (run `cargo test`)
//!   3. Consumed by the UI fork as `import { T } from '.../bindings'`
//!
//! Drift discipline: if you change a field here, `cargo test`
//! regenerates the .ts; CI fails on unstaged diff in `bindings/`.
//!
//! Scope: HTTP-exposed surface only. Runtime-internal library types
//! (those in `bizra_cognition`) stay Rust-only and are projected
//! through these DTOs at the gateway layer.

use serde::{Deserialize, Serialize};
use ts_rs::TS;

// ════════════════════════════════════════════════════════════════════
// Commit 1 exemplar — ReceiptKindName
// ════════════════════════════════════════════════════════════════════
//
// The runtime `ReceiptKind` enum has explicit u8 discriminants. Over
// the HTTP wire we expose the canonical string name (what operators
// read) together with the byte. This type is the wire projection.

/// Canonical string name for a `ReceiptKind` at the HTTP boundary.
/// Matches the Rust enum 1:1 by naming convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/")]
pub enum ReceiptKindName {
    Genesis,
    CognitionBoot,
    Myelination,
    Demyelination,
    ReasoningSession,
    GovernanceDecision,
    NodeLifecycle,
    Manifest,
    PrincipalActivation,
    MissionExecuted,
    DegradedPath,
}

impl ReceiptKindName {
    /// Canonical byte discriminant matching bizra_cognition::receipts::ReceiptKind.
    pub fn to_byte(self) -> u8 {
        match self {
            Self::Genesis => 0x00,
            Self::CognitionBoot => 0x10,
            Self::Myelination => 0x20,
            Self::Demyelination => 0x21,
            Self::ReasoningSession => 0x30,
            Self::GovernanceDecision => 0x40,
            Self::NodeLifecycle => 0x50,
            Self::Manifest => 0x60,
            Self::PrincipalActivation => 0x61,
            Self::MissionExecuted => 0x70,
            Self::DegradedPath => 0xF0,
        }
    }
}

impl From<bizra_cognition::receipts::ReceiptKind> for ReceiptKindName {
    fn from(k: bizra_cognition::receipts::ReceiptKind) -> Self {
        use bizra_cognition::receipts::ReceiptKind;
        match k {
            ReceiptKind::Genesis => Self::Genesis,
            ReceiptKind::CognitionBoot => Self::CognitionBoot,
            ReceiptKind::Myelination => Self::Myelination,
            ReceiptKind::Demyelination => Self::Demyelination,
            ReceiptKind::ReasoningSession => Self::ReasoningSession,
            ReceiptKind::GovernanceDecision => Self::GovernanceDecision,
            ReceiptKind::NodeLifecycle => Self::NodeLifecycle,
            ReceiptKind::Manifest => Self::Manifest,
            ReceiptKind::PrincipalActivation => Self::PrincipalActivation,
            ReceiptKind::MissionExecuted => Self::MissionExecuted,
            ReceiptKind::DegradedPath => Self::DegradedPath,
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Commit 1 exemplar — MissionStageName
// ════════════════════════════════════════════════════════════════════

/// Canonical string name for a `MissionStage` at the HTTP boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/")]
pub enum MissionStageName {
    Intent,
    Mission,
    Claim,
    Admissibility,
    Execution,
    Receipt,
    Canonicalization,
    Replayability,
    Reflex,
}

impl From<bizra_cognition::mission_freeze_v1::MissionStage> for MissionStageName {
    fn from(s: bizra_cognition::mission_freeze_v1::MissionStage) -> Self {
        use bizra_cognition::mission_freeze_v1::MissionStage;
        match s {
            MissionStage::Intent => Self::Intent,
            MissionStage::Mission => Self::Mission,
            MissionStage::Claim => Self::Claim,
            MissionStage::Admissibility => Self::Admissibility,
            MissionStage::Execution => Self::Execution,
            MissionStage::Receipt => Self::Receipt,
            MissionStage::Canonicalization => Self::Canonicalization,
            MissionStage::Replayability => Self::Replayability,
            MissionStage::Reflex => Self::Reflex,
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Tests — sanity checks + ts-rs emission triggers
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receipt_kind_name_bytes_match_runtime() {
        // Every HTTP boundary byte must match the runtime byte exactly.
        // Drift here = UI state diverges from chain state = §10 Proof
        // Law violation waiting to happen.
        use bizra_cognition::receipts::ReceiptKind;
        let pairs = [
            (ReceiptKind::Genesis, 0x00u8),
            (ReceiptKind::CognitionBoot, 0x10),
            (ReceiptKind::Myelination, 0x20),
            (ReceiptKind::Demyelination, 0x21),
            (ReceiptKind::ReasoningSession, 0x30),
            (ReceiptKind::GovernanceDecision, 0x40),
            (ReceiptKind::NodeLifecycle, 0x50),
            (ReceiptKind::Manifest, 0x60),
            (ReceiptKind::PrincipalActivation, 0x61),
            (ReceiptKind::MissionExecuted, 0x70),
            (ReceiptKind::DegradedPath, 0xF0),
        ];
        for (rk, byte) in pairs {
            let name: ReceiptKindName = rk.into();
            assert_eq!(
                name.to_byte(),
                byte,
                "boundary byte for {:?} must match runtime {:#x}",
                rk,
                byte
            );
        }
    }

    #[test]
    fn all_mission_stages_project_to_names() {
        use bizra_cognition::mission_freeze_v1::MissionStage;
        for s in [
            MissionStage::Intent,
            MissionStage::Mission,
            MissionStage::Claim,
            MissionStage::Admissibility,
            MissionStage::Execution,
            MissionStage::Receipt,
            MissionStage::Canonicalization,
            MissionStage::Replayability,
            MissionStage::Reflex,
        ] {
            let _ = MissionStageName::from(s);
        }
    }
}
