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
// Admissibility group
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/")]
pub enum VerdictName {
    Permit,
    Reject,
    Review,
    ScoreOnly,
}

impl From<bizra_cognition::admissibility_freeze_v1::Verdict> for VerdictName {
    fn from(v: bizra_cognition::admissibility_freeze_v1::Verdict) -> Self {
        use bizra_cognition::admissibility_freeze_v1::Verdict;
        match v {
            Verdict::Permit => Self::Permit,
            Verdict::Reject => Self::Reject,
            Verdict::Review => Self::Review,
            Verdict::ScoreOnly => Self::ScoreOnly,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/")]
pub enum InvariantName {
    ZannZero,
    ClaimMustBind,
    RibaZero,
    NoShadowState,
    IhsanFloor,
}

impl InvariantName {
    /// Canonical operator-facing scorer_id exactly as the gateway emits.
    pub fn scorer_id(self) -> &'static str {
        match self {
            Self::ZannZero => "ZANN_ZERO",
            Self::ClaimMustBind => "CLAIM_MUST_BIND",
            Self::RibaZero => "RIBA_ZERO",
            Self::NoShadowState => "NO_SHADOW_STATE",
            Self::IhsanFloor => "IHSAN_FLOOR",
        }
    }
}

/// Gate verdict row as it appears inside AdmissibilityResult.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/", rename_all = "camelCase")]
pub struct GateVerdictContract {
    #[serde(rename = "scorerId")]
    pub scorer_id: String,
    pub invariant: Option<String>,
    pub verdict: VerdictName,
    pub reason: String,
    pub score: Option<f64>,
}

/// Structured rejection detail when admissibility refuses a claim.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/", rename_all = "camelCase")]
pub struct RejectedClaimContract {
    pub invariant: String,
    pub reason: String,
    #[serde(rename = "remediationPath")]
    pub remediation_path: String,
    #[serde(rename = "escalationAllowed")]
    pub escalation_allowed: bool,
}

/// Full admissibility verdict shipped by every mission endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/", rename_all = "camelCase")]
pub struct AdmissibilityContract {
    pub verdict: VerdictName,
    #[serde(rename = "gateVerdicts")]
    pub gate_verdicts: Vec<GateVerdictContract>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub rejected: Option<RejectedClaimContract>,
}

// ════════════════════════════════════════════════════════════════════
// Resource / URP group (G4)
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/")]
pub struct ResourceContract {
    pub kind: String,
    pub id: String,
    pub summary: String,
    pub allowlisted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/")]
pub struct UrpBucketContract {
    pub kind: String,
    pub resources: Vec<ResourceContract>,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/", rename_all = "camelCase")]
pub struct UrpViewContract {
    #[serde(rename = "totalCount")]
    pub total_count: usize,
    #[serde(rename = "allowlistedCount")]
    pub allowlisted_count: usize,
    pub buckets: Vec<UrpBucketContract>,
}

// ════════════════════════════════════════════════════════════════════
// Organize action group (G5)
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/", rename_all = "lowercase")]
pub enum OrganizeEntryKind {
    #[serde(rename = "file")]
    File,
    #[serde(rename = "directory")]
    Directory,
    #[serde(rename = "symlink")]
    Symlink,
    #[serde(rename = "other")]
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/")]
pub struct OrganizeEntryContract {
    pub name: String,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/", rename_all = "camelCase")]
pub struct OrganizeResponseContract {
    #[serde(rename = "missionId")]
    pub mission_id: String,
    #[serde(rename = "missionReceiptId")]
    pub mission_receipt_id: String,
    #[serde(rename = "organizeReceiptId")]
    pub organize_receipt_id: String,
    #[serde(rename = "chainHead")]
    pub chain_head: String,
    pub path: String,
    #[serde(rename = "listingDigest")]
    pub listing_digest: String,
    #[serde(rename = "fileCount")]
    pub file_count: u32,
    #[serde(rename = "dirCount")]
    pub dir_count: u32,
    #[serde(rename = "entryCount")]
    pub entry_count: u32,
    pub entries: Vec<OrganizeEntryContract>,
    #[serde(rename = "timestampNs")]
    pub timestamp_ns: u64,
    pub admissibility: AdmissibilityContract,
}

// ════════════════════════════════════════════════════════════════════
// Proof-of-Impact group (G6)
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/", rename_all = "camelCase")]
pub struct PoiEntryContract {
    #[serde(rename = "receiptId")]
    pub receipt_id: String,
    #[serde(rename = "receiptKindByte")]
    pub receipt_kind_byte: u8,
    #[serde(rename = "receiptKindName")]
    pub receipt_kind_name: String,
    #[serde(rename = "qualityScore")]
    pub quality_score: f64,
    #[serde(rename = "gateMinScore")]
    pub gate_min_score: f64,
    #[serde(rename = "entryCount")]
    pub entry_count: u32,
    #[serde(rename = "impactScore")]
    pub impact_score: f64,
    #[serde(rename = "timestampNs")]
    pub timestamp_ns: u64,
    #[serde(rename = "principalId", skip_serializing_if = "Option::is_none")]
    #[ts(rename = "principalId", optional)]
    pub principal_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/", rename_all = "camelCase")]
pub struct PoiLedgerResponseContract {
    #[serde(rename = "chainHead")]
    pub chain_head: String,
    pub entries: Vec<PoiEntryContract>,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/", rename_all = "camelCase")]
pub struct PoiPerKindContract {
    pub kind: String,
    pub count: usize,
    #[serde(rename = "totalImpact")]
    pub total_impact: f64,
    #[serde(rename = "avgImpact")]
    pub avg_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/", rename_all = "camelCase")]
pub struct PoiSummaryResponseContract {
    #[serde(rename = "chainHead")]
    pub chain_head: String,
    #[serde(rename = "totalEntries")]
    pub total_entries: usize,
    #[serde(rename = "totalImpact")]
    pub total_impact: f64,
    #[serde(rename = "avgImpact")]
    pub avg_impact: f64,
    #[serde(rename = "maxImpact")]
    pub max_impact: f64,
    #[serde(rename = "byKind")]
    pub by_kind: Vec<PoiPerKindContract>,
}

// ════════════════════════════════════════════════════════════════════
// Principal activation group (G2)
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/", rename_all = "camelCase")]
pub struct ActivatePrincipalResponseContract {
    #[serde(rename = "missionId")]
    pub mission_id: String,
    #[serde(rename = "missionReceiptId")]
    pub mission_receipt_id: String,
    #[serde(rename = "principalActivationReceiptId")]
    pub principal_activation_receipt_id: String,
    #[serde(rename = "principalId")]
    pub principal_id: String,
    #[serde(rename = "profileHash")]
    pub profile_hash: String,
    #[serde(rename = "chainHead")]
    pub chain_head: String,
    #[serde(rename = "finalStage")]
    pub final_stage: MissionStageName,
    pub admissibility: AdmissibilityContract,
    // ts-rs does NOT automatically honor `#[serde(rename)]`; without the
    // twin `#[ts(...)]` directives the generated .ts would emit these
    // fields as snake_case required strings while the actual JSON wire
    // is camelCase and may be omitted. `optional` marks them `?: T`
    // so TS consumers correctly handle the omission.
    #[serde(rename = "cacheWarning", skip_serializing_if = "Option::is_none")]
    #[ts(rename = "cacheWarning", optional)]
    pub cache_warning: Option<String>,
    #[serde(rename = "effectiveCacheDir", skip_serializing_if = "Option::is_none")]
    #[ts(rename = "effectiveCacheDir", optional)]
    pub effective_cache_dir: Option<String>,
}

// ════════════════════════════════════════════════════════════════════
// Error group
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/")]
pub struct ErrorBodyContract {
    pub code: String,
    pub message: String,
    pub domain: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub admissibility: Option<AdmissibilityContract>,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../bindings/")]
pub struct ErrorResponseContract {
    pub error: ErrorBodyContract,
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

    #[test]
    fn invariant_scorer_ids_match_frozen_canon() {
        // Drift guard: the gateway currently emits these literal strings.
        // If either side moves, this test fails and the UI is not
        // silently shown a wrong label.
        assert_eq!(InvariantName::ZannZero.scorer_id(), "ZANN_ZERO");
        assert_eq!(InvariantName::ClaimMustBind.scorer_id(), "CLAIM_MUST_BIND");
        assert_eq!(InvariantName::RibaZero.scorer_id(), "RIBA_ZERO");
        assert_eq!(InvariantName::NoShadowState.scorer_id(), "NO_SHADOW_STATE");
        assert_eq!(InvariantName::IhsanFloor.scorer_id(), "IHSAN_FLOOR");
    }

    #[test]
    fn verdict_name_serializes_as_camel_case_variant() {
        let v = VerdictName::Permit;
        let j = serde_json::to_string(&v).unwrap();
        assert_eq!(j, "\"Permit\"");
    }

    #[test]
    fn gate_verdict_contract_serializes_with_camel_case_scorer_id() {
        let g = GateVerdictContract {
            scorer_id: "IHSAN_FLOOR".into(),
            invariant: Some("IhsanFloor".into()),
            verdict: VerdictName::Permit,
            reason: "score 0.98 >= floor 0.95".into(),
            score: Some(0.98),
        };
        let j: serde_json::Value = serde_json::to_value(&g).unwrap();
        assert_eq!(j["scorerId"], "IHSAN_FLOOR");
        assert_eq!(j["verdict"], "Permit");
        assert_eq!(j["score"], 0.98);
    }

    #[test]
    fn admissibility_contract_round_trip() {
        let a = AdmissibilityContract {
            verdict: VerdictName::Reject,
            gate_verdicts: vec![GateVerdictContract {
                scorer_id: "IHSAN_FLOOR".into(),
                invariant: Some("IhsanFloor".into()),
                verdict: VerdictName::Reject,
                reason: "score 0.40 < floor 0.95".into(),
                score: Some(0.40),
            }],
            rejected: Some(RejectedClaimContract {
                invariant: "IHSAN_FLOOR".into(),
                reason: "below floor".into(),
                remediation_path: "raise quality_score to >= 0.95".into(),
                escalation_allowed: true,
            }),
        };
        let j = serde_json::to_string(&a).unwrap();
        let back: AdmissibilityContract = serde_json::from_str(&j).unwrap();
        assert_eq!(back.verdict, VerdictName::Reject);
        assert_eq!(back.gate_verdicts.len(), 1);
        assert!(back.rejected.is_some());
    }

    #[test]
    fn poi_entry_contract_matches_gateway_dto_shape() {
        // Any future rename of these field names will break the UI.
        // This test locks the camelCase wire shape.
        let e = PoiEntryContract {
            receipt_id: "deadbeef".into(),
            receipt_kind_byte: 0x70,
            receipt_kind_name: "MissionExecuted".into(),
            quality_score: 0.98,
            gate_min_score: 0.97,
            entry_count: 3,
            impact_score: 0.9743,
            timestamp_ns: 1_700_000_000_000_000_000,
            principal_id: Some("cafebabe".into()),
        };
        let v = serde_json::to_value(&e).unwrap();
        assert!(v.get("receiptId").is_some());
        assert!(v.get("receiptKindByte").is_some());
        assert!(v.get("receiptKindName").is_some());
        assert!(v.get("qualityScore").is_some());
        assert!(v.get("gateMinScore").is_some());
        assert!(v.get("entryCount").is_some());
        assert!(v.get("impactScore").is_some());
        assert!(v.get("timestampNs").is_some());
        assert!(v.get("principalId").is_some());
    }

    #[test]
    fn urp_view_contract_camel_case_counts() {
        let u = UrpViewContract {
            total_count: 3,
            allowlisted_count: 2,
            buckets: vec![],
        };
        let v = serde_json::to_value(&u).unwrap();
        assert_eq!(v["totalCount"], 3);
        assert_eq!(v["allowlistedCount"], 2);
    }

    #[test]
    fn organize_response_contract_has_all_required_camel_case_fields() {
        let o = OrganizeResponseContract {
            mission_id: "a".into(),
            mission_receipt_id: "b".into(),
            organize_receipt_id: "c".into(),
            chain_head: "d".into(),
            path: "/tmp/target".into(),
            listing_digest: "e".into(),
            file_count: 2,
            dir_count: 1,
            entry_count: 3,
            entries: vec![OrganizeEntryContract {
                name: "alpha.txt".into(),
                kind: "file".into(),
            }],
            timestamp_ns: 123,
            admissibility: AdmissibilityContract {
                verdict: VerdictName::Permit,
                gate_verdicts: vec![],
                rejected: None,
            },
        };
        let v = serde_json::to_value(&o).unwrap();
        for field in [
            "missionId",
            "missionReceiptId",
            "organizeReceiptId",
            "chainHead",
            "path",
            "listingDigest",
            "fileCount",
            "dirCount",
            "entryCount",
            "entries",
            "timestampNs",
            "admissibility",
        ] {
            assert!(
                v.get(field).is_some(),
                "OrganizeResponseContract missing field '{}'",
                field
            );
        }
    }

    /// Drift-guard: generated TS bindings must emit camelCase field names
    /// (via `#[ts(rename_all = "camelCase")]` at struct level + explicit
    /// `#[ts(rename = "…")]` on optional fields) that match the serde
    /// wire shape. If ts-rs starts emitting snake_case or a dropped
    /// rename_all attribute is reintroduced, TS consumers silently read
    /// `undefined` from the wire. This test locks the invariant.
    #[test]
    fn ts_bindings_emit_wire_compatible_camel_case_keys() {
        // Binding files are generated by the other #[test]s in this module
        // via `#[ts(export, export_to = "../bindings/")]`. Test order is
        // not guaranteed, so we instantiate the struct here which triggers
        // the ts-rs export side effect synchronously for this check.
        let _ = ActivatePrincipalResponseContract {
            mission_id: "x".into(),
            mission_receipt_id: "x".into(),
            principal_activation_receipt_id: "x".into(),
            principal_id: "x".into(),
            profile_hash: "x".into(),
            chain_head: "x".into(),
            final_stage: MissionStageName::Replayability,
            admissibility: AdmissibilityContract {
                verdict: VerdictName::Permit,
                gate_verdicts: vec![],
                rejected: None,
            },
            cache_warning: None,
            effective_cache_dir: None,
        };

        // Cross-crate path: tests run from the crate root (CARGO_MANIFEST_DIR).
        let binding_file = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("bindings")
            .join("ActivatePrincipalResponseContract.ts");
        let binding_src = std::fs::read_to_string(&binding_file).unwrap_or_else(|e| {
            panic!(
                "could not read generated binding {}: {} — did ts-rs run? (cargo test regenerates)",
                binding_file.display(),
                e
            )
        });

        // Camel-case wire keys that MUST appear (serde JSON-level).
        for key in [
            "missionId: string",
            "missionReceiptId: string",
            "principalActivationReceiptId: string",
            "principalId: string",
            "profileHash: string",
            "chainHead: string",
            "finalStage: MissionStageName",
            "admissibility: AdmissibilityContract",
        ] {
            assert!(
                binding_src.contains(key),
                "binding missing required camelCase key '{}'.\n\nFull binding:\n{}",
                key,
                binding_src
            );
        }

        // Skip-serializing-if Option fields MUST be marked optional (`?:`) with
        // the camelCase name. Both skip-option fields on this contract must
        // assert — if either regresses to `T | null` (required + nullable) or
        // to snake_case, TS consumers silently read `undefined` at runtime.
        assert!(
            binding_src.contains("cacheWarning?:"),
            "binding must mark cacheWarning as optional (cacheWarning?: …). Found:\n{}",
            binding_src
        );
        assert!(
            binding_src.contains("effectiveCacheDir?:"),
            "binding must mark effectiveCacheDir as optional (effectiveCacheDir?: …). Found:\n{}",
            binding_src
        );

        // Negative guard: the old snake_case drift must NOT reappear for any
        // field that has a #[serde(rename = "camelCase")] on the Rust side.
        for bad_key in [
            "cache_warning:",
            "effective_cache_dir:",
            "mission_id:",
            "chain_head:",
        ] {
            assert!(
                !binding_src.contains(bad_key),
                "binding regressed to snake_case — found forbidden key '{}'.\n\nFull:\n{}",
                bad_key,
                binding_src
            );
        }
    }
}
