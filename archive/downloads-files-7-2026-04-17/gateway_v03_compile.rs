//! Gateway v0.3 — TrustCompiler integration
//!
//! بسم الله الرحمن الرحيم
//!
//! This file contains the NEW routes that use TrustCompiler::compile()
//! instead of raw CognitionRuntime::submit_mission(). The existing
//! /mission route is preserved for backward compatibility.
//!
//! New routes:
//!   POST /compile    — universal trust compilation endpoint
//!   POST /organize   — convenience wrapper for filesystem organization
//!
//! Integration path for Claude Code:
//!   1. Add these routes to the existing main.rs router
//!   2. Replace the Arc<RwLock<CognitionRuntime>> with Arc<RwLock<TrustCompiler>>
//!      (TrustCompiler owns the chain; runtime methods migrate gradually)
//!   3. Keep /mission route working via a shim that calls compile() internally

use axum::{
    extract::State,
    http::StatusCode,
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use bizra_cognition::trust_compiler::{
    TrustCompiler, CompilationRequest, CompilationKind,
    CompilationReceipt, FilesystemExecutor, IdentityExecutor,
};
use bizra_cognition::canonical_hasher::blake3_domain;

// ============================================================================
// DTOs
// ============================================================================

#[derive(Deserialize)]
pub struct CompileRequest {
    /// What the operator wants to do
    pub intent: String,
    /// Operation type: "mission", "filesystem", "tool", "lifecycle"
    pub kind: String,
    /// Quality score (0.0 - 1.0). Must be ≥ 0.95
    #[serde(default = "default_quality")]
    pub quality_score: f64,
    /// For filesystem ops: target directory path
    pub target_path: Option<String>,
}

fn default_quality() -> f64 { 0.98 }

#[derive(Serialize)]
pub struct CompileResponse {
    /// Whether the compilation was rejected
    pub rejected: bool,
    /// Rejection reason (if rejected)
    pub rejection_reason: Option<String>,
    /// Remediation path (if rejected)
    pub remediation: Option<String>,
    /// Receipt ID (if permitted)
    pub receipt_id: Option<String>,
    /// Number of sub-operations performed
    pub sub_operations: usize,
    /// Sub-operation summaries
    pub operations: Vec<OperationSummary>,
    /// Admissibility verdict
    pub verdict: String,
    /// Gate count that passed
    pub gates_passed: usize,
    /// Final stage reached
    pub stage: String,
    /// Chain head after compilation
    pub chain_head: String,
    /// Chain length after compilation
    pub chain_length: usize,
}

#[derive(Serialize)]
pub struct OperationSummary {
    pub tool: String,
    pub input_hash: String,
    pub output_hash: String,
}

#[derive(Serialize)]
pub struct CompileErrorResponse {
    pub error: CompileErrorBody,
}

#[derive(Serialize)]
pub struct CompileErrorBody {
    pub code: &'static str,
    pub message: String,
    pub domain: &'static str,
}

const DOMAIN: &str = "bizra-trust-compiler-v1";

// ============================================================================
// Route handler
// ============================================================================

pub async fn handle_compile(
    State(state): State<Arc<RwLock<TrustCompiler>>>,
    Json(req): Json<CompileRequest>,
) -> Result<Json<CompileResponse>, (StatusCode, Json<CompileErrorResponse>)> {

    let kind = match req.kind.as_str() {
        "mission" | "lifecycle" => CompilationKind::Mission,
        "filesystem" => CompilationKind::FilesystemOp,
        "tool" => CompilationKind::ToolExecution,
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(CompileErrorResponse {
                    error: CompileErrorBody {
                        code: "UNKNOWN_KIND",
                        message: format!("unknown compilation kind: {}", req.kind),
                        domain: DOMAIN,
                    },
                }),
            ));
        }
    };

    // Build the compilation request
    let evidence = blake3_domain("bizra-compile-evidence-v1", req.intent.as_bytes());
    let request_id = blake3_domain("bizra-compile-request-v1", &evidence);
    let ts_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let compilation_request = CompilationRequest {
        request_id,
        kind,
        intent: req.intent.clone(),
        quality_score: req.quality_score,
        evidence_hash: evidence,
        parent_id: None,
        timestamp_ns: ts_ns,
    };

    // Select executor based on kind
    let mut compiler = state.write().await;

    let result = match kind {
        CompilationKind::FilesystemOp => {
            let target = req.target_path.as_deref().unwrap_or(".");
            let executor = FilesystemExecutor::new(target.into());
            compiler.compile(compilation_request, &executor)
        }
        _ => {
            let executor = IdentityExecutor;
            compiler.compile(compilation_request, &executor)
        }
    };

    match result {
        Ok(receipt) => {
            let chain_head = hex::encode(compiler.chain().head());
            let chain_length = compiler.chain().len();

            if receipt.rejected {
                // Structured rejection — HTTP 422
                let rejection_reason = receipt.admissibility.rejected
                    .as_ref()
                    .map(|r| format!("{}: {}", r.invariant, r.reason));
                let remediation = receipt.admissibility.rejected
                    .as_ref()
                    .map(|r| r.remediation_path.clone());

                Ok(Json(CompileResponse {
                    rejected: true,
                    rejection_reason,
                    remediation,
                    receipt_id: None,
                    sub_operations: 0,
                    operations: Vec::new(),
                    verdict: "Reject".to_string(),
                    gates_passed: receipt.admissibility.gate_verdicts
                        .iter()
                        .filter(|g| g.verdict == bizra_cognition::admissibility_freeze_v1::Verdict::Permit)
                        .count(),
                    stage: "Admissibility".to_string(),
                    chain_head,
                    chain_length,
                }))
            } else {
                Ok(Json(CompileResponse {
                    rejected: false,
                    rejection_reason: None,
                    remediation: None,
                    receipt_id: receipt.receipt_id.map(hex::encode),
                    sub_operations: receipt.sub_receipts.len(),
                    operations: receipt.sub_receipts.iter().map(|s| OperationSummary {
                        tool: s.tool_name.clone(),
                        input_hash: hex::encode(&s.input_hash[..16]),
                        output_hash: hex::encode(&s.output_hash[..16]),
                    }).collect(),
                    verdict: "Permit".to_string(),
                    gates_passed: 5,
                    stage: "Canonicalization".to_string(),
                    chain_head,
                    chain_length,
                }))
            }
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(CompileErrorResponse {
                error: CompileErrorBody {
                    code: "COMPILATION_ERROR",
                    message: format!("{:?}", e),
                    domain: DOMAIN,
                },
            }),
        )),
    }
}

// ============================================================================
// Convenience: POST /organize
// ============================================================================

#[derive(Deserialize)]
pub struct OrganizeRequest {
    /// Directory to organize (absolute path)
    pub path: String,
    /// Quality score (default 0.98)
    #[serde(default = "default_quality")]
    pub quality_score: f64,
}

pub async fn handle_organize(
    State(state): State<Arc<RwLock<TrustCompiler>>>,
    Json(req): Json<OrganizeRequest>,
) -> Result<Json<CompileResponse>, (StatusCode, Json<CompileErrorResponse>)> {
    let compile_req = CompileRequest {
        intent: format!("organize directory: {}", req.path),
        kind: "filesystem".to_string(),
        quality_score: req.quality_score,
        target_path: Some(req.path),
    };
    handle_compile(State(state), Json(compile_req)).await
}
