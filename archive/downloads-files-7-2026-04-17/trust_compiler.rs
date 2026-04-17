//! BIZRA Admissibility-First Trust Compilation
//!
//! بسم الله الرحمن الرحيم
//!
//! File: bizra-omega/bizra-cognition/src/trust_compiler.rs
//! Authority: Manifest v0.2 §3 (Five Invariants), §6 (Lawful Loop)
//! Cycle: 6 — first impact receipt
//!
//! The Trust Compiler is the single entry point for ALL state-mutating
//! operations in BIZRA. Nothing bypasses admissibility. The pattern is:
//!
//!   1. Caller constructs a CompilationRequest (intent + evidence + quality)
//!   2. Trust compiler evaluates admissibility (5 gates)
//!   3. If REJECT → structured rejection returned, NO state mutation
//!   4. If PERMIT → executor runs, each sub-action receipted
//!   5. Final receipt binds all sub-receipts to parent
//!   6. Chain advanced atomically
//!
//! This module generalizes submit_mission()'s pattern into a reusable
//! compilation pipeline that works for missions, tool calls, agent
//! delegations, and any future action type.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::receipts::{
    ReceiptChain, ReceiptPayload, ReceiptKind,
    ChainError, Blake3Hash,
};
use crate::receipt_freeze_v1::ReceiptArtifact;
use crate::admissibility_freeze_v1::{
    AdmissibilityChain, AdmissibilityClaim, AdmissibilityResult,
    GateVerdict, Verdict,
};
use crate::canonical_hasher::blake3_domain;

// ============================================================================
// Core types
// ============================================================================

/// The universal input to the trust compiler. Any operation that wants
/// to mutate state must be expressed as a CompilationRequest.
#[derive(Debug, Clone)]
pub struct CompilationRequest {
    /// Unique identifier for this request (BLAKE3 of intent + context)
    pub request_id: Blake3Hash,
    /// What kind of operation this is
    pub kind: CompilationKind,
    /// Human-readable intent description
    pub intent: String,
    /// Quality score asserted by the caller (0.0 - 1.0)
    /// Must be ≥ IHSAN_FLOOR (0.95) to pass gate 1
    pub quality_score: f64,
    /// Evidence hash binding the request to proof
    /// Must be non-zero to pass ZANN_ZERO and CLAIM_MUST_BIND
    pub evidence_hash: Blake3Hash,
    /// Optional parent request (for sub-operations)
    pub parent_id: Option<Blake3Hash>,
    /// Timestamp of request creation (nanoseconds)
    pub timestamp_ns: u64,
}

/// The kind of operation being compiled. Each kind follows the same
/// admissibility path but may have different execution semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationKind {
    /// Full mission (intent → state gap → execution → receipt)
    Mission,
    /// Single tool invocation within a mission
    ToolExecution,
    /// Agent-to-agent delegation (future: A2A)
    AgentDelegation,
    /// Filesystem operation (Cycle-6 primary)
    FilesystemOp,
    /// LLM inference call (Cycle-7)
    LlmInference,
    /// Node lifecycle event (activation, shutdown)
    NodeLifecycle,
}

impl CompilationKind {
    /// Map to the ReceiptKind that this compilation produces
    pub fn receipt_kind(&self) -> ReceiptKind {
        match self {
            CompilationKind::Mission => ReceiptKind::NodeLifecycle,
            CompilationKind::ToolExecution => ReceiptKind::GovernanceDecision, // TODO: ReceiptKind::ToolExecution = 0x80
            CompilationKind::AgentDelegation => ReceiptKind::GovernanceDecision,
            CompilationKind::FilesystemOp => ReceiptKind::GovernanceDecision, // TODO: ReceiptKind::FilesystemOp = 0x81
            CompilationKind::LlmInference => ReceiptKind::GovernanceDecision, // TODO: ReceiptKind::LlmInference = 0x82
            CompilationKind::NodeLifecycle => ReceiptKind::NodeLifecycle,
        }
    }
}

/// The output of a successful trust compilation.
#[derive(Debug, Clone)]
pub struct CompilationReceipt {
    /// The request that was compiled
    pub request: CompilationRequest,
    /// Admissibility evaluation result (always present, even on PERMIT)
    pub admissibility: AdmissibilityResult,
    /// Whether the request was rejected
    pub rejected: bool,
    /// Gate verdict hashes (empty on reject)
    pub gate_receipt_hashes: Vec<Blake3Hash>,
    /// Sub-operation receipts (e.g., per-file-move receipts)
    pub sub_receipts: Vec<SubReceipt>,
    /// Final receipt hash (None on reject)
    pub receipt_id: Option<Blake3Hash>,
    /// Final receipt artifact (None on reject)
    pub final_receipt: Option<ReceiptArtifact>,
}

/// A sub-operation receipt — one atomic action within a compiled operation.
#[derive(Debug, Clone)]
pub struct SubReceipt {
    /// What tool/action produced this
    pub tool_name: String,
    /// BLAKE3 of the input parameters
    pub input_hash: Blake3Hash,
    /// BLAKE3 of the output/result
    pub output_hash: Blake3Hash,
    /// Parent compilation request ID
    pub parent_id: Blake3Hash,
    /// Chain hash after this sub-receipt was appended
    pub chain_hash: Blake3Hash,
    /// Timestamp
    pub timestamp_ns: u64,
}

impl ReceiptPayload for SubReceipt {
    fn kind(&self) -> ReceiptKind {
        ReceiptKind::GovernanceDecision // TODO: ReceiptKind::ToolExecution
    }

    fn timestamp_ns(&self) -> u64 {
        self.timestamp_ns
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(192);
        buf.extend_from_slice(self.tool_name.as_bytes());
        buf.push(0x00); // separator
        buf.extend_from_slice(&self.input_hash);
        buf.extend_from_slice(&self.output_hash);
        buf.extend_from_slice(&self.parent_id);
        buf.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        buf
    }

    fn hash(&self) -> Blake3Hash {
        blake3_domain("bizra-sub-receipt-v1", &self.canonical_bytes())
    }
}

// ============================================================================
// Compilation errors
// ============================================================================

#[derive(Debug)]
pub enum CompilationError {
    /// Chain operation failed
    Chain(ChainError),
    /// Clock failure
    Clock(String),
    /// Executor returned an error
    ExecutionFailed(String),
}

impl From<ChainError> for CompilationError {
    fn from(e: ChainError) -> Self { CompilationError::Chain(e) }
}

// ============================================================================
// The executor trait — what happens AFTER admissibility permits
// ============================================================================

/// Implementors define what actually happens when a request is permitted.
/// The trust compiler calls execute() only after all 5 gates return PERMIT.
///
/// The executor returns a Vec<SubReceipt> — one for each atomic action taken.
/// Each sub-receipt is independently verifiable and bound to the parent request.
pub trait Executor {
    /// Execute the permitted request and return sub-receipts.
    ///
    /// Contract:
    /// - MUST NOT mutate the receipt chain (the compiler does that)
    /// - MUST return one SubReceipt per atomic action
    /// - MUST hash inputs and outputs for each sub-receipt
    /// - MAY fail — failure returns Err, no sub-receipts are chained
    fn execute(
        &self,
        request: &CompilationRequest,
        timestamp_ns: u64,
    ) -> Result<Vec<SubReceipt>, String>;
}

/// A no-op executor for operations where submission IS the action
/// (e.g., principal activation, node lifecycle events).
pub struct IdentityExecutor;

impl Executor for IdentityExecutor {
    fn execute(
        &self,
        request: &CompilationRequest,
        timestamp_ns: u64,
    ) -> Result<Vec<SubReceipt>, String> {
        // No sub-operations — the mission itself is the action
        Ok(vec![SubReceipt {
            tool_name: "identity".to_string(),
            input_hash: request.evidence_hash,
            output_hash: request.request_id,
            parent_id: request.request_id,
            chain_hash: [0u8; 32], // filled by compiler after chain append
            timestamp_ns,
        }])
    }
}

// ============================================================================
// The trust compiler
// ============================================================================

/// The Trust Compiler — admissibility-first state mutation.
///
/// Usage:
/// ```ignore
/// let compiler = TrustCompiler::new(chain);
/// let receipt = compiler.compile(request, &executor)?;
/// if receipt.rejected {
///     // show rejection reason + remediation path
/// } else {
///     // receipt.receipt_id is the canonical proof
/// }
/// ```
pub struct TrustCompiler {
    chain: ReceiptChain,
    admissibility: AdmissibilityChain,
}

impl TrustCompiler {
    /// Create a new trust compiler with a receipt chain and canonical
    /// admissibility gates.
    pub fn new(chain: ReceiptChain) -> Self {
        Self {
            chain,
            admissibility: AdmissibilityChain::canonical(),
        }
    }

    /// Access the underlying chain (read-only projection for gateway)
    pub fn chain(&self) -> &ReceiptChain {
        &self.chain
    }

    /// Compile a request through the admissibility-first pipeline.
    ///
    /// This is the SINGLE function that ALL state mutations flow through.
    /// No bypass. No side channel. No shortcut.
    ///
    /// Pipeline:
    ///   1. Build AdmissibilityClaim from request
    ///   2. Evaluate against 5 constitutional gates
    ///   3. If REJECT: return structured rejection (chain UNCHANGED)
    ///   4. If PERMIT: run executor
    ///   5. Append each sub-receipt to chain
    ///   6. Append gate verdicts to chain
    ///   7. Mint final ReceiptArtifact binding everything
    ///   8. Append final receipt to chain
    ///   9. Return CompilationReceipt with all hashes
    pub fn compile<E: Executor>(
        &mut self,
        request: CompilationRequest,
        executor: &E,
    ) -> Result<CompilationReceipt, CompilationError> {
        let timestamp_ns = Self::now_ns()?;

        // ── Step 1: Build claim ──────────────────────────────────────
        let claim = AdmissibilityClaim {
            claim_id: request.request_id,
            has_evidence: request.evidence_hash != [0u8; 32],
            evidence_hash: Some(request.evidence_hash),
            economic_pattern: None,
            state_mutation: None,
            quality_score: request.quality_score,
            timestamp_ns,
        };

        // ── Step 2: Evaluate admissibility ───────────────────────────
        let admissibility = self.admissibility.evaluate(&claim);

        // ── Step 3: REJECT path ──────────────────────────────────────
        // Chain stays CLEAN. §10: "chain reflects what actually happened
        // by ABSENCE, not by presence of a rejection receipt."
        if admissibility.verdict != Verdict::Permit {
            return Ok(CompilationReceipt {
                request,
                admissibility,
                rejected: true,
                gate_receipt_hashes: Vec::new(),
                sub_receipts: Vec::new(),
                receipt_id: None,
                final_receipt: None,
            });
        }

        // ── Step 4: Execute (ONLY after PERMIT) ──────────────────────
        let mut sub_receipts = executor.execute(&request, timestamp_ns)
            .map_err(CompilationError::ExecutionFailed)?;

        // ── Step 5: Append sub-receipts to chain ─────────────────────
        for sub in &mut sub_receipts {
            let hash = self.chain.append_with_payload(sub.clone())?;
            sub.chain_hash = hash;
        }

        // ── Step 6: Append gate verdicts to chain ────────────────────
        let mut gate_receipt_hashes = Vec::with_capacity(
            admissibility.gate_verdicts.len()
        );
        for verdict in &admissibility.gate_verdicts {
            let hash = self.chain.append_with_payload(verdict.clone())?;
            gate_receipt_hashes.push(hash);
        }

        // ── Step 7: Mint final receipt ───────────────────────────────
        // Evidence chain: sub-receipt hashes + gate verdict hashes
        let mut evidence_chain = Vec::new();
        for sub in &sub_receipts {
            evidence_chain.push(sub.chain_hash);
        }
        for gh in &gate_receipt_hashes {
            evidence_chain.push(*gh);
        }

        let final_receipt = ReceiptArtifact::new(
            request.kind.receipt_kind(),
            request.request_id, // claim_ref
            request.evidence_hash,
            evidence_chain,
            self.chain.head(),
            Self::now_ns()?,
        );
        let receipt_id = final_receipt.receipt_id;

        // ── Step 8: Append final receipt ─────────────────────────────
        self.chain.append_artifact(final_receipt.clone())?;

        // ── Step 9: Return compilation receipt ───────────────────────
        Ok(CompilationReceipt {
            request,
            admissibility,
            rejected: false,
            gate_receipt_hashes,
            sub_receipts,
            receipt_id: Some(receipt_id),
            final_receipt: Some(final_receipt),
        })
    }

    fn now_ns() -> Result<u64, CompilationError> {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .map_err(|e| CompilationError::Clock(e.to_string()))
    }
}

// ============================================================================
// Filesystem executor (Cycle-6: organize Downloads)
// ============================================================================

use std::path::{Path, PathBuf};
use std::fs;

/// A single file move operation with before/after hash verification.
#[derive(Debug, Clone)]
pub struct FileMoveOp {
    pub source: PathBuf,
    pub destination: PathBuf,
    pub file_hash: Blake3Hash,
    pub category: String,
    pub size_bytes: u64,
}

/// Categories files by extension and moves them into subdirectories.
/// Each move is independently receipted and hash-verified.
pub struct FilesystemExecutor {
    pub target_dir: PathBuf,
}

impl FilesystemExecutor {
    pub fn new(target_dir: PathBuf) -> Self {
        Self { target_dir }
    }

    /// Categorize a file by extension using the 14-category classifier.
    fn categorize(ext: &str) -> &'static str {
        match ext.to_lowercase().as_str() {
            // Documents
            "pdf" | "doc" | "docx" | "txt" | "md" | "rtf" | "odt" => "documents",
            // Spreadsheets
            "xls" | "xlsx" | "csv" | "tsv" | "ods" => "spreadsheets",
            // Presentations
            "ppt" | "pptx" | "key" | "odp" => "presentations",
            // Images
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "svg" | "webp" | "ico" => "images",
            // Videos
            "mp4" | "avi" | "mkv" | "mov" | "wmv" | "flv" | "webm" => "videos",
            // Audio
            "mp3" | "wav" | "flac" | "aac" | "ogg" | "wma" | "m4a" => "audio",
            // Archives
            "zip" | "tar" | "gz" | "bz2" | "7z" | "rar" | "xz" => "archives",
            // Code
            "rs" | "py" | "ts" | "tsx" | "js" | "jsx" | "c" | "cpp" | "h"
            | "java" | "go" | "rb" | "swift" | "kt" => "code",
            // Config
            "toml" | "yaml" | "yml" | "json" | "xml" | "ini" | "env" => "config",
            // Data
            "sql" | "db" | "sqlite" | "parquet" | "feather" => "data",
            // Executables
            "exe" | "msi" | "deb" | "rpm" | "appimage" | "dmg" => "executables",
            // Fonts
            "ttf" | "otf" | "woff" | "woff2" => "fonts",
            // Models (AI/3D)
            "onnx" | "pt" | "safetensors" | "gguf" | "obj" | "stl" | "fbx" => "models",
            // Everything else
            _ => "other",
        }
    }

    /// Hash file contents using BLAKE3 (for verification before and after move)
    fn hash_file(path: &Path) -> Result<Blake3Hash, String> {
        let bytes = fs::read(path)
            .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
        Ok(blake3_domain("bizra-file-content-v1", &bytes))
    }
}

impl Executor for FilesystemExecutor {
    fn execute(
        &self,
        request: &CompilationRequest,
        timestamp_ns: u64,
    ) -> Result<Vec<SubReceipt>, String> {
        let dir = &self.target_dir;
        if !dir.exists() {
            return Err(format!("target directory does not exist: {}", dir.display()));
        }

        let entries: Vec<_> = fs::read_dir(dir)
            .map_err(|e| format!("cannot read directory: {}", e))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .collect();

        if entries.is_empty() {
            return Ok(Vec::new()); // nothing to organize — honest empty
        }

        let mut sub_receipts = Vec::with_capacity(entries.len());

        for entry in &entries {
            let source = entry.path();
            let ext = source.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            let category = Self::categorize(ext);

            // Create category directory
            let category_dir = dir.join(category);
            if !category_dir.exists() {
                fs::create_dir_all(&category_dir)
                    .map_err(|e| format!("cannot create {}: {}", category_dir.display(), e))?;
            }

            let filename = source.file_name()
                .ok_or_else(|| "file has no name".to_string())?;
            let destination = category_dir.join(filename);

            // Hash BEFORE move (CLAIM_MUST_BIND: evidence of original state)
            let file_hash = Self::hash_file(&source)?;

            // Compute input hash (source path + file hash)
            let mut input_buf = Vec::new();
            input_buf.extend_from_slice(source.to_string_lossy().as_bytes());
            input_buf.extend_from_slice(&file_hash);
            let input_hash = blake3_domain("bizra-fs-input-v1", &input_buf);

            // Execute the move
            fs::rename(&source, &destination)
                .map_err(|e| format!("move failed {} → {}: {}",
                    source.display(), destination.display(), e))?;

            // Hash AFTER move (verify no corruption)
            let post_hash = Self::hash_file(&destination)?;
            if post_hash != file_hash {
                // CONSTITUTIONAL HALT: file corrupted during move
                // Attempt to move back
                let _ = fs::rename(&destination, &source);
                return Err(format!(
                    "CORRUPTION DETECTED: {} hash changed during move (pre={:x?} post={:x?})",
                    filename.to_string_lossy(),
                    &file_hash[..8], &post_hash[..8]
                ));
            }

            // Compute output hash (destination path + verified hash)
            let mut output_buf = Vec::new();
            output_buf.extend_from_slice(destination.to_string_lossy().as_bytes());
            output_buf.extend_from_slice(&post_hash);
            let output_hash = blake3_domain("bizra-fs-output-v1", &output_buf);

            sub_receipts.push(SubReceipt {
                tool_name: format!("filesystem.move_file:{}", category),
                input_hash,
                output_hash,
                parent_id: request.request_id,
                chain_hash: [0u8; 32], // filled by compiler
                timestamp_ns,
            });
        }

        Ok(sub_receipts)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receipts::ReceiptChain;
    use std::io::Write;
    use tempfile::TempDir;

    fn test_request(quality: f64) -> CompilationRequest {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let intent = "test compilation request";
        let evidence = blake3_domain("test-evidence", intent.as_bytes());
        let request_id = blake3_domain("test-request", &evidence);

        CompilationRequest {
            request_id,
            kind: CompilationKind::Mission,
            intent: intent.to_string(),
            quality_score: quality,
            evidence_hash: evidence,
            parent_id: None,
            timestamp_ns: ts,
        }
    }

    // ── Test 1: PERMIT path produces receipt ─────────────────────────

    #[test]
    fn compile_permit_produces_receipt_and_advances_chain() {
        let chain = ReceiptChain::new();
        let mut compiler = TrustCompiler::new(chain);
        let request = test_request(0.98);
        let executor = IdentityExecutor;

        let result = compiler.compile(request, &executor).unwrap();

        assert!(!result.rejected);
        assert!(result.receipt_id.is_some());
        assert!(result.final_receipt.is_some());
        assert_eq!(result.gate_receipt_hashes.len(), 5);
        assert_eq!(result.sub_receipts.len(), 1); // identity executor
        // Chain: 1 sub-receipt + 5 gates + 1 final = 7
        assert_eq!(compiler.chain().len(), 7);
    }

    // ── Test 2: REJECT path leaves chain untouched ───────────────────

    #[test]
    fn compile_reject_leaves_chain_clean() {
        let chain = ReceiptChain::new();
        let mut compiler = TrustCompiler::new(chain);
        let request = test_request(0.50); // below IHSAN_FLOOR
        let executor = IdentityExecutor;

        let pre_len = compiler.chain().len();
        let result = compiler.compile(request, &executor).unwrap();

        assert!(result.rejected);
        assert!(result.receipt_id.is_none());
        assert!(result.final_receipt.is_none());
        assert_eq!(result.gate_receipt_hashes.len(), 0);
        assert_eq!(result.sub_receipts.len(), 0);
        assert_eq!(compiler.chain().len(), pre_len); // UNCHANGED
    }

    // ── Test 3: Zero evidence hash triggers ZANN_ZERO rejection ──────

    #[test]
    fn compile_rejects_zero_evidence() {
        let chain = ReceiptChain::new();
        let mut compiler = TrustCompiler::new(chain);
        let mut request = test_request(0.98);
        request.evidence_hash = [0u8; 32]; // no evidence
        let executor = IdentityExecutor;

        let result = compiler.compile(request, &executor).unwrap();
        assert!(result.rejected);
    }

    // ── Test 4: Filesystem executor organizes files correctly ─────────

    #[test]
    fn filesystem_executor_categorizes_and_moves() {
        let dir = TempDir::new().unwrap();
        let dir_path = dir.path();

        // Create test files
        let files = vec![
            ("report.pdf", b"pdf content" as &[u8]),
            ("photo.jpg", b"jpeg content"),
            ("main.rs", b"fn main() {}"),
            ("data.csv", b"a,b,c"),
            ("song.mp3", b"audio bytes"),
        ];

        for (name, content) in &files {
            let mut f = fs::File::create(dir_path.join(name)).unwrap();
            f.write_all(content).unwrap();
        }

        let executor = FilesystemExecutor::new(dir_path.to_path_buf());
        let request = test_request(0.98);
        let ts = request.timestamp_ns;

        let sub_receipts = executor.execute(&request, ts).unwrap();

        assert_eq!(sub_receipts.len(), 5);

        // Verify files moved to correct categories
        assert!(dir_path.join("documents/report.pdf").exists());
        assert!(dir_path.join("images/photo.jpg").exists());
        assert!(dir_path.join("code/main.rs").exists());
        assert!(dir_path.join("spreadsheets/data.csv").exists());
        assert!(dir_path.join("audio/song.mp3").exists());

        // Verify originals are gone
        assert!(!dir_path.join("report.pdf").exists());
        assert!(!dir_path.join("photo.jpg").exists());

        // Verify each sub-receipt has non-zero hashes
        for sub in &sub_receipts {
            assert_ne!(sub.input_hash, [0u8; 32]);
            assert_ne!(sub.output_hash, [0u8; 32]);
            assert!(sub.tool_name.starts_with("filesystem.move_file:"));
        }
    }

    // ── Test 5: Full pipeline — filesystem through trust compiler ────

    #[test]
    fn full_pipeline_filesystem_through_trust_compiler() {
        let dir = TempDir::new().unwrap();
        let dir_path = dir.path();

        // Create 3 test files
        fs::write(dir_path.join("notes.txt"), b"hello world").unwrap();
        fs::write(dir_path.join("logo.png"), b"PNG fake header").unwrap();
        fs::write(dir_path.join("backup.zip"), b"PK zip fake").unwrap();

        let chain = ReceiptChain::new();
        let mut compiler = TrustCompiler::new(chain);
        let mut request = test_request(0.98);
        request.kind = CompilationKind::FilesystemOp;
        let executor = FilesystemExecutor::new(dir_path.to_path_buf());

        let result = compiler.compile(request, &executor).unwrap();

        // Not rejected
        assert!(!result.rejected);
        // 3 files = 3 sub-receipts
        assert_eq!(result.sub_receipts.len(), 3);
        // Chain: 3 sub-receipts + 5 gates + 1 final = 9
        assert_eq!(compiler.chain().len(), 9);
        // Files organized
        assert!(dir_path.join("documents/notes.txt").exists());
        assert!(dir_path.join("images/logo.png").exists());
        assert!(dir_path.join("archives/backup.zip").exists());
        // Receipt exists
        assert!(result.receipt_id.is_some());
    }

    // ── Test 6: Low quality filesystem op is rejected ────────────────

    #[test]
    fn filesystem_op_rejected_at_low_quality() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("test.txt"), b"data").unwrap();

        let chain = ReceiptChain::new();
        let mut compiler = TrustCompiler::new(chain);
        let request = test_request(0.50); // below floor
        let executor = FilesystemExecutor::new(dir.path().to_path_buf());

        let result = compiler.compile(request, &executor).unwrap();

        assert!(result.rejected);
        // File NOT moved (execution never happened)
        assert!(dir.path().join("test.txt").exists());
        // Chain unchanged
        assert_eq!(compiler.chain().len(), 0);
    }

    // ── Test 7: Empty directory produces zero sub-receipts ───────────

    #[test]
    fn empty_directory_produces_no_sub_receipts() {
        let dir = TempDir::new().unwrap();

        let chain = ReceiptChain::new();
        let mut compiler = TrustCompiler::new(chain);
        let request = test_request(0.98);
        let executor = FilesystemExecutor::new(dir.path().to_path_buf());

        let result = compiler.compile(request, &executor).unwrap();

        assert!(!result.rejected);
        assert_eq!(result.sub_receipts.len(), 0);
        // Chain: 0 sub-receipts + 5 gates + 1 final = 6
        assert_eq!(compiler.chain().len(), 6);
    }

    // ── Test 8: Sub-receipts are bound to parent ─────────────────────

    #[test]
    fn sub_receipts_bound_to_parent_request() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.rs"), b"code").unwrap();
        fs::write(dir.path().join("b.py"), b"code").unwrap();

        let chain = ReceiptChain::new();
        let mut compiler = TrustCompiler::new(chain);
        let request = test_request(0.98);
        let parent_id = request.request_id;
        let executor = FilesystemExecutor::new(dir.path().to_path_buf());

        let result = compiler.compile(request, &executor).unwrap();

        for sub in &result.sub_receipts {
            assert_eq!(sub.parent_id, parent_id);
        }
    }
}
